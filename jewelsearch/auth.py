"""Login and registration backed by Supabase, enforced entirely on the server.

The browser never talks to Supabase. It only sees this app's own /api/auth/*
endpoints and an HttpOnly session cookie signed with a server-side secret.
Supabase Auth stores the credentials (hashed); the `profiles` table holds the
admin-approval flag (see supabase/schema.sql). Nothing is served until a user
is signed in AND approved, and approval is re-checked every minute, so
un-ticking `approved` in Supabase locks a user out within a minute.
"""
import asyncio
import base64
import hashlib
import hmac
import json
import os
import re
import secrets
import time
from collections import defaultdict, deque
from datetime import datetime, timezone

import httpx

from .config import DATA

SUPABASE_URL = os.environ.get("SUPABASE_URL", "").rstrip("/")
SUPABASE_KEY = os.environ.get("SUPABASE_SECRET_KEY", "")
COOKIE = "df_session"
SESSION_SECONDS = 7 * 24 * 3600
APPROVAL_TTL = 60   # seconds between re-checks of the approved flag
STALE_OK = 600      # keep using a known answer this long if Supabase is unreachable

EMAIL_RE = re.compile(r"^[^@\s]{1,64}@[^@\s]+\.[^@\s]{2,}$")
UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")


class AuthError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status, self.message = status, message


def configured() -> bool:
    return bool(SUPABASE_URL and SUPABASE_KEY)


# ---------- session cookie ----------

def _load_secret() -> bytes:
    if os.environ.get("SESSION_SECRET"):
        return os.environ["SESSION_SECRET"].encode()
    path = DATA / "session_secret"
    if not path.exists():
        DATA.mkdir(parents=True, exist_ok=True)
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as f:
            f.write(secrets.token_urlsafe(48))
    return path.read_text().strip().encode()


_SECRET = _load_secret()


def _b64(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode()


def _unb64(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


def _sign(body: str) -> str:
    return _b64(hmac.new(_SECRET, body.encode(), hashlib.sha256).digest())


def make_session(user: dict, now: float | None = None) -> str:
    exp = int((now or time.time()) + SESSION_SECONDS)
    body = _b64(json.dumps({"uid": user["id"], "email": user["email"], "name": user["name"], "exp": exp},
                           separators=(",", ":")).encode())
    return f"{body}.{_sign(body)}"


def read_session(token: str | None, now: float | None = None) -> dict | None:
    """The session payload if the cookie is authentic and unexpired, else None."""
    if not token or token.count(".") != 1 or len(token) > 2048:
        return None
    body, sig = token.split(".")
    if not hmac.compare_digest(sig, _sign(body)):
        return None
    try:
        data = json.loads(_unb64(body))
    except ValueError:
        return None
    if data.get("exp", 0) < (now or time.time()) or not UUID_RE.match(str(data.get("uid", ""))):
        return None
    return data


# ---------- input checks ----------

def clean_email(email: str) -> str:
    email = email.strip().lower()
    if not EMAIL_RE.match(email) or len(email) > 254:
        raise AuthError(400, "Enter a valid email address.")
    return email


def check_password(password: str):
    if len(password) < 8:
        raise AuthError(400, "Password must be at least 8 characters.")
    if len(password) > 72:
        raise AuthError(400, "Password must be at most 72 characters.")
    if password.isalpha() or password.isdigit():
        raise AuthError(400, "Use a mix of letters and numbers or symbols.")


def clean_name(name: str) -> str:
    name = " ".join(name.split())
    if not 1 <= len(name) <= 80:
        raise AuthError(400, "Enter your name (up to 80 characters).")
    return name


# ---------- brute-force limits ----------

class RateLimiter:
    def __init__(self):
        self.hits: dict[str, deque] = defaultdict(deque)

    def check(self, key: str, limit: int, window: float, now: float | None = None):
        now = now or time.monotonic()
        q = self.hits[key]
        while q and now - q[0] > window:
            q.popleft()
        if len(q) >= limit:
            raise AuthError(429, f"Too many attempts. Try again in {int(window - (now - q[0])) // 60 + 1} min.")
        q.append(now)


limiter = RateLimiter()


# ---------- Supabase (server-side only) ----------

_http = httpx.AsyncClient(timeout=10)


def _headers() -> dict:
    h = {"apikey": SUPABASE_KEY, "Content-Type": "application/json"}
    if SUPABASE_KEY.startswith("eyJ"):   # legacy service_role JWT key
        h["Authorization"] = f"Bearer {SUPABASE_KEY}"
    return h


def _error(r: httpx.Response) -> tuple[str, str]:
    try:
        j = r.json()
    except ValueError:
        return "", ""
    return str(j.get("error_code") or j.get("code") or j.get("error") or ""), str(j.get("msg") or j.get("error_description") or j.get("message") or "")


async def _call(method: str, path: str, headers: dict | None = None, **kw) -> httpx.Response:
    if not configured():
        raise AuthError(503, "Sign-in is not set up yet. Ask the administrator.")
    try:
        return await _http.request(method, SUPABASE_URL + path, headers={**_headers(), **(headers or {})}, **kw)
    except httpx.HTTPError:
        raise AuthError(503, "Account service is unreachable. Try again shortly.") from None


async def sign_up(email: str, password: str, name: str):
    r = await _call("POST", "/auth/v1/signup", json={"email": email, "password": password, "data": {"full_name": name}})
    if r.status_code < 300:
        return
    code, msg = _error(r)
    if code in ("user_already_exists", "email_exists"):
        return   # same answer as a new sign-up, so the form can't be used to probe emails
    if code == "weak_password":
        raise AuthError(400, msg or "Choose a stronger password.")
    if code in ("email_address_invalid", "validation_failed"):
        raise AuthError(400, "Enter a valid email address.")
    if r.status_code == 429:
        raise AuthError(429, "Too many registrations right now. Try again later.")
    if code == "signup_disabled":
        raise AuthError(403, "Registration is closed.")
    raise AuthError(502, "Registration failed. Try again later.")


async def sign_in(email: str, password: str) -> dict:
    r = await _call("POST", "/auth/v1/token", params={"grant_type": "password"}, json={"email": email, "password": password})
    if r.status_code >= 300:
        code, _ = _error(r)
        if code == "email_not_confirmed":
            raise AuthError(403, "Please confirm your email first (check your inbox).")
        if r.status_code == 429:
            raise AuthError(429, "Too many attempts. Try again later.")
        if r.status_code in (400, 401, 422):
            raise AuthError(401, "Email or password is incorrect.")
        raise AuthError(502, "Sign-in failed. Try again later.")
    return _user_from(r.json().get("user") or {})


def _user_from(u: dict) -> dict:
    if not UUID_RE.match(str(u.get("id", ""))):
        raise AuthError(502, "Sign-in failed. Try again later.")
    meta = u.get("user_metadata") or {}
    name = meta.get("full_name") or meta.get("name") or u.get("email", "").split("@")[0]
    return {"id": u["id"], "email": u.get("email", ""), "name": " ".join(str(name).split())[:80]}


# ---------- Google sign-in (OAuth with PKCE, exchanged on the server) ----------

def sign_blob(data: dict, ttl: int) -> str:
    body = _b64(json.dumps({**data, "exp": int(time.time()) + ttl}, separators=(",", ":")).encode())
    return f"{body}.{_sign(body)}"


def read_blob(token: str | None) -> dict | None:
    if not token or token.count(".") != 1 or len(token) > 2048:
        return None
    body, sig = token.split(".")
    if not hmac.compare_digest(sig, _sign(body)):
        return None
    try:
        data = json.loads(_unb64(body))
    except ValueError:
        return None
    return data if data.get("exp", 0) >= time.time() else None


def pkce_pair() -> tuple[str, str]:
    verifier = secrets.token_urlsafe(48)
    challenge = _b64(hashlib.sha256(verifier.encode()).digest())
    return verifier, challenge


def google_authorize_url(redirect_to: str, challenge: str) -> str:
    from urllib.parse import urlencode
    return SUPABASE_URL + "/auth/v1/authorize?" + urlencode({
        "provider": "google", "redirect_to": redirect_to,
        "code_challenge": challenge, "code_challenge_method": "s256",
    })


_settings: tuple[dict, float] | None = None


async def google_enabled() -> bool:
    """Whether the Google provider is switched on in Supabase (checked at most once a minute)."""
    global _settings
    if _settings and time.monotonic() - _settings[1] < 60:
        return bool(_settings[0].get("google"))
    try:
        r = await _call("GET", "/auth/v1/settings")
        external = r.json().get("external", {}) if r.status_code == 200 else {}
    except (AuthError, ValueError):
        return False
    _settings = (external, time.monotonic())
    return bool(external.get("google"))


async def exchange_code(code: str, verifier: str) -> dict:
    r = await _call("POST", "/auth/v1/token", params={"grant_type": "pkce"},
                    json={"auth_code": code, "code_verifier": verifier})
    if r.status_code >= 300:
        raise AuthError(401, "Google sign-in failed. Please try again.")
    return _user_from(r.json().get("user") or {})


_approval: dict[str, tuple[bool, float]] = {}
_locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)


async def is_approved(uid: str, fresh: bool = False) -> bool:
    """Whether an admin has approved this user (cached for APPROVAL_TTL seconds)."""
    if not configured() or not UUID_RE.match(uid):
        return False
    async with _locks[uid]:
        hit, now = _approval.get(uid), time.monotonic()
        if hit and not fresh and now - hit[1] < APPROVAL_TTL:
            return hit[0]
        try:
            r = await _call("GET", "/rest/v1/profiles", params={"id": f"eq.{uid}", "select": "approved"})
            r.raise_for_status()
            rows = r.json()
            ok = bool(rows) and rows[0].get("approved") is True
        except (AuthError, httpx.HTTPError, ValueError):
            if hit and now - hit[1] < STALE_OK:
                return hit[0]
            raise AuthError(503, "Account service is unreachable. Try again shortly.") from None
        _approval[uid] = (ok, now)
        return ok


_flags: dict[tuple[str, str], tuple[bool, float]] = {}


async def _flag(uid: str, column: str) -> bool:
    """Whether a role column of profiles is ticked for this user (cached like approval).

    Fails closed: if the column is missing (its .sql file not run yet) or
    Supabase is unreachable with no recent answer, the answer is no.
    """
    if not configured() or not UUID_RE.match(uid):
        return False
    hit, now = _flags.get((uid, column)), time.monotonic()
    if hit and now - hit[1] < APPROVAL_TTL:
        return hit[0]
    try:
        r = await _call("GET", "/rest/v1/profiles", params={"id": f"eq.{uid}", "select": column})
        rows = r.json() if r.status_code == 200 else []
        ok = bool(rows) and rows[0].get(column) is True
    except (AuthError, ValueError):
        return bool(hit and now - hit[1] < STALE_OK and hit[0])
    _flags[(uid, column)] = (ok, now)
    return ok


async def is_admin(uid: str) -> bool:
    """profiles.is_admin (supabase/admin.sql): the upload panel."""
    return await _flag(uid, "is_admin")


async def is_jeweler(uid: str) -> bool:
    """profiles.is_jeweler (supabase/orders.sql): the manufacturing side, the jeweler panel."""
    return await _flag(uid, "is_jeweler")


async def staff_role(uid: str) -> str | None:
    """"admin" or "jeweler" for people who may run orders, else None."""
    if await is_admin(uid):
        return "admin"
    if await is_jeweler(uid):
        return "jeweler"
    return None


async def record_login(uid: str):
    """Best effort: shows admins when each user last signed in."""
    try:
        await _call("PATCH", "/rest/v1/profiles", params={"id": f"eq.{uid}"},
                    json={"last_login_at": datetime.now(timezone.utc).isoformat()})
    except AuthError:
        pass


def safe_next(target: str | None) -> str:
    """Only allow redirects back into this site (blocks //evil.com and similar)."""
    if not target or not target.startswith("/") or target.startswith("//") or "\\" in target:
        return "/"
    return target
