"""FastAPI app: search API + dashboard, behind login.

Run:  .venv/bin/uvicorn jewelsearch.server:app --port 8765

Every route (pages, API, thumbnails, original renders, videos) goes through
`guard`, which lets a request through only with a valid session cookie of an
approved user. The only public routes are the sign-in page and its API.
"""
import base64
import json
import os
import re
import secrets
import threading
import time
from collections import OrderedDict
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal
from urllib.parse import quote

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Path as PathParam, Query, Request
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, StringConstraints, ValidationError
from starlette.concurrency import run_in_threadpool

from . import (auth, brands, webarchive, favorites, history, linksearch, manual, orders, phone, photo, purchase, recommend, storage,
               sketch, suggest, tryon, uploads, variation, webproducts)
from .config import CATEGORIES, CROPS, METALS, thumb_name
from .search import SearchEngine

STATIC = Path(__file__).resolve().parent / "static"
ORIGINAL_CACHE = "private, max-age=86400"
engine: SearchEngine | None = None
suggester: suggest.Suggester | None = None
recommender: recommend.Recommender | None = None
manual_panel: manual.Manual | None = None
web_pool: webproducts.WebProducts | None = None
studio: sketch.Studio | None = None
variation_jobs: variation.Jobs | None = None
sketch_tasks = sketch.Tasks()


@asynccontextmanager
async def lifespan(_app):
    global engine, suggester, recommender, manual_panel, web_pool, studio, variation_jobs
    engine = SearchEngine()
    if os.environ.get("JEWEL_JUDGE", "1") == "0":   # small servers (Render): words + picture check only
        engine.domain._judge = None
    else:
        engine.domain.load_judge()   # the local model that keeps searches to jewellery (domain.py), ~3.5 GB
    engine.warm_up()             # so the first photo or link after a restart doesn't wait
    suggester = suggest.Suggester(engine)
    recommender = recommend.Recommender(engine, suggester)
    manual_panel = manual.Manual(engine)
    web_pool = webproducts.WebProducts(engine)   # "From the web" panel (demo); empty until its build script ran
    studio = sketch.Studio()                     # Sketch to Design (Pollinations / Google wrapper); data/sketch
    variation_jobs = variation.Jobs(studio)      # Design Variations: same wrapper, sets made in the background
    yield


# No /docs or /openapi.json: the API shape is not advertised to visitors.
app = FastAPI(title="Jewellery design search", lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
app.mount("/crops", StaticFiles(directory=CROPS), name="crops")
# three.js, MediaPipe and the hand model are self-hosted (no CDN: privacy + CSP)
app.mount("/vendor", StaticFiles(directory=STATIC / "vendor"), name="vendor")
app.mount("/tryon-app", StaticFiles(directory=STATIC / "tryon"), name="tryon-app")
app.mount("/demo-app", StaticFiles(directory=STATIC / "demo"), name="demo-app")   # 3D look demo (6 designs)
TRYON_MODELS = Path(__file__).resolve().parent.parent / "data" / "tryon" / "models"

def _storage_origin() -> str:
    """The bucket's https address: pages may load original renders and 3D videos
    from it (through signed links the server hands out), nothing else."""
    try:
        origin = storage.get().media_origin()
    except Exception as e:   # e.g. no S3 keys in .env: media links will fail, pages still work
        print(f"storage: no media origin ({e})", flush=True)
        origin = None
    return f" {origin}" if origin else ""


MEDIA_ORIGIN = _storage_origin()
PAGES = {"/", "/admin", "/tryon", "/tryon/me", "/buy", "/jeweler", "/jeweler/jobcard", "/demo/look",
         "/brand-import", "/brand-design", "/sketch", "/variation"}   # sent to the sign-in page instead of getting a 401
PUBLIC = {"/login", "/api/auth/login", "/api/auth/register", "/api/auth/logout",
          "/api/auth/google", "/api/auth/google/callback"}
SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "same-origin",
    "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
    "Content-Security-Policy": (
        f"default-src 'self'; img-src 'self' data: blob:{MEDIA_ORIGIN}; media-src 'self'{MEDIA_ORIGIN}; connect-src 'self'; "
        "script-src 'self' 'unsafe-inline' 'wasm-unsafe-eval'; style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
        "font-src https://fonts.gstatic.com; frame-ancestors 'none'; base-uri 'none'; form-action 'self'; object-src 'none'"
    ),
}


# The try-on page is the only one allowed to use the camera. It also needs
# WebAssembly (MediaPipe, meshopt decoder) and blob: video frames.
TRYON_HEADERS = {
    **SECURITY_HEADERS,
    "Permissions-Policy": "camera=(self), microphone=(), geolocation=()",
    "Content-Security-Policy": (
        f"default-src 'self'; img-src 'self' data: blob:{MEDIA_ORIGIN}; media-src 'self' blob: mediastream:{MEDIA_ORIGIN}; "
        "connect-src 'self' data: blob:; worker-src 'self' blob:; "
        "script-src 'self' 'unsafe-inline' 'wasm-unsafe-eval'; style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
        "font-src https://fonts.gstatic.com; frame-ancestors 'none'; base-uri 'none'; form-action 'self'; object-src 'none'"
    ),
}


CAMERA_PAGES = {"/tryon", "/tryon/me", "/sketch", "/variation"}


def _secure(resp, path: str = ""):
    resp.headers.update(TRYON_HEADERS if path in CAMERA_PAGES else SECURITY_HEADERS)
    # the app's pages and scripts change often: the browser must check for a
    # newer copy every time (a quick 304 when unchanged), never run a stale one
    if (path in PAGES or path.startswith("/tryon-app/")) and "cache-control" not in resp.headers:
        resp.headers["Cache-Control"] = "no-cache"
    return resp


@app.middleware("http")
async def guard(request: Request, call_next):
    path = request.url.path
    # uvicorn prints http://0.0.0.0:8765, but browsers only allow the camera
    # on https or localhost: send the same page to localhost instead
    if request.url.hostname == "0.0.0.0" and request.method == "GET":
        return _secure(RedirectResponse(str(request.url.replace(hostname="localhost")), 307))
    if path not in PUBLIC:
        user = auth.read_session(request.cookies.get(auth.COOKIE))
        try:
            ok = bool(user) and await auth.is_approved(user["uid"])
        except auth.AuthError as e:
            return _secure(JSONResponse({"detail": e.message}, e.status))
        if not ok:
            if request.method == "GET" and path in PAGES:
                target = path + ("?" + request.url.query if request.url.query else "")
                resp = RedirectResponse("/login?next=" + quote(target, safe=""), 303)
            else:
                resp = JSONResponse({"detail": "Sign in required"}, 401)
            if user:   # authentic cookie, but approval was withdrawn
                resp.delete_cookie(auth.COOKIE, path="/")
            return _secure(resp)
        request.state.user = user
    return _secure(await call_next(request), path)


@app.exception_handler(auth.AuthError)
async def auth_error(_request, e: auth.AuthError):
    return JSONResponse({"detail": e.message}, e.status)


# ---------- auth API ----------

def json_only(request: Request):
    """Cross-site pages can't send JSON without a CORS preflight (which is never
    allowed), so requiring it blocks CSRF on these POSTs."""
    if request.headers.get("content-type", "").split(";")[0].strip().lower() != "application/json":
        raise HTTPException(415, "Expected JSON")


def client_ip(request: Request) -> str:
    host = request.client.host if request.client else ""
    if host in ("127.0.0.1", "::1"):   # arrived through the Cloudflare tunnel
        return request.headers.get("cf-connecting-ip", host)
    return host


class Credentials(BaseModel):
    email: str = Field(max_length=254)
    password: str = Field(max_length=128)


class Registration(Credentials):
    name: str = Field(max_length=120)


@app.post("/api/auth/register", dependencies=[Depends(json_only)])
async def register(body: Registration, request: Request):
    auth.limiter.check("reg:" + client_ip(request), 5, 3600)
    email, name = auth.clean_email(body.email), auth.clean_name(body.name)
    auth.check_password(body.password)
    await auth.sign_up(email, body.password, name)
    return {"ok": True, "message": "You can sign in once an administrator approves your account."}


@app.post("/api/auth/login", dependencies=[Depends(json_only)])
async def login(body: Credentials, request: Request):
    ip = client_ip(request)
    email = auth.clean_email(body.email)
    auth.limiter.check("ip:" + ip, 30, 300)
    auth.limiter.check(f"acct:{ip}:{email}", 8, 300)
    user = await auth.sign_in(email, body.password)
    if not await auth.is_approved(user["id"], fresh=True):
        raise auth.AuthError(403, "Your account is waiting for administrator approval.")
    await auth.record_login(user["id"])
    return _start_session(JSONResponse({"ok": True, "name": user["name"]}), request, user)


def _is_https(request: Request) -> bool:
    return request.url.scheme == "https" or request.headers.get("x-forwarded-proto") == "https"


def _start_session(resp, request: Request, user: dict):
    resp.set_cookie(auth.COOKIE, auth.make_session(user), max_age=auth.SESSION_SECONDS, path="/",
                    httponly=True, secure=_is_https(request), samesite="lax")
    return resp


def _public_origin(request: Request) -> str:
    """Where Google should send people back to. PUBLIC_URL in .env pins it; otherwise
    use the address the visitor used (Supabase only accepts allow-listed ones)."""
    if os.environ.get("PUBLIC_URL"):
        return os.environ["PUBLIC_URL"].rstrip("/")
    return f"{'https' if _is_https(request) else 'http'}://{request.headers.get('host', request.url.netloc)}"


OAUTH_COOKIE, OAUTH_PATH = "df_oauth", "/api/auth/google"


@app.get("/api/auth/google")
async def google_start(request: Request, next: str = "/"):
    auth.limiter.check("google:" + client_ip(request), 20, 300)
    if not await auth.google_enabled():
        return RedirectResponse("/login?error=google_disabled", 303)
    verifier, challenge = auth.pkce_pair()
    redirect_to = _public_origin(request) + "/api/auth/google/callback"
    resp = RedirectResponse(auth.google_authorize_url(redirect_to, challenge), 303)
    # The verifier never leaves the server except inside this signed, HttpOnly, 10-minute cookie.
    resp.set_cookie(OAUTH_COOKIE, auth.sign_blob({"v": verifier, "next": auth.safe_next(next)}, 600),
                    max_age=600, path=OAUTH_PATH, httponly=True, secure=_is_https(request), samesite="lax")
    return resp


@app.get("/api/auth/google/callback")
async def google_callback(request: Request, code: str | None = None, error: str | None = None):
    flow = auth.read_blob(request.cookies.get(OAUTH_COOKIE))

    def back(target: str):
        resp = RedirectResponse(target, 303)
        resp.delete_cookie(OAUTH_COOKIE, path=OAUTH_PATH)
        return resp

    if error == "access_denied":
        return back("/login?error=google_cancelled")
    if error or not code or not flow or len(code) > 512:
        return back("/login?error=google_failed")
    try:
        user = await auth.exchange_code(code, flow["v"])
        approved = await auth.is_approved(user["id"], fresh=True)
    except auth.AuthError:
        return back("/login?error=google_failed")
    if not approved:
        return back("/login?status=pending")
    await auth.record_login(user["id"])
    return _start_session(back(flow["next"]), request, user)


@app.post("/api/auth/logout")
def logout():
    resp = JSONResponse({"ok": True})
    resp.delete_cookie(auth.COOKIE, path="/")
    return resp


@app.get("/api/auth/me")
async def me(request: Request):
    u = request.state.user
    return {"name": u["name"], "email": u["email"], "is_admin": await auth.is_admin(u["uid"]),
            "is_jeweler": await auth.is_jeweler(u["uid"]),
            "originals": storage.get().ready()}   # false: pictures fall back to small previews (see /media)


# ---------- search API ----------

def _check(category, metal):
    """Filters: omitted = take from the prompt, "any" = no filter, else a fixed value."""
    if category and category not in CATEGORIES + ["any"]:
        raise HTTPException(400, f"category must be 'any' or one of {CATEGORIES}")
    if metal and metal not in METALS + ["any"]:
        raise HTTPException(400, f"metal must be 'any' or one of {METALS}")


def _uid(uid: int):
    if uid not in engine.by_uid:
        raise HTTPException(404, "unknown design")
    return uid


@app.get("/api/search")
def search(request: Request, background: BackgroundTasks, q: str = Query(..., min_length=1, max_length=500),
           category: str | None = None, metal: str | None = None, page: int = Query(0, ge=0, le=20)):
    _check(category, metal)
    t = time.perf_counter()
    res = engine.search(q, category=category, metal=metal, page=page)
    res["ms"] = round((time.perf_counter() - t) * 1000)
    if page == 0:   # saved after the response is sent, so it adds no delay
        background.add_task(history.record, request.state.user["uid"], q.strip(), category, metal, res)
    return res


class ManualIn(BaseModel):
    picks: list[Annotated[str, StringConstraints(max_length=40)]] = Field(default_factory=list,
                                                                          max_length=manual.MAX_PICKS)


@app.post("/api/manual", dependencies=[Depends(json_only)])
def manual_options(body: ManualIn):
    """Manual search: every option with how many designs it leaves, and the picks
    written as the prompt the search runs (jewelsearch/manual.py). The search
    itself is /api/search with that prompt."""
    return JSONResponse(manual_panel(body.picks), headers={"Cache-Control": "no-store"})


@app.get("/api/suggest")
def suggestions(q: str = Query("", max_length=500), photo: bool = False,
                category: str | None = Query(None, max_length=20)):
    """Suggestions under the search box while typing (jewelsearch/suggest.py).
    Made from the collection, never from other users' searches; each one is
    checked to fill a page of results. With a photo: photo=1 and the type the
    photo was read as (category), when sure. Not cached by the browser."""
    if category and category not in CATEGORIES:
        raise HTTPException(400, f"category must be one of {CATEGORIES}")
    return JSONResponse(suggester(q, photo=photo, category=category), headers={"Cache-Control": "no-store"})


# ---------- suggested prompts under the search bar (recommend.py) ----------
POPULAR_TTL = 600                       # everyone's recent searches are re-read every 10 minutes
_popular: dict = {"at": 0.0, "rows": []}
_popular_lock = None
_prompt_cache: OrderedDict = OrderedDict()   # (user, newest search, favourites) -> answer


async def _popular_rows() -> list[dict]:
    global _popular_lock
    import asyncio
    _popular_lock = _popular_lock or asyncio.Lock()
    async with _popular_lock:
        if time.time() - _popular["at"] > POPULAR_TTL:
            try:
                _popular["rows"] = await history.recent_meanings(recommend.TRENDING_DAYS)
            except auth.AuthError:
                pass   # keep the last good copy; popular prompts are optional
            _popular["at"] = time.time()
    return _popular["rows"]


@app.get("/api/prompt-suggestions")
async def prompt_suggestions(request: Request):
    """The 5 suggested prompts for this shopper: their own searches and saved
    designs, what other shoppers search (meaning only) and the collection."""
    uid = request.state.user["uid"]
    try:
        rows = await history.list_for(uid, 200)
    except auth.AuthError:
        rows = []
    try:
        favs = await favorites.list_for(uid)
    except auth.AuthError:
        favs = []
    popular = await _popular_rows()
    key = (uid, rows[0]["id"] if rows else None, rows[0]["created_at"] if rows else None,
           tuple(sorted(f["design_uid"] for f in favs)), _popular["at"])
    res = _prompt_cache.get(key)
    if res is None:
        res = await run_in_threadpool(recommender, rows, favs, popular, uid)
        _prompt_cache[key] = res
        while len(_prompt_cache) > 500:
            _prompt_cache.popitem(last=False)
    return JSONResponse(res, headers={"Cache-Control": "no-store"})


# ---------- search by photo (+ optional words) ----------
# The photo is read once (vector + design DNA) and kept in memory for an hour
# under a random token bound to the user, so changing a filter or loading more
# results doesn't send the photo again. Photos are never written to disk.

PHOTO_BODY_MAX = 12 * 1024 * 1024    # JSON with a base64 data URL; the page sends a ~1024px JPEG
PHOTO_TTL, PHOTO_KEEP = 3600, 300
_photos: OrderedDict[str, tuple] = OrderedDict()   # token -> (user id, time, PhotoQuery)
_photos_lock = threading.Lock()


def _photo_put(owner: str, pq) -> str:
    token = secrets.token_urlsafe(18)
    with _photos_lock:
        _photos[token] = (owner, time.time(), pq)
        while len(_photos) > PHOTO_KEEP:
            _photos.popitem(last=False)
    return token


def _photo_get(owner: str, token: str):
    with _photos_lock:
        hit = _photos.get(token)
        if not hit or hit[0] != owner or time.time() - hit[1] > PHOTO_TTL:
            return None
        _photos.move_to_end(token)
        return hit[2]


@app.exception_handler(photo.PhotoError)
async def photo_error(_request, e: photo.PhotoError):
    return JSONResponse({"detail": e.message}, e.status)


class PhotoSearchIn(BaseModel):
    image: str = Field(max_length=PHOTO_BODY_MAX)
    q: str = Field("", max_length=500)
    category: str | None = Field(None, max_length=20)
    metal: str | None = Field(None, max_length=20)


@app.post("/api/photo-search", dependencies=[Depends(json_only)])
async def photo_search(request: Request):
    """Upload a photo (plus optional words and filters): its design DNA and the closest designs."""
    length = request.headers.get("content-length", "")
    if not length.isdigit() or int(length) > PHOTO_BODY_MAX:
        raise HTTPException(413, "The photo is too large. Use one under 8 MB.")
    raw = await request.body()
    if len(raw) > PHOTO_BODY_MAX:
        raise HTTPException(413, "The photo is too large. Use one under 8 MB.")
    try:
        body = PhotoSearchIn.model_validate_json(raw)
    except ValidationError:
        raise HTTPException(422, "Bad request")
    _check(body.category, body.metal)
    user = request.state.user
    auth.limiter.check("photo:" + user["uid"], 40, 600)

    def work():
        t = time.perf_counter()
        pq = engine.read_photo(photo.read(photo.from_data_url(body.image)))
        res = engine.search_photo(pq, body.q.strip(), category=body.category, metal=body.metal)
        res["ms"] = round((time.perf_counter() - t) * 1000)
        res["token"] = _photo_put(user["uid"], pq)
        return res
    return await run_in_threadpool(work)


class LinkSearchIn(BaseModel):
    text: str = Field(max_length=2500)   # the pasted link, with the shopper's words around it
    category: str | None = Field(None, max_length=20)
    metal: str | None = Field(None, max_length=20)


@app.post("/api/link-search", dependencies=[Depends(json_only)])
async def link_search(request: Request, body: LinkSearchIn):
    """A pasted product link (plus optional words): the jewellery picture on that page,
    matched like an uploaded photo; else its jewellery words; else a clear message
    (jewelsearch/linksearch.py). The picture is kept like a photo (memory only, 1 h)."""
    _check(body.category, body.metal)
    user = request.state.user
    auth.limiter.check("link:" + user["uid"], 20, 600)

    def work():
        t = time.perf_counter()
        try:
            res = linksearch.search(engine, body.text, body.category, body.metal)
        except linksearch.LinkError as e:
            raise HTTPException(400, e.message)
        pq = res.pop("_photo", None)
        if pq is not None:
            res["token"] = _photo_put(user["uid"], pq)
        res["ms"] = round((time.perf_counter() - t) * 1000)
        return res
    return await run_in_threadpool(work)


@app.get("/api/photo-search/{token}")
def photo_search_again(request: Request, token: str = PathParam(..., max_length=40),
                       q: str = Query("", max_length=500), category: str | None = None,
                       metal: str | None = None, page: int = Query(0, ge=0, le=50)):
    """Same photo, new words / filters / page."""
    _check(category, metal)
    pq = _photo_get(request.state.user["uid"], token)
    if pq is None:
        raise HTTPException(410, "This photo search has expired. Add the photo again.")
    t = time.perf_counter()
    res = engine.search_photo(pq, q.strip(), category=category, metal=metal, page=page)
    res["ms"] = round((time.perf_counter() - t) * 1000)
    res["token"] = token
    return res


# ---------- browsing without a prompt ----------

@app.get("/api/categories")
def categories():
    return {"items": engine.categories()}


@app.get("/api/browse")
def browse(category: str = Query(..., max_length=20), page: int = Query(0, ge=0, le=500)):
    if category not in CATEGORIES:
        raise HTTPException(400, f"category must be one of {CATEGORIES}")
    return engine.browse(category, page)


# ---------- search history (per user, stored in Supabase) ----------

def _cards(uids: list[int], metal: str | None) -> list[dict]:
    return [engine.card(u, metal) for u in uids if u in engine.by_uid]


@app.get("/api/history")
async def history_list(request: Request, limit: int = Query(50, ge=1, le=200)):
    rows = await history.list_for(request.state.user["uid"], limit)
    items = []
    for r in rows:
        u = r["understood"] or {}
        first = _cards(r["result_uids"][:1], u.get("metal"))
        items.append({
            "id": r["id"], "query": r["query"], "created_at": r["created_at"], "matches": r["matches"],
            "category": u.get("category"), "metal": u.get("metal"), "count": len(r["result_uids"]),
            "thumb": first[0]["thumb"] if first else None,
        })
    return {"items": items}


@app.get("/api/history/{entry_id}")
async def history_entry(request: Request, entry_id: int = PathParam(..., ge=1)):
    r = await history.get(request.state.user["uid"], entry_id)
    if not r:
        raise HTTPException(404, "not found")
    u = r["understood"] or {}
    return {
        "id": r["id"], "query": r["query"], "created_at": r["created_at"], "understood": u, "notes": r["notes"],
        "category_override": r["category_override"], "metal_override": r["metal_override"],
        "matches": r["matches"], "results": _cards(r["result_uids"], u.get("metal")),
    }


@app.delete("/api/history/{entry_id}")
async def history_delete(request: Request, entry_id: int = PathParam(..., ge=1)):
    await history.delete(request.state.user["uid"], entry_id)
    return {"ok": True}


@app.delete("/api/history")
async def history_clear(request: Request):
    await history.delete(request.state.user["uid"])
    return {"ok": True}


# ---------- favourites (per user, stored in Supabase) ----------

_by_design_id: dict[str, int] = {}


def _resolve(design_uid: int, design_id: str) -> int | None:
    """uids come from the index; if it was rebuilt, find the design by its id instead."""
    if design_uid in engine.by_uid and engine.by_uid[design_uid]["design_id"] == design_id:
        return design_uid
    if not _by_design_id:
        _by_design_id.update({m["design_id"]: u for u, m in engine.by_uid.items()})
    return _by_design_id.get(design_id)


@app.get("/api/favorites")
async def favorites_list(request: Request):
    items = []
    for r in await favorites.list_for(request.state.user["uid"]):
        uid = _resolve(r["design_uid"], r["design_id"])
        if uid is not None:
            items.append({**engine.card(uid), "liked_at": r["created_at"]})
    return {"items": items}


@app.get("/api/favorites/ids")
async def favorites_ids(request: Request):
    rows = await favorites.list_for(request.state.user["uid"])
    return {"uids": [u for r in rows if (u := _resolve(r["design_uid"], r["design_id"])) is not None]}


@app.put("/api/favorites/{uid}")
async def favorites_add(request: Request, uid: int = PathParam(..., ge=0)):
    await favorites.add(request.state.user["uid"], _uid(uid), engine.by_uid[uid]["design_id"])
    return {"ok": True}


@app.delete("/api/favorites/{uid}")
async def favorites_remove(request: Request, uid: int = PathParam(..., ge=0)):
    await favorites.remove(request.state.user["uid"], uid)
    return {"ok": True}


@app.get("/api/similar/{uid}")
def similar(uid: int, metal: str | None = None):
    if metal and metal not in METALS:
        raise HTTPException(400, f"metal must be one of {METALS}")
    return engine.similar(_uid(uid), metal=metal)


# ---------- "From the web" panel (demo, jewelsearch/webproducts.py) ----------
# Read-only: nothing here records history or touches user data.

WEB_ID = r"^[0-9a-f]{12}$"


def _web():
    if web_pool is None or not web_pool.ready:
        raise HTTPException(503, "The web collection isn't ready yet.")
    return web_pool


@app.get("/api/web-products")
def web_products(q: str = Query(..., min_length=1, max_length=500), category: str | None = None,
                 metal: str | None = None):
    _check(category, metal)
    return _web().for_prompt(q, category=category, metal=metal)


@app.get("/api/web-products/photo/{token}")
def web_products_like_photo(request: Request, token: str = PathParam(..., max_length=40),
                            q: str = Query("", max_length=500), category: str | None = None,
                            metal: str | None = None):
    """Web products like the photo (or linked picture) of a photo search: same token, no upload."""
    _check(category, metal)
    pq = _photo_get(request.state.user["uid"], token)
    if pq is None:
        raise HTTPException(410, "This photo search has expired. Add the photo again.")
    return _web().for_photo(pq, q, category=category, metal=metal)


@app.get("/api/web-products/design/{uid}")
def web_products_like_design(uid: int, metal: str | None = None):
    if metal and metal not in METALS:
        raise HTTPException(400, f"metal must be one of {METALS}")
    return _web().like_design(_uid(uid), metal=metal)


@app.get("/api/web-products/item/{pid}")
def web_products_like_item(pid: Annotated[str, PathParam(pattern=WEB_ID)]):
    try:
        return _web().like_product(pid)
    except KeyError:
        raise HTTPException(404, "Unknown product")


@app.get("/web-img/{pid}/{size}")
def web_picture(pid: Annotated[str, PathParam(pattern=WEB_ID)], size: int):
    if size not in webproducts.SIZES:
        raise HTTPException(404, "Unknown size")
    try:
        f = _web().picture(pid, size)
    except KeyError:
        raise HTTPException(404, "Unknown product")
    except (linksearch.LinkError, photo.PhotoError):
        raise HTTPException(502, "The shop's picture couldn't be loaded.")
    return FileResponse(f, media_type="image/webp", headers={"Cache-Control": "private, max-age=604800"})


@app.get("/api/phone-link")
def phone_link(request: Request, path: str = Query("/", max_length=500)):
    """The public https address of a page of this app, and its QR code, so
    it can be opened on a phone (whose camera needs https)."""
    if not phone.safe_path(path):
        raise HTTPException(400, "Bad path")
    base = phone.public_base(request.headers.get("host"), request.headers.get("x-forwarded-proto"))
    if not base:
        raise HTTPException(503, "The app isn't live yet. Start it with scripts/start_live.sh to open it on a phone.")
    url = base + path
    return {"url": url, "svg": phone.qr_svg(url)}


@app.get("/api/design/{uid}")
def design(uid: int):
    return engine.detail(_uid(uid))


@app.get("/api/buy/{uid}")
def buy_details(uid: int = PathParam(..., ge=0), id: str | None = Query(None, max_length=200)):
    """Everything the buy page shows for one design: its pictures, and its
    gold weight, stones and size from the job card or the CAD file. The
    design id guards against a link made before the index was rebuilt."""
    found = _resolve(uid, id) if id else _uid(uid)
    if found is None:
        raise HTTPException(404, "This design is no longer in the catalogue")
    return {**engine.detail(found), "specs": purchase.details(engine.by_uid[found])}


@app.get("/media/{token}")
def media(request: Request, token: str = PathParam(..., pattern=r"^[0-9a-f]{16}$")):
    """A full-size render or 3D video, by the opaque ID the search API gave the page.

    Pages never see or send dataset paths: the ID maps to a catalogue file on
    the server, so nothing else can be requested. With S3 storage the browser
    is sent to a signed link (valid for an hour) and downloads straight from
    the bucket, the smaller web copy when there is one; with a dataset folder
    the file is sent from here.
    """
    path = engine.media_by_token.get(token)
    if path is None:
        raise HTTPException(404, "not found")
    st = storage.get()
    try:
        url = st.media_url(path)
        if url:
            # the same link is reused for 45 min, so the browser's own cache keeps working
            return RedirectResponse(url, 302, headers={"Cache-Control": "private, max-age=600"})
        # the file exactly as it is in the dataset; a day in the browser cache, so a 2 MB
        # render is downloaded once, not on every search that shows it, and after that day
        # an unchanged file is answered "not modified" (a few bytes) instead of sent again
        f = st.local_copy(path)
        stat = f.stat()
        etag = f'"{stat.st_size:x}-{int(stat.st_mtime):x}"'
        headers = {"Cache-Control": ORIGINAL_CACHE, "ETag": etag}
        if etag in request.headers.get("if-none-match", ""):
            return Response(status_code=304, headers=headers)
        return FileResponse(f, headers=headers)
    except (FileNotFoundError, storage.StorageUnavailable) as e:
        # a catalogue picture whose original can't be read (the dataset drive isn't connected):
        # its small preview, so the page still works, never cached in place of the original
        preview = CROPS / thumb_name(path)
        if storage.kind_of(path) == "image" and preview.is_file():
            return FileResponse(preview, headers={"Cache-Control": "no-store", "X-Original": "unavailable"})
        if isinstance(e, FileNotFoundError):
            raise HTTPException(404, "This file is not in the dataset storage yet")
        raise HTTPException(503, "dataset storage not available")


# ---------- try-on models (made by scripts/cad_to_glb.py) ----------

_SLUG = re.compile(r"^[A-Za-z0-9_-]{1,120}$")


@app.get("/api/tryon/models")
def tryon_models(category: str | None = None):
    items = []
    for f in sorted(TRYON_MODELS.glob("*.json")):
        meta = json.loads(f.read_text())
        if meta.get("status") == "ok" and (category is None or meta["category"] == category) and tryon.passed(meta["slug"]):
            meta.pop("source", None)   # dataset paths stay on the server
            meta["version"] = _model_version(meta["slug"])
            items.append(meta)
    return {"items": items}


def _model_version(slug: str) -> int:
    """Changes when the model is converted again: clients put it in the .glb
    URL, so a day-long browser cache never shows an old model."""
    f = TRYON_MODELS / f"{slug}.glb"
    return int(f.stat().st_mtime) if f.exists() else 0


@app.get("/api/tryon/models/{slug}.glb")
def tryon_model(slug: str):
    f = TRYON_MODELS / f"{slug}.glb"
    if not _SLUG.match(slug) or not f.is_file():
        raise HTTPException(404, "no try-on model")
    # private: only signed-in users may fetch; a day in the browser cache is fine
    return FileResponse(f, media_type="model/gltf-binary", headers={"Cache-Control": "private, max-age=86400"})


@app.get("/api/tryon/design/{uid}")
def tryon_design(uid: int = PathParam(..., ge=0)):
    """What one catalogue design can be tried on with (404 if nothing):
    the 3D model if it passed its check, and/or its own catalogue render."""
    m = engine.by_uid[_uid(uid)]
    slug = tryon.model_for(m)
    photo = tryon.photo_ok(m)
    if not slug and not photo:
        raise HTTPException(404, "This design has no try-on yet")
    meta_slug = slug or tryon.converted(m)   # real size, even when the 3D look failed its check
    meta = json.loads((TRYON_MODELS / f"{meta_slug}.json").read_text()) if meta_slug else {}
    meta.pop("source", None)
    if slug:
        meta["version"] = _model_version(slug)
    card = engine.card(uid)
    # the catalogue render per metal, for the "real photo" try-on
    photos = {mt: f"/api/tryon/photo/{uid}?metal={mt}" for mt in card["metals"]
              if photo and tryon.front_image(m, mt)}
    return {"slug": slug, "has3d": bool(slug), "part": tryon.PART_FOR[m["category"]], "model": meta,
            "card": card, "photos": photos}


@app.get("/api/tryon/photo/{uid}")
def tryon_photo(uid: int = PathParam(..., ge=0), metal: str = Query(..., max_length=20)):
    """The design's own front render, trimmed, for the "real photo" try-on."""
    m = engine.by_uid[_uid(uid)]
    rel = tryon.front_image(m, metal) if m["category"] in tryon.PHOTO_PARTS else None
    if not rel or rel not in engine.allowed_media:
        raise HTTPException(404, "no front photo")
    try:
        f = tryon.cutout(rel)
    except FileNotFoundError:
        raise HTTPException(404, "no front photo")
    except storage.StorageUnavailable:
        raise HTTPException(503, "dataset storage not available")
    return FileResponse(f, media_type="image/png", headers={"Cache-Control": "private, max-age=86400"})


# ---------- body photos for the instant try-on (per user, this server only) ----------

class BodyPhoto(BaseModel):
    image: str = Field(max_length=4_500_000)     # data URL of a JPEG the page re-encoded
    width: int
    height: int
    anchors: dict
    consent: bool


@app.get("/api/body")
def body_list(request: Request):
    return {"parts": tryon.list_photos(request.state.user["uid"])}


@app.put("/api/body/{part}", dependencies=[Depends(json_only)])
def body_save(request: Request, body: BodyPhoto, part: str = PathParam(..., max_length=10)):
    tryon.save_photo(request.state.user["uid"], part, body.image, body.width, body.height, body.anchors, body.consent)
    return {"parts": tryon.list_photos(request.state.user["uid"])}


@app.get("/api/body/{part}.jpg")
def body_photo(request: Request, part: str = PathParam(..., max_length=10)):
    # personal photo: only for this user, never kept by shared caches
    return FileResponse(tryon.photo_path(request.state.user["uid"], part), media_type="image/jpeg",
                        headers={"Cache-Control": "private, no-store"})


@app.delete("/api/body/{part}")
def body_delete(request: Request, part: str = PathParam(..., max_length=10)):
    tryon.delete_photo(request.state.user["uid"], part)
    return {"parts": tryon.list_photos(request.state.user["uid"])}


@app.delete("/api/body")
def body_delete_all(request: Request):
    tryon.delete_photo(request.state.user["uid"])
    return {"parts": {}}


# ---------- admin: upload new design files to the dataset drive ----------

async def admin_only(request: Request):
    if not await auth.is_admin(request.state.user["uid"]):
        raise HTTPException(403, "Admins only")


@app.exception_handler(uploads.UploadError)
async def upload_error(_request, e: uploads.UploadError):
    return JSONResponse({"detail": e.message}, e.status)


@app.get("/api/admin/uploads", dependencies=[Depends(admin_only)])
def admin_info(limit: int = Query(100, ge=1, le=500)):
    ready = uploads.drive_ready()
    free = uploads.free_bytes()
    return {
        "drive_ready": ready, "free_bytes": free, "reserve_bytes": uploads.RESERVE if free is not None else 0,
        "storage": storage.kind(),
        "root": uploads.new_root(), "folders": uploads.FOLDERS, "custom_categories": uploads.custom_categories(),
        "chunk_bytes": uploads.CHUNK, "max_file_bytes": uploads.MAX_FILE, "recent": uploads.recent(limit),
    }


class UploadStart(BaseModel):
    category: str = Field(max_length=20)
    name: str = Field(max_length=255)
    size: int = Field(ge=0)
    batch: str = Field("", max_length=120)
    custom: str = Field("", max_length=120)   # category folder name when category is "other"
    subdirs: str = Field("", max_length=1000)


class FolderCheck(BaseModel):
    category: str = Field(max_length=20)
    custom: str = Field("", max_length=120)
    folders: list[str] = Field(max_length=200)   # relative to the category folder


@app.post("/api/admin/uploads/check", dependencies=[Depends(admin_only), Depends(json_only)])
def admin_folder_check(body: FolderCheck):
    """Folders that already exist, so the panel can ask before mixing new files into them."""
    return {"existing": uploads.existing_folders(body.category, body.custom, body.folders)}


@app.post("/api/admin/uploads", dependencies=[Depends(admin_only), Depends(json_only)])
async def admin_upload_start(body: UploadStart, request: Request):
    u = request.state.user
    up = uploads.start(u["uid"], u["email"], body.category, body.name, body.size, body.batch, body.subdirs, body.custom)
    return {"id": up.id, "chunk_bytes": uploads.CHUNK}


@app.put("/api/admin/uploads/{upload_id}", dependencies=[Depends(admin_only)])
async def admin_upload_chunk(request: Request, upload_id: str = PathParam(..., max_length=40),
                             offset: int = Query(..., ge=0)):
    # Raw bytes, not a form: a cross-site page can't send a PUT without a CORS preflight.
    length = request.headers.get("content-length", "0")
    if not length.isdigit() or int(length) > uploads.CHUNK:
        raise HTTPException(413, "Chunk too large")
    return await uploads.append(upload_id, request.state.user["uid"], offset, request.stream())


@app.delete("/api/admin/uploads/{upload_id}", dependencies=[Depends(admin_only)])
async def admin_upload_cancel(request: Request, upload_id: str = PathParam(..., max_length=40)):
    uploads.cancel(upload_id, request.state.user["uid"])
    return {"ok": True}


# ---------- orders: customers order from the buy page, jewelers run them ----------

@app.exception_handler(orders.OrderError)
async def order_error(_request, e: orders.OrderError):
    return JSONResponse({"detail": e.message}, e.status)


class OrderIn(BaseModel):
    uid: int = Field(ge=0)
    design_id: str = Field(max_length=200)
    metal: str = Field(max_length=20)
    purity: str = Field(max_length=5)
    ring_size: int | None = Field(None, ge=0, le=60)
    quantity: int = Field(1, ge=1, le=orders.MAX_QTY)
    phone: str = Field(max_length=30)
    note: str = Field("", max_length=1000)


@app.post("/api/orders", dependencies=[Depends(json_only)])
async def order_place(body: OrderIn, request: Request):
    """A customer orders a design as it is. The choices are checked again here
    (the page could be old or edited) and frozen into the order."""
    u = request.state.user
    auth.limiter.check("order:" + u["uid"], 10, 3600)
    found = _resolve(body.uid, body.design_id)
    if found is None:
        raise HTTPException(404, "This design is no longer in the catalogue")
    m = engine.by_uid[found]
    # the order keeps preview links (they outlive catalogue changes); pages show the originals
    d = engine.detail(found)
    frozen = {**d, "thumb": engine.preview_url(d["thumb"]),
              "thumbs_by_metal": {mt: engine.preview_url(t) for mt, t in d["thumbs_by_metal"].items()},
              "views": {mt: [{**v, "thumb": engine.preview_url(v["thumb"])} for v in vs] for mt, vs in d["views"].items()}}
    row = orders.build_order(u, m, frozen, purchase.details(m), metal=body.metal, purity=body.purity,
                             ring_size=body.ring_size, quantity=body.quantity, phone=body.phone, note=body.note)
    order = await orders.create(row, {"id": u["uid"], "name": u["name"], "role": "customer"})
    return {"id": order["id"], "order_no": orders.order_no(order["id"]), "phone": row["customer_phone"]}


async def jeweler_only(request: Request):
    """Jewelers (profiles.is_jeweler) and admins."""
    role = await auth.staff_role(request.state.user["uid"])
    if not role:
        raise HTTPException(403, "Jewelers only")
    u = request.state.user
    request.state.actor = {"id": u["uid"], "name": u["name"] or u["email"], "role": role}


def _actor(request: Request) -> dict:
    return request.state.actor


def _originals(order: dict) -> dict:
    """An order's pictures as the original renders (it stores their previews)."""
    if order.get("thumb"):
        order["thumb"] = engine.original_url(order["thumb"])
    design = (order.get("snapshot") or {}).get("design") or {}
    if design.get("views"):
        design["views"] = [engine.original_url(v) for v in design["views"]]
    return order


@app.get("/api/jeweler/orders", dependencies=[Depends(jeweler_only)])
async def jeweler_orders(show: str = Query("active", max_length=20), q: str = Query("", max_length=100),
                         limit: int = Query(200, ge=1, le=500)):
    res = await orders.list_orders(show, q, limit)
    res["items"] = [_originals(o) for o in res.get("items", [])]
    return {**res, "stages": orders.STAGES}


async def _order_page(order_id: int, order: dict | None = None) -> dict:
    """One order with its history, for the panel and the job card."""
    order = order or await orders.get(order_id)
    brand = order.get("kind") == "brand"   # our version of another jeweller's design (brands.py)
    uid = None if brand else _resolve(order["design_uid"], order["design_id"])
    return {
        "order": _originals(orders.view(order)), "events": await orders.events(order_id),
        "stages": orders.STAGES, "labels": orders.LABELS,
        "pay_methods": orders.PAY_METHODS, "making_modes": orders.MAKING_MODES,
        "catalogue_uid": uid,   # None: the design has left the catalogue (the order keeps its frozen copy)
        "brand_id": order["design_uid"] if brand else None,
    }


@app.get("/api/jeweler/orders/{order_id}", dependencies=[Depends(jeweler_only)])
async def jeweler_order(order_id: int = PathParam(..., ge=1)):
    return await _order_page(order_id)


class StageMove(BaseModel):
    to: str = Field(max_length=20)
    note: str = Field("", max_length=500)
    seen: str = Field(max_length=64)   # the order's updated_at as the jeweler saw it


@app.post("/api/jeweler/orders/{order_id}/stage", dependencies=[Depends(jeweler_only), Depends(json_only)])
async def jeweler_stage(body: StageMove, request: Request, order_id: int = PathParam(..., ge=1)):
    saved = await orders.move(order_id, body.to, body.note, body.seen, _actor(request))
    return await _order_page(order_id, saved)


class QuoteIn(BaseModel):
    gold_weight_g: float
    gold_rate: float
    making_mode: Literal["per_g", "percent", "fixed"]
    making_value: float
    diamond_value: float = 0
    diamond_quality: str = Field("", max_length=60)
    other_label: str = Field("", max_length=60)
    other_value: float = 0
    gst_pct: float
    note: str = Field("", max_length=500)
    seen: str = Field(max_length=64)


@app.put("/api/jeweler/orders/{order_id}/quote", dependencies=[Depends(jeweler_only), Depends(json_only)])
async def jeweler_quote(body: QuoteIn, request: Request, order_id: int = PathParam(..., ge=1)):
    saved = await orders.save_quote(order_id, body.model_dump(exclude={"seen"}), body.seen, _actor(request))
    return await _order_page(order_id, saved)


class PaymentIn(BaseModel):
    amount: float
    method: str = Field(max_length=20)
    ref: str = Field("", max_length=80)
    seen: str = Field(max_length=64)


@app.post("/api/jeweler/orders/{order_id}/payments", dependencies=[Depends(jeweler_only), Depends(json_only)])
async def jeweler_payment(body: PaymentIn, request: Request, order_id: int = PathParam(..., ge=1)):
    saved = await orders.add_payment(order_id, body.amount, body.method, body.ref, body.seen, _actor(request))
    return await _order_page(order_id, saved)


@app.delete("/api/jeweler/orders/{order_id}/payments/{pay_id}", dependencies=[Depends(jeweler_only)])
async def jeweler_payment_remove(request: Request, order_id: int = PathParam(..., ge=1),
                                 pay_id: str = PathParam(..., max_length=16), seen: str = Query(..., max_length=64)):
    saved = await orders.remove_payment(order_id, pay_id, seen, _actor(request))
    return await _order_page(order_id, saved)


class WeightIn(BaseModel):
    grams: float
    note: str = Field("", max_length=300)
    seen: str = Field(max_length=64)


@app.put("/api/jeweler/orders/{order_id}/weight", dependencies=[Depends(jeweler_only), Depends(json_only)])
async def jeweler_weight(body: WeightIn, request: Request, order_id: int = PathParam(..., ge=1)):
    saved = await orders.set_weight(order_id, body.grams, body.note, body.seen, _actor(request))
    return await _order_page(order_id, saved)


class NoteIn(BaseModel):
    text: str = Field(max_length=1000)


@app.post("/api/jeweler/orders/{order_id}/notes", dependencies=[Depends(jeweler_only), Depends(json_only)])
async def jeweler_note(body: NoteIn, request: Request, order_id: int = PathParam(..., ge=1)):
    await orders.add_note(order_id, body.text, _actor(request))
    return await _order_page(order_id)


# ---------- brand designs: other jewellers' designs, listed with our own version (brands.py) ----------

@app.exception_handler(brands.BrandError)
async def brand_error(_request, e: brands.BrandError):
    return JSONResponse({"detail": e.message}, e.status)


class BrandFetchIn(BaseModel):
    url: str = Field(max_length=2500)


@app.exception_handler(webarchive.ArchiveError)
async def archive_error(_request, e: webarchive.ArchiveError):
    return JSONResponse({"detail": e.message}, e.status)


def _archive_info(folder: str, web_id: int | None, status: str = "staged") -> dict:
    return {"folder": folder, "web_id": web_id, "recorded": web_id is not None,
            "where": webarchive.where({"folder": folder, "archive_status": status})}


@app.post("/api/brand-designs/fetch", dependencies=[Depends(jeweler_only), Depends(json_only)])
async def brand_fetch(body: BrandFetchIn, request: Request, background: BackgroundTasks):
    """The team pastes a product link: its details and pictures, and a starting point
    for our version. Every fetched page is kept: its files go to the dataset storage
    (webarchive.py, in the background) and a row to the scraped-designs collection."""
    u = request.state.user
    auth.limiter.check("brandfetch:" + u["uid"], 30, 600)
    res = await run_in_threadpool(brands.fetch_link, body.url, u["uid"], u.get("name") or u.get("email") or "")
    d = res.pop("_draft")
    d.web_id = await webarchive.record(d.original, d.folder, d.pictures, u)
    if d.web_id:
        webarchive.set_record(d.folder, d.web_id)
    background.add_task(webarchive.push_and_mark, d.folder, d.web_id)
    return {**res, "web_id": d.web_id, "archive": _archive_info(d.folder, d.web_id)}


@app.get("/api/web-designs", dependencies=[Depends(jeweler_only)])
async def web_designs(q: str = Query("", max_length=100), limit: int = Query(60, ge=1, le=300)):
    """The collection: every product page the team has fetched."""
    rows = await webarchive.listing(limit, q)
    return {"items": [{**{k: r[k] for k in ("id", "url", "site", "title", "brand", "price", "currency", "folder",
                                           "archive_status", "brand_design_id", "scraped_by_name", "created_at")},
                       "pictures": len(r.get("pictures") or []),
                       "bytes": sum(p.get("bytes") or 0 for p in r.get("pictures") or []),
                       "largest": max((f"{p['width']}×{p['height']}" for p in r.get("pictures") or []),
                                      key=lambda x: int(x.split("×")[0]) * int(x.split("×")[1]), default=None),
                       "where": webarchive.where(r), "preview": f"/api/web-designs/{r['id']}/preview"} for r in rows],
            "storage": storage.get().root_label()}


@app.post("/api/web-designs/{web_id}/open", dependencies=[Depends(jeweler_only)])
async def web_design_open(request: Request, web_id: int = PathParam(..., ge=1)):
    """A design of the collection back in the team's form, without visiting the shop again."""
    row = await webarchive.get(web_id)
    res = await run_in_threadpool(brands.reopen, row, request.state.user["uid"])
    return {**res, "archive": _archive_info(row["folder"], web_id, row.get("archive_status"))}


@app.get("/api/web-designs/{web_id}/preview", dependencies=[Depends(jeweler_only)])
async def web_design_preview(web_id: int = PathParam(..., ge=1)):
    row = await webarchive.get(web_id)
    name = "preview.jpg"
    local = webarchive.local_file(row["folder"], name)
    if local:
        return FileResponse(local, media_type="image/jpeg", headers={"Cache-Control": ORIGINAL_CACHE})
    st = storage.get()
    try:
        url = st.media_url(f"{row['folder']}/{name}")
        if url:
            return RedirectResponse(url, 302, headers={"Cache-Control": "private, max-age=600"})
        return FileResponse(st.local_copy(f"{row['folder']}/{name}"), media_type="image/jpeg",
                            headers={"Cache-Control": ORIGINAL_CACHE})
    except (FileNotFoundError, storage.StorageUnavailable):
        raise HTTPException(404, "Not found")


class BrandCreateIn(BaseModel):
    token: str = Field(max_length=40)
    pictures: list[int] = Field(max_length=brands.MAX_LISTED)   # which fetched pictures to show, in order
    ours: dict


@app.post("/api/brand-designs", dependencies=[Depends(jeweler_only), Depends(json_only)])
async def brand_create(body: BrandCreateIn, request: Request):
    """"Submit and display in catalog": our version is checked and priced, the chosen
    pictures are written, and the page goes to the catalogue with the design in it."""
    u = request.state.user
    draft = brands.take_draft(body.token, u["uid"])
    ours = brands.clean_ours(body.ours)
    picks = list(dict.fromkeys(i for i in body.pictures if 0 <= i < len(draft.pictures)))
    if not picks:
        raise HTTPException(400, "Keep at least one picture.")
    files = await run_in_threadpool(lambda: brands.save_pictures(brands.chosen_pictures(draft, picks)))
    original = {**draft.original, "archive": {"folder": draft.folder, "web_design_id": draft.web_id}}
    try:
        row = await brands.create(original, ours, files, u)
    except Exception:
        brands.remove_pictures(files)
        raise
    brands.drop_draft(body.token)
    if draft.web_id:
        await webarchive.update(draft.web_id, {"brand_design_id": row["id"]})
    return {"id": row["id"], "redirect": "/?" + "&".join((f"browse={ours['category']}", f"new={row['id']}"))}


@app.get("/api/brand-designs")
async def brand_list(category: str | None = Query(None, max_length=20)):
    if category is not None and category not in CATEGORIES:
        raise HTTPException(400, f"category must be one of {CATEGORIES}")
    return {"items": [brands.card(r) for r in await brands.listed(category)]}


@app.get("/api/brand-designs/{design_id}")
async def brand_view(request: Request, design_id: int = PathParam(..., ge=1)):
    staff = bool(await auth.staff_role(request.state.user["uid"]))
    return brands.view(await brands.get(design_id, include_hidden=staff), can_manage=staff)


@app.get("/api/brand-designs/{design_id}/buy")
async def brand_buy(design_id: int = PathParam(..., ge=1)):
    """Our version of the design in the buy page's shape: "Buy with us" is the usual buy flow."""
    return brands.buy_view(await brands.get(design_id))


class BrandOrderIn(BaseModel):
    metal: str = Field(max_length=20)
    purity: str = Field(max_length=5)
    ring_size: int | None = Field(None, ge=0, le=60)
    quantity: int = Field(1, ge=1, le=orders.MAX_QTY)
    phone: str = Field(max_length=30)
    note: str = Field("", max_length=1000)


@app.post("/api/brand-designs/{design_id}/order", dependencies=[Depends(json_only)])
async def brand_order(body: BrandOrderIn, request: Request, design_id: int = PathParam(..., ge=1)):
    u = request.state.user
    auth.limiter.check("order:" + u["uid"], 10, 3600)
    row = brands.build_order(u, await brands.get(design_id), metal=body.metal, purity=body.purity,
                             ring_size=body.ring_size, quantity=body.quantity, phone=body.phone, note=body.note)
    order = await orders.create(row, {"id": u["uid"], "name": u["name"], "role": "customer"})
    return {"id": order["id"], "order_no": orders.order_no(order["id"]), "phone": row["customer_phone"]}


@app.delete("/api/brand-designs/{design_id}", dependencies=[Depends(jeweler_only)])
async def brand_hide(design_id: int = PathParam(..., ge=1)):
    """Off the catalogue (kept for orders that point to it)."""
    await brands.hide(design_id)
    return {"ok": True}


@app.get("/brand-media/{name}")
def brand_media(name: str = PathParam(..., max_length=40)):
    """A brand design's picture, the file as it was downloaded."""
    path = brands.picture_path(name)
    if path is None:
        raise HTTPException(404, "Not found")
    return FileResponse(path, media_type=brands.MEDIA_TYPES[path.suffix[1:]],
                        headers={"Cache-Control": ORIGINAL_CACHE})


# ---------- pages ----------
# no-cache: browsers must revalidate, so UI updates show up without a hard refresh

# ---------- Sketch to Design: sketch / photo / words -> picture by Nano Banana (sketch.py) ----------

@app.exception_handler(sketch.SketchError)
async def sketch_error(_request, e: sketch.SketchError):
    return JSONResponse({"detail": e.message}, e.status)


class SketchIn(BaseModel):
    image: str | None = Field(None, max_length=12_000_000)   # data URL; None = words only
    kind: str = Field("auto", max_length=20)
    background: str = Field("luxury", max_length=20)
    custom_bg: str = Field("", max_length=200)
    notes: str = Field("", max_length=1000)
    sparkle: bool = True
    model: str = Field("", max_length=30)   # "" = the default (first connected free-credit model)


async def _sketch_body(request: Request, cls):
    length = request.headers.get("content-length", "")
    if not length.isdigit() or int(length) > PHOTO_BODY_MAX:
        raise HTTPException(413, "The image is too large. Use one under 8 MB.")
    raw = await request.body()
    if len(raw) > PHOTO_BODY_MAX:
        raise HTTPException(413, "The image is too large. Use one under 8 MB.")
    try:
        return cls.model_validate_json(raw)
    except ValidationError:
        raise HTTPException(422, "Bad request")


def _sketch_prompt(body: SketchIn) -> tuple[str, bytes | None]:
    picture = sketch.prepare(sketch.from_data_url(body.image)) if body.image else None
    if picture is None:
        notes = " ".join(body.notes.split())
        if len(notes) < 3:
            raise sketch.SketchError("Add a sketch or a picture, or describe the design in words.")
        v = engine.domain.check(notes)   # local judge, free: no credit spent on non-jewellery words
        if not v.ok:
            raise sketch.SketchError("Describe a jewellery design (ring, earrings, pendant...).")
    prompt = sketch.build_prompt(body.kind, body.background, body.custom_bg, body.notes, body.sparkle,
                                 has_picture=picture is not None)
    return prompt, picture


def _sketch_model(key: str) -> str:
    key = key or sketch.default_model() or ""
    m = sketch.BY_KEY.get(key)
    if m is None or m.provider not in sketch.connected():
        raise sketch.SketchError("AI picture making is not connected on this server yet." if not sketch.connected()
                                 else "Pick a model from the list.", 503 if not sketch.connected() else 400)
    return key


@app.get("/api/sketch/options")
def sketch_options(request: Request):
    return sketch.options(studio, request.state.user["uid"])


@app.post("/api/sketch/generate", dependencies=[Depends(json_only)])
async def sketch_generate(request: Request):
    body = await _sketch_body(request, SketchIn)
    uid = request.state.user["uid"]
    auth.limiter.check("sketch:" + uid, 12, 600)

    def check():   # quick checks answer at once; the picture itself is made in the background
        model = _sketch_model(body.model)
        prompt, picture = _sketch_prompt(body)
        return model, prompt, picture
    model, prompt, picture = await run_in_threadpool(check)
    return sketch_tasks.start(uid, lambda: {**studio.generate(uid, model, prompt, picture), "budget": studio.public_budget(uid)})


class SketchRefineIn(BaseModel):
    change: str = Field(max_length=400)
    model: str = Field("", max_length=30)   # "" = the default (first connected free-credit model)


@app.post("/api/sketch/{gid}/refine", dependencies=[Depends(json_only)])
async def sketch_refine(request: Request, body: SketchRefineIn, gid: str = PathParam(..., max_length=20)):
    uid = request.state.user["uid"]
    auth.limiter.check("sketch:" + uid, 12, 600)

    model = _sketch_model(body.model)
    sketch.edit_prompt(body.change)   # refuse an empty change at once
    if studio.file(gid, uid) is None:
        raise HTTPException(404, "That design is no longer available.")
    return sketch_tasks.start(uid, lambda: {**studio.refine(uid, gid, body.change, model), "budget": studio.public_budget(uid)})


@app.get("/api/sketch/job/{job_id}")
def sketch_job(request: Request, job_id: str = PathParam(..., max_length=20)):
    j = sketch_tasks.get(job_id, request.state.user["uid"])
    if j is None:
        raise HTTPException(404, "This job has finished long ago or never existed.")
    return j


@app.get("/api/sketch/mine")
def sketch_mine(request: Request, panel: Literal["sketch", "variation"] | None = None):
    uid = request.state.user["uid"]
    return {"items": studio.mine(uid, panel=panel), "budget": studio.public_budget(uid)}


class SketchCellIn(BaseModel):
    index: int = Field(ge=0, le=63)


@app.post("/api/sketch/{gid}/cell", dependencies=[Depends(json_only)])
def sketch_cell(request: Request, body: SketchCellIn, gid: str = PathParam(..., max_length=20)):
    """One variation cut out of a set, kept as its own design (no AI call)."""
    return studio.cell(gid, request.state.user["uid"], body.index)


# ---------- Design Variations: one picture -> a labelled set of variations (variation.py) ----------

class VariationIn(BaseModel):
    image: str = Field(max_length=12_000_000)
    count: int = Field(4, ge=1, le=36)
    direction: str = Field("", max_length=1500)   # the person's own words (optional)
    look: str = Field("illustration", max_length=20)
    model: str = Field("", max_length=30)   # "" = the best connected model for 2x2 tiles


def _job_out(j: dict | None) -> dict:
    if j is None:
        raise HTTPException(404, "This job has finished long ago or never existed.")
    return {k: v for k, v in j.items() if k != "uid"}


@app.get("/api/variation/options")
def variation_options(request: Request):
    return variation.options(studio, request.state.user["uid"])


@app.post("/api/variation/generate", dependencies=[Depends(json_only)])
async def variation_generate(request: Request):
    body = await _sketch_body(request, VariationIn)
    uid = request.state.user["uid"]
    auth.limiter.check("sketch:" + uid, 12, 600)

    def work():
        model = _sketch_model(body.model or variation.default_model() or "")
        picture = sketch.prepare(sketch.from_data_url(body.image))
        return variation_jobs.start(uid, picture, body.count, body.direction, body.look, model, auto=not body.model)
    return _job_out(await run_in_threadpool(work))


@app.get("/api/variation/job/{job_id}")
def variation_job(request: Request, job_id: str = PathParam(..., max_length=20)):
    uid = request.state.user["uid"]
    out = _job_out(variation_jobs.get(job_id, uid))
    out["budget"] = studio.public_budget(uid)
    return out


@app.get("/api/sketch/img/{gid}")
def sketch_img(request: Request, gid: str = PathParam(..., max_length=20), download: int = 0):
    f = studio.file(gid, request.state.user["uid"])
    if f is None:
        raise HTTPException(404, "Not found")
    headers = {"Cache-Control": "private, max-age=604800"}
    if download:
        headers["Content-Disposition"] = f'attachment; filename="design-{gid}{f.suffix}"'
    return FileResponse(f, headers=headers)


@app.delete("/api/sketch/{gid}")
def sketch_delete(request: Request, gid: str = PathParam(..., max_length=20)):
    if not studio.delete(gid, request.state.user["uid"]):
        raise HTTPException(404, "Not found")
    return {"ok": True}


@app.get("/api/sketch/{gid}/similar")
def sketch_similar(request: Request, gid: str = PathParam(..., max_length=20)):
    """Closest designs in our own collection to a made picture (local models, free)."""
    uid = request.state.user["uid"]
    f = studio.file(gid, uid)
    if f is None:
        raise HTTPException(404, "Not found")
    auth.limiter.check("photo:" + uid, 40, 600)
    pq = engine.read_photo(photo.read(f.read_bytes()))
    res = engine.search_photo(pq, "")
    res["token"] = _photo_put(uid, pq)
    return res


@app.get("/login")
def login_page(request: Request, next: str = "/"):
    if auth.read_session(request.cookies.get(auth.COOKIE)):
        return RedirectResponse(auth.safe_next(next), 303)
    return FileResponse(STATIC / "login.html", headers={"Cache-Control": "no-cache"})


@app.get("/admin")
async def admin_page(request: Request):
    if not await auth.is_admin(request.state.user["uid"]):
        return RedirectResponse("/", 303)
    return FileResponse(STATIC / "admin.html", headers={"Cache-Control": "no-cache"})


@app.get("/jeweler")
async def jeweler_page(request: Request):
    if not await auth.staff_role(request.state.user["uid"]):
        return RedirectResponse("/", 303)
    return FileResponse(STATIC / "jeweler.html", headers={"Cache-Control": "no-cache"})


@app.get("/jeweler/jobcard")
async def jobcard_page(request: Request):
    if not await auth.staff_role(request.state.user["uid"]):
        return RedirectResponse("/", 303)
    return FileResponse(STATIC / "jobcard.html", headers={"Cache-Control": "no-cache"})


@app.get("/brand-import")
async def brand_import_page(request: Request):
    if not await auth.staff_role(request.state.user["uid"]):
        return RedirectResponse("/", 303)
    return FileResponse(STATIC / "brand_import.html", headers={"Cache-Control": "no-cache"})


@app.get("/brand-design")
def brand_design_page():
    return FileResponse(STATIC / "brand_design.html", headers={"Cache-Control": "no-cache"})


@app.get("/buy")
def buy_page():
    return FileResponse(STATIC / "buy.html", headers={"Cache-Control": "no-cache"})


@app.get("/demo/look")
def demo_look_page():
    """3D look demo: 6 designs as catalogue photo, today's 3D and the new look (not used by live pages)."""
    return FileResponse(STATIC / "demo" / "look.html", headers={"Cache-Control": "no-cache"})


@app.get("/sketch")
def sketch_page():
    return FileResponse(STATIC / "sketch.html", headers={"Cache-Control": "no-cache"})


@app.get("/variation")
def variation_page():
    return FileResponse(STATIC / "variation.html", headers={"Cache-Control": "no-cache"})


@app.get("/tryon/me")
def body_page():
    return FileResponse(STATIC / "body.html", headers={"Cache-Control": "no-cache"})


@app.get("/tryon")
def tryon_page():
    return FileResponse(STATIC / "tryon.html", headers={"Cache-Control": "no-cache"})


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-cache"})
