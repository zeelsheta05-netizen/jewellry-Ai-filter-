import time

import pytest

from jewelsearch import auth

USER = {"id": "0f8fad5b-d9cb-469f-a165-70867728950e", "email": "a@b.co", "name": "Asha"}


def test_session_round_trip():
    s = auth.read_session(auth.make_session(USER))
    assert s["uid"] == USER["id"] and s["name"] == "Asha"


def test_tampered_session_rejected():
    token = auth.make_session(USER)
    body, sig = token.split(".")
    forged = auth._b64(auth._unb64(body).replace(b"Asha", b"Evil"))
    assert auth.read_session(f"{forged}.{sig}") is None
    assert auth.read_session(token[:-2] + "xx") is None
    assert auth.read_session("garbage") is None
    assert auth.read_session(None) is None


def test_expired_session_rejected():
    token = auth.make_session(USER, now=time.time() - auth.SESSION_SECONDS - 5)
    assert auth.read_session(token) is None


def test_rate_limiter_blocks_after_limit():
    rl = auth.RateLimiter()
    for i in range(3):
        rl.check("k", 3, 60, now=100 + i)
    with pytest.raises(auth.AuthError) as e:
        rl.check("k", 3, 60, now=104)
    assert e.value.status == 429
    rl.check("k", 3, 60, now=200)   # window passed


@pytest.mark.parametrize("target,expected", [
    ("/?q=ring", "/?q=ring"), ("//evil.com", "/"), ("https://evil.com", "/"),
    ("/\\evil.com", "/"), ("", "/"), (None, "/"),
])
def test_safe_next(target, expected):
    assert auth.safe_next(target) == expected


def test_input_checks():
    assert auth.clean_email("  Asha@Example.COM ") == "asha@example.com"
    for bad in ("nope", "a@b", "a b@c.de"):
        with pytest.raises(auth.AuthError):
            auth.clean_email(bad)
    for bad in ("short1", "lettersonly", "12345678"):
        with pytest.raises(auth.AuthError):
            auth.check_password(bad)
    auth.check_password("jewel2026")
    assert auth.clean_name("  Asha   Patel ") == "Asha Patel"


def test_oauth_blob_round_trip_and_tamper():
    token = auth.sign_blob({"v": "verifier", "next": "/?q=ring"}, 600)
    assert auth.read_blob(token)["v"] == "verifier"
    body, sig = token.split(".")
    assert auth.read_blob(body + "." + sig[:-2] + "xx") is None
    assert auth.read_blob(auth.sign_blob({"v": "x"}, -1)) is None   # expired


def test_pkce_challenge_matches_verifier():
    import base64, hashlib
    v, c = auth.pkce_pair()
    assert 43 <= len(v) <= 128
    assert c == base64.urlsafe_b64encode(hashlib.sha256(v.encode()).digest()).rstrip(b"=").decode()
