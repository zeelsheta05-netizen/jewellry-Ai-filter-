"""Per-user favourite designs, stored in Supabase (public.favorites).

Server-only, like history: every read and write is filtered by the user id
from the signed session cookie.
"""
from .auth import AuthError, _call

TABLE = "/rest/v1/favorites"


async def _checked(method: str, params: dict, **kw):
    r = await _call(method, TABLE, params=params, **kw)
    if r.status_code == 404:
        raise AuthError(503, "Favourites aren't set up yet (run supabase/favorites.sql).")
    if r.status_code >= 300:
        raise AuthError(502, "Favourites are unavailable right now.")
    return r


async def list_for(uid: str) -> list[dict]:
    r = await _checked("GET", {"user_id": f"eq.{uid}", "select": "design_uid,design_id,created_at",
                               "order": "created_at.desc", "limit": "1000"})
    return r.json()


async def add(uid: str, design_uid: int, design_id: str):
    await _checked("POST", {}, json={"user_id": uid, "design_uid": design_uid, "design_id": design_id},
                   headers={"Prefer": "resolution=ignore-duplicates,return=minimal"})


async def remove(uid: str, design_uid: int):
    await _checked("DELETE", {"user_id": f"eq.{uid}", "design_uid": f"eq.{design_uid}"},
                   headers={"Prefer": "return=minimal"})
