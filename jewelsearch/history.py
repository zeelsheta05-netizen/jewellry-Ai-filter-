"""Per-user search history, stored in Supabase (public.search_history).

Only the server touches the table, and every query is filtered by the user id
from the signed session cookie, never by anything the browser sends.
"""
import logging
from datetime import datetime, timezone

from .auth import AuthError, _call

log = logging.getLogger(__name__)
TABLE = "/rest/v1/search_history"
LIST_FIELDS = "id,query,category_override,metal_override,understood,matches,result_uids,created_at"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


async def _checked(method: str, params: dict, **kw):
    r = await _call(method, TABLE, params=params, **kw)
    if r.status_code == 404:
        raise AuthError(503, "Search history isn't set up yet (run supabase/search_history.sql).")
    if r.status_code >= 300:
        raise AuthError(502, "Search history is unavailable right now.")
    return r


async def record(uid: str, query: str, category: str | None, metal: str | None, res: dict):
    """Save a search. One entry per prompt: running the same prompt again (for
    example with another metal picked) updates that entry and moves it to the top."""
    query = " ".join(query.split())
    row = {
        "category_override": category, "metal_override": metal,
        "understood": res["query"], "notes": res["notes"], "matches": res["matches_in_filter"],
        "result_uids": [c["uid"] for c in res["results"]],
    }
    try:
        r = await _checked("GET", {"user_id": f"eq.{uid}", "query": f"eq.{query}", "select": "id",
                                   "order": "created_at.desc"})
        ids = [x["id"] for x in r.json()]
        if ids:
            await _checked("PATCH", {"id": f"eq.{ids[0]}", "user_id": f"eq.{uid}"},
                           json={**row, "created_at": _now()}, headers={"Prefer": "return=minimal"})
            if ids[1:]:   # tidy up any older copies of the same prompt
                await _checked("DELETE", {"id": f"in.({','.join(map(str, ids[1:]))})", "user_id": f"eq.{uid}"},
                               headers={"Prefer": "return=minimal"})
        else:
            await _checked("POST", {}, json={**row, "user_id": uid, "query": query},
                           headers={"Prefer": "return=minimal"})
    except AuthError as e:   # history must never break search
        log.warning("search history not saved: %s", e.message)


# Other people's searches are read only as their parsed meaning (type, metal, styles, cut):
# the words they typed are never selected, so they can't leak into anyone's suggestions.
MEANING_FIELDS = ("user_id,created_at,category:understood->>category,metal:understood->>metal,"
                  "intents:understood->intents,shape:understood->>shape,refused:understood->refused")


async def recent_meanings(days: int, limit: int = 5000) -> list[dict]:
    """Every shopper's searches of the last `days` days, as meaning only (for "popular" suggestions)."""
    since = datetime.fromtimestamp(datetime.now(timezone.utc).timestamp() - days * 86400, timezone.utc).isoformat()
    r = await _checked("GET", {"created_at": f"gte.{since}", "select": MEANING_FIELDS,
                               "order": "created_at.desc", "limit": str(limit)})
    return r.json()


async def list_for(uid: str, limit: int) -> list[dict]:
    r = await _checked("GET", {"user_id": f"eq.{uid}", "select": LIST_FIELDS,
                               "order": "created_at.desc", "limit": str(limit)})
    return r.json()


async def get(uid: str, entry_id: int) -> dict | None:
    r = await _checked("GET", {"id": f"eq.{entry_id}", "user_id": f"eq.{uid}", "select": LIST_FIELDS + ",notes"})
    rows = r.json()
    return rows[0] if rows else None


async def delete(uid: str, entry_id: int | None = None):
    params = {"user_id": f"eq.{uid}"}
    if entry_id is not None:
        params["id"] = f"eq.{entry_id}"
    await _checked("DELETE", params, headers={"Prefer": "return=minimal"})
