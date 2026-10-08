"""Suggested prompts under the search bar (jewelsearch/recommend.py).

Runs against the built index with a scripted jewellery judge. How often the
suggestions anticipate the next search: scripts/eval_prompt_suggestions.py.
Run: .venv/bin/python -m pytest tests/test_recommend.py -q
"""
import re
from datetime import datetime, timedelta, timezone

import pytest

from jewelsearch import recommend
from jewelsearch.config import INDEX
from jewelsearch.query import parse
from jewelsearch.search import TOP_K

needs_index = pytest.mark.skipif(not (INDEX / "front.npy").exists(), reason="index not built")
pytestmark = needs_index
NOW = datetime.now(timezone.utc)
LATIN = re.compile(r"[A-Za-z]")


@pytest.fixture(scope="module")
def engine():
    from jewelsearch.search import SearchEngine
    return SearchEngine(judge=lambda t: "no" if "pizza" in t else "yes")


@pytest.fixture(scope="module")
def rec(engine):
    return recommend.Recommender(engine)


def hist(*queries, refused=()):
    """Search history rows, newest first, one a day."""
    return [{"id": len(queries) - i, "query": q, "understood": {"refused": {"reason": "x"}} if q in refused else {},
             "created_at": (NOW - timedelta(days=i)).isoformat()} for i, q in enumerate(queries)]


def others(*specs, users=("o1", "o2")):
    """Other shoppers' searches as the server reads them: meaning only."""
    return [{"user_id": u, "category": c, "metal": m, "intents": list(i), "shape": None, "refused": None,
             "created_at": (NOW - timedelta(days=1)).isoformat()} for u in users for c, m, i in specs]


def texts(res):
    return [i["text"] for i in res["items"]]


def test_new_shopper_gets_every_type_once(rec):
    res = rec([], [], others(("earrings", None, ["stud"])), "me")
    assert len(res["items"]) == 5 and not res["personal"]
    assert {i["kind"] for i in res["items"]} <= {"popular", "explore"}
    assert len({parse(t).categories[0] for t in texts(res)}) == 5
    assert "stud earrings" in texts(res)


def test_every_suggestion_fills_a_page_and_is_jewellery(engine, rec):
    cases = [([], []), (hist("rose gold ring for my mother", "thin rose gold ring", "oval ring"), []),
             (hist("નાની બુટ્ટી", "ફૂલવાળી કાનની બુટ્ટી"), []), (hist("statement necklace for wedding"), [])]
    for h, f in cases:
        for it in rec(h, f, others(("pendant", "yellow_gold", ["heart"])), "me")["items"]:
            assert engine.domain.check(it["text"]).ok, it
            if it["kind"] != "recent":
                q = parse(it["text"])
                relaxed = []
                mask, _ = engine._filters(q, relaxed)
                assert mask.sum() >= TOP_K and not relaxed, it


def test_latest_search_comes_first_then_new_ideas_and_discovery(rec):
    res = rec(hist("heart pendants", "yellow gold minimal rings", "yellow gold minimal rings",
                   "yellow gold minimal rings", "yellow gold floral rings"), [], others(), "me")
    items = res["items"]
    assert items[0] == {**items[0], "text": "heart pendants", "kind": "recent"}
    assert sum(i["kind"] == "recent" for i in items) <= recommend.MAX_RECENT_REPEATER
    assert any(i["kind"] == "for_you" for i in items)
    assert any(i["kind"] in ("popular", "explore") for i in items)


def test_how_many_to_run_again_follows_the_habit(rec):
    explorer = rec(hist("rose gold ring", "solitaire ring", "thin ring", "floral earrings", "heart pendants"), [], [], "me")
    repeater = rec(hist("rose gold ring", "solitaire ring", "rose gold ring", "solitaire ring", "rose gold ring",
                        "thin ring"), [], [], "me")
    assert sum(i["kind"] == "recent" for i in explorer["items"]) == recommend.MAX_RECENT
    assert sum(i["kind"] == "recent" for i in repeater["items"]) == recommend.MAX_RECENT_REPEATER


def test_refused_and_non_jewellery_searches_are_never_suggested(rec):
    res = rec(hist("boxing ring", "pizza", "rose gold ring", refused=("boxing ring",)), [], [], "me")
    assert not [t for t in texts(res) if "boxing" in t or "pizza" in t]


def test_no_two_suggestions_mean_the_same(rec):
    res = rec(hist("rose gold rings", "rose gold ring", "ROSE GOLD RINGS", "rose gold solitaire ring"), [], [], "me")
    meanings = [(tuple(parse(t).categories), parse(t).metal, frozenset(set(parse(t).intents) - recommend.NOT_STYLE))
                for t in texts(res)]
    assert len(meanings) == len(set(meanings))


def test_suggestions_are_written_in_the_shoppers_script(rec):
    for h, script in ((hist("નાની બુટ્ટી", "ફૂલવાળી કાનની બુટ્ટી", "રોઝ ગોલ્ડ વીંટી"), "gu"),
                      (hist("छोटी बाली", "रोज़ गोल्ड अंगूठी"), "hi")):
        res = rec(h, [], others(("pendant", "yellow_gold", ["heart"])), "me")
        assert res["lang"] == script
        assert all(not LATIN.search(t) for t in texts(res)), texts(res)


def test_popular_needs_two_other_people_and_never_shows_their_words(rec):
    secret = "ring for my sister Priya's engagement"
    rows = others(("bracelet", None, ["cuff"]), users=("o1", "o2")) + others(("necklace", None, ["choker"]), users=("o3",))
    rows += others(("earrings", None, ["hoop"]), users=("me", "o4"))     # the shopper's own search doesn't count
    for r in rows:
        r["query"] = secret                                             # even if the words were there
    res = rec(hist("rose gold ring"), [], rows, "me")
    popular = [i["text"] for i in res["items"] if i["kind"] == "popular"]
    assert popular in ([], ["cuff bracelets"])
    assert all("Priya" not in t for t in texts(res))
    fresh = rec([], [], rows, "someone else")
    assert "cuff bracelets" in texts(fresh)
    assert "choker necklaces" not in [i["text"] for i in fresh["items"] if i["kind"] == "popular"]
    # "me" and o4 searched hoops: popular for someone else (two other people), not for "me" (only o4)
    assert "hoop earrings" in [i["text"] for i in fresh["items"] if i["kind"] == "popular"]


def test_favourites_shape_the_suggestions(engine, rec):
    uid = next(u for u, m in engine.by_uid.items() if m["category"] == "bracelet")
    res = rec([], [{"design_uid": uid, "design_id": engine.by_uid[uid]["design_id"], "created_at": NOW.isoformat()}],
              [], "me")
    assert res["personal"]
    assert "bracelet" in parse(texts(res)[0]).categories or any("bracelet" in parse(t).categories for t in texts(res)[:2])


@pytest.mark.parametrize("lang", ["en", "gu", "hi"])
def test_written_prompts_mean_what_they_were_made_from(rec, lang):
    for spec in (recommend.Spec("ring", "rose_gold", ("solitaire",)), recommend.Spec("earrings", None, ("floral",)),
                 recommend.Spec("pendant", "yellow_gold", ("heart",)), recommend.Spec("necklace", None, ("bridal",))):
        text = rec._render(spec, lang)
        if text is None:
            continue
        q = parse(text)
        assert q.categories == [spec.cat] and q.metal == spec.metal and set(spec.intents) <= set(q.intents), (lang, text)
    assert rec._render(recommend.Spec("ring", "rose_gold", ("solitaire",)), "en") == "rose gold solitaire rings"
    assert rec._render(recommend.Spec("pendant", "yellow_gold", ("heart",)), "gu") == "યલો ગોલ્ડ દિલ પેન્ડન્ટ"


# ---- the API route and the database query (no index needed for these two) ----------

def test_everyones_searches_are_read_as_meaning_only(monkeypatch):
    import asyncio

    from jewelsearch import history
    seen = {}

    async def fake_checked(method, params, **kw):
        seen.update(params)

        class R:
            def json(self):
                return []
        return R()
    monkeypatch.setattr(history, "_checked", fake_checked)
    asyncio.run(history.recent_meanings(30))
    assert "query" not in seen["select"] and "raw" not in seen["select"]
    assert "understood->>category" in seen["select"] and seen["created_at"].startswith("gte.")


def test_prompt_suggestions_route(monkeypatch):
    from fastapi.testclient import TestClient

    from jewelsearch import auth, favorites, history, server
    monkeypatch.setattr(auth, "read_session", lambda c: {"uid": "u-1", "name": "A", "email": "a@x.in"} if c == "ok" else None)

    async def yes(uid, fresh=False):
        return True
    monkeypatch.setattr(auth, "is_approved", yes)

    async def rows(uid, limit):
        return [{"id": 7, "query": "rose gold ring", "understood": {}, "created_at": NOW.isoformat()}]

    async def favs(uid):
        return []

    async def meanings(days, limit=5000):
        return []
    monkeypatch.setattr(history, "list_for", rows)
    monkeypatch.setattr(favorites, "list_for", favs)
    monkeypatch.setattr(history, "recent_meanings", meanings)
    calls = []
    monkeypatch.setattr(server, "recommender", lambda h, f, p, uid: calls.append((h, f, p, uid)) or {
        "items": [{"text": "rose gold ring", "kind": "recent", "why": "Your latest search", "count": None}],
        "personal": True, "lang": "en"})
    monkeypatch.setattr(server, "_prompt_cache", type(server._prompt_cache)())
    c = TestClient(server.app)
    assert c.get("/api/prompt-suggestions").status_code == 401
    c.cookies.set(auth.COOKIE, "ok")
    r = c.get("/api/prompt-suggestions")
    assert r.status_code == 200 and r.json()["items"][0]["kind"] == "recent"
    assert r.headers["cache-control"] == "no-store"
    c.get("/api/prompt-suggestions")
    assert len(calls) == 1 and calls[0][3] == "u-1"   # the second call came from the cache
