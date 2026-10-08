"""Manual search: picked properties become the prompt the AI search runs
(jewelsearch/manual.py).

The wording is checked without the index; the numbers and the search run
against the built index (slow: loads the model).
Run: .venv/bin/python -m pytest tests/test_manual.py -q
"""
import itertools

import pytest

from jewelsearch import manual as M
from jewelsearch.config import INDEX
from jewelsearch.query import parse

needs_index = pytest.mark.skipif(not (INDEX / "front.npy").exists(), reason="index not built")


def with_type(key):
    o = M.BY_KEY[key]
    return [f"type:{o.types[0]}", key] if o.types else [key]


def has(q, key) -> bool:
    o = M.BY_KEY[key]
    return {"type": lambda: q.categories == [o.value], "metal": lambda: q.metal == o.value,
            "cut": lambda: q.shape == o.value}.get(o.group, lambda: o.value in q.intents)()


# ---- the wording (no index needed) -------------------------------------------------

@pytest.mark.parametrize("key", list(M.BY_KEY))
def test_every_option_is_understood_exactly(key):
    text = M.compose(with_type(key))
    q = parse(text)
    assert has(q, key) and not q.free_text(), (key, text)


def test_every_pair_of_options_is_understood_exactly():
    """Any two compatible picks from different groups: both are read, nothing is left over."""
    checked = 0
    for a, b in itertools.combinations(M.BY_KEY, 2):
        picks = list(dict.fromkeys(with_type(a) + with_type(b)))
        if M.BY_KEY[a].group == M.BY_KEY[b].group or set(M.clean(picks)) != set(picks):
            continue   # same group, two different types, or picks that contradict each other
        q = parse(M.compose(picks))
        assert has(q, a) and has(q, b) and not q.free_text(), (picks, M.compose(picks))
        checked += 1
    assert checked > 2000


def test_the_prompt_reads_naturally():
    picks = ["metal:rose_gold", "type:ring", "stones:solitaire", "cut:oval", "shape:thin", "wearer:women"]
    assert M.compose(picks) == "rose gold oval solitaire thin rings for women"
    assert M.compose(["type:bracelet", "shape:flexible", "setting:tennis"]) == "tennis chain bracelet"
    assert M.compose(["style:floral"]) == "floral jewellery"
    assert M.compose([]) == ""


def test_picks_are_cleaned():
    # unknown keys and junk are ignored; one pick per group, the last one wins
    assert M.clean(["metal:gold", 7, "type:ring", "type:pendant", "metal:rose_gold"]) == ["type:pendant", "metal:rose_gold"]
    # a shape of another type is dropped
    assert M.clean(["type:pendant", "shape:stud"]) == ["type:pendant"]
    # settings and cuts can't go with "no stones"; many-diamond settings can't go with a solitaire
    assert M.clean(["stones:plain", "setting:halo", "cut:oval"]) == ["stones:plain"]
    assert M.clean(["stones:solitaire", "setting:halo", "setting:bezel"]) == ["stones:solitaire", "setting:bezel"]
    assert M.clean(["x"] * 100 + ["type:ring"]) == []   # at most MAX_PICKS are read


def test_how_each_pick_acts():
    assert M.KIND["type:ring"] == M.KIND["metal:rose_gold"] == M.KIND["wearer:men"] == "filter"
    assert M.KIND["stones:plain"] == M.KIND["stones:solitaire"] == M.KIND["shape:stud"] == "filter"
    assert M.KIND["stones:side_stones"] == M.KIND["stones:cluster"] == "filter"   # the trained diamond reader
    assert M.KIND["cut:oval"] == "cut"
    assert M.KIND["setting:halo"] == M.KIND["style:floral"] == M.KIND["finish:brushed"] == "ai"


# ---- numbers and search (index needed) ---------------------------------------------

@pytest.fixture(scope="module")
def engine():
    from jewelsearch.search import SearchEngine
    return SearchEngine(judge=lambda t: "yes")


@pytest.fixture(scope="module")
def panel(engine):
    return M.Manual(engine)


def options(res):
    return {o["key"]: o for g in res["groups"] for o in g["options"]}


@needs_index
def test_numbers_are_strict_counts(engine, panel):
    res = panel(["type:ring", "metal:rose_gold"])
    assert res["prompt"] == "rose gold rings"
    assert res["count"] == int(engine.strict_mask(parse("rose gold rings")).sum())
    opts = options(res)
    assert opts["type:ring"]["picked"] and opts["type:ring"]["count"] == res["count"]
    assert opts["type:earrings"]["count"] == int(engine.strict_mask(parse("rose gold earrings")).sum())
    assert opts["shape:thin"]["count"] == int(engine.strict_mask(parse("rose gold thin rings")).sum())
    assert opts["cut:oval"]["count"] == int((engine.strict_mask(parse("rose gold rings")) & (engine.known_cut == "oval")).sum())
    assert opts["setting:halo"]["count"] is None and opts["style:floral"]["count"] is None   # the AI ranks by these


@needs_index
def test_strict_counts_never_loosen_a_filter(engine):
    """'no stones' leaves only a few designs; the count says so instead of loosening it."""
    q = parse("thin rings without stones")
    relaxed = []
    loose, _ = engine._filters(q, relaxed)
    strict = engine.strict_mask(q)
    assert relaxed and strict.sum() < loose.sum() and not (strict & ~loose).any()


@needs_index
def test_options_follow_the_type(panel):
    assert not [k for k in options(panel([])) if k.startswith("shape:")]   # shapes come with a type
    rings = options(panel(["type:ring"]))
    assert "shape:thin" in rings and "shape:stud" not in rings
    assert {k for k in options(panel(["type:earrings"])) if k.startswith("shape:")} >= {"shape:stud", "shape:hoop"}


@needs_index
def test_contradictions_and_dead_ends_are_shown(panel):
    opts = options(panel(["stones:plain"]))
    assert opts["setting:halo"]["blocked"] and opts["cut:oval"]["blocked"]
    men = options(panel(["wearer:men"]))
    assert men["type:earrings"]["count"] == 0          # no men's earrings in the collection
    res = panel(["type:earrings", "wearer:men"])
    assert res["count"] == 0


@needs_index
def test_a_pick_ruled_out_by_later_picks_is_flagged(engine, panel):
    """Oval first, then thin: no thin rose gold solitaire is known to have an oval cut."""
    res = panel(["type:ring", "metal:rose_gold", "stones:solitaire", "cut:oval", "shape:thin"])
    if options(res)["cut:oval"]["count"] == 0:
        assert any("oval cut" in w for w in res["warnings"])
        assert any("oval centre stone" in n for n in engine.search(res["prompt"])["notes"])
    assert panel(["type:ring", "metal:rose_gold"])["warnings"] == []


@needs_index
def test_a_manual_search_returns_what_was_picked(engine, panel):
    picks = ["type:ring", "stones:solitaire", "shape:thin", "metal:white_gold"]
    res = panel(picks)
    found = engine.search(res["prompt"])
    assert len(found["results"]) == 8 and not found.get("refused")
    keep = engine.strict_mask(parse(res["prompt"]))
    assert all(keep[c["uid"]] for c in found["results"])
    q = found["query"]
    assert q["categories"] == ["ring"] and q["metal"] == "white_gold" and {"solitaire", "thin"} <= set(q["intents"])


# ---- the API route (no index needed) ---------------------------------------------

def test_manual_route(monkeypatch):
    from fastapi.testclient import TestClient

    from jewelsearch import auth, server
    monkeypatch.setattr(auth, "read_session", lambda c: {"uid": "u-1", "name": "A", "email": "a@x.in"} if c == "ok" else None)

    async def yes(uid, fresh=False):
        return True
    monkeypatch.setattr(auth, "is_approved", yes)
    calls = []
    monkeypatch.setattr(server, "manual_panel", lambda picks: calls.append(picks) or {"picks": picks, "prompt": "rings"})
    c = TestClient(server.app)
    assert c.post("/api/manual", json={"picks": ["type:ring"]}).status_code == 401
    c.cookies.set(auth.COOKIE, "ok")
    r = c.post("/api/manual", json={"picks": ["type:ring"]})
    assert r.status_code == 200 and r.json()["prompt"] == "rings" and r.headers["cache-control"] == "no-store"
    assert calls == [["type:ring"]]
    assert c.post("/api/manual", json={"picks": ["x" * 41]}).status_code == 422
    assert c.post("/api/manual", json={"picks": ["type:ring"] * (M.MAX_PICKS + 1)}).status_code == 422
    assert c.post("/api/manual", content="picks=type:ring", headers={"Content-Type": "text/plain"}).status_code == 415
