"""Search-box suggestions (jewelsearch/suggest.py).

The word lists are checked without the index; the rest runs against the built
index (slow: loads the model).
Run: .venv/bin/python -m pytest tests/test_suggest.py -q
"""
import re

import pytest

from jewelsearch import suggest
from jewelsearch.config import INDEX
from jewelsearch.query import parse

needs_index = pytest.mark.skipif(not (INDEX / "front.npy").exists(), reason="index not built")
LATIN = re.compile(r"[A-Za-z]")


# ---- word lists (no index needed) ----------------------------------------------

def test_every_completion_is_understood_and_correctly_spelt():
    for lang, phrases in suggest.VOCAB.items():
        assert len(phrases) == len(set(phrases)), lang
        for p in phrases:
            q = parse(p)
            assert suggest._understood(q), (lang, p)
            assert not q.corrections, (lang, p)
            assert suggest.script(p) == lang, (lang, p)


@pytest.mark.parametrize("lang", ["en", "gu", "hi"])
def test_refinement_words_add_what_they_claim(lang):
    for facet, claim, words, only in suggest.REFINEMENTS:
        if lang not in words:
            continue
        noun = suggest.TYPE_WORDS[lang][only or "ring"]
        noun = noun[1] if isinstance(noun, tuple) else noun
        where, w = words[lang]
        text = suggest._place(noun, where, w, lang)
        q = parse(text)
        assert suggest._has(q, claim), (lang, text, claim)
        assert q.categories == [only or "ring"], (lang, text)


@pytest.mark.parametrize("text,words,expected", [
    ("bold rings", "rose gold", "rose gold bold rings"),
    ("a ring for my mother", "rose gold", "a rose gold ring for my mother"),
    ("an earring", "rose gold", "a rose gold earring"),
    ("a ring", "oval", "an oval ring"),
    ("show me bold rings", "rose gold", "show me rose gold bold rings"),
    ("a simple and thin ring", "solitaire", "a simple and thin solitaire ring"),
    ("single stone ring", "rose gold", "rose gold single stone ring"),
    ("floral", "rose gold", "rose gold floral"),
])
def test_words_go_in_front_of_the_piece(text, words, expected):
    assert suggest._insert_pre(text, words, "en") == expected


def test_indian_scripts_put_words_right_before_the_piece():
    assert suggest._insert_pre("વીંટી", "રોઝ ગોલ્ડ", "gu") == "રોઝ ગોલ્ડ વીંટી"
    assert (suggest._insert_pre("મારી બહેન માટે નાની અને સાદી કાનની બુટ્ટી", "રોઝ ગોલ્ડ", "gu")
            == "મારી બહેન માટે નાની અને સાદી રોઝ ગોલ્ડ કાનની બુટ્ટી")
    assert suggest._insert_pre("शादी के लिए भारी हीरे का हार", "रोज़ गोल्ड", "hi") == "शादी के लिए भारी हीरे का रोज़ गोल्ड हार"


def test_type_words_are_placed_naturally():
    assert suggest._add_type("floral", "ring", "en") == "floral rings"
    assert suggest._add_type("for my mother", "pendant", "en") == "pendants for my mother"
    assert suggest._add_type("a floral", "ring", "en") == "a floral ring"
    assert suggest._add_type("ફૂલ", "ring", "gu") == "ફૂલ વીંટી"


# ---- against the index ---------------------------------------------------------

@pytest.fixture(scope="module")
def engine():
    from jewelsearch.search import SearchEngine
    return SearchEngine()


@pytest.fixture(scope="module")
def sugg(engine):
    return suggest.Suggester(engine)


def texts(res):
    return [i["text"] for i in res["items"]]


PROMPTS = ["bold rings", "bold ri", "r", "earrings", "bracelets", "necklace", "pendants", "floral", "for my mother",
           "a ring for my mother", "gents", "oval rings", "વીંટી", "अंगूठी", "अंगू", "anguthi", "rings for "]


@needs_index
def test_count_reports_relaxed_filters(engine):
    n, relaxed = engine.count(parse("rings"))
    assert n == int((engine.cat == "ring").sum()) and not relaxed
    assert engine.count(parse("solitaire necklaces"))[1]      # too few: the strict filter falls back
    assert engine.count(parse("thin rings for men"))[1]


@needs_index
@pytest.mark.parametrize("prompt", PROMPTS)
def test_every_suggestion_fills_a_page_without_relaxing(engine, sugg, prompt):
    for text in texts(sugg(prompt)):
        n, relaxed = engine.count(parse(text))
        assert n >= 8 and not relaxed, text
        res = engine.search(text)
        assert len(res["results"]) == 8, text
        assert not [x for x in res["notes"] if "Fewer" in x or "Only" in x or "No " in x or "no men" in x], text


@needs_index
@pytest.mark.parametrize("prompt", ["bold rings", "earrings", "a ring for my mother", "વીંટી", "अंगूठी", "bracelets"])
def test_refinements_keep_everything_asked_for(sugg, prompt):
    base = parse(prompt)
    for item in sugg(prompt)["items"]:
        assert suggest._keeps(base, parse(item["text"])), item["text"]
        assert suggest._sig(parse(item["text"])) != suggest._sig(base), item["text"]


@needs_index
def test_for_men_only_where_mens_designs_exist(engine, sugg):
    assert not [t for t in texts(sugg("earrings")) if "men" in t.split()]
    assert not [t for t in texts(sugg("pendants")) if "men" in t.split()]
    rows = {i["text"]: i for i in sugg("bold rings")["items"]}
    assert rows["bold rings for men"]["count"] == int(((engine.cat == "ring") & engine.mens).sum())


@needs_index
def test_no_stones_only_where_enough_plain_designs_exist(sugg):
    for prompt in ("bracelets", "bold rings", "earrings"):
        assert not [t for t in texts(sugg(prompt)) if "without stones" in t]


@needs_index
def test_counts_show_only_when_the_suggestion_narrows(sugg):
    rows = {i["text"]: i for i in sugg("bold rings")["items"]}
    assert rows["rose gold bold rings"]["count"] is None        # nearly every ring comes in rose gold
    assert rows["thin bold rings"]["count"] < 3338 * suggest.COUNT_SHOWN_BELOW


@needs_index
def test_completes_the_word_being_typed(sugg):
    first = sugg("bold ri")["items"][0]
    assert first == {"text": "bold rings", "count": first["count"], "kind": "complete"}
    assert texts(sugg("a ri"))[0] == "a ring"
    assert texts(sugg("rose g"))[0] == "rose gold"
    assert texts(sugg("વીં"))[0] == "વીંટી"
    assert texts(sugg("rings for "))[:2] == ["rings for women", "rings for men"]
    # "hoop" alone already means hoop earrings: the half-typed word is finished all the same
    assert texts(sugg("hoop ear"))[0] == "hoop earrings"
    assert texts(sugg("stud ear"))[0] == "stud earrings"
    assert "rings with" not in texts(sugg("rings wi"))   # a finished word that names nothing isn't offered


@needs_index
def test_a_finished_word_is_not_completed_again(sugg):
    assert "single stone rings" not in texts(sugg("single stone ring"))
    assert not [i for i in sugg("necklace")["items"] if i["kind"] == "complete"]


@needs_index
def test_unfinished_word_with_no_good_ending_gives_nothing(sugg):
    assert sugg("gents ear")["items"] == []     # men's earrings don't exist
    assert sugg("rings go")["items"] == []
    assert sugg("hello")["items"] == []


@needs_index
def test_no_type_named_offers_the_types(engine, sugg):
    rows = sugg("floral")["items"]
    assert texts({"items": rows}) == ["floral rings", "floral earrings", "floral pendants", "floral bracelets",
                                      "floral necklaces"]
    assert rows[0]["count"] == int((engine.cat == "ring").sum())
    assert texts(sugg("gents")) == ["gents rings", "gents bracelets"]


@needs_index
def test_stone_cut_only_where_designs_have_it(sugg):
    assert texts(sugg("oval")) == ["oval rings"]
    assert all(i["count"] is None for i in sugg("oval rings")["items"])


@needs_index
@pytest.mark.parametrize("prompt,lang,category", [("વીંટી", "gu", "ring"), ("अंगूठी", "hi", "ring"),
                                                  ("મારી બહેન માટે નાની અને સાદી કાનની બુટ્ટી", "gu", "earrings")])
def test_suggestions_stay_in_the_shoppers_script(sugg, prompt, lang, category):
    items = texts(sugg(prompt))
    assert len(items) >= 4
    for t in items:
        assert not LATIN.search(t) and suggest.script(t) == lang, t
        assert parse(t).categories == [category], t


@needs_index
def test_metal_goes_before_the_piece_and_fixes_the_article(sugg):
    assert "a rose gold ring for my mother" in texts(sugg("a ring for my mother"))
    assert "a rose gold earring" in texts(sugg("an earring"))
    assert not [t for t in texts(sugg("plain gold")) if "rose gold" in t]


@needs_index
def test_spelling_is_fixed_in_suggestions(sugg):
    first = sugg("bracelate for party")["items"][0]
    assert first["text"] == "bracelet for party" and first["kind"] == "fix"
    assert all("bracelate" not in t for t in texts(sugg("bracelate for party")))


@needs_index
def test_photo_suggestions_have_no_counts_and_no_types(sugg):
    items = sugg("rose gold", photo=True)["items"]
    assert items and all(i["count"] is None for i in items)
    assert not [i for i in items if parse(i["text"]).categories]


@needs_index
def test_photo_suggestions_are_checked_against_the_photos_type(engine, sugg):
    texts_ = texts(sugg("rose gold", photo=True, category="earrings"))
    assert "rose gold for men" not in texts_          # no men's earrings
    assert "stud rose gold" in texts_ or "rose gold stud" in texts_ or any("stud" in t for t in texts_)
    for t in texts_:
        q = parse(t)
        q.categories, q.category = q.categories or ["earrings"], q.category or "earrings"
        n, relaxed = engine.count(q)
        assert n >= 8 and not relaxed, t
    assert sugg("rose gold", photo=False, category="earrings") == sugg("rose gold")   # the type only applies to a photo


@needs_index
def test_empty_or_long_text_gets_nothing(sugg):
    assert sugg("")["items"] == [] and sugg("   ")["items"] == []
    assert sugg("rings " * 40)["items"] == []


# ---- the API route (no index needed) ----------------------------------------------

@pytest.fixture
def client(monkeypatch):
    from fastapi.testclient import TestClient

    from jewelsearch import auth, server
    monkeypatch.setattr(auth, "read_session", lambda cookie: {"uid": "u-1", "name": "A", "email": "a@x.in"}
                        if cookie == "ok" else None)

    async def approved(uid, fresh=False):
        return True
    monkeypatch.setattr(auth, "is_approved", approved)
    calls = []

    def fake(q, photo=False, category=None):
        calls.append((q, photo, category))
        return {"items": [{"text": q + " for men", "count": 87, "kind": "refine"}]}
    monkeypatch.setattr(server, "suggester", fake)
    c = TestClient(server.app)
    c.calls = calls
    return c


def test_suggest_route_needs_sign_in_and_is_not_cached(client):
    from jewelsearch import auth
    assert client.get("/api/suggest", params={"q": "rings"}).status_code == 401
    client.cookies.set(auth.COOKIE, "ok")
    r = client.get("/api/suggest", params={"q": "rings", "photo": "1", "category": "earrings"})
    assert r.status_code == 200
    assert r.json()["items"][0]["text"] == "rings for men"
    assert r.headers["cache-control"] == "no-store"
    assert client.calls == [("rings", True, "earrings")]
    assert client.get("/api/suggest", params={"q": "rings", "category": "shoes"}).status_code == 400
    assert client.get("/api/suggest", params={"q": "x" * 501}).status_code == 422
