"""End-to-end ranking tests against the built index (slow: loads the model).

Run: .venv/bin/python -m pytest tests/test_search.py -q
"""
import pytest

from jewelsearch.config import INDEX

pytestmark = pytest.mark.skipif(not (INDEX / "front.npy").exists(), reason="index not built")


@pytest.fixture(scope="module")
def engine():
    from jewelsearch.search import SearchEngine
    return SearchEngine()


def attr(engine, card, a, c):
    return engine.by_uid[card["uid"]].get("attrs", {}).get(a, {}).get(c, 0.0)


@pytest.mark.parametrize("prompt,category", [
    ("bracelate for party", "bracelet"),
    ("tennis bracelet with diamonds", "bracelet"),
    ("bangle for my wife", "bracelet"),
    ("braclet", "bracelet"),
    ("neckless for wedding", "necklace"),
    ("small pendent with heart", "pendant"),
    ("earings for office", "earrings"),
    ("rign for daily wear", "ring"),
    ("મારી બહેન માટે નાની અને સાદી કાનની બુટ્ટી", "earrings"),
    ("शादी के लिए भारी हीरे का हार", "necklace"),
    ("ચાંદી જેવી બંગડી", "bracelet"),
])
def test_category_is_strict(engine, prompt, category):
    res = engine.search(prompt)
    assert len(res["results"]) == 8
    assert {c["category"] for c in res["results"]} == {category}


@pytest.mark.parametrize("prompt,metal", [
    ("rose gold ring for my mother", "rose_gold"),
    ("yellow gold earrings", "yellow_gold"),
    ("silver pendant", "white_gold"),
])
def test_metal_filter(engine, prompt, metal):
    res = engine.search(prompt)
    assert all(metal in c["metals"] and c["metal_shown"] == metal for c in res["results"])


def test_client_prompt_solitaire_without_side_stones(engine):
    res = engine.search("silver ring + single dimond on center (no more dimonds on ring)"
                        "and simple without dimond thin width")
    cards = res["results"]
    assert {c["category"] for c in cards} == {"ring"}
    assert sum(attr(engine, c, "stones", "solitaire") >= 0.3 for c in cards) >= 6
    assert all(attr(engine, c, "stones", "accented") + attr(engine, c, "stones", "pave") <= 0.6 for c in cards)
    assert any("silver" in n for n in res["notes"])


def test_several_categories(engine):
    res = engine.search("gift for my sister, pendant or earrings")
    assert {c["category"] for c in res["results"]} <= {"pendant", "earrings"}


def test_mens_ring_is_mens(engine):
    res = engine.search("gents ring bina stone")
    assert all(engine.mens[c["uid"]] for c in res["results"])


def test_womens_ring_excludes_mens(engine):
    res = engine.search("ring for my wife")
    assert not any(engine.mens[c["uid"]] for c in res["results"])


def test_excluded_metal(engine):
    res = engine.search("ring, not yellow gold")
    assert all(set(c["metals"]) - {"yellow_gold"} for c in res["results"])


def test_similar_keeps_category(engine):
    first = engine.search("tennis bracelet")["results"][0]
    res = engine.similar(first["uid"])
    assert {c["category"] for c in res["results"]} == {"bracelet"}
    assert first["uid"] not in {c["uid"] for c in res["results"]}


def test_results_are_distinct_designs(engine):
    res = engine.search("oval diamond engagement ring")
    fams = [engine.by_uid[c["uid"]]["family"] for c in res["results"]]
    assert len(set(fams)) == len(fams)


@pytest.mark.parametrize("prompt,form", [
    ("Hoop earrings in white gold", "hoop"), ("कुंडल", "hoop"), ("stud earrings", "stud"),
    ("drop earrings", "drop"), ("bangle in rose gold", "bangle"), ("cuff bracelet", "cuff"),
])
def test_forms(engine, prompt, form):
    res = engine.search(prompt)
    p = engine.attr[("form", form)]
    assert sum(p[c["uid"]] >= 0.5 for c in res["results"]) >= 7


def test_missing_jhumka_is_explained(engine):
    res = engine.search("bridal jhumka heavy wala")
    assert any("jhumka" in n for n in res["notes"])


def test_next_page_has_new_designs(engine):
    first = engine.search("simple earrings for office")
    second = engine.search("simple earrings for office", page=1)
    assert len(second["results"]) == 8 and first["has_more"]
    assert not {c["uid"] for c in first["results"]} & {c["uid"] for c in second["results"]}


def test_big_centre_diamond_is_solitaire(engine):
    res = engine.search("ek patli ring chahiye jisme beech me ek bada round diamond ho aur side me koi stone na ho")
    assert sum(engine.attr[("stones", "solitaire")][c["uid"]] >= 0.3 for c in res["results"]) >= 6


def test_no_stones_is_strict(engine):
    """'bina stone' must never pad the page with stone designs, even when few plain ones exist."""
    from jewelsearch.search import PLAIN_MAX_STONE, PLAIN_MIN_PROB
    plain = engine.attr[("stones", "plain")]
    for q in ("gents ring bina stone", "plain gold ring without stones", "earrings no stones"):
        res = engine.search(q)
        for c in res["results"]:
            assert engine.stone[c["uid"]] <= PLAIN_MAX_STONE and plain[c["uid"]] >= PLAIN_MIN_PROB, (q, c["design_id"])
        if len(res["results"]) < 8:
            assert res["notes"], q   # the short page is explained


def test_mens_is_strict(engine):
    """'for men' never pads with women's designs; categories without men's pieces say so."""
    for q in ("gents earrings", "pendant for men"):
        res = engine.search(q)
        assert res["results"] == [] and "no men's" in " ".join(res["notes"]), q
    res = engine.search("gents ring bina stone", category="earrings", metal="any")
    assert res["results"] == []
    res = engine.search("gents bracelet")
    assert res["results"] and all(engine.mens[c["uid"]] for c in res["results"])


@pytest.mark.parametrize("prompt,layout", [("solitaire rings", "solitaire"), ("rings with side stones", "centre_side"),
                                           ("earrings with side stones", "centre_side"), ("cluster rings", "all_small"),
                                           ("rose gold thin solitaire rings", "solitaire")])
def test_diamond_layouts_are_what_was_asked(engine, prompt, layout):
    """Checked against the job cards / CAD (scripts/diamond_labels.py; the search never reads them):
    the first two pages have the asked layout. "Side stones" found none before the trained reader."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
    from diamond_labels import labels
    got = [labels(engine.meta[c["uid"]])[1] for page in (0, 1) for c in engine.search(prompt, page=page)["results"]]
    checkable = [g for g in got if g]
    assert len(checkable) >= 6 and sum(g == layout for g in checkable) >= 0.75 * len(checkable), (prompt, got)


def test_a_loosened_filter_always_says_so(engine):
    """Filters from the prompt are never relaxed silently: each one loosened for lack of
    designs is named in a note, in the shopper's own word."""
    from pathlib import Path

    from jewelsearch.query import parse
    for prompt, word in (("kada for men", "kada"), ("solitaire bracelet", "solitaire"),
                         ("हीरे के बिना पतली अंगूठी", "पतली")):
        relaxed = []
        _, notes = engine._filters(parse(prompt), relaxed)
        assert relaxed and any(f"“{word}”" in n for n in notes), (prompt, notes)
    for prompt in ("thin rings", "stud earrings", "cuff bracelet", "choker necklace"):
        relaxed = []
        _, notes = engine._filters(parse(prompt), relaxed)
        assert not relaxed and not any("clearly" in n for n in notes), (prompt, notes)
    here = Path(__file__).parent
    for f in ("prompts_100.txt", "prompts_in_domain_extra.txt", "prompts_structural.txt"):
        for line in (here / f).read_text().splitlines():
            prompt = line.split("|")[0].strip()
            if prompt and not prompt.startswith("#"):
                relaxed = []
                _, notes = engine._filters(parse(prompt), relaxed)
                assert notes or not relaxed, (prompt, relaxed)


# ---- "Browse all categories": a whole category without a prompt ----

@pytest.mark.parametrize("category", ["ring", "earrings", "pendant", "necklace", "bracelet"])
def test_browse_pages_through_the_whole_category_once(engine, category):
    order = engine.browse_order(category)
    assert order and len(order) == len(set(order))
    assert {engine.meta[u]["category"] for u in order} == {category}
    assert len({engine.meta[u]["family"] for u in order}) == len(order)   # one per stone-cut family
    first, second = engine.browse(category, 0), engine.browse(category, 1)
    assert [c["uid"] for c in first["results"]] == order[:24]
    assert not {c["uid"] for c in first["results"]} & {c["uid"] for c in second["results"]}
    assert first["total"] == len(order) and first["has_more"] == (len(order) > 24)


def test_browse_shows_the_richest_pieces_first(engine):
    import numpy as np
    for category in ("ring", "necklace", "pendant"):
        order = engine.browse_order(category)
        assert np.nanmedian(engine.stone[order[:24]]) > 2 * np.nanmedian(engine.stone[order[-24:]])


def test_categories_list_every_category(engine):
    items = engine.categories()
    assert [c["category"] for c in items] == ["ring", "earrings", "pendant", "necklace", "bracelet"]
    assert all(c["designs"] > 0 and c["thumb"].startswith("/media/") for c in items)   # the original render



# ---- diamonds, read from pictures ---------------------------------------------------

def test_search_never_looks_diamonds_up():
    # the client asked for diamonds read from images, not looked up in job cards / CAD files
    import inspect

    from jewelsearch import search
    src = inspect.getsource(search)
    assert "purchase" not in src and "specs.json" not in src and "diamond_labels" not in src


def test_photo_search_reads_diamonds_from_pixels(engine):
    import io
    import json

    from PIL import Image

    from jewelsearch import photo
    from jewelsearch.config import CROPS, thumb_name
    if not engine.dia_model:
        pytest.skip("diamond reader not trained (scripts/train_diamond_dna.py)")
    uid = next(u for u in engine.by_uid if engine._design_cut(u) == "oval" and engine.cat[u] == "ring")
    m = engine.by_uid[uid]
    crop = Image.open(CROPS / thumb_name(m["images"][m["embed_metal"]][m["front_view"]])).convert("RGBA")
    flat = Image.new("RGBA", crop.size, "white")
    flat.alpha_composite(crop)
    buf = io.BytesIO()
    flat.convert("RGB").save(buf, "JPEG", quality=90)
    pq = engine.read_photo(photo.read(buf.getvalue()))
    res = engine.search_photo(pq)
    json.dumps(res)   # the API must be able to send it (no numpy values)
    dia = res["dna"]["diamonds"]
    assert dia["how"] == "trained" and dia["centre_cut"]["value"] == "oval"
    assert res["results"][0]["uid"] in engine.families[m["family"]]
    assert any("oval centre" in c["tags"] for c in res["results"])
