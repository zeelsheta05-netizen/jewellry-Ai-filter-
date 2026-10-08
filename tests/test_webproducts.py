"""The "From the web" panel (jewelsearch/webproducts.py): reading shop feeds,
the type and colour filters, the matching, and picture sizes. No network, no models."""
import json
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from jewelsearch import webproducts as wp


@pytest.mark.parametrize("title,ptype,tags,want", [
    ("Rose Gold Halo Diamond Ring", "Rings", [], "ring"),
    ("Silver Hoop Earrings", "Earrings", [], "earrings"),
    ("Gold Ear Cuff", "", [], "earrings"),
    ("Green Enamel Cuff", "Bracelet", [], "bracelet"),
    ("Open Dome River Pendant Necklace", "Chain Pendant", [], "pendant"),
    ("Paper Link Chain Bracelet", "", [], "bracelet"),
    ("Zircon Mangalsutra 201982", "", [], "necklace"),
    ("Rose Gold Suchismita Mangalsutra", "Pendants", ["pendant"], "necklace"),
    ("Blink Detail Bracelet And Ring Set", "Jewelry Sets", [], None),      # a set
    ("Diamond Nose Pin", "NOSE PIN", [], None),                            # a type we don't have
    ("Gift Card", "", ["rings"], None),
    ("Lunessa", "", ["Pendants"], "pendant"),                              # only the tags say it
])
def test_category_of(title, ptype, tags, want):
    assert wp.category_of(title, ptype, tags) == want


def test_colours_of():
    assert wp.colours_of(["Rose Gold Kamala Necklace"]) == ["rose_gold"]
    assert wp.colours_of(["Golden Moon Hoop Earrings"]) == ["yellow_gold"]
    assert wp.colours_of(["925 Sterling Silver Ring"]) == ["white_gold"]          # silver looks white
    assert wp.colours_of(["Ring", "Yellow Gold", "White Gold", "Rose Gold"]) == ["rose_gold", "white_gold", "yellow_gold"]
    assert wp.colours_of(["Crystal Threader Earrings"]) == []                      # not said: not guessed here


def test_material_of():
    assert wp.material_of("Classic Round 9KT Gold Laboratory Grown Diamond Ring") == "9KT gold · Lab-grown diamond"
    assert wp.material_of("Pearl Bypass 925 Sterling Silver Ring") == "925 silver · Pearl"
    assert wp.material_of("Gold Plated Hoops with zircon") == "Gold plated · Zircon"
    assert wp.material_of("Black Bloom Bracelet") == ""


def test_sized_url_replaces_the_width():
    u = wp.sized_url("//www.giva.co/cdn/shop/files/a.jpg?v=12&width=100", 640)
    assert u == "https://www.giva.co/cdn/shop/files/a.jpg?v=12&width=640"


def test_from_feed():
    shop = wp.SHOP_BY_KEY["giva"]
    prod = {"id": 7, "handle": "rose-ring", "title": "Rose Gold  Halo Ring", "product_type": "Rings", "tags": ["925"],
            "variants": [{"price": "2499.00", "available": True}, {"price": "1999.00", "available": False}],
            "images": [{"src": "https://cdn.example/a.jpg", "width": 1500, "height": 1500}],
            "options": [{"name": "Metal Colour", "values": ["Rose Gold", "Silver"]}], "published_at": "2026-10-01T10:00:00Z"}
    p = wp.from_feed(shop, prod, "INR")
    assert p["url"] == "https://www.giva.co/products/rose-ring"
    assert p["category"] == "ring" and p["title"] == "Rose Gold Halo Ring"
    assert p["colours"] == ["rose_gold", "white_gold"] and p["price"] == 2499.0   # sold-out variant's price ignored
    assert p["id"] == wp.product_id("giva", 7)
    sold_out = {**prod, "variants": [{"price": "1", "available": False}]}
    assert wp.from_feed(shop, sold_out, "INR") is None
    assert wp.from_feed(shop, {**prod, "images": []}, "INR") is None


# ---- matching, on a tiny pool ----

def _unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


@pytest.fixture
def pool(tmp_path):
    items = [
        ("a", "giva", "ring", ["rose_gold"], [1, 0, 0]),
        ("b", "salty", "ring", ["yellow_gold"], [0.7, 0.7, 0]),
        ("c", "giva", "ring", ["rose_gold"], [1, 0.001, 0]),       # the same picture as "a" (a colour listing)
        ("d", "isharya", "earrings", ["yellow_gold"], [1, 0, 0]),
        ("e", "palmonas", "ring", [], [0, 1, 0]),
    ]
    rows = [{"id": f"{n}" * 12, "shop": s, "title": n, "url": f"https://x/{n}", "category": c, "colours": col,
             "material": "", "price": 100.0, "currency": "INR", "image": "https://x/i.jpg", "published": None}
            for n, s, c, col, _ in items]
    vecs = np.stack([_unit(v) for *_, v in items])
    (tmp_path / "products.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    for name in ("siglip.npy", "title.npy", "dino.npy"):
        np.save(tmp_path / name, vecs)
    (tmp_path / "info.json").write_text(json.dumps({"built": "x", "shops": {}}))
    q = SimpleNamespace(categories=["ring"], category="ring", exclude_categories=[], metal=None, exclude_metals=[], raw="",
                        intents=[])
    engine = SimpleNamespace(
        domain=SimpleNamespace(check=lambda prompt: SimpleNamespace(ok="boxing" not in prompt, text=prompt)),
        _query_vector=lambda q: _unit([1, 0, 0]))
    return wp.WebProducts(engine, tmp_path), q


def test_prompt_keeps_the_type_and_skips_repeat_pictures(pool, monkeypatch):
    web, q = pool
    monkeypatch.setattr("jewelsearch.query.parse", lambda text: q)
    r = web.for_prompt("rose gold ring")
    ids = [c["title"] for c in r["results"]]
    assert "d" not in ids                       # earrings never show for a ring search
    assert ids[0] == "a" and "c" not in ids     # "c" is the same picture as "a"
    assert set(ids) == {"a", "b", "e"}


def test_prompt_metal_is_strict(pool, monkeypatch):
    web, q = pool
    q.metal = "rose_gold"
    monkeypatch.setattr("jewelsearch.query.parse", lambda text: q)
    assert [c["title"] for c in web.for_prompt("rose gold ring")["results"]] == ["a"]


def test_not_jewellery_is_refused(pool):
    web, _ = pool
    r = web.for_prompt("boxing ring")
    assert r["refused"] and r["results"] == []


def test_like_product_same_type_not_itself(pool):
    web, _ = pool
    r = web.like_product("a" * 12)
    ids = [c["title"] for c in r["results"]]
    assert "a" not in ids and "d" not in ids and ids[0] == "c"
    with pytest.raises(KeyError):
        web.like_product("f" * 12)


def test_missing_pool_is_not_ready(tmp_path):
    assert not wp.WebProducts(SimpleNamespace(), tmp_path).ready


# ---- picture sizes ----

def test_pictures_shrink_and_enlarge_at_most_twice(tmp_path, monkeypatch):
    monkeypatch.setattr(wp, "IMG_DIR", tmp_path)
    big = wp.save_picture("big", Image.new("RGB", (1500, 1000), "white"), 640)
    assert Image.open(big).size == (640, 427)
    small = wp.save_picture("small", Image.new("RGB", (200, 150), "white"), 640)
    assert Image.open(small).size == (400, 300)          # 2x, not stretched to 640
    near = wp.save_picture("near", Image.new("RGB", (630, 630), "white"), 640)
    assert Image.open(near).size == (630, 630)           # not worth enlarging
    assert not list(tmp_path.glob("*.tmp"))


def test_specific_words_of_the_request_count(pool):
    web, _ = pool
    web.items[1]["title"] = "Pearl Mangalsutra"
    s = web._word_score("gold mangalsutras with pearls")
    assert s[1] == 2 and s[0] == 0                       # plural or not; "gold" is a filter word, not counted
    assert not web._word_score("gold ring for women").any()


def test_mens_pieces(pool, monkeypatch):
    web, q = pool
    web.items[0]["title"] = "Bold Men's Band"
    web.mens[0] = True
    monkeypatch.setattr("jewelsearch.query.parse", lambda text: q)
    assert [c["title"] for c in web.for_prompt("ring")["results"]][0] != "Bold Men's Band"   # pushed down
    q.intents = ["men"]
    assert [c["title"] for c in web.for_prompt("ring for men")["results"]] == ["Bold Men's Band"]


def test_picture_colour_reading(pool):
    web, _ = pool
    # metal text vectors in METALS order (rose, white, yellow): picture [1,0,0] reads rose, [0,1,0] white
    web.engine._dna_vectors = lambda: {"metal": np.stack([_unit([1, 0, 0]), _unit([0, 1, 0]), _unit([0, 0, 1])])}
    p = web._picture_colour("rose_gold")
    assert p[0] > 0.99 and p[4] < 0.01      # "a" shows rose, "e" shows white


def test_photo_uses_its_type_and_picture(pool):
    web, _ = pool
    pq = SimpleNamespace(vec=_unit([1, 0, 0]), dvec=_unit([1, 0, 0]), category="ring", metal=None)
    ids = [c["title"] for c in web.for_photo(pq)["results"]]
    assert "d" not in ids and ids[0] == "a"                 # earrings left out: the photo is a ring
    assert web.for_photo(pq)["matches"] == 4 and web.for_photo(pq, category="any")["matches"] == 5
    unsure = SimpleNamespace(vec=_unit([1, 0, 0]), dvec=None, category=None, metal=None)
    assert web.for_photo(unsure)["matches"] == 5            # type not sure: no type filter
    assert [c["title"] for c in web.for_photo(pq, metal="yellow_gold")["results"]] == ["b"]   # asked metal is strict


def test_our_design_on_the_web(pool):
    web, _ = pool
    e = web.engine
    e.by_uid = {0: {"category": "ring", "design_id": "DR-1"}}
    e.vecs = np.stack([_unit([0, 1, 0])])
    e.front = np.stack([_unit([0, 1, 0])])
    e.dino_views = None
    r = web.like_design(0)
    ids = [c["title"] for c in r["results"]]
    assert ids[0] == "e" and "d" not in ids and r["source"]["design_id"] == "DR-1"   # rings only, closest first
    # the picked colour breaks near-ties only: a clearly closer shape stays first
    e._dna_vectors = lambda: {"metal": np.stack([_unit([1, 0, 0]), _unit([0, 1, 0]), _unit([0, 0, 1])])}
    assert [c["title"] for c in web.like_design(0, metal="rose_gold")["results"]][0] == "e"


def test_our_design_diamond_readings_count(pool):
    web, _ = pool
    e = web.engine
    e.by_uid = {0: {"category": "ring", "design_id": "DR-1"}}
    e.vecs = e.front = np.stack([_unit([0.5, 0.5, 0.7])])            # equally like every ring
    e.dino_views = None
    # cut reader: picture direction x = "pear", y = "round"; our design is a pear
    e.dia_model = {"x": 1}
    e._diamond_probs = lambda head, X: np.stack([X[:, 0] ** 2, X[:, 1] ** 2 + X[:, 2] ** 2], 1)
    e.dia_cut = np.array([[1.0, 0.0]])
    e.dia_layout = np.array([[0.5, 0.5]])
    ids = [c["title"] for c in web.like_design(0)["results"]]
    assert ids.index("a" if "a" in ids else "c") < ids.index("e")    # the pear-reading picture ranks above


def test_our_design_and_mens_pieces(pool):
    web, _ = pool
    e = web.engine
    e.by_uid = {0: {"category": "ring", "design_id": "DR-1"}}
    e.vecs = e.front = np.stack([_unit([1, 0, 0])])
    e.dino_views = None
    web.mens[0] = True                                     # "a" is a men's piece
    e.mens = np.array([False])
    assert [c["title"] for c in web.like_design(0)["results"]][0] != "a"
    e.mens = np.array([True])
    assert [c["title"] for c in web.like_design(0)["results"]][0] == "a"
