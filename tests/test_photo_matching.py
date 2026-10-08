"""Photo matching with DINOv2 + SigLIP2, match labels, and design details.

Full measurement: scripts/eval_photo_search.py. The detail reader is a stand-in here.
Run: .venv/bin/python -m pytest tests/test_photo_matching.py -q
"""
import numpy as np
import pytest
from PIL import Image

from jewelsearch import details, search
from jewelsearch.config import CROPS, INDEX, thumb_name

needs_dino = pytest.mark.skipif(not (INDEX / "dino_views.npy").exists(), reason="DINOv2 views not built")


# ---- details (no model) -------------------------------------------------------------

def test_agreement_counts_the_photos_clear_details():
    designs = np.array([[0.95, 0.9, 0.1], [0.95, 0.05, 0.1], [0.1, 0.05, 0.9]])
    keys = ["halo", "clusters", "pave_band"]
    a = details.agreement({"halo": 0.99, "clusters": 0.97, "pave_band": 0.5, "two_tone": 1.0}, designs, keys)
    assert list(np.round(a, 3)) == [0.925, 0.5, 0.075]          # unsure pave and two-tone are not used
    assert details.agreement({"halo": 0.5}, designs, keys) is None


def test_only_sure_trusted_details_are_shown():
    """Halo and clusters are matched but not named (the reader's names were often wrong)."""
    shown = details.shown({"halo": 0.99, "clusters": 0.95, "pave_band": 0.9, "two_tone": 0.85})
    assert [d["key"] for d in shown] == ["pave_band"]
    assert shown[0]["label"] == "Diamond-set band"
    assert details.shown({"pave_band": 0.6}) == []


# ---- the engine (index) ---------------------------------------------------------------

class FakeReader:
    def __init__(self, reading):
        self.reading, self.calls = reading, 0

    def __call__(self, im, keys=None, side=None):
        self.calls += 1
        return dict(self.reading)


@pytest.fixture(scope="module")
def engine():
    return search.SearchEngine(judge=None, details=None)


def render(engine, uid, metal=None):
    m = engine.by_uid[uid]
    metal = metal or m["embed_metal"]
    views = m["images"][metal]
    return Image.open(CROPS / thumb_name(views[m["front_view"]] if m["front_view"] in views else views[sorted(views)[0]]))


@needs_dino
def test_both_models_rank_the_photo(engine):
    """Renders in another metal than the indexed one (a clean "catalogue" photo): measured 90% first."""
    rng = np.random.default_rng(3)
    uids = [u for u, m in engine.by_uid.items() if len(m["images"]) > 1]
    first = on_page = 0
    for uid in rng.choice(uids, 10, replace=False):
        m = engine.by_uid[int(uid)]
        other = next(mt for mt in m["images"] if mt != m["embed_metal"])
        pq = engine.read_photo(render(engine, int(uid), other).convert("RGB"))
        assert pq.dino is not None and pq.fused is not None
        fams = [engine.by_uid[c["uid"]]["family"] for c in engine.search_photo(pq)["results"]]
        first += fams[0] == m["family"]
        on_page += m["family"] in fams
    assert first >= 7 and on_page >= 9, (first, on_page)


@needs_dino
def test_only_the_first_card_can_carry_a_match_label(engine):
    pq = engine.read_photo(render(engine, 0).convert("RGB"))
    res = engine.search_photo(pq)
    assert res["results"][0]["match"] in ("same", "very_close", "close", "similar")
    assert all(c["match"] in ("same", "similar") for c in res["results"][1:])
    assert all(c["match"] != "close" for c in res["results"])
    assert all(c["match"] == "similar" for c in engine.search_photo(pq, page=1)["results"] if c["match"] != "same")


def _scores(engine, mask, sim_top, dino_top, sim_next, dino_next):
    """Two picture-model scores where design 0 and design 1 (other families) stand out."""
    rng = np.random.default_rng(0)
    sim, dino = rng.normal(0, 1, len(engine.meta)), rng.normal(0, 1, len(engine.meta))
    a, b = np.flatnonzero(mask)[:2]
    assert engine.meta[a]["family"] != engine.meta[b]["family"]
    sim[a], dino[a], sim[b], dino[b] = sim_top, dino_top, sim_next, dino_next
    return a, sim, dino


@needs_dino
def test_first_label_comes_from_its_lead_when_both_models_agree(engine):
    mask = engine.cat == "ring"
    pq = engine.read_photo(render(engine, 0).convert("RGB"))
    a, pq.sim, pq.dino = _scores(engine, mask, 4.0, 4.0, 3.5, 3.5)            # leads by 1.0 on z = 4 + 4
    assert engine._first_label(pq, mask, a) == "very_close"
    a, pq.sim, pq.dino = _scores(engine, mask, 4.0, 4.0, 3.7, 3.7)            # leads by 0.6: not enough
    assert engine._first_label(pq, mask, a) == "similar"
    a, pq.sim, pq.dino = _scores(engine, mask, 4.0, 4.0, 3.9, 3.9)            # leads by 0.2
    assert engine._first_label(pq, mask, a) == "similar"


@needs_dino
def test_one_models_spike_is_not_a_close_match(engine):
    """The hand photo: DINOv2 rated a wide multi-row ring far above every other (the fingers),
    SigLIP2 barely above average, and it was called a "Close match"."""
    mask = engine.cat == "ring"
    pq = engine.read_photo(render(engine, 0).convert("RGB"))
    a, pq.sim, pq.dino = _scores(engine, mask, 0.9, 9.0, 2.0, 2.5)
    assert engine._first_label(pq, mask, a) == "similar"


@needs_dino
def test_no_clear_match_says_so(engine, monkeypatch):
    pq = engine.read_photo(render(engine, 0).convert("RGB"))
    monkeypatch.setattr(search, "TIERS_LEAD", ((99.0, "very_close"),))   # nothing qualifies
    pq.dna["same"] = None
    res = engine.search_photo(pq)
    assert all(c["match"] == "similar" for c in res["results"])
    assert any(n.startswith("No design here is a clear match") for n in res["notes"])


# ---- finding the piece in a busy photo ----------------------------------------------------

class _Finder:
    """Stands in for the engine in SearchEngine._find_piece: a fixed patch map."""
    def __init__(self, heat, sbox):
        self.found = (heat, sbox)

    def piece_heat(self, im):
        return self.found


def _heat(cells, grid=24):
    h = np.full((grid, grid), 0.05)
    for r, c in cells:
        h[r, c] = 0.95
    return h


def test_busy_photo_is_cropped_around_the_jewellery_patches():
    im = Image.new("RGB", (480, 480), (200, 150, 120))
    v = {"full": im, "box": None}                                   # the background cut-out failed
    f = _Finder(_heat([(10, 10), (10, 11), (11, 10), (11, 11)]), (0, 0, 480, 480))
    crop, box = search.SearchEngine._find_piece(f, im, v)
    assert crop.size[0] == crop.size[1] and 40 <= crop.size[0] <= 50     # 2 cells of 20 px, padded 6% a side
    assert box[0] < 0.46 < box[2] and box[1] < 0.46 < box[3]


def test_cut_out_is_kept_when_both_find_the_same_piece():
    im = Image.new("RGB", (480, 480), "white")
    piece = Image.new("RGB", (100, 100), "gold")
    v = {"full": im, "piece": piece, "box": [0.4, 0.4, 0.6, 0.6]}
    f = _Finder(_heat([(10, 10), (10, 11), (11, 10), (11, 11), (12, 12), (13, 13)]), (0, 0, 480, 480))
    crop, box = search.SearchEngine._find_piece(f, im, v)
    assert crop is piece and box == [0.4, 0.4, 0.6, 0.6]


def _mask(box, shape=(192, 192)):
    m = np.zeros(shape, bool)
    m[int(box[1] * shape[0]):int(box[3] * shape[0]), int(box[0] * shape[1]):int(box[2] * shape[1])] = True
    return m


def test_finder_wins_when_the_cut_out_found_the_person():
    """On worn photos the cut-out "found" the person every time it found anything."""
    im = Image.new("RGB", (480, 480), "white")
    person = [0.0, 0.3, 0.6, 1.0]
    v = {"full": im, "piece": Image.new("RGB", (300, 300)), "box": person, "mask": _mask(person)}
    crop, box = search.SearchEngine._find_piece(_Finder(_heat([(3, 18), (3, 19), (4, 18), (4, 19)]), (0, 0, 480, 480)), im, v)
    assert box[0] > 0.65 and box[3] < 0.3


def test_cut_out_wins_when_it_is_jewellery_too():
    """A necklace on a plain background: the finder sees the pendant, the cut-out the whole necklace."""
    im = Image.new("RGB", (480, 480), "white")
    necklace = [0.2, 0.1, 0.8, 0.9]
    v = {"full": im, "piece": Image.new("RGB", (300, 300)), "box": necklace, "mask": _mask(necklace, (12, 12))}
    heat = np.full((24, 24), 0.05)
    heat[2:18, 5:19] = 0.5                       # the chain, read as jewellery but not surely
    heat[15:18, 10:13] = 0.95                    # the pendant
    crop, box = search.SearchEngine._find_piece(_Finder(heat, (0, 0, 480, 480)), im, v)
    assert crop is v["piece"] and box == necklace


def test_no_jewellery_patches_keeps_the_old_crop():
    im = Image.new("RGB", (480, 480), "white")
    v = {"full": im, "box": None}
    assert search.SearchEngine._find_piece(_Finder(_heat([]), (0, 0, 480, 480)), im, v) == (im, None)
    assert search.SearchEngine._find_piece(type("NoFinder", (), {"piece_heat": lambda self, im: None})(), im, v) == (im, None)


@needs_dino
def test_details_are_read_shown_and_shared(engine, monkeypatch):
    reader = FakeReader({"halo": 0.99, "clusters": 0.95, "pave_band": 0.97})
    monkeypatch.setattr(engine, "detail_reader", reader)
    keys = ["halo", "clusters", "pave_band"]
    fake = np.full((len(engine.meta), 3), 0.05)
    rings = np.flatnonzero(engine.cat == "ring")
    fake[rings[:40]] = [0.97, 0.96, 0.95]    # 40 rings that share all three readings
    monkeypatch.setattr(engine, "details", fake)
    monkeypatch.setattr(engine, "detail_keys", keys)
    pq = engine.read_photo(render(engine, int(rings[0])).convert("RGB"))
    assert reader.calls == 1
    assert pq.detail_agree is not None and pq.detail_agree[rings[0]] > 0.9
    assert [d["key"] for d in pq.dna["details"]] == ["pave_band"]       # halo / clusters matched, not named
    res = engine.search_photo(pq, category="ring")
    with_all = [c for c in res["results"] if c["uid"] in set(rings[:40].tolist())]
    assert with_all and all(c["shares"] == ["Diamond-set band"] for c in with_all)
    assert all(c["shares"] == [] for c in res["results"] if c["uid"] not in set(rings[:40].tolist()))
