"""Buy page figures: the job card wins; CAD estimates only where they were
checked against job cards; nothing is guessed."""
import json

import pytest

from jewelsearch import purchase

RING = {"design_id": "R1", "folders": ["Rings/R1"], "category": "ring", "metals": ["rose_gold", "white_gold", "yellow_gold"]}
EARRING = {"design_id": "E1", "folders": ["Ear/E1"], "category": "earrings", "metals": ["yellow_gold"]}
PENDANT = {"design_id": "P1", "folders": ["Pend/P1"], "category": "pendant", "metals": ["white_gold"]}
NECKLACE = {"design_id": "N1", "folders": ["Neck/N1"], "category": "necklace", "metals": ["yellow_gold"]}

CARD = {
    "design_number": "R1", "category": "Ladies Ring", "design_type": "Simple",
    "ring_size": {"unit": "US", "size": 7.0},
    "weights_g": {"14k": 7.735, "18k": 9.1, "22k": 10.374},
    "stones": [
        {"shape": "OVAL", "sieve": "0", "gem": "Diamond", "size": "12.00X8.00", "count": 1, "carat_each": 3.63, "carat": 3.63, "setting": "COLLET SETTING"},
        {"shape": "RND", "sieve": "+3.5", "gem": "Diamond", "size": "1.4", "count": 42, "carat_each": 0.012, "carat": 0.504, "setting": "U CUT "},
    ],
}
CAD_RING = {
    "status": "ok", "category": "ring", "size_mm": [20.1, 7.87, 24.4],
    "metal_volume_mm3": 250.0, "metal_open_share": 0.0,
    "stone_groups": [{"shape": "round", "size_mm": [1.4, 1.4], "depth_mm": 0.87, "count": 10, "volume_mm3": 5.8},
                     {"shape": "pear", "size_mm": [6.0, 4.0], "depth_mm": 2.4, "count": 1, "volume_mm3": 18.75}],
    "ring": {"inner_diameter_mm": 17.03, "estimated": False, "us_size": 6.5},
}


@pytest.fixture
def specs(tmp_path, monkeypatch):
    f = tmp_path / "specs.json"
    monkeypatch.setattr(purchase, "SPECS", f)
    monkeypatch.setattr(purchase, "_cache", {"mtime": None, "data": {}})

    def write(designs, calibration={"k18_g_per_cm3": 12.0, "rings": 94, "within_10pct": 0.89}):
        f.write_text(json.dumps({"calibration": calibration, "designs": designs}))
    return write


def test_the_job_card_wins_over_the_cad(specs):
    specs({"R1|Rings/R1": {"card": CARD, "cad": CAD_RING}})
    d = purchase.details(RING)
    assert d["source"] == "card"
    assert d["gold"] == {"source": "card", "by_purity": {"14k": 7.74, "18k": 9.1, "22k": 10.37}}
    s = d["stones"]
    assert s["source"] == "card" and s["counted"]
    assert [(g["shape"], g["size"], g["count"], g["setting"]) for g in s["groups"]] == [
        ("Oval", "12 × 8 mm", 1, "Collet setting"), ("Round", "1.4 mm", 42, "U cut")]
    assert s["count"] == 43 and s["carat"] == 4.13
    assert d["ring_size"] == {"us": 7.0, "in": 14, "diameter_mm": 17.3, "estimated": False}
    assert d["design_type"] == "Simple"
    assert d["dimensions"] == [{"label": "Width on the finger", "value": "7.9 mm"}]


def test_a_card_without_stones_is_a_plain_piece(specs):
    specs({"R1|Rings/R1": {"card": {**CARD, "stones": []}}})
    s = purchase.details(RING)["stones"]
    assert s["count"] == 0 and s["groups"] == []


def test_a_ring_without_a_card_gets_calibrated_estimates(specs):
    specs({"R1|Rings/R1": {"cad": CAD_RING}})
    d = purchase.details(RING)
    assert d["source"] == "cad"
    assert d["gold"]["source"] == "cad"
    assert d["gold"]["by_purity"]["18k"] == 3.0     # 0.25 cm3 x 12 g/cm3
    assert d["gold"]["by_purity"]["14k"] == 2.55
    rnd, pear = d["stones"]["groups"]
    assert rnd == {"shape": "Round", "size": "1.4 mm", "count": 10, "carat": 0.12, "setting": "", "gem": "Diamond"}
    assert pear["shape"] == "Pear" and pear["size"] == "6.0 × 4.0 mm"
    assert pear["carat"] == pytest.approx(0.33, abs=0.001)   # volume x 0.0176 ct/mm3
    assert d["ring_size"]["us"] == 6.5 and not d["ring_size"]["estimated"]


def test_no_weight_from_an_open_metal_surface(specs):
    specs({"R1|Rings/R1": {"cad": {**CAD_RING, "metal_open_share": 0.2}}})
    assert purchase.details(RING)["gold"] is None


def test_no_weight_without_a_calibration(specs):
    specs({"R1|Rings/R1": {"cad": CAD_RING}}, calibration=None)
    assert purchase.details(RING)["gold"] is None


def test_pendants_get_estimates_but_not_unchecked_categories(specs):
    cad = {**CAD_RING, "category": "pendant", "ring": None}
    specs({"P1|Pend/P1": {"cad": cad}, "N1|Neck/N1": {"cad": {**cad, "category": "necklace"}}})
    p = purchase.details(PENDANT)
    assert p["gold"]["source"] == "cad"
    assert p["dimensions"] == [{"label": "Height × width", "value": "7.9 × 20.1 mm"}, {"label": "Depth", "value": "24.4 mm"}]
    n = purchase.details(NECKLACE)
    assert n["gold"] is None
    assert n["dimensions"] == [{"label": "Overall size", "value": "24.4 × 20.1 × 7.9 mm"}]


def test_earrings_from_cad_show_stone_sizes_but_no_counts_or_weight(specs):
    cad = {**CAD_RING, "category": "earring", "ring": None, "pair_in_file": False,
           "stone_groups": CAD_RING["stone_groups"] + [{"shape": "round", "size_mm": [1.4, 1.4], "depth_mm": 0.87, "count": 4, "volume_mm3": 2.3}]}
    specs({"E1|Ear/E1": {"cad": cad}})
    d = purchase.details(EARRING)
    assert d["gold"] is None
    s = d["stones"]
    assert not s["counted"] and s["count"] is None and s["carat"] is None
    assert [(g["shape"], g["size"], g["count"]) for g in s["groups"]] == [("Round", "1.4 mm", None), ("Pear", "6.0 × 4.0 mm", None)]
    assert d["dimensions"] == []   # one earring or the pair side by side: unknown
    specs({"E1|Ear/E1": {"cad": {**cad, "pair_in_file": True}}})
    assert purchase.details(EARRING)["dimensions"][0]["label"] == "Height × width (one earring)"


def test_a_design_with_nothing_measured(specs):
    specs({})
    d = purchase.details(RING)
    assert d["source"] is None and d["gold"] is None and d["stones"] is None and d["ring_size"] is None
    assert d["dimensions"] == []


def test_22k_is_offered_in_yellow_gold_only(specs):
    specs({})
    d = purchase.details(RING)
    assert d["purities_for"] == {"rose_gold": ["14k", "18k"], "white_gold": ["14k", "18k"],
                                 "yellow_gold": ["14k", "18k", "22k"]}


@pytest.mark.parametrize("kwargs, us, ind", [
    ({"us": 7}, 7.0, 14), ({"us": 6}, 6.0, 12), ({"indian": 14}, 7.0, 14), ({"diameter": 17.03}, 6.5, 14),
])
def test_ring_sizes(kwargs, us, ind):
    r = purchase.ring_size(**kwargs)
    assert (r["us"], r["in"]) == (us, ind)


def test_impossible_ring_sizes_are_dropped():
    assert purchase.ring_size(diameter=40) is None
    assert purchase.ring_size() is None


@pytest.mark.parametrize("raw, label", [("1.4", "1.4 mm"), ("12.00X8.00", "12 × 8 mm"), ("4.00 x 2.50", "4 × 2.5 mm"),
                                        ("MELE", "MELE"), ("", "")])
def test_stone_size_labels(raw, label):
    assert purchase.size_label(raw) == label


def test_round_carats_follow_the_trade_chart():
    assert purchase.round_carat(1.4) == pytest.approx(0.012)
    assert purchase.round_carat(1.425) == pytest.approx(0.0125)
    assert purchase.round_carat(6.4) == pytest.approx(1.0)


def test_card_shape_codes():
    assert [purchase.shape_name(c) for c in ("RND", "OVAL", "PEAR", "EMR", "MQ", "BAG", "", "Tanzanite")] == [
        "Round", "Oval", "Pear", "Emerald", "Marquise", "Baguette", "Diamond", "Tanzanite"]


def test_a_cad_file_without_stones_is_not_called_plain(specs):
    specs({"R1|Rings/R1": {"cad": {**CAD_RING, "stone_groups": []}}})
    assert purchase.details(RING)["stones"] is None
