"""Which designs get a "Try on" button: only those whose 3D model passed the
check against the catalogue photo (scripts/check_tryon_models.py)."""
import json
import os

import pytest

from jewelsearch import tryon

RING = {"design_id": "R1", "folders": ["Rings/R1"], "category": "ring"}
EARRING = {"design_id": "E1", "folders": ["Earrings/E1"], "category": "earrings"}


@pytest.fixture(autouse=True)
def files(tmp_path, monkeypatch):
    mapping = tmp_path / "designs.json"
    mapping.write_text(json.dumps({"R1|Rings/R1": "r1-abc", "E1|Earrings/E1": "e1-def"}))
    monkeypatch.setattr(tryon, "DESIGN_MAP", mapping)
    monkeypatch.setattr(tryon, "FIDELITY", tmp_path / "fidelity.json")
    monkeypatch.setattr(tryon, "_map", {"mtime": None, "data": {}})
    monkeypatch.setattr(tryon, "_checks", {"mtime": None, "data": None})
    return tmp_path


def _checks(files, data, bump=0):
    f = files / "fidelity.json"
    f.write_text(json.dumps(data))
    t = f.stat().st_mtime + bump
    os.utime(f, (t, t))


def test_before_any_check_every_model_is_offered():
    assert tryon.model_for(RING) == "r1-abc"
    assert tryon.part_for(EARRING) is None   # no catalogue picture: nothing for the AI to draw from


def test_only_models_that_passed_are_offered(files):
    _checks(files, {"r1-abc": {"sim": 0.83, "pass": True}, "e1-def": {"sim": 0.70, "pass": False}})
    assert tryon.model_for(RING) == "r1-abc"
    assert tryon.model_for(EARRING) is None


def test_a_model_added_after_the_check_waits_for_it(files):
    _checks(files, {"r1-abc": {"sim": 0.83, "pass": True}})
    assert tryon.model_for(EARRING) is None


def test_a_new_check_is_picked_up_without_a_restart(files):
    _checks(files, {"r1-abc": {"pass": True}, "e1-def": {"pass": False}})
    assert tryon.model_for(EARRING) is None
    _checks(files, {"r1-abc": {"pass": True}, "e1-def": {"pass": True}}, bump=5)
    assert tryon.model_for(EARRING) == "e1-def"


def test_a_model_converted_again_after_its_check_waits_for_a_new_check(files, monkeypatch):
    models = files / "models"
    models.mkdir()
    monkeypatch.setattr(tryon, "MODELS", models)
    glb = models / "r1-abc.glb"
    glb.write_bytes(b"glb")
    mtime = int(glb.stat().st_mtime)
    _checks(files, {"r1-abc": {"pass": True, "mtime": mtime}})
    assert tryon.model_for(RING) == "r1-abc"
    os.utime(glb, (mtime + 60, mtime + 60))
    assert tryon.model_for(RING) is None


def test_every_design_with_a_front_picture_can_be_tried_on(files):
    """The model try-on draws the design from its catalogue front picture, so the
    button doesn't depend on a 3D model or its check."""
    _checks(files, {"r1-abc": {"pass": False}, "e1-def": {"pass": False}})
    with_pic = {**EARRING, "images": {"yellow_gold": {"4": "Earrings/E1/E1_YG_4.png"}}, "front_view": "4"}
    assert tryon.part_for(with_pic) == "face"
    pendant = {"design_id": "P1", "folders": ["Pendants/P1"], "category": "pendant",
               "images": {"rose_gold": {"4": "Pendants/P1/P1_RG_4.png"}}}
    assert tryon.part_for(pendant) == "neck"
