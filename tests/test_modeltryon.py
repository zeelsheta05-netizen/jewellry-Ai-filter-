"""Try-on on ready-made models (jewelsearch/modeltryon.py): the AI puts the
design on the model photo in one step (no pre-placed layer); its picture must
show the piece at the measured spot with the model unchanged, and only the
drawn piece is taken back into the original photo. A ring never goes on an ear."""
import base64
import io
import json

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFilter

from jewelsearch import modeltryon, sketch
from jewelsearch.auth import AuthError

UID = "12345678-1234-1234-1234-123456789abc"
OTHER = "87654321-4321-4321-4321-cba987654321"
RING = {"uid": 7, "design_id": "R1", "category": "ring"}


def _hand(lean=0.0):
    """21 hand landmarks (normalised) of an upright left hand, fingers up;
    lean shifts the fingertips sideways (a finger leaning right)."""
    lm = [[0.5, 0.9, 0]] + [[0.5, 0.8, 0]] * 4
    for x in (0.62, 0.54, 0.46, 0.38):                         # index, middle, ring, little
        for y in (0.55, 0.42, 0.33, 0.25):                     # MCP, PIP, DIP, tip
            lm.append([x + lean * (0.55 - y), y, 0])
    return {"landmarks": lm, "isRight": False}


HAND = {"id": "hand-a", "label": "Hand", "width": 600, "height": 750, "parts": {"hand": _hand()}}
PORTRAIT = {"id": "face-a", "label": "Portrait", "width": 600, "height": 750,
            "parts": {"face": {"lobeL": {"x": 0.3, "y": 0.4}, "lobeR": {"x": 0.7, "y": 0.4}, "pxPerMm": 4.0},
                      "neck": {"notch": {"x": 0.5, "y": 0.6}, "neckL": {"x": 0.42, "y": 0.55},
                               "neckR": {"x": 0.58, "y": 0.55}, "pxPerMm": 4.0}}}


@pytest.fixture(autouse=True)
def dirs(tmp_path, monkeypatch):
    lib = tmp_path / "library"
    lib.mkdir()
    monkeypatch.setattr(modeltryon, "LIBRARY", lib)
    monkeypatch.setattr(modeltryon, "CUSTOM", tmp_path / "custom")
    monkeypatch.setattr(modeltryon, "RESULTS", tmp_path / "results")
    monkeypatch.setattr(modeltryon, "_lib", {"mtime": None, "items": []})
    monkeypatch.setattr(sketch, "local_ready", lambda: True)
    return tmp_path


def _photo(w=600, h=750, seed=0):
    """A stand-in model photo: skin-coloured noise, so every pixel is distinct."""
    rng = np.random.default_rng(seed)
    a = np.clip(rng.normal([200, 160, 135], 12, (h, w, 3)), 0, 255).astype(np.uint8)
    return Image.fromarray(a)


def _design():
    """A stand-in catalogue picture: a gold ring with a stone, transparent around it."""
    im = Image.new("RGBA", (300, 300), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.ellipse((60, 60, 240, 240), outline=(225, 180, 80, 255), width=22)
    d.ellipse((120, 30, 180, 90), fill=(245, 245, 250, 255))
    return im


def _library(dirs, items):
    for it in items:
        _photo(it["width"], it["height"]).save(dirs / "library" / f"{it['id']}.jpg", quality=95)
    (dirs / "library" / "library.json").write_text(json.dumps({"items": items}))


def _fake_ai(kind):
    """A stand-in for the AI. It gets [model crop, design picture] and returns:
    'good'     the crop with a ring drawn at its centre (and slightly softened)
    'far'      the ring drawn in a corner (on another finger)
    'nothing'  the crop, unchanged
    'moved'    the crop shifted 12 px with the ring (the AI moved the hand)"""
    calls = []

    def ai(images, prompt, seed):
        calls.append(prompt)
        im = images[0].filter(ImageFilter.GaussianBlur(0.4))
        n = im.width
        if kind == "nothing":
            return im
        cx, cy = (n // 2, n // 2) if kind != "far" else (n // 8, n // 8)
        r = n // 14
        d = ImageDraw.Draw(im)
        d.ellipse((cx - r, cy - r, cx + r, cy + r), outline=(225, 180, 80), width=max(3, r // 3))
        d.ellipse((cx - r // 3, cy - r - r // 3, cx + r // 3, cy - r + r // 3), fill=(245, 245, 250))
        if kind == "moved":
            im = Image.fromarray(np.roll(np.asarray(im), 12, axis=1))
        return im
    ai.calls = calls
    return ai


def test_only_models_measured_for_the_body_part_are_offered(dirs):
    _library(dirs, [HAND, PORTRAIT])
    assert [m["id"] for m in modeltryon.models_for(UID, "hand")["recommended"]] == ["hand-a"]
    assert [m["id"] for m in modeltryon.models_for(UID, "face")["recommended"]] == ["face-a"]
    m = modeltryon.models_for(UID, "neck")["recommended"][0]
    assert m["ref"] == "lib:face-a" and m["anchors"]["notch"] == {"x": 0.5, "y": 0.6}


def test_a_ring_is_never_made_on_a_portrait_whatever_the_page_sends(dirs):
    _library(dirs, [HAND, PORTRAIT])
    ai = _fake_ai("good")
    with pytest.raises(AuthError) as e:
        modeltryon.make(UID, "lib:face-a", RING, "yellow_gold", _design(), ai=ai)
    assert e.value.status == 400 and not ai.calls


def test_unknown_or_unsafe_model_references_are_refused(dirs):
    _library(dirs, [HAND])
    for ref in ("lib:../../etc/passwd", "lib:nope", "custom:abc", "web:hand-a", "lib:HAND-A"):
        with pytest.raises(AuthError):
            modeltryon.resolve(UID, ref)


def test_the_spots_come_from_the_landmarks():
    (ring,) = modeltryon.spots("ring", _hand(), 600, 750)
    l, t, r, b = ring["box"]
    assert r - l == b - t and 0 <= l and 0 <= t and r <= 600 and b <= 750
    # the base of the ring finger: between its knuckle (0.46, 0.55) and middle joint (0.46, 0.42)
    x, y = l + ring["target"][0], t + ring["target"][1]
    assert abs(x - 0.46 * 600) < 2 and 0.42 * 750 < y < 0.55 * 750
    ears = modeltryon.spots("earrings", PORTRAIT["parts"]["face"], 600, 750)
    assert len(ears) == 2 and [round(s["box"][0] + s["target"][0]) for s in ears] == [180, 420]
    (pendant,) = modeltryon.spots("pendant", PORTRAIT["parts"]["neck"], 600, 750)
    assert pendant["box"][1] + pendant["target"][1] > 0.6 * 750   # below the neck notch


def test_the_design_is_drawn_and_the_rest_of_the_photo_is_untouched():
    base = _photo()
    ai = _fake_ai("good")
    shown, report = modeltryon.generate(base, "ring", "yellow_gold", _hand(), _design(), ai=ai)
    assert shown.size == base.size and len(ai.calls) == 1
    a, b = np.asarray(base), np.asarray(shown)
    changed = np.abs(a.astype(int) - b.astype(int)).sum(-1) > 30
    ys, xs = np.nonzero(changed)
    # the drawn ring is at the base of the ring finger
    assert abs(xs.mean() - 0.46 * 600) < 25 and 0.40 * 750 < ys.mean() < 0.56 * 750
    # far from it, the photo is the original pixel for pixel
    far = np.ones(changed.shape, bool)
    far[int(0.3 * 750):int(0.65 * 750), int(0.3 * 600):int(0.62 * 600)] = False
    assert np.array_equal(a[far], b[far])


def test_the_ai_gets_the_model_crop_and_the_design_picture():
    seen = []

    def ai(images, prompt, seed):
        seen.append([im.size for im in images])
        return _fake_ai("good")(images, prompt, seed)
    modeltryon.generate(_photo(), "ring", "rose_gold", _hand(), _design(), ai=ai)
    (sizes,) = seen
    assert len(sizes) == 2 and sizes[0] == sizes[1]


@pytest.mark.parametrize("kind", ["far", "nothing", "moved"])
def test_a_wrong_picture_is_drawn_again_then_reported(kind):
    ai = _fake_ai(kind)
    with pytest.raises(sketch.SketchError):
        modeltryon.generate(_photo(), "ring", "yellow_gold", _hand(), _design(), ai=ai)
    assert len(ai.calls) == modeltryon.TRIES
    assert ai.calls[1].startswith("Most important")


def test_a_piece_not_recognised_as_this_design_is_rejected():
    def no(piece):
        return {"rank": 900, "of": 3000, "ok": False}
    with pytest.raises(sketch.SketchError):
        modeltryon.generate(_photo(), "ring", "yellow_gold", _hand(), _design(), ai=_fake_ai("good"), judge=no)
    pieces = []

    def yes(piece):
        pieces.append(piece.size)
        return {"rank": 3, "of": 3000, "ok": True}
    modeltryon.generate(_photo(), "ring", "yellow_gold", _hand(), _design(), ai=_fake_ai("good"), judge=yes)
    (w, h), = pieces
    assert 20 < w < 300 and 20 < h < 300   # the judge sees the drawn piece, not the whole crop


def test_a_leaning_finger_is_turned_up_for_the_ai_and_back():
    base = _photo()
    hand = _hand(lean=0.6)
    assert modeltryon.piece_angle("ring", hand, 600, 750) != 0
    shown, _ = modeltryon.generate(base, "ring", "yellow_gold", hand, _design(), ai=_fake_ai("good"))
    (s,) = modeltryon.spots("ring", hand, 600, 750)
    spot = (s["box"][0] + s["target"][0], s["box"][1] + s["target"][1])
    changed = np.abs(np.asarray(base, int) - np.asarray(shown, int)).sum(-1) > 30
    ys, xs = np.nonzero(changed)
    assert np.hypot(xs.mean() - spot[0], ys.mean() - spot[1]) < 20


def test_without_a_design_picture_nothing_is_made():
    with pytest.raises(AuthError):
        modeltryon.generate(_photo(), "ring", "yellow_gold", _hand(), None, ai=_fake_ai("good"))


def test_make_saves_the_picture_for_this_user_only(dirs):
    _library(dirs, [HAND])
    r = modeltryon.make(UID, "lib:hand-a", RING, "yellow_gold", _design(), ai=_fake_ai("good"))
    assert r["model"] == "lib:hand-a" and r["design"] == 7 and "exact_url" not in r
    assert Image.open(modeltryon.result_file(UID, r["id"])).size == (600, 750)
    assert [x["id"] for x in modeltryon.results(UID, 7)] == [r["id"]]
    assert modeltryon.results(UID, 8) == []
    with pytest.raises(AuthError):
        modeltryon.result_file(OTHER, r["id"])
    modeltryon.delete_result(UID, r["id"])
    assert modeltryon.results(UID) == []


def _png_url(im):
    b = io.BytesIO()
    im.save(b, "PNG")
    return "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()


def _jpeg_url(im):
    b = io.BytesIO()
    im.save(b, "JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(b.getvalue()).decode()


def test_uploaded_models_are_kept_per_user_and_checked(dirs):
    im = _photo(500, 640)
    m = modeltryon.add_custom(UID, _jpeg_url(im), 500, 640, {"hand": {"landmarks": [], "isRight": True}})
    assert m["ref"].startswith("custom:") and m["parts"] == ["hand"]
    assert [x["id"] for x in modeltryon.models_for(UID, "hand")["custom"]] == [m["id"]]
    assert modeltryon.models_for(UID, "face")["custom"] == []
    assert modeltryon.models_for(OTHER, "hand")["custom"] == []
    with pytest.raises(AuthError):   # not what the page measured
        modeltryon.add_custom(UID, _jpeg_url(im), 640, 500, {"hand": {}})
    with pytest.raises(AuthError):   # no body part found
        modeltryon.add_custom(UID, _jpeg_url(im), 500, 640, {})
    with pytest.raises(AuthError):   # an unknown body part
        modeltryon.add_custom(UID, _jpeg_url(im), 500, 640, {"foot": {}})
    with pytest.raises(AuthError):   # a PNG (the page always re-encodes to JPEG)
        modeltryon.add_custom(UID, _png_url(im.convert("RGBA")), 500, 640, {"hand": {}})
    modeltryon.delete_custom(UID, m["id"])
    assert modeltryon.models_for(UID, "hand")["custom"] == []


def test_the_ai_is_told_to_put_exactly_this_design_on_the_model():
    p = modeltryon.prompt_for("ring", "rose_gold")
    assert "ring shown in image 2" in p and "finger in the centre of image 1" in p
    assert "exactly the design in image 2" in p and "do not add or remove stones" in p
    assert "18k rose gold" in p and "Do not change the person" in p


def test_the_finger_angle_is_read_from_the_hand_landmarks():
    lm = [[0.5, 0.9, 0]] * 21
    up = [p[:] for p in lm]
    up[13], up[14] = [0.5, 0.6, 0], [0.5, 0.5, 0]          # straight up
    assert modeltryon.piece_angle("ring", {"landmarks": up}, 600, 600) == 0.0
    right = [p[:] for p in lm]
    right[13], right[14] = [0.5, 0.6, 0], [0.6, 0.5, 0]    # leaning right by 45 degrees
    assert round(modeltryon.piece_angle("ring", {"landmarks": right}, 600, 600)) == 45
    assert modeltryon.piece_angle("earrings", {"lobeL": {}}, 600, 600) == 0.0
