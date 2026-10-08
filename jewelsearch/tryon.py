"""Try-on data: which designs have a 3D model, and each user's body photos.

Models are made by scripts/build_tryon_models.py (data/tryon/designs.json maps
a design to its model). Body photos are the user's own hand / face / neck
pictures used for the instant "Try on" preview. They are personal (a face is
biometric data), so:

- stored only on this server, one folder per user: data/bodyphotos/<user id>/
- served only to that same signed-in user, never cached by shared caches
- saved only with the user's explicit consent, deletable at any time
- stored with the landmarks the browser measured, so a preview needs no
  model or upload at view time
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import time
from pathlib import Path

from .auth import AuthError
from .config import DATA

MODELS = DATA / "tryon" / "models"
DESIGN_MAP = DATA / "tryon" / "designs.json"
FIDELITY = DATA / "tryon" / "fidelity.json"   # scripts/check_tryon_models.py
PHOTOS = DATA / "bodyphotos"
CUTOUTS = DATA / "tryon" / "photos"   # catalogue front views, trimmed for the "real photo" try-on
CUTOUT_SIDE = 1100                    # px: sharp on a phone at close-up, ~0.3-1 MB

# which body photo each category is tried on
PART_FOR = {"ring": "hand", "bracelet": "hand", "earrings": "face", "pendant": "neck", "necklace": "neck"}
PARTS = ("hand", "face", "neck")
MAX_PHOTO_BYTES = 3 * 1024 * 1024
UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")

_map = {"mtime": None, "data": {}}


def _mapping() -> dict:
    """designs.json, re-read when the batch converter updates it."""
    try:
        mtime = DESIGN_MAP.stat().st_mtime
    except FileNotFoundError:
        return {}
    if mtime != _map["mtime"]:
        _map["data"] = json.loads(DESIGN_MAP.read_text())
        _map["mtime"] = mtime
    return _map["data"]


_checks = {"mtime": None, "data": None}


def _fidelity() -> dict | None:
    """Model checks against the catalogue photos, re-read when they change.
    None until the check has been run at all."""
    try:
        mtime = FIDELITY.stat().st_mtime
    except FileNotFoundError:
        return None
    if mtime != _checks["mtime"]:
        _checks["data"] = json.loads(FIDELITY.read_text())
        _checks["mtime"] = mtime
    return _checks["data"]


def passed(slug: str) -> bool:
    """A model that doesn't look like its design (a missing ring head, both
    earrings of a pair, a flat bracelet) is never shown: once the check has
    run, only models that passed it, and not ones added since, are offered."""
    checks = _fidelity()
    if checks is None:
        return True
    c = checks.get(slug, {})
    if not c.get("pass"):
        return False
    # converted again after it was checked: hidden until it is re-checked
    glb = MODELS / f"{slug}.glb"
    return "mtime" not in c or not glb.exists() or int(glb.stat().st_mtime) == c["mtime"]


def model_for(meta: dict) -> str | None:
    slug = _mapping().get(f"{meta['design_id']}|{meta['folders'][0]}")
    return slug if slug and passed(slug) else None


def converted(meta: dict) -> str | None:
    """The design's converted model, passed the look check or not: its real
    size (from the CAD) is still right, which the real-photo try-on needs."""
    slug = _mapping().get(f"{meta['design_id']}|{meta['folders'][0]}")
    if not slug:
        return None
    try:
        return slug if json.loads((MODELS / f"{slug}.json").read_text()).get("status") == "ok" else None
    except FileNotFoundError:
        return None


def photo_ok(meta: dict) -> bool:
    """The real-photo try-on needs the catalogue front render and the real
    size: a ring takes it from the finger, earrings from their CAD model."""
    if meta["category"] not in PHOTO_PARTS or not any(front_image(meta, mt) for mt in meta.get("images", {})):
        return False
    return meta["category"] == "ring" or converted(meta) is not None


def live_model(meta: dict) -> str | None:
    """The 3D model for the live camera try-on (rings only, checked models)."""
    return model_for(meta) if meta["category"] == "ring" else None


def part_for(meta: dict) -> str | None:
    """Body photo this design can be tried on: with a 3D model that passed its
    check, or with its own catalogue render."""
    return PART_FOR.get(meta["category"]) if model_for(meta) or photo_ok(meta) else None


# ---------- body photos ----------

def _dir(uid: str) -> Path:
    if not UUID_RE.match(uid):   # user ids are UUIDs; never a path fragment
        raise AuthError(400, "Bad user id")
    return PHOTOS / uid


def _check_part(part: str):
    if part not in PARTS:
        raise AuthError(404, "Unknown body part")


def list_photos(uid: str) -> dict:
    out = {}
    d = _dir(uid)
    for part in PARTS:
        f = d / f"{part}.json"
        if f.is_file() and (d / f"{part}.jpg").is_file():
            info = json.loads(f.read_text())
            out[part] = {"updated": info["updated"], "width": info["width"], "height": info["height"],
                         "anchors": info["anchors"], "url": f"/api/body/{part}.jpg?v={int(info['updated'])}"}
    return out


def save_photo(uid: str, part: str, image_b64: str, width: int, height: int, anchors: dict, consent: bool):
    _check_part(part)
    if not consent:
        raise AuthError(400, "Please agree to store the photo first.")
    try:
        raw = base64.b64decode(image_b64.split(",", 1)[-1], validate=True)
    except ValueError:
        raise AuthError(400, "The photo could not be read.") from None
    if len(raw) > MAX_PHOTO_BYTES:
        raise AuthError(413, "The photo is too large.")
    if not raw.startswith(b"\xff\xd8\xff"):   # the page always sends a JPEG it re-encoded itself
        raise AuthError(400, "The photo must be a JPEG.")
    if not (64 <= width <= 4096 and 64 <= height <= 4096):
        raise AuthError(400, "Unexpected photo size.")
    blob = json.dumps(anchors)
    if len(blob) > 64 * 1024:
        raise AuthError(400, "Too much landmark data.")
    d = _dir(uid)
    d.mkdir(parents=True, exist_ok=True)
    os.chmod(d, 0o700)
    # write-then-rename so a half-written photo is never served
    tmp = d / f".{part}.jpg.tmp"
    tmp.write_bytes(raw)
    os.chmod(tmp, 0o600)
    tmp.replace(d / f"{part}.jpg")
    info = {"updated": time.time(), "width": width, "height": height, "anchors": anchors,
            "consent": {"given": True, "at": time.time()}}
    tmp = d / f".{part}.json.tmp"
    tmp.write_text(json.dumps(info))
    os.chmod(tmp, 0o600)
    tmp.replace(d / f"{part}.json")


def photo_path(uid: str, part: str) -> Path:
    _check_part(part)
    f = _dir(uid) / f"{part}.jpg"
    if not f.is_file():
        raise AuthError(404, "No photo yet")
    return f


def delete_photo(uid: str, part: str | None = None):
    d = _dir(uid)
    for p in ([part] if part else PARTS):
        _check_part(p)
        for ext in ("jpg", "json"):
            (d / f"{p}.{ext}").unlink(missing_ok=True)
    if d.is_dir() and not any(d.iterdir()):
        d.rmdir()


# ---------- "real photo" try-on: the catalogue render itself ----------

# The 2D try-on shows the design's own photo-real catalogue render (a front
# view on a transparent background) instead of the 3D model: it is exactly
# the design and looks real. Only parts shot facing the camera work this way.
PHOTO_PARTS = {"ring", "earrings"}


def front_image(meta: dict, metal: str) -> str | None:
    """Dataset path of the design's front view in one metal, if there is one."""
    views = meta.get("images", {}).get(metal)
    if not views:
        return None
    return views.get(meta.get("front_view") or "4") or views.get("4")


def cutout(rel: str) -> Path:
    """The front view trimmed to the piece and scaled for phones, cached.
    Transparency is kept: the page places it on the body photo. Read once
    from the dataset storage (the web copy is plenty for 1100 px)."""
    import io

    from PIL import Image

    from . import storage

    out = CUTOUTS / (hashlib.sha1(rel.encode()).hexdigest()[:20] + ".png")
    if out.exists():
        return out
    im = Image.open(io.BytesIO(storage.get().read_bytes(rel, prefer="web"))).convert("RGBA")
    box = im.getchannel("A").point(lambda a: 255 if a > 8 else 0).getbbox()
    if box:
        pad = int(0.02 * max(box[2] - box[0], box[3] - box[1]))
        im = im.crop((max(0, box[0] - pad), max(0, box[1] - pad), min(im.width, box[2] + pad), min(im.height, box[3] + pad)))
    im.thumbnail((CUTOUT_SIDE, CUTOUT_SIDE), Image.LANCZOS)
    CUTOUTS.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    im.save(tmp, "PNG", optimize=True)
    tmp.replace(out)
    return out
