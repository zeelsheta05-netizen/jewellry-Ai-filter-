"""Try-on on ready-made models: the AI puts a catalogue design on a model photo
in one step.

The customer picks one or more model photos (our library, or photos they
upload, e.g. a model picture from the web) and presses Generate. For each:

1. WHERE: the spot on the photo (the base of the ring finger, each earlobe,
   the neck) comes from the landmarks measured on that photo when it was
   added. The server checks the body part fits the piece, so a ring can never
   go on an ear.
2. GENERATE (FLUX.2 Klein on this Mac): a square crop centred on the spot
   (a finger turned to point up) and the design's own catalogue picture go to
   the AI together, with one instruction: put exactly this design on the
   model at the centre. Nothing is pre-placed: the AI draws the piece worn.
3. CHECK: the piece must be drawn at the spot, the model unchanged around it,
   and the drawn piece must be recognised as this design among all designs of
   its kind (SigLIP2, the search model). A miss is drawn once more; a second
   miss is reported to the customer instead of showing a wrong picture.
4. KEEP THE PHOTO: only the piece the AI drew (and its shadow) is taken back
   into the photo, colour-matched; every other pixel is the original model
   photo at its full resolution.

Storage, per signed-in user only:
    data/tryon/library/            the shared model library (scripts/build_model_library.py)
    data/tryon/custom/<uid>/       photos the user uploaded, with their landmarks
    data/tryon/results/<uid>/      generated pictures: <rid>.jpg, <rid>.json
"""
from __future__ import annotations

import base64
import io
import json
import os
import re
import secrets
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter, ImageOps

from . import sketch, tryon
from .auth import AuthError
from .config import DATA

LIBRARY = DATA / "tryon" / "library"
CUSTOM = DATA / "tryon" / "custom"
RESULTS = DATA / "tryon" / "results"
UUID_RE = tryon.UUID_RE
ID_RE = re.compile(r"^[a-z0-9-]{1,60}$")
CUSTOM_MAX = 60                      # uploaded models per user
CUSTOM_BYTES = 8 * 1024 * 1024       # one uploaded photo (the page re-encodes to JPEG)
GEN = 768                            # px: the square the AI repaints (memory-safe next to the live app)
GEN_SMALL = 512                      # for small crops (an earring): less blow-up, the AI keeps the ear as it is
SMALL_CROP = 250                     # photo px: crops below this (an earring, 224 px) are made at GEN_SMALL; rings (>= 256) stay at GEN
TRIES = 2                            # AI attempts per spot before the picture is reported as failed
ACTIVE_MAX = 16                      # queued + running pictures per user

# ---------- the library ----------

_lib = {"mtime": None, "items": []}


def library() -> list[dict]:
    f = LIBRARY / "library.json"
    try:
        mtime = f.stat().st_mtime
    except FileNotFoundError:
        return []
    if mtime != _lib["mtime"]:
        _lib["items"] = json.loads(f.read_text())["items"]
        _lib["mtime"] = mtime
    return _lib["items"]


def _public(item: dict, kind: str, part: str | None, v: str = "") -> dict:
    base = f"/api/tryon/{kind}/{item['id']}"
    return {"ref": f"{'lib' if kind == 'library' else 'custom'}:{item['id']}", "id": item["id"],
            "label": item.get("label", ""), "tags": item.get("tags", []), "width": item["width"], "height": item["height"],
            "parts": sorted(item["parts"]), "anchors": item["parts"].get(part) if part else None,
            "url": f"{base}.jpg{v}", "thumb": f"{base}.thumb.jpg{v}"}


def models_for(uid: str, part: str | None) -> dict:
    """Recommended (library) and Custom (uploaded) models. With a part, only
    models measured for that body part: a ring is never offered an ear photo."""
    lib = [_public(it, "library", part) for it in library() if part is None or part in it["parts"]]
    own = [_public(it, "custom", part, f"?v={int(it['created'])}") for it in _custom_items(uid)
           if part is None or part in it["parts"]]
    return {"recommended": lib, "custom": own}


def library_file(mid: str, thumb: bool = False) -> Path:
    if not ID_RE.match(mid):
        raise AuthError(404, "Unknown model")
    f = LIBRARY / f"{mid}{'.thumb' if thumb else ''}.jpg"
    if not f.is_file():
        raise AuthError(404, "Unknown model")
    return f


# ---------- the user's own uploaded models ----------

def _udir(root: Path, uid: str) -> Path:
    if not UUID_RE.match(uid):
        raise AuthError(400, "Bad user id")
    return root / uid


def _custom_items(uid: str) -> list[dict]:
    d = _udir(CUSTOM, uid)
    if not d.is_dir():
        return []
    items = []
    for f in sorted(d.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            items.append(json.loads(f.read_text()))
        except (OSError, ValueError):
            continue
    return items


def _check_anchors(parts: dict) -> dict:
    if not isinstance(parts, dict) or not parts:
        raise AuthError(400, "No body part was found in this photo.")
    out = {}
    for part, anchors in parts.items():
        if part not in tryon.PARTS or not isinstance(anchors, dict):
            raise AuthError(400, "Unknown body part")
        out[part] = anchors
    if len(json.dumps(out)) > 96 * 1024:
        raise AuthError(400, "Too much landmark data.")
    return out


def _jpeg(data_url: str, limit: int) -> bytes:
    try:
        raw = base64.b64decode(data_url.split(",", 1)[-1], validate=True)
    except ValueError:
        raise AuthError(400, "The photo could not be read.") from None
    if len(raw) > limit:
        raise AuthError(413, "The photo is too large.")
    return raw


def add_custom(uid: str, image: str, width: int, height: int, parts: dict, label: str = "") -> dict:
    raw = _jpeg(image, CUSTOM_BYTES)
    if not raw.startswith(b"\xff\xd8\xff"):   # the page re-encodes every upload to JPEG
        raise AuthError(400, "The photo must be a JPEG.")
    with Image.open(io.BytesIO(raw)) as im:
        if im.size != (width, height) or not (128 <= width <= 4096 and 128 <= height <= 4096):
            raise AuthError(400, "Unexpected photo size.")
    parts = _check_anchors(parts)
    d = _udir(CUSTOM, uid)
    if len(list(d.glob("*.json"))) >= CUSTOM_MAX if d.is_dir() else False:
        raise AuthError(400, f"You can keep up to {CUSTOM_MAX} of your own models. Delete one first.")
    d.mkdir(parents=True, exist_ok=True)
    os.chmod(d, 0o700)
    cid = secrets.token_hex(6)
    (d / f"{cid}.jpg").write_bytes(raw)
    with Image.open(io.BytesIO(raw)) as im:
        th = ImageOps.exif_transpose(im).convert("RGB")
        th.thumbnail((480, 960), Image.LANCZOS)
        th.save(d / f"{cid}.thumb.jpg", "JPEG", quality=85)
    item = {"id": cid, "label": label[:60], "width": width, "height": height, "parts": parts, "created": time.time()}
    tmp = d / f".{cid}.json.tmp"
    tmp.write_text(json.dumps(item))
    tmp.replace(d / f"{cid}.json")
    return _public(item, "custom", None, f"?v={int(item['created'])}")


def custom_file(uid: str, cid: str, thumb: bool = False) -> Path:
    if not re.fullmatch(r"[0-9a-f]{12}", cid):
        raise AuthError(404, "Unknown model")
    f = _udir(CUSTOM, uid) / f"{cid}{'.thumb' if thumb else ''}.jpg"
    if not f.is_file():
        raise AuthError(404, "Unknown model")
    return f


def delete_custom(uid: str, cid: str):
    custom_file(uid, cid)
    d = _udir(CUSTOM, uid)
    for name in (f"{cid}.jpg", f"{cid}.thumb.jpg", f"{cid}.json"):
        (d / name).unlink(missing_ok=True)


def resolve(uid: str, ref: str) -> tuple[Path, dict]:
    """A model reference from the page -> (photo file, its measured item)."""
    kind, _, mid = ref.partition(":")
    if kind == "lib":
        f = library_file(mid)
        item = next((it for it in library() if it["id"] == mid), None)
    elif kind == "custom":
        f = custom_file(uid, mid)
        item = json.loads((f.with_suffix(".json")).read_text())
    else:
        raise AuthError(404, "Unknown model")
    if item is None:
        raise AuthError(404, "Unknown model")
    return f, item


# ---------- picture helpers ----------

def _dilate(mask: Image.Image, r: int) -> Image.Image:
    """Grow a mask by about r px (a box blur reaches r px in every direction: fast at any r)."""
    if r <= 0:
        return mask
    return mask.filter(ImageFilter.BoxBlur(r)).point(lambda v: 255 if v > 0 else 0)


def _lab(rgb: np.ndarray) -> np.ndarray:
    c = rgb / 255.0
    c = np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)
    xyz = c @ np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]]).T
    xyz /= np.array([0.9505, 1.0, 1.089])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def _blur(a: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur of a float image (H, W) or (H, W, 3), 0..255 range."""
    if a.ndim == 3:
        return np.stack([_blur(a[..., c], sigma) for c in range(3)], -1)
    im = Image.fromarray(a.clip(0, 255).astype(np.uint8))
    return np.asarray(im.filter(ImageFilter.GaussianBlur(sigma)), np.float32)


def _mask_img(m: np.ndarray) -> Image.Image:
    return Image.fromarray((m > 0).astype(np.uint8) * 255)


def _grow_into(seed: np.ndarray, allowed: np.ndarray, limit: int = 400) -> np.ndarray:
    """The parts of 'allowed' connected to 'seed' (grown 1 px at a time)."""
    m = seed & allowed
    img = _mask_img(m)
    allow = _mask_img(allowed)
    from PIL import ImageChops
    for _ in range(limit):
        nxt = ImageChops.multiply(img.filter(ImageFilter.MaxFilter(3)), allow)
        if nxt.getbbox() == img.getbbox() and ImageChops.difference(nxt, img).getbbox() is None:
            break
        img = nxt
    return np.asarray(img) > 0


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / (np.sqrt((a * a).sum() * (b * b).sum()) + 1e-6))


def call_ai(images: list[Image.Image], prompt: str, seed: int) -> Image.Image:
    """FLUX.2 Klein edit on this Mac with several reference pictures (image 1 is
    edited, the others guide it). Shares the generator lock with the AI
    generation panel: one picture at a time on this machine."""
    if not sketch.local_ready():
        raise sketch.SketchError("The AI try-on is not installed on this server.", 503)
    with sketch.drawing("try-on edit", "The AI is busy with other pictures. Try again in a few minutes."):
        # its turn on this Mac, the app's idle models out of memory, free memory: sketch.drawing
        tmp = DATA / "tryon" / "tmp"
        tmp.mkdir(parents=True, exist_ok=True)
        tag = secrets.token_hex(6)
        files = [tmp / f"{tag}-{i}.png" for i in range(len(images))]
        out, txt = tmp / f"{tag}-out.png", tmp / f"{tag}-prompt.txt"
        try:
            for im, f in zip(images, files):
                im.convert("RGB").save(f, "PNG")
            txt.write_text(prompt)
            cmd = [str(sketch.LOCAL_BIN / "mflux-generate-flux2-edit"), "--low-ram", "--image-paths", *map(str, files),
                   "--model", sketch.LOCAL_WEIGHTS, "--base-model", "flux2-klein-4b", "--prompt-file", str(txt),
                   "--width", str(images[0].width), "--height", str(images[0].height), "--steps", "4", "--seed", str(seed),
                   "--output", str(out), "--no-exif"]
            try:
                r = subprocess.run(cmd, capture_output=True, text=True, timeout=sketch.LOCAL_TIMEOUT,
                                   env={**os.environ, "HF_HUB_OFFLINE": "1"})
            except subprocess.TimeoutExpired:
                raise sketch.SketchError("The picture took too long. Try again.", 502)
            if r.returncode != 0 or not out.is_file():
                print(f"try-on AI failed ({r.returncode}): {r.stderr[-600:]}", flush=True)
                raise sketch.SketchError("The AI could not make the picture. Try again.", 502)
            with Image.open(out) as im:
                return im.convert("RGB")
        finally:
            for f in [*files, out, txt]:
                f.unlink(missing_ok=True)


def _square_ref(ref: Image.Image) -> Image.Image:
    """The catalogue render centred on white, as the AI's second picture."""
    ref = ref.convert("RGBA")
    box = ref.getchannel("A").getbbox() or (0, 0, ref.width, ref.height)
    ref = ref.crop(box)
    side = int(max(ref.size) * 1.15)
    sq = Image.new("RGB", (side, side), "white")
    sq.paste(ref, ((side - ref.width) // 2, (side - ref.height) // 2), ref)
    return sq.resize((GEN, GEN), Image.LANCZOS)


def _upright(img: Image.Image, angle: float, box, fill) -> Image.Image:
    """The crop 'box' of img, turned by angle degrees (counter-clockwise) about its centre."""
    if not angle:
        return img.crop(box)
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    return img.rotate(angle, resample=Image.BICUBIC, center=(cx, cy), fillcolor=fill).crop(box)


def _turn_back(crop: Image.Image, angle: float, box, size, fill) -> Image.Image:
    """Undo _upright: the crop put back at 'box' in a picture of 'size', turned back."""
    if not angle:
        return crop
    canvas = Image.new(crop.mode, size, fill)
    canvas.paste(crop, box[:2])
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    return canvas.rotate(-angle, resample=Image.BICUBIC, center=(cx, cy), fillcolor=fill).crop(box)


def piece_angle(category: str, anchors: dict | None, W: int, H: int) -> float:
    """How far the finger (ring) or forearm (bracelet) leans from upright, in degrees
    counter-clockwise to turn it upright. The AI draws rings upright: given a
    leaning finger it tends to straighten the ring, i.e. move the design."""
    ids = {"ring": (13, 14), "bracelet": (0, 9)}.get(category)
    lm = (anchors or {}).get("landmarks") or []
    if not ids or len(lm) < 21:
        return 0.0
    (x0, y0), (x1, y1) = lm[ids[0]][:2], lm[ids[1]][:2]
    import math
    a = math.degrees(math.atan2((x1 - x0) * W, -(y1 - y0) * H))
    return a if abs(a) >= 8 else 0.0


# ---------- direct generation: model photo + design picture -> AI ----------

def _px(p, W, H):
    if isinstance(p, dict):
        return np.array([p["x"] * W, p["y"] * H], float)
    return np.array([p[0] * W, p[1] * H], float)


def spots(category: str, anchors: dict, W: int, H: int) -> list[dict]:
    """Where on this model photo the AI draws the piece, from the landmarks
    measured on the photo: one square crop per spot, with the spot itself
    (the base of the ring finger, an earlobe, the neck) and how far to turn
    the crop so a finger points up. Nothing is drawn here: the AI draws."""
    out = []

    def crop(cx, cy, side, angle=0.0):
        side = int(min(max(side, 64), W, H))
        left = int(min(max(cx - side / 2, 0), W - side))
        top = int(min(max(cy - side / 2, 0), H - side))
        out.append({"box": (left, top, left + side, top + side), "target": (cx - left, cy - top), "angle": angle})

    if category in ("ring", "bracelet"):
        lm = anchors.get("landmarks") or []
        if len(lm) < 21:
            raise AuthError(400, "No hand was measured on this model photo.")
        p = [_px(q, W, H) for q in lm]
        spacing = (np.linalg.norm(p[13] - p[9]) + np.linalg.norm(p[13] - p[17])) / 2
        if category == "ring":
            at = p[13] + 0.57 * (p[14] - p[13])          # where a ring sits (same as the live try-on)
            crop(at[0], at[1], max(spacing * 4.5, 256), piece_angle("ring", anchors, W, H))
        else:
            at = p[0] + 0.15 * (p[0] - p[9])             # just past the wrist
            crop(at[0], at[1], max(np.linalg.norm(p[5] - p[17]) * 3.0, 300), piece_angle("bracelet", anchors, W, H))
    elif category == "earrings":
        mm = float(anchors.get("pxPerMm") or 0) or W / 160
        for lobe in ("lobeL", "lobeR"):
            at = _px(anchors[lobe], W, H)
            crop(at[0], at[1] + 6 * mm, max(42 * mm, 224))
    else:   # pendant, necklace
        mm = float(anchors.get("pxPerMm") or 0) or W / 160
        notch = _px(anchors["notch"], W, H)
        neck = abs(anchors["neckR"]["x"] - anchors["neckL"]["x"]) * W
        drop = 32 if category == "pendant" else 12      # mm below the neck notch the piece hangs
        crop(notch[0], notch[1] + drop * mm, max(neck * 2.6, 320))
    return out


def prompt_for(category: str, metal: str) -> str:
    """One instruction: put THIS design (image 2) on the model (image 1), at the
    spot in the centre of image 1 (the crop is centred on it)."""
    metal_words = sketch.METAL_WORDS.get(metal, "gold")
    where = {
        "ring": ("Image 1 is a close-up photograph of a hand. Put the ring shown in image 2 on the finger in the "
                 "centre of image 1, at the base of that finger, worn naturally: the band wraps around the finger and "
                 "the finger hides the back of the band, at the real size of a ring on that finger"),
        "bracelet": ("Image 1 is a close-up photograph of a hand and wrist. Put the bracelet shown in image 2 around "
                     "the wrist in the centre of image 1, worn naturally: it wraps around the wrist and the wrist hides "
                     "its back"),
        "earrings": ("Image 1 is a close-up photograph of an ear. Put one earring shown in image 2 (if it shows a pair, "
                     "use one of them) on the earlobe in the centre of image 1, worn naturally, hanging from the "
                     "earlobe at its real small size"),
        "pendant": ("Image 1 is a photograph of a model's neck and chest. Put the pendant shown in image 2 on her, on a "
                    f"fine {metal_words} chain around her neck, the pendant hanging in the centre of image 1 on the "
                    "skin of the chest at its real size"),
        "necklace": ("Image 1 is a photograph of a model's neck and chest. Put the necklace shown in image 2 around her "
                     "neck, lying on the skin, centred in image 1, at its real size"),
    }.get(category, "Put the jewellery shown in image 2 on the model in image 1, in the centre")
    return (
        f"{where}. The jewellery must be exactly the design in image 2: the same shape, the same stones (cut, number "
        f"and placement), the same setting and {metal_words} metal colour. Do not redesign it, do not add or remove "
        "stones. Add natural contact shadows where the metal touches the skin, reflections in the polished metal, "
        "real diamond sparkle, and match the light direction, colour, softness and grain of the photograph. Do not "
        "change the person: keep the skin, nails, hair, clothes, background and framing exactly as in image 1. "
        "Photorealistic, high-end jewellery campaign photo."
    )


ZONES = {   # half-width, half-height of the area the piece may take, share of the crop (finger upright)
    "ring": (0.22, 0.13), "bracelet": (0.34, 0.17), "earrings": (0.17, 0.30),
    "pendant": (0.45, 0.45), "necklace": (0.48, 0.45),
}


def _disk(shape, cx, cy, r) -> np.ndarray:
    yy, xx = np.ogrid[:shape[0], :shape[1]]
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r


def assess(orig: Image.Image, made: Image.Image, target: tuple[float, float], category: str) -> dict:
    """Did the AI draw the piece AT the spot, and leave the model alone?

      skin    structure of the photo around the spot kept (blurred NCC); a hand
              moved by 12 px scores ~0.6, a kept one ~0.99
      stray   share of the photo, away from the spot, that the AI changed a lot
              (a ring drawn on another finger, a second pendant)
      drawn   px the AI added at the spot (0: it drew nothing there)
      off     how far the drawn piece's centre is from the spot, share of the crop
    Also returns "matched" (the AI picture colour-matched to the photo), "take"
    (0..1, where its pixels are taken back) and "piece" (the drawn piece, cut
    out, for the design check)."""
    a = np.asarray(orig.convert("RGB"), np.float32)
    b = np.asarray(made.convert("RGB"), np.float32)
    n = a.shape[0]
    tx, ty = target
    # where the piece may be drawn, around the spot: a ring / bracelet goes across the
    # (upright) finger / wrist, an earring hangs from the lobe, a pendant / necklace
    # spreads over the chest; the AI also redraws finger outlines a little, which
    # must stay the photo's own
    rx, ry = ZONES.get(category, (0.3, 0.3))
    yy, xx = np.ogrid[:n, :n]
    zone = ((xx - tx) / (rx * n)) ** 2 + ((yy - ty) / (ry * n)) ** 2 <= 1
    reach = max(rx, ry) * n
    skin = ~_disk(a.shape[:2], tx, ty, reach * 1.25)
    if skin.sum() < 0.1 * n * n:
        skin = ~zone
    ga, gb = _blur(a.mean(-1), 2), _blur(b.mean(-1), 2)
    skin_ncc = _ncc(ga[skin], gb[skin])
    matched = b.copy()
    for c in range(3):
        sa, sb = a[..., c][skin], b[..., c][skin]
        g = float(np.clip(sa.std() / (sb.std() + 1e-6), 0.7, 1.4))
        matched[..., c] = (b[..., c] - sb.mean()) * g + sa.mean()
    matched = matched.clip(0, 255)
    diff = np.abs(_blur(matched, 1.5) - _blur(a, 1.5)).sum(-1)
    stray = float(((diff > 70) & skin).mean())
    changed = diff > 28
    changed = np.asarray(_mask_img(changed).filter(ImageFilter.MinFilter(3)).filter(ImageFilter.MaxFilter(3))) > 0
    seed = changed & _disk(a.shape[:2], tx, ty, 0.12 * n)
    take = _grow_into(seed, changed & zone) if seed.any() else seed
    drawn = int(take.sum())
    if drawn:
        ys, xs = np.nonzero(take)
        off = float(np.hypot(xs.mean() - tx, ys.mean() - ty) / n)
        strong = take & (diff > 60)   # the metal and stones, not the soft shadow around them
        if strong.sum() >= 30:
            ys, xs = np.nonzero(strong)
        pad = int(0.1 * max(np.ptp(xs), np.ptp(ys))) + 4
        piece = Image.fromarray(matched.astype(np.uint8)).crop(
            (max(0, xs.min() - pad), max(0, ys.min() - pad), min(n, xs.max() + pad), min(n, ys.max() + pad)))
    else:
        off, piece = 1.0, None
    takef = np.asarray(_dilate(_mask_img(take), 2).filter(ImageFilter.GaussianBlur(max(1.5, n * 0.004))), np.float32) / 255
    ok = (skin_ncc >= float(os.environ.get("TRYON_MIN_SKIN", "0.85")) and stray <= 0.01
          and drawn >= max(40, int(0.0015 * n * n)) and off <= 0.18)
    return {"skin": round(skin_ncc, 3), "stray": round(stray, 4), "drawn": drawn, "off": round(off, 3), "ok": ok,
            "matched": Image.fromarray(matched.astype(np.uint8)), "take": takef, "piece": piece}


def _plain(score: dict) -> dict:
    return {k: v for k, v in score.items() if k not in ("matched", "take", "piece")}


def generate(base: Image.Image, category: str, metal: str, anchors: dict, reference: Image.Image | None,
             ai=call_ai, judge=None, stage=lambda s: None, tries: int = TRIES) -> tuple[Image.Image, dict]:
    """The model photo with the design drawn on it by the AI in one step.

    For each spot: the crop around it (turned so a finger points up) and the
    design's catalogue picture go to the AI with one instruction. Its picture
    must pass assess() (drawn at the spot, model unchanged) and, with a judge,
    the design check (the drawn piece is recognised as THIS design among all
    designs of its kind). Only the piece the AI drew is taken back into the
    photo: every other pixel is the original model photo."""
    base = base.convert("RGB")
    W, H = base.size
    if reference is None:
        raise AuthError(404, "This design has no catalogue picture for the try-on.")
    ref = _square_ref(reference)
    prompt = prompt_for(category, metal)
    shown = base.copy()
    report = {"spots": []}
    for si, s in enumerate(spot_list := spots(category, anchors, W, H)):
        box, angle = s["box"], s["angle"]
        side = box[2] - box[0]
        gen = GEN_SMALL if side < SMALL_CROP else GEN
        fill = tuple(int(v) for v in np.asarray(base.crop(box)).reshape(-1, 3).mean(0))
        orig = _upright(base, angle, box, fill).resize((gen, gen), Image.LANCZOS)
        tx, ty = (gen / 2, gen / 2) if angle else (s["target"][0] * gen / side, s["target"][1] * gen / side)
        if angle:   # turning is about the crop centre; the spot moves with it
            import math
            dx, dy = s["target"][0] - side / 2, s["target"][1] - side / 2
            t = math.radians(angle)
            tx = (side / 2 + dx * math.cos(t) + dy * math.sin(t)) * gen / side
            ty = (side / 2 - dx * math.sin(t) + dy * math.cos(t)) * gen / side
        best = None
        for k in range(tries):
            stage(f"ai {si + 1}/{len(spot_list)} try {k + 1}")
            made = ai([orig, ref.resize((gen, gen), Image.LANCZOS)],
                      prompt if k == 0 else "Most important: put exactly the design of image 2 at the centre of image 1 "
                      "and change nothing else. " + prompt, secrets.randbelow(2**31))
            if made.size != (gen, gen):
                made = made.resize((gen, gen), Image.LANCZOS)
            score = assess(orig, made, (tx, ty), category)
            if score["ok"] and judge is not None and score["piece"] is not None:
                score["design"] = judge(score["piece"])
                score["ok"] = bool(score["design"]["ok"])
            if os.environ.get("TRYON_DEBUG_DIR"):   # tuning only: every crop, AI picture and score
                dbg = Path(os.environ["TRYON_DEBUG_DIR"])
                dbg.mkdir(parents=True, exist_ok=True)
                tag = f"{int(time.time())}-{si}-{k}"
                orig.save(dbg / f"{tag}-model.png")
                made.save(dbg / f"{tag}-ai.png")
                (dbg / f"{tag}-score.json").write_text(json.dumps(_plain(score)))
            rank = (score["ok"], score["skin"] - score["off"])
            if best is None or rank > (best["ok"], best["skin"] - best["off"]):
                best = score
            if score["ok"]:
                break
        report["spots"].append(_plain(best))
        if not best["ok"]:
            raise sketch.SketchError("The AI could not put this design on this model accurately. Try again, "
                                     "or pick another model.", 422)
        made_full = _turn_back(best["matched"].resize((side, side), Image.LANCZOS), angle, box, base.size, fill)
        mask = _turn_back(Image.fromarray((best["take"] * 255).astype(np.uint8)).resize((side, side), Image.LANCZOS),
                          angle, box, base.size, 0)
        region = shown.crop(box)
        region.paste(made_full, (0, 0), mask)
        shown.paste(region, box[:2])
    return shown, report


# ---------- results + background jobs ----------

def _rdir(uid: str) -> Path:
    return _udir(RESULTS, uid)


def save_result(uid: str, shown: Image.Image, meta: dict) -> dict:
    d = _rdir(uid)
    d.mkdir(parents=True, exist_ok=True)
    os.chmod(d, 0o700)
    rid = secrets.token_hex(8)
    shown.save(d / f"{rid}.jpg", "JPEG", quality=97, subsampling=0)
    meta = {**meta, "id": rid, "created": time.time()}
    (d / f"{rid}.json").write_text(json.dumps(meta))
    return public_result(meta)


def public_result(meta: dict) -> dict:
    rid = meta["id"]
    return {"id": rid, "design": meta["design"], "design_id": meta["design_id"], "metal": meta["metal"],
            "model": meta["model"], "label": meta.get("label", ""), "created": meta["created"],
            "url": f"/api/tryon/result/{rid}.jpg"}


def results(uid: str, design: int | None = None, limit: int = 40) -> list[dict]:
    d = _rdir(uid)
    if not d.is_dir():
        return []
    out = []
    for f in sorted(d.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            meta = json.loads(f.read_text())
        except (OSError, ValueError):
            continue
        if design is None or meta.get("design") == design:
            out.append(public_result(meta))
        if len(out) >= limit:
            break
    return out


def result_file(uid: str, rid: str) -> Path:
    if not re.fullmatch(r"[0-9a-f]{16}", rid):
        raise AuthError(404, "Not found")
    f = _rdir(uid) / f"{rid}.jpg"
    if not f.is_file():
        raise AuthError(404, "Not found")
    return f


def delete_result(uid: str, rid: str):
    result_file(uid, rid)
    for suffix in (".jpg", ".exact.jpg", ".json"):   # .exact.jpg: pictures from the earlier two-step try-on
        (_rdir(uid) / f"{rid}{suffix}").unlink(missing_ok=True)


class Jobs:
    """Pictures being made, polled by the page (each takes minutes; the tunnel
    cuts requests at 100 s). Jobs wait their turn for the one AI on this Mac."""

    def __init__(self, ai=call_ai):
        self.lock = threading.Lock()
        self.jobs: dict[str, dict] = {}
        self.ai = ai

    def active(self, uid: str) -> int:
        with self.lock:
            return sum(1 for j in self.jobs.values() if j["uid"] == uid and j["status"] in ("queued", "running"))

    def start(self, uid: str, work) -> dict:
        """work(stage) -> result dict, run in its own thread."""
        if self.active(uid) >= ACTIVE_MAX:
            raise AuthError(429, "Many pictures are already being made. Wait for some to finish.")
        jid = secrets.token_urlsafe(9)
        cut = time.time() - 3 * 3600
        with self.lock:
            for k in [k for k, j in self.jobs.items() if j["ts"] < cut and j["status"] not in ("queued", "running")]:
                del self.jobs[k]
            self.jobs[jid] = {"id": jid, "uid": uid, "ts": time.time(), "status": "queued", "stage": "queued",
                              "result": None, "error": None}

        def set_stage(s: str):
            with self.lock:
                self.jobs[jid].update(status="running", stage=s)

        def run():
            try:
                res = work(set_stage)
                with self.lock:
                    self.jobs[jid].update(status="done", stage="done", result=res)
            except (sketch.SketchError, AuthError) as e:
                with self.lock:
                    self.jobs[jid].update(status="error", error=e.message)
            except Exception as e:   # never leave the page polling forever
                print(f"try-on job failed: {e!r}", flush=True)
                with self.lock:
                    self.jobs[jid].update(status="error", error="The picture could not be made. Try again.")
        threading.Thread(target=run, daemon=True).start()
        return self.get(jid, uid)

    def get(self, jid: str, uid: str) -> dict | None:
        with self.lock:
            j = self.jobs.get(jid)
            return {k: v for k, v in j.items() if k != "uid"} if j and j["uid"] == uid else None


def make(uid: str, ref: str, design: dict, metal: str, reference: Image.Image | None,
         ai=call_ai, judge=None, stage=lambda s: None) -> dict:
    """One picture: the design (its catalogue picture) on one model photo, drawn by
    the AI in one step. design: uid, design_id, category."""
    photo, item = resolve(uid, ref)
    part = tryon.PART_FOR.get(design["category"])
    if part not in item["parts"]:   # never a ring on an ear, whatever the page sends
        raise AuthError(400, "This model photo doesn't fit this piece.")
    stage("queued")
    with Image.open(photo) as im:
        base = ImageOps.exif_transpose(im).convert("RGB")
    shown, report = generate(base, design["category"], metal, item["parts"][part], reference, ai=ai, judge=judge,
                             stage=stage)
    return save_result(uid, shown, {
        "design": design["uid"], "design_id": design["design_id"], "metal": metal, "model": ref,
        "label": item.get("label", ""), "checks": report["spots"]})
