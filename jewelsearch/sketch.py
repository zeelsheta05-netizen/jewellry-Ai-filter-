"""Sketch to Design: a hand sketch (or photo, or words) -> a photoreal jewellery
design, made on the server through one wrapper over two image providers:

- Pollinations (gen.pollinations.ai, POLLINATIONS_KEY): a free account's quest
  credits pay for FLUX Kontext / FLUX.2 Klein / GPT Image mini (image-to-image);
  paid credits there also give Google's Nano Banana models.
- Google direct (GEMINI_API_KEY): the Nano Banana models; Google has no free tier
  for image models, so the key's project needs billing.

Shoppers never leave the page: the server sends the request and keeps the result.

Every paid call costs money, so this module spends as little as possible:

1. The prompt is built here, from a fixed template and the shopper's picks, with
   no AI text call. Notes in Gujarati / Hindi / English are read by the local
   parser (query.py) for type and metal, so the image model gets short, exact
   English. Prompts stay around 80-120 words.
2. The picture is shrunk to at most 1024 px JPEG before sending (input tokens and
   upload time), and an empty canvas is refused before any call.
3. One picture per call, 1:1, 1K, a free-credit model by default.
4. Same sketch + same choices -> the stored picture comes back, free (cache).
5. "Change this" edits the last picture with a one-line instruction instead of a
   new full generation from scratch.
6. Budget guard before every call: per-user daily count, a daily and a monthly
   dollar cap (.env), one call at a time per user.
7. Only models whose provider has a key are offered; free-credit models first.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import re
import secrets
import sqlite3
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import httpx
from PIL import Image, ImageOps, ImageStat, UnidentifiedImageError

from . import memory
from .config import DATA, ROOT

DIR = DATA / "sketch"
OUT = DIR / "out"
LEDGER = DIR / "ledger.sqlite"
API_BASE = "https://generativelanguage.googleapis.com/v1beta"
SEND_SIDE = 1024          # longest side of the picture sent to the model
MAX_BYTES = 8 * 1024 * 1024
NOTES_MAX = 300
BLANK_STD = 3.0           # a canvas this flat has nothing drawn on it


class SketchError(Exception):
    def __init__(self, message: str, status: int = 400, reason: str = ""):
        super().__init__(message)
        self.message, self.status, self.reason = message, status, reason   # reason "balance": the account ran out


# Prices per picture, read 2026-10-07: Google = ai.google.dev/gemini-api/docs/pricing
# (Standard tier, plus a small allowance for the input); Pollinations =
# gen.pollinations.ai/image/models in pollen (1 pollen is about 1 USD: their
# Kontext 0.03 vs ~0.04 USD elsewhere). "free" = quest credits of a free account pay for it.
@dataclass(frozen=True)
class Model:
    key: str
    label: str
    provider: str      # "pollinations" | "google"
    api: str
    size: str
    usd: float
    note: str = ""
    free: bool = False
    quality: str = ""  # Pollinations gptimage: low / medium / high


MODELS = [
    # on our own Mac (FLUX.2 Klein 4B, 4-bit, mflux in .venv-imagegen): no account, no daily limit, a few minutes each
    Model("local-klein", "FLUX.2 Klein · our server", "local", "flux2-klein-4b", "768", 0.0, "Runs on our computer · about 3-4 min", True),
    Model("local-klein-hd", "FLUX.2 Klein HD · our server", "local", "flux2-klein-4b", "1024", 0.0, "Sharper · about 5-7 min", True),
    Model("p-klein", "FLUX.2 Klein · 1K", "pollinations", "klein", "1K", 0.005, "Fast, good first try", True),
    Model("p-kontext", "FLUX Kontext · 1K", "pollinations", "kontext", "1K", 0.03, "Follows the sketch well", True),
    Model("p-gptimage", "GPT Image mini · 1K", "pollinations", "gptimage", "1K", 0.012, "Keeps the sketch's view", True, "medium"),
    Model("p-nanobanana", "Nano Banana · 1K (Pollinations)", "pollinations", "nanobanana", "1K", 0.041, "Google model"),
    Model("p-nanobanana-2", "Nano Banana 2 · 1K (Pollinations)", "pollinations", "nanobanana-2", "1K", 0.067, "Google model, sharper"),
    Model("p-nanobanana-pro", "Nano Banana Pro · 1K (Pollinations)", "pollinations", "nanobanana-pro", "1K", 0.142, "Highest quality"),
    Model("lite-1k", "Nano Banana 2 Lite · 1K", "google", "gemini-3.1-flash-lite-image", "1K", 0.034, "Fast"),
    Model("nb21-1k", "Nano Banana 2.1 · 1K", "google", "gemini-nano-banana-2.1", "1K", 0.036, "Best value, sharper detail"),
    Model("nb21-2k", "Nano Banana 2.1 · 2K", "google", "gemini-nano-banana-2.1", "2K", 0.053, "Final picture"),
    Model("nb21-4k", "Nano Banana 2.1 · 4K", "google", "gemini-nano-banana-2.1", "4K", 0.116, "Print / catalogue"),
    Model("pro-2k", "Nano Banana Pro · 2K", "google", "gemini-3-pro-image", "2K", 0.138, "Highest quality"),
    Model("pro-4k", "Nano Banana Pro · 4K", "google", "gemini-3-pro-image", "4K", 0.244, "Highest quality, print"),
]
BY_KEY = {m.key: m for m in MODELS}
PROVIDERS = {"local": ("", "our server"), "pollinations": ("POLLINATIONS_KEY", "Pollinations"), "google": ("GEMINI_API_KEY", "Google")}


TYPES = {   # value -> words for the model
    "auto": None, "ring": "ring", "earrings": "pair of earrings", "pendant": "pendant on a fine chain",
    "necklace": "necklace", "bracelet": "bracelet", "bangle": "bangle", "brooch": "brooch",
    "other": "piece of jewellery",
}
TYPE_LABELS = {"auto": "Auto-detect", "ring": "Ring", "earrings": "Earrings", "pendant": "Pendant",
               "necklace": "Necklace", "bracelet": "Bracelet", "bangle": "Bangle", "brooch": "Brooch",
               "other": "Other"}
BACKGROUNDS = {   # value -> (label, words for the model)
    "luxury": ("Luxury jewellery display", "on an elegant jewellery display stand, soft boutique lighting"),
    "white": ("White studio", "on a pure white seamless studio background, soft even light, catalogue style"),
    "velvet": ("Black velvet", "on black velvet, dramatic soft spotlight"),
    "nature": ("Nature backdrop", "on a natural stone surface with soft blurred greenery behind"),
    "water": ("Water reflection", "on a glossy wet surface with a gentle water reflection"),
    "rustic": ("Rustic elegance", "on aged wood and linen, warm natural light"),
    "geometric": ("Geometric modern", "on minimal geometric plinths in soft neutral tones"),
    "custom": ("Custom background…", None),
}
METAL_WORDS = {"rose_gold": "18k rose gold", "white_gold": "18k white gold", "yellow_gold": "18k yellow gold"}
SPARKLE = "Brilliant diamond sparkle with crisp facets and polished metal highlights."


def _key(provider: str) -> str:
    if provider == "local":
        return "on" if local_ready() else ""
    return os.environ.get(PROVIDERS[provider][0], "").strip()


def connected() -> list[str]:
    return [p for p in PROVIDERS if _key(p)]


def available() -> list[Model]:
    """Models of connected providers. Pollinations' paid-credit models (Nano Banana
    there) only when POLLINATIONS_PAID=1: a free account gets 402 for them."""
    on = set(connected())
    paid = os.environ.get("POLLINATIONS_PAID", "").strip() == "1"
    return [m for m in MODELS if m.provider in on and (m.free or m.provider != "pollinations" or paid)]


def default_model() -> str | None:
    models = available()
    return models[0].key if models else None


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


def limits() -> dict:
    return {"per_user_day": int(_env_float("SKETCH_PER_USER_DAY", 0)),   # 0 = no per-user daily limit
            "usd_day": _env_float("SKETCH_USD_DAY", 2.0),
            "usd_month": _env_float("SKETCH_USD_MONTH", 10.0),
            "inr_per_usd": _env_float("SKETCH_INR_PER_USD", 88.0)}


# ---------- the prompt (no AI call) ----------

def _clean(text: str, n: int) -> str:
    return " ".join((text or "").split())[:n]


def build_prompt(kind: str = "auto", background: str = "luxury", custom_bg: str = "", notes: str = "",
                 sparkle: bool = True, has_picture: bool = True) -> str:
    """Short, fixed-order prompt. Type and metal written in any language in the
    notes are turned into English words by the local parser."""
    if kind not in TYPES:
        raise SketchError("Pick a jewellery type from the list.")
    if background not in BACKGROUNDS:
        raise SketchError("Pick a background from the list.")
    notes = _clean(notes, NOTES_MAX)
    noun, metal = TYPES[kind], None
    if notes:
        from .query import parse   # local lexicon, no model
        q = parse(notes)
        metal = METAL_WORDS.get(q.metal or "")
        if noun is None and q.category:
            noun = TYPES.get(q.category) or q.category
    lines = []
    if has_picture:
        what = f"a {noun}" if noun else "the jewellery piece drawn"
        lines.append(f"Turn this hand sketch into a photorealistic product photo of {what}.")
        lines.append("Follow the sketch's outline, proportions, motifs and stone positions exactly; "
                     "do not add or remove elements.")
    else:
        lines.append(f"Photorealistic product photo of a {noun or 'piece of jewellery'}.")
    if metal:
        lines.append(f"Metal: {metal}.")
    if notes:
        lines.append(f"Details: {notes}")
    bg = _clean(custom_bg, 120) if background == "custom" else BACKGROUNDS[background][1]
    if background == "custom" and not bg:
        raise SketchError("Describe the custom background, or pick one from the list.")
    lines.append(f"Background: {bg}.")
    if sparkle:
        lines.append(SPARKLE)
    lines.append("One piece only, centred, sharp focus, realistic proportions for real jewellery manufacture. "
                 "No text, no hands, no watermark.")
    return "\n".join(lines)


def edit_prompt(change: str) -> str:
    change = _clean(change, 200)
    if not change:
        raise SketchError("Write what to change.")
    return (f"Edit this jewellery photo: {change}. Keep the design, angle, lighting and background otherwise "
            "exactly the same. No text, no watermark.")


# ---------- the picture ----------

def from_data_url(s: str) -> bytes:
    head, _, body = (s or "").partition(",")
    if not (head.startswith("data:image/") and head.endswith(";base64")):
        raise SketchError("Send the sketch as an image.")
    try:
        data = base64.b64decode(body, validate=True)
    except ValueError:
        raise SketchError("The image could not be read. Try another file.")
    if len(data) > MAX_BYTES:
        raise SketchError("The image is too large. Use one under 8 MB.", 413)
    return data


def prepare(data: bytes) -> bytes:
    """Any upload -> upright RGB JPEG, longest side <= SEND_SIDE. Strips metadata.
    Refuses an empty canvas (no credit spent on nothing)."""
    try:
        im = Image.open(io.BytesIO(data))
        if im.format not in ("PNG", "JPEG", "WEBP"):
            raise SketchError("Use a JPG, PNG or WebP image.")
        if im.width * im.height > 40_000_000:
            raise SketchError("The image has too many pixels. Use a smaller one.", 413)
        im = ImageOps.exif_transpose(im)
        if im.mode in ("RGBA", "LA", "P"):
            im = im.convert("RGBA")
            flat = Image.new("RGB", im.size, "white")   # transparent sketch -> on white paper
            flat.paste(im, mask=im.split()[-1])
            im = flat
        im = im.convert("RGB")
    except SketchError:
        raise
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
        raise SketchError("This file isn't an image that can be read. Use a JPG, PNG or WebP.")
    if min(im.size) < 64:
        raise SketchError("The image is too small. Use one at least 64 px wide.")
    im.thumbnail((SEND_SIDE, SEND_SIDE), Image.LANCZOS)
    if max(ImageStat.Stat(im.convert("L")).stddev) < BLANK_STD:
        raise SketchError("The sketch is empty. Draw or add a picture first.")
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=88, optimize=True)
    return buf.getvalue()


# ---------- Google call ----------

def _find_image(obj):
    """First base64 picture in a response, for both the Interactions and the
    generateContent shapes."""
    if isinstance(obj, dict):
        inline = obj.get("inlineData") or obj.get("inline_data")
        if isinstance(inline, dict) and inline.get("data"):
            return inline["data"], inline.get("mimeType") or inline.get("mime_type") or "image/png"
        if obj.get("type") == "image" and isinstance(obj.get("data"), str):
            return obj["data"], obj.get("mime_type") or "image/png"
        for v in obj.values():
            hit = _find_image(v)
            if hit:
                return hit
    elif isinstance(obj, list):
        for v in obj:
            hit = _find_image(v)
            if hit:
                return hit
    return None


def _google_error(r: httpx.Response) -> SketchError:
    try:
        msg = r.json().get("error", {}).get("message", "")
    except ValueError:
        msg = ""
    low = msg.lower()
    if r.status_code == 429 or "quota" in low:
        return SketchError("The AI service limit is reached. Try again later.", 429)
    if r.status_code in (401, 403) or "api key" in low:
        return SketchError("Google refused the API key. Check GEMINI_API_KEY in .env.", 502)
    if "billing" in low or "free tier" in low:
        return SketchError("Image models are not enabled for this Google key.", 502)
    if "safety" in low or "blocked" in low:
        return SketchError("Google declined this picture or text. Change the notes and try again.", 422)
    return SketchError(f"Google could not make the picture ({r.status_code}). {msg[:160]}".strip(), 502)


def call_google(model: Model, prompt: str, picture: bytes | None) -> tuple[bytes, str]:
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        raise SketchError("Google is not connected on this server (GEMINI_API_KEY).", 503)
    b64 = base64.b64encode(picture).decode() if picture else None
    head = {"x-goog-api-key": key, "Content-Type": "application/json"}
    body = {"model": model.api,
            "input": [{"type": "text", "text": prompt}] + ([{"type": "image", "mime_type": "image/jpeg", "data": b64}] if b64 else []),
            "response_format": {"type": "image", "mime_type": "image/png", "aspect_ratio": "1:1", "image_size": model.size}}
    with httpx.Client(timeout=httpx.Timeout(180, connect=15)) as c:
        r = c.post(f"{API_BASE}/interactions", headers=head, json=body)
        if r.status_code == 404:   # older endpoint for keys/models without the Interactions API
            parts = [{"text": prompt}] + ([{"inline_data": {"mime_type": "image/jpeg", "data": b64}}] if b64 else [])
            r = c.post(f"{API_BASE}/models/{model.api}:generateContent", headers=head,
                       json={"contents": [{"parts": parts}],
                             "generationConfig": {"responseModalities": ["IMAGE"],
                                                  "imageConfig": {"aspectRatio": "1:1", "imageSize": model.size}}})
    if r.status_code != 200:
        raise _google_error(r)
    hit = _find_image(r.json())
    if not hit:
        raise SketchError("Google answered without a picture (it may have declined the request). "
                          "Change the notes and try again.", 502)
    try:
        return base64.b64decode(hit[0]), hit[1]
    except ValueError:
        raise SketchError("Google's picture could not be read. Try again.", 502)


POLLINATIONS = "https://gen.pollinations.ai"
_POLL_MEDIA = re.compile(r"^https://([a-z0-9-]+\.)?pollinations\.ai/")


def _pollinations_error(r: httpx.Response, model: Model | None = None) -> SketchError:
    try:
        msg = (r.json().get("error") or {}).get("message", "")
    except (ValueError, AttributeError):
        msg = ""
    if r.status_code == 401:
        return SketchError("Pollinations refused the key. Check POLLINATIONS_KEY in .env.", 502)
    if r.status_code in (402, 403) and model is not None and not model.free:
        return SketchError(f"{model.label} is not available on this account. Pick another model.", 400)
    if r.status_code == 402:
        return SketchError("The AI service limit is reached. Try again later.", 429, "balance")
    if r.status_code == 403:
        return SketchError("This model is not available on this account. Pick another model.", 400)
    if r.status_code == 429:
        return SketchError("Too many pictures at once. Try again in a minute.", 429)
    if r.status_code == 422:
        return SketchError("The AI declined this picture or text. Change the notes and try again.", 422)
    return SketchError(f"The AI could not make the picture ({r.status_code}). {msg[:160]}".strip(), 502)


def call_pollinations(model: Model, prompt: str, picture: bytes | None) -> tuple[bytes, str]:
    """OpenAI-style endpoints: /v1/images/edits takes the sketch as an upload (no
    public link needed), /v1/images/generations for words only. Answer: base64."""
    key = _key("pollinations")
    if not key:
        raise SketchError("Pollinations is not connected on this server (POLLINATIONS_KEY).", 503)
    head = {"Authorization": f"Bearer {key}"}
    fields = {"prompt": prompt, "model": model.api, "size": "1024x1024", "n": "1", "response_format": "b64_json"}
    if model.quality:
        fields["quality"] = model.quality
    with httpx.Client(timeout=httpx.Timeout(240, connect=15)) as c:
        if picture:
            r = c.post(f"{POLLINATIONS}/v1/images/edits", headers=head, data=fields,
                       files={"image": ("sketch.jpg", picture, "image/jpeg")})
        else:
            r = c.post(f"{POLLINATIONS}/v1/images/generations", headers=head, json={**fields, "n": 1})
        if r.status_code != 200:
            raise _pollinations_error(r, model)
        try:
            item = (r.json().get("data") or [{}])[0]
        except (ValueError, AttributeError, IndexError):
            item = {}
        if item.get("b64_json"):
            try:
                data = base64.b64decode(item["b64_json"])
            except ValueError:
                raise SketchError("The AI's picture could not be read. Try again.", 502)
        elif isinstance(item.get("url"), str) and _POLL_MEDIA.match(item["url"]):   # only their own media host
            g = c.get(item["url"])
            if g.status_code != 200:
                raise SketchError("The AI's picture could not be downloaded. Try again.", 502)
            data = g.content
        else:
            raise SketchError("The AI answered without a picture. Change the notes and try again.", 502)
    try:
        fmt = Image.open(io.BytesIO(data)).format
    except (UnidentifiedImageError, OSError):
        raise SketchError("The AI's picture could not be read. Try again.", 502)
    return data, {"JPEG": "image/jpeg", "WEBP": "image/webp"}.get(fmt, "image/png")


PLANNER_MODEL = "openai"     # Pollinations alias of GPT-5.4 Nano: reads pictures, very cheap
PLANNER_USD = 0.001          # allowance per planning call (~2k tokens in, ~1k out)


def call_pollinations_text(prompt: str, picture: bytes | None = None, max_tokens: int = 1800) -> str:
    """One chat answer from Pollinations' text model (optionally looking at a picture)."""
    key = _key("pollinations")
    if not key:
        raise SketchError("Pollinations is not connected on this server (POLLINATIONS_KEY).", 503)
    content = [{"type": "text", "text": prompt}]
    if picture:
        content.append({"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(picture).decode()}})
    body = {"model": PLANNER_MODEL, "messages": [{"role": "user", "content": content}], "max_tokens": max_tokens}
    with httpx.Client(timeout=httpx.Timeout(90, connect=15)) as c:
        r = c.post(f"{POLLINATIONS}/v1/chat/completions", headers={"Authorization": f"Bearer {key}"}, json=body)
    if r.status_code != 200:
        raise _pollinations_error(r)
    try:
        return r.json()["choices"][0]["message"]["content"] or ""
    except (ValueError, KeyError, IndexError, TypeError):
        raise SketchError("The AI planner answered in an unexpected way.", 502)


# ---------- on our own computer (mflux, separate process, one picture at a time) ----------

LOCAL_BIN = ROOT / ".venv-imagegen" / "bin"
LOCAL_WEIGHTS = "mflux-community/flux2-klein-4b-mflux-q4"   # Apache-2.0, 4-bit, 4.6 GB in the Hugging Face cache
LOCAL_TIMEOUT = 25 * 60
LOCAL_LOCK = threading.Lock()
_HF_CACHE = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"


def local_ready() -> bool:
    """The generator is installed and its weights are on disk (LOCAL_IMAGEGEN=0 switches it off)."""
    if os.environ.get("LOCAL_IMAGEGEN", "1").strip() == "0":
        return False
    weights = _HF_CACHE / ("models--" + LOCAL_WEIGHTS.replace("/", "--")) / "snapshots"
    return (LOCAL_BIN / "mflux-generate-flux2-edit").is_file() and weights.is_dir() and any(weights.iterdir())


def free_memory_gb() -> float:
    """Memory macOS can hand out right away (free + inactive + speculative pages)."""
    try:
        out = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=5).stdout
    except (OSError, subprocess.SubprocessError):
        return 99.0
    page = int(re.search(r"page size of (\d+)", out).group(1)) if "page size of" in out else 16384
    total = 0
    for name in ("Pages free", "Pages inactive", "Pages speculative"):
        m = re.search(name + r":\s+(\d+)", out)
        total += int(m.group(1)) if m else 0
    return total * page / 1e9


DRAW_LOCK_FILE = DIR / "drawing.lock"


class _MacLock:
    """One drawing at a time on this whole Mac, across processes (a second server, a
    library build, a test copy): two FLUX runs at once push a 16 GB Mac deep into swap
    and each takes several times longer (measured 2026-10-09)."""

    def __init__(self, path: Path):
        self.path, self.fh = path, None

    def acquire(self, timeout: float) -> bool:
        import fcntl
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fh = open(self.path, "a+")
        end = time.time() + timeout
        while True:
            try:
                fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
                self.fh = fh
                return True
            except BlockingIOError:
                if time.time() >= end:
                    fh.close()
                    return False
                time.sleep(2)

    def release(self):
        import fcntl
        if self.fh is not None:
            fcntl.flock(self.fh, fcntl.LOCK_UN)
            self.fh.close()
            self.fh = None


def _other_drawing() -> bool:
    """A FLUX run started outside the lock (a script run by hand)."""
    r = subprocess.run(["pgrep", "-f", r"Python.*bin/mflux-generate"], capture_output=True)
    return r.returncode == 0


class drawing:
    """Everything a local drawing needs around it, for every tool (sketch, variations,
    Design Generator, try-on): its turn on this Mac, the app's idle models out of memory
    (memory.py), enough free memory, and a timing line in the log."""

    def __init__(self, label: str, busy_message: str = "The design studio is busy with other designs. "
                                                         "Try again in a few minutes."):
        self.label, self.busy = label, busy_message
        self.mac = _MacLock(DRAW_LOCK_FILE)   # read at each drawing (tests point it elsewhere)

    def __enter__(self):
        t = time.time()
        if not LOCAL_LOCK.acquire(timeout=LOCAL_TIMEOUT):
            raise SketchError(self.busy, 429)
        if not self.mac.acquire(timeout=max(1.0, LOCAL_TIMEOUT - (time.time() - t))):
            LOCAL_LOCK.release()
            raise SketchError(self.busy, 429)
        try:
            for _ in range(int(LOCAL_TIMEOUT / 5)):   # a FLUX run outside the lock: let it finish first
                if not _other_drawing():
                    break
                time.sleep(5)
            memory.drawing_starts()   # the app's big idle models let go of their memory meanwhile
            need = float(os.environ.get("LOCAL_MIN_FREE_GB", "2.0"))
            for _ in range(12):   # up to 2 min for memory to free up
                if free_memory_gb() >= need:
                    break
                time.sleep(10)
            else:
                raise SketchError("The design studio is busy right now. Try again in a few minutes.", 429)
        except BaseException:
            self._release()
            raise
        self.waited, self.t0 = time.time() - t, time.time()
        return self

    def _release(self):
        memory.drawing_ends()
        self.mac.release()
        LOCAL_LOCK.release()

    def __exit__(self, *exc):
        print(f"local imagegen: {self.label} {time.time() - self.t0:.0f} s (waited {self.waited:.0f} s)", flush=True)
        self._release()
        return False


def call_local(model: Model, prompt: str, picture: bytes | None) -> tuple[bytes, str]:
    """FLUX.2 Klein on this Mac, as a separate process that only exists while it draws
    (measured next to the live app: 768 px 5.8 GB / ~3.5 min, 1024 px 10.3 GB / ~6 min)."""
    if not local_ready():
        raise SketchError("The design studio on our server is not installed.", 503)
    with drawing(f"{model.key}{' edit' if picture else ''}"):
        tmp = DIR / "tmp"
        tmp.mkdir(parents=True, exist_ok=True)
        tag = secrets.token_hex(6)
        src, out, txt = tmp / f"{tag}-in.jpg", tmp / f"{tag}-out.png", tmp / f"{tag}-prompt.txt"
        try:
            txt.write_text(prompt)
            size = int(model.size)
            cmd = [str(LOCAL_BIN / ("mflux-generate-flux2-edit" if picture else "mflux-generate-flux2")), "--low-ram",
                   "--model", LOCAL_WEIGHTS, "--base-model", model.api, "--prompt-file", str(txt),
                   "--width", str(size), "--height", str(size), "--steps", "4",
                   "--seed", str(secrets.randbelow(2**31)), "--output", str(out), "--no-exif"]
            if picture:
                with Image.open(io.BytesIO(picture)) as im:   # reference at the output size: less memory
                    im.convert("RGB").resize((size, size), Image.LANCZOS).save(src, "JPEG", quality=92)
                cmd[2:2] = ["--image-paths", str(src)]
            env = {**os.environ, "HF_HUB_OFFLINE": "1"}
            try:
                r = subprocess.run(cmd, capture_output=True, text=True, timeout=LOCAL_TIMEOUT, env=env)
            except subprocess.TimeoutExpired:
                raise SketchError("The design took too long on our server. Try again.", 502)
            if r.returncode != 0 or not out.is_file():
                print(f"local imagegen failed ({r.returncode}): {r.stderr[-600:]}", flush=True)
                raise SketchError("The design studio could not make the picture. Try again.", 502)
            return out.read_bytes(), "image/png"
        finally:
            for f in DIR.glob("tmp/*"):
                if f.is_file() and time.time() - f.stat().st_mtime > 60:
                    f.unlink(missing_ok=True)
            for f in (src, out, txt):
                f.unlink(missing_ok=True)


def call_provider(model: Model, prompt: str, picture: bytes | None) -> tuple[bytes, str]:
    if model.provider == "local":
        return call_local(model, prompt, picture)
    return (call_pollinations if model.provider == "pollinations" else call_google)(model, prompt, picture)


# ---------- ledger, cache, budget ----------

class Studio:
    """Stores every picture made (data/sketch/out) and every spend (ledger)."""

    def __init__(self, root: Path | None = None, caller=call_provider):
        self.root = Path(root) if root else DIR
        self.out = self.root / "out"
        self.out.mkdir(parents=True, exist_ok=True)
        self.caller = caller
        self.lock = threading.Lock()
        self.busy: set[str] = set()
        self.db = sqlite3.connect(self.root / "ledger.sqlite", check_same_thread=False)
        self.db.execute("""CREATE TABLE IF NOT EXISTS gens (
            id TEXT PRIMARY KEY, uid TEXT NOT NULL, ts REAL NOT NULL, model TEXT NOT NULL, usd REAL NOT NULL,
            source TEXT NOT NULL, cache_key TEXT, file TEXT NOT NULL, prompt TEXT, parent TEXT, deleted INTEGER DEFAULT 0)""")
        self.db.execute("CREATE INDEX IF NOT EXISTS gens_uid ON gens(uid, ts)")
        self.db.execute("CREATE INDEX IF NOT EXISTS gens_key ON gens(cache_key)")
        cols = {r[1] for r in self.db.execute("PRAGMA table_info(gens)")}
        if "panel" not in cols:   # which page made it: sketch / variation
            self.db.execute("ALTER TABLE gens ADD COLUMN panel TEXT DEFAULT 'sketch'")
        if "meta" not in cols:    # JSON: grid layout + captions for variation sets
            self.db.execute("ALTER TABLE gens ADD COLUMN meta TEXT")
        self.db.commit()

    # spending
    def spent(self, since: float, uid: str | None = None) -> tuple[float, int]:
        sql, args = "SELECT COALESCE(SUM(usd),0), COUNT(*) FROM gens WHERE ts>=? AND source='ai'", [since]
        if uid:
            sql += " AND uid=?"
            args.append(uid)
        with self.lock:
            usd, n = self.db.execute(sql, args).fetchone()
        return float(usd), int(n)

    @staticmethod
    def _day_start() -> float:
        t = time.localtime()
        return time.mktime((t.tm_year, t.tm_mon, t.tm_mday, 0, 0, 0, 0, 0, -1))

    @staticmethod
    def _month_start() -> float:
        t = time.localtime()
        return time.mktime((t.tm_year, t.tm_mon, 1, 0, 0, 0, 0, 0, -1))

    def budget(self, uid: str) -> dict:
        lim = limits()
        day_usd, _ = self.spent(self._day_start())
        month_usd, _ = self.spent(self._month_start())
        _, mine = self.spent(self._day_start(), uid)
        per = lim["per_user_day"] if lim["per_user_day"] > 0 else None   # None = no per-user limit
        return {"left_today": None if per is None else max(0, per - mine), "per_user_day": per,
                "usd_today": round(day_usd, 3), "usd_day": lim["usd_day"],
                "usd_month_spent": round(month_usd, 3), "usd_month": lim["usd_month"]}

    def public_budget(self, uid: str) -> dict:
        """What the page shows: only the person's designs left today (no money)."""
        b = self.budget(uid)
        return {"left_today": b["left_today"], "per_user_day": b["per_user_day"]}

    def check_budget(self, uid: str, usd: float):
        b = self.budget(uid)
        if b["left_today"] is not None and b["left_today"] <= 0:
            raise SketchError(f"You have made {b['per_user_day']} designs today, the daily limit. Try again tomorrow.", 429)
        if b["usd_today"] + usd > b["usd_day"]:
            raise SketchError("Today's design limit is reached. Try again tomorrow.", 429)
        if b["usd_month_spent"] + usd > b["usd_month"]:
            raise SketchError("This month's design limit is reached.", 429)

    # storage
    def _save(self, uid, model, usd, source, cache_key, data, mime, prompt, parent=None, panel="sketch", meta=None) -> dict:
        ext = {"image/jpeg": "jpg", "image/webp": "webp"}.get(mime, "png")
        gid = secrets.token_urlsafe(9)
        name = f"{gid}.{ext}"
        (self.out / name).write_bytes(data)
        with self.lock:
            self.db.execute("INSERT INTO gens(id,uid,ts,model,usd,source,cache_key,file,prompt,parent,panel,meta) "
                            "VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                            (gid, uid, time.time(), model, usd, source, cache_key, name, prompt, parent, panel,
                             json.dumps(meta) if meta else None))
            self.db.commit()
        return self.row(gid, uid)

    def _cached(self, cache_key: str) -> sqlite3.Row | None:
        with self.lock:
            r = self.db.execute("SELECT file, model FROM gens WHERE cache_key=? ORDER BY ts LIMIT 1", (cache_key,)).fetchone()
        return r if r and (self.out / r[0]).is_file() else None

    def row(self, gid: str, uid: str) -> dict | None:
        with self.lock:
            r = self.db.execute("SELECT id,ts,model,usd,source,file,prompt,parent,panel,meta FROM gens "
                                "WHERE id=? AND uid=? AND deleted=0", (gid, uid)).fetchone()
        if not r:
            return None
        m = BY_KEY.get(r[2])
        return {"id": r[0], "ts": r[1], "model": r[2], "model_label": m.label if m else r[2],
                "source": r[4], "image": f"/api/sketch/img/{r[0]}", "prompt": r[6], "parent": r[7],
                "panel": r[8] or "sketch", "meta": json.loads(r[9]) if r[9] else None}

    def file(self, gid: str, uid: str) -> Path | None:
        with self.lock:
            r = self.db.execute("SELECT file FROM gens WHERE id=? AND uid=? AND deleted=0", (gid, uid)).fetchone()
        p = self.out / r[0] if r else None
        return p if p and p.is_file() else None

    def mine(self, uid: str, limit: int = 60, panel: str | None = None) -> list[dict]:
        sql, args = "SELECT id FROM gens WHERE uid=? AND deleted=0", [uid]
        if panel:
            sql += " AND COALESCE(panel,'sketch')=?"
            args.append(panel)
        with self.lock:
            ids = [r[0] for r in self.db.execute(sql + " ORDER BY ts DESC LIMIT ?", args + [limit])]
        return [x for x in (self.row(i, uid) for i in ids) if x]

    def delete(self, gid: str, uid: str) -> bool:
        """Hides it from the gallery. The file stays (another request may share it
        through the cache) and the spend stays in the ledger."""
        with self.lock:
            n = self.db.execute("UPDATE gens SET deleted=1 WHERE id=? AND uid=?", (gid, uid)).rowcount
            self.db.commit()
        return n > 0

    # making pictures
    def _run(self, uid: str, model_key: str, prompt: str, picture: bytes | None, parent: str | None = None,
             panel: str = "sketch", salt: str = "") -> dict:
        model = BY_KEY.get(model_key)
        if not model:
            raise SketchError("Pick a model from the list.")
        parts = [model.api.encode(), model.size.encode(), prompt.encode(), picture or b""]
        if salt:   # a second design from the same words must be a new picture, not the stored one
            parts.append(salt.encode())
        cache_key = hashlib.sha256(b"\0".join(parts)).hexdigest()
        with self.lock:
            own = self.db.execute("SELECT id FROM gens WHERE cache_key=? AND uid=? AND deleted=0 ORDER BY ts DESC LIMIT 1",
                                  (cache_key, uid)).fetchone()
        if own and self.file(own[0], uid):   # they made this exact one already: show it again, no new entry
            return {**self.row(own[0], uid), "cached": True}
        hit = self._cached(cache_key)
        if hit:   # same sketch, same choices: the stored picture, no credit used
            data = (self.out / hit[0]).read_bytes()
            mime = "image/jpeg" if hit[0].endswith(".jpg") else "image/png"
            res = self._save(uid, model.key, 0.0, "cache", cache_key, data, mime, prompt, parent, panel)
            return {**res, "cached": True}
        with self.lock:
            if uid in self.busy:
                raise SketchError("Your previous design is still being made. Wait for it to finish.", 429)
            self.busy.add(uid)
        try:
            self.check_budget(uid, model.usd)
            data, mime = self.caller(model, prompt, picture)
            return {**self._save(uid, model.key, model.usd, "ai", cache_key, data, mime, prompt, parent, panel), "cached": False}
        finally:
            with self.lock:
                self.busy.discard(uid)

    def generate(self, uid: str, model_key: str, prompt: str, picture: bytes | None) -> dict:
        return self._run(uid, model_key, prompt, picture)

    def make(self, uid: str, model_key: str, prompt: str, picture: bytes | None, panel: str, salt: str = "") -> dict:
        """One picture for another panel (design.py): same cache, limits and ledger."""
        return self._run(uid, model_key, prompt, picture, panel=panel, salt=salt)

    def set_meta(self, gid: str, uid: str, meta: dict):
        with self.lock:
            self.db.execute("UPDATE gens SET meta=? WHERE id=? AND uid=?", (json.dumps(meta), gid, uid))
            self.db.commit()

    def refine(self, uid: str, gid: str, change: str, model_key: str) -> dict:
        src = self.file(gid, uid)
        if src is None:
            raise SketchError("That design is no longer available.", 404)
        panel = (self.row(gid, uid) or {}).get("panel", "sketch")
        res = self._run(uid, model_key, edit_prompt(change), prepare(src.read_bytes()), parent=gid, panel=panel)
        if not res.get("meta"):   # name the edit after the change, for the galleries
            label = "Edited: " + _clean(change, 200)
            meta = {"n": 1, "label": label[:80], "edit": _clean(change, 200)}
            self.set_meta(res["id"], uid, meta)
            res = {**res, "meta": meta}
        return res

    # helpers for multi-call work (variation.py)
    def acquire(self, uid: str):
        with self.lock:
            if uid in self.busy:
                raise SketchError("Your previous design is still being made. Wait for it to finish.", 429)
            self.busy.add(uid)

    def release(self, uid: str):
        with self.lock:
            self.busy.discard(uid)

    def own_cached(self, uid: str, cache_key: str) -> dict | None:
        with self.lock:
            own = self.db.execute("SELECT id FROM gens WHERE cache_key=? AND uid=? AND deleted=0 ORDER BY ts DESC LIMIT 1",
                                  (cache_key, uid)).fetchone()
        return self.row(own[0], uid) if own and self.file(own[0], uid) else None

    def keep_tile(self, job_id: str, data: bytes):
        """Raw tiles of variation sets (data/sketch/tiles), newest 300 kept."""
        d = self.root / "tiles"
        d.mkdir(exist_ok=True)
        (d / f"{job_id}-{secrets.token_hex(3)}.img").write_bytes(data)
        old = sorted(d.iterdir(), key=lambda p: p.stat().st_mtime)[:-300]
        for p in old:
            p.unlink(missing_ok=True)

    def labels_with_prefix(self, uid: str, prefix: str) -> list[str]:
        """Captions of the sets this person already made from the same picture (newest first)."""
        with self.lock:
            metas = [r[0] for r in self.db.execute(
                "SELECT meta FROM gens WHERE uid=? AND cache_key LIKE ? AND meta IS NOT NULL ORDER BY ts DESC LIMIT 20",
                (uid, prefix + "%"))]
        out = []
        for m in metas:
            try:
                out += [c.get("label", "") for c in json.loads(m).get("cells", [])]
            except (ValueError, AttributeError):
                pass
        return [x for x in out if x]

    def count_key_prefix(self, uid: str, prefix: str) -> int:
        """How many sets this person already made from the same picture + style (new ideas each time)."""
        with self.lock:
            return self.db.execute("SELECT COUNT(*) FROM gens WHERE uid=? AND cache_key LIKE ?",
                                   (uid, prefix + "%")).fetchone()[0]

    def record_lost_spend(self, uid: str, model_key: str, usd: float):
        """Calls that were paid for but whose set failed still count toward the limits."""
        if usd <= 0:
            return
        with self.lock:
            self.db.execute("INSERT INTO gens(id,uid,ts,model,usd,source,file,deleted,panel) VALUES(?,?,?,?,?,'ai','',1,'variation')",
                            (secrets.token_urlsafe(9), uid, time.time(), model_key, usd))
            self.db.commit()

    def save(self, uid, model_key, usd, source, cache_key, data, mime, prompt, parent=None, panel="sketch", meta=None) -> dict:
        return self._save(uid, model_key, usd, source, cache_key, data, mime, prompt, parent, panel, meta)

    def cell(self, gid: str, uid: str, index: int) -> dict:
        """One variation cut out of a set (no AI call), kept as its own design."""
        row = self.row(gid, uid)
        src = self.file(gid, uid)
        cells = ((row or {}).get("meta") or {}).get("cells") or []
        if row is None or src is None:
            raise SketchError("That design is no longer available.", 404)
        if not 0 <= index < len(cells):
            raise SketchError("Pick one of the variations in the set.")
        c = cells[index]
        with Image.open(src) as im:
            part = im.convert("RGB").crop((c["x"], c["y"], c["x"] + c["w"], c["y"] + c["h"]))
            buf = io.BytesIO()
            part.save(buf, "PNG")
        label = c.get("label") or f"Variation {index + 1}"
        return self._save(uid, row["model"], 0.0, "cell", None, buf.getvalue(), "image/png", label, parent=gid,
                          panel=row["panel"], meta={"n": 1, "rows": 1, "cols": 1, "label": label,
                                                    "cells": [{"x": 0, "y": 0, "w": c["w"], "h": c["h"], "label": label}]})


class Tasks:
    """Background work for the Sketch to Design panel, polled by the page: a picture
    made on our server takes minutes, and the tunnel ends requests at 100 s."""

    def __init__(self):
        self.lock = threading.Lock()
        self.jobs: dict[str, dict] = {}

    def start(self, uid: str, fn) -> dict:
        cut = time.time() - 3600
        job_id = secrets.token_urlsafe(9)
        with self.lock:
            for k in [k for k, j in self.jobs.items() if j["ts"] < cut and j["status"] != "running"]:
                del self.jobs[k]
            self.jobs[job_id] = {"id": job_id, "uid": uid, "ts": time.time(), "status": "running",
                                 "result": None, "error": None}

        def run():
            try:
                res = fn()
                with self.lock:
                    self.jobs[job_id].update(status="done", result=res)
            except SketchError as e:
                with self.lock:
                    self.jobs[job_id].update(status="error", error=e.message)
            except Exception as e:   # never leave the page polling forever
                print(f"sketch job failed: {e!r}", flush=True)
                with self.lock:
                    self.jobs[job_id].update(status="error", error="The design could not be made. Try again.")
        threading.Thread(target=run, daemon=True).start()
        return self.get(job_id, uid)

    def get(self, job_id: str, uid: str) -> dict | None:
        with self.lock:
            j = self.jobs.get(job_id)
            return {k: v for k, v in j.items() if k != "uid"} if j and j["uid"] == uid else None


def options(studio: Studio, uid: str) -> dict:
    return {
        "live": bool(connected()),
        "providers": [PROVIDERS[p][1] for p in connected()],
        "models": [{"key": m.key, "label": m.label, "provider": m.provider, "note": m.note} for m in available()],
        "default_model": default_model(),
        "types": [{"key": k, "label": v} for k, v in TYPE_LABELS.items()],
        "backgrounds": [{"key": k, "label": v[0]} for k, v in BACKGROUNDS.items()],
        "budget": studio.public_budget(uid),
    }
