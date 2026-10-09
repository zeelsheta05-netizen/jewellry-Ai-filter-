"""Design Variations: one jewellery picture (sketch or photo) -> a set of 1-36
variations, shown as one labelled grid.

Same wrapper as Sketch to Design (sketch.py: providers, ledger, cache, limits),
different input and context logic:

- No fixed style lists: one cheap planning call (Pollinations' text model, which
  reads the picture) names N variation ideas made for THIS piece, following the
  person's own words, and avoiding ideas already used on the same picture. If
  the planner is unavailable, ideas are combined at random from design axes, so
  sets still differ every time.
- Free image models draw a 2x2 grid reliably but break down on bigger ones
  (tested 2026-10-07: a 4x4 request came back 3x3 or 4x5, stones lost). So each
  call asks for one 2x2 tile = 4 named variations, and the server joins the
  tiles: 16 variations = 4 calls (not 16), ~512 px per cell.
- The model writes no text (it garbles captions); the server draws them.
- Long sets run as a background job the page polls (the tunnel ends requests at 100 s).
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import random
import re
import secrets
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import sketch
from .sketch import BY_KEY, SketchError

COUNTS = (1, 4, 9, 16, 25, 36)
PER_CALL = 4            # one 2x2 tile per call
PARALLEL = 3            # calls at once (provider rate limits)
RETRY_WAIT = 6          # seconds before the one retry after a rate limit
CELL = 512
CAPTION = 46
GAP = 12
PAPER = (248, 243, 232)
INK = (60, 45, 25)
DIRECTION_MAX = 600
DEFAULT_MODELS = ("local-klein", "local-klein-hd", "p-gptimage")   # our server first (no daily limit), fast size first:
# a 2x2 tile at 768 px takes about 2 min, at 1024 px 6-7 min next to the live app; HD stays selectable
LOW_BALANCE_MODELS = ("local-klein", "local-klein-hd", "p-klein")  # when the account runs low mid-set (Auto only)

GEM_RULE = ("Keep every gemstone exactly the same — same type, cut, colour, count, size and position — "
            "unless the direction says otherwise. Vary the metalwork and the design around the stones.")

LOOKS = {
    "illustration": ("Design illustration", "as a detailed coloured jewellery design illustration on cream paper"),
    "photo": ("Realistic photo", "as a photorealistic jewellery product photo on a soft light background"),
    "pencil": ("Pencil sketch", "as a clean, detailed pencil jewellery sketch on white paper"),
}
TILE_SPOTS = ["top-left", "top-right", "bottom-left", "bottom-right"]

# fallback when the planner can't be reached: random combinations, different every time
_AXES = {
    "technique": ["filigree", "hand engraving", "milgrain", "granulation", "open-work", "repoussé relief", "hammered texture",
                  "polished high-shine", "brushed satin", "rope-twist", "pierced lattice", "enamel accents", "beaded wire",
                  "knife-edge", "two-tone metal", "cut-out negative space"],
    "motif": ["vine", "lotus", "peacock", "paisley", "wave", "feather", "sunburst", "leaf", "scroll", "geometric", "floral",
              "teardrop", "crescent", "knot", "honeycomb", "ribbon", "star", "shell", "temple arch", "petal"],
    "form": ["slimmer and lighter", "bolder and fuller", "layered two-tier", "asymmetric", "elongated", "rounder",
             "more open and airy", "denser and richer", "minimal clean", "sculptural 3D"],
}


def calls_for(n: int) -> int:
    return 1 if n == 1 else math.ceil(n / PER_CALL)


def _clean(text: str, n: int) -> str:
    return " ".join((text or "").split())[:n]


# ---------- planning the ideas ----------

def _plan_prompt(n: int, direction: str, avoid: list[str], piece: str = "") -> str:
    lines = [
        f"You are a senior jewellery designer. The piece: {piece}" if piece else
        "You are a senior jewellery designer. Look at the jewellery in the picture.",
        f"Propose exactly {n} clearly different design variations of THIS piece, made for its type, shape, stones and style. "
        "Every variation must stay the same kind of piece (a ring stays a ring).",
        "Each must be realistic to manufacture and wearable. Spread them widely: change silhouette, metalwork technique, "
        "motifs, setting style and proportions; no two alike, no generic filler.",
        "Use plain, concrete jewellery words for names and briefs (band, shank, shoulders, setting, prongs, bezel, halo, "
        "gallery, openwork, engraving, texture, milgrain): no poetic or abstract names, no scenes or objects that are not "
        "parts of the piece.",
        GEM_RULE,
    ]
    if direction:
        lines.append(f"The client's direction (follow it): {direction}")
    if avoid:
        lines.append("Already shown for this piece, do not repeat or closely resemble: " + "; ".join(avoid[:80]))
    lines.append('Reply with JSON only: {"piece": "<one sentence describing the original piece>", "ideas": '
                 '[{"name": "<2-4 word name>", "brief": "<max 18 words: what changes>"}]}')
    return "\n".join(lines)


def _parse_plan(text: str, n: int, keep_all: bool = False) -> tuple[str, list[dict]]:
    m = re.search(r"\{.*\}", text or "", re.S)
    if not m:
        raise ValueError("no JSON")
    try:
        data = json.loads(m.group(0))
    except ValueError:
        # small local models often write almost-JSON (e.g. each idea in its own list): take the pairs
        pairs = re.findall(r'"name"\s*:\s*"([^"]+)"\s*,\s*"brief"\s*:\s*"([^"]*)"', text)
        piece = re.search(r'"piece"\s*:\s*"([^"]*)"', text)
        data = {"piece": piece.group(1) if piece else "", "ideas": [{"name": a, "brief": b} for a, b in pairs]}
    ideas, seen = [], set()
    for it in data.get("ideas") or []:
        if not isinstance(it, dict):
            continue
        name = _clean(str(it.get("name", "")), 40).strip(" .")
        brief = _clean(str(it.get("brief", "")), 160)
        if name and name.lower() not in seen:
            seen.add(name.lower())
            ideas.append({"name": name, "brief": brief or name})
    if len(ideas) < n:
        raise ValueError(f"only {len(ideas)} ideas")
    return _clean(str(data.get("piece", "")), 200), ideas if keep_all else ideas[:n]


def random_ideas(n: int, avoid: list[str] | None = None, rng: random.Random | None = None, kind: str = "piece") -> list[dict]:
    """Last fallback: random technique x motif x form combinations applied to the same
    kind of piece, never the same set twice."""
    rng = rng or random.Random(secrets.randbits(64))
    used = {a.lower() for a in (avoid or [])}
    out = []
    for _ in range(n * 40):
        t, mo, f = rng.choice(_AXES["technique"]), rng.choice(_AXES["motif"]), rng.choice(_AXES["form"])
        name = f"{mo.capitalize()} {t}"
        if name.lower() in used:
            continue
        used.add(name.lower())
        out.append({"name": name, "brief": f"the same {kind}, reworked with {t} metalwork and {mo} motifs, {f}"})
        if len(out) == n:
            break
    return out


def read_piece(engine, picture: bytes) -> dict | None:
    """The piece in words, from our own photo reading (Design DNA, local models, free):
    {"piece": "a yellow gold ring, no gemstones, ...", "kind": "ring", "stones": False}."""
    if engine is None:
        return None
    try:
        from . import photo
        d = engine.read_photo(photo.read(picture)).dna
    except Exception as e:
        print(f"variation: photo reading failed ({e!r})", flush=True)
        return None
    t = d.get("type") or {}
    kind = {"earrings": "pair of earrings"}.get(t.get("value"), t.get("value") or "piece of jewellery")
    traits = d.get("traits") or []
    plain = any(x.get("group") == "Stones" and "no stone" in x.get("label", "").lower() for x in traits)
    metal = ((d.get("metal") or {}).get("metal") or "").replace("_", " ")
    words = [f"a {metal + ' ' if metal else ''}{kind}" + ("" if t.get("sure", True) else " (probably)")]
    dd = d.get("diamonds") or {}
    stones = [x["label"] for x in traits if x.get("group") == "Stones"]
    stones += [dd[k]["label"] + (" centre stone" if k == "centre_cut" else "") for k in ("layout", "centre_cut") if dd.get(k)]
    words.append("no gemstones" if plain else ", ".join(stones))
    words += [x["label"] for x in traits if x.get("group") != "Stones"][:4]
    words += [m["label"] for m in (d.get("motifs") or [])][:3] + [x["label"] for x in (d.get("details") or [])][:2]
    return {"piece": ", ".join(w for w in words if w).lower(), "kind": kind, "stones": not plain}


def local_writer(engine):
    """The small language model the app already has loaded (Qwen3-1.7B, the jewellery
    judge) writing a short answer: free, on this Mac, no extra memory. None if not loaded."""
    ask = getattr(getattr(engine, "domain", None), "_judge", None)
    ask = getattr(ask, "__wrapped__", ask)            # Judge.ask is lru_cache(Judge._ask)
    judge = getattr(ask, "__self__", None)
    if judge is None or not hasattr(judge, "model"):
        return None

    def write(prompt: str, max_tokens: int) -> str:
        msgs = [{"role": "user", "content": prompt}]
        text = judge.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        x = judge.tok(text, return_tensors="pt").to(judge.dev)
        with judge.gpu, judge.torch.no_grad():
            out = judge.model.generate(**x, max_new_tokens=max_tokens, do_sample=True, temperature=0.8, top_p=0.95)
        return judge.tok.decode(out[0][x["input_ids"].shape[1]:], skip_special_tokens=True)
    return write


def plan_ideas(picture: bytes, n: int, direction: str, avoid: list[str], planner=None, engine=None,
               writer=None) -> tuple[str, list[dict], str]:
    """(piece description, n ideas, who planned: "ai" / "local" / "random").
    1. Pollinations' text model, which looks at the picture (when it has credit);
    2. our own photo reading + the local language model (free, always here);
    3. random combinations kept to the same kind of piece."""
    info = read_piece(engine, picture)
    use = planner or (sketch.call_pollinations_text if "pollinations" in sketch.connected() else None)
    if use is not None:
        for _ in range(2):
            try:
                piece, ideas = _parse_plan(use(_plan_prompt(n, direction, avoid), picture, 400 + 60 * n), n)
                return piece or (info or {}).get("piece", ""), ideas, "ai"
            except SketchError as e:
                print(f"variation planner: {e.message}", flush=True)
                if e.reason == "balance":
                    break   # no credit: don't ask again
            except ValueError as e:
                print(f"variation planner: {e}", flush=True)
    write = writer or local_writer(engine)
    if write is not None and info:
        ideas, seen = [], list(avoid)
        for _ in range(6):   # asks in small batches: a small model keeps JSON tidy for ~6 ideas
            k = min(6, n - len(ideas))
            if k <= 0:
                break
            try:
                _, got = _parse_plan(write(_plan_prompt(k, direction, seen, info["piece"]), 120 + 45 * k), 1, keep_all=True)
            except Exception as e:
                print(f"variation local planner: {e!r}", flush=True)
                continue
            for i in got:
                if i["name"].lower() not in {x.lower() for x in seen} and len(ideas) < n:
                    ideas.append(i)
                    seen.append(i["name"])
        if len(ideas) >= max(1, n // 2):
            ideas += random_ideas(n - len(ideas), seen, kind=info["kind"])
            return info["piece"], ideas, "local"
    return (info or {}).get("piece", ""), random_ideas(n, avoid, kind=(info or {}).get("kind", "piece")), "random"


def effective_look(look: str, direction: str) -> str:
    """Words in the direction win over the Look list (e.g. "realistic" with Design illustration)."""
    if re.search(r"\b(realistic|real[- ]?life|photo\w*|lifelike)\b", direction or "", re.I):
        return "photo"
    if re.search(r"\b(pencil|sketch\w*|drawing)\b", direction or "", re.I) and look == "photo":
        return "pencil"
    return look


NO_STONES = ("The original has no gemstones: do not add any. Vary the metalwork, surface texture and shape, "
             "and keep it the same kind of piece.")


def tile_prompt(ideas: list[dict], direction: str, look: str, piece: str = "", stones: bool = True) -> str:
    """One call = one 2x2 tile of up to four named variations (or one picture for a single variation)."""
    look_words = LOOKS[effective_look(look, direction)][1]
    rule = GEM_RULE if stones else NO_STONES
    base = f"The original piece: {piece} " if piece else ""
    extra = f"Direction: {direction} " if direction else ""
    if len(ideas) == 1:
        i = ideas[0]
        return (f"Using the attached jewellery picture as the base design, create one variation: {i['brief'] or i['name']}. "
                f"{base}{extra}{rule} Show the complete piece, same viewpoint and framing as the original, {look_words}. "
                "One piece only. No text, no captions, no numbers.")
    # the drawing model gets only the concrete brief: names are captions (a poetic name like
    # "Nebula Niche" was drawn literally as curtains and a galaxy)
    spots = "; ".join(f"{TILE_SPOTS[k]}: {i['brief'] or i['name']}" for k, i in enumerate(ideas))
    # the piece and the stone rule come first: smaller models weigh the start of the prompt most
    return (f"Using the attached jewellery picture as the base design, make ONE image split into a 2x2 grid of four equal "
            f"square panels separated by clear white gutters. {base}Every panel shows this same kind of piece"
            f"{' with the same gemstones in the same places' if stones else ''}. {rule} {extra}Panel designs — {spots}. "
            f"Same viewpoint and framing as the original in every panel, {look_words}. No text, no captions, no numbers.")


# ---------- reading a 2x2 tile ----------

def _line(gray: np.ndarray, axis: int) -> int | None:
    """Where the panels meet near the middle, along one axis (None = no dividing line).
    A gutter: a light, even band across the whole picture (brightest wins among ties).
    A hairline: panels touching with a thin light line, lighter than both sides."""
    a = gray if axis == 0 else gray.T
    W = a.shape[1]
    best_g, best_h, lines = None, None, {}
    for x in range(int(W * .28), int(W * .72)):
        # a line lighter than both sides (hairline or gutter up to ~12 px): models don't always centre it
        h = max(float(((a[:, x] - np.maximum(a[:, x - k], a[:, x + k])) > 6).mean()) for k in (3, 6))
        lines[x] = h
        if best_h is None or h > best_h[0]:
            best_h = (h, x)
        if W * .40 <= x < W * .60:
            band = a[:, max(0, x - 2):x + 3]
            share = float(((band.mean(axis=1) > 205) & (band.std(axis=1) < 14)).mean())
            g = (share, float(band.mean()))
            if best_g is None or g > best_g[0]:
                best_g = (g, x)
    if best_h[0] >= .9:        # real lines .99; single pictures <= .27
        x = best_h[1]
        if not W * .40 <= x < W * .60:   # off-centre: a 3-across grid has a twin line on the other side
            twin = max((lines.get(m, 0) for m in range(W - x - int(W * .04), W - x + int(W * .04))), default=0)
            if twin >= .9:
                return None
        return x
    if best_g[0][0] >= .985:   # light, even band in the middle (panels on the same white as the gutter)
        return best_g[1]
    return None


def split_tile(data: bytes, wanted: int) -> list[Image.Image]:
    """2x2 tile -> its panels (reading order). Not a grid -> the whole picture as one panel."""
    im = Image.open(io.BytesIO(data)).convert("RGB")
    if wanted == 1:
        return [im]
    small = im.copy()
    small.thumbnail((512, 512))
    gray = np.asarray(small.convert("L"), dtype=np.float32)
    gx, gy = _line(gray, 0), _line(gray, 1)
    if gx is None or gy is None:
        return [im]
    quads = [gray[:gy, :gx], gray[:gy, gx:], gray[gy:, :gx], gray[gy:, gx:]]
    if any(q.size == 0 or (q < 235).mean() < .005 for q in quads):   # an empty quarter: not four pieces
        return [im]
    kx, ky = im.width / small.width, im.height / small.height
    X, Y = int(gx * kx), int(gy * ky)
    pad = max(4, im.width // 128)
    boxes = [(0, 0, X - pad, Y - pad), (X + pad, 0, im.width, Y - pad),
             (0, Y + pad, X - pad, im.height), (X + pad, Y + pad, im.width, im.height)]
    return [_trim(im.crop(b)) for b in boxes][:wanted]


def _trim(im: Image.Image) -> Image.Image:
    """Cut white gutter strips left at the panel's edges and the light frame around the piece."""
    gray = np.asarray(im.convert("L"), dtype=np.float32)
    top, bottom, left, right = 0, gray.shape[0], 0, gray.shape[1]
    lim_y, lim_x = int(gray.shape[0] * .12), int(gray.shape[1] * .12)

    def white(v):
        return v.mean() > 246 and v.std() < 8
    while top < lim_y and white(gray[top]):
        top += 1
    while gray.shape[0] - bottom < lim_y and white(gray[bottom - 1]):
        bottom -= 1
    while left < lim_x and white(gray[:, left]):
        left += 1
    while gray.shape[1] - right < lim_x and white(gray[:, right - 1]):
        right -= 1
    im = im.crop((left, top, right, bottom))
    g = gray[top:bottom, left:right]
    ink = np.argwhere(g < 235)
    if len(ink) < 50:
        return im
    (y0, x0), (y1, x1) = ink.min(0), ink.max(0)
    m = int(max(im.size) * .04)
    return im.crop((max(0, x0 - m), max(0, y0 - m), min(im.width, x1 + m), min(im.height, y1 + m)))


# ---------- the set ----------

FONT_FILES = ["/System/Library/Fonts/Supplemental/Arial.ttf", "/Library/Fonts/Arial Unicode.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "C:/Windows/Fonts/arial.ttf"]


def _font(size: int):
    """A font with accented letters (é in "Pavé"); Pillow's built-in one lacks them."""
    for f in FONT_FILES:
        try:
            return ImageFont.truetype(f, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:   # very old Pillow
        return ImageFont.load_default()


def _plain(text: str) -> str:
    """Only used with the built-in font: é -> e, so no empty boxes appear."""
    import unicodedata
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode() or text


def _fit_label(draw, text: str, font, width: int) -> str:
    if draw.textlength(text, font=font) <= width:
        return text
    while text and draw.textlength(text + "…", font=font) > width:
        text = text[:-1]
    return text.rstrip() + "…"


def compose(panels: list[Image.Image], ideas: list[dict]) -> tuple[bytes, dict]:
    """Panels -> one grid picture with captions; meta has each cell's box (without caption) for cut-outs."""
    n = len(panels)
    if n == 1:
        buf = io.BytesIO()
        panels[0].save(buf, "PNG")
        w, h = panels[0].size
        i = ideas[0]
        return buf.getvalue(), {"n": 1, "rows": 1, "cols": 1, "label": i["name"],
                                "cells": [{"x": 0, "y": 0, "w": w, "h": h, "label": i["name"], "brief": i["brief"]}]}
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    W = GAP + cols * (CELL + GAP)
    H = GAP + rows * (CELL + CAPTION + GAP)
    sheet = Image.new("RGB", (W, H), PAPER)
    draw = ImageDraw.Draw(sheet)
    font = _font(24)
    cells = []
    for k, (p, idea) in enumerate(zip(panels, ideas)):
        r, c = divmod(k, cols)
        x, y = GAP + c * (CELL + GAP), GAP + r * (CELL + CAPTION + GAP)
        cell = Image.new("RGB", (CELL, CELL), PAPER)
        p = p.copy()
        p.thumbnail((CELL, CELL), Image.LANCZOS)
        cell.paste(p, ((CELL - p.width) // 2, (CELL - p.height) // 2))
        sheet.paste(cell, (x, y))
        draw.rectangle((x, y, x + CELL - 1, y + CELL - 1), outline=(226, 212, 182))
        name = idea["name"] if isinstance(font, ImageFont.FreeTypeFont) else _plain(idea["name"])
        label = _fit_label(draw, name, font, CELL - 16)
        draw.text((x + (CELL - draw.textlength(label, font=font)) / 2, y + CELL + 10), label, fill=INK, font=font)
        cells.append({"x": x, "y": y, "w": CELL, "h": CELL, "label": idea["name"], "brief": idea["brief"]})
    buf = io.BytesIO()
    sheet.save(buf, "JPEG", quality=90, optimize=True)
    return buf.getvalue(), {"n": n, "rows": rows, "cols": cols, "cells": cells, "size": [W, H]}


class Jobs:
    """Background sets, polled by the page (memory only; finished jobs kept 1 h)."""

    def __init__(self, studio: sketch.Studio, planner=None, engine=None, writer=None):
        self.studio = studio
        self.planner = planner      # picture-reading planner (default: Pollinations' text model)
        self.engine = engine        # our search engine: photo reading + the local language model
        self.writer = writer
        self.lock = threading.Lock()
        self.jobs: dict[str, dict] = {}

    def get(self, job_id: str, uid: str) -> dict | None:
        with self.lock:
            j = self.jobs.get(job_id)
            return dict(j) if j and j["uid"] == uid else None

    def _set(self, job_id: str, **kw):
        with self.lock:
            self.jobs[job_id].update(kw)

    def _sweep(self):
        cut = time.time() - 3600
        with self.lock:
            for k in [k for k, j in self.jobs.items() if j["ts"] < cut and j["status"] != "running"]:
                del self.jobs[k]

    def start(self, uid: str, picture: bytes, n: int, direction: str, look: str, model_key: str,
              wait: bool = False, auto: bool = False) -> dict:
        if n not in COUNTS:
            raise SketchError("Pick how many variations from the list.")
        if look not in LOOKS:
            raise SketchError("Pick a look from the list.")
        model = BY_KEY.get(model_key)
        if model is None:
            raise SketchError("Pick a model from the list.")
        direction = _clean(direction, DIRECTION_MAX)
        calls = calls_for(n)
        self.studio.acquire(uid)
        try:
            self.studio.check_budget(uid, model.usd * calls + sketch.PLANNER_USD)
        except SketchError:
            self.studio.release(uid)
            raise
        self._sweep()
        job_id = secrets.token_urlsafe(9)
        with self.lock:
            self.jobs[job_id] = {"id": job_id, "uid": uid, "ts": time.time(), "status": "running", "stage": "planning",
                                 "calls": calls, "done_calls": 0, "result": None, "error": None, "n": n}
        args = (job_id, uid, picture, n, direction, look, model, auto)
        if wait:
            self._run(*args)
        else:
            threading.Thread(target=self._run, args=args, daemon=True).start()
        return self.get(job_id, uid)

    def _run(self, job_id, uid, picture, n, direction, look, model, auto=False):
        base = hashlib.sha256(b"\0".join([b"variation", picture])).hexdigest()[:24]
        spent, planned_cost = 0.0, 0.0
        state = {"model": model, "fell_back": False}
        cheap = next((BY_KEY[k] for k in LOW_BALANCE_MODELS if BY_KEY[k].provider in sketch.connected()), None)
        try:
            avoid = self.studio.labels_with_prefix(uid, base)   # ideas already shown for this picture
            piece, ideas, planned = plan_ideas(picture, n, direction, avoid, self.planner, self.engine, self.writer)
            planned_cost = sketch.PLANNER_USD if planned == "ai" else 0.0
            info_stones = not (piece and "no gemstones" in piece.lower())
            self._set(job_id, stage="drawing")
            groups = [ideas] if n == 1 else [ideas[i:i + PER_CALL] for i in range(0, n, PER_CALL)]

            def one(group):
                nonlocal spent
                prompt = tile_prompt(group, direction, look, piece, info_stones)
                for attempt in range(3):
                    use = state["model"]
                    try:
                        data, _mime = self.studio.caller(use, prompt, picture)
                        with self.lock:
                            spent += use.usd
                            self.jobs[job_id]["done_calls"] += 1
                        self.studio.keep_tile(job_id, data)   # the raw answer, for checking splits later
                        return split_tile(data, len(group))
                    except SketchError as e:
                        if e.reason == "balance":
                            # the account is low: with Auto, finish the set on the cheapest good model
                            if auto and cheap and use.key != cheap.key:
                                with self.lock:
                                    state["model"], state["fell_back"] = cheap, True
                                continue
                            raise
                        if attempt == 0 and e.status in (429, 502):
                            time.sleep(RETRY_WAIT)
                            continue
                        raise
                raise SketchError("The variations could not be made. Try again.", 502)

            with ThreadPoolExecutor(max_workers=PARALLEL) as pool:
                results = list(pool.map(one, groups))
            panels, used = [], []
            for group, got in zip(groups, results):
                panels += got
                used += group[:len(got)] if len(got) == len(group) else [group[0]]
            data, meta = compose(panels, used)
            meta.update({"look": effective_look(look, direction), "asked": n, "piece": piece, "planned": planned,
                         "direction": direction,
                         "model": state["model"].key, "fell_back": state["fell_back"]})
            if len(panels) < n:
                meta["note"] = f"{len(panels)} of {n} variations came out; the AI merged some panels."
            key = base + secrets.token_hex(8)
            mime = "image/png" if n == 1 else "image/jpeg"
            row = self.studio.save(uid, state["model"].key, spent + planned_cost, "ai", key, data, mime,
                                   tile_prompt(groups[0], direction, look, piece, info_stones), panel="variation", meta=meta)
            self._set(job_id, status="done", result=row)
        except SketchError as e:
            self.studio.record_lost_spend(uid, model.key, spent + planned_cost)
            self._set(job_id, status="error", error=e.message)
        except Exception as e:   # never leave the page polling forever
            self.studio.record_lost_spend(uid, model.key, spent + planned_cost)
            print(f"variation job failed: {e!r}", flush=True)
            self._set(job_id, status="error", error="The variations could not be made. Try again.")
        finally:
            self.studio.release(uid)


def default_model() -> str | None:
    keys = [m.key for m in sketch.available()]
    return next((k for k in DEFAULT_MODELS if k in keys), keys[0] if keys else None)


def options(studio: sketch.Studio, uid: str) -> dict:
    models = sketch.available()
    return {
        "live": bool(models),
        "providers": [sketch.PROVIDERS[p][1] for p in sketch.connected()],
        "counts": [{"n": n, "label": str(n) if n == 1 else f"{n} ({int(math.sqrt(n))}×{int(math.sqrt(n))})"} for n in COUNTS],
        "looks": [{"key": k, "label": v[0]} for k, v in LOOKS.items()],
        "models": [{"key": m.key, "label": m.label, "provider": m.provider, "note": m.note} for m in models],
        "default_model": default_model(),
        "budget": studio.public_budget(uid),
    }
