"""Design Generator: a jewellery spec (category, metal, up to 3 stone types,
optional style words and reference picture) -> realistic design pictures.

Accuracy comes from four places:
  1. plan_design reasons about the piece like a jeweller before anything is
     drawn: the real size of the piece, how much of it the stones cover
     (count x size), which known form that makes (solitaire, three-stone,
     tennis / station bracelet, eternity band...), how big each stone looks
     next to the metal, and the colour the metal shows after plating.
  2. build_prompt turns that plan into fixed, exact words for the picture
     model (no AI call), specification first so a long style text can never
     push it out of the 512 tokens the local model reads.
  3. Each picture is read back by a vision-language model (Qwen3-VL, already
     loaded for photo search) with multiple-choice questions built from the
     spec: piece type, metal colour, stone shape, stone size, stone count.
     The image-embedding reader is the fallback when it isn't loaded.
  4. A clear miss of type, metal or stones is remade (DESIGN_TRIES pictures
     at most, default 2) with repair words for each miss (and a different view
     when the type was missed); the closest picture is kept. Shape, size and
     count misses are shown on the picture, not remade (each picture costs
     minutes on this Mac; DESIGN_SOFT_REMAKE=1 turns that remake on).

Requests wait in a per-person queue ("queue stack") and run one after another.
Stones can be typed freely or picked from the team's stone inventory.
"""

from __future__ import annotations

import csv
import inspect
import io
import os
import re
import secrets
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from . import sketch
from .config import DATA
from .sketch import BACKGROUNDS, BY_KEY, SketchError, _clean

DIR = DATA / "design"
MAX_STONES = 3
STYLE_MAX = 5000          # what the page accepts
PLACEMENT_MAX = 1000
PROMPT_BUDGET = 1900      # characters: about the 512 tokens FLUX.2 Klein reads (chat template included)
STYLE_MIN_ROOM = 250
DESIGNS_MAX = 4
QUEUE_KEEP = 40           # requests remembered per person (newest)
SAME_REQUEST_S = 3        # a double click is one request

# key -> (label, noun for the model, how the piece is shown, reader type or None, typical metal grams light..heavy)
CATEGORIES = {
    "ring": ("Ring", "ring", "shown upright at a three-quarter angle so the top and the band are both visible", "ring", (2.0, 6.0)),
    "necklace": ("Necklace", "necklace", "laid out flat and fully visible, centred", "necklace", (8.0, 30.0)),
    "bracelet": ("Bracelet", "bracelet", "laid flat and opened out into a long gentle curve with its clasp visible, the whole length in frame", "bracelet", (5.0, 18.0)),
    "earrings": ("Earrings", "matching pair of earrings", "both earrings side by side, front view", "earrings", (2.0, 7.0)),
    "pendant": ("Pendant", "pendant on a fine chain", "front view, the pendant large in frame with the chain loop above", "pendant", (1.0, 5.0)),
    "brooch": ("Brooch", "brooch", "front view, flat, fully visible", None, (4.0, 15.0)),
    "bangle": ("Bangle", "rigid bangle", "standing upright at a slight angle, large in frame, its wide round opening clearly visible", "bracelet", (8.0, 25.0)),
    "chain": ("Chain", "chain", "laid out flat in a loose loop, links clearly visible", "necklace", (4.0, 20.0)),
    "anklet": ("Anklet", "anklet", "laid out flat in a loose loop, fully visible", "bracelet", (4.0, 15.0)),
    "cufflink": ("Cufflink", "matching pair of cufflinks", "both cufflinks side by side, three-quarter view showing face and back toggle", None, (6.0, 16.0)),
    "tiara": ("Tiara", "tiara", "front view, standing upright, fully visible", None, (30.0, 120.0)),
}
PAIRS = {"earrings", "cufflink"}   # stone counts are for the whole pair

# key -> (what the piece is, with its real size: picture models confuse a small bracelet with a ring,
#         length the stones are set along (mm, per piece), usual metal width (mm), the metal part stones sit on)
GEOMETRY = {
    "ring": ("a finger ring, sized for one finger (about 2 cm across)", 55, 2.5, "band"),
    "necklace": ("a necklace worn around the neck, about 42 cm long", 420, 3.0, "chain"),
    "bracelet": ("a flexible wrist bracelet about 17 cm long with a clasp, big enough to go around a wrist, "
                 "far longer and larger than a ring", 170, 4.0, "links"),
    "earrings": ("a matching pair of earrings", 20, 3.0, "earring's metal"),
    "pendant": ("a pendant hanging on a fine chain", 20, 3.0, "pendant frame"),
    "brooch": ("a brooch with a pin on the back", 40, 4.0, "brooch frame"),
    "bangle": ("a rigid round wrist bangle about 6.5 cm across, large enough for a hand to slide through, "
               "far larger than a ring", 200, 4.0, "bangle's metal"),
    "chain": ("a neck chain about 45 cm long", 450, 3.0, "links"),
    "anklet": ("a flexible ankle chain about 24 cm long with a clasp", 240, 3.0, "chain"),
    "cufflink": ("a matching pair of cufflinks", 15, 10.0, "cufflink face"),
    "tiara": ("a tiara worn on the head, about 15 cm wide", 300, 6.0, "tiara's metal"),
}
# a second view, used when a picture was read as the wrong kind of piece
ALT_VIEW = {
    "bracelet": "laid completely flat and straight from end to end with the clasp open, the full 17 cm length across the frame",
    "anklet": "laid completely flat and straight from end to end with the clasp open, the full length across the frame",
    "bangle": "standing upright, seen from the front, large in frame, the big round opening in the middle",
    "necklace": "laid flat in a wide U shape, the whole necklace in frame",
    "chain": "laid flat in a wide U shape, the whole chain in frame",
    "pendant": "front view, the pendant hanging from its chain, chain visible going up out of the frame",
    "ring": "a single small ring standing upright, seen from the front so the finger hole is clearly visible",
}

# key -> (label, exact words for the model)
SHAPES = {
    "round": ("Round", "round brilliant-cut"),
    "oval": ("Oval", "oval brilliant-cut"),
    "pear": ("Pear", "pear-shaped (teardrop) brilliant-cut"),
    "marquise": ("Marquise", "marquise-cut (long boat shape with two points)"),
    "emerald": ("Emerald", "emerald-cut (rectangular step-cut with cut corners)"),
    "princess": ("Princess", "princess-cut (square brilliant with sharp corners)"),
    "cushion": ("Cushion", "cushion-cut (square with rounded corners)"),
    "radiant": ("Radiant", "radiant-cut (rectangular brilliant with cut corners)"),
    "asscher": ("Asscher", "Asscher-cut (square step-cut with cut corners)"),
    "heart": ("Heart", "heart-shaped brilliant-cut"),
    "trillion": ("Trillion", "trillion-cut (triangular)"),
    "baguette": ("Baguette", "baguette-cut (long narrow rectangle, step-cut)"),
    "tapered_baguette": ("Tapered baguette", "tapered baguette-cut (narrow trapezoid, step-cut)"),
    "half_moon": ("Half moon", "half-moon-cut (half circle)"),
    "kite": ("Kite", "kite-shaped cut"),
    "hexagon": ("Hexagon", "hexagonal step-cut"),
    "old_mine": ("Old mine", "old mine-cut (antique cushion with large facets)"),
    "rose_cut": ("Rose cut", "rose-cut (flat base, domed top of triangular facets)"),
    "cabochon": ("Cabochon", "smooth polished cabochon (domed, no facets)"),
    "briolette": ("Briolette", "briolette (faceted teardrop drop)"),
}
READER_CUTS = {"round", "oval", "pear", "princess", "cushion", "emerald", "marquise", "radiant", "asscher", "heart"}
FACETED_DIAMOND = re.compile(r"\b(diamond|moissanite|cz|cubic zirconia|lab[- ]?grown)\b", re.I)
SUGGEST = {
    "metal_types": ["18k yellow gold", "18k white gold", "18k rose gold", "14k yellow gold", "14k white gold",
                    "14k rose gold", "22k yellow gold", "9k yellow gold", "Platinum 950", "925 sterling silver", "18k alloy"],
    "platings": ["Rhodium", "Black rhodium", "Yellow gold", "Rose gold", "None"],
    "stone_types": ["Diamond", "Lab-grown diamond", "Moissanite", "Ruby", "Blue sapphire", "Emerald", "Pink sapphire",
                    "Tanzanite", "Amethyst", "Aquamarine", "Topaz", "Garnet", "Citrine", "Peridot", "Pearl", "Opal", "Onyx"],
    "colors": ["White", "Colourless (D-F)", "Near colourless (G-J)", "Blue", "Red", "Green", "Pink", "Yellow", "Black"],
    "clarities": ["FL", "IF", "VVS1", "VVS2", "VS1", "VS2", "SI1", "SI2", "I1"],
}


# ---------- the request ----------

@dataclass
class Stone:
    type: str
    count: int
    color: str
    shape: str
    size_mm: float
    clarity: str = ""
    inventory_id: str = ""


@dataclass
class Spec:
    category: str
    metal_type: str = ""
    plating: str = ""
    weight_g: float | None = None
    stones: list[Stone] = field(default_factory=list)
    style: str = ""
    placement: str = ""
    regional: str = ""
    gross_weight_g: float | None = None
    reference_url: str = ""
    background: str = "white"


def check_spec(spec: Spec) -> Spec:
    """Plain-language refusals for anything the picture model can't use."""
    if spec.category not in CATEGORIES:
        raise SketchError("Pick a jewellery category.")
    if spec.background not in BACKGROUNDS or spec.background == "custom":
        raise SketchError("Pick a background from the list.")
    if len(spec.stones) > MAX_STONES:
        raise SketchError(f"Up to {MAX_STONES} stone types per design.")
    for w, name in ((spec.weight_g, "Metal weight"), (spec.gross_weight_g, "Gross weight")):
        if w is not None and not 0.2 <= w <= 2000:
            raise SketchError(f"{name} should be between 0.2 and 2000 g.")
    for i, s in enumerate(spec.stones, 1):
        s.type, s.color, s.clarity = _clean(s.type, 40), _clean(s.color, 40), _clean(s.clarity, 20)
        missing = [n for n, v in (("type", s.type), ("color", s.color), ("shape", s.shape)) if not v]
        if missing:
            raise SketchError(f"Stone {i}: fill in the {' and '.join(missing)}.")
        if s.shape not in SHAPES:
            raise SketchError(f"Stone {i}: pick a shape from the list.")
        if not 1 <= s.count <= 999:
            raise SketchError(f"Stone {i}: the count should be between 1 and 999.")
        if not 0.5 <= s.size_mm <= 40:
            raise SketchError(f"Stone {i}: the size should be between 0.5 and 40 mm.")
    spec.metal_type, spec.plating = _clean(spec.metal_type, 60), _clean(spec.plating, 40)
    spec.regional = _clean(spec.regional, 60)
    spec.placement = _clean(spec.placement, PLACEMENT_MAX)
    spec.style = _clean(spec.style, STYLE_MAX)
    return spec


# ---------- spec -> exact words ----------

METAL_LOOK = {   # colour the reader checks -> plain words for how the finished metal must look
    "yellow_gold": "warm yellow gold",
    "white_gold": "bright white, like white gold or platinum, not yellow",
    "rose_gold": "warm pinkish rose gold, not yellow",
    "black": "dark gunmetal black",
    "two_tone": "two metal colours together",
}
CLIP_METALS = ("yellow_gold", "white_gold", "rose_gold")   # what the image-embedding reader can tell apart


def read_metal(metal_type: str, plating: str) -> tuple[str, str | None, list[str]]:
    """-> (words for the model, colour the finished piece shows, notes for the person).
    A plating that changes the colour hides the alloy's own colour: the alloy's colour
    word is then left out, since "yellow gold, rhodium plated" draws yellow metal."""
    t, p = metal_type.lower(), plating.lower().strip()
    notes = []
    karat = re.search(r"\b(9|10|14|18|20|22|24)\s*(k|kt|ct|karat|carat)\b", t)
    k = f"{karat.group(1)}k " if karat else ""
    plated = bool(p) and p not in ("none", "no", "-")
    if re.search(r"platinum|\bpt\s*\d{3}\b|\bpt\b", t):
        base, colour, words = "platinum", "white_gold", "platinum (cool bright white metal)"
    elif re.search(r"silver|\b925\b|sterling", t):
        base, colour, words = "silver", "white_gold", "925 sterling silver (bright white metal)"
    elif re.search(r"palladium", t):
        base, colour, words = "palladium", "white_gold", "palladium (white metal)"
    elif re.search(r"two[- ]?tone|tri[- ]?colou?r|bi[- ]?colou?r", t):
        base, colour, words = "gold", "two_tone", f"{k}two-tone gold (yellow and white gold together)"
    elif re.search(r"rose|pink|red gold", t):
        base, colour, words = "gold", "rose_gold", f"{k}rose gold (warm pinkish gold)"
    elif re.search(r"white", t):
        base, colour, words = "gold", "white_gold", f"{k}white gold"
    elif re.search(r"yellow", t) or (karat and karat.group(1) in ("22", "24")):
        base, colour, words = "gold", "yellow_gold", f"{k}yellow gold" + (" (rich deep yellow)" if karat and karat.group(1) in ("22", "24") else "")
    elif karat or "gold" in t or "alloy" in t:
        base, colour, words = "gold", "yellow_gold", f"{k}yellow gold"
        if not plated:
            notes.append("The metal colour wasn't given, so yellow gold was used. Write yellow, white or rose to choose.")
    elif t:
        base, colour, words = "other", None, metal_type
    else:
        base, colour, words = "gold", "yellow_gold", "18k yellow gold"
        notes.append("No metal was given, so 18k yellow gold was used.")
    if plated:
        own = colour
        if "black" in p:
            finish, colour = "black rhodium plated (dark gunmetal black finish)", "black"
        elif "rhodium" in p or "white" in p:
            finish, colour = "rhodium plated (bright mirror-white finish)", "white_gold"
        elif "rose" in p or "pink" in p:
            finish, colour = "rose gold plated (pinkish gold finish)", "rose_gold"
        elif "yellow" in p or "gold" in p:
            finish, colour = "yellow gold plated (yellow finish)", "yellow_gold"
        else:
            finish = f"{plating} plated"
        if colour != own and base == "gold":
            words = f"{k}gold"          # the plating's colour is all that shows
        words += ", " + finish
    return words, colour, notes


def weight_words(category: str, grams: float | None) -> str:
    if grams is None:
        return ""
    light, heavy = CATEGORIES[category][4]
    if grams < light * 0.6:
        feel = "very fine and delicate, thin metal"
    elif grams < light:
        feel = "delicate, slim metal"
    elif grams <= heavy:
        feel = "balanced, medium-weight metal"
    elif grams <= heavy * 1.6:
        feel = "bold, substantial metal"
    else:
        feel = "very heavy, chunky statement metal"
    return f"about {grams:g} g of metal: {feel}"


def _carat(size_mm: float) -> float:
    return 0.004 * size_mm ** 3   # round brilliant diamond: 6.5 mm is about 1 ct


def metal_width(category: str, grams: float | None) -> float:
    """Usual width (mm) of the metal the stones sit on, a little wider for a heavier piece."""
    width = GEOMETRY[category][2]
    if grams:
        light, heavy = CATEGORIES[category][4]
        width *= min(1.5, max(0.7, (grams / ((light + heavy) / 2)) ** 0.5))
    return width


def size_class(mm: float, width: float) -> str:
    """How big a stone looks next to the metal: tiny / small / medium / large."""
    if mm < 1.8:
        return "tiny"
    if mm >= 6.0:
        return "large"
    return "small" if mm <= width * 1.3 and mm < 3.5 else "medium"


SIZE_WORDS = {
    "tiny": "tiny melee stones that look like small sparkling points, much narrower than the {part}",
    "small": "small stones, about as wide as the {part}",
    "medium": "clearly wider than the {part}",
    "large": "a large stone, the dominant feature of the piece",
}


def stone_words(s: Stone, category: str = "ring", width: float | None = None) -> str:
    stone = s.type.strip().lower()
    colour = s.color.strip()
    low = colour.lower()
    diamondish = bool(FACETED_DIAMOND.search(stone))
    if diamondish:
        if low in ("white", "colourless", "colorless") or re.fullmatch(r"[d-f](\s*-\s*[d-f])?", low) or "d-f" in low:
            colour = "colourless icy-white"
        elif re.fullmatch(r"[g-j](\s*-\s*[g-j])?", low) or "g-j" in low or "near" in low:
            colour = "near-colourless white"
        elif low and low not in ("white",):
            colour = f"fancy {colour}" if "fancy" not in low else colour
    words = f"{SHAPES[s.shape][1]} {colour} {stone}".replace("  ", " ")
    if s.shape == "cabochon" or "pearl" in stone.lower():
        words = f"{colour} {stone}" + (f", {SHAPES[s.shape][0].lower()} shape" if s.shape not in ("round", "cabochon") else "")
    size = f"{s.size_mm:g} mm" + (" each" if s.count > 1 else "")
    if diamondish and s.shape == "round" and s.size_mm >= 3:
        size += f" (about {_carat(s.size_mm):.2g} ct{' each' if s.count > 1 else ''})"
    cls = size_class(s.size_mm, width or GEOMETRY[category][2])
    look = SIZE_WORDS[cls].format(part=GEOMETRY[category][3])
    if cls == "large" and s.count > 1:
        look = "large stones, the dominant feature of the piece"
    size += f": {look}"
    clarity = s.clarity.upper()
    if re.match(r"^(FL|IF|VVS)", clarity):
        size += ", flawless-looking, very bright"
    elif re.match(r"^VS|^SI", clarity):
        size += ", eye-clean"
    elif re.match(r"^I\d", clarity):
        size += ", slightly included"
    return f"{words}, {size}"


MAIN_ROLES = ("centre stone", "two main stones", "three main stones")


def stone_roles(category: str, stones: list[Stone]) -> list[tuple[str, Stone]]:
    """Main stone(s) vs accents, from count and size (largest first)."""
    pair = category in PAIRS
    out = []
    order = sorted(stones, key=lambda s: -s.size_mm)
    for i, s in enumerate(order):
        per_piece = s.count / 2 if pair and s.count % 2 == 0 else s.count
        big = s.size_mm >= 3.0 or (i == 0 and len(order) > 1 and s.size_mm >= order[-1].size_mm * 1.8)
        if i == 0 and per_piece <= 3 and big:
            role = {1: "centre stone", 2: "two main stones", 3: "three main stones"}[int(per_piece)]
        elif s.count >= 12 and s.size_mm <= 2.5:
            role = "pavé accents"
        elif per_piece <= 3 and s.size_mm >= 3.0:
            role = "side stones"
        else:
            role = "accent stones"
        out.append((role, s))
    return out


def _per_piece(category: str, s: Stone) -> float:
    return s.count / 2 if category in PAIRS and s.count % 2 == 0 else s.count


def plan_form(category: str, roles: list[tuple[str, Stone]]) -> str | None:
    """The known jewellery form the stones make, from how much of the piece they cover.
    Picture models know these forms by name far better than they follow counts:
    10 stones of 1 mm on a 17 cm bracelet cover 7% of it, a station bracelet, not a tennis one."""
    if not roles:
        return None
    noun = CATEGORIES[category][1]
    length = GEOMETRY[category][1]
    main = [(r, s) for r, s in roles if r in MAIN_ROLES]
    rest = [(r, s) for r, s in roles if r not in MAIN_ROLES]
    n_rest = int(sum(_per_piece(category, s) for _, s in rest))
    cover = sum(_per_piece(category, s) * s.size_mm * 1.15 for _, s in rest) / length
    lead = main[0][0] if main else None
    if category == "ring":
        if lead == "centre stone":
            return ("solitaire ring: one centre stone on a plain polished band" if not rest else
                    "a centre stone ring, the smaller stones set beside or around the centre stone")
        if lead == "two main stones":
            return "two-stone (toi et moi) ring: two stones side by side on top"
        if lead == "three main stones":
            return "three-stone ring: three stones side by side across the top"
        if cover >= 0.8:
            return f"full eternity band: one continuous row of {n_rest} identical stones all the way round the band"
        if cover >= 0.25:
            return f"half-eternity band: one row of {n_rest} stones across the top half only, the rest is plain polished band"
        return (f"slim band with {n_rest} tiny stones set close together in a short row on top; "
                "the rest of the band is plain polished metal")
    if category in ("bracelet", "anklet", "necklace", "chain"):
        if main:
            return f"a {noun} with the main stone{'s' if lead != 'centre stone' else ''} at the centre front" + \
                   (", the smaller stones beside them" if rest else ", the rest plain metal")
        if cover >= 0.7:
            name = {"bracelet": "tennis bracelet", "anklet": "tennis anklet", "necklace": "rivière necklace"}.get(category, f"stone-set {noun}")
            return f"{name}: one continuous line of {n_rest} matching stones along the whole length"
        return (f"a fine station {noun}: {n_rest} small stones set as separate stations spaced evenly along it; "
                "most of its length is plain metal links")
    if category == "bangle":
        if cover >= 0.7 and not main:
            return f"bangle set all the way round with one row of {n_rest} stones"
        return "bangle with the stones set across the front only; the rest is plain polished metal"
    if category == "earrings":
        if lead == "centre stone":
            return "stud earrings: one stone on each earring" if not rest else \
                   "each earring has its main stone with the smaller stones around or below it"
        return None
    if category == "pendant" and lead == "centre stone":
        return "solitaire pendant: one stone in a simple setting" if not rest else \
               "a pendant with its main stone in the middle and the smaller stones around it"
    return None


def _count_words(category: str, s: Stone) -> str:
    if category in PAIRS and s.count % 2 == 0:
        return f"{s.count} in total ({s.count // 2} on each of the pair)"
    if category in PAIRS:
        return f"{s.count} in total across the pair"
    return f"exactly {s.count}" if s.count <= 12 else f"{s.count}"


def _cut_at_word(text: str, n: int) -> str:
    if len(text) <= n:
        return text
    cut = text[:n].rsplit(" ", 1)[0]
    return cut.rstrip(",;:") + "…"


SHAPE_FAMILY = {"round": "round", "oval": "oval", "pear": "pear", "marquise": "marquise", "heart": "heart",
                "princess": "square", "asscher": "square", "cushion": "cushion",
                "emerald": "rect", "baguette": "rect", "tapered_baguette": "rect", "radiant": "rect"}


@dataclass
class Built:
    prompt: str
    expect: dict          # what the reader should find: type, metal, stones, centre_cut, shape, size, count
    notes: list[str]      # told to the person with the request


def build_prompt(spec: Spec, has_picture: bool = False, stress: list[str] | None = None,
                 alt_view: bool = False) -> Built:
    label, noun, view, reader_type, _ = CATEGORIES[spec.category]
    anchor = GEOMETRY[spec.category][0]
    metal, colour, notes = read_metal(spec.metal_type, spec.plating)
    grams = spec.weight_g if spec.weight_g is not None else spec.gross_weight_g
    width = metal_width(spec.category, grams)
    lines = []
    if stress:   # a remake: the points the earlier picture missed come first
        lines.append("Most important: " + "; ".join(stress) + ".")
    if has_picture:
        lines.append(f"Create a new photorealistic jewellery product photo of {anchor}. Use the reference picture "
                     "only as inspiration for the overall style and silhouette; the specification below wins wherever they differ.")
    else:
        lines.append(f"Photorealistic high-end jewellery product photo of {anchor}.")
    roles = stone_roles(spec.category, spec.stones) if spec.stones else []
    form = None if spec.placement else plan_form(spec.category, roles)
    if form:
        lines.append(f"Design: {form}.")
    weight = weight_words(spec.category, grams)
    look = f" The metal must look {METAL_LOOK[colour]}." if colour in METAL_LOOK else ""
    lines.append(f"Metal: {metal}, high polish.{look}" + (f" Weight {weight}." if weight else ""))
    expect = {"type": reader_type, "type_label": label.lower(), "category": spec.category, "metal": colour,
              "stones": bool(spec.stones), "centre_cut": None, "shape": None, "size": None, "count": 0}
    if spec.stones:
        lines.append("Gemstones, exactly these and no others:")
        for role, s in roles:
            lines.append(f"- {role.capitalize()}: {_count_words(spec.category, s)} {stone_words(s, spec.category, width)}.")
        role0, s0 = roles[0]
        if role0 == "centre stone" and s0.shape in READER_CUTS and spec.category in ("ring", "pendant"):
            expect["centre_cut"] = s0.shape
        expect["shape"], expect["shape_key"] = SHAPE_FAMILY.get(s0.shape), s0.shape
        cls = size_class(s0.size_mm, width)
        expect["size"] = cls if cls in ("tiny", "large") else None
        expect["count"] = sum(s.count for s in spec.stones)
        if spec.placement:
            lines.append(f"Stone placement: {_cut_at_word(spec.placement, 500)}.")
            if len(spec.placement) > 500:
                notes.append("Stone placement was shortened to its first 500 characters for the picture model.")
        elif not form:
            main = [r for r, _ in roles if r in MAIN_ROLES]
            if main and len(roles) > 1:
                lines.append("Placement: the main stone(s) at the focal point, smaller stones arranged around or beside them.")
    else:
        lines.append("Plain polished metal design with no gemstones at all.")
    if spec.regional:
        lines.append(f"Regional style: {spec.regional}.")
    shown = ALT_VIEW.get(spec.category, view) if alt_view else view
    tail = [f"Composition: one {noun}, {shown}. Background: {BACKGROUNDS[spec.background][1]}.",
            "Accurate real-world proportions and stone sizes, manufacturable by a jeweller, crisp facets, sharp focus. "
            "No text, no hands, no watermark."]
    fixed = "\n".join(lines + tail)
    if spec.style:
        room = max(STYLE_MIN_ROOM, PROMPT_BUDGET - len(fixed) - 10)
        style = _cut_at_word(spec.style, room)
        if len(style) < len(spec.style):
            notes.append(f"Your style text was shortened to its first {len(style)} characters, as much as the picture "
                         "model reads after the specification.")
        lines.append(f"Design style: {style}")
    return Built("\n".join(lines + tail), expect, notes)


# ---------- reading the picture back ----------

SURE = 0.7   # a reading this sure (or surer) can count as a miss; below it, "can't tell"

# the vision-language reader's choices: (key, words shown to it)
TYPE_OPTS = [("ring", "ring"), ("bracelet", "bracelet or bangle"), ("necklace", "necklace or chain"),
             ("earrings", "earrings"), ("pendant", "pendant on a chain"), ("brooch", "brooch"),
             ("cufflink", "cufflinks"), ("tiara", "tiara or crown")]
TYPE_OK = {"ring": {"ring"}, "bracelet": {"bracelet"}, "bangle": {"bracelet"}, "anklet": {"bracelet"},
           "necklace": {"necklace", "pendant"}, "chain": {"necklace"}, "pendant": {"pendant", "necklace"},
           "earrings": {"earrings"}, "brooch": {"brooch", "pendant"}, "cufflink": {"cufflink", "earrings"},
           "tiara": {"tiara"}}
TYPE_Q = ("What kind of jewellery piece is this? A ring fits one finger (about 2 cm across). A bracelet or bangle "
          "goes around a wrist (about 6-7 cm across). A necklace goes around the neck.")
METAL_OPTS = [("yellow_gold", "yellow gold"), ("white_gold", "white / silver / platinum"),
              ("rose_gold", "rose gold (pinkish)"), ("two_tone", "two different metal colours"), ("black", "black")]
METAL_NAMES = {"yellow_gold": "Yellow gold", "white_gold": "White metal", "rose_gold": "Rose gold",
               "two_tone": "Two-tone", "black": "Black"}
STONES_OPTS = [("yes", "yes, gemstones or diamonds are set in it"), ("no", "no, it is plain metal only")]
SHAPE_OPTS = [("round", "round"), ("oval", "oval"), ("pear", "pear / teardrop"),
              ("marquise", "marquise (long, pointed at both ends)"), ("heart", "heart"),
              ("squarish", "square or rectangular"), ("none", "no gemstones")]
SQUARE_OPTS = [("rect", "long rectangle (emerald-cut or baguette)"), ("square", "square with sharp corners (princess or Asscher)"),
               ("cushion", "square with rounded corners (cushion)")]
SIZE_OPTS = [("tiny", "tiny sparkling points, much narrower than the metal"), ("same", "about as wide as the metal"),
             ("large", "clearly wider than the metal, large stones"), ("none", "no gemstones")]
COUNT_OPTS = [("0", "none"), ("1", "1"), ("2-3", "2 to 3"), ("4-8", "4 to 8"), ("9-20", "9 to 20"), ("21+", "more than 20")]
COUNT_EDGES = [(0, 0), (1, 1), (2, 3), (4, 8), (9, 20), (21, 10**6)]
WEIGHT = {"type": 5, "metal": 4, "stones": 4, "cut": 2, "shape": 2, "size": 2, "count": 1}
HARD = ("type", "metal", "stones")      # worth a remake (up to DESIGN_TRIES pictures)
SOFT = ("shape", "size", "count")       # shown on the picture; a remake only when DESIGN_SOFT_REMAKE=1


def max_tries() -> int:
    """Pictures per design at most: the first plus one remake for a clear miss of type, metal or stones."""
    try:
        return max(1, min(5, int(os.environ.get("DESIGN_TRIES", "2"))))
    except ValueError:
        return 2


def _bucket(n: int) -> int:
    return next(i for i, (lo, hi) in enumerate(COUNT_EDGES) if lo <= n <= hi)


def vlm_questions(expect: dict) -> list[tuple[str, str, list[tuple[str, str]]]]:
    """The multiple-choice questions a picture of this spec is read with: (key, question, options)."""
    qs = [("type", TYPE_Q, TYPE_OPTS), ("metal", "What colour is the metal?", METAL_OPTS),
          ("stones", "Does this piece have gemstones or diamonds set in it?", STONES_OPTS)]
    if expect.get("stones"):
        if expect.get("shape") and expect.get("size") != "tiny":   # tiny stones' shapes can't be seen
            qs.append(("shape", "What shape is the largest gemstone, seen from the top?", SHAPE_OPTS))
            if expect["shape"] in ("rect", "square", "cushion"):
                qs.append(("square", "Which best describes the largest gemstone's outline?", SQUARE_OPTS))
        if expect.get("size"):
            qs.append(("size", "How big are the gemstones compared with the width of the metal they sit on?", SIZE_OPTS))
        if expect.get("count"):
            qs.append(("count", "About how many gemstones can you see?", COUNT_OPTS))
    return qs


def vlm_reading(answers: dict[str, list[float]], questions) -> dict:
    """Per question: the chosen option's key and how sure the reader was."""
    out = {"by": "vlm"}
    for key, _, opts in questions:
        p = answers[key]
        i = max(range(len(p)), key=p.__getitem__)
        out[key], out[key + "_p"] = opts[i][0], round(float(p[i]), 3)
        if key == "stones":
            out["stones_yes"] = round(float(p[0]), 3)
    return out


def _check(key, label, ok, detail=""):
    return {"key": key, "label": label, "ok": ok, "detail": detail if ok is False else ""}


def compare_vlm(expect: dict, r: dict) -> list[dict]:
    checks = []
    if expect.get("category") or expect.get("type"):
        want = TYPE_OK.get(expect.get("category") or "", {expect.get("type")})
        got = r.get("type")
        ok = True if got in want else (False if r.get("type_p", 0) >= SURE else None)
        checks.append(_check("type", f"Looks like {_a(expect['type_label'])}", ok,
                             f"read as {_a(dict(TYPE_OPTS).get(got, got))}"))
    if expect.get("metal") in METAL_NAMES:
        got = r.get("metal")
        ok = True if got == expect["metal"] else (False if r.get("metal_p", 0) >= SURE else None)
        checks.append(_check("metal", f"{METAL_NAMES[expect['metal']]} colour", ok,
                             f"read as {METAL_NAMES.get(got, got or '?').lower()}"))
    if "stones_yes" in r:
        yes = r["stones_yes"]
        has = True if yes >= SURE else (False if yes <= 1 - SURE else None)
        if expect.get("stones"):
            checks.append(_check("stones", "Has gemstones", has))
        else:
            checks.append(_check("stones", "Plain metal, no stones", None if has is None else not has, "stones were drawn"))
    shape = expect.get("shape")
    if shape and r.get("shape"):
        label = f"{SHAPES[expect['shape_key']][0]}-shaped stones" if expect.get("shape_key") else "Stone shape"
        got = r["shape"]
        if shape in ("rect", "square", "cushion"):
            fam_ok = got == "squarish"
            sub = r.get("square")
            ok = True if fam_ok and sub == shape else (
                False if (not fam_ok and r["shape_p"] >= SURE) or (fam_ok and sub != shape and r.get("square_p", 0) >= SURE) else None)
            detail = f"read as {dict(SQUARE_OPTS).get(sub, sub)}" if fam_ok else f"read as {dict(SHAPE_OPTS).get(got, got)}"
        else:
            ok = True if got == shape else (False if r["shape_p"] >= SURE else None)
            detail = f"read as {dict(SHAPE_OPTS).get(got, got)}"
        if got == "none":
            ok = None
        checks.append(_check("shape", label, ok, detail))
    size = expect.get("size")
    if size and r.get("size") and r["size"] != "none":
        got = r["size"]
        if size == "tiny":
            ok = True if got == "tiny" else (False if got == "large" and r["size_p"] >= SURE else None)
        else:
            ok = True if got == "large" else (False if got == "tiny" and r["size_p"] >= SURE else None)
        checks.append(_check("size", "Tiny stones" if size == "tiny" else "Large stones", ok,
                             "drawn as large stones" if size == "tiny" else "drawn as tiny stones"))
    n = expect.get("count")
    if n and r.get("count"):
        got = [k for k, _ in COUNT_OPTS].index(r["count"])
        off = abs(got - _bucket(n))
        ok = True if off == 0 else (False if off >= 2 and r["count_p"] >= SURE else None)
        checks.append(_check("count", f"About {n} stone{'s' if n > 1 else ''}", ok,
                             f"looks like {dict(COUNT_OPTS)[r['count']]} stones"))
    return checks


def compare(expect: dict, reading: dict | None) -> list[dict]:
    """Reader's view of a made picture vs the spec -> checks for the page.
    ok: True match, False clear miss, None can't tell."""
    if not reading:
        return []
    if reading.get("by") == "vlm":
        return compare_vlm(expect, reading)
    checks = []   # the image-embedding reader (fallback): type, metal, stones, centre cut
    if expect.get("type"):
        got, sure = reading.get("type"), reading.get("type_sure")
        ok = None if not sure else got == expect["type"]
        checks.append({"key": "type", "label": f"Looks like {_a(expect['type_label'])}", "ok": ok,
                       "detail": "" if ok is not False else f"read as {_a(got)}"})
    if expect.get("metal") in CLIP_METALS:
        got = reading.get("metal")
        ok = None if not got else got == expect["metal"]
        checks.append({"key": "metal", "label": f"{METAL_NAMES[expect['metal']]} colour", "ok": ok,
                       "detail": "" if ok is not False else f"read as {got.replace('_', ' ')}"})
    if "plain" in reading and reading["plain"] is not None:
        if expect.get("stones"):
            checks.append({"key": "stones", "label": "Has gemstones", "ok": not reading["plain"], "detail": ""})
        else:
            checks.append({"key": "stones", "label": "Plain metal, no stones", "ok": bool(reading["plain"]), "detail": ""})
    if expect.get("centre_cut") and reading.get("centre_cut"):
        ok = reading["centre_cut"] == expect["centre_cut"]
        checks.append({"key": "cut", "label": f"{SHAPES[expect['centre_cut']][0]} centre stone", "ok": ok,
                       "detail": "" if ok else f"read as {reading['centre_cut']}"})
    return checks


def misses(checks: list[dict]) -> list[str]:
    """Clear misses worth a remake: the hard ones (type, metal, stones) and the soft ones (shape, size, count)."""
    return [c["key"] for c in checks if c["ok"] is False and c["key"] in HARD + SOFT]


def score(checks: list[dict]) -> float:
    return sum(WEIGHT.get(c["key"], 1) * (1 if c["ok"] else (-1.5 if c["ok"] is False else 0)) for c in checks)


def stress_words(expect: dict, missed: list[str], noun: str, spec: Spec | None = None) -> list[str]:
    """Repair words for each missed point, put first in the next picture's prompt."""
    out = []
    big = max(spec.stones, key=lambda s: s.size_mm) if spec and spec.stones else None
    if "type" in missed:
        out.append(f"the piece must clearly be a {noun} at its real size")
    if "metal" in missed and expect.get("metal") in METAL_LOOK:
        out.append(f"the metal must look {METAL_LOOK[expect['metal']]}")
    if "stones" in missed:
        out.append("gemstones must be clearly visible" if expect.get("stones") else "no gemstones anywhere, plain metal only")
    if "shape" in missed and big:
        out.append(f"every main stone must be {SHAPES[big.shape][1]}")
    if "size" in missed and big:
        out.append(f"the stones must be tiny, only {big.size_mm:g} mm, small sparkling points; most of the piece is plain "
                   "polished metal" if expect.get("size") == "tiny" else
                   f"the main stone must be large, {big.size_mm:g} mm across")
    if "count" in missed and expect.get("count"):
        out.append(f"exactly {expect['count']} stones in total")
    return out


def _a(word: str | None) -> str:
    word = word or "something else"
    return ("an " if word[:1] in "aeiou" else "a ") + word


def reader_from_engine(engine, photo_mod):
    """-> read(picture bytes, expect) -> reading dict for compare().
    The vision-language reader (Qwen3-VL) when photo search has it loaded, else the
    image-embedding reader (type, metal, stones, centre cut)."""
    vlm = getattr(getattr(engine, "detail_reader", None), "__self__", None)
    vlm = vlm if hasattr(vlm, "choose") else None

    def clip_read(data: bytes) -> dict:
        pq = engine.read_photo(photo_mod.read(data))
        t, dia = pq.dna["type"], pq.dna["diamonds"]
        return {"type": t["value"], "type_sure": bool(t["sure"]), "metal": pq.metal,
                "plain": dia.get("plain"), "centre_cut": (dia.get("centre_cut") or {}).get("value")}

    def read(data: bytes, expect: dict | None = None) -> dict:
        if vlm is not None and expect is not None:
            try:
                qs = vlm_questions(expect)
                probs = vlm.choose(photo_mod.read(data), [(q, [w for _, w in opts]) for _, q, opts in qs])
                return vlm_reading({k: p for (k, _, _), p in zip(qs, probs)}, qs)
            except Exception as e:   # fall back to the embedding reader
                print(f"design: vision-language check failed ({e!r})", flush=True)
        return clip_read(data)
    return read


# ---------- stone inventory ----------

class Inventory:
    """The team's loose stones (data/design/inventory.sqlite). Everyone may pick
    from it; staff (admins, jewelers) keep it up to date."""

    FIELDS = ("code", "type", "shape", "color", "size_mm", "clarity", "qty", "note")

    def __init__(self, root: Path | None = None):
        self.root = Path(root) if root else DIR
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.db = sqlite3.connect(self.root / "inventory.sqlite", check_same_thread=False)
        self.db.execute("""CREATE TABLE IF NOT EXISTS stones (
            id TEXT PRIMARY KEY, code TEXT, type TEXT NOT NULL, shape TEXT NOT NULL, color TEXT NOT NULL,
            size_mm REAL NOT NULL, clarity TEXT, qty INTEGER NOT NULL, note TEXT, ts REAL NOT NULL, by TEXT)""")
        self.db.commit()

    @staticmethod
    def clean(item: dict) -> dict:
        out = {k: _clean(str(item.get(k, "") or ""), 60) for k in ("code", "type", "color", "clarity", "note")}
        shape = str(item.get("shape", "")).strip().lower().replace(" ", "_").replace("-", "_")
        shape = {"tapered": "tapered_baguette", "rose": "rose_cut", "oldmine": "old_mine", "halfmoon": "half_moon"}.get(shape, shape)
        if shape not in SHAPES:
            raise SketchError(f"Unknown shape “{item.get('shape', '')}”. Use one of: {', '.join(v[0] for v in SHAPES.values())}.")
        try:
            size, qty = float(item.get("size_mm")), int(float(item.get("qty")))
        except (TypeError, ValueError):
            raise SketchError("Size (mm) and quantity must be numbers.")
        if not out["type"] or not out["color"]:
            raise SketchError("Each stone needs a type and a colour.")
        if not 0.5 <= size <= 40 or not 0 <= qty <= 1_000_000:
            raise SketchError("Size should be 0.5-40 mm and quantity 0 or more.")
        return {**out, "shape": shape, "size_mm": size, "qty": qty}

    def _row(self, r) -> dict:
        return {"id": r[0], "code": r[1] or "", "type": r[2], "shape": r[3], "shape_label": SHAPES.get(r[3], (r[3],))[0],
                "color": r[4], "size_mm": r[5], "clarity": r[6] or "", "qty": r[7], "note": r[8] or ""}

    def all(self) -> list[dict]:
        with self.lock:
            rows = self.db.execute("SELECT id,code,type,shape,color,size_mm,clarity,qty,note FROM stones "
                                   "ORDER BY type COLLATE NOCASE, shape, size_mm").fetchall()
        return [self._row(r) for r in rows]

    def get(self, sid: str) -> dict | None:
        with self.lock:
            r = self.db.execute("SELECT id,code,type,shape,color,size_mm,clarity,qty,note FROM stones WHERE id=?",
                                (sid,)).fetchone()
        return self._row(r) if r else None

    def add(self, item: dict, by: str = "") -> dict:
        c = self.clean(item)
        sid = secrets.token_urlsafe(6)
        with self.lock:
            self.db.execute("INSERT INTO stones VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                            (sid, c["code"], c["type"], c["shape"], c["color"], c["size_mm"], c["clarity"], c["qty"],
                             c["note"], time.time(), by))
            self.db.commit()
        return self.get(sid)

    def update(self, sid: str, item: dict, by: str = "") -> dict:
        c = self.clean(item)
        with self.lock:
            n = self.db.execute("UPDATE stones SET code=?,type=?,shape=?,color=?,size_mm=?,clarity=?,qty=?,note=?,ts=?,by=? "
                                "WHERE id=?", (c["code"], c["type"], c["shape"], c["color"], c["size_mm"], c["clarity"],
                                               c["qty"], c["note"], time.time(), by, sid)).rowcount
            self.db.commit()
        if not n:
            raise SketchError("That stone is no longer in the inventory.", 404)
        return self.get(sid)

    def delete(self, sid: str) -> bool:
        with self.lock:
            n = self.db.execute("DELETE FROM stones WHERE id=?", (sid,)).rowcount
            self.db.commit()
        return n > 0

    def import_csv(self, text: str, by: str = "") -> dict:
        """Columns: code, type, shape, color, size_mm, clarity, qty, note (header row needed).
        Every row is checked first; nothing is added if any row is wrong."""
        rows = list(csv.DictReader(io.StringIO(text.strip())))
        if not rows:
            raise SketchError("The CSV has no rows. The first line must be the column names.")
        if len(rows) > 2000:
            raise SketchError("Up to 2000 stones per import.")
        norm = lambda d: {re.sub(r"[^a-z_]", "", (k or "").strip().lower().replace(" ", "_")): v for k, v in d.items()}
        clean = []
        for i, r in enumerate(rows, 2):
            r = norm(r)
            r.setdefault("size_mm", r.get("size") or r.get("sizemm"))
            r.setdefault("qty", r.get("quantity") or r.get("count"))
            r.setdefault("color", r.get("colour"))
            try:
                clean.append(self.clean(r))
            except SketchError as e:
                raise SketchError(f"Line {i}: {e.message}")
        for c in clean:
            self.add(c, by)
        return {"added": len(clean)}

    def resolve(self, stones: list[Stone]) -> list[Stone]:
        """Inventory picks -> full stones from the stored record (the page can't change them)."""
        out = []
        for i, s in enumerate(stones, 1):
            if not s.inventory_id:
                raise SketchError(f"Stone {i}: pick a stone from the inventory.")
            rec = self.get(s.inventory_id)
            if rec is None:
                raise SketchError(f"Stone {i}: that stone is no longer in the inventory.")
            if s.count > rec["qty"]:
                raise SketchError(f"Stone {i}: only {rec['qty']} of {rec['code'] or rec['type']} in stock.")
            out.append(Stone(rec["type"], s.count, rec["color"], rec["shape"], rec["size_mm"], rec["clarity"], rec["id"]))
        return out


# ---------- the queue stack ----------

def summary(spec: Spec) -> str:
    bits = [CATEGORIES[spec.category][0]]
    if spec.metal_type or spec.plating:
        bits.append(" ".join(x for x in (spec.metal_type, f"{spec.plating} plated" if spec.plating and spec.plating.lower() != "none" else "") if x))
    for s in spec.stones:
        bits.append(f"{s.count}× {s.size_mm:g} mm {SHAPES[s.shape][0].lower()} {s.color.lower()} {s.type.lower()}")
    return " · ".join(bits)


def spec_dict(spec: Spec) -> dict:
    d = dict(spec.__dict__)
    d["stones"] = [dict(s.__dict__) for s in spec.stones]
    return d


class Queue:
    """Each person's design requests, made one after another in the background.
    Kept in memory (the made pictures themselves are stored by the Studio)."""

    def __init__(self, studio: sketch.Studio, reader=None):
        self.studio = studio
        self.reader = reader          # bytes -> reading dict, or None (no checks)
        self.lock = threading.Lock()
        self.items: dict[str, list[dict]] = {}
        self.workers: set[str] = set()

    def list(self, uid: str) -> list[dict]:
        with self.lock:
            return [self._public(i) for i in reversed(self.items.get(uid, []))]

    @staticmethod
    def _public(item: dict) -> dict:
        out = {k: v for k, v in item.items() if not k.startswith("_")}
        if out["status"] in ("queued", "making", "checking"):
            out["elapsed"] = round(time.time() - (item.get("started") or item["ts"]))
        return out

    def position(self, uid: str, item_id: str) -> int:
        with self.lock:
            waiting = [i["id"] for i in self.items.get(uid, []) if i["status"] == "queued"]
        return waiting.index(item_id) + 1 if item_id in waiting else 0

    def submit(self, uid: str, spec: Spec, model_key: str, designs: int, picture: bytes | None,
               wait: bool = False) -> dict:
        if not 1 <= designs <= DESIGNS_MAX:
            raise SketchError(f"Make 1 to {DESIGNS_MAX} designs per request.")
        model = BY_KEY.get(model_key)
        if model is None:
            raise SketchError("Pick a model from the list.")
        built = build_prompt(spec, has_picture=picture is not None)
        b = self.studio.budget(uid)
        with self.lock:
            mine = self.items.setdefault(uid, [])
            pending = sum(i["designs"] for i in mine if i["status"] in ("queued", "making", "checking"))
            if b["left_today"] is not None and pending + designs > b["left_today"]:
                raise SketchError(f"You can make {b['left_today']} more designs today and {pending} are already "
                                  "in your queue. Make fewer, or try again tomorrow.", 429)
            if mine and mine[-1]["_prompt"] == built.prompt and time.time() - mine[-1]["ts"] < SAME_REQUEST_S:
                return self._public(mine[-1])
            item = {"id": secrets.token_urlsafe(8), "ts": time.time(), "status": "queued", "summary": summary(spec),
                    "spec": spec_dict(spec), "model": model.key, "model_label": model.label, "designs": designs,
                    "done": 0, "attempt": 0, "results": [], "notes": built.notes, "error": None, "started": None,
                    "_prompt": built.prompt, "_expect": built.expect, "_picture": picture, "_spec": spec}
            mine.append(item)
            del mine[:-QUEUE_KEEP]
            start = uid not in self.workers
            if start:
                self.workers.add(uid)
        if start:
            if wait:
                self._work(uid)
            else:
                threading.Thread(target=self._work, args=(uid,), daemon=True).start()
        return self._public(item)

    def cancel(self, uid: str, item_id: str) -> bool:
        with self.lock:
            for i in self.items.get(uid, []):
                if i["id"] == item_id and i["status"] == "queued":
                    i.update(status="cancelled")
                    return True
        return False

    def _next(self, uid: str) -> dict | None:
        with self.lock:
            item = next((i for i in self.items.get(uid, []) if i["status"] == "queued"), None)
            if item is None:
                self.workers.discard(uid)
            else:
                item.update(status="making", started=time.time())
            return item

    def _set(self, item: dict, **kw):
        with self.lock:
            item.update(kw)

    def _work(self, uid: str):
        while True:
            item = self._next(uid)
            if item is None:
                return
            try:
                self._make(uid, item)
                self._set(item, status="done" if item["results"] else "error",
                          error=None if item["results"] else "No design could be made. Try again.")
            except SketchError as e:
                self._set(item, status="done" if item["results"] else "error",
                          error=None if item["results"] else e.message,
                          notes=item["notes"] + ([f"Stopped early: {e.message}"] if item["results"] else []))
            except Exception as e:   # never leave the page waiting forever
                print(f"design request failed: {e!r}", flush=True)
                self._set(item, status="done" if item["results"] else "error",
                          error=None if item["results"] else "The design could not be made. Try again.")
            finally:
                with self.lock:
                    item.pop("_picture", None)

    def _wait_turn(self, uid: str):
        """Another tool of the same person may be drawing: wait for it (max 30 min)."""
        end = time.time() + 1800
        while time.time() < end:
            with self.studio.lock:
                if uid not in self.studio.busy:
                    return
            time.sleep(3)
        raise SketchError("Your other design is taking too long. Try again later.", 429)

    def _one(self, uid: str, item: dict, prompt: str, salt: str) -> dict:
        for _ in range(5):
            self._wait_turn(uid)
            try:
                return self.studio.make(uid, item["model"], prompt, item.get("_picture"), panel="design", salt=salt)
            except SketchError as e:
                if "still being made" not in e.message:   # another tool started in between: wait again
                    raise
        raise SketchError("Your other designs are keeping the studio busy. Try again later.", 429)

    def _make(self, uid: str, item: dict):
        spec, expect = item["_spec"], item["_expect"]
        noun = CATEGORIES[spec.category][1]
        has_picture = item.get("_picture") is not None
        limit = max_tries() if self.reader else 1
        for n in range(item["designs"]):
            self._set(item, attempt=n + 1, stage="making")
            salt = f"{item['id']}:{n}"
            t0 = time.time()
            best = self._one(uid, item, item["_prompt"], salt)
            print(f"design {item['id']}#{n + 1}: picture {time.time() - t0:.0f} s ({item['model']})", flush=True)
            checks, tries, missed_ever = [], 1, []
            if self.reader:
                self._set(item, status="checking")
                checks = self._check(uid, best["id"], expect)
                latest = checks
                while tries < limit:
                    missed = misses(latest)
                    hard = [k for k in missed if k in HARD]
                    soft_ok = os.environ.get("DESIGN_SOFT_REMAKE") == "1" and tries < 2
                    if not hard and not soft_ok:   # a picture takes minutes: remake only clear big misses
                        break
                    missed_ever += [k for k in missed if k not in missed_ever]
                    self._set(item, status="making", stage="remaking")
                    prompt = build_prompt(spec, has_picture, stress_words(expect, missed_ever, noun, spec),
                                          alt_view="type" in missed_ever).prompt
                    again = self._one(uid, item, prompt, f"{salt}:try{tries + 1}")
                    tries += 1
                    self._set(item, status="checking")
                    latest = self._check(uid, again["id"], expect)
                    if score(latest) > score(checks):
                        self.studio.delete(best["id"], uid)
                        best, checks = again, latest
                    else:
                        self.studio.delete(again["id"], uid)
            result = {"id": best["id"], "image": best["image"], "model_label": best["model_label"],
                      "cached": best.get("cached", False), "checks": checks, "remade": tries > 1, "tries": tries}
            self.studio.set_meta(best["id"], uid, {"design": {"summary": item["summary"], "checks": checks,
                                                              "spec": item["spec"], "tries": tries}})
            with self.lock:
                item["results"].append(result)
                item["done"] = n + 1
                item["status"] = "making" if n + 1 < item["designs"] else "checking"

    def _check(self, uid: str, gid: str, expect: dict) -> list[dict]:
        f = self.studio.file(gid, uid)
        if f is None:
            return []
        try:
            data, t0 = f.read_bytes(), time.time()
            takes_spec = len(inspect.signature(self.reader).parameters) > 1
            checks = compare(expect, self.reader(data, expect) if takes_spec else self.reader(data))
            print(f"design check {gid}: {time.time() - t0:.0f} s, misses {misses(checks)}", flush=True)
            return checks
        except Exception as e:   # a check that fails must not lose the picture
            print(f"design check failed: {e!r}", flush=True)
            return []


def default_model() -> str | None:
    """Fast first: on this Mac a 768 px picture takes about 2 min, 1024 px 6-7 min next to the
    live app (it pushes the app's models into swap). HD stays in the list for the final look."""
    keys = [m.key for m in sketch.available()]
    for k in ("local-klein", "local-klein-hd", "p-gptimage", "nb21-1k"):
        if k in keys:
            return k
    return keys[0] if keys else None


def options(studio: sketch.Studio, uid: str, staff: bool, inventory: Inventory) -> dict:
    return {
        "live": bool(sketch.connected()),
        "categories": [{"key": k, "label": v[0], "pair": k in PAIRS} for k, v in CATEGORIES.items()],
        "shapes": [{"key": k, "label": v[0]} for k, v in SHAPES.items()],
        "models": [{"key": m.key, "label": m.label, "note": m.note} for m in sketch.available()],
        "default_model": default_model(),
        "backgrounds": [{"key": k, "label": v[0]} for k, v in BACKGROUNDS.items() if k != "custom"],
        "suggest": SUGGEST,
        "max_stones": MAX_STONES, "style_max": STYLE_MAX, "placement_max": PLACEMENT_MAX, "designs_max": DESIGNS_MAX,
        "staff": staff,
        "inventory_count": len(inventory.all()),
        "budget": studio.public_budget(uid),
    }
