"""Buy page: what a design is made of, from its job card or its CAD file.

scripts/build_cad_specs.py writes data/cad/specs.json. This module decides
what the page may show, and says where each figure comes from:

- A job card (the client's own spreadsheet) is shown as it is: gold weight
  per purity after polishing, and the diamond list with setting types.
- Without one, figures are measured from the CAD file, and only where they
  were checked against the job cards:
    gold weight   rings and pendants with a closed metal surface (for rings,
                  ~90% land within 10% of the card). Earrings are left out:
                  their cards include findings (screw posts) that the CAD
                  doesn't, and a file may hold one earring or the pair.
    stones        cut and size always; count and carats except for earrings
                  (pair or single is unknown).
    size          ring size and width; height x width for other pieces.
- Anything else is "confirmed when you order", never guessed.
"""
from __future__ import annotations

import json
import math
import re

from .config import DATA

SPECS = DATA / "cad" / "specs.json"

# weight of each purity relative to 18K: the fixed ratios on the client's job cards
PURITIES = [
    {"key": "14k", "label": "14K", "fineness": "585", "ratio": 0.85},
    {"key": "18k", "label": "18K", "fineness": "750", "ratio": 1.00},
    {"key": "22k", "label": "22K", "fineness": "916", "ratio": 1.14},
]
# 22K is too soft to hold stones well and can't be made properly white or rose
PURITIES_FOR = {"yellow_gold": ["14k", "18k", "22k"], "rose_gold": ["14k", "18k"], "white_gold": ["14k", "18k"]}
DEFAULT_PURITY = "18k"

CAD_WEIGHT_CATEGORIES = {"ring", "pendant"}   # where the CAD estimate was checked against job cards
MAX_OPEN_SHARE = 0.01          # metal surface that may be open before the volume isn't trusted
DIAMOND_CT_PER_MM3 = 0.0176    # 3.52 g/cm3 x 5 ct/g

# Round diamonds are weighed by the trade's size chart, the one in the client's
# job cards ("Diamond Detail's-1"): mm -> carats per stone.
ROUND_CHART = [
    (0.7, .0021), (0.8, .003), (0.9, .004), (1.0, .005), (1.1, .006), (1.15, .007), (1.2, .008),
    (1.25, .009), (1.3, .010), (1.35, .011), (1.4, .012), (1.45, .013), (1.5, .014), (1.55, .015),
    (1.6, .019), (1.7, .022), (1.8, .026), (1.9, .030), (2.0, .035), (2.1, .040), (2.2, .045),
    (2.3, .053), (2.4, .060), (2.5, .065), (2.6, .070), (2.7, .080), (2.8, .090), (2.9, .100),
    (3.0, .110), (3.1, .120), (3.2, .135), (3.3, .145), (3.4, .160), (3.5, .190), (3.6, .200),
    (3.7, .210), (3.9, .250), (4.2, .330), (4.5, .400), (4.8, .470), (5.1, .520), (5.3, .630),
    (5.5, .750), (5.8, .850), (6.4, 1.0), (7.3, 1.5), (8.0, 2.0), (8.75, 2.5), (9.3, 3.0),
    (10.2, 4.0), (11.0, 5.0), (11.6, 6.0),
]

SHAPE_CODES = [("Round", r"^(rnd|rd|round|brill)"), ("Oval", r"^ov"), ("Pear", r"^pe"), ("Marquise", r"^(mq|marq)"),
               ("Emerald", r"^(em|emr|emer)"), ("Princess", r"^(pr|prn|princ)"), ("Cushion", r"^(cu|cus|cush)"),
               ("Radiant", r"^(ra|rad)"), ("Heart", r"^(hr|hrt|heart)"), ("Baguette", r"^(bag|bg|tap|tpr)"),
               ("Asscher", r"^(as|assch)"), ("Trillion", r"^(tr|tril)")]

_cache = {"mtime": None, "data": {"designs": {}, "calibration": None}}


def _specs() -> dict:
    """specs.json, re-read when the build script updates it (no restart)."""
    try:
        mtime = SPECS.stat().st_mtime
    except FileNotFoundError:
        return {"designs": {}, "calibration": None}
    if mtime != _cache["mtime"]:
        _cache["data"] = json.loads(SPECS.read_text())
        _cache["mtime"] = mtime
    return _cache["data"]


def design_key(m: dict) -> str:
    return f"{m['design_id']}|{m['folders'][0]}"


# ---------------------------------------------------------------- formatting

def shape_name(code: str | None) -> str:
    code = (code or "").strip().lower()
    for name, pat in SHAPE_CODES:
        if re.match(pat, code):
            return name
    return code.title() if code else "Diamond"


def size_label(size) -> str:
    """ "1.4" -> "1.4 mm", "12.00X8.00" -> "12 × 8 mm" """
    parts = [p for p in re.split(r"\s*[xX×*]\s*", str(size or "").strip()) if p]
    nums = []
    for p in parts:
        try:
            nums.append(float(p))
        except ValueError:
            return str(size)
    if not nums:
        return ""
    return " × ".join(f"{n:g}" for n in nums) + " mm"


def setting_label(s: str) -> str:
    s = re.sub(r"\s+", " ", (s or "").strip())
    return s[:1].upper() + s[1:].lower() if s else ""


def round_carat(d_mm: float) -> float:
    xs, ys = zip(*ROUND_CHART)
    if d_mm <= xs[0]:
        return ys[0] * (d_mm / xs[0]) ** 3
    if d_mm >= xs[-1]:
        return ys[-1] * (d_mm / xs[-1]) ** 3
    for (x0, y0), (x1, y1) in zip(ROUND_CHART, ROUND_CHART[1:]):
        if x0 <= d_mm <= x1:
            return y0 + (y1 - y0) * (d_mm - x0) / (x1 - x0)
    return ys[-1]


def ring_size(us: float | None = None, indian: float | None = None, diameter: float | None = None,
              estimated: bool = False) -> dict | None:
    """One ring size in every system: US (ISO 8653 steps), Indian (the
    circumference minus 40 mm, as Indian jewellers size), inner diameter."""
    if diameter is None:
        if us is not None:
            diameter = 11.63 + 0.8128 * us
        elif indian is not None:
            diameter = (indian + 40) / math.pi
        else:
            return None
    if not 12 <= diameter <= 25:
        return None
    return {"us": round((diameter - 11.63) / 0.8128 * 2) / 2, "in": round(math.pi * diameter - 40),
            "diameter_mm": round(diameter, 1), "estimated": estimated}


# ---------------------------------------------------------------- the page data

def _gold(card, cad, category, cal):
    if card and card["weights_g"].get("18k"):
        w = card["weights_g"]   # the card's own figure for each purity it lists
        return {"source": "card", "by_purity": {p["key"]: round(w.get(p["key"]) or w["18k"] * p["ratio"], 2)
                                                for p in PURITIES}}
    if (cad and cal and category in CAD_WEIGHT_CATEGORIES and cad.get("metal_volume_mm3", 0) > 0
          and cad.get("metal_open_share", 1) <= MAX_OPEN_SHARE):
        k18 = cad["metal_volume_mm3"] / 1000 * cal["k18_g_per_cm3"]
        return {"source": "cad", "by_purity": {p["key"]: round(k18 * p["ratio"], 2) for p in PURITIES}}
    return None


def _stones(card, cad, category):
    if card and card["stones"]:
        groups = [{"shape": shape_name(s["shape"]), "size": size_label(s["size"]), "count": s["count"],
                   "carat": round(s["carat"], 3) if s.get("carat") else None,
                   "setting": setting_label(s["setting"]), "gem": (s.get("gem") or "Diamond").title()}
                  for s in card["stones"]]
        return {"source": "card", "counted": True, "groups": groups,
                "count": sum(g["count"] for g in groups),
                "carat": round(sum(g["carat"] or 0 for g in groups), 2)}
    if card and card["weights_g"].get("18k"):   # a job card with no stones: a plain metal piece
        return {"source": "card", "counted": True, "groups": [], "count": 0, "carat": 0}
    if not cad or not cad.get("stone_groups"):
        return None   # no stones in a CAD file doesn't prove the piece is plain (casting files lack them)
    counted = category != "earrings"
    groups = []
    for g in cad.get("stone_groups", []):
        shape = (g.get("shape") or "").title() or "Diamond"
        L, W = g["size_mm"]
        size = f"{L:.1f} mm" if shape == "Round" or abs(L - W) < 0.05 else f"{L:.1f} × {W:.1f} mm"
        if shape == "Round":
            carat = g["count"] * round_carat((L + W) / 2)
        else:
            carat = g["volume_mm3"] * DIAMOND_CT_PER_MM3
        groups.append({"shape": shape, "size": size, "count": g["count"] if counted else None,
                       "carat": round(carat, 3) if counted else None, "setting": "", "gem": "Diamond"})
    if not counted:   # one row per cut and size, without per-file counts
        seen, uniq = set(), []
        for g in groups:
            if (g["shape"], g["size"]) not in seen:
                seen.add((g["shape"], g["size"]))
                uniq.append(g)
        groups = uniq
    return {"source": "cad", "counted": counted, "groups": groups,
            "count": sum(g["count"] or 0 for g in groups) if counted else None,
            "carat": round(sum(g["carat"] or 0 for g in groups), 2) if counted else None}


def _ring(card, cad):
    if card and card.get("ring_size"):
        r = card["ring_size"]
        if r["unit"] == "US":
            return ring_size(us=r["size"])
        if r["unit"] == "IN":
            return ring_size(indian=r["size"])
    if cad and cad.get("ring"):
        r = cad["ring"]
        return ring_size(diameter=r["inner_diameter_mm"], estimated=r.get("estimated", False))
    return None


def _dimensions(cad, category):
    """Size of one piece in the try-on frame: rings +Y along the finger;
    earrings and pendants +Y up, +X across."""
    if not cad or not cad.get("size_mm"):
        return []
    x, y, z = cad["size_mm"]
    mm = lambda v: f"{v:.1f} mm"
    if category == "ring":
        return [{"label": "Width on the finger", "value": mm(y)}]
    if category == "earrings":
        if not cad.get("pair_in_file"):   # the file may hold one earring or the pair side by side
            return []
        return [{"label": "Height × width (one earring)", "value": f"{y:.1f} × {x:.1f} mm"}]
    if category == "pendant":
        return [{"label": "Height × width", "value": f"{y:.1f} × {x:.1f} mm"},
                {"label": "Depth", "value": mm(z)}]
    a, b, c = sorted((x, y, z), reverse=True)   # bracelets, necklaces: orientation varies
    return [{"label": "Overall size", "value": f"{a:.1f} × {b:.1f} × {c:.1f} mm"}]


def details(m: dict) -> dict:
    data = _specs()
    entry = data.get("designs", {}).get(design_key(m), {})
    card, cad = entry.get("card"), entry.get("cad")
    category = m["category"]
    gold = _gold(card, cad, category, data.get("calibration"))
    stones = _stones(card, cad, category)
    ring = _ring(card, cad) if category == "ring" else None
    return {
        "source": "card" if card else "cad" if cad else None,
        "design_type": (card or {}).get("design_type") or None,
        "gold": gold,
        "stones": stones,
        "ring_size": ring,
        "dimensions": _dimensions(cad, category),
        "purities": [{k: p[k] for k in ("key", "label", "fineness")} for p in PURITIES],
        "purities_for": {mt: PURITIES_FOR[mt] for mt in m["metals"] if mt in PURITIES_FOR},
        "default_purity": DEFAULT_PURITY,
    }
