"""Training and test answers for the diamond reader (scripts only).

Search never reads these: they only teach the image model what each cut and
layout looks like (train_diamond_dna.py) and check how often it is right
(eval_photo_search.py). The answers come from the client's job cards and the
CAD files (data/cad/specs.json), and from a cut named in the design id
(DDLR-092-OV).

  cut     the centre stone's cut, when one stone stands out: the largest stone
          (one per piece, two for a pair of earrings) at least 1.6 times longer
          than any other stone
  layout  solitaire    that centre stone and no other diamonds
          centre_side  a centre stone with smaller diamonds (halo, side stones)
          all_small    no stone stands out (pave, eternity, cluster, tennis, an even three-stone)
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from jewelsearch import dna, purchase  # noqa: E402
from jewelsearch.search import family_and_shape  # noqa: E402

CUTS = [c for c, _ in dna.SHAPES]
LAYOUTS = list(dna.LAYOUT_PROMPTS)
CENTRE_RATIO = 1.6


def _long(size) -> float:
    if isinstance(size, (list, tuple)):
        return float(max(size)) if size else 0.0
    nums = []
    for part in re.split(r"\s*[xX×*]\s*", str(size or "").strip()):
        try:
            nums.append(float(part))
        except ValueError:
            pass
    return max(nums) if nums else 0.0


def labels(m: dict) -> tuple[str | None, str | None]:
    """(centre cut or None, layout or None) for one design of the index."""
    entry = purchase._specs().get("designs", {}).get(purchase.design_key(m), {})
    card, cad = entry.get("card"), entry.get("cad")
    if card and card.get("stones"):
        groups = [(purchase.shape_name(s["shape"]).lower(), _long(s["size"]), s["count"]) for s in card["stones"]]
    elif cad and cad.get("stone_groups"):
        groups = [((g.get("shape") or "").lower(), _long(g["size_mm"]), g["count"]) for g in cad["stone_groups"]]
    else:
        groups = []
    groups = [g for g in groups if g[2]]
    cut = layout = None
    if groups:
        groups.sort(key=lambda g: -g[1])
        (top_cut, top_long, top_n), rest = groups[0], groups[1:]
        next_long = max((g[1] for g in rest), default=0.0)
        if top_n <= (2 if m["category"] == "earrings" else 1) and top_long >= CENTRE_RATIO * max(next_long, 1e-6):
            layout = "centre_side" if rest else "solitaire"
            cut = top_cut
        else:
            layout = "all_small"
    named = family_and_shape(m["design_id"], m["folders"][0].rsplit("/", 1)[0])[1]
    cut = named or cut   # the designer's own label first
    return (cut if cut in CUTS else None), layout
