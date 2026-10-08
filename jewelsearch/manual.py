"""Manual search: pick a design's properties instead of describing it.

The picks are written as the prompt the AI search already understands ("rose
gold halo rings with side stones"), so both modes share the same filters,
ranking and notes, a manual search lands in the search history like any other,
and it can be edited in words. Every option's words were chosen so the prompt
parses back to exactly the picks (tests/test_manual.py checks every option and
every pair of options).

How a pick acts, the same as when its words are typed:
  filter  designs without it are left out: type, metal, wearer, no stones,
          diamond layout (solitaire, centre + side stones, many small diamonds),
          band width, form (stud, bangle, choker, ...)
  cut     designs with the cut are ranked first; the number is how many are known to
          have it (named in the id, or read clearly from the renders)
  ai      the image model ranks designs by how much they show it: settings,
          motifs, look, occasion and finish (read from the renders, so no
          design is left out for it and no number is shown)

The numbers next to options are strict: how many designs pass every filter
with that option picked, none loosened (SearchEngine.strict_mask). When a
search still has to loosen a filter, its results say so (search.py).
"""
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from .attributes import STRICT
from .config import CATEGORIES
from .query import parse
from .search import LAYOUT_INTENTS

FILTER_INTENTS = set(STRICT) | set(LAYOUT_INTENTS) | {"plain", "men", "women"}
MAX_PICKS = 16
NO_TYPE_NOUN = "jewellery"   # "rose gold floral jewellery": every type


@dataclass(frozen=True)
class Option:
    group: str
    value: str
    label: str
    words: str                 # what it adds to the prompt
    where: str = "pre"         # "pre" (before the piece's name), "post" (after it) or "noun" (the name itself)
    types: tuple = ()          # the types it applies to (none = every type)

    @property
    def key(self) -> str:
        return f"{self.group}:{self.value}"


# (group key, title, hint shown under the title)
GROUPS = [
    ("type", "Jewellery type", None),
    ("metal", "Metal colour", None),
    ("wearer", "For", None),
    ("stones", "Diamonds & gemstones", None),
    ("setting", "Diamond setting", None),
    ("cut", "Centre stone cut", "The number: designs known to have this cut, from their id or read from their pictures. "
                                "The AI ranks designs with it first."),
    ("shape", "Shape", None),
    ("style", "Style & motif", None),
    ("look", "Look & occasion", None),
    ("finish", "Finish", "Gold purity (14K, 18K, 22K) is chosen when ordering: every design is made in each."),
]
# where pre-words go, nearest the piece's name last: "minimal rose gold floral matte halo oval solitaire thin rings"
PRE_ORDER = {"look": 0, "metal": 1, "style": 2, "finish": 3, "setting": 4, "cut": 5, "stones": 6, "shape": 7}
POST_ORDER = {"stones": 0, "setting": 1, "wearer": 2, "look": 3}

O = Option
OPTIONS = [
    O("type", "ring", "Ring", "rings", "noun"),
    O("type", "earrings", "Earrings", "earrings", "noun"),
    O("type", "pendant", "Pendant", "pendants", "noun"),
    O("type", "necklace", "Necklace", "necklaces", "noun"),
    O("type", "bracelet", "Bracelet", "bracelets", "noun"),

    O("metal", "yellow_gold", "Yellow gold", "yellow gold"),
    O("metal", "white_gold", "White gold", "white gold"),
    O("metal", "rose_gold", "Rose gold", "rose gold"),

    O("wearer", "women", "Women", "for women", "post"),
    O("wearer", "men", "Men", "for men", "post"),
    O("wearer", "kids", "Kids", "for kids", "post"),

    O("stones", "diamond", "With diamonds", "with diamonds", "post"),
    O("stones", "plain", "No stones", "without stones", "post"),
    O("stones", "solitaire", "Solitaire (one diamond)", "solitaire"),
    O("stones", "side_stones", "Centre + side stones", "with side stones", "post"),
    O("stones", "cluster", "Many small diamonds", "cluster"),
    O("stones", "coloured_stone", "Coloured gemstones", "gemstone"),

    O("setting", "halo", "Halo", "halo"),
    O("setting", "three_stone", "Three stone", "three stone"),
    O("setting", "tennis", "Tennis line", "tennis", types=("ring", "bracelet", "necklace")),
    O("setting", "rows", "Rows of diamonds", "with rows of diamonds", "post"),
    O("setting", "single_row", "Single row", "single row"),
    O("setting", "bezel", "Bezel", "bezel"),
    O("setting", "prong", "Prong / claw", "prong"),
    O("setting", "channel", "Channel", "channel set"),

    O("cut", "round", "Round", "round"),
    O("cut", "oval", "Oval", "oval"),
    O("cut", "pear", "Pear", "pear"),
    O("cut", "princess", "Princess", "princess cut"),
    O("cut", "cushion", "Cushion", "cushion cut"),
    O("cut", "emerald", "Emerald", "emerald cut"),
    O("cut", "marquise", "Marquise", "marquise"),
    O("cut", "radiant", "Radiant", "radiant cut"),
    O("cut", "asscher", "Asscher", "asscher cut"),

    O("shape", "thin", "Thin band", "thin", types=("ring",)),
    O("shape", "wide", "Wide band", "wide", types=("ring",)),
    O("shape", "split_shank", "Split shank", "split shank", types=("ring",)),
    O("shape", "flat_top", "Signet / flat top", "signet", types=("ring",)),
    O("shape", "open_design", "Open ended", "open ended", types=("ring",)),
    O("shape", "stud", "Stud", "stud", types=("earrings",)),
    O("shape", "hoop", "Hoop", "hoop", types=("earrings",)),
    O("shape", "drop", "Drop", "drop", types=("earrings",)),
    O("shape", "chandelier", "Chandelier", "chandelier", types=("earrings",)),
    O("shape", "jhumka", "Jhumka", "jhumka", types=("earrings",)),
    O("shape", "bangle", "Bangle", "bangle", types=("bracelet",)),
    O("shape", "cuff", "Cuff", "cuff", types=("bracelet",)),
    O("shape", "flexible", "Flexible chain", "chain bracelet", "noun", types=("bracelet",)),
    O("shape", "choker", "Choker", "choker", types=("necklace",)),
    O("shape", "rani_haar", "Rani haar (long, layered)", "rani haar", "noun", types=("necklace",)),

    O("style", "floral", "Floral", "floral"),
    O("style", "heart", "Heart", "heart"),
    O("style", "infinity", "Infinity", "infinity"),
    O("style", "leaf", "Leaf", "leaf"),
    O("style", "twisted", "Twisted", "twisted"),
    O("style", "geometric", "Geometric", "geometric"),
    O("style", "butterfly", "Butterfly", "butterfly"),
    O("style", "star", "Star", "star"),
    O("style", "moon", "Moon", "moon"),
    O("style", "bow", "Bow / knot", "bow"),
    O("style", "evil_eye", "Evil eye", "evil eye"),
    O("style", "initial", "Initial letter", "initial", types=("pendant", "necklace", "bracelet", "ring")),
    O("style", "traditional", "Traditional", "traditional"),
    O("style", "modern", "Modern", "modern"),

    O("look", "minimal", "Minimal", "minimal"),
    O("look", "statement", "Statement", "statement"),
    O("look", "ornate", "Ornate", "ornate"),
    O("look", "everyday", "Everyday wear", "for everyday wear", "post"),
    O("look", "bridal", "Bridal", "bridal"),
    O("look", "engagement", "Engagement", "engagement"),
    O("look", "party", "Party", "party"),

    O("finish", "brushed", "Matte / brushed", "matte"),
    O("finish", "vintage", "Milgrain edges", "milgrain"),
    O("finish", "openwork", "Openwork / filigree", "filigree"),
]
BY_KEY = {o.key: o for o in OPTIONS}

# picks that contradict each other: the first can't be combined with any of the second
MANY_DIAMONDS = {"setting:halo", "setting:three_stone", "setting:tennis", "setting:rows", "setting:single_row",
                 "setting:channel"}
CONFLICTS = {
    "stones:plain": {k for k in BY_KEY if BY_KEY[k].group in ("setting", "cut")},
    "stones:solitaire": MANY_DIAMONDS,
}
CONFLICT_WHY = {"stones:plain": "Not with “No stones”", "stones:solitaire": "Not with a solitaire (one diamond)"}


def conflict(key: str, picks) -> str | None:
    """Why an option can't be added to these picks, or None."""
    for k in picks:
        if key in CONFLICTS.get(k, ()):
            return CONFLICT_WHY[k]
        if k in CONFLICTS.get(key, ()):
            return f"Not with “{BY_KEY[k].label}”"
    return None


def kind(o: Option) -> str:
    """How the option acts on the search: "filter", "cut" or "ai" (see the module notes)."""
    if o.group in ("type", "metal"):
        return "filter"
    q = parse(o.words if o.where == "noun" else f"{o.words} jewellery")
    if q.shape:
        return "cut"
    return "filter" if set(q.intents) & FILTER_INTENTS else "ai"


KIND = {o.key: kind(o) for o in OPTIONS}


def clean(picks) -> list[str]:
    """Known options only, one per group (the last wins), none for another type,
    and no setting or cut against "no stones" or "solitaire" (the stones pick stays)."""
    by_group = {}
    for k in list(picks)[:MAX_PICKS]:
        if isinstance(k, str) and k in BY_KEY:
            by_group[BY_KEY[k].group] = k
    cat = BY_KEY[by_group["type"]].value if "type" in by_group else None
    out = [k for g, _, _ in GROUPS if (k := by_group.get(g))
           and (not BY_KEY[k].types or cat in BY_KEY[k].types)]
    stones = by_group.get("stones")
    return [k for k in out if k not in CONFLICTS.get(stones, ())]


def compose(picks: list[str]) -> str:
    """The picks as a prompt: "minimal rose gold halo rings with side stones for women"."""
    opts = [BY_KEY[k] for k in clean(picks)]
    if not opts:
        return ""
    noun = next((o.words for o in opts if o.where == "noun" and o.group == "shape"), None) \
        or next((o.words for o in opts if o.where == "noun"), NO_TYPE_NOUN)
    pre = sorted((o for o in opts if o.where == "pre"), key=lambda o: PRE_ORDER[o.group])
    post = sorted((o for o in opts if o.where == "post"), key=lambda o: POST_ORDER[o.group])
    return " ".join([o.words for o in pre] + [noun] + [o.words for o in post])


class Manual:
    """The options with their numbers for the current picks."""

    def __init__(self, engine):
        self.engine = engine
        self.panel = lru_cache(maxsize=2048)(self._panel)

    def _strict(self, picks: tuple) -> np.ndarray:
        text = compose(list(picks))
        return self.engine.strict_mask(parse(text)) if text else np.ones(len(self.engine.meta), dtype=bool)

    def __call__(self, picks) -> dict:
        return self.panel(tuple(clean(picks)))

    def _panel(self, picks: tuple) -> dict:
        chosen = {BY_KEY[k].group: k for k in picks}
        cat = BY_KEY[chosen["type"]].value if "type" in chosen else None
        mask = self._strict(picks)
        groups = []
        for g, title, hint in GROUPS:
            others = tuple(k for k in picks if BY_KEY[k].group != g)
            base = self._strict(others) if g in chosen else mask
            items = []
            for o in (o for o in OPTIONS if o.group == g):
                if o.types and cat not in o.types:
                    continue
                k = KIND[o.key]
                why = None if o.key in picks else conflict(o.key, others)
                if why:
                    items.append({"key": o.key, "label": o.label, "kind": k, "count": None, "picked": False,
                                  "blocked": why})
                    continue
                if k == "filter":
                    n = int(self._strict(others + (o.key,)).sum())
                elif k == "cut":
                    n = int((base & (self.engine.known_cut == o.value)).sum())
                else:
                    n = None
                items.append({"key": o.key, "label": o.label, "kind": k, "count": n, "picked": o.key in picks})
            if items:
                groups.append({"key": g, "title": title, "hint": hint, "options": items})
        prompt = compose(list(picks))
        # a pick that other picks have since ruled out (it was picked first)
        warnings = [f"No design with your other picks is known to have {'an' if o['label'][0] in 'AEIOU' else 'a'} "
                    f"{o['label'].lower()} cut, so the AI ranks them by how much they look like one." if o["kind"] == "cut"
                    else f"No design has “{o['label']}” together with your other picks."
                    for g in groups for o in g["options"] if o["picked"] and o["count"] == 0]
        return {"picks": list(picks), "prompt": prompt, "count": int(mask.sum()) if prompt else len(mask),
                "groups": groups, "warnings": warnings, "types": list(CATEGORIES)}
