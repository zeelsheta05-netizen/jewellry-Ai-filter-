"""Design DNA: what the search reads from a shopper's photo.

The DNA of a photo is
  type      ring / earrings / pendant / necklace / bracelet
  metal     the metal colour in the photo
  traits    stones, band width, weight, form: the same zero-shot attributes the
            index stores for every design (attributes.py)
  motifs    style details (floral, halo, twisted, ...) and the centre stone's cut
plus the closest designs in the collection.

Two readings are combined for type and traits, because a phone photo is not a
studio render and neither reading is reliable alone:
  zero-shot  the image model compares the photo with text prompts
  neighbours the photo's nearest catalogue designs vote with their own
             (indexed, render-based) values, weighted by similarity
Motifs are calibrated against the collection: a motif is shown only when the
photo scores higher on it than nearly all designs of its type (a percentile),
because raw zero-shot scores for different words aren't comparable. Motifs that
proved unreliable on photos are left out (see MOTIFS).

Each trait carries the search word that the text parser (query.py) turns into
the same filter, so the page can add a trait to the search as text that the
shopper can read and edit.
"""
import numpy as np

from .attributes import ATTRIBUTES, FORMS, NOUN, TEMPERATURE

CATEGORY_PROMPTS = {
    "ring": ["a photo of a finger ring", "a close-up of a ring"],
    "earrings": ["a photo of a pair of earrings", "a pair of stud or drop earrings"],
    "pendant": ["a photo of a pendant", "a pendant charm on a chain"],
    "necklace": ["a photo of a necklace", "a necklace worn around the neck"],
    "bracelet": ["a photo of a bracelet", "a bangle or tennis bracelet"],
}
# Is it jewellery at all? (a selfie, a cat, a screenshot of text)
JEWELLERY_PROMPTS = {
    "yes": ["a photo of jewellery", "a photo of a gold ring with diamonds", "a pair of earrings",
            "a gold necklace", "a pendant on a chain", "a bracelet", "a hand wearing a ring",
            "a woman wearing earrings and a necklace"],
    "no": ["a photo of a person's face", "a photo of an animal", "a landscape photo", "a screenshot of text",
           "a photo of food", "a photo of a room", "a photo of a car", "a photo of clothes", "a photo of a phone"],
}

# Metal colour: zero-shot on the cropped piece. On simulated photos it named the
# metal of 99.7% of rings, earrings and bracelets in all three golds; reading the
# hue of the pixels managed 83% (skin and wood look like gold). Thin necklaces
# show little metal and rose gold chains often read as white, so the colour is
# only stated when the reading is at least 0.7 sure: then it was right 98% of the
# time (shown for 85% of photos).
METAL_PROMPT = "a piece of jewellery made of {metal}"
METAL_WORDS = {"rose_gold": "rose gold", "white_gold": "white gold", "yellow_gold": "yellow gold"}
METAL_MIN = 0.7

# trait class -> (group, label, search word or None). The words parse into
# the intents of attributes.INTENT_ATTRS (strict ones filter, others rank).
TRAITS = {
    ("stones", "plain"): ("Stones", "No stones", "no stones"),
    ("stones", "solitaire"): ("Stones", "Solitaire", "solitaire"),
    ("stones", "accented"): ("Stones", "Centre + side stones", "side stones"),
    ("stones", "pave"): ("Stones", "Many diamonds", "many diamonds"),
    ("band", "thin"): ("Band", "Thin band", "thin band"),
    ("band", "medium"): ("Band", "Medium band", None),
    ("band", "wide"): ("Band", "Wide band", "wide band"),
    ("weight", "delicate"): ("Look", "Delicate", "delicate"),
    ("weight", "statement"): ("Look", "Statement", "statement"),
    ("form", "stud"): ("Form", "Stud", "stud"),
    ("form", "hoop"): ("Form", "Hoop", "hoop"),
    ("form", "drop"): ("Form", "Drop", "drop"),
    ("form", "jhumka"): ("Form", "Jhumka", "jhumka"),
    ("form", "bangle"): ("Form", "Bangle", "bangle"),
    ("form", "flexible"): ("Form", "Flexible chain", "tennis"),
    ("form", "cuff"): ("Form", "Open cuff", "cuff"),
    ("form", "choker"): ("Form", "Choker", "choker"),
    ("form", "pendant_chain"): ("Form", "Fine chain", None),
    ("form", "collar"): ("Form", "Collar / haar", "statement"),
}
TRAIT_MIN = 0.5       # a trait is shown when its class wins its group with at least this probability

# Motifs: (key, label, prompt phrase, search word, types it applies to or None).
# A motif is shown when the photo outscores MOTIF_PCT of the designs of its type.
# On 559 simulated photos, checked against each source design's own motifs, most
# motifs were right 77-100% of the time at 0.97; on 11 real photos, background
# and lighting push some higher (a wooden or stone background reads as "leaf"),
# so photos need 0.99 (leaf 0.995). "Traditional", "modern", "coloured stones"
# and "moon" were right under half the time even at 0.99 and are not read from
# photos; words in the search box still use them.
MOTIFS = [
    ("floral", "Floral", "floral flower motif design", "floral", None),
    ("heart", "Heart", "heart shaped design", "heart", None),
    ("infinity", "Infinity", "infinity symbol design", "infinity", None),
    ("leaf", "Leaf", "leaf motif design", "leaf", None),
    ("twisted", "Twisted", "twisted intertwined crossover band", "twisted", None),
    ("geometric", "Geometric", "modern geometric design", "geometric", None),
    ("butterfly", "Butterfly", "butterfly motif", "butterfly", None),
    ("star", "Star", "star motif", "star", None),
    ("bow", "Bow / knot", "bow and knot motif", "bow", None),
    ("evil_eye", "Evil eye", "evil eye motif", "evil eye", None),
    ("initial", "Initial letter", "initial letter alphabet design", "initial", ("pendant", "necklace", "bracelet", "ring")),
    ("halo", "Halo", "halo setting, centre stone surrounded by a ring of small diamonds", "halo", None),
    ("three_stone", "Three stone", "three stone design, a larger centre diamond with a smaller diamond on each side", "three stone",
     ("ring", "pendant")),
    ("split_shank", "Split shank", "split shank band that divides into two near the top", "split shank", ("ring",)),
    ("tennis", "Tennis line", "tennis style, a continuous line of identical diamonds", "tennis", ("bracelet", "necklace", "ring")),
    ("rows", "Rows of diamonds", "multiple rows of diamonds, layered design", "rows", None),
    ("openwork", "Openwork", "openwork design with cut-out gaps and filigree", "openwork", None),
    ("vintage", "Vintage / milgrain", "vintage design with milgrain beaded edges", "vintage", None),
    ("bezel", "Bezel set", "stone set in a smooth metal bezel rim", "bezel", None),
]
MOTIF_PCT = 0.99
MOTIF_PCT_OVERRIDE = {"leaf": 0.995}
SHAPES = [
    ("round", "round cut centre stone"), ("oval", "oval cut centre stone"), ("pear", "pear cut centre stone"),
    ("princess", "princess cut centre stone"), ("cushion", "cushion cut centre stone"),
    ("emerald", "emerald cut centre stone"), ("marquise", "marquise cut centre stone"),
    ("radiant", "radiant cut centre stone"), ("asscher", "asscher cut centre stone"),
    ("heart", "heart shaped centre stone"),
]
MOTIF_MAX = 2
# motifs that describe how the diamonds are set: shown with the diamonds, not with the style
DIAMOND_MOTIFS = {"halo", "three_stone", "tennis", "rows", "bezel"}

# ---- diamonds -----------------------------------------------------------------
# Read from pictures only: the uploaded photo, and every catalogue design's own
# renders, by two small readers trained on top of the image model
# (scripts/train_diamond_dna.py, which also prints how often they are right).
# Without the trained readers, the photo is read zero-shot with the prompts below.
LAYOUT_PROMPTS = {
    "solitaire": ["a {n} with one single diamond and no other stones", "a solitaire {n}, one centre diamond only"],
    "centre_side": ["a {n} with one large centre diamond and many small diamonds around it or on the sides",
                    "a {n} with a big centre stone and a halo or pave side stones"],
    "all_small": ["a {n} covered with many small diamonds of the same size, no large centre stone",
                  "a {n} with a row of equal sized diamonds, eternity or pave"],
}
LAYOUT_LABELS = {"solitaire": "Solitaire", "centre_side": "Centre + smaller diamonds", "all_small": "Even-sized diamonds"}
LAYOUT_WORDS = {"solitaire": "solitaire", "centre_side": "side stones", "all_small": "many diamonds"}
CUT_MIN = 0.6          # a photo's centre cut is shown from this probability (91% right on unseen simulated photos)
LAYOUT_MIN = 0.6       # (89% right on unseen simulated photos, shown for 91%)
CENTRE_MIN = 0.5       # ...and only when solitaire + centre layouts together are at least this likely
CARD_CUT_MIN = 0.7     # a catalogue design gets an "oval centre" tag from this probability (99% right on unseen renders)


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def knn_vote(sim: np.ndarray, labels: np.ndarray, classes: list, k: int, sharp: float = 50.0) -> np.ndarray:
    """Similarity-weighted vote of the k nearest designs -> probabilities over classes."""
    top = np.argsort(-sim)[:k]
    w = softmax(sim[top] * sharp)
    out = np.zeros(len(classes))
    for j, wt in zip(top, w):
        if labels[j] in classes:
            out[classes.index(labels[j])] += wt
    return out / max(out.sum(), 1e-9)


def class_prompts(category: str | None) -> dict:
    """{attr: {class: [prompts]}} for the attributes that apply to a type."""
    noun = NOUN.get(category, "piece of jewellery")
    out = {}
    for attr, classes in ATTRIBUTES.items():
        if attr == "band" and category != "ring":
            continue
        out[attr] = {c: [p.format(noun=noun) for p in ps] for c, ps in classes.items()}
    if category in FORMS:
        out["form"] = FORMS[category]
    return out


def zero_shot(vec: np.ndarray, class_vecs: dict) -> dict:
    """{attr: {class: p}} from the photo vector and averaged class prompt vectors."""
    res = {}
    for attr, cv in class_vecs.items():
        names = list(cv)
        p = softmax(np.array([vec @ cv[n] for n in names]) * TEMPERATURE)
        res[attr] = dict(zip(names, p))
    return res


def neighbour_attrs(sim: np.ndarray, attr_arrays: dict, mask: np.ndarray, k: int) -> dict:
    """{(attr, class): p} averaged over the k nearest designs within mask
    (similarity-weighted), from the values the index stores per design."""
    idx = np.flatnonzero(mask)
    if not len(idx):
        return {}
    top = idx[np.argsort(-sim[idx])[:k]]
    w = softmax(sim[top] * 50.0)
    out = {}
    for key, arr in attr_arrays.items():
        vals = arr[top]
        ok = ~np.isnan(vals)
        if ok.any():
            out[key] = float((vals[ok] * w[ok]).sum() / w[ok].sum())
    return out


def percentile(value: float, population: np.ndarray) -> float:
    if not len(population):
        return 0.0
    return float((population < value).mean())


def centred(scores: np.ndarray) -> np.ndarray:
    """Remove each item's mean over all prompts (works on (k,) or (n, k))."""
    return scores - scores.mean(-1, keepdims=True)
