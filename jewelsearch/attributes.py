"""Design attributes scored zero-shot by the image model.

Each attribute is a set of mutually exclusive classes described by a few
prompts; a design's class probabilities are a softmax over its similarity to
each class's averaged prompt vector. Scored on the front-view crop, where
stones and band width are clearest.

ATTRIBUTES[attr][class] = prompts; "{noun}" is replaced per category.
"""
import numpy as np

ATTRIBUTES = {
    "stones": {
        "plain": [
            "a plain polished gold {noun} with no gemstones at all",
            "a metal-only {noun} without any diamonds",
            "a simple gold {noun} made only of metal, no stones",
        ],
        "solitaire": [
            "a solitaire {noun} with one single large diamond in the centre and a plain band",
            "a {noun} with just one big diamond held by prongs",
            "a single stone {noun}, one centre diamond only",
        ],
        "accented": [
            "a {noun} with a centre diamond and small diamonds set along the band",
            "a {noun} with a main stone surrounded by a halo of small diamonds",
            "a {noun} with a few small diamonds as accents",
        ],
        "pave": [
            "a {noun} fully covered with many small pave diamonds",
            "a {noun} with rows of tiny diamonds all over",
            "a cluster {noun} with lots of diamonds",
        ],
    },
    "band": {  # rings only
        "thin": ["a very thin delicate ring band", "a slim narrow ring", "a fine wire-thin ring"],
        "medium": ["a ring with a medium width band", "a regular width ring band"],
        "wide": ["a wide thick ring band", "a broad chunky ring", "a heavy wide band ring"],
    },
    "weight": {
        "delicate": ["a small delicate lightweight {noun}", "a minimal dainty {noun} for everyday wear"],
        "statement": ["a big heavy ornate statement {noun}", "a large bold intricate {noun} for a wedding"],
    },
}
NOUN = {"ring": "ring", "earrings": "pair of earrings", "pendant": "pendant", "necklace": "necklace",
        "bracelet": "bracelet"}
TEMPERATURE = 100.0   # SigLIP similarities sit in a narrow band; sharpen before softmax

# How query intents map to attribute classes: intent -> [(attr, class, weight)]
INTENT_ATTRS = {
    "plain": [("stones", "plain", 1.0)],
    "solitaire": [("stones", "solitaire", 1.0)],
    "side_stones": [("stones", "accented", 0.6), ("stones", "pave", 0.6)],
    "halo": [("stones", "accented", 1.0)],
    "diamond": [("stones", "plain", -1.0)],
    "tennis": [("stones", "pave", 0.5), ("form", "flexible", 0.5)],
    "cluster": [("stones", "pave", 1.0)],
    "thin": [("band", "thin", 1.0)],
    "wide": [("band", "wide", 1.0)],
    "men": [("band", "wide", 0.5), ("weight", "statement", 0.3)],
    "minimal": [("weight", "delicate", 0.5)],
    "everyday": [("weight", "delicate", 0.5)],
    "statement": [("weight", "statement", 0.7)],
    "bridal": [("weight", "statement", 0.7), ("stones", "plain", -0.5)],
    "rani_haar": [("weight", "statement", 1.0), ("stones", "pave", 0.5)],
    "big_stone": [("stones", "solitaire", 0.6)],
    "three_stone": [("stones", "accented", 1.0)],
    "chandelier": [("form", "drop", 1.0), ("weight", "statement", 0.7)],
    "rows": [("stones", "pave", 0.5)],
    "ornate": [("weight", "statement", 0.8)],
    "stud": [("form", "stud", 1.0)],
    "hoop": [("form", "hoop", 1.0)],
    "drop": [("form", "drop", 1.0)],
    "jhumka": [("form", "drop", 1.0), ("weight", "statement", 0.3)],
    "bangle": [("form", "bangle", 1.0)],
    "cuff": [("form", "cuff", 1.0)],
    "flexible": [("form", "flexible", 1.0)],
    "choker": [("form", "choker", 1.0)],
    "pendant_chain": [("form", "pendant_chain", 1.0)],
}
# Intents strict enough to act as filters (designs clearly lacking them are
# dropped when enough others remain).
STRICT = {"solitaire": ("stones", "solitaire"),
          "thin": ("band", "thin"), "wide": ("band", "wide"),
          "stud": ("form", "stud"), "hoop": ("form", "hoop"), "drop": ("form", "drop"),
          "jhumka": ("form", "drop"), "chandelier": ("form", "drop"), "bangle": ("form", "bangle"), "cuff": ("form", "cuff"),
          "flexible": ("form", "flexible"), "choker": ("form", "choker"),
          "pendant_chain": ("form", "pendant_chain")}
# forms the catalogue does not have: what is shown instead, and the note
MISSING_FORMS = {"jhumka": "The catalogue has no jhumka-style earrings; showing drop earrings."}


def class_vectors(embedder, category: str) -> dict:
    noun = NOUN.get(category, "piece of jewellery")
    out = {}
    for attr, classes in ATTRIBUTES.items():
        if attr == "band" and category != "ring":
            continue
        vecs = {}
        for cls, prompts in classes.items():
            v = embedder.texts([p.format(noun=noun) for p in prompts]).mean(0)
            vecs[cls] = v / np.linalg.norm(v)
        out[attr] = vecs
    return out


def score(design_vecs: np.ndarray, cvecs: dict) -> dict:
    """-> {attr: {class: probs array (n,)}}"""
    res = {}
    for attr, vecs in cvecs.items():
        names = list(vecs)
        logits = design_vecs @ np.stack([vecs[n] for n in names]).T * TEMPERATURE
        logits -= logits.max(1, keepdims=True)
        p = np.exp(logits)
        p /= p.sum(1, keepdims=True)
        res[attr] = {n: p[:, i] for i, n in enumerate(names)}
    return res


# Sub-type ("form") of a piece, per category. Scored on the front view at
# engine start-up (cheap: text prompts only), so no re-index is needed.
FORMS = {
    "earrings": {
        "stud": ["a pair of small stud earrings that sit on the earlobe", "a pair of button tops earrings"],
        "hoop": ["a pair of round hoop earrings", "a pair of circular huggie hoop earrings, bali"],
        "drop": ["a pair of long drop earrings dangling below the ear", "a pair of dangler earrings"],
        "jhumka": ["a pair of indian jhumka earrings with a bell shaped dome at the bottom",
                   "a pair of traditional jhumki bell earrings"],
    },
    "bracelet": {
        "bangle": ["a rigid round bangle bracelet", "a solid gold kada bangle"],
        "flexible": ["a flexible chain bracelet", "a tennis bracelet, a flexible line of diamonds with a clasp"],
        "cuff": ["an open cuff bracelet with a gap", "an open ended cuff bangle"],
    },
    "necklace": {
        "choker": ["a short choker necklace that sits tight around the neck"],
        "pendant_chain": ["a thin simple chain necklace with a small pendant", "a delicate fine chain necklace"],
        "collar": ["a large heavy collar necklace, an indian bridal haar",
                   "a wide statement necklace covered in stones"],
    },
}


def form_scores(embedder, vecs, cats) -> dict:
    """-> {("form", class): probs (n,), NaN outside the class's category}"""
    import numpy as np
    out = {}
    for cat, classes in FORMS.items():
        idx = np.flatnonzero(cats == cat)
        names = list(classes)
        cv = np.stack([embedder.texts(p).mean(0) for p in classes.values()])
        cv /= np.linalg.norm(cv, axis=1, keepdims=True)
        logits = vecs[idx] @ cv.T * TEMPERATURE
        logits -= logits.max(1, keepdims=True)
        p = np.exp(logits)
        p /= p.sum(1, keepdims=True)
        for i, n in enumerate(names):
            arr = out.setdefault(("form", n), np.full(len(vecs), np.nan))
            arr[idx] = p[:, i]
    return out
