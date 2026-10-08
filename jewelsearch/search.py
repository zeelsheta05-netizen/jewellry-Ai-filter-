"""Ranking: filters -> text/image similarity + attribute match -> diversity -> 8 designs."""
import hashlib
import json
import re
import threading
from dataclasses import dataclass

import numpy as np
from PIL import ImageOps

from . import dna, photo, tryon
from . import details as details_mod
from .domain import Domain
from .attributes import INTENT_ATTRS, MISSING_FORMS, STRICT, TEMPERATURE, form_scores
from .config import CATEGORIES, INDEX, METALS, media_token, thumb_name


def media_url(relpath: str) -> str:
    """The link a page uses for a full-size render or 3D video (server decides where it comes from)."""
    return "/media/" + media_token(relpath)
from .embedder import Embedder
from .query import CATEGORY_NOUN, ParsedQuery, bank_queries, parse

TOP_K = 8
BUDGET_NOTE = "Prices aren't in the catalogue yet, so “{}” was left out."
POOL = 80             # candidates considered for diversity re-ranking
MMR_LAMBDA = 0.8      # 1.0 = pure relevance, lower = more varied results
ATTR_WEIGHT = 1.5     # attribute match, in units of similarity standard deviations
NEG_TEXT_WEIGHT = 0.5
SHAPE_BOOST = 1.0
STRICT_MIN_PROB = 0.3     # below this a design "clearly lacks" a strict attribute
STRICT_MIN_LEFT = 2 * TOP_K
STRICT_FALLBACK_SHARE = 0.3
# Diamond layouts are filtered by the trained reader (scripts/train_diamond_dna.py) when it
# is there: every design read from all its renders. Checked against job cards / CAD on
# 3,282 designs, the zero-shot "solitaire" filter it replaces kept 1,039 designs, 39.5% of
# them solitaires; the reader keeps 500, 92.6% right, and finds 95% of the solitaires.
# Centre + side stones: 88.5% right (the image model's words found none of them);
# many small diamonds: 94.0% right.
LAYOUT_INTENTS = {"solitaire": 0, "side_stones": 1, "cluster": 2}   # -> dna.LAYOUT_PROMPTS order
LAYOUT_MIN_PROB = 0.5
# how a style is named in a note when its filter has to be loosened
STYLE_WORD = {"pendant_chain": "pendant on a chain", "side_stones": "side stones", "big_stone": "big stone",
              "three_stone": "three stone", "rani_haar": "rani haar"}
PLAIN_MAX_STONE = 0.03        # stone share (of the piece's pixels) that counts as "no stones"
PLAIN_MIN_PROB = 0.5          # ...and the image model must also see it as plain metal
STONE_WEIGHT = 3.0
HUB_WEIGHT = 1.0      # hubness correction strength (see _hub_bias)
HUB_TOP_K = 30
DUPLICATE_COS = 0.995 # front views this alike are the same render in two folders
BASE_WEIGHT = 0.5     # how much of the generic "a photo of a <type>" to remove
UNWANTED_MAX_PROB = 0.6   # above this a design "clearly has" an unwanted attribute
BROWSE_PAGE = 24      # designs per page when browsing a category without a prompt
BROWSE_SPREAD = 6     # a design is kept apart from the last few shown when it looks nearly the same...
BROWSE_ALIKE = 0.93   # ...as one of them (front-view cosine)

# ---- search by photo ----
# Chosen on 559 simulated shopper photos (see scripts/eval_photo_search.py). The
# design was found on the first page for 69% of them with the whole photo, 79%
# with the photo cropped to the piece, and 86% when the crop is also compared
# with each view of a design (its closest view counts). Query expansion and a
# hubness correction were tried and made it worse, so they are not used.
VIEW_MIX = 1.0            # weight of a design's closest single view against the design as a whole
TYPE_ZS = 0.3             # type: share of the zero-shot reading; the rest is the nearest designs' own types
TYPE_K = 10
TYPE_SURE = 0.6           # below this the photo's type is not used as a filter (right 97% of the time above it)
TRAIT_ZS = 0.5
TRAIT_K = 10
JEWEL_MIN = 0.5
SAME_SIM, SAME_GAP = 0.85, 0.03   # "same design": right ~97% of the time when it is shown
TIERS = ((0.87, "very_close"), (0.82, "close"))   # photo similarity -> match label, else "similar"
PHOTO_TEXT_WEIGHT = 0.6   # words next to a photo refine it; the photo leads
PHOTO_DNA_WEIGHT = 0.3    # agreement of stones / band / weight / form with the photo's
# agreement of each design's diamonds with the photo's, both read from pictures by the trained readers.
# On 300 simulated photos (scripts/eval_photo_search.py --set PHOTO_DIAMOND_WEIGHT=...):
#   weight   design on page 1   results with the photo's layout / centre cut
#   0        86.3%              81.7% / 61.1%
#   0.15     87.0%              83.6% / 65.4%
#   0.3      87.0%              83.9% / 67.9%
#   0.5      87.0%              84.6% / 70.1%
# Those designs were in the readers' training, so readings there are better than on new pieces: 0.3, not more.
PHOTO_DIAMOND_WEIGHT = 0.3
PHOTO_MMR = 0.9           # closeness matters more than variety when matching a photo
# DINOv2 (dino.py) next to SigLIP2, each standardised within the filters: on 200 simulated photos
# the design came first 58% -> 78.5% of the time, on the first page 86% -> 93.5% (dino.py has more).
DINO_WEIGHT = 1.0
# Match labels from that combined score (z within the photo's type), not from one model's similarity:
# shop renders of designs NOT in the collection reach SigLIP 0.92 / DINOv2 0.95 with ordinary halo
# rings, so similarity alone called them "very close". What sets a real match apart is how far it
# leads the next design (200 simulated photos + 8 real product pages of outside designs):
#   lead >= 0.5: called "same" 120 times, right 96%;  >= 1.0: 87 times, right 99%;  >= 1.3: 77, 100%
SAME_LEAD = 1.3   # full engine, two runs of 200 photos: shown for 36-37%, right 95.9% / 98.6%
# "Very close" / "Close match" were given to every card whose combined score was high (z >= 5 / 4).
# Scores are relative, so the top of every page is high: of the cards labelled close or better, 25%
# were the photo's design on clean simulated photos and 12% on worn ones, and one model's spike
# (DINOv2 matching the fingers of a hand photo with a ring's rows of diamonds) put a wrong ring first
# as a "Close match". Now only the first card can carry a match label, from how far it leads the
# next design family on the combined score, and only when both models rate it well. First cards
# per lead (225 clean / 240 worn simulated photos): 0.5-0.8 right 77% / 60%, 0.8-1.3 88% / 62%,
# 1.3 and more 97.5% / 93.5% (that is SAME_LEAD, "Same design"); under 0.5 about 30% on worn photos.
# On 198 real shop photos (a product's styled and worn photos, checked against what its own studio shot
# finds first) the 0.5-0.8 band agreed 5 times in 16, so "Close match" is no longer given.
TIERS_LEAD = ((0.8, "very_close"),)   # else "similar"
LABEL_BOTH_MIN = 1.5      # each model's own standardised score must be at least this
# Finding the piece in a busy photo (worn on a hand, by a face, on silk next to tweezers): the
# background cut-out (photo.py) needs a plain background; without it the whole photo was compared
# with the catalogue, and when it "found" something on such photos it was the person. A trained
# reader of DINOv2's patches (scripts/train_piece_finder.py) marks the patches that are jewellery,
# and the search reads the crop around them (when the cut-out and the finder disagree, see
# CUT_MIN_HEAT). Measured with the engine (JEWEL_DETAILS=0), before -> after:
#   240 worn-style photos (catalogue pieces on real hand / face photos, scratch test set):
#       design first 3.8% -> 37.9%, on the first page 8.8% -> 61.7% (rings 1.2% -> 32.5%, 3.1% -> 56.2%),
#       type right 52.9% -> 92.1%
#   300 simulated photos (scripts/eval_photo_search.py): first 75.3% -> 78.3%, page 92.0% -> 96.0%
#       (wood / fabric / stone 60.0% -> 70.7%, 80.0% -> 93.3%; white catalogue screenshots unchanged)
#   about 0.2 s more per photo.
PIECE_FINDER = 1          # 0: the background cut-out alone, as before
PIECE_MIN_P = 0.7         # a patch counts as jewellery when the reader is at least this sure
# Groups of jewellery patches kept besides the most certain one (photo.heat_box). With 0.25 (a pendant and
# its chain, both earrings) worn photos lost: design first 37.9% -> 34.6%, rings on the first page
# 56.2% -> 49.4%; clean photos gained 1 point first and none on the first page. 1.0 = that group only.
PIECE_KEEP = 1.0
# When the cut-out and the finder disagree, the cut-out is kept if it is mostly jewellery by the finder's
# map. Mean map over the cut-out (5th / 50th / 95th percentile): the piece on 276 simulated photos
# 0.45 / 0.91 / 0.98; the person on 84 worn-style photos 0.03 / 0.10 / 0.22. (The collection's nearest
# design was tried as the judge: a real hand photo came closer to it whole than its ring did, 0.857 / 0.843.)
CUT_MIN_HEAT = 0.3
# design details read by the vision-language model (details.py) from the photo and from every design:
# designs sharing the photo's clear details (its "halo", "clusters", diamond-set band readings) rank higher.
# Measured 2026-10-05 (found first / on the first page; real: agrees with the product's own studio shot):
#   setting                 300 simulated     88 styled shop     100 worn shop photos
#   off                     78.0% / 94.7%     23.9% / 65.9%      19.0% / 55.0%
#   band only, 0.5          78.7% / 93.7%     22.7% / 61.4%      22.0% / 56.0%
#   all three, 0.5          78.0% / 93.3%     25.0% / 68.2%      23.0% / 62.0%
#   band only, 1.0          75.7% / 93.0%     21.6% / 58.0%      21.0% / 54.0%
# A photo with details takes ~3.2 s to read instead of ~0.8 s.
DETAILS_WEIGHT = 0.5

_SHAPE_SUFFIX = re.compile(r"^(.*?)[\s\-]*\b(AC|CU|EM|PE|RA|PR|RD|RND|HR|MQ|OV|OVAL|PEAR)$")
_SHAPE_NAMES = {
    "AC": "asscher", "CU": "cushion", "EM": "emerald", "PE": "pear", "PEAR": "pear", "RA": "radiant",
    "PR": "princess", "RD": "round", "RND": "round", "HR": "heart", "MQ": "marquise",
    "OV": "oval", "OVAL": "oval",
}
PLURAL = {"ring": "rings", "earrings": "earrings", "pendant": "pendants", "necklace": "necklaces", "bracelet": "bracelets"}
DISPLAY_METAL_ORDER = ["yellow_gold", "rose_gold", "white_gold"]
ATTR_LABELS = {
    ("stones", "plain"): "no stones", ("stones", "solitaire"): "solitaire",
    ("stones", "accented"): "centre + side stones", ("stones", "pave"): "many diamonds",
    ("band", "thin"): "thin band", ("band", "wide"): "wide band",
    ("weight", "delicate"): "delicate", ("weight", "statement"): "statement",
    ("form", "stud"): "stud", ("form", "hoop"): "hoop", ("form", "drop"): "drop",
    ("form", "bangle"): "bangle", ("form", "cuff"): "cuff", ("form", "choker"): "choker",
}


def index_key(meta: list[dict]) -> str:
    """Fingerprint of the index's design order (see scripts/build_view_index.py)."""
    ids = "\n".join(f"{m['design_id']}|{m['folders'][0]}" for m in meta)
    return hashlib.sha1(ids.encode()).hexdigest()


@dataclass
class PhotoQuery:
    """A shopper's photo, read once (SearchEngine.read_photo)."""
    vec: np.ndarray        # the photo's vector
    sim: np.ndarray        # its similarity to every design
    category: str | None   # its type when the reading is sure
    metal: str | None      # its metal colour, when the reading is clear
    attrs: dict            # (attr, class) -> probability, for DNA agreement
    dna: dict              # what the page shows
    diamond_agree: np.ndarray | None = None   # how well each design's diamonds match the photo's
    dino: np.ndarray | None = None            # DINOv2 similarity to every design (its closest view)
    details: dict | None = None               # the photo's details: question key -> probability of yes
    detail_agree: np.ndarray | None = None    # share of the photo's clear details each design has
    fused: np.ndarray | None = None           # SigLIP2 + DINOv2, standardised within the photo's type
    dvec: np.ndarray | None = None            # the photo's own DINOv2 vector (web panel: webproducts.for_photo)


def _overlap(a, b) -> float:
    """Intersection over union of two boxes (x0, y0, x1, y1)."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _a(category: str) -> str:
    return {"earrings": "a pair of earrings"}.get(category, "an " + category if category[0] in "aeiou" else "a " + category)


def _tier_lead(lead: float) -> str:
    for t, name in TIERS_LEAD:
        if lead >= t:
            return name
    return "similar"


def _tier(sim: float) -> str:
    for t, name in TIERS:
        if sim >= t:
            return name
    return "similar"


def family_and_shape(design_id: str, folder: str):
    """Stone-cut variants (DDLR-092-AC, -CU, ...) share a family key."""
    m = _SHAPE_SUFFIX.match(design_id.strip().upper())
    if not m:
        return f"{folder}|{design_id}", None
    base = re.sub(r"\s+", "", m.group(1))
    return f"{folder}|{base}", _SHAPE_NAMES[m.group(2)]


def _id_key(s: str) -> str:
    return re.sub(r"[^0-9a-z#]", "", s.lower())


class SearchEngine:
    def __init__(self, embedder: Embedder | None = None, judge="auto", details="auto"):
        self.emb = embedder or Embedder()
        # the image model's calibrated match score: sigmoid(scale * cosine + bias) (SigLIP)
        m = self.emb.model
        self.match_scale = float(m.logit_scale.detach().exp()) if hasattr(m, "logit_scale") else None
        self.match_bias = float(m.logit_bias.detach()) if getattr(m, "logit_bias", None) is not None else 0.0
        self.vecs = np.load(INDEX / "embeddings.npy")
        self.front = np.load(INDEX / "front.npy")
        self.meta = [json.loads(l) for l in (INDEX / "meta.jsonl").read_text().splitlines()]
        self.by_uid = {}
        self.families: dict[str, list[int]] = {}
        for i, m in enumerate(self.meta):
            m["uid"] = i
            parent = m["folders"][0].rsplit("/", 1)[0]
            m["family"], m["stone_shape"] = family_and_shape(m["design_id"], parent)
            self.families.setdefault(m["family"], []).append(i)
            self.by_uid[i] = m
        self._merge_duplicates()
        self.cat = np.array([m["category"] for m in self.meta])
        self.metal_mask = {mt: np.array([mt in m["metals"] for m in self.meta]) for mt in METALS}
        self.shape = np.array([m["stone_shape"] or "" for m in self.meta])
        # men's designs: filed under "Mens ring" / "Gent's Ring" folders; the
        # SBMR and GGR series are men's rings throughout
        self.mens = np.array([any(re.search(r"\bmen|gent", f, re.I) for f in m["folders"])
                              or bool(re.match(r"^\s*(SBMR|GGR)\b", m["design_id"], re.I)) for m in self.meta])
        # measured share of diamond pixels (scripts/stone_pixels.py); NaN if unmeasured
        self.stone = np.array([np.nan if m.get("stone_frac") is None else m["stone_frac"] for m in self.meta])
        self.attr = {}   # (attr, class) -> probs; NaN where the attribute doesn't apply
        for m_i, m in enumerate(self.meta):
            for attr, classes in m.get("attrs", {}).items():
                for cls, p in classes.items():
                    self.attr.setdefault((attr, cls), np.full(len(self.meta), np.nan))[m_i] = p
        self.attr.update(form_scores(self.emb, self.front, self.cat))
        self.hub = self._hub_bias()
        self._browse: dict[str, list[int]] = {}
        self._load_views()
        self._load_dino()
        self._load_details(details)
        self._load_piece_finder()
        self._load_diamond_reader()
        # each design's centre cut as far as it is known: named in its id, else read clearly
        # from its renders (both right every time they could be checked against job cards / CAD).
        # Ranking still boosts the id-named ones only: adding the read ones didn't rank better.
        self.known_cut = np.array([s or (self._design_cut(u) or "") for u, s in enumerate(self.shape)])
        self._photo_lock = threading.Lock()   # one photo through the model at a time
        self._dna_text: dict = {}             # cached text vectors of the DNA prompts
        # the only dataset files pages may open, by opaque ID (a page never names a dataset path)
        self.allowed_media = set()
        for m in self.meta:
            for views in m["images"].values():
                self.allowed_media.update(views.values())
            self.allowed_media.update(m["videos"].values())
        self.media_by_token = {media_token(p): p for p in self.allowed_media}
        # orders saved before the catalogue showed originals hold crop links: crop name -> render
        self.path_by_crop = {thumb_name(p): p for p in self.allowed_media}
        # design ids typed into the box ("DDLR-092", "ddlr 092 ov"): looked up, never refused
        self.by_id_key: dict[str, list[int]] = {}
        for u, m in enumerate(self.meta):
            base = _SHAPE_SUFFIX.match(m["design_id"].strip().upper())
            for k in {_id_key(m["design_id"]), _id_key(base.group(1)) if base else ""} - {""}:
                self.by_id_key.setdefault(k, []).append(u)
        self.domain = Domain(self, judge=judge)   # jewellery prompts only (jewelsearch/domain.py)

    # ---- public API --------------------------------------------------------
    def search(self, prompt: str, category: str | None = None, metal: str | None = None,
               k: int = TOP_K, page: int = 0) -> dict:
        ids = self.find_ids(prompt)
        if ids:
            return self._id_results(prompt, ids, metal, k, page)
        v = self.domain.check(prompt)
        q = parse(v.text)
        q.raw = prompt
        if not v.ok:   # not a jewellery search: say so, show nothing
            r = v.refusal(prompt)
            return {"query": {**self._explain(q), "refused": r}, "refused": r, "notes": [], "page": page,
                    "has_more": False, "matches_in_filter": 0, "results": []}
        # None = keep what the prompt said, "any" = drop the filter
        if category:
            q.category = None if category == "any" else category
            q.categories = [q.category] if q.category else []
        if metal:
            q.metal = None if metal == "any" else metal
            q.metal_note = None

        mask, notes = self._filters(q)
        if v.budget:
            notes.append(BUDGET_NOTE.format(v.budget))
        if q.shape and mask.any() and not (mask & (self.known_cut == q.shape)).any():
            # a cut only orders the results: say so when no design here is known to have it
            notes.append(f"No design here is known to have {'an' if q.shape[0] in 'aeiou' else 'a'} {q.shape} centre "
                         "stone, so they are ranked by how much they look like one.")
        qvec = self._query_vector(q)
        sim = 0.5 * (self.vecs @ qvec) + 0.5 * (self.front @ qvec) - HUB_WEIGHT * self.hub
        score = self._standardise(sim, mask) + self._intent_score(q, mask)
        picked = self._mmr(score, mask, k * (page + 1), MMR_LAMBDA)[k * page:]
        return {
            "query": self._explain(q),
            "notes": notes,
            "page": page,
            "has_more": int(mask.sum()) > k * (page + 1),
            "matches_in_filter": int(mask.sum()),
            "results": [self.card(i, q.metal) for i in picked],
        }

    def find_ids(self, prompt: str) -> list[int]:
        """Designs whose id is exactly what was typed (ignoring case, spaces, dashes)."""
        key = _id_key(prompt)
        if len(prompt) > 40 or len(key) < 3 or not re.search(r"\d", key):
            return []
        return self.by_id_key.get(key, [])

    def _id_results(self, prompt: str, uids: list[int], metal: str | None, k: int, page: int) -> dict:
        q = parse("")
        q.raw = prompt
        n = len(uids)
        return {"query": self._explain(q), "notes": [f"Design id: {n} design{'s' if n > 1 else ''} with this id."],
                "page": page, "has_more": n > k * (page + 1), "matches_in_filter": n,
                "results": [self.card(u, metal if metal not in (None, "any") else None)
                            for u in uids[k * page:k * (page + 1)]]}

    def picture_score(self, text: str, q: ParsedQuery) -> float:
        """How well the best design here matches the prompt as a picture caption,
        against a plain "a photo of a <piece>": the image model's calibrated
        match logits (SigLIP), the design's best view each. Far below 0 means no
        design looks like what the prompt describes (see jewelsearch/domain.py)."""
        if self.match_scale is None:
            return 0.0
        mask, _ = self._filters(q)
        if not mask.any():
            mask = np.ones(len(self.meta), dtype=bool)
        noun = CATEGORY_NOUN.get(q.category, "piece of jewellery")
        t = self.emb.texts(["a photo of " + text, f"a photo of a {noun}"])
        cos = np.maximum(self.front @ t.T, self.vecs @ t.T)
        if self.views is not None:
            best = np.maximum.reduceat(self.views @ t.T, self._view_starts, axis=0)
            cos[self._view_designs] = np.maximum(cos[self._view_designs], best)
        best = (self.match_scale * cos[mask] + self.match_bias).max(0)
        return float(best[0] - best[1])

    def categories(self) -> list[dict]:
        """Every category with its number of designs and its first design when
        browsing, for the "Browse all categories" tiles."""
        out = []
        for c in CATEGORIES:
            order = self.browse_order(c)
            if order:
                out.append({"category": c, "count": int((self.cat == c).sum()), "designs": len(order),
                            "thumb": self._cover(order)})
        return out

    def _cover(self, order: list[int]) -> str:
        """A category tile's picture: the richest design whose photo isn't
        mostly a flat grey box (a few renders sit on one)."""
        from PIL import Image

        from .config import CROPS
        for u in order[:40]:
            card = self.card(u)
            path = self._front_path(self.by_uid[u], card["metal_shown"])
            try:   # judged on the small copy the models read; the tile shows the original
                a = np.asarray(Image.open(CROPS / thumb_name(path)).convert("RGBA")).astype(int)
            except OSError:
                continue
            rgb = a[..., :3]
            flat_grey = ((a[..., 3] > 200) & (rgb.min(-1) > 150) & (rgb.max(-1) < 235) & (np.ptp(rgb, -1) < 12)).mean()
            if flat_grey < 0.25:
                return card["thumb"]
        return self.card(order[0])["thumb"]

    def browse(self, category: str, page: int = 0, k: int = BROWSE_PAGE) -> dict:
        order = self.browse_order(category)
        return {"category": category, "total": len(order), "page": page,
                "has_more": len(order) > k * (page + 1),
                "results": [self.card(u) for u in order[k * page:k * (page + 1)]]}

    def browse_order(self, category: str) -> list[int]:
        """A whole category without a prompt: richest pieces first (measured
        stone share: the heavy, fully set designs), one per stone-cut family,
        and a design that looks nearly the same as one just shown waits a few
        places. Deliberately no hubness correction: it pushes exactly the
        classic, richly set pieces down (see _hub_bias)."""
        if category in self._browse:
            return self._browse[category]
        idx = np.flatnonzero(self.cat == category)
        rich = np.nan_to_num(self.stone[idx], nan=float(np.nanmedian(self.stone)))
        pending, families = [], set()
        for u in idx[np.argsort(-rich, kind="stable")]:
            if self.meta[u]["family"] not in families:
                families.add(self.meta[u]["family"])
                pending.append(int(u))
        v = self.front / np.linalg.norm(self.front, axis=1, keepdims=True)
        order = []
        while pending:
            pick = 0
            if order:
                window = pending[:50]
                sims = v[window] @ v[order[-BROWSE_SPREAD:]].T
                ok = np.flatnonzero(sims.max(axis=1) < BROWSE_ALIKE)
                pick = int(ok[0]) if len(ok) else 0
            order.append(pending.pop(pick))
        self._browse[category] = order
        return order

    def similar(self, uid: int, metal: str | None = None, k: int = TOP_K) -> dict:
        src = self.by_uid[uid]
        mask = self.cat == src["category"]
        for j in self.families[src["family"]]:
            mask[j] = False   # skip the design itself and its stone-cut variants
        score = 0.5 * (self.vecs @ self.vecs[uid]) + 0.5 * (self.front @ self.front[uid])
        picked = self._mmr(score, mask, k, 0.85)
        return {"source": self.card(uid, metal), "results": [self.card(i, metal) for i in picked]}

    def detail(self, uid: int) -> dict:
        m = self.by_uid[uid]
        variants = [self.card(j) for j in self.families[m["family"]] if j != uid]
        order = [m["front_view"]] + [v for v in sorted(m["images"][m["embed_metal"]]) if v != m["front_view"]]
        return {
            **self.card(uid),
            "folders": m["folders"],
            "has_cad": m["has_cad"],
            "category_source": m["category_source"],
            "views": {mt: [{"thumb": media_url(vs[v]), "full": media_url(vs[v])} for v in order if v in vs]
                      for mt, vs in m["images"].items()},
            "videos": {mt: media_url(p) for mt, p in m["videos"].items()},
            "variants": variants,
        }

    def original_url(self, url):
        """A preview link (data/crops) -> the original render's link, while the design is in
        the catalogue (orders keep preview links: they outlive catalogue changes)."""
        if isinstance(url, str) and url.startswith("/crops/"):
            path = self.path_by_crop.get(url.rsplit("/", 1)[1])
            return media_url(path) if path else url
        return url

    def preview_url(self, url):
        """An original render's link -> its small preview (data/crops), for records that must
        keep a picture after the design leaves the catalogue."""
        if isinstance(url, str) and url.startswith("/media/"):
            path = self.media_by_token.get(url.rsplit("/", 1)[1])
            return "/crops/" + thumb_name(path) if path else url
        return url

    def card(self, uid: int, metal: str | None = None) -> dict:
        m = self.by_uid[uid]
        show = metal if metal in m["images"] else next(
            (mt for mt in DISPLAY_METAL_ORDER if mt in m["images"]), None)
        tags = [label for key, label in ATTR_LABELS.items()
                if key in self.attr and np.nan_to_num(self.attr[key][uid]) >= 0.5]
        cut = self._design_cut(uid)
        if cut:
            tags.append(f"{'emerald-cut' if cut == 'emerald' else cut} centre")
        return {
            "uid": uid, "design_id": m["design_id"], "category": m["category"],
            "metals": m["metals"], "metal_shown": show, "stone_shape": m["stone_shape"],
            "thumb": self._front_url(m, show),
            "thumbs_by_metal": {mt: self._front_url(m, mt) for mt in m["images"]},
            "tags": tags,
            "has_video": bool(m["videos"]),
            "tryon": tryon.part_for(m),   # "hand" / "face" / "neck": which saved photo it is tried on
            "live": tryon.live_model(m),  # model slug for the live camera (rings), else None
            "variant_count": len(self.families[m["family"]]) - 1,
        }

    # ---- search by photo ---------------------------------------------------
    def warm_up(self):
        """Make now what the first photo of each type would otherwise wait for: the
        types' text vectors and the models' first run on the GPU (35 s after a restart)."""
        from PIL import Image, ImageDraw
        self._dna_vectors()
        for c in CATEGORIES:
            self._category_text(c)
        im = Image.new("RGB", (512, 512), "white")
        ImageDraw.Draw(im).ellipse((140, 140, 372, 372), outline=(212, 175, 55), width=24)   # a plain gold band
        self.read_photo(im)

    def read_photo(self, im) -> "PhotoQuery":
        """A shopper's photo -> its vector, its similarity to every design and
        its design DNA. Done once per upload; the search page then refines it
        with words and filters (search_photo) without sending the photo again."""
        v = photo.views(im)
        crop, box = self._find_piece(im, v)       # the piece, cropped like the catalogue (else the whole photo)
        with self._photo_lock:
            e = self.emb.images([crop, ImageOps.mirror(crop)])   # and its mirror image: either side may face the camera
        q = e.mean(0)
        q /= np.linalg.norm(q)
        sim = self._photo_sim(q)
        dsim = dq = None
        if self.dino is not None:
            with self._photo_lock:
                dq = self.dino.images([crop, ImageOps.mirror(crop)]).mean(0)
            dq = dq / np.linalg.norm(dq)
            dsim = self._dino_sim(dq)
        text = self._dna_vectors()
        # type: zero-shot and the nearest designs' own types, together
        p_type = (TYPE_ZS * dna.softmax(q @ text["type"].T * TEMPERATURE)
                  + (1 - TYPE_ZS) * dna.knn_vote(sim, self.cat, CATEGORIES, TYPE_K))
        order = np.argsort(-p_type)
        cat = CATEGORIES[int(order[0])]
        sure = bool(p_type[order[0]] >= TYPE_SURE)
        jewel = float(dna.softmax(q @ text["jewellery"].T * TEMPERATURE)[0])
        mp = dna.softmax(q @ text["metal"].T * TEMPERATURE)
        best = int(np.argmax(mp))
        metal = {"metal": METALS[best] if mp[best] >= dna.METAL_MIN else None, "p": round(float(mp[best]), 3)}
        attrs, traits = self._photo_traits(q, sim, cat)
        motifs = self._photo_motifs(q, cat)
        dia, agree = self._photo_diamonds(q, sim, cat, attrs)
        if dia["layout"]:   # the layout reading (checked against CAD data) replaces the zero-shot "stones" trait
            traits = [t for t in traits if t["group"] != "Stones"]
        dia["settings"] = [m for m in motifs if m["key"] in dna.DIAMOND_MOTIFS]
        motifs = [m for m in motifs if m["key"] not in dna.DIAMOND_MOTIFS]
        reading = agree_details = None
        if self.detail_reader is not None:
            reading = self.detail_reader(crop)
            if self.details is not None:
                agree_details = details_mod.agreement(reading, self.details, self.detail_keys)
        fused = None
        if dsim is not None:
            in_type = (self.cat == cat) if sure else np.ones(len(self.meta), dtype=bool)
            fused = self._standardise(sim, in_type) + DINO_WEIGHT * self._standardise(dsim, in_type)
            same = self._same_by_lead(fused, in_type)
        else:
            same = self._same_design(sim)
        return PhotoQuery(vec=q, sim=sim, category=cat if sure else None, metal=metal["metal"], attrs=attrs,
                          diamond_agree=agree, dino=dsim, fused=fused, dvec=dq,
                          details=reading, detail_agree=agree_details, dna={
            "details": details_mod.shown(reading) if reading else [],
            "type": {"value": cat, "p": round(float(p_type[order[0]]), 3), "sure": sure,
                     "others": [{"value": CATEGORIES[int(i)], "p": round(float(p_type[i]), 3)}
                                for i in order[1:3] if p_type[i] >= 0.05]},
            "metal": metal,
            "traits": traits,
            "motifs": motifs,
            "diamonds": dia,
            "jewellery": jewel >= JEWEL_MIN,
            "box": box,
            "same": self.card(same, metal["metal"]) if same is not None else None,
        })

    def picture_readings(self, ims) -> list[dict]:
        """For pictures found on a web page (linksearch.py): how much each looks like
        jewellery, and how close it comes to the collection's nearest design."""
        crops = []
        for im in ims:
            v = photo.views(im)
            crops.append(v.get("piece", v["full"]))
        with self._photo_lock:
            vecs = self.emb.images(crops)
        text = self._dna_vectors()
        return [{"jewellery": float(dna.softmax(q @ text["jewellery"].T * TEMPERATURE)[0]),
                 "closest": float(self._photo_sim(q).max())} for q in vecs]

    def search_photo(self, pq: "PhotoQuery", prompt: str = "", category: str | None = None,
                     metal: str | None = None, k: int = TOP_K, page: int = 0) -> dict:
        """Designs like the photo, filtered and refined by optional words.

        Type: a type the shopper picked or wrote wins; otherwise the photo's
        type is a filter when the reading is sure, and no type filter (with a
        note) when it isn't. Metal: words or a pick filter as in a text search;
        the photo's own colour only chooses which metal the cards show,
        because every design can be made in any colour."""
        notes = []
        if prompt and prompt.strip():
            v = self.domain.check(prompt)
            if v.ok:
                prompt = v.text
                if v.budget:
                    notes.append(BUDGET_NOTE.format(v.budget))
            else:
                notes.append(f"No designs match “{' '.join(prompt.split())}”, so those words were left out "
                             "and the photo alone is used.")
                prompt = ""
        q = parse(prompt or "")
        if not pq.dna["jewellery"]:
            notes.append("This photo doesn't look like jewellery, so the matches may not be close. "
                         "A clear photo of one piece works best.")
        if category:
            q.category = None if category == "any" else category
            q.categories = [q.category] if q.category else []
            type_from = "you"
        elif q.categories:
            type_from = "words"
            if pq.category and pq.category not in q.categories:
                notes.append(f"Your photo looks like {_a(pq.category)}; showing "
                             f"{' and '.join(PLURAL[c] for c in q.categories)} because you asked for them.")
        elif pq.category:
            q.category, q.categories, type_from = pq.category, [pq.category], "photo"
        else:
            type_from = None
            t = pq.dna["type"]
            guess = " or ".join([PLURAL[t["value"]]] + [PLURAL[o["value"]] for o in t["others"][:1]])
            notes.append(f"The type of piece isn't clear from the photo (perhaps {guess}), "
                         "so every type is shown. Pick one above to narrow it.")
        if metal:
            q.metal = None if metal == "any" else metal
            q.metal_note = None
        mask, more = self._filters(q)
        notes += more
        score = self._standardise(pq.sim, mask) + self._intent_score(q, mask)
        if pq.dino is not None and DINO_WEIGHT:
            score = score + DINO_WEIGHT * self._standardise(pq.dino, mask)
        if q.phrases or q.free_text():
            tv = self._query_vector(q)
            tsim = 0.5 * (self.vecs @ tv) + 0.5 * (self.front @ tv) - HUB_WEIGHT * self.hub
            score = score + PHOTO_TEXT_WEIGHT * self._standardise(tsim, mask)
        if PHOTO_DNA_WEIGHT:
            score = score + PHOTO_DNA_WEIGHT * self._standardise(self._dna_agreement(pq.attrs), mask)
        if PHOTO_DIAMOND_WEIGHT and pq.diamond_agree is not None:
            score = score + PHOTO_DIAMOND_WEIGHT * self._standardise(pq.diamond_agree, mask)
        if DETAILS_WEIGHT and pq.detail_agree is not None:
            score = score + DETAILS_WEIGHT * self._standardise(pq.detail_agree, mask)
        picked = self._mmr(score, mask, k * (page + 1), PHOTO_MMR)[k * page:]
        show = q.metal or pq.metal
        same = pq.dna["same"]["uid"] if pq.dna["same"] else None
        results = []
        first_label = self._first_label(pq, mask, picked[0]) if page == 0 and picked and pq.fused is not None else "similar"
        for n, i in enumerate(picked):
            c = self.card(i, show)
            if i == same:
                c["match"] = "same"
            elif pq.fused is None:   # SigLIP2 alone (no DINOv2 index): its similarity, as before
                c["match"] = _tier(float(pq.sim[i]))
            else:
                c["match"] = first_label if (page == 0 and n == 0) else "similar"
            c["shares"] = self._shared_details(pq, i)
            results.append(c)
        if page == 0 and pq.fused is not None and results and all(c["match"] == "similar" for c in results):
            shared = [x.lower() for x in results[0]["shares"]]
            notes.append("No design here is a clear match for this one; these are the closest in style"
                         + (f", sharing its {' and '.join(shared)}." if shared else "."))
        return {
            "query": self._explain(q),
            "type_from": type_from,
            "dna": pq.dna,
            "notes": notes,
            "page": page,
            "has_more": int(mask.sum()) > k * (page + 1),
            "matches_in_filter": int(mask.sum()),
            "results": results,
        }

    # ---- internals ---------------------------------------------------------
    def _merge_duplicates(self):
        """The same render filed in two folders becomes two designs; put such
        near-identical pairs in one family so they never both appear."""
        v = self.front / np.linalg.norm(self.front, axis=1, keepdims=True)
        for i in range(0, len(v), 1000):
            sims = v[i:i + 1000] @ v.T
            for a, b in zip(*np.nonzero(sims > DUPLICATE_COS)):
                a += i
                fa, fb = self.meta[a]["family"], self.meta[b]["family"]
                if a < b and fa != fb:
                    for j in self.families.pop(fb):
                        self.meta[j]["family"] = fa
                        self.families[fa].append(j)

    def _hub_bias(self) -> np.ndarray:
        """How strongly each design matches generic queries of its own type.

        Some designs sit close to almost any text ("hubs") and would appear
        for every prompt. Subtracting their mean similarity to their top
        HUB_TOP_K matches in a bank of generic queries (CSLS-style hubness
        correction) makes a design win only when it fits the prompt itself.
        The bank vectors are cached next to the index.
        """
        bank_texts = {c: bank_queries(c) for c in CATEGORIES}
        cache = INDEX / "hub_bank.npz"
        key = hashlib.sha1(json.dumps(bank_texts, sort_keys=True).encode()).hexdigest()
        if cache.exists() and np.load(cache)["key"].item() == key:
            data = np.load(cache)
            bank = {c: data[c] for c in CATEGORIES}
        else:
            bank = {c: self.emb.texts(t) for c, t in bank_texts.items()}
            np.savez(cache, key=np.array(key), **bank)
        dv = 0.5 * self.vecs + 0.5 * self.front
        hub = np.zeros(len(self.meta), dtype=np.float32)
        for c in CATEGORIES:
            idx = np.flatnonzero(self.cat == c)
            if len(idx):
                sims = dv[idx] @ bank[c].T
                hub[idx] = np.sort(sims, axis=1)[:, -HUB_TOP_K:].mean(1)
        return hub

    def _design_cut(self, uid: int) -> str | None:
        """A design's centre cut as read from its own renders, when the reading is clear."""
        if not self.dia_model:
            return None
        lp = self.dia_layout[uid]
        if lp[0] + lp[1] < dna.CENTRE_MIN:   # solitaire + centre_side
            return None
        i = int(np.argmax(self.dia_cut[uid]))
        return dna.SHAPES[i][0] if self.dia_cut[uid][i] >= dna.CARD_CUT_MIN else None

    # ---- photo search internals ---------------------------------------------
    def _load_views(self):
        """Per-view vectors (scripts/build_view_index.py), only if they were
        made from this very index: a stale file would put views on the wrong designs."""
        self.views = None
        f, j = INDEX / "views.npy", INDEX / "views.json"
        if not (f.exists() and j.exists()):
            return
        info = json.loads(j.read_text())
        if info.get("key") != index_key(self.meta):
            print("views.npy is from an older index: run scripts/build_view_index.py again", flush=True)
            return
        design = np.asarray(info["design"])
        if len(design) == 0 or np.any(np.diff(design) < 0):
            return
        self.views = np.load(f)
        self._view_starts = np.r_[0, np.flatnonzero(np.diff(design)) + 1]
        self._view_designs = design[self._view_starts]

    def _load_dino(self):
        """DINOv2 vectors of every view (scripts/build_dino_index.py), only if made from
        this very index with this model and size; else photo search uses SigLIP2 alone."""
        from .dino import MODEL as DINO_MODEL, SIZE as DINO_SIZE
        self.dino, self.dino_views = None, None
        f, j = INDEX / "dino_views.npy", INDEX / "dino_views.json"
        if self.views is None or not (f.exists() and j.exists()):
            return
        info = json.loads(j.read_text())
        if (info.get("key") != index_key(self.meta) or info.get("model") != DINO_MODEL or info.get("size") != DINO_SIZE
                or info.get("rows") != len(self.views)):
            print("dino_views.npy is from an older index or model: run scripts/build_dino_index.py again", flush=True)
            return
        from .dino import Dino
        self.dino_views = np.load(f)
        self.dino = Dino()

    def _load_piece_finder(self):
        """The patch reader that finds the piece in a busy photo (scripts/train_piece_finder.py),
        if it was made for this DINOv2 model and size."""
        from .dino import MODEL as DINO_MODEL, SIZE as DINO_SIZE
        self.piece_finder = None
        f = INDEX / "piece_finder.npz"
        if self.dino is None or not f.exists():
            return
        d = np.load(f)
        if str(d["model"]) != DINO_MODEL or int(d["size"]) != DINO_SIZE:
            print("piece_finder.npz is for another DINOv2 model: run scripts/train_piece_finder.py again", flush=True)
            return
        self.piece_finder = {"W": d["W"].astype(np.float32), "b": float(d["b"]), "scale": float(d["scale"])}

    def piece_heat(self, im) -> tuple[np.ndarray, tuple] | None:
        """How likely each patch of the (squared) photo is jewellery -> (GRID x GRID map, the square's box)."""
        if not PIECE_FINDER or self.piece_finder is None:
            return None
        from .dino import GRID
        sq, sbox, _ = photo.square_full(im)
        with self._photo_lock:
            f = self.dino.patches([sq])[0]
        pf = self.piece_finder
        heat = 1 / (1 + np.exp(-((f * pf["scale"]) @ pf["W"] + pf["b"])))
        return heat.reshape(GRID, GRID), sbox

    def _find_piece(self, im, v: dict):
        """The crop of the photo to compare with the catalogue, and its box (fractions of
        the photo, for the page to outline). The background cut-out when it found the
        piece; else the crop around the patches the piece finder reads as jewellery;
        else the whole photo."""
        own = (v.get("piece", v["full"]), v["box"])
        found = self.piece_heat(im)
        if found is None:
            return own
        from .dino import GRID
        heat, sbox = found
        cells = photo.heat_box(heat, PIECE_MIN_P, PIECE_KEEP)
        if cells is None:
            return own
        box = photo.grid_to_photo(cells, GRID, sbox)
        frac = [round(max(0, box[0]) / im.width, 4), round(max(0, box[1]) / im.height, 4),
                round(min(im.width, box[2]) / im.width, 4), round(min(im.height, box[3]) / im.height, 4)]
        if v["box"] is not None:
            if _overlap(v["box"], frac) >= 0.3:
                return own   # both found the same piece: the cut-out is tighter
            if photo.heat_on_mask(heat, sbox, im.size, v["mask"]) >= CUT_MIN_HEAT:
                return own   # the cut-out is jewellery too: a whole necklace whose thin chain the finder misses
            # else the cut-out found something else: the person wearing the ring
        elif (frac[2] - frac[0]) * (frac[3] - frac[1]) >= 0.8:
            return own   # the piece fills the photo
        return photo.square_crop(im, box, v.get("fill", (255, 255, 255))), frac

    def _load_details(self, details):
        """The detail reader (details.py) and every design's details (scripts/build_details_index.py).
        details="auto": load the reader (~4.5 GB) unless JEWEL_DETAILS=0; None: no reader; or a callable."""
        import os
        self.detail_reader, self.details, self.detail_keys = None, None, []
        if details == "auto":
            if os.environ.get("JEWEL_DETAILS", "1") != "0":
                try:
                    self.detail_reader = details_mod.DetailReader().read
                except Exception as e:   # photo search works without it
                    print(f"details: reader not available ({e})", flush=True)
        elif details is not None:
            self.detail_reader = details
        f, j = INDEX / "details.npy", INDEX / "details.json"
        if self.detail_reader and f.exists() and j.exists():
            info = json.loads(j.read_text())
            if info.get("key") == index_key(self.meta) and info.get("model") == details_mod.MODEL:
                d = np.load(f)
                self.detail_keys = info["keys"]
                self.details = np.where(np.isnan(d), 0.5, d)   # not read yet: neither shares nor lacks a detail
            else:
                print("details.npy is from an older index or model: run scripts/build_details_index.py again", flush=True)

    def _shared_details(self, pq, uid: int) -> list[str]:
        """The photo's clear details this design has too (labels, for its card)."""
        if not pq.details or self.details is None:
            return []
        out = []
        for j, k in enumerate(self.detail_keys):
            if k not in details_mod.SHOWN:
                continue
            if pq.details.get(k, 0) >= details_mod.YES and self.details[uid, j] >= details_mod.YES:
                out.append(details_mod.QUESTIONS[k][1])
        return out

    def _dino_sim(self, qv: np.ndarray) -> np.ndarray:
        """DINOv2 similarity of a photo to every design: its closest view (designs without
        views get the median, so they neither win nor vanish)."""
        best = np.maximum.reduceat(self.dino_views @ qv, self._view_starts)
        out = np.full(len(self.meta), float(np.median(best)))
        out[self._view_designs] = best
        return out

    def _first_label(self, pq, mask: np.ndarray, uid: int) -> str:
        """The first card's match label: how far it leads the next design family on the two
        picture models together (within the filters), when both models rate it well."""
        zs, zd = self._standardise(pq.sim, mask), self._standardise(pq.dino, mask)
        if min(zs[uid], zd[uid]) < LABEL_BOTH_MIN:
            return "similar"
        fused = zs + zd
        fam = self.meta[uid]["family"]
        order = np.argsort(-np.where(mask, fused, -np.inf))[:200]
        second = next((fused[j] for j in order if self.meta[j]["family"] != fam), None)
        if second is None:
            return "similar"
        return _tier_lead(float(fused[uid] - second))

    def _same_by_lead(self, fused: np.ndarray, mask: np.ndarray):
        """The photo shows a design of the collection when that design leads the next
        design family by SAME_LEAD on the combined score."""
        order = np.argsort(-np.where(mask, fused, -np.inf))[:200]
        first = int(order[0])
        fam = self.meta[first]["family"]
        second = next((fused[j] for j in order if self.meta[j]["family"] != fam), None)
        return first if second is not None and fused[first] - second >= SAME_LEAD else None

    def _photo_sim(self, qv: np.ndarray) -> np.ndarray:
        """Similarity of a photo vector to every design: the design as a whole
        (mean of views + front view), mixed with its single closest view, since
        a photo may show any angle."""
        base = 0.5 * (self.vecs @ qv) + 0.5 * (self.front @ qv)
        if self.views is None or not VIEW_MIX:
            return base
        best = np.maximum.reduceat(self.views @ qv, self._view_starts)
        out = base.copy()
        out[self._view_designs] = VIEW_MIX * best + (1 - VIEW_MIX) * base[self._view_designs]
        return out

    def _dna_vectors(self) -> dict:
        """Text vectors of the type and jewellery prompts (made once)."""
        if "type" not in self._dna_text:
            def mean(prompts):
                v = self.emb.texts(prompts).mean(0)
                return v / np.linalg.norm(v)
            with self._photo_lock:
                self._dna_text["jewellery"] = np.stack([mean(dna.JEWELLERY_PROMPTS["yes"]),
                                                        mean(dna.JEWELLERY_PROMPTS["no"])])
                self._dna_text["type"] = np.stack([mean(dna.CATEGORY_PROMPTS[c]) for c in CATEGORIES])
                self._dna_text["metal"] = np.stack([mean([dna.METAL_PROMPT.format(metal=dna.METAL_WORDS[m])])
                                                    for m in METALS])
        return self._dna_text

    def _category_text(self, cat: str) -> dict:
        """Per type: trait class vectors, motif and cut vectors, and how every
        design of the type scores on the motifs and cuts (the calibration)."""
        key = "cat:" + cat
        if key not in self._dna_text:
            noun = dna.NOUN.get(cat, "piece of jewellery")
            with self._photo_lock:
                classes = {}
                for attr, cls in dna.class_prompts(cat).items():
                    vs = {}
                    for c, prompts in cls.items():
                        v = self.emb.texts(prompts).mean(0)
                        vs[c] = v / np.linalg.norm(v)
                    classes[attr] = vs
                motif = self.emb.texts([f"a photo of a {noun}, {m[2]}" for m in dna.MOTIFS])
                shape = self.emb.texts([f"a photo of a {noun}, {s[1]}" for s in dna.SHAPES])
                layout = np.stack([self.emb.texts([t.format(n=noun) for t in prompts]).mean(0)
                                   for prompts in dna.LAYOUT_PROMPTS.values()])
            idx = np.flatnonzero(self.cat == cat)
            dv = 0.5 * self.vecs[idx] + 0.5 * self.front[idx]
            self._dna_text[key] = {"classes": classes, "motif": motif, "shape": shape, "layout": layout,
                                   "motif_pop": dna.centred(dv @ motif.T)}
        return self._dna_text[key]

    def _photo_traits(self, q: np.ndarray, sim: np.ndarray, cat: str):
        """Stones / band / weight / form of the photo: zero-shot on the photo,
        averaged with its nearest designs' indexed values."""
        ct = self._category_text(cat)
        zs = dna.zero_shot(q, ct["classes"])
        keys = [(a, c) for a, cls in ct["classes"].items() for c in cls]
        nb = dna.neighbour_attrs(sim, {kk: self.attr[kk] for kk in keys if kk in self.attr}, self.cat == cat, TRAIT_K)
        attrs, traits = {}, []
        for a, cls in zs.items():
            p = {c: TRAIT_ZS * zs[a][c] + (1 - TRAIT_ZS) * nb.get((a, c), zs[a][c]) for c in cls}
            total = sum(p.values())
            p = {c: v / total for c, v in p.items()}
            attrs.update({(a, c): v for c, v in p.items()})
            best = max(p, key=p.get)
            if p[best] >= dna.TRAIT_MIN and (a, best) in dna.TRAITS:
                group, label, word = dna.TRAITS[(a, best)]
                traits.append({"key": f"{a}:{best}", "group": group, "label": label, "p": round(float(p[best]), 3),
                               "word": word})
        return attrs, traits

    def _photo_motifs(self, q: np.ndarray, cat: str) -> list:
        """Motifs where the photo outscores nearly every design of its type."""
        ct = self._category_text(cat)
        mine = dna.centred(q @ ct["motif"].T)
        pct = [dna.percentile(mine[i], ct["motif_pop"][:, i]) for i in range(len(dna.MOTIFS))]
        motifs = sorted(({"key": m[0], "label": m[1], "word": m[3], "pct": round(p, 3)}
                         for m, p in zip(dna.MOTIFS, pct)
                         if p >= dna.MOTIF_PCT_OVERRIDE.get(m[0], dna.MOTIF_PCT) and (m[4] is None or cat in m[4])),
                        key=lambda x: -x["pct"])
        return motifs[:dna.MOTIF_MAX]

    def _photo_diamonds(self, q: np.ndarray, sim: np.ndarray, cat: str, attrs: dict):
        """The photo's diamonds, read from its pixels: layout (solitaire / centre +
        smaller / even-sized) and the centre stone's cut. -> (reading for the page,
        how well each design's diamonds, read from its own renders, agree with it)."""
        out = {"plain": bool(attrs.get(("stones", "plain"), 0) >= 0.5), "layout": None, "centre_cut": None,
               "how": "trained" if self.dia_model else "zero-shot"}
        if out["plain"]:
            return out, None
        names, cuts = list(dna.LAYOUT_PROMPTS), [c for c, _ in dna.SHAPES]
        if self.dia_model:
            lp, cp = self._diamond_probs("layout", q[None])[0], self._diamond_probs("cut", q[None])[0]
        else:
            ct = self._category_text(cat)
            lp = dna.softmax(q @ ct["layout"].T * TEMPERATURE)
            cp = dna.softmax(q @ ct["shape"].T * TEMPERATURE)
        li, ci = int(np.argmax(lp)), int(np.argmax(cp))
        centre = float(lp[names.index("solitaire")] + lp[names.index("centre_side")])
        if lp[li] >= dna.LAYOUT_MIN:
            out["layout"] = {"value": names[li], "label": dna.LAYOUT_LABELS[names[li]], "p": round(float(lp[li]), 3),
                             "word": dna.LAYOUT_WORDS[names[li]]}
        if centre >= dna.CENTRE_MIN and cp[ci] >= dna.CUT_MIN:
            cut = cuts[ci]
            out["centre_cut"] = {"value": cut, "label": "Heart shape" if cut == "heart" else f"{cut.title()} cut",
                                 "p": round(float(cp[ci]), 3), "word": "emerald cut" if cut == "emerald" else cut}
        if not self.dia_model:
            return out, None
        # agreement with each design's own diamonds (both read from pictures): overlap of the
        # layout probabilities, plus the cut's overlap weighted by how likely the photo has a centre stone
        agree = np.sqrt(self.dia_layout) @ np.sqrt(lp) + centre * (np.sqrt(self.dia_cut) @ np.sqrt(cp))
        return out, agree

    def _load_diamond_reader(self):
        """The trained diamond readers (scripts/train_diamond_dna.py), only if made
        from this very index, and every design's diamonds read from its renders."""
        self.dia_model = None
        f = INDEX / "diamond_dna.npz"
        if f.exists():
            d = np.load(f)
            if d["key"].item() == index_key(self.meta) and self.views is not None:
                self.dia_model = {k: d[k] for k in ("cut_W", "cut_b", "layout_W", "layout_b")}
                self.dia_model["scale"] = float(d["scale"])
            else:
                print("diamond_dna.npz is from an older index: run scripts/train_diamond_dna.py again", flush=True)
        if not self.dia_model:
            return
        counts = np.diff(np.r_[self._view_starts, len(self.views)])
        for head in ("layout", "cut"):
            p = self._diamond_probs(head, self.front)                 # designs without views: their front view
            pv = np.add.reduceat(self._diamond_probs(head, self.views), self._view_starts) / counts[:, None]
            p[self._view_designs] = pv                                # the mean over all of a design's views
            setattr(self, "dia_" + head, p)

    def _diamond_probs(self, head: str, X: np.ndarray) -> np.ndarray:
        m = self.dia_model
        return dna.softmax((X * m["scale"]) @ m[head + "_W"].T + m[head + "_b"])

    def _same_design(self, sim: np.ndarray):
        """The photo shows a design of the collection when its best match is
        both very close and clearly ahead of every other design family."""
        order = np.argsort(-sim)[:100]
        first = int(order[0])
        fam = self.meta[first]["family"]
        second = next((sim[j] for j in order if self.meta[j]["family"] != fam), -1.0)
        if sim[first] >= SAME_SIM and sim[first] - second >= SAME_GAP:
            return first
        return None

    def _dna_agreement(self, attrs: dict) -> np.ndarray:
        """How well each design's stones / band / weight / form match the
        photo's: Bhattacharyya overlap of the class probabilities per group."""
        total = np.zeros(len(self.meta))
        groups = {}
        for (a, c), p in attrs.items():
            groups.setdefault(a, []).append((c, p))
        for a, cls in groups.items():
            part = np.zeros(len(self.meta))
            seen = np.zeros(len(self.meta), bool)
            for c, p in cls:
                d = self.attr.get((a, c))
                if d is None:
                    continue
                ok = ~np.isnan(d)
                part[ok] += np.sqrt(p * d[ok])
                seen |= ok
            part[~seen] = np.median(part[seen]) if seen.any() else 0
            total += part
        return total

    @staticmethod
    def _front_path(m: dict, metal: str) -> str:
        views = m["images"][metal]
        v = m.get("front_view")
        return views[v] if v in views else views[sorted(views)[-1]]

    def _front_url(self, m: dict, metal: str) -> str:
        """The design's front render as it is in the dataset (catalogue pictures are never
        cropped, resized or re-encoded; data/crops is only what the image models read)."""
        return media_url(self._front_path(m, metal))

    def _query_vector(self, q: ParsedQuery) -> np.ndarray:
        texts, weights = [q.visual_text()], [1.0]
        free = q.free_text()
        if free:
            texts.append(f"a photo of jewellery, {free}")
            # when the lexicon understood little, lean on the free text
            weights.append(0.5 if (q.category or q.intents) else 3.0)
        neg = q.negative_text()
        if neg:
            texts.append(neg)
            weights.append(-NEG_TEXT_WEIGHT)
        if q.phrases or free:
            # the type is already a hard filter, so the generic "a photo of a
            # ring" part only adds bias; remove it to rank on what is specific
            texts.append(q.base_text())
            weights.append(-BASE_WEIGHT)
        t = self.emb.texts(texts)
        v = (np.array(weights)[:, None] * t).sum(0)
        return v / np.linalg.norm(v)

    def count(self, q: ParsedQuery) -> tuple[int, bool]:
        """How many designs a parsed prompt's filters keep, and whether any
        filter had to be relaxed to get there (used by search suggestions)."""
        relaxed = []
        mask, _ = self._filters(q, relaxed)
        return int(mask.sum()), bool(relaxed)

    def strict_mask(self, q: ParsedQuery) -> np.ndarray:
        """The designs that pass every filter of a prompt, none loosened even when
        few pass (the numbers next to manual search options)."""
        return self._filters(q, strict=True)[0]

    def _filters(self, q: ParsedQuery, relaxed: list | None = None, strict: bool = False):
        """-> (mask, notes). Every filter that is loosened or dropped because
        too few designs pass it is named in `relaxed` when a list is given;
        strict: none is loosened, however few designs pass."""
        relaxed = [] if relaxed is None else relaxed
        notes = [q.metal_note] if q.metal_note else []
        mask = np.ones(len(self.meta), dtype=bool)
        if q.categories:
            mask &= np.isin(self.cat, q.categories)
        for c in q.exclude_categories:
            mask &= self.cat != c
        for mt in q.exclude_metals:
            # keep designs that also come in some other metal
            mask &= np.any([self.metal_mask[o] for o in METALS if o != mt], axis=0)
        if q.metal:
            with_metal = mask & self.metal_mask[q.metal]
            if with_metal.sum() >= TOP_K or strict:
                mask = with_metal
            else:
                notes.append(f"Fewer than {TOP_K} matching designs exist in {q.metal.replace('_', ' ')}, "
                             "so other metals are included.")
                relaxed.append("metal")
        # "for men" is strict too: only the men's series (never women's designs as
        # filler). The collection has men's rings and bracelets only.
        if "men" in q.intents:
            mask &= self.mens
            if not mask.any():
                what = PLURAL.get(q.categories[0], "designs") if len(q.categories) == 1 else "designs for this search"
                notes.append(f"The collection has no men's {what}. Men's designs exist only as rings and bracelets.")
        elif "women" in q.intents:
            mask &= ~self.mens
        # "no stones" is strict: only designs with no stones at all, even if that
        # is fewer than a full page. Both signals must agree: the measured stone
        # share (sees small accent stones) and the image model's "plain metal".
        if "plain" in q.intents and mask.any():
            s = np.where(np.isnan(self.stone), np.inf, self.stone)
            plain = np.nan_to_num(self.attr.get(("stones", "plain"), np.ones(len(s))), nan=0.0)
            mask = mask & (s <= PLAIN_MAX_STONE) & (plain >= PLAIN_MIN_PROB)
            n = int(mask.sum())
            if n == 0:
                notes.append("No designs without stones match this search. Try removing \"no stones\" or another filter.")
            elif n < TOP_K:
                notes.append(f"Only {n} design{'' if n == 1 else 's'} here {'has' if n == 1 else 'have'} no stones at all, "
                             f"so {'it is' if n == 1 else 'they are'} all shown.")
        notes += [MISSING_FORMS[i] for i in q.intents if i in MISSING_FORMS]
        # strict attributes: drop designs that clearly lack / clearly have them.
        # A filter loosened for lack of designs always says so (never silently),
        # in the shopper's own word: the typed style, or the type word that implies it ("kada")
        def word(intent):
            for t, k, v, neg in q.terms:
                if not neg and (k == "intent" and v == intent or k == "category" and intent in parse(t).intents):
                    return t
            return STYLE_WORD.get(intent, intent.replace("_", " "))
        for intent in q.intents:
            if intent in LAYOUT_INTENTS and self.dia_model:
                p, least = self.dia_layout[:, LAYOUT_INTENTS[intent]], LAYOUT_MIN_PROB
            elif intent in STRICT and self.attr.get(STRICT[intent]) is not None:
                p, least = np.nan_to_num(self.attr[STRICT[intent]], nan=1.0), STRICT_MIN_PROB
            else:
                continue
            if mask.any():
                keep = mask & (p >= least)
                if strict:
                    mask = keep
                    continue
                if keep.sum() < STRICT_MIN_LEFT:
                    # rare attribute (few truly plain rings exist): keep the
                    # designs that have the most of it instead of none
                    keep = mask & (p >= np.quantile(p[mask], 1 - STRICT_FALLBACK_SHARE))
                    relaxed.append(intent)
                if keep.sum() >= STRICT_MIN_LEFT:
                    if intent in relaxed:
                        notes.append(f"Few designs here are clearly “{word(intent)}”, so the ones that look most like it "
                                     "are shown.")
                    mask = keep
                else:
                    if intent not in relaxed:
                        relaxed.append(intent)
                    notes.append(f"Too few designs here are clearly “{word(intent)}” to keep only those, so it orders "
                                 "the results instead.")
        for intent in q.not_intents:
            parts = [self.attr[(a, c)] for a, c, w in INTENT_ATTRS.get(intent, []) if w > 0 and (a, c) in self.attr]
            if not parts or not mask.any():
                continue
            keep = mask & (np.nansum(parts, axis=0) <= UNWANTED_MAX_PROB)
            if keep.sum() >= STRICT_MIN_LEFT or strict:
                mask = keep
            else:
                relaxed.append("not " + intent)
                unwanted = next((t for t, k, v, neg in q.terms if neg and k == "intent" and v == intent),
                                STYLE_WORD.get(intent, intent.replace("_", " ")))
                notes.append(f"Too few designs here are clearly without “{unwanted}” to keep only those, so designs "
                             "with it are ranked lower.")
        return mask, notes

    def _intent_score(self, q: ParsedQuery, mask: np.ndarray) -> np.ndarray:
        """What the words ask for beyond the picture: attributes, stones, cut."""
        score = self._attr_score(q)
        if "plain" in q.intents or "diamond" in q.intents:
            stones = self._standardise(np.nan_to_num(self.stone, nan=np.nanmedian(self.stone)), mask)
            score = score + STONE_WEIGHT * (-stones if "plain" in q.intents else 0.5 * stones)
        if q.shape:
            # (adding the cut read from each design's renders was tried: it made "oval ring" less exact)
            score = score + SHAPE_BOOST * (self.shape == q.shape)
        return score

    def _attr_score(self, q: ParsedQuery) -> np.ndarray:
        total = np.zeros(len(self.meta))
        for sign, intents in ((1, q.intents), (-1, q.not_intents)):
            for intent in intents:
                for attr, cls, w in INTENT_ATTRS.get(intent, []):
                    p = self.attr.get((attr, cls))
                    if p is not None:
                        total += sign * w * ATTR_WEIGHT * np.nan_to_num(p, nan=0.0)
        return total

    @staticmethod
    def _standardise(sim: np.ndarray, mask: np.ndarray) -> np.ndarray:
        s = sim[mask] if mask.any() else sim
        return (sim - s.mean()) / (s.std() + 1e-9)

    def _mmr(self, scores: np.ndarray, mask: np.ndarray, k: int, lam: float) -> list[int]:
        idx = np.flatnonzero(mask)
        if not len(idx):
            return []
        pool = idx[np.argsort(-scores[idx])[:max(POOL, 4 * k)]]
        rel = scores[pool]
        rel = (rel - rel.min()) / (rel.max() - rel.min() + 1e-9)
        sim = self.vecs[pool] @ self.vecs[pool].T
        chosen, families = [], set()
        while len(chosen) < k:
            best, best_val = None, -np.inf
            for a in range(len(pool)):
                if a in chosen or self.meta[pool[a]]["family"] in families:
                    continue
                red = max((sim[a, c] for c in chosen), default=0.0)
                val = lam * rel[a] - (1 - lam) * red
                if val > best_val:
                    best, best_val = a, val
            if best is None:
                break
            chosen.append(best)
            families.add(self.meta[pool[best]]["family"])
        return [int(pool[a]) for a in chosen]

    @staticmethod
    def _explain(q: ParsedQuery) -> dict:
        return {"raw": q.raw, "corrections": q.corrections, "category": q.category, "categories": q.categories,
                "exclude_categories": q.exclude_categories, "metal": q.metal,
                "exclude_metals": q.exclude_metals, "shape": q.shape, "intents": q.intents,
                "not_intents": q.not_intents, "free_text": q.free_text(), "model_text": q.visual_text(),
                "negative_text": q.negative_text()}
