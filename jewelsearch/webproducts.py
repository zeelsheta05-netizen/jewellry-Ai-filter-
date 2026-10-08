"""From the web (DEMO): jewellery products of other shops, shown in a side panel
next to the search results, like Google's product view. Each card shows the
shop's own picture, name and price, and opens the shop's page when clicked.

Nothing here looks for copies of a design. The panel shows what looks most
like what the shopper asked for:
  - for words: the same query vector the collection search uses (SigLIP2),
    against each product's picture, plus a little of its name;
  - for one of our designs: SigLIP2 and DINOv2 together, like photo search;
  - for a web product: the products that look most like it.
The type the shopper asked for is a hard filter (a ring search shows rings);
a metal colour is too, when the shop says the colour.

The pool is a small file set made by scripts/build_web_products.py from the
public product feeds of a few shops (their robots rules are checked first):

    data/web_products/products.jsonl   one product per line (the card's facts)
    data/web_products/siglip.npy       SigLIP2 vector of each product's first picture
    data/web_products/title.npy        SigLIP2 vector of each product's name
    data/web_products/dino.npy         DINOv2 vector of each product's first picture
    data/web_products/info.json        when it was made, how many per shop, models

Pictures: the page may only load pictures from this app, so they come through
/web-img/<id>/<size>. Each is fetched once from the shop's image server at the
size the frame needs (the shop makes it from its original), kept in
data/web_products/img/, and never stretched more than 2x (Lanczos) when the
shop has nothing larger. The page fits every picture whole inside a square
frame, with a soft blurred copy of itself behind it: nothing is cropped,
squashed or redrawn by an AI upscaler (that would invent design details).
"""
from __future__ import annotations

import hashlib
import io
import json
import re
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import numpy as np

from .config import CATEGORIES, DATA, METALS

DIR = DATA / "web_products"
IMG_DIR = DIR / "img"
SIZES = (320, 640)        # picture sizes the page asks for (card, large preview); 2x screens included
MAX_UPSCALE = 2.0
WEBP_QUALITY = 90
NEW_DAYS = 30             # "New" badge: first published by the shop within this many days
TITLE_WEIGHT = 0.4        # how much a product's name counts next to its picture
DINO_WEIGHT = 1.0         # same weight as photo search
WORD_BOOST = 1.5          # per specific word of the request found in the product's name (at most 2)
PICTURE_COLOUR = 3.0      # a shop may sell a ring in three colours but photograph one: prefer the asked colour
DESIGN_CUT = 1.5          # our design -> web: agreement of the diamond cut reading (measured, see like_design)
DESIGN_LAYOUT = 1.0       # ...and of the diamond layout (solitaire / centre + side / many small)
DESIGN_MOTIF = 1.0        # ...and of the motifs the design clearly has (heart, butterfly, floral, halo, ...)
MOTIF_CLEAR = 0.97        # a design "has" a motif when it outscores this share of the designs of its type
DESIGN_DINO = 0.5         # ...and a little DINOv2: keeps motif shapes (butterflies) at almost no cost
DESIGN_COLOUR = 0.5       # the card's picked colour: a tie-breaker, never above shape
MENS_PENALTY = 1.5        # men's pieces when the request isn't for men (a request for men shows only those)
MENS = re.compile(r"\b(men|mens|men's|man's|gents?|for him|male|boys?)\b", re.I)
SHOP_SPREAD = 0.12        # small push towards a mix of shops on one page
SAME_PICTURE = 0.97       # a colour variant listed as its own product: show one of them
K = 24

# Shops with a public product feed (Shopify /products.json), checked 2026-10-06.
SHOPS = [
    {"key": "giva", "name": "GIVA", "base": "https://www.giva.co"},
    {"key": "palmonas", "name": "Palmonas", "base": "https://palmonas.com"},
    {"key": "isharya", "name": "Isharya", "base": "https://isharya.com"},
    {"key": "limelight", "name": "Limelight Diamonds", "base": "https://www.limelightdiamonds.com"},
    {"key": "kushals", "name": "Kushal's", "base": "https://www.kushals.com"},
    {"key": "salty", "name": "Salty", "base": "https://www.salty.co.in"},
]
SHOP_BY_KEY = {s["key"]: s for s in SHOPS}


# ---- reading a product's words ------------------------------------------------------

SKIP_WORDS = re.compile(
    r"\b(gift ?cards?|gift ?box|boxes|pouch|cleaner|cleaning|polish(?:ing)? cloth|nose ?pins?|nose ?rings?|"
    r"septum|anklets?|payal|toe ?rings?|maang ?tikk?a|mang ?tikk?a|tikka|brooch|hair|watch(?:es)?|"
    r"keychain|key chain|bag charm|sets?|combo|hampers?|kit|jewellery box|organi[sz]er|"
    r"belly|waist|hath ?phool|haathphool|cufflinks?)\b", re.I)
TYPE_WORDS = [   # checked in this order; an earring word wins over "ring", ear cuffs are earrings
    ("earrings", re.compile(r"\b(ear ?rings?|earings?|studs?|hoops?|huggies?|jhumk[ai]s?|jhumkas?|ear ?cuffs?|"
                            r"danglers?|drop earrings?|ear ?climbers?|chandbali)\b", re.I)),
    ("ring", re.compile(r"\b(rings?|bands?|solitaire ring)\b", re.I)),
    ("pendant", re.compile(r"\b(pendants?|lockets?)\b", re.I)),
    ("bracelet", re.compile(r"\b(bracelets?|bangles?|kadas?|(?<!ear )(?<!ear-)cuffs?|charm bracelet)\b", re.I)),
    ("necklace", re.compile(r"\b(necklaces?|chokers?|mangalsutras?|chains?|lariats?|neckpieces?)\b", re.I)),
]
GENERIC = {"ring", "rings", "earring", "earrings", "necklace", "necklaces", "pendant", "pendants", "bracelet",
           "bracelets", "jewellery", "jewelry", "gold", "silver", "white", "yellow", "rose", "with", "without",
           "for", "the", "and", "design", "designs", "simple", "nice", "beautiful", "women", "ladies", "daily",
           "wear", "everyday", "piece", "pieces", "want", "show", "like", "something", "very", "some"}
KT = re.compile(r"\b(9|10|14|18|22|24)\s?(?:k|kt|karat|carat)\b", re.I)


def category_of(title: str, product_type: str, tags: list[str]) -> str | None:
    """One of our types from the shop's words, or None (sets, other types, unclear)."""
    if re.search(r"mangalsutra", title, re.I) and not SKIP_WORDS.search(f"{title} {product_type}"):
        return "necklace"   # the search treats a mangalsutra as a necklace too, whatever the shop files it under
    for text in (f"{title} {product_type}", " ".join(tags)):
        if SKIP_WORDS.search(f"{title} {product_type}"):
            return None
        found = [cat for cat, rx in TYPE_WORDS if rx.search(text)]
        if "earrings" in found:
            found = [c for c in found if c != "ring"]      # "earrings" also contains "rings"
        if "pendant" in found:
            found = [c for c in found if c != "necklace"]  # "pendant necklace", "chain pendant"
        if "necklace" in found and len(found) > 1 and re.search(r"\bchains?\b", text, re.I) \
                and not re.search(r"necklace|choker|mangalsutra", text, re.I):
            found.remove("necklace")                       # "chain bracelet"
        if len(found) == 1:
            return found[0]
        if len(found) > 1:
            return None                                    # several types: a set or a mixed listing
    return None


def colours_of(texts: list[str]) -> list[str]:
    """Metal colours the shop names (rose / white / yellow gold); silver and platinum
    look white. Empty when the shop doesn't say."""
    out = set()
    for t in texts:
        t = (t or "").lower()
        if not t:
            continue
        if "rose" in t:
            out.add("rose_gold")
        if re.search(r"white gold|platinum|rhodium", t):
            out.add("white_gold")
        rest = re.sub(r"rose gold|white gold|rose|platinum|rhodium", " ", t)
        if re.search(r"\b(yellow|gold|golden)\b", rest):
            out.add("yellow_gold")
        elif re.search(r"\b(silver|sterling|925)\b", rest) and "rose" not in t:
            out.add("white_gold")
    return sorted(out)


def material_of(text: str) -> str:
    """A short honest label of what it's made of: "18KT gold · Lab-grown diamond"."""
    t = text.lower()
    parts = []
    m = KT.search(t)
    if m:
        parts.append(f"{m.group(1)}KT gold")
    elif re.search(r"\b(925|sterling silver|pure silver)\b", t):
        parts.append("925 silver" + (" · gold plated" if "plated" in t and "gold" in t else ""))
    elif "plated" in t or "plating" in t:
        parts.append("Gold plated" if "gold" in t else "Plated")
    elif "silver" in t:
        parts.append("Silver")
    elif "platinum" in t:
        parts.append("Platinum")
    if re.search(r"lab[- ]?grown|laboratory grown|lgd", t):
        parts.append("Lab-grown diamond")
    elif "moissanite" in t:
        parts.append("Moissanite")
    elif re.search(r"\b(zircon|zirconia|cz)\b", t):
        parts.append("Zircon")
    elif "diamond" in t:
        parts.append("Diamond")
    elif "pearl" in t:
        parts.append("Pearl")
    return " · ".join(parts)


def product_id(shop: str, shop_product_id) -> str:
    return hashlib.sha1(f"{shop}:{shop_product_id}".encode()).hexdigest()[:12]


def sized_url(src: str, width: int) -> str:
    """The shop's image server makes this width from its original (Shopify: ?width=)."""
    p = urlsplit(src if not src.startswith("//") else "https:" + src)
    q = [(k, v) for k, v in parse_qsl(p.query) if k not in ("width", "height", "crop")]
    q.append(("width", str(width)))
    return urlunsplit((p.scheme or "https", p.netloc, p.path, urlencode(q), ""))


def from_feed(shop: dict, prod: dict, currency: str) -> dict | None:
    """One product of a shop's /products.json -> the facts a card needs (None: skip it)."""
    imgs = [i for i in prod.get("images") or [] if i.get("src")]
    variants = prod.get("variants") or []
    if not imgs or not variants or not any(v.get("available", True) for v in variants):
        return None
    tags = prod.get("tags") or []
    if isinstance(tags, str):
        tags = [t.strip() for t in tags.split(",")]
    title = " ".join((prod.get("title") or "").split())
    ptype = prod.get("product_type") or ""
    cat = category_of(title, ptype, tags)
    if not cat or not prod.get("handle"):
        return None
    options = [str(v) for o in prod.get("options") or [] for v in (o.get("values") or [])
               if re.search(r"metal|colou?r|material|tone|finish", o.get("name") or "", re.I)]
    prices = []
    for v in [v for v in variants if v.get("available", True)]:   # the price a shopper can buy at
        try:
            prices.append(float(v.get("price")))
        except (TypeError, ValueError):
            pass
    first = imgs[0]
    words = " ".join([title, ptype, " ".join(tags), " ".join(options)])
    return {
        "id": product_id(shop["key"], prod.get("id")),
        "shop": shop["key"],
        "title": title[:160],
        "url": f"{shop['base']}/products/{prod['handle']}",
        "category": cat,
        "colours": colours_of([title, *options]),
        "material": material_of(words),
        "price": min(prices) if prices else None,
        "currency": currency,
        "image": first["src"] if not first["src"].startswith("//") else "https:" + first["src"],
        "image_w": first.get("width"),
        "image_h": first.get("height"),
        "published": prod.get("published_at") or prod.get("created_at"),
    }


# ---- the pool, and matching ------------------------------------------------------------

def _z(sim: np.ndarray, mask: np.ndarray) -> np.ndarray:
    s = sim[mask] if mask.any() else sim
    return (sim - s.mean()) / (s.std() + 1e-9)


def _when(s: str | None):
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")) if s else None
    except ValueError:
        return None


class WebProducts:
    def __init__(self, engine, folder: Path = DIR):
        self.engine = engine
        self.folder = folder
        self.items: list[dict] = []
        self.ready = False
        f = folder / "products.jsonl"
        try:
            items = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
            sig, title, dino = (np.load(folder / n) for n in ("siglip.npy", "title.npy", "dino.npy"))
            self.info = json.loads((folder / "info.json").read_text())
        except (OSError, ValueError) as e:
            print(f"web products: not loaded ({e}); run scripts/build_web_products.py", flush=True)
            return
        if not (len(items) == len(sig) == len(title) == len(dino)) or not items:
            print("web products: files don't match each other; run scripts/build_web_products.py again", flush=True)
            return
        self.items, self.sig, self.title, self.dino = items, sig, title, dino
        self.by_id = {p["id"]: i for i, p in enumerate(items)}
        self.cat = np.array([p["category"] for p in items])
        self.colour = {m: np.array([m in p["colours"] for p in items]) for m in METALS}
        self.shop = np.array([p["shop"] for p in items])
        self.mens = np.array([bool(MENS.search(p["title"])) for p in items])
        now = datetime.now(timezone.utc)
        self.new = np.array([bool((w := _when(p.get("published"))) and now - w < timedelta(days=NEW_DAYS))
                             for p in items])
        self.ready = True

    # ---- cards ----
    def card(self, i: int) -> dict:
        p = self.items[i]
        shop = SHOP_BY_KEY.get(p["shop"], {"name": p["shop"], "base": ""})
        return {"id": p["id"], "title": p["title"], "url": p["url"], "shop": shop["name"],
                "site": urlsplit(p["url"]).hostname.removeprefix("www."), "category": p["category"],
                "material": p["material"], "price": p["price"], "currency": p["currency"],
                "new": bool(self.new[i]),
                "img": f"/web-img/{p['id']}/{SIZES[0]}", "img_large": f"/web-img/{p['id']}/{SIZES[1]}"}

    def _pick(self, score: np.ndarray, mask: np.ndarray, k: int) -> list[int]:
        """Best first, with a small push towards a mix of shops, and one of each set
        of near-identical pictures (the same piece listed per colour)."""
        score = np.where(mask, score, -np.inf)
        order = np.argsort(-score)[: k * 6]
        order = [int(i) for i in order if np.isfinite(score[i])]
        picked, per_shop = [], {}
        while order and len(picked) < k:
            best = max(order, key=lambda i: score[i] - SHOP_SPREAD * per_shop.get(self.shop[i], 0))
            order.remove(best)
            if picked and float((self.sig[picked] @ self.sig[best]).max()) > SAME_PICTURE:
                continue
            picked.append(best)
            per_shop[self.shop[best]] = per_shop.get(self.shop[best], 0) + 1
        return picked

    def _answer(self, picked: list[int], mask: np.ndarray, **extra) -> dict:
        return {"results": [self.card(i) for i in picked], "matches": int(mask.sum()),
                "shops": self.info.get("shops", {}), "built": self.info.get("built"), **extra}

    def _picture_colour(self, metal: str) -> np.ndarray:
        """How sure the image model is that each picture shows this metal colour (0..1)."""
        if not hasattr(self, "_colour_p"):
            vecs = getattr(self.engine, "_dna_vectors", None)
            if vecs is None:
                return np.zeros(len(self.items))
            from . import dna
            from .attributes import TEMPERATURE
            self._colour_p = dna.softmax(self.sig @ vecs()["metal"].T * TEMPERATURE)
        return self._colour_p[:, METALS.index(metal)]

    def _word_score(self, text: str) -> np.ndarray:
        """Specific words of the request ("mangalsutra", "tennis", "pearl") found in product names.
        The type words are filters already, and the image model can't read every word."""
        words = {w.rstrip("s") for w in re.findall(r"[a-z]{4,}", text.lower()) if w not in GENERIC}
        if not words:
            return np.zeros(len(self.items))
        if not hasattr(self, "_title_words"):
            self._title_words = [{w.rstrip("s") for w in re.findall(r"[a-z]{4,}", p["title"].lower())}
                                 for p in self.items]
        return np.array([min(2, len(words & t)) for t in self._title_words], dtype=float)

    def _type_mask(self, cats: list[str], exclude: list[str] = ()) -> np.ndarray:
        mask = np.isin(self.cat, cats) if cats else np.ones(len(self.items), dtype=bool)
        if exclude:
            mask &= ~np.isin(self.cat, list(exclude))
        return mask

    def for_prompt(self, prompt: str, category: str | None = None, metal: str | None = None, k: int = K) -> dict:
        from .query import parse
        v = self.engine.domain.check(prompt)
        if not v.ok:
            return {"results": [], "matches": 0, "refused": True}
        q = parse(v.text)
        q.raw = prompt
        if category:
            q.category = None if category == "any" else category
            q.categories = [q.category] if q.category else []
        if metal:
            q.metal = None if metal == "any" else metal
        mask = self._type_mask(q.categories, q.exclude_categories)
        note = None
        if q.metal:
            mask &= self.colour[q.metal]
        for m in q.exclude_metals:
            mask &= ~self.colour[m]
        if "men" in q.intents:
            mask &= self.mens
        qvec = self.engine._query_vector(q)
        colour = PICTURE_COLOUR * self._picture_colour(q.metal) if q.metal else 0
        score = colour + (_z(self.sig @ qvec, mask) + TITLE_WEIGHT * _z(self.title @ qvec, mask)
                 + WORD_BOOST * self._word_score(v.text)
                 - (0 if "men" in q.intents else MENS_PENALTY) * self.mens)
        if not mask.any():
            note = "No product in the web collection fits these filters."
        return self._answer(self._pick(score, mask, k), mask, category=q.category, metal=q.metal, note=note)

    def for_photo(self, pq, words: str = "", category: str | None = None, metal: str | None = None,
                  k: int = K) -> dict:
        """Web products that look like a shopper's photo (or a linked picture), read once by
        SearchEngine.read_photo: SigLIP2 and DINOv2 together, like photo search. Its type is a
        filter when the reading is sure; its colour is preferred, not required (a shop may not
        say a colour). Words added to the photo refine it like a prompt."""
        from .query import parse
        q = None
        words = (words or "").strip()
        if words:
            v = self.engine.domain.check(words)
            if v.ok:
                q = parse(v.text)
        cats = [] if category == "any" else [category] if category else \
            (q.categories if q and q.categories else [pq.category] if pq.category else [])
        mask = self._type_mask(cats, q.exclude_categories if q else [])
        asked = None if metal == "any" else metal or (q.metal if q else None)
        if asked:
            mask &= self.colour[asked]
        score = _z(self.sig @ pq.vec, mask)
        if getattr(pq, "dvec", None) is not None:
            score = score + DINO_WEIGHT * _z(self.dino @ pq.dvec, mask)
        colour = asked or (None if metal == "any" else pq.metal)
        if colour:
            score = score + PICTURE_COLOUR * self._picture_colour(colour)
        if q is not None:
            qvec = self.engine._query_vector(q)
            score = score + 0.5 * _z(self.sig @ qvec, mask) + WORD_BOOST * self._word_score(words)
        note = None if mask.any() else "No product in the web collection fits these filters."
        return self._answer(self._pick(score, mask, k), mask, category=cats[0] if len(cats) == 1 else None,
                            metal=asked, note=note)

    def like_design(self, uid: int, metal: str | None = None, k: int = K) -> dict:
        """Web products that look like one of our designs.

        Our designs are studio renders, shop products are photos. Measured 2026-10-06 on
        the 2,921-product pool (scratchpad eval: top 8 results vs web titles naming a cut
        or a diamond layout; 686 designs with a known cut, 3,084 with a clear layout):

                                                   same cut   same layout   same motif
            SigLIP2 + DINOv2 (first version)        0.182       0.501        0.242
            SigLIP2 + 1.5 cut + 1 layout            0.342       0.682          -
            ... + 0.5 DINOv2                        0.340       0.678        0.233
            ... + 1 motif agreement (now)           0.337       0.666        0.360
        (motif: 296 designs with a clear heart / butterfly / floral / star / bow / infinity /
        leaf / evil-eye reading, against web titles naming one)

        DINOv2 tells near-identical photos apart but not a render from a photo (alone it
        was below chance there), so it only gets a small weight. The diamond readers
        (trained on our renders, scripts/train_diamond_dna.py) and the motif prompts read
        the web pictures' SigLIP2 vectors; motifs bring back shapes like butterflies, which
        the diamond readers don't know.
        The colour picked on the card only breaks near-ties: shape comes first."""
        e = self.engine
        src = e.by_uid[uid]
        mask = self._type_mask([src["category"]])
        qv = e.vecs[uid] + e.front[uid]
        qv = qv / np.linalg.norm(qv)
        score = _z(self.sig @ qv, mask)
        dia = self._diamonds()
        if dia is not None and getattr(e, "dia_cut", None) is not None:
            score = score + DESIGN_CUT * _z((dia["cut"] * e.dia_cut[uid]).sum(1), mask) \
                          + DESIGN_LAYOUT * _z((dia["layout"] * e.dia_layout[uid]).sum(1), mask)
        motif = self._motif_agreement(uid, src["category"])
        if motif is not None:
            score = score + DESIGN_MOTIF * _z(motif, mask)
        rows = self._design_dino_rows(uid)
        if rows is not None:
            score = score + DESIGN_DINO * _z((self.dino @ rows.T).max(1), mask)
        if metal:
            score = score + DESIGN_COLOUR * self._picture_colour(metal)
        mens = getattr(e, "mens", None)
        ours_for_men = bool(mens[uid]) if mens is not None else False
        score = score + MENS_PENALTY * (1 if ours_for_men else -1) * self.mens   # same rule as typed searches
        return self._answer(self._pick(score, mask, k), mask, category=src["category"], metal=metal,
                            source={"design_id": src["design_id"], "uid": uid})

    def _motif_agreement(self, uid: int, cat: str):
        """How strongly each web picture shows the motifs our design clearly has (the
        motif prompts of photo search's design DNA, read on SigLIP2 vectors)."""
        e = self.engine
        if not hasattr(e, "_category_text"):
            return None
        from . import dna
        ct = e._category_text(cat)
        pop = ct["motif_pop"]                                   # designs of this type x motifs
        k = int(np.searchsorted(np.flatnonzero(e.cat == cat), uid))
        clear = np.flatnonzero((pop < pop[k]).mean(0) >= MOTIF_CLEAR)
        if not len(clear):
            return None
        if not hasattr(self, "_motif_z"):
            self._motif_z = {}
        if cat not in self._motif_z:
            ws = dna.centred(self.sig @ ct["motif"].T)
            inside = self.cat == cat
            mu, sd = ws[inside].mean(0), ws[inside].std(0) + 1e-9
            self._motif_z[cat] = (ws - mu) / sd
        return self._motif_z[cat][:, clear].sum(1)

    def _diamonds(self):
        """The diamond readers' cut and layout readings of every web picture (made once)."""
        if not hasattr(self, "_dia"):
            e = self.engine
            self._dia = None
            if getattr(e, "dia_model", None):
                self._dia = {h: e._diamond_probs(h, self.sig) for h in ("cut", "layout")}
        return self._dia

    def like_product(self, pid: str, k: int = 12) -> dict:
        """Web products that look like a web product (the preview's "Related" grid)."""
        i = self.by_id.get(pid)
        if i is None:
            raise KeyError(pid)
        mask = self.cat == self.cat[i]
        mask[i] = False
        score = _z(self.sig @ self.sig[i], mask) + DINO_WEIGHT * _z(self.dino @ self.dino[i], mask)
        return self._answer(self._pick(score, mask, k), mask, source=self.card(i))

    def _design_dino_rows(self, uid: int):
        e = self.engine
        if getattr(e, "dino_views", None) is None:
            return None
        j = int(np.searchsorted(e._view_designs, uid))
        if j >= len(e._view_designs) or e._view_designs[j] != uid:
            return None
        end = e._view_starts[j + 1] if j + 1 < len(e._view_starts) else len(e.dino_views)
        return e.dino_views[e._view_starts[j]:end]

    # ---- pictures ----
    _fetching = threading.BoundedSemaphore(6)   # at most this many picture fetches to shops at once

    def picture(self, pid: str, size: int) -> Path:
        """The product's picture at this size (WebP), made once and kept."""
        if size not in SIZES:
            raise ValueError(size)
        i = self.by_id.get(pid)
        if i is None:
            raise KeyError(pid)
        out = IMG_DIR / f"{pid}_{size}.webp"
        if out.exists():
            return out
        from . import linksearch, photo
        with self._fetching:
            if out.exists():
                return out
            f = linksearch.fetch(sized_url(self.items[i]["image"], size),
                                 "image/avif,image/webp,image/png,image/jpeg,image/*;q=0.8",
                                 photo.MAX_BYTES, time.monotonic() + 20)
        return save_picture(pid, photo.read(f.data), size)


def save_picture(pid: str, im, size: int) -> Path:
    """Keep the picture at this size: shrunk with Lanczos when larger, enlarged at
    most 2x (Lanczos) when the shop has nothing as large, never cropped."""
    from PIL import Image
    im = im.convert("RGBA") if im.mode in ("RGBA", "LA", "P") else im.convert("RGB")
    long = max(im.size)
    if long > size:
        im = im.copy()
        im.thumbnail((size, size), Image.LANCZOS)
    elif long < size:
        scale = min(MAX_UPSCALE, size / long)
        if scale > 1.05:
            im = im.resize((round(im.width * scale), round(im.height * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "WEBP", quality=WEBP_QUALITY, method=6)
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    out = IMG_DIR / f"{pid}_{size}.webp"
    tmp = out.with_suffix(f".{threading.get_ident()}.tmp")
    tmp.write_bytes(buf.getvalue())
    tmp.replace(out)
    return out


__all__ = ["WebProducts", "SHOPS", "from_feed", "category_of", "colours_of", "material_of", "sized_url", "CATEGORIES"]
