"""Designs from other jewellers' websites ("brand designs", a demo).

The team (jewelers and admins) pastes a product link from another jeweller's
shop. The page is read with the same safe fetcher as link search
(linksearch.fetch: public addresses only, size and time limits) and gives:

  - the product's name, brand, SKU, description and price, from its
    structured data (JSON-LD), shop tags (og:price...) and, on Shopify shops,
    the product's own JSON (/products/<handle>.js);
  - the specification the page shows as label / value pairs ("Purity · 18 KT",
    "Gross Weight (g) · 3.2") or as tables (a price breakup);
  - its pictures, downloaded here: pages never load pictures from the other
    site, and the files are kept exactly as downloaded (no re-encoding).

The team then fills in our own version of the design (gold weight per purity,
diamonds, size, our pricing) and lists it. The listing keeps the original
details as they were read, and our details beside them. Customers can buy it
from us (the usual order flow, kind "brand") or open the seller's page.

Stored in Supabase (public.brand_designs, supabase/brand_designs.sql), only
by this server; pictures in data/brand_designs/.
"""
from __future__ import annotations

import base64
import html as htmlmod
import io
import json
import logging
import re
import secrets
import threading
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from html.parser import HTMLParser
from urllib.parse import parse_qsl, urljoin, urlsplit, urlunsplit

import numpy as np
from PIL import Image

from . import linksearch, orders, photo, purchase, webarchive
from .auth import _call
from .config import CATEGORIES, DATA, METALS
from .query import parse

log = logging.getLogger(__name__)
TABLE = "/rest/v1/brand_designs"
PICTURES = DATA / "brand_designs"

BUDGET = 45.0              # seconds for one link: the page, the shop's JSON and the pictures
MAX_PICTURES = 12          # pictures kept from a page (all go to the dataset storage)
MAX_LISTED = 8             # of which a listing shows at most this many
MAX_CANDIDATES = 16        # pictures tried
MIN_PICTURE_SIDE = 300     # product photos; smaller ones are thumbnails and icons
MAX_PICTURE_BYTES = 30 * 1024 * 1024   # full-size originals (a 2600 px PNG is a few MB)
DRAFT_TTL = 3600           # a fetched page waits this long for the team's form
MAX_DRAFTS = 24            # in memory, all users together (previews only: the files are on disk)
# The shop's own file, not a re-encoding of it: image servers (Shopify, Thumbor, ...) answer a
# browser that accepts WebP / AVIF with a smaller, lossy copy (measured: Melorra 2.6 MB PNG vs
# 143 KB WebP at the same size). Servers with only WebP files still send them (image/*).
PICTURE_ACCEPT = "image/png,image/jpeg;q=0.9,image/gif;q=0.8,image/*;q=0.5"
KEEP_FORMATS = {"JPEG": ("jpg", "image/jpeg"), "MPO": ("jpg", "image/jpeg"), "PNG": ("png", "image/png"),
                "WEBP": ("webp", "image/webp"), "GIF": ("gif", "image/gif")}
MEDIA_TYPES = {ext: ctype for ext, ctype in KEEP_FORMATS.values()}
FILE_RE = re.compile(r"^[0-9a-f]{32}\.(?:jpg|png|webp|gif)$")

PURITY_KEYS = [p["key"] for p in purchase.PURITIES]
FINENESS = {p["key"]: p["fineness"] for p in purchase.PURITIES}
RATIO = {p["key"]: p["ratio"] for p in purchase.PURITIES}


class BrandError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status, self.message = status, message


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _text(s, limit: int) -> str:
    return " ".join(str(s or "").split())[:limit]


# ---------------------------------------------------------------- reading the page

# Labels of a jewellery specification, as shops write them ("Gross Weight (g)", "Diamond Clarity").
SPEC_LABEL = re.compile(r"""(?x)
  (?:metal|gold|silver|platinum)(?:\s+(?:type|colou?r|purity|karat(?:age)?|carat(?:age)?|weight|tone|finish))?
| purity | karat(?:age)? | caratage | kt | fineness | material | plating | finish | hallmark(?:ed|ing)?
| (?:net|gross|total|product|approx\.?|approximate)\s+(?:metal\s+|gold\s+)?weight | weight
| (?:total\s+)?(?:diamond|stone|gem\s?stone|gem|solitaire|pearl)s?(?:\s+(?:weight|carats?|carat\s+weight|ct|count|pieces|
     pcs|quantity|nos?\.?|type|shape|colou?r|clarity|quality|setting|size|cut|grade|certificate|certification))?
| (?:no\.?|number)\s+of\s+(?:diamonds|stones|gems|pieces) | (?:total\s+)?carat(?:\s+weight)? | total\s+ct
| clarity | colou?r(?:\s+grade)? | cut | shape | setting(?:\s+type)?
| certificat(?:e|ion)(?:\s+(?:by|no\.?|number))? | certified\s+by
| (?:ring|band|chain|bangle|bracelet|pendant|earring|necklace|product)?\s*(?:size|width|height|length|diameter|thickness|
     dimensions?)
| sku | product\s+(?:code|id) | (?:style|design|item|model)\s+(?:no\.?|number|code) | model
| collection | occasion | gender | style | type | category
| making(?:\s+charges?)? | gst | tax(?:es)? | discount | mrp | price | sub\s*total | total | grand\s+total
| (?:gold|diamond|stone|metal|gem\s?stone)\s+(?:value|price|rate|charges?)
""")
UNIT_IN_LABEL = re.compile(r"\s*\((?:g|gm|gms|grams?|ct|cts|carats?|mm|cm|in|inr|rs\.?|₹)\)\s*$", re.I)
SKIP_TAGS = {"script", "style", "noscript", "svg", "template", "head", "iframe", "button", "select", "option",
             "nav", "header", "footer"}   # page menus and footers are full of jewellery words
MAX_SPECS, MAX_TABLES, MAX_ROWS = 50, 4, 25


# On their own these name a shop's menus ("Gold · Gold Rings", "Gemstone · Shop by price"), not a spec.
MENU_WORDS = re.compile(r"(?:gold|silver|platinum|diamonds?|gem\s?stones?|gems?|solitaires?|pearls?|style|type|"
                        r"category|collection|price|total)")
TAG_KEYS = {"metal", "stone", "gemstone", "material", "plating", "purity", "karat", "colour", "color", "finish",
            "occasion", "gender", "setting", "shape"}


def _is_label(text: str, strict: bool = True) -> bool:
    t = UNIT_IN_LABEL.sub("", text.strip().rstrip(":").strip()).lower()
    if strict and MENU_WORDS.fullmatch(t):
        return False
    return 0 < len(t) <= 40 and bool(SPEC_LABEL.fullmatch(t))


def _is_value(text: str) -> bool:
    t = text.strip()
    words = len(t.split())
    return (0 < len(t) <= 80 and words <= 12 and not t.endswith("?") and not (t.endswith(".") and words > 5)
            and bool(re.search(r"\w", t)) and not _is_label(t, strict=False)
            and not re.search(r"\b(?:shop by|add to cart|buy now|view all|click|checked items)\b", t, re.I))


class _Specs(HTMLParser):
    """The page's visible text, in order, and its tables (rows of cell texts). A table's
    text is kept out of the running text: a wide table (a price breakup with weight,
    rate and value columns) would pair the wrong cells; its two-column rows come back
    as label and value."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.nodes, self.tables = [], []
        self._skip = 0
        self._depth = 0          # tables inside tables: only the outermost is read
        self._table = self._row = self._cell = None

    def handle_starttag(self, tag, attrs):
        if tag in SKIP_TAGS:
            self._skip += 1
        elif tag == "table":
            self._depth += 1
            if self._depth == 1:
                self._table = []
        elif tag == "tr" and self._table is not None:
            self._row = []
        elif tag in ("td", "th") and self._row is not None:
            self._cell = []
        elif tag == "br" and self._cell is not None:
            self._cell.append(" ")

    def handle_endtag(self, tag):
        if tag in SKIP_TAGS:
            self._skip = max(0, self._skip - 1)
        elif tag in ("td", "th") and self._cell is not None and self._row is not None:
            self._row.append(_text("".join(self._cell), 80))
            self._cell = None
        elif tag == "tr" and self._row is not None and self._table is not None:
            if any(self._row):
                self._table.append(self._row)
            self._row = None
        elif tag == "table":
            self._depth = max(0, self._depth - 1)
            if self._depth == 0 and self._table is not None:
                if len(self._table) >= 2:
                    self.tables.append(self._table[:MAX_ROWS + 1])
                for row in self._table:
                    if len(row) == 2 and all(row):
                        self.nodes += row
                self._table = None

    def handle_data(self, data):
        if self._skip:
            return
        t = " ".join(data.split())
        if not t:
            return
        if self._cell is not None:
            self._cell.append(data)
        if self._table is None and len(self.nodes) < 20000:
            self.nodes.append(t[:300])


def _pairs_from_nodes(nodes: list[str]) -> list[tuple[str, str]]:
    """Label / value pairs: a label node followed by its value node ("Purity", "18 KT"),
    or one node "Label: value"."""
    out, i = [], 0
    while i < len(nodes):
        t = nodes[i]
        if _is_label(t):
            j = i + 1
            while j < len(nodes) and nodes[j] in (":", "-", "|", "·"):
                j += 1
            if j < len(nodes) and _is_value(nodes[j]):
                out.append((t.rstrip(": ").strip(), nodes[j].lstrip(": ").strip()))
                i = j + 1
                continue
        elif ":" in t:
            label, _, value = t.partition(":")
            if _is_label(label) and _is_value(value):
                out.append((label.strip(), value.strip()))
        i += 1
    return out


def _pairs_from_text(text: str) -> list[tuple[str, str]]:
    """ "Ring Diameter: 1.66 cm" lines of a description."""
    out = []
    for line in re.split(r"[\n\r•|]+", text or ""):
        label, sep, value = line.partition(":")
        if sep and _is_label(label) and _is_value(value):
            out.append((label.strip(), value.strip()))
    return out


def _useful_table(rows: list[list[str]]) -> bool:
    flat = " ".join(" ".join(r) for r in rows).lower()
    return (max(len(r) for r in rows) <= 8 and len(flat) <= 4000
            and bool(re.search(r"gold|diamond|metal|stone|weight|making|gst|carat|purity|total|silver|platinum", flat)))


def _ld_products(raw_blocks: list[str]) -> list[dict]:
    out = []
    for raw in raw_blocks:
        try:
            linksearch._ld_products(json.loads(raw), out)
        except ValueError:
            continue
    return out


def _money(v) -> float | None:
    """ "1,099.00" / 25828 / "₹ 34,132" -> a number."""
    if isinstance(v, bool) or v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v) if v > 0 else None
    m = re.search(r"\d[\d,]*(?:\.\d+)?", str(v))
    if not m:
        return None
    try:
        n = float(Decimal(m.group().replace(",", "")))
    except InvalidOperation:
        return None
    return n if n > 0 else None


def _offer(prod: dict) -> tuple[float | None, str]:
    offers = prod.get("offers")
    for o in (offers if isinstance(offers, list) else [offers]):
        if isinstance(o, dict):
            price = _money(o.get("price")) or _money(o.get("lowPrice"))
            if price:
                return price, _text(o.get("priceCurrency"), 5).upper()
    return None, ""


def _brand_name(prod: dict) -> str:
    b = prod.get("brand")
    if isinstance(b, dict):
        return _text(b.get("name"), 80)
    if isinstance(b, list) and b and isinstance(b[0], dict):
        return _text(b[0].get("name"), 80)
    return _text(b, 80) if isinstance(b, str) else ""


def _plain(html_text: str) -> str:
    """A product description's HTML -> plain text with its line breaks."""
    t = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", html_text or "")
    t = re.sub(r"(?i)<\s*(br|/p|/li|/div|/h\d)\s*/?>", "\n", t)
    t = htmlmod.unescape(re.sub(r"<[^>]+>", " ", t))
    lines = [" ".join(line.split()) for line in t.split("\n")]
    return "\n".join(line for line in lines if line)[:3000]


def _shopify(url: str, html: str, deadline: float) -> dict | None:
    """A Shopify shop's product JSON (/products/<handle>.js), when the page is one."""
    p = urlsplit(url)
    m = re.search(r"(/(?:[a-z]{2}(?:-[a-z]{2})?/)?products/[^/?#]+)", p.path)
    if not m or not re.search(r"cdn\.shopify\.com|Shopify\.shop|shopify-section", html):
        return None
    try:
        f = linksearch.fetch(urlunsplit((p.scheme, p.netloc, m.group(1).removesuffix(".js") + ".js", "", "")),
                             "application/json,text/javascript,*/*;q=0.5", linksearch.MAX_HTML, deadline)
        data = json.loads(f.data)
    except (linksearch.LinkError, ValueError):
        return None
    return data if isinstance(data, dict) and data.get("title") else None


@dataclass
class Scraped:
    original: dict
    picture_urls: list = field(default_factory=list)
    product: dict | None = None     # the shop's own product data (Shopify), kept with the page


def read_product(url: str, html: str, deadline: float) -> Scraped:
    """Everything the product page says about the design (no pictures downloaded yet)."""
    page = linksearch.read_page(html, url)
    reader = linksearch._Reader()
    specs_reader = _Specs()
    for r in (reader, specs_reader):
        try:
            r.feed(html)
            r.close()
        except Exception:   # broken markup: keep what was read
            pass
    meta = {}
    for k, v in reader.meta:
        meta.setdefault(k, v)
    prods = _ld_products(reader.ld)
    prod = prods[0] if prods else {}
    shop = _shopify(url, html, deadline)

    title = _text(prod.get("name") or (shop or {}).get("title") or meta.get("og:title") or page.title, 200)
    title = re.split(r"\s[|–—]\s", title)[0].strip() or title
    price, currency = _offer(prod)
    mrp = None
    if shop:
        if not price and _money(shop.get("price")):
            price = _money(shop["price"]) / 100          # Shopify keeps prices in paise / cents
        if (shop.get("compare_at_price") or 0) > (shop.get("price") or 0):
            mrp = _money(shop["compare_at_price"]) / 100
    if not price:
        price = _money(meta.get("product:price:amount") or meta.get("og:price:amount"))
    currency = currency or _text(meta.get("product:price:currency") or meta.get("og:price:currency"), 5).upper()
    if price and not currency and re.search(r"₹|\bINR\b|\bRs\.?", html[:400000]):
        currency = "INR"
    description = _plain(prod.get("description") if isinstance(prod.get("description"), str) else "")
    if shop and len(_plain(shop.get("description") or "")) > len(description):
        description = _plain(shop.get("description") or "")
    if not description:
        description = _text(meta.get("og:description") or meta.get("description"), 1500)

    pairs: list[tuple[str, str]] = []
    for prop in (prod.get("additionalProperty") or []) if isinstance(prod.get("additionalProperty"), list) else []:
        if isinstance(prop, dict) and prop.get("name") and prop.get("value") not in (None, ""):
            pairs.append((_text(prop["name"], 40), _text(prop["value"], 80)))
    for key, label in (("material", "Material"), ("color", "Colour"), ("category", "Category")):
        if isinstance(prod.get(key), str) and prod[key].strip():
            pairs.append((label, _text(prod[key], 80)))
    w = prod.get("weight")
    if isinstance(w, dict) and w.get("value"):
        pairs.append(("Weight", _text(f"{w['value']} {w.get('unitText') or w.get('unitCode') or ''}", 40)))
    if shop:
        for tag in shop.get("tags") or []:   # "Metal_925 Silver", "Stone_Zircon"
            m = re.fullmatch(r"([A-Za-z][A-Za-z ]{1,30})_(.{1,60})", str(tag))
            if m and m.group(1).strip().lower() in TAG_KEYS:
                pairs.append((m.group(1).strip(), m.group(2).strip()))
        for opt in shop.get("options") or []:
            vals = [v for v in opt.get("values") or [] if v != "Default Title"]
            if vals:
                pairs.append((_text(opt.get("name"), 40), _text(", ".join(map(str, vals)), 80)))
    pairs += _pairs_from_nodes(specs_reader.nodes) + _pairs_from_text(description)
    specs, seen = [], set()
    for label, value in pairs:
        key = (re.sub(r"\W+", " ", label.lower()).strip(), value.lower())
        if label and value and key not in seen:
            seen.add(key)
            specs.append({"label": label, "value": value})
        if len(specs) >= MAX_SPECS:
            break
    tables = [{"rows": t} for t in specs_reader.tables if _useful_table(t)][:MAX_TABLES]

    sku = _text(prod.get("sku") or prod.get("mpn") or "", 60)
    if not sku and shop:
        sku = _text(next((v.get("sku") for v in shop.get("variants") or [] if v.get("sku")), ""), 60)
    original = {
        "url": url, "site": (urlsplit(url).hostname or "").removeprefix("www."),
        "title": title, "brand": _brand_name(prod) or _text((shop or {}).get("vendor") or meta.get("og:site_name"), 80),
        "sku": sku, "price": price, "mrp": mrp, "currency": currency or ("INR" if price else ""),
        "description": description[:3000], "specs": specs, "tables": tables, "fetched_at": _now(),
        "via": "shopify" if shop else "json-ld" if prod else "page",
    }
    original["key"] = key_figures(original)

    urls = []
    if shop:
        for img in shop.get("images") or []:
            if isinstance(img, str):
                urls.append(urljoin(url, "https:" + img if img.startswith("//") else img))
    urls += [p.url for p in page.pictures]
    seen_u, picture_urls = set(), []
    for u in urls:
        norm = re.sub(r"([?&])(width|height|w|h|crop|v)=[^&]*", r"\1", u).rstrip("?&")
        if norm not in seen_u:
            seen_u.add(norm)
            picture_urls.append(u)
    return Scraped(original, picture_urls[:MAX_CANDIDATES], shop)


# ---------------------------------------------------------------- key figures

PURITY_RE = re.compile(r"\b(9|10|14|18|22|24)\s*(?:k|kt|kts|karat|carat|ct)\b", re.I)
NUM_RE = re.compile(r"(\d+(?:\.\d+)?)")


def _num(value: str) -> float | None:
    m = NUM_RE.search(value.replace(",", ""))
    return float(m.group(1)) if m else None


def key_figures(original: dict) -> dict:
    """The few figures that can be compared with our version: purity, metal, gold weight,
    diamond carats and count, read from the page's labels (never guessed from prose)."""
    specs = original.get("specs") or []
    text = " ".join([original.get("title", "")] + [f"{s['label']} {s['value']}" for s in specs])
    out = {"purity": None, "metal": None, "gold_g": None, "gross_g": None, "diamond_ct": None, "diamond_count": None,
           "diamond_quality": None}
    plated = re.search(r"plated|plating|gold[- ]tone|vermeil|stainless|brass|base metal|alloy", text, re.I)
    m = PURITY_RE.search(text)
    if re.search(r"\b925\b|sterling|\bsilver\b", text, re.I):
        out["metal"], out["purity"] = "silver", "925 silver" if re.search(r"\b925\b|sterling", text, re.I) else None
    elif plated:
        out["metal"] = "gold plated"
    elif re.search(r"\bplatinum\b", text, re.I):
        out["metal"] = "platinum"
    else:
        q = parse(text[:500])
        out["metal"] = q.metal if q.metal in METALS else None
        if m:
            out["purity"] = f"{m.group(1)}K"
    # a price breakup table: "9 KT Yellow Gold | 1.490 g | ...", "SI IJ round - 18 No.s | 0.180 ct | ..."
    for t in original.get("tables") or []:
        for row in t["rows"]:
            line = " | ".join(row)
            g = next((float(x.group(1)) for c in row if (x := re.fullmatch(r"(\d+(?:\.\d+)?)\s*(?:g|gm|gms|grams?)", c.strip(), re.I))), None)
            ct = next((float(x.group(1)) for c in row if (x := re.fullmatch(r"(\d+(?:\.\d+)?)\s*(?:ct|cts|carats?)", c.strip(), re.I))), None)
            pm = PURITY_RE.search(line)
            if g is not None and pm and re.search(r"gold|platinum", line, re.I) and out["gold_g"] is None:
                out["gold_g"] = g
                out["purity"] = out["purity"] or f"{pm.group(1)}K"
                if out["metal"] is None:
                    out["metal"] = parse(line).metal if parse(line).metal in METALS else None
            elif ct is not None and re.search(r"diamond|solitaire|\b(?:SI|VS|VVS|IF)\d?\b|\b[D-K]{2}\b", line):
                out["diamond_ct"] = round((out["diamond_ct"] or 0) + ct, 3)
                n = re.search(r"(\d+)\s*(?:no\.?s?|pcs|pieces|stones)\b", line, re.I)
                if n:
                    out["diamond_count"] = (out["diamond_count"] or 0) + int(n.group(1))
    # the stated figures of a description: "Set in 18 KT Yellow Gold(2.150 g) with diamonds (0.280 ct ,FG-SI)"
    desc = original.get("description") or ""
    if (g := re.search(r"(?:(\d{1,2})\s*KT?\s+)?[A-Za-z ]{0,20}(?:gold|silver|platinum)\s*\(\s*(\d+(?:\.\d+)?)\s*g\s*\)",
                       desc, re.I)):
        if out["metal"] not in ("silver", "gold plated") and out["gold_g"] is None:
            out["gold_g"] = float(g.group(2))
            if g.group(1) and not out["purity"]:
                out["purity"] = f"{g.group(1)}K"
    if (d := re.search(r"diamonds?\s*\(\s*(\d+(?:\.\d+)?)\s*ct\b\s*,?\s*([A-Z]{1,3}-[A-Z0-9]{1,5})?", desc, re.I)):
        out["diamond_ct"] = out["diamond_ct"] or float(d.group(1))
        if d.group(2):
            out["diamond_quality"] = d.group(2)
    for s in specs:
        label, v = s["label"].lower(), s["value"]
        n = _num(v)
        if n is None:
            continue
        grams = bool(re.search(r"\bg(?:m|ms|rams?)?\b|\(g\)", f"{label} {v.lower()}"))
        if re.search(r"(gold|metal|net)\s+weight", label) and out["gold_g"] is None and (grams or n < 200) \
                and out["metal"] not in ("silver", "gold plated"):
            out["gold_g"] = n
        elif re.search(r"gross\s+weight|^weight$|product\s+weight", label) and out["gross_g"] is None and grams:
            out["gross_g"] = n
        elif re.search(r"(diamond|stone).*(weight|carat|ct)|total\s+(carat|ct)|carat\s+weight", label) \
                and out["diamond_ct"] is None and n < 100:
            out["diamond_ct"] = n
        elif re.search(r"(no\.?|number)\s+of\s+(diamonds|stones)|(diamond|stone)s?\s+(count|pieces|pcs|nos?\.?|quantity)",
                       label) and out["diamond_count"] is None and n == int(n):
            out["diamond_count"] = int(n)
    return out


def prefill(original: dict) -> dict:
    """A starting point for our version, from the original's figures. The team checks
    and changes every value before it is listed."""
    k = original.get("key") or key_figures(original)
    q = parse(f"{original.get('title', '')} {original.get('description', '')[:200]}")
    category = q.categories[0] if q.categories else None
    metals = [k["metal"]] if k.get("metal") in METALS else list(METALS)
    purity = (k.get("purity") or "").lower().replace(" ", "")
    purity = purity if purity in PURITY_KEYS else None
    purities = sorted({"14k", "18k"} | ({purity} if purity else set()), key=PURITY_KEYS.index)
    if "22k" in purities and metals != ["yellow_gold"]:
        purities.remove("22k")
    gold = {}
    if k.get("gold_g") and purity:   # another purity (9K, silver) can't be converted: the team enters it
        k18 = k["gold_g"] / RATIO[purity]
        gold = {p: round(k18 * RATIO[p], 3) for p in purities}
    stones = []
    if k.get("diamond_ct") or k.get("diamond_count"):
        stones.append({"shape": (q.shape or "Round").title() if isinstance(q.shape, str) else "Round", "size": "",
                       "count": k.get("diamond_count"), "carat": k.get("diamond_ct"), "setting": ""})
    dims = [{"label": s["label"], "value": s["value"]} for s in original.get("specs") or []
            if re.search(r"size|width|height|length|diameter|thickness|dimension", s["label"], re.I)][:4]
    return {"name": original.get("title", "")[:120], "design_no": "", "category": category, "metals": metals,
            "purities": purities, "gold_g": gold, "stones": stones, "diamond_quality": k.get("diamond_quality") or "",
            "dimensions": dims,
            "ring_size_in": None, "note": "",
            "pricing": {"gold_rate_24k": None, "making_mode": "per_g", "making_value": None, "diamond_value": 0,
                        "other_label": "", "other_value": 0, "gst_pct": 3}}


# ---------------------------------------------------------------- pictures, at their best

# Query parameters and path parts with which image servers make a smaller copy of a file.
SIZE_PARAMS = {"width", "height", "w", "h", "sw", "sh", "sm", "size", "resize", "fit", "crop", "quality", "q", "dpr",
               "auto", "fm", "format", "im", "imwidth", "imheight", "wid", "hei", "qlt", "fmt", "odnwidth",
               "odnheight", "odnbg", "tr", "tx", "scale"}
IMG_EXT = r"(?=\.(?:jpe?g|png|webp|gif)$)"


def upgrade_urls(url: str) -> list[str]:
    """A picture's link -> links to try, the full-size original first:
    ?width=600 (Shopify, imgix), /fit-in/515x515/ (Thumbor), /upload/w_500,c_fill/ (Cloudinary),
    /cache/<hash>/ (Magento), -300x300.jpg (WordPress), _600x.jpg (old Shopify)."""
    p = urlsplit(url)
    query = "&".join(f"{k}={v}" if v else k for k, v in parse_qsl(p.query, keep_blank_values=True)
                     if k.lower() not in SIZE_PARAMS)
    path = p.path
    path = re.sub(r"/fit-in/\d{1,5}x\d{1,5}/", "/", path)
    path = re.sub(r"^/(?:unsafe/)?\d{2,5}x\d{2,5}/", "/", path)
    path = re.sub(r"/upload/(?:[a-z]{1,3}_[^/]+/)+", "/upload/", path)
    path = re.sub(r"/cache/(?:\d+/)?(?:[a-z_]+/)?[0-9a-f]{32}/", "/", path)
    path = re.sub(r"-\d{2,4}x\d{2,4}" + IMG_EXT, "", path, flags=re.I)
    path = re.sub(r"_(?:pico|icon|thumb|small|compact|medium|large|grande|\d{2,4}x\d{0,4}|x\d{2,4})(?:_crop_\w+)?(?:@\dx)?"
                  + IMG_EXT, "", path, flags=re.I)
    best = urlunsplit((p.scheme, p.netloc, path, query, ""))
    return [best, url] if best != url else [url]


@dataclass
class Pic:
    data: bytes
    ext: str
    ctype: str
    width: int
    height: int
    small: object      # PIL image, at most 1024 px: previews and duplicates only
    source: str        # where the file came from
    asked: str         # the link on the page
    sign: object = None   # 32 x 32 colour thumbnail: the same photo at another size


SAME_PHOTO = 4.0   # mean colour difference (0-255) of two thumbnails of one photo at two sizes


def _sign(im):
    return np.asarray(_flat(im).resize((32, 32), Image.LANCZOS), dtype=np.float32)


def _same_photo(a: Pic, b: Pic) -> bool:
    """The same photo at another size. In colour: a ring photographed in yellow, white
    and rose gold is three photos (greyscale would make them one)."""
    return (abs(a.width / a.height - b.width / b.height) < 0.01
            and float(np.abs(a.sign - b.sign).mean()) < SAME_PHOTO)


def _download(url: str, deadline: float) -> Pic | None:
    """A picture of the page, the largest copy its server gives, exactly as sent."""
    for link in upgrade_urls(url):
        try:
            f = linksearch.fetch(link, PICTURE_ACCEPT, MAX_PICTURE_BYTES, deadline)
            with Image.open(io.BytesIO(f.data)) as im:
                fmt, (w, h) = im.format, im.size
            if fmt not in KEEP_FORMATS or min(w, h) < MIN_PICTURE_SIDE or w * h > photo.MAX_PIXELS:
                continue
            small = photo.read(f.data)      # decodes the whole file: a broken or hostile file fails here
        except (linksearch.LinkError, photo.PhotoError, OSError, Image.DecompressionBombError, ValueError):
            continue
        ext, ctype = KEEP_FORMATS[fmt]
        return Pic(f.data, ext, ctype, w, h, small, f.url, url, _sign(small))
    return None


def download_pictures(urls: list[str], deadline: float) -> list[Pic]:
    """The page's pictures, each once: the same photo at two sizes keeps the larger."""
    with ThreadPoolExecutor(4) as pool:
        got = [g for g in pool.map(lambda u: _download(u, deadline), urls) if g]
    kept: list[Pic] = []
    for g in got:
        twin = next((i for i, k in enumerate(kept) if k.data == g.data or _same_photo(k, g)), None)
        if twin is None:
            kept.append(g)
        elif g.width * g.height > kept[twin].width * kept[twin].height:
            kept[twin] = g
    return kept[:MAX_PICTURES]


def _flat(im):
    if im.mode in ("RGBA", "LA", "P"):
        bg = Image.new("RGB", im.size, "white")
        rgba = im.convert("RGBA")
        bg.paste(rgba, mask=rgba.split()[-1])
        return bg
    return im.convert("RGB")


def _jpeg(im, side: int) -> bytes:
    im = _flat(im.copy())
    im.thumbnail((side, side))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=82)
    return buf.getvalue()


def _preview(im) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(_jpeg(im, 360)).decode()


# ---------------------------------------------------------------- fetched pages waiting for the form

@dataclass
class Draft:
    owner: str
    created: float
    original: dict
    folder: str          # where the page's files are (webarchive)
    pictures: list       # [{file, width, height, ...}] as stored
    web_id: int | None = None


_drafts: "OrderedDict[str, Draft]" = OrderedDict()
_lock = threading.Lock()


def _keep_draft(d: Draft) -> str:
    token = secrets.token_hex(12)
    with _lock:
        now = time.monotonic()
        for k in [k for k, v in _drafts.items() if now - v.created > DRAFT_TTL]:
            del _drafts[k]
        while len(_drafts) >= MAX_DRAFTS:
            _drafts.popitem(last=False)
        _drafts[token] = d
    return token


def take_draft(token: str, owner: str) -> Draft:
    with _lock:
        d = _drafts.get(token)
        if not d or d.owner != owner or time.monotonic() - d.created > DRAFT_TTL:
            raise BrandError(410, "This fetched page has expired. Paste the link and fetch it again, "
                                  "or open it from the scraped designs below.")
        return d


def drop_draft(token: str):
    with _lock:
        _drafts.pop(token, None)


def _answer(token: str, d: Draft, previews: list[str]) -> dict:
    return {"token": token, "original": d.original, "folder": d.folder, "web_id": d.web_id,
            "pictures": [{"i": i, "preview": previews[i], "width": p["width"], "height": p["height"],
                          "bytes": p.get("bytes")} for i, p in enumerate(d.pictures)],
            "prefill": prefill(d.original), "options": form_options(), "max_listed": MAX_LISTED}


def fetch_link(text: str, owner: str, owner_name: str = "") -> dict:
    """A product link -> its original details, its pictures (previews) and a starting
    point for our version. Everything fetched is written to data/web_designs/ at once
    (webarchive.stage); the caller sends it on to the dataset storage."""
    try:
        url = linksearch.normalise(text)
    except linksearch.LinkError as e:
        raise BrandError(400, "Paste the link of the design's product page. " + e.message)
    deadline = time.monotonic() + BUDGET
    try:
        f = linksearch.fetch(url, "text/html,application/xhtml+xml;q=0.9,*/*;q=0.5", linksearch.MAX_HTML, deadline)
    except linksearch.LinkError as e:
        raise BrandError(400, e.message)
    if f.ctype.startswith("image/"):
        raise BrandError(400, "That link is a picture. Paste the link of the product page, so its price and "
                              "details can be read too.")
    if "html" not in f.ctype and not f.data[:300].lstrip().lower().startswith((b"<!doctype html", b"<html")):
        raise BrandError(400, "That link isn't a web page.")
    scraped = read_product(f.url, f.data.decode("utf-8", "replace"), deadline)
    pictures = download_pictures(scraped.picture_urls, deadline)
    o = scraped.original
    if not pictures and not o["title"]:
        raise BrandError(422, "No product could be read from that page. Some shops build their pages in the "
                              "browser or block automatic visits.")
    folder = webarchive.folder_for(o)
    files = webarchive.stage(folder, pictures, o, f.data, scraped.product, owner_name,
                             _jpeg(pictures[0].small, 480) if pictures else None)
    d = Draft(owner, time.monotonic(), o, folder, files)
    return _answer(_keep_draft(d), d, [_preview(p.small) for p in pictures]) | {"_draft": d}


def reopen(row: dict, owner: str) -> dict:
    """A design of the collection (public.web_designs) back in the team's form,
    its pictures read from the dataset storage: no new visit to the shop."""
    previews, kept = [], []
    for p in (row.get("pictures") or [])[:MAX_PICTURES]:
        try:
            previews.append(_preview(photo.read(webarchive.read(row["folder"], p["file"]))))
            kept.append(p)
        except Exception as e:   # a file the storage can't give now: left out
            log.warning("brand designs: %s/%s unreadable (%s)", row["folder"], p.get("file"), e)
    if not kept:
        raise BrandError(503, "The pictures of this design can't be read from the dataset storage right now.")
    d = Draft(owner, time.monotonic(), row["original"], row["folder"], kept, row["id"])
    return _answer(_keep_draft(d), d, previews)


def chosen_pictures(d: Draft, picks: list[int]) -> list[tuple]:
    """The pictures the team kept, in their order -> [(bytes, ext, content type, width, height)]."""
    out = []
    for i in picks:
        p = d.pictures[i]
        ext = p["file"].rsplit(".", 1)[-1]
        out.append((webarchive.read(d.folder, p["file"]), ext, MEDIA_TYPES.get(ext, "image/jpeg"), p["width"], p["height"]))
    return out


def form_options() -> dict:
    return {"categories": CATEGORIES, "metals": METALS,
            "purities": [{k: p[k] for k in ("key", "label", "fineness")} for p in purchase.PURITIES],
            "purities_for": purchase.PURITIES_FOR, "ratio": RATIO, "making_modes": orders.MAKING_MODES}


# ---------------------------------------------------------------- our version

def _f(v, lo, hi, what) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not lo <= v <= hi:
        raise BrandError(400, f"Check {what}.")
    return float(v)


def clean_ours(raw: dict) -> dict:
    """The team's form, checked: what our version of the design is and costs."""
    if not isinstance(raw, dict):
        raise BrandError(400, "Fill in our details.")
    name = orders.clean_text(raw.get("name"), 120).replace("\n", " ")
    if len(name) < 2:
        raise BrandError(400, "Give the design a name.")
    category = raw.get("category")
    if category not in CATEGORIES:
        raise BrandError(400, "Choose the category.")
    metals = [m for m in METALS if m in (raw.get("metals") or [])]
    if not metals:
        raise BrandError(400, "Choose at least one metal colour.")
    purities = [p for p in PURITY_KEYS if p in (raw.get("purities") or [])]
    if not purities:
        raise BrandError(400, "Choose at least one gold purity.")
    for m in metals:
        if not set(purities) & set(purchase.PURITIES_FOR[m]):
            raise BrandError(400, f"No chosen purity is made in {m.replace('_', ' ')} (22K is yellow gold only).")
    if "22k" in purities and "yellow_gold" not in metals:
        raise BrandError(400, "22K is made in yellow gold only.")
    gold_in = raw.get("gold_g") or {}
    gold = {p: round(_f(gold_in.get(p), 0.01, 2000, f"the gold weight for {p.upper()}"), 3) for p in purities}
    stones = []
    for s in (raw.get("stones") or [])[:30]:
        if not isinstance(s, dict):
            continue
        shape = orders.clean_text(s.get("shape"), 30)
        if not shape:
            continue
        count = s.get("count")
        if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= 10000:
            raise BrandError(400, f"Check the number of {shape} stones.")
        stones.append({"shape": shape, "size": orders.clean_text(s.get("size"), 30), "count": count,
                       "carat": round(_f(s.get("carat"), 0, 1000, f"the carats of the {shape} stones"), 3),
                       "setting": orders.clean_text(s.get("setting"), 40), "gem": "Diamond"})
    dims = []
    for d in (raw.get("dimensions") or [])[:8]:
        if isinstance(d, dict) and orders.clean_text(d.get("label"), 40) and orders.clean_text(d.get("value"), 40):
            dims.append({"label": orders.clean_text(d["label"], 40), "value": orders.clean_text(d["value"], 40)})
    ring = raw.get("ring_size_in")
    if category != "ring" or ring in (None, ""):
        ring = None
    elif isinstance(ring, bool) or not isinstance(ring, int) or ring not in orders.RING_SIZES:
        raise BrandError(400, "Choose the ring size the design is made at.")
    p = raw.get("pricing") or {}
    if p.get("making_mode") not in orders.MAKING_MODES:
        raise BrandError(400, "Choose how the making charge is counted.")
    pricing = {
        "gold_rate_24k": round(_f(p.get("gold_rate_24k"), 100, 1_000_000, "the 24K gold rate"), 2),
        "making_mode": p["making_mode"],
        "making_value": round(_f(p.get("making_value"), 0, 100_000_000, "the making charge"), 2),
        "diamond_value": round(_f(p.get("diamond_value") or 0, 0, 1_000_000_000, "the diamond value"), 2),
        "other_label": orders.clean_text(p.get("other_label"), 60),
        "other_value": round(_f(p.get("other_value") or 0, 0, 100_000_000, "the other charges"), 2),
        "gst_pct": round(_f(p.get("gst_pct"), 0, 28, "the GST"), 2),
    }
    if pricing["making_mode"] == "percent" and pricing["making_value"] > 100:
        raise BrandError(400, "A making charge in percent must be 100 or less.")
    ours = {"name": name, "design_no": orders.clean_text(raw.get("design_no"), 40).replace("\n", " "),
            "category": category, "metals": metals, "purities": purities, "gold_g": gold, "stones": stones,
            "diamond_quality": orders.clean_text(raw.get("diamond_quality"), 60), "dimensions": dims,
            "ring_size_in": ring, "pricing": pricing, "note": orders.clean_text(raw.get("note"), 1500)}
    ours["prices"] = prices(ours)
    return ours


def purity_rate(rate_24k: float, purity: str) -> float:
    """The gold rate of a purity from the 24K rate, by its gold content (18K = 750/1000)."""
    return float((Decimal(str(rate_24k)) * Decimal(FINENESS[purity]) / 1000).quantize(Decimal("0.01")))


def quote_for(ours: dict, purity: str) -> dict:
    """Our pricing for one purity, in the jeweler panel's quote form (orders.bill)."""
    p = ours["pricing"]
    return {"gold_weight_g": ours["gold_g"][purity], "gold_rate": purity_rate(p["gold_rate_24k"], purity),
            "making_mode": p["making_mode"], "making_value": p["making_value"], "diamond_value": p["diamond_value"],
            "diamond_quality": ours.get("diamond_quality", ""), "other_label": p["other_label"],
            "other_value": p["other_value"], "gst_pct": p["gst_pct"]}


def prices(ours: dict) -> dict:
    """Our price of one piece in each purity, with its breakup."""
    out = {}
    for pur in ours["purities"]:
        q = quote_for(ours, pur)
        out[pur] = {**orders.bill(orders.build_quote(q, "")), "gold_rate": q["gold_rate"]}
    return out


# ---------------------------------------------------------------- pictures on disk

def save_pictures(pictures: list) -> list[dict]:
    """The chosen pictures, written as downloaded -> [{file, width, height}]."""
    PICTURES.mkdir(parents=True, exist_ok=True)
    out = []
    for data, ext, _ctype, w, h in pictures:
        name = f"{secrets.token_hex(16)}.{ext}"
        (PICTURES / name).write_bytes(data)
        out.append({"file": name, "width": w, "height": h})
    return out


def remove_pictures(files: list[dict]):
    for f in files:
        if FILE_RE.match(f.get("file", "")):
            (PICTURES / f["file"]).unlink(missing_ok=True)


def picture_path(name: str):
    if not FILE_RE.match(name):
        return None
    path = PICTURES / name
    return path if path.is_file() else None


def picture_url(f: dict) -> str:
    return "/brand-media/" + f["file"]


# ---------------------------------------------------------------- Supabase

async def _req(method: str, params: dict | None = None, **kw):
    r = await _call(method, TABLE, params=params or {}, **kw)
    if r.status_code == 404:
        raise BrandError(503, "Brand designs aren't set up yet (run supabase/brand_designs.sql in Supabase).")
    if r.status_code >= 300:
        log.warning("brand designs: %s -> %s %s", method, r.status_code, r.text[:300])
        raise BrandError(502, "Brand designs are unavailable right now. Try again shortly.")
    return r


async def create(original: dict, ours: dict, pictures: list[dict], user: dict) -> dict:
    row = {"status": "listed", "source_url": original["url"], "site": original["site"], "original": original,
           "ours": ours, "pictures": pictures, "category": ours["category"], "created_by": user["uid"],
           "created_by_name": user.get("name") or user.get("email") or "", "updated_at": _now()}
    r = await _req("POST", json=row, headers={"Prefer": "return=representation"})
    return r.json()[0]


async def get(design_id: int, include_hidden: bool = False) -> dict:
    params = {"id": f"eq.{design_id}", "select": "*"}
    if not include_hidden:
        params["status"] = "eq.listed"
    rows = (await _req("GET", params)).json()
    if not rows:
        raise BrandError(404, "This design is no longer listed.")
    return rows[0]


async def listed(category: str | None = None, limit: int = 200) -> list[dict]:
    params = {"select": "*", "status": "eq.listed", "order": "created_at.desc", "limit": str(limit)}
    if category:
        params["category"] = f"eq.{category}"
    return (await _req("GET", params)).json()


async def hide(design_id: int) -> dict:
    r = await _req("PATCH", {"id": f"eq.{design_id}"}, json={"status": "hidden", "updated_at": _now()},
                   headers={"Prefer": "return=representation"})
    rows = r.json()
    if not rows:
        raise BrandError(404, "Design not found.")
    return rows[0]


# ---------------------------------------------------------------- what pages show

def _seller_link(url: str) -> str | None:
    p = urlsplit(url or "")
    return url if p.scheme in ("http", "https") and p.hostname else None


def card(row: dict) -> dict:
    """A brand design as a catalogue card."""
    o, ours = row["original"], row["ours"]
    pics = row.get("pictures") or []
    totals = [v["total"] for v in ours.get("prices", {}).values()]
    return {"id": row["id"], "name": ours["name"], "design_no": ours.get("design_no", ""), "category": ours["category"],
            "metals": ours["metals"], "site": row["site"], "brand": o.get("brand") or row["site"],
            "thumb": picture_url(pics[0]) if pics else None,
            "seller_price": o.get("price"), "currency": o.get("currency") or "",
            "our_price_from": min(totals) if totals else None, "seller_url": _seller_link(row["source_url"]),
            "created_at": row.get("created_at")}


def view(row: dict, can_manage: bool = False) -> dict:
    """Everything the design's page shows: the original as read, and our version."""
    return {**card(row), "original": row["original"], "ours": row["ours"],
            "pictures": [{"url": picture_url(p), "width": p["width"], "height": p["height"]}
                         for p in row.get("pictures") or []],
            "purities": [{k: p[k] for k in ("key", "label", "fineness")} for p in purchase.PURITIES],
            "making_modes": orders.MAKING_MODES, "can_manage": can_manage, "status": row.get("status")}


def specs_for_buy(ours: dict) -> dict:
    """Our version in the buy page's shape (purchase.details)."""
    st = ours.get("stones") or []
    stones = {"source": "ours", "counted": True, "groups": st, "count": sum(s["count"] for s in st),
              "carat": round(sum(s["carat"] for s in st), 3)}
    return {
        "source": "ours", "design_type": None,
        "gold": {"source": "ours", "by_purity": dict(ours["gold_g"])},
        "stones": stones,
        "ring_size": purchase.ring_size(indian=ours["ring_size_in"]) if ours.get("ring_size_in") else None,
        "dimensions": ours.get("dimensions") or [],
        "purities": [{k: p[k] for k in ("key", "label", "fineness")} for p in purchase.PURITIES],
        "purities_for": {m: [p for p in purchase.PURITIES_FOR[m] if p in ours["purities"]] for m in ours["metals"]},
        "default_purity": "18k" if "18k" in ours["purities"] else ours["purities"][0],
        "diamond_quality": ours.get("diamond_quality", ""),
    }


def buy_view(row: dict) -> dict:
    """A brand design in the shape of /api/buy/{uid}, so the buy page works the same."""
    ours, o = row["ours"], row["original"]
    views = [{"thumb": picture_url(p), "full": picture_url(p)} for p in row.get("pictures") or []]
    return {
        "uid": None, "brand_id": row["id"], "design_id": ours.get("design_no") or ours["name"], "name": ours["name"],
        "category": ours["category"], "metals": ours["metals"], "metal_shown": ours["metals"][0],
        "views": {m: views for m in ours["metals"]}, "videos": {}, "tags": [f"Based on a {o.get('brand') or row['site']} design"],
        "stone_shape": None, "specs": specs_for_buy(ours),
        "prices": {p: v["total"] for p, v in ours.get("prices", {}).items()},
        "brand": {"id": row["id"], "site": row["site"], "brand": o.get("brand") or row["site"],
                  "title": o.get("title", ""), "seller_url": _seller_link(row["source_url"])},
    }


def build_order(user: dict, row: dict, *, metal: str, purity: str, ring_size: int | None, quantity: int,
                phone: str, note: str) -> dict:
    """An order for our version of a brand design: the usual order (checked the same
    way), with the listing it came from and our price frozen in it."""
    ours, o = row["ours"], row["original"]
    b = buy_view(row)
    m = {"design_id": b["design_id"][:200], "folders": [f"brand:{row['id']}"], "category": ours["category"],
         "metals": ours["metals"], "stone_shape": None}
    first = b["views"][ours["metals"][0]]
    detail = {"uid": row["id"], "views": b["views"], "thumb": first[0]["thumb"] if first else None,
              "thumbs_by_metal": {}}
    out = orders.build_order(user, m, detail, b["specs"], metal=metal, purity=purity, ring_size=ring_size,
                             quantity=quantity, phone=phone, note=note)
    price = ours["prices"][purity]
    out["kind"] = "brand"
    out["design_uid"] = row["id"]
    out["design_key"] = f"brand:{row['id']}"
    out["snapshot"]["gold_source"] = "ours"
    out["snapshot"]["brand"] = {
        "id": row["id"], "name": ours["name"], "site": row["site"], "brand": o.get("brand") or row["site"],
        "title": o.get("title", ""), "seller_url": _seller_link(row["source_url"]),
        "seller_price": o.get("price"), "currency": o.get("currency") or "",
        "listed_price": price["total"], "quote": quote_for(ours, purity),
    }
    return out
