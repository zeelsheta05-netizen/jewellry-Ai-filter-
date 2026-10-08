"""Search by a web link: a shopper pastes a product page (or an image link),
the jewellery picture is found on it and matched against the collection
exactly like a photo the shopper uploaded (SearchEngine.read_photo).

1. Fetch the page safely (below). A link straight to an image is used as is.
2. Collect the page's pictures, best guesses first: the product image of its
   structured data (JSON-LD), Open Graph / Twitter images, itemprop="image",
   <link rel="image_src">, then <img> / <picture> elements (lazy-loading
   attributes and srcset included), skipping logos, icons and tracking pixels.
3. Download the top few, read them like an upload (size and format checks),
   and let the image model choose: the picture that looks most like jewellery
   and most like a design of the collection.
4. No usable picture (the site blocks visits, builds its pictures with
   JavaScript, or shows none): read the jewellery words of the shopper's text,
   the product's name, the page title and the link itself
   ("/rose-gold-solitaire-ring-123") and run a normal search with them.
5. Nothing at all: a clear message asking for a photo or a description.

Fetching a link the shopper chose is a way into this Mac's network (SSRF), so:
only http(s) on ports 80 / 443, no user:password in links; every host is
resolved and must be a public address, on every redirect (at most 4); the
connection goes to the address that was checked (no DNS rebinding) with the
certificate checked for the real name; pages are cut at 3 MB and pictures at
8 MB after decompression; short time limits; nothing is written to disk.
Pictures are never shown to the shopper from the other site: the page gets a
copy made here.
"""
import base64
import io
import ipaddress
import json
import re
import socket
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass, field
from html.parser import HTMLParser
from urllib.parse import urljoin, urlsplit, urlunsplit

import httpx

from . import photo
from .domain import understood
from .query import parse

MAX_HTML = 3 * 1024 * 1024
MAX_IMAGE = photo.MAX_BYTES
MAX_REDIRECTS = 4
DNS_TIMEOUT = 5.0
TIMEOUT = httpx.Timeout(8.0, connect=5.0)
BUDGET = 25.0             # seconds for the whole link, all fetches included
MAX_PICTURES = 8          # pictures downloaded and compared per page
MIN_SIDE = 150            # smaller pictures are icons and thumbnails
JEWEL_MIN = 0.5           # the image model's "this is jewellery" reading a picture needs...
# ...and on a page whose words (and the shopper's) never mention jewellery, a product photo's:
# measured on shop pages 1.00 / 0.79-0.90; news, sari, watch pages with jewellery worn in the
# photo at most 0.69 close (a pearl stud in a BBC news photo: 0.92 / 0.65)
STRONG_JEWEL, STRONG_CLOSEST = 0.9, 0.75
PORTS = {"http": 80, "https": 443}
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) "
      "Chrome/126.0 Safari/537.36 DesignFinder/1.0")
URL_IN_TEXT = re.compile(r"(?:https?://|www\.)[^\s<>\"']+", re.I)
SKIP = re.compile(r"logo|icon|sprite|avatar|favicon|banner|badge|payment|flag|placeholder|loader|spinner|blank|"
                  r"pixel|tracking|facebook|twitter|instagram|whatsapp|pinterest|youtube|linkedin|arrow|rating|"
                  r"emoji|captcha|qr[-_]?code|\.svg(\?|$)", re.I)
JEWEL_WORDS = re.compile(r"ring|earring|stud|hoop|jhumk|necklace|pendant|locket|chain|bracelet|bangle|kada|"
                         r"jewel|diamond|solitaire|gold|platinum|silver|mangalsutra|product|zoom|large", re.I)
# words of a request that say "like this", not what the piece should be
FILLER = re.compile(r"\b(?:similar(?: to)?|same|matching|look ?alikes?|alternatives?|options?|like this|this one|"
                    r"this|that|link|url|website|page|product|item|find|show|me|designs?)\b", re.I)


_RESOLVER = ThreadPoolExecutor(4, thread_name_prefix="link-dns")


class LinkError(Exception):
    def __init__(self, message: str, short: str = ""):
        super().__init__(message)
        self.message = message           # a sentence for the shopper
        self.short = short or message    # for "no photo could be used (…)"


@dataclass
class Fetched:
    url: str
    ctype: str
    data: bytes


@dataclass
class Picture:
    url: str
    how: str        # json-ld | og:image | twitter:image | itemprop | image_src | img | link
    rank: float     # higher = more likely the product picture, before looking at it
    alt: str = ""


@dataclass
class Page:
    url: str
    title: str = ""
    names: list = field(default_factory=list)       # product names / descriptions from metadata
    pictures: list = field(default_factory=list)


# ---- fetching safely ------------------------------------------------------------

def normalise(text: str) -> str:
    """The link in what was pasted, with https:// added to a bare "www."."""
    m = URL_IN_TEXT.search(text or "")
    if not m:
        raise LinkError("There's no web link in the text.")
    url = m.group().rstrip(".,;:!?)]}'\"")
    return url if re.match(r"https?://", url, re.I) else "https://" + url


def split_link(text: str) -> tuple[str, str]:
    """Pasted text -> (link, the shopper's own words around it)."""
    url = normalise(text)
    words = re.sub(r"[()\[\]{}<>\"“”]", " ", URL_IN_TEXT.sub(" ", text))
    return url, " ".join(FILLER.sub(" ", words).split())


def _check_url(url: str):
    p = urlsplit(url)
    if p.scheme.lower() not in PORTS or not p.hostname:
        raise LinkError("Only web links (http or https) can be read.", "not a web link")
    if p.username or p.password:
        raise LinkError("Links with a user name or password can't be read.", "the link has a password")
    port = p.port or PORTS[p.scheme.lower()]
    if port != PORTS[p.scheme.lower()]:
        raise LinkError("Only links on the usual web ports can be read.", "unusual web port")
    return p, port


def public_address(host: str, port: int) -> str:
    """The host's address, if every address it has is on the public internet."""
    # the system resolver has no time limit of its own: an unknown name can take 30 s
    future = _RESOLVER.submit(socket.getaddrinfo, host, port, type=socket.SOCK_STREAM)
    try:
        infos = future.result(timeout=DNS_TIMEOUT)
    except FutureTimeout:
        raise LinkError("The website's address couldn't be found in time.", "the website couldn't be found")
    except (socket.gaierror, UnicodeError, OSError):
        raise LinkError("That website couldn't be found.", "the website couldn't be found")
    addrs = []
    for *_, sockaddr in infos:
        ip = ipaddress.ip_address(sockaddr[0].split("%")[0])
        if ip.version == 6 and ip.ipv4_mapped:
            ip = ip.ipv4_mapped
        if not ip.is_global or ip.is_multicast:
            raise LinkError("That link points inside a private network, so it can't be read.", "private network")
        addrs.append(str(ip))
    if not addrs:
        raise LinkError("That website couldn't be found.", "the website couldn't be found")
    return addrs[0]


def fetch(url: str, accept: str, max_bytes: int, deadline: float) -> Fetched:
    for _ in range(MAX_REDIRECTS + 1):
        p, port = _check_url(url)
        host = p.hostname
        ip = public_address(host, port)
        netloc = f"[{ip}]" if ":" in ip else ip
        target = urlunsplit((p.scheme.lower(), netloc, p.path or "/", p.query, ""))
        left = deadline - time.monotonic()
        if left <= 0:
            raise LinkError("The website took too long to answer.", "the website was too slow")
        timeout = httpx.Timeout(min(TIMEOUT.read, left), connect=min(TIMEOUT.connect, left))
        headers = {"Host": host, "User-Agent": UA, "Accept": accept, "Accept-Language": "en-IN,en;q=0.9,hi;q=0.6"}
        try:
            # a new client per request: no pooled connection can skip the address and certificate checks
            with httpx.Client(timeout=timeout, follow_redirects=False) as c, \
                    c.stream("GET", target, headers=headers, extensions={"sni_hostname": host}) as r:
                if r.status_code in (301, 302, 303, 307, 308) and r.headers.get("location"):
                    url = urljoin(url, r.headers["location"])
                    continue
                if r.status_code >= 400:
                    raise LinkError(f"The website refused to open the page (error {r.status_code}). "
                                    "Some shops block automatic visits.", "the shop's website blocks automatic visits")
                data = bytearray()
                for chunk in r.iter_bytes():   # decoded: a compressed bomb is cut off here too
                    data += chunk
                    if len(data) > max_bytes:
                        raise LinkError("The page is too large to read.", "the page is too large")
                    if time.monotonic() > deadline:
                        raise LinkError("The website took too long to answer.", "the website was too slow")
                return Fetched(url, r.headers.get("content-type", "").split(";")[0].strip().lower(), bytes(data))
        except httpx.TimeoutException:
            raise LinkError("The website took too long to answer.", "the website was too slow")
        except httpx.HTTPError:
            raise LinkError("The website couldn't be reached.", "the website couldn't be reached")
    raise LinkError("The link redirects too many times.", "too many redirects")


# ---- reading the page -------------------------------------------------------------

IMG_ATTRS = ("data-zoom-image", "data-large_image", "data-large-image", "data-src", "data-original",
             "data-lazy-src", "data-image", "src")
SRCSET_ATTRS = ("data-srcset", "srcset", "data-lazy-srcset")


def _largest(srcset: str) -> str | None:
    best, best_w = None, -1.0
    for part in srcset.split(","):
        bits = part.strip().split()
        if not bits:
            continue
        w = 1.0
        if len(bits) > 1:
            m = re.match(r"([\d.]+)([wx])", bits[1])
            if m:
                w = float(m.group(1)) * (1 if m.group(2) == "w" else 1000)
        if w > best_w:
            best, best_w = bits[0], w
    return best


class _Reader(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.meta, self.links, self.imgs, self.ld = [], [], [], []
        self.title, self._in_title, self._in_ld, self._buf = "", False, False, []

    def handle_starttag(self, tag, attrs):
        a = {k.lower(): (v or "") for k, v in attrs}
        if tag == "meta":
            key = (a.get("property") or a.get("name") or a.get("itemprop") or "").lower()
            if key and a.get("content"):
                self.meta.append((key, a["content"].strip()))
        elif tag == "link":
            rel, href = a.get("rel", "").lower(), a.get("href", "")
            if href and ("image_src" in rel or (rel == "preload" and a.get("as") == "image")):
                self.links.append(("image_src" if "image_src" in rel else "preload", href))
        elif tag in ("img", "source") and len(self.imgs) < 400:
            self.imgs.append(a)
        elif tag == "title" and not self.title:   # the page's title, not an SVG icon's
            self._in_title = True
        elif tag == "script" and "ld+json" in a.get("type", "").lower():
            self._in_ld, self._buf = True, []

    def handle_endtag(self, tag):
        if tag == "title":
            self._in_title = False
        elif tag == "script" and self._in_ld:
            self._in_ld = False
            self.ld.append("".join(self._buf))

    def handle_data(self, data):
        if self._in_title and len(self.title) < 300:
            self.title += data
        elif self._in_ld:
            self._buf.append(data)


def _ld_products(node, out):
    """Product nodes of JSON-LD, however they are nested (@graph, lists, offers...)."""
    if isinstance(node, list):
        for n in node:
            _ld_products(n, out)
    elif isinstance(node, dict):
        t = node.get("@type")
        types = t if isinstance(t, list) else [t]
        if any(str(x).lower() in ("product", "productgroup", "individualproduct") for x in types):
            out.append(node)
        for v in node.values():
            if isinstance(v, (dict, list)):
                _ld_products(v, out)


def _ld_images(value) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [value.get("contentUrl") or value.get("url") or ""]
    if isinstance(value, list):
        return [u for v in value for u in _ld_images(v)]
    return []


def read_page(html: str, base: str) -> Page:
    r = _Reader()
    try:
        r.feed(html)
        r.close()       # whatever is still buffered (a page cut off at the size limit)
    except Exception:   # broken markup: keep what was read
        pass
    page = Page(url=base, title=" ".join(r.title.split()))
    seen = set()

    def add(src, how, rank, alt=""):
        if not src or src.startswith("data:"):
            return
        u = urljoin(base, src.strip())
        if not u.lower().startswith(("http://", "https://")) or u in seen or SKIP.search(u):
            return
        seen.add(u)
        page.pictures.append(Picture(u, how, rank, alt))

    for raw in r.ld:
        try:
            data = json.loads(raw)
        except ValueError:
            continue
        products = []
        _ld_products(data, products)
        for prod in products:
            for k in ("name", "description"):
                if isinstance(prod.get(k), str):
                    page.names.append(prod[k][:300])
            for u in _ld_images(prod.get("image")):
                add(u, "json-ld", 6)
    rank = {"og:image:secure_url": 5, "og:image": 5, "og:image:url": 5, "twitter:image": 4, "twitter:image:src": 4,
            "image": 4, "thumbnailurl": 3}
    for key, value in r.meta:
        if key in rank:
            add(value, "og:image" if key.startswith("og:") else "twitter:image" if key.startswith("twitter") else "itemprop",
                rank[key])
        elif key in ("og:title", "twitter:title", "og:description", "description", "product:category"):
            page.names.append(value[:300])
    for kind, href in r.links:
        add(href, "image_src" if kind == "image_src" else "link", 3 if kind == "image_src" else 1)
    title_words = {w for w in re.findall(r"[a-z]{4,}", page.title.lower())}
    for a in r.imgs:
        src = next((a[k] for k in IMG_ATTRS if a.get(k)), None)
        big = next((_largest(a[k]) for k in SRCSET_ATTRS if a.get(k)), None)
        alt = " ".join((a.get("alt") or a.get("title") or "").split())[:200]
        hay = f"{alt} {src or ''} {a.get('class', '')} {a.get('id', '')}".lower()
        score = 0.0
        if JEWEL_WORDS.search(hay):
            score += 1.0
        if title_words & set(re.findall(r"[a-z]{4,}", alt.lower())):
            score += 1.0
        try:
            w, h = int(a.get("width", "0") or 0), int(a.get("height", "0") or 0)
        except ValueError:
            w = h = 0
        if 0 < min(w or 9999, h or 9999) < 100:
            continue   # a declared small picture: an icon
        if max(w, h) >= 400:
            score += 1.0
        if a.get("itemprop", "").lower() == "image":
            score += 2.0
        add(big or src, "img", score, alt)
    page.pictures.sort(key=lambda p: -p.rank)   # stable: page order within a rank
    return page


# ---- the search -------------------------------------------------------------------

def _slug_words(url: str) -> str:
    p = urlsplit(url)
    words = re.split(r"[^a-z]+", (p.path + " " + p.query).lower())
    return " ".join(w for w in words if len(w) > 1)


def text_intent(url: str, page: Page | None, words: str, domain=None) -> str | None:
    """A search from words: the shopper's own, then the product name, the page title
    and the link, keeping only what the search understands about jewellery. Wording
    that isn't about jewellery as a whole ("Boxing ring - Wikipedia") is skipped."""
    sources = [words] + ((page.names[:4] + [page.title]) if page else []) + [_slug_words(url)]
    terms, seen = [], set()
    for text in sources:
        text = " ".join(re.split(r"\s[|\-–—:]\s", text or "")[:1]) if text is not words else text   # "Name | Shop"
        if not text or (domain and text is not words and not domain.check(text[:200]).ok):
            continue
        for term, kind, value, neg in parse(text).terms:
            # a type may be named twice in different words ("diamond stud earrings"), once per wording
            key = (kind, value, term.lower().rstrip("s")) if kind == "category" else (kind, value)
            if not neg and term != "bigstone" and key not in seen:
                seen.add(key)
                terms.append(term)
    prompt = " ".join(([words] if words and understood(parse(words)) else [])
                      + [t for t in terms if t.lower() not in (words or "").lower()])
    return prompt if prompt and understood(parse(prompt)) else None


def thumbnail(im) -> str:
    """The picture as the page shows it (a copy made here, never the other site's link)."""
    im = im.copy()
    im.thumbnail((photo.WORK_SIDE, photo.WORK_SIDE))
    buf = io.BytesIO()
    im.convert("RGB").save(buf, "JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def download_pictures(page: Page, deadline: float) -> list:
    """The likeliest pictures of the page, downloaded -> [(Picture, image)]. Done before
    any model runs: a model may wait for the GPU behind other shoppers' searches, and
    that wait must not use up the time the downloads have."""
    todo = page.pictures[:MAX_PICTURES]
    if not todo:
        return []

    def get(pic):
        try:
            f = fetch(pic.url, "image/avif,image/webp,image/png,image/jpeg,image/*;q=0.8", MAX_IMAGE, deadline)
            im = photo.read(f.data)
            return (pic, im) if min(im.size) >= MIN_SIDE else None
        except (LinkError, photo.PhotoError):
            return None
    with ThreadPoolExecutor(4) as pool:
        return [g for g in pool.map(get, todo) if g]


def read_pictures(engine, got: list) -> list:
    """[(Picture, image)] -> [(Picture, image, the image model's reading)]"""
    return [(pic, im, r) for (pic, im), r in zip(got, engine.picture_readings([im for _, im in got]))] if got else []


def pick_picture(read: list, about_jewellery: bool):
    """Of the read pictures, the one the image model finds most like jewellery and
    like the collection's designs. On a page whose words don't mention jewellery the
    picture must look like a product photo (STRONG_*). -> (Picture, image, reading) or None."""
    best = None
    for pic, im, r in read:
        if r["jewellery"] < JEWEL_MIN:
            continue
        if not about_jewellery and (r["jewellery"] < STRONG_JEWEL or r["closest"] < STRONG_CLOSEST):
            continue
        # how jewellery-like and how close to the collection, plus a little for where the page put it
        score = r["jewellery"] + 2.0 * r["closest"] + 0.05 * pic.rank
        if best is None or score > best[0]:
            best = (score, pic, im, r)
    return best[1:] if best else None


def search(engine, text: str, category=None, metal=None) -> dict:
    """Pasted text with a link -> a result for the page. The caller keeps
    result["_photo"] (the read picture) for filters and "show more"."""
    url, words = split_link(text)
    deadline = time.monotonic() + BUDGET
    source = {"url": url, "site": (urlsplit(url).hostname or "").removeprefix("www.")}
    page, problem = None, None
    try:
        f = fetch(url, "text/html,application/xhtml+xml,image/*;q=0.9,*/*;q=0.5", MAX_HTML, deadline)
        source["url"] = f.url
        if f.ctype.startswith("image/"):          # a link straight to a picture
            page = Page(url=f.url, pictures=[Picture(f.url, "link", 10)])
        elif "html" in f.ctype or f.data[:200].lstrip().lower().startswith((b"<!doctype html", b"<html")):
            page = read_page(f.data.decode("utf-8", "replace"), f.url)
            source["title"] = page.title[:200]
        else:
            problem = "isn't a web page or a picture"
    except LinkError as e:
        problem = e
    read = read_pictures(engine, download_pictures(page, deadline) if page else [])
    # a clear product photo needs no reading of the page's words, the slow part (the
    # jewellery judge, one GPU run per name, title and link): 25 of 32 s on a shop page
    found, prompt = pick_picture(read, about_jewellery=False), None
    if not found:
        prompt = text_intent(source["url"], page, words, engine.domain)
        found = pick_picture(read, about_jewellery=prompt is not None)
    if found:
        pic, im, reading = found
        pq = engine.read_photo(im)
        res = engine.search_photo(pq, words, category=category, metal=metal)
        where = source.get("title") or source["site"]
        res["notes"] = [f"Matched to the picture on {source['site']}" + (f": “{where[:80]}”" if where != source["site"] else "")
                        + "."] + res["notes"]
        source.update({"picture": pic.url, "how": pic.how})
        return {**res, "mode": "image", "image": thumbnail(im), "source": source, "words": words, "_photo": pq}
    if prompt:
        res = engine.search(prompt, category=category, metal=metal)
        if not res.get("refused"):
            why = problem.short if isinstance(problem, LinkError) else problem or "no jewellery photo on the page"
            res["notes"] = [f"No photo could be used from that link ({why}), so designs are matched to its words: "
                            f"“{prompt}”."] + res["notes"]
            return {**res, "mode": "text", "prompt": prompt, "source": source}
    reason = (problem.message if isinstance(problem, LinkError)
              else f"The link {problem}." if problem else "No jewellery photo or jewellery words were found on that page.")
    return {"mode": "failed", "source": source, "results": [], "notes": [], "matches_in_filter": 0,
            "failed": {"title": "Couldn't find a jewellery design in that link",
                       "hint": f"{reason} Add a photo of the piece instead (a screenshot works), or describe it, "
                               "for example “rose gold solitaire ring”."}}
