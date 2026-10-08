"""Search by a product link (jewelsearch/linksearch.py).

No test reaches the internet: name lookups and web requests are replaced.
Run: .venv/bin/python -m pytest tests/test_linksearch.py -q
"""
import io
import socket
import time

import httpx
import pytest

from jewelsearch import linksearch as L
from jewelsearch.config import INDEX

needs_index = pytest.mark.skipif(not (INDEX / "front.npy").exists(), reason="index not built")
ADDRS = {"shop.example": "93.184.216.34", "cdn.example": "93.184.216.35", "evil.example": "127.0.0.1",
         "lan.example": "192.168.1.10", "meta.example": "169.254.169.254", "v6loop.example": "::1",
         "mapped.example": "::ffff:10.0.0.1", "mixed.example": ["93.184.216.36", "10.0.0.5"]}


@pytest.fixture
def dns(monkeypatch):
    def fake(host, port, type=0, **kw):
        if host not in ADDRS:
            raise socket.gaierror("unknown")
        addrs = ADDRS[host] if isinstance(ADDRS[host], list) else [ADDRS[host]]
        return [(socket.AF_INET6 if ":" in a else socket.AF_INET, socket.SOCK_STREAM, 6, "", (a, port)) for a in addrs]
    monkeypatch.setattr(L.socket, "getaddrinfo", fake)


@pytest.fixture
def web(monkeypatch, dns):
    """Web requests answered by `routes`: (ip, path) -> httpx.Response. Records every request."""
    routes, seen = {}, []

    def handler(request):
        seen.append(request)
        key = (request.url.host, request.url.path)
        if key not in routes:
            return httpx.Response(404)
        return routes[key]() if callable(routes[key]) else routes[key]
    real = httpx.Client
    monkeypatch.setattr(L.httpx, "Client", lambda **kw: real(transport=httpx.MockTransport(handler), **kw))
    return routes, seen


def deadline():
    return time.monotonic() + 10


# ---- the link in the text ----------------------------------------------------------------

@pytest.mark.parametrize("text,url,words", [
    ("https://shop.example/ring-123", "https://shop.example/ring-123", ""),
    ("find similar designs like this https://shop.example/x in white gold", "https://shop.example/x", "in white gold"),
    ("www.shop.example/p/rose-gold-ring.", "https://www.shop.example/p/rose-gold-ring", ""),
    ("similar earrings (https://shop.example/e)", "https://shop.example/e", "earrings"),
])
def test_the_link_and_the_words_are_split(text, url, words):
    u, w = L.split_link(text)
    assert u == url and w == words


def test_text_without_a_link_is_refused():
    with pytest.raises(L.LinkError):
        L.split_link("file:///etc/passwd")


@pytest.mark.parametrize("url", ["ftp://shop.example/x", "https://user:pw@shop.example/", "https://shop.example:8443/",
                                 "http://shop.example:22/", "javascript:alert(1)"])
def test_only_plain_web_links(url):
    with pytest.raises(L.LinkError):
        L._check_url(url)


# ---- addresses ------------------------------------------------------------------------------

@pytest.mark.parametrize("host", ["evil.example", "lan.example", "meta.example", "v6loop.example", "mapped.example",
                                  "mixed.example", "nowhere.example"])
def test_private_or_unknown_addresses_are_refused(dns, host):
    with pytest.raises(L.LinkError):
        L.public_address(host, 443)


def test_public_address(dns):
    assert L.public_address("shop.example", 443) == "93.184.216.34"


def test_a_slow_name_lookup_gives_up(monkeypatch):
    monkeypatch.setattr(L, "DNS_TIMEOUT", 0.2)
    monkeypatch.setattr(L.socket, "getaddrinfo", lambda *a, **k: time.sleep(2))
    t = time.monotonic()
    with pytest.raises(L.LinkError):
        L.public_address("slow.example", 443)
    assert time.monotonic() - t < 1


# ---- fetching -------------------------------------------------------------------------------

def test_the_request_goes_to_the_checked_address_with_the_real_name(web):
    routes, seen = web
    routes[("93.184.216.34", "/p")] = httpx.Response(200, html="<title>x</title>")
    f = L.fetch("https://shop.example/p", "text/html", 1000, deadline())
    assert f.ctype == "text/html"
    r = seen[0]
    assert r.url.host == "93.184.216.34" and r.headers["host"] == "shop.example"
    assert r.extensions["sni_hostname"] == "shop.example"


def test_a_redirect_into_the_private_network_is_refused(web):
    routes, seen = web
    routes[("93.184.216.34", "/p")] = httpx.Response(302, headers={"location": "http://evil.example/admin"})
    with pytest.raises(L.LinkError, match="private network"):
        L.fetch("https://shop.example/p", "text/html", 1000, deadline())
    assert all(r.url.host != "127.0.0.1" for r in seen)


def test_too_many_redirects(web):
    routes, _ = web
    routes[("93.184.216.34", "/loop")] = httpx.Response(302, headers={"location": "/loop"})
    with pytest.raises(L.LinkError, match="redirects"):
        L.fetch("https://shop.example/loop", "text/html", 1000, deadline())


def test_a_large_answer_is_cut_off(web):
    routes, _ = web
    routes[("93.184.216.34", "/big")] = httpx.Response(200, content=b"x" * 5000)
    with pytest.raises(L.LinkError, match="too large"):
        L.fetch("https://shop.example/big", "text/html", 1000, deadline())


def test_a_refused_visit_says_so(web):
    routes, _ = web
    routes[("93.184.216.34", "/blocked")] = httpx.Response(403)
    with pytest.raises(L.LinkError) as e:
        L.fetch("https://shop.example/blocked", "text/html", 1000, deadline())
    assert "403" in e.value.message and "blocks automatic visits" in e.value.short


# ---- reading the page -------------------------------------------------------------------------

PAGE = """<html><head><title>Rose Gold Solitaire Ring | Shop</title>
<meta property="og:image" content="/img/og.jpg"><meta name="twitter:image" content="https://cdn.example/tw.jpg">
<meta property="og:title" content="Rose Gold Solitaire Ring">
<script type="application/ld+json">{"@context":"https://schema.org","@graph":[{"@type":"Organization","logo":"/logo.png"},
 {"@type":"Product","name":"Rose Gold Solitaire Ring","image":[{"@type":"ImageObject","url":"https://cdn.example/main.jpg"}]}]}</script>
</head><body><svg><title>Caret</title></svg>
<img src="/static/logo.png" alt="Shop"><img src="/i/tiny.jpg" width="20" height="20">
<img data-src="/i/lazy.jpg" alt="rose gold solitaire ring side view" width="800">
<picture><source srcset="/i/s-400.jpg 400w, /i/s-1200.jpg 1200w"></picture>
<img src="/i/footer-banner.jpg"></body></html>"""


def test_the_page_is_read_best_pictures_first():
    page = L.read_page(PAGE, "https://shop.example/p/ring")
    urls = [p.url for p in page.pictures]
    assert page.title == "Rose Gold Solitaire Ring | Shop"            # not the SVG icon's <title>
    assert urls[0] == "https://cdn.example/main.jpg" and page.pictures[0].how == "json-ld"
    assert urls[1] == "https://shop.example/img/og.jpg"
    assert "https://shop.example/i/lazy.jpg" in urls and "https://shop.example/i/s-1200.jpg" in urls
    assert not [u for u in urls if "logo" in u or "tiny" in u or "banner" in u]
    assert "Rose Gold Solitaire Ring" in page.names


def test_broken_markup_does_not_break_reading():
    page = L.read_page("<html><head><title>Ring</title><script type=application/ld+json>{bad json</script></head>"
                       "<body><div><img src='/a.jpg'><p>cut off here <script>var x = '",
                       "https://shop.example/")
    assert page.pictures and page.pictures[0].url == "https://shop.example/a.jpg"


class FakeDomain:
    def check(self, text):
        class V:
            ok = "boxing" not in text.lower()
        return V()


def test_words_from_the_page_and_the_link():
    page = L.Page("https://shop.example/x", title="Diamond Stud Earrings | Shop")
    assert L.text_intent("https://shop.example/p/123", page, "", FakeDomain()) == "diamond stud earrings"
    assert L.text_intent("https://blocked.example/rose-gold-solitaire-ring-9", None, "", FakeDomain()) == \
        "rose gold solitaire ring"
    assert L.text_intent("https://en.example/wiki/Boxing_ring", L.Page("x", title="Boxing ring - Wiki"), "",
                         FakeDomain()) is None
    assert L.text_intent("https://news.example/today", L.Page("x", title="Today's news"), "", FakeDomain()) is None
    assert L.text_intent("https://news.example/today", None, "earrings", FakeDomain()) == "earrings"


# ---- the whole search (index) -------------------------------------------------------------------

@pytest.fixture(scope="module")
def engine():
    from jewelsearch.search import SearchEngine
    return SearchEngine(judge=lambda t: "no" if "boxing" in t.lower() or "news" in t.lower() else "yes")


def jpeg(im):
    buf = io.BytesIO()
    im.convert("RGB").save(buf, "JPEG", quality=90)
    return buf.getvalue()


def ring_picture(engine):
    from PIL import Image

    from jewelsearch.config import CROPS, thumb_name
    m = next(m for m in engine.meta if m["category"] == "ring")
    views = m["images"][m["embed_metal"]]
    return jpeg(Image.open(CROPS / thumb_name(views[m["front_view"]])))


def noise_picture():
    import numpy as np
    from PIL import Image
    rng = np.random.default_rng(1)
    return jpeg(Image.fromarray(rng.integers(0, 255, (600, 800, 3), dtype=np.uint8)))


def serve(monkeypatch, pages):
    def fake(url, accept, max_bytes, deadline):
        if url not in pages:
            raise L.LinkError("The website refused to open the page (error 403).", "the shop's website blocks automatic visits")
        ctype, data = pages[url]
        return L.Fetched(url, ctype, data)
    monkeypatch.setattr(L, "fetch", fake)


@needs_index
def test_a_product_page_is_matched_by_its_picture(engine, monkeypatch):
    serve(monkeypatch, {"https://shop.example/p": ("text/html", b'<title>Our best piece</title>'
                                                   b'<meta property="og:image" content="https://cdn.example/a.jpg">'),
                        "https://cdn.example/a.jpg": ("image/jpeg", ring_picture(engine))})
    res = L.search(engine, "https://shop.example/p in rose gold")
    assert res["mode"] == "image" and res["source"]["how"] == "og:image"
    assert res["_photo"] is not None and res["image"].startswith("data:image/jpeg;base64,")
    assert len(res["results"]) == 8 and {c["category"] for c in res["results"]} == {"ring"}
    assert res["words"] == "in rose gold" and res["query"]["metal"] == "rose_gold"


@needs_index
def test_a_link_straight_to_a_picture(engine, monkeypatch):
    serve(monkeypatch, {"https://cdn.example/r.jpg": ("image/jpeg", ring_picture(engine))})
    assert L.search(engine, "https://cdn.example/r.jpg")["mode"] == "image"


@needs_index
def test_waiting_for_the_gpu_does_not_cost_the_picture(engine, monkeypatch):
    """Models wait for the GPU behind other shoppers' searches; under load the live app
    once ran out of time for the downloads and fell back to the link's words."""
    pages = {"https://shop.example/p": ("text/html", b'<title>Our best ring</title>'
                                        b'<meta property="og:image" content="https://cdn.example/a.jpg">'),
             "https://cdn.example/a.jpg": ("image/jpeg", ring_picture(engine))}

    def fake(url, accept, max_bytes, deadline):
        if time.monotonic() > deadline:
            raise L.LinkError("The website took too long to answer.", "the website was too slow")
        return L.Fetched(url, *pages[url])
    monkeypatch.setattr(L, "fetch", fake)
    monkeypatch.setattr(L, "BUDGET", 0.5)
    real = engine.picture_readings
    monkeypatch.setattr(engine, "picture_readings", lambda ims: time.sleep(1) or real(ims))
    assert L.search(engine, "https://shop.example/p")["mode"] == "image"


@needs_index
def test_a_clear_product_photo_needs_no_reading_of_the_words(engine, monkeypatch):
    """The page's words are read by the jewellery judge, one GPU run each: the slow part."""
    serve(monkeypatch, {"https://shop.example/p": ("text/html", b'<title>Our best piece</title>'
                                                   b'<meta property="og:image" content="https://cdn.example/a.jpg">'),
                        "https://cdn.example/a.jpg": ("image/jpeg", ring_picture(engine))})

    def never(*a, **k):
        raise AssertionError("the page's words were read")
    monkeypatch.setattr(L, "text_intent", never)
    assert L.search(engine, "https://shop.example/p")["mode"] == "image"


@needs_index
def test_a_blocked_shop_falls_back_to_the_links_words(engine, monkeypatch):
    serve(monkeypatch, {})
    res = L.search(engine, "https://www.shop.example/en/diamond-stud-earrings-14k-123")
    assert res["mode"] == "text" and res["prompt"] == "diamond stud earrings"
    assert {c["category"] for c in res["results"]} == {"earrings"}
    assert "blocks automatic visits" in res["notes"][0]


@needs_index
def test_a_page_that_is_not_about_jewellery_says_so(engine, monkeypatch):
    serve(monkeypatch, {"https://news.example/today": ("text/html", b'<title>Today</title><img src="/n.jpg" width="900">'),
                        "https://news.example/n.jpg": ("image/jpeg", noise_picture())})
    res = L.search(engine, "https://news.example/today")
    assert res["mode"] == "failed" and res["results"] == []
    assert "photo" in res["failed"]["hint"]
    assert res["failed"]["title"] == "Couldn't find a jewellery design in that link"


# ---- the API route --------------------------------------------------------------------------------

@pytest.fixture
def client(monkeypatch):
    from fastapi.testclient import TestClient

    from jewelsearch import auth, server
    monkeypatch.setattr(auth, "read_session", lambda c: {"uid": "u-1", "name": "A", "email": "a@x.in"} if c == "ok" else None)

    async def yes(uid, fresh=False):
        return True
    monkeypatch.setattr(auth, "is_approved", yes)
    monkeypatch.setattr(auth, "limiter", auth.RateLimiter())
    monkeypatch.setattr(server, "_photos", type(server._photos)())
    c = TestClient(server.app)
    c.cookies.set(auth.COOKIE, "ok")
    return c


def test_link_route(client, monkeypatch):
    from jewelsearch import server
    answers = iter([{"mode": "image", "results": [], "_photo": "PQ"}, {"mode": "failed", "results": []}])
    monkeypatch.setattr(L, "search", lambda engine, text, category=None, metal=None: next(answers))
    r = client.post("/api/link-search", json={"text": "https://shop.example/p"})
    assert r.status_code == 200 and r.json()["token"] and "_photo" not in r.json()
    assert server._photo_get("u-1", r.json()["token"]) == "PQ"            # filters and "show more" reuse it
    assert "token" not in client.post("/api/link-search", json={"text": "https://shop.example/q"}).json()

    def bad(engine, text, category=None, metal=None):
        raise L.LinkError("There's no web link in the text.")
    monkeypatch.setattr(L, "search", bad)
    assert client.post("/api/link-search", json={"text": "hello"}).status_code == 400
    assert client.post("/api/link-search", content="x", headers={"content-type": "text/plain"}).status_code == 415
