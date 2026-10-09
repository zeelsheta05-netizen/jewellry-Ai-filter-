"""Social post captions (jewelsearch/social.py): trends calendar, facts, SEO keywords,
hashtags, platform limits, invented-claim filter, writers' fallback, jobs and routes."""
import base64
import datetime as dt
import io
import time
from types import SimpleNamespace

import pytest
from PIL import Image, ImageDraw

from jewelsearch import social as S
from jewelsearch.sketch import SketchError

OCT9 = dt.date(2026, 10, 9)


def ring_brief(**kw):
    base = dict(details="18k rose gold, 0.5 ct diamonds, BIS hallmarked", brand="Shree Jewels", city="Surat",
                tone="festive", today=OCT9)
    base.update(kw)
    b = S.make_brief(kw.pop("platforms", list(S.PLATFORMS)), **{k: v for k, v in base.items() if k != "platforms"})
    b.facts = S.read_facts(None, None, b.facts.details + " ring")
    return b


def jpeg() -> bytes:
    im = Image.new("RGB", (200, 200), "white")
    ImageDraw.Draw(im).ellipse((50, 50, 150, 150), outline=(200, 150, 60), width=12)
    buf = io.BytesIO()
    im.save(buf, "JPEG")
    return buf.getvalue()


class FakeEngine:
    """read_photo -> Design DNA like the real engine's."""
    def read_photo(self, _im):
        return SimpleNamespace(dna={"type": {"value": "ring", "sure": True}, "metal": {"metal": "yellow_gold"},
                                    "traits": [{"group": "Stones", "label": "Solitaire"}, {"group": "Form", "label": "Thin band"}],
                                    "motifs": [{"label": "Floral"}]})


# ---------- trends ----------

def test_trending_by_date_and_market():
    assert [o.key for o in S.trending(OCT9)][:2] == ["dhanteras", "diwali"]   # biggest jewellery days first
    assert "karwachauth" in [o.key for o in S.trending(OCT9)]
    assert S.trending(OCT9, "global") == []                                  # Indian festivals stay in India
    assert [o.key for o in S.trending(dt.date(2026, 12, 20), "global")] == ["christmas"]
    assert "wedding26" in [o.key for o in S.trending(dt.date(2027, 1, 10))]
    # fixed-date holidays come back every year
    assert "valentine" in [o.key for o in S.trending(dt.date(2029, 2, 1), "global")]
    assert S.trending(dt.date(2027, 7, 1)) == []


# ---------- facts, keywords, hashtags ----------

def test_facts_from_the_picture_and_user_words_win():
    f = S.read_facts(FakeEngine(), jpeg())
    assert (f.kind, f.metal, f.stone_style, f.keyword()) == ("ring", "yellow gold", "solitaire", "yellow gold solitaire ring")
    assert "thin band" in f.traits and "floral" in f.traits
    f = S.read_facts(FakeEngine(), jpeg(), "Rose gold emerald earrings")
    assert f.keyword() == "rose gold emerald earrings"


def test_keywords_and_hashtags():
    b = ring_brief()
    keys = S.keywords(b.facts, b.occasions(), b.city)
    assert keys[0] == "rose gold diamond ring" and "dhanteras gold buying" in keys and "jewellery shop in surat" in keys
    tags = S.hashtags(b.facts, b.occasions(), b.brand, b.city)
    assert tags[:3] == ["#ShreeJewels", "#RoseGoldDiamondRing", "#DiamondRing"] and "#Dhanteras" in tags
    assert len({t.lower() for t in tags}) == len(tags)
    g = S.keywords(b.facts, [], "", "global")
    assert "fine jewelry" in g and not any("jewellery" in k for k in g)


# ---------- claims ----------

def test_invented_claims_are_dropped_but_users_own_details_stay():
    details = "18k rose gold, 0.5 ct diamonds, bis hallmarked"
    text, dropped = S.scrub("Now 20% off! Certified by GIA. Set with 0.5 ct diamonds and BIS hallmark. "
                            "A 2 ct centre stone. Handcrafted by artisans. Show it off tonight.", details)
    assert "0.5 ct diamonds" in text and "Show it off tonight." in text
    assert {"Now 20% off!", "Certified by GIA.", "A 2 ct centre stone.", "Handcrafted by artisans."} == set(dropped)
    assert S.scrub("Free delivery across India.", "")[0] == ""
    assert S.scrub("Made by our skilled artisans. A smart investment. The festive season returns.", "")[0] \
        == "The festive season returns."
    assert S.scrub("Free delivery across India.", "free delivery")[0] == "Free delivery across India."


# ---------- writing ----------

def test_every_platform_fits_its_rules_with_the_template():
    b = ring_brief()
    for key, p in S.PLATFORMS.items():
        post = S.write_post(b, key, None, [])
        assert post["writer"] == "template" and post["chars"] <= p.limit
        assert post["keyword_in_start"], key
        assert len(post["hashtags"]) == p.tags and all(t in post["text"] for t in post["hashtags"])
        assert "DM us to order" in post["text"]
        if p.title:
            assert post["title"] and len(post["title"]) <= p.title
    assert S.write_post(b, "google", None, [])["hashtags"] == []
    post = S.write_post(b, "pinterest", None, [("ai", lambda *a: "Fine jewelry: a rose gold diamond ring with a soft glow for you.")])
    assert "jewellery" in post["text"] and "jewelry" not in post["text"]
    assert post["title"] == "Rose gold diamond ring for Dhanteras"


def test_writer_answer_is_cleaned_and_limited():
    b = ring_brief(platforms=["x"])
    long = "Here's the caption: " + "This rose gold diamond ring shines. " * 20 + "#ring #gold #love"
    post = S.write_post(b, "x", None, [("ai", lambda pr, pic, n: long)])
    assert post["writer"] == "ai" and post["chars"] <= 270 and "#love" not in post["text"]
    assert not post["text"].lower().startswith("here")
    assert post["text"].endswith("#ShreeJewels #RoseGoldDiamondRing")


def test_broken_writer_falls_back():
    b = ring_brief(platforms=["instagram"])

    def away(*a):
        raise SketchError("away", 503)
    assert S.write_post(b, "instagram", None, [("ai", away), ("local", lambda *a: "ok")])["writer"] == "template"
    assert S.write_post(b, "instagram", None, [("ai", away), ("local", lambda *a: "A rose gold diamond ring that glows "
                                                                              "for every festive evening.")])["writer"] == "local"


def test_title_platform_and_prompt():
    b = ring_brief()
    post = S.write_post(b, "pinterest", None, [("ai", lambda *a: "TITLE: Rose Gold Diamond Ring for Diwali\n"
                                                                  "TEXT: A festive rose gold diamond ring with a soft glow.")])
    assert post["title"] == "Rose Gold Diamond Ring for Diwali" and post["text"].startswith("A festive rose gold")
    pr = S.prompt_for(b, S.PLATFORMS["pinterest"], S.keywords(b.facts, b.occasions(), b.city))
    assert '"rose gold diamond ring"' in pr and "Dhanteras" in pr and "TITLE:" in pr and "Do not invent prices" in pr


# ---------- jobs ----------

def test_job_writes_each_platform_and_rewrites():
    calls = []

    def writer(prompt, pic, n):
        calls.append(prompt)
        return f"A yellow gold solitaire ring for festive evenings, version {len(calls)} with a fine thin band."
    jobs = S.Jobs(FakeEngine(), writer=writer)
    j = jobs.start("u1", jpeg(), S.make_brief(["instagram", "x"], tone="elegant", today=OCT9))
    for _ in range(100):
        j = jobs.get(j["id"], "u1")
        if j["status"] != "running":
            break
        time.sleep(0.02)
    assert j["status"] == "done" and set(j["posts"]) == {"instagram", "x"}
    assert j["facts"]["keyword"] == "yellow gold solitaire ring" and j["trends"][0]["key"] == "dhanteras"
    assert jobs.get(j["id"], "someone-else") is None
    again = jobs.rewrite(j["id"], "u1", "x", 2)
    assert "version 3" in again["text"] and "different version" in calls[-1]
    with pytest.raises(SketchError):
        jobs.rewrite(j["id"], "u2", "x", 1)


def test_brief_needs_a_platform():
    with pytest.raises(SketchError):
        S.make_brief(["myspace"])


# ---------- routes ----------

@pytest.fixture
def client(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from jewelsearch import auth, server, sketch

    monkeypatch.setattr(auth, "read_session", lambda c: {"uid": "u-1", "name": "A", "email": "a@x.in"} if c == "ok" else None)

    async def yes(_uid):
        return True
    monkeypatch.setattr(auth, "is_approved", yes)
    monkeypatch.setattr(auth, "limiter", auth.RateLimiter())
    monkeypatch.setattr(server, "studio", sketch.Studio(tmp_path, caller=lambda *a: (jpeg(), "image/jpeg")))
    monkeypatch.setattr(server, "social_jobs", S.Jobs(FakeEngine(), writer=lambda *a: "A yellow gold solitaire ring "
                                                                                       "made for every festive evening."))
    c = TestClient(server.app)
    c.cookies.set(auth.COOKIE, "ok")
    return c


def test_routes(client):
    r = client.get("/social", headers={"sec-fetch-dest": "iframe"})
    assert r.status_code == 200 and "Social post" in r.text and r.headers["x-frame-options"] == "SAMEORIGIN"
    r = client.get("/social", headers={"sec-fetch-dest": "document"}, follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == "/ai?tab=social"
    assert "Social post" in client.get("/ai").text
    o = client.get("/api/social/options").json()
    assert {p["key"] for p in o["platforms"]} == set(S.PLATFORMS) and o["designs"] == []
    image = "data:image/jpeg;base64," + base64.b64encode(jpeg()).decode()
    assert client.post("/api/social/start", json={"platforms": ["instagram"]}).status_code == 400   # no picture
    assert client.post("/api/social/start", json={"image": image, "platforms": []}).status_code == 422
    j = client.post("/api/social/start", json={"image": image, "platforms": ["instagram", "linkedin"], "tone": "luxury"}).json()
    for _ in range(100):
        j = client.get(f"/api/social/job/{j['id']}").json()
        if j["status"] != "running":
            break
        time.sleep(0.02)
    assert j["status"] == "done" and j["posts"]["instagram"]["writer"] == "ai"
    r = client.post(f"/api/social/job/{j['id']}/rewrite", json={"platform": "linkedin", "variant": 1})
    assert r.status_code == 200 and r.json()["platform"] == "linkedin"
    assert client.get("/api/social/job/nope").status_code == 404
    assert client.post("/api/social/start", json={"gid": "missing", "platforms": ["x"]}).status_code == 404
