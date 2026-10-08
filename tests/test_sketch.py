"""Sketch to Design (jewelsearch/sketch.py): prompt, picture checks, cache, budget, Google call shape."""
import base64
import os
import io
import json

import httpx
import pytest
from PIL import Image, ImageDraw

from jewelsearch import sketch


def png(blank=False, size=(600, 600)) -> bytes:
    im = Image.new("RGBA", size, (255, 255, 255, 0))
    if not blank:
        d = ImageDraw.Draw(im)
        d.ellipse((150, 150, 450, 450), outline=(0, 0, 0, 255), width=8)
        d.polygon([(260, 90), (340, 90), (300, 160)], outline=(0, 0, 0, 255), width=6)
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return buf.getvalue()


def data_url(b: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(b).decode()


class FakeGoogle:
    def __init__(self):
        self.calls = []

    def __call__(self, model, prompt, picture):
        self.calls.append((model.key, prompt, picture))
        return png(), "image/png"


@pytest.fixture
def studio(tmp_path, monkeypatch):
    for k in ("SKETCH_PER_USER_DAY", "SKETCH_USD_DAY", "SKETCH_USD_MONTH"):
        monkeypatch.delenv(k, raising=False)
    return sketch.Studio(tmp_path, caller=FakeGoogle())


# ---------- prompt ----------

def test_prompt_is_short_and_follows_the_sketch():
    p = sketch.build_prompt("ring", "white", notes="rose gold, small round diamonds on the band")
    assert "photorealistic product photo of a ring" in p
    assert "Follow the sketch's outline" in p
    assert "18k rose gold" in p
    assert "white seamless studio" in p
    assert len(p.split()) < 130


def test_gujarati_notes_give_type_and_metal_without_any_ai_call():
    p = sketch.build_prompt("auto", "luxury", notes="રોઝ ગોલ્ડ વીંટી")
    assert "of a ring" in p and "18k rose gold" in p


def test_auto_type_without_words_lets_the_model_read_the_sketch():
    assert "the jewellery piece drawn" in sketch.build_prompt("auto", "velvet")


def test_words_only_prompt_has_no_sketch_lines():
    p = sketch.build_prompt("pendant", "luxury", notes="heart shaped", has_picture=False)
    assert "sketch" not in p.lower() and "pendant" in p


def test_sparkle_and_custom_background():
    p = sketch.build_prompt("ring", "custom", custom_bg="on marble   with rose petals", sparkle=False)
    assert "on marble with rose petals" in p and sketch.SPARKLE not in p
    with pytest.raises(sketch.SketchError):
        sketch.build_prompt("ring", "custom", custom_bg="  ")


def test_unknown_choices_refused():
    with pytest.raises(sketch.SketchError):
        sketch.build_prompt("crown", "luxury")
    with pytest.raises(sketch.SketchError):
        sketch.build_prompt("ring", "moon")


def test_notes_are_cut_to_a_short_length():
    p = sketch.build_prompt("ring", "white", notes="diamond " * 200)
    assert len(p) < 1000


# ---------- picture ----------

def test_prepare_shrinks_to_1024_jpeg_on_white():
    out = sketch.prepare(png(size=(3000, 2000)))
    im = Image.open(io.BytesIO(out))
    assert im.format == "JPEG" and max(im.size) == 1024 and im.mode == "RGB"
    assert im.getpixel((5, 5))[0] > 240   # transparent -> white paper, not black


def test_empty_canvas_is_refused_before_any_call():
    with pytest.raises(sketch.SketchError, match="empty"):
        sketch.prepare(png(blank=True))


def test_not_an_image_refused():
    with pytest.raises(sketch.SketchError):
        sketch.prepare(b"hello")
    with pytest.raises(sketch.SketchError):
        sketch.from_data_url("data:text/plain;base64,aGk=")


# ---------- cache, budget, gallery ----------

def test_same_request_twice_costs_once(studio):
    pic = sketch.prepare(png())
    a = studio.generate("u1", "lite-1k", "prompt", pic)
    b = studio.generate("u2", "lite-1k", "prompt", pic)
    assert len(studio.caller.calls) == 1
    assert a["cached"] is False and b["cached"] is True and b["source"] == "cache"
    assert "usd" not in a   # no prices go to the page
    assert studio.budget("u2")["left_today"] == studio.budget("u2")["per_user_day"]   # cache hits don't count
    again = studio.generate("u1", "lite-1k", "prompt", pic)
    assert again["id"] == a["id"] and again["cached"] and len(studio.mine("u1")) == 1   # no duplicate entry


def test_different_model_or_prompt_is_a_new_call(studio):
    pic = sketch.prepare(png())
    studio.generate("u1", "lite-1k", "prompt", pic)
    studio.generate("u1", "nb21-1k", "prompt", pic)
    studio.generate("u1", "lite-1k", "prompt 2", pic)
    assert len(studio.caller.calls) == 3


def test_daily_count_limit(studio, monkeypatch):
    monkeypatch.setenv("SKETCH_PER_USER_DAY", "2")
    pic = sketch.prepare(png())
    studio.generate("u1", "lite-1k", "a", pic)
    studio.generate("u1", "lite-1k", "b", pic)
    with pytest.raises(sketch.SketchError, match="daily limit"):
        studio.generate("u1", "lite-1k", "c", pic)
    assert len(studio.caller.calls) == 2
    studio.generate("u2", "lite-1k", "c", pic)   # someone else still can


def test_dollar_caps_checked_before_the_call(studio, monkeypatch):
    monkeypatch.setenv("SKETCH_USD_DAY", "0.05")
    pic = sketch.prepare(png())
    studio.generate("u1", "lite-1k", "a", pic)            # 0.034
    with pytest.raises(sketch.SketchError, match="Today"):
        studio.generate("u2", "lite-1k", "b", pic)        # would be 0.068
    monkeypatch.setenv("SKETCH_USD_DAY", "100")
    monkeypatch.setenv("SKETCH_USD_MONTH", "0.1")
    with pytest.raises(sketch.SketchError, match="month"):
        studio.generate("u2", "pro-2k", "b", pic)
    assert len(studio.caller.calls) == 1


def test_unknown_model_refused(studio):
    with pytest.raises(sketch.SketchError):
        studio.generate("u1", "dall-e", "a", None)


def test_refine_sends_the_last_picture_and_a_short_edit(studio):
    first = studio.generate("u1", "lite-1k", "a", sketch.prepare(png()))
    second = studio.refine("u1", first["id"], "make the band thinner", "lite-1k")
    model, prompt, picture = studio.caller.calls[-1]
    assert prompt.startswith("Edit this jewellery photo: make the band thinner.")
    assert Image.open(io.BytesIO(picture)).format == "JPEG"
    assert second["parent"] == first["id"]
    with pytest.raises(sketch.SketchError):
        studio.refine("u2", first["id"], "x", "lite-1k")   # not their design


def test_gallery_is_per_user_and_delete_hides(studio):
    a = studio.generate("u1", "lite-1k", "a", sketch.prepare(png()))
    assert [x["id"] for x in studio.mine("u1")] == [a["id"]]
    assert studio.mine("u2") == [] and studio.file(a["id"], "u2") is None
    assert studio.delete(a["id"], "u1") and studio.mine("u1") == []
    assert studio.budget("u1")["left_today"] == studio.budget("u1")["per_user_day"] - 1   # spend stays


# ---------- Google call ----------

def _transport(handler):
    real = httpx.Client

    def client(*a, **kw):
        kw["transport"] = httpx.MockTransport(handler)
        return real(*a, **kw)
    return client


def test_google_call_shape_and_parse(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k-test")
    seen = {}

    def handler(req):
        seen["url"], seen["key"], seen["body"] = str(req.url), req.headers["x-goog-api-key"], json.loads(req.content)
        img = base64.b64encode(png()).decode()
        return httpx.Response(200, json={"steps": [{"type": "model_output", "content": [{"type": "image", "mime_type": "image/png", "data": img}]}]})
    monkeypatch.setattr(sketch.httpx, "Client", _transport(handler))
    data, mime = sketch.call_google(sketch.BY_KEY["nb21-2k"], "p", b"\xff\xd8jpeg")
    assert seen["url"].endswith("/v1beta/interactions") and seen["key"] == "k-test"
    b = seen["body"]
    assert b["model"] == "gemini-nano-banana-2.1"
    assert b["response_format"] == {"type": "image", "mime_type": "image/png", "aspect_ratio": "1:1", "image_size": "2K"}
    assert [x["type"] for x in b["input"]] == ["text", "image"]
    assert Image.open(io.BytesIO(data)).format == "PNG" and mime == "image/png"


def test_google_falls_back_to_generate_content_on_404(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    urls = []

    def handler(req):
        urls.append(str(req.url))
        if req.url.path.endswith("/interactions"):
            return httpx.Response(404, json={"error": {"message": "not found"}})
        body = json.loads(req.content)
        assert body["generationConfig"]["imageConfig"]["imageSize"] == "1K"
        return httpx.Response(200, json={"candidates": [{"content": {"parts": [{"inlineData": {"mimeType": "image/png", "data": base64.b64encode(png()).decode()}}]}}]})
    monkeypatch.setattr(sketch.httpx, "Client", _transport(handler))
    sketch.call_google(sketch.BY_KEY["lite-1k"], "p", None)
    assert urls[-1].endswith("/models/gemini-3.1-flash-lite-image:generateContent")


@pytest.mark.parametrize("status,msg,match", [(429, "Resource exhausted", "limit"), (403, "API key not valid", "API key"),
                                              (400, "billing required", "not enabled")])
def test_google_errors_read_plainly(monkeypatch, status, msg, match):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.setattr(sketch.httpx, "Client", _transport(lambda req: httpx.Response(status, json={"error": {"message": msg}})))
    with pytest.raises(sketch.SketchError, match=match):
        sketch.call_google(sketch.BY_KEY["lite-1k"], "p", None)


def test_no_key_means_nothing_is_sent(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(sketch.httpx, "Client", None)   # would crash if used
    with pytest.raises(sketch.SketchError) as e:
        sketch.call_google(sketch.BY_KEY["lite-1k"], "p", None)
    assert e.value.status == 503


# ---------- routes ----------

@pytest.fixture
def client(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from fastapi.testclient import TestClient
    from jewelsearch import auth, server

    monkeypatch.setattr(auth, "read_session", lambda c: {"uid": "u-1", "name": "A", "email": "a@x.in"} if c == "ok" else None)

    async def yes(_uid):
        return True
    monkeypatch.setattr(auth, "is_approved", yes)
    monkeypatch.setattr(auth, "limiter", auth.RateLimiter())
    monkeypatch.setenv("POLLINATIONS_KEY", "sk_test")
    fake = FakeGoogle()
    monkeypatch.setattr(server, "studio", sketch.Studio(tmp_path, caller=fake))
    judge = SimpleNamespace(check=lambda t: SimpleNamespace(ok="ring" in t or "pendant" in t, text=t))
    monkeypatch.setattr(server, "engine", SimpleNamespace(domain=judge))
    c = TestClient(server.app)
    c.cookies.set(auth.COOKIE, "ok")
    c.fake = fake
    return c


def test_page_and_options_need_sign_in(client):
    from fastapi.testclient import TestClient
    from jewelsearch import server
    anon = TestClient(server.app)
    assert anon.get("/sketch", follow_redirects=False).status_code == 303
    assert anon.get("/api/sketch/options").status_code == 401
    r = client.get("/sketch")
    assert r.status_code == 200 and "Sketch to Design" in r.text
    assert "camera=(self)" in r.headers["permissions-policy"]


def test_options_list_only_connected_models_free_first(client, monkeypatch):
    o = client.get("/api/sketch/options").json()
    assert o["live"] and o["providers"] == ["Pollinations"] and o["default_model"] == "p-klein"
    assert all(m["provider"] == "pollinations" for m in o["models"])
    assert not any(k in m for m in o["models"] for k in ("usd", "inr", "free"))   # no prices or free/paid notes
    assert set(o["budget"]) == {"left_today", "per_user_day"}
    monkeypatch.setenv("GEMINI_API_KEY", "g")
    keys = [m["key"] for m in client.get("/api/sketch/options").json()["models"]]
    assert "nb21-2k" in keys and keys.index("p-kontext") < keys.index("lite-1k")


def test_nothing_connected_refuses_before_any_work(client, monkeypatch):
    monkeypatch.delenv("POLLINATIONS_KEY")
    o = client.get("/api/sketch/options").json()
    assert o["live"] is False and o["models"] == [] and o["default_model"] is None
    r = client.post("/api/sketch/generate", json={"image": data_url(png())})
    assert r.status_code == 503 and client.fake.calls == []


def test_model_of_unconnected_provider_refused(client):
    r = client.post("/api/sketch/generate", json={"image": data_url(png()), "model": "nb21-2k"})
    assert r.status_code == 400 and client.fake.calls == []


def wait_job(client, job):
    import time
    for _ in range(200):
        if job["status"] != "running":
            break
        time.sleep(.03)
        job = client.get(f"/api/sketch/job/{job['id']}").json()
    assert job["status"] == "done", job
    return job["result"]


def test_generate_route_then_image_similar_owner_only(client):
    r = client.post("/api/sketch/generate", json={"image": data_url(png()), "kind": "ring"})
    assert r.status_code == 200, r.text
    assert "uid" not in r.json()
    j = wait_job(client, r.json())
    assert j["budget"]["left_today"] == j["budget"]["per_user_day"] - 1
    img = client.get(j["image"])
    assert img.status_code == 200 and img.content[:4] == b"\x89PNG"
    assert "attachment" in client.get(j["image"] + "?download=1").headers["content-disposition"]
    assert client.get("/api/sketch/mine").json()["items"][0]["id"] == j["id"]
    assert client.delete(f"/api/sketch/{j['id']}").status_code == 200
    assert client.get(j["image"]).status_code == 404


def test_empty_sketch_and_off_topic_words_spend_nothing(client):
    r = client.post("/api/sketch/generate", json={"image": data_url(png(blank=True))})
    assert r.status_code == 400 and "empty" in r.json()["detail"]
    r = client.post("/api/sketch/generate", json={"notes": "a red sports car"})
    assert r.status_code == 400
    r = client.post("/api/sketch/generate", json={"notes": "heart pendant", "kind": "pendant"})
    assert r.status_code == 200
    wait_job(client, r.json())
    assert len(client.fake.calls) == 1 and client.fake.calls[0][2] is None
    assert client.fake.calls[0][0] == "p-klein"   # the default model


# ---------- Pollinations call ----------

def test_pollinations_sends_sketch_as_upload_and_reads_base64(monkeypatch):
    monkeypatch.setenv("POLLINATIONS_KEY", "sk_x")
    seen = {}

    def handler(req):
        seen["url"], seen["auth"], seen["type"], seen["body"] = str(req.url), req.headers["authorization"], req.headers["content-type"], req.content
        return httpx.Response(200, json={"data": [{"b64_json": base64.b64encode(png()).decode()}]})
    monkeypatch.setattr(sketch.httpx, "Client", _transport(handler))
    data, mime = sketch.call_provider(sketch.BY_KEY["p-kontext"], "the prompt", b"\xff\xd8\xffjpegbytes")
    assert seen["url"] == "https://gen.pollinations.ai/v1/images/edits" and seen["auth"] == "Bearer sk_x"
    assert seen["type"].startswith("multipart/form-data")
    assert b'name="model"\r\n\r\nkontext' in seen["body"] and b"jpegbytes" in seen["body"] and b"the prompt" in seen["body"]
    assert mime == "image/png" and Image.open(io.BytesIO(data)).format == "PNG"


def test_pollinations_words_only_uses_generations(monkeypatch):
    monkeypatch.setenv("POLLINATIONS_KEY", "sk_x")
    seen = {}

    def handler(req):
        seen["url"], seen["body"] = str(req.url), json.loads(req.content)
        return httpx.Response(200, json={"data": [{"b64_json": base64.b64encode(png()).decode()}]})
    monkeypatch.setattr(sketch.httpx, "Client", _transport(handler))
    sketch.call_provider(sketch.BY_KEY["p-gptimage"], "p", None)
    assert seen["url"].endswith("/v1/images/generations")
    assert seen["body"]["model"] == "gptimage" and seen["body"]["quality"] == "medium" and seen["body"]["n"] == 1


def test_pollinations_url_answer_only_from_their_host(monkeypatch):
    monkeypatch.setenv("POLLINATIONS_KEY", "sk_x")
    asked = []

    def handler(req):
        asked.append(str(req.url))
        if req.url.host == "media.pollinations.ai":
            return httpx.Response(200, content=png())
        return httpx.Response(200, json={"data": [{"url": answer}]})
    monkeypatch.setattr(sketch.httpx, "Client", _transport(handler))
    answer = "https://media.pollinations.ai/abc"
    sketch.call_provider(sketch.BY_KEY["p-klein"], "p", b"x")
    assert asked[-1] == answer
    answer = "http://169.254.169.254/latest"
    with pytest.raises(sketch.SketchError):
        sketch.call_provider(sketch.BY_KEY["p-klein"], "p", b"x")
    assert answer not in asked


@pytest.mark.parametrize("status,match", [(401, "key"), (402, "limit"), (403, "not available"), (429, "Too many"), (422, "declined")])
def test_pollinations_errors_read_plainly(monkeypatch, status, match):
    monkeypatch.setenv("POLLINATIONS_KEY", "sk_x")
    monkeypatch.setattr(sketch.httpx, "Client", _transport(lambda req: httpx.Response(status, json={"error": {"message": "m"}})))
    with pytest.raises(sketch.SketchError, match=match):
        sketch.call_provider(sketch.BY_KEY["p-kontext"], "p", b"x")


def test_pollinations_no_key_sends_nothing(monkeypatch):
    monkeypatch.setattr(sketch.httpx, "Client", None)
    with pytest.raises(sketch.SketchError) as e:
        sketch.call_provider(sketch.BY_KEY["p-kontext"], "p", None)
    assert e.value.status == 503


def test_paid_pollinations_models_hidden_until_enabled(monkeypatch):
    monkeypatch.setenv("POLLINATIONS_KEY", "sk_x")
    monkeypatch.delenv("POLLINATIONS_PAID", raising=False)
    assert all(m.free for m in sketch.available())
    monkeypatch.setenv("POLLINATIONS_PAID", "1")
    assert "p-nanobanana" in [m.key for m in sketch.available()]


def test_paid_model_refusal_says_not_available(monkeypatch):
    monkeypatch.setenv("POLLINATIONS_KEY", "sk_x")
    monkeypatch.setattr(sketch.httpx, "Client", _transport(lambda req: httpx.Response(402, json={"error": {"message": "m"}})))
    with pytest.raises(sketch.SketchError, match="not available on this account"):
        sketch.call_provider(sketch.BY_KEY["p-nanobanana"], "p", b"x")
    with pytest.raises(sketch.SketchError, match="limit is reached"):
        sketch.call_provider(sketch.BY_KEY["p-kontext"], "p", b"x")



# ---------- on our own computer (mflux) ----------

@pytest.fixture
def local_on(tmp_path, monkeypatch):
    """Pretend the generator and its weights are installed; record the command instead of running it."""
    import subprocess as sp
    monkeypatch.setenv("LOCAL_IMAGEGEN", "1")
    monkeypatch.setattr(sketch, "local_ready", lambda: True)
    monkeypatch.setattr(sketch, "DIR", tmp_path)
    monkeypatch.setattr(sketch, "free_memory_gb", lambda: 8.0)
    seen = []

    def run(cmd, **kw):
        seen.append((cmd, kw))
        out = cmd[cmd.index("--output") + 1]
        Image.new("RGB", (int(cmd[cmd.index("--width") + 1]),) * 2, "white").save(out)
        return sp.CompletedProcess(cmd, 0, "", "")
    monkeypatch.setattr(sketch.subprocess, "run", run)
    return seen


def test_local_is_offered_first_and_is_the_default(local_on, monkeypatch):
    monkeypatch.setattr(sketch, "_key", lambda p: "on" if p == "local" else "")
    assert sketch.connected() == ["local"] and sketch.default_model() == "local-klein"
    assert [m.key for m in sketch.available()] == ["local-klein", "local-klein-hd"]


def test_local_call_runs_mflux_edit_with_low_ram_and_reference(local_on):
    data, mime = sketch.call_provider(sketch.BY_KEY["local-klein"], "a prompt", sketch.prepare(png()))
    cmd, kw = local_on[0]
    assert cmd[0].endswith("mflux-generate-flux2-edit") and "--low-ram" in cmd
    assert cmd[cmd.index("--model") + 1] == sketch.LOCAL_WEIGHTS and cmd[cmd.index("--width") + 1] == "768"
    ref = cmd[cmd.index("--image-paths") + 1]
    assert kw["env"]["HF_HUB_OFFLINE"] == "1" and kw["timeout"] == sketch.LOCAL_TIMEOUT
    assert Image.open(io.BytesIO(data)).size == (768, 768) and mime == "image/png"
    assert not list(sketch.DIR.glob("tmp/*"))   # temporary files cleaned up
    assert not os.path.exists(ref)


def test_local_words_only_uses_text_to_image(local_on):
    sketch.call_provider(sketch.BY_KEY["local-klein-hd"], "p", None)
    cmd = local_on[0][0]
    assert cmd[0].endswith("mflux-generate-flux2") and "--image-paths" not in cmd and cmd[cmd.index("--width") + 1] == "1024"


def test_local_refuses_when_memory_stays_low(local_on, monkeypatch):
    monkeypatch.setattr(sketch, "free_memory_gb", lambda: 0.5)
    monkeypatch.setattr(sketch.time, "sleep", lambda s: None)
    with pytest.raises(sketch.SketchError, match="busy"):
        sketch.call_provider(sketch.BY_KEY["local-klein"], "p", None)
    assert local_on == [] and not sketch.LOCAL_LOCK.locked()


def test_local_failure_reads_plainly_and_frees_the_lock(local_on, monkeypatch):
    import subprocess as sp
    monkeypatch.setattr(sketch.subprocess, "run", lambda cmd, **kw: sp.CompletedProcess(cmd, 1, "", "boom"))
    with pytest.raises(sketch.SketchError, match="could not make"):
        sketch.call_provider(sketch.BY_KEY["local-klein"], "p", None)
    assert not sketch.LOCAL_LOCK.locked()


def test_local_switch_off(monkeypatch):
    monkeypatch.setenv("LOCAL_IMAGEGEN", "0")
    assert sketch.local_ready() is False and "local" not in sketch.connected()
