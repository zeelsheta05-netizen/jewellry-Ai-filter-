"""Design Generator (jewelsearch/design.py): spec -> exact prompt, checks, queue, inventory, routes."""
import io
import time

import pytest
from PIL import Image

from jewelsearch import design as D
from jewelsearch import sketch
from jewelsearch.sketch import SketchError


def png(color=(250, 250, 250)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), color).save(buf, "PNG")
    return buf.getvalue()


class FakeAI:
    def __init__(self):
        self.prompts = []

    def __call__(self, model, prompt, picture):
        self.prompts.append(prompt)
        return png((len(self.prompts) * 30 % 255, 200, 200)), "image/png"


def ring_spec(**kw):
    base = dict(category="ring", metal_type="18k white gold", plating="rhodium", weight_g=3.2,
                stones=[D.Stone("Diamond", 1, "D-F", "round", 6.5, "VVS1"), D.Stone("diamond", 24, "white", "round", 1.3, "VS1")])
    base.update(kw)
    return D.check_spec(D.Spec(**base))


# ---------- prompt ----------

def test_prompt_has_every_detail_in_exact_words():
    b = D.build_prompt(ring_spec(style="Art deco with milgrain", regional="Italian"))
    p = b.prompt
    assert p.startswith("Photorealistic high-end jewellery product photo of a finger ring, sized for one finger")
    assert "Design: a centre stone ring" in p
    assert "18k white gold, rhodium plated (bright mirror-white finish)" in p and "3.2 g" in p
    assert "Centre stone: exactly 1 round brilliant-cut colourless icy-white diamond, 6.5 mm (about 1.1 ct)" in p
    assert "Pavé accents: 24 round brilliant-cut colourless icy-white diamond, 1.3 mm each" in p
    assert "exactly these and no others" in p and "Regional style: Italian." in p and "Design style: Art deco with milgrain" in p
    assert b.expect == {"type": "ring", "type_label": "ring", "category": "ring", "metal": "white_gold", "stones": True,
                        "centre_cut": "round", "shape": "round", "shape_key": "round", "size": "large", "count": 25}
    assert len(p) <= D.PROMPT_BUDGET


def test_spec_comes_first_and_long_style_is_shortened_openly():
    b = D.build_prompt(ring_spec(style="vintage filigree scrollwork " * 300))
    assert len(b.prompt) <= D.PROMPT_BUDGET + 20
    assert b.prompt.index("Centre stone") < b.prompt.index("Design style")
    assert any("shortened" in n for n in b.notes)


def test_metal_reading():
    assert D.read_metal("18k alloy", "")[1] == "yellow_gold" and D.read_metal("18k alloy", "")[2]   # told it was assumed
    assert D.read_metal("18k alloy", "rhodium")[1] == "white_gold"
    assert D.read_metal("14KT Rose", "")[1] == "rose_gold"
    assert D.read_metal("Platinum 950", "")[1] == "white_gold"
    assert D.read_metal("925 sterling silver", "black rhodium")[1] == "black"
    words, colour, _ = D.read_metal("18k yellow gold", "Rhodium")     # the plating is all that shows
    assert colour == "white_gold" and "yellow" not in words and words.startswith("18k gold, rhodium plated")
    assert "rich deep yellow" in D.read_metal("22kt gold", "")[0]
    assert D.read_metal("", "")[2]   # nothing given: a note says what was used


def test_pairs_count_per_piece_and_plain_design():
    e = D.check_spec(D.Spec("earrings", "18k yellow gold", stones=[D.Stone("ruby", 2, "red", "pear", 7)]))
    p = D.build_prompt(e).prompt
    assert "2 in total (1 on each of the pair)" in p and "pear-shaped" in p and "matching pair of earrings" in p
    plain = D.build_prompt(D.check_spec(D.Spec("bangle", "22k gold", weight_g=30)))
    assert "no gemstones at all" in plain.prompt and "bold, substantial" in plain.prompt and plain.expect["type"] == "bracelet"


def test_check_spec_refuses_bad_input():
    with pytest.raises(SketchError, match="category"):
        D.check_spec(D.Spec("spaceship"))
    with pytest.raises(SketchError, match="Stone 1: fill in the color"):
        D.check_spec(D.Spec("ring", stones=[D.Stone("diamond", 1, "", "round", 2)]))
    with pytest.raises(SketchError, match="size"):
        D.check_spec(D.Spec("ring", stones=[D.Stone("diamond", 1, "white", "round", 0.1)]))
    with pytest.raises(SketchError, match="Up to 3"):
        D.check_spec(D.Spec("ring", stones=[D.Stone("d", 1, "w", "round", 2)] * 4))


def test_compare_and_remake_words():
    exp = D.build_prompt(ring_spec()).expect
    good = D.compare(exp, {"type": "ring", "type_sure": True, "metal": "white_gold", "plain": False, "centre_cut": "round"})
    assert all(c["ok"] for c in good) and not D.misses(good)
    bad = D.compare(exp, {"type": "pendant", "type_sure": True, "metal": "yellow_gold", "plain": False, "centre_cut": None})
    assert set(D.misses(bad)) == {"type", "metal"} and D.score(bad) < D.score(good)
    unsure = D.compare(exp, {"type": "pendant", "type_sure": False, "metal": None, "plain": None, "centre_cut": None})
    assert not D.misses(unsure)   # can't tell is not a miss
    words = D.stress_words(exp, ["type", "metal"], "ring")
    assert "clearly be a ring" in words[0] and "white gold" in words[1]
    assert D.build_prompt(ring_spec(), stress=words).prompt.startswith("Most important:")


def test_plan_reads_the_pieces_real_size():
    """The user's 2026-10-08 bracelet: 10 x 1 mm emerald-cut diamonds on 10 g rose gold came out as a ring
    with big stones. The plan now says what that spec really is."""
    sp = D.check_spec(D.Spec("bracelet", "18k rose gold", "Rose gold", 10, [D.Stone("Diamond", 10, "White", "emerald", 1, "VVS2")]))
    b = D.build_prompt(sp)
    assert "wrist bracelet about 17 cm long with a clasp" in b.prompt and "far longer and larger than a ring" in b.prompt
    assert "station bracelet" in b.prompt and "tiny melee stones" in b.prompt and "closed loop" not in b.prompt
    assert b.expect["size"] == "tiny" and b.expect["shape"] == "rect" and b.expect["count"] == 10
    alt = D.build_prompt(sp, alt_view=True).prompt
    assert "laid completely flat and straight" in alt
    tennis = D.build_prompt(D.check_spec(D.Spec("bracelet", "18k white gold", stones=[D.Stone("Diamond", 50, "White", "round", 2.5)])))
    assert "tennis bracelet" in tennis.prompt
    assert "three-stone ring" in D.build_prompt(D.check_spec(D.Spec("ring", "18k white gold", stones=[D.Stone("Diamond", 3, "White", "oval", 5)]))).prompt
    assert "eternity band" in D.build_prompt(D.check_spec(D.Spec("ring", "18k white gold", stones=[D.Stone("Diamond", 22, "White", "round", 2.2)]))).prompt
    placed = D.build_prompt(D.check_spec(D.Spec("bracelet", "18k white gold", placement="all on the clasp",
                                                stones=[D.Stone("Diamond", 50, "White", "round", 2.5)])))
    assert "tennis" not in placed.prompt and "all on the clasp" in placed.prompt    # the person's placement wins


def vlm(**kw):
    r = {"by": "vlm", "type": "bracelet", "type_p": 0.99, "metal": "rose_gold", "metal_p": 0.99, "stones": "yes",
         "stones_p": 0.99, "stones_yes": 0.99, "size": "tiny", "size_p": 0.9, "count": "9-20", "count_p": 0.8}
    r.update(kw)
    return r


def test_vision_language_checks_catch_the_screenshot_misses():
    sp = D.check_spec(D.Spec("bracelet", "18k rose gold", "Rose gold", 10, [D.Stone("Diamond", 10, "White", "emerald", 1, "VVS2")]))
    exp = D.build_prompt(sp).expect
    keys = [k for k, _, _ in D.vlm_questions(exp)]
    assert keys == ["type", "metal", "stones", "size", "count"]          # tiny stones: shape isn't asked
    good = D.compare(exp, vlm())
    assert all(c["ok"] for c in good) and not D.misses(good)
    # what the reader said about the real picture: a ring, rose gold, large stones, more than 20
    bad = D.compare(exp, vlm(type="ring", type_p=1.0, size="large", size_p=0.86, count="21+", count_p=0.95))
    assert set(D.misses(bad)) == {"type", "size"} and D.score(bad) < D.score(good)   # 21+ is next to 9-20: not a miss
    words = D.stress_words(exp, ["type", "size"], "bracelet", sp)
    assert "clearly be a bracelet" in words[0] and "only 1 mm" in words[1]
    unsure = D.compare(exp, vlm(type="ring", type_p=0.5))
    assert "type" not in D.misses(unsure)
    # rhodium over yellow gold read as yellow: a miss (the old reader said "can't tell")
    ring = D.build_prompt(D.check_spec(D.Spec("ring", "18k yellow gold", "Rhodium", 4.9, [D.Stone("Diamond", 1, "Pink", "emerald", 7)]))).expect
    r = D.compare(ring, vlm(type="ring", metal="yellow_gold", shape="squarish", shape_p=0.9, square="cushion", square_p=0.8,
                            size="large", count="1"))
    assert set(D.misses(r)) == {"metal", "shape"}
    assert [k for k, _, _ in D.vlm_questions(ring)][3:5] == ["shape", "square"]


def test_reader_uses_the_vision_language_model_when_loaded():
    class FakeVLM:
        def __init__(self):
            self.asked = []

        def choose(self, im, questions):
            self.asked.append(questions)
            return [[1.0] + [0.0] * (len(o) - 1) for _, o in questions]   # always the first option

        def read(self, im):
            return {}

    fv = FakeVLM()
    engine = type("E", (), {"detail_reader": fv.read})()
    photo_mod = type("P", (), {"read": staticmethod(lambda b: Image.open(io.BytesIO(b)))})
    read = D.reader_from_engine(engine, photo_mod)
    exp = D.build_prompt(ring_spec()).expect
    r = read(png(), exp)
    assert r["by"] == "vlm" and r["type"] == "ring" and r["metal"] == "yellow_gold" and r["stones_yes"] == 1.0
    assert len(fv.asked[0]) == len(D.vlm_questions(exp))


# ---------- queue ----------

@pytest.fixture
def studio(tmp_path, monkeypatch):
    for k in ("SKETCH_PER_USER_DAY", "SKETCH_USD_DAY", "SKETCH_USD_MONTH"):
        monkeypatch.delenv(k, raising=False)
    return sketch.Studio(tmp_path, caller=FakeAI())


def test_queue_makes_checked_designs(studio):
    q = D.Queue(studio, reader=lambda b: {"type": "ring", "type_sure": True, "metal": "white_gold", "plain": False, "centre_cut": "round"})
    item = q.submit("u", ring_spec(), "p-klein", 2, None, wait=True)
    got = q.list("u")[0]
    assert got["id"] == item["id"] and got["status"] == "done" and len(got["results"]) == 2
    assert all(c["ok"] for r in got["results"] for c in r["checks"]) and not got["results"][0]["remade"]
    assert got["results"][0]["image"] != got["results"][1]["image"]          # two different pictures, not the cache
    assert len(studio.caller.prompts) == 2
    assert "_prompt" not in got and studio.mine("u", panel="design")[0]["meta"]["design"]["summary"].startswith("Ring")


def test_queue_remakes_a_clear_miss_once_and_keeps_the_better(studio):
    reads = iter([{"type": "pendant", "type_sure": True, "metal": "white_gold", "plain": False, "centre_cut": None},
                  {"type": "ring", "type_sure": True, "metal": "white_gold", "plain": False, "centre_cut": "round"}])
    q = D.Queue(studio, reader=lambda b: next(reads))
    q.submit("u", ring_spec(), "p-klein", 1, None, wait=True)
    r = q.list("u")[0]["results"][0]
    assert r["remade"] and all(c["ok"] for c in r["checks"])
    assert studio.caller.prompts[1].startswith("Most important: the piece must clearly be a ring")
    assert len(studio.mine("u", panel="design")) == 1    # the missed picture is hidden


def test_queue_keeps_trying_hard_misses_up_to_the_limit(studio, monkeypatch):
    monkeypatch.setenv("DESIGN_TRIES", "3")
    sp = D.check_spec(D.Spec("bracelet", "18k rose gold", "Rose gold", 10, [D.Stone("Diamond", 10, "White", "emerald", 1)]))
    reads = iter([vlm(type="ring"), vlm(type="ring", size="large"), vlm(size="large")])
    q = D.Queue(studio, reader=lambda b, e: next(reads))
    q.submit("u", sp, "p-klein", 1, None, wait=True)
    r = q.list("u")[0]["results"][0]
    assert r["tries"] == 3 and len(studio.caller.prompts) == 3
    assert "laid completely flat and straight" in studio.caller.prompts[1]           # a new view after a type miss
    assert "only 1 mm" in studio.caller.prompts[2] and "clearly be a bracelet" in studio.caller.prompts[2]
    assert [c["key"] for c in r["checks"] if c["ok"] is False] == ["size"]           # the closest one is kept
    assert len(studio.mine("u", panel="design")) == 1


def test_queue_soft_miss_is_shown_not_remade_by_default(studio, monkeypatch):
    sp = D.check_spec(D.Spec("bracelet", "18k rose gold", "Rose gold", 10, [D.Stone("Diamond", 10, "White", "emerald", 1)]))
    q = D.Queue(studio, reader=lambda b, e: vlm(size="large"))
    q.submit("u", sp, "p-klein", 1, None, wait=True)
    r = q.list("u")[0]["results"][0]
    assert r["tries"] == 1 and len(studio.caller.prompts) == 1 and not r["remade"]
    assert [c["key"] for c in r["checks"] if c["ok"] is False] == ["size"]      # still shown on the picture
    monkeypatch.setenv("DESIGN_TRIES", "3")
    monkeypatch.setenv("DESIGN_SOFT_REMAKE", "1")
    sp.weight_g = 11   # not the same request again (a double click is one request)
    q.submit("u", sp, "p-klein", 1, None, wait=True)
    assert q.list("u")[0]["results"][0]["tries"] == 2 and len(studio.caller.prompts) == 3   # one remake only


def test_defaults_are_the_fast_ones(monkeypatch):
    monkeypatch.delenv("DESIGN_TRIES", raising=False)
    assert D.max_tries() == 2
    monkeypatch.setattr(sketch, "_key", lambda p: "on" if p == "local" else "")
    assert D.default_model() == "local-klein"


def test_queue_respects_daily_limit_and_cancel(studio, monkeypatch):
    monkeypatch.setenv("SKETCH_PER_USER_DAY", "2")
    q = D.Queue(studio)
    with pytest.raises(SketchError, match="2 more designs today"):
        q.submit("u", ring_spec(), "p-klein", 3, None)
    q.workers.add("u")   # pretend a request is already being made: the next one waits
    item = q.submit("u", ring_spec(metal_type="18k rose gold"), "p-klein", 1, None)
    assert item["status"] == "queued" and q.cancel("u", item["id"]) and q.list("u")[0]["status"] == "cancelled"
    assert not q.cancel("u", item["id"])


def test_queue_error_is_reported(studio):
    def boom(model, prompt, picture):
        raise SketchError("limit is reached", 402)
    studio.caller = boom
    q = D.Queue(studio)
    q.submit("u", ring_spec(), "p-klein", 1, None, wait=True)
    got = q.list("u")[0]
    assert got["status"] == "error" and got["error"] == "limit is reached"


# ---------- inventory ----------

def test_inventory_crud_resolve_and_csv(tmp_path):
    inv = D.Inventory(tmp_path)
    s = inv.add({"code": "RD-650", "type": "Diamond", "shape": "Round", "color": "white", "size_mm": 6.5, "clarity": "VVS1", "qty": 3})
    assert s["shape"] == "round" and inv.all()[0]["qty"] == 3
    got = inv.resolve([D.Stone("", 2, "", "", 0, "", s["id"])])[0]
    assert (got.type, got.shape, got.size_mm, got.count) == ("Diamond", "round", 6.5, 2)
    with pytest.raises(SketchError, match="only 3"):
        inv.resolve([D.Stone("", 5, "", "", 0, "", s["id"])])
    inv.update(s["id"], {**s, "qty": 10})
    assert inv.get(s["id"])["qty"] == 10
    with pytest.raises(SketchError, match="Line 3"):
        inv.import_csv("code,type,shape,color,size_mm,clarity,qty\nA,ruby,oval,red,5,,4\nB,ruby,blob,red,5,,4")
    assert len(inv.all()) == 1   # nothing added from a CSV with a wrong line
    assert inv.import_csv("Code,Type,Shape,Colour,Size,Clarity,Quantity\nA,ruby,oval,red,5,,4\nB,pearl,round,white,8,,20")["added"] == 2
    assert inv.delete(s["id"]) and len(inv.all()) == 2


# ---------- routes ----------

@pytest.fixture
def client(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from fastapi.testclient import TestClient
    from jewelsearch import auth, server

    monkeypatch.setattr(auth, "read_session", lambda c: {"uid": "u-1", "name": "A", "email": "a@x.in"} if c == "ok" else None)

    async def yes(_uid):
        return True

    async def role(_uid):
        return client_role["role"]
    client_role = {"role": None}
    monkeypatch.setattr(auth, "is_approved", yes)
    monkeypatch.setattr(auth, "staff_role", role)
    monkeypatch.setattr(auth, "limiter", auth.RateLimiter())
    monkeypatch.setenv("POLLINATIONS_KEY", "sk_test")
    st = sketch.Studio(tmp_path, caller=FakeAI())
    monkeypatch.setattr(server, "studio", st)
    monkeypatch.setattr(server, "design_inventory", D.Inventory(tmp_path))
    monkeypatch.setattr(server, "design_queue", D.Queue(st, reader=lambda b: {"type": "ring", "type_sure": True,
                                                                              "metal": "white_gold", "plain": False, "centre_cut": "round"}))
    monkeypatch.setattr(server, "engine", SimpleNamespace(domain=SimpleNamespace(check=lambda t: SimpleNamespace(ok=True, text=t))))
    c = TestClient(server.app)
    c.cookies.set(auth.COOKIE, "ok")
    c.role = client_role
    return c


def wait_done(client, n=50):
    for _ in range(n):
        items = client.get("/api/generator/queue").json()["items"]
        if items and items[0]["status"] in ("done", "error"):
            return items[0]
        time.sleep(0.05)
    raise AssertionError("queue did not finish")


def test_page_options_and_generate_flow(client):
    r = client.get("/design", headers={"sec-fetch-dest": "iframe"})
    assert r.status_code == 200 and "Design Generator" in r.text and "frame-ancestors 'self'" in r.headers["content-security-policy"]
    assert client.get("/design", headers={"sec-fetch-dest": "document"}, follow_redirects=False).headers["location"] == "/ai?tab=design"
    assert "Design generator AI" in client.get("/ai").text
    o = client.get("/api/generator/options").json()
    assert [c["label"] for c in o["categories"]][:3] == ["Ring", "Necklace", "Bracelet"] and len(o["categories"]) == 11
    assert [s["label"] for s in o["shapes"]][:11] == ["Round", "Oval", "Pear", "Marquise", "Emerald", "Princess", "Cushion",
                                                     "Radiant", "Asscher", "Heart", "Trillion"]
    assert not o["staff"] and o["max_stones"] == 3 and o["style_max"] == 5000
    assert not any(k in m for m in o["models"] for k in ("usd", "inr", "free"))
    body = {"category": "ring", "metal_type": "18k white gold", "plating": "rhodium", "weight_g": 3.2,
            "stones": [{"type": "diamond", "count": 1, "color": "white", "shape": "round", "size_mm": 6.5, "clarity": "VVS1"}],
            "style": "Art deco", "model": "p-klein"}
    r = client.post("/api/generator/generate", json=body)
    assert r.status_code == 200, r.text
    done = wait_done(client)
    assert done["status"] == "done" and done["results"][0]["checks"][0]["ok"]
    assert client.get(done["results"][0]["image"]).status_code == 200
    bad = client.post("/api/generator/generate", json={**body, "stones": [{**body["stones"][0], "shape": "blob"}]})
    assert bad.status_code == 400 and "shape" in bad.json()["detail"]


def test_inventory_routes_staff_only_and_inventory_mode(client):
    stone = {"code": "RD-650", "type": "diamond", "shape": "round", "color": "white", "size_mm": 6.5, "clarity": "VVS1", "qty": 2}
    assert client.post("/api/generator/inventory", json=stone).status_code == 403
    client.role["role"] = "jeweler"
    sid = client.post("/api/generator/inventory", json=stone).json()["id"]
    assert client.get("/api/generator/inventory").json()["items"][0]["id"] == sid
    body = {"mode": "inventory", "category": "ring", "metal_type": "platinum", "model": "p-klein",
            "stones": [{"inventory_id": sid, "count": 3}]}
    r = client.post("/api/generator/generate", json=body)
    assert r.status_code == 400 and "only 2" in r.json()["detail"]
    body["stones"][0]["count"] = 1
    assert client.post("/api/generator/generate", json=body).status_code == 200
    assert "6.5 mm" in wait_done(client)["summary"]
    client.role["role"] = None
    assert client.delete(f"/api/generator/inventory/{sid}").status_code == 403
