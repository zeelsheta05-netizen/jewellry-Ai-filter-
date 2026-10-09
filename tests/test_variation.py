"""Design Variations (jewelsearch/variation.py): ideas, 2x2 tiles joined into sets, jobs, cut-outs, routes."""
import base64
import io
import json
import re
import time
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from jewelsearch import sketch, variation as V


def tile(panels=4) -> bytes:
    """What a model returns for a 2x2 request: four pieces on cream with white gutters."""
    im = Image.new("RGB", (1024, 1024), (255, 255, 255))
    d = ImageDraw.Draw(im)
    boxes = [(0, 0, 505, 505), (519, 0, 1024, 505), (0, 519, 505, 1024), (519, 519, 1024, 1024)] if panels == 4 else [(0, 0, 1024, 1024)]
    for x0, y0, x1, y1 in boxes:
        d.rectangle((x0, y0, x1, y1), fill=(246, 238, 220))
        cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
        d.ellipse((cx - 120, cy - 120, cx + 120, cy + 120), outline=(150, 110, 40), width=14)
        d.ellipse((cx - 25, cy + 120, cx + 25, cy + 190), fill=(60, 170, 120))
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return buf.getvalue()


def sketch_png() -> bytes:
    im = Image.new("RGB", (600, 600), "white")
    ImageDraw.Draw(im).ellipse((150, 150, 450, 450), outline="black", width=8)
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return buf.getvalue()


class FakeAI:
    def __init__(self, fail_first=0, panels=4):
        self.calls, self.fail_first, self.panels = [], fail_first, panels

    def __call__(self, model, prompt, picture):
        self.calls.append((model.key, prompt))
        if len(self.calls) <= self.fail_first:
            raise sketch.SketchError("Too many pictures at once.", 429)
        return tile(1 if "create one variation" in prompt else self.panels), "image/png"


class FakePlanner:
    """Stands in for the text model: names ideas from a counter, records what it was told."""
    def __init__(self, fail=False):
        self.prompts, self.k, self.fail = [], 0, fail

    def __call__(self, prompt, picture, max_tokens):
        self.prompts.append(prompt)
        if self.fail:
            raise sketch.SketchError("planner down", 502)
        n = int(re.search(r"exactly (\d+)", prompt).group(1))
        ideas = []
        for _ in range(n):
            self.k += 1
            ideas.append({"name": f"Idea {self.k}", "brief": f"brief {self.k}"})
        return "Sure! " + json.dumps({"piece": "A green ring.", "ideas": ideas})


@pytest.fixture
def jobs(tmp_path, monkeypatch):
    for k in ("SKETCH_PER_USER_DAY", "SKETCH_USD_DAY", "SKETCH_USD_MONTH"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("SKETCH_PER_USER_DAY", "20")   # these tests check the counter; the default is now no limit
    monkeypatch.setattr(V, "RETRY_WAIT", 0)
    return V.Jobs(sketch.Studio(tmp_path, caller=FakeAI()), planner=FakePlanner())


PIC = None


def pic():
    global PIC
    PIC = PIC or sketch.prepare(sketch_png())
    return PIC


def labels(row):
    return [c["label"] for c in row["meta"]["cells"]]


# ---------- planning ----------

def test_calls_are_one_per_four_variations():
    assert [V.calls_for(n) for n in V.COUNTS] == [1, 1, 3, 4, 7, 9]


def test_no_fixed_style_lists_remain():
    assert not hasattr(V, "STYLES")


def test_plan_prompt_is_about_this_piece_with_direction_and_used_ideas():
    p = V._plan_prompt(9, "lighter for daily wear", ["Basket Halo", "Twisted Shoulders"])
    assert "exactly 9" in p and "THIS piece" in p and "lighter for daily wear" in p
    assert "Basket Halo; Twisted Shoulders" in p and "gemstone" in p and "JSON" in p


def test_parse_plan_tolerates_chatter_and_drops_duplicates():
    text = 'Here: {"piece": "p", "ideas": [{"name": "A", "brief": "a"}, {"name": "a", "brief": "dup"}, {"name": "B"}]}'
    piece, ideas = V._parse_plan(text, 2)
    assert piece == "p" and [i["name"] for i in ideas] == ["A", "B"] and ideas[1]["brief"] == "B"
    with pytest.raises(ValueError):
        V._parse_plan(text, 3)
    with pytest.raises(ValueError):
        V._parse_plan("no json here", 1)


def test_planner_down_falls_back_to_random_ideas_that_differ_each_time():
    _, a, who = V.plan_ideas(b"x", 9, "", [], planner=FakePlanner(fail=True))
    _, b, _ = V.plan_ideas(b"x", 9, "", [i["name"] for i in a], planner=FakePlanner(fail=True))
    assert who == "random" and len(a) == len(b) == 9
    assert not {i["name"] for i in a} & {i["name"] for i in b}


def test_tile_prompt_names_each_panel_with_its_brief():
    ideas = [{"name": f"N{i}", "brief": f"b{i}"} for i in range(4)]
    p = V.tile_prompt(ideas, "Direction here.", "photo", "A green ring.")
    assert "2x2 grid" in p and "top-left: b0" in p and "bottom-right: b3" in p and "N0" not in p   # names are captions only
    assert "The original piece: A green ring." in p and "Direction: Direction here." in p
    assert "photorealistic" in p and "No text" in p
    one = V.tile_prompt(ideas[:1], "", "pencil")
    assert "create one variation: b0" in one and "grid" not in one


# ---------- tiles and sets ----------

def test_split_tile_finds_four_panels_and_keeps_single_pictures_whole():
    assert len(V.split_tile(tile(4), 4)) == 4
    assert len(V.split_tile(tile(1), 4)) == 1
    assert len(V.split_tile(tile(4), 2)) == 2   # last tile of a set may need fewer


@pytest.mark.parametrize("name,parts", [("hairline_necklaces", 4), ("hairline_rings", 4), ("single_ring", 1)])
def test_split_real_tiles_without_gutters(name, parts):
    """Real 2x2 photo tiles where panels touch with only a hairline (these came out merged before)."""
    data = (Path(__file__).parent / "data" / f"{name}.jpg").read_bytes()
    assert len(V.split_tile(data, 4)) == parts


def test_compose_grid_with_captions_and_cell_boxes():
    panels = V.split_tile(tile(4), 4) + V.split_tile(tile(4), 4)
    ideas = [{"name": f"Idea {i}", "brief": f"b{i}"} for i in range(6)]
    data, meta = V.compose(panels[:6], ideas)
    im = Image.open(io.BytesIO(data))
    assert (meta["rows"], meta["cols"]) == (2, 3) and im.size == tuple(meta["size"])
    c = meta["cells"][4]
    assert c["label"] == "Idea 4" and c["brief"] == "b4" and c["x"] + c["w"] <= im.width


# ---------- jobs ----------

def test_sixteen_variations_one_plan_four_drawing_calls(jobs):
    j = jobs.start("u1", pic(), 16, "", "illustration", "p-gptimage", wait=True)
    assert j["status"] == "done", j
    assert len(jobs.studio.caller.calls) == 4 and len(jobs.planner.prompts) == 1
    r = j["result"]
    assert r["panel"] == "variation" and r["meta"]["n"] == 16 and r["meta"]["planned"] == "ai"
    assert labels(r)[:3] == ["Idea 1", "Idea 2", "Idea 3"] and r["meta"]["piece"] == "A green ring."
    assert "The original piece: A green ring." in jobs.studio.caller.calls[0][1]
    assert jobs.studio.spent(0)[0] == pytest.approx(4 * sketch.BY_KEY["p-gptimage"].usd + sketch.PLANNER_USD)
    assert jobs.studio.budget("u1")["left_today"] == jobs.studio.budget("u1")["per_user_day"] - 1   # one set = one design


def test_second_set_tells_the_planner_what_was_shown(jobs):
    a = jobs.start("u1", pic(), 4, "", "illustration", "p-gptimage", wait=True)["result"]
    jobs.start("u1", pic(), 4, "temple style", "illustration", "p-gptimage", wait=True)
    second = jobs.planner.prompts[1]
    assert all(x in second for x in labels(a)) and "temple style" in second
    other = sketch.prepare(Path(__file__).parent.joinpath("data", "single_ring.jpg").read_bytes())
    jobs.start("u1", other, 4, "", "illustration", "p-gptimage", wait=True)
    assert "Already shown" not in jobs.planner.prompts[2]   # a different picture starts fresh


def test_planner_failure_still_makes_a_set(tmp_path, monkeypatch):
    monkeypatch.setattr(V, "RETRY_WAIT", 0)
    jobs = V.Jobs(sketch.Studio(tmp_path, caller=FakeAI()), planner=FakePlanner(fail=True))
    r = jobs.start("u1", pic(), 4, "", "illustration", "p-gptimage", wait=True)["result"]
    assert r["meta"]["planned"] == "random" and len(set(labels(r))) == 4


def test_single_variation_is_one_plain_picture(jobs):
    r = jobs.start("u1", pic(), 1, "", "photo", "p-kontext", wait=True)["result"]
    assert r["meta"]["n"] == 1 and "create one variation" in jobs.studio.caller.calls[0][1]


def test_merged_panels_still_make_a_set_with_a_note(tmp_path, monkeypatch):
    monkeypatch.setattr(V, "RETRY_WAIT", 0)
    jobs = V.Jobs(sketch.Studio(tmp_path, caller=FakeAI(panels=1)), planner=FakePlanner())
    r = jobs.start("u1", pic(), 4, "", "illustration", "p-klein", wait=True)["result"]
    assert r["meta"]["n"] == 1 and "1 of 4" in r["meta"]["note"]


def test_rate_limit_retried_once(tmp_path, monkeypatch):
    monkeypatch.setattr(V, "RETRY_WAIT", 0)
    jobs = V.Jobs(sketch.Studio(tmp_path, caller=FakeAI(fail_first=1)), planner=FakePlanner())
    j = jobs.start("u1", pic(), 4, "", "illustration", "p-klein", wait=True)
    assert j["status"] == "done" and len(jobs.studio.caller.calls) == 2


def test_failed_set_still_counts_paid_calls(tmp_path, monkeypatch):
    monkeypatch.setattr(V, "RETRY_WAIT", 0)
    monkeypatch.setattr(V, "PARALLEL", 1)
    studio = sketch.Studio(tmp_path)
    n = {"c": 0}

    def caller(model, prompt, picture):
        n["c"] += 1
        if n["c"] > 1:
            raise sketch.SketchError("The AI declined this picture or text.", 422)
        return tile(4), "image/png"
    studio.caller = caller
    jobs = V.Jobs(studio, planner=FakePlanner())
    j = jobs.start("u1", pic(), 9, "", "illustration", "p-klein", wait=True)
    assert j["status"] == "error" and "declined" in j["error"]
    assert studio.spent(0)[0] == pytest.approx(sketch.BY_KEY["p-klein"].usd + sketch.PLANNER_USD)
    assert studio.mine("u1") == [] and "u1" not in studio.busy


def test_budget_checked_for_all_calls_before_starting(jobs, monkeypatch):
    monkeypatch.setenv("SKETCH_USD_DAY", "0.04")
    with pytest.raises(sketch.SketchError, match="limit"):
        jobs.start("u1", pic(), 16, "", "illustration", "p-gptimage", wait=True)   # 4 x 0.012 + plan
    assert jobs.studio.caller.calls == [] and jobs.planner.prompts == [] and "u1" not in jobs.studio.busy


def test_bad_choices_refused(jobs):
    for args in [(5, "illustration", "p-klein"), (4, "oil", "p-klein"), (4, "illustration", "nope")]:
        with pytest.raises(sketch.SketchError):
            jobs.start("u1", pic(), args[0], "", args[1], args[2], wait=True)


def test_cell_cut_out_is_a_free_design_of_its_own(jobs):
    r = jobs.start("u1", pic(), 4, "", "illustration", "p-gptimage", wait=True)["result"]
    c = jobs.studio.cell(r["id"], "u1", 2)
    im = Image.open(jobs.studio.file(c["id"], "u1"))
    assert im.size == (V.CELL, V.CELL) and c["prompt"] == labels(r)[2]
    assert c["source"] == "cell" and c["panel"] == "variation" and c["parent"] == r["id"]
    with pytest.raises(sketch.SketchError):
        jobs.studio.cell(r["id"], "u1", 9)
    with pytest.raises(sketch.SketchError):
        jobs.studio.cell(r["id"], "u2", 0)


def test_galleries_are_split_by_panel(jobs):
    jobs.studio.generate("u1", "p-klein", "a sketch prompt", pic())
    jobs.start("u1", pic(), 4, "", "illustration", "p-gptimage", wait=True)
    assert [x["panel"] for x in jobs.studio.mine("u1", panel="sketch")] == ["sketch"]
    assert [x["panel"] for x in jobs.studio.mine("u1", panel="variation")] == ["variation"]


def test_background_job_finishes(jobs):
    j = jobs.start("u1", pic(), 4, "", "illustration", "p-kontext")
    for _ in range(100):
        if jobs.get(j["id"], "u1")["status"] != "running":
            break
        time.sleep(.05)
    assert jobs.get(j["id"], "u1")["status"] == "done"
    assert jobs.get(j["id"], "u2") is None


# ---------- routes ----------

@pytest.fixture
def client(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from fastapi.testclient import TestClient
    from jewelsearch import auth, server
    monkeypatch.setattr(auth, "read_session", lambda c: {"uid": "u-1", "name": "A", "email": "a@x.in"} if c == "ok" else None)

    async def yes(*a, **k):
        return True
    monkeypatch.setattr(auth, "is_approved", yes)
    monkeypatch.setattr(auth, "limiter", auth.RateLimiter())
    monkeypatch.setenv("POLLINATIONS_KEY", "sk_test")
    monkeypatch.setattr(V, "RETRY_WAIT", 0)
    studio = sketch.Studio(tmp_path, caller=FakeAI())
    monkeypatch.setattr(server, "studio", studio)
    monkeypatch.setattr(server, "variation_jobs", V.Jobs(studio, planner=FakePlanner()))
    monkeypatch.setattr(server, "engine", SimpleNamespace(domain=SimpleNamespace(check=lambda t: SimpleNamespace(ok=True, text=t))))
    c = TestClient(server.app)
    c.cookies.set(auth.COOKIE, "ok")
    c.studio = studio
    return c


def data_url(b: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(b).decode()


def test_page_and_options(client):
    r = client.get("/variation")
    assert r.status_code == 200 and "Variation (Sketch)" in r.text and "camera=(self)" in r.headers["permissions-policy"]
    assert 'data-style' not in r.text   # no fixed style chips
    o = client.get("/api/variation/options").json()
    assert [c["n"] for c in o["counts"]] == list(V.COUNTS) and o["counts"][3]["label"] == "16 (4×4)"
    assert "styles" not in o and o["default_model"] == "p-gptimage"
    assert not any(k in m for m in o["models"] for k in ("usd", "inr", "free"))


def test_generate_poll_cell_route_flow(client):
    j = client.post("/api/variation/generate", json={"image": data_url(sketch_png()), "count": 4, "direction": "temple style"}).json()
    assert "uid" not in j
    for _ in range(100):
        j = client.get(f"/api/variation/job/{j['id']}").json()
        if j["status"] != "running":
            break
        time.sleep(.05)
    assert j["status"] == "done", j
    assert client.studio.caller.calls[0][0] == "p-gptimage"   # default model
    assert "Direction: temple style" in client.studio.caller.calls[0][1]
    gid = j["result"]["id"]
    assert client.get(j["result"]["image"]).status_code == 200
    c = client.post(f"/api/sketch/{gid}/cell", json={"index": 1}).json()
    assert c["prompt"] == j["result"]["meta"]["cells"][1]["label"]
    assert [x["id"] for x in client.get("/api/sketch/mine?panel=variation").json()["items"]][:2] == [c["id"], gid]
    assert client.get("/api/sketch/mine?panel=sketch").json()["items"] == []
    assert client.get("/api/variation/job/nope").status_code == 404


def test_generate_refuses_bad_input_before_any_call(client, monkeypatch):
    r = client.post("/api/variation/generate", json={"image": data_url(sketch_png()), "count": 5})
    assert r.status_code == 400
    r = client.post("/api/variation/generate", json={"image": "data:text/plain;base64,aGk=", "count": 4})
    assert r.status_code == 400
    monkeypatch.delenv("POLLINATIONS_KEY")
    r = client.post("/api/variation/generate", json={"image": data_url(sketch_png()), "count": 4})
    assert r.status_code == 503
    assert client.studio.caller.calls == []


class BrokeOn:
    """The account has no balance left for one model (Pollinations 402)."""
    def __init__(self, broke_key):
        self.broke, self.calls = broke_key, []

    def __call__(self, model, prompt, picture):
        self.calls.append(model.key)
        if model.key == self.broke:
            raise sketch.SketchError("The AI service limit is reached. Try again later.", 429, "balance")
        return tile(4), "image/png"


def test_low_balance_with_auto_finishes_on_the_cheapest_model(tmp_path, monkeypatch):
    monkeypatch.setenv("POLLINATIONS_KEY", "sk_x")
    jobs = V.Jobs(sketch.Studio(tmp_path, caller=BrokeOn("p-gptimage")), planner=FakePlanner())
    j = jobs.start("u1", pic(), 9, "", "illustration", "p-gptimage", wait=True, auto=True)
    assert j["status"] == "done", j
    r = j["result"]
    assert r["meta"]["n"] == 9 and r["meta"]["fell_back"] is True and r["model"] == "p-klein"
    assert jobs.studio.caller.calls.count("p-klein") == 3
    assert jobs.studio.spent(0)[0] == pytest.approx(3 * sketch.BY_KEY["p-klein"].usd + sketch.PLANNER_USD)


def test_low_balance_with_a_picked_model_stops_with_a_plain_message(tmp_path, monkeypatch):
    monkeypatch.setenv("POLLINATIONS_KEY", "sk_x")
    jobs = V.Jobs(sketch.Studio(tmp_path, caller=BrokeOn("p-gptimage")), planner=FakePlanner())
    j = jobs.start("u1", pic(), 4, "", "illustration", "p-gptimage", wait=True)
    assert j["status"] == "error" and "limit" in j["error"] and "pollen" not in j["error"].lower()
    assert jobs.studio.caller.calls == ["p-gptimage"]   # no retry loop on an empty balance



def test_variations_default_to_our_server_when_installed(monkeypatch):
    monkeypatch.setattr(sketch, "_key", lambda p: "on" if p in ("local", "pollinations") else "")
    assert V.default_model() == "local-klein"       # the fast size first (HD stays selectable)
    monkeypatch.setattr(sketch, "_key", lambda p: "on" if p == "pollinations" else "")
    assert V.default_model() == "p-gptimage"


def test_low_balance_falls_back_to_our_server_first(tmp_path, monkeypatch):
    monkeypatch.setattr(sketch, "_key", lambda p: "on" if p in ("local", "pollinations") else "")
    jobs = V.Jobs(sketch.Studio(tmp_path, caller=BrokeOn("p-gptimage")), planner=FakePlanner())
    r = jobs.start("u1", pic(), 4, "", "illustration", "p-gptimage", wait=True, auto=True)["result"]
    assert r["model"] == "local-klein" and jobs.studio.caller.calls == ["p-gptimage", "local-klein"]



# ---------- planning without Pollinations: our photo reading + the local language model ----------

RING_DNA = {"type": {"value": "ring", "sure": True}, "metal": {"metal": "yellow_gold"},
            "traits": [{"group": "Stones", "label": "No stones"}, {"group": "Band", "label": "Wide band"},
                       {"group": "Look", "label": "Textured"}],
            "diamonds": {}, "motifs": [{"label": "Organic"}], "details": []}


class FakeEngine:
    def __init__(self, dna=RING_DNA):
        self.dna = dna

    def read_photo(self, im):
        from types import SimpleNamespace
        return SimpleNamespace(dna=self.dna)


class FakeWriter:
    def __init__(self):
        self.prompts, self.k = [], 0

    def __call__(self, prompt, max_tokens):
        self.prompts.append(prompt)
        n = int(re.search(r"exactly (\d+)", prompt).group(1))
        out = []
        for _ in range(n):
            self.k += 1
            out.append({"name": f"Band idea {self.k}", "brief": f"wide band change {self.k}"})
        return json.dumps({"piece": "x", "ideas": out})


def test_read_piece_describes_type_metal_and_no_stones():
    info = V.read_piece(FakeEngine(), sketch_png())
    assert info["kind"] == "ring" and info["stones"] is False
    assert info["piece"].startswith("a yellow gold ring") and "no gemstones" in info["piece"] and "wide band" in info["piece"]


def test_without_credit_the_local_model_plans_from_the_photo_reading():
    w = FakeWriter()
    piece, ideas, who = V.plan_ideas(sketch_png(), 9, "realistic", ["Old idea"], planner=FakePlanner(fail=True),
                                     engine=FakeEngine(), writer=w)
    assert who == "local" and len(ideas) == 9 and ideas[0]["name"] == "Band idea 1"
    assert "a yellow gold ring" in w.prompts[0] and "Old idea" in w.prompts[0] and "a ring stays a ring" in w.prompts[0]
    assert len(w.prompts) == 2   # asked in batches of up to 6
    assert "Band idea 1" in w.prompts[1]   # the second batch avoids the first


def test_no_balance_does_not_ask_pollinations_twice():
    p = FakePlanner(fail=True)
    p.fail = False
    calls = []

    def broke(prompt, picture, mt):
        calls.append(1)
        raise sketch.SketchError("limit", 429, "balance")
    V.plan_ideas(sketch_png(), 4, "", [], planner=broke, engine=FakeEngine(), writer=FakeWriter())
    assert len(calls) == 1


def test_random_fallback_keeps_the_kind_of_piece():
    piece, ideas, who = V.plan_ideas(sketch_png(), 4, "", [], planner=FakePlanner(fail=True), engine=FakeEngine())
    assert who == "random" and all("the same ring" in i["brief"] for i in ideas)


def test_stone_free_piece_gets_the_no_stones_rule_and_realistic_words_win():
    ideas = [{"name": f"N{i}", "brief": "b"} for i in range(4)]
    p = V.tile_prompt(ideas, "make realistic real life ring design", "illustration", "a yellow gold ring, no gemstones", False)
    assert "do not add any" in p and "same gemstones" not in p and "photorealistic" in p and "illustration" not in p
    assert V.effective_look("photo", "pencil sketch please") == "pencil"
    assert V.effective_look("illustration", "") == "illustration"


def test_job_uses_local_planning_and_stone_rule(tmp_path, monkeypatch):
    monkeypatch.setattr(V, "RETRY_WAIT", 0)
    jobs = V.Jobs(sketch.Studio(tmp_path, caller=FakeAI()), planner=FakePlanner(fail=True),
                  engine=FakeEngine(), writer=FakeWriter())
    r = jobs.start("u1", pic(), 4, "make realistic ring", "illustration", "local-klein-hd", wait=True)["result"]
    assert r["meta"]["planned"] == "local" and r["meta"]["look"] == "photo"
    prompt = jobs.studio.caller.calls[0][1]
    assert "no gemstones" in prompt and "do not add any" in prompt and "wide band change 1" in prompt



def test_local_writer_finds_the_judge_behind_its_cache():
    from functools import lru_cache
    from types import SimpleNamespace

    class J:
        model = tok = dev = gpu = torch = object()

        def __init__(self):
            self.ask = lru_cache(maxsize=8)(self._ask)

        def _ask(self, p):
            return "yes"
    eng = SimpleNamespace(domain=SimpleNamespace(_judge=J().ask))
    assert V.local_writer(eng) is not None
    assert V.local_writer(SimpleNamespace(domain=SimpleNamespace(_judge=None))) is None



def test_parse_plan_takes_pairs_from_almost_json():
    text = ('{"piece": "A ring.", "ideas": [{"name": "Eclipse Flow", "brief": "organic flow"}], '
            '[{"name": "Ripple Flow", "brief": "wavy band"}], [{"name": "Lunar Tide", "brief": "crescent')
    piece, ideas = V._parse_plan(text, 2, keep_all=True)
    assert piece == "A ring." and [i["name"] for i in ideas] == ["Eclipse Flow", "Ripple Flow"]
