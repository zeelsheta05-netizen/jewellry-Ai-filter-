"""Keeping the search to jewellery (jewelsearch/domain.py).

The routing is tested with a stand-in judge and picture check (fast). The
search itself runs against the built index with a scripted judge, and a few
clear prompts go through the real local judge when it is downloaded.
Full measurement: scripts/eval_domain.py.
Run: .venv/bin/python -m pytest tests/test_domain.py -q
"""
from pathlib import Path

import pytest

from jewelsearch import domain
from jewelsearch.config import INDEX

needs_index = pytest.mark.skipif(not (INDEX / "front.npy").exists(), reason="index not built")
JUDGE_CACHED = any((Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots").glob("*/config.json"))


class FakeEngine:
    def __init__(self, score=0.0):
        self.score, self.seen = score, []

    def picture_score(self, text, q):
        self.seen.append(text)
        return self.score


class FakeJudge:
    def __init__(self, answer="yes"):
        self.answer, self.seen = answer, []

    def __call__(self, text):
        self.seen.append(text)
        return self.answer


def make(answer="yes", score=0.0):
    judge, engine = FakeJudge(answer), FakeEngine(score)
    return domain.Domain(engine, judge=judge), judge, engine


# ---- routing (no models) -------------------------------------------------------

@pytest.mark.parametrize("prompt", ["pizza", "weather today", "hello", "123456", "asdfgh", "આજનું હવામાન",
                                    "मौसम कैसा है", "how to learn java", "   "])
def test_no_jewellery_word_is_refused_without_asking_the_judge(prompt):
    d, judge, _ = make("yes")
    v = d.check(prompt)
    assert not v.ok and v.reason == "nothing" and v.via == "words"
    assert judge.seen == []


@pytest.mark.parametrize("prompt", ["rose gold ring for my mother", "wedding", "for my mother", "jhumka",
                                    "મારી બહેન માટે નાની અને સાદી કાનની બુટ્ટી", "शादी के लिए भारी हीरे का हार",
                                    "neckless for my wife", "gift for my dad"])
def test_prompts_made_of_known_words_are_accepted_without_the_judge(prompt):
    d, judge, engine = make("no", score=-99)
    v = d.check(prompt)
    assert v.ok and v.via == "words"
    assert judge.seen == [] and engine.seen == []


def test_unknown_words_go_to_the_judge_with_piece_names_in_english():
    d, judge, engine = make("yes")
    assert d.check("heavy haar for reception").ok
    assert judge.seen == ["heavy necklace for reception"]
    d, judge, _ = make("yes")
    d.check("bali for daily use")
    assert judge.seen == ["earrings for daily use"]


@pytest.mark.parametrize("answer,reason", [("no", "not_jewellery"), ("other", "other_type")])
def test_the_judge_refuses(answer, reason):
    d, _, engine = make(answer)
    v = d.check("boxing ring")
    assert not v.ok and v.reason == reason and v.via == "judge"
    assert engine.seen == []   # no picture check needed


def test_the_picture_check_overrules_a_yes():
    d, _, _ = make("yes", score=domain.PICTURE_MIN - 0.1)
    v = d.check("mehndi design for bride")
    assert not v.ok and v.reason == "no_match" and v.via == "picture"
    d, _, _ = make("yes", score=domain.PICTURE_MIN + 0.1)
    assert d.check("mehndi design for bride").ok


def test_without_a_judge_the_picture_check_decides():
    d = domain.Domain(FakeEngine(-9), judge=None)
    assert not d.check("boxing ring").ok
    d = domain.Domain(FakeEngine(0), judge=None)
    v = d.check("snake ring")
    assert v.ok and v.via == "picture"


@pytest.mark.parametrize("prompt,clean,budget", [
    ("rose gold ring under 30000", "rose gold ring", "under 30000"),
    ("earrings below 50k for my wife", "earrings for my wife", "below 50k"),
    ("pendant rs 20,000", "pendant", "rs 20,000"),
    ("bracelet 25000 rs", "bracelet", "25000 rs"),
    ("rose gold ring", "rose gold ring", None),
    ("18k rose gold ring", "18k rose gold ring", None),     # a gold purity, not a budget
    ("22K bangle", "22K bangle", None),
])
def test_a_budget_is_taken_out_not_refused(prompt, clean, budget):
    d, judge, _ = make("no")
    v = d.check(prompt)
    assert v.ok and v.text == clean and v.budget == budget
    assert judge.seen == []


def test_refusal_messages():
    d, _, _ = make("no")
    r = d.check("pizza").refusal("pizza")
    assert r["title"] == "Can't find designs like “pizza”" and "only this jewellery collection" in r["hint"]
    r = d.check("silver price").refusal("silver price")
    assert "not prices" in r["hint"]
    d, _, _ = make("other")
    assert "rings, earrings, pendants" in d.check("toe ring").refusal("toe ring")["hint"]
    long = "pizza " * 30
    assert d.check(long).refusal(long)["title"].endswith("…”")


# ---- the search (index) ----------------------------------------------------------

def scripted(text):
    """A stand-in for the local judge, for the search tests."""
    return "no" if any(w in text for w in ("boxing", "onion", "doorbell", "dress", "cake")) else "yes"


@pytest.fixture(scope="module")
def engine():
    from jewelsearch.search import SearchEngine
    return SearchEngine(judge=scripted)


@needs_index
@pytest.mark.parametrize("prompt", ["pizza", "weather today", "boxing ring", "wedding dress", "hello"])
def test_search_refuses_with_a_message_and_no_designs(engine, prompt):
    res = engine.search(prompt)
    assert res["results"] == [] and res["matches_in_filter"] == 0 and not res["has_more"]
    assert res["refused"]["title"] == f"Can't find designs like “{prompt}”"
    assert res["query"]["refused"] == res["refused"]   # kept with the saved search, so history shows it too


@needs_index
def test_search_with_a_budget_shows_designs_and_says_so(engine):
    res = engine.search("rose gold ring under 30000")
    assert "refused" not in res and len(res["results"]) == 8
    assert {c["category"] for c in res["results"]} == {"ring"}
    assert any("under 30000" in n for n in res["notes"])


@needs_index
def test_design_id_is_looked_up_never_refused(engine):
    m = next(m for m in engine.meta if "-" in m["design_id"])
    res = engine.search(m["design_id"].lower().replace("-", " "))
    assert "refused" not in res
    assert m["uid"] in [c["uid"] for c in res["results"]]


@needs_index
def test_picture_check_separates_look_alikes(engine):
    from jewelsearch.query import parse
    assert engine.picture_score("a boxing ring", parse("boxing ring")) < domain.PICTURE_MIN
    assert engine.picture_score("Plain gold band for daily use", parse("Plain gold band for daily use")) > domain.PICTURE_MIN


@needs_index
def test_photo_search_leaves_out_words_that_are_not_jewellery(engine):
    from PIL import Image

    from jewelsearch.config import CROPS
    pq = engine.read_photo(Image.open(next(CROPS.glob("*.webp"))))
    res = engine.search_photo(pq, "pizza near me")
    assert len(res["results"]) == 8
    assert any("pizza near me" in n and "photo alone" in n for n in res["notes"])
    assert res["query"]["raw"] == ""


@pytest.fixture(scope="module")
def real_engine():
    from jewelsearch.search import SearchEngine
    return SearchEngine()


@needs_index
@pytest.mark.skipif(not JUDGE_CACHED, reason="the local judge (Qwen3-1.7B) is not downloaded")
@pytest.mark.parametrize("prompt,ok", [
    ("boxing ring", False), ("onion rings", False), ("heart attack symptoms", False), ("silver price", False),
    ("Plain gold band for daily use", True), ("snake ring", True), ("heavy haar for reception", True),
    ("pretty ring for my girlfriend", True),
])
def test_real_judge_on_clear_prompts(real_engine, prompt, ok):
    assert real_engine.domain.check(prompt).ok is ok
