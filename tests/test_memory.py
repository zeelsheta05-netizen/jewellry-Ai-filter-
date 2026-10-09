"""Big idle models let go of their memory while the local picture model draws (jewelsearch/memory.py)."""
import time

import pytest

from jewelsearch import domain, memory


class FakeModel:
    def __init__(self):
        self.loaded, self.events = True, []

    def release(self):
        self.events.append("release")
        was, self.loaded = self.loaded, False
        return was

    def load(self):
        self.events.append("load")
        was, self.loaded = self.loaded, True
        return not was


@pytest.fixture
def fresh(monkeypatch):
    monkeypatch.setattr(memory, "_models", [])
    monkeypatch.setattr(memory, "RELOAD_AFTER_S", 0.05)
    monkeypatch.delenv("LOCAL_RELEASE_MODELS", raising=False)
    memory.DRAWING.clear()
    yield
    memory.DRAWING.clear()


def test_models_let_go_while_drawing_and_come_back_after(fresh):
    m = FakeModel()
    memory.register(m)
    memory.register(m)                      # once only
    memory.drawing_starts()
    assert memory.DRAWING.is_set() and not m.loaded and m.events == ["release"]
    memory.drawing_ends()
    assert not memory.DRAWING.is_set()
    for _ in range(50):
        if m.loaded:
            break
        time.sleep(0.02)
    assert m.loaded and m.events == ["release", "load"]


def test_back_to_back_drawings_reload_once_after_the_last(fresh):
    m = FakeModel()
    memory.register(m)
    memory.drawing_starts()
    memory.drawing_ends()
    memory.drawing_starts()                 # the next picture starts before the reload
    time.sleep(0.15)
    assert not m.loaded                     # the first drawing's reload stood down
    memory.drawing_ends()
    time.sleep(0.15)
    assert m.loaded and m.events.count("load") == 1


def test_switch_off(fresh, monkeypatch):
    monkeypatch.setenv("LOCAL_RELEASE_MODELS", "0")
    m = FakeModel()
    memory.register(m)
    memory.drawing_starts()
    assert m.loaded
    memory.drawing_ends()


class AwayJudge:
    """Stands in for domain.Judge.ask while the judge has let go of its memory."""
    def __init__(self):
        self.away, self.calls = True, 0

    def __call__(self, prompt):
        self.calls += 1
        if self.away:
            raise domain.JudgeAway()
        return "other"


class Engine:
    def picture_score(self, text, q):
        return 0.0


def test_search_screening_while_the_judge_is_away_is_not_cached():
    judge = AwayJudge()
    d = domain.Domain(Engine(), judge=judge)
    prompt = "something unusual gold ring for my sister"
    v = d.check(prompt)
    assert v.ok and v.via == "picture"          # words + picture check meanwhile
    judge.away = False
    v = d.check(prompt)                          # not the cached drawing-time answer
    assert not v.ok and v.via == "judge" and judge.calls == 2
