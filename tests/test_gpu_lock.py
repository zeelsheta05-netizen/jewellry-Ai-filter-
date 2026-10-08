"""Models used from many threads at once (jewelsearch/config.py GPU).

The live app crashed when two requests ran models at the same moment on the
Mac's GPU ("failed assertion _status < MTLCommandBufferStatusCommitted"): a
crash ends this test run instead of failing one test.
Run: .venv/bin/python -m pytest tests/test_gpu_lock.py -q
"""
import threading

from PIL import Image


def test_models_can_be_used_from_many_threads_at_once():
    from jewelsearch.dino import Dino
    from jewelsearch.embedder import Embedder
    emb, dino = Embedder(), Dino()
    im = Image.new("RGB", (300, 300), (210, 180, 120))
    calls = [lambda: emb.texts(["rose gold ring"]), lambda: emb.images([im]), lambda: dino.images([im, im])]
    errors, done = [], []

    def work(i):
        try:
            for j in range(6):
                calls[(i + j) % len(calls)]()
                done.append(1)
        except Exception as e:   # noqa: BLE001 - reported below
            errors.append(repr(e))
    threads = [threading.Thread(target=work, args=(i,)) for i in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors and len(done) == 36
