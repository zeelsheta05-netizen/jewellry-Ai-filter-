"""Sharing this Mac's 16 GB between the live app and the local picture model.

The live app keeps two language models loaded: the details reader (Qwen3-VL-2B,
~4.5 GB, photo search + Design Generator checks) and the search judge
(Qwen3-1.7B, ~3.5 GB). FLUX.2 Klein needs another 6-10 GB while it draws. With
all of them in memory the Mac swapped hard (measured 2026-10-08: swap 15.7 of
16 GB used, a 768 px picture 4-4.7 min instead of ~100 s, an edit ~15 min).

So while a picture is being drawn (sketch.call_local), those two models let go
of their memory. Both have a fallback meanwhile (photo search without details,
search screening by words + picture check), and they load again on their own a
little after the drawing ends, or at their next use.
"""
from __future__ import annotations

import gc
import os
import threading
import time

DRAWING = threading.Event()      # the local picture model is drawing right now
RELOAD_AFTER_S = 20              # quiet time after a drawing before the models come back
_lock = threading.Lock()
_models: list = []               # objects with release() and load()
_gen = 0                         # counts drawings, so only the last one's reload runs


def enabled() -> bool:
    return os.environ.get("LOCAL_RELEASE_MODELS", "1") != "0"


def register(model) -> None:
    """A big model that can let go of its memory: needs release() and load()."""
    with _lock:
        if model not in _models:
            _models.append(model)


def free_device_cache() -> None:
    gc.collect()
    try:
        import torch
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def drawing_starts() -> None:
    global _gen
    with _lock:
        _gen += 1
        models = list(_models) if enabled() else []
    DRAWING.set()
    t = time.time()
    released = [m for m in models if _safe(m.release)]
    if released:
        free_device_cache()
        print(f"memory: let go of {len(released)} model(s) for drawing in {time.time() - t:.1f} s", flush=True)


def drawing_ends() -> None:
    DRAWING.clear()
    with _lock:
        gen = _gen
    threading.Thread(target=_reload_later, args=(gen,), daemon=True).start()


def _reload_later(gen: int) -> None:
    time.sleep(RELOAD_AFTER_S)
    with _lock:
        if gen != _gen or DRAWING.is_set():   # another drawing started: it reloads after itself
            return
        models = list(_models)
    t = time.time()
    loaded = [m for m in models if _safe(m.load)]
    if loaded:
        print(f"memory: models back after drawing in {time.time() - t:.1f} s", flush=True)


def _safe(fn) -> bool:
    try:
        return bool(fn())
    except Exception as e:   # never break a drawing or a search over this
        print(f"memory: {getattr(fn, '__qualname__', fn)} failed ({e!r})", flush=True)
        return False
