#!/usr/bin/env python3
"""Read every design's details (jewelsearch/details.py) from its front view, once.

Reads  data/index/meta.jsonl, data/crops/ (no dataset drive)
Writes data/index/details.npy   (n_designs, n_keys) float32: probability of "yes"; NaN = not read yet
       data/index/details.json  {"key", "model", "side", "keys", "done"}

About 2.2 s a design on an M2 (~2.7 h for the collection). It saves every 100
designs and continues where it stopped when run again. Run it again after
build_index.py; the search ignores readings made for another index or model.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from jewelsearch.config import CROPS, INDEX, thumb_name  # noqa: E402
from jewelsearch.details import MODEL, QUESTIONS, SIDE, DetailReader  # noqa: E402
from jewelsearch.search import index_key  # noqa: E402

KEYS = [k for k, (_, _, matched) in QUESTIONS.items() if matched]   # two-tone: every render is in one metal


def main():
    meta = [json.loads(l) for l in (INDEX / "meta.jsonl").read_text().splitlines()]
    key = index_key(meta)
    npy, js = INDEX / "details.npy", INDEX / "details.json"
    info = json.loads(js.read_text()) if js.exists() else {}
    if npy.exists() and info.get("key") == key and info.get("model") == MODEL and info.get("keys") == KEYS \
            and info.get("side") == SIDE:
        out = np.load(npy)
    else:
        out = np.full((len(meta), len(KEYS)), np.nan, dtype=np.float32)
    reader = DetailReader()

    def save():
        np.save(INDEX / "details.tmp.npy", out)
        (INDEX / "details.tmp.npy").replace(npy)
        js.write_text(json.dumps({"key": key, "model": MODEL, "side": SIDE, "keys": KEYS,
                                  "done": int((~np.isnan(out[:, 0])).sum())}))
    todo = [i for i in range(len(meta)) if np.isnan(out[i, 0])]
    print(f"{len(todo)} of {len(meta)} designs to read", flush=True)
    t0 = time.time()
    for n, i in enumerate(todo, 1):
        m = meta[i]
        views = m["images"][m["embed_metal"]]
        path = CROPS / thumb_name(views[m["front_view"]] if m["front_view"] in views else views[sorted(views)[-1]])
        if path.exists():
            r = reader.read(Image.open(path), KEYS)
            out[i] = [r[k] for k in KEYS]
        if n % 100 == 0 or n == len(todo):
            save()
            rate = (time.time() - t0) / n
            print(f"{n}/{len(todo)}  {rate:.2f}s each, ~{rate * (len(todo) - n) / 60:.0f} min left", flush=True)
    save()
    print("done", flush=True)


if __name__ == "__main__":
    main()
