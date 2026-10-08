#!/usr/bin/env python3
"""Embed every view of every design with DINOv2, for search by photo (jewelsearch/dino.py).

Reads  data/index/views.json (the views and their order) and data/crops/ (no dataset drive)
Writes data/index/dino_views.npy   (n_views, 768) float32, rows in the order of views.json
       data/index/dino_views.json  {"key", "model", "size", "rows"}

Run it after scripts/build_view_index.py (which runs after build_index.py); about
25 minutes on an M2. The search ignores this file when the index, the model or
the size changed, so a stale file can never attach a view to the wrong design.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from jewelsearch.config import CROPS, INDEX, thumb_name  # noqa: E402
from jewelsearch.dino import MODEL, SIZE, Dino  # noqa: E402

BATCH = 64


def main():
    meta = [json.loads(l) for l in (INDEX / "meta.jsonl").read_text().splitlines()]
    info = json.loads((INDEX / "views.json").read_text())
    paths = [CROPS / thumb_name(meta[r]["images"][meta[r]["embed_metal"]][v]) for r, v in zip(info["design"], info["view"])]
    dino = Dino()
    out = np.zeros((len(paths), dino.dim), dtype=np.float32)
    t0 = time.time()
    for i in range(0, len(paths), BATCH):
        out[i:i + BATCH] = dino.images([Image.open(p) for p in paths[i:i + BATCH]])
        if (i // BATCH) % 25 == 0:
            print(f"{i}/{len(paths)} views  {time.time() - t0:.0f}s", flush=True)
    np.save(INDEX / "dino_views.tmp.npy", out)
    (INDEX / "dino_views.json.tmp").write_text(json.dumps({"key": info["key"], "model": MODEL, "size": SIZE, "rows": len(paths)}))
    (INDEX / "dino_views.tmp.npy").replace(INDEX / "dino_views.npy")
    (INDEX / "dino_views.json.tmp").replace(INDEX / "dino_views.json")
    print(f"embedded {len(paths)} views in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
