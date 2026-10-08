#!/usr/bin/env python3
"""Embed every view of every design, for photo search.

The main index (build_index.py) keeps one vector per design (the mean of its
views) and its front view. A customer's photo can show any angle, so photo
search also compares it with each view on its own and keeps the best one.

Reads  data/index/meta.jsonl and data/crops/ (no dataset drive needed)
Writes data/index/views.npy    (n_views, d) float32, rows grouped by design
       data/index/views.json   {"key": ..., "design": [row in meta per view], "view": [view name]}

"key" fingerprints meta.jsonl's design order. The search engine ignores the
views when the key no longer matches (the main index was rebuilt), so a stale
file can never attach a view to the wrong design. Run it after every
build_index.py run (about 10 minutes on an M2).
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from jewelsearch.config import CROPS, INDEX, thumb_name  # noqa: E402
from jewelsearch.embedder import Embedder  # noqa: E402
from jewelsearch.search import index_key  # noqa: E402

BATCH = 64


def main():
    meta = [json.loads(l) for l in (INDEX / "meta.jsonl").read_text().splitlines()]
    jobs = []
    for row, m in enumerate(meta):
        views = m["images"][m["embed_metal"]]
        for v in sorted(views):
            path = CROPS / thumb_name(views[v])
            if path.exists():
                jobs.append((row, v, path))
    emb = Embedder()
    out = np.zeros((len(jobs), emb.texts(["x"]).shape[1]), dtype=np.float32)
    t0 = time.time()
    for i in range(0, len(jobs), BATCH):
        chunk = jobs[i:i + BATCH]
        out[i:i + len(chunk)] = emb.images([Image.open(p) for _, _, p in chunk])
        if (i // BATCH) % 25 == 0:
            print(f"{i}/{len(jobs)} views  {time.time() - t0:.0f}s", flush=True)
    tmp = INDEX / "views.tmp.npy"
    np.save(tmp, out)
    (INDEX / "views.json.tmp").write_text(json.dumps(
        {"key": index_key(meta), "design": [j[0] for j in jobs], "view": [j[1] for j in jobs]}))
    tmp.replace(INDEX / "views.npy")
    (INDEX / "views.json.tmp").replace(INDEX / "views.json")
    print(f"embedded {len(jobs)} views of {len(meta)} designs in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
