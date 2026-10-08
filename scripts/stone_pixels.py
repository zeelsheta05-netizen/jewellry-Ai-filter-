#!/usr/bin/env python3
"""Measure how much of each design is diamond, by colour.

On a yellow- or rose-gold render the metal is strongly coloured while
diamonds are colourless and bright, so the share of low-saturation bright
pixels inside the piece is a direct stone measure - far more reliable than
asking the image model "does it have stones?", which misses a few small
stones. (White-gold renders cannot be measured this way; designs available
only in white gold keep the model's estimate.)

Adds "stone_frac" to every record in data/index/meta.jsonl.
"""
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from jewelsearch.config import CROPS, INDEX, thumb_name  # noqa: E402

COLOURED = ["yellow_gold", "rose_gold"]


def stone_frac(m: dict):
    metal = next((mt for mt in COLOURED if mt in m["images"]), None)
    if metal is None:
        return None
    views = m["images"][metal]
    path = views.get(m.get("front_view")) or views[sorted(views)[-1]]
    p = CROPS / thumb_name(path)
    if not p.exists():
        return None
    im = Image.open(p).convert("RGBA")
    a = np.asarray(im)
    piece = a[..., 3] > 128
    if piece.sum() < 200:
        return None
    hsv = np.asarray(im.convert("RGB").convert("HSV")).astype(np.float32) / 255
    stone = piece & (hsv[..., 1] < 0.06) & (hsv[..., 2] > 0.7)
    return round(float(stone.sum() / piece.sum()), 4)


def main():
    path = INDEX / "meta.jsonl"
    metas = [json.loads(l) for l in path.read_text().splitlines()]
    with ProcessPoolExecutor(8) as pool:
        fracs = list(pool.map(stone_frac, metas, chunksize=32))
    for m, f in zip(metas, fracs):
        m["stone_frac"] = f
    path.write_text("".join(json.dumps(m) + "\n" for m in metas))
    got = [f for f in fracs if f is not None]
    print(f"measured {len(got)}/{len(metas)}; quantiles:",
          {q: round(float(np.quantile(got, q)), 4) for q in (0.05, 0.1, 0.2, 0.5, 0.9)})


if __name__ == "__main__":
    main()
