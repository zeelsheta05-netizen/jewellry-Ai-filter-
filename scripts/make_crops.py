#!/usr/bin/env python3
"""Tight crops of every render, plus per-image view statistics.

Pendants and necklaces are rendered small inside a 2600px frame, so a plain
downscale leaves the piece a few dozen pixels wide - too small for the model
to see stones or motifs, and poor for display. Each original is cropped to
the jewellery's bounding box (from the alpha channel, or from non-white
pixels for the few opaque renders), padded to a square and saved as a 512px
WebP in data/crops/ under the same name as its thumbnail.

Alongside, data/crops/stats.jsonl records per image:
  sym   left-right mirror overlap of the silhouette (front views score high)
  fill  fraction of the original frame the piece occupies
  ink   fraction of the crop covered by the piece
Re-running skips images already processed.
"""
import argparse
import io
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from jewelsearch import storage  # noqa: E402
from jewelsearch.config import CATALOG, CROPS, INDEX, thumb_name  # noqa: E402

SIZE = 512
PAD = 0.06


def mask_of(im: Image.Image) -> np.ndarray:
    a = np.asarray(im)
    alpha = a[..., 3]
    if (alpha < 250).mean() > 0.02:          # transparent background
        return alpha > 128                   # >128 drops the soft drop-shadow
    return a[..., :3].min(axis=2) < 238      # opaque render on white


def body_mask(m: np.ndarray) -> np.ndarray:
    """Drop thin structures (the chain) so a pendant crop centres on the pendant."""
    eroded = np.asarray(Image.fromarray(m.astype(np.uint8) * 255).filter(ImageFilter.MinFilter(11))) > 0
    return eroded if eroded.sum() > 50 else m


def work(job):
    rel, category = job
    dst = CROPS / thumb_name(rel)
    try:
        # read from the dataset storage by name (the original; the web copy if
        # the original isn't moved yet)
        with Image.open(io.BytesIO(storage.get().read_bytes(rel))) as im:
            im = im.convert("RGBA")
        small = im.resize((im.width // 4, im.height // 4))
        m = mask_of(small)
        if not m.any():
            return {"path": rel, "sym": 0.0, "fill": 0.0, "ink": 0.0}
        if category == "pendant":
            m = body_mask(m)
        ys, xs = np.nonzero(m)
        y0, y1, x0, x1 = ys.min() * 4, (ys.max() + 1) * 4, xs.min() * 4, (xs.max() + 1) * 4
        side = int(max(y1 - y0, x1 - x0) * (1 + 2 * PAD))
        cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
        box = (cx - side // 2, cy - side // 2, cx - side // 2 + side, cy - side // 2 + side)
        crop = Image.new("RGBA", (side, side), (0, 0, 0, 0))
        crop.paste(im.crop(box), (0, 0))  # crop() pads out-of-frame areas with transparency
        crop = crop.resize((SIZE, SIZE), Image.LANCZOS)
        if not dst.exists():
            tmp = dst.with_suffix(".tmp")
            crop.save(tmp, "WEBP", quality=85, method=4)
            tmp.rename(dst)
        cm = mask_of(crop.resize((128, 128)))
        sym = float((cm & cm[:, ::-1]).sum() / max((cm | cm[:, ::-1]).sum(), 1))
        return {"path": rel, "sym": round(sym, 4),
                "fill": round(float((x1 - x0) * (y1 - y0) / (im.width * im.height)), 4),
                "ink": round(float(cm.mean()), 4)}
    except Exception as exc:
        print(f"fail {rel}: {exc}", file=sys.stderr)
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    CROPS.mkdir(parents=True, exist_ok=True)
    stats_path = CROPS / "stats.jsonl"
    done = set()
    if stats_path.exists():
        done = {json.loads(l)["path"] for l in stats_path.read_text().splitlines()}

    jobs = []
    # category from the index where available (it includes model-filled ones)
    cats = {}
    if (INDEX / "meta.jsonl").exists():
        cats = {json.loads(l)["design_id"]: json.loads(l)["category"]
                for l in (INDEX / "meta.jsonl").read_text().splitlines()}
    for line in CATALOG.read_text().splitlines():
        rec = json.loads(line)
        cat = cats.get(rec["design_id"], rec["category"])
        for views in rec["images"].values():
            jobs.extend((p, cat) for p in views.values() if p not in done)

    t0 = time.time()
    with open(stats_path, "a") as out, ProcessPoolExecutor(args.workers) as pool:
        for i, r in enumerate(pool.map(work, jobs, chunksize=8), 1):
            if r:
                out.write(json.dumps(r) + "\n")
            if i % 2000 == 0:
                out.flush()
                print(f"{i}/{len(jobs)}  {time.time() - t0:.0f}s", flush=True)
    print(f"done {len(jobs)} images in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
