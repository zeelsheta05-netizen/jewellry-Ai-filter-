#!/usr/bin/env python3
"""Teach search by photo to find the piece in a busy photo (worn on a hand, next to a
face, on a table with other things).

The background cut-out (photo.py) only works when the piece stands out from a plain
background. Shop "lifestyle" photos and photos of a piece being worn fail it, and then
the whole photo (face, fingers, silk, tweezers) was compared with the catalogue: on
240 such photos the design came first 3.8% of the time.

This trains a single logistic layer (a "linear probe") on DINOv2's patch vectors
(dino.py, one vector per 14 px patch of the photo at 336 px): is this patch part of a
piece of jewellery? Training photos are made by pasting catalogue crops (transparent
PNG/WebP, so every patch's answer is known) onto ordinary photos of people, hands and
surfaces. The background photos must be your own or freely licensed (Wikimedia Commons
was used); they are only read here, never shown or stored with the index.

  .venv/bin/python scripts/train_piece_finder.py --backgrounds data/piece_finder/backgrounds [--photos 2400]

Writes data/index/piece_finder.npz (index-fingerprinted like diamond_dna.npz).
A fifth of the background photos is kept aside to check the result, and the report
gives how often the found box is on the piece and how often a photo with no piece
gets a box anyway.
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from jewelsearch import photo  # noqa: E402
from jewelsearch.config import CROPS, INDEX, thumb_name  # noqa: E402
from jewelsearch.dino import GRID, MODEL, SIZE, Dino  # noqa: E402
from jewelsearch.search import index_key  # noqa: E402

SCALE = 10.0        # patch vectors are unit length; the layer sees them x10 (as the diamond probe does)
POS_COVER = 0.25    # a patch at least this much covered by the piece is "jewellery"
CHUNK = 16          # pictures per model run: fixed small batches keep GPU memory flat


def composite(bg: Image.Image | None, crops: list[Image.Image], rng) -> tuple[Image.Image, np.ndarray]:
    """Pieces pasted on a photo (or a plain studio background) -> (photo, piece mask 0..1)."""
    long = int(rng.integers(640, 1200))
    if bg is None:   # studio-like: plain or gradient
        c1 = np.array(rng.integers(150, 256, 3), np.float32)
        c2 = np.clip(c1 + rng.normal(0, 25, 3), 0, 255)
        w, h = (long, int(long * rng.uniform(0.6, 1.0))) if rng.random() < 0.5 else (int(long * rng.uniform(0.6, 1.0)), long)
        t = np.linspace(0, 1, h)[:, None, None]
        canvas = Image.fromarray(((1 - t) * c1 + t * c2).repeat(w, 1).astype(np.uint8))
    else:
        canvas = bg.copy()
        canvas.thumbnail((long, long), Image.LANCZOS)
    w, h = canvas.size
    canvas = canvas.convert("RGBA")
    mask = np.zeros((h, w), np.float32)
    for crop in crops:
        side = int(min(w, h) * rng.uniform(0.10, 0.6 if bg is None else 0.45))
        piece = crop.resize((side, side), Image.LANCZOS).rotate(float(rng.uniform(-30, 30)), resample=Image.BICUBIC)
        x, y = int(rng.integers(0, max(1, w - side))), int(rng.integers(0, max(1, h - side)))
        a = np.asarray(piece, np.float32)[..., 3] / 255
        sh = Image.fromarray((a * 255 * 0.3).astype(np.uint8)).filter(ImageFilter.GaussianBlur(side / 40))
        shadow = Image.new("RGBA", piece.size, (0, 0, 0, 0))
        shadow.putalpha(sh)
        canvas.alpha_composite(shadow, (min(w - side, x + side // 40), min(h - side, y + side // 30)))
        canvas.alpha_composite(piece, (x, y))
        mask[y:y + side, x:x + side] = np.maximum(mask[y:y + side, x:x + side], (a > 0.5).astype(np.float32))
    im = ImageEnhance.Brightness(canvas.convert("RGB")).enhance(float(rng.uniform(0.8, 1.15)))
    im = ImageEnhance.Contrast(im).enhance(float(rng.uniform(0.85, 1.15)))
    arr = np.asarray(im, np.float32)
    warm = rng.uniform(-0.07, 0.07)
    arr[..., 0] *= 1 + warm
    arr[..., 2] *= 1 - warm
    im = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    if rng.random() < 0.6:
        im = im.filter(ImageFilter.GaussianBlur(float(rng.uniform(0.3, 1.3))))
    return im, mask


def grid_cover(im: Image.Image, mask: np.ndarray) -> tuple[Image.Image, np.ndarray, tuple]:
    """The square the model sees, and how much of each of its GRID x GRID patches the piece covers."""
    sq, box, _ = photo.square_full(im)
    side = box[2] - box[0]
    full = np.zeros((side, side), np.float32)
    x0, y0 = -box[0], -box[1]
    full[y0:y0 + mask.shape[0], x0:x0 + mask.shape[1]] = mask
    cover = np.asarray(Image.fromarray(full).resize((GRID, GRID), Image.BOX), np.float32)
    return sq, cover, box


def make_set(dino, bgs, crop_paths, n, rng, keep_all=False):
    """n training photos -> sampled patch rows (X, y), plus per-photo heat maps' inputs for checking."""
    X, y, checks = [], [], []
    batch = []

    def flush():
        feats = dino.patches([b[0] for b in batch])
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        for (sq, cover, box, has, im_size, true_box), f in zip(batch, feats):
            c = cover.ravel()
            pos = np.flatnonzero(c >= POS_COVER)
            neg = np.flatnonzero(c == 0)
            if not keep_all:
                neg = rng.choice(neg, size=min(len(neg), max(3 * len(pos), 40)), replace=False)
            rows = np.r_[pos, neg]
            X.append(f[rows].astype(np.float16))
            y.append(np.r_[np.ones(len(pos)), np.zeros(len(neg))].astype(np.float32))
            if keep_all:
                checks.append({"feats": f.astype(np.float16), "box": box, "has": has, "true": true_box, "size": im_size})
        batch.clear()

    for i in range(n):
        kind = rng.random()
        bg = None if kind < 0.15 else bgs[int(rng.integers(len(bgs)))]
        k = 0 if 0.15 <= kind < 0.25 else (2 if rng.random() < 0.2 else 1)
        crops = [Image.open(crop_paths[int(rng.integers(len(crop_paths)))]).convert("RGBA") for _ in range(k)]
        im, mask = composite(bg, crops, rng)
        sq, cover, box = grid_cover(im, mask)
        ys, xs = np.nonzero(mask > 0)
        true_box = (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1) if len(xs) else None
        batch.append((sq, cover, box, k > 0, im.size, true_box))
        if len(batch) == CHUNK:
            flush()
        if i % 400 == 0:
            print(f"  {i}/{n}", flush=True)
    if batch:
        flush()
    return np.concatenate(X), np.concatenate(y), checks


def train(X, y, epochs=300):
    X, y = torch.tensor(X.astype(np.float32)) * SCALE, torch.tensor(y)
    pos_weight = torch.tensor([(y == 0).sum() / max(1, (y == 1).sum())]) ** 0.5
    layer = torch.nn.Linear(X.shape[1], 1)
    opt = torch.optim.Adam(layer.parameters(), lr=0.01, weight_decay=1e-4)
    for _ in range(epochs):
        opt.zero_grad()
        torch.nn.functional.binary_cross_entropy_with_logits(layer(X)[:, 0], y, pos_weight=pos_weight).backward()
        opt.step()
    return layer.weight.detach().numpy()[0].astype(np.float32), float(layer.bias.detach().numpy()[0])


def on_piece(box, true) -> bool:
    """The found box holds at least half of the piece's box, and the piece is not lost in it
    (at least 1/8 of the found box). Plain overlap (IoU) undercounts a small piece, whose box
    is padded to a square and rounded to whole 14 px patches."""
    ix = max(0, min(box[2], true[2]) - max(box[0], true[0]))
    iy = max(0, min(box[3], true[3]) - max(box[1], true[1]))
    t = (true[2] - true[0]) * (true[3] - true[1])
    b = (box[2] - box[0]) * (box[3] - box[1])
    return ix * iy >= 0.5 * t and t >= b / 8


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--backgrounds", required=True, help="folder of ordinary photos (jpg/png) with no jewellery in them")
    ap.add_argument("--photos", type=int, default=2400)
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--exclude", help="tests.json of a test set whose designs must not be used here")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    meta = [json.loads(l) for l in (INDEX / "meta.jsonl").read_text().splitlines()]
    skip = {t["uid"] for t in json.load(open(args.exclude))} if args.exclude else set()
    crop_paths = []
    for u, m in enumerate(meta):
        if u in skip:
            continue
        for metal, views in m["images"].items():
            for v in views.values():
                p = CROPS / thumb_name(v)
                if p.exists():
                    crop_paths.append(p)
    random.Random(args.seed).shuffle(crop_paths)
    files = sorted(p for p in Path(args.backgrounds).iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    random.Random(args.seed).shuffle(files)
    cut = max(1, len(files) // 5)
    held, used = files[:cut], files[cut:]
    load = lambda fs: [Image.open(f).convert("RGB") for f in fs]
    print(f"{len(crop_paths)} catalogue crops, {len(used)} background photos to learn from, {len(held)} kept aside")
    dino = Dino()
    t0 = time.time()
    X, y, _ = make_set(dino, load(used), crop_paths, args.photos, rng)
    print(f"{len(y)} patches ({int(y.sum())} on a piece) in {time.time() - t0:.0f}s")
    W, b = train(X, y)
    # check on the photos kept aside (unseen backgrounds)
    _, _, checks = make_set(dino, load(held), crop_paths, max(200, args.photos // 6), rng, keep_all=True)
    report = {}
    for thr in (0.5, 0.6, 0.7, 0.8, 0.9):
        hits = with_piece = false_box = without = 0
        for c in checks:
            heat = 1 / (1 + np.exp(-((c["feats"].astype(np.float32) * SCALE) @ W + b)))
            cells = photo.heat_box(heat.reshape(GRID, GRID), thr)
            if c["has"]:
                with_piece += 1
                if cells is not None:
                    box = photo.grid_to_photo(cells, GRID, c["box"])
                    hits += on_piece(box, c["true"])
            else:
                without += 1
                false_box += cells is not None
        report[thr] = {"box on the piece": round(hits / max(1, with_piece), 3),
                       "box on a photo with no piece": round(false_box / max(1, without), 3)}
        print(f"threshold {thr}: box on the piece {hits / max(1, with_piece):5.1%} of {with_piece}   "
              f"box on a photo with no piece {false_box / max(1, without):5.1%} of {without}")
    np.savez(INDEX / "piece_finder.npz", key=np.array(index_key(meta)), model=np.array(MODEL), size=SIZE,
             W=W, b=b, scale=SCALE, report=np.array(json.dumps(report)))
    print(f"saved {INDEX / 'piece_finder.npz'}")


if __name__ == "__main__":
    main()
