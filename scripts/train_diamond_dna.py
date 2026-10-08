#!/usr/bin/env python3
"""Teach the search to read a piece's diamonds from its picture.

Two small readers sit on top of the image model's vector (SigLIP2, the same as
the index): one for the centre stone's cut (round, oval, pear, princess, ...),
one for the diamond layout (solitaire / centre + smaller diamonds / even-sized
diamonds). Each is a single softmax layer (a "linear probe"): it learns what an
oval centre stone or a halo looks like to the image model.

Search reads only the picture: a shopper's photo, and every catalogue design's
own renders. The job cards and CAD files (diamond_labels.py) are used here only
as the answers to learn from and to check against.

Training pictures, per design with an answer:
  * every catalogue view in the index (data/index/views.npy)
  * PHOTOS simulated shopper photos (another metal colour, another angle, a
    background, cropped like a real upload; eval_photo_search.fake_photo),
    so the readers also work on phone photos, not only on studio renders

Checked with 5 folds by design family (a design and its stone-cut variants
are never on both sides), on renders and on simulated photos of designs the
reader never saw. The report is printed and saved in the model file.

  .venv/bin/python scripts/train_diamond_dna.py     # ~20 min the first time (simulated photos are cached)

Writes data/index/diamond_dna.npz; the server picks it up at its next start
(and ignores it if the index was rebuilt since: run this again then).
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from diamond_labels import CUTS, LAYOUTS, labels  # noqa: E402
from eval_photo_search import CONDITIONS, fake_photo  # noqa: E402
from jewelsearch import photo  # noqa: E402
from jewelsearch.config import CROPS, INDEX, thumb_name  # noqa: E402
from jewelsearch.search import SearchEngine, index_key  # noqa: E402

OUT = INDEX / "diamond_dna.npz"
SCALE = 10.0           # vectors are unit length; scaled so the layer trains quickly
FOLDS = 5
THRESHOLDS = (0.5, 0.6, 0.7, 0.8)


def simulated_photos(engine, rows, per_design, seed):
    """[(row, vector)] for per_design simulated shopper photos of each design (cached)."""
    cache = INDEX / "diamond_aug.npz"
    key = f"{index_key(engine.meta)}|{per_design}|{seed}"
    if cache.exists():
        data = np.load(cache)
        if data["key"].item() == key:
            return data["rows"], data["vecs"]
    rnd, rng = random.Random(seed), np.random.default_rng(seed)
    out_rows, out_vecs, t0 = [], [], time.time()
    for n, row in enumerate(rows):
        m = engine.meta[row]
        metals = sorted(mt for mt in m["images"] if mt != m["embed_metal"]) or [m["embed_metal"]]
        for k in range(per_design):
            views = m["images"][rnd.choice(metals)]
            crop = CROPS / thumb_name(views[rnd.choice(sorted(views))])
            if not crop.exists():
                continue
            im = photo.read(fake_photo(Image.open(crop).convert("RGBA"), CONDITIONS[(n + k) % 4], rng))
            v = photo.views(im)
            piece = v.get("piece", v["full"])
            e = engine.emb.images([piece, ImageOps.mirror(piece)]).mean(0)
            out_rows.append(row)
            out_vecs.append(e / np.linalg.norm(e))
        if n % 200 == 0:
            print(f"  simulated photos: {n}/{len(rows)} designs  {time.time() - t0:.0f}s", flush=True)
    rows_a, vecs_a = np.array(out_rows), np.stack(out_vecs).astype(np.float32)
    np.savez(cache, key=np.array(key), rows=rows_a, vecs=vecs_a)
    return rows_a, vecs_a


def train(X, y, ncls, epochs=400):
    X, y = torch.tensor(X), torch.tensor(y)
    count = torch.bincount(y, minlength=ncls).float().clamp(min=1)
    w = (count.sum() / count) ** 0.5            # rare cuts count more, but not fully (they're also rarer to see)
    w = w / w.mean()
    layer = torch.nn.Linear(X.shape[1], ncls)
    opt = torch.optim.Adam(layer.parameters(), lr=0.01, weight_decay=1e-3)
    for _ in range(epochs):
        opt.zero_grad()
        torch.nn.functional.cross_entropy(layer(X * SCALE), y, weight=w).backward()
        opt.step()
    return layer.weight.detach().numpy().astype(np.float32), layer.bias.detach().numpy().astype(np.float32)


def probs(W, b, X):
    z = (X * SCALE) @ W.T + b
    z -= z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def confidence_table(p, y):
    out = {"accuracy": round(float((p.argmax(1) == y).mean()), 3), "n": int(len(y))}
    for t in THRESHOLDS:
        s = p.max(1) >= t
        out[f"p>={t}"] = {"shown": round(float(s.mean()), 3),
                          "right": round(float((p[s].argmax(1) == y[s]).mean()), 3) if s.any() else None}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--photos", type=int, default=2, help="simulated shopper photos per design (default 2)")
    ap.add_argument("--seed", type=int, default=21)
    args = ap.parse_args()
    engine = SearchEngine()
    if engine.views is None:
        raise SystemExit("Run scripts/build_view_index.py first (per-view vectors).")
    n = len(engine.meta)
    y = {"cut": np.full(n, -1), "layout": np.full(n, -1)}
    for row, m in enumerate(engine.meta):
        cut, layout = labels(m)
        if cut:
            y["cut"][row] = CUTS.index(cut)
        if layout:
            y["layout"][row] = LAYOUTS.index(layout)
    classes = {"cut": CUTS, "layout": LAYOUTS}
    print(f"designs with an answer: cut {int((y['cut'] >= 0).sum())}, layout {int((y['layout'] >= 0).sum())}")

    starts = engine._view_starts
    view_rows = np.repeat(engine._view_designs, np.diff(np.r_[starts, len(engine.views)]))
    labelled = np.flatnonzero((y["cut"] >= 0) | (y["layout"] >= 0))
    print(f"making {args.photos} simulated photos for each of {len(labelled)} designs (cached after the first run)")
    aug_rows, aug_vecs = simulated_photos(engine, labelled, args.photos, args.seed)
    fold = np.array([hash_fold(m["family"]) for m in engine.meta])

    model, report = {}, {}
    for head in ("cut", "layout"):
        Y = y[head]
        X_all = np.concatenate([engine.views, aug_vecs])
        r_all = np.concatenate([view_rows, aug_rows])
        is_photo = np.r_[np.zeros(len(view_rows), bool), np.ones(len(aug_rows), bool)]
        have = Y[r_all] >= 0
        render_p, photo_p, photo_y = np.zeros((n, len(classes[head]))), [], []
        for f in range(FOLDS):
            tr = have & (fold[r_all] != f)
            W, b = train(X_all[tr], Y[r_all][tr], len(classes[head]))
            te = have & (fold[r_all] == f)
            # renders: average the design's views; photos: each one on its own
            v = te & ~is_photo
            np.add.at(render_p, r_all[v], probs(W, b, X_all[v]))
            p = te & is_photo
            photo_p.append(probs(W, b, X_all[p]))
            photo_y.append(Y[r_all][p])
        designs = np.flatnonzero((Y >= 0) & (render_p.sum(1) > 0))
        render_p = render_p[designs] / render_p[designs].sum(1, keepdims=True)
        report[head] = {"renders": confidence_table(render_p, Y[designs]),
                        "photos": confidence_table(np.concatenate(photo_p), np.concatenate(photo_y))}
        print(f"\n{head}: renders of unseen designs   {json.dumps(report[head]['renders'])}")
        print(f"{head}: simulated photos, unseen   {json.dumps(report[head]['photos'])}")
        W, b = train(X_all[have], Y[r_all][have], len(classes[head]))
        model[f"{head}_W"], model[f"{head}_b"] = W, b

    np.savez(OUT, key=np.array(index_key(engine.meta)), scale=np.array(SCALE),
             cut_classes=np.array(CUTS), layout_classes=np.array(LAYOUTS),
             report=np.array(json.dumps(report)), **model)
    print(f"\nwrote {OUT}")


def hash_fold(family: str) -> int:
    """Stable fold per design family (Python's hash() changes per run)."""
    import hashlib
    return int(hashlib.sha1(family.encode()).hexdigest(), 16) % FOLDS


if __name__ == "__main__":
    main()
