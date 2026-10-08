#!/usr/bin/env python3
"""Embed every design and write the search index.

Inputs: data/catalog.jsonl, data/crops/*.webp and data/crops/stats.jsonl
        (from make_crops.py).

Per design (using one metal's renders, white gold preferred, so shapes are
compared in the same colour):
  * every view's tight crop is embedded
  * design vector = mean of all views ("design DNA" across angles)
  * front view = the most left-right symmetric view, smoothed by a majority
    vote across the design's series (a series renders its angles in the same
    order); front vector = that view's embedding
  * category: filename rules, or the model where the rules had no answer
  * attributes (stones / band / weight) scored on the front vector

Writes data/index/embeddings.npy        design vectors (n, d)
       data/index/front.npy             front-view vectors (n, d)
       data/index/meta.jsonl            catalog record + front view + attributes
"""
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from jewelsearch import attributes  # noqa: E402
from jewelsearch.config import CATALOG, CATEGORIES, CROPS, EMBED_METAL_ORDER, INDEX, thumb_name  # noqa: E402
from jewelsearch.dna import CATEGORY_PROMPTS  # noqa: E402  (photo search reads a photo's type with the same prompts)
from jewelsearch.embedder import Embedder  # noqa: E402
BATCH = 64


def series_key(design_id: str, category: str) -> str:
    """Designs sharing a naming prefix come from one render batch."""
    did = design_id.strip().upper()
    m = re.match(r"^(\d+#)", did) or re.match(r"^([A-Z]+)", did)
    return f"{category}|{m.group(1) if m else did}"


def main():
    INDEX.mkdir(parents=True, exist_ok=True)
    stats = {}
    for line in (CROPS / "stats.jsonl").read_text().splitlines():
        s = json.loads(line)
        stats[s["path"]] = s
    recs = [json.loads(l) for l in CATALOG.read_text().splitlines()]
    recs = [r for r in recs if r["has_images"]]

    jobs, metas = [], []
    for r in recs:
        metal = next(m for m in EMBED_METAL_ORDER if r["images"].get(m))
        views = r["images"][metal]
        r["embed_metal"] = metal
        r["_views"] = sorted(views)
        for v in r["_views"]:
            jobs.append((len(metas), v, CROPS / thumb_name(views[v])))
        metas.append(r)

    emb = Embedder()
    view_vecs = {}
    t0 = time.time()
    for i in range(0, len(jobs), BATCH):
        chunk = [j for j in jobs[i:i + BATCH] if j[2].exists()]
        if not chunk:
            continue
        vecs = emb.images([Image.open(p) for _, _, p in chunk])
        for (row, v, _), vec in zip(chunk, vecs):
            view_vecs[(row, v)] = vec
        if (i // BATCH) % 25 == 0:
            print(f"{i}/{len(jobs)} views  {time.time() - t0:.0f}s", flush=True)

    keep = [row for row in range(len(metas)) if any((row, v) in view_vecs for v in metas[row]["_views"])]
    metas = [metas[row] | {"_row": row} for row in keep]

    # ---- category (needed before the front-view vote, which is per series)
    dvecs = np.stack([np.mean([view_vecs[(m["_row"], v)] for v in m["_views"] if (m["_row"], v) in view_vecs], 0)
                      for m in metas])
    dvecs /= np.linalg.norm(dvecs, axis=1, keepdims=True)
    cat_vecs = np.stack([emb.texts(p).mean(0) for p in CATEGORY_PROMPTS.values()])
    cat_vecs /= np.linalg.norm(cat_vecs, axis=1, keepdims=True)
    pred = (dvecs @ cat_vecs.T).argmax(1)
    for m, p in zip(metas, pred):
        model_cat = CATEGORIES[p]
        m["category_model"] = model_cat
        if m["category"] == "unknown":
            m["category"], m["category_source"] = model_cat, "model"
        elif model_cat != m["category"] and model_cat in m.get("path_categories", []):
            # the picture agrees with a folder name (e.g. GNK-0006 filed under
            # PANDANT, or an "…E" earring in "Earring photo"): trust both over
            # the id rule. When the model agrees with nothing, the rule stands
            # (it mistakes eternity rings for bracelets, for instance).
            m["category"], m["category_source"] = model_cat, "model+folder"

    # ---- front view: per-design most symmetric view, then series majority
    def own_best(m):
        views = m["images"][m["embed_metal"]]
        return max(m["_views"], key=lambda v: stats.get(views[v], {}).get("sym", 0))

    votes = defaultdict(Counter)
    for m in metas:
        votes[series_key(m["design_id"], m["category"])][own_best(m)] += 1
    fvecs = []
    for m in metas:
        key = series_key(m["design_id"], m["category"])
        v, n = votes[key].most_common(1)[0]
        share = n / sum(votes[key].values())
        if not ((m["_row"], v) in view_vecs and share >= 0.5):
            v = own_best(m)
        m["front_view"] = v
        m["front_source"] = "series" if v != own_best(m) or share >= 0.5 else "own"
        fvecs.append(view_vecs[(m["_row"], v)])
    fvecs = np.stack(fvecs)

    # ---- attributes, per category so prompts use the right noun
    cats = np.array([m["category"] for m in metas])
    for cat in CATEGORIES:
        idx = np.flatnonzero(cats == cat)
        if not len(idx):
            continue
        scores = attributes.score(fvecs[idx], attributes.class_vectors(emb, cat))
        for attr, classes in scores.items():
            for k, row in enumerate(idx):
                metas[row].setdefault("attrs", {})[attr] = {c: round(float(p[k]), 4) for c, p in classes.items()}

    np.save(INDEX / "embeddings.npy", dvecs.astype(np.float32))
    np.save(INDEX / "front.npy", fvecs.astype(np.float32))
    with open(INDEX / "meta.jsonl", "w") as fh:
        for m in metas:
            m.pop("_views"), m.pop("_row")
            fh.write(json.dumps(m) + "\n")

    decided = [m for m in metas if m["category_source"] != "model"]
    agree = sum(m["category"] == m["category_model"] for m in decided)
    fronts = Counter((m["category"], m["front_view"]) for m in metas)
    print(f"indexed {len(metas)} designs ({len(view_vecs)} views) in {time.time() - t0:.0f}s; "
          f"model agrees with rules on {agree}/{len(decided)}")
    print("front view by category:", dict(sorted(fronts.items())))


if __name__ == "__main__":
    main()
