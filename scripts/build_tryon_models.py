"""Convert every catalogue design that has CAD into a try-on model.

    .venv-cad/bin/python scripts/build_tryon_models.py              # all designs, resumable
    .venv-cad/bin/python scripts/build_tryon_models.py --limit 50   # quick sample
    .venv-cad/bin/python scripts/build_tryon_models.py --retry      # redo failed ones
    .venv-cad/bin/python scripts/build_tryon_models.py --force      # redo all (converter changed)
    .venv-cad/bin/python scripts/build_tryon_models.py --older-than 2026-09-29T17:20   # finish a stopped --force

Reads the search index (data/index/meta.jsonl), finds each design's .3dm in
its folders on the dataset drive (read-only), converts it with cad_to_glb.py
and writes:

    data/tryon/models/<slug>.glb/.json   one per design
    data/tryon/designs.json              {"<design_id>|<first folder>": slug}

The server uses designs.json to show a "Try on" button on a card only when
that design has a working model. Already converted designs are skipped, so
the run can be stopped and started again.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cad_to_glb  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
META = ROOT / "data" / "index" / "meta.jsonl"
MAP = ROOT / "data" / "tryon" / "designs.json"
LOG = ROOT / "data" / "tryon" / "build.log"

# search-index category -> converter category
CATEGORY = {"ring": "ring", "earrings": "earring", "pendant": "pendant",
            "necklace": "necklace", "bracelet": "bracelet"}
# file-name tags that mark a non-primary copy of the same design
SECONDARY = re.compile(r"@(stl|re)\b|photo|[-_ ]r$", re.I)


def design_key(m: dict) -> str:
    return f"{m['design_id']}|{m['folders'][0]}"


def slug_for(m: dict) -> str:
    base = re.sub(r"[^A-Za-z0-9_-]+", "_", m["design_id"]).strip("_")[:60] or "design"
    return f"{base}-{hashlib.sha1(design_key(m).encode()).hexdigest()[:6]}"


def find_cad_name(m: dict) -> str | None:
    """The dataset name of the design's own .3dm: same name as the design id; a
    folder shared by many designs (e.g. PHOTOS) must match by name, a design's
    own folder may hold one differently named file. Looks in the folder and two
    levels below it, from the storage's listing (nothing is downloaded)."""
    want = m["design_id"].lower()
    named, others = [], []
    st = cad_to_glb.ds.get()
    for folder in m["folders"]:
        found = [rel for rel in st.walk(folder, exts={".3dm"}) if rel[len(folder) + 1:].count("/") <= 2]
        for rel in sorted(found, key=lambda r: (r.count("/"), r)):   # shallow first, like a top-down walk
            stem = Path(rel).stem
            if stem.lower() == want:
                return rel
            if stem.lower().startswith(want) and not SECONDARY.search(stem):
                named.append(rel)
            elif not SECONDARY.search(stem):
                others.append(rel)
    if named:
        return sorted(named, key=lambda r: len(r.rsplit("/", 1)[-1]))[0]
    return others[0] if len(others) == 1 else None


def find_cad(m: dict) -> Path | None:
    """The design's .3dm as a local file (fetched from the storage when needed)."""
    rel = find_cad_name(m)
    return cad_to_glb.fetch(rel) if rel else None


def cad_size(m: dict) -> int:
    """Bytes of the design's .3dm plus the same-named .stl next to it, from the listing."""
    rel = find_cad_name(m)
    if rel is None:
        return 0
    st = cad_to_glb.ds.get()
    folder, _, name = rel.rpartition("/")
    n = st.size(rel) or 0
    for other in st.walk(folder, exts={".stl"}):
        o_dir, _, o_name = other.rpartition("/")
        if o_dir == folder and Path(o_name).stem.lower() == Path(name).stem.lower():
            n += st.size(other) or 0
    return n


def work(args):
    m, slug, cat = args
    try:
        cad = find_cad(m)
    except cad_to_glb.ds.StorageUnavailable:   # the storage went away: not the design's fault
        return m, slug, {"status": "error: dataset storage not available"}
    if cad is None:
        if not cad_to_glb.ds.get().ready():
            return m, slug, {"status": "error: dataset storage not available"}
        return m, slug, {"status": "no_cad_file"}
    try:
        return m, slug, cad_to_glb.convert(cad, cat, slug)
    except Exception as e:   # one bad file must not stop the batch
        return m, slug, {"status": f"error: {type(e).__name__}: {e}"[:200]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--retry", action="store_true", help="convert again designs that failed before")
    ap.add_argument("--force", action="store_true", help="convert everything again (after a converter change)")
    ap.add_argument("--older-than", help="convert again designs last converted before this time "
                                         "(e.g. 2026-09-29T17:20): finishes a --force run that was stopped")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    if not cad_to_glb.ds.get().ready():
        raise SystemExit(f"{cad_to_glb.ds.get().label()} is not available")
    older_than = datetime.fromisoformat(args.older_than).timestamp() if args.older_than else 0
    meta = [json.loads(l) for l in META.read_text().splitlines()]
    mapping = json.loads(MAP.read_text()) if MAP.exists() else {}
    todo = []
    for m in meta:
        cat = CATEGORY.get(m["category"])
        if not cat or not m.get("has_cad"):
            continue
        slug = slug_for(m)
        done = cad_to_glb.OUT / f"{slug}.json"
        stale = older_than and done.exists() and done.stat().st_mtime < older_than
        if done.exists() and not args.force and not stale:
            status = json.loads(done.read_text()).get("status")
            if status == "ok" or not args.retry:
                if status == "ok":
                    mapping[design_key(m)] = slug
                continue
        todo.append((m, slug, cat))
    if args.limit:
        todo = todo[: args.limit]
    print(f"{len(todo)} designs to convert, {len(mapping)} already done", flush=True)

    t0, counts = time.time(), {}
    with ProcessPoolExecutor(args.workers) as pool, LOG.open("a") as log:
        futures = [pool.submit(work, a) for a in todo]
        for n, fut in enumerate(as_completed(futures), 1):
            m, slug, res = fut.result()
            status = res.get("status", "?")
            key = status.split(":")[0]
            counts[key] = counts.get(key, 0) + 1
            if status == "ok":
                mapping[design_key(m)] = slug
            else:   # remember failures so a plain re-run skips them
                mapping.pop(design_key(m), None)   # a model that used to convert may not any more
                (cad_to_glb.OUT / f"{slug}.glb").unlink(missing_ok=True)
                (cad_to_glb.OUT / f"{slug}.json").write_text(json.dumps(
                    {"status": status, "slug": slug, "design_id": m["design_id"], "category": m["category"]}))
            log.write(json.dumps({"design_id": m["design_id"], "slug": slug, "status": status}) + "\n")
            if n % 25 == 0 or n == len(todo):
                MAP.write_text(json.dumps(mapping))
                rate = n / (time.time() - t0)
                print(f"{n}/{len(todo)}  {rate:.1f}/s  eta {(len(todo) - n) / rate / 60:.0f} min  {counts}", flush=True)
    MAP.write_text(json.dumps(mapping))
    print("done", counts, flush=True)


if __name__ == "__main__":
    main()
