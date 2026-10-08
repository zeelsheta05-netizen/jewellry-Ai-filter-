#!/usr/bin/env python3
"""Read-only audit of the jewellery dataset.

Lists the dataset storage (jewelsearch/storage.py: the S3 storage index, or a
folder), groups render files into "designs" and writes
  data/audit/designs.jsonl   one JSON record per design
  data/audit/summary.json    counts by bucket / category / metal / extension

Nothing on the dataset volume is modified.

Filename convention (observed):
  <design_key>[-<variant 2 digits>]@<R|W|Y>-#<viwe|view><N>.<png|mp4>
  ("viwe" is a typo used throughout the dataset; both spellings occur)
  Coloured-gemstone versions carry a tag: <design_id>-01@R-GEM-#viwe1.png
  R = rose gold, W = white gold, Y = yellow gold; views 1-4 png, view 5 mp4.
"""
import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from jewelsearch import storage  # noqa: E402

VIEW_RE = re.compile(
    r"^(?P<key>.+?)(?:-(?P<var>\d\d))?@(?P<metal>[RWY])-(?:(?P<tag>GEM)-)?#vi(?:ew|we)(?P<view>\d)\.(?P<ext>png|mp4)$",
    re.IGNORECASE,
)
METALS = {"R": "rose_gold", "W": "white_gold", "Y": "yellow_gold"}
CAD_EXT = {".3dm", ".stl"}


def walk(st: storage.Storage):
    """Yield (folder name, [file names]) for every folder holding files, parents
    first, from the storage's listing (junk and system files are never listed)."""
    groups = defaultdict(list)
    for path in st.walk():
        d, _, name = path.rpartition("/")
        groups[d].append(name)
    for d in sorted(groups):
        yield d, sorted(groups[d])


def audit(st: storage.Storage):
    designs = {}
    ext_counter = Counter()
    for d, files in walk(st):
        rel = Path(d) if d else Path(".")
        cad_here = any(Path(f).suffix.lower() in CAD_EXT for f in files)
        for f in files:
            ext_counter[Path(f).suffix.lower()] += 1
            m = VIEW_RE.match(f)
            if not m:
                continue
            key = m["key"]
            rec = designs.setdefault((str(rel), key), {
                "design_id": key,
                "dir": str(rel),
                "bucket": rel.parts[0] if rel.parts else "",
                "category_path": list(rel.parts[:-1]) if len(rel.parts) > 1 else [],
                "images": {},   # metal -> {view: relative path}
                "videos": {},   # metal -> relative path
                "variants": set(),
                "has_cad": False,
            })
            metal = METALS[m["metal"].upper()]
            path = str(rel / f)
            # "-GEM" renders are the coloured-gemstone version; used when the
            # plain render of that metal/view is absent
            views = rec["images"].setdefault(metal, {}) if m["ext"].lower() == "png" else None
            if views is not None:
                if not m["tag"] or int(m["view"]) not in views:
                    views[int(m["view"])] = path
            elif not m["tag"] or metal not in rec["videos"]:
                rec["videos"][metal] = path
            if m["var"]:
                rec["variants"].add(m["var"])
        # CAD files usually sit in the design folder or its same-named inner folder
        if cad_here:
            for (drel, _), rec in designs.items():
                if drel == str(rel) or str(rel).startswith(drel + os.sep):
                    rec["has_cad"] = True
    return designs, ext_counter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="", help="a dataset folder to audit instead of the configured storage")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent.parent / "data" / "audit"))
    args = ap.parse_args()

    st, out = storage.LocalStorage(args.root) if args.root else storage.get(), Path(args.out)
    if not st.ready():
        raise SystemExit(f"{st.label()} is not available")
    out.mkdir(parents=True, exist_ok=True)
    designs, ext_counter = audit(st)

    by_bucket, by_metal_count, img_per_design = Counter(), Counter(), Counter()
    with open(out / "designs.jsonl", "w") as fh:
        for rec in designs.values():
            rec["variants"] = sorted(rec["variants"])
            n_img = sum(len(v) for v in rec["images"].values())
            img_per_design[n_img] += 1
            by_bucket[rec["bucket"]] += 1
            by_metal_count[len(rec["images"])] += 1
            fh.write(json.dumps(rec) + "\n")

    summary = {
        "designs_total": len(designs),
        "designs_with_video": sum(1 for r in designs.values() if r["videos"]),
        "designs_with_cad": sum(1 for r in designs.values() if r["has_cad"]),
        "by_bucket": dict(by_bucket.most_common()),
        "metals_per_design": dict(sorted(by_metal_count.items())),
        "images_per_design": dict(sorted(img_per_design.items())),
        "extensions": dict(ext_counter.most_common(15)),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
