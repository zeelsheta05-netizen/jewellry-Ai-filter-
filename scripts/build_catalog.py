#!/usr/bin/env python3
"""Merge the raw audit into one record per unique design_id.

Reads  data/audit/designs.jsonl   (one record per design *folder*)
Writes data/catalog.jsonl         (one record per design_id)
       data/catalog_summary.json

Merge rules
  * The same design_id can appear in several folders (e.g. a PHOTOS copy).
    Images / videos are unioned per (metal, view); on conflict the file from
    the folder that is *not* named PHOTOS wins.
  * "NOT UPLOAD" in a folder name is the client's own listing label and is
    ignored everywhere (it is stripped before any category lookup).
  * Category is a rule-based first guess with a recorded source; designs the
    rules cannot resolve stay "unknown" and are filled in by the image model
    during indexing.
"""
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
AUDIT = ROOT / "data" / "audit" / "designs.jsonl"

# Rules on the design_id, checked in order. Pattern is matched against the id
# upper-cased. Kept deliberately conservative: anything ambiguous stays unknown.
ID_RULES = [
    ("earrings", re.compile(r"^(?:[A-Z]*ER\b|[A-Z]*ER[-( ]|\d+#\d+E\b|\d+#\d+E_|\d+#\d+@E\b)")),
    ("pendant",  re.compile(r"^(?:[A-Z]*(?:PD|PO|PS|PND|NP)\b|[A-Z]*(?:PD|PO|PS|PND|NP)[-( ]|\d+#\d+P\b|\d+#\d+[A-Z]?P_)")),
    ("necklace", re.compile(r"^(?:[A-Z]*(?:NK|WN)\b|[A-Z]*(?:NK|WN)[-( ]|\d+#\d+N\b)")),
    ("bracelet", re.compile(r"^(?:[A-Z]*(?:BR|WB|MB)\b|[A-Z]*(?:BR|WB|MB)[-( ]|\d+#\d+@?L?B\b)")),
    ("ring",     re.compile(r"^(?:[A-Z]*(?:LR|DR|MR|GR|WR|ER_?RING)\b|[A-Z]*(?:LR|DR|MR|GR|WR)[-( ]|R\d+\b|\d+#\d+L\b)")),
]
# Keywords in the folder path (after removing "NOT UPLOAD").
PATH_RULES = [
    ("earrings", re.compile(r"earring|\bstud\b|\bhoop\b|jhumk|bali")),
    ("pendant",  re.compile(r"pandant|pendant")),
    ("necklace", re.compile(r"neckl|neckless|mangal ?sutra|\bchain\b(?!.*brace)")),
    ("bracelet", re.compile(r"brelcate|bracelet|bangle|\bcuff\b")),
    ("ring",     re.compile(r"\bring|rings\b|ledis|gent")),
]


def clean_path(parts):
    out = []
    for p in parts:
        p = re.sub(r"\s*NOT\s*UPLOAD\s*", "", p, flags=re.I).strip()
        if p:
            out.append(p)
    return out


def path_categories(path_parts) -> set:
    joined = " / ".join(clean_path(path_parts)).lower()
    return {cat for cat, rx in PATH_RULES if rx.search(joined)}


def guess_category(design_id, path_parts):
    did = design_id.strip().upper()
    for cat, rx in ID_RULES:
        if rx.search(did):
            return cat, "id"
    hits = path_categories(path_parts)
    if len(hits) == 1:
        return hits.pop(), "path"
    return "unknown", "none"


def merge_key(design_id: str, folder: str) -> str:
    """Coded ids (DDLR-123, 12#10003P, GNK-0001) are unique across the drive,
    so their copies in different folders are merged. Bare numbers ("11",
    "1 (303)", "10 AC") are reused by unrelated batches, so they are only
    unique within their folder."""
    if "#" in design_id or re.match(r"^\s*[A-Za-z]{2,}", design_id):
        return design_id
    return f"{folder}/{design_id}"


def main():
    merged = {}
    for line in AUDIT.read_text().splitlines():
        r = json.loads(line)
        did = r["design_id"]
        m = merged.setdefault(merge_key(did, r["dir"]), {
            "design_id": did, "folders": [], "images": defaultdict(dict),
            "videos": {}, "variants": set(), "has_cad": False,
            "_path_parts": [],
        })
        is_photos = "photos" in r["dir"].lower().split("/")[-1]
        m["folders"].append(r["dir"])
        m["has_cad"] |= r["has_cad"]
        m["variants"].update(r["variants"])
        if len(r["dir"].split("/")) > len(m["_path_parts"]) and not is_photos:
            m["_path_parts"] = r["dir"].split("/")
        for metal, views in r["images"].items():
            for v, p in views.items():
                if v not in m["images"][metal] or not is_photos:
                    m["images"][metal][v] = p
        for metal, p in r["videos"].items():
            if metal not in m["videos"] or not is_photos:
                m["videos"][metal] = p

    cats, srcs = Counter(), Counter()
    out = ROOT / "data" / "catalog.jsonl"
    with open(out, "w") as fh:
        for key, m in sorted(merged.items()):
            parts = m.pop("_path_parts") or m["folders"][0].split("/")
            cat, src = guess_category(m["design_id"], parts)
            m["key"] = key
            # every category any of its folders names; the index uses this to
            # let the image model overrule a wrong id rule
            m["path_categories"] = sorted(set().union(*(path_categories(f.split("/")) for f in m["folders"])))
            cats[cat] += 1
            srcs[src] += 1
            rec = {
                **m,
                "images": {k: {str(v): p for v, p in sorted(vs.items())} for k, vs in m["images"].items()},
                "variants": sorted(m["variants"]),
                "metals": sorted(m["images"]),
                "category": cat, "category_source": src,
                "has_images": any(m["images"].values()),
            }
            fh.write(json.dumps(rec) + "\n")

    summary = {
        "designs": len(merged),
        "with_images": sum(1 for m in merged.values() if any(m["images"].values())),
        "with_video": sum(1 for m in merged.values() if m["videos"]),
        "category_guess": dict(cats.most_common()),
        "category_source": dict(srcs),
    }
    (ROOT / "data" / "catalog_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
