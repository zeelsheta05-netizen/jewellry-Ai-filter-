#!/usr/bin/env python3
"""Score the ranking on the 100-prompt set, so changes can be compared.

  diversity   distinct designs across all result slots, and the worst repeat
  fit         for prompts whose intents map to a measurable property, how
              well the 8 results have it (0..1, higher is better):
                plain / "no diamonds"  share of results with few stones
                diamond / cluster      share of results with many stones
                solitaire              mean solitaire probability
                thin / wide            mean band-class probability
                everyday / minimal     mean "delicate" probability
                statement / bridal     mean "statement" probability
                men / women            share of results of that gender
  category    share of results whose type was asked for (must stay 1.0)

Usage: .venv/bin/python scripts/eval_metrics.py
"""
import collections
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from jewelsearch.search import PLAIN_MAX_STONE, SearchEngine  # noqa: E402

PROMPTS = Path(__file__).resolve().parent.parent / "tests" / "prompts_100.txt"
FEW_STONES = 0.06
MANY_STONES = 0.25


def fits(e: SearchEngine, intents: list[str], uids: list[int]) -> dict:
    ms = [e.by_uid[u] for u in uids]
    a = lambda m, at, c: m.get("attrs", {}).get(at, {}).get(c, 0.0)  # noqa: E731
    stone = [m.get("stone_frac") or 0.0 for m in ms]
    out = {}
    if "plain" in intents:
        out["plain"] = np.mean([s <= FEW_STONES for s in stone])
    if "diamond" in intents or "cluster" in intents:
        out["diamond"] = np.mean([s >= MANY_STONES for s in stone])
    if "solitaire" in intents:
        out["solitaire"] = np.mean([a(m, "stones", "solitaire") for m in ms])
    for k in ("thin", "wide"):
        if k in intents and all(m["category"] == "ring" for m in ms):
            out[k] = np.mean([a(m, "band", k) for m in ms])
    if {"everyday", "minimal"} & set(intents):
        out["delicate"] = np.mean([a(m, "weight", "delicate") for m in ms])
    if {"statement", "bridal"} & set(intents):
        out["statement"] = np.mean([a(m, "weight", "statement") for m in ms])
    if "men" in intents:
        out["men"] = np.mean([e.mens[u] for u in uids])
    if "women" in intents:
        out["women"] = np.mean([not e.mens[u] for u in uids])
    return out


def main():
    e = SearchEngine()
    lines = [l.strip() for l in PROMPTS.read_text().splitlines() if l.strip() and not l.startswith("#")]
    count, fit, cat_ok, slots = collections.Counter(), collections.defaultdict(list), [], 0
    for l in lines:
        r = e.search(l)
        uids = [c["uid"] for c in r["results"]]
        slots += len(uids)
        count.update(uids)
        cats = set(r["query"]["categories"])
        cat_ok += [not cats or c["category"] in cats for c in r["results"]]
        for k, v in fits(e, r["query"]["intents"], uids).items():
            fit[k].append(v)
    print(f"diversity: {len(count)} distinct designs in {slots} slots; worst repeat {count.most_common(1)[0][1]}x; "
          f"designs shown 5+ times: {sum(n >= 5 for n in count.values())}")
    print(f"category:  {np.mean(cat_ok):.3f}")
    allfit = []
    for k in sorted(fit):
        allfit += fit[k]
        print(f"fit {k:10s} {np.mean(fit[k]):.2f}  (n={len(fit[k])})")
    print(f"fit overall {np.mean(allfit):.3f}")


if __name__ == "__main__":
    main()
