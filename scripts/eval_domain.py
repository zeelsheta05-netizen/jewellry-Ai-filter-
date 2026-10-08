"""How well the search keeps to jewellery (jewelsearch/domain.py).

Runs every prompt of the jewellery lists (must be accepted) and of the
out-of-domain list (must be refused) through the same check the search uses,
and reports each section, what decided (words / judge / picture) and the
mistakes. Loads the index, the image model and the judge (~25 s).

    .venv/bin/python scripts/eval_domain.py
"""
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from jewelsearch.search import SearchEngine  # noqa: E402


def sections(path: Path) -> dict[str, list[str]]:
    out, name = {}, "all"
    for line in path.read_text().splitlines():
        line = line.strip()
        if line.startswith("# ---"):
            name = line[5:].strip()
        elif line and not line.startswith("#"):
            out.setdefault(name, []).append(line)
    return out


def main():
    e = SearchEngine()
    e.domain.load_judge()
    t = ROOT / "tests"
    main_set = [p for f in ("prompts_100.txt", "prompts_structural.txt") for ps in sections(t / f).values() for p in ps]
    runs = [("jewellery: prompts_100 + structural", True, {"all": main_set})]
    runs += [("jewellery extra", True, sections(t / "prompts_in_domain_extra.txt"))]
    runs += [("not jewellery", False, sections(t / "prompts_out_of_domain.txt"))]
    t0, n = time.perf_counter(), 0
    for title, want, secs in runs:
        for name, prompts in secs.items():
            wrong, via = [], {}
            for p in prompts:
                v = e.domain.check(p)
                n += 1
                via[v.via] = via.get(v.via, 0) + 1
                if v.ok != want:
                    wrong.append(f"{'accepted' if v.ok else 'refused'} ({v.via}{', ' + v.reason if v.reason else ''}): {p}")
            print(f"{title} / {name}: {len(prompts) - len(wrong)}/{len(prompts)} right   decided by {via}")
            for w in wrong:
                print("    " + w)
    print(f"\n{n} prompts, {(time.perf_counter() - t0) / n * 1000:.0f} ms each on average")


if __name__ == "__main__":
    main()
