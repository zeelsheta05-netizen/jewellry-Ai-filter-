"""Buy-page details for every catalogue design: gold weight, stones, size.

    .venv-cad/bin/python scripts/build_cad_specs.py             # resumable
    .venv-cad/bin/python scripts/build_cad_specs.py --force     # measure every CAD file again
    .venv-cad/bin/python scripts/build_cad_specs.py --cards-only

Two sources, both read-only on the dataset drive:

1. Job cards: the client's own "Job Card" spreadsheets (<design>@5.XLSX,
   about 180 designs). Gold weight per purity after polishing, and the
   diamond list with sieve size, count, carats and setting type. These are
   the client's numbers and always win.
2. The CAD file: cad_to_glb.convert(measure_only=True) keeps the same single
   piece as the try-on model and measures its metal volume, each stone's cut,
   size and volume, the ring size and the piece's size. Nothing is written
   to the drive or to the try-on models.

Where a design has both, the pair calibrates the CAD numbers: grams of 18K
per cm3 of CAD metal, fitted on rings with a closed metal surface. The factor
(~12 g/cm3, not 18K's ~15.5) includes the 0.2-0.3 mm the client's cards say
is lost in polishing. The server decides what is shown, see
jewelsearch/purchase.py.

Output: data/cad/specs.json, plus one cached measurement per design in
data/cad/measured/<slug>.json.
"""

from __future__ import annotations

import argparse
import html
import json
import re
import statistics
import sys
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_tryon_models as tm  # noqa: E402
import cad_to_glb  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "cad"
MEASURED = OUT / "measured"
SPECS = OUT / "specs.json"

# ---------------------------------------------------------------- job cards


def _col(letters: str) -> int:
    n = 0
    for ch in letters:
        n = n * 26 + ord(ch) - 64
    return n


def read_first_sheet(path: Path) -> dict[tuple[int, int], str]:
    """Cells of an .xlsx's first sheet as {(row, col): text}. The job cards
    are plain values, so a zip + XML read is enough (no openpyxl needed)."""
    z = zipfile.ZipFile(path)
    names = z.namelist()
    shared = []
    if "xl/sharedStrings.xml" in names:
        x = z.read("xl/sharedStrings.xml").decode("utf8", "replace")
        for si in re.findall(r"<si>(.*?)</si>", x, re.S):
            shared.append(html.unescape("".join(re.findall(r"<t[^>]*>(.*?)</t>", si, re.S))))
    wb = z.read("xl/workbook.xml").decode("utf8", "replace")
    rels = z.read("xl/_rels/workbook.xml.rels").decode("utf8", "replace")
    first = re.search(r"<sheet [^>]*r:id=\"([^\"]+)\"", wb).group(1)
    target = None
    for rel in re.findall(r"<Relationship [^>]*>", rels):
        if f'Id="{first}"' in rel:
            target = re.search(r'Target="([^"]+)"', rel).group(1).lstrip("/")
    if not target:
        return {}
    target = target if target.startswith("xl/") else "xl/" + target
    x = z.read(target).decode("utf8", "replace")
    cells = {}
    for ref, attrs, inner in re.findall(r'<c r="([A-Z]+\d+)"([^>]*?)(?:/>|>(.*?)</c>)', x, re.S):
        v = re.search(r"<v>(.*?)</v>", inner or "")
        if v:
            val = v.group(1)
            if 't="s"' in attrs:
                val = shared[int(val)]
        else:
            t = re.search(r"<t[^>]*>(.*?)</t>", inner or "")
            if not t:
                continue
            val = html.unescape(t.group(1))
        m = re.match(r"([A-Z]+)(\d+)", ref)
        cells[(int(m.group(2)), _col(m.group(1)))] = val.strip()
    return cells


def _num(v) -> float | None:
    try:
        f = float(str(v).replace(",", ""))
    except (TypeError, ValueError):
        return None
    return f if f == f else None   # NaN


PURITY_LABELS = {"sterling 925": "silver_925", "10 kt": "10k", "14 kt": "14k", "18 kt": "18k",
                 "22 kt": "22k", "platinum": "platinum", "24 kt fine gold": "24k"}


def parse_card(path: Path) -> dict | None:
    """One job card: design number, category, ring size, weight per purity,
    diamond rows. None when the card is empty (a blank template)."""
    cells = read_first_sheet(path)
    at = {}
    for k, v in sorted(cells.items()):
        at.setdefault(v.lower().rstrip(": ").strip(), k)

    def right(label, n=1):
        k = at.get(label)
        return cells.get((k[0], k[1] + n)) if k else None

    weights = {}
    for label, key in PURITY_LABELS.items():
        w = _num(right(label))
        if w and w > 0:
            weights[key] = round(w, 3)
    ring = None
    unit, size = right("ring size"), _num(right("ring size", 2))
    if unit and size:
        unit = unit.strip().upper()
        ring = {"unit": {"IND": "IN"}.get(unit, unit), "size": size}

    stones = []
    head, total = at.get("diam shape"), at.get("total")
    if head:
        r = head[0] + 1
        end = total[0] if total else head[0] + 40
        c0 = head[1]
        while r < end:
            qty = _num(cells.get((r, c0 + 4)))
            if qty and qty > 0:
                stones.append({
                    "shape": cells.get((r, c0), ""), "sieve": cells.get((r, c0 + 1), ""),
                    "gem": cells.get((r, c0 + 2), ""), "size": cells.get((r, c0 + 3), ""),
                    "count": int(qty), "carat_each": _num(cells.get((r, c0 + 5))),
                    "carat": _num(cells.get((r, c0 + 6))), "setting": cells.get((r, c0 + 7), ""),
                })
            r += 1
    if not weights.get("18k") and not stones:
        return None
    return {
        "design_number": right("design number") or "",
        "category": (right("category") or "").title(),
        "design_type": (right("design type") or "").title(),
        "ring_size": ring,
        "weights_g": weights,
        "stones": stones,
        "file": cad_to_glb.dataset_name(path),
    }


def _norm(design_id: str) -> str:
    return re.sub(r"\s+", "", design_id).upper()


def _lot(rel: str) -> str:
    """The batch folder a card belongs to ("04/04/Loat - 33"), or the folder
    it sits in (without a trailing "xlsx") when it isn't in a batch."""
    parts = rel.split("/")[:-1]
    for i, p in enumerate(parts):
        if re.match(r"^loat\s*-\s*\d+", p.strip(), re.I):
            return "/".join(parts[: i + 1])
    if parts and parts[-1].lower() == "xlsx":
        parts = parts[:-1]
    return "/".join(parts)


def find_cards(meta: list[dict]) -> dict[str, dict]:
    """Job cards on the drive matched to catalogue designs by design id, in
    the same batch folder (bare-number ids such as "14" are reused by
    unrelated batches). A coded id found only once may be anywhere."""
    by_id: dict[str, list[dict]] = {}
    for m in meta:
        by_id.setdefault(_norm(m["design_id"]), []).append(m)
    found = {}
    for rel in cad_to_glb.ds.get().walk(exts={".xlsx"}):
        if "$RECYCLE" in rel:
            continue
        stem = re.sub(r"@\d+$", "", Path(rel).stem)
        cands = by_id.get(_norm(stem), [])
        if not cands:
            continue
        lot = _lot(rel)
        near = [m for m in cands if any(f == lot or f.startswith(lot + "/") for f in m["folders"])]
        if not near and len(cands) == 1 and re.search(r"[A-Za-z]", stem):
            near = cands
        if not near:
            continue
        try:
            card = parse_card(cad_to_glb.fetch(rel))
        except Exception as e:   # a damaged file must not stop the build
            print(f"skipped card {rel}: {e!r}", flush=True)
            continue
        if card:
            for m in near:
                found[tm.design_key(m)] = card
    return found

# ---------------------------------------------------------------- CAD files


def measure(args):
    m, slug, cat = args
    try:
        cad = tm.find_cad(m)
    except cad_to_glb.ds.StorageUnavailable:
        return m, slug, {"status": "error: dataset storage not available"}
    if cad is None:
        if not cad_to_glb.ds.get().ready():
            return m, slug, {"status": "error: dataset storage not available"}
        return m, slug, {"status": "no_cad_file"}
    try:
        return m, slug, cad_to_glb.convert(cad, cat, slug, measure_only=True)
    except Exception as e:   # one bad file must not stop the batch
        return m, slug, {"status": f"error: {type(e).__name__}: {e}"[:200]}


BIG_BYTES = 150_000_000   # CAD this large (STLs of 13M triangles exist) takes ~10 GB to load: one at a time


def cad_bytes(m: dict) -> int:
    """Size of a design's .3dm plus the .stl next to it (from the storage listing)."""
    return tm.cad_size(m)


def run(batch, workers, per_worker, progress):
    """Measure a batch in worker processes that are replaced every few files
    (rhino3dm/trimesh don't give memory back between files)."""
    with ProcessPoolExecutor(workers, max_tasks_per_child=per_worker) as pool:
        for fut in as_completed([pool.submit(measure, a) for a in batch]):
            m, slug, res = fut.result()
            res["design_id"] = m["design_id"]
            (MEASURED / f"{slug}.json").write_text(json.dumps(res))
            progress(res)


def calibrate(designs: dict) -> dict | None:
    """Grams of 18K per cm3 of CAD metal, from rings that have both a job card
    and a closed CAD metal surface. The median ignores the few files whose
    CAD holds a different variant than the card."""
    pairs = []
    for d in designs.values():
        card, cad = d.get("card"), d.get("cad")
        if not card or not cad or cad.get("category") != "ring":
            continue
        k18, vol = card["weights_g"].get("18k"), cad.get("metal_volume_mm3", 0)
        if k18 and vol > 0 and cad.get("metal_open_share", 1) <= 0.01:
            pairs.append(k18 / (vol / 1000))
    if len(pairs) < 20:
        return None
    med = statistics.median(pairs)
    return {"k18_g_per_cm3": round(med, 3), "rings": len(pairs),
            "within_10pct": round(sum(abs(p / med - 1) <= 0.1 for p in pairs) / len(pairs), 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="measure every CAD file again")
    ap.add_argument("--cards-only", action="store_true", help="re-read the job cards, keep the CAD measurements")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--tasks-per-worker", type=int, default=10,
                    help="replace each worker after this many files: rhino3dm/trimesh memory is not given back "
                         "between files, and six long-lived workers grew to 8-14 GB each and swapped")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    if not cad_to_glb.ds.get().ready():
        raise SystemExit(f"{cad_to_glb.ds.get().label()} is not available")

    meta = [json.loads(line) for line in tm.META.read_text().splitlines()]
    MEASURED.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    cards = find_cards(meta)
    print(f"{len(cards)} designs have a job card ({time.time() - t0:.0f} s)", flush=True)

    todo = []
    for m in meta:
        cat = tm.CATEGORY.get(m["category"])
        if not cat or not m.get("has_cad"):
            continue
        slug = tm.slug_for(m)
        if args.cards_only or ((MEASURED / f"{slug}.json").exists() and not args.force):
            continue
        todo.append((m, slug, cat))
    if args.limit:
        todo = todo[: args.limit]
    sizes = {a[1]: cad_bytes(a[0]) for a in todo}
    small = [a for a in todo if sizes[a[1]] < BIG_BYTES]
    big = sorted((a for a in todo if sizes[a[1]] >= BIG_BYTES), key=lambda a: sizes[a[1]])
    print(f"{len(todo)} CAD files to measure ({len(big)} over {BIG_BYTES // 1_000_000} MB, measured one at a time last)",
          flush=True)
    t0, counts, done = time.time(), {}, [0]

    def progress(res):
        done[0] += 1
        key = res.get("status", "?").split(":")[0]
        counts[key] = counts.get(key, 0) + 1
        n = done[0]
        if n % 50 == 0 or n == len(todo) or n > len(small):
            rate = n / (time.time() - t0)
            print(f"{n}/{len(todo)}  {rate:.1f}/s  {counts}", flush=True)

    if small:
        run(small, args.workers, args.tasks_per_worker, progress)
    if big:
        run(big, 1, 1, progress)

    designs = {}
    for m in meta:
        key = tm.design_key(m)
        entry = {}
        if key in cards:
            entry["card"] = cards[key]
        f = MEASURED / f"{tm.slug_for(m)}.json"
        if f.exists():
            res = json.loads(f.read_text())
            if res.get("status") == "ok":
                res.pop("slug", None)
                res.pop("design_id", None)
                entry["cad"] = res
        if entry:
            designs[key] = entry
    cal = calibrate(designs)
    SPECS.write_text(json.dumps({
        "built": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "calibration": cal,
        "designs": designs,
    }))
    n_cad = sum("cad" in d for d in designs.values())
    print(f"wrote {SPECS.relative_to(ROOT)}: {len(designs)} designs, {len(cards)} job cards, "
          f"{n_cad} measured CAD files; calibration {cal}", flush=True)


if __name__ == "__main__":
    main()
