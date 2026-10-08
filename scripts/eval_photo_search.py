#!/usr/bin/env python3
"""How well search by photo finds a design, measured on simulated shopper photos.

There are no real shopper photos with known answers yet, so each test photo is
made from a catalogue render the index has NOT seen: the same design in another
metal colour (the index embeds white gold), often from another angle, placed on
one of four kinds of background, rotated, resized, recoloured, blurred and
JPEG-compressed:

  catalogue  on white, filling the frame (a screenshot from a website or catalogue)
  plain      on a velvet / card / gradient background with a soft shadow
  skin       on a skin tone, as if held in the hand, with a shadow
  busy       small, on a wood / fabric / stone surface

The full engine runs on each (SearchEngine.read_photo + search_photo, no words)
and the report gives, per background and per type:
  found@1 / found@8   the design (or its stone-cut variant) is the first result / on the first page
  type                the type read from the photo is right; how often it is sure enough to filter
  metal               the metal colour read from the photo is the colour of the render
  same design         how often "Same design" is shown, and how often it is right

Real photos (a hand, a shop window, a phone screen) are harder than these, so
treat the numbers as an upper bound and a regression check.

  .venv/bin/python scripts/eval_photo_search.py                # ~60 photos per type, ~5 min
  .venv/bin/python scripts/eval_photo_search.py --per-type 20  # quick check
  .venv/bin/python scripts/eval_photo_search.py --set PHOTO_DIAMOND_WEIGHT=0.6   # try another setting
"""
import argparse
import io
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from diamond_labels import labels as diamond_answers  # noqa: E402  (job card / CAD: only to check, never to search)
from jewelsearch import photo  # noqa: E402
from jewelsearch.config import CROPS, thumb_name  # noqa: E402
from jewelsearch.search import SearchEngine  # noqa: E402

CONDITIONS = ["catalogue", "plain", "skin", "busy"]
PLAIN = [(18, 18, 22), (25, 30, 60), (90, 20, 35), (215, 200, 175), (120, 120, 125), (40, 60, 45), (235, 235, 230)]
SKIN = [(224, 172, 140), (198, 134, 94), (160, 105, 70), (241, 194, 160), (120, 80, 55)]
SURFACES = [((120, 80, 50), (170, 120, 80)), ((60, 60, 70), (110, 100, 120)), ((200, 195, 190), (150, 145, 150)),
            ((80, 30, 40), (140, 70, 80)), ((30, 50, 40), (90, 110, 90)), ((210, 180, 150), (160, 110, 90))]
SCALE = {"catalogue": (0.8, 1.0), "plain": (0.5, 0.85), "skin": (0.4, 0.75), "busy": (0.3, 0.6)}


def _gradient(w, h, c1, c2, rng):
    t = np.linspace(0, 1, h)[:, None, None] if rng.random() < 0.5 else np.linspace(0, 1, w)[None, :, None]
    return np.broadcast_to((1 - t) * np.asarray(c1)[None, None] + t * np.asarray(c2)[None, None], (h, w, 3)).astype(np.float32)


def background(cond, w, h, rng):
    if cond == "catalogue":
        return np.full((h, w, 3), 255, np.float32)
    if cond in ("plain", "skin"):
        c = np.array((PLAIN if cond == "plain" else SKIN)[rng.integers(len(PLAIN if cond == "plain" else SKIN))], np.float32)
        lo, hi = (0.8, 1.3) if cond == "plain" else (0.85, 1.1)
        return _gradient(w, h, c * lo, np.minimum(255, c * hi), rng) + rng.normal(0, 3.5, (h, w, 3))
    c1, c2 = (np.array(c, np.float32) for c in SURFACES[rng.integers(len(SURFACES))])
    t = np.asarray(Image.fromarray((rng.random((6, 6)) * 255).astype(np.uint8)).resize((w, h), Image.BICUBIC),
                   np.float32)[..., None] / 255
    grain = np.sin(np.linspace(0, rng.uniform(20, 60), w))[None, :, None] * 9 if rng.random() < 0.5 else 0
    return np.broadcast_to((1 - t) * c1 + t * c2 + grain, (h, w, 3)) + rng.normal(0, 6, (h, w, 3))


def fake_photo(crop: Image.Image, cond: str, rng) -> bytes:
    """A catalogue crop (RGBA) -> JPEG bytes of a simulated shopper photo."""
    a, b = [(1, 1), (3, 4), (4, 3)][rng.integers(3)]
    long = int(rng.integers(640, 1000))
    w, h = (long, long * b // a) if a >= b else (long * a // b, long)
    canvas = Image.fromarray(np.clip(background(cond, w, h, rng), 0, 255).astype(np.uint8)).convert("RGBA")
    side = int(min(w, h) * rng.uniform(*SCALE[cond]))
    piece = crop.resize((side, side), Image.LANCZOS)
    if cond != "catalogue":
        piece = piece.rotate(float(rng.uniform(-18, 18)), resample=Image.BICUBIC)
    x, y = int(rng.integers(0, max(1, w - side))), int(rng.integers(0, max(1, h - side)))
    if cond in ("plain", "skin"):   # soft cast shadow
        sh = Image.fromarray((np.asarray(piece)[..., 3] * 0.35).astype(np.uint8)).filter(ImageFilter.GaussianBlur(side / 40))
        shadow = Image.new("RGBA", piece.size, (0, 0, 0, 0))
        shadow.putalpha(sh)
        canvas.alpha_composite(shadow, (min(w - side, x + side // 40), min(h - side, y + side // 30)))
    canvas.alpha_composite(piece, (x, y))
    im = ImageEnhance.Brightness(canvas.convert("RGB")).enhance(float(rng.uniform(0.85, 1.12)))
    im = ImageEnhance.Contrast(im).enhance(float(rng.uniform(0.85, 1.15)))
    arr = np.asarray(im, np.float32)
    warm = rng.uniform(-0.06, 0.06)   # white balance
    arr[..., 0] *= 1 + warm
    arr[..., 2] *= 1 - warm
    im = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    blur = float(rng.uniform(0, 1.2))
    if blur > 0.3:
        im = im.filter(ImageFilter.GaussianBlur(blur))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=int(rng.integers(55, 90)))
    return buf.getvalue()


def pick(engine: SearchEngine, per_type: int, seed: int) -> list[dict]:
    """Designs that have a render in a metal other than the indexed one."""
    rnd = random.Random(seed)
    by_type = defaultdict(list)
    for uid, m in engine.by_uid.items():
        if any(mt != m["embed_metal"] for mt in m["images"]):
            by_type[m["category"]].append(uid)
    tests = []
    for cat in sorted(by_type):
        uids = by_type[cat]
        rnd.shuffle(uids)
        for uid in uids[:per_type]:
            m = engine.by_uid[uid]
            metal = rnd.choice(sorted(mt for mt in m["images"] if mt != m["embed_metal"]))
            views = m["images"][metal]
            view = m["front_view"] if (rnd.random() < 0.5 and m["front_view"] in views) else rnd.choice(sorted(views))
            crop = CROPS / thumb_name(views[view])
            if crop.exists():
                tests.append({"uid": uid, "type": cat, "metal": metal, "crop": crop, "cond": CONDITIONS[len(tests) % 4]})
    return tests


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--per-type", type=int, default=60)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--json", help="also write the per-photo results here")
    ap.add_argument("--set", action="append", default=[], metavar="NAME=VALUE",
                    help="try another value of a search.py setting, e.g. --set PHOTO_DIAMOND_WEIGHT=0.6")
    args = ap.parse_args()
    from jewelsearch import search
    for item in args.set:
        name, value = item.split("=", 1)
        if not hasattr(search, name):
            raise SystemExit(f"search.py has no setting {name}")
        setattr(search, name, type(getattr(search, name))(value))
        print(f"{name} = {getattr(search, name)}")
    engine = SearchEngine()
    tests = pick(engine, args.per_type, args.seed)
    answers = {u: diamond_answers(m) for u, m in engine.by_uid.items()}
    rng = np.random.default_rng(args.seed)
    rows, t0 = [], time.time()
    for i, t in enumerate(tests):
        data = fake_photo(Image.open(t["crop"]).convert("RGBA"), t["cond"], rng)
        t1 = time.perf_counter()
        pq = engine.read_photo(photo.read(data))
        res = engine.search_photo(pq)
        ms = (time.perf_counter() - t1) * 1000
        fam = engine.by_uid[t["uid"]]["family"]
        fams = [engine.by_uid[c["uid"]]["family"] for c in res["results"]]
        same = pq.dna["same"]
        # do the shown designs share the photo's diamonds? (checked against job cards / CAD files)
        true_cut, true_lay = answers[t["uid"]]
        shown = [c["uid"] for c in res["results"]]
        lay_known = [answers[u][1] for u in shown if answers[u][1]]
        cut_known = [answers[u][0] for u in shown if answers[u][0]]
        dia = pq.dna["diamonds"]
        rows.append({
            "dia_read_layout": dia["layout"]["value"] if dia.get("layout") else None, "dia_true_layout": true_lay,
            "dia_read_cut": dia["centre_cut"]["value"] if dia.get("centre_cut") else None, "dia_true_cut": true_cut,
            "dia_layout": (sum(x == true_lay for x in lay_known) / len(lay_known)) if true_lay and lay_known else None,
            "dia_cut": (sum(x == true_cut for x in cut_known) / len(cut_known)) if true_cut and cut_known else None,
            "uid": t["uid"], "type": t["type"], "cond": t["cond"], "metal": t["metal"], "ms": round(ms),
            "found1": bool(fams) and fams[0] == fam, "found8": fam in fams,
            "type_read": pq.dna["type"]["value"], "type_sure": pq.dna["type"]["sure"],
            "metal_read": pq.dna["metal"]["metal"], "metal_p": pq.dna["metal"]["p"],
            "same_shown": same is not None, "same_right": same is not None and engine.by_uid[same["uid"]]["family"] == fam,
        })
        if i % 50 == 0:
            print(f"{i}/{len(tests)}  {time.time() - t0:.0f}s", flush=True)

    def line(label, rs):
        n = len(rs)
        sure = [r for r in rs if r["type_sure"]]
        return (f"{label:12s} {n:4d}   found@1 {sum(r['found1'] for r in rs) / n:5.1%}   found@8 {sum(r['found8'] for r in rs) / n:5.1%}"
                f"   type {sum(r['type_read'] == r['type'] for r in rs) / n:5.1%} (sure {len(sure) / n:4.0%},"
                f" right when sure {sum(r['type_read'] == r['type'] for r in sure) / max(len(sure), 1):5.1%})"
                f"   metal {sum(r['metal_read'] == r['metal'] for r in rs) / n:5.1%}")
    print()
    print(line("all", rows))
    for c in CONDITIONS:
        print(line(c, [r for r in rows if r["cond"] == c]))
    for cat in sorted({r["type"] for r in rows}):
        print(line(cat, [r for r in rows if r["type"] == cat]))
    shown = [r for r in rows if r["same_shown"]]
    print(f"\n\"Same design\" shown for {len(shown) / len(rows):.1%} of photos, right {sum(r['same_right'] for r in shown) / max(len(shown), 1):.1%} of the time")
    print("(the diamond reader was trained on these designs; train_diamond_dna.py reports it on unseen ones)")
    for what in ("layout", "cut"):
        known = [r for r in rows if r[f"dia_true_{what}"]]
        shown = [r for r in known if r[f"dia_read_{what}"]]
        if known:
            print(f"diamond {what} read from the photo: shown for {len(shown) / len(known):.0%} of {len(known)} photos with an answer, "
                  f"right {np.mean([r[f'dia_read_{what}'] == r[f'dia_true_{what}'] for r in shown]) if shown else float('nan'):.0%} of the time")
    for key, label in (("dia_layout", "same diamond layout"), ("dia_cut", "same centre-stone cut")):
        vals = [r[key] for r in rows if r[key] is not None]
        if vals:
            print(f"results with the photo's {label} (checked against job cards / CAD): {np.mean(vals):.1%} (over {len(vals)} photos)")
    print(f"median time per photo {np.median([r['ms'] for r in rows]):.0f} ms (read + search)")
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
