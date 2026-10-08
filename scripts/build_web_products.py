#!/usr/bin/env python3
"""Collect the "From the web" pool (DEMO): current products of a few jewellery
shops, read once by the image models (jewelsearch/webproducts.py).

For each shop in webproducts.SHOPS:
  1. read its robots.txt; skip the shop if the product feed or product pages are
     disallowed for us;
  2. read its public product feed (/products.json, 250 per page, a pause between
     pages) up to --per-shop products of our five types that are in stock;
  3. download each product's first picture at 640 px from the shop's image
     server, keep it as the card (320) and preview (640) pictures, and read it
     with SigLIP2 and DINOv2 (small batches, like the other index scripts).

Writes data/web_products/{products.jsonl, siglip.npy, title.npy, dino.npy, info.json}
(all replaced together at the end) and data/web_products/img/. The app reads the
pool at start-up. Run:  .venv/bin/python scripts/build_web_products.py [--per-shop 500] [--cap kushals=3000]
Pictures already in data/web_products/img/ are reused, so a stopped run can be started again.
"""
import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from urllib.robotparser import RobotFileParser

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from jewelsearch import linksearch, photo, webproducts as wp  # noqa: E402

ROBOT_NAME = "DesignFinder"
FEED_PAGE = 100      # Limelight's 250-product pages passed 20 MB
PAUSE = 1.0          # seconds between feed pages of one shop
EMBED_SIZE = 640     # picture downloaded once: the preview size, read by the models too
BATCH = 16


def get(url: str, accept: str, limit: int, seconds: float = 30):
    return linksearch.fetch(url, accept, limit, time.monotonic() + seconds)


def robots_for(base: str) -> RobotFileParser:
    rp = RobotFileParser()
    try:
        rp.parse(get(base + "/robots.txt", "text/plain", 1_000_000).data.decode("utf-8", "replace").splitlines())
    except linksearch.LinkError:
        rp.parse([])   # no robots.txt: nothing is disallowed
    return rp


def collect(shop: dict, per_shop: int) -> tuple[list[dict], str]:
    base = shop["base"]
    rp = robots_for(base)
    if not rp.can_fetch(ROBOT_NAME, base + "/products.json") or not rp.can_fetch(ROBOT_NAME, base + "/products/x"):
        return [], "robots.txt disallows the product feed"
    try:
        currency = json.loads(get(base + "/meta.json", "application/json", 500_000).data).get("currency") or "INR"
    except (linksearch.LinkError, ValueError):
        currency = "INR"
    out, seen, page = [], set(), 1
    while len(out) < per_shop and page <= 300:
        try:
            data = json.loads(get(f"{base}/products.json?limit={FEED_PAGE}&page={page}", "application/json",
                                  20_000_000, 60).data)
        except (linksearch.LinkError, ValueError) as e:
            return out, f"feed stopped at page {page}: {getattr(e, 'message', e)}"
        prods = data.get("products") or []
        if not prods:
            break
        for prod in prods:
            p = wp.from_feed(shop, prod, currency)
            if p and p["id"] not in seen:
                seen.add(p["id"])
                out.append(p)
        page += 1
        time.sleep(PAUSE)
    return out[:per_shop], "ok"


def download(p: dict):
    from PIL import Image
    have = [wp.IMG_DIR / f"{p['id']}_{size}.webp" for size in wp.SIZES]
    if all(f.exists() for f in have):   # kept from an earlier run: read it, don't fetch it again
        try:
            with Image.open(have[-1]) as im:
                return im.convert("RGB")
        except OSError:
            pass
    try:
        f = get(wp.sized_url(p["image"], EMBED_SIZE), "image/avif,image/webp,image/png,image/jpeg,image/*;q=0.8",
                photo.MAX_BYTES, 25)
        im = photo.read(f.data)
    except (linksearch.LinkError, photo.PhotoError):
        return None
    for size in wp.SIZES:
        wp.save_picture(p["id"], im, size)
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-shop", type=int, default=500)
    ap.add_argument("--shops", nargs="*", help="shop keys (default: all)")
    ap.add_argument("--cap", nargs="*", default=[], help="per-shop limits, e.g. kushals=3000")
    args = ap.parse_args()

    shops = [s for s in wp.SHOPS if not args.shops or s["key"] in args.shops]
    products, report = [], {}
    for shop in shops:
        caps = dict(c.split("=") for c in args.cap)
        got, status = collect(shop, int(caps.get(shop["key"], args.per_shop)))
        report[shop["key"]] = {"name": shop["name"], "found": len(got), "status": status}
        print(f"{shop['name']}: {len(got)} products ({status})", flush=True)
        products += got

    import torch
    from jewelsearch.dino import MODEL as DINO_MODEL, Dino
    from jewelsearch.embedder import Embedder
    from jewelsearch.config import MODEL_NAME
    emb, dino = Embedder(), Dino()
    keep, sig, title, dv = [], [], [], []
    t0 = time.time()
    with ThreadPoolExecutor(4) as pool:
        for start in range(0, len(products), BATCH):
            chunk = products[start:start + BATCH]
            ims = list(pool.map(download, chunk))
            pairs = [(p, im) for p, im in zip(chunk, ims) if im is not None]
            if not pairs:
                continue
            crops = []
            for _, im in pairs:
                v = photo.views(im)
                crops.append(v.get("piece", v["full"]))
            sig.append(emb.images(crops))
            dv.append(dino.images(crops))
            title.append(emb.texts([f"a photo of {p['title']}" for p, _ in pairs]))
            keep += [p for p, _ in pairs]
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            done = start + len(chunk)
            if done % (BATCH * 10) == 0 or done == len(products):
                print(f"  {done}/{len(products)} read, {len(keep)} kept, {time.time() - t0:.0f}s", flush=True)

    # colour the shop didn't name: read from the picture, only when the model is sure
    from jewelsearch import dna
    from jewelsearch.config import METALS
    from jewelsearch.attributes import TEMPERATURE
    mt = np.stack([emb.texts([dna.METAL_PROMPT.format(metal=dna.METAL_WORDS[m])])[0] for m in METALS])
    sig_all = np.concatenate(sig)
    for p, q in zip(keep, sig_all):
        if not p["colours"]:
            mp = dna.softmax(q @ mt.T * TEMPERATURE)
            best = int(np.argmax(mp))
            if mp[best] >= dna.METAL_MIN:
                p["colours"], p["colour_from"] = [METALS[best]], "picture"

    for p in keep:
        report[p["shop"]]["kept"] = report[p["shop"]].get("kept", 0) + 1
    by_cat = {}
    for p in keep:
        by_cat[p["category"]] = by_cat.get(p["category"], 0) + 1
    wp.DIR.mkdir(parents=True, exist_ok=True)
    tmp = {n: wp.DIR / f".{n}.tmp" for n in ("products.jsonl", "siglip.npy", "title.npy", "dino.npy", "info.json")}
    tmp["products.jsonl"].write_text("".join(json.dumps(p, ensure_ascii=False) + "\n" for p in keep))
    for name, arrs in (("siglip.npy", sig), ("title.npy", title), ("dino.npy", dv)):
        with open(tmp[name], "wb") as f:
            np.save(f, np.concatenate(arrs).astype(np.float32))
    tmp["info.json"].write_text(json.dumps({
        "built": datetime.now(timezone.utc).isoformat(timespec="seconds"), "count": len(keep),
        "shops": {k: {"name": v["name"], "count": v.get("kept", 0)} for k, v in report.items()},
        "categories": by_cat, "report": report, "siglip": MODEL_NAME, "dino": DINO_MODEL}, indent=1))
    for name, t in tmp.items():
        t.replace(wp.DIR / name)
    print(f"done: {len(keep)} products, {by_cat}, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
