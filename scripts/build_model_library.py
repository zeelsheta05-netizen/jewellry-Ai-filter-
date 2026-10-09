"""Build the ready-made try-on model library: model photos the customer picks
in the "Try on" dialog instead of taking photos of themselves.

    .venv/bin/python scripts/build_model_library.py SOURCE_DIR [--labels labels.json]

SOURCE_DIR holds the model photos (jpg / png / webp). Each photo is measured
with the SAME analysis the site uses for body photos (MediaPipe hand / face /
pose in headless Chrome, scripts in jewelsearch/static/tryon/), so a piece is
placed on a library model exactly like on a customer's own photo:

    hand photo   -> rings, bracelets   (21 hand landmarks)
    portrait     -> earrings           (earlobes + real scale from the iris)
                 -> pendants, necklaces (neck sides + neck notch)

A part is offered only when every check of that part passed (the same checks
a customer's photo must pass), so a model is never offered for a piece it
can't wear correctly. The photo itself is stored unchanged in quality (JPEG
q95 of the original pixels, never resized); a small thumbnail is made for
the picker.

Output: data/tryon/library/<id>.jpg, <id>.thumb.jpg, library.json
labels.json (optional): {"file stem": "Label"} or {"file stem": {"label": "...", "tags": ["Type filter values"]}}
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent.parent
STATIC = ROOT / "jewelsearch" / "static"
OUT = ROOT / "data" / "tryon" / "library"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
THUMB = 480

PAGE = """<!doctype html><html><body><script type="module">
import { analyse } from "/tryon/body-analysis.js";
const todo = await (await fetch("/todo")).json();
for (const t of todo) {
  const img = new Image();
  img.src = "/photos/" + encodeURIComponent(t.file);
  await img.decode();
  const out = {};
  for (const part of ["hand", "face", "neck"]) {
    try {
      const r = await analyse(part, img, "IMAGE");
      out[part] = { found: r.found, allOk: r.allOk, checks: r.checks, tip: r.tip, anchors: r.anchors };
    } catch (e) { out[part] = { error: String(e) }; }
  }
  await fetch("/result", { method: "POST", body: JSON.stringify({ file: t.file, width: img.naturalWidth, height: img.naturalHeight, out }) });
}
await fetch("/done", { method: "POST" });
</script></body></html>"""


def slug(stem: str) -> str:
    return re.sub(r"[^a-z0-9-]+", "-", stem.lower()).strip("-")[:60] or "model"


def analyse_all(src: Path, files: list[str]) -> dict[str, dict]:
    results: dict[str, dict] = {}
    done = threading.Event()
    site = Path(tempfile.mkdtemp(prefix="model-lib-"))
    (site / "index.html").write_text(PAGE)
    os.symlink(STATIC / "vendor", site / "vendor")
    os.symlink(STATIC / "tryon", site / "tryon")
    os.symlink(src.resolve(), site / "photos")

    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=str(site), **kw)

        def log_message(self, *a):
            pass

        def end_headers(self):
            self.send_header("Cache-Control", "no-store")
            super().end_headers()

        def do_GET(self):
            if self.path == "/todo":
                body = json.dumps([{"file": f} for f in files]).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            super().do_GET()

        def do_POST(self):
            data = self.rfile.read(int(self.headers.get("Content-Length") or 0))
            if self.path == "/result":
                d = json.loads(data)
                results[d["file"]] = d
                print(f"  measured {d['file']}", flush=True)
            elif self.path == "/done":
                done.set()
            self.send_response(204)
            self.end_headers()

    Handler.extensions_map = {**SimpleHTTPRequestHandler.extensions_map, ".mjs": "text/javascript",
                              ".js": "text/javascript", ".wasm": "application/wasm"}
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    profile = tempfile.mkdtemp(prefix="model-lib-chrome-")
    chrome = subprocess.Popen([CHROME, "--headless=new", f"--user-data-dir={profile}", "--use-angle=metal", "--enable-gpu",
                               "--remote-debugging-port=0", f"http://127.0.0.1:{server.server_address[1]}/index.html"],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        t0 = time.time()
        while not done.is_set():
            if chrome.poll() is not None:
                raise SystemExit("Chrome exited before finishing")
            if time.time() - t0 > 60 + 30 * len(files):
                raise SystemExit("Timed out measuring the photos")
            time.sleep(0.5)
    finally:
        chrome.kill()
        server.shutdown()
        shutil.rmtree(site, ignore_errors=True)
        shutil.rmtree(profile, ignore_errors=True)
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("source", type=Path)
    ap.add_argument("--labels", type=Path)
    args = ap.parse_args()
    labels = json.loads(args.labels.read_text()) if args.labels else {}
    files = sorted(f.name for f in args.source.iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
    if not files:
        raise SystemExit("No photos in " + str(args.source))
    print(f"measuring {len(files)} photos", flush=True)
    measured = analyse_all(args.source, files)

    OUT.mkdir(parents=True, exist_ok=True)
    items, skipped = [], []
    for name in files:
        m = measured.get(name)
        stem = Path(name).stem
        parts = {p: r["anchors"] for p, r in (m or {}).get("out", {}).items() if r.get("allOk") and r.get("anchors")}
        if not parts:
            why = {p: r.get("tip") or r.get("error") for p, r in (m or {}).get("out", {}).items() if r.get("found") or r.get("error")}
            skipped.append((name, why))
            continue
        mid = slug(stem)
        with Image.open(args.source / name) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            if im.size != (m["width"], m["height"]):
                raise SystemExit(f"{name}: size changed between measuring and saving")
            im.save(OUT / f"{mid}.jpg", "JPEG", quality=95, subsampling=0)
            th = im.copy()
            th.thumbnail((THUMB, THUMB * 2), Image.LANCZOS)
            th.save(OUT / f"{mid}.thumb.jpg", "JPEG", quality=85)
        lab = labels.get(stem, "")
        lab = lab if isinstance(lab, dict) else {"label": lab}
        items.append({"id": mid, "label": lab.get("label", ""), "tags": lab.get("tags", []), "width": m["width"],
                      "height": m["height"], "parts": parts})
        print(f"{mid}: {', '.join(parts)}", flush=True)
    (OUT / "library.json").write_text(json.dumps({"built": time.time(), "items": items}, indent=1))
    for name, why in skipped:
        print(f"SKIPPED {name}: {why}", flush=True)
    print(f"{len(items)} models in the library, {len(skipped)} skipped", flush=True)


if __name__ == "__main__":
    sys.exit(main())
