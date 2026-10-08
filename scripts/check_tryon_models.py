"""Check every try-on model against its own catalogue photo, so a model that
doesn't look like the design never reaches a customer.

    .venv/bin/python scripts/check_tryon_models.py            # new / changed models
    .venv/bin/python scripts/check_tryon_models.py --all      # re-check everything
    .venv/bin/python scripts/check_tryon_models.py --since 2026-09-29T17:20   # only a partial run's models

Why: the CAD -> 3D conversion sometimes changes a design (a ring head that
doesn't touch the band is dropped, a file holds both earrings of a pair or a
different variant, a bracelet is modelled flat). Showing that on the
customer's hand is worse than showing no try-on at all.

How: each model is rendered on white in plain views (headless Chrome, the
same three.js the site uses) and embedded with the search model (SigLIP2).
The best view's similarity to the design's own front photo in the search
index decides. Checked on a hand-labelled sample: >= 0.78 kept 18 of 21
correct models and blocked 8 of 10 broken ones.

Output: data/tryon/fidelity.json {slug: {"sim", "pass", "mtime"}}. The
server shows "Try on" only for models that passed.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from datetime import datetime
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from jewelsearch.embedder import Embedder  # noqa: E402

MODELS = ROOT / "data" / "tryon" / "models"
MAP = ROOT / "data" / "tryon" / "designs.json"
OUT = ROOT / "data" / "tryon" / "fidelity.json"
PARTIAL = OUT.with_suffix(".partial.json")   # progress; published to OUT only when complete
META = ROOT / "data" / "index" / "meta.jsonl"
FRONT = ROOT / "data" / "index" / "front.npy"
VENDOR = ROOT / "jewelsearch" / "static" / "vendor"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
PASS_SIM = 0.78
KIND = {"ring": "ring", "bracelet": "ring", "earrings": "earring"}   # catalogue category -> how to shoot it

# Renders each model in plain views, the way the catalogue photographs that
# kind of piece, and posts them back:
#   ring, bracelet  upright with the head on top: from the front a little above
#                   (the catalogue's usual shot), 3/4, along the finger, from above
#   earring         a pair side by side (the catalogue always shows the pair),
#                   front and turned
#   others          front, turned 30 degrees, and nearly side-on
STUDIO = """<!doctype html><html><head><script type="importmap">
{"imports":{"three":"/vendor/three/three.module.js","three/addons/":"/vendor/three/addons/"}}</script></head>
<body style="margin:0"><canvas id="c" width="256" height="256"></canvas><script type="module">
import * as THREE from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { MeshoptDecoder } from "three/addons/libs/meshopt_decoder.module.js";
import { RoomEnvironment } from "three/addons/environments/RoomEnvironment.js";
const r = new THREE.WebGLRenderer({ canvas: document.getElementById("c"), antialias: true, preserveDrawingBuffer: true });
r.setClearColor(0xffffff, 1); r.outputColorSpace = THREE.SRGBColorSpace; r.toneMapping = THREE.NeutralToneMapping;
const scene = new THREE.Scene(), pm = new THREE.PMREMGenerator(r);
scene.environment = pm.fromScene(new RoomEnvironment(), 0.03).texture;
const metal = new THREE.MeshPhysicalMaterial({ color: new THREE.Color(1, .766, .336), metalness: 1, roughness: .15, envMapIntensity: 1.6 });
const stone = new THREE.MeshPhysicalMaterial({ color: 0xffffff, metalness: .9, roughness: .02, envMapIntensity: 2.4, emissive: 0x5a5a5a });
const loader = new GLTFLoader().setMeshoptDecoder(MeshoptDecoder);
const VIEWS = {
  ring: [[0, -.93, .37], [.45, -.83, .33], [0, -1, .001], [0, .001, 1]],
  earring: [[0, .001, 1], [.5, .001, .87]],
  other: [[0, .001, 1], [.5, .001, .87], [.97, .001, .25]],
};
async function shots(slug, kind) {
  const g = (await loader.loadAsync(`/models/${slug}.glb`)).scene;
  g.traverse(o => { if (o.isMesh) o.material = (o.isInstancedMesh || /stone/i.test(o.material?.name || "")) ? stone : metal; });
  scene.clear(); scene.add(g);
  if (kind === "earring") {   // the pair: a mirrored twin to the left
    const w = new THREE.Box3().setFromObject(g).getSize(new THREE.Vector3()).x;
    const twin = g.clone(true);
    twin.scale.x = -1;
    g.position.x = w * .65; twin.position.x = -w * .65;
    scene.add(twin);
  }
  const box = new THREE.Box3().setFromObject(scene), c = box.getCenter(new THREE.Vector3()), rad = box.getSize(new THREE.Vector3()).length() / 2;
  const out = [];
  for (const v of VIEWS[kind]) {
    const cam = new THREE.PerspectiveCamera(25, 1, rad / 50, rad * 50);
    cam.position.copy(c).addScaledVector(new THREE.Vector3(...v).normalize(), rad / Math.tan(THREE.MathUtils.degToRad(12.5)) * 1.05);
    const headUp = kind === "ring" && Math.abs(v[1]) > .3;   // ring frame: +Y finger axis, +Z head
    cam.up.set(0, headUp ? 0 : 1, headUp ? 1 : 0);
    cam.lookAt(c);
    r.render(scene, cam); out.push(r.domElement.toDataURL("image/png"));
  }
  g.traverse(o => o.geometry?.dispose());
  return out;
}
const todo = await (await fetch("/todo")).json();
for (const t of todo) {
  let views = [];
  try { views = await shots(t.slug, t.kind); } catch (e) { console.error(t.slug, e); }
  await fetch("/shot", { method: "POST", body: JSON.stringify({ slug: t.slug, views }) });
}
await fetch("/done", { method: "POST" });
</script></body></html>"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="re-check models that were already checked")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--software", action="store_true", help="render on the CPU (no GPU)")
    ap.add_argument("--save", help="also write every render to this folder (for review)")
    ap.add_argument("--since", help="only models converted after this time (e.g. 2026-09-29T17:20); "
                                    "older ones stay hidden. For reviewing part of a conversion run.")
    args = ap.parse_args()
    since = datetime.fromisoformat(args.since).timestamp() if args.since else 0

    meta = [json.loads(l) for l in META.read_text().splitlines()]
    row_of = {f'{m["design_id"]}|{m["folders"][0]}': i for i, m in enumerate(meta)}
    front = np.load(FRONT)
    mapping = json.loads(MAP.read_text())
    # resume an interrupted run; the server keeps using the last complete check
    last = PARTIAL if PARTIAL.exists() else OUT
    results = json.loads(last.read_text()) if last.exists() else {}
    live = {slug for slug in mapping.values()
            if (MODELS / f"{slug}.glb").exists() and (MODELS / f"{slug}.glb").stat().st_mtime >= since}
    results = {k: v for k, v in results.items() if k in live}   # gone, or older than --since

    todo, own = [], {}
    for key, slug in mapping.items():
        glb = MODELS / f"{slug}.glb"
        if key not in row_of or slug not in live:
            continue
        own[slug] = row_of[key]
        done = results.get(slug)
        if done and not args.all and done.get("mtime") == int(glb.stat().st_mtime):
            continue
        cat = meta[row_of[key]]["category"]
        todo.append({"slug": slug, "kind": KIND.get(cat, "other")})
    if args.limit:
        todo = todo[: args.limit]
    print(f"{len(todo)} models to check, {len(results)} checked before", flush=True)
    if not todo:
        if PARTIAL.exists():
            PARTIAL.replace(OUT)
        return

    shots: list[tuple[str, list[bytes]]] = []
    finished = threading.Event()
    lock = threading.Lock()

    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=str(site), **kw)

        def log_message(self, *a):
            pass

        def do_GET(self):
            if self.path == "/todo":
                body = json.dumps(todo).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            super().do_GET()

        def do_POST(self):
            data = self.rfile.read(int(self.headers.get("Content-Length") or 0))
            if self.path == "/shot":
                d = json.loads(data)
                pngs = [base64.b64decode(u.split(",", 1)[1]) for u in d["views"]]
                with lock:
                    shots.append((d["slug"], pngs))
            elif self.path == "/done":
                finished.set()
            self.send_response(204)
            self.end_headers()

    if args.save:
        Path(args.save).mkdir(parents=True, exist_ok=True)
    site = Path(tempfile.mkdtemp(prefix="tryon-check-"))
    (site / "studio.html").write_text(STUDIO)
    os.symlink(VENDOR, site / "vendor")
    os.symlink(MODELS, site / "models")
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    profile = tempfile.mkdtemp(prefix="tryon-check-chrome-")
    gl = ["--use-angle=swiftshader", "--enable-unsafe-swiftshader"] if args.software else ["--use-angle=metal", "--enable-gpu"]
    chrome = subprocess.Popen([CHROME, "--headless=new", f"--user-data-dir={profile}", *gl, "--hide-scrollbars", "--remote-debugging-port=0",
                               f"http://127.0.0.1:{server.server_address[1]}/studio.html"],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    emb = Embedder()
    checked, t0 = 0, time.time()

    def score(batch):
        nonlocal checked
        for slug, pngs in batch:
            glb = MODELS / f"{slug}.glb"
            if not pngs:   # the model failed to load or render
                results[slug] = {"sim": 0.0, "pass": False, "mtime": int(glb.stat().st_mtime)}
                continue
            if args.save:
                for k, p in enumerate(pngs):
                    Path(args.save, f"{slug}_{k}.png").write_bytes(p)
            e = emb.images([Image.open(io.BytesIO(p)) for p in pngs])
            sim = float((e @ front[own[slug]]).max())
            results[slug] = {"sim": round(sim, 3), "pass": sim >= PASS_SIM, "mtime": int(glb.stat().st_mtime)}
        checked += len(batch)
        PARTIAL.write_text(json.dumps(results))
        rate = checked / (time.time() - t0)
        passed = sum(1 for s, _ in batch if results[s]["pass"])
        print(f"{checked}/{len(todo)}  {rate:.1f}/s  eta {(len(todo) - checked) / rate / 60:.0f} min  "
              f"(last batch {passed}/{len(batch)} passed)", flush=True)

    try:
        while not (finished.is_set() and not shots):
            if chrome.poll() is not None and not finished.is_set():
                raise SystemExit("Chrome exited before finishing")
            with lock:
                batch, shots[:] = shots[:], []
            if batch:
                score(batch)
            else:
                time.sleep(0.5)
    finally:
        chrome.kill()
        server.shutdown()
        shutil.rmtree(site, ignore_errors=True)
        shutil.rmtree(profile, ignore_errors=True)
    ok = sum(1 for r in results.values() if r["pass"])
    if args.limit:   # a sample: never publish a check that skipped models
        print(f"sample done: {ok}/{len(results)} pass; not published (run without --limit)", flush=True)
        return
    PARTIAL.replace(OUT)
    print(f"done: {ok}/{len(results)} models pass (similarity >= {PASS_SIM})", flush=True)


if __name__ == "__main__":
    main()
