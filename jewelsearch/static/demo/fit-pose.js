// The catalogue photo's camera angle for a 3D model: the direction and roll
// whose silhouette best overlaps the photo's outline (its alpha). Both
// outlines are cropped to their bounding box and scaled to the same square,
// so only the shape counts, not size or position. Run once per design
// (look.html?fit=1); the result goes into designs.json.

import * as THREE from "three";

const N = 48;          // coarse search resolution
const FINE = 96;

function normalise(get, w, h, box, n) {
  const out = new Uint8Array(n * n);
  if (box.x1 < 0) return out;
  const bw = box.x1 - box.x0 + 1, bh = box.y1 - box.y0 + 1, s = Math.max(bw, bh) / n;
  const ox = box.x0 - (Math.max(bw, bh) - bw) / 2, oy = box.y0 - (Math.max(bw, bh) - bh) / 2;
  for (let j = 0; j < n; j++) for (let i = 0; i < n; i++) {
    const x = Math.floor(ox + (i + 0.5) * s), y = Math.floor(oy + (j + 0.5) * s);
    out[j * n + i] = x >= 0 && y >= 0 && x < w && y < h && get(x, y) ? 1 : 0;
  }
  return out;
}

function iou(a, b) {
  let i = 0, u = 0;
  for (let k = 0; k < a.length; k++) { i += a[k] & b[k]; u += a[k] | b[k]; }
  return u ? i / u : 0;
}

async function photoMask(url, n) {
  const img = new Image();
  img.src = url;
  await img.decode();
  const W = 400, H = Math.round(400 * img.naturalHeight / img.naturalWidth);
  const c = document.createElement("canvas"); c.width = W; c.height = H;
  const x = c.getContext("2d"); x.drawImage(img, 0, 0, W, H);
  const px = x.getImageData(0, 0, W, H).data;
  // an opaque photo (a few renders sit on white): the piece is what isn't near-white
  let transparent = 0;
  for (let k = 3; k < px.length; k += 4) transparent += px[k] < 250;
  const inside = transparent > px.length / 40
    ? (X, Y) => px[(Y * W + X) * 4 + 3] > 40
    : (X, Y) => { const k = (Y * W + X) * 4; return Math.min(px[k], px[k + 1], px[k + 2]) < 235; };
  let x0 = W, y0 = H, x1 = -1, y1 = -1;
  for (let Y = 0; Y < H; Y++) for (let X = 0; X < W; X++) if (inside(X, Y)) { x0 = Math.min(x0, X); x1 = Math.max(x1, X); y0 = Math.min(y0, Y); y1 = Math.max(y1, Y); }
  return normalise(inside, W, H, { x0, y0, x1, y1 }, n);
}

// front: only angles from the front (+Z): a pair of earrings has the same outline
// from behind, where only the posts would show
export async function fitPose(renderer, root, photoUrl, front = false) {
  const white = new THREE.MeshBasicMaterial({ color: 0xffffff });
  const sil = root.clone(true);
  sil.traverse((o) => { if (o.isMesh) o.material = white; });
  const scene = new THREE.Scene();
  scene.add(sil);
  const box = new THREE.Box3().setFromObject(sil), c = box.getCenter(new THREE.Vector3()), rad = box.getSize(new THREE.Vector3()).length() / 2;
  const cam = new THREE.PerspectiveCamera(25, 1, rad / 50, rad * 50);
  const dist = rad / Math.tan(THREE.MathUtils.degToRad(12.5)) * 1.05;
  const targets = { [N]: new THREE.WebGLRenderTarget(N, N), [FINE]: new THREE.WebGLRenderTarget(FINE, FINE) };
  const buf = { [N]: new Uint8Array(N * N * 4), [FINE]: new Uint8Array(FINE * FINE * 4) };
  const photo = { [N]: await photoMask(photoUrl, N), [FINE]: await photoMask(photoUrl, FINE) };
  const prevClear = renderer.getClearAlpha();
  renderer.setClearColor(0x000000, 0);

  const score = (dir, roll, n) => {
    cam.position.copy(c).addScaledVector(dir, dist);
    // an up vector perpendicular to the view, turned by the roll angle
    const ref = Math.abs(dir.z) < 0.9 ? new THREE.Vector3(0, 0, 1) : new THREE.Vector3(0, 1, 0);
    const up = ref.clone().sub(dir.clone().multiplyScalar(ref.dot(dir))).normalize().applyAxisAngle(dir, roll);
    cam.up.copy(up);
    cam.lookAt(c);
    renderer.setRenderTarget(targets[n]);
    renderer.render(scene, cam);
    renderer.readRenderTargetPixels(targets[n], 0, 0, n, n, buf[n]);
    const px = buf[n];
    // render targets are read bottom-up: flip rows to match the photo
    const get = (x, y) => px[((n - 1 - y) * n + x) * 4] > 127;
    let x0 = n, y0 = n, x1 = -1, y1 = -1;
    for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) if (get(x, y)) { x0 = Math.min(x0, x); x1 = Math.max(x1, x); y0 = Math.min(y0, y); y1 = Math.max(y1, y); }
    return { s: iou(normalise(get, n, n, { x0, y0, x1, y1 }, n), photo[n]), up };
  };

  // coarse: 400 directions on the sphere x 24 rolls
  let best = { s: -1 };
  const K = 400;
  for (let k = 0; k < K; k++) {
    const y = 1 - (k + 0.5) / K * 2, r = Math.sqrt(1 - y * y), phi = k * Math.PI * (3 - Math.sqrt(5));
    const dir = new THREE.Vector3(Math.cos(phi) * r, y, Math.sin(phi) * r);
    if (front && dir.z < 0.2) continue;
    for (let j = 0; j < 24; j++) {
      const roll = j / 24 * Math.PI * 2;
      const res = score(dir, roll, N);
      if (res.s > best.s) best = { ...res, dir: dir.clone(), roll };
    }
  }
  // fine: around the best, at a higher resolution
  let fine = { ...score(best.dir, best.roll, FINE), dir: best.dir, roll: best.roll };
  for (let step = 0.12; step > 0.01; step /= 2) {
    let moved = true;
    while (moved) {
      moved = false;
      const axes = [new THREE.Vector3(1, 0, 0), new THREE.Vector3(0, 1, 0), new THREE.Vector3(0, 0, 1)];
      for (const a of axes) for (const sgn of [-1, 1]) {
        const dir = fine.dir.clone().addScaledVector(a, sgn * step).normalize();
        if (front && dir.z < 0.2) continue;
        for (const dr of [-step, 0, step]) {
          const res = score(dir, fine.roll + dr * 2, FINE);
          if (res.s > fine.s + 1e-4) { fine = { ...res, dir, roll: fine.roll + dr * 2 }; moved = true; }
        }
      }
    }
  }
  renderer.setRenderTarget(null);
  renderer.setClearColor(0xffffff, prevClear);
  Object.values(targets).forEach((t) => t.dispose());
  return { dir: fine.dir.toArray().map((v) => +v.toFixed(4)), up: fine.up.toArray().map((v) => +v.toFixed(4)), iou: +fine.s.toFixed(3), coarse: +best.s.toFixed(3) };
}
