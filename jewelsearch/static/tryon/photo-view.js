// Instant try-on on a saved body photo: draws a design on the user's own
// hand / face / neck picture using the anchors stored with the photo.
//
//   ring      on the ring finger (same pose + finger occluders as the live view)
//   bracelet  around the wrist, occluded by a wrist cylinder
//   earrings  one on each earlobe (mirrored for the other ear), hanging straight
//             down; small studs sit centred on the lobe
//   pendant   on a thin chain from the sides of the neck
//   necklace  the flat CAD layout, cut where it goes behind the neck
//
// Scale is real: px per mm comes from the hand (knuckle spacing) or the iris.

import * as THREE from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { MeshoptDecoder } from "three/addons/libs/meshopt_decoder.module.js";
import { RoomEnvironment } from "three/addons/environments/RoomEnvironment.js";
import { handFrame, toPixels, wristFrame } from "./hand-tracker.js";
import { NECK_VERSION, neckAnchors } from "./neck-geometry.js";

export const METALS = {
  yellow_gold: { label: "Yellow gold", color: [1.0, 0.766, 0.336] },
  rose_gold: { label: "Rose gold", color: [0.955, 0.637, 0.538] },
  white_gold: { label: "White gold", color: [0.86, 0.855, 0.84] },
};
const AVG_FINGER_MM = 16.9;   // ring size India 13 / US 6.5
const WRIST_MM = 56;          // average adult wrist width, used for bracelets
const STUD_MAX_MM = 13;       // shorter earrings are studs: centred on the lobe
const CHAIN_DROP_MM = 55;     // 45 cm chain: pendant bail ~5.5 cm below the neck notch
const MIN_DROP_MM = 12;       // shortest chain (a photo with little chest below the neck)
const LINK_MM = 1.6;          // cable chain: link length; wire is ~0.4 mm

const loader = new GLTFLoader().setMeshoptDecoder(MeshoptDecoder);

export class PhotoTryOn {
  constructor(canvas) {
    this.renderer = new THREE.WebGLRenderer({ canvas, alpha: true, antialias: true, preserveDrawingBuffer: true });
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.toneMapping = THREE.NeutralToneMapping;
    this.renderer.localClippingEnabled = true;
    this.renderer.setClearColor(0x000000, 0);
    this.scene = new THREE.Scene();
    const pmrem = new THREE.PMREMGenerator(this.renderer);
    this.scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.03).texture;
    pmrem.dispose();
    this.camera = new THREE.OrthographicCamera(0, 1, 0, -1, -1e5, 1e5);
    this.camera.position.z = 1e4;
    this.metal = new THREE.MeshPhysicalMaterial({ metalness: 1, roughness: 0.1, envMapIntensity: 1.9, clearcoat: 0.4, clearcoatRoughness: 0.05 });
    this.stone = new THREE.MeshPhysicalMaterial({ color: 0xffffff, metalness: 0.9, roughness: 0.015, envMapIntensity: 2.6,
      iridescence: 0.12, iridescenceIOR: 2.0, iridescenceThicknessRange: [300, 600],
      emissive: 0x5a5a5a });   // a diamond never looks black: its facets always return some light
    this.occluderMat = new THREE.MeshBasicMaterial({ colorWrite: false });
    this.root = new THREE.Group();
    this.scene.add(this.root);
    this.gain = [1, 1, 1];
    this.setMetal("yellow_gold");
  }

  setMetal(key) {
    this.metalKey = key;
    const [r, g, b] = (METALS[key] || METALS.yellow_gold).color;
    const [gr, gg, gb] = this.gain;
    this.metal.color.setRGB(r * gr, g * gg, b * gb);
    this.stone.color.setRGB(gr, gg, gb);
    // real-photo mode: each metal is its own catalogue render
    if (this.photoMode && this.photoMode.urls[key] && this.photoMode.metal !== key) {
      this.photoMode.metal = key;
      this._showPhoto(this.photoMode.urls[key]);
      return;
    }
    this.render();
  }

  // Light the piece like the photo: the same colour cast (grey-world white
  // balance) and brightness as the skin it sits on. A dim, warm office photo
  // gets dimmer, warmer gold instead of a studio-bright sticker.
  _matchPhoto(image, a) {
    this.gain = [1, 1, 1];
    this.renderer.toneMappingExposure = 1;
    if (!image?.naturalWidth) return this.setMetal(this.metalKey);
    const c = document.createElement("canvas");
    c.width = 48; c.height = 48;
    const ctx = c.getContext("2d", { willReadFrequently: true });
    ctx.drawImage(image, 0, 0, 48, 48);
    const d = ctx.getImageData(0, 0, 48, 48).data;
    const mean = [0, 0, 0];
    for (let i = 0; i < d.length; i += 4) for (let k = 0; k < 3; k++) mean[k] += d[i + k];
    const grey = (mean[0] + mean[1] + mean[2]) / 3;
    this.gain = mean.map((m) => Math.min(1.12, Math.max(0.88, m / grey)));
    // brightness around where the piece sits (whole photo for hands)
    const at = a ? (a.notch || a.lobeL) : null;
    let sum = 0, n = 0;
    for (let y = 0; y < 48; y++) for (let x = 0; x < 48; x++) {
      if (at && Math.hypot(x / 48 - at.x, y / 48 - at.y) > 0.15) continue;
      const i = (y * 48 + x) * 4;
      sum += 0.2126 * d[i] + 0.7152 * d[i + 1] + 0.0722 * d[i + 2]; n++;
    }
    const luma = n ? sum / n / 255 : 0.5;
    this.renderer.toneMappingExposure = Math.min(1.1, Math.max(0.6, 0.5 + luma * 0.8));
    this.setMetal(this.metalKey);
  }

  // photo: { width, height, anchors }, design: { category, model(meta), url },
  // opts.image: the loaded photo, to match its light and colour
  async show(photo, design, opts = {}) {
    const W = photo.width, H = photo.height;
    // Render at the photo's own resolution: a sharper render than the photo
    // around it is what makes jewellery look pasted on.
    this.renderer.setPixelRatio(1);
    this.renderer.setSize(W, H, false);
    Object.assign(this.camera, { left: 0, right: W, top: 0, bottom: -H });
    this.camera.updateProjectionMatrix();

    this.photoMode = null;
    if (opts.metal) this.metalKey = opts.metal;
    let a = photo.anchors;
    const meta = design.model, cat = design.category;
    const px = (p) => new THREE.Vector3(p.x * W, -p.y * H, 0);
    // neck photos saved before the anchors were corrected: recompute them
    if (a.shoulderL && (a.v || 0) < NECK_VERSION) {
      a = { ...a, ...neckAnchors({ chin: a.chin, shoulderL: a.shoulderL, shoulderR: a.shoulderR,
        halfWidth: (a.neckR.x - a.neckL.x) / 2 * 0.95, pxPerMm: a.pxPerMm, H }) };
    }
    this._matchPhoto(opts.image, cat === "ring" || cat === "bracelet" ? null : a);

    // Real-photo mode: the design's own catalogue render instead of the 3D model
    if (design.photos?.[this.metalKey] && (cat === "ring" || cat === "earrings")) {
      this.photoMode = { urls: design.photos, metal: this.metalKey, a, cat, meta, W, H };
      return this._showPhoto(design.photos[this.metalKey]);
    }
    const gltf = await loader.loadAsync(design.url);
    this.root.clear();

    // Flat pieces face the camera and mirror the dim wall behind it: light them
    // from the room's bright side, like a jewellery photo.
    const flat = cat !== "ring" && cat !== "bracelet";
    this.scene.environmentIntensity = flat ? 1.5 : 1;
    this.scene.environmentRotation.set(flat ? -0.5 : 0, flat ? Math.PI * 0.8 : 0, 0);
    this.metal.clippingPlanes = [];   // only a necklace clips (behind the neck)
    this.stone.clippingPlanes = [];

    if (cat === "ring" || cat === "bracelet") {
      const pts = toPixels(a.landmarks.map(([x, y, z]) => ({ x, y, z })), W, H);
      const isRing = cat === "ring";
      const f = isRing ? handFrame(pts, a.isRight) : wristFrame(pts, a.isRight);
      // The ring is shown at the size it was designed in, fitting the finger:
      // on a photo we can't know the user's ring size better than that.
      const bodyMm = isRing ? (meta.ring?.inner_diameter_mm || AVG_FINGER_MM) : WRIST_MM;
      const bodyPx = isRing ? f.fingerPx : f.wristPx;
      const piece = this._piece(gltf);
      const s = (bodyPx / bodyMm) * 1000;   // px per metre of model
      const { position: p, xAxis: x, yAxis: y, zAxis: z } = f.pose;
      piece.matrixAutoUpdate = false;
      piece.matrix.set(x.x * s, y.x * s, z.x * s, p.x, x.y * s, y.y * s, z.y * s, p.y, x.z * s, y.z * s, z.z * s, p.z, 0, 0, 0, 1);
      this.root.add(piece);
      // the finger / wrist hides the back of the band
      const occ = new THREE.Mesh(new THREE.CylinderGeometry(1, 1, 1, 32), this.occluderMat);
      occ.renderOrder = -1;
      occ.matrixAutoUpdate = false;
      const r = (bodyPx / 2) * (isRing ? 0.94 : 0.9), h = bodyPx * 2.5;
      const q = new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 1, 0), new THREE.Vector3(y.x, y.y, y.z));
      occ.matrix.compose(new THREE.Vector3(p.x, p.y, p.z), q, new THREE.Vector3(r, h, r));
      this.root.add(occ);
    } else if (cat === "earrings") {
      const s = a.pxPerMm * 1000;
      const heightMm = meta.size_mm?.[1] || 20;
      const stud = heightMm <= STUD_MAX_MM;
      for (const [lobe, mirror] of [[a.lobeL, 1], [a.lobeR, -1]]) {
        const piece = this._piece(gltf);
        piece.scale.set(s * mirror, s, s);        // the other ear gets the mirror image
        const pos = px(lobe);
        if (stud) pos.y += (heightMm / 2) * a.pxPerMm;   // model origin is its top: centre it on the lobe
        piece.position.copy(pos);
        this.root.add(piece);
      }
    } else if (cat === "pendant" || cat === "necklace") {
      const s = a.pxPerMm * 1000;
      if (cat === "pendant") {
        // The pendant must hang fully inside the photo: shorten the chain on a
        // photo cropped close under the neck instead of cutting the pendant off.
        const heightMm = meta.size_mm?.[1] || 20;
        const roomMm = (H * (1 - a.notch.y)) / a.pxPerMm - heightMm - 4;
        const dropMm = Math.max(MIN_DROP_MM, Math.min(CHAIN_DROP_MM, roomMm));
        const bail = px(a.notch).add(new THREE.Vector3(0, -dropMm * a.pxPerMm, 0));
        const piece = this._piece(gltf);
        piece.scale.setScalar(s);
        piece.position.copy(bail);
        piece.position.z = 5;
        this.root.add(piece);
        // the chain runs through the bail, ~1 mm below the pendant's top
        const through = bail.clone().add(new THREE.Vector3(0, -1 * a.pxPerMm, 0));
        this.root.add(this._chain(px(a.neckL), through, px(a.neckR), a.pxPerMm));
      } else {
        this.root.add(this._necklace(gltf, meta, a, px, s));
      }
    }
    this.render();
  }

  // ---------- real-photo mode ----------
  // The catalogue's front render (transparent PNG) placed on the body photo.
  // It looks real and is exactly the design; it can't turn, so it is used
  // for the saved-photo try-on of pieces that face the camera.
  async _showPhoto(url) {
    const pm = this.photoMode, token = (this._photoToken = (this._photoToken || 0) + 1);
    // the catalogue render is far sharper than the photo: draw it at 2x so
    // close-up and saved pictures keep its detail
    this.renderer.setPixelRatio(2);
    this.renderer.setSize(pm.W, pm.H, false);
    const img = new Image();
    img.src = url;
    await img.decode();
    if (token !== this._photoToken) return;   // a newer metal was picked meanwhile
    const { a, cat, meta, W, H } = pm;
    const px = (p) => new THREE.Vector3(p.x * W, -p.y * H, 0);
    this.root.clear();
    if (cat === "earrings") {
      // the render shows the pair: left one on the image-left ear, right one on the other
      const [left, right] = splitPair(img);
      const heightMm = meta.size_mm?.[1] || 14;
      for (const [lobe, part] of [[a.lobeL, left], [a.lobeR, right]]) {
        const s = (heightMm * a.pxPerMm) / part.height;
        const w = part.width * s, h = part.height * s;
        const pos = px(lobe);
        if (heightMm > STUD_MAX_MM) pos.y -= h / 2;   // a drop hangs from the lobe; a stud is centred on it
        this.root.add(this._plane(part, w, h, pos, 0));
      }
    } else {
      const pts = toPixels(a.landmarks.map(([x, y, z]) => ({ x, y, z })), W, H);
      const f = handFrame(pts, a.isRight);
      const ring = ringFront(img);
      // the band's outer width over the finger: inner = finger, plus the band
      const outerPerInner = meta.size_mm?.[0] && meta.ring?.inner_diameter_mm
        ? meta.size_mm[0] / meta.ring.inner_diameter_mm : 1.2;
      const s = (f.fingerPx * outerPerInner) / ring.canvas.width;
      const { position: p, yAxis: y } = f.pose;
      const angle = Math.atan2(y.y, y.x) - Math.PI / 2;   // the image's up = towards the fingertip
      this.root.add(this._plane(ring.canvas, ring.canvas.width * s, ring.canvas.height * s, new THREE.Vector3(p.x, p.y, 0), angle));
    }
    this.render();
  }

  _plane(canvas, w, h, pos, angle) {
    const tex = new THREE.CanvasTexture(canvas);
    tex.colorSpace = THREE.SRGBColorSpace;
    tex.anisotropy = this.renderer.capabilities.getMaxAnisotropy();
    // the photo's colour cast and brightness, like the 3D pieces get
    const e = this.renderer.toneMappingExposure;
    const [gr, gg, gb] = this.gain;
    const mat = new THREE.MeshBasicMaterial({ map: tex, transparent: true, toneMapped: false, depthWrite: false });
    mat.color.setRGB(Math.min(1, gr * e), Math.min(1, gg * e), Math.min(1, gb * e));
    const mesh = new THREE.Mesh(new THREE.PlaneGeometry(w, h), mat);
    mesh.position.copy(pos);
    mesh.position.z = 5;
    mesh.rotation.z = angle;
    return mesh;
  }

  // The loaded model with our materials.
  _piece(gltf) {
    const g = gltf.scene.clone(true);
    g.traverse((o) => {
      if (!o.isMesh) return;
      const stone = o.isInstancedMesh || /stone/i.test(o.material?.name || "");
      o.material = stone ? this.stone : this.metal;
    });
    return g;
  }

  // Cable chain hanging from both sides of the neck to the pendant's bail.
  // Real links, alternately turned 90 degrees, so the chain catches the
  // light like metal and not a drawn line.
  _chain(left, bail, right, pxPerMm) {
    const half = (side) => {
      const pts = [];
      for (let i = 0; i <= 24; i++) {
        const t = i / 24;
        // leaves the neck already heading inward, arms bowing slightly out
        // over the collarbones, meeting at an angle under the pendant's weight
        pts.push(new THREE.Vector3(side.x + (bail.x - side.x) * (0.45 * t + 0.55 * t * t),
          side.y + (bail.y - side.y) * (1 - (1 - t) ** 1.12), 2));
      }
      return pts;
    };
    const curve = new THREE.CatmullRomCurve3([...half(left), ...half(right).reverse().slice(1)], false, "centripetal");
    const pitch = LINK_MM * 0.8 * pxPerMm;   // links overlap by the wire thickness
    const count = Math.max(2, Math.round(curve.getLength() / pitch));
    const link = new THREE.TorusGeometry(0.5, 0.14, 6, 12).scale(1.6 / 1.0, 1, 1);   // oval, 1.6 x 1 (units: link widths)
    const mesh = new THREE.InstancedMesh(link, this.metal, count);
    const w = Math.max(1.4, LINK_MM / 1.6 * pxPerMm);   // link width in px, never thinner than a hairline
    const m = new THREE.Matrix4(), q = new THREE.Quaternion(), twist = new THREE.Quaternion();
    const X = new THREE.Vector3(1, 0, 0), scale = new THREE.Vector3(w, w, w);
    for (let i = 0; i < count; i++) {
      const u = (i + 0.5) / count;
      q.setFromUnitVectors(X, curve.getTangentAt(u));
      twist.setFromAxisAngle(X, i % 2 ? Math.PI / 2 : 0.35);   // every other link turned
      m.compose(curve.getPointAt(u), q.multiply(twist), scale);
      mesh.setMatrixAt(i, m);
    }
    return mesh;
  }

  // A necklace CAD file is laid out flat. Worn, its upper part goes round the
  // back of the neck: find the height where the layout is as wide as the
  // neck, pin that height to the sides of the neck, and clip everything above.
  _necklace(gltf, meta, a, px, s) {
    const g = this._piece(gltf);
    g.scale.setScalar(s);
    g.updateMatrixWorld(true);
    const verts = [];
    g.traverse((o) => {
      if (!o.isMesh || o.isInstancedMesh) return;
      const pos = o.geometry.getAttribute("position"), v = new THREE.Vector3();
      for (let i = 0; i < pos.count; i += 3) verts.push(v.fromBufferAttribute(pos, i).applyMatrix4(o.matrixWorld).clone());
    });
    const box = new THREE.Box3().setFromPoints(verts);
    const neckL = px(a.neckL), neckR = px(a.neckR);
    const neckW = neckR.x - neckL.x;
    // width of the layout at each height, from the top down
    const bins = 60, step = (box.max.y - box.min.y) / bins;
    let cut = null, widest = { w: -1 };
    for (let b = 0; b < bins; b++) {
      const y0 = box.max.y - (b + 1) * step, y1 = box.max.y - b * step;
      let lo = Infinity, hi = -Infinity;
      for (const v of verts) if (v.y >= y0 && v.y < y1) { lo = Math.min(lo, v.x); hi = Math.max(hi, v.x); }
      if (lo > hi) continue;
      const row = { y: (y0 + y1) / 2, cx: (lo + hi) / 2, w: hi - lo };
      if (row.w > widest.w) widest = row;
      if (row.w >= neckW) { cut = row; break; }
    }
    // a choker narrower than this neck: pin its widest point to the neck sides
    cut ??= widest;
    const cutY = cut.y, cx = cut.cx;
    // move so the cut height sits on the neck sides, centred on the neck
    const target = neckL.clone().lerp(neckR, 0.5);
    g.position.set(target.x - cx, target.y - cutY, 5);
    const clip = new THREE.Plane(new THREE.Vector3(0, -1, 0), target.y);   // keep y <= neck-side height
    this.metal.clippingPlanes = [clip];
    this.stone.clippingPlanes = [clip];
    return g;
  }

  render() {
    this.renderer.render(this.scene, this.camera);
  }

  // Photo + jewellery as one JPEG blob.
  async snapshot(img) {
    const c = document.createElement("canvas");
    const gl = this.renderer.domElement;   // may be 2x the photo (real-photo mode)
    c.width = gl.width; c.height = gl.height;
    const ctx = c.getContext("2d");
    ctx.imageSmoothingQuality = "high";
    ctx.drawImage(img, 0, 0, c.width, c.height);
    this.render();
    ctx.drawImage(this.renderer.domElement, 0, 0, c.width, c.height);
    return new Promise((r) => c.toBlob(r, "image/jpeg", 0.92));
  }
}

// ---------- catalogue render helpers (real-photo mode) ----------

export { ringFront, splitPair };   // also used to preview a cut-out

function alphaOf(img) {
  const c = document.createElement("canvas");
  c.width = img.naturalWidth || img.width; c.height = img.naturalHeight || img.height;
  const ctx = c.getContext("2d", { willReadFrequently: true });
  ctx.drawImage(img, 0, 0);
  return { c, ctx, data: ctx.getImageData(0, 0, c.width, c.height) };
}

// Crop a canvas to its visible pixels.
function trim(c, ctx) {
  const { width: w, height: h } = c, d = ctx.getImageData(0, 0, w, h).data;
  let x0 = w, y0 = h, x1 = -1, y1 = -1;
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    if (d[(y * w + x) * 4 + 3] > 10) { if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y; }
  }
  const out = document.createElement("canvas");
  out.width = Math.max(1, x1 - x0 + 1); out.height = Math.max(1, y1 - y0 + 1);
  out.getContext("2d").drawImage(c, x0, y0, out.width, out.height, 0, 0, out.width, out.height);
  return out;
}

// The catalogue shows earrings as a pair: split at the emptiest column near the middle.
function splitPair(img) {
  const { c, ctx } = alphaOf(img), { width: w, height: h } = c;
  dropShadow(ctx, w, h);
  const d = ctx.getImageData(0, 0, w, h).data;
  let best = Math.round(w / 2), least = Infinity;
  for (let x = Math.round(w * 0.3); x < w * 0.7; x++) {
    let sum = 0;
    for (let y = 0; y < h; y++) sum += d[(y * w + x) * 4 + 3];
    if (sum < least) { least = sum; best = x; }
  }
  const half = (x0, x1) => {
    const k = document.createElement("canvas");
    k.width = x1 - x0; k.height = h;
    const kctx = k.getContext("2d", { willReadFrequently: true });
    kctx.drawImage(c, x0, 0, x1 - x0, h, 0, 0, x1 - x0, h);
    return trim(k, kctx);
  };
  return [half(0, best), half(best, w)];
}

// A ring's front render is taken from the front and a little above: the top
// arc is the back of the band (seen from inside), the lower arc its front,
// and the head stands in front of both. On the back of a hand the back of the
// band is under the finger, so it is removed:
//   1. its lower edge is found in clean columns (where nothing overlaps it,
//      its first opaque stretch is short) and fitted with a parabola;
//   2. everything above that curve, across the hole, is cleared;
//   3. except where the head overlaps it: there, from the topmost diamond down
//      is kept (a halo), so only the band above the head goes.
function ringFront(img) {
  const { c, ctx, data } = alphaOf(img), { width: w, height: h } = c, d = data.data;
  dropShadow(ctx, w, h);
  const px = ctx.getImageData(0, 0, w, h).data;
  const opaque = (x, y) => px[(y * w + x) * 4 + 3] >= 128;
  const stoneLike = (x, y) => {   // bright and colourless: a diamond, not gold
    const i = (y * w + x) * 4, r = px[i], g = px[i + 1], b = px[i + 2];
    // gold highlights are bright too, but stay warm (blue well below red)
    return px[i + 3] >= 128 && Math.min(r, g, b) > 150 && Math.max(r, g, b) - Math.min(r, g, b) < 28 && b >= r - 12;
  };
  const merge = Math.max(2, Math.round(h * 0.012));
  const firstRun = (x) => {   // [start, end) of the first opaque stretch, tiny gaps ignored
    let y = 0;
    while (y < h && !opaque(x, y)) y++;
    const start = y;
    for (;;) {
      while (y < h && opaque(x, y)) y++;
      let g = 0;
      while (y + g < h && !opaque(x, y + g) && g <= merge) g++;
      if (g > merge || y + g >= h) return [start, y];
      y += g;
    }
  };
  const edges = [], overlapped = [];
  for (let x = Math.round(w * 0.1); x < w * 0.9; x += Math.max(1, Math.round(w * 0.01))) {
    const [a, b] = firstRun(x);
    if (b >= h) continue;
    if (b - a > h * 0.3) overlapped.push(x); else edges.push([x, b]);
  }
  const left = edges.filter(([x]) => x < w / 2).length, right = edges.length - left;
  if (left >= 2 && right >= 2) {
    const m = w / 2;
    let s0 = 0, s1 = 0, s2 = 0, t0 = 0, t1 = 0;
    for (const [x, y] of edges) { const u = (x - m) ** 2; s0++; s1 += u; s2 += u * u; t0 += y; t1 += u * y; }
    const den = s0 * s2 - s1 * s1;
    const a = den ? (s0 * t1 - s1 * t0) / den : 0, b = (t0 - a * s1) / s0;
    const curve = (x) => Math.round(a * (x - m) ** 2 + b);
    // the head: the overlapped columns around the middle (the band's sides
    // are long stretches too, but out at the edges)
    const mid = overlapped.filter((x) => Math.abs(x - m) < w * 0.25);
    const headL = mid.length ? Math.min(...mid) : Infinity, headR = mid.length ? Math.max(...mid) : -Infinity;
    // the head's top, column by column: the highest diamond above the curve
    // in or next to that column (a halo); elsewhere the curve decides
    // A diamond is a chunky patch of such pixels; a highlight on the gold
    // rim is only a thin line of them: require a dense square around the pixel.
    const sat = new Uint32Array((w + 1) * (h + 1));   // summed-area table of stoneLike
    for (let y = 0; y < h; y++) {
      let row = 0;
      for (let x = 0; x < w; x++) { row += stoneLike(x, y) ? 1 : 0; sat[(y + 1) * (w + 1) + x + 1] = sat[y * (w + 1) + x + 1] + row; }
    }
    const r = Math.max(3, Math.round(h * 0.02));
    const dense = (x, y) => {
      const x0 = Math.max(0, x - r), x1 = Math.min(w, x + r + 1), y0 = Math.max(0, y - r), y1 = Math.min(h, y + r + 1);
      const n = sat[y1 * (w + 1) + x1] - sat[y0 * (w + 1) + x1] - sat[y1 * (w + 1) + x0] + sat[y0 * (w + 1) + x0];
      return n >= 0.3 * (x1 - x0) * (y1 - y0);
    };
    const top = new Array(w).fill(Infinity);
    for (let x = Math.max(0, headL); x <= Math.min(w - 1, headR); x++)
      for (let y = 0; y < curve(x); y++) if (stoneLike(x, y) && dense(x, y)) { top[x] = y; break; }
    // join the diamond tops into the arch's outline (straight across the
    // gaps between stones), so the band showing around the arch goes too
    const cols = [];
    for (let x = 0; x < w; x++) if (top[x] < Infinity) cols.push(x);
    const arch = new Array(w).fill(Infinity);
    for (let i = 0; i + 1 < cols.length; i++) {
      const [xa, xb] = [cols[i], cols[i + 1]];
      for (let x = xa; x <= xb; x++) arch[x] = top[xa] + (top[xb] - top[xa]) * (x - xa) / Math.max(1, xb - xa);
    }
    if (cols.length === 1) arch[cols[0]] = top[cols[0]];
    // Never cut into the head: in its columns only the band above a
    // diamond arch goes; a gold head (loops, a crown) is kept whole, even if
    // a little band shows behind it. A changed design is worse than that.
    const margin = Math.round(h * 0.01), pad = Math.round(w * 0.02);
    for (let x = 0; x < w; x++) {
      const inHead = x >= headL - pad && x <= headR + pad;
      const cut = !inHead ? curve(x) : arch[x] < curve(x) ? Math.max(0, Math.round(arch[x]) - margin) : 0;
      if (cut > 0) ctx.clearRect(x, 0, 1, cut);
    }
  }
  return { canvas: trim(c, ctx) };
}

// The renders sit on a soft grey floor shadow: semi-transparent pixels. Keep
// only the piece (its anti-aliased edge stays, slightly firmer).
function dropShadow(ctx, w, h) {
  const im = ctx.getImageData(0, 0, w, h), d = im.data;
  for (let i = 3; i < d.length; i += 4) d[i] = d[i] < 110 ? 0 : Math.min(255, Math.round((d[i] - 110) * 255 / 110));
  ctx.putImageData(im, 0, 0);
}
