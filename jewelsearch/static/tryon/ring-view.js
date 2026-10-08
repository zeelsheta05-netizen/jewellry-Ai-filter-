// Draws the ring over the camera picture.
//
// - Orthographic camera in video pixels, so landmark coordinates are used as-is.
// - Ring model: metres, +Y = finger axis, +Z = head (see scripts/cad_to_glb.py).
// - Resizing: the shank is pushed radially so the inner diameter matches the
//   user's ring size while the head keeps its real size (how a jeweller resizes).
// - Occlusion: invisible capsules on every finger write depth only, so the
//   back of the band disappears behind the finger.
// - Lighting: studio environment for reflections, exposure and tint follow
//   the camera picture around the hand.
// - Blending in: the ring is softened to the camera's own sharpness and
//   fades in/out instead of popping. (A contact shadow was tried and removed:
//   on real users it read as a dark smear around the band.)

import * as THREE from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { MeshoptDecoder } from "three/addons/libs/meshopt_decoder.module.js";
import { RoomEnvironment } from "three/addons/environments/RoomEnvironment.js";

// Linear-light base colours of polished alloys (PBR F0 values)
export const METALS = {
  yellow: { label: "Yellow gold", color: [1.0, 0.766, 0.336] },
  rose: { label: "Rose gold", color: [0.955, 0.637, 0.538] },
  white: { label: "White gold", color: [0.86, 0.855, 0.84] },
};

// Finger segments for the occluders: [from, to, radius relative to the ring finger]
const SEGMENTS = [
  [1, 2, 1.1], [2, 3, 1.05], [3, 4, 0.95],          // thumb
  [5, 6, 1.0], [6, 7, 0.9], [7, 8, 0.8],            // index
  [9, 10, 1.05], [10, 11, 0.95], [11, 12, 0.85],    // middle
  [13, 14, 1.0], [14, 15, 0.9], [15, 16, 0.8],      // ring
  [17, 18, 0.85], [18, 19, 0.78], [19, 20, 0.7],    // little
];
// The occluder sits a touch inside the skin so the band never looks sunk in
const OCCLUDER_SHRINK = 0.94;

export class RingView {
  constructor(canvas) {
    this.renderer = new THREE.WebGLRenderer({ canvas, alpha: true, antialias: true, powerPreference: "high-performance" });
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.toneMapping = THREE.NeutralToneMapping;
    this.renderer.setClearColor(0x000000, 0);

    this.scene = new THREE.Scene();
    const pmrem = new THREE.PMREMGenerator(this.renderer);
    this.scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.03).texture;
    pmrem.dispose();

    this.camera = new THREE.OrthographicCamera(0, 1, 0, -1, -1e5, 1e5);
    this.camera.position.z = 1e4;

    this.anchor = new THREE.Group();          // follows the ring finger
    this.anchor.matrixAutoUpdate = false;
    this.scene.add(this.anchor);
    this.ring = null;                          // model, scaled mm -> px inside the anchor

    this.occluderMat = new THREE.MeshBasicMaterial({ colorWrite: false });
    this.occluders = SEGMENTS.map(() => {
      const m = new THREE.Mesh(new THREE.CylinderGeometry(1, 1, 1, 16), this.occluderMat);
      m.renderOrder = -1;
      m.matrixAutoUpdate = false;
      this.scene.add(m);
      return m;
    });

    this.metalMat = new THREE.MeshPhysicalMaterial({
      metalness: 1, roughness: 0.1, envMapIntensity: 1.9, clearcoat: 0.4, clearcoatRoughness: 0.05,
    });
    // Cheap diamond: a clear stone shows almost no body colour, just the
    // facets reflecting the room. So: near-mirror facets (flat normals come
    // from the model), cool white, plus a thin-film sheen for a hint of fire.
    // Real refraction is reserved for centre stones (later part).
    this.stoneMat = stoneMaterial();
    this.canvas = canvas;
    this.alpha = 0;          // fade in / out
    this.lastHand = null;    // pose kept while fading out, and for a frozen photo
    this.setMetal("yellow");
    this.tint = new THREE.Color(1, 1, 1);
    this.size = [1, 1];
  }

  // w, h: video pixels; displayScale: how much CSS enlarges the stage, so the
  // ring is rendered at screen resolution rather than camera resolution
  setSize(w, h, displayScale = 1) {
    // Screen resolution, but never more than ~2.2 MP: a 3200x1800 canvas made
    // laptops drop frames (ring lagged behind the hand) and can crash phones.
    const budget = Math.sqrt(2.2e6 / (w * h));
    const ratio = THREE.MathUtils.clamp(displayScale * (window.devicePixelRatio || 1), 1, Math.min(2, budget));
    if (w === this.size[0] && h === this.size[1] && ratio === this.ratio) return;
    this.size = [w, h];
    this.ratio = ratio;
    this.renderer.setPixelRatio(ratio);
    this.renderer.setSize(w, h, false);
    Object.assign(this.camera, { left: 0, right: w, top: 0, bottom: -h });
    this.camera.updateProjectionMatrix();
  }

  setMetal(key) {
    this.metal = key;
    this._applyColour();
  }

  _applyColour() {
    const [r, g, b] = METALS[this.metal].color;
    const t = this.tint || new THREE.Color(1, 1, 1);
    this.metalMat.color.setRGB(r * t.r, g * t.g, b * t.b);
  }

  // Scene light follows the camera picture: dim room -> darker ring, warm
  // light -> warmer reflections. Only a partial tint, or white gold would turn orange.
  matchLighting(image) {
    if (!image) return;
    // gentle: a dim room should not make polished gold look dull
    const target = THREE.MathUtils.clamp(0.85 + image.luma * 0.5, 0.9, 1.35);
    this.renderer.toneMappingExposure += (target - this.renderer.toneMappingExposure) * 0.15;
    const [r, g, b] = image.rgb, mean = (r + g + b) / 3 || 1;
    const k = 0.12;
    this.tint.setRGB(1 + k * (r / mean - 1), 1 + k * (g / mean - 1), 1 + k * (b / mean - 1));
    this._applyColour();
    // A razor-sharp ring on a soft phone picture looks pasted on: blur it
    // to roughly the camera's own sharpness (in video pixels).
    const blur = THREE.MathUtils.clamp(0.25 + 40 / Math.max(image.sharpness, 1), 0.25, 1.0);
    this.canvas.style.filter = `blur(${blur.toFixed(2)}px)`;
  }

  async loadRing(url, meta) {
    const loader = new GLTFLoader().setMeshoptDecoder(MeshoptDecoder);
    const gltf = await loader.loadAsync(url);
    if (this.ring) this.anchor.remove(this.ring);
    this.ring = new THREE.Group();
    this.designInnerMm = meta.ring?.inner_diameter_mm || 16.9;
    this.parts = [];
    gltf.scene.updateMatrixWorld(true);
    gltf.scene.traverse((o) => {
      if (!o.isMesh) return;
      const isStone = /stone/i.test(o.material?.name || "") || /stone/i.test(o.name);
      this.parts.push(this._bake(o, isStone));
    });
    this.parts.forEach((p) => this.ring.add(p.mesh));
    this.anchor.add(this.ring);
    this.setRingSize(this.fingerMm || this.designInnerMm);
  }

  // Flatten a loaded mesh into ring space with float positions, remembering
  // the original positions so resizing can be re-applied from scratch.
  _bake(src, isStone) {
    if (src.isInstancedMesh) {
      const base = [];
      const m = new THREE.Matrix4();
      for (let i = 0; i < src.count; i++) {
        src.getMatrixAt(i, m);
        base.push(src.matrixWorld.clone().multiply(m));
      }
      const mesh = new THREE.InstancedMesh(src.geometry, this.stoneMat, src.count);
      return { mesh, isStone: true, base };
    }
    const geo = new THREE.BufferGeometry();
    for (const name of ["position", "normal"]) {
      const a = src.geometry.getAttribute(name);
      if (!a) continue;
      const f = new Float32Array(a.count * 3);
      for (let i = 0; i < a.count; i++) { f[i * 3] = a.getX(i); f[i * 3 + 1] = a.getY(i); f[i * 3 + 2] = a.getZ(i); }
      geo.setAttribute(name, new THREE.BufferAttribute(f, 3));
    }
    if (src.geometry.index) geo.setIndex(src.geometry.index);
    geo.applyMatrix4(src.matrixWorld);
    if (!geo.getAttribute("normal")) geo.computeVertexNormals();
    const mesh = new THREE.Mesh(geo, isStone ? this.stoneMat : this.metalMat);
    return { mesh, isStone, base: geo.getAttribute("position").array.slice() };
  }

  // Resize the ring to the user's finger (inner diameter in mm). The radial
  // push is applied to every vertex, so the head moves out with the band but
  // keeps its size: a 1 ct stone stays a 1 ct stone on any finger.
  setRingSize(fingerMm) {
    this.fingerMm = fingerMm;
    if (!this.parts) return;
    const delta = ((fingerMm - this.designInnerMm) / 2) / 1000;   // radial offset, metres
    const push = (x, z) => { const r = Math.hypot(x, z) || 1; return [x + (x / r) * delta, z + (z / r) * delta]; };
    for (const p of this.parts) {
      if (p.mesh.isInstancedMesh) {
        const m = new THREE.Matrix4(), v = new THREE.Vector3();
        p.base.forEach((b, i) => {
          m.copy(b);
          v.setFromMatrixPosition(m);
          const [x, z] = push(v.x, v.z);
          m.setPosition(x, v.y, z);
          p.mesh.setMatrixAt(i, m);
        });
        p.mesh.instanceMatrix.needsUpdate = true;
        p.mesh.computeBoundingSphere();
      } else {
        const pos = p.mesh.geometry.getAttribute("position");
        for (let i = 0; i < pos.count; i++) {
          const [x, z] = push(p.base[i * 3], p.base[i * 3 + 2]);
          pos.array[i * 3] = x; pos.array[i * 3 + 2] = z;
        }
        pos.needsUpdate = true;
        p.mesh.geometry.computeBoundingSphere();
      }
    }
  }

  // hand: output of HandTracker.update()
  update(hand) {
    const show = !!(hand?.found && hand.ready && this.ring);
    if (show) this.lastHand = hand;
    this.alpha += ((show ? 1 : 0) - this.alpha) * (show ? 0.25 : 0.12);
    const visible = this.alpha > 0.02 && !!this.lastHand && !!this.ring;
    this.anchor.visible = visible;
    this.occluders.forEach((o) => { o.visible = visible; });
    this.canvas.style.opacity = visible ? this.alpha.toFixed(3) : "0";
    if (!show) return;   // fading out: keep the last pose

    const { position: p, xAxis: x, yAxis: y, zAxis: z } = hand.pose;
    // px per metre of the model: the finger is fingerPx wide and fingerMm thick
    const s = (hand.fingerPx / this.fingerMm) * 1000;
    this.anchor.matrix.set(
      x.x * s, y.x * s, z.x * s, p.x,
      x.y * s, y.y * s, z.y * s, p.y,
      x.z * s, y.z * s, z.z * s, p.z,
      0, 0, 0, 1,
    );
    this.anchor.matrixWorldNeedsUpdate = true;

    const radius = (hand.fingerPx / 2) * OCCLUDER_SHRINK;
    const a = new THREE.Vector3(), b = new THREE.Vector3(), dir = new THREE.Vector3();
    const up = new THREE.Vector3(0, 1, 0), q = new THREE.Quaternion();
    SEGMENTS.forEach(([i, j, k], n) => {
      const pi = hand.points[i], pj = hand.points[j];
      a.set(pi.x, pi.y, pi.z); b.set(pj.x, pj.y, pj.z);
      dir.subVectors(b, a);
      const length = dir.length();
      q.setFromUnitVectors(up, dir.normalize());
      const r = radius * k;
      // CylinderGeometry(1, 1, 1): radius 1, height 1 along Y
      const o = this.occluders[n];
      o.matrix.compose(a.clone().add(b).multiplyScalar(0.5), q, new THREE.Vector3(r, length, r));
      o.matrixWorldNeedsUpdate = true;
    });
  }

  render() {
    this.renderer.render(this.scene, this.camera);
  }

  // Photo of what the user sees: camera frame + ring, mirrored like the screen.
  snapshot(frame, mirrored) {
    const [W, H] = this.size, k = this.ratio || 1;
    const c = document.createElement("canvas");
    c.width = Math.round(W * k); c.height = Math.round(H * k);
    const ctx = c.getContext("2d");
    if (mirrored) { ctx.translate(c.width, 0); ctx.scale(-1, 1); }
    ctx.drawImage(frame, 0, 0, c.width, c.height);
    this.render();   // read the WebGL canvas in the same task, before it is cleared
    ctx.filter = this.canvas.style.filter || "none";
    ctx.globalAlpha = Math.min(1, this.alpha);
    ctx.drawImage(this.renderer.domElement, 0, 0, c.width, c.height);
    return c;
  }
}

// ---------- materials ----------

function stoneMaterial() {
  // A clear stone shows almost no body colour, just the facets reflecting the
  // room: near-mirror facets (flat normals come from the model), neutral white,
  // a faint thin-film sheen for fire. (Stronger sheen read as blue sapphire.)
  return new THREE.MeshPhysicalMaterial({
    color: 0xffffff, metalness: 0.9, roughness: 0.015, envMapIntensity: 2.6,
    iridescence: 0.12, iridescenceIOR: 2.0, iridescenceThicknessRange: [300, 600],
  });
}

// ---------- design thumbnails ----------
// Drawn with the page's own renderer into an offscreen target: one WebGL
// context fewer, which matters on phones (MediaPipe already uses one).

let thumb = null;
export async function renderThumbnail(view, url, metal = "yellow", size = 192) {
  const renderer = view.renderer;
  if (!thumb) {
    const [r, g, b] = METALS[metal].color;
    thumb = {
      target: new THREE.WebGLRenderTarget(size, size, { samples: 4, colorSpace: THREE.SRGBColorSpace }),
      camera: new THREE.PerspectiveCamera(28, 1, 0.001, 10),
      metal: new THREE.MeshPhysicalMaterial({ color: new THREE.Color().setRGB(r, g, b), metalness: 1, roughness: 0.1, envMapIntensity: 1.9, clearcoat: 0.4 }),
      stone: stoneMaterial(),
      pixels: new Uint8Array(size * size * 4),
      canvas: Object.assign(document.createElement("canvas"), { width: size, height: size }),
    };
  }
  const gltf = await new GLTFLoader().setMeshoptDecoder(MeshoptDecoder).loadAsync(url);
  const scene = new THREE.Scene();
  scene.environment = view.scene.environment;
  gltf.scene.traverse((o) => {
    if (o.isMesh) o.material = o.isInstancedMesh || /stone/i.test(o.material?.name || "") ? thumb.stone : thumb.metal;
  });
  scene.add(gltf.scene);
  // catalogue angle: looking down the finger axis, from a little above the head
  const box = new THREE.Box3().setFromObject(gltf.scene);
  const centre = box.getCenter(new THREE.Vector3());
  const radius = box.getBoundingSphere(new THREE.Sphere()).radius;
  const cam = thumb.camera;
  cam.up.set(0, 0, 1);
  cam.position.copy(centre).addScaledVector(new THREE.Vector3(0.35, 1, 0.55).normalize(), radius / Math.sin(THREE.MathUtils.degToRad(14)));
  cam.lookAt(centre);

  const exposure = renderer.toneMappingExposure;
  renderer.toneMappingExposure = 1;
  renderer.setRenderTarget(thumb.target);
  renderer.setClearColor(0x000000, 0);
  renderer.clear();
  renderer.render(scene, cam);
  renderer.readRenderTargetPixels(thumb.target, 0, 0, size, size, thumb.pixels);
  renderer.setRenderTarget(null);
  renderer.toneMappingExposure = exposure;
  gltf.scene.traverse((o) => o.geometry?.dispose());

  // WebGL rows are bottom-up
  const img = new ImageData(size, size), row = size * 4;
  for (let y = 0; y < size; y++) img.data.set(thumb.pixels.subarray((size - 1 - y) * row, (size - y) * row), y * row);
  thumb.canvas.getContext("2d").putImageData(img, 0, 0);
  return thumb.canvas.toDataURL("image/png");
}
