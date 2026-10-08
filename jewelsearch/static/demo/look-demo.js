// 3D look demo: each design as its catalogue photo, today's 3D and the new 3D
// look (jewel-look.js), turned together. A demo on 6 designs: the live pages
// are not changed.

import * as THREE from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { MeshoptDecoder } from "three/addons/libs/meshopt_decoder.module.js";
import { RoomEnvironment } from "three/addons/environments/RoomEnvironment.js";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { EffectComposer } from "three/addons/postprocessing/EffectComposer.js";
import { RenderPass } from "three/addons/postprocessing/RenderPass.js";
import { UnrealBloomPass } from "three/addons/postprocessing/UnrealBloomPass.js";
import { OutputPass } from "three/addons/postprocessing/OutputPass.js";
import { applyLook, buildStudio, isStone, setAlloy } from "./jewel-look.js";

const $ = (s) => document.querySelector(s);
const params = new URLSearchParams(location.search);
const FOV = 25;
// today's look, exactly as the live try-on draws it (tryon/ring-view.js)
const TODAY_METAL = { white_gold: [0.86, 0.855, 0.84], yellow_gold: [1.0, 0.766, 0.336], rose_gold: [0.955, 0.637, 0.538] };
const LABEL = { white_gold: "White gold", yellow_gold: "Yellow gold", rose_gold: "Rose gold" };

const loader = new GLTFLoader().setMeshoptDecoder(MeshoptDecoder);
const designs = await (await fetch("/demo-app/designs.json", { cache: "no-store" })).json();
const state = { i: 0, alloy: ["white_gold", "yellow_gold", "rose_gold"].includes(params.get("metal")) ? params.get("metal") : "white_gold", turning: false, show: "new" };
const cache = new Map();   // uid -> { info, gltf }

async function load(d) {
  if (cache.has(d.uid)) return cache.get(d.uid);
  const info = await (await fetch(`/api/tryon/design/${d.uid}`)).json();
  const url = `/api/tryon/models/${encodeURIComponent(d.slug)}.glb${info.model?.version ? `?v=${info.model.version}` : ""}`;
  const gltf = await loader.loadAsync(url);
  const out = { info, gltf };
  cache.set(d.uid, out);
  return out;
}

// the model as shown: earrings as the pair the catalogue shows (a mirrored twin)
function arrange(scene, kind) {
  const root = new THREE.Group();
  const g = scene.clone(true);
  root.add(g);
  if (kind === "earring") {
    const w = new THREE.Box3().setFromObject(g).getSize(new THREE.Vector3()).x;
    const twin = g.clone(true);
    twin.scale.x = -1;
    g.position.x = w * 0.65; twin.position.x = -w * 0.65;
    root.add(twin);
  }
  root.updateMatrixWorld(true);
  return root;
}

// a soft shadow on the floor under the piece, like the catalogue's. The floor is
// level in the photo: its "up" is the camera's up tilted back by the camera's
// height above the piece (about 25 degrees for the catalogue's rings, 0 for the rest).
function contactShadow(root, pose, kind) {
  const c = document.createElement("canvas");
  c.width = c.height = 128;
  const x = c.getContext("2d"), grad = x.createRadialGradient(64, 64, 2, 64, 64, 64);
  grad.addColorStop(0, "rgba(40,40,40,0.38)"); grad.addColorStop(0.55, "rgba(40,40,40,0.12)"); grad.addColorStop(1, "rgba(40,40,40,0)");
  x.fillStyle = grad; x.fillRect(0, 0, 128, 128);
  const tilt = THREE.MathUtils.degToRad(kind === "ring" ? 25 : 0);
  const dir = new THREE.Vector3(...pose.dir).normalize();
  const up = new THREE.Vector3(...pose.up).normalize().multiplyScalar(Math.cos(tilt)).addScaledVector(dir, Math.sin(tilt)).normalize();
  const box = new THREE.Box3().setFromObject(root), mid = box.getCenter(new THREE.Vector3()), rad = box.getSize(new THREE.Vector3()).length() / 2;
  let low = Infinity;   // the piece's lowest point (its box's corners float below a tilted piece)
  const v = new THREE.Vector3(), m4 = new THREE.Matrix4();
  root.traverse((o) => {
    if (!o.isMesh) return;
    const pos = o.geometry.getAttribute("position"), step = Math.max(1, Math.floor(pos.count / 4000));
    const n = o.isInstancedMesh ? o.count : 1;
    for (let k = 0; k < n; k++) {
      if (o.isInstancedMesh) o.getMatrixAt(k, m4).premultiply(o.matrixWorld); else m4.copy(o.matrixWorld);
      for (let i = 0; i < pos.count; i += step) low = Math.min(low, v.fromBufferAttribute(pos, i).applyMatrix4(m4).sub(mid).dot(up));
    }
  });
  const m = new THREE.Mesh(new THREE.PlaneGeometry(1, 1), new THREE.MeshBasicMaterial({ map: new THREE.CanvasTexture(c), transparent: true, depthWrite: false, toneMapped: false }));
  m.quaternion.setFromUnitVectors(new THREE.Vector3(0, 0, 1), up);
  m.position.copy(mid).addScaledVector(up, low - rad * 0.01);
  m.scale.set(rad * 1.8, rad * 1.2, 1);
  return m;
}

class Viewer {
  constructor(canvas, look) {
    this.look = look;
    this.canvas = canvas;
    // transparent: the page's white shows through untouched (tone mapping would grey a white background)
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: look === "today", alpha: true, powerPreference: "high-performance" });
    this.renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.toneMapping = THREE.NeutralToneMapping;
    this.renderer.setClearColor(0xffffff, 0);
    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(FOV, 1, 0.001, 100);
    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.enableDamping = true;
    this.controls.enablePan = false;
    if (look === "today") {
      const pm = new THREE.PMREMGenerator(this.renderer);
      this.scene.environment = pm.fromScene(new RoomEnvironment(), 0.03).texture;
      pm.dispose();
    } else {
      this.studio = buildStudio(this.renderer);
      const rt = new THREE.WebGLRenderTarget(1, 1, { type: THREE.HalfFloatType, samples: 4 });
      this.composer = new EffectComposer(this.renderer, rt);
      this.composer.addPass(new RenderPass(this.scene, this.camera));
      // only diamond glints glow: polished metal peaks below 3.6 (soft boxes up to 3.8 x 0.8 reflectance)
      this.bloom = new UnrealBloomPass(new THREE.Vector2(1, 1), 0.3, 0.2, 3.6);
      this.composer.addPass(this.bloom);
      this.composer.addPass(new OutputPass());
    }
    this.root = null;
    this.frameMs = 0;
  }

  show(gltf, kind, alloy, pose) {
    if (this.root) this.scene.remove(this.root);
    this.root = arrange(gltf.scene, kind);
    if (this.look === "today") {
      const metal = new THREE.MeshPhysicalMaterial({ color: new THREE.Color().setRGB(...TODAY_METAL[alloy]), metalness: 1, roughness: 0.1, envMapIntensity: 1.9, clearcoat: 0.4, clearcoatRoughness: 0.05 });
      const stone = new THREE.MeshPhysicalMaterial({ color: 0xffffff, metalness: 0.9, roughness: 0.015, envMapIntensity: 2.6, iridescence: 0.12, iridescenceIOR: 2.0, iridescenceThicknessRange: [300, 600] });
      this.root.traverse((o) => { if (o.isMesh) o.material = isStone(o) ? stone : metal; });
      this.todayMetal = metal;
    } else {
      this.facets = applyLook(this.root, this.studio, alloy).facets;
    }
    this.scene.add(this.root);
    this.shadow && this.scene.remove(this.shadow);
    this.shadow = contactShadow(this.root, pose, kind);
    this.scene.add(this.shadow);
  }

  setAlloy(alloy) {
    if (!this.root) return;
    if (this.look === "today") this.todayMetal.color.setRGB(...TODAY_METAL[alloy]);
    else setAlloy(this.root, alloy);
  }

  // look from a direction (the catalogue photo's angle), the whole piece in view
  frame(dir, up) {
    const box = new THREE.Box3().setFromObject(this.root), c = box.getCenter(new THREE.Vector3());
    const rad = box.getSize(new THREE.Vector3()).length() / 2;
    this.camera.near = rad / 50; this.camera.far = rad * 50; this.camera.updateProjectionMatrix();
    this.camera.position.copy(c).addScaledVector(new THREE.Vector3(...dir).normalize(), rad / Math.tan(THREE.MathUtils.degToRad(FOV / 2)) * 1.05);
    this.camera.up.set(...up);
    this.controls.target.copy(c);
    this.camera.lookAt(c);
    this.camera.updateMatrixWorld();
    // move in until the piece fills about 80% of the view, as in the catalogue photos
    let ext = 0;
    for (let i = 0; i < 8; i++) {
      const p = new THREE.Vector3(i & 1 ? box.max.x : box.min.x, i & 2 ? box.max.y : box.min.y, i & 4 ? box.max.z : box.min.z).project(this.camera);
      ext = Math.max(ext, Math.abs(p.x), Math.abs(p.y));
    }
    if (ext > 0) this.camera.position.sub(c).multiplyScalar(Math.max(0.35, ext / 0.8)).add(c);
    this.controls.update();
  }

  resize() {
    const w = this.canvas.clientWidth, h = this.canvas.clientHeight;
    if (!w || !h) return;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h; this.camera.updateProjectionMatrix();
    if (this.composer) {
      const pr = this.renderer.getPixelRatio();
      this.composer.setSize(w, h);
      this.bloom.resolution.set(w * pr, h * pr);
    }
  }

  render() {
    const t = performance.now();
    if (this.composer) this.composer.render(); else this.renderer.render(this.scene, this.camera);
    this.frameMs = this.frameMs * 0.9 + (performance.now() - t) * 0.1;
  }
}

// ---------- the page ----------

const views = { today: new Viewer($("#today"), "today"), new: new Viewer($("#new"), "new") };
let syncing = false;
for (const [k, v] of Object.entries(views)) {
  v.controls.addEventListener("change", () => {   // turn both views together
    if (syncing) return;
    syncing = true;
    const o = k === "new" ? views.today : views.new;
    o.camera.position.copy(v.camera.position); o.camera.up.copy(v.camera.up); o.controls.target.copy(v.controls.target);
    o.camera.lookAt(o.controls.target); o.controls.update();
    syncing = false;
  });
  v.controls.addEventListener("start", () => setTurning(false));
}

function setTurning(on) {
  state.turning = on;
  $("#turn").setAttribute("aria-pressed", String(on));
}

function phone() { return matchMedia("(max-width: 760px)").matches; }

async function open(i) {
  state.i = i;
  const d = designs[i];
  document.querySelectorAll("#designs button").forEach((b, j) => b.setAttribute("aria-pressed", String(j === i)));
  $("#title").textContent = `${d.design_id} · ${d.label}`;
  $("#busy").hidden = false;
  try {
    const { info, gltf } = await load(d);
    const metals = info.card.metals;
    if (!metals.includes(state.alloy)) state.alloy = info.card.metal_shown;
    renderMetals(metals);
    $("#photo").src = info.card.thumbs_by_metal[state.alloy];
    for (const v of Object.values(views)) { v.show(gltf, d.kind, state.alloy, d.pose); v.resize(); v.frame(d.pose.dir, d.pose.up); }
    const f = views.new.facets || [];
    $("#facets").textContent = f.length ? `Light traced through every facet of ${f.length === 1 ? "the stone" : `${f.length} stone shapes`}.` : "";
  } catch (e) {
    console.error(e);
    $("#title").textContent = `${d.design_id}: couldn't load (${e.message || e})`;
  } finally { $("#busy").hidden = true; }
}

function renderMetals(metals) {
  $("#metals").innerHTML = metals.map((m) => `<button type="button" data-m="${m}" aria-pressed="${m === state.alloy}"><span class="sw ${m}"></span>${LABEL[m]}</button>`).join("");
}

$("#metals").addEventListener("click", (e) => {
  const b = e.target.closest("[data-m]");
  if (!b) return;
  state.alloy = b.dataset.m;
  renderMetals(cache.get(designs[state.i].uid).info.card.metals);
  $("#photo").src = cache.get(designs[state.i].uid).info.card.thumbs_by_metal[state.alloy];
  for (const v of Object.values(views)) v.setAlloy(state.alloy);
});
$("#turn").onclick = () => setTurning(!state.turning);
$("#reset").onclick = () => { const d = designs[state.i]; for (const v of Object.values(views)) v.frame(d.pose.dir, d.pose.up); };
$("#show").addEventListener("click", (e) => {   // phones: one view at a time
  const b = e.target.closest("[data-show]");
  if (!b) return;
  state.show = b.dataset.show;
  document.body.dataset.show = state.show;
  document.querySelectorAll("#show button").forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
  for (const v of Object.values(views)) v.resize();
});
$("#designs").innerHTML = designs.map((d, i) => `<button type="button" data-i="${i}">${d.label}</button>`).join("");
$("#designs").addEventListener("click", (e) => { const b = e.target.closest("[data-i]"); if (b) open(+b.dataset.i); });
addEventListener("resize", () => Object.values(views).forEach((v) => v.resize()));

let last = performance.now();
function loop(t) {
  const dt = Math.min(0.05, (t - last) / 1000);
  last = t;
  if (state.turning) {   // turn about the view's up axis, both views
    const v = views.new;
    v.camera.position.sub(v.controls.target).applyAxisAngle(v.camera.up, dt * 0.45).add(v.controls.target);
    v.camera.lookAt(v.controls.target);
    v.controls.update();
    v.controls.dispatchEvent({ type: "change" });
  }
  for (const [k, v] of Object.entries(views)) {
    v.controls.update();
    if (!phone() || state.show === k) v.render();
  }
  requestAnimationFrame(loop);
}
document.body.dataset.show = state.show;
if (params.get("fit") === "1") {   // once per design list: the catalogue photo's angle (fit-pose.js)
  const { fitPose } = await import("./fit-pose.js");
  window.fitResults = [];
  const only = params.get("only") === null ? null : +params.get("only");
  for (let i = 0; i < designs.length; i++) {
    if (only !== null && i !== only) continue;
    await open(i);
    const { info } = cache.get(designs[i].uid);
    const pose = await fitPose(views.new.renderer, views.new.root, info.card.thumbs_by_metal[info.card.metal_shown], designs[i].kind !== "ring");
    window.fitResults.push({ uid: designs[i].uid, ...pose });
  }
  window.fitDone = true;
}
await open(Math.min(designs.length - 1, Math.max(0, +params.get("d") || 0)));
setTurning(params.get("turn") === "1");
requestAnimationFrame(loop);
window.demoReady = true;
window.demoViews = views;   // for checks from the browser console
