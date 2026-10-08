// The realistic jewellery look (demo, not used by the live pages yet).
//
// Diamonds: light is followed inside each stone. A cut stone is convex, so it
// is exactly the space inside its facet planes: a ray that enters through a
// facet leaves through the plane it reaches first. At every facet it reaches
// it either reflects inside (total internal reflection: a diamond's brilliance)
// or partly leaves, split into red, green and blue at slightly different
// angles (dispersion: a diamond's fire). The planes are taken from the stone's
// own CAD facets, so every cut (round, emerald, marquise, pear ...) behaves as
// it was designed.
//
// Metal: polished gold in a bright studio with soft darker bands, like the
// catalogue renders. Stones: their own darker studio with strong light boxes
// and hot spots (see studioScene).

import * as THREE from "three";
import { toCreasedNormals } from "three/addons/utils/BufferGeometryUtils.js";

export const IOR = { r: 2.407, g: 2.417, b: 2.451 };   // diamond, at red / green / blue light
const MAX_PLANES = 96;     // facets kept per stone, the largest (a round brilliant has 57, plus its girdle)
const MAX_BOUNCES = 6;

// linear reflectance of polished 18K alloys (white gold is rhodium plated), a little
// richer than measured: the bright studio washes colour out, and the catalogue's gold is rich
export const ALLOYS = {
  white_gold: [0.80, 0.80, 0.81],
  yellow_gold: [1.0, 0.70, 0.28],
  rose_gold: [0.98, 0.58, 0.45],
};

// ---------- the studio ----------

// Two studios, as a jewellery photographer lights metal and stones differently:
//   gems:  a bright room with strong soft boxes, a few black cards so the facets
//          read crisp white and black (all-dark made stones look black), and small
//          hot spots for sparkle
//   metal: bright, with long soft boxes and mid-grey bands that give the
//          polished surface its shape (black showed as blotches)
function studioScene(kind) {
  const scene = new THREE.Scene();
  const [floor, top] = kind === "gems" ? [0.3, 0.85] : [0.45, 1.0];
  const room = new THREE.Mesh(new THREE.SphereGeometry(20, 64, 32), new THREE.ShaderMaterial({
    side: THREE.BackSide, depthWrite: false,
    vertexShader: "varying vec3 vDir; void main() { vDir = normalize(position); gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }",
    fragmentShader: `varying vec3 vDir; void main() { float h = vDir.y * 0.5 + 0.5; gl_FragColor = vec4(vec3(mix(${floor.toFixed(3)}, ${top.toFixed(3)}, h)), 1.0); }`,
  }));
  scene.add(room);
  const box = (dir, w, h, power, color = 0xffffff) => {
    const m = new THREE.Mesh(new THREE.PlaneGeometry(w, h), new THREE.MeshBasicMaterial({ color: new THREE.Color(color).multiplyScalar(power), side: THREE.DoubleSide }));
    m.position.copy(new THREE.Vector3(...dir).normalize().multiplyScalar(14));
    m.lookAt(0, 0, 0);
    scene.add(m);
  };
  if (kind === "gems") {
    box([0, 1, 0.1], 10, 10, 4);            // overhead
    box([-1, 0.3, 0.45], 3.5, 12, 4.5);     // left strip
    box([1, 0.3, 0.45], 3.5, 12, 4.5);      // right strip
    box([0, 0.35, 1], 7, 4, 3);             // front: the table's reflection
    box([0.3, 0.5, -1], 6, 4, 3.5);         // back
    box([-0.75, 0.05, -0.65], 5, 10, 0, 0x000000);   // black cards
    box([0.85, 0.0, -0.5], 4, 10, 0, 0x000000);
    box([0, -0.3, 1], 10, 3, 0, 0x000000);
    box([0.6, 0.9, 0.6], 2.5, 2.5, 0, 0x000000);
    const rnd = mulberry(7);                // small hot spots all around: sparkle
    for (let i = 0; i < 36; i++) {
      const v = new THREE.Vector3(rnd() * 2 - 1, rnd() * 1.5 - 0.3, rnd() * 2 - 1);
      box(v.toArray(), 0.4, 0.4, 25);
    }
  } else {
    box([0, 1, 0.15], 14, 14, 3.4);         // overhead
    box([-1, 0.35, 0.5], 4, 14, 3.8);       // left strip
    box([1, 0.35, 0.5], 4, 14, 3.8);        // right strip
    box([0, 0.15, 1], 12, 3, 2.2);          // low front fill
    box([0.2, 0.6, -1], 8, 5, 2.8);         // back kicker
    box([-0.7, -0.1, -0.7], 6, 10, 0.3);    // mid-grey bands
    box([0.8, -0.05, -0.5], 5, 10, 0.3);
  }
  return scene;
}

function mulberry(seed) {
  return () => { seed |= 0; seed = (seed + 0x6d2b79f5) | 0; let t = Math.imul(seed ^ (seed >>> 15), 1 | seed); t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}

// -> { gems: sharp cube map the stones see, metal: PMREM map for the metal }
export function buildStudio(renderer, size = 512) {
  const target = new THREE.WebGLCubeRenderTarget(size, { type: THREE.HalfFloatType, generateMipmaps: false });
  const cam = new THREE.CubeCamera(0.1, 100, target);
  cam.update(renderer, studioScene("gems"));
  const pmrem = new THREE.PMREMGenerator(renderer);
  const metal = pmrem.fromScene(studioScene("metal"), 0).texture;
  pmrem.dispose();
  return { gems: target.texture, metal };
}

// ---------- the stones ----------

// The stone's facet planes (outward normal, offset) in its own coordinates.
export function facetPlanes(geometry) {
  const pos = geometry.getAttribute("position"), idx = geometry.index;
  const n = idx ? idx.count : pos.count;
  geometry.computeBoundingBox();
  const box = geometry.boundingBox, centre = box.getCenter(new THREE.Vector3()), size = box.getSize(new THREE.Vector3()).length();
  const a = new THREE.Vector3(), b = new THREE.Vector3(), c = new THREE.Vector3(), nn = new THREE.Vector3();
  const planes = [];
  const at = (i, v) => v.fromBufferAttribute(pos, idx ? idx.getX(i) : i);
  for (let i = 0; i < n; i += 3) {
    at(i, a); at(i + 1, b); at(i + 2, c);
    nn.subVectors(b, a).cross(c.clone().sub(a));
    const area = nn.length() / 2;
    if (area < 1e-12) continue;
    nn.normalize();
    let d = nn.dot(a);
    if (d - nn.dot(centre) < 0) { nn.negate(); d = -d; }          // outward
    // the model's positions are compressed (16 bit), which tilts the triangles of one facet
    // by up to a few degrees: those are one facet again
    const p = planes.find((q) => q.n.dot(nn) > 0.9986 && Math.abs(q.d - d) < 0.012 * size);
    if (p) { p.n.multiplyScalar(p.area).addScaledVector(nn, area).normalize(); p.d = (p.d * p.area + d * area) / (p.area + area); p.area += area; }
    else planes.push({ n: nn.clone(), d, area });
  }
  // only the stone's outer faces: a plane with corners of the stone beyond it is an inner or
  // concave face, which would trap light (the stone is then traced as its convex outline)
  const v = new THREE.Vector3();
  let cutting = 0;
  const outer = planes.filter((p) => {
    for (let i = 0; i < pos.count; i++) if (p.n.dot(v.fromBufferAttribute(pos, i)) - p.d > 0.006 * size) { cutting++; return false; }
    return true;
  });
  outer.sort((p, q) => q.area - p.area);
  const kept = outer.slice(0, MAX_PLANES);
  return { planes: kept.map((p) => new THREE.Vector4(p.n.x, p.n.y, p.n.z, p.d)), count: kept.length, found: outer.length, cutting, vertices: pos.count };
}

const GEM_VERT = /* glsl */ `
varying vec3 vLocalPos;
varying vec3 vLocalCam;
varying mat3 vToWorld;
void main() {
  mat4 m = modelMatrix;
  #ifdef USE_INSTANCING
    m = m * instanceMatrix;
  #endif
  vLocalPos = position;
  vLocalCam = (inverse(m) * vec4(cameraPosition, 1.0)).xyz;
  vToWorld = mat3(m);
  gl_Position = projectionMatrix * viewMatrix * m * vec4(position, 1.0);
}`;

const GEM_FRAG = /* glsl */ `
uniform samplerCube gemEnv;
uniform vec4 planes[${MAX_PLANES}];
uniform int planeCount;
uniform int bounces;
uniform vec3 ior;          // red, green, blue
uniform float envPower;
varying vec3 vLocalPos;
varying vec3 vLocalCam;
varying mat3 vToWorld;

vec3 envAt(vec3 localDir) { return texture(gemEnv, normalize(vToWorld * localDir)).rgb * envPower; }

// unpolarised Fresnel reflectance; eta = n(outside the surface) / n(this side)
float fresnel(float cosI, float eta) {
  float c = abs(cosI);
  float g2 = eta * eta - 1.0 + c * c;
  if (g2 < 0.0) return 1.0;
  float g = sqrt(g2);
  float A = (g - c) / (g + c);
  float B = (c * (g + c) - 1.0) / (c * (g - c) + 1.0);
  return 0.5 * A * A * (1.0 + B * B);
}

void main() {
  vec3 rd = normalize(vLocalPos - vLocalCam);
  vec3 n = normalize(cross(dFdx(vLocalPos), dFdy(vLocalPos)));     // this facet
  if (dot(n, rd) > 0.0) n = -n;
  float F = fresnel(dot(-rd, n), ior.g);
  vec3 col = F * envAt(reflect(rd, n));                             // the facet's own reflection
  vec3 dir = refract(rd, n, 1.0 / ior.g);
  vec3 p = vLocalPos;
  float thr = 1.0 - F;
  for (int b = 0; b < ${MAX_BOUNCES}; b++) {
    if (b >= bounces) break;
    float tMin = 1e20;
    vec3 nOut = -dir;
    for (int i = 0; i < ${MAX_PLANES}; i++) {
      if (i >= planeCount) break;
      float dn = dot(planes[i].xyz, dir);
      if (dn > 1e-5) {
        float t = (planes[i].w - dot(planes[i].xyz, p)) / dn;
        if (t > 1e-6 && t < tMin) { tMin = t; nOut = planes[i].xyz; }
      }
    }
    if (tMin > 1e19) break;
    p += dir * tMin;
    float Fi = fresnel(dot(dir, nOut), 1.0 / ior.g);
    if (Fi < 1.0) {   // part of the light leaves here, split into colours
      vec3 oR = refract(dir, -nOut, ior.r), oG = refract(dir, -nOut, ior.g), oB = refract(dir, -nOut, ior.b);
      vec3 out_ = vec3(dot(oR, oR) > 0.0 ? envAt(oR).r : 0.0, envAt(oG).g, dot(oB, oB) > 0.0 ? envAt(oB).b : 0.0);
      col += thr * (1.0 - Fi) * out_;
      thr *= Fi;
    }
    dir = reflect(dir, nOut);
  }
  col += thr * envAt(dir) * 0.5;   // what is still inside leaves somewhere else
  gl_FragColor = vec4(col, 1.0);
  #include <tonemapping_fragment>
  #include <colorspace_fragment>
}`;

export function gemMaterial(geometry, gemEnv, { bounces = 5, envPower = 1.0 } = {}) {
  const facetInfo = facetPlanes(geometry);
  const { planes, count, found } = facetInfo;
  while (planes.length < MAX_PLANES) planes.push(new THREE.Vector4(0, 0, 0, 0));
  const mat = new THREE.ShaderMaterial({
    uniforms: {
      gemEnv: { value: gemEnv }, planes: { value: planes }, planeCount: { value: count },
      bounces: { value: bounces }, ior: { value: new THREE.Vector3(IOR.r, IOR.g, IOR.b) }, envPower: { value: envPower },
    },
    vertexShader: GEM_VERT, fragmentShader: GEM_FRAG,
  });
  mat.userData.facets = { kept: count, found, cutting: facetInfo.cutting, vertices: facetInfo.vertices };
  return mat;
}

// ---------- the metal ----------

export function metalMaterial(alloy, env) {
  return new THREE.MeshPhysicalMaterial({
    color: new THREE.Color().setRGB(...ALLOYS[alloy]), metalness: 1, roughness: 0.055,
    envMap: env, envMapIntensity: 0.9,
    // some CAD triangles face inwards: drawn from one side they vanish and the band
    // shows dark patches (today's look has them too)
    side: THREE.DoubleSide,
  });
}

// ---------- a loaded model in the new look ----------

export const isStone = (o) => o.isInstancedMesh || /stone/i.test(o.material?.name || "") || /stone/i.test(o.name || "");

// Stones get a gem material per stone shape (instanced stones share theirs);
// the small stones of a pave bounce light fewer times than a centre stone.
export function applyLook(root, studio, alloy) {
  const metal = metalMaterial(alloy, studio.metal);
  const facets = [];
  let biggest = 0;
  root.traverse((o) => { if (o.isMesh && isStone(o)) { o.geometry.computeBoundingSphere(); biggest = Math.max(biggest, o.geometry.boundingSphere.radius * o.matrixWorld.getMaxScaleOnAxis()); } });
  root.traverse((o) => {
    if (!o.isMesh) return;
    if (isStone(o)) {
      const r = o.geometry.boundingSphere.radius * o.matrixWorld.getMaxScaleOnAxis();
      o.material = gemMaterial(o.geometry, studio.gems, { bounces: r > 0.5 * biggest ? 6 : 3, envPower: 1.2 });
      facets.push(o.material.userData.facets);
    } else {
      // the file's normals are compressed to a few bits: polished curves reflected in
      // blocks. Smooth normals from the positions, sharp edges (over 35 degrees) kept.
      if (!o.userData.smoothed) {
        o.geometry = toCreasedNormals(o.geometry, THREE.MathUtils.degToRad(35));
        o.userData.smoothed = true;
      }
      o.material = metal;
    }
  });
  return { metal, facets };
}

export function setAlloy(root, alloy) {
  root.traverse((o) => { if (o.isMesh && !isStone(o)) o.material.color.setRGB(...ALLOYS[alloy]); });
}
