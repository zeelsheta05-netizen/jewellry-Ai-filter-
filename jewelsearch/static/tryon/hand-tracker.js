// Hand tracking for ring try-on: MediaPipe Hand Landmarker (21 points),
// One-Euro smoothing, the ring pose on the ring finger, and the scan
// verification checks that decide whether the hand is good enough to use.
//
// Coordinates handed to the renderer are video pixels with y up and z
// towards the camera: P = (x * W, -y * H, -z * W * Z_SCALE). MediaPipe's z uses the
// same scale as x (damped by Z_SCALE below), so distances are roughly isotropic.

import { FilesetResolver, HandLandmarker } from "/vendor/mediapipe/vision_bundle.mjs";
import { PointsFilter } from "./one-euro.js";

const WASM = "/vendor/mediapipe/wasm";
const MODEL = "/vendor/models/hand_landmarker.task";

// Landmark ids
const WRIST = 0, INDEX_MCP = 5, MIDDLE_MCP = 9, RING_MCP = 13, RING_PIP = 14, RING_TIP = 16, PINKY_MCP = 17;
// Calibrated on a real user frame (2026-09-29): a worn ring sat at 0.56 of
// the knuckle -> middle-joint bone, and the finger was 0.97x the knuckle
// spacing wide at that point.
// Where a ring sits on the proximal phalanx, from the knuckle (0) to the middle joint (1)
const RING_POSITION = 0.57;
// Visible finger width vs. knuckle spacing (MCP centre to MCP centre)
const FINGER_PER_SPACING = 0.8;   // the finger shaft under a ring is narrower than the knuckles are apart
// MediaPipe's z overstates depth for a hand close to the camera: a ring that
// looked flat on the finger came out tilted ~13 degrees. Damp it.
const Z_SCALE = 0.6;

// Checks in the order the user should fix them. `hard` ones gate the ring.
export const CHECKS = [
  { id: "hand", hard: true, tip: "Show one hand to the camera" },
  { id: "frame", hard: true, tip: "Keep all your fingers inside the frame" },
  { id: "distance", hard: true, tip: "Move your hand closer" },
  { id: "back", hard: true, tip: "Turn the back of your hand to the camera" },
  { id: "straight", hard: true, tip: "Straighten your ring finger" },
  { id: "light", hard: false, tip: "More light please: move somewhere brighter" },
  { id: "sharp", hard: false, tip: "Hold still, the picture is blurry" },
  { id: "steady", hard: false, tip: "Hold your hand steady" },
];

const sub = (a, b) => ({ x: a.x - b.x, y: a.y - b.y, z: a.z - b.z });
const add = (a, b) => ({ x: a.x + b.x, y: a.y + b.y, z: a.z + b.z });
const mul = (a, s) => ({ x: a.x * s, y: a.y * s, z: a.z * s });
const dot = (a, b) => a.x * b.x + a.y * b.y + a.z * b.z;
const cross = (a, b) => ({ x: a.y * b.z - a.z * b.y, y: a.z * b.x - a.x * b.z, z: a.x * b.y - a.y * b.x });
const len = (a) => Math.hypot(a.x, a.y, a.z);
const norm = (a) => mul(a, 1 / (len(a) || 1));

// mode: "VIDEO" for the live camera, "IMAGE" for an uploaded photo
export async function createHandLandmarker(mode = "VIDEO") {
  const files = await FilesetResolver.forVisionTasks(WASM);
  const opts = (delegate) => ({
    baseOptions: { modelAssetPath: MODEL, delegate },
    runningMode: mode,
    numHands: 1,
    minHandDetectionConfidence: 0.6,
    minHandPresenceConfidence: 0.6,
    minTrackingConfidence: 0.6,
  });
  try {
    return await HandLandmarker.createFromOptions(files, opts("GPU"));
  } catch (e) {
    // iOS Safari's GPU delegate is unreliable; the CPU path always works
    console.warn("GPU delegate failed, using CPU", e);
    return await HandLandmarker.createFromOptions(files, opts("CPU"));
  }
}

// Brightness and sharpness (variance of the Laplacian) of a small crop.
class ImageProbe {
  constructor(size = 96) {
    this.size = size;
    this.canvas = document.createElement("canvas");
    this.canvas.width = this.canvas.height = size;
    this.ctx = this.canvas.getContext("2d", { willReadFrequently: true });
  }
  measure(video, cx, cy, half) {
    const s = this.size, W = video.videoWidth || video.width, H = video.videoHeight || video.height;
    const x0 = Math.max(0, cx - half), y0 = Math.max(0, cy - half);
    const w = Math.min(W - x0, half * 2), h = Math.min(H - y0, half * 2);
    if (w < 8 || h < 8) return null;
    this.ctx.drawImage(video, x0, y0, w, h, 0, 0, s, s);
    const px = this.ctx.getImageData(0, 0, s, s).data;
    const g = new Float32Array(s * s);
    let r = 0, gg = 0, b = 0;
    for (let i = 0; i < s * s; i++) {
      r += px[i * 4]; gg += px[i * 4 + 1]; b += px[i * 4 + 2];
      g[i] = 0.299 * px[i * 4] + 0.587 * px[i * 4 + 1] + 0.114 * px[i * 4 + 2];
    }
    let sum = 0, sum2 = 0, n = 0;
    for (let y = 1; y < s - 1; y++) {
      for (let x = 1; x < s - 1; x++) {
        const i = y * s + x;
        const lap = g[i - 1] + g[i + 1] + g[i - s] + g[i + s] - 4 * g[i];
        sum += lap; sum2 += lap * lap; n++;
      }
    }
    const n2 = s * s;
    return {
      luma: (0.299 * r + 0.587 * gg + 0.114 * b) / n2 / 255,
      rgb: [r / n2 / 255, gg / n2 / 255, b / n2 / 255],
      sharpness: sum2 / n - (sum / n) ** 2,
    };
  }
}

// Landmarks (normalised, as MediaPipe returns them) -> the renderer's pixel space.
export const toPixels = (raw, W, H) => raw.map((p) => ({ x: p.x * W, y: -p.y * H, z: -p.z * W * Z_SCALE }));

// Ring pose on the ring finger, and the finger's width. pts: toPixels() output.
// Shared by the live camera, photo checks and the saved-photo preview.
export function handFrame(pts, isRight) {
  const spacing = (len(sub(pts[RING_MCP], pts[MIDDLE_MCP])) + len(sub(pts[RING_MCP], pts[PINKY_MCP]))) / 2;
  const fingerPx = spacing * FINGER_PER_SPACING;

  // Back-of-hand normal. For a right hand, (index - wrist) x (little - wrist)
  // points out of the palm, so flip it; a left hand is the mirror image.
  let dorsal = norm(cross(sub(pts[INDEX_MCP], pts[WRIST]), sub(pts[PINKY_MCP], pts[WRIST])));
  if (isRight) dorsal = mul(dorsal, -1);

  const yAxis = norm(sub(pts[RING_PIP], pts[RING_MCP]));
  const zAxis = norm(sub(dorsal, mul(yAxis, dot(dorsal, yAxis))));
  const xAxis = cross(yAxis, zAxis);
  const position = add(pts[RING_MCP], mul(sub(pts[RING_PIP], pts[RING_MCP]), RING_POSITION));
  return { points: pts, fingerPx, dorsal, pose: { position, xAxis, yAxis, zAxis } };
}

// Bracelet pose: around the wrist, axis along the forearm (continuing the
// middle-knuckle -> wrist line), sitting just past the wrist crease.
// Wrist width ~ knuckle width (index knuckle to little knuckle).
export function wristFrame(pts, isRight) {
  const { dorsal } = handFrame(pts, isRight);
  const yAxis = norm(sub(pts[WRIST], pts[MIDDLE_MCP]));
  const zAxis = norm(sub(dorsal, mul(yAxis, dot(dorsal, yAxis))));
  const xAxis = cross(yAxis, zAxis);
  const palmLen = len(sub(pts[MIDDLE_MCP], pts[WRIST]));
  const position = add(pts[WRIST], mul(yAxis, palmLen * 0.06));
  return { points: pts, wristPx: len(sub(pts[INDEX_MCP], pts[PINKY_MCP])), pose: { position, xAxis, yAxis, zAxis } };
}

// The hand verification checks for one frame or photo (all but "steady").
export function handChecks(raw, pts, isRight, W, H, img) {
  const { dorsal, pose: { yAxis } } = handFrame(pts, isRight);
  // A ring only needs the knuckles and fingers in view; the wrist and thumb
  // base are often cut off when the hand is close to a laptop camera.
  const m = 0.02;
  const inFrame = raw.slice(INDEX_MCP).every((p) => p.x > m && p.x < 1 - m && p.y > m && p.y < 1 - m);
  const palm = Math.hypot(raw[MIDDLE_MCP].x * W - raw[WRIST].x * W, raw[MIDDLE_MCP].y * H - raw[WRIST].y * H) / Math.min(W, H);
  const bend = dot(yAxis, norm(sub(pts[RING_TIP], pts[RING_PIP])));
  return {
    hand: true,
    frame: inFrame,
    distance: palm > 0.16 && palm < 0.6,
    back: dorsal.z > 0.3,   // +1 = back of the hand straight at the camera
    straight: bend > Math.cos((40 * Math.PI) / 180),
    light: !img || (img.luma > 0.16 && img.luma < 0.92),
    sharp: !img || img.sharpness > 35,
  };
}

export { ImageProbe };

export class HandTracker {
  constructor(landmarker) {
    this.landmarker = landmarker;
    // Responsive when moving (ring must stay on the finger), still heavy at rest
    this.filter = new PointsFilter(21, { minCutoff: 1.5, beta: 6 });
    this.probe = new ImageProbe();
    this.frame = 0;
    this.image = null;        // last brightness / sharpness measurement
    this.lastRing = null;     // for the steadiness check
    this.goodStreak = 0;
    this.badStreak = 0;
    this.ready = false;       // hysteresis over the hard checks, so the ring doesn't flicker
  }

  // source: the <video> or a canvas holding the current frame
  update(video, timeMs) {
    const W = video.videoWidth || video.width, H = video.videoHeight || video.height;
    const res = this.landmarker.detectForVideo(video, timeMs);
    const out = { width: W, height: H, checks: {}, found: false };
    const hand = res.landmarks?.[0];
    if (!hand) {
      this.filter.reset();
      this.lastRing = null;
      return this._finish(out, { hand: false });
    }
    const raw = hand;
    const pts = toPixels(this.filter.apply(raw, timeMs), W, H);
    // Verified on MediaPipe's own right_hands.jpg / left_hands.jpg test photos:
    // on our (never mirrored) frames the label is the physical hand.
    const label = res.handedness?.[0]?.[0]?.categoryName;
    const isRight = label === "Right";

    const { fingerPx, pose: { position, xAxis, yAxis, zAxis } } = handFrame(pts, isRight);

    let steady = true;
    const palm = Math.hypot(raw[MIDDLE_MCP].x * W - raw[WRIST].x * W, raw[MIDDLE_MCP].y * H - raw[WRIST].y * H) / Math.min(W, H);
    if (this.lastRing) {
      const dt = Math.max(1e-3, (timeMs - this.lastRing.t) / 1000);
      steady = len(sub(position, this.lastRing.p)) / dt / (palm * Math.min(W, H)) < 1.5;
    }
    this.lastRing = { p: position, t: timeMs };

    // brightness/sharpness twice a second is enough; each read stalls the GPU
    if (this.frame++ % 15 === 0) {
      this.image = this.probe.measure(video, raw[RING_MCP].x * W, raw[RING_MCP].y * H, Math.max(24, fingerPx * 1.6));
    }
    const img = this.image;

    Object.assign(out, {
      found: true, points: pts, isRight, fingerPx, palm,
      pose: { position, xAxis, yAxis, zAxis },
      image: img,
    });
    return this._finish(out, { ...handChecks(raw, pts, isRight, W, H, img), steady }, palm);
  }

  _finish(out, result, palm = 0) {
    for (const c of CHECKS) out.checks[c.id] = result[c.id] ?? false;
    const hardOk = CHECKS.every((c) => !c.hard || out.checks[c.id]);
    if (hardOk) { this.goodStreak++; this.badStreak = 0; } else { this.badStreak++; this.goodStreak = 0; }
    if (!this.ready && this.goodStreak >= 5) this.ready = true;
    if (this.ready && this.badStreak >= 8) this.ready = false;
    out.ready = this.ready;
    out.allOk = CHECKS.every((c) => out.checks[c.id]);
    const failing = CHECKS.find((c) => !out.checks[c.id]);
    out.tip = failing ? failing.tip : "Looking good";
    if (failing?.id === "distance" && palm >= 0.6) out.tip = "Move your hand a little farther away";
    if (failing?.id === "light" && out.image?.luma >= 0.92) out.tip = "Too bright: avoid direct light on the hand";
    return out;
  }
}
