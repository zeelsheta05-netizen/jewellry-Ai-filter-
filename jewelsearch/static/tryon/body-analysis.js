// Body photo analysis for the instant try-on: find the body part, run the
// verification checks, and compute the anchor points the preview needs.
//
//   hand -> rings, bracelets        (MediaPipe Hand Landmarker)
//   face -> earrings                (Face Landmarker: 478 points incl. irises)
//   neck -> necklaces, pendants     (Pose Landmarker shoulders + Face chin)
//
// Anchors are stored with the photo (normalised 0..1 image coordinates), so
// the preview on the dashboard needs no model at all. Every photo gets a
// real-size scale: the iris is ~11.7 mm across in every adult, which gives
// pixels per mm for face and neck photos.

import { FaceLandmarker, FilesetResolver, HandLandmarker, PoseLandmarker } from "/vendor/mediapipe/vision_bundle.mjs";
import { ImageProbe, handChecks, toPixels } from "./hand-tracker.js";
import { neckAnchors } from "./neck-geometry.js";

const WASM = "/vendor/mediapipe/wasm";
const MODELS = {
  hand: "/vendor/models/hand_landmarker.task",
  face: "/vendor/models/face_landmarker.task",
  pose: "/vendor/models/pose_landmarker_lite.task",
};
const IRIS_MM = 11.7;

// What each part is for, and the checks shown to the user, in fix order.
export const PARTS = {
  hand: {
    title: "Hand", use: "Rings & bracelets",
    how: "Back of your hand, fingers straight and a little apart, about 30 cm from the camera. Remove any ring from the ring finger.",
    checks: [
      ["hand", "Show one hand"], ["frame", "Keep all fingers in the photo"], ["wrist", "Wrist in the photo too (for bracelets)"],
      ["distance", "Hand at a good distance"],
      ["back", "Back of the hand to the camera"], ["straight", "Ring finger straight"],
      ["light", "Enough light"], ["sharp", "Sharp, not blurry"],
    ],
  },
  face: {
    title: "Face", use: "Earrings",
    how: "Look straight at the camera, hair behind your ears so both earlobes show. Remove earrings.",
    checks: [
      ["face", "Show your face"], ["frontal", "Look straight at the camera"], ["size", "Face at a good distance"],
      ["ears", "Both ears inside the photo"], ["earsVisible", "Earlobes not covered by hair"],
      ["light", "Enough light"], ["sharp", "Sharp, not blurry"],
    ],
  },
  neck: {
    title: "Neck", use: "Necklaces & pendants",
    how: "Face the camera, chin up a little, with both shoulders and your upper chest in the photo. A round or V neckline works best, with no collar covering the neck. Take off any necklace you are wearing.",
    checks: [
      ["face", "Show your face and chin"], ["shoulders", "Both shoulders in the photo"], ["chest", "Step back: upper chest in the photo too"], ["frontal", "Face the camera straight on"],
      ["neckVisible", "Neck not covered by a collar or scarf"], ["light", "Enough light"], ["sharp", "Sharp, not blurry"],
    ],
  },
};

let files = null;
const cache = {};
async function landmarker(kind, mode) {
  const key = kind + mode;
  if (cache[key]) return cache[key];
  files ??= await FilesetResolver.forVisionTasks(WASM);
  const Task = { hand: HandLandmarker, face: FaceLandmarker, pose: PoseLandmarker }[kind];
  const opts = (delegate) => ({
    baseOptions: { modelAssetPath: MODELS[kind], delegate }, runningMode: mode,
    ...(kind === "hand" ? { numHands: 1 } : kind === "face" ? { numFaces: 1 } : { numPoses: 1 }),
  });
  try {
    cache[key] = await Task.createFromOptions(files, opts("GPU"));
  } catch {
    cache[key] = await Task.createFromOptions(files, opts("CPU"));   // iOS Safari
  }
  return cache[key];
}

// Models a part needs, loaded up front so the camera check runs smoothly.
export async function prepare(part, mode) {
  if (part === "hand") await landmarker("hand", mode);
  if (part === "face" || part === "neck") await landmarker("face", mode);
  if (part === "neck") await landmarker("pose", mode);
}

const probe = new ImageProbe(96);
const dist = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);
const lerp = (a, b, t) => ({ x: a.x + (b.x - a.x) * t, y: a.y + (b.y - a.y) * t });

// Share of skin-coloured pixels in a box (YCrCb rule, works across skin tones).
// Used to tell whether hair covers an earlobe or a collar covers the neck.
const skinCanvas = document.createElement("canvas");
function skinShare(source, W, H, x0, y0, x1, y1) {
  const w = 32, h = 32, c = skinCanvas;
  c.width = w; c.height = h;
  const ctx = c.getContext("2d", { willReadFrequently: true });
  const sx = Math.max(0, x0 * W), sy = Math.max(0, y0 * H);
  const sw = Math.min(W, x1 * W) - sx, sh = Math.min(H, y1 * H) - sy;
  if (sw < 4 || sh < 4) return 0;
  ctx.drawImage(source, sx, sy, sw, sh, 0, 0, w, h);
  const d = ctx.getImageData(0, 0, w, h).data;
  let skin = 0;
  for (let i = 0; i < d.length; i += 4) {
    const r = d[i], g = d[i + 1], b = d[i + 2];
    const cr = 128 + 0.5 * r - 0.4187 * g - 0.0813 * b;
    const cb = 128 - 0.1687 * r - 0.3313 * g + 0.5 * b;
    if (cr > 133 && cr < 180 && cb > 77 && cb < 135) skin++;
  }
  return skin / (w * h);
}

function detect(lm, source, mode, t) {
  return mode === "VIDEO" ? lm.detectForVideo(source, t) : lm.detect(source);
}

function irisScale(f, W) {
  // iris edge points 469/471 (one eye) and 474/476 (the other), horizontal
  const a = dist({ x: f[469].x * W, y: 0 }, { x: f[471].x * W, y: 0 });
  const b = dist({ x: f[474].x * W, y: 0 }, { x: f[476].x * W, y: 0 });
  return ((a + b) / 2) / IRIS_MM;   // px per mm
}

// Analyse one image or video frame. Returns { found, checks, tip, allOk, anchors }.
export async function analyse(part, source, mode = "IMAGE", t = performance.now()) {
  const W = source.videoWidth || source.naturalWidth || source.width;
  const H = source.videoHeight || source.naturalHeight || source.height;
  const checks = {};
  let anchors = null, image = null;

  if (part === "hand") {
    const res = detect(await landmarker("hand", mode), source, mode, t);
    const raw = res.landmarks?.[0];
    if (raw) {
      const isRight = res.handedness?.[0]?.[0]?.categoryName === "Right";
      const pts = toPixels(raw, W, H);
      image = probe.measure(source, raw[13].x * W, raw[13].y * H, Math.max(24, dist(pts[13], pts[9]) * 1.6));
      Object.assign(checks, handChecks(raw, pts, isRight, W, H, image));
      // The same photo is used for bracelets: the wrist and a bit of forearm
      // below it must be in the picture (the live ring view doesn't need this).
      const palmY = raw[0].y - raw[9].y, palmX = raw[0].x - raw[9].x;
      // bracelet sits ~6% of a palm length past the wrist point and is ~7 mm wide
      const below = { x: raw[0].x + palmX * 0.15, y: raw[0].y + palmY * 0.15 };
      checks.wrist = [raw[0], below].every((p) => p.x > 0.02 && p.x < 0.98 && p.y > 0.02 && p.y < 0.98);
      anchors = { landmarks: raw.map((p) => [+p.x.toFixed(5), +p.y.toFixed(5), +p.z.toFixed(5)]), isRight };
    }
  }

  if (part === "face" || part === "neck") {
    const res = detect(await landmarker("face", mode), source, mode, t);
    const f = res.faceLandmarks?.[0];
    if (f) {
      checks.face = true;
      const faceW = dist(f[234], f[454]);          // cheek edge to cheek edge (normalised x)
      const mid = lerp(f[234], f[454], 0.5);
      checks.frontal = Math.abs(f[1].x - mid.x) / Math.max(faceW, 1e-3) < 0.09;
      const pxPerMm = irisScale(f, W);
      image = probe.measure(source, f[1].x * W, f[1].y * H, faceW * W * 0.5);

      if (part === "face") {
        checks.size = faceW > 0.16 && faceW < 0.62;
        // Earlobes are not in the face mesh: just outside the jaw contour,
        // below the ear opening (234/454), about level with the mouth (132/361).
        const out = faceW * 0.045;
        const lobeL = { x: lerp(f[234], f[132], 0.75).x - out, y: lerp(f[234], f[132], 0.75).y };
        const lobeR = { x: lerp(f[454], f[361], 0.75).x + out, y: lerp(f[454], f[361], 0.75).y };
        const m = faceW * 0.12;
        checks.ears = lobeL.x - m > 0 && lobeR.x + m < 1 && Math.max(lobeL.y, lobeR.y) + m < 1;
        const box = (p) => [p.x - m * 0.6, p.y - m * 0.9, p.x + m * 0.6, p.y + m * 0.4];
        checks.earsVisible = skinShare(source, W, H, ...box(lobeL)) > 0.3 && skinShare(source, W, H, ...box(lobeR)) > 0.3;
        const roll = Math.atan2((f[263].y - f[33].y) * H, (f[263].x - f[33].x) * W);
        anchors = { lobeL, lobeR, pxPerMm, roll, faceW };
      } else {
        const pose = detect(await landmarker("pose", mode), source, mode, t).landmarks?.[0];
        if (pose) {
          const sL = pose[12], sR = pose[11];   // image-left / image-right shoulder
          const vis = (p) => (p.visibility ?? 1) > 0.5 && p.x > 0.01 && p.x < 0.99 && p.y < 0.99;
          checks.shoulders = vis(sL) && vis(sR);
          checks.frontal = checks.frontal && Math.abs(sL.y - sR.y) < 0.12 * dist(sL, sR) * (W / H);
          const chin = f[152];
          // neck ~ 0.78 x face width; the chain leaves the skin where the neck meets the shoulders
          const { neckL, neckR, notch, v } = neckAnchors({ chin, shoulderL: sL, shoulderR: sR, halfWidth: faceW * 0.37, pxPerMm, H });
          const half = faceW * 0.37, down = notch.y - chin.y;
          // a pendant hangs 4-6 cm below the neck notch and is up to 3 cm long
          checks.chest = (1 - notch.y) * H >= 75 * pxPerMm;
          checks.neckVisible = skinShare(source, W, H, chin.x - half * 0.7, chin.y + down * 0.3, chin.x + half * 0.7, notch.y) > 0.45;
          anchors = { chin: { x: chin.x, y: chin.y }, neckL, neckR, notch, v, pxPerMm,
            shoulderL: { x: sL.x, y: sL.y }, shoulderR: { x: sR.x, y: sR.y } };
        }
      }
    }
  }

  if (image) {
    checks.light = image.luma > 0.16 && image.luma < 0.92;
    checks.sharp = image.sharpness > 35;
  }
  const list = PARTS[part].checks;
  for (const [id] of list) checks[id] = !!checks[id];
  const failing = list.find(([id]) => !checks[id]);
  return { found: !!anchors, checks, allOk: !failing, tip: failing ? failing[1] : "Perfect: all checks passed", anchors, width: W, height: H };
}
