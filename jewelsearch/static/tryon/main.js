// Ring try-on page: camera -> hand tracker -> ring renderer, plus the
// verification guide, capture (freeze) mode, and metal / size / design pickers.

import { CHECKS, HandTracker, createHandLandmarker } from "./hand-tracker.js";
import { METALS, RingView, renderThumbnail } from "./ring-view.js";

const $ = (s) => document.querySelector(s);
const params = new URLSearchParams(location.search);

const video = $("#video");
const stage = $("#stage");
// The screen shows this canvas, not the <video>: each camera frame is drawn
// here in the same step as it is analysed, so the ring and the hand are always
// from the same frame. (With the live <video> the picture ran 1-2 frames ahead
// of the tracking and the ring trailed behind a moving hand.)
const frameCanvas = $("#frame");
const frameCtx = frameCanvas.getContext("2d", { alpha: false });
const DEBUG = params.get("debug") === "1";   // ?debug=1: fps and detection time
$("#debug").hidden = !DEBUG;
const view = new RingView($("#gl"));

// ---------- ring sizes (US scale, with Indian size and mm) ----------
const usToMm = (us) => 11.63 + 0.8128 * us;
const inSize = (mm) => Math.round(Math.PI * mm - 40);
const sizeSel = $("#size");
for (let us = 3; us <= 13; us += 0.5) {
  const mm = usToMm(us);
  sizeSel.add(new Option(`India ${inSize(mm)} · US ${us} · ${mm.toFixed(1)} mm`, mm.toFixed(2)));
}
sizeSel.value = usToMm(6.5).toFixed(2);   // an average size until the user picks theirs
const showSize = () => { $("#sizeNow").textContent = `India ${inSize(+sizeSel.value)}`; };
view.setRingSize(+sizeSel.value);
showSize();
sizeSel.onchange = () => { view.setRingSize(+sizeSel.value); showSize(); };

// ---------- metal ----------
const metalBox = $("#metals");
for (const [key, m] of Object.entries(METALS)) {
  const b = document.createElement("button");
  b.className = "metal";
  b.dataset.metal = key;
  b.title = m.label;
  b.innerHTML = `<i class="dot ${key}"></i><span>${m.label.split(" ")[0]}</span>`;   // "Yellow", fits a phone row
  b.onclick = () => selectMetal(key);
  metalBox.append(b);
}
function selectMetal(key) {
  view.setMetal(key);
  metalBox.querySelectorAll("button").forEach((b) => b.setAttribute("aria-pressed", b.dataset.metal === key));
}
selectMetal(["yellow", "rose", "white"].includes(params.get("metal")) ? params.get("metal") : "yellow");

// ---------- designs ----------
let models = [];
const prettyName = (slug) => slug.replace(/_/g, " ");
// the version changes when a model is converted again (browser cache)
const modelUrl = (slug) => `/api/tryon/models/${encodeURIComponent(slug)}.glb?v=${models.find((m) => m.slug === slug)?.version ?? 0}`;

async function loadModels() {
  const r = await fetch("/api/tryon/models?category=ring");
  if (r.status === 401) { location.href = "/login?next=" + encodeURIComponent(location.pathname + location.search); return; }
  models = (await r.json()).items;
  const box = $("#designs");
  box.innerHTML = "";
  for (const m of models) {
    const b = document.createElement("button");
    b.className = "design";
    b.dataset.slug = m.slug;
    b.title = prettyName(m.slug);
    b.innerHTML = `<span class="thumb"></span><small>${prettyName(m.slug)}</small>`;
    b.onclick = () => selectDesign(m.slug);
    box.append(b);
  }
  $("#designCount").textContent = `${models.length}`;
  const want = params.get("model");
  if (models.length) await selectDesign(models.some((m) => m.slug === want) ? want : models[0].slug);
  else $("#title").textContent = "No ring models yet";
}

// Thumbnails are rendered here from the same 3D models, one after another,
// so the design picker shows the actual rings instead of code names. Started
// only once hand tracking is running: doing both at once is the memory peak
// that can crash a phone tab.
let thumbsStarted = false;
async function drawThumbnails() {
  if (thumbsStarted) return;
  thumbsStarted = true;
  for (const b of document.querySelectorAll(".design")) {
    try {
      const src = await renderThumbnail(view, modelUrl(b.dataset.slug));
      b.querySelector(".thumb").style.backgroundImage = `url(${src})`;
    } catch (e) { console.warn("thumbnail", b.dataset.slug, e); }
  }
}

let loadingSlug = null;
async function selectDesign(slug) {
  const meta = models.find((m) => m.slug === slug);
  loadingSlug = slug;
  $("#title").textContent = prettyName(slug);
  const stones = Object.values(meta.stones || {}).reduce((a, n) => a + n, 0);
  $("#subtitle").textContent = stones ? `${stones} diamond${stones > 1 ? "s" : ""}` : "Plain metal";
  document.querySelectorAll(".design").forEach((b) => b.setAttribute("aria-pressed", b.dataset.slug === slug));
  document.querySelector(`.design[data-slug="${CSS.escape(slug)}"]`)?.scrollIntoView({ inline: "center", block: "nearest", behavior: "smooth" });
  const u = new URL(location.href);
  u.searchParams.set("model", slug);
  history.replaceState(null, "", u);
  await view.loadRing(modelUrl(slug), meta);
  if (loadingSlug === slug) view.setRingSize(+sizeSel.value);
}

// ---------- guide: one short tip at a time; tap for the full checklist ----------
const checkBox = $("#checks");
const LABELS = { hand: "Hand", frame: "In frame", distance: "Distance", back: "Back of hand",
  straight: "Finger straight", light: "Light", sharp: "Sharp", steady: "Steady" };
for (const c of CHECKS) {
  const li = document.createElement("li");
  li.dataset.id = c.id;
  li.textContent = LABELS[c.id];
  checkBox.append(li);
}
$("#pill").onclick = () => $("#guide").classList.toggle("open");

let lastTip = "", lastState = "";
function setGuide(state, tip) {
  if (tip !== lastTip) { $("#tip").textContent = tip; lastTip = tip; }
  if (state !== lastState) { $("#guide").dataset.state = state; lastState = state; }
}
function showChecks(hand) {
  for (const li of checkBox.children) li.classList.toggle("ok", !!hand.checks[li.dataset.id]);
  const done = CHECKS.filter((c) => hand.checks[c.id]).length;
  $("#progress").style.setProperty("--p", done / CHECKS.length);
  $("#handHint").classList.toggle("show", !hand.found);
  if (hand.ready) setGuide("ready", hand.allOk ? "Ready: tap the button to take a photo" : hand.tip);
  else setGuide(hand.found ? "fix" : "search", hand.tip);
  $("#shutter").disabled = !hand.ready;
}

// ---------- camera ----------
let facing = params.get("camera") === "user" ? "user" : "environment";
let stream = null, mirrored = false;

async function startCamera() {
  stream?.getTracks().forEach((t) => t.stop());
  stream = await navigator.mediaDevices.getUserMedia({
    audio: false,
    video: { facingMode: { ideal: facing }, width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: 30 } },
  });
  video.srcObject = stream;
  await video.play();
  // a laptop has only a front camera even if we asked for the back one
  const actual = stream.getVideoTracks()[0].getSettings().facingMode;
  mirrored = (actual || facing) === "user";
  stage.classList.toggle("mirror", mirrored);
  fit();
}

// Stage = video pixels; CSS scales it to cover the camera area.
function fit() {
  const W = video.videoWidth, H = video.videoHeight;
  if (!W) return;
  const area = $("#cameraArea").getBoundingClientRect();
  const s = Math.max(area.width / W, area.height / H);
  stage.style.width = W + "px";
  stage.style.height = H + "px";
  if (frameCanvas.width !== W || frameCanvas.height !== H) { frameCanvas.width = W; frameCanvas.height = H; }
  stage.style.setProperty("--s", s);
  view.setSize(W, H, s);
}
addEventListener("resize", fit);

// ---------- capture: freeze the frame, then compare designs on the photo ----------
let frozen = false, frozenHand = null;

function capture() {
  if (frozen || !view.lastHand?.ready) return;
  frozen = true;
  frozenHand = view.lastHand;
  video.pause();
  document.body.classList.add("frozen");
  setGuide("photo", "Photo mode: switch design, metal or size");
}
function retake() {
  frozen = false;
  frozenHand = null;
  document.body.classList.remove("frozen");
  tracker?.filter.reset();
  video.play();
}
async function savePhoto() {
  const canvas = view.snapshot(frameCanvas, mirrored);
  const blob = await new Promise((r) => canvas.toBlob(r, "image/jpeg", 0.92));
  const file = new File([blob], `ring-${loadingSlug || "try-on"}.jpg`, { type: "image/jpeg" });
  // Phone: share sheet (WhatsApp, Photos...). Laptop: plain download. Nothing is uploaded.
  if (navigator.canShare?.({ files: [file] })) {
    try { await navigator.share({ files: [file], title: "My ring try-on" }); return; }
    catch (e) { if (e.name === "AbortError") return; }
  }
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = file.name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 5000);
}
$("#shutter").onclick = capture;
$("#retake").onclick = retake;
$("#save").onclick = savePhoto;

// ---------- main loop ----------
let tracker = null, frame = 0, lastVideoTime = -1;
const perf = { frames: 0, detectMs: 0, since: performance.now() };
function loop() {
  if (frozen) {
    view.update(frozenHand);
  } else if (video.readyState >= 2 && video.currentTime !== lastVideoTime) {
    lastVideoTime = video.currentTime;
    if (video.videoWidth !== view.size[0] || video.videoHeight !== view.size[1]) fit();
    frameCtx.drawImage(video, 0, 0, frameCanvas.width, frameCanvas.height);
    const t0 = performance.now();
    const hand = tracker.update(frameCanvas, t0);
    perf.detectMs += performance.now() - t0;
    perf.frames++;
    view.update(hand);
    if (frame++ % 8 === 0) view.matchLighting(hand.image);
    showChecks(hand);
  }
  view.render();
  if (DEBUG && performance.now() - perf.since > 1000) {
    const secs = (performance.now() - perf.since) / 1000;
    $("#debug").textContent = `${Math.round(perf.frames / secs)} fps · detect ${Math.round(perf.detectMs / Math.max(perf.frames, 1))} ms · ${view.size[0]}x${view.size[1]} @${view.ratio?.toFixed(2)}`;
    Object.assign(perf, { frames: 0, detectMs: 0, since: performance.now() });
  }
  requestAnimationFrame(loop);
}

function step(id, state) { $(`#steps [data-step="${id}"]`).dataset.state = state; }

async function start() {
  const btn = $("#start");
  btn.disabled = true;
  if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia) {
    $("#startMsg").textContent = "The camera needs a secure (https) link. Open the app through its https address.";
    btn.disabled = false;
    return;
  }
  $("#steps").hidden = false;
  try {
    step("camera", "busy");
    await startCamera();
    step("camera", "done");
    step("tracking", "busy");
    tracker = new HandTracker(await createHandLandmarker());
    step("tracking", "done");
    $("#intro").hidden = true;
    fit();
    requestAnimationFrame(loop);
    setTimeout(drawThumbnails, 800);
  } catch (e) {
    console.error(e);
    $("#startMsg").textContent = e.name === "NotAllowedError"
      ? "Camera permission was blocked. Allow the camera for this site in your browser settings, then try again."
      : "Couldn't start: " + (e.message || e);
    btn.disabled = false;
  }
}

$("#start").onclick = start;
$("#flip").onclick = async () => {
  facing = facing === "user" ? "environment" : "user";
  if (frozen) retake();
  if (stream) { await startCamera(); tracker?.filter.reset(); }
};
$("#more").onclick = () => document.body.classList.toggle("panel-open");

// ?autostart=1: coming from a "Try on" button, the user has already tapped once
if (params.get("autostart") === "1") start();

loadModels().catch((e) => { console.error(e); $("#title").textContent = "Couldn't load designs"; });
