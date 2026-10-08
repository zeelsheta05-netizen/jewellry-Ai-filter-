// "My try-on photos": take or upload a hand / face / neck photo, verify it,
// let the user fine-tune the anchor points, and save it (with consent) for
// the instant try-on on the dashboard.

import { PARTS, analyse, prepare } from "./body-analysis.js";

const $ = (s) => document.querySelector(s);
const params = new URLSearchParams(location.search);
// where to go after saving (a dashboard design waiting for this photo)
const back = /^\/(?!\/)[^\\]*$/.test(params.get("return") || "") ? params.get("return") : "/";
$("#back").href = back;

const MAX_SIDE = 1600;          // saved photo size: plenty for a phone screen
const AUTO_CAPTURE_FRAMES = 6;  // all checks green this many analysed frames in a row
const ICON = { hand: "✋", face: "🙂", neck: "💎" };
let photos = {};

// ---------- cards ----------
async function load() {
  const r = await fetch("/api/body");
  if (r.status === 401) { location.href = "/login?next=" + encodeURIComponent(location.pathname + location.search); return; }
  photos = (await r.json()).parts;
  render();
}

function render() {
  const box = $("#cards");
  box.innerHTML = "";
  for (const [part, info] of Object.entries(PARTS)) {
    const have = photos[part];
    const card = document.createElement("article");
    card.className = "card";
    card.innerHTML = `
      <div class="pic">${have ? "" : ICON[part]}</div>
      <div class="body">
        <span class="use">${info.use}</span>
        <h2>${info.title}</h2>
        <span class="status ${have ? "ok" : ""}">${have ? "✓ Ready for try-on" : "No photo yet"}</span>
        <div class="actions">
          <button class="btn primary" data-act="camera">${have ? "Retake" : "Take photo"}</button>
          <button class="btn" data-act="upload">Upload</button>
          ${have ? '<button class="btn danger" data-act="delete">Delete</button>' : ""}
        </div>
      </div>`;
    if (have) {
      const pic = card.querySelector(".pic");
      pic.style.backgroundImage = `url(${have.url})`;
      // keep the face in the crop: anchor at the upper part of the photo
      if (part !== "hand") pic.style.backgroundPosition = `center ${part === "face" ? 15 : 30}%`;
    }
    card.querySelector('[data-act="camera"]').onclick = () => openSheet(part, "camera");
    card.querySelector('[data-act="upload"]').onclick = () => openSheet(part, "upload");
    card.querySelector('[data-act="delete"]')?.addEventListener("click", () => remove(part));
    box.append(card);
  }
}

async function remove(part) {
  if (!confirm(`Delete your ${PARTS[part].title.toLowerCase()} photo?`)) return;
  photos = (await (await fetch(`/api/body/${part}`, { method: "DELETE" })).json()).parts;
  render();
}
$("#deleteAll").onclick = async () => {
  if (!confirm("Delete all your try-on photos from the server?")) return;
  await fetch("/api/body", { method: "DELETE" });
  photos = {};
  render();
};

// ---------- capture sheet ----------
const video = $("#video"), shot = $("#shot");
let part = null, stream = null, facing = "user", running = false, goodRun = 0;
let result = null;   // last analysis of the photo being reviewed

function showChecks(res) {
  const list = $("#checks");
  if (list.dataset.part !== part) {
    list.innerHTML = PARTS[part].checks.map(([id, label]) => `<li data-id="${id}">${label}</li>`).join("");
    list.dataset.part = part;
  }
  for (const li of list.children) li.classList.toggle("ok", !!res?.checks[li.dataset.id]);
  $("#tip").textContent = res ? res.tip : "Looking for you…";
}

async function openSheet(p, how) {
  part = p;
  result = null;
  $("#sheetTitle").textContent = `${PARTS[p].title} photo`;
  $("#how").textContent = PARTS[p].how;
  $("#err").textContent = "";
  $("#sheet").hidden = false;
  document.body.style.overflow = "hidden";
  review(false);
  showChecks(null);
  if (how === "upload") { $("#file").click(); return; }
  // the hand is easiest with the back camera; face and neck are selfies
  facing = p === "hand" ? "environment" : "user";
  await startCamera();
}

function closeSheet() {
  running = false;
  stream?.getTracks().forEach((t) => t.stop());
  stream = null;
  $("#sheet").hidden = true;
  document.body.style.overflow = "";
}
$("#close").onclick = closeSheet;

async function startCamera() {
  stream?.getTracks().forEach((t) => t.stop());
  review(false);
  $("#tip").textContent = "Starting camera…";
  if (!window.isSecureContext || !navigator.mediaDevices) {
    // browsers only allow the camera on https:// or http://localhost
    $("#err").textContent = "The browser blocks the camera on this http:// address. On a phone, open the https:// link "
      + `instead. On the computer running Design Finder, open localhost:${location.port || 80}. Or tap "Upload photo" below, which works here.`;
    $("#tip").textContent = "";
    return;
  }
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: false,
      video: { facingMode: { ideal: facing }, width: { ideal: 1920 }, height: { ideal: 1080 } } });
  } catch (e) {
    $("#err").textContent = e.name === "NotAllowedError"
      ? "Camera permission was blocked. Allow it in the browser settings, or upload a photo instead."
      : "Couldn't open the camera. You can upload a photo instead.";
    return;
  }
  video.srcObject = stream;
  await video.play();
  const actual = stream.getVideoTracks()[0].getSettings().facingMode;
  video.classList.toggle("mirror", (actual || facing) === "user");
  $("#tip").textContent = "Loading…";
  await prepare(part, "VIDEO");
  running = true;
  goodRun = 0;
  tick();
}
$("#flip").onclick = () => { facing = facing === "user" ? "environment" : "user"; startCamera(); };

// Analyse ~6 frames a second; auto-capture once every check has been green for a moment.
const frameCanvas = document.createElement("canvas");
async function tick() {
  if (!running || !video.videoWidth) { if (running) setTimeout(tick, 150); return; }
  frameCanvas.width = video.videoWidth;
  frameCanvas.height = video.videoHeight;
  frameCanvas.getContext("2d").drawImage(video, 0, 0);
  const res = await analyse(part, frameCanvas, "VIDEO", performance.now());
  if (!running) return;
  showChecks(res);
  goodRun = res.allOk ? goodRun + 1 : 0;
  $("#shutter").disabled = !res.allOk;
  if (goodRun >= AUTO_CAPTURE_FRAMES) return capture(frameCanvas, res);
  setTimeout(tick, 150);
}
$("#shutter").onclick = async () => {
  const res = await analyse(part, frameCanvas, "VIDEO", performance.now());
  if (res.allOk) capture(frameCanvas, res);
};

// Keep the checked frame (not a later one) and switch to review.
function capture(source, res) {
  running = false;
  stream?.getTracks().forEach((t) => t.stop());
  const k = Math.min(1, MAX_SIDE / Math.max(source.width, source.height));
  shot.width = Math.round(source.width * k);
  shot.height = Math.round(source.height * k);
  shot.getContext("2d").drawImage(source, 0, 0, shot.width, shot.height);
  result = res;
  review(true);
}

// ---------- upload ----------
$("#upload").onclick = () => $("#file").click();
$("#file").onchange = async () => {
  const f = $("#file").files[0];
  $("#file").value = "";
  if (!f) return;
  running = false;
  stream?.getTracks().forEach((t) => t.stop());
  $("#err").textContent = "";
  $("#tip").textContent = "Checking your photo…";
  const img = await createImageBitmap(f, { imageOrientation: "from-image" }).catch(() => null);
  if (!img) { $("#err").textContent = "That file isn't a photo this browser can read."; return; }
  const k = Math.min(1, MAX_SIDE / Math.max(img.width, img.height));
  shot.width = Math.round(img.width * k);
  shot.height = Math.round(img.height * k);
  shot.getContext("2d").drawImage(img, 0, 0, shot.width, shot.height);
  await prepare(part, "IMAGE");
  result = await analyse(part, shot, "IMAGE");
  showChecks(result);
  review(true);
};

// ---------- review: adjust the points, consent, save ----------
function review(on) {
  video.hidden = on;
  shot.hidden = !on;
  $("#shutter").hidden = on;
  $("#upload").hidden = on;
  $("#retake").hidden = !on;
  $("#save").hidden = !on;
  $("#consentRow").hidden = !on;
  $("#flip").hidden = on;
  document.querySelectorAll("#frameBox .dot").forEach((d) => d.remove());
  if (!on) return;
  $("#consent").checked = false;
  if (!result?.allOk) {
    // verification layer: a photo that fails a check cannot be saved
    $("#err").textContent = result?.found
      ? `Not quite: ${result.tip.toLowerCase()}. Please retake or upload another photo.`
      : `We couldn't find your ${PARTS[part].title.toLowerCase()} in this photo. Please retake or upload another.`;
    $("#save").disabled = true;
    $("#consentRow").hidden = true;
    return;
  }
  $("#err").textContent = "";
  $("#tip").textContent = part === "hand" ? "Looks good" : "Drag the dots if they are not exactly right";
  placeDots();
  updateSave();
}
$("#retake").onclick = () => { result = null; showChecks(null); startCamera(); };
$("#consent").onchange = updateSave;
function updateSave() { $("#save").disabled = !($("#consent").checked && result?.allOk); }

const DOTS = {
  face: [["lobeL", "earlobe"], ["lobeR", "earlobe"]],
  neck: [["neckL", "neck side"], ["neckR", "neck side"], ["notch", "neck base"]],
};
function placeDots() {
  const box = $("#frameBox");
  for (const [key, label] of DOTS[part] || []) {
    const p = result.anchors[key];
    const d = document.createElement("div");
    d.className = "dot";
    d.innerHTML = `<span>${label}</span>`;
    const put = () => { d.style.left = `${p.x * 100}%`; d.style.top = `${p.y * 100}%`; };
    put();
    d.onpointerdown = (e) => {
      d.setPointerCapture(e.pointerId);
      d.onpointermove = (ev) => {
        const r = shot.getBoundingClientRect();
        p.x = Math.min(1, Math.max(0, (ev.clientX - r.left) / r.width));
        p.y = Math.min(1, Math.max(0, (ev.clientY - r.top) / r.height));
        put();
      };
      d.onpointerup = () => { d.onpointermove = null; };
    };
    box.append(d);
  }
}

$("#save").onclick = async () => {
  const btn = $("#save");
  btn.disabled = true;
  btn.textContent = "Saving…";
  try {
    const r = await fetch(`/api/body/${part}`, {
      method: "PUT", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image: shot.toDataURL("image/jpeg", 0.88), width: shot.width, height: shot.height,
        anchors: result.anchors, consent: $("#consent").checked }),
    });
    const data = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(data.detail || "Couldn't save the photo.");
    photos = data.parts;
    closeSheet();
    if (params.get("return")) { location.href = back; return; }
    render();
  } catch (e) {
    $("#err").textContent = e.message;
  } finally {
    btn.textContent = "Save photo";
    updateSave();
  }
};

await load();
// ?part=hand: opened from a design's "Try on" button without a photo yet
if (PARTS[params.get("part")]) openSheet(params.get("part"), params.get("how") === "upload" ? "upload" : "camera");
