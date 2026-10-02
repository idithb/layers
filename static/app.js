"use strict";

const $ = (id) => document.getElementById(id);
const canvas = $("canvas");
const ctx = canvas.getContext("2d");

const state = {
  file: null,
  width: 0,
  height: 0,
  background: null,      // HTMLImageElement
  bgVisible: true,
  layers: [],            // {id, name, kind, img, alpha, w, h, x, y, scale, visible, orig}
  selected: null,
  drag: null,
};

// ---------- helpers ----------

function loadImage(src) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = reject;
    img.src = src;
  });
}

function alphaOf(img) {
  const c = document.createElement("canvas");
  c.width = img.naturalWidth;
  c.height = img.naturalHeight;
  const g = c.getContext("2d");
  g.drawImage(img, 0, 0);
  return g.getImageData(0, 0, c.width, c.height).data;
}

function canvasToBlob(c) {
  return new Promise((resolve) => c.toBlob(resolve, "image/png"));
}

function imageToCanvas(img, w = img.naturalWidth, h = img.naturalHeight) {
  const c = document.createElement("canvas");
  c.width = w;
  c.height = h;
  c.getContext("2d").drawImage(img, 0, 0, w, h);
  return c;
}

function setBusy(on, text = "מעבד…") {
  $("busy").hidden = !on;
  $("busyText").textContent = text;
}

function setStatus(text) {
  $("status").textContent = text;
}

async function fetchJSON(path, options) {
  const res = await fetch(path, options);
  if (!res.ok) {
    let msg = res.statusText;
    try { msg = (await res.json()).detail || msg; } catch (_) {}
    throw new Error(msg);
  }
  return res.json();
}

// The server runs either our own job queue (/api/*, local) or the Gradio API
// (Hugging Face ZeroGPU Space, where the GPU is only available through Gradio).
const backend = fetchJSON("/api/status").then((s) => s.backend).catch(() => "jobs");

async function api(name, params, busyText) {
  return (await backend) === "gradio" ? gradioApi(name, params, busyText) : jobsApi(name, params, busyText);
}

// Submit a job, then poll until it finishes (processing can take minutes)
async function jobsApi(name, params, busyText) {
  const form = new FormData();
  for (const [k, v] of Object.entries(params)) form.append(k, v);
  const { job } = await fetchJSON(`/api/${name}`, { method: "POST", body: form });
  for (;;) {
    await new Promise((r) => setTimeout(r, 1500));
    const s = await fetchJSON(`/api/jobs/${job}`);
    if (s.status === "done") return s.result;
    if (s.status === "error") throw new Error(s.error);
    setBusy(true, s.ahead > 0 ? `ממתין בתור – ${s.ahead} לפניך…` : busyText);
  }
}

function blobToDataURL(blob) {
  return new Promise((resolve, reject) => {
    const r = new FileReader();
    r.onload = () => resolve(r.result);
    r.onerror = reject;
    r.readAsDataURL(blob);
  });
}

let gradioClient = null;
async function gradioApi(name, params, busyText) {
  if (!gradioClient) {
    const { Client } = await import("/ui/vendor/gradio-client.js");
    gradioClient = await Client.connect(new URL("/", location.href).href);
  }
  const data = {};
  for (const [k, v] of Object.entries(params)) data[k] = v instanceof Blob ? await blobToDataURL(v) : v;
  const job = gradioClient.submit(`/${name}`, data);
  for await (const msg of job) {
    if (msg.type === "status") {
      if (msg.stage === "error") throw new Error(msg.message || "שגיאה בשרת");
      if (msg.stage === "pending" && msg.position > 0) setBusy(true, `ממתין בתור – ${msg.position} לפניך…`);
      else setBusy(true, busyText);
    } else if (msg.type === "data") {
      return msg.data[0];
    }
  }
  throw new Error("no result");
}

function download(href, name) {
  const a = document.createElement("a");
  a.href = href;
  a.download = name;
  document.body.appendChild(a);
  a.click();
  a.remove();
}

// ---------- rendering ----------

function render(withUI = true, target = ctx) {
  target.clearRect(0, 0, state.width, state.height);
  if (state.background && state.bgVisible) {
    target.drawImage(state.background, 0, 0, state.width, state.height);
  }
  for (const l of state.layers) {
    if (!l.visible) continue;
    target.drawImage(l.img, l.x, l.y, l.w * l.scale, l.h * l.scale);
  }
  if (withUI && state.selected && state.selected.visible) {
    const l = state.selected;
    const px = pxPerCss();
    target.save();
    target.lineWidth = 1.5 * px;
    target.setLineDash([6 * px, 4 * px]);
    target.strokeStyle = "#4f46e5";
    target.strokeRect(l.x, l.y, l.w * l.scale, l.h * l.scale);
    target.setLineDash([]);
    target.fillStyle = "#fff";
    const hs = handleSize();
    const [hx, hy] = handlePos(l);
    target.fillRect(hx - hs / 2, hy - hs / 2, hs, hs);
    target.strokeRect(hx - hs / 2, hy - hs / 2, hs, hs);
    target.restore();
  }
  $("dims").textContent = state.width ? `${state.width} × ${state.height}` : "";
}

function pxPerCss() {
  return canvas.clientWidth ? state.width / canvas.clientWidth : 1;
}
function handleSize() { return 12 * pxPerCss(); }
function handlePos(l) { return [l.x + l.w * l.scale, l.y + l.h * l.scale]; }

// ---------- layers panel ----------

function renderLayerList() {
  const ul = $("layers");
  ul.innerHTML = "";
  if (!state.background) {
    ul.innerHTML = '<li class="empty">אין שכבות עדיין</li>';
    $("props").hidden = true;
    return;
  }
  // top-most first
  for (const l of [...state.layers].reverse()) {
    const li = document.createElement("li");
    if (l === state.selected) li.classList.add("selected");
    li.innerHTML = `<button class="eye" title="הצג/הסתר">${l.visible ? "👁" : "◌"}</button>
      <img alt=""><span class="name"></span><span class="kind">${l.kind === "text" ? "טקסט" : "תמונה"}</span>`;
    li.querySelector("img").src = l.img.src;
    li.querySelector(".name").textContent = l.name;
    li.querySelector(".eye").onclick = (e) => {
      e.stopPropagation();
      l.visible = !l.visible;
      renderLayerList();
      render();
    };
    li.onclick = () => select(l);
    ul.appendChild(li);
  }
  const bg = document.createElement("li");
  bg.className = "locked";
  bg.innerHTML = `<button class="eye" title="הצג/הסתר">${state.bgVisible ? "👁" : "◌"}</button>
    <img alt=""><span class="name">רקע</span><span class="kind">LaMa</span>`;
  bg.querySelector("img").src = state.background.src;
  bg.querySelector(".eye").onclick = () => {
    state.bgVisible = !state.bgVisible;
    renderLayerList();
    render();
  };
  ul.appendChild(bg);
  updateProps();
}

function select(l) {
  state.selected = l;
  renderLayerList();
  render();
}

function updateProps() {
  const l = state.selected;
  $("props").hidden = !l;
  if (!l) return;
  $("propX").value = Math.round(l.x);
  $("propY").value = Math.round(l.y);
  $("propScale").value = Math.round(l.scale * 100);
  $("propScaleVal").textContent = `${Math.round(l.scale * 100)}%`;
}

function setScale(l, scale) {
  // scale around the layer centre
  const cx = l.x + (l.w * l.scale) / 2;
  const cy = l.y + (l.h * l.scale) / 2;
  l.scale = Math.max(0.02, scale);
  l.x = cx - (l.w * l.scale) / 2;
  l.y = cy - (l.h * l.scale) / 2;
}

$("propX").oninput = (e) => { if (state.selected) { state.selected.x = +e.target.value; render(); } };
$("propY").oninput = (e) => { if (state.selected) { state.selected.y = +e.target.value; render(); } };
$("propScale").oninput = (e) => {
  if (!state.selected) return;
  setScale(state.selected, e.target.value / 100);
  updateProps();
  render();
};
$("btnUp").onclick = () => moveZ(1);
$("btnDown").onclick = () => moveZ(-1);
$("btnResetLayer").onclick = () => {
  const l = state.selected;
  if (!l) return;
  Object.assign(l, l.orig);
  updateProps();
  render();
};
$("btnDelete").onclick = () => {
  const l = state.selected;
  if (!l) return;
  state.layers = state.layers.filter((x) => x !== l);
  state.selected = null;
  renderLayerList();
  render();
};
$("btnFlatten").onclick = async () => {
  const l = state.selected;
  if (!l) return;
  const c = imageToCanvas(state.background, state.width, state.height);
  c.getContext("2d").drawImage(l.img, l.x, l.y, l.w * l.scale, l.h * l.scale);
  state.background = await loadImage(c.toDataURL("image/png"));
  state.layers = state.layers.filter((x) => x !== l);
  state.selected = null;
  renderLayerList();
  render();
};

function moveZ(dir) {
  const l = state.selected;
  if (!l) return;
  const i = state.layers.indexOf(l);
  const j = i + dir;
  if (j < 0 || j >= state.layers.length) return;
  [state.layers[i], state.layers[j]] = [state.layers[j], state.layers[i]];
  renderLayerList();
  render();
}

// ---------- canvas interaction ----------

function toCanvas(e) {
  const r = canvas.getBoundingClientRect();
  return [((e.clientX - r.left) / r.width) * state.width, ((e.clientY - r.top) / r.height) * state.height];
}

function hitTest(x, y) {
  for (let i = state.layers.length - 1; i >= 0; i--) {
    const l = state.layers[i];
    if (!l.visible) continue;
    const lx = (x - l.x) / l.scale;
    const ly = (y - l.y) / l.scale;
    if (lx < 0 || ly < 0 || lx >= l.w || ly >= l.h) continue;
    const a = l.alpha[(Math.floor(ly) * l.w + Math.floor(lx)) * 4 + 3];
    // generous for thin text strokes: accept anything inside a text box
    if (a > 20 || l.kind === "text") return l;
  }
  return null;
}

canvas.addEventListener("pointerdown", (e) => {
  if (!state.background) return;
  const [x, y] = toCanvas(e);
  const s = state.selected;
  if (s && s.visible) {
    const [hx, hy] = handlePos(s);
    const hs = handleSize();
    if (Math.abs(x - hx) <= hs && Math.abs(y - hy) <= hs) {
      state.drag = { mode: "scale", l: s, x0: x, y0: y, scale0: s.scale };
      canvas.setPointerCapture(e.pointerId);
      return;
    }
  }
  const hit = hitTest(x, y);
  if (hit) {
    if (hit !== state.selected) select(hit);
    state.drag = { mode: "move", l: hit, dx: x - hit.x, dy: y - hit.y };
    canvas.setPointerCapture(e.pointerId);
  } else if (state.selected) {
    select(null);
  }
});

canvas.addEventListener("pointermove", (e) => {
  if (!state.background) return;
  const [x, y] = toCanvas(e);
  const d = state.drag;
  if (!d) {
    const s = state.selected;
    let cursor = "default";
    if (s && s.visible) {
      const [hx, hy] = handlePos(s);
      if (Math.abs(x - hx) <= handleSize() && Math.abs(y - hy) <= handleSize()) cursor = "nwse-resize";
    }
    if (cursor === "default" && hitTest(x, y)) cursor = "move";
    canvas.style.cursor = cursor;
    return;
  }
  if (d.mode === "move") {
    d.l.x = x - d.dx;
    d.l.y = y - d.dy;
  } else {
    // keep top-left fixed, scale with the handle
    const w0 = d.l.w * d.scale0;
    const h0 = d.l.h * d.scale0;
    const f = Math.max((x - d.l.x) / w0, (y - d.l.y) / h0);
    d.l.scale = Math.max(0.02, d.scale0 * f);
  }
  updateProps();
  render();
});

const endDrag = () => { state.drag = null; };
canvas.addEventListener("pointerup", endDrag);
canvas.addEventListener("pointercancel", endDrag);

window.addEventListener("keydown", (e) => {
  const l = state.selected;
  if (!l || ["INPUT", "SELECT", "TEXTAREA"].includes(document.activeElement.tagName)) return;
  const step = e.shiftKey ? 10 : 1;
  const moves = { ArrowLeft: [-step, 0], ArrowRight: [step, 0], ArrowUp: [0, -step], ArrowDown: [0, step] };
  if (moves[e.key]) {
    l.x += moves[e.key][0];
    l.y += moves[e.key][1];
    e.preventDefault();
  } else if (e.key === "Delete" || e.key === "Backspace") {
    $("btnDelete").click();
    e.preventDefault();
  } else return;
  updateProps();
  render();
});

window.addEventListener("resize", () => render());

// ---------- upload & separate ----------

function setFile(file) {
  if (!file || !file.type.startsWith("image/")) return;
  state.file = file;
  $("btnSeparate").disabled = false;
  $("drop").querySelector("span").textContent = file.name;
  // preview the original right away
  loadImage(URL.createObjectURL(file)).then((img) => {
    if (state.layers.length) return;
    state.width = img.naturalWidth;
    state.height = img.naturalHeight;
    canvas.width = state.width;
    canvas.height = state.height;
    canvas.classList.add("ready");
    $("placeholder").hidden = true;
    ctx.drawImage(img, 0, 0);
    $("dims").textContent = `${state.width} × ${state.height}`;
  });
}

$("file").onchange = (e) => setFile(e.target.files[0]);
const drop = $("drop");
["dragenter", "dragover"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
drop.addEventListener("drop", (e) => setFile(e.dataTransfer.files[0]));
$("stage").addEventListener("dragover", (e) => e.preventDefault());
$("stage").addEventListener("drop", (e) => { e.preventDefault(); setFile(e.dataTransfer.files[0]); });

$("btnSeparate").onclick = async () => {
  if (!state.file) return;
  const params = {
    image: state.file,
    detect_text: $("optText").checked,
    detect_subject: $("optSubject").checked,
    merge_text_lines: $("optMerge").checked,
  };
  const busyText = "מפריד שכבות… (זה יכול לקחת דקה-שתיים)";
  setBusy(true, busyText);
  try {
    const data = await api("separate", params, busyText);
    state.width = data.width;
    state.height = data.height;
    state.background = await loadImage(data.background);
    state.bgVisible = true;
    state.layers = await Promise.all(
      data.layers.map(async (l) => {
        const img = await loadImage(l.image);
        return {
          id: l.id, name: l.name, kind: l.kind, img, alpha: alphaOf(img),
          w: l.width, h: l.height, x: l.x, y: l.y, scale: 1, visible: true,
          orig: { x: l.x, y: l.y, scale: 1 },
        };
      })
    );
    state.selected = null;
    canvas.width = state.width;
    canvas.height = state.height;
    canvas.classList.add("ready");
    $("placeholder").hidden = true;
    $("outW").value = state.width;
    $("outH").value = state.height;
    ["btnResize", "btnExport", "btnExportLayers"].forEach((id) => ($(id).disabled = false));
    const i = data.info || {};
    const parts = [`${data.layers.filter((l) => l.kind === "text").length} שכבות טקסט`];
    parts.push(data.layers.some((l) => l.kind === "subject") ? "תמונה הופרדה" : (i.subject ? `תמונה: ${i.subject}` : "ללא תמונה"));
    parts.push(`רקע: ${i.inpainter === "lama" ? "LaMa" : "OpenCV (LaMa לא זמין)"}`);
    setStatus(parts.join(" · "));
    renderLayerList();
    render();
  } catch (err) {
    alert("שגיאה בהפרדה: " + err.message);
  } finally {
    setBusy(false);
  }
};

// ---------- resize ----------

$("presets").addEventListener("click", (e) => {
  const b = e.target.closest("button");
  if (!b) return;
  $("outW").value = b.dataset.w;
  $("outH").value = b.dataset.h;
});

$("btnResize").onclick = async () => {
  const w = parseInt($("outW").value, 10);
  const h = parseInt($("outH").value, 10);
  if (!(w >= 16 && h >= 16 && w <= 4096 && h <= 4096)) {
    alert("הגודל צריך להיות בין 16 ל-4096 פיקסלים");
    return;
  }
  const params = {
    background: await canvasToBlob(imageToCanvas(state.background, state.width, state.height)),
    width: w,
    height: h,
    mode: $("resizeMode").value,
  };
  const busyText = "משנה גודל ומשלים רקע…";
  setBusy(true, busyText);
  try {
    const data = await api("resize", params, busyText);
    const t = data.transform;
    const s = Math.min(t.sx, t.sy);
    for (const l of state.layers) {
      const cx = l.x + (l.w * l.scale) / 2;
      const cy = l.y + (l.h * l.scale) / 2;
      l.scale *= s;
      l.x = cx * t.sx + t.ox - (l.w * l.scale) / 2;
      l.y = cy * t.sy + t.oy - (l.h * l.scale) / 2;
      l.orig = { x: l.x, y: l.y, scale: l.scale };
    }
    state.background = await loadImage(data.background);
    state.width = data.width;
    state.height = data.height;
    canvas.width = state.width;
    canvas.height = state.height;
    renderLayerList();
    render();
  } catch (err) {
    alert("שגיאה בשינוי גודל: " + err.message);
  } finally {
    setBusy(false);
  }
};

// ---------- export ----------

function baseName() {
  return (state.file ? state.file.name.replace(/\.[^.]+$/, "") : "image") + "-layers";
}

$("btnExport").onclick = () => {
  const c = document.createElement("canvas");
  c.width = state.width;
  c.height = state.height;
  render(false, c.getContext("2d"));
  download(c.toDataURL("image/png"), `${baseName()}.png`);
};

$("btnExportLayers").onclick = async () => {
  download(imageToCanvas(state.background, state.width, state.height).toDataURL("image/png"), `${baseName()}-background.png`);
  for (const l of state.layers) {
    await new Promise((r) => setTimeout(r, 250)); // browsers throttle rapid downloads
    // each layer on a full-size transparent canvas, at its current position
    const c = document.createElement("canvas");
    c.width = state.width;
    c.height = state.height;
    c.getContext("2d").drawImage(l.img, l.x, l.y, l.w * l.scale, l.h * l.scale);
    download(c.toDataURL("image/png"), `${baseName()}-${l.id}.png`);
  }
};

fetch("/api/status").then((r) => r.json()).then((s) => {
  const missing = [];
  if (!s.birefnet) missing.push("BiRefNet");
  if (!s.lama) missing.push("LaMa");
  if (missing.length) setStatus(`חסרים מודלים: ${missing.join(", ")} – הרץ scripts/download_models.py`);
}).catch(() => {});
