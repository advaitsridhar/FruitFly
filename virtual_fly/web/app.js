// app.js — Virtual Fly: the page that shows a simulated whole fly nervous system driving a fly in a dish.
// Data comes from the Python server (see docs/API.md): one layout fetch, then a stream of state ticks.
"use strict";
import { $, setText, setClass, setShown, esc, fmt, post, getJSON, el } from "./util.js";
import { Arena, lerpAngle } from "./arena.js";
import { BrainView, REGION_COLORS } from "./brain3d.js";
import { RetinaView } from "./retina.js";
import * as P from "./panels.js";

// ---------------------------------------------------------------- state
let L = null;                       // layout (brain map, arena size, readouts, odours ...)
let S = null, Sprev = null, Stime = 0, Sgap = 25, lastSeq = 0, lastStateAt = 0, firstState = false;
let tool = "lure", odourFood = "", pointer = null, lastSent = 0, sees = false;
let arena = null, brain = null, retina = null;
const panels = {};
let toolOrder = ["lure", "hand", "sugar", "bitter", "water", "dust", "shock", "post"];
const HINTS = {
  lure: "Wiggle the lure slowly beside the fly: it turns toward small moving things (a courtship-chase circuit).",
  hand: "Swoop the hand straight at the fly, fast. A slow hand doesn't scare it.",
  sugar: "Click just in front of the fly's head to drop sugar. A hungry fly eats more eagerly.",
  bitter: "Click to drop bitter food. Try it right next to sugar.",
  water: "Click to drop water: a thirsty fly drinks it.",
  dust: "Click near the fly to puff dust at its antennae.",
  shock: "Click anywhere: an electric shock drives the PPL1 punishment dopamine neurons and pairs with whatever it smells now.",
  post: "Click to plant a post. The fly can see it and bump into it.",
};

// ---------------------------------------------------------------- start-up
async function loadLayout() {
  while (!L) {
    try { const r = await fetch("api/layout"); if (!r.ok) throw new Error(r.status); L = await r.json(); }
    catch (e) { await new Promise((res) => setTimeout(res, 1000)); }
  }
  setText($("sub"), `${L.n.toLocaleString()} neurons · ${(L.edges / 1e6).toFixed(1)} M connections · ${(L.synapses / 1e6).toFixed(0)} M synapses · MaleCNS v1.0`);
  buildToolbar();
  arena = new Arena($("arena"), $("stage"), L);
  new ResizeObserver(() => arena.resize()).observe($("stage"));
  retina = new RetinaView($("retina"), L.retina || {});
  brain = new BrainView($("brain"), $("brainOverlay"), L);
  setText($("brainCount"), `${brain.m.toLocaleString()} somas`);
  $("legend").innerHTML = L.regions.map((r, i) => `<span><i style="background:${REGION_COLORS[i]}"></i>${esc(r)}</span>`).join("");
  wireBrain();
  panels.why = new P.WhyPanel(L);
  panels.keys = new P.KeyNeurons(L);
  panels.drives = new P.DrivesPanel();
  panels.learning = new P.LearningPanel(L);
  panels.checks = new P.ChecksPanel(L);
  panels.scenarios = new P.ScenariosPanel(L);
  panels.lab = new P.LabPanel(L);
  panels.paths = new P.PathwayPanel(brain, panels.lab);
  panels.events = new P.EventsPanel();
  panels.recording = new P.RecordingPanel();
  panels.model = new P.ModelPanel(L);
  buildDialogs();
  setTool("lure");
  connect();
}

function buildToolbar() {
  const box = $("odourTools"); box.innerHTML = "";
  for (const o of L.odours || []) {
    const b = el("button", "tool", `<i class="dot" style="background:${o.colour};color:${o.colour}"></i>${esc(o.id)}`);
    b.dataset.tool = o.id; b.title = `${o.name} (innately ${o.innate}): ${o.note}`;
    box.appendChild(b); toolOrder.push(o.id);
    HINTS[o.id] = `Click to place ${o.name} (innately ${o.innate}). Shift-click adds sugar, Alt-click bitter. Turn the wind on to blow a plume.`;
  }
  document.querySelectorAll(".tool[data-tool]").forEach((b) => {
    const k = toolOrder.indexOf(b.dataset.tool);
    if (k >= 0 && k < 10) b.insertAdjacentHTML("beforeend", ` <span class="k">${(k + 1) % 10}</span>`);
    b.onclick = () => setTool(b.dataset.tool);
  });
  $("odourFood").querySelectorAll("button").forEach((b) => (b.onclick = () => {
    odourFood = b.dataset.food; $("odourFood").querySelectorAll("button").forEach((x) => x.classList.toggle("on", x === b));
  }));
}
function setTool(t) {
  tool = t; post({ type: "tool", tool: t });
  document.querySelectorAll(".tool[data-tool]").forEach((b) => b.classList.toggle("on", b.dataset.tool === t));
  setText($("hint"), HINTS[t] || "");
  setShown($("odourFood"), !!(L && L.odours.some((o) => o.id === t)));
}

// ---------------------------------------------------------------- live data: SSE with polling fallback
let es = null, polling = false, pollTimer = 0;
function connect() {
  try { es = new EventSource("api/stream"); } catch (e) { startPolling(); return; }
  es.onmessage = (e) => {
    let s; try { s = JSON.parse(e.data); } catch (err) { return; }
    if (polling) stopPolling();
    gotState(s);
  };
  es.onerror = () => { if (!polling) startPolling(); };
}
function startPolling() { polling = true; pollLoop(); }
function stopPolling() { polling = false; clearTimeout(pollTimer); }
async function pollLoop() {
  if (!polling) return;
  try { const r = await fetch("api/state", { cache: "no-store" }); const s = await r.json(); if (s && s.seq) gotState(s); } catch (e) { /* banner handles it */ }
  if (polling) pollTimer = setTimeout(pollLoop, 30);
}
function gotState(s) {
  if (!s || !s.seq || !s.fly) return;
  if (s.seq < lastSeq - 200) { location.reload(); return; }        // the server was restarted: its layout may differ
  if (s.seq === lastSeq) return;
  lastSeq = s.seq;
  onState(s);
}

// ---------------------------------------------------------------- per-tick UI update (kept cheap: text and widths only when they change)
function onState(s) {
  const now = performance.now();
  if (S) Sgap = Math.min(200, Math.max(15, 0.7 * Sgap + 0.3 * (now - Stime)));
  Sprev = S; S = s; Stime = now; lastStateAt = now;
  if (!firstState) { firstState = true; setShown($("loading"), false); }
  brain.setSpikes(s.spikes, now / 1000);
  setText($("stSps"), (s.sps || 0).toLocaleString());
  setText($("stRtf"), s.paused ? "paused" : (s.rtf >= 0.97 ? "real time" : fmt(s.rtf, 2) + "×") + (s.speed !== 1 ? ` (×${fmt(s.speed, 2)})` : ""));
  setText($("stT"), fmt(s.t, 1));
  slowHint(s);
  for (const k in panels) panels[k].update(s);
  if (retinaOn) retina.update(s.retina, s.senses);
  syncControls(s);
  toast(s.msg);
}
function syncControls(s) {
  const ap = $("autopilot"); if (document.activeElement !== ap) ap.checked = !!s.autopilot;
  setText($("pauseBtn"), s.paused ? "Resume" : "Pause");
  const fem = $("femaleToggle"); if (document.activeElement !== fem) fem.checked = !!(s.world && s.world.female);
  if (!speedDrag && !held("speed")) { const sp = $("speed"); if (Math.abs(parseFloat(sp.value) - s.speed) > 0.01) sp.value = s.speed; setText($("speedVal"), fmt(s.speed, 2) + "×"); }
  const w = s.world && s.world.wind;
  const st = s.world && s.world.stripes;
  if (st && !drumDrag && !held("drum")) {
    const cnt = $("stripesCount"), sp = $("drumSpeed");
    if (parseInt(cnt.value) !== st.count) {
      if (![...cnt.options].some((o) => parseInt(o.value) === st.count)) cnt.add(new Option(String(st.count), String(st.count)));
      cnt.value = String(st.count);
    }
    if (Math.abs(parseFloat(sp.value) - st.drum_speed) > 0.05) sp.value = st.drum_speed;
    setText($("drumVal"), Math.abs(st.drum_speed) < 0.05 ? "still" : `${fmt(st.drum_speed, 1)} rad/s`);
  }
  if (w && !windDrag && !held("wind")) { wind.angle = w.angle; wind.speed = w.speed; const ws = $("windSpeed"); if (Math.abs(parseFloat(ws.value) - w.speed) > 0.5) ws.value = w.speed; drawWindDial(); setText($("windVal"), w.speed > 0 ? `${Math.round(w.speed)} mm/s` : "off"); }
}
let slowSince = 0;
function slowHint(s) {
  if (s.paused || s.rtf >= 0.6) { slowSince = 0; if ($("hint").__slow) { $("hint").__slow = false; setText($("hint"), HINTS[tool] || ""); } return; }
  if (!slowSince) slowSince = performance.now();
  if (performance.now() - slowSince > 3000) { $("hint").__slow = true; setText($("hint"), `Your computer is running the brain at ${fmt(s.rtf, 2)}× real time, so the fly's world is in slow motion to keep up. Closing other programs helps.`); }
}
let toastText = "";
function toast(msg) {
  if (msg && msg !== toastText) { $("toast").textContent = msg; $("toast").classList.add("show"); }
  if (!msg && toastText) $("toast").classList.remove("show");
  toastText = msg || "";
}

// ---------------------------------------------------------------- poses, interpolated between ticks
function flyPose() {
  const f = S.fly, walking = f.mode === "walk" || f.mode === "backward" || f.mode === "court" || Math.abs(f.v) > 0.5;
  if (!Sprev) return { ...f, walking };
  const t = Math.min(1, (performance.now() - Stime) / Sgap), p = Sprev.fly;
  return { ...f, walking, x: p.x + (f.x - p.x) * t, y: p.y + (f.y - p.y) * t, h: lerpAngle(p.h, f.h, t) };
}
function femalePose() {
  const f = S.world && S.world.female; if (!f) return null;
  const p = Sprev && Sprev.world && Sprev.world.female;
  if (!p) return { ...f, walking: false };
  const t = Math.min(1, (performance.now() - Stime) / Sgap);
  return { ...f, walking: Math.abs(f.legs - p.legs) > 0.01, x: p.x + (f.x - p.x) * t, y: p.y + (f.y - p.y) * t, h: lerpAngle(p.h, f.h, t) };
}
let handAng = 0;
function serverHandAngle() {
  const h = S.world && S.world.hand, p = Sprev && Sprev.world && Sprev.world.hand;
  if (h && p && Math.hypot(h[0] - p[0], h[1] - p[1]) > 0.05) handAng = Math.atan2(-(h[1] - p[1]), h[0] - p[0]);
  return handAng;
}

// ---------------------------------------------------------------- arena input
const arenaEl = $("arena");
function pointerAt(e) {
  const r = arenaEl.getBoundingClientRect(), px = e.clientX - r.left, py = e.clientY - r.top;
  const [x, y] = arena.C2W(px, py);
  return { px, py, x, y, inside: Math.hypot(x, y) < L.arena_r };
}
arenaEl.addEventListener("pointermove", (e) => {
  if (!arena) return;
  const p = pointerAt(e), now = performance.now();
  if (pointer && pointer.t) {
    const d = Math.hypot(p.x - pointer.x, p.y - pointer.y), dt = Math.max(1, now - pointer.t) / 1000;
    p.speed = 0.6 * (pointer.speed || 0) + 0.4 * (d / dt);
    p.ang = d > 0.05 ? Math.atan2(p.py - pointer.py, p.px - pointer.px) : pointer.ang;
  }
  p.t = now; p.food = odourFood || (e.shiftKey ? "sugar" : e.altKey ? "bitter" : ""); pointer = p;
  if (now - lastSent > 20) {
    lastSent = now;
    post(p.inside && (tool === "lure" || tool === "hand") ? { type: "hand", x: p.x, y: p.y } : { type: "hand_off" });
  }
});
arenaEl.addEventListener("pointerleave", () => { pointer = null; post({ type: "hand_off" }); });
arenaEl.addEventListener("pointerdown", (e) => {
  if (!arena) return;
  const p = pointerAt(e);
  if (!p.inside) return;
  if (tool === "sugar" || tool === "bitter" || tool === "water") post({ type: "drop", kind: tool, x: p.x, y: p.y });
  else if (tool === "dust") { arena.puff(p.x, p.y); post({ type: "dust", x: p.x, y: p.y }); }
  else if (tool === "shock") { arena.shock(); post({ type: "shock" }); }
  else if (tool === "post") post({ type: "drop", kind: "post", x: p.x, y: p.y });
  else if (L.odours.some((o) => o.id === tool)) {
    const food = odourFood || (e.shiftKey ? "sugar" : e.altKey ? "bitter" : "");
    post(food ? { type: "drop", kind: tool, x: p.x, y: p.y, food } : { type: "drop", kind: tool, x: p.x, y: p.y });
  }
});

// ---------------------------------------------------------------- controls
let speedDrag = false, windDrag = false, retinaOn = true;
const wind = { angle: Math.PI, speed: 0 };
const holds = {};                                    // control name -> time until which the state sync leaves it alone
const hold = (name, ms = 800) => (holds[name] = performance.now() + ms);
const held = (name) => (holds[name] || 0) > performance.now();
$("autopilot").onchange = (e) => post({ type: "autopilot", on: e.target.checked });
$("learnToggle").onchange = (e) => post({ type: "learning", on: e.target.checked });
$("femaleToggle").onchange = (e) => post({ type: "female", on: e.target.checked });
$("pauseBtn").onclick = () => post({ type: "pause", on: !(S && S.paused) });
$("resetBtn").onclick = () => post({ type: "reset" });
$("clearBtn").onclick = (e) => { e.stopPropagation(); $("clearMenu").hidden = !$("clearMenu").hidden; };
$("clearMenu").querySelectorAll("button").forEach((b) => (b.onclick = () => { post({ type: "clear", what: b.dataset.what }); $("clearMenu").hidden = true; }));
document.addEventListener("click", (e) => { if (!$("clearMenu").hidden && !e.target.closest(".menuwrap")) $("clearMenu").hidden = true; });
{
  const sp = $("speed"); let t = 0;
  sp.addEventListener("pointerdown", () => (speedDrag = true));
  sp.addEventListener("input", () => { hold("speed"); const v = parseFloat(sp.value); setText($("speedVal"), fmt(v, 2) + "×"); clearTimeout(t); t = setTimeout(() => post({ type: "speed", value: v }), 80); });
  sp.addEventListener("change", () => { hold("speed"); clearTimeout(t); post({ type: "speed", value: parseFloat(sp.value) }); speedDrag = false; });
  window.addEventListener("pointerup", () => { speedDrag = false; windDrag = false; });
}
$("seesToggle").onchange = (e) => (sees = e.target.checked);
function clap() { post({ type: "sound" }); if (arena) arena.clap(); }
$("clapBtn").onclick = clap;
// the optomotor drum: stripes on the wall, spinning
let drumDrag = false;
{
  const cnt = $("stripesCount"), sp = $("drumSpeed"); let t = 0;
  const send = (now) => { hold("drum"); const a = { type: "stripes", count: parseInt(cnt.value), drum_speed: parseFloat(sp.value) }; clearTimeout(t); if (now) post(a); else t = setTimeout(() => post(a), 60); };
  const label = () => setText($("drumVal"), Math.abs(parseFloat(sp.value)) < 0.05 ? "still" : `${fmt(parseFloat(sp.value), 1)} rad/s`);
  cnt.onchange = () => { if (parseInt(cnt.value) > 0 && Math.abs(parseFloat(sp.value)) < 0.05) { sp.value = 1; label(); } send(true); };
  sp.addEventListener("pointerdown", () => (drumDrag = true));
  sp.addEventListener("input", () => { if (parseInt(cnt.value) === 0 && Math.abs(parseFloat(sp.value)) >= 0.05) cnt.value = "16"; label(); send(false); });
  sp.addEventListener("change", () => { drumDrag = false; send(true); });
}
function stripesNow() {
  const st = S.world && S.world.stripes; if (!st || !st.count) return null;
  const p = Sprev && Sprev.world && Sprev.world.stripes;
  if (!p || !p.count) return st;
  return { count: st.count, phase: lerpAngle(p.phase, st.phase, Math.min(1, (performance.now() - Stime) / Sgap)) };
}
$("retinaToggle").onchange = (e) => { retinaOn = e.target.checked; setShown($("retinaBox"), retinaOn); };

// wind dial: a compass showing the direction the wind blows toward
const dial = $("windDial"), dctx = dial.getContext("2d");
{
  const dpr = window.devicePixelRatio || 1; dial.width = dial.height = Math.round(36 * dpr);
  let t = 0;
  const sendWind = (now) => { hold("wind"); const a = { type: "wind", angle: wind.angle, speed: wind.speed }; clearTimeout(t); if (now) post(a); else t = setTimeout(() => post(a), 60); };
  const setFrom = (e) => {
    const r = dial.getBoundingClientRect(), dx = e.clientX - r.left - r.width / 2, dy = e.clientY - r.top - r.height / 2;
    if (Math.hypot(dx, dy) < 3) return;
    wind.angle = Math.atan2(-dy, dx);
    if (wind.speed <= 0) { wind.speed = 15; $("windSpeed").value = 15; }
    setText($("windVal"), `${Math.round(wind.speed)} mm/s`); drawWindDial(); sendWind(false);
  };
  dial.addEventListener("pointerdown", (e) => { windDrag = true; dial.setPointerCapture(e.pointerId); setFrom(e); });
  dial.addEventListener("pointermove", (e) => { if (windDrag) setFrom(e); });
  dial.addEventListener("pointerup", () => { windDrag = false; sendWind(true); });
  const ws = $("windSpeed");
  ws.addEventListener("pointerdown", () => (windDrag = true));
  ws.addEventListener("input", () => { wind.speed = parseFloat(ws.value); setText($("windVal"), wind.speed > 0 ? `${Math.round(wind.speed)} mm/s` : "off"); drawWindDial(); sendWind(false); });
  ws.addEventListener("change", () => { windDrag = false; wind.speed = parseFloat(ws.value); sendWind(true); });
}
function drawWindDial() {
  const c = dctx, dpr = window.devicePixelRatio || 1, r = 18, on = wind.speed > 0;
  c.setTransform(dpr, 0, 0, dpr, 0, 0); c.clearRect(0, 0, 36, 36);
  c.strokeStyle = "#223042"; c.lineWidth = 1;
  for (let k = 0; k < 4; k++) { const a = k * Math.PI / 2; c.beginPath(); c.moveTo(r + Math.cos(a) * 13, r + Math.sin(a) * 13); c.lineTo(r + Math.cos(a) * 16, r + Math.sin(a) * 16); c.stroke(); }
  c.save(); c.translate(r, r); c.rotate(-wind.angle);
  const len = on ? 8 + Math.min(6, wind.speed * 0.15) : 8;
  c.strokeStyle = on ? "#8fb8ff" : "#8b98a9"; c.lineWidth = 2; c.lineCap = "round";
  c.beginPath(); c.moveTo(-len, 0); c.lineTo(len, 0); c.moveTo(len - 4, -3.5); c.lineTo(len, 0); c.lineTo(len - 4, 3.5); c.stroke();
  c.restore();
}
drawWindDial();

// ---------------------------------------------------------------- brain: view buttons and the neuron popover
function wireBrain() {
  const vb = $("brainView"), sb = $("brainSpin");
  const refresh = () => { setText(vb, brain.mode3d ? "3-D" : "front"); setClass(vb, "on", brain.mode3d); setClass(sb, "on", brain.spin); sb.disabled = !brain.gl; };
  vb.onclick = () => { brain.setView(!brain.mode3d); refresh(); };
  sb.onclick = () => { brain.spin = !brain.spin; if (brain.spin) brain.setView(true); refresh(); };
  $("brainReset").onclick = () => { brain.resetView(); refresh(); };
  if (!brain.gl) setText($("brainHint"), "WebGL is unavailable here, so this is the flat map. Click a dot to look up that neuron.");
  brain.spin = true; brain.pitch = 0.3; refresh();
  const pop = $("neuronPop");
  brain.onPick = async (i, px, py) => {
    if (i < 0) { setShown(pop, false); return; }
    const wrap = $("brainCard").querySelector(".brainwrap").getBoundingClientRect();
    pop.style.left = Math.max(4, Math.min(wrap.width - 268, px > wrap.width / 2 ? px - 272 : px + 12)) + "px";
    pop.style.top = Math.max(4, Math.min(wrap.height - 40, py - 20)) + "px";
    pop.innerHTML = `<button class="close">✕</button><h5>neuron #${i}</h5><div class="feedback">looking it up…</div>`;
    setShown(pop, true);
    pop.querySelector(".close").onclick = () => { setShown(pop, false); brain.picked = -1; };
    const r = await getJSON(`api/neuron?index=${i}`);
    if (!r || !r.ok) { pop.querySelector(".feedback").textContent = (r && r.error) || "no answer"; return; }
    const n = r.neuron, spec = n.side ? `${n.type}/${n.side}` : n.type;
    const list = (rows) => rows.slice(0, 4).map((p) => `<li><b>${esc(p.type)}${p.side ? "/" + p.side : ""}</b> <span class="${p.sign > 0 ? "pos" : "neg"}">${p.sign > 0 ? "+" : "−"}</span> ${p.synapses} syn · ${p.neurons} cell${p.neurons === 1 ? "" : "s"}</li>`).join("") || "<li>none</li>";
    pop.innerHTML = `<button class="close">✕</button>
      <h5>${esc(n.type || "(unannotated)")}${n.side ? " / " + esc(n.side) : ""} <small style="color:var(--muted)">#${n.index}</small></h5>
      <div class="kv">
        <span class="k">class</span><span class="v">${esc([n.superclass, n.class, n.subclass].filter(Boolean).join(" · ") || "–")}</span>
        <span class="k">transmitter</span><span class="v">${esc(n.nt || "?")} (${n.sign > 0 ? "excitatory" : "inhibitory"})</span>
        <span class="k">connections</span><span class="v">${n.n_inputs} in · ${n.n_outputs} out</span>
        <span class="k">firing now</span><span class="v">${fmt(n.rate_hz, 1)} Hz</span>
        <span class="k">region</span><span class="v">${esc(L.regions[L.region[i]] || "")}</span>
      </div>
      <b>strongest inputs</b><ul>${list(n.inputs || [])}</ul>
      <b>strongest outputs</b><ul>${list(n.outputs || [])}</ul>
      <div class="row wrap"><a href="https://neuprint.janelia.org/view?bodyid=${n.body_id}&dataset=male-cns%3Av1.0" target="_blank" rel="noopener">neuPrint ↗</a>
        <button class="mini" data-act="lab">to the lab</button><button class="mini" data-act="watch">watch ${esc(spec)}</button></div>`;
    pop.querySelector(".close").onclick = () => { setShown(pop, false); brain.picked = -1; };
    pop.querySelector("[data-act=lab]").onclick = () => { $("spec").value = spec; $("spec").focus(); };
    pop.querySelector("[data-act=watch]").onclick = () => post({ type: "watch", spec });
  };
}

// ---------------------------------------------------------------- dialogs and keyboard
function buildDialogs() {
  const wr = L.whats_real || {};
  $("realWiring").innerHTML = (wr.wiring || []).map((t) => `<li>${esc(t)}</li>`).join("");
  $("realHand").innerHTML = (wr.hand_built || []).map((t) => `<li>${esc(t)}</li>`).join("");
  $("realNot").innerHTML = (wr.not_modelled || []).map((t) => `<li>${esc(t)}</li>`).join("");
}
$("realBtn").onclick = () => $("realDlg").showModal();
$("keysBtn").onclick = () => $("keysDlg").showModal();
document.querySelectorAll("[data-close]").forEach((b) => (b.onclick = () => b.closest("dialog").close()));
window.addEventListener("keydown", (e) => {
  const tag = e.target.tagName;
  if (tag === "INPUT" || tag === "SELECT" || tag === "TEXTAREA" || e.ctrlKey || e.metaKey) return;
  if (!L) return;
  if (/^[0-9]$/.test(e.key)) { const k = e.key === "0" ? 9 : parseInt(e.key) - 1; if (toolOrder[k]) setTool(toolOrder[k]); return; }
  switch (e.key) {
    case " ": e.preventDefault(); post({ type: "pause", on: !(S && S.paused) }); break;
    case "f": case "F": post({ type: "female", on: !(S && S.world && S.world.female) }); break;
    case "c": case "C": clap(); break;
    case "s": case "S": $("seesToggle").checked = sees = !sees; break;
    case "e": case "E": $("retinaToggle").checked = retinaOn = !retinaOn; setShown($("retinaBox"), retinaOn); break;
    case "w": case "W": post({ type: "autopilot", on: !(S && S.autopilot) }); break;
    case "n": case "N": post({ type: "reset" }); break;
    case "Escape": setShown($("neuronPop"), false); if (brain) brain.picked = -1; $("clearMenu").hidden = true; break;
  }
});

// ---------------------------------------------------------------- main loop
let last = performance.now();
function frame(now) {
  const dt = Math.min(0.1, (now - last) / 1000); last = now;
  if (L && arena) {
    const pose = S ? flyPose() : null, female = S ? femalePose() : null;
    arena.draw(dt, { S, pose, female, pointer, tool, sees, handAng: S ? serverHandAngle() : 0, stripes: S ? stripesNow() : null });
    brain.frame(dt, now / 1000);
    panels.keys.drawSparks(now);
    setShown($("disc"), lastStateAt > 0 && now - lastStateAt > 3000);
  }
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);
loadLayout();
