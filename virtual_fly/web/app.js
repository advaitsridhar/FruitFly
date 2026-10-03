// app.js — Virtual Fly: the page that shows a simulated whole fly nervous system driving a fly in a dish.
// Data comes from the Python server (see docs/API.md): one layout fetch, then a stream of state ticks.
"use strict";
import { $, setText, setClass, setShown, esc, fmt, post, getJSON, el, setActionFly } from "./util.js";
import { Arena, lerpAngle } from "./arena.js";
import { BrainView, REGION_COLORS } from "./brain3d.js";
import { RetinaView } from "./retina.js";
import * as P from "./panels.js";
import { PanelManager } from "./layout.js";

// ---------------------------------------------------------------- state
let L = null;                       // layout (brain map, arena size, readouts, odours ...)
let S = null, Sprev = null, Stime = 0, Sgap = 25, lastSeq = 0, lastStateAt = 0, firstState = false;
let tool = "lure", odourFood = "", pointer = null, lastSent = 0, sees = false;
let arena = null, brain = null, retina = null, layout = null;
// the 3-D dish (arena3d.js, docs/TWO_FLIES_PLAN.md 7.4): built on the first press of the 3-D button, which is when three.js
// and the model are fetched; "off" | "loading" | "on" | "unavailable" (no WebGL2, or the model did not load)
let view3d = null, view3dState = "off", view3dReason = "";
/** The dish that is showing: the 3-D one while it is on, else the 2-D canvas (both answer the same calls). */
function activeArena() { return view3dState === "on" && view3d ? view3d : arena; }
const panels = {};
// two flies (docs/API.md "Two flies"): the state then lists `flies`; `focus` is the one the panels, the eye inset, the brain
// map and the camera follow; each fly has its own layout (api/layout?fly=k) and its own brain map, kept alive side by side
let focus = 0, pairReady = false;
const layouts = {}, brains = [];
const ZOOMS = [1, 1.5, 2, 3, 4];
let toolOrder = ["lure", "hand", "sugar", "bitter", "water", "dust", "shock", "post"];
const HINTS = {
  lure: "Wiggle the lure slowly beside the fly: it turns toward small moving things (a courtship-chase circuit).",
  hand: "Swoop the hand straight at the fly, fast. A slow hand doesn't scare it.",
  sugar: "Click just in front of the fly's head to drop sugar. A hungry fly eats more eagerly, in bouts of a few seconds.",
  bitter: "Click to drop bitter food. Try it right next to sugar.",
  water: "Click to drop water: a thirsty fly tastes it, but in this model its water cells do not reach MN9, so it does not drink.",
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
  setText($("sub"), `${L.n.toLocaleString()} neurons · ${(L.edges / 1e6).toFixed(1)} M connections · ${(L.synapses / 1e6).toFixed(0)} M synapses · ${L.sex === "female" ? "FlyWire 783 (female)" : "MaleCNS v1.0"}`);
  buildToolbar();
  arena = new Arena($("arena"), $("stage"), L);
  new ResizeObserver(() => { arena.resize(); if (view3d) view3d.resize(); }).observe($("stage"));
  retina = new RetinaView($("retina"), L.retina || {});
  layouts[0] = L; L.fly_id = 0;
  brain = brains[0] = new BrainView($("brain"), $("brainOverlay"), L);
  setText($("brainCount"), `${brain.m.toLocaleString()} somas`);
  setLegend(L);
  setText($("genomeSyn"), `${Math.round(L.synapses / 1e6)} million`);
  wireBrain();
  attachPick(brain);
  $("focusSel").onchange = (e) => { setFocus(parseInt(e.target.value) || 0); e.target.blur(); };   // (keys are ignored while a menu has the focus)
  panels.why = new P.WhyPanel(L);
  panels.keys = new P.KeyNeurons(L);
  panels.drives = new P.DrivesPanel();
  panels.learning = new P.LearningPanel(L);
  panels.checks = new P.ChecksPanel(L);
  panels.scenarios = new P.ScenariosPanel(L);
  panels.lab = new P.LabPanel(L);
  panels.paths = new P.PathwayPanel(brain, panels.lab);
  panels.genetics = new P.GeneticsPanel(L, panels.lab);
  panels.genome = new P.GenomePanel(L);
  panels.events = new P.EventsPanel();
  panels.recording = new P.RecordingPanel();
  panels.model = new P.ModelPanel(L);
  layout = new PanelManager($("aside"), $("panelsBtn"), $("panelMenu"));
  buildDialogs();
  setTool("lure");
  connect();
}

function setLegend(Lk) {
  const has = new Set(Lk.region);                    // only the regions this fly has (the female fly has no nerve cord)
  $("legend").innerHTML = Lk.regions.map((r, i) => has.has(i) ? `<span><i style="background:${REGION_COLORS[i]}"></i>${esc(r)}</span>` : "").join("");
}

// ---------------------------------------------------------------- two flies: the second fly's layout and map, the focus
async function setupPair(s) {
  pairReady = true;
  for (const f of s.flies) {
    if (layouts[f.id]) continue;
    let Lk = null;
    while (!Lk) {
      try { const r = await fetch(`api/layout?fly=${f.id}`); if (!r.ok) throw new Error(r.status); Lk = await r.json(); }
      catch (e) { await new Promise((res) => setTimeout(res, 1000)); }
    }
    Lk.fly_id = f.id; layouts[f.id] = Lk;
    if (f.id === 1 && !brains[1]) brains[1] = new BrainView($("brain2"), $("brainOverlay2"), Lk);   // kept alive from now on
    if (brains[f.id]) attachPick(brains[f.id]);
  }
  const sel = $("focusSel"); sel.innerHTML = "";
  for (const f of s.flies) {
    const o = document.createElement("option"); o.value = String(f.id);
    o.textContent = `${f.sex === "female" ? "♀" : "♂"} ${f.sex} (fly ${f.id}${f.id === 0 ? ", the one you play" : ", the partner"})`;
    sel.appendChild(o);
  }
  sel.value = String(focus); setShown(sel, true);
  // the female toggle: the scripted female is single-fly play's; here she is simulated and always in the dish
  const fem = $("femaleToggle"); fem.checked = true; fem.disabled = true;
  setText($("femaleText"), "♀ female: simulated");
  $("femaleLabel").title = "A simulated female with a brain of her own is in the dish (--partner); the scripted female is for single-fly play";
  setShown($("cuesNote"), s.flies.some((f) => f.sex === "female"));       // her abdomen marks are hand-built cues: say so under the dish
  const names = s.flies.map((f) => `${f.sex} (${layouts[f.id] ? layouts[f.id].dataset : "…"})`).join(" and ");
  setText($("sub"), `two flies: ${names} · ${(L.n + (layouts[1] ? layouts[1].n : 0)).toLocaleString()} neurons in two brains`);
  buildDialogs();
}
function setFocus(k) {
  if (!layouts[k] || !brains[k]) return;
  focus = k; setActionFly(k);
  const Lk = layouts[k];
  brains.forEach((b, i) => { if (!b) return; setShown(b.canvas, i === k); setShown(b.overlay, i === k); });
  brain = brains[k]; brain._resize(); wireBrain();
  setText($("brainCount"), `${brain.m.toLocaleString()} somas`);
  setLegend(Lk);
  // the panels built from the layout are rebuilt for this fly; the ones that only read the state keep their DOM
  panels.why = new P.WhyPanel(Lk);
  panels.keys = new P.KeyNeurons(Lk);
  panels.checks = new P.ChecksPanel(Lk);
  panels.lab.relayout(Lk);
  panels.paths.refocus(brain);
  panels.genetics.relayout(Lk);
  panels.genome = new P.GenomePanel(Lk);
  panels.model = new P.ModelPanel(Lk);
  setText($("genomeSyn"), `${Math.round(Lk.synapses / 1e6)} million`);
  if (S) { const V = viewOf(S); for (const key in panels) panels[key].update(V); panels.why.update(V); }
  $("focusSel").value = String(k);
}
/** The state as the focused fly sees it: with two flies, fly k's entry laid over the shared fields (docs/API.md). */
function viewOf(s) { return focus && s.flies && s.flies[focus] ? { ...s, ...s.flies[focus] } : s; }

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

// ---------------------------------------------------------------- per-tick ingestion (cheap) and per-frame rendering
// The server sends 40 ticks a second. Every tick is taken in (spike flashes, sparkline history,
// the event log), but the page is written only once per animation frame, from the latest tick:
// when a machine draws 30 frames a second it no longer tries to lay the panels out 40 times.
let renderPending = false;
function onState(s) {
  const now = performance.now();
  if (S) Sgap = Math.min(200, Math.max(15, 0.7 * Sgap + 0.3 * (now - Stime)));
  Sprev = S; S = s; Stime = now; lastStateAt = now;
  if (s.flies && !pairReady) setupPair(s);
  const V = viewOf(s);
  if (s.flies) for (const f of s.flies) { if (brains[f.id]) brains[f.id].setSpikes(f.spikes, now / 1000); }
  else brain.setSpikes(s.spikes, now / 1000);
  panels.keys.ingest(V);
  panels.events.ingest(s);
  renderPending = true;
}
function renderState(s0) {
  if (!firstState) { firstState = true; setShown($("loading"), false); }
  const s = viewOf(s0);                              // the focused fly's numbers over the shared ones
  setText($("stSps"), (s.sps || 0).toLocaleString());
  const partsOn = !!(s.genome && s.genome.parts && s.genome.parts.on);
  const graded = Math.min(s.graded_eps || 0, s.sps || 0);   // the parts list's graded cells release quanta, not spikes
  setText($("stSpsUnit"), partsOn ? "events/s" : "spikes/s");
  const tip = partsOn ? `events per second in the whole brain: ${((s.sps || 0) - graded).toLocaleString()} spikes and ` +
    `${graded.toLocaleString()} release quanta of the graded cells (parts list)` : "spikes per second in the whole brain";
  if ($("stSpsBox").title !== tip) $("stSpsBox").title = tip;
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
  const fem = $("femaleToggle"); if (!s.flies && document.activeElement !== fem) fem.checked = !!(s.world && s.world.female);
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
let slowSince = 0, bodyNoted = false;
const PHYSICS_PACE = "The physics body (NeuroMechFly's legs in MuJoCo) runs at about a tenth of real time, so the fly's world is in slow motion: that is the body's pace, not your computer's.";
const PAIR_PACE = "Both flies are physics bodies (NeuroMechFly in one MuJoCo world, drawn at real size), which runs well under real time, about a tenth with two flies: that is the bodies' pace, not your computer's.";
function slowHint(s) {
  if (s.paused || s.rtf >= 0.6) { slowSince = 0; if ($("hint").__slow) { $("hint").__slow = false; setText($("hint"), HINTS[tool] || ""); } return; }
  if (!slowSince) slowSince = performance.now();
  if (performance.now() - slowSince <= 3000) return;
  if (s.fly && s.fly.physics) {                     // expected with this body: said once, then the line is the tools' again
    const pace = s.fly.physics.pair ? PAIR_PACE : PHYSICS_PACE;
    if (!bodyNoted) { bodyNoted = true; setText($("hint"), pace); $("stRtf").parentElement.title = pace; }
    return;
  }
  $("hint").__slow = true; setText($("hint"), `Your computer is running the brain at ${fmt(s.rtf, 2)}× real time, so the fly's world is in slow motion to keep up. Closing other programs helps.`);
}
let toastText = "", toastHold = 0;
/** Show the server's message (sent with every tick), or, with `holdMs`, a page-local one that the next ticks'
 *  empty `msg` leaves in place until the hold runs out (the server's own message still replaces it). */
function toast(msg, holdMs = 0) {
  const now = performance.now();
  if (holdMs) toastHold = now + holdMs;
  else if (!msg && now < toastHold) return;
  if (msg && msg !== toastText) { $("toast").textContent = msg; $("toast").classList.add("show"); }
  if (!msg && toastText) $("toast").classList.remove("show");
  toastText = msg || "";
}

// ---------------------------------------------------------------- poses, interpolated between ticks
function onePose(f, p, sex, id, hz, senses) {
  const walking = f.mode === "walk" || f.mode === "backward" || f.mode === "court" || Math.abs(f.v) > 0.5;
  if (!p) return { ...f, walking, sex, id, hz, senses };
  const t = Math.min(1, (performance.now() - Stime) / Sgap);
  return { ...f, walking, sex, id, hz, senses, x: p.x + (f.x - p.x) * t, y: p.y + (f.y - p.y) * t, h: lerpAngle(p.h, f.h, t) };
}
/** Every simulated fly's pose this frame (one with a single fly; each of `S.flies` with two), by fly id. */
function posesOf(s) {
  if (!s.flies) return [onePose(s.fly, Sprev && Sprev.fly, L.sex, 0, s.hz, s.senses)];
  return s.flies.map((f) => onePose(f.fly, Sprev && Sprev.flies && Sprev.flies[f.id] && Sprev.flies[f.id].fly, f.sex, f.id, f.hz, f.senses));
}
function scriptedFemalePose() {
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
const arenaEl = $("arena"), arena3dEl = $("arena3d");
function pointerAt(e) {
  const r = e.currentTarget.getBoundingClientRect(), px = e.clientX - r.left, py = e.clientY - r.top;
  const [x, y] = activeArena().C2W(px, py);
  return { px, py, x, y, inside: Math.hypot(x, y) < L.arena_r };
}
// the pointer works on whichever dish is showing: the same handlers on both canvases
const onDish = (type, fn, opts) => { arenaEl.addEventListener(type, fn, opts); arena3dEl.addEventListener(type, fn, opts); };
onDish("pointermove", (e) => {
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
onDish("pointerleave", () => { pointer = null; post({ type: "hand_off" }); });
// zoom: the wheel over the dish, the button, or + / -; above 1x the view follows the fly (in 3-D the wheel is the camera's)
function setZoom(z) {
  if (!arena) return;
  const zoom = activeArena().setZoom(z);
  setText($("zoomBtn"), `🔍 ${zoom % 1 ? zoom.toFixed(1) : zoom}×`);
  setClass($("zoomBtn"), "on", zoom > 1);
}
function zoomStep(dir) {
  if (!arena) return;
  const z = activeArena().zoom, next = dir > 0 ? ZOOMS.find((v) => v > z + 1e-6) : [...ZOOMS].reverse().find((v) => v < z - 1e-6);
  setZoom(next == null ? z : next);                     // already past the last step: stay there
}
// the wheel is claimed only when it changes the zoom, so a horizontal swipe or scrolling at 1x keeps working
arenaEl.addEventListener("wheel", (e) => {
  if (!arena || !e.deltaY) return;
  const z = Math.max(1, Math.min(4, arena.zoom * (e.deltaY < 0 ? 1.12 : 1 / 1.12)));
  if (z === arena.zoom) return;
  e.preventDefault(); setZoom(z);
}, { passive: false });
$("zoomBtn").onclick = () => { if (!arena) return; const z = activeArena().zoom, i = ZOOMS.findIndex((v) => v > z + 1e-6); setZoom(i < 0 ? 1 : ZOOMS[i]); };
onDish("pointerdown", (e) => {
  if (!arena) return;
  const p = pointerAt(e);
  if (!p.inside) return;
  if (tool === "sugar" || tool === "bitter" || tool === "water") post({ type: "drop", kind: tool, x: p.x, y: p.y });
  else if (tool === "dust") { activeArena().puff(p.x, p.y); post({ type: "dust", x: p.x, y: p.y }); }
  else if (tool === "shock") { activeArena().shock(); post({ type: "shock" }); }
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
function clap() { post({ type: "sound" }); if (arena) activeArena().clap(); }
$("clapBtn").onclick = clap;
// the optomotor drum: stripes on the wall, spinning
let drumDrag = false;
{
  const cnt = $("stripesCount"), sp = $("drumSpeed"); let t = 0;
  const send = (now) => { hold("drum"); const a = { type: "stripes", count: parseInt(cnt.value), drum_speed: parseFloat(sp.value) }; clearTimeout(t); if (now) post(a); else t = setTimeout(() => post(a), 60); };
  const label = () => setText($("drumVal"), Math.abs(parseFloat(sp.value)) < 0.05 ? "still" : `${fmt(parseFloat(sp.value), 1)} rad/s`);
  cnt.onchange = () => { if (parseInt(cnt.value) > 0 && Math.abs(parseFloat(sp.value)) < 0.05) { sp.value = 1; label(); } send(true); cnt.blur(); };
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

// ---------------------------------------------------------------- the 3-D dish (arena3d.js), loaded on the first press
function show3d(on) {
  view3dState = on ? "on" : "off";
  if (on) view3d.show(); else if (view3d) view3d.hide();
  setShown(arenaEl, !on); setShown($("badge3d"), on);
  setClass($("view3dBtn"), "on", on); setText($("view3dBtn"), on ? "🧊 2-D" : "🧊 3-D");
  if (!on) setShown($("msg3d"), false);
  setZoom(activeArena().zoom);                         // the button shows the dish's own zoom
}
async function toggle3d() {
  if (!L || !arena || view3dState === "loading") return;
  if (view3dState === "on") { show3d(false); return; }
  if (view3dState === "unavailable") { toast(view3dReason, 4000); return; }
  if (!view3d) {
    view3dState = "loading"; setText($("view3dBtn"), "🧊 loading…"); toast("Loading the 3-D view: three.js and the fly model…", 10000);
    try {
      const mod = await import("./arena3d.js");       // nothing of three.js or the model is fetched before this
      view3d = await mod.Arena3D.create($("arena3d"), $("stage"), L, { msg: $("msg3d"), fps: $("fps3d") });
    } catch (e) {
      view3dState = "unavailable"; view3dReason = `The 3-D view cannot run here: ${e && e.message ? e.message : e}`;
      console.warn(view3dReason, e);
      setText($("view3dBtn"), "🧊 3-D"); $("view3dBtn").title = view3dReason; toast(view3dReason, 6000);
      return;
    }
    $("cam3d").onchange = (e) => { view3d.setCamera(e.target.value); setZoom(view3d.zoom); e.target.blur(); };
  }
  show3d(true);
  toastHold = 0; toast("");                            // the loading message goes
}
$("view3dBtn").onclick = toggle3d;

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
  setText($("brainHint"), brain.gl ? "Drag to rotate, wheel to zoom, click a dot to look up that neuron."
    : "WebGL is unavailable here, so this is the flat map. Click a dot to look up that neuron.");
  if (brain.__wired) { refresh(); return; }
  brain.__wired = true; brain.spin = true; brain.pitch = 0.3; refresh();
}
/** The neuron popover of one brain map (each fly's map looks its neurons up in its own connectome: api/neuron?fly=k). */
function attachPick(view) {
  if (view.onPick) return;
  const pop = $("neuronPop"), Lk = view.L, fly = Lk.fly_id ? `&fly=${Lk.fly_id}` : "";
  view.onPick = async (i, px, py) => {
    if (i < 0) { setShown(pop, false); return; }
    const wrap = $("brainCard").querySelector(".brainwrap").getBoundingClientRect();
    pop.style.left = Math.max(4, Math.min(wrap.width - 268, px > wrap.width / 2 ? px - 272 : px + 12)) + "px";
    pop.style.top = Math.max(4, Math.min(wrap.height - 40, py - 20)) + "px";
    pop.innerHTML = `<button class="close">✕</button><h5>neuron #${i}</h5><div class="feedback">looking it up…</div>`;
    setShown(pop, true);
    pop.querySelector(".close").onclick = () => { setShown(pop, false); view.picked = -1; };
    const r = await getJSON(`api/neuron?index=${i}${fly}`);
    if (!r || !r.ok) { pop.querySelector(".feedback").textContent = (r && r.error) || "no answer"; return; }
    const n = r.neuron, spec = !n.type ? `index:${n.index}` : n.side ? `${n.type}/${n.side}` : n.type;   // an unannotated cell by itself
    const list = (rows) => rows.slice(0, 4).map((p) => `<li><b>${esc(p.type)}${p.side ? "/" + p.side : ""}</b> <span class="${p.sign > 0 ? "pos" : "neg"}">${p.sign > 0 ? "+" : "−"}</span> ${p.synapses} syn · ${p.neurons} cell${p.neurons === 1 ? "" : "s"}</li>`).join("") || "<li>none</li>";
    pop.innerHTML = `<button class="close">✕</button>
      <h5>${esc(n.type || "(unannotated)")}${n.side ? " / " + esc(n.side) : ""} <small style="color:var(--muted)">#${n.index}</small></h5>
      <div class="kv">
        <span class="k">class</span><span class="v">${esc([n.superclass, n.class, n.subclass].filter(Boolean).join(" · ") || "–")}</span>
        <span class="k">transmitter</span><span class="v">${esc(n.nt || "?")} (${n.sign > 0 ? "excitatory" : "inhibitory"})${partsRole(n.parts)}</span>
        ${vfbRows(n)}
        <span class="k">connections</span><span class="v">${n.n_inputs} in · ${n.n_outputs} out</span>
        <span class="k">firing now</span><span class="v">${fmt(n.rate_hz, 1)} Hz</span>
        <span class="k">region</span><span class="v">${esc(Lk.regions[Lk.region[i]] || "")}</span>
        <span class="k">genes</span><span class="v">${(n.genes || []).length ? n.genes.map((g) => `<a href="${g.flybase}" target="_blank" rel="noopener" title="${esc(g.why)} · FlyBase">${esc(g.symbol)}</a>`).join(", ") : "none known here"}${n.dimorphism ? ` · ${esc(n.dimorphism)}` : ""}</span>
        ${receptorRow(n.receptors)}
      </div>
      ${n.vfb && n.vfb.definition ? `<details><summary>what is this cell type?</summary><div class="def">${esc(n.vfb.definition)}</div></details>` : ""}
      <b>strongest inputs</b><ul>${list(n.inputs || [])}</ul>
      <b>strongest outputs</b><ul>${list(n.outputs || [])}</ul>
      <div class="row wrap">${Lk.sex === "female"
        ? `<span class="muted" title="FlyWire root id (release 783): search for it in FlyWire Codex">root id <code>${esc(n.body_ref || String(n.body_id))}</code></span> <a href="https://codex.flywire.ai" target="_blank" rel="noopener">FlyWire Codex ↗</a>`
        : `<a href="https://neuprint.janelia.org/view?bodyid=${esc(n.body_ref || String(n.body_id))}&dataset=male-cns%3Av1.0" target="_blank" rel="noopener">neuPrint ↗</a>`}
        ${n.vfb ? `<a href="${esc(n.vfb.url)}" target="_blank" rel="noopener" title="${esc(n.vfb.label)} on Virtual Fly Brain">VFB ↗</a>` : ""}
        <button class="mini" data-act="lab">to the lab</button><button class="mini" data-act="watch">watch ${esc(spec)}</button></div>`;
    pop.querySelector(".close").onclick = () => { setShown(pop, false); view.picked = -1; };
    pop.querySelector("[data-act=lab]").onclick = () => { $("spec").value = spec; $("spec").focus(); };
    pop.querySelector("[data-act=watch]").onclick = () => post({ type: "watch", spec });
    pop.querySelectorAll(".crumb").forEach((a) => (a.onclick = () => { $("spec").value = `fbbt:${a.dataset.fbbt}`; $("spec").focus(); }));
  };
}

// the popover's Virtual Fly Brain lines: the ontology class, its parents (click = select that class),
// the transmitter the literature asserts, the lineage and peptides, the receptors the type expresses
function partsRole(p) {
  if (!p) return "";
  const bits = [];
  if (p.modulator) bits.push(`${p.modulator} tone${p.keep_fast ? " + fast synapses" : ""}`);
  if (p.curated) bits.push(`the literature says ${esc(p.curated.curated.join(" + "))}: ${esc(p.curated.action)}`);
  else if (p.sign !== undefined && p.modulator === null) bits.push(`fast ${p.sign > 0 ? "+" : "−"}`);
  if (p.graded) bits.push("graded");
  if (p.local) bits.push("releases locally, following the Kenyon cells around each target");
  if (p.receptor_fact) bits.push(`<span title="${esc(p.receptor_fact.why)}">from the literature: ${esc(p.receptor_fact.what || p.receptor_fact.receptors.join(", "))}</span>`);
  return bits.length ? ` <small class="muted">· parts list: ${bits.join("; ")}</small>` : "";
}
function vfbRows(n) {
  const v = n.vfb;
  if (!v) return L.vfb && L.vfb.available ? `<span class="k">ontology</span><span class="v muted">no FBbt class matched this type</span>` : "";
  const crumbs = (v.breadcrumb || []).slice(0, 3).map((b) => `<a class="crumb" data-fbbt="${esc(b.fbbt)}" title="select every ${esc(b.label)} (fbbt:${esc(b.fbbt)})">${esc(b.label)}</a>`).join(" › ");
  let out = `<span class="k">ontology</span><span class="v"><a href="${esc(v.url)}" target="_blank" rel="noopener" title="${esc(v.fbbt[0])} on Virtual Fly Brain">${esc(v.label)}</a>${v.coarse ? ` <small class="muted">(class of ${v.shared_by || "several"} types)</small>` : v.shared_by ? ` <small class="muted">(shared by ${v.shared_by} types)</small>` : ""}${crumbs ? `<br><small>${crumbs}</small>` : ""}</span>`;
  if (v.curated_nt && v.curated_nt.length) {
    const agrees = v.curated_nt.includes(n.nt);
    out += `<span class="k">${v.evidence === "literature" ? "literature" : "elsewhere"}</span><span class="v">${esc(v.curated_nt.join(" + "))} <small class="${agrees ? "agree" : "warn"}">${agrees ? "agrees" : "differs from the prediction"}</small> <small class="muted">(${v.evidence === "literature" ? "curated in the ontology" : "another connectome's prediction, via the ontology"})</small></span>`;
  }
  const extra = [];
  if (v.lineage && v.lineage.length) extra.push(esc(v.lineage[0]));
  if (v.birth) extra.push(`${v.birth} neuron`);
  if (v.peptides && v.peptides.length) extra.push(`peptides: ${esc(v.peptides.join(", "))}`);
  if (extra.length) out += `<span class="k">also</span><span class="v">${extra.join(" · ")}</span>`;
  return out;
}
function receptorRow(rx) {
  if (!rx || !rx.receptors.length) return "";
  const cells = rx.receptors.map((r) => {
    const tip = esc(r.modulator ? `${r.modulator} receptor, ${r.sign > 0 ? "raises" : "lowers"} the gain` : "not a modelled receptor");
    const name = r.flybase ? `<a href="${esc(r.flybase)}" target="_blank" rel="noopener" title="${tip} · FlyBase">${esc(r.gene)}</a>` : `<span title="${tip}">${esc(r.gene)}</span>`;
    return `${name} ${Math.round(100 * r.extent)} %`;
  }).join(", ");
  return `<span class="k">receptors</span><span class="v">${cells} <small class="muted">(${esc(rx.family_label)}${rx.depth ? ", from " + esc(rx.label) : ""}; % of cells)</small></span>`;
}

// ---------------------------------------------------------------- dialogs and keyboard
const INTRO_FEMALE = (Lk) => `The fly's brain is a simulation of the whole female fruit-fly brain: all ${Lk.n.toLocaleString()} ` +
  `neurons of FlyWire's connectome, release 783 (Dorkenwald et al., Nature 2024), without the nerve cord. Every neuron is the same ` +
  `simple "leaky integrate-and-fire" unit, and every connection's strength comes from the synapse count in the wiring diagram ` +
  `(Shiu et al., Nature 2024, the model as published). Nothing is trained.`;
function buildDialogs() {
  const other = layouts[1];
  if (other) {
    const who = (Lk) => Lk.sex === "female" ? `the whole female brain, FlyWire release 783 (${Lk.n.toLocaleString()} neurons, no nerve cord)`
      : `the whole male central nervous system, MaleCNS v1.0 (${Lk.n.toLocaleString()} neurons, nerve cord included)`;
    setText($("realIntro"), `Two simulated flies share this dish, each a whole connectome run as the same simple "leaky integrate-and-fire" ` +
      `network (Shiu et al., Nature 2024): fly 0 is ${who(L)}, fly 1 is ${who(other)}. They reach each other only through the world, ` +
      `through hand-built senses listed below; nothing links one brain to the other, and nothing is trained.`);
  } else if (L.sex === "female") setText($("realIntro"), INTRO_FEMALE(L));
  const wr = L.whats_real || {};
  $("realWiring").innerHTML = (wr.wiring || []).map((t) => `<li>${esc(t)}</li>`).join("");
  $("realHand").innerHTML = (wr.hand_built || []).map((t) => `<li>${esc(t)}</li>`).join("");
  $("realNot").innerHTML = (wr.not_modelled || []).map((t) => `<li>${esc(t)}</li>`).join("");
  setShown($("realOtherHead"), !!other); setShown($("realOther"), !!other);
  if (other) {                                          // the other fly's own wiring facts, the ones the first fly's list lacks
    const mine = new Set([...(wr.wiring || []), ...(wr.hand_built || []), ...(wr.not_modelled || [])]), ow = other.whats_real || {};
    const items = [["wiring", ow.wiring], ["hand-built", ow.hand_built], ["not modelled", ow.not_modelled]]
      .flatMap(([tag, list]) => (list || []).filter((t) => !mine.has(t)).map((t) => `<li><small class="tag">${tag}</small> ${esc(t)}</li>`));
    $("realOther").innerHTML = items.join("");
  }
}
$("realBtn").onclick = () => $("realDlg").showModal();
$("keysBtn").onclick = () => $("keysDlg").showModal();
document.querySelectorAll("[data-close]").forEach((b) => (b.onclick = () => b.closest("dialog").close()));
window.addEventListener("keydown", (e) => {
  const tag = e.target.tagName;
  const typing = (tag === "INPUT" && !/^(checkbox|radio|range|button)$/.test(e.target.type)) || tag === "SELECT" || tag === "TEXTAREA";
  if (typing || e.ctrlKey || e.metaKey) return;
  if (e.key === " " && e.target.closest("button, a, input, select, summary, [role=button]")) return;   // Space activates a focused control
  if (!L) return;
  if (/^[0-9]$/.test(e.key)) { const k = e.key === "0" ? 9 : parseInt(e.key) - 1; if (toolOrder[k]) setTool(toolOrder[k]); return; }
  switch (e.key) {
    case " ": e.preventDefault(); post({ type: "pause", on: !(S && S.paused) }); break;
    case "f": case "F": if (S && S.flies) toast("The female is simulated here (--partner); the scripted female is for single-fly play.", 3000); else post({ type: "female", on: !(S && S.world && S.world.female) }); break;
    case "c": case "C": clap(); break;
    case "s": case "S": $("seesToggle").checked = sees = !sees; break;
    case "e": case "E": $("retinaToggle").checked = retinaOn = !retinaOn; setShown($("retinaBox"), retinaOn); break;
    case "w": case "W": { const V = S ? viewOf(S) : null; post({ type: "autopilot", on: !(V && V.autopilot) }); break; }
    case "n": case "N": post({ type: "reset" }); break;
    case "h": case "H": if (layout) layout.toggleSidebar(); break;
    case "d": case "D": toggle3d(); break;
    case "+": case "=": zoomStep(1); break;
    case "-": case "_": zoomStep(-1); break;
    case "Escape": setShown($("neuronPop"), false); for (const b of brains) if (b) b.picked = -1; $("clearMenu").hidden = true; break;
  }
});

// ---------------------------------------------------------------- main loop
let last = performance.now();
const failed = new Set();
// one broken drawer must not stop the others, and nothing may stop the loop itself: each part is guarded
// and the next frame is always scheduled (a failure is logged once)
function guard(name, fn) {
  try { fn(); } catch (e) { if (!failed.has(name)) { failed.add(name); console.error(`${name} failed; the rest of the page keeps running`, e); } }
}
function frame(now) {
  try {
    const dt = Math.min(0.1, (now - last) / 1000); last = now;
    if (L && arena) {
      if (renderPending && S) { renderPending = false; guard("panels", () => renderState(S)); }
      const poses = S ? posesOf(S) : [], pose = poses[focus] || poses[0] || null, female = S ? scriptedFemalePose() : null;
      const view = { S, pose, flies: poses, female, pointer, tool, sees, cues: true, handAng: S ? serverHandAngle() : 0, stripes: S ? stripesNow() : null };
      if (view3dState === "on" && view3d) guard("the 3-D dish", () => view3d.draw(dt, view));
      else guard("the dish", () => arena.draw(dt, view));
      guard("the brain map", () => brain.frame(dt, now / 1000));
      guard("the key-neuron sparklines", () => panels.keys.drawSparks(now));
      setShown($("disc"), lastStateAt > 0 && now - lastStateAt > 3000);
    }
  } catch (e) {
    guard("the frame", () => { throw e; });
  } finally {
    requestAnimationFrame(frame);
  }
}
requestAnimationFrame(frame);
loadLayout();
