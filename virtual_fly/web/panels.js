// panels.js — the right-hand column: each panel builds its DOM once and updates values in place.
"use strict";
import { $, setText, setWidth, setClass, setShown, esc, fmt, post, getJSON, debounce, el } from "./util.js";

const MODE_TEXT = { idle: "resting", walk: "walking", feed: "eating", groom: "grooming", escape: "ESCAPE!", backward: "backing up", court: "courting" };
const MODE_COLOR = { idle: "#1c2633", walk: "#16302a", feed: "#3a2c0e", groom: "#12283a", escape: "#3a1414", backward: "#3a260f", court: "#3a1a30" };
const SIDE = { L: "on its left", R: "on its right", LR: "ahead", RL: "ahead" };

// ================================================================= 1. Why it's doing that
export class WhyPanel {
  constructor(L) {
    this.odourName = {}; for (const o of L.odours || []) this.odourName[o.id] = o.name;
    const flowText = (s) => { const out = []; for (const m of String(s).matchAll(/([LR])([ab])/g)) out.push(`${m[1]} ${m[2] === "a" ? "front→back" : "back→front"}`); return out.join(", "); };
    this.defs = [
      ["sugar", "taste_sugar", "tastes sugar", () => "tastes sugar"],
      ["bitter", "taste_bitter", "tastes bitter", () => "tastes bitter"],
      ["water", "taste_water", "tastes water", () => "tastes water"],
      ["", "small", "sees small object", (f) => "sees small object " + (SIDE[f.small] || "")],
      ["danger", "loom", "sees looming", (f) => "sees looming " + (SIDE[f.loom] || "")],
      ["", "flow", "wide-field motion", (f) => "wide-field motion: " + flowText(f.flow)],
      ["smell", "smell", "smells", (f) => "smells " + (this.odourName[f.smell] || f.smell)],
      ["pink", "pheromone", "pheromone", () => "tastes the female's pheromone"],
      ["pink", "courting", "courting", () => "courtship arousal (pC1 driven)"],
      ["wind", "sound", "sound", () => "hears a loud sound"],
      ["wind", "wind", "wind", (f) => `wind from ${Math.abs(f.wind)}° ${f.wind > 5 ? "left" : f.wind < -5 ? "right" : "ahead"}`],
      ["", "dust", "dust", () => "dust on antennae"],
      ["", "touch", "touch", () => "head touching something"],
      ["reward", "reward", "reward", () => "sugar reward → PAM dopamine"],
      ["danger", "shock", "shock", () => "shock → PPL1 dopamine"],
      ["", "zap", "zap", (f) => "zap: " + f.zap],
    ];
    const box = $("senses"); box.innerHTML = "";
    // the chip's label never changes (a chip that grew a few words every tick re-wrapped the row and
    // moved every card below it); the details go on one fixed-height line underneath
    this.chips = this.defs.map(([cls, key, off, on]) => { const e = el("span", "chip " + cls, esc(off)); e.__t = off; box.appendChild(e); return { e, key, off, on }; });
    this.detail = $("senseDetail");
  }
  update(S) {
    const m = $("mode"), mode = S.mode || (S.fly && S.fly.mode) || "idle";
    setText(m, MODE_TEXT[mode] || mode);
    if (m.__bg !== mode) { m.__bg = mode; m.style.background = MODE_COLOR[mode] || "#1c2633"; }
    setText($("driver"), S.driver || "");
    const f = S.senses || {}, details = [];
    for (const ch of this.chips) {
      const on = ch.key in f && f[ch.key] !== false && f[ch.key] !== "" && f[ch.key] != null;
      setClass(ch.e, "on", on);
      if (on) { const d = ch.on(f); if (d !== ch.off) details.push(d); }
    }
    setText(this.detail, details.join(" · "));
  }
}

// ================================================================= 2. Key neurons (bars + sparklines)
const HIST = 240;   // samples kept per readout (6 s at 40 ticks/s)
export class KeyNeurons {
  constructor(L) {
    this.L = L; this.rows = {}; this.hist = {}; this.customKeys = ""; this.lastSpark = 0;
    this.box = $("rows");
    const groups = new Map();
    for (const r of L.readouts) { if (!groups.has(r.group)) groups.set(r.group, []); groups.get(r.group).push(r); }
    for (const [g, rs] of groups) {
      this.box.appendChild(el("div", "grp", `<span>${esc(g)}</span>`));
      for (const r of rs) this.addRow(r, false);
    }
    this.customGrp = el("div", "grp", `<span>Custom watches</span>`); this.customGrp.hidden = true;
    this.box.appendChild(this.customGrp);
    this.seed();
  }
  addRow(r, custom) {
    const name = el("div", "name");
    const shown = custom ? r.key : r.key.replace(/(L|R)$/, " $1").replace(/^GF$/, "DNp01");
    name.innerHTML = `<span>${esc(shown)} <small>${esc(r.label)}</small></span>` + (custom ? `<button title="stop watching">✕</button>` : "");
    name.title = `${r.key}: ${r.spec}`;
    if (custom) name.querySelector("button").onclick = () => post({ type: "unwatch", key: r.key });
    const bar = el("div", "bar"), fill = document.createElement("div"); fill.style.background = r.colour; bar.appendChild(fill);
    const spark = el("canvas", "spark"); spark.width = 64 * (window.devicePixelRatio || 1); spark.height = 16 * (window.devicePixelRatio || 1);
    const val = el("div", "val", "0");
    this.box.append(name, bar, spark, val);
    this.rows[r.key] = { fill, val, spark, max: r.max, colour: r.colour, custom, els: [name, bar, spark, val], peak: 1 };
    if (!this.hist[r.key]) this.hist[r.key] = { a: new Float32Array(HIST), i: 0, n: 0 };
  }
  removeRow(key) { const r = this.rows[key]; if (!r) return; r.els.forEach((e) => e.remove()); delete this.rows[key]; delete this.hist[key]; }
  async seed() {
    const keys = this.L.readouts.map((r) => r.key);
    const h = await getJSON(`api/history?keys=${encodeURIComponent(keys.join(","))}&n=${HIST}`);
    if (!h || !h.history) return;
    for (const k of keys) { const arr = h.history[k]; if (!arr) continue; const H = this.hist[k]; if (!H || H.n > 40) continue; for (const v of arr) this.push(H, v); }
  }
  push(H, v) { H.a[H.i] = v; H.i = (H.i + 1) % HIST; if (H.n < HIST) H.n++; }
  /** Called for every state tick (cheap): keep the sparkline history complete even when the page
   *  draws fewer frames than the server sends ticks. */
  ingest(S) {
    const hz = S.hz || {}, custom = S.custom || {};
    const ck = Object.keys(custom).join("|");
    if (ck !== this.customKeys) {                       // watches added or removed: rebuild only those rows
      this.customKeys = ck;
      for (const k of Object.keys(this.rows)) if (this.rows[k].custom && !(k in custom)) this.removeRow(k);
      for (const k of Object.keys(custom)) if (!this.rows[k]) this.addRow({ key: k, spec: custom[k], label: custom[k], max: 50, colour: "#c792ff" }, true);
      this.customGrp.hidden = !ck;
    }
    for (const k in this.rows) this.push(this.hist[k], hz[k] || 0);
  }
  /** Called once per animation frame with the latest state: the DOM writes. */
  update(S) {
    const hz = S.hz || {};
    if (this.customKeys !== Object.keys(S.custom || {}).join("|")) this.ingest(S);
    for (const k in this.rows) {
      const r = this.rows[k], v = hz[k] || 0;
      if (r.custom) r.peak = Math.max(r.peak * 0.999, v, 5), r.max = r.peak;
      setWidth(r.fill, (100 * v) / r.max);
      setText(r.val, String(Math.round(v)));
    }
  }
  drawSparks(now) {
    if (now - this.lastSpark < 120) return;
    this.lastSpark = now;
    const dpr = window.devicePixelRatio || 1;
    for (const k in this.rows) {
      const r = this.rows[k], H = this.hist[k], c = r.spark.getContext("2d"), W = r.spark.width, Hh = r.spark.height;
      c.clearRect(0, 0, W, Hh);
      if (H.n < 2) continue;
      c.beginPath();
      const step = W / (HIST - 1), start = (H.i - H.n + HIST) % HIST, x0 = W - (H.n - 1) * step;
      for (let j = 0; j < H.n; j++) {
        const v = H.a[(start + j) % HIST], y = Hh - 1 - Math.min(1, v / r.max) * (Hh - 2);
        if (j === 0) c.moveTo(x0, y); else c.lineTo(x0 + j * step, y);
      }
      c.strokeStyle = r.colour; c.globalAlpha = 0.9; c.lineWidth = dpr; c.stroke();
      c.lineTo(W, Hh); c.lineTo(x0, Hh); c.closePath(); c.fillStyle = r.colour; c.globalAlpha = 0.15; c.fill(); c.globalAlpha = 1;
    }
  }
}

// ================================================================= 3. Internal state
export class DrivesPanel {
  constructor() {
    this.dragging = null;
    for (const k of ["hunger", "thirst"]) {
      const s = $(k + "Set");
      s.addEventListener("pointerdown", () => (this.dragging = k));
      s.addEventListener("input", debounce(() => post({ type: "state", [k]: parseFloat(s.value) }), 60));
      s.addEventListener("change", () => { post({ type: "state", [k]: parseFloat(s.value) }); this.dragging = null; });
      s.addEventListener("pointerup", () => (this.dragging = null));
    }
  }
  update(S) {
    const st = S.state || {};
    for (const k of ["hunger", "thirst", "arousal"]) {
      const v = st[k] || 0;
      setWidth($(k + "Bar"), 100 * v); setText($(k + "Val"), fmt(v, 2));
      const s = $(k + "Set"); if (s && this.dragging !== k) s.value = v;
    }
  }
}

// ================================================================= 4. Learning
export class LearningPanel {
  constructor(L) {
    this.odourName = {}; for (const o of L.odours || []) this.odourName[o.id] = o.name;
    this.bars = {}; this.keys = "";
    $("learnBtn").onclick = () => post({ type: "learning", on: !this.enabled });
    $("forgetBtn").onclick = () => { if (confirm("Reset every KC→MBON synapse to its original strength?")) post({ type: "learning", forget: true }); };
  }
  update(S) {
    const l = S.learning;
    setShown($("learnBody"), !!l); setShown($("learnOff"), !l);
    const lt = $("learnToggle");
    if (!l) { setText($("learnStatus"), "off in this profile"); lt.disabled = true; lt.checked = false; return; }
    lt.disabled = false; if (document.activeElement !== lt) lt.checked = !!l.enabled;
    this.enabled = !!l.enabled;
    setText($("learnStatus"), l.enabled ? "plasticity on" : "plasticity paused");
    setText($("learnBtn"), l.enabled ? "Learning: on" : "Learning: off");
    const mb = l.mbon || {}, keys = Object.keys(mb).sort();
    if (keys.join() !== this.keys) {
      this.keys = keys.join(); const box = $("mbons"); box.innerHTML = ""; this.bars = {};
      for (const k of keys) {
        const d = el("div", "mbon" + (mb[k].valence < 0 ? " av" : ""));
        d.innerHTML = `<span class="tip"></span><div class="b"></div><div class="d"></div><i class="m"></i>`;
        box.appendChild(d); this.bars[k] = { d, b: d.querySelector(".b"), dop: d.querySelector(".d"), tip: d.querySelector(".tip"), m: d.querySelector(".m") };
      }
    }
    for (const k of keys) {
      // `now`: strength weighted by the Kenyon cells active right now (what the current odour experiences);
      // `strength`: the mean over all Kenyon cells (drawn as a faint tick). Older servers only send `strength`.
      const b = this.bars[k], m = mb[k], now = m.now == null ? m.strength : m.now;
      const h = Math.round(100 * Math.max(0, Math.min(1, now))), hs = Math.round(100 * Math.max(0, Math.min(1, m.strength)));
      if (b.h !== h) { b.h = h; b.b.style.height = Math.max(4, h * 0.4) + "px"; }
      if (b.hs !== hs) { b.hs = hs; b.m.style.bottom = (4 + hs * 0.4) + "px"; b.m.hidden = m.now == null; }
      const dop = Math.round(10 * Math.min(1, m.dopamine)) / 10;
      if (b.dop.__o !== dop) { b.dop.__o = dop; b.dop.style.opacity = dop; }
      setText(b.tip, `${k}: ${m.valence > 0 ? "approach" : "avoid"} · now ${fmt(now, 2)} (for the odour it smells)` + (m.now == null ? "" : ` · mean ${fmt(m.strength, 2)}`) + ` · dopamine ${fmt(m.dopamine, 2)}`);
      setClass(b.d, "av", m.valence < 0);
    }
    const bias = Math.max(-1, Math.min(1, l.learned_bias || 0));
    const n = $("gaugeNeedle"), left = Math.round(50 + 50 * bias);
    if (n.__l !== left) { n.__l = left; n.style.left = left + "%"; }
    setText($("gaugeText"), l.smelling ? `smelling ${this.odourName[l.smelling] || l.smelling}: learned bias ${bias >= 0 ? "+" : ""}${fmt(bias, 2)}` : "not smelling anything");
    setText($("learnText"), `${fmt(100 * (l.depressed_fraction || 0), 1)} % of KC→MBON synapses depressed · ${l.events || 0} learning event${l.events === 1 ? "" : "s"}`);
  }
}

// ================================================================= 5. Experiments checklist
export class ChecksPanel {
  constructor(L) {
    this.els = {}; const ul = $("checks"); ul.innerHTML = "";
    for (const c of L.checks || []) { const li = el("li", "", `<span class="box">✓</span><span>${esc(c.text)}</span>`); ul.appendChild(li); this.els[c.id] = li; }
    this.total = (L.checks || []).length;
  }
  update(S) {
    const done = S.done || []; let n = 0;
    for (const id in this.els) { const on = done.includes(id); if (on) n++; setClass(this.els[id], "done", on); }
    setText($("checksDone"), `${n} / ${this.total}`);
  }
}

// ================================================================= 6. Scenarios
export class ScenariosPanel {
  constructor(L) {
    const box = $("scenarios"); box.innerHTML = ""; this.buttons = {};
    for (const s of L.scenarios || []) {
      const d = el("div", "scn", `<span>${esc(s.name)}</span><button class="btn">Run</button><span class="desc">${esc(s.description)}</span>`);
      d.querySelector("button").onclick = () => post({ type: "scenario", id: s.id });
      box.appendChild(d); this.buttons[s.id] = d.querySelector("button");
    }
    $("scnStop").onclick = () => post({ type: "scenario" });
    this.stepKey = ""; this.stepMax = 0;
  }
  update(S) {
    const sc = S.scenario;
    setShown($("scenarioStatus"), !!sc);
    for (const id in this.buttons) { const b = this.buttons[id]; const running = sc && sc.id === id; setText(b, running ? "Running…" : "Run"); b.disabled = !!running; }
    if (!sc) return;
    setText($("scnName"), sc.name); setText($("scnStep"), `step ${sc.step} / ${sc.steps}`);
    setText($("scnCaption"), sc.caption || ""); setText($("scnLeft"), sc.left > 0 ? `${fmt(sc.left, 0)} s left in this step` : "");
    const key = sc.id + ":" + sc.step;
    if (key !== this.stepKey) { this.stepKey = key; this.stepMax = sc.left || 0; }
    this.stepMax = Math.max(this.stepMax, sc.left || 0);
    const within = this.stepMax > 0 ? 1 - sc.left / this.stepMax : 1;
    setWidth($("scnProgress"), 100 * ((sc.step - 1 + within) / Math.max(1, sc.steps)));
    const m = sc.measure || {}, box = $("scnMeasure"), keys = Object.keys(m);
    if (box.__keys !== keys.join()) { box.__keys = keys.join(); box.innerHTML = keys.map((k) => `<span class="k">${esc(k.replace(/_/g, " "))}</span><span class="v" data-k="${esc(k)}"></span>`).join(""); }
    for (const e of box.querySelectorAll(".v")) setText(e, typeof m[e.dataset.k] === "number" ? fmt(m[e.dataset.k], 2) : String(m[e.dataset.k]));
  }
}

// ================================================================= 7. Neuron lab
export class LabPanel {
  constructor(L) {
    this.L = L;
    const dl = $("typelist"), frag = document.createDocumentFragment();
    for (const t of L.types || []) { const o = document.createElement("option"); o.value = t; frag.appendChild(o); }
    dl.appendChild(frag);
    this.baseOptions = dl.innerHTML;
    const search = debounce(async (q) => {
      if (q.length < 2) { if (dl.innerHTML !== this.baseOptions) dl.innerHTML = this.baseOptions; return; }
      const r = await getJSON(`api/types?q=${encodeURIComponent(q)}&limit=40`);
      if (!r || !r.types) return;
      const seen = new Set(), opts = [];
      for (const t of r.types) { if (!seen.has(t.type)) { seen.add(t.type); opts.push(`<option value="${esc(t.type)}">${t.n} neurons</option>`); } }
      dl.innerHTML = opts.join("") + this.baseOptions;
    }, 180);
    for (const id of ["spec", "traceFrom", "traceTo"]) $(id).addEventListener("input", (e) => search(e.target.value.trim().replace(/^prefix:|\/[LRM]$/g, "")));
    $("spec").addEventListener("keydown", (e) => { if (e.key === "Enter") this.zap(); });
    const pr = $("presets"); pr.innerHTML = "";
    for (const p of L.presets || []) {
      const b = document.createElement("button"); b.textContent = p.spec; b.title = p.label;
      b.onclick = () => { $("spec").value = p.spec; $("hz").value = p.hz; this.zap(); };
      pr.appendChild(b);
    }
    $("zapBtn").onclick = () => this.zap();
    $("silenceBtn").onclick = () => this.silence();
    $("modBtn").onclick = () => this.modulate();
    $("watchBtn").onclick = () => this.watch();
    this.silKey = ""; this.modKey = "";
  }
  spec() { const s = $("spec").value.trim(); if (!s) this.feedback("Type a neuron type first, or pick one of the presets.", true); return s; }
  feedback(text, err) { setText($("feedback"), text); setClass($("feedback"), "err", !!err); }
  async zap() {
    const spec = this.spec(); if (!spec) return;
    const hz = parseFloat($("hz").value) || 80;
    const r = await post({ type: "zap", spec, hz, secs: 2 });
    this.feedback(r.ok ? `Zapping ${spec}: ${r.n} neuron${r.n === 1 ? "" : "s"} at ${hz} spikes/s for 2 s.` : r.error, !r.ok);
  }
  async silence(spec) {
    spec = spec || this.spec(); if (!spec) return;
    const r = await post({ type: "silence", spec });
    this.feedback(r.ok ? `Silenced ${spec} (${r.n} neurons). Click ✕ to undo.` : r.error, !r.ok);
  }
  async modulate() {
    const spec = this.spec(); if (!spec) return;
    const factor = Math.max(0, parseFloat($("factor").value)); if (isNaN(factor)) return;
    const r = await post({ type: "modulate", spec, factor });
    this.feedback(r.ok ? `Output of ${spec} (${r.n} neurons) scaled ×${factor}.` : r.error, !r.ok);
  }
  async watch() {
    const spec = this.spec(); if (!spec) return;
    const r = await post({ type: "watch", spec });
    this.feedback(r.ok ? `Watching ${spec} (${r.n} neurons): see the bar under 'Custom watches'.` : r.error, !r.ok);
  }
  update(S) {
    const sil = S.silenced || [], base = S.baseline || [], mod = S.modulated || {};
    const sk = sil.join("|") + "#" + base.join("|");
    if (sk !== this.silKey) {
      this.silKey = sk; const box = $("silenced");
      box.innerHTML = sil.map((x) => `<span>🚫 ${esc(x)}<button title="unsilence" data-spec="${esc(x)}">✕</button></span>`).join("")
        + base.map((x) => `<span class="base" title="Silenced by the game to stop runaway loops (see 'What's real here?')">${esc(x)} (game default)</span>`).join("");
      box.querySelectorAll("button").forEach((b) => (b.onclick = () => post({ type: "unsilence", spec: b.dataset.spec })));
    }
    const mk = Object.entries(mod).map(([k, v]) => k + "=" + v).join("|");
    if (mk !== this.modKey) {
      this.modKey = mk; const box = $("modulated");
      box.innerHTML = Object.entries(mod).map(([k, v]) => `<span class="mod">×${v} ${esc(k)}<button title="back to normal" data-spec="${esc(k)}">✕</button></span>`).join("");
      box.querySelectorAll("button").forEach((b) => (b.onclick = () => post({ type: "modulate", spec: b.dataset.spec, factor: 1 })));
    }
  }
}

// ================================================================= 8. Pathway explorer
export class PathwayPanel {
  constructor(brain, lab) {
    this.brain = brain; this.lab = lab; this.result = null; this.selected = -1;
    $("traceBtn").onclick = () => this.trace();
    for (const id of ["traceFrom", "traceTo"]) $(id).addEventListener("keydown", (e) => { if (e.key === "Enter") this.trace(); });
  }
  update() { /* nothing per tick: results come from an explicit trace */ }
  async trace() {
    const from = $("traceFrom").value.trim(), to = $("traceTo").value.trim(), hops = $("traceHops").value;
    if (!from || !to) return;
    const btn = $("traceBtn"); btn.disabled = true; setText($("traceInfo"), "searching the wiring…");
    const r = await getJSON(`api/trace?from=${encodeURIComponent(from)}&to=${encodeURIComponent(to)}&hops=${hops}&top=8`);
    btn.disabled = false;
    if (!r || !r.ok) { setText($("traceInfo"), (r && r.error) || "no answer"); setClass($("traceInfo"), "err", true); return; }
    setClass($("traceInfo"), "err", false);
    this.result = r;
    setText($("traceInfo"), r.paths.length ? `${r.paths.length} route${r.paths.length === 1 ? "" : "s"} in ${r.secs} s` : `no route within ${hops} hops (${r.secs} s)`);
    const box = $("paths"); box.innerHTML = "";
    r.paths.forEach((p, i) => {
      const d = el("div", "path");
      const chain = p.nodes.map((n, j) => {
        const h = p.hops[j - 1], hop = j === 0 ? "" : `<span class="hop ${h.sign > 0 ? "pos" : "neg"}" title="${h.synapses} synapses, ${fmt(100 * h.fraction, 1)} % of ${esc(h.dst)}'s input (${h.src_size} → ${h.dst_size} cells)">→(${h.sign > 0 ? "+" : "−"})<small>${h.synapses}</small>→</span>`;
        return hop + `<span class="node ${j === 0 || j === p.nodes.length - 1 ? "end" : ""}">${esc(n)}</span>`;
      }).join("");
      d.innerHTML = `<div class="chain">${chain}</div><div class="meta"><span>score ${p.score.toExponential(2)}</span><span>${p.net_sign > 0 ? "net excitatory" : "net inhibitory"}</span><span>${p.hops.length} hops</span></div>`;
      d.onclick = () => this.select(i);
      box.appendChild(d);
    });
    const rel = $("relays");
    rel.innerHTML = r.relays && r.relays.length ? `<h5>Relays: where the routes converge (lesion candidates)</h5>` + r.relays.slice(0, 6).map(([name, sc]) =>
      `<div class="relay"><span>${esc(name)}</span><span class="sc">${sc.toExponential(2)}</span><button class="mini" data-spec="${esc(name)}">Silence</button></div>`).join("") : "";
    rel.querySelectorAll("button").forEach((b) => (b.onclick = () => this.lab.silence(b.dataset.spec)));
    if (r.paths.length) this.select(0);
  }
  select(i) {
    this.selected = i;
    $("paths").querySelectorAll(".path").forEach((e, j) => e.classList.toggle("on", j === i));
    const r = this.result; if (!r || !r.neurons || !r.neurons[i]) { this.brain.setPath(null); return; }
    this.brain.pathNames = r.paths[i].nodes;
    this.brain.setPath(r.neurons[i]);
    this.brain.setView(true);
  }
}

// ================================================================= 9. Event log
export class EventsPanel {
  constructor() { this.lastId = 0; this.count = 0; this.queue = []; this.ul = $("events"); $("eventsClear").onclick = () => { this.ul.innerHTML = ""; this.count = 0; }; }
  /** Every tick: remember the events not seen yet (the DOM is written once per frame). */
  ingest(S) {
    for (const e of S.events || []) { if (e.id > this.lastId) { this.lastId = e.id; this.queue.push(e); } }
  }
  update(S) {
    if (!this.queue.length) { if (S.events && S.events.length && !this.lastId) this.ingest(S); if (!this.queue.length) return; }
    const ul = this.ul, atBottom = ul.scrollHeight - ul.scrollTop - ul.clientHeight < 24;
    const frag = document.createDocumentFragment();
    for (const e of this.queue) {
      const li = el("li", e.kind, `<i></i><span class="t">${fmt(e.t, 1)} s</span><span>${esc(e.text)}</span>`);
      li.title = e.kind; frag.appendChild(li); this.count++;
    }
    this.queue.length = 0;
    ul.appendChild(frag);
    while (this.count > 200) { ul.firstChild.remove(); this.count--; }
    if (atBottom) ul.scrollTop = ul.scrollHeight;
  }
}

// ================================================================= 10. Recording
export class RecordingPanel {
  // `on` is tracked here: the server keeps `state.recording` (the kept frames) after a stop.
  constructor() {
    this.on = null; this.holdUntil = 0;
    $("recBtn").onclick = () => { this.on = !this.on; this.holdUntil = performance.now() + 2000; post({ type: "record", on: this.on, spikes: $("recSpikes").checked }); this.render(this.last); };
  }
  update(S) {
    this.last = S.recording;
    if (this.on === null) this.on = !!S.recording;                                   // first state: adopt what the server says
    else if (!S.recording && performance.now() > this.holdUntil) this.on = false;   // (after a grace period: the action lands on the next brain tick)
    this.render(S.recording);
  }
  render(r) {
    setText($("recBtn"), this.on ? "■ Stop" : "● Record"); setClass($("recBtn"), "rec", this.on);
    setText($("recInfo"), r ? `${r.frames.toLocaleString()} frames${r.spikes ? " + every spike" : ""}${this.on ? "" : " kept: download below"}` : "not recording");
  }
}

// ================================================================= 11. Model
export class ModelPanel {
  constructor(L) {
    setText($("modelProfile"), `profile: ${L.profile}`);
    const s = L.settings || {}, p = s.plasticity;
    const rows = [
      ["neurons", `${L.n.toLocaleString()} (${(L.edges / 1e6).toFixed(2)} M connections, ${(L.synapses / 1e6).toFixed(1)} M synapses)`],
      ["time step", `${s.dt} ms brain time, ${L.tick_ms} ms per world tick`],
      ["synaptic gain", `${s.gain}` + (s.kenyon_gain != null && s.kenyon_gain !== 1 ? ` (Kenyon cells ×${s.kenyon_gain})` : "")],
      ["fatigue", `${s.fatigue_mv} mV per spike, fading over ${s.fatigue_ms} ms`],
      ["noise", s.noise_hz ? `${s.noise_hz} Hz × ${s.noise_mv} mV (${s.noise_spec})` : "none"],
      ["silenced by profile", (s.silenced || []).join(", ") || "nothing"],
      ["plasticity", p ? `rate ${p.rate}, KC trace ${p.kc_tau_ms} ms, DAN trace ${p.dan_tau_ms} ms, floor ${p.floor}, recovery ${p.recover_min} min, ${Number(p.plastic_synapses).toLocaleString()} plastic KC→MBON synapses` : "off"],
      ["columnar vision", L.columnar_vision ? "on: T4/T5 columns driven from the retina" : "off: LC4/LPLC2/LC10a/HS driven from feature detectors"],
      ["brain speed", `<span id="modelRtf">–</span>`],
    ];
    $("modelKv").innerHTML = rows.map(([k, v]) => `<span class="k">${esc(k)}</span><span class="v">${k === "brain speed" ? v : esc(v)}</span>`).join("");
    const dec = L.decoder || {}, t = $("decoderTable");
    t.innerHTML = `<tr><th>DN</th><th>direct → motor</th><th>2-hop motor synapses by neuromere</th></tr>` + Object.entries(dec).map(([k, v]) =>
      `<tr><td>${esc(k)}</td><td>${v.direct_motor_synapses}</td><td>${Object.entries(v.two_hop_motor_synapses_by_neuromere || {}).map(([n, c]) => `${esc(n)} ${c}`).join(", ")}</td></tr>`).join("");
  }
  update(S) { setText($("modelRtf"), S.paused ? "paused" : `${fmt(S.rtf, 2)}× real time at speed ${fmt(S.speed, 2)}× · ${S.stims} stimulated populations · brain calmed ${S.calms}×`); }
}
