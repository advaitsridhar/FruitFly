// brain3d.js — every neuron with a known soma as a WebGL point cloud, flashing when it spikes.
// Falls back to the 2-D canvas approach of the first version when WebGL is unavailable.
"use strict";

export const REGION_COLORS = [
  "#5b8cff", // optic lobes
  "#4de2c5", // central brain
  "#ff5d8f", // descending
  "#9be15d", // nerve cord
  "#ffc857", // ascending
  "#ff7a45", // motor
  "#6f7c8c", // other
  "#ff9ad5", // Kenyon cells
  "#c792ff", // MBON / DAN
];

const hex2rgb = (h) => [parseInt(h.slice(1, 3), 16), parseInt(h.slice(3, 5), 16), parseInt(h.slice(5, 7), 16)];

const VS = `
attribute vec3 aPos; attribute vec3 aCol; attribute float aT;
uniform mat4 uMVP; uniform float uNow, uTau, uSize;
varying vec3 vCol; varying float vK;
void main() {
  gl_Position = uMVP * vec4(aPos, 1.0);
  float k = clamp(exp(-(uNow - aT) / uTau), 0.0, 1.0);
  gl_PointSize = uSize * (1.0 + 1.6 * k);
  vCol = aCol; vK = k;
}`;
const FS = `
precision mediump float;
varying vec3 vCol; varying float vK; uniform float uBase, uGain;
void main() {
  vec2 d = gl_PointCoord - 0.5; float r2 = dot(d, d);
  if (r2 > 0.25) discard;
  float soft = 1.0 - smoothstep(0.10, 0.25, r2);
  float a = uBase * soft;                                   // dim base coverage ("over" blending)
  vec3 c = vCol * a + (vCol * 0.8 + 0.4) * vK * uGain * soft;   // plus an additive flash when it spiked
  gl_FragColor = vec4(c, a);
}`;

export class BrainView {
  constructor(canvas, overlay, L) {
    this.canvas = canvas; this.overlay = overlay; this.L = L;
    this.w = L.w; this.h = L.h; this.d = L.d;
    this.yaw = 0; this.pitch = 0; this.zoom = 1; this.spin = true; this.mode3d = true;
    this.path = null; this.picked = -1; this.onPick = null;
    this.dpr = window.devicePixelRatio || 1;
    this.nowSec = 0;
    this._buildArrays();
    this.gl = null;
    try { this.gl = canvas.getContext("webgl", { antialias: false, alpha: false, premultipliedAlpha: false, preserveDrawingBuffer: false }); } catch (e) { this.gl = null; }
    if (this.gl) {
      try { this._initGL(); } catch (e) { console.warn("WebGL init failed, using 2-D map", e); this.gl = null; }
    }
    if (!this.gl) this._init2D();
    this._resize();
    this._ro = new ResizeObserver(() => this._resize());
    this._ro.observe(canvas.parentElement);
    this._input();
  }

  // ---------------------------------------------------------------- data
  _buildArrays() {
    const L = this.L, n = L.n;
    let m = 0;
    for (let i = 0; i < n; i++) if (L.x[i] >= 0) m++;
    this.m = m;
    this.pos = new Float32Array(m * 3);
    this.col = new Uint8Array(m * 3);
    this.lastT = new Float32Array(m).fill(-1e6);
    this.slotOf = new Int32Array(n).fill(-1);
    this.neuronOf = new Int32Array(m);
    const rgb = REGION_COLORS.map(hex2rgb);
    let k = 0;
    for (let i = 0; i < n; i++) {
      if (L.x[i] < 0) continue;
      this.pos[3 * k] = L.x[i] - this.w / 2;
      this.pos[3 * k + 1] = this.h / 2 - L.y[i];
      this.pos[3 * k + 2] = L.z[i] - this.d / 2;
      const c = rgb[L.region[i]] || rgb[6];
      this.col[3 * k] = c[0]; this.col[3 * k + 1] = c[1]; this.col[3 * k + 2] = c[2];
      this.slotOf[i] = k; this.neuronOf[k] = i; k++;
    }
  }

  // ---------------------------------------------------------------- WebGL
  _initGL() {
    const gl = this.gl;
    const sh = (type, src) => { const s = gl.createShader(type); gl.shaderSource(s, src); gl.compileShader(s);
      if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s)); return s; };
    const p = gl.createProgram();
    gl.attachShader(p, sh(gl.VERTEX_SHADER, VS)); gl.attachShader(p, sh(gl.FRAGMENT_SHADER, FS)); gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p));
    gl.useProgram(p);
    this.prog = p;
    this.u = {};
    for (const n of ["uMVP", "uNow", "uTau", "uSize", "uBase", "uGain"]) this.u[n] = gl.getUniformLocation(p, n);
    const buf = (data, loc, size, type, norm) => {
      const b = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, b); gl.bufferData(gl.ARRAY_BUFFER, data, type === gl.FLOAT && size === 1 ? gl.DYNAMIC_DRAW : gl.STATIC_DRAW);
      gl.enableVertexAttribArray(loc); gl.vertexAttribPointer(loc, size, type, norm, 0, 0); return b;
    };
    this.bPos = buf(this.pos, gl.getAttribLocation(p, "aPos"), 3, gl.FLOAT, false);
    this.bCol = buf(this.col, gl.getAttribLocation(p, "aCol"), 3, gl.UNSIGNED_BYTE, true);
    this.bT = buf(this.lastT, gl.getAttribLocation(p, "aT"), 1, gl.FLOAT, false);
    gl.disable(gl.DEPTH_TEST);
    gl.enable(gl.BLEND); gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
    gl.clearColor(6 / 255, 9 / 255, 13 / 255, 1);
    this.dirtyT = false;
  }

  // ---------------------------------------------------------------- 2-D fallback
  _init2D() {
    this.ctx = this.canvas.getContext("2d");
    this.bbg = document.createElement("canvas"); this.bfx = document.createElement("canvas");
    this.spin = false; this.mode3d = false;
    this.pending = [];
  }
  _build2D() {
    const W = this.canvas.width, H = this.canvas.height;
    this.bbg.width = this.bfx.width = W; this.bbg.height = this.bfx.height = H;
    const g = this.bbg.getContext("2d");
    g.fillStyle = "#06090d"; g.fillRect(0, 0, W, H);
    const s = Math.max(1, this.dpr), M = this._matrix(), rgb = REGION_COLORS;
    g.globalAlpha = 0.16;
    for (let region = 0; region < rgb.length; region++) {
      g.fillStyle = rgb[region];
      for (let k = 0; k < this.m; k++) {
        if (this.L.region[this.neuronOf[k]] !== region) continue;
        const [x, y] = this._proj(k, M);
        g.fillRect(x * this.dpr, y * this.dpr, s, s);
      }
    }
    g.globalAlpha = 1;
  }

  // ---------------------------------------------------------------- geometry
  _resize() {
    const r = this.canvas.getBoundingClientRect();
    if (!r.width) return;
    this.dpr = window.devicePixelRatio || 1;
    this.cssW = r.width; this.cssH = r.height;
    const W = Math.round(r.width * this.dpr), H = Math.round(r.height * this.dpr);
    if (this.canvas.width !== W || this.canvas.height !== H) {
      this.canvas.width = W; this.canvas.height = H;
      this.overlay.width = W; this.overlay.height = H;
      if (this.gl) this.gl.viewport(0, 0, W, H); else this._build2D();
    }
  }
  _matrix() {
    // rows of the 3x3 rotation Rx(pitch) * Ry(yaw), then scale in css px per map unit
    const cy = Math.cos(this.yaw), sy = Math.sin(this.yaw), cp = Math.cos(this.pitch), sp = Math.sin(this.pitch);
    const fit = Math.min(this.cssW / (this.w * 1.08), this.cssH / (this.h * 1.06)) * this.zoom;
    return { r: [cy, 0, sy, sp * sy, cp, -sp * cy, -cp * sy, sp, cp * cy], s: fit };
  }
  _proj(k, M) {
    const x = this.pos[3 * k], y = this.pos[3 * k + 1], z = this.pos[3 * k + 2], r = M.r;
    return [this.cssW / 2 + (r[0] * x + r[1] * y + r[2] * z) * M.s, this.cssH / 2 - (r[3] * x + r[4] * y + r[5] * z) * M.s];
  }
  projectNeuron(i) { const k = this.slotOf[i]; return k < 0 ? null : this._proj(k, this._matrix()); }

  pick(px, py) {
    const M = this._matrix(); let best = -1, bd = 36;
    for (let k = 0; k < this.m; k++) {
      const [x, y] = this._proj(k, M), d = (x - px) * (x - px) + (y - py) * (y - py);
      if (d < bd) { bd = d; best = k; }
    }
    return best < 0 ? -1 : this.neuronOf[best];
  }

  // ---------------------------------------------------------------- input
  _input() {
    const c = this.canvas; let down = null, moved = 0;
    c.addEventListener("pointerdown", (e) => { down = { x: e.clientX, y: e.clientY, yaw: this.yaw, pitch: this.pitch }; moved = 0; c.setPointerCapture(e.pointerId); c.classList.add("drag"); });
    c.addEventListener("pointermove", (e) => {
      if (!down) return;
      const dx = e.clientX - down.x, dy = e.clientY - down.y; moved = Math.max(moved, Math.abs(dx), Math.abs(dy));
      if (this.gl) { this.yaw = down.yaw + dx * 0.01; this.pitch = Math.max(-1.3, Math.min(1.3, down.pitch + dy * 0.01)); if (moved > 3) this.mode3d = true; }
    });
    const up = (e) => {
      if (!down) return; c.classList.remove("drag");
      if (moved < 4) {
        const r = c.getBoundingClientRect(), i = this.pick(e.clientX - r.left, e.clientY - r.top);
        this.picked = i; if (this.onPick) this.onPick(i, e.clientX - r.left, e.clientY - r.top);
      }
      down = null;
    };
    c.addEventListener("pointerup", up); c.addEventListener("pointercancel", () => { down = null; c.classList.remove("drag"); });
    c.addEventListener("wheel", (e) => { e.preventDefault(); this.zoom = Math.max(0.5, Math.min(8, this.zoom * Math.exp(-e.deltaY * 0.0012))); if (!this.gl) this._build2D(); }, { passive: false });
  }
  setView(mode3d) {
    this.mode3d = mode3d && !!this.gl;
    if (!this.mode3d) { this.yaw = 0; this.pitch = 0; this.spin = false; } else if (this.pitch === 0 && this.yaw === 0) { this.pitch = 0.3; }
  }
  resetView() { this.yaw = 0; this.pitch = this.mode3d ? 0.3 : 0; this.zoom = 1; if (!this.gl) this._build2D(); }

  // ---------------------------------------------------------------- per tick / per frame
  setSpikes(list, nowSec) {
    if (!list || !list.length) return;
    const lt = this.lastT, so = this.slotOf;
    for (let j = 0; j < list.length; j++) { const k = so[list[j]]; if (k >= 0) lt[k] = nowSec; }
    if (this.gl) this.dirtyT = true; else this.pending.push(list);
  }
  setPath(indices) { this.path = indices && indices.length ? indices : null; }

  frame(dt, nowSec) {
    // 141k points are redrawn at most 30 times a second (the flashes decay in the shader, so they
    // stay smooth); on a laptop this halves what the map costs the page every frame
    this._acc = (this._acc || 0) + dt;
    if (this._acc < 1 / 30 && !this.path && this.picked < 0) return;
    const step = this._acc; this._acc = 0;
    this.nowSec = nowSec;
    if (this.spin && this.gl && this.mode3d) this.yaw += step * 0.25;
    if (this.gl) this._drawGL(); else this._draw2D(step);
    this._drawOverlay();
  }
  _mvp(M) {
    const r = M.r, sx = M.s * 2 / this.cssW, sy = M.s * 2 / this.cssH, sz = 1 / 1500;
    // column-major 4x4: NDC = S * R * p
    return new Float32Array([
      r[0] * sx, r[3] * sy, r[6] * sz, 0,
      r[1] * sx, r[4] * sy, r[7] * sz, 0,
      r[2] * sx, r[5] * sy, r[8] * sz, 0,
      0, 0, 0, 1]);
  }
  _drawGL() {
    const gl = this.gl;
    if (this.dirtyT) { gl.bindBuffer(gl.ARRAY_BUFFER, this.bT); gl.bufferSubData(gl.ARRAY_BUFFER, 0, this.lastT); this.dirtyT = false; }
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.uniformMatrix4fv(this.u.uMVP, false, this._mvp(this._matrix()));
    gl.uniform1f(this.u.uNow, this.nowSec);
    gl.uniform1f(this.u.uTau, 0.13);
    gl.uniform1f(this.u.uSize, 1.7 * this.dpr * Math.sqrt(Math.max(0.6, this.zoom)));
    gl.uniform1f(this.u.uBase, 0.3);
    gl.uniform1f(this.u.uGain, 1.0);
    gl.drawArrays(gl.POINTS, 0, this.m);
  }
  _draw2D(dt) {
    const g = this.bfx.getContext("2d"), W = this.canvas.width, H = this.canvas.height, M = this._matrix();
    if (this.pending.length) {
      const size = 2.2 * this.dpr;
      g.globalCompositeOperation = "lighter"; g.globalAlpha = 0.5;
      for (const list of this.pending) for (const i of list) {
        const k = this.slotOf[i]; if (k < 0) continue;
        const [x, y] = this._proj(k, M);
        g.fillStyle = REGION_COLORS[this.L.region[i]];
        g.fillRect(x * this.dpr - size / 2, y * this.dpr - size / 2, size, size);
      }
      g.globalAlpha = 1; g.globalCompositeOperation = "source-over";
      this.pending.length = 0;
    }
    g.globalCompositeOperation = "destination-out";
    g.fillStyle = `rgba(0,0,0,${Math.min(1, dt / 0.35)})`; g.fillRect(0, 0, W, H);
    g.globalCompositeOperation = "source-over";
    const c = this.ctx;
    c.drawImage(this.bbg, 0, 0);
    c.globalCompositeOperation = "lighter"; c.drawImage(this.bfx, 0, 0); c.globalCompositeOperation = "source-over";
  }
  _drawOverlay() {
    const o = this.overlay, c = o.getContext("2d"), dpr = this.dpr;
    const need = this.path || this.picked >= 0 || true;
    c.setTransform(dpr, 0, 0, dpr, 0, 0);
    c.clearRect(0, 0, this.cssW, this.cssH);
    if (!need) return;
    c.fillStyle = "rgba(231,237,244,.5)"; c.font = "10.5px system-ui";
    c.fillText("brain", 6, 14); c.fillText("nerve cord", 6, this.cssH * 0.66);
    if (this.mode3d && this.gl) { c.fillStyle = "rgba(139,152,169,.7)"; c.fillText(`yaw ${Math.round(((this.yaw * 180 / Math.PI) % 360 + 360) % 360)}°`, 6, this.cssH - 8); }
    const M = this._matrix();
    if (this.path) {
      const pts = this.path.map((i) => (i == null || this.slotOf[i] < 0) ? null : this._proj(this.slotOf[i], M));
      c.lineWidth = 2.2; c.strokeStyle = "rgba(255,209,102,.9)"; c.shadowColor = "#ffd166"; c.shadowBlur = 8;
      c.beginPath(); let started = false;
      for (const p of pts) { if (!p) continue; if (!started) { c.moveTo(p[0], p[1]); started = true; } else c.lineTo(p[0], p[1]); }
      c.stroke(); c.shadowBlur = 0;
      pts.forEach((p, j) => {
        if (!p) return;
        c.fillStyle = j === 0 ? "#4de2c5" : j === pts.length - 1 ? "#ff5d8f" : "#ffd166";
        c.beginPath(); c.arc(p[0], p[1], 4, 0, 2 * Math.PI); c.fill();
        c.strokeStyle = "#0a0e13"; c.lineWidth = 1; c.stroke();
        if (this.pathNames && this.pathNames[j]) {
          c.font = "600 10px system-ui"; c.fillStyle = "rgba(10,14,19,.75)";
          const w = c.measureText(this.pathNames[j]).width + 6;
          const lx = Math.max(2, Math.min(this.cssW - w - 2, p[0] + 6)), ly = Math.max(14, Math.min(this.cssH - 2, p[1] - 2));
          c.fillRect(lx, ly - 10, w, 13);
          c.fillStyle = "#ffe8b0"; c.fillText(this.pathNames[j], lx + 3, ly);
        }
      });
    }
    if (this.picked >= 0 && this.slotOf[this.picked] >= 0) {
      const [x, y] = this._proj(this.slotOf[this.picked], M);
      c.strokeStyle = "#fff"; c.lineWidth = 1.5; c.beginPath(); c.arc(x, y, 6, 0, 2 * Math.PI); c.stroke();
      c.strokeStyle = "rgba(255,255,255,.4)"; c.beginPath(); c.arc(x, y, 11, 0, 2 * Math.PI); c.stroke();
    }
  }
}
