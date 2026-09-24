// retina.js — what the two compound eyes see: one dot per facet at its (azimuth, elevation).
"use strict";

export class RetinaView {
  constructor(canvas, layout) {
    this.canvas = canvas; this.ctx = canvas.getContext("2d");
    this.lay = layout; this.dpr = window.devicePixelRatio || 1;
    this.panelW = 112; this.panelH = 54; this.gap = 8;
    const W = this.panelW * 2 + this.gap, H = this.panelH;
    canvas.width = Math.round(W * this.dpr); canvas.height = Math.round(H * this.dpr);
    canvas.style.width = W + "px"; canvas.style.height = H + "px";
    this.pts = {};
    for (const side of ["L", "R"]) {
      const e = layout[side]; if (!e) continue;
      const az = e.az, el = e.el, n = az.length;
      let azMin = Infinity, azMax = -Infinity, elMin = Infinity, elMax = -Infinity;
      for (let i = 0; i < n; i++) { azMin = Math.min(azMin, az[i]); azMax = Math.max(azMax, az[i]); elMin = Math.min(elMin, el[i]); elMax = Math.max(elMax, el[i]); }
      const xs = new Float32Array(n), ys = new Float32Array(n), x0 = side === "L" ? 0 : this.panelW + this.gap;
      const pad = 3;
      for (let i = 0; i < n; i++) {
        // the left eye looks left: its most lateral facets go to the far left, the frontal ones to the middle
        const u = (az[i] - azMin) / Math.max(1e-6, azMax - azMin);
        const fx = side === "L" ? 1 - u : u;     // right eye: az is negative, most lateral = azMin = far right
        xs[i] = x0 + pad + fx * (this.panelW - 2 * pad);
        ys[i] = pad + (1 - (el[i] - elMin) / Math.max(1e-6, elMax - elMin)) * (this.panelH - 2 * pad);
      }
      this.pts[side] = { xs, ys, n, x0, cell: Math.max(2.5, (this.panelW - 2 * pad) / (e.n_az || 26) * 0.95) };
    }
    this.bytes = { L: null, R: null };
    this.decode = (b64) => { const s = atob(b64), a = new Uint8Array(s.length); for (let i = 0; i < s.length; i++) a[i] = s.charCodeAt(i); return a; };
  }
  update(retina, senses) {
    if (!retina) return;
    const c = this.ctx, dpr = this.dpr;
    c.setTransform(dpr, 0, 0, dpr, 0, 0);
    c.clearRect(0, 0, this.canvas.width, this.canvas.height);
    for (const side of ["L", "R"]) {
      const p = this.pts[side]; if (!p || !retina[side]) continue;
      let bytes; try { bytes = this.decode(retina[side]); } catch (e) { continue; }
      // panel background and border (tinted when this eye reports a small object or looming)
      const small = ((senses && senses.small) || "").includes(side), loom = ((senses && senses.loom) || "").includes(side);
      c.fillStyle = "#0b1016"; c.fillRect(p.x0, 0, this.panelW, this.panelH);
      const cell = p.cell, n = Math.min(p.n, bytes.length);
      for (let i = 0; i < n; i++) {
        const v = bytes[i];
        c.fillStyle = `rgb(${v},${v},${Math.min(255, v + 18)})`;
        c.fillRect(p.xs[i] - cell / 2, p.ys[i] - cell / 2, cell, cell);
      }
      c.strokeStyle = loom ? "#ff5d5d" : small ? "#4de2c5" : "#223042"; c.lineWidth = loom || small ? 1.5 : 1;
      c.strokeRect(p.x0 + 0.5, 0.5, this.panelW - 1, this.panelH - 1);
    }
    // the midline marker between the eyes
    c.fillStyle = "#3a4a60"; c.fillRect(this.panelW + this.gap / 2 - 0.5, 4, 1, this.panelH - 8);
  }
}
