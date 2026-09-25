// arena.js — the petri dish: world millimetres -> pixels (y up), the fly, the female, food, odour plumes, wind.
"use strict";

const FOOD_COLS = {
  sugar: ["#ffe3a3", "#f2b134", "#9b6a10"],
  bitter: ["#dcc9ff", "#a77bff", "#5b3aa6"],
  water: ["#d8eeff", "#4da3ff", "#1b4f8a"],
};
export const TOOL_COLS = { sugar: "#f2b134", bitter: "#a77bff", water: "#4da3ff", dust: "#c9d4e2", post: "#8b98a9", shock: "#ffd166" };

export function lerpAngle(a, b, t) { let d = ((b - a + Math.PI) % (2 * Math.PI)) - Math.PI; if (d < -Math.PI) d += 2 * Math.PI; return a + d * t; }
export const wrap = (a) => { a = (a + Math.PI) % (2 * Math.PI); if (a < 0) a += 2 * Math.PI; return a - Math.PI; };

export class Arena {
  constructor(canvas, stage, L) {
    this.canvas = canvas; this.stage = stage; this.L = L; this.ctx = canvas.getContext("2d");
    this.R = L.arena_r || 50; this.w = this.h = 600; this.base = 6; this.scale = 6; this.dpr = 1;
    this.zoom = 1; this.cam = { x: 0, y: 0 };         // the view: zoom 1 shows the whole dish; above it the camera follows the fly
    this.particles = []; this.windDots = []; this.shockUntil = 0; this.clapAt = 0;
    this.sprites = {};
    this.odourColour = {}; for (const o of L.odours || []) this.odourColour[o.id] = o.colour;
    this.resize();
  }
  resize() {
    // the canvas fills the stage; the dish is drawn centred, as big as the shorter side allows
    const st = this.stage.getBoundingClientRect();
    this.dpr = window.devicePixelRatio || 1;
    this.w = Math.max(280, Math.floor(st.width || 280)); this.h = Math.max(280, Math.floor(st.height || st.width || 280));
    this.canvas.width = Math.round(this.w * this.dpr); this.canvas.height = Math.round(this.h * this.dpr);
    this.canvas.style.width = this.w + "px"; this.canvas.style.height = this.h + "px";
    this.base = (Math.min(this.w, this.h) / 2 - 10) / this.R;
    this.scale = this.base * this.zoom;
  }
  /** Zoom the view (1 = the whole dish, up to 4x); above 1 the camera follows the fly. */
  setZoom(z) {
    this.zoom = Math.max(1, Math.min(4, z));
    this.scale = this.base * this.zoom;
    if (this.zoom === 1) this.cam = { x: 0, y: 0 };
    return this.zoom;
  }
  _camera(dt, pose) {
    if (this.zoom <= 1) { this.cam.x = this.cam.y = 0; return; }
    // keep the fly in the middle, but never look past the wall: the camera centre stays at least half the
    // shorter canvas side from the wall in every direction (a radial limit, then a per-axis one for the
    // long axis, which may already show the whole dish)
    const lim = Math.max(0, this.R - Math.min(this.w, this.h) / 2 / this.scale);
    let tx = pose ? pose.x : 0, ty = pose ? pose.y : 0;
    const d = Math.hypot(tx, ty);
    if (d > lim) { tx *= lim / d; ty *= lim / d; }
    const lx = Math.max(0, this.R - this.w / 2 / this.scale), ly = Math.max(0, this.R - this.h / 2 / this.scale);
    tx = Math.max(-lx, Math.min(lx, tx)); ty = Math.max(-ly, Math.min(ly, ty));
    const k = Math.min(1, dt * 4);
    this.cam.x += (tx - this.cam.x) * k; this.cam.y += (ty - this.cam.y) * k;
  }
  W2C(x, y) { return [this.w / 2 + (x - this.cam.x) * this.scale, this.h / 2 - (y - this.cam.y) * this.scale]; }
  C2W(px, py) { return [(px - this.w / 2) / this.scale + this.cam.x, -(py - this.h / 2) / this.scale + this.cam.y]; }

  sprite(col) {
    if (this.sprites[col]) return this.sprites[col];
    const s = document.createElement("canvas"); s.width = s.height = 64;
    const g = s.getContext("2d"), grd = g.createRadialGradient(32, 32, 2, 32, 32, 32);
    grd.addColorStop(0, col + "70"); grd.addColorStop(0.5, col + "30"); grd.addColorStop(1, col + "00");
    g.fillStyle = grd; g.fillRect(0, 0, 64, 64);
    return (this.sprites[col] = s);
  }

  // ---------------------------------------------------------------- effects
  puff(x, y) {
    for (let i = 0; i < 70; i++) {
      const a = Math.random() * 2 * Math.PI, v = 4 + Math.random() * 14;
      this.particles.push({ x, y, vx: Math.cos(a) * v, vy: Math.sin(a) * v, life: 0.8 + Math.random() * 0.7, age: 0, r: 0.25 + Math.random() * 0.5 });
    }
  }
  shock() { this.shockUntil = performance.now() + 500; }
  clap() { this.clapAt = performance.now(); }

  // ---------------------------------------------------------------- main draw
  draw(dt, view) {
    const { S, pose, female, pointer, tool, sees, hz, stripes } = view, c = this.ctx, R = this.R;
    c.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    c.clearRect(0, 0, this.w, this.h);
    this._camera(dt, pose);
    const [cx, cy] = this.W2C(0, 0), rr = R * this.scale;
    const grd = c.createRadialGradient(cx, cy - rr * 0.3, rr * 0.1, cx, cy, rr);
    grd.addColorStop(0, "#222d3b"); grd.addColorStop(1, "#141b25");
    c.fillStyle = grd; c.beginPath(); c.arc(cx, cy, rr, 0, 2 * Math.PI); c.fill();
    c.strokeStyle = "rgba(255,255,255,.035)"; c.lineWidth = 1;
    for (let k = 1; k < 4; k++) { c.beginPath(); c.arc(cx, cy, (rr * k) / 4, 0, 2 * Math.PI); c.stroke(); }
    if (!S) return;
    const w = S.world || {};
    // plume, odour sources, wind streaks (under everything else)
    c.save(); c.beginPath(); c.arc(cx, cy, rr, 0, 2 * Math.PI); c.clip();
    if (stripes && stripes.count > 0) this.drawStripes(stripes.count, stripes.phase);
    this.drawWindDots(dt, w.wind);
    for (const p of w.puffs || []) this.drawPuff(p);
    for (const o of w.odours || []) this.drawOdourSource(o);
    c.restore();
    // the wall
    c.strokeStyle = "#3a4a60"; c.lineWidth = 3; c.beginPath(); c.arc(cx, cy, rr, 0, 2 * Math.PI); c.stroke();
    c.strokeStyle = "rgba(255,255,255,.08)"; c.lineWidth = 1; c.beginPath(); c.arc(cx, cy, rr - 4, Math.PI * 1.1, Math.PI * 1.6); c.stroke();
    for (const o of w.obstacles || []) this.drawObstacle(o);
    for (const f of w.food || []) this.drawFood(f);
    this.drawParticles(dt);
    if (female) this.drawFly(female, { female: true });
    if (pose) {
      if (sees) this.drawSees(pose, S, w, female);
      this.drawFly(pose, { female: false, hz });
      if (performance.now() < this.shockUntil) this.drawShock(pose);
    }
    if (w.wind && w.wind.speed > 0) this.drawWindArrow(w.wind);
    if (pose && performance.now() - this.clapAt < 600) this.drawClap(pose);
    if (pointer && pointer.inside) this.drawPointer(pointer.px, pointer.py, tool, pointer.ang || 0, pointer.speed || 0, pointer.food);
    else if (w.hand && (w.tool === "lure" || w.tool === "hand")) {
      const [px, py] = this.W2C(w.hand[0], w.hand[1]); this.drawPointer(px, py, w.tool, view.handAng || 0, 200, "");
    }
  }

  drawPuff(p) {
    const [x, y] = this.W2C(p[0], p[1]), r = p[2] * this.scale, col = this.odourColour[p[4]] || "#9be15d";
    const c = this.ctx; c.globalAlpha = Math.min(1, p[3]) * 0.28;
    c.drawImage(this.sprite(col), x - r, y - r, 2 * r, 2 * r);
    c.globalAlpha = 1;
  }
  drawOdourSource(o) {
    const c = this.ctx, [x, y] = this.W2C(o.x, o.y), k = this.scale, col = this.odourColour[o.odour] || "#9be15d";
    c.globalAlpha = 0.1; c.fillStyle = col; c.beginPath(); c.arc(x, y, 6 * k, 0, 2 * Math.PI); c.fill(); c.globalAlpha = 1;
    c.strokeStyle = col; c.lineWidth = 2; c.beginPath(); c.arc(x, y, 2.4 * k, 0, 2 * Math.PI); c.stroke();
    c.fillStyle = col; c.beginPath(); c.arc(x, y, 0.8 * k, 0, 2 * Math.PI); c.fill();
    c.strokeStyle = col + "66"; c.lineWidth = 1; c.setLineDash([2, 3]); c.beginPath(); c.arc(x, y, 3.6 * k, 0, 2 * Math.PI); c.stroke(); c.setLineDash([]);
  }
  drawStripes(count, phase) {
    // the optomotor drum: alternating dark and bright bands painted on the inside of the wall
    const c = this.ctx, [cx, cy] = this.W2C(0, 0), r1 = (this.R - 3.2) * this.scale, r0 = this.R * this.scale + 2, step = 2 * Math.PI / count;
    for (let i = 0; i < count; i++) {
      const a0 = -(phase + i * step), a1 = a0 - step;      // world angles are CCW, the canvas y axis points down
      c.fillStyle = i % 2 ? "#cfd8e3" : "#0d1218";
      c.beginPath(); c.arc(cx, cy, r0, a0, a1, true); c.arc(cx, cy, r1, a1, a0, false); c.closePath(); c.fill();
    }
    c.strokeStyle = "rgba(0,0,0,.35)"; c.lineWidth = 1; c.beginPath(); c.arc(cx, cy, r1, 0, 2 * Math.PI); c.stroke();
  }
  drawClap(f) {
    const c = this.ctx, [X, Y] = this.W2C(f.x, f.y), k = this.scale, u = (performance.now() - this.clapAt) / 600;
    c.strokeStyle = `rgba(143,184,255,${0.7 * (1 - u)})`; c.lineWidth = 2;
    for (let j = 0; j < 3; j++) { const r = (4 + 26 * ((u + j * 0.33) % 1)) * k; c.beginPath(); c.arc(X, Y, r, 0, 2 * Math.PI); c.stroke(); }
  }
  drawObstacle(o) {
    const c = this.ctx, [x, y] = this.W2C(o.x, o.y), r = o.r * this.scale;
    c.fillStyle = "rgba(0,0,0,.35)"; c.beginPath(); c.ellipse(x + r * 0.18, y + r * 0.22, r, r * 0.95, 0, 0, 2 * Math.PI); c.fill();
    c.fillStyle = "#1a2230"; c.beginPath(); c.arc(x, y, r, 0, 2 * Math.PI); c.fill();
    const g = c.createRadialGradient(x - r * 0.3, y - r * 0.35, r * 0.1, x, y, r * 0.85);
    g.addColorStop(0, "#4a5a72"); g.addColorStop(1, "#2a3547");
    c.fillStyle = g; c.beginPath(); c.arc(x - r * 0.08, y - r * 0.1, r * 0.82, 0, 2 * Math.PI); c.fill();
    c.strokeStyle = "rgba(255,255,255,.12)"; c.lineWidth = 1; c.beginPath(); c.arc(x, y, r, 0, 2 * Math.PI); c.stroke();
  }
  drawFood(f) {
    const c = this.ctx, [x, y] = this.W2C(f.x, f.y), r = f.r * this.scale, col = FOOD_COLS[f.kind] || FOOD_COLS.sugar;
    const g = c.createRadialGradient(x - r * 0.35, y - r * 0.35, r * 0.1, x, y, r);
    g.addColorStop(0, col[0]); g.addColorStop(0.55, col[1]); g.addColorStop(1, col[2]);
    c.fillStyle = "rgba(0,0,0,.25)"; c.beginPath(); c.ellipse(x + r * 0.15, y + r * 0.2, r, r * 0.9, 0, 0, 2 * Math.PI); c.fill();
    c.fillStyle = g; c.beginPath(); c.arc(x, y, r, 0, 2 * Math.PI); c.fill();
    c.fillStyle = "rgba(255,255,255,.55)"; c.beginPath(); c.ellipse(x - r * 0.35, y - r * 0.4, r * 0.28, r * 0.16, -0.6, 0, 2 * Math.PI); c.fill();
  }
  drawParticles(dt) {
    const c = this.ctx, ps = this.particles;
    for (let i = ps.length - 1; i >= 0; i--) {
      const p = ps[i]; p.age += dt; p.x += p.vx * dt; p.y += p.vy * dt; p.vx *= 0.93; p.vy *= 0.93;
      if (p.age > p.life) { ps.splice(i, 1); continue; }
      const [x, y] = this.W2C(p.x, p.y);
      c.fillStyle = `rgba(214,222,232,${0.7 * (1 - p.age / p.life)})`;
      c.beginPath(); c.arc(x, y, p.r * this.scale, 0, 2 * Math.PI); c.fill();
    }
  }
  drawWindDots(dt, wind) {
    if (!wind || wind.speed <= 0) { this.windDots.length = 0; return; }
    const c = this.ctx, R = this.R, vx = wind.speed * Math.cos(wind.angle), vy = wind.speed * Math.sin(wind.angle);
    while (this.windDots.length < 70) { const a = Math.random() * 2 * Math.PI, r = R * Math.sqrt(Math.random()); this.windDots.push({ x: r * Math.cos(a), y: r * Math.sin(a) }); }
    const len = Math.min(5, 1 + wind.speed * 0.12);
    c.strokeStyle = "rgba(200,220,255,.13)"; c.lineWidth = 1; c.beginPath();
    for (const d of this.windDots) {
      d.x += vx * dt; d.y += vy * dt;
      if (Math.hypot(d.x, d.y) > R) { // respawn on the upwind rim
        const a = wind.angle + Math.PI + (Math.random() - 0.5) * 2.6; d.x = R * 0.98 * Math.cos(a); d.y = R * 0.98 * Math.sin(a);
      }
      const [x, y] = this.W2C(d.x, d.y), [x2, y2] = this.W2C(d.x - len * Math.cos(wind.angle), d.y - len * Math.sin(wind.angle));
      c.moveTo(x, y); c.lineTo(x2, y2);
    }
    c.stroke();
  }
  drawWindArrow(wind) {
    const c = this.ctx, x = 44, y = this.h - 44, len = 10 + Math.min(26, wind.speed * 0.7), a = -wind.angle;
    c.fillStyle = "rgba(10,14,19,.7)"; c.beginPath(); c.arc(x, y, 30, 0, 2 * Math.PI); c.fill();
    c.strokeStyle = "rgba(143,184,255,.35)"; c.lineWidth = 1; c.beginPath(); c.arc(x, y, 30, 0, 2 * Math.PI); c.stroke();
    c.save(); c.translate(x, y); c.rotate(a);
    c.strokeStyle = "#8fb8ff"; c.lineWidth = 2.5; c.lineCap = "round";
    c.beginPath(); c.moveTo(-len / 2, 0); c.lineTo(len / 2, 0); c.moveTo(len / 2 - 6, -5); c.lineTo(len / 2, 0); c.lineTo(len / 2 - 6, 5); c.stroke();
    c.restore();
    c.fillStyle = "#8fb8ff"; c.font = "600 10px system-ui"; c.textAlign = "center"; c.textBaseline = "top";
    c.fillText(`wind ${Math.round(wind.speed)} mm/s`, x, y + 33);
  }
  drawShock(f) {
    const c = this.ctx, [X, Y] = this.W2C(f.x, f.y), k = this.scale, t = performance.now();
    c.save(); c.translate(X, Y); c.strokeStyle = "#ffd166"; c.lineWidth = 2; c.shadowColor = "#ffd166"; c.shadowBlur = 10;
    for (let j = 0; j < 6; j++) {
      const a = j * Math.PI / 3 + t / 90; c.beginPath(); c.moveTo(Math.cos(a) * 6 * k, Math.sin(a) * 6 * k);
      c.lineTo(Math.cos(a + 0.25) * 8 * k, Math.sin(a + 0.25) * 8 * k); c.lineTo(Math.cos(a - 0.1) * 10.5 * k, Math.sin(a - 0.1) * 10.5 * k); c.stroke();
    }
    c.restore();
  }
  drawSees(pose, S, w, female) {
    const sm = (S.senses && S.senses.small) || "", lo = (S.senses && S.senses.loom) || "";
    if (!sm && !lo) return;
    const c = this.ctx, cands = [];
    if (w.hand && (w.tool === "lure" || w.tool === "hand")) cands.push({ x: w.hand[0], y: w.hand[1], kind: w.tool });
    if (female) cands.push({ x: female.x, y: female.y, kind: "fly" });
    for (const o of w.obstacles || []) cands.push({ x: o.x, y: o.y, kind: "post" });
    for (const o of cands) {
      const b = wrap(Math.atan2(o.y - pose.y, o.x - pose.x) - pose.h), side = b > 0 ? "L" : "R";
      const small = sm.includes(side), loom = lo.includes(side);
      if (!small && !loom) continue;
      const s = side === "L" ? 1 : -1;
      const ex = pose.x + 2.9 * Math.cos(pose.h) - s * 0.65 * Math.sin(pose.h), ey = pose.y + 2.9 * Math.sin(pose.h) + s * 0.65 * Math.cos(pose.h);
      const [x1, y1] = this.W2C(ex, ey), [x2, y2] = this.W2C(o.x, o.y);
      c.strokeStyle = loom ? "rgba(255,93,93,.55)" : "rgba(77,226,197,.5)"; c.lineWidth = 1; c.setLineDash([4, 4]);
      c.beginPath(); c.moveTo(x1, y1); c.lineTo(x2, y2); c.stroke(); c.setLineDash([]);
      c.strokeStyle = loom ? "rgba(255,93,93,.7)" : "rgba(77,226,197,.7)"; c.beginPath(); c.arc(x2, y2, Math.max(8, 1.6 * this.scale), 0, 2 * Math.PI); c.stroke();
    }
  }

  // ---------------------------------------------------------------- pointer
  drawPointer(x, y, tool, ang, speed, food) {
    const c = this.ctx, k = this.scale;
    if (tool === "lure") {
      c.save(); c.translate(x, y); c.rotate(ang);
      c.fillStyle = "rgba(0,0,0,.35)"; c.beginPath(); c.ellipse(1.5, 2, 1.3 * k, 0.8 * k, 0, 0, 2 * Math.PI); c.fill();
      c.fillStyle = "rgba(210,225,240,.35)";
      c.beginPath(); c.ellipse(-0.6 * k, 0.55 * k, 1.1 * k, 0.4 * k, 0.5, 0, 2 * Math.PI); c.fill();
      c.beginPath(); c.ellipse(-0.6 * k, -0.55 * k, 1.1 * k, 0.4 * k, -0.5, 0, 2 * Math.PI); c.fill();
      c.fillStyle = "#11151b"; c.beginPath(); c.ellipse(0, 0, 1.2 * k, 0.55 * k, 0, 0, 2 * Math.PI); c.fill();
      c.fillStyle = "#b8322a"; c.beginPath(); c.arc(0.95 * k, 0.3 * k, 0.25 * k, 0, 2 * Math.PI); c.arc(0.95 * k, -0.3 * k, 0.25 * k, 0, 2 * Math.PI); c.fill();
      c.restore();
    } else if (tool === "hand") {
      const r = 6 * k, a = Math.min(0.55, 0.12 + speed / 400);
      const g = c.createRadialGradient(x, y, r * 0.2, x, y, r);
      g.addColorStop(0, `rgba(10,12,16,${a + 0.15})`); g.addColorStop(1, "rgba(10,12,16,0)");
      c.fillStyle = g; c.beginPath(); c.arc(x, y, r, 0, 2 * Math.PI); c.fill();
      c.font = `${Math.round(r * 0.9)}px system-ui`; c.textAlign = "center"; c.textBaseline = "middle";
      c.globalAlpha = 0.85; c.fillText("✋", x, y); c.globalAlpha = 1;
    } else if (tool === "shock") {
      c.font = `${Math.round(3.5 * k)}px system-ui`; c.textAlign = "center"; c.textBaseline = "middle";
      c.globalAlpha = 0.9; c.fillText("⚡", x, y); c.globalAlpha = 1;
    } else {
      const col = TOOL_COLS[tool] || this.odourColour[tool] || "#9be15d";
      const r = tool === "dust" ? 6 : tool === "post" ? 4 : tool in TOOL_COLS ? 3 : 2.4;
      c.strokeStyle = col; c.lineWidth = 1.5; c.setLineDash([3, 3]);
      c.beginPath(); c.arc(x, y, r * k, 0, 2 * Math.PI); c.stroke(); c.setLineDash([]);
      if (food) { c.fillStyle = TOOL_COLS[food]; c.beginPath(); c.arc(x, y, 1.2 * k, 0, 2 * Math.PI); c.fill(); }
    }
  }

  // ---------------------------------------------------------------- the fly, drawn top-down in mm (+x forward, +y its right on screen)
  drawFly(f, o) {
    const c = this.ctx, k = this.scale, [X, Y] = this.W2C(f.x, f.y), fem = !!o.female, t = performance.now() / 1000;
    const jump = f.jump == null ? 0 : Math.sin(Math.PI * f.jump), lift = 1 + 0.35 * jump;
    const walking = f.walking, mode = f.mode || "walk";
    const wl = f.wingL || 0, wr = f.wingR || 0, ab = f.abdomen || 0;
    const song = !fem && mode !== "escape" && f.jump == null && (wl > 0.35 || wr > 0.35);
    const sz = fem ? 1.32 : 1.25;
    c.save();
    c.translate(X, Y);
    if (fem && f.receptive > 0.05) {  // her interest: a warm glow
      const g = c.createRadialGradient(0, 0, 2 * k, 0, 0, 9 * k);
      g.addColorStop(0, `rgba(255,154,213,${0.35 * f.receptive})`); g.addColorStop(1, "rgba(255,154,213,0)");
      c.fillStyle = g; c.beginPath(); c.arc(0, 0, 9 * k, 0, 2 * Math.PI); c.fill();
    }
    c.fillStyle = `rgba(0,0,0,${0.35 - 0.15 * jump})`;
    c.beginPath(); c.ellipse(4 + 10 * jump, 5 + 10 * jump, 4.2 * k, 2.4 * k, -f.h, 0, 2 * Math.PI); c.fill();
    c.rotate(-f.h); c.scale(sz * k * lift, sz * k * lift);
    c.lineCap = "round"; c.lineJoin = "round";
    // legs: tripod gait (front-left + mid-right + hind-left swing together)
    const legs = [[1.0, 0.95, 1.9], [0.6, 0.0, 0.2], [0.2, -0.95, -1.9]];
    c.strokeStyle = fem ? "#3a2a1c" : "#2a1d14"; c.lineWidth = 0.16;
    legs.forEach(([ax, kneeX, footX], j) => {
      for (const side of [-1, 1]) {
        const group = (j % 2 === 0) === (side === -1) ? 0 : Math.PI;
        const sw = walking ? 0.45 * Math.sin(f.legs * 1.6 + group) : 0;
        let kx = kneeX + sw * 0.5, fx = footX + sw, ky = side * 1.55, fy = side * 2.35;
        if (j === 0 && mode === "groom") {       // front legs rub the head
          const g = Math.sin(f.groom * 1.6 + (side > 0 ? 0 : 1.2));
          kx = 1.7; ky = side * 0.9; fx = 2.9 + 0.35 * g; fy = side * (0.25 + 0.2 * g);
        }
        if (mode === "escape" || f.jump != null) { fx *= 1.15; fy *= 1.15; }
        c.beginPath(); c.moveTo(ax, side * 0.45); c.lineTo(kx, ky); c.lineTo(fx, fy); c.stroke();
      }
    });
    // wings: folded back at rest, out to 90 degrees when extended; a singing wing vibrates
    for (const side of [-1, 1]) {
      const e = side > 0 ? wr : wl;
      let ang = Math.PI - side * (0.2 + (Math.PI / 2 - 0.2) * e);
      if (song && e > 0.35) ang += side * 0.12 * Math.sin(t * 60);
      c.save(); c.translate(0.3, side * 0.2); c.rotate(ang);
      c.fillStyle = fem ? "rgba(215,225,240,.45)" : "rgba(205,220,235,.42)"; c.strokeStyle = "rgba(230,240,250,.55)"; c.lineWidth = 0.05;
      c.beginPath(); c.ellipse(2.0, 0, 2.1, 0.72, 0, 0, 2 * Math.PI); c.fill(); c.stroke();
      c.strokeStyle = "rgba(120,135,150,.5)"; c.beginPath(); c.moveTo(0.2, 0); c.lineTo(3.8, 0.1); c.moveTo(0.5, 0.2); c.lineTo(3.5, 0.5); c.stroke();
      c.restore();
    }
    // abdomen (shorter and curled when bent for a courtship attempt)
    const abLen = 1.6 * (1 - 0.3 * ab), abX = -1.5 + 0.45 * ab;
    let g = c.createLinearGradient(-3.1, 0, 0, 0);
    if (fem) { g.addColorStop(0, "#5a4230"); g.addColorStop(1, "#a88a63"); } else { g.addColorStop(0, "#3b2717"); g.addColorStop(1, "#8b6a45"); }
    c.fillStyle = g; c.beginPath(); c.ellipse(abX, 0, abLen, 0.95 + 0.1 * ab, 0, 0, 2 * Math.PI); c.fill();
    c.strokeStyle = fem ? "rgba(60,40,25,.6)" : "rgba(30,18,10,.8)"; c.lineWidth = 0.22;
    for (const sx of [-2.3, -1.7, -1.1]) { const px = abX + (sx + 1.5) * (abLen / 1.6); c.beginPath(); c.ellipse(px, 0, 0.18, 0.85, 0, -Math.PI / 2, Math.PI / 2); c.stroke(); }
    if (ab > 0.1) { c.fillStyle = "#2a1a10"; c.beginPath(); c.ellipse(abX - abLen + 0.1, 0.15 * ab, 0.35, 0.3, 0, 0, 2 * Math.PI); c.fill(); }
    // thorax
    g = c.createRadialGradient(0.8, -0.2, 0.1, 0.6, 0, 1.2);
    if (fem) { g.addColorStop(0, "#c9a878"); g.addColorStop(1, "#8a6a48"); } else { g.addColorStop(0, "#b08a5a"); g.addColorStop(1, "#6d4f31"); }
    c.fillStyle = g; c.beginPath(); c.ellipse(0.6, 0, 1.05, 0.85, 0, 0, 2 * Math.PI); c.fill();
    // proboscis
    if (f.prob > 0.02) {
      c.strokeStyle = "#8a5a3a"; c.lineWidth = 0.32;
      c.beginPath(); c.moveTo(2.8, 0); c.lineTo(2.8 + 1.5 * f.prob, 0); c.stroke();
      c.fillStyle = "#b07a50"; c.beginPath(); c.ellipse(2.85 + 1.5 * f.prob, 0, 0.22, 0.34, 0, 0, 2 * Math.PI); c.fill();
    }
    // head, eyes, antennae
    c.fillStyle = fem ? "#b2916a" : "#9a7a52"; c.beginPath(); c.ellipse(2.25, 0, 0.62, 0.75, 0, 0, 2 * Math.PI); c.fill();
    for (const side of [-1, 1]) {
      const eg = c.createRadialGradient(2.35, side * 0.5, 0.05, 2.3, side * 0.52, 0.5);
      eg.addColorStop(0, "#ff6b57"); eg.addColorStop(1, "#8e1c12");
      c.fillStyle = eg; c.beginPath(); c.ellipse(2.3, side * 0.52, 0.42, 0.36, 0, 0, 2 * Math.PI); c.fill();
      c.strokeStyle = "#5d4630"; c.lineWidth = 0.08; c.beginPath(); c.moveTo(2.8, side * 0.12); c.lineTo(3.15, side * 0.35); c.stroke();
    }
    c.restore();
    // labels
    if (fem) {
      if (f.receptive > 0.5) {
        c.font = `${Math.round(9 + 8 * f.receptive)}px system-ui`; c.textAlign = "center"; c.textBaseline = "bottom";
        c.fillStyle = `rgba(255,120,180,${0.5 + 0.5 * f.receptive})`; c.fillText("♥", X + 4 * k, Y - 5 * k - 2 * Math.sin(t * 3));
      }
      c.font = "600 10px system-ui"; c.textAlign = "center"; c.textBaseline = "bottom"; c.fillStyle = "rgba(255,154,213,.7)";
      c.fillText("♀", X, Y - 6.5 * k);
      return;
    }
    const label = (mode === "escape" || f.jump != null) ? "ESCAPE!" : song ? "♪ singing" : mode === "feed" ? "eating" :
      mode === "groom" ? "grooming" : mode === "backward" ? "backing up" : mode === "court" ? "courting" : "";
    if (label) {
      c.font = "600 12px system-ui"; c.textAlign = "center"; c.textBaseline = "bottom";
      const w = c.measureText(label).width + 12;
      c.fillStyle = "rgba(10,14,19,.8)"; c.fillRect(X - w / 2, Y - 6.2 * k - 18, w, 17);
      c.fillStyle = mode === "escape" ? "#ff8a8a" : song ? "#ff9ad5" : "#e7edf4"; c.fillText(label, X, Y - 6.2 * k - 3);
    }
  }
}
