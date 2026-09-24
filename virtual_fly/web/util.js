// util.js — tiny helpers shared by the page's modules.
"use strict";

export const $ = (id) => document.getElementById(id);

/** Set an element's text only when it changed (the state handler runs 40 times a second). */
export function setText(el, s) { if (el && el.__t !== s) { el.__t = s; el.textContent = s; } }
/** Set a bar fill's length in percent, rounded to half a percent, only when it changed.
 *  The fill is scaled with a transform rather than resized: a transform is animated by the
 *  compositor and never makes the page lay itself out again (thirty bars did, forty times a second). */
export function setWidth(el, pct) {
  const v = Math.max(0, Math.min(100, Math.round(pct * 2) / 2));
  if (el && el.__w !== v) { el.__w = v; el.style.transform = `scaleX(${v / 100})`; }
}
export function setClass(el, cls, on) { if (el && el.__c?.[cls] !== !!on) { (el.__c ??= {})[cls] = !!on; el.classList.toggle(cls, !!on); } }
export function setShown(el, on) { if (el && el.__s !== !!on) { el.__s = !!on; el.hidden = !on; } }

export const esc = (s) => String(s).replace(/[&<>"']/g, (ch) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[ch]));
export const fmt = (v, d = 1) => (v == null || isNaN(v)) ? "–" : Number(v).toFixed(d);
export const pct = (v) => Math.round(100 * v) + " %";

/** POST an action to the game. Never throws. */
export async function post(data) {
  try {
    const r = await fetch("api/action", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(data) });
    return await r.json();
  } catch (e) { return { ok: false, error: "no connection" }; }
}
export async function getJSON(url) {
  try { const r = await fetch(url, { cache: "no-store" }); return await r.json(); }
  catch (e) { return { ok: false, error: "no connection" }; }
}

export function debounce(fn, ms) { let t = 0; return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); }; }
export function el(tag, cls, html) { const e = document.createElement(tag); if (cls) e.className = cls; if (html != null) e.innerHTML = html; return e; }
