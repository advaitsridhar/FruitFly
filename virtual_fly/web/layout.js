// layout.js — the panel manager: hide the side panels so the dish takes the screen, choose which
// panels show, and drag any panel by its title to reorder it in the sidebar or to float it over
// the dish and put it wherever you like. The arrangement is remembered in this browser.
"use strict";
import { $, el, esc } from "./util.js";

const KEY = "vf.layout.v1";
const DRAG_START = 6;                 // px of movement before a press becomes a drag
const FLOAT_W = 400;

function titleOf(card) {
  const h2 = card.querySelector(":scope > h2");
  const first = h2 && (h2.querySelector("span") || h2);
  return ((first && first.textContent) || card.id || "panel").trim();
}
function readStore() {
  try { const s = JSON.parse(localStorage.getItem(KEY) || "null"); return s && typeof s === "object" ? s : null; } catch (e) { return null; }
}
function writeStore(state) {
  try { localStorage.setItem(KEY, JSON.stringify(state)); } catch (e) { /* private window or storage blocked: the layout just is not remembered */ }
}
const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));

export class PanelManager {
  constructor(aside, button, menu) {
    this.aside = aside; this.button = button; this.menu = menu;
    this.cards = [...aside.querySelectorAll(":scope > .card")];
    this.cards.forEach((c, i) => { if (!c.id) c.id = `card${i}`; c.dataset.title = titleOf(c); this.decorate(c); });
    this.byId = Object.fromEntries(this.cards.map((c) => [c.id, c]));
    this.zTop = 30;
    this.drag = null;
    const saved = readStore();
    this.state = { sidebar: true, order: this.cards.map((c) => c.id), hidden: [], floating: {} };
    if (saved) {
      if (typeof saved.sidebar === "boolean") this.state.sidebar = saved.sidebar;
      if (Array.isArray(saved.order)) this.state.order = [...saved.order.filter((id) => this.byId[id]), ...this.state.order.filter((id) => !saved.order.includes(id))];
      if (Array.isArray(saved.hidden)) this.state.hidden = saved.hidden.filter((id) => this.byId[id]);
      if (saved.floating && typeof saved.floating === "object") for (const [id, p] of Object.entries(saved.floating)) if (this.byId[id] && p && typeof p === "object") this.state.floating[id] = p;
    }
    this.sizeWatch = new ResizeObserver((entries) => {
      for (const en of entries) {
        const card = en.target, p = this.state.floating[card.id];
        if (p && card.classList.contains("floating") && !this.drag) { p.w = Math.round(card.offsetWidth); p.h = Math.round(card.offsetHeight); this.saveSoon(); }
      }
    });
    this.apply();
    this.buildMenu();
    button.onclick = (e) => { e.stopPropagation(); this.refreshMenu(); menu.hidden ? (menu.hidden = false) : this.closeMenu(); };
    document.addEventListener("click", (e) => { if (!menu.hidden && !e.target.closest(".menuwrap")) this.closeMenu(); });
    window.addEventListener("keydown", (e) => { if (e.key === "Escape") this.closeMenu(); });
    window.addEventListener("resize", () => this.keepOnScreen());
  }

  closeMenu() {
    this.menu.hidden = true;
    // a checkbox in the menu keeps the focus after a click, which would swallow the keyboard shortcuts
    if (document.activeElement && this.menu.contains(document.activeElement)) document.activeElement.blur();
  }

  // ---------------------------------------------------------------- per-card controls
  decorate(card) {
    const h2 = card.querySelector(":scope > h2");
    if (!h2) return;
    const ctl = el("span", "pctl");
    const fl = el("button", "", "⧉"); fl.type = "button"; fl.dataset.act = "float"; fl.title = "Pop this panel out over the dish (drag its title to move it; the same button docks it again)";
    const hd = el("button", "", "×"); hd.type = "button"; hd.dataset.act = "hide"; hd.title = "Hide this panel (bring it back from the Panels menu)";
    ctl.append(fl, hd);
    h2.appendChild(ctl);
    fl.onclick = (e) => { e.stopPropagation(); this.isFloating(card) ? this.dock(card) : this.float(card); this.save(); };
    hd.onclick = (e) => { e.stopPropagation(); this.hide(card, true); this.save(); };
    h2.addEventListener("pointerdown", (e) => this.pointerDown(e, card, h2));
    card.addEventListener("pointerdown", () => { if (this.isFloating(card)) card.style.zIndex = ++this.zTop; });
  }
  isFloating(card) { return card.classList.contains("floating"); }
  isHidden(card) { return this.state.hidden.includes(card.id); }

  // ---------------------------------------------------------------- state -> DOM
  apply() {
    for (const id of this.state.order) {                  // floating cards leave the sidebar ...
      const card = this.byId[id];
      if (this.state.floating[id]) this.float(card, this.state.floating[id]); else if (this.isFloating(card)) this.dock(card, false);
    }
    for (const id of this.state.order) {                  // ... and the rest line up in the saved order
      const card = this.byId[id];
      if (!this.isFloating(card)) this.aside.appendChild(card);
    }
    for (const card of this.cards) card.hidden = this.isHidden(card);
    document.body.classList.toggle("nosidebar", !this.state.sidebar);
  }
  float(card, pos) {
    const r = card.getBoundingClientRect();
    if (!this.isFloating(card)) {
      card.classList.add("floating");
      document.body.appendChild(card);
      card.style.zIndex = ++this.zTop;
    }
    const p = pos || this.state.floating[card.id] || { x: Math.max(8, r.left - 40), y: Math.max(8, r.top), w: Math.max(FLOAT_W, Math.round(r.width)) };
    this.state.floating[card.id] = p;
    card.style.left = `${p.x}px`; card.style.top = `${p.y}px`;
    card.style.width = `${p.w || FLOAT_W}px`;
    card.style.height = p.h ? `${p.h}px` : "";
    this.keepOnScreen(card);
    this.sizeWatch.observe(card);
    card.querySelector('.pctl [data-act="float"]').textContent = "⇤";
    this.refreshMenu();
  }
  dock(card, reposition = true) {
    delete this.state.floating[card.id];
    this.sizeWatch.unobserve(card);
    card.classList.remove("floating");
    card.style.left = card.style.top = card.style.width = card.style.height = card.style.maxHeight = card.style.zIndex = "";
    // back into the sidebar at its place in the order
    const order = this.state.order, k = order.indexOf(card.id);
    const next = order.slice(k + 1).map((id) => this.byId[id]).find((c) => c.parentElement === this.aside);
    if (next) this.aside.insertBefore(card, next); else this.aside.appendChild(card);
    card.querySelector('.pctl [data-act="float"]').textContent = "⧉";
    if (reposition) this.refreshMenu();
  }
  hide(card, on) {
    const h = new Set(this.state.hidden);
    on ? h.add(card.id) : h.delete(card.id);
    this.state.hidden = [...h];
    card.hidden = on;
    this.refreshMenu();
  }
  setSidebar(on) {
    this.state.sidebar = !!on;
    document.body.classList.toggle("nosidebar", !this.state.sidebar);
    this.refreshMenu();
    this.save();
  }
  toggleSidebar() { this.setSidebar(!this.state.sidebar); }
  showAll() { for (const c of this.cards) this.hide(c, false); this.setSidebar(true); }
  dockAll() { for (const c of this.cards) if (this.isFloating(c)) this.dock(c); this.save(); }
  reset() {
    this.state = { sidebar: true, order: this.cards.map((c) => c.id), hidden: [], floating: {} };
    this.apply();
    this.refreshMenu();
    this.save();
  }
  keepOnScreen(one) {
    for (const card of one ? [one] : this.cards) {
      if (!this.isFloating(card)) continue;
      const p = this.state.floating[card.id]; if (!p) continue;
      const w = card.offsetWidth || p.w || FLOAT_W;
      p.x = clamp(p.x, 8 - w + 60, window.innerWidth - 60);
      p.y = clamp(p.y, 8, Math.max(8, window.innerHeight - 40));
      card.style.left = `${p.x}px`; card.style.top = `${p.y}px`;
      card.style.maxHeight = `${Math.max(120, window.innerHeight - p.y - 8)}px`;     // never past the bottom edge: it scrolls instead
    }
  }
  save() {
    this.state.order = [...this.aside.querySelectorAll(":scope > .card")].map((c) => c.id)
      .concat(this.state.order.filter((id) => this.byId[id].parentElement !== this.aside));
    writeStore(this.state);
  }
  saveSoon() { clearTimeout(this._t); this._t = setTimeout(() => this.save(), 300); }

  // ---------------------------------------------------------------- the Panels menu
  buildMenu() {
    const m = this.menu; m.innerHTML = "";
    const side = el("label", "", `<input type="checkbox" id="sidebarToggle"> side panels <span class="float" title="keyboard: H">H</span>`);
    side.querySelector("input").onchange = (e) => this.setSidebar(e.target.checked);
    m.appendChild(side);
    m.appendChild(el("div", "sep"));
    this.menuRows = {};
    for (const id of this.state.order) {
      const card = this.byId[id];
      const row = el("label", "", `<input type="checkbox"> ${esc(card.dataset.title)} <span class="float"></span>`);
      row.querySelector("input").onchange = (e) => { this.hide(card, !e.target.checked); if (e.target.checked && !this.isFloating(card)) this.setSidebar(true); this.save(); };
      m.appendChild(row); this.menuRows[id] = row;
    }
    m.appendChild(el("div", "sep"));
    const all = el("button", "", "Show all panels"); all.onclick = () => { this.showAll(); this.save(); };
    const dock = el("button", "", "Dock all floating panels"); dock.onclick = () => this.dockAll();
    const reset = el("button", "", "Reset the layout"); reset.onclick = () => this.reset();
    m.append(all, dock, reset);
    const tip = el("div", "tip", "Drag a panel by its title to reorder it, or drag it over the dish to float it. Hover a title for its hide (×) and pop-out (⧉) buttons.");
    m.appendChild(tip);
    this.refreshMenu();
  }
  refreshMenu() {
    if (!this.menuRows) return;
    const sb = $("sidebarToggle"); if (sb) sb.checked = this.state.sidebar;
    for (const [id, row] of Object.entries(this.menuRows)) {
      const card = this.byId[id];
      row.querySelector("input").checked = !this.isHidden(card);
      row.querySelector(".float").textContent = this.isFloating(card) ? "floating" : "";
    }
    const n = this.cards.filter((c) => !this.isHidden(c)).length;
    this.button.textContent = `Panels ${n < this.cards.length ? `${n}/${this.cards.length} ` : ""}▾`;
    this.button.classList.toggle("dim", !this.state.sidebar);
  }

  // ---------------------------------------------------------------- dragging a card by its title
  pointerDown(e, card, h2) {
    if (e.button !== 0 || e.target.closest("button, input, select, a, label, textarea")) return;
    const r = card.getBoundingClientRect();
    this.drag = { card, h2, x0: e.clientX, y0: e.clientY, dx: e.clientX - r.left, dy: e.clientY - r.top, active: false, id: e.pointerId };
    // the listeners live on the window: moving the card in the DOM (reordering, floating) would release a
    // pointer capture held by the title, and the drag would stall on the first move
    const move = (ev) => { if (ev.pointerId === this.drag?.id) this.pointerMove(ev); };
    const up = (ev) => {
      if (this.drag && ev.pointerId !== this.drag.id) return;
      window.removeEventListener("pointermove", move); window.removeEventListener("pointerup", up); window.removeEventListener("pointercancel", up);
      this.pointerUp(ev);
    };
    window.addEventListener("pointermove", move); window.addEventListener("pointerup", up); window.addEventListener("pointercancel", up);
    e.preventDefault();
  }
  pointerMove(e) {
    const d = this.drag; if (!d) return;
    if (!d.active) {
      if (Math.hypot(e.clientX - d.x0, e.clientY - d.y0) < DRAG_START) return;
      d.active = true;
      d.card.classList.add("dragging"); document.body.classList.add("dragging");
      if (this.isFloating(d.card)) d.card.style.zIndex = ++this.zTop;
    }
    const a = this.aside.getBoundingClientRect();
    const overSidebar = this.state.sidebar && e.clientX >= a.left - 24 && e.clientX <= a.right + 24 && e.clientY >= a.top - 24 && e.clientY <= a.bottom + 24;
    if (overSidebar) {
      if (this.isFloating(d.card)) this.dock(d.card);
      this.placeInSidebar(d.card, e.clientY, a);
    } else {
      if (!this.isFloating(d.card)) this.float(d.card, { x: e.clientX - d.dx, y: e.clientY - d.dy, w: Math.max(FLOAT_W, Math.round(d.card.offsetWidth)) });
      const p = this.state.floating[d.card.id];
      p.x = e.clientX - d.dx; p.y = e.clientY - d.dy;
      d.card.style.left = `${p.x}px`; d.card.style.top = `${p.y}px`;
    }
  }
  placeInSidebar(card, y, a) {
    // the other docked cards, top to bottom: drop before the first whose middle is below the pointer
    const others = [...this.aside.querySelectorAll(":scope > .card")].filter((c) => c !== card && !c.hidden);
    let before = null;
    for (const c of others) { const r = c.getBoundingClientRect(); if (y < r.top + r.height / 2) { before = c; break; } }
    if (before ? before.previousElementSibling !== card : this.aside.lastElementChild !== card) {
      if (before) this.aside.insertBefore(card, before); else this.aside.appendChild(card);
    }
    if (y < a.top + 40) this.aside.scrollTop -= 10; else if (y > a.bottom - 40) this.aside.scrollTop += 10;   // auto-scroll near the edges
  }
  pointerUp(e) {
    const d = this.drag; this.drag = null;
    if (!d) return;
    if (d.active) {
      d.card.classList.remove("dragging"); document.body.classList.remove("dragging");
      if (this.isFloating(d.card)) this.keepOnScreen(d.card);
      this.refreshMenu();
      this.save();
    }
  }
}
