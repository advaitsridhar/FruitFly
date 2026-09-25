// layout.js — the panel manager: hide the side panels so the dish takes the screen, choose which
// panels show, and drag any panel by its title to reorder it in the sidebar or to float it over
// the dish and put it wherever you like. The arrangement is remembered in this browser.
"use strict";
import { $, el, esc } from "./util.js";

const KEY = "vf.layout.v1";
const DRAG_START = 6;                 // px of movement before a press becomes a drag
const FLOAT_W = 400;                  // a floating card's width unless it was resized
const HEADER = 54;                    // floating cards stay below the header, whose buttons and menus must stay reachable
const Z0 = 30;                        // floating cards stack from here; header menus sit at 1000, overlays at 2000 (style.css)

function titleOf(card) {
  const h2 = card.querySelector(":scope > h2");
  const first = h2 && (h2.querySelector("span") || h2);
  return ((first && first.textContent) || card.id || "panel").trim();
}
function readStore() {
  try { const s = JSON.parse(localStorage.getItem(KEY) || "null"); return s && typeof s === "object" && !Array.isArray(s) ? s : null; } catch (e) { return null; }
}
function writeStore(state) {
  try { localStorage.setItem(KEY, JSON.stringify(state)); } catch (e) { /* private window or storage blocked: the layout just is not remembered */ }
}
const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
const num = (v) => (typeof v === "number" && Number.isFinite(v) ? v : null);
const floatWidth = () => Math.min(FLOAT_W, window.innerWidth - 16);

export class PanelManager {
  constructor(aside, button, menu) {
    this.aside = aside; this.button = button; this.menu = menu;
    this.cards = [...aside.querySelectorAll(":scope > .card")];
    this.cards.forEach((c, i) => { if (!c.id) c.id = `card${i}`; c.dataset.title = titleOf(c); this.decorate(c); });
    this.byId = Object.fromEntries(this.cards.map((c) => [c.id, c]));
    this.zTop = Z0;
    this.drag = null;
    const saved = readStore();
    this.state = { sidebar: true, order: this.cards.map((c) => c.id), hidden: [], floating: {} };
    if (saved) {                                        // whatever is stored is data: filtered, deduplicated, validated
      if (typeof saved.sidebar === "boolean") this.state.sidebar = saved.sidebar;
      if (Array.isArray(saved.order)) {
        const known = [...new Set(saved.order.filter((id) => typeof id === "string" && this.byId[id]))];
        this.state.order = [...known, ...this.state.order.filter((id) => !known.includes(id))];
      }
      if (Array.isArray(saved.hidden)) this.state.hidden = [...new Set(saved.hidden.filter((id) => typeof id === "string" && this.byId[id]))];
      if (saved.floating && typeof saved.floating === "object" && !Array.isArray(saved.floating)) {
        for (const [id, p] of Object.entries(saved.floating)) {
          if (!this.byId[id] || !p || typeof p !== "object" || Array.isArray(p)) continue;
          const q = { x: num(p.x) ?? 8, y: num(p.y) ?? HEADER, w: clamp(num(p.w) ?? FLOAT_W, 260, Math.max(260, Math.min(900, window.innerWidth - 16))) };
          if (num(p.h) > 0) q.h = Math.min(p.h, window.innerHeight);
          this.state.floating[id] = q;
        }
      }
    }
    // a floating card's size is remembered only when the user set it with the resize handle (which writes an
    // inline height); a natural height must not be pinned, or the card would stop growing with its content
    this.sizeWatch = new ResizeObserver((entries) => {
      for (const en of entries) {
        const card = en.target, p = this.state.floating[card.id];
        if (!p || !this.isFloating(card) || this.drag || card.hidden || !card.offsetWidth) continue;
        p.w = Math.round(card.offsetWidth);
        if (card.style.height) p.h = Math.round(card.offsetHeight); else delete p.h;
        this.saveSoon();
      }
    });
    this.apply();
    this.buildMenu();
    button.setAttribute("aria-haspopup", "true"); button.setAttribute("aria-controls", menu.id); button.setAttribute("aria-expanded", "false");
    menu.setAttribute("role", "group"); menu.setAttribute("aria-label", "Panels");
    button.onclick = (e) => { e.stopPropagation(); this.refreshMenu(); menu.hidden ? this.openMenu() : this.closeMenu(); };
    document.addEventListener("click", (e) => { if (!menu.hidden && !e.target.closest(".menuwrap")) this.closeMenu(); });
    window.addEventListener("keydown", (e) => { if (e.key === "Escape" && !menu.hidden) this.closeMenu(true); });
    menu.addEventListener("focusout", (e) => { if (!e.relatedTarget || !e.relatedTarget.closest(".menuwrap")) this.closeMenu(); });
    button.parentElement.addEventListener("keydown", (e) => {   // arrow keys walk the rows, as in a menu (from the button too)
      if (!["ArrowDown", "ArrowUp", "Home", "End"].includes(e.key)) return;
      if (menu.hidden) { if (e.key === "ArrowDown") this.openMenu(); else return; }
      const items = [...menu.querySelectorAll("input, button")].filter((x) => !x.hidden && x.offsetParent !== null);
      const i = items.indexOf(document.activeElement);
      const j = e.key === "Home" || i < 0 ? 0 : e.key === "End" ? items.length - 1 : (i + (e.key === "ArrowDown" ? 1 : -1) + items.length) % items.length;
      if (items[j]) { items[j].focus(); e.preventDefault(); }
    });
    window.addEventListener("resize", () => this.keepOnScreen());
  }
  openMenu() { this.menu.hidden = false; this.button.setAttribute("aria-expanded", "true"); }
  closeMenu(toButton = false) {
    if (this.menu.hidden) return;
    this.menu.hidden = true; this.button.setAttribute("aria-expanded", "false");
    // a checkbox in the menu keeps the focus after a click, which would swallow the keyboard shortcuts
    if (document.activeElement && this.menu.contains(document.activeElement)) { if (toButton) this.button.focus(); else document.activeElement.blur(); }
    else if (toButton) this.button.focus();
  }

  // ---------------------------------------------------------------- per-card controls
  decorate(card) {
    const h2 = card.querySelector(":scope > h2");
    if (!h2) return;
    const ctl = el("span", "pctl");
    const fl = el("button", "", "⧉"); fl.type = "button"; fl.dataset.act = "float"; fl.title = "Pop this panel out over the dish (drag its title to move it; the same button docks it again)"; fl.setAttribute("aria-label", "Pop this panel out");
    const hd = el("button", "", "×"); hd.type = "button"; hd.dataset.act = "hide"; hd.title = "Hide this panel (bring it back from the Panels menu)"; hd.setAttribute("aria-label", "Hide this panel");
    ctl.append(fl, hd);
    h2.appendChild(ctl);
    fl.onclick = (e) => {
      e.stopPropagation();
      const hadFocus = document.activeElement === fl;
      this.isFloating(card) ? this.dock(card) : this.float(card); this.save();
      if (hadFocus) fl.focus({ preventScroll: true });      // the DOM move dropped the focus
    };
    hd.onclick = (e) => { e.stopPropagation(); this.hide(card, true); this.save(); this.button.focus({ preventScroll: true }); };   // the Panels button can bring it back
    h2.addEventListener("pointerdown", (e) => this.pointerDown(e, card, h2));
    card.addEventListener("pointerdown", () => { if (this.isFloating(card)) this.raise(card); });
  }
  isFloating(card) { return card.classList.contains("floating"); }
  isHidden(card) { return this.state.hidden.includes(card.id); }
  raise(card) {
    card.style.zIndex = ++this.zTop;
    if (this.zTop > Z0 + 500) {                         // keep the numbers bounded: renumber from the bottom up
      const fl = this.cards.filter((c) => this.isFloating(c)).sort((a, b) => (+a.style.zIndex || 0) - (+b.style.zIndex || 0));
      fl.forEach((c, i) => { c.style.zIndex = Z0 + i; });
      this.zTop = Z0 + fl.length;
    }
  }

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
      this.raise(card);
    }
    // a popped-out card lands over the right edge of the dish, clear of the sidebar and never on the header
    const st = ($("stage") || this.aside).getBoundingClientRect(), w0 = floatWidth();
    const p = pos || this.state.floating[card.id] || { x: Math.max(8, st.right - w0 - 16), y: Math.max(HEADER, Math.min(r.top, st.bottom - 200)), w: w0 };
    this.state.floating[card.id] = p;
    card.style.left = `${p.x}px`; card.style.top = `${p.y}px`;
    card.style.width = `${p.w || floatWidth()}px`;
    card.style.height = p.h ? `${p.h}px` : "";
    this.keepOnScreen(card);
    this.sizeWatch.observe(card);
    const fb = card.querySelector('.pctl [data-act="float"]'); fb.textContent = "⇤"; fb.setAttribute("aria-label", "Dock this panel");
    this.refreshMenu();
  }
  dock(card, byUser = true) {
    delete this.state.floating[card.id];
    this.sizeWatch.unobserve(card);
    card.classList.remove("floating");
    card.style.left = card.style.top = card.style.width = card.style.height = card.style.maxHeight = card.style.zIndex = "";
    // back into the sidebar at its own slot: before the next card of the order that is docked
    const order = this.state.order, k = order.indexOf(card.id);
    const next = order.slice(k + 1).map((id) => this.byId[id]).find((c) => c.parentElement === this.aside);
    if (next) this.aside.insertBefore(card, next); else this.aside.appendChild(card);
    const fb = card.querySelector('.pctl [data-act="float"]'); fb.textContent = "⧉"; fb.setAttribute("aria-label", "Pop this panel out");
    if (byUser) {
      if (!this.state.sidebar) this.setSidebar(true);   // a card docked into a hidden sidebar would vanish
      this.refreshMenu();
    }
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
  dockAll() {
    let any = false;
    for (const c of this.cards) if (this.isFloating(c)) { this.dock(c, false); any = true; }
    if (any && !this.state.sidebar) this.setSidebar(true);
    this.refreshMenu();
    this.save();
  }
  reset() {
    this.state = { sidebar: true, order: this.cards.map((c) => c.id), hidden: [], floating: {} };
    this.apply();
    this.refreshMenu();
    this.save();
  }
  keepOnScreen(one) {
    // a floating card that fits stays whole on the screen; a bigger one keeps at least its title and
    // controls reachable, and never runs past the bottom edge (it scrolls inside instead)
    for (const card of one ? [one] : this.cards) {
      if (!this.isFloating(card)) continue;
      const p = this.state.floating[card.id]; if (!p) continue;
      const w = card.offsetWidth || p.w || FLOAT_W, h = card.offsetHeight || p.h || 120;
      p.x = clamp(p.x, 8, Math.max(8, window.innerWidth - Math.min(w, window.innerWidth - 16) - 8));
      p.y = clamp(p.y, HEADER, Math.max(HEADER, window.innerHeight - Math.min(h, 60) - 8));
      card.style.left = `${p.x}px`; card.style.top = `${p.y}px`;
      card.style.maxHeight = `${Math.max(60, window.innerHeight - p.y - 8)}px`;
    }
  }
  save() {
    // docked cards take the docked slots of the order in their sidebar sequence; a floating card keeps its
    // slot, so that docking it again puts it back where it was
    const docked = [...this.aside.querySelectorAll(":scope > .card")].map((c) => c.id);
    let i = 0;
    const order = this.state.order.map((id) => (this.byId[id].parentElement === this.aside ? docked[i++] : id));
    this.state.order = order.concat(docked.slice(i));
    writeStore(this.state);
  }
  saveSoon() { clearTimeout(this._t); this._t = setTimeout(() => this.save(), 300); }

  // ---------------------------------------------------------------- the Panels menu
  buildMenu() {
    const m = this.menu; m.innerHTML = "";
    const side = el("label", "", `<input type="checkbox" id="sidebarToggle"> side panels <span class="float" title="keyboard: H" aria-hidden="true">H</span>`);
    side.querySelector("input").onchange = (e) => { this.setSidebar(e.target.checked); e.target.blur(); };
    m.appendChild(side);
    m.appendChild(el("div", "sep"));
    this.menuRows = {};
    for (const id of this.state.order) {
      const card = this.byId[id];
      const row = el("div", "prow");
      const lab = el("label", "", `<input type="checkbox"> ${esc(card.dataset.title)} <span class="float" aria-hidden="true"></span>`);
      lab.querySelector("input").onchange = (e) => {
        this.hide(card, !e.target.checked);
        if (e.target.checked && !this.isFloating(card)) this.setSidebar(true);
        this.save(); e.target.blur();                    // the focus must not stay on the checkbox: it would swallow the shortcuts
      };
      const up = el("button", "mv", "▲"); up.type = "button"; up.title = "Move up"; up.setAttribute("aria-label", `Move ${card.dataset.title} up`);
      const dn = el("button", "mv", "▼"); dn.type = "button"; dn.title = "Move down"; dn.setAttribute("aria-label", `Move ${card.dataset.title} down`);
      up.onclick = () => this.move(card, -1); dn.onclick = () => this.move(card, 1);
      row.append(lab, up, dn);
      m.appendChild(row); this.menuRows[id] = row;
    }
    m.appendChild(el("div", "sep"));
    const all = el("button", "", "Show all panels"); all.onclick = () => { this.showAll(); this.save(); };
    const dock = el("button", "", "Dock all floating panels"); dock.onclick = () => this.dockAll();
    const reset = el("button", "", "Reset the layout"); reset.onclick = () => this.reset();
    m.append(all, dock, reset);
    const tip = el("div", "tip", "Drag a panel by its title to reorder it, or drag it out of the sidebar to float it over the dish (▲ ▼ reorder from here). Hover a title for its hide and pop-out buttons. Zoom (the wheel over the dish, 🔍, +) is what uses the extra width.");
    m.appendChild(tip);
    this.refreshMenu();
  }
  /** Move a docked card one place up or down the sidebar (the keyboard's way to reorder). */
  move(card, dir) {
    if (this.isFloating(card)) return;
    const docked = [...this.aside.querySelectorAll(":scope > .card")].filter((c) => !c.hidden || c === card);
    const i = docked.indexOf(card), j = i + dir;
    if (i < 0 || j < 0 || j >= docked.length) return;
    if (dir < 0) this.aside.insertBefore(card, docked[j]); else this.aside.insertBefore(docked[j], card);
    this.save();
    const focused = document.activeElement && document.activeElement.classList.contains("mv") ? (dir < 0 ? "▲" : "▼") : null;
    this.buildMenu();                                   // the rows follow the new order
    if (focused) { const b = [...this.menuRows[card.id].querySelectorAll(".mv")].find((x) => x.textContent === focused); if (b) b.focus(); }
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
    this.button.textContent = !this.state.sidebar ? "Panels hidden ▾" : `Panels ${n < this.cards.length ? `${n}/${this.cards.length} ` : ""}▾`;
    this.button.title = !this.state.sidebar ? "The side panels are hidden: press H or tick 'side panels' to show them"
      : "Hide the side panels so the dish takes the screen, choose which panels show, or drag a panel by its title to move it";
    this.button.classList.toggle("dim", !this.state.sidebar);
  }

  // ---------------------------------------------------------------- dragging a card by its title
  pointerDown(e, card, h2) {
    if (e.button !== 0 || e.target.closest("button, input, select, a, label, textarea")) return;
    const ae = document.activeElement;                  // a click on a title takes the focus like any other click
    if (ae && ae !== document.body && ae.matches("input, textarea, select")) ae.blur();
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
      if (this.isFloating(d.card)) this.raise(d.card);
    }
    // "over the sidebar" means over the part of it that is on screen (in the single-column layout the
    // sidebar runs on below the window)
    const a = this.aside.getBoundingClientRect();
    const top = Math.max(a.top, HEADER), bottom = Math.min(a.bottom, window.innerHeight);
    const overSidebar = this.state.sidebar && e.clientX >= a.left - 24 && e.clientX <= a.right + 24 && e.clientY >= top - 24 && e.clientY <= bottom + 24;
    if (overSidebar) {
      if (this.isFloating(d.card)) this.dock(d.card, false);
      this.placeInSidebar(d.card, e.clientY, a);
    } else {
      if (!this.isFloating(d.card)) {
        const w = floatWidth();
        d.dx = Math.min(d.dx, w - 40);                    // a wide docked card narrows when it floats: keep the pointer on it
        this.float(d.card, { x: e.clientX - d.dx, y: e.clientY - d.dy, w });
      }
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
    // auto-scroll near the edges: the sidebar when it scrolls, else the page (single-column layout)
    const top = Math.max(a.top, HEADER), bottom = Math.min(a.bottom, window.innerHeight);
    const dy = y < top + 40 ? -10 : y > bottom - 40 ? 10 : 0;
    if (dy) { if (this.aside.scrollHeight > this.aside.clientHeight + 1) this.aside.scrollTop += dy; else window.scrollBy(0, dy); }
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
