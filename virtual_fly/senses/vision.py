"""
Vision: a compound-eye retina model, and the visual neurons it drives.

The starter kit *injected* vision: it computed looming from the pointer's angular size and drove
the looming detectors directly. Here the fly has a **retina**: each eye is a grid of ommatidia
(facets) that each look in one direction and report the brightness there. The arena floor is
bright; the wall, obstacles, the other fly, the lure and the hand are dark shapes of given heights.
Every tick the retina is re-rendered from the fly's position and heading, so what the fly sees
depends on where it looks, how it moves, and what moves.

From the two retinal images, hand-built *feature computations* derive the firing rates of the real
visual neuron types, the way the fly's optic lobe would:

* **Expansion** (a dark shape growing fast) -> ``LC4`` and ``LPLC2``, the looming detectors that
  drive the giant fibre (Ache et al. 2019; Klapoetke et al. 2017).
* **A small moving object** (a blob of 1-12 facets that moves) -> ``LC10a`` (the courtship-chase
  detector, Ribeiro et al. 2018) and ``LC11`` (Keles & Frye 2017) on the eye that sees it.
* **Wide-field motion** (the whole image sliding front-to-back or back-to-front, as when the fly
  turns) -> the columnar motion detectors ``T4``/``T5`` of the matching direction, uniformly over
  the eye. What the wiring does with that (HS/VS cells, optomotor turning) is up to the connectome.

Why not compute motion per column and inject it into the connectome's own T4/T5 columns? The data
has medulla column coordinates for ~24,000 columnar cells, and :class:`ColumnarMotion` does exactly
that when enabled: it maps every facet to the nearest column and drives that column's T4a-d/T5a-d
by a Hassenstein-Reichardt correlator. The feature detectors above are still driven in parallel,
because in this model the T4/T5 -> LPLC2 -> giant fibre route needs a stronger, more specific
input than the correlator provides (see README, "Computed vision").
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..world import FLY_HALF

EYE_HEIGHT = 1.0        # mm above the floor
WALL_HEIGHT = 30.0      # mm: a tall wall, so the drum stripes fill a good part of the visual field
DARK = 0.15             # brightness of a dark object (floor = 1.0)


@dataclass
class VisibleObject:
    x: float
    y: float
    r: float            # horizontal radius (mm)
    h: float            # height above the floor (mm)
    kind: str
    dark: float = DARK


class Eye:
    """One compound eye: a grid of facets with fixed viewing directions in the fly's frame."""

    def __init__(self, side: str, n_az: int = 30, n_el: int = 16):
        self.side = side
        s = 1.0 if side == "L" else -1.0
        # azimuth from 8 degrees across the midline to 165 degrees to the side; elevation -55..+60
        az = np.linspace(math.radians(-8), math.radians(165), n_az) * s
        el = np.linspace(math.radians(-55), math.radians(60), n_el)
        AZ, EL = np.meshgrid(az, el, indexing="ij")
        # a plain rectangular grid: staggering alternate rows (as a real hex lattice does) makes a purely
        # horizontal edge produce a vertical component in every column-to-column comparison
        self.n_az, self.n_el = n_az, n_el
        self.az = AZ.ravel()
        self.el = EL.ravel()
        self.n = self.az.size
        self.image = np.ones(self.n, dtype=np.float32)
        self.prev = np.ones(self.n, dtype=np.float32)
        self.object_mask = np.zeros(self.n, dtype=bool)       # facets covered by an object (not the wall)
        self.facet_deg = math.degrees(az[1] - az[0]) * s

    def render(self, x: float, y: float, h: float, objects: list[VisibleObject], arena_r: float,
               stripes: int = 0, stripe_phase: float = 0.0):
        """Brightness of every facet, given the fly's pose and the visible objects."""
        self.prev = self.image
        img = np.ones(self.n, dtype=np.float32)
        # the wall: a dark band at the horizon whose height depends on distance in each direction
        world_az = self.az + h
        cos_a, sin_a = np.cos(world_az), np.sin(world_az)
        # distance along each ray to the circular wall: solve |p + t d| = R
        b = x * cos_a + y * sin_a
        c = x * x + y * y - arena_r * arena_r
        t = -b + np.sqrt(np.maximum(b * b - c, 0.0))
        top = np.arctan2(WALL_HEIGHT - EYE_HEIGHT, np.maximum(t, 0.1))
        bottom = np.arctan2(-EYE_HEIGHT, np.maximum(t, 0.1))
        wall = (self.el <= top) & (self.el >= bottom)
        if stripes > 0:                                                   # a striped drum: dark/bright bands
            wall_x = x + t * cos_a
            wall_y = y + t * sin_a
            band = np.floor((np.arctan2(wall_y, wall_x) - stripe_phase) / (2 * math.pi) * stripes).astype(int)
            img[wall] = np.where(band[wall] % 2 == 0, 0.85, 0.25)
        else:
            img[wall] = 0.55                                              # the wall is grey, not black
        # floor below the horizon is bright; sky above the wall is bright
        obj_mask = np.zeros(self.n, dtype=bool)
        for o in objects:
            dx, dy = o.x - x, o.y - y
            d = math.hypot(dx, dy)
            if d < 0.3:
                continue
            bearing = (math.atan2(dy, dx) - h + math.pi) % (2 * math.pi) - math.pi
            half = math.atan2(o.r, d)
            daz = (self.az - bearing + math.pi) % (2 * math.pi) - math.pi
            in_az = np.abs(daz) <= half
            if not in_az.any():
                continue
            el_top = math.atan2(o.h - EYE_HEIGHT, max(d - o.r, 0.1))
            el_bot = math.atan2(-EYE_HEIGHT, max(d - o.r, 0.1))
            hit = in_az & (self.el <= el_top) & (self.el >= el_bot)
            img[hit] = np.minimum(img[hit], o.dark)
            obj_mask |= hit
        self.image = img
        self.object_mask = obj_mask
        return img

    def dark_mask(self) -> np.ndarray:
        return self.image < 0.4

    def grid(self, arr: np.ndarray) -> np.ndarray:
        return arr.reshape(self.n_az, self.n_el)


def _blobs(mask2d: np.ndarray) -> list[tuple]:
    """Connected dark regions (4-connected) as (size, mean_i, mean_j, streak=0, touches_edge)."""
    n_i, n_j = mask2d.shape
    seen = np.zeros_like(mask2d, dtype=bool)
    out = []
    for i in range(n_i):
        for j in range(n_j):
            if not mask2d[i, j] or seen[i, j]:
                continue
            stack = [(i, j)]
            seen[i, j] = True
            cells = []
            while stack:
                a, b = stack.pop()
                cells.append((a, b))
                for da, db in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    na, nb = a + da, b + db
                    if 0 <= na < n_i and 0 <= nb < n_j and mask2d[na, nb] and not seen[na, nb]:
                        seen[na, nb] = True
                        stack.append((na, nb))
            rows = [c[0] for c in cells]
            # only the rear edge of the eye counts as "entering the field": column 0 is the frontal
            # overlap of both eyes, where an approaching object is exactly what must be detected
            out.append((len(cells), float(np.mean(rows)), float(np.mean([c[1] for c in cells])), 0,
                        max(rows) == n_i - 1))
    return out


class FeatureDetectors:
    """Hand-built feature computations from the retinal images -> rates of real visual neurons.

    Dark blobs are tracked from frame to frame (matched by position). Only a blob seen in both
    frames can loom, so an object that merely enters the field of view, or sweeps across it because
    the fly turned, does not trigger the escape circuit. Real looming detectors are likewise tuned
    to expansion, not to translation (Klapoetke et al. 2017).
    """

    LOOM_THRESHOLD_DEG = 60.0   # edge speed (deg/s) below which nothing looms
    LOOM_HALF_DEG = 150.0       # edge speed at which LC4 is half-driven
    SMALL_MAX = 14              # facets: largest blob that still counts as a "small object"
    BIG_MIN = 6                 # facets: smallest blob that can loom (real LPLC2: objects > ~15 deg)

    def __init__(self, eyes: dict[str, Eye]):
        self.eyes = eyes
        self.blobs = {s: [] for s in eyes}               # last frame's blobs: (size, i, j)
        self.expansion = {s: 0.0 for s in eyes}          # smoothed edge speed, deg/s
        self.loom_area = {s: 0.0 for s in eyes}
        self.small_speed = {s: 0.0 for s in eyes}        # facets/s
        self.small_pos = {s: None for s in eyes}
        self.flow = {s: 0.0 for s in eyes}               # facets/s, + = front-to-back
        self.felt: dict[str, str] = {}

    @staticmethod
    def _match(blob, prev_blobs, max_dist=3.5):
        best, best_d = None, max_dist ** 2
        for pb in prev_blobs:
            d = (blob[1] - pb[1]) ** 2 + (blob[2] - pb[2]) ** 2
            if d < best_d:
                best, best_d = pb, d
        return best

    def rates(self, dt: float, self_turn: float = 0.0, self_speed: float = 0.0) -> dict[str, float]:
        out: dict[str, float] = {}
        felt: dict[str, str] = {}
        own_shift = abs(self_turn) * dt        # radians the image shifted because the fly turned
        # walking toward something makes it expand too, and turning makes objects slide into view;
        # raise the looming threshold with the fly's own speed and turn rate so that approaching a
        # post or a mate, or looking round, is not an attack (a hand-built efference copy)
        loom_threshold = self.LOOM_THRESHOLD_DEG + 6.0 * abs(self_speed) + 0.5 * math.degrees(abs(self_turn))
        for s, eye in self.eyes.items():
            objects = eye.object_mask                   # things in the dish, never the wall or its stripes
            blobs = _blobs(eye.grid(objects))
            prev = self.blobs[s]
            facet = math.radians(abs(eye.facet_deg))
            # --- looming: sustained growth of a matched blob (two frames in a row, so the facet
            # quantisation of a nearby object shifting sideways does not read as expansion)
            edge_deg, area = 0.0, 0.0
            tracked = []
            for b in blobs:
                pb = self._match(b, prev, max_dist=3.5 + own_shift / facet)
                streak = 0
                if pb is not None and b[0] > pb[0]:
                    streak = pb[3] + 1
                tracked.append((b[0], b[1], b[2], streak, b[4]))
                # a blob cut off by the front or back edge of the eye is entering or leaving the
                # field of view: its apparent growth is not expansion
                if b[0] < self.BIG_MIN or pb is None or streak < 2 or b[4] or pb[4]:
                    continue
                r_now, r_prev = math.sqrt(b[0] / math.pi), math.sqrt(pb[0] / math.pi)
                # an approaching object grows in place; one passing by shifts more than it grows
                shift = max(0.0, math.hypot(b[1] - pb[1], b[2] - pb[2]) - own_shift / facet)
                if r_now - r_prev < 0.6 * shift:
                    continue
                e = (r_now - r_prev) / dt * abs(eye.facet_deg) if dt > 0 else 0.0
                if e > edge_deg:
                    edge_deg, area = e, b[0]
            blobs = tracked
            self.expansion[s] += (edge_deg - self.expansion[s]) * min(1.0, dt / 0.04)
            self.loom_area[s] = area
            if self.expansion[s] > loom_threshold:
                x = (self.expansion[s] - loom_threshold) / (2 * self.LOOM_HALF_DEG)
                out[f"LC4/{s}"] = 160.0 * sat(x)
                out[f"LPLC2/{s}"] = 160.0 * sat(x) * min(1.0, area / 14.0)
                felt["loom"] = felt.get("loom", "") + s
            # --- small moving object: a small blob whose centroid moved (beyond self-motion)
            small = [b for b in blobs if 1 <= b[0] <= self.SMALL_MAX]
            best = None
            if small:
                if self.small_pos[s] is not None:
                    best = self._match(self.small_pos[s], small, max_dist=4.0 + own_shift / facet)
                if best is None:
                    best = max(small, key=lambda b: b[0])
            if best is not None and self.small_pos[s] is not None:
                di = best[1] - self.small_pos[s][1]
                dj = best[2] - self.small_pos[s][2]
                speed = max(0.0, math.hypot(di, dj) - 0.9 * own_shift / facet) / dt if dt > 0 else 0.0
                self.small_speed[s] += (speed - self.small_speed[s]) * min(1.0, dt / 0.08)
            else:
                self.small_speed[s] *= math.exp(-dt / 0.15)
            self.small_pos[s] = best
            if best is not None and self.small_speed[s] > 0.15:
                deg_s = self.small_speed[s] * abs(eye.facet_deg)
                hz = 70.0 * (0.35 + 0.65 * min(1.0, deg_s / 60.0)) * min(1.0, best[0] / 2.0)
                out[f"LC10a/{s}"] = hz
                out[f"LC11/{s}"] = 0.6 * hz
                felt["small"] = felt.get("small", "") + s
            self.blobs[s] = blobs
            # --- wide-field horizontal flow from the row-averaged image (1-D shift estimate)
            prof_now = eye.grid(eye.image).mean(axis=1)
            prof_prev = eye.grid(eye.prev).mean(axis=1)
            d_now = prof_now - prof_now.mean()
            d_prev = prof_prev - prof_prev.mean()
            best_shift, best_corr = 0, -1.0
            for shift in (-2, -1, 0, 1, 2):
                a = d_now[max(0, shift):len(d_now) + min(0, shift)]
                b = d_prev[max(0, -shift):len(d_prev) + min(0, -shift)]
                if a.size < 4 or float(np.abs(a).sum()) < 0.3:
                    continue
                corr = float((a * b).sum())
                if corr > best_corr:
                    best_corr, best_shift = corr, shift
            flow = best_shift / dt if dt > 0 else 0.0          # + = image moved toward larger index = back
            self.flow[s] += (flow - self.flow[s]) * min(1.0, dt / 0.06)
            if abs(self.flow[s]) > 8.0 and best_corr > 0.2:
                sub = "a" if self.flow[s] > 0 else "b"           # a: front-to-back, b: back-to-front
                out[f"T4{sub}/{s},T5{sub}/{s}"] = 40.0 * min(1.0, abs(self.flow[s]) / 40.0)
                felt["flow"] = felt.get("flow", "") + f"{s}{sub}"
        self.felt = felt
        return out


def sat(s: float) -> float:
    """Saturating response 0..1 (same shape as the fly-brain-minecraft encoders)."""
    s = max(0.0, min(1.0, s))
    return s ** 1.5 / (0.2 ** 1.5 + s ** 1.5) / (1.0 / (0.2 ** 1.5 + 1.0))


class ColumnarMotion:
    """Per-column Hassenstein-Reichardt motion detection injected into the connectome's own T4/T5.

    Each facet of the retina is mapped to the nearest medulla column (hex1, hex2) of that eye; each
    column's T4 and T5 neurons of subtype a/b/c/d (front-to-back, back-to-front, up, down) get a
    rate proportional to the correlator output in their preferred direction. ON edges (brightening)
    go to T4, OFF edges (darkening) to T5, as in the real fly.

    **Calibration** (``axes``): the column lattice uses hex axes at 60 degrees. The preferred
    direction of each T4 subtype was measured from the data itself, from the offset between a T4's
    Mi9 inputs (the preferred side of the dendrite, where an edge moving in the preferred direction
    enters) and its Mi4/C3 inputs (the null side): motion in the preferred direction runs from the
    Mi9 side to the Mi4 side, so the preferred direction is Mi4 - Mi9. In the hex-lattice plane the
    *a* subtype points at ``a_deg`` and the *c* subtype at ``c_deg``. Because *a* means front-to-back
    and *c* means upward, these two vectors define the azimuth and elevation axes of the visual
    field. Three independent checks of the frame are in docs/SCIENCE.md 5.3 (HSN's inputs are
    dorsal, LPLC2's inputs are arranged for outward motion, Mi1 somata run dorsal with hex1+hex2).
    Columns are then spread linearly over the eye's field (``az_range``,
    ``el_range``). Distances along the lattice are not exact angles (the real eye's facets are not
    uniformly spaced), but every column lands on the correct part of the visual field and the
    subtypes point the right way, which is what the downstream wiring needs.
    """

    SUBTYPES = ("a", "b", "c", "d")

    def __init__(self, conn, eyes: dict[str, Eye], axes: dict, gain_hz: float = 120.0):
        self.conn = conn
        self.eyes = eyes
        self.gain_hz = gain_hz
        self.axes = axes
        self.cols: dict[str, dict] = {}
        for side, eye in eyes.items():
            self.cols[side] = self._map_eye(side, eye, axes)
        self.hp = {s: np.zeros(e.n, dtype=np.float32) for s, e in eyes.items()}   # delayed arm
        self.last_idx = np.zeros(0, dtype=np.int64)
        self.last_hz = np.zeros(0)

    # columnar input types that carry medulla column coordinates in this dataset
    T4_REFERENCE = ("Mi1", "Mi9", "Mi4", "C3")
    T5_REFERENCE = ("Tm1", "Tm2", "Tm9", "Tm4")

    def _home_column(self, idx, reference):
        """T4/T5 cells carry no column coordinate themselves; their home column is the
        synapse-weighted mean column of their columnar inputs (Mi1/Mi9/Mi4/C3 for T4, Tm1/2/4/9
        for T5). Returns (hex1, hex2) floats per neuron, NaN where no such input exists."""
        conn = self.conn
        e = conn.in_edges(idx)
        pre, post = conn.pre_idx[e], conn.post_idx[e]
        ok = (conn.hex1[pre] >= 0) & np.isin(conn.types[pre], reference)
        e, pre, post = e[ok], pre[ok], post[ok]
        local = np.searchsorted(idx, post)
        w = conn.n_syn[e].astype(np.float64)
        tot = np.bincount(local, weights=w, minlength=idx.size)
        h1 = np.bincount(local, weights=w * conn.hex1[pre], minlength=idx.size) / np.maximum(tot, 1e-9)
        h2 = np.bincount(local, weights=w * conn.hex2[pre], minlength=idx.size) / np.maximum(tot, 1e-9)
        h1[tot == 0] = np.nan
        h2[tot == 0] = np.nan
        return h1, h2

    def _map_eye(self, side, eye, axes):
        conn = self.conn
        idx_by_sub, home = {}, {}
        for cell, ref in (("T4", self.T4_REFERENCE), ("T5", self.T5_REFERENCE)):
            for sub in self.SUBTYPES:
                idx = conn.select(f"{cell}{sub}/{side}")
                idx_by_sub[cell + sub] = idx
                home[cell + sub] = self._home_column(idx, ref)
        # the column lattice of this eye: the reference cells' own coordinates
        ref_idx = conn.select(",".join(f"{t}/{side}" for t in self.T4_REFERENCE + self.T5_REFERENCE))
        h1 = conn.hex1[ref_idx].astype(np.int64)
        h2 = conn.hex2[ref_idx].astype(np.int64)
        ok = (h1 >= 0) & (h2 >= 0)
        keys = np.unique(h1[ok] * 1000 + h2[ok])
        ch1, ch2 = (keys // 1000).astype(float), (keys % 1000).astype(float)
        # hex lattice -> plane (axes at 60 degrees), then project onto the measured PD axes
        cx = ch1 + 0.5 * ch2
        cy = (math.sqrt(3) / 2) * ch2
        a = math.radians(axes["a_deg"])                   # front-to-back (T4a preferred direction)
        c = math.radians(axes["c_deg"])                   # upward (T4c preferred direction)
        p_az = cx * math.cos(a) + cy * math.sin(a)
        p_el = cx * math.cos(c) + cy * math.sin(c)
        az_lo, az_hi = (math.radians(v) for v in axes.get("az_range", (-8.0, 165.0)))
        el_lo, el_hi = (math.radians(v) for v in axes.get("el_range", (-55.0, 60.0)))
        col_az = az_lo + (p_az - p_az.min()) / max(p_az.max() - p_az.min(), 1e-9) * (az_hi - az_lo)
        col_el = el_lo + (p_el - p_el.min()) / max(p_el.max() - p_el.min(), 1e-9) * (el_hi - el_lo)
        if side == "R":
            col_az = -col_az                              # the eye's azimuth is signed (+ = left)
        # nearest column for each facet
        d = (col_az[None, :] - eye.az[:, None]) ** 2 + (col_el[None, :] - eye.el[:, None]) ** 2
        facet_col = np.argmin(d, axis=1).astype(np.int64)
        # T4/T5 neurons -> nearest lattice column to their home column
        neurons = {}
        for name, idx in idx_by_sub.items():
            hh1, hh2 = home[name]
            good = ~np.isnan(hh1)
            kk = (np.rint(hh1[good]).astype(np.int64) * 1000 + np.rint(hh2[good]).astype(np.int64))
            pos = np.searchsorted(keys, kk)
            pos = np.clip(pos, 0, keys.size - 1)
            # a rounded home column may not exist: fall back to the nearest one in the plane
            miss = keys[pos] != kk
            if miss.any():
                gx = hh1[good][miss] + 0.5 * hh2[good][miss]
                gy = (math.sqrt(3) / 2) * hh2[good][miss]
                dd = (cx[None, :] - gx[:, None]) ** 2 + (cy[None, :] - gy[:, None]) ** 2
                pos[miss] = np.argmin(dd, axis=1)
            neurons[name] = (idx[good], pos)
        return {"facet_col": facet_col, "n_cols": keys.size, "neurons": neurons,
                "col_az": col_az, "col_el": col_el, "hex": keys}

    DELAY_TAU = 0.06        # s: the "delayed arm" of the correlator (a low-pass of recent changes)
    efference_gain = 1.0    # set by the game: <1 while the fly turns on purpose (efference copy)

    @staticmethod
    def _pool(r):
        """T4/T5 dendrites span a few columns: pool each facet's response with its neighbours."""
        out = r.copy()
        out[1:, :] += 0.5 * r[:-1, :]
        out[:-1, :] += 0.5 * r[1:, :]
        out[:, 1:] += 0.5 * r[:, :-1]
        out[:, :-1] += 0.5 * r[:, 1:]
        return out

    def rates(self, dt: float):
        """Returns (neuron indices, rates) for T4/T5 neurons of both eyes.

        Hassenstein-Reichardt correlator on the *change* of brightness at each facet: a change now at
        facet i, times the recent change at the neighbouring facet the motion came from, minus the
        mirror term (the change now at the neighbour times the recent change at i). Interiors of
        objects, which do not change, give nothing; only moving edges do. ON edges (brightening)
        drive T4, OFF edges (darkening) drive T5, each in the subtype of the motion's direction.
        """
        idx_all, hz_all = [], []
        for side, eye in self.eyes.items():
            n_az, n_el = eye.n_az, eye.n_el
            d = (eye.image - eye.prev).reshape(n_az, n_el)             # change this frame
            lp = self.hp[side].reshape(n_az, n_el)                      # recent changes (delayed arm)
            k = min(1.0, dt / self.DELAY_TAU)
            fb = np.zeros_like(d)                                       # a: front-to-back (index i grows)
            fb[1:, :] = d[1:, :] * lp[:-1, :] - d[:-1, :] * lp[1:, :]
            bf = -fb                                                    # b: back-to-front
            up = np.zeros_like(d)                                       # c: upward (index j grows)
            up[:, 1:] = d[:, 1:] * lp[:, :-1] - d[:, :-1] * lp[:, 1:]
            dn = -up                                                    # d: downward
            self.hp[side] = (lp + (d - lp) * k).ravel()
            on = (d > 0).astype(np.float32)
            resp = {}
            for sub, r in (("a", fb), ("b", bf), ("c", up), ("d", dn)):
                r = np.maximum(r, 0.0)
                resp["T4" + sub] = self._pool(r * on).ravel()
                resp["T5" + sub] = self._pool(r * (1 - on)).ravel()
            cols = self.cols[side]
            cnt = np.bincount(cols["facet_col"], minlength=cols["n_cols"])
            for name, (idx, col) in cols["neurons"].items():
                if not resp[name].any():
                    continue
                per_col = np.bincount(cols["facet_col"], weights=resp[name], minlength=cols["n_cols"])
                per_col = per_col / np.maximum(cnt, 1)
                hz = self.gain_hz * np.minimum(1.0, per_col[col] / 0.015) * self.efference_gain
                keep = hz > 2.0
                if keep.any():
                    idx_all.append(idx[keep])
                    hz_all.append(hz[keep])
        if idx_all:
            self.last_idx = np.concatenate(idx_all)
            self.last_hz = np.concatenate(hz_all)
        else:
            self.last_idx, self.last_hz = np.zeros(0, dtype=np.int64), np.zeros(0)
        return self.last_idx, self.last_hz


# Preferred-direction axes of T4 subtypes in the hex-lattice plane, measured from the MaleCNS data
# (offset Mi4 - Mi9 of each T4's inputs, ~450-700 cells per subtype and side, both sides agree
# within 3 degrees): a (front-to-back) +106 deg, b -80 deg, c (up) +26 deg, d -148 deg.
# (An earlier build used Mi9 - Mi4, the same axes rotated by 180 degrees, which put HSN's inputs in
# the ventral field and LPLC2's inputs in the contraction arrangement; see docs/SCIENCE.md 5.3.)
COLUMNAR_AXES = {"a_deg": 106.0, "c_deg": 26.0, "az_range": (-8.0, 165.0), "el_range": (-55.0, 60.0)}


class Retina:
    """Both eyes plus the feature detectors; call :meth:`look` once per tick."""

    def __init__(self, world, conn=None, columnar: bool = False):
        self.world = world
        self.eyes = {"L": Eye("L"), "R": Eye("R")}
        self.features = FeatureDetectors(self.eyes)
        self.columnar = ColumnarMotion(conn, self.eyes, COLUMNAR_AXES) if (columnar and conn is not None) else None

    def objects(self, pose) -> list[VisibleObject]:
        w = self.world
        objs = [VisibleObject(o.x, o.y, o.r, 15.0, "post", 0.1) for o in w.obstacles]
        objs += [VisibleObject(f.x, f.y, f.r, 0.6, "food", 0.5) for f in w.food]
        if w.female is not None:
            objs.append(VisibleObject(w.female.x, w.female.y, 1.6, 2.2, "fly", 0.12))
        if w.hand is not None and pose.jump is None:
            hx, hy = w.hand
            if w.tool == "lure":
                objs.append(VisibleObject(hx, hy, 1.3, 2.0, "lure", 0.12))
            elif w.tool == "hand":
                objs.append(VisibleObject(hx, hy, 6.0, 5.0, "hand", 0.1))
        return objs

    def look(self, pose, dt: float, turn_command: float = 0.0) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
        """Render both eyes and compute the visual neuron rates. ``turn_command`` is the fly's own
        commanded yaw (-1..1): an *efference copy* damps the wide-field motion signal while the fly
        turns on purpose, as in real flies (Kim, Fitzgerald & Maimon 2015). Hand-built."""
        objs = self.objects(pose)
        for eye in self.eyes.values():
            eye.render(pose.x, pose.y, pose.h, objs, self.world.arena_r,
                       getattr(self.world, "stripes", 0), getattr(self.world, "stripe_phase", 0.0))
        rates = self.features.rates(dt, self_turn=pose.w, self_speed=pose.v)
        if self.columnar is not None:
            # the connectome's own T4/T5 carry wide-field motion: drop the uniform stand-in
            rates = {k: v for k, v in rates.items() if not k.startswith("T4")}
            self.columnar.efference_gain = max(0.15, 1.0 - 2.5 * abs(turn_command))
            idx, hz = self.columnar.rates(dt)
        else:
            idx, hz = np.zeros(0, dtype=np.int64), np.zeros(0)
        return rates, idx, hz

    def images_b64(self) -> dict[str, str]:
        import base64
        out = {}
        for s, eye in self.eyes.items():
            out[s] = base64.b64encode((eye.image * 255).astype(np.uint8).tobytes()).decode()
        return out

    def layout(self) -> dict:
        return {s: {"n_az": e.n_az, "n_el": e.n_el, "az": [round(float(a), 3) for a in e.az],
                    "el": [round(float(a), 3) for a in e.el]} for s, e in self.eyes.items()}
