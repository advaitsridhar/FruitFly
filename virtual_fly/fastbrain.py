"""
A compiled integrator for :class:`virtual_fly.brain.FlyBrain` (optional; needs ``numba``).

The NumPy step makes about ten passes over the 176k-element state arrays and scatters every
spike's synaptic kicks with ``np.add.at``. The kernel here does the same arithmetic, on the same
float32 values, in the same order, in one fused pass plus a plain loop over the spiking neurons'
outgoing connections, so it produces **bit-identical spikes** to the NumPy path (the test suite
checks this on every seed and every optional mechanism) while touching memory once. The Poisson
draws for stimulated neurons and the background noise stay in NumPy, on the brain's own random
generator, so a seed means the same thing on both backends.

``available()`` says whether numba could be imported; :class:`FlyBrain` picks this backend
automatically when it can (``backend="auto"``) and falls back to NumPy otherwise.
"""

from __future__ import annotations

import math

import numpy as np

try:
    from numba import njit
    HAVE_NUMBA = True
except Exception:                       # pragma: no cover - exercised only where numba is missing
    HAVE_NUMBA = False

    def njit(*args, **kwargs):          # type: ignore[misc]
        def deco(f):
            return f
        return deco if not (args and callable(args[0])) else args[0]


def available() -> bool:
    return HAVE_NUMBA


@njit(cache=True, nogil=True)
def _merge_sorted(a, b, out):
    """Union of two sorted, unique int64 arrays into ``out`` (sorted, unique); returns the count."""
    i = j = k = 0
    na, nb = a.shape[0], b.shape[0]
    while i < na and j < nb:
        x, y = a[i], b[j]
        if x < y:
            out[k] = x; i += 1
        elif y < x:
            out[k] = y; j += 1
        else:
            out[k] = x; i += 1; j += 1
        k += 1
    while i < na:
        out[k] = a[i]; i += 1; k += 1
    while j < nb:
        out[k] = b[j]; j += 1; k += 1
    return k


@njit(cache=True, nogil=True)
def lif_step(v, g, arriving, has_arriving, refr, decay_m, coupling, decay_s,
             thr, flush, do_flush, forced, spk, cand, spikes_out):
    """One integration step for every neuron. Returns the number of spikes written to ``spikes_out``.

    Order of operations (identical to ``FlyBrain.step``): add the arriving delayed input, freeze the
    refractory neurons (they keep their input and sit at reset), leak and integrate everyone else,
    snap values below ``flush`` to zero when ``do_flush``, then threshold; ``forced`` (sorted, unique)
    are the stimulated neurons that fire this step whatever their voltage. The dense pass is a plain
    vectorisable loop on one core: a version split over the cores was 1.3x faster on an idle
    machine and 10x *slower* as soon as one other process (a browser, say) used a core, because
    every 50 µs parallel region then waited for a descheduled worker.
    """
    n = v.shape[0]
    n_refr = refr.shape[0]
    zero = np.float32(0.0)
    # what the refractory neurons keep: their input including what arrives now (NumPy: g += arriving
    # first, then g[refractory] is restored after the decay)
    frozen = np.empty(n_refr, dtype=np.float32)
    for k in range(n_refr):
        r = refr[k]
        gr = g[r]
        if has_arriving:
            gr = gr + arriving[r]
        frozen[k] = gr
    # the dense pass: one plain loop the compiler vectorises (a blocked variant that also recorded
    # which blocks spiked, to shorten the scan below, ran three times slower: the reduction broke
    # the vectorisation, and the scan itself is only a tenth of this pass)
    if has_arriving:
        for i in range(n):
            gi = g[i] + arriving[i]
            arriving[i] = zero
            vi = v[i] * decay_m
            vi = vi + gi * coupling
            gi = gi * decay_s
            v[i] = vi
            g[i] = gi
            spk[i] = vi >= thr[i]
    else:
        for i in range(n):
            gi = g[i]
            vi = v[i] * decay_m
            vi = vi + gi * coupling
            gi = gi * decay_s
            v[i] = vi
            g[i] = gi
            spk[i] = vi >= thr[i]
    # refractory neurons: frozen at reset with their input kept; they cannot fire
    for k in range(n_refr):
        r = refr[k]
        v[r] = zero
        g[r] = frozen[k]
        spk[r] = False
    if do_flush:                                       # same flush as NumPy, for everyone, after the restore
        for i in range(n):
            if abs(v[i]) < flush:
                v[i] = zero
            if abs(g[i]) < flush:
                g[i] = zero
    n_cand = 0
    for i in range(n):                                 # compaction, in index order
        if spk[i]:
            cand[n_cand] = i
            n_cand += 1
    if forced.shape[0] == 0:
        for k in range(n_cand):
            spikes_out[k] = cand[k]
        return n_cand
    return _merge_sorted(cand[:n_cand], forced, spikes_out)


@njit(cache=True, nogil=True)
def fire_and_send(spikes, v, g, thr, fatigue_mv, spike_count, row_ptr, post_idx, w, target,
                  use_std, std_x, std_t, t, dt, std_tau_ms, std_u):
    """Reset the spiking neurons and add their synaptic kicks to ``target`` (a slot of the delay
    queue), connection by connection in the same order ``np.add.at`` would."""
    zero = np.float32(0.0)
    one32 = np.float32(1.0)
    for k in range(spikes.shape[0]):
        s = spikes[k]
        v[s] = zero
        g[s] = zero
        if fatigue_mv > zero:                         # fatigue_mv is float32, like NumPy's thr[s] += float32
            thr[s] = thr[s] + fatigue_mv
        spike_count[s] += 1
        a, b = row_ptr[s], row_ptr[s + 1]
        if use_std:                                   # Tsodyks-Markram: recover, then use some resource
            elapsed = (t - std_t[s]) * dt
            rec = math.exp(-elapsed / std_tau_ms)
            xf = 1.0 - (one32 - std_x[s]) * rec       # float32 (1 - x) times float64 -> float64
            factor = np.float32(xf)
            std_x[s] = np.float32(xf * (1.0 - std_u))
            std_t[s] = t
            for e in range(a, b):
                p = post_idx[e]
                target[p] = target[p] + w[e] * factor
        else:
            for e in range(a, b):
                p = post_idx[e]
                target[p] = target[p] + w[e]


@njit(cache=True, nogil=True)
def step_kernel(v, g, arriving, has_arriving, refr, decay_m, coupling, decay_s, thr, flush, do_flush,
                forced, spk, cand, spikes_out, fatigue_mv, spike_count, row_ptr, post_idx, w, target,
                use_std, std_x, std_t, t, dt, std_tau_ms, std_u):
    """``lif_step`` then ``fire_and_send`` in one call (one dispatch per step instead of two)."""
    n_spk = lif_step(v, g, arriving, has_arriving, refr, decay_m, coupling, decay_s, thr, flush, do_flush,
                     forced, spk, cand, spikes_out)
    if n_spk:
        fire_and_send(spikes_out[:n_spk], v, g, thr, fatigue_mv, spike_count, row_ptr, post_idx, w, target,
                      use_std, std_x, std_t, t, dt, std_tau_ms, std_u)
    return n_spk


def warm_up():
    """Compile (or load from the on-disk cache) the kernels on tiny arrays, so the first real step
    of a game does not stall. Takes a few seconds the very first time on a machine."""
    if not HAVE_NUMBA:
        return
    n = 4
    f32 = np.zeros(n, dtype=np.float32)
    i64 = np.zeros(0, dtype=np.int64)
    cand = np.zeros(n, dtype=np.int64)
    out = np.zeros(n, dtype=np.int64)
    spk = np.zeros(n, dtype=np.bool_)
    for has_arr in (False, True):
        lif_step(f32.copy(), f32.copy(), f32.copy(), has_arr, i64, np.float32(0.9), np.float32(0.1),
                 np.float32(0.9), f32 + np.float32(7.0), np.float32(1e-6), True, i64, spk, cand, out)
    row_ptr = np.array([0, 1, 1, 1, 1], dtype=np.int64)
    post = np.array([1], dtype=np.int64)
    w = np.array([0.1], dtype=np.float32)
    for use_std in (False, True):
        fire_and_send(np.array([0], dtype=np.int64), f32.copy(), f32.copy(), f32.copy(), np.float32(0.05), np.zeros(n, np.int32),
                      row_ptr, post, w, f32.copy(), use_std, np.ones(n, np.float32), np.zeros(n, np.int64),
                      3, 0.5, 500.0, 0.1)
        step_kernel(f32.copy(), f32.copy(), f32.copy(), True, i64, np.float32(0.9), np.float32(0.1), np.float32(0.9),
                    f32 + np.float32(7.0), np.float32(1e-6), True, np.array([0], dtype=np.int64), spk, cand, out,
                    np.float32(0.05), np.zeros(n, np.int32), row_ptr, post, w, f32.copy(), use_std,
                    np.ones(n, np.float32), np.zeros(n, np.int64), 3, 0.5, 500.0, 0.1)
