"""
A GPU backend for :class:`virtual_fly.brain.FlyBrain` (optional; needs ``cupy`` and an NVIDIA GPU).

``FlyBrain(conn, backend="cupy")`` builds :class:`GpuFlyBrain`, a subclass whose step runs as hand-written CUDA
kernels compiled at run time by CuPy (NVRTC), **bit-identical** to the NumPy and numba steps: the same spikes on
every step and the same bits in every state array. How that is kept true (docs/TWO_FLIES_PLAN.md section 6):

* **The same float32 operations in the same order.** The kernels are compiled with ``--fmad=false`` and spell every
  operation out with the rounded intrinsics (``__fmul_rn``, ``__fadd_rn``, ...), so nothing is fused, reassociated
  or flushed; the leak, the integration, the fatigue fade, the flush, the graded release and the resets are the CPU's
  expressions operand for operand, and the constants (decay factors, the fade, the coupling) are the host's float32
  values, never recomputed on the device.
* **Every random draw stays on the host**, from the brain's own generator, in the CPU's order (noise count, noise
  neurons, forced spikes), so a seed means the same thing on every backend; the device only adds the host's noise
  kicks (one addition per hit, as ``np.add.at`` does) and marks the host's forced neurons.
* **Synaptic kicks are pulled, in order.** The CPU scatters each spike's kicks into the delayed-input slot, spike by
  spike and edge by edge, so every target accumulates its kicks in ascending presynaptic order from the slot's value.
  The device marks the spiking neurons' edges in a bitmap over the edges grouped by target (the connectome's stable
  in-edge order), then one thread per hit target walks its bits upward and adds the kicks in that same order from the
  slot's value: deterministic, and identical to the CPU, because the data has one connection per (pre, post) pair.
* **The depression factor keeps the CPU's mixed precision**: ``1 - x`` in float32, promoted, times a float64
  ``exp`` from a host-built table (the CPU's own ``math.exp`` values), then rounded to float32 for the kick.
* **The host replays its own work from the spike log.** Tone deposits, the APL tally, plasticity, monitors, the
  recording, callbacks, ``spike_count`` and the refractory bookkeeping are run on the host from each step's sorted
  spike list, by the same code the CPU runs, so every host-side number is the CPU's. The device keeps ``v``, ``g``,
  ``thr``, the delay ring, the resets, the depression state and the graded cells' release; the host reads them on
  demand (properties that download).

``available()`` says whether CuPy and a GPU are there; ``unavailable_reason()`` says in one line what is missing.
Nothing here is imported unless the backend is asked for, so a machine without CUDA never pays for it.
"""
from __future__ import annotations

import math

import numpy as np

from . import fastbrain
from .brain import FlyBrain

_cp = None


def _cupy():
    global _cp
    if _cp is None:
        import cupy
        _cp = cupy
    return _cp


def unavailable_reason() -> str | None:
    """None when CuPy imports and sees an NVIDIA GPU; otherwise one line naming what failed."""
    try:
        cp = _cupy()
    except Exception as e:                      # pragma: no cover - depends on the machine
        return f"the cupy backend needs the cupy package (pip install cupy-cuda13x, or cupy-cuda12x for an older driver): {type(e).__name__}: {e}"
    try:
        n = cp.cuda.runtime.getDeviceCount()
    except Exception as e:                      # pragma: no cover - depends on the machine
        return f"the cupy backend needs an NVIDIA GPU and its driver: {type(e).__name__}: {e}"
    if n < 1:                                   # pragma: no cover
        return "the cupy backend needs an NVIDIA GPU: CUDA reports no device"
    return None


def available() -> bool:
    return unavailable_reason() is None


# ---------------------------------------------------------------------------------------------------- the kernels
# Every kernel reads the step it is in from ``ctrl`` (device memory: [t, k, n_logged, n_hit, ...]) so that the same
# launches, with the same arguments, serve every step and can be captured into a CUDA graph. The input buffer
# ``inbuf`` holds, per chunk: the control words, the per-step offsets of the noise and forced lists, and the lists.
_HDR = 8                       # ctrl[0..3] = t, k, logged, n_hit; [4] noise idx start, [5] noise count start, [6] forced start, [7] spare

_SRC = r'''
extern "C" {

__device__ __forceinline__ int grid_stride() { return gridDim.x * blockDim.x; }
__device__ __forceinline__ int grid_tid() { return blockIdx.x * blockDim.x + threadIdx.x; }

// the delayed input that arrives this step (one slot of the ring), scaled per modulated target by the tone gain in
// force at the START of the step, added to g; the slot is emptied. Nothing happens when nothing is pending (as on the
// CPU, so a -0.0 in g stays what it is).
__global__ void k_arrive(const int n, const int n_slots, const int* ctrl, float* queue, const unsigned char* pending,
                         const int* tpos, const float* gain, float* g)
{
    int t = ctrl[0];
    int slot = t % n_slots;
    if (!pending[slot]) return;
    float* arr = queue + (size_t)slot * n;
    for (int i = grid_tid(); i < n; i += grid_stride()) {
        float a = arr[i];
        int tp = tpos[i];
        if (tp >= 0) a = __fmul_rn(a, gain[tp]);
        g[i] = __fadd_rn(g[i], a);
        arr[i] = 0.0f;
    }
}

// the tone gain the host staged becomes the live one (right after the first arrival of a chunk: the gain a block
// computes applies from the next step on, as on the CPU)
__global__ void k_copy_gain(const int m, const float* staged, float* live)
{
    for (int i = grid_tid(); i < m; i += grid_stride()) live[i] = staged[i];
}

// the host's noise hits: the same neuron hit c times gets c separate additions, as np.add.at gives it
__global__ void k_noise(const int C, const int* inbuf, const float noise_mv, float* g)
{
    int k = inbuf[1];
    const int* noise_off = inbuf + 8;
    int a = noise_off[k], b = noise_off[k + 1];
    const int* idx = inbuf + inbuf[4];
    const int* cnt = inbuf + inbuf[5];
    for (int j = a + grid_tid(); j < b; j += grid_stride()) {
        int i = idx[j];
        int c = cnt[j];
        float gi = g[i];
        for (int r = 0; r < c; r++) gi = __fadd_rn(gi, noise_mv);
        g[i] = gi;
    }
}

// the host's forced spikes of this step (stimulated neurons the host's dice chose)
__global__ void k_force(const int C, const int* inbuf, unsigned char* forced_mask)
{
    int k = inbuf[1];
    const int* forced_off = inbuf + 8 + C + 1;
    int a = forced_off[k], b = forced_off[k + 1];
    const int* forced = inbuf + inbuf[6];
    for (int j = a + grid_tid(); j < b; j += grid_stride()) forced_mask[forced[j]] = 1;
}

// the dense pass: the fatigue fade at a block step, the refractory freeze, the leak and the integration, the flush at a
// block step, the threshold, the forced spikes, the graded cells' release; writes the spike flags
__global__ void k_dense(const int n, const int* ctrl, const int ref_steps, const int fatigue_block,
                        const float decay_m, const float decay_s, const float coupling, const float flush,
                        const float fatigue_mv, const float fade20, const float gr_c, const float gr_sat,
                        float* v, float* g, float* thr, const float* theta_i, const int* last_reset,
                        const unsigned char* gmask, const int* gpos, float* rel, const unsigned char* forced_mask,
                        unsigned char* spk_flag)
{
    int t = ctrl[0];
    bool block = (t % fatigue_block) == 0;
    for (int i = grid_tid(); i < n; i += grid_stride()) {
        float vi = v[i], gi = g[i], th = thr[i];
        if (block && fatigue_mv > 0.0f) {                       // thr -= theta_i; thr *= f; thr += theta_i
            th = __fsub_rn(th, theta_i[i]);
            th = __fmul_rn(th, fade20);
            th = __fadd_rn(th, theta_i[i]);
            thr[i] = th;
        }
        int d = t - last_reset[i];
        bool refr = (d > 0) && (d < ref_steps);                  // reset in one of the last ref_steps-1 steps
        if (refr) {
            vi = 0.0f;                                           // frozen at reset; g keeps its input (incl. this step's)
        } else {
            vi = __fmul_rn(vi, decay_m);
            float tmp = __fmul_rn(gi, coupling);
            vi = __fadd_rn(vi, tmp);
            gi = __fmul_rn(gi, decay_s);
        }
        if (block) {                                             // the denormal guard, for everyone, after the restore
            if (fabsf(vi) < flush) vi = 0.0f;
            if (fabsf(gi) < flush) gi = 0.0f;
        }
        v[i] = vi;
        g[i] = gi;
        bool spk = (!refr) && (vi >= th);
        if (forced_mask[i]) spk = true;                          // stimulated neurons fire even while refractory
        if (gmask[i]) {                                          // graded cells: release in proportion to v
            if (vi > 0.0f) {
                float r = rel[gpos[i]];
                float vc = (vi > gr_sat) ? gr_sat : vi;
                r = __fadd_rn(r, __fmul_rn(vc, gr_c));
                if (r >= 1.0f) { r = __fsub_rn(r, 1.0f); spk = true; }
                rel[gpos[i]] = r;
            }
        }
        spk_flag[i] = spk ? 1 : 0;
    }
}

// the step's spikes into the log (any order: the host sorts); the forced marks are cleared on the way
__global__ void k_compact(const int n, int* ctrl, const unsigned char* spk_flag, unsigned char* forced_mask, int* logbuf)
{
    for (int i = grid_tid(); i < n; i += grid_stride()) {
        if (spk_flag[i]) {
            int p = atomicAdd(&ctrl[2], 1);
            logbuf[p] = i;
        }
        forced_mask[i] = 0;
    }
}

// one warp per spiking neuron: the reset (not for graded cells), the fatigue step, the depression bookkeeping, and
// its outgoing edges marked in the bitmap with their targets put on the hit list
__global__ void k_send_mark(int* ctrl, const int* step_end, const int* logbuf,
                            const int* row_ptr, const int* post_idx, const int* csr_to_csc,
                            unsigned int* bits, int* hit, int* hit_list,
                            float* v, float* g, float* thr, int* last_reset, const unsigned char* gmask,
                            const float fatigue_mv, const int use_std, float* std_x, long long* std_t,
                            float* std_factor, const double* std_table, const int K, const double one_minus_u)
{
    int t = ctrl[0], k = ctrl[1];
    int s0 = (k == 0) ? 0 : step_end[k - 1];
    int s1 = ctrl[2];
    int lane = threadIdx.x & 31;
    int warp = grid_tid() >> 5;
    int nwarps = grid_stride() >> 5;
    for (int j = s0 + warp; j < s1; j += nwarps) {
        int s = logbuf[j];
        if (lane == 0) {
            if (!gmask[s]) {
                v[s] = 0.0f;
                g[s] = 0.0f;
                if (fatigue_mv > 0.0f) thr[s] = __fadd_rn(thr[s], fatigue_mv);
                last_reset[s] = t;
            }
            if (use_std) {                                       // Tsodyks-Markram in the CPU's mixed precision
                long long kk = (long long)t - std_t[s];
                double rec = (kk < K) ? std_table[kk] : std_table[K - 1];
                float d1 = __fsub_rn(1.0f, std_x[s]);            // float32 (1 - x) ...
                double xf = __dsub_rn(1.0, __dmul_rn((double)d1, rec));   // ... promoted, times the recovery, in double
                std_factor[s] = __double2float_rn(xf);           // the kick factor
                std_x[s] = __double2float_rn(__dmul_rn(xf, one_minus_u));
                std_t[s] = t;
            }
        }
        int a = row_ptr[s], b = row_ptr[s + 1];
        for (int e = a + lane; e < b; e += 32) {
            int q = csr_to_csc[e];
            atomicOr(&bits[q >> 5], 1u << (q & 31));
            int p = post_idx[e];
            if (atomicExch(&hit[p], 1) == 0) {
                int pos = atomicAdd(&ctrl[3], 1);
                hit_list[pos] = p;
            }
        }
    }
}

// one thread per hit target: its set bits in ascending edge (= presynaptic) order, each kick added to the slot's
// value in that order, exactly the CPU's scatter; the bits and the hit flag are cleared on the way
__global__ void k_pull(const int n, const int n_slots, const int delay_steps, const int* ctrl, const int* hit_list,
                       const int* col_ptr, const int* csc_pre, const float* w_csc, unsigned int* bits, int* hit,
                       float* queue, const int use_std, const float* std_factor)
{
    int t = ctrl[0];
    int out = (t + delay_steps) % n_slots;
    int nh = ctrl[3];
    float* slotp = queue + (size_t)out * n;
    for (int j = grid_tid(); j < nh; j += grid_stride()) {
        int p = hit_list[j];
        float acc = slotp[p];
        int q0 = col_ptr[p], q1 = col_ptr[p + 1];
        if (q1 > q0) {
            int w0 = q0 >> 5, w1 = (q1 - 1) >> 5;
            for (int wi = w0; wi <= w1; wi++) {
                unsigned int word = bits[wi];
                unsigned int lo = (wi == w0) ? (0xffffffffu << (q0 & 31)) : 0xffffffffu;
                unsigned int hi = (wi == w1) ? (0xffffffffu >> (31 - ((q1 - 1) & 31))) : 0xffffffffu;
                unsigned int mask = lo & hi;
                unsigned int mine = word & mask;
                if (mine) {
                    unsigned int m = mine;
                    while (m) {
                        int b = __ffs(m) - 1;
                        m &= m - 1;
                        int q = (wi << 5) + b;
                        float kick = use_std ? __fmul_rn(w_csc[q], std_factor[csc_pre[q]]) : w_csc[q];
                        acc = __fadd_rn(acc, kick);
                    }
                    if (mask == 0xffffffffu) bits[wi] = 0u; else atomicAnd(&bits[wi], ~mask);
                }
            }
        }
        slotp[p] = acc;
        hit[p] = 0;
    }
}

// one thread, at the end of the step: the ring's flags, the log's step boundary, the counters for the next step
__global__ void k_end(const int n_slots, const int delay_steps, int* ctrl, int* step_end, unsigned char* pending)
{
    int t = ctrl[0], k = ctrl[1];
    int s0 = (k == 0) ? 0 : step_end[k - 1];
    int slot = t % n_slots, out = (t + delay_steps) % n_slots;
    pending[slot] = 0;
    if (ctrl[2] > s0) pending[out] = 1;
    step_end[k] = ctrl[2];
    ctrl[3] = 0;
    ctrl[0] = t + 1;
    ctrl[1] = k + 1;
}

// the analytic fatigue fade on waking: thr = ((thr - theta_i) * f) + theta_i, three float32 operations
__global__ void k_fade(const int n, const float f, float* thr, const float* theta_i)
{
    for (int i = grid_tid(); i < n; i += grid_stride()) {
        float th = __fsub_rn(thr[i], theta_i[i]);
        th = __fmul_rn(th, f);
        thr[i] = __fadd_rn(th, theta_i[i]);
    }
}

}
'''

_BLOCK = 256


class GpuEngine:
    """The device side of one :class:`GpuFlyBrain`: the arrays, the kernels, the chunk graph and the uploads."""

    def __init__(self, brain, use_graph: bool = True):
        cp = _cupy()
        self.cp = cp
        self.use_graph = use_graph
        self.launches = 0                                   # how many times the kernels ran (tests read it)
        self.stream = cp.cuda.Stream(non_blocking=True)
        n, conn = brain.n, brain.conn
        self.n, self.n_slots, self.delay_steps, self.ref_steps = n, brain.n_slots, brain.delay_steps, brain.ref_steps
        self.fatigue_block = brain._fatigue_block
        tick_steps = int(round(25.0 / brain.dt))
        self.C = math.gcd(math.gcd(tick_steps, self.fatigue_block), 200)   # 10 at dt 0.5, 5 at dt 1.0
        if self.fatigue_block % self.C or 200 % self.C:
            raise ValueError("the chunk must divide the 20-step block and the 200-step quiet check")
        mod = cp.RawModule(code=_SRC, options=("--fmad=false",))
        self.k = {name: mod.get_function(name) for name in
                  ("k_arrive", "k_copy_gain", "k_noise", "k_force", "k_dense", "k_compact", "k_send_mark", "k_pull", "k_end", "k_fade")}
        self.grid_n = max(1, min(1024, -(-n // _BLOCK)))
        # ---- the wiring: CSR as the connectome keeps it, CSC in the connectome's stable in-edge order
        E = int(conn.row_ptr[-1])
        self.E = E
        row_ptr = np.asarray(conn.row_ptr, dtype=np.int64)
        post = np.asarray(conn.post_idx, dtype=np.int64)
        if E:
            d = np.diff(post)
            same = np.ones(E - 1, dtype=bool)
            rs = row_ptr[1:-1]
            rs = rs[(rs > 0) & (rs < E)]
            same[rs - 1] = False
            if not np.all(d[same] > 0):
                raise ValueError("the cupy backend needs every row's targets in ascending order with one connection per "
                                 "(pre, post) pair, which this wiring does not have")
        order = np.asarray(conn.in_edge_order, dtype=np.int64)     # CSC position -> CSR edge
        csr_to_csc = np.empty(E, dtype=np.int32)
        csr_to_csc[order] = np.arange(E, dtype=np.int32)
        self.d_row_ptr = cp.asarray(row_ptr.astype(np.int32))
        self.d_post = cp.asarray(post.astype(np.int32))
        self.d_col_ptr = cp.asarray(np.asarray(conn.col_ptr, dtype=np.int32))
        self.d_csc_pre = cp.asarray(np.asarray(conn.pre_idx, dtype=np.int32)[order])
        self.d_csr_to_csc = cp.asarray(csr_to_csc)
        self._order = order
        self.d_w_csr = cp.empty(max(E, 1), dtype=cp.float32)
        self.d_w_csc = cp.empty(max(E, 1), dtype=cp.float32)
        self.d_bits = cp.zeros(max((E + 31) // 32, 1), dtype=cp.uint32)
        self.d_hit = cp.zeros(n, dtype=cp.int32)
        self.d_hit_list = cp.zeros(n, dtype=cp.int32)
        # ---- per-neuron constants
        self.d_gmask = cp.asarray(np.asarray(brain._gmask, dtype=np.uint8))
        gpos = np.full(n, -1, dtype=np.int32)
        gpos[brain._graded_idx] = np.arange(brain._graded_idx.size, dtype=np.int32)
        self.d_gpos = cp.asarray(gpos)
        tpos = np.full(n, -1, dtype=np.int32)
        if brain._mod_targets.size:
            tpos[brain._mod_targets] = np.arange(brain._mod_targets.size, dtype=np.int32)
        self.d_tpos = cp.asarray(tpos)
        self.n_targets = int(brain._mod_targets.size)
        self.d_gain_live = cp.ones(max(self.n_targets, 1), dtype=cp.float32)
        self.d_gain_staged = cp.ones(max(self.n_targets, 1), dtype=cp.float32)
        self._gain_staged_host = np.ones(self.n_targets, dtype=np.float32)
        self.d_theta_i = cp.asarray(np.asarray(brain._theta_i, dtype=np.float32))
        self.d_std_factor = cp.zeros(n, dtype=cp.float32)
        self.d_spk_flag = cp.zeros(n, dtype=cp.uint8)
        self.d_forced_mask = cp.zeros(n, dtype=cp.uint8)
        self.d_last_reset = cp.full(n, -(2 ** 30), dtype=cp.int32)
        self.d_pending = cp.zeros(self.n_slots, dtype=cp.uint8)
        # ---- the state arrays (uploaded from the brain's host copies)
        self.d_v = cp.zeros(n, dtype=cp.float32)
        self.d_g = cp.zeros(n, dtype=cp.float32)
        self.d_thr = cp.zeros(n, dtype=cp.float32)
        self.d_queue = cp.zeros((self.n_slots, n), dtype=cp.float32)
        self.use_std = brain.std_u > 0
        self.d_std_x = cp.ones(n, dtype=cp.float32) if self.use_std else cp.zeros(1, dtype=cp.float32)
        self.d_std_t = cp.zeros(n, dtype=cp.int64)
        self.n_graded = int(brain._graded_idx.size)
        self.d_rel = cp.zeros(max(self.n_graded, 1), dtype=cp.float32)
        self.std_table, self.std_table_mismatch = self._build_std_table(brain)
        self.d_std_table = cp.asarray(self.std_table)
        self.one_minus_u = float(1.0 - brain.std_u)
        # ---- the constants the kernels take
        self.decay_m, self.decay_s, self.coupling = np.float32(brain.decay_m), np.float32(brain.decay_s), np.float32(brain.coupling)
        self.flush = np.float32(brain.FLUSH_MV)
        self.fatigue_mv = np.float32(brain.fatigue_mv)
        self.fade20 = np.float32(brain.decay_f ** self.fatigue_block)
        self.gr_c = np.float32(brain._gr_c) if brain._parts is not None else np.float32(0.0)
        self.gr_sat = np.float32(brain._gr_sat) if brain._parts is not None else np.float32(0.0)
        self.noise_mv = np.float32(brain.noise_mv)
        # ---- the chunk buffers
        self.d_step_end = cp.zeros(self.C, dtype=cp.int32)
        self.d_log = cp.zeros(max(self.C * n, 1), dtype=cp.int32)
        self._inbuf_cap = 0
        self._ensure_inbuf(_HDR + 2 * (self.C + 1) + 4096)
        self._graph = None
        self._w_ref = None
        self._w_pe_last = None
        self._w_loc_last = None
        self._args = None
        self.upload_state(brain)
        self.upload_weights(brain, full=True)

    # ------------------------------------------------------------------ building blocks
    def _build_std_table(self, brain):
        """``exp(-((k * dt) / tau))`` for every step count a neuron can wait, with the CPU's own ``math.exp``, up to
        the first entry at or below 2^-54: from there on ``1.0 - x * rec`` is exactly 1.0 in double for every x in
        [0, 1], so the CPU's results no longer depend on the entry and the last one stands in for all later steps."""
        if not self.use_std:
            return np.ones(1, dtype=np.float64), None
        dt, tau = brain.dt, brain.std_tau_ms
        bound = 2.0 ** -54
        K = int(math.ceil(54.0 * math.log(2.0) * tau / dt)) + 2
        table = np.array([math.exp(-((k * dt) / tau)) for k in range(K)], dtype=np.float64)
        while table[-1] > bound:                                 # (the estimate is only an estimate: make sure)
            K += 64
            table = np.array([math.exp(-((k * dt) / tau)) for k in range(K)], dtype=np.float64)
        table = table[:int(np.argmax(table <= bound)) + 1]       # up to the first entry at the bound
        K = table.size
        other = np.exp(-(np.arange(K, dtype=np.float64) * dt) / tau)   # what the NumPy path computes
        bad = np.flatnonzero(other != table)
        return table, (int(bad[0]) if bad.size else None)

    def _ensure_inbuf(self, size: int):
        cp = self.cp
        if size > self._inbuf_cap:
            self._inbuf_cap = max(size, 2 * self._inbuf_cap)
            self.d_inbuf = cp.zeros(self._inbuf_cap, dtype=cp.int32)
            self._graph = None                                  # the launches' arguments changed
            self._args = None

    def _launch_args(self):
        """The argument tuples of the per-step launches (fixed: every per-step number is read from device memory)."""
        if self._args is None:
            C = np.int32(self.C)
            ctrl = self.d_inbuf
            n = np.int32(self.n)
            self._args = {
                "k_arrive": (n, np.int32(self.n_slots), ctrl, self.d_queue, self.d_pending, self.d_tpos, self.d_gain_live, self.d_g),
                "k_copy_gain": (np.int32(max(self.n_targets, 1)), self.d_gain_staged, self.d_gain_live),
                "k_noise": (C, ctrl, self.noise_mv, self.d_g),
                "k_force": (C, ctrl, self.d_forced_mask),
                "k_dense": (n, ctrl, np.int32(self.ref_steps), np.int32(self.fatigue_block), self.decay_m, self.decay_s,
                            self.coupling, self.flush, self.fatigue_mv, self.fade20, self.gr_c, self.gr_sat,
                            self.d_v, self.d_g, self.d_thr, self.d_theta_i, self.d_last_reset, self.d_gmask, self.d_gpos,
                            self.d_rel, self.d_forced_mask, self.d_spk_flag),
                "k_compact": (n, ctrl, self.d_spk_flag, self.d_forced_mask, self.d_log),
                "k_send_mark": (ctrl, self.d_step_end, self.d_log, self.d_row_ptr, self.d_post, self.d_csr_to_csc,
                                self.d_bits, self.d_hit, self.d_hit_list, self.d_v, self.d_g, self.d_thr, self.d_last_reset,
                                self.d_gmask, self.fatigue_mv, np.int32(1 if self.use_std else 0), self.d_std_x, self.d_std_t,
                                self.d_std_factor, self.d_std_table, np.int32(self.std_table.size), np.float64(self.one_minus_u)),
                "k_pull": (n, np.int32(self.n_slots), np.int32(self.delay_steps), ctrl, self.d_hit_list, self.d_col_ptr,
                           self.d_csc_pre, self.d_w_csc, self.d_bits, self.d_hit, self.d_queue,
                           np.int32(1 if self.use_std else 0), self.d_std_factor),
                "k_end": (np.int32(self.n_slots), np.int32(self.delay_steps), ctrl, self.d_step_end, self.d_pending),
            }
        return self._args

    def _launch_step(self, first: bool):
        a = self._launch_args()
        k = self.k
        gn = (self.grid_n,)
        b = (_BLOCK,)
        k["k_arrive"](gn, b, a["k_arrive"])
        if first and self.n_targets:
            k["k_copy_gain"]((max(1, min(64, -(-self.n_targets // _BLOCK))),), b, a["k_copy_gain"])
        k["k_noise"]((64,), b, a["k_noise"])
        k["k_force"]((64,), b, a["k_force"])
        k["k_dense"](gn, b, a["k_dense"])
        k["k_compact"](gn, b, a["k_compact"])
        k["k_send_mark"]((256,), b, a["k_send_mark"])
        k["k_pull"]((256,), b, a["k_pull"])
        k["k_end"]((1,), (1,), a["k_end"])

    def _chunk_graph(self):
        if self._graph is None:
            self._launch_args()
            with self.stream:
                self.stream.begin_capture()
                for s in range(self.C):
                    self._launch_step(s == 0)
                self._graph = self.stream.end_capture()
        return self._graph

    # ------------------------------------------------------------------ uploads and downloads
    _NAMES = ("v", "g", "thr", "queue", "std_x", "std_t", "_rel")

    def _dev(self, name):
        return {"v": self.d_v, "g": self.d_g, "thr": self.d_thr, "queue": self.d_queue, "std_x": self.d_std_x,
                "std_t": self.d_std_t, "_rel": self.d_rel}[name]

    def upload(self, name: str, value):
        if value is None:
            return
        dev = self._dev(name)
        if name == "_rel" and self.n_graded == 0:
            return
        with self.stream:
            dev.set(np.ascontiguousarray(value, dtype=dev.dtype))
        self.stream.synchronize()

    def download(self, name: str):
        dev = self._dev(name)
        if name == "_rel":
            dev = dev[:self.n_graded]
        with self.stream:
            out = dev.get()
        self.stream.synchronize()
        out.flags.writeable = False
        return out

    def upload_state(self, brain):
        """Every device-resident array from the brain's host copies, plus the resets, the ring's flags and the step."""
        cp = self.cp
        for name in self._NAMES:
            value = brain._host.get(name)
            if value is not None:
                self.upload(name, value)
        last = np.full(self.n, -(2 ** 30), dtype=np.int32)
        for k in range(len(brain._recent) - 1, -1, -1):          # oldest first, so the newest reset wins
            arr = brain._recent[k]
            if arr.size:
                last[arr] = brain.t - 1 - k
        with self.stream:
            self.d_last_reset.set(last)
            self.d_pending.set(np.asarray(brain._pending, dtype=np.uint8))
            if self.n_targets:
                gain = np.asarray(brain._mod_gain, dtype=np.float32)
                self.d_gain_live.set(gain)
                self.d_gain_staged.set(gain)
                self._gain_staged_host = gain.copy()
            self.d_hit.fill(0)
            self.d_bits.fill(0)
            self.d_forced_mask.fill(0)
        self.stream.synchronize()
        brain._stale.clear()

    def upload_weights(self, brain, full: bool = False):
        """The host's ``w`` to both edge layouts: all of it after a rebuild, else the subsets the blocks rewrite."""
        w = brain.w
        if full or w is not self._w_ref:
            self._w_ref = w
            with self.stream:
                if self.E:
                    self.d_w_csr.set(np.ascontiguousarray(w, dtype=np.float32))
                    self.d_w_csc.set(np.ascontiguousarray(w[self._order], dtype=np.float32))
            self.stream.synchronize()
            pl = brain.plasticity
            self._w_pe_last = w[pl.pe].copy() if pl is not None else None
            self._w_loc_last = [w[loc.edges].copy() for loc in brain._local]
            return
        pl = brain.plasticity
        if pl is not None:
            cur = w[pl.pe]
            if self._w_pe_last is None or not np.array_equal(cur.view(np.uint32), self._w_pe_last.view(np.uint32)):
                self._upload_subset(pl.pe, cur)
                self._w_pe_last = cur.copy()
        for j, loc in enumerate(brain._local):
            cur = w[loc.edges]
            if len(self._w_loc_last) <= j:
                self._w_loc_last.append(None)
            last = self._w_loc_last[j]
            if last is None or not np.array_equal(cur.view(np.uint32), last.view(np.uint32)):
                self._upload_subset(loc.edges, cur)
                self._w_loc_last[j] = cur.copy()

    def _upload_subset(self, edges, values):
        cp = self.cp
        if not len(edges):
            return
        with self.stream:
            e = cp.asarray(np.asarray(edges, dtype=np.int64))
            vals = cp.asarray(np.ascontiguousarray(values, dtype=np.float32))
            self.d_w_csr[e] = vals
            self.d_w_csc[self.d_csr_to_csc[e].astype(cp.int64)] = vals
        self.stream.synchronize()

    def sync_gain(self, brain):
        """Stage the host's tone gain when a block changed it (it goes live after the next step's arrivals)."""
        if not self.n_targets:
            return
        gain = np.asarray(brain._mod_gain, dtype=np.float32)
        if gain.shape != self._gain_staged_host.shape or not np.array_equal(gain, self._gain_staged_host):
            with self.stream:
                self.d_gain_staged.set(gain)
            self.stream.synchronize()
            self._gain_staged_host = gain.copy()

    def set_gain_now(self, brain):
        """The host's tone gain in force at once (after a restore or a reset: the next arrivals use it)."""
        if not self.n_targets:
            return
        gain = np.asarray(brain._mod_gain, dtype=np.float32)
        with self.stream:
            self.d_gain_live.set(gain)
            self.d_gain_staged.set(gain)
        self.stream.synchronize()
        self._gain_staged_host = gain.copy()

    def fade(self, factor: np.float32):
        with self.stream:
            self.k["k_fade"]((self.grid_n,), (_BLOCK,), (np.int32(self.n), np.float32(factor), self.d_thr, self.d_theta_i))
        self.stream.synchronize()

    def max_abs(self):
        cp = self.cp
        with self.stream:
            mv = float(cp.abs(self.d_v).max())
            mg = float(cp.abs(self.d_g).max())
        self.stream.synchronize()
        return np.float32(mv), np.float32(mg)

    def zero_v_g(self):
        with self.stream:
            self.d_v.fill(0.0)
            self.d_g.fill(0.0)
        self.stream.synchronize()

    # ------------------------------------------------------------------ running steps
    def run(self, t0: int, count: int, noise, forced):
        """Launch ``count`` steps from step ``t0`` (a whole chunk as a graph when it is one) with the host's draws
        (``noise``: per step a (neurons, counts) pair or None; ``forced``: per step an index array or None).
        Returns the per-step spike lists, each sorted, int64."""
        C = self.C
        n_off = [0]
        f_off = [0]
        n_idx, n_cnt, f_parts = [], [], []
        for k in range(C):
            if k < count and noise[k] is not None:
                n_idx.append(noise[k][0]); n_cnt.append(noise[k][1])
                n_off.append(n_off[-1] + int(noise[k][0].size))
            else:
                n_off.append(n_off[-1])
            if k < count and forced[k] is not None:
                f_parts.append(forced[k])
                f_off.append(f_off[-1] + int(forced[k].size))
            else:
                f_off.append(f_off[-1])
        hdr = _HDR + 2 * (C + 1)
        nn, nf = n_off[-1], f_off[-1]
        used = hdr + 2 * nn + nf
        self._ensure_inbuf(used)
        host = np.empty(used, dtype=np.int32)
        host[0:8] = (t0, 0, 0, 0, hdr, hdr + nn, hdr + 2 * nn, 0)
        host[8:8 + C + 1] = n_off
        host[8 + C + 1:hdr] = f_off
        if nn:
            host[hdr:hdr + nn] = np.concatenate(n_idx)
            host[hdr + nn:hdr + 2 * nn] = np.concatenate(n_cnt)
        if nf:
            host[hdr + 2 * nn:used] = np.concatenate(f_parts)
        with self.stream:
            self.d_inbuf[:used].set(host)
            if count == C and self.use_graph:
                self._chunk_graph().launch(self.stream)
            else:
                for s in range(count):
                    self._launch_step(s == 0)
            step_end = self.d_step_end[:count].get()
            total = int(step_end[count - 1])
            log = self.d_log[:total].get() if total else np.zeros(0, dtype=np.int32)
        self.stream.synchronize()
        self.launches += 1
        out = []
        start = 0
        for k in range(count):
            end = int(step_end[k])
            out.append(np.sort(log[start:end]).astype(np.int64))
            start = end
        return out

    def set_t(self, t: int):
        """Nothing to do: every launch carries the step in its input buffer."""


# ---------------------------------------------------------------------------------------------------- the brain
def _device_property(name: str):
    def get(self):
        if self._gpu is not None and name in self._stale:
            if self._host.get(name) is not None:              # (std_x is None without depression: nothing to fetch)
                self._host[name] = self._gpu.download(name)
            self._stale.discard(name)
        return self._host.get(name)

    def set(self, value):
        self._host[name] = value
        self._stale.discard(name)
        if self._gpu is not None:
            self._gpu.upload(name, value)
    return property(get, set)


class GpuFlyBrain(FlyBrain):
    """The brain with its dynamical state on the GPU (built by ``FlyBrain(conn, backend="cupy")``).

    ``v``, ``g``, ``thr``, ``queue``, ``std_x``, ``std_t`` and ``_rel`` live on the device; reading them downloads a
    copy (read-only, so a host write that the device would not see fails loudly); assigning them uploads. Everything
    else is the CPU's host state, replayed from the spike log (see the module docstring)."""

    v = _device_property("v")
    g = _device_property("g")
    thr = _device_property("thr")
    queue = _device_property("queue")
    std_x = _device_property("std_x")
    std_t = _device_property("std_t")
    _rel = _device_property("_rel")

    def __init__(self, *args, use_graph: bool = True, **kwargs):
        reason = unavailable_reason()
        if reason is not None:
            raise RuntimeError(reason)
        self._gpu = None
        self._host = {}
        self._stale = set()
        super().__init__(*args, **kwargs)
        self._gpu = GpuEngine(self, use_graph=use_graph)

    # ------------------------------------------------------------------ the state
    def reset(self):
        super().reset()                                      # the setters uploaded the arrays (when the engine exists)
        if self._gpu is not None:
            self._gpu.upload_state(self)
            self._gpu.set_gain_now(self)
            self._gpu.upload_weights(self)

    def restore(self, snap: dict):
        for name in GpuEngine._NAMES:                        # writeable host copies for the base's in-place writes
            cur = getattr(self, name)
            if cur is not None:
                self._host[name] = cur.copy()
        super().restore(snap)
        self._gpu.upload_state(self)
        self._gpu.set_gain_now(self)
        self._gpu.upload_weights(self)

    def _rebuild_weights(self):
        super()._rebuild_weights()
        if self._gpu is not None:
            self._gpu.upload_weights(self, full=True)

    def _fade_fatigue(self, factor):
        if self._gpu is None:                                # (not reached: the engine exists before any step)
            return super()._fade_fatigue(factor)
        self._gpu.fade(np.float32(factor))
        self._stale.add("thr")

    def _check_quiet(self):
        if self._stim_idx.size or self._noise_idx.size or self._pending.any() or any(r.size for r in self._recent):
            return
        mv, mg = self._gpu.max_abs()
        if mv < np.float32(self.REST_MV) and mg < np.float32(self.REST_MV):
            self._gpu.zero_v_g()
            self._stale.update(("v", "g"))
            self.quiet = True
            self._quiet_since = self.t

    # ------------------------------------------------------------------ stepping
    def _step_gpu(self, slot: int) -> np.ndarray:
        """One step, launched on its own (the base ``step`` has handled the quiet path and the wake-up fade)."""
        out = self._run_steps(1)
        return out[0][1] if out else self._empty

    def advance_steps(self, n_steps: int) -> list:
        out = []
        n = int(n_steps)
        C = self._gpu.C
        while n > 0:
            slot = self.t % self.n_slots
            if self.quiet and not self._stim_idx.size and not self._pending[slot] and not self._noise_idx.size:
                self.step()                                  # the quiet path: host only, nothing launched
                n -= 1
                continue
            if self.t % C != 0 or n < C:
                t = self.t
                s = self.step()
                if s.size:
                    out.append((t, s))
                n -= 1
                continue
            if self.quiet and self.fatigue_mv > 0:           # waking up: fatigue kept fading while we slept
                self._fade_fatigue(self.decay_f ** (self.t - self._quiet_since))
            self.quiet = False
            out.extend(self._run_steps(C))
            n -= C
        return out

    def _draw(self, count: int):
        """The chunk's random numbers, in the CPU's per-step order, from the brain's own generator."""
        noise, forced = [], []
        for _ in range(count):
            hit = None
            if self._noise_idx.size:
                k = self.rng.poisson(self._noise_idx.size * self.noise_hz * self.dt / 1000.0)
                if k:
                    h = self._noise_idx[self.rng.integers(0, self._noise_idx.size, k)]
                    u, c = np.unique(h, return_counts=True)
                    hit = (u.astype(np.int32), c.astype(np.int32))
            noise.append(hit)
            f = None
            if self._stim_idx.size:
                f = self._stim_idx[self.rng.random(self._stim_idx.size) < self._stim_p].astype(np.int32)
            forced.append(f)
        return noise, forced

    def _run_steps(self, count: int) -> list:
        """``count`` steps from the current step on the device, then the host's work for each in the CPU's order."""
        t0 = self.t
        gpu = self._gpu
        if t0 % self._fatigue_block == 0:
            if self._parts is not None:
                self._mod_block()
                if self._local_active:
                    self._local_block()
        gpu.sync_gain(self)
        gpu.upload_weights(self)
        noise, forced = self._draw(count)
        steps = gpu.run(t0, count, noise, forced)
        self._stale.update(("v", "g", "thr", "queue", "std_x", "std_t", "_rel"))
        out = []
        graded = self._graded_idx.size
        for spikes in steps:
            t = self.t
            slot = t % self.n_slots
            self._pending[slot] = False
            resets = spikes
            if spikes.size:
                if graded:
                    resets = spikes[~self._gmask[spikes]]
                self.spike_count[spikes] += 1
                self.total_spikes += spikes.size
                self._pending[(t + self.delay_steps) % self.n_slots] = True
                if self._mod_targets.size:
                    if fastbrain.available():
                        if fastbrain.deposit_tone(spikes, self._mod_kind, self.row_ptr, self.post_idx, self._n_syn, self.out_scale,
                                                  self._parts.target_pos, self._mod_synref_k, self._mod_level):
                            self._mod_active = True
                    else:
                        self._deposit(spikes)
                if self._local:
                    self._local_tally(spikes)
                if self.recording is not None:
                    self.recording.append((t, spikes.astype(np.int32)))
                for cb in self.on_spikes:
                    cb(spikes, t)
                out.append((t, spikes))
            if self._recent:
                self._recent = [resets] + self._recent[:-1]
            if self.plasticity is not None:
                self.plasticity.step(self, spikes)
            self.t += 1
            self.window_ms += self.dt
            self.last_spikes = spikes
            if self.monitors:
                self._tick_monitors()
            if self.t % 200 == 0:
                self._check_quiet()
        return out
