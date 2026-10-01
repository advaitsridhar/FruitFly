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
  in-edge order), then one warp per hit target walks its bits upward (the words and the marked kicks loaded in
  parallel) and adds the kicks one by one in that same order from the slot's value, every lane repeating the same
  chain: deterministic, and identical to the CPU, because the data has one connection per (pre, post) pair.
* **The depression factor keeps the CPU's mixed precision**: ``1 - x`` in float32, promoted, times a float64
  ``exp`` from a host-built table (the CPU's own ``math.exp`` values), then rounded to float32 for the kick.
* **The host replays its own work from the spike log.** Tone deposits, the APL tally, plasticity, monitors, the
  recording, callbacks, ``spike_count`` and the refractory bookkeeping are run on the host from each step's sorted
  spike list, by the same code the CPU runs, so every host-side number is the CPU's. The device keeps ``v``, ``g``,
  ``thr``, the delay ring, the resets, the depression state and the graded cells' release; the host reads them on
  demand (properties that download).

* **Six kernels per step, chunks of steps as one graph.** A step is the scatter of the host's draws, the dense pass
  (arrivals, noise, leak, threshold, forced spikes, graded release, the log), the send and mark, the hit list, the
  pull and the step's end; ten steps (five at ``dt`` 1.0), or twenty where a 20-step block starts and the 25 ms tick
  allows, are captured once into a CUDA graph and replayed with one upload (pinned memory) and one download per
  chunk, so a tick is three launches. Every device array is made and used on the engine's own stream, so nothing
  from another stream can overtake an upload.

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

// this step's host-drawn noise (how many kicks each hit neuron gets) and forced spikes, scattered into two per-neuron
// arrays the dense pass reads and clears (a sparse kernel: a few hundred entries)
__global__ void k_scatter(const int C, const int* inbuf, int* noise_cnt, unsigned char* forced_mask)
{
    int k = inbuf[1];
    const int* noise_off = inbuf + 8;
    const int* forced_off = inbuf + 8 + C + 1;
    int a = noise_off[k], b = noise_off[k + 1];
    const int* idx = inbuf + inbuf[4];
    const int* cnt = inbuf + inbuf[5];
    for (int j = a + grid_tid(); j < b; j += grid_stride()) noise_cnt[idx[j]] = cnt[j];
    int fa = forced_off[k], fb = forced_off[k + 1];
    const int* forced = inbuf + inbuf[6];
    for (int j = fa + grid_tid(); j < fb; j += grid_stride()) forced_mask[forced[j]] = 1;
}

// the tone gain the host staged becomes the live one (after the first step of a chunk has read the old one: the gain a
// block computes applies from the next step on, as on the CPU)
__global__ void k_copy_gain(const int m, const float* staged, float* live)
{
    for (int i = grid_tid(); i < m; i += grid_stride()) live[i] = staged[i];
}

// the dense pass, one thread per neuron, in the CPU's order: the arriving slot (scaled by the tone gain on a modulated
// target, when the slot is pending; the slot entry emptied), the noise kicks one by one (as np.add.at adds them), the
// fatigue fade at a block step, the refractory freeze or the leak and integration, the flush at a block step, the
// threshold, the forced spike, the graded cells' release; a spiking neuron appends itself to the log (any order: the
// host sorts) and clears its own noise count and forced mark on the way
__global__ void k_dense(const int n, const int n_slots, int* ctrl, const int ref_steps, const int fatigue_block,
                        const float decay_m, const float decay_s, const float coupling, const float flush,
                        const float fatigue_mv, const float fade20, const float gr_c, const float gr_sat,
                        const float noise_mv, float* queue, const unsigned char* pending, const int* tpos,
                        const float* gain, int* noise_cnt,
                        float* v, float* g, float* thr, const float* theta_i, const int* last_reset,
                        const unsigned char* gmask, const int* gpos, float* rel, unsigned char* forced_mask, int* logbuf)
{
    int t = ctrl[0];
    int slot = t % n_slots;
    bool block = (t % fatigue_block) == 0;
    bool pend = pending[slot] != 0;
    float* arr = queue + (size_t)slot * n;
    for (int i = grid_tid(); i < n; i += grid_stride()) {
        float vi = v[i], gi = g[i], th = thr[i];
        if (pend) {                                              // g += arriving (the tone scales a modulated target's)
            float a = arr[i];
            int tp = tpos[i];
            if (tp >= 0) a = __fmul_rn(a, gain[tp]);
            gi = __fadd_rn(gi, a);
            arr[i] = 0.0f;
        }
        int c = noise_cnt[i];
        if (c) {                                                 // then the noise: one addition per hit
            for (int r = 0; r < c; r++) gi = __fadd_rn(gi, noise_mv);
            noise_cnt[i] = 0;
        }
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
        if (forced_mask[i]) { spk = true; forced_mask[i] = 0; }  // stimulated neurons fire even while refractory
        if (gmask[i]) {                                          // graded cells: release in proportion to v
            if (vi > 0.0f) {
                float r = rel[gpos[i]];
                float vc = (vi > gr_sat) ? gr_sat : vi;
                r = __fadd_rn(r, __fmul_rn(vc, gr_c));
                if (r >= 1.0f) { r = __fsub_rn(r, 1.0f); spk = true; }
                rel[gpos[i]] = r;
            }
        }
        if (spk) {
            int p = atomicAdd(&ctrl[2], 1);
            logbuf[p] = i;
        }
    }
}

// one block per spiking neuron: thread 0 does the reset (not for graded cells), the fatigue step and the depression
// bookkeeping; every thread marks a share of the neuron's outgoing edges in the bitmap (atomicOr without a return
// value: fire and forget) and flags their targets as hit (a plain store: every writer writes 1), so a neuron with
// thousands of outgoing edges is marked by 256 threads at once and nothing in the loop waits for an answer
__global__ void k_send_mark(int* ctrl, const int* step_end, const int* logbuf,
                            const int* row_ptr, const int* post_idx, const int* csr_to_csc,
                            unsigned int* bits, int* hit,
                            float* v, float* g, float* thr, int* last_reset, const unsigned char* gmask,
                            const float fatigue_mv, const int use_std, float* std_x, long long* std_t,
                            float* std_factor, const double* std_table, const int K, const double one_minus_u)
{
    int t = ctrl[0], k = ctrl[1];
    int s0 = (k == 0) ? 0 : step_end[k - 1];
    int s1 = ctrl[2];
    for (int j = s0 + blockIdx.x; j < s1; j += gridDim.x) {
        int s = logbuf[j];
        if (threadIdx.x == 0) {
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
        for (int e = a + threadIdx.x; e < b; e += blockDim.x) {
            int q = csr_to_csc[e];
            atomicOr(&bits[q >> 5], 1u << (q & 31));
            hit[post_idx[e]] = 1;
        }
    }
}

// the hit targets of this step gathered into a list: each warp reads 32 flags at once and the warp's hits are appended
// together (one atomic on the counter per warp with a hit); the list's order does not matter, every target is pulled
// on its own
__global__ void k_hits(const int n, int* ctrl, const int* hit, int* hit_list)
{
    int lane = threadIdx.x & 31;
    for (int base = (grid_tid() & ~31); base < n; base += grid_stride()) {
        int i = base + lane;
        bool h = (i < n) && hit[i];
        unsigned int ball = __ballot_sync(0xffffffffu, h);
        if (ball) {
            int leader = __ffs(ball) - 1;
            int pos = 0;
            if (lane == leader) pos = atomicAdd(&ctrl[3], __popc(ball));
            pos = __shfl_sync(0xffffffffu, pos, leader);
            if (h) hit_list[pos + __popc(ball & ((1u << lane) - 1u))] = i;
        }
    }
}

// the pull: a warp per hit target walks its segment of the bitmap, 32 words at a time, one word per lane; the marked
// kicks of each word are loaded by one lane each, in parallel, then added to the accumulator one by one in ascending
// edge (= presynaptic) order, every lane repeating the same chain of __fadd_rn so that all of them hold the CPU's
// value; the words and the flag are cleared on the way
__global__ void k_pull(const int n, const int n_slots, const int delay_steps, const int* ctrl, const int* hit_list,
                       const int* col_ptr, const int* csc_pre, const float* w_csc, unsigned int* bits, int* hit,
                       float* queue, const int use_std, const float* std_factor)
{
    int t = ctrl[0];
    int out = (t + delay_steps) % n_slots;
    int nh = ctrl[3];
    float* slotp = queue + (size_t)out * n;
    int lane = threadIdx.x & 31;
    int warp = grid_tid() >> 5;
    int nwarps = grid_stride() >> 5;
    for (int j = warp; j < nh; j += nwarps) {
        int p = hit_list[j];
        float acc = slotp[p];
        int q0 = col_ptr[p], q1 = col_ptr[p + 1];
        int w0 = q0 >> 5, w1 = (q1 - 1) >> 5;
        for (int wb = w0; wb <= w1; wb += 32) {
            int wi = wb + lane;
            unsigned int mine = 0u, mask = 0u;
            if (wi <= w1) {
                unsigned int word = bits[wi];
                unsigned int lo = (wi == w0) ? (0xffffffffu << (q0 & 31)) : 0xffffffffu;
                unsigned int hi = (wi == w1) ? (0xffffffffu >> (31 - ((q1 - 1) & 31))) : 0xffffffffu;
                mask = lo & hi;
                mine = word & mask;
            }
            unsigned int active = __ballot_sync(0xffffffffu, mine != 0u);
            while (active) {                                     // the words with marked kicks, in ascending order
                int wl = __ffs(active) - 1;
                active &= active - 1;
                unsigned int wbits = __shfl_sync(0xffffffffu, mine, wl);
                int m = __popc(wbits);
                float kick = 0.0f;
                if (lane < m) {                                  // lane j loads the j-th marked kick of this word
                    int b = __fns(wbits, 0, lane + 1);
                    int q = ((wb + wl) << 5) + b;
                    kick = use_std ? __fmul_rn(w_csc[q], std_factor[csc_pre[q]]) : w_csc[q];
                }
                for (int jj = 0; jj < m; jj++)                   // the CPU's order, one addition at a time
                    acc = __fadd_rn(acc, __shfl_sync(0xffffffffu, kick, jj));
            }
            if (mine) {                                          // clear my word's marks (a boundary word is shared)
                if (mask == 0xffffffffu) bits[wi] = 0u; else atomicAnd(&bits[wi], ~mask);
            }
        }
        if (lane == 0) { slotp[p] = acc; hit[p] = 0; }
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

// a subset of weights written into both edge layouts (the plastic edges after a learning block, the local neurons'
// outputs after a release block): the values come through a pinned staging buffer, stream-ordered before the next step
__global__ void k_scatter_w(const int m, const int* e_csr, const int* e_csc, const float* vals, float* w_csr, float* w_csc)
{
    for (int i = grid_tid(); i < m; i += grid_stride()) {
        float x = vals[i];
        w_csr[e_csr[i]] = x;
        w_csc[e_csc[i]] = x;
    }
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
        self.profile = False                                # True: time the device's work with events (device_ms adds up)
        self.device_ms = 0.0
        # one stream per engine, and EVERY device operation of the engine on it: an array made on another stream
        # (CuPy's default one) is initialised by a memset that nothing orders before this stream's first upload or
        # kernel, and under a busy GPU that memset was seen to land late and wipe the uploaded thresholds. A blocking
        # stream (the default kind) also waits for anything CuPy itself puts on the legacy default stream.
        self.stream = cp.cuda.Stream()
        with self.stream:
            self._build(brain)
        self.stream.synchronize()

    def _build(self, brain):
        cp = self.cp
        n, conn = brain.n, brain.conn
        self.n, self.n_slots, self.delay_steps, self.ref_steps = n, brain.n_slots, brain.delay_steps, brain.ref_steps
        self.fatigue_block = brain._fatigue_block
        tick_steps = int(round(25.0 / brain.dt))
        self.tick_steps = tick_steps
        self.C = math.gcd(math.gcd(tick_steps, self.fatigue_block), 200)   # 10 at dt 0.5, 5 at dt 1.0
        self.C2 = 2 * self.C                                # the long chunk, where a block boundary allows it
        if self.fatigue_block % self.C or 200 % self.C:
            raise ValueError("the chunk must divide the 20-step block and the 200-step quiet check")
        mod = cp.RawModule(code=_SRC, options=("--fmad=false",))
        self.k = {name: mod.get_function(name) for name in
                  ("k_scatter", "k_copy_gain", "k_dense", "k_send_mark", "k_hits", "k_pull", "k_end", "k_fade", "k_scatter_w")}
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
        self._pinned_mems = []
        self.h_gain = self._pinned_f32(max(self.n_targets, 1))
        self.d_theta_i = cp.asarray(np.asarray(brain._theta_i, dtype=np.float32))
        self.d_std_factor = cp.zeros(n, dtype=cp.float32)
        self.d_noise_cnt = cp.zeros(n, dtype=cp.int32)
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
        # ---- the chunk buffers: one device array for the step boundaries and the log (downloaded in one copy into
        # pinned host memory), one pinned host buffer for the inputs (uploaded in one copy); sized for the long chunk
        self.d_out = cp.zeros(self.C2 + max(self.C2 * n, 1), dtype=cp.int32)
        self.d_step_end = self.d_out[:self.C2]
        self.d_log = self.d_out[self.C2:]
        self.h_out = self._pinned(self.d_out.size)
        self._out_guess = 4096                             # how much of the log the one download fetches (grows to fit)
        self._inbuf_cap = 0
        self._ensure_inbuf(_HDR + 2 * (self.C + 1) + 4096)
        self._graphs = {}
        self._w_ref = None
        self._w_pe_last = None
        self._w_loc_last = None
        self._subset_index = {}
        self._args = {}
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

    def _pinned(self, count: int):
        """A page-locked host buffer of ``count`` int32 (a DMA can go straight to it, without a staging copy); the
        memory stays alive as long as the engine keeps it in ``_pinned_mems``."""
        mem = self.cp.cuda.alloc_pinned_memory(count * 4)
        self._pinned_mems.append(mem)
        return np.frombuffer(mem, dtype=np.int32, count=count)

    def _pinned_f32(self, count: int):
        mem = self.cp.cuda.alloc_pinned_memory(count * 4)
        self._pinned_mems.append(mem)
        return np.frombuffer(mem, dtype=np.float32, count=count)

    def _ensure_inbuf(self, size: int):
        cp = self.cp
        if size > self._inbuf_cap:
            self._inbuf_cap = max(size, 2 * self._inbuf_cap)
            with self.stream:                                   # (made and zeroed on the engine's own stream)
                self.d_inbuf = cp.zeros(self._inbuf_cap, dtype=cp.int32)
            self.h_in = self._pinned(self._inbuf_cap)
            self._graphs = {}                                   # the launches' arguments changed
            self._args = {}

    def _launch_args(self, count: int):
        """The argument tuples of the per-step launches of a chunk of ``count`` steps (fixed: every per-step number
        is read from device memory; the count sets the input buffer's layout)."""
        if count not in self._args:
            C = np.int32(count)
            ctrl = self.d_inbuf
            n = np.int32(self.n)
            self._args[count] = {
                "k_scatter": (C, ctrl, self.d_noise_cnt, self.d_forced_mask),
                "k_copy_gain": (np.int32(max(self.n_targets, 1)), self.d_gain_staged, self.d_gain_live),
                "k_dense": (n, np.int32(self.n_slots), ctrl, np.int32(self.ref_steps), np.int32(self.fatigue_block),
                            self.decay_m, self.decay_s, self.coupling, self.flush, self.fatigue_mv, self.fade20,
                            self.gr_c, self.gr_sat, self.noise_mv, self.d_queue, self.d_pending, self.d_tpos,
                            self.d_gain_live, self.d_noise_cnt, self.d_v, self.d_g, self.d_thr, self.d_theta_i,
                            self.d_last_reset, self.d_gmask, self.d_gpos, self.d_rel, self.d_forced_mask, self.d_log),
                "k_send_mark": (ctrl, self.d_step_end, self.d_log, self.d_row_ptr, self.d_post, self.d_csr_to_csc,
                                self.d_bits, self.d_hit, self.d_v, self.d_g, self.d_thr, self.d_last_reset,
                                self.d_gmask, self.fatigue_mv, np.int32(1 if self.use_std else 0), self.d_std_x, self.d_std_t,
                                self.d_std_factor, self.d_std_table, np.int32(self.std_table.size), np.float64(self.one_minus_u)),
                "k_hits": (n, ctrl, self.d_hit, self.d_hit_list),
                "k_pull": (n, np.int32(self.n_slots), np.int32(self.delay_steps), ctrl, self.d_hit_list, self.d_col_ptr,
                           self.d_csc_pre, self.d_w_csc, self.d_bits, self.d_hit, self.d_queue,
                           np.int32(1 if self.use_std else 0), self.d_std_factor),
                "k_end": (np.int32(self.n_slots), np.int32(self.delay_steps), ctrl, self.d_step_end, self.d_pending),
            }
        return self._args[count]

    def _launch_step(self, first: bool, count: int):
        """One step: six launches (the scatter of the host's draws, the dense pass, the send and mark, the hit list, the
        pull, the step's end), plus, after the first step of a chunk, the staged tone gain going live."""
        a = self._launch_args(count)
        k = self.k
        gn = (self.grid_n,)
        b = (_BLOCK,)
        k["k_scatter"]((64,), b, a["k_scatter"])
        k["k_dense"](gn, b, a["k_dense"])
        if first and self.n_targets:
            k["k_copy_gain"]((max(1, min(64, -(-self.n_targets // _BLOCK))),), b, a["k_copy_gain"])
        k["k_send_mark"]((256,), b, a["k_send_mark"])        # a block per spiking neuron
        k["k_hits"](gn, b, a["k_hits"])                      # the hit flags compacted into the list
        k["k_pull"]((256,), b, a["k_pull"])                  # a warp per hit target
        k["k_end"]((1,), (1,), a["k_end"])

    def _chunk_graph(self, count: int):
        """The captured graph of a chunk of ``count`` steps (one per chunk length in use)."""
        if count not in self._graphs:
            self._launch_args(count)
            with self.stream:
                self.stream.begin_capture()
                for s in range(count):
                    self._launch_step(s == 0, count)
                self._graphs[count] = self.stream.end_capture()
        return self._graphs[count]

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
            self.d_hit.fill(0)
            self.d_bits.fill(0)
            self.d_forced_mask.fill(0)
            self.d_noise_cnt.fill(0)
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
                self._upload_subset("pe", pl.pe, cur)
                self._w_pe_last = cur.copy()
        self.upload_local(brain)

    def upload_plastic(self, brain):
        """The plastic edges' weights after a block that changed them (no comparison: the block said so)."""
        pl = brain.plasticity
        if pl is not None:
            cur = brain.w[pl.pe]
            self._upload_subset("pe", pl.pe, cur)
            self._w_pe_last = cur.copy()

    def upload_local(self, brain):
        """The local neurons' output weights where a block changed them (it rewrites them every block, mostly with
        the same values: compared with the last upload first)."""
        for j, loc in enumerate(brain._local):
            cur = brain.w[loc.edges]
            if len(self._w_loc_last) <= j:
                self._w_loc_last.append(None)
            last = self._w_loc_last[j]
            if last is None or not np.array_equal(cur.view(np.uint32), last.view(np.uint32)):
                self._upload_subset(("loc", j), loc.edges, cur)
                self._w_loc_last[j] = cur.copy()

    def _upload_subset(self, key, edges, values):
        """``values`` into both edge layouts at ``edges`` (a fixed subset: its device index arrays, a pinned staging
        buffer and a device one are kept under ``key``): one asynchronous copy and one scatter kernel, stream-ordered
        before the next launch, no synchronisation (the staging buffer is rewritten only after a later launch's
        synchronisation)."""
        cp = self.cp
        m = len(edges)
        if not m:
            return
        cached = self._subset_index.get(key)
        with self.stream:
            if cached is None or cached[0] is not edges:
                e = cp.asarray(np.asarray(edges, dtype=np.int32))
                ec = self.d_csr_to_csc[e.astype(cp.int64)]
                mem = cp.cuda.alloc_pinned_memory(m * 4)
                self._pinned_mems.append(mem)
                host = np.frombuffer(mem, dtype=np.float32, count=m)
                cached = (edges, e, ec, host, cp.empty(m, dtype=cp.float32))
                self._subset_index[key] = cached
                self.stream.synchronize()                       # (the index copies came from pageable memory)
            _, e, ec, host, dvals = cached
            host[:] = values
            dvals.data.copy_from_host_async(host.ctypes.data, m * 4, self.stream)
            self.k["k_scatter_w"]((max(1, min(256, -(-m // _BLOCK))),), (_BLOCK,), (np.int32(m), e, ec, dvals, self.d_w_csr, self.d_w_csc))

    def sync_gain(self, brain):
        """Stage the host's tone gain after a block recomputed it (it goes live after the next step's arrivals):
        one asynchronous copy from a pinned buffer, no comparison, no synchronisation."""
        if not self.n_targets:
            return
        self.h_gain[:] = brain._mod_gain
        self.d_gain_staged.data.copy_from_host_async(self.h_gain.ctypes.data, self.n_targets * 4, self.stream)

    def set_gain_now(self, brain):
        """The host's tone gain in force at once (after a restore or a reset: the next arrivals use it)."""
        if not self.n_targets:
            return
        gain = np.asarray(brain._mod_gain, dtype=np.float32)
        with self.stream:
            self.d_gain_live.set(gain)
            self.d_gain_staged.set(gain)
        self.stream.synchronize()

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
        """Launch ``count`` steps from step ``t0`` (a whole chunk, C or 2C steps, as a graph; fewer steps one by one)
        with the host's draws (``noise``: per step a (neurons, counts) pair or None; ``forced``: per step an index
        array or None). Returns the per-step spike lists, each sorted, int64."""
        C = count if count in (self.C, self.C2) else self.C
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
        host = self.h_in                                         # the pinned input buffer, filled in place
        host[0:8] = (t0, 0, 0, 0, hdr, hdr + nn, hdr + 2 * nn, 0)
        host[8:8 + C + 1] = n_off
        host[8 + C + 1:hdr] = f_off
        if nn:
            np.concatenate(n_idx, out=host[hdr:hdr + nn])
            np.concatenate(n_cnt, out=host[hdr + nn:hdr + 2 * nn])
        if nf:
            np.concatenate(f_parts, out=host[hdr + 2 * nn:used])
        stream = self.stream
        guess = min(self._out_guess, self.d_log.size)
        with stream:
            self.d_inbuf.data.copy_from_host_async(host.ctypes.data, used * 4, stream)
            if self.profile:
                e0, e1 = self.cp.cuda.Event(), self.cp.cuda.Event()
                e0.record()
            if count == C and self.use_graph:
                self._chunk_graph(count).launch(stream)
            else:
                for s in range(count):
                    self._launch_step(s == 0, C)
            if self.profile:
                e1.record()
            # one download: the step boundaries and the first part of the log (a second one only when it overflows);
            # the log sits after the C2 step boundaries in d_out, whatever this chunk's length
            L = self.C2
            self.d_out.data.copy_to_host_async(self.h_out.ctypes.data, (L + guess) * 4, stream)
        stream.synchronize()
        step_end = self.h_out[:count]
        total = int(step_end[count - 1])
        if total > guess:
            with stream:
                self.d_log[guess:total].data.copy_to_host_async(self.h_out[L + guess:].ctypes.data, (total - guess) * 4, stream)
            stream.synchronize()
            self._out_guess = min(2 * total, self.d_log.size)
        if self.profile:
            self.device_ms += self.cp.cuda.get_elapsed_time(e0, e1)
        self.launches += 1
        log = self.h_out[L:L + total]
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
        C2, tick = self._gpu.C2, self._gpu.tick_steps
        self._gpu.upload_weights(self)                       # weights rewritten in place since the last call (attach,
        self._weights_fresh = True                           # forget, load, restore): compared once per call
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
            # the long chunk when it starts at a block (so the blocks fall at its start and end) and ends within the
            # tick (so a 25 ms monitor bin closes at a chunk end, never inside one)
            count = C2 if (self.t % C2 == 0 and n >= C2 and (self.t % tick) + C2 <= tick) else C
            out.extend(self._run_steps(count))
            n -= count
        self._weights_fresh = False
        return out

    def _draw(self, count: int):
        """The chunk's random numbers, in the CPU's per-step order, from the brain's own generator. Without noise
        the only draw per step is ``random(n_stim)``, and one ``random((count, n_stim))`` is the same sequence of
        doubles (the generator fills them one after another), so the whole chunk is drawn at once."""
        noise, forced = [], []
        if not self._noise_idx.size:
            if self._stim_idx.size:
                u = self.rng.random((count, self._stim_idx.size))
                forced = [self._stim_idx[u[k] < self._stim_p].astype(np.int32) for k in range(count)]
            else:
                forced = [None] * count
            return [None] * count, forced
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
        """``count`` steps from the current step on the device, then the host's work for each in the CPU's order.
        What must stay per step is (the recording, the callbacks); the rest is done once for the run where that is
        exactly the CPU's result: the spike counts and the local tally are integer sums, the tone deposits keep the
        spike order in one call, the refractory lists are the last steps' resets, the plasticity's tallies are
        exact counts (``step_block``), and a monitor bin can only close at a step the run ends with (else the slow
        exact path is taken)."""
        t0 = self.t
        gpu = self._gpu
        if t0 % self._fatigue_block == 0:
            if self._parts is not None:
                self._mod_block()
                gpu.sync_gain(self)                          # (the gain only changes here)
                if self._local_active:
                    self._local_block()
                    gpu.upload_local(self)
        if not getattr(self, "_weights_fresh", False):
            gpu.upload_weights(self)
        noise, forced = self._draw(count)
        steps = gpu.run(t0, count, noise, forced)
        self._stale.update(("v", "g", "thr", "queue", "std_x", "std_t", "_rel"))
        dt, n_slots, delay = self.dt, self.n_slots, self.delay_steps
        out = []
        for k, spikes in enumerate(steps):                   # what is per step by nature
            if spikes.size:
                t = t0 + k
                if self.recording is not None:
                    self.recording.append((t, spikes.astype(np.int32)))
                for cb in self.on_spikes:
                    cb(spikes, t)
                out.append((t, spikes))
        for k, spikes in enumerate(steps):                   # the ring's flags, in step order
            t = t0 + k
            self._pending[t % n_slots] = False
            if spikes.size:
                self._pending[(t + delay) % n_slots] = True
        lists = [s for _, s in out]
        allspk = np.concatenate(lists) if len(lists) > 1 else (lists[0] if lists else self._empty)
        # a monitor whose bin closes before the run's last step needs the counts of that step: the per-step path
        early = False
        if self.monitors and count > 1:
            end1 = (t0 + count - 1) * dt
            early = any(end1 - m._t_start >= m.bin_ms - 1e-9 for m in self.monitors.values())
        if early:
            for k, spikes in enumerate(steps):
                if spikes.size:
                    self.spike_count[spikes] += 1
                self.t = t0 + k + 1
                self._tick_monitors()
            self.t = t0
        else:
            for spikes in lists:                             # (each list is unique: a plain fancy add; np.add.at is slow)
                self.spike_count[spikes] += 1
        if allspk.size:
            self.total_spikes += allspk.size
            if self._mod_targets.size:                       # one call keeps the order: step by step, spikes ascending
                if fastbrain.available():
                    if fastbrain.deposit_tone(allspk, self._mod_kind, self.row_ptr, self.post_idx, self._n_syn, self.out_scale,
                                              self._parts.target_pos, self._mod_synref_k, self._mod_level):
                        self._mod_active = True
                else:
                    for spikes in lists:
                        self._deposit(spikes)
            if self._local:
                self._local_tally(allspk)                    # integer sums: the order does not matter
        if self._recent:                                     # the resets of the last ref_steps - 1 steps, newest first
            graded = self._graded_idx.size
            recent = []
            for k in range(count - 1, max(-1, count - 1 - len(self._recent)), -1):
                spikes = steps[k]
                recent.append(spikes[~self._gmask[spikes]] if (graded and spikes.size) else spikes)
            self._recent = (recent + self._recent)[:len(self._recent)]
        if self.plasticity is not None:
            if self.plasticity.step_block(self, steps, t0 + count - 1):
                gpu.upload_plastic(self)
        self.t = t0 + count
        self.window_ms += count * dt
        self.last_spikes = steps[-1]
        if self.monitors and not early:
            self._tick_monitors()
        if self.t % 200 == 0:
            self._check_quiet()
        return out
