"""
The brain simulator: every neuron of the connectome as a leaky integrate-and-fire unit.

The core model is exactly Shiu et al., *Nature* 2024 ("A Drosophila computational brain model
reveals sensorimotor processing"), with the male-connectome gain calibration of the
fly-brain-minecraft project. On top of that, this version adds optional mechanisms that real
neurons have and the paper's model lacks. **Every one of them is off by default**, so
``FlyBrain(conn)`` is still the paper's model; the game switches on the ones that are documented
in :mod:`virtual_fly.settings`:

* **Short-term synaptic depression** (Tsodyks & Markram 1997): a neuron that fires in a burst runs
  low on releasable vesicles, so its later spikes have less effect. This is the main physiological
  brake against the runaway loops that the pure model shows after bitter taste or dust.
* **Spike-frequency adaptation** ("fatigue"): every spike raises that neuron's threshold a little,
  fading over a couple of seconds.
* **Background noise**: a weak Poisson bombardment of every neuron, standing in for the
  spontaneous activity real neurons have. With it, the fly is never completely silent.
* **Output modulation**: scale (or zero) the output of any population. Zero = silencing, like
  expressing tetanus toxin in a real fly; 1.5 = the population's synapses are stronger, which is
  how the game models hunger sharpening the sugar sense (dopamine/NPF do this in real flies).
* **Synaptic plasticity**: a hook for :class:`virtual_fly.plasticity.MushroomBodyPlasticity`,
  dopamine-gated depression of Kenyon-cell-to-MBON synapses, i.e. associative learning.
* **Recording** of every spike, **monitors** of population rates over time, and **checkpoints**
  that capture the whole brain state so an experiment can be replayed or branched.

The integrator itself is the starter kit's dense NumPy loop (the fastest way to update 176k
identical neurons), plus one guard the starter lacks: values that have decayed below a microvolt
are snapped to zero, because float32 numbers drifting into the denormal range slow every array
operation several-fold, which halved the speed of a busy, never-quiet game brain. A brain at rest
with nothing on the way costs nothing, and ``dt=1.0`` halves the cost when a machine cannot keep up.

Use it from your own code::

    from virtual_fly import load_connectome, FlyBrain
    brain = FlyBrain(load_connectome())
    brain.stimulate("LB3b,LB3c", 120)            # drive sugar-sensing neurons at 120 spikes/s
    brain.run(500)                               # simulate 500 ms
    print(brain.rate("MN9"))                     # proboscis motor neuron firing rate (Hz)
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .connectome import Connectome

MV_PER_SYNAPSE = 0.275       # Shiu et al. 2024
TAU_M, TAU_S = 20.0, 5.0      # ms
THETA = 7.0                   # mV above rest
REFRACTORY_MS, DELAY_MS = 2.2, 1.8


@dataclass
class Stimulus:
    """A population made to fire as a Poisson process at ``hz`` spikes/s."""
    spec: str
    idx: np.ndarray
    hz: float


@dataclass
class Monitor:
    """A rolling record of one population's firing rate, one value per ``bin_ms``."""
    name: str
    spec: str
    idx: np.ndarray
    bin_ms: float
    history: list = field(default_factory=list)
    _count: int = 0            # spike_count total of the population at the last bin boundary
    _t_start: float = 0.0


class FlyBrain:
    """Leaky integrate-and-fire network with the parameters of Shiu et al. 2024.

    Each neuron has a voltage ``v`` and a synaptic input ``g`` (both in mV, measured from rest)::

        tau_m dv/dt = -v + g          tau_m = 20 ms   (the leak pulls v back to rest)
        tau_s dg/dt = -g              tau_s = 5 ms    (incoming kicks fade out)

    When ``v`` crosses threshold (7 mV above rest, i.e. -45 mV) the neuron spikes: ``v`` and ``g``
    reset to 0, it is silent for 2.2 ms, and 1.8 ms later every neuron it connects to gets
    ``g += sign x synapses x 0.275 mV x gain``.

    ``gain = 0.65`` for this male connectome: it has more synapses per neuron than the female brain
    the 0.275 mV figure was tuned on, so the literal value over-excites it. Kenyon cells (the
    learning centre) get their input scaled by 0.25 so that their odour code stays sparse.

    Optional mechanisms (all off unless you ask, see the module docstring):

    ``fatigue_mv`` / ``fatigue_ms``
        threshold increase per spike and its decay time.
    ``std_u`` / ``std_tau_ms``
        short-term depression: each spike uses a fraction ``std_u`` of the neuron's synaptic
        resource, which recovers with time constant ``std_tau_ms``. ``std_u = 0`` disables it.
    ``noise_hz`` / ``noise_mv``
        every neuron receives independent random kicks of ``noise_mv`` at ``noise_hz`` per second.
    ``threshold_jitter``
        standard deviation (mV) of a fixed per-neuron threshold offset, so neurons are not all
        identical (seeded, so reproducible).
    """

    def __init__(self, conn: Connectome, dt: float = 0.5, gain: float = 0.65, kenyon_gain: float = 0.25,
                 fatigue_mv: float = 0.0, fatigue_ms: float = 2000.0,
                 std_u: float = 0.0, std_tau_ms: float = 500.0,
                 noise_hz: float = 0.0, noise_mv: float = 1.0, noise_spec: str = "all",
                 threshold_jitter: float = 0.0, seed: int = 0):
        self.conn = conn
        self.dt = float(dt)
        self.gain, self.kenyon_gain = float(gain), float(kenyon_gain)
        self.fatigue_mv, self.fatigue_ms = float(fatigue_mv), float(fatigue_ms)
        self.decay_f = float(np.exp(-dt / fatigue_ms))
        self._fatigue_block = 20                                  # fatigue fades slowly: update it every 20 steps
        self.std_u, self.std_tau_ms = float(std_u), float(std_tau_ms)
        self.noise_hz, self.noise_mv, self.noise_spec = float(noise_hz), float(noise_mv), noise_spec
        self.threshold_jitter = float(threshold_jitter)
        self.seed = int(seed)
        n = conn.n
        self.n = n
        self.theta = np.float32(THETA)
        self.decay_m = np.float32(np.exp(-dt / TAU_M))
        self.decay_s = np.float32(np.exp(-dt / TAU_S))
        # exact contribution of g to v over one step (solution of the two linear equations above)
        self.coupling = np.float32(TAU_S / (TAU_M - TAU_S) * (np.exp(-dt / TAU_M) - np.exp(-dt / TAU_S)))
        self.ref_steps = max(1, int(round(REFRACTORY_MS / dt)))
        self.delay_steps = max(1, int(round(DELAY_MS / dt)))
        self.n_slots = self.delay_steps + 1
        self.lock = threading.RLock()       # hold it while stepping or changing wiring from another thread

        # synaptic weights, one per connection (mV)
        post_gain = np.ones(n, dtype=np.float32)
        post_gain[conn.select("class:Kenyon_Cell")] = kenyon_gain
        pre_sign = np.repeat(conn.sign, np.diff(conn.row_ptr))
        self._w_original = (conn.n_syn.astype(np.float32) * pre_sign * np.float32(MV_PER_SYNAPSE * gain)
                            * post_gain[conn.post_idx])
        self.row_ptr, self.post_idx = conn.row_ptr, conn.post_idx.astype(np.int64)
        self.out_scale = np.ones(n, dtype=np.float32)       # per-neuron output multiplier (0 = silenced)
        self.silenced: dict[str, np.ndarray] = {}           # spec -> neuron indices
        self.modulated: dict[str, tuple[np.ndarray, float]] = {}   # spec -> (indices, factor)
        self.plasticity = None                              # set by MushroomBodyPlasticity.attach()
        self.w = self._w_original.copy()

        self.rng = np.random.default_rng(seed)
        self.stim: dict[str, Stimulus] = {}
        self._empty = np.zeros(0, dtype=np.int64)
        self._stim_idx = self._empty
        self._stim_p = np.zeros(0)
        self._noise_idx = conn.select(noise_spec) if noise_hz > 0 else self._empty
        self.monitors: dict[str, Monitor] = {}
        self.recording: list | None = None
        self.on_spikes: list[Callable[[np.ndarray, int], None]] = []   # callbacks (spikes, t) each step
        # fixed per-neuron threshold offsets (heterogeneity), seeded separately so stimulation
        # randomness does not change when jitter is toggled
        if threshold_jitter > 0:
            self._theta_i = (THETA + np.random.default_rng(seed + 1).normal(0, threshold_jitter, n)
                             ).clip(2.0, None).astype(np.float32)
        else:
            self._theta_i = np.full(n, THETA, dtype=np.float32)
        self.reset()

    # ------------------------------------------------------------------ state
    def reset(self):
        """Put every neuron back at rest and clear all input (wiring changes are kept)."""
        n = self.n
        self.v = np.zeros(n, dtype=np.float32)
        self.g = np.zeros(n, dtype=np.float32)
        self._tmp = np.zeros(n, dtype=np.float32)
        self.queue = np.zeros((self.n_slots, n), dtype=np.float32)   # delayed synaptic input, one slot per step
        self._pending = np.zeros(self.n_slots, dtype=bool)     # does a queue slot hold input?
        self._recent = [self._empty] * (self.ref_steps - 1)   # who spiked in the last few steps
        self.thr = self._theta_i.copy()                        # current threshold incl. fatigue
        self._thr_varies = self.fatigue_mv > 0 or self.threshold_jitter > 0
        self.std_x = np.ones(n, dtype=np.float32) if self.std_u > 0 else None   # synaptic resource
        self.std_t = np.zeros(n, dtype=np.int64)               # step at which std_x was last updated
        self.spike_count = np.zeros(n, dtype=np.int32)
        self.window_ms = 0.0
        self.t = 0
        self.total_spikes = 0
        self.quiet = True           # True while the whole brain is at rest (lets us skip the maths)
        self._quiet_since = 0
        self.last_spikes = self._empty
        for m in self.monitors.values():
            m.history.clear()
            m._count, m._t_start = 0, 0.0
        if self.plasticity is not None:
            self.plasticity.reset_traces()

    @property
    def time_ms(self) -> float:
        return self.t * self.dt

    # ------------------------------------------------------------------ input
    def stimulate(self, spec: str, hz: float):
        """Make every neuron in ``spec`` fire as a Poisson process at ``hz`` spikes/s (0 = stop)."""
        if hz and hz > 0:
            idx = self.conn.select(spec)
            if idx.size == 0:
                raise ValueError(f"no neurons match '{spec}'")
            self.stim[spec] = Stimulus(spec, idx, float(hz))
        else:
            self.stim.pop(spec, None)
        self._rebuild_stim()

    def set_stimuli(self, rates: dict[str, float]):
        """Replace all stimulation at once: ``{spec: hz}``. Specs with hz <= 0 are ignored."""
        self.stim = {s: Stimulus(s, self.conn.select(s), float(hz)) for s, hz in rates.items() if hz > 0}
        self._rebuild_stim()

    def set_stimulus_arrays(self, idx: np.ndarray, hz: np.ndarray, extra: dict[str, float] | None = None):
        """Low-level input: per-neuron rates (used by the retina, which drives thousands of
        columnar neurons at different rates). ``extra`` adds ordinary spec-based stimuli."""
        self.stim = {}
        if extra:
            self.stim = {s: Stimulus(s, self.conn.select(s), float(h)) for s, h in extra.items() if h > 0}
        self._rebuild_stim(extra_idx=np.asarray(idx, dtype=np.int64), extra_hz=np.asarray(hz, dtype=np.float64))

    def clear_stimuli(self):
        self.stim.clear()
        self._rebuild_stim()

    def _rebuild_stim(self, extra_idx=None, extra_hz=None):
        parts_i = [s.idx for s in self.stim.values()]
        parts_h = [np.full(s.idx.size, s.hz) for s in self.stim.values()]
        if extra_idx is not None and extra_idx.size:
            keep = extra_hz > 0
            parts_i.append(extra_idx[keep])
            parts_h.append(extra_hz[keep])
        if not parts_i:
            self._stim_idx, self._stim_p = self._empty, np.zeros(0)
            return
        idx = np.concatenate(parts_i)
        hz = np.concatenate(parts_h)
        order = np.lexsort((-hz, idx))                 # by neuron, highest rate first
        idx, hz = idx[order], hz[order]
        first = np.ones(idx.size, dtype=bool)
        first[1:] = idx[1:] != idx[:-1]                # a neuron in two populations keeps the higher rate
        self._stim_idx = idx[first]
        self._stim_p = np.minimum(1.0, hz[first] * self.dt / 1000.0)

    # ------------------------------------------------------------------ changing the wiring
    def silence(self, spec: str) -> int:
        """Block the output synapses of a population, like expressing tetanus toxin in real flies.

        The neurons still fire (you can watch them), but nothing downstream hears them.
        Returns the number of neurons silenced."""
        idx = self.conn.select(spec)
        if idx.size == 0:
            raise ValueError(f"no neurons match '{spec}'")
        self.silenced[spec] = idx
        self._rebuild_weights()
        return int(idx.size)

    def unsilence(self, spec: str | None = None):
        """Undo ``silence(spec)``, or all silencing when spec is None."""
        if spec is None:
            self.silenced.clear()
        else:
            self.silenced.pop(spec, None)
        self._rebuild_weights()

    def modulate(self, spec: str, factor: float) -> int:
        """Scale the output strength of a population (1 = normal, 0 = silent, 2 = twice as strong).

        A crude stand-in for neuromodulation: real flies sharpen their sugar sense when hungry by
        releasing dopamine onto the taste neurons' terminals; here ``modulate("LB3b,LB3c", 1.4)``."""
        idx = self.conn.select(spec)
        if idx.size == 0:
            raise ValueError(f"no neurons match '{spec}'")
        if abs(factor - 1.0) < 1e-6:
            self.modulated.pop(spec, None)
        else:
            self.modulated[spec] = (idx, float(factor))
        self._rebuild_weights()
        return int(idx.size)

    def unmodulate(self, spec: str | None = None):
        if spec is None:
            self.modulated.clear()
        else:
            self.modulated.pop(spec, None)
        self._rebuild_weights()

    def silenced_mask(self) -> np.ndarray:
        mask = np.zeros(self.n, dtype=bool)
        for idx in self.silenced.values():
            mask[idx] = True
        return mask

    def _rebuild_weights(self):
        """w = original x per-neuron output scale (silencing, modulation) x plastic scale."""
        scale = np.ones(self.n, dtype=np.float32)
        for idx, factor in self.modulated.values():
            scale[idx] *= np.float32(factor)
        for idx in self.silenced.values():
            scale[idx] = 0.0
        self.out_scale = scale
        self.w = self._w_original * np.repeat(scale, np.diff(self.row_ptr))
        if self.plasticity is not None:
            self.plasticity.reapply(self)

    # ------------------------------------------------------------------ stepping
    REST_MV = 0.01           # below this (v and g) the whole brain counts as back at rest
    FLUSH_MV = 1e-6          # values below this are snapped to 0 every 20 steps (denormal guard)

    @property
    def active(self) -> np.ndarray:
        """Neurons not at rest (non-zero voltage or synaptic input)."""
        return (np.abs(self.v) >= self.REST_MV) | (np.abs(self.g) >= self.REST_MV)

    def step(self) -> np.ndarray:
        """Advance the whole brain by one time step (dt ms). Returns indices of neurons that spiked.

        The integration is dense (a handful of passes over the 176k-element state arrays, which is
        what NumPy does fastest); a brain that is completely at rest with nothing on the way costs
        nothing at all.
        """
        v, g, tmp = self.v, self.g, self._tmp
        slot = self.t % self.n_slots
        if (self.quiet and not self._stim_idx.size and not self._pending[slot] and not self._noise_idx.size):
            if self.plasticity is not None:              # traces keep decaying, memories keep fading
                self.plasticity.step(self, self._empty)
            self.t += 1                                  # nothing is happening anywhere: skip the maths
            self.window_ms += self.dt
            self.last_spikes = self._empty
            if self.monitors:
                self._tick_monitors()
            return self._empty
        if self.quiet and self.fatigue_mv > 0:           # waking up: fatigue kept fading while we slept
            self._fade_fatigue(self.decay_f ** (self.t - self._quiet_since))
        self.quiet = False
        if self._pending[slot]:
            arriving = self.queue[slot]
            g += arriving                                # synaptic kicks that were sent 1.8 ms ago
            arriving.fill(0.0)
            self._pending[slot] = False
        if self._noise_idx.size:                         # background bombardment: a few random kicks
            k = self.rng.poisson(self._noise_idx.size * self.noise_hz * self.dt / 1000.0)
            if k:
                hit = self._noise_idx[self.rng.integers(0, self._noise_idx.size, k)]
                np.add.at(g, hit, np.float32(self.noise_mv))
        # neurons that spiked in the last 2 ms are refractory: frozen at reset, input piles up in g
        refractory = np.concatenate(self._recent) if self._recent else self._empty
        g_frozen = g[refractory]
        v *= self.decay_m                                # leak towards rest ...
        np.multiply(g, self.coupling, out=tmp)
        v += tmp                                         # ... plus synaptic drive
        g *= self.decay_s                                # synaptic input fades
        v[refractory] = 0.0
        g[refractory] = g_frozen
        if self.t % self._fatigue_block == 0:
            if self.fatigue_mv > 0:
                self._fade_fatigue(self.decay_f ** self._fatigue_block)
            # Values that have decayed below a microvolt are snapped to zero. Left alone they drift
            # into the denormal float range, where the CPU slows every array operation several-fold
            # (a busy brain ran at half speed before this). A microvolt is 7,000x below threshold.
            v[np.abs(v) < self.FLUSH_MV] = 0.0
            g[np.abs(g) < self.FLUSH_MV] = 0.0
        spikes = np.flatnonzero(v >= (self.thr if self._thr_varies else self.theta))
        if self._stim_idx.size:                          # stimulated sensory neurons fire at random
            forced = self._stim_idx[self.rng.random(self._stim_idx.size) < self._stim_p]
            if forced.size:
                spikes = np.union1d(spikes, forced)
        if spikes.size:
            v[spikes] = 0.0
            g[spikes] = 0.0
            if self.fatigue_mv > 0:
                self.thr[spikes] += np.float32(self.fatigue_mv)
            self.spike_count[spikes] += 1
            self.total_spikes += spikes.size
            out = (self.t + self.delay_steps) % self.n_slots
            self._send(spikes, self.queue[out])
            self._pending[out] = True
            if self.recording is not None:
                self.recording.append((self.t, spikes.astype(np.int32)))
            for cb in self.on_spikes:
                cb(spikes, self.t)
        if self._recent:
            self._recent = [spikes] + self._recent[:-1]
        if self.plasticity is not None:
            self.plasticity.step(self, spikes)
        self.t += 1
        self.window_ms += self.dt
        self.last_spikes = spikes
        if self.monitors:
            self._tick_monitors()
        if self.t % 200 == 0:
            self._check_quiet()
        return spikes

    def _fade_fatigue(self, factor):
        self.thr -= self._theta_i
        self.thr *= np.float32(factor)
        self.thr += self._theta_i

    def _check_quiet(self):
        """Every 100 ms: if no input is arriving and every neuron is back at rest, go to sleep."""
        if self._stim_idx.size or self._noise_idx.size or self._pending.any() or any(r.size for r in self._recent):
            return
        if np.abs(self.v).max() < self.REST_MV and np.abs(self.g).max() < self.REST_MV:
            self.v.fill(0.0)
            self.g.fill(0.0)
            self.quiet = True
            self._quiet_since = self.t

    def _send(self, spikes, target):
        """Add the synaptic kicks of all spiking neurons to a slot of the delayed-input queue."""
        starts = self.row_ptr[spikes]
        lengths = self.row_ptr[spikes + 1] - starts
        total = int(lengths.sum())
        if total == 0:
            return
        # positions of every outgoing connection of every spiking neuron, back to back
        edges = np.repeat(starts - np.cumsum(lengths) + lengths, lengths) + np.arange(total)
        weights = self.w[edges]
        if self.std_x is not None:                       # short-term depression (Tsodyks-Markram)
            x = self.std_x[spikes]
            elapsed = (self.t - self.std_t[spikes]) * self.dt
            x = 1.0 - (1.0 - x) * np.exp(-elapsed / self.std_tau_ms)     # recovery since last spike
            weights = weights * np.repeat(x, lengths).astype(np.float32)
            self.std_x[spikes] = x * (1.0 - self.std_u)                  # this spike used some resource
            self.std_t[spikes] = self.t
        np.add.at(target, self.post_idx[edges], weights)

    def run(self, ms: float, record: bool = False):
        """Simulate ``ms`` milliseconds. With ``record=True`` returns a list of (time_ms, spike indices)."""
        steps = int(round(ms / self.dt))
        rec = [] if record else None
        for _ in range(steps):
            s = self.step()
            if record and s.size:
                rec.append((self.t * self.dt, s))
        return rec

    def run_until_quiet(self, max_ms: float = 2000.0, check_ms: float = 100.0) -> float:
        """Run with the current input until the brain is silent (or ``max_ms``); returns ms simulated."""
        done = 0.0
        while done < max_ms:
            before = self.total_spikes
            self.run(check_ms)
            done += check_ms
            if self.total_spikes == before:
                break
        return done

    # ------------------------------------------------------------------ output
    def rate(self, spec) -> float:
        """Mean firing rate (Hz per neuron) of a population since the last ``reset_counts()``."""
        idx = self.conn.select(spec)
        if idx.size == 0 or self.window_ms <= 0:
            return 0.0
        return float(self.spike_count[idx].sum()) / idx.size / (self.window_ms / 1000.0)

    def rates(self, specs: dict[str, str] | list[str]) -> dict[str, float]:
        if isinstance(specs, dict):
            return {k: self.rate(v) for k, v in specs.items()}
        return {s: self.rate(s) for s in specs}

    def active_fraction(self, spec) -> float:
        """Fraction of the population that fired at least once since ``reset_counts()``."""
        idx = self.conn.select(spec)
        return float((self.spike_count[idx] > 0).mean()) if idx.size else 0.0

    def top_neurons(self, k: int = 15, exclude_stimulated: bool = False) -> list[tuple[int, float]]:
        """The ``k`` most active neurons since ``reset_counts()`` as (index, Hz)."""
        counts = self.spike_count.astype(np.float64)
        if exclude_stimulated and self._stim_idx.size:
            counts = counts.copy()
            counts[self._stim_idx] = 0
        top = np.argsort(counts)[::-1][:k]
        secs = max(self.window_ms, self.dt) / 1000.0
        return [(int(i), float(counts[i] / secs)) for i in top if counts[i] > 0]

    def top_types(self, k: int = 15, exclude_stimulated: bool = False, by_side: bool = True) -> list[dict]:
        """The most active cell types since ``reset_counts()`` (mean Hz per neuron of the type)."""
        counts = self.spike_count.astype(np.float64)
        if exclude_stimulated and self._stim_idx.size:
            counts = counts.copy()
            counts[self._stim_idx] = 0
        conn = self.conn
        secs = max(self.window_ms, self.dt) / 1000.0
        key = conn.type_idx.astype(np.int64) * (8 if by_side else 1) + (conn.side_idx if by_side else 0)
        uniq, inv = np.unique(key, return_inverse=True)
        tot = np.bincount(inv, weights=counts, minlength=uniq.size)
        size = np.bincount(inv, minlength=uniq.size)
        active = np.bincount(inv, weights=(counts > 0).astype(np.float64), minlength=uniq.size)
        mean_hz = tot / size / secs
        order = np.argsort(-mean_hz)
        out = []
        for u in order[: k + 1]:
            if tot[u] <= 0:
                break
            t_idx = int(uniq[u] // 8) if by_side else int(uniq[u])
            if t_idx == 0:
                continue
            side = conn.tables["sides"][int(uniq[u] % 8)] if by_side else ""
            out.append({"type": conn.tables["types"][t_idx], "side": side, "hz": float(mean_hz[u]),
                        "neurons": int(size[u]), "active": int(active[u])})
        return out[:k]

    def reset_counts(self):
        self.spike_count[:] = 0
        self.window_ms = 0.0
        for m in self.monitors.values():
            m._count = 0

    # ------------------------------------------------------------------ monitors and recording
    def add_monitor(self, name: str, spec: str, bin_ms: float = 10.0) -> Monitor:
        """Keep a rolling history of a population's rate (Hz per neuron), one value per bin."""
        m = Monitor(name, spec, self.conn.select(spec), float(bin_ms))
        m._t_start = self.time_ms
        # start from the population's current count: spikes fired before the monitor existed must not
        # be booked to its first bin
        m._count = int(self.spike_count[m.idx].sum()) if m.idx.size else 0
        self.monitors[name] = m
        return m

    def remove_monitor(self, name: str):
        self.monitors.pop(name, None)

    def _tick_monitors(self, spikes=None):
        """Cheap: only at bin boundaries, from spike_count deltas (reset-aware)."""
        t_ms = self.t * self.dt                          # called after t was advanced
        for m in self.monitors.values():
            if t_ms - m._t_start >= m.bin_ms - 1e-9:
                total = int(self.spike_count[m.idx].sum()) if m.idx.size else 0
                delta = total - m._count
                m._count = total
                hz = delta / max(1, m.idx.size) / (m.bin_ms / 1000.0)
                m.history.append(round(hz, 2))
                if len(m.history) > 100000:
                    del m.history[:50000]
                m._t_start = t_ms

    def start_recording(self):
        """Record every spike (step, indices) until ``stop_recording()``. Memory: ~5 bytes per spike."""
        self.recording = []

    def stop_recording(self):
        rec, self.recording = self.recording, None
        return rec or []

    def recording_arrays(self, rec=None):
        """Turn a recording into two arrays (time_ms float32, neuron int32) for saving with np.savez."""
        rec = self.recording if rec is None else rec
        if not rec:
            return np.zeros(0, np.float32), np.zeros(0, np.int32)
        times = np.concatenate([np.full(s.size, t * self.dt, dtype=np.float32) for t, s in rec])
        idx = np.concatenate([s for _, s in rec])
        return times, idx

    # ------------------------------------------------------------------ checkpoints
    def snapshot(self) -> dict:
        """Capture the full dynamical state (not the wiring) so it can be restored later."""
        snap = {"t": self.t, "v": self.v.copy(), "g": self.g.copy(),
                "queue": self.queue.copy(),
                "pending": self._pending.copy(), "recent": [r.copy() for r in self._recent],
                "thr": self.thr.copy(), "std_x": None if self.std_x is None else self.std_x.copy(),
                "std_t": self.std_t.copy(), "spike_count": self.spike_count.copy(),
                "window_ms": self.window_ms, "total_spikes": self.total_spikes,
                "quiet": self.quiet, "quiet_since": self._quiet_since,
                "rng": self.rng.bit_generator.state,
                "monitors": {k: (m._count, m._t_start, len(m.history)) for k, m in self.monitors.items()}}
        if self.plasticity is not None:
            snap["plasticity"] = self.plasticity.snapshot()
        return snap

    def restore(self, snap: dict):
        self.t = snap["t"]
        self.v[:] = snap["v"]
        self.g[:] = snap["g"]
        self.queue[:] = snap["queue"]
        self._pending[:] = snap["pending"]
        self._recent = [r.copy() for r in snap["recent"]]
        self.thr[:] = snap["thr"]
        if self.std_x is not None and snap["std_x"] is not None:
            self.std_x[:] = snap["std_x"]
        self.std_t[:] = snap["std_t"]
        self.spike_count[:] = snap["spike_count"]
        self.window_ms, self.total_spikes = snap["window_ms"], snap["total_spikes"]
        self.quiet, self._quiet_since = snap["quiet"], snap["quiet_since"]
        self.rng.bit_generator.state = snap["rng"]
        for k, m in self.monitors.items():             # monitors bin from spike_count deltas: rewind them too
            if k in snap.get("monitors", {}):
                m._count, m._t_start, n_hist = snap["monitors"][k]
                del m.history[n_hist:]
            else:                                      # added after the snapshot: restart from the restored counts
                m._count = int(self.spike_count[m.idx].sum()) if m.idx.size else 0
                m._t_start = self.time_ms
        if self.plasticity is not None and "plasticity" in snap:
            self.plasticity.restore(self, snap["plasticity"])

    def settings(self) -> dict:
        """The model parameters, for logging and for the game's 'what is running' panel."""
        return {"dt": self.dt, "gain": self.gain, "kenyon_gain": self.kenyon_gain,
                "fatigue_mv": self.fatigue_mv, "fatigue_ms": self.fatigue_ms,
                "std_u": self.std_u, "std_tau_ms": self.std_tau_ms,
                "noise_hz": self.noise_hz, "noise_mv": self.noise_mv, "noise_spec": self.noise_spec,
                "threshold_jitter": self.threshold_jitter, "seed": self.seed,
                "silenced": sorted(self.silenced), "modulated": {k: v[1] for k, v in self.modulated.items()},
                "plasticity": None if self.plasticity is None else self.plasticity.settings()}
