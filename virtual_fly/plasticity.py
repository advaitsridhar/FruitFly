"""
Associative learning in the mushroom body: dopamine-gated depression of KC-to-MBON synapses.

This is the one place in the fly brain where learning is understood well enough to write down:

* ~4,000 **Kenyon cells** (KCs) carry a sparse code for what the fly smells.
* Their axons run through 15 **compartments** of the mushroom-body lobes. In each compartment
  they synapse onto one or two **mushroom body output neurons** (MBONs), and each compartment is
  watered by its own **dopaminergic neurons** (DANs): the PPL1 cluster signals punishment, the
  PAM cluster signals reward.
* The learning rule (Hige et al. 2015; Handler et al. 2019): a KC-to-MBON synapse is *depressed*
  when the KC was active shortly *before* dopamine arrives in that compartment. Nothing else
  changes. Odour + punishment therefore weakens the odour's drive onto the MBONs in the
  PPL1 compartments, which happen to be the approach-promoting ones (Aso et al. 2014), so the fly
  now avoids that odour; odour + reward weakens its drive onto the avoidance-promoting MBONs in the
  PAM compartments, so the fly approaches it.

Everything structural here is read from the wiring diagram, not hand-coded: which synapses are
plastic (KC -> MBON), and which DANs gate which MBON (the DANs that synapse directly onto that
MBON, which in this data are its compartment's DANs). Only the rule's constants are chosen
by hand and labelled below.
"""

from __future__ import annotations

import numpy as np

from .connectome import Connectome

APPROACH_NTS = ("acetylcholine", "gaba")     # MBON valence by transmitter (Aso et al. 2014, eLife)
AVOID_NTS = ("glutamate",)
FAST_NTS = ("acetylcholine", "gaba", "glutamate")


def mbon_transmitters(conn: Connectome, idx) -> np.ndarray:
    """The MBONs' transmitters, taking the file's literature transmitter where it names exactly one fast one.
    The female fly's FlyWire prediction leaves MBON03, MBON05 and MBON07 'unclear', calls one MBON02 GABAergic
    and three MBON10 cells glutamatergic, where her literature column (``meta["known_nt"]``) says glutamate
    and GABA; the male file has no such table, so its predictions are used as they are."""
    idx = np.asarray(idx)
    nt = conn.nt[idx].copy()
    known = (getattr(conn, "meta", None) or {}).get("known_nt") or {}
    if known:
        for k, i in enumerate(idx.tolist()):
            fast = [x for x in known.get(conn.types[i], {}).get("nt", []) if x in FAST_NTS]
            if len(fast) == 1:
                nt[k] = fast[0]
    return nt


def mbon_valence(conn: Connectome, mbon_spec: str = "class:MBON") -> tuple[np.ndarray, np.ndarray]:
    """Per-MBON valence: +1 approach-promoting (cholinergic, GABAergic), -1 avoidance-promoting
    (glutamatergic), 0 unknown. Returns (indices, valence)."""
    idx = conn.select(mbon_spec)
    nt = mbon_transmitters(conn, idx)
    val = np.where(np.isin(nt, APPROACH_NTS), 1, np.where(np.isin(nt, AVOID_NTS), -1, 0))
    return idx, val.astype(np.int8)


class MushroomBodyPlasticity:
    """Dopamine-gated depression of KC->MBON synapses (attach to a :class:`FlyBrain`).

    Hand-chosen constants:

    ``rate``          fraction of remaining strength lost per second at full KC eligibility x full
                      dopamine (0.5 = a strongly paired odour loses half its drive in ~1 s of pairing)
    ``kc_tau_ms``     how long a KC spike leaves the synapse "eligible" for depression (real flies:
                      dopamine within ~1-2 s after the odour; forward pairing)
    ``dan_tau_ms``    how long a DAN spike's dopamine lingers in the compartment
    ``floor``         a synapse can lose at most this much (0.15 = never below 15 % of original)
    ``recover_min``   memory decay: minutes for a depressed synapse to recover most of its strength
                      (short-term memory in flies fades over tens of minutes; 0 = never)
    ``dan_scale``     dopamine trace is normalised so that this many DAN spikes/s in a compartment
                      count as "full" dopamine
    ``dan_floor``     dopamine below this fraction of "full" does nothing. Odours alone make some
                      PPL1 neurons fire in this model (PPL102 at ~100 Hz); the floor keeps that from
                      teaching the fly to avoid everything it smells, while bitter taste, shock and
                      reward (which saturate their compartments) get through.
    """

    def __init__(self, kc_spec: str = "class:Kenyon_Cell", mbon_spec: str = "class:MBON",
                 dan_spec: str = "class:DAN", rate: float = 0.5, kc_tau_ms: float = 1500.0,
                 dan_tau_ms: float = 400.0, floor: float = 0.15, recover_min: float = 20.0,
                 dan_scale: float = 40.0, dan_floor: float = 0.5, block_steps: int = 20):
        self.kc_spec, self.mbon_spec, self.dan_spec = kc_spec, mbon_spec, dan_spec
        self.rate, self.kc_tau_ms, self.dan_tau_ms = float(rate), float(kc_tau_ms), float(dan_tau_ms)
        self.floor, self.recover_min, self.dan_scale = float(floor), float(recover_min), float(dan_scale)
        self.dan_floor = float(dan_floor)
        self.block_steps = int(block_steps)
        self.enabled = True
        self.brain = None

    # ------------------------------------------------------------------ wiring
    def attach(self, brain):
        conn: Connectome = brain.conn
        self.brain = brain
        self.kc = conn.select(self.kc_spec)
        self.mbon = conn.select(self.mbon_spec)
        self.dan = conn.select(self.dan_spec)
        n = conn.n
        self.kc_local = np.full(n, -1, dtype=np.int64)
        self.kc_local[self.kc] = np.arange(self.kc.size)
        self.dan_local = np.full(n, -1, dtype=np.int64)
        self.dan_local[self.dan] = np.arange(self.dan.size)
        mbon_local = np.full(n, -1, dtype=np.int64)
        mbon_local[self.mbon] = np.arange(self.mbon.size)
        # plastic synapses: every KC -> MBON connection
        self.pe = conn.edges_between(self.kc, self.mbon)
        self.pe_kc = self.kc_local[conn.pre_idx[self.pe]]
        self.pe_mbon = mbon_local[conn.post_idx[self.pe]]
        self.mbon_nt = mbon_transmitters(conn, self.mbon)       # for the valence (the literature's where it has one)
        # dopamine gating: DAN -> MBON direct synapses define each MBON's compartment DANs
        dm = conn.edges_between(self.dan, self.mbon)
        self.dm_dan = self.dan_local[conn.pre_idx[dm]]
        self.dm_mbon = mbon_local[conn.post_idx[dm]]
        w = conn.n_syn[dm].astype(np.float64)
        tot = np.bincount(self.dm_mbon, weights=w, minlength=self.mbon.size)
        self.dm_w = (w / np.maximum(tot[self.dm_mbon], 1.0)).astype(np.float32)   # sums to 1 per MBON
        self.mbon_has_dan = tot > 0
        self.scale = np.ones(self.pe.size, dtype=np.float32)
        self.base = None
        self.reapply(brain)
        self.reset_traces()
        brain.plasticity = self
        return self

    def reapply(self, brain):
        """Recompute the plastic edges' base weights (after silencing/modulation) and apply the scale."""
        self.base = (brain._w_original[self.pe] * brain.out_scale[brain.conn.pre_idx[self.pe]]).astype(np.float32)
        brain.w[self.pe] = self.base * self.scale

    def reset_traces(self):
        self.kc_trace = np.zeros(self.kc.size, dtype=np.float32)
        self.dan_trace = np.zeros(self.dan.size, dtype=np.float32)
        self._kc_acc = np.zeros(self.kc.size, dtype=np.float32)
        self._dan_acc = np.zeros(self.dan.size, dtype=np.float32)
        self.dopamine = np.zeros(self.mbon.size, dtype=np.float32)
        self._last_block = 0
        self.events = 0                       # number of blocks in which some synapse was depressed

    def reset_weights(self):
        """Forget everything: every KC->MBON synapse back to its original strength."""
        self.scale[:] = 1.0
        if self.brain is not None:
            self.reapply(self.brain)

    # ------------------------------------------------------------------ dynamics
    def step(self, brain, spikes: np.ndarray):
        if spikes.size:
            k = self.kc_local[spikes]
            k = k[k >= 0]
            if k.size:
                np.add.at(self._kc_acc, k, 1.0)
            d = self.dan_local[spikes]
            d = d[d >= 0]
            if d.size:
                np.add.at(self._dan_acc, d, 1.0)
        if (brain.t + 1) % self.block_steps:
            return
        block_ms = self.block_steps * brain.dt
        # traces: exponential decay plus the spikes of this block
        self.kc_trace *= np.float32(np.exp(-block_ms / self.kc_tau_ms))
        self.dan_trace *= np.float32(np.exp(-block_ms / self.dan_tau_ms))
        if self._kc_acc.any():
            self.kc_trace += self._kc_acc
            self._kc_acc.fill(0.0)
        if self._dan_acc.any():
            self.dan_trace += self._dan_acc
            self._dan_acc.fill(0.0)
        if not self.enabled:
            return
        changed = False
        if self.dan_trace.any() and self.kc_trace.any():
            # dopamine per MBON compartment, normalised: dan_scale spikes/s sustained == 1.0
            dan_rate = self.dan_trace / (self.dan_tau_ms / 1000.0)          # ~ spikes per second
            self.dopamine = np.bincount(self.dm_mbon, weights=self.dm_w * dan_rate[self.dm_dan],
                                        minlength=self.mbon.size).astype(np.float32)
            self.dopamine = np.minimum(1.0, self.dopamine / np.float32(self.dan_scale))
            if self.dan_floor > 0:
                self.dopamine = np.maximum(0.0, self.dopamine - np.float32(self.dan_floor)) / np.float32(1.0 - self.dan_floor)
            elig = np.minimum(1.0, self.kc_trace / np.float32(3.0))         # ~3 recent spikes = full
            drive = elig[self.pe_kc] * self.dopamine[self.pe_mbon]
            if drive.any():
                dw = np.float32(self.rate * block_ms / 1000.0) * drive
                self.scale *= (1.0 - dw)
                np.maximum(self.scale, np.float32(self.floor), out=self.scale)
                changed = True
                self.events += 1
        else:
            self.dopamine.fill(0.0)
        if self.recover_min > 0 and (self.scale < 1.0).any():
            k = np.float32(block_ms / (self.recover_min * 60000.0))
            self.scale += (1.0 - self.scale) * k
            changed = True
        if changed:
            brain.w[self.pe] = self.base * self.scale

    # ------------------------------------------------------------------ inspection
    def summary(self, conn: Connectome | None = None) -> dict:
        """Per MBON type: mean remaining strength of its KC inputs (all of them, and weighted by the
        Kenyon cells active right now, i.e. what the current odour experiences), valence, dopamine."""
        conn = conn or self.brain.conn
        types = conn.types[self.mbon]
        nt = self.mbon_nt if conn is self.brain.conn else mbon_transmitters(conn, self.mbon)
        per_mbon = np.bincount(self.pe_mbon, weights=self.scale, minlength=self.mbon.size)
        cnt = np.bincount(self.pe_mbon, minlength=self.mbon.size)
        mean = np.where(cnt > 0, per_mbon / np.maximum(cnt, 1), 1.0)
        act = np.minimum(1.0, self.kc_trace / 3.0)[self.pe_kc]
        w_act = np.bincount(self.pe_mbon, weights=act, minlength=self.mbon.size)
        s_act = np.bincount(self.pe_mbon, weights=act * self.scale, minlength=self.mbon.size)
        mean_act = np.where(w_act > 1e-6, s_act / np.maximum(w_act, 1e-6), mean)
        out = {}
        for t in sorted(set(types.tolist())):
            m = types == t
            valence = 1 if nt[m][0] in APPROACH_NTS else -1 if nt[m][0] in AVOID_NTS else 0
            out[t] = {"strength": float(mean[m].mean()), "now": float(mean_act[m].mean()),
                      "valence": valence, "nt": nt[m][0],
                      "dopamine": float(self.dopamine[m].mean()), "kc_synapses": int(cnt[m].sum())}
        return out

    def depressed_fraction(self) -> float:
        return float((self.scale < 0.999).mean()) if self.scale.size else 0.0

    def settings(self) -> dict:
        return {"rate": self.rate, "kc_tau_ms": self.kc_tau_ms, "dan_tau_ms": self.dan_tau_ms,
                "floor": self.floor, "recover_min": self.recover_min, "dan_scale": self.dan_scale,
                "dan_floor": self.dan_floor,
                "plastic_synapses": int(self.pe.size), "enabled": self.enabled}

    def snapshot(self) -> dict:
        return {"scale": self.scale.copy(), "kc_trace": self.kc_trace.copy(), "dan_trace": self.dan_trace.copy(),
                "kc_acc": self._kc_acc.copy(), "dan_acc": self._dan_acc.copy(), "events": self.events}

    def restore(self, brain, snap: dict):
        self.scale[:] = snap["scale"]
        self.kc_trace[:] = snap["kc_trace"]
        self.dan_trace[:] = snap["dan_trace"]
        self._kc_acc[:] = snap["kc_acc"]
        self._dan_acc[:] = snap["dan_acc"]
        self.events = snap["events"]
        brain.w[self.pe] = self.base * self.scale

    def save(self, path):
        np.savez_compressed(path, edges=self.pe, scale=self.scale)

    def load(self, path):
        z = np.load(path)
        if z["edges"].shape != self.pe.shape or not np.array_equal(z["edges"], self.pe):
            raise ValueError("this weight file was made with a different connectome or plastic set")
        self.scale[:] = z["scale"]
        self.reapply(self.brain)
