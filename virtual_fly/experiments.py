"""
Experiments: stimulate, wait, listen, and compare with what real flies do.

An :class:`Experiment` is a stimulus (population specs and rates), a duration, and a list of
readouts with the range a real fly (or the published model) would show. :func:`run_experiment`
runs one across several random seeds and reports mean and spread; :func:`sweep` measures a
dose-response curve; :func:`lesion_scan` asks which relay neurons a behaviour actually needs by
silencing candidates one at a time.

The six *classic* experiments are those of Shiu et al. 2024 and the fly-brain-minecraft benches.
The *extended* set covers the circuits this kit adds (odour coding, courtship, wind, hearing,
computed vision). Every expected range is labelled with where it comes from.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

import numpy as np

from .brain import FlyBrain

# --- stimulus sets (real sensory neuron types from neuPrint) ---
SUGAR = {"LB3b,LB3c": 120, "PhG1a,PhG1b,PhG1c": 100, "LgLG3": 80}   # labellar, pharyngeal and leg sugar cells
BITTER = {"LB1a,LB1b,LB1c,LB1d": 120}
LOOM_RIGHT = {"LC4/R,LPLC2/R": 150}
DUST = {"subclass:wind_gravity,subclass:grooming": 150}
ODOUR_VINEGAR = {"ORN_DM1,ORN_DM4,ORN_VM7d,ORN_DP1m": 80}            # glomeruli tuned to vinegar / fruit esters
ODOUR_GEOSMIN = {"ORN_DA2": 100}                                     # geosmin (mould): innately aversive
SONG = {"prefix:JO-A,prefix:JO-B": 100}                              # Johnston's organ auditory neurons
WIND_LEFT = {"prefix:JO-C/L,prefix:JO-E/L": 100}                     # antennal deflection, left antenna
PHEROMONE = {"prefix:pC1_": 80}                                      # (stand-in) courtship command neurons


@dataclass
class Readout:
    spec: str
    label: str
    lo: float
    hi: float
    source: str = ""


@dataclass
class Experiment:
    name: str
    stimulus: dict[str, float]
    ms: float
    readouts: list[Readout]
    settle_ms: float = 100.0
    tags: tuple[str, ...] = ()
    note: str = ""


def R(spec, label, lo, hi, source=""):
    return Readout(spec, label, lo, hi, source)


SHIU = "Shiu et al. 2024 / fly-brain-minecraft bench"
CLASSIC: list[Experiment] = [
    Experiment("Silence (no input)", {}, 300, [R("all", "whole brain", 0, 0, "model property")], tags=("classic",)),
    Experiment("Sugar on the mouthparts", SUGAR, 600,
               [R("GNG232", "G2N-1 taste interneuron", 20, 60, SHIU), R("DNg67", "Fudog", 10, 40, SHIU),
                R("MN9", "MN9 proboscis motor neuron", 30, 90, SHIU), R("class:Kenyon_Cell", "Kenyon cells", 0, 1, SHIU)],
               tags=("classic", "taste")),
    Experiment("Bitter taste", BITTER, 600,
               [R("GNG087", "Scapula (bitter pathway)", 100, 400, SHIU), R("MN9", "MN9 proboscis motor neuron", 0, 5, SHIU)],
               tags=("classic", "taste")),
    Experiment("Sugar + bitter together", {**SUGAR, **BITTER}, 600,
               [R("MN9", "MN9 proboscis motor neuron", 0, 5, SHIU)], tags=("classic", "taste")),
    Experiment("Something looming on the right", LOOM_RIGHT, 400,
               [R("DNp01", "giant fibre (escape)", 250, 400, SHIU), R("TTMn", "TTMn jump muscle motor neuron", 40, 100, SHIU)],
               tags=("classic", "vision")),
    Experiment("Dust on the antennae", DUST, 600,
               [R("DNg62", "aDN1 grooming neuron", 100, 260, SHIU), R("DNge078", "aDN2 grooming neuron", 80, 200, SHIU)],
               tags=("classic", "touch")),
]

# Extended experiments are filled in by settings.EXTENDED after the probe-calibrated ranges; kept here
# so that `run_all` can find them by tag.
EXTENDED: list[Experiment] = []


def all_experiments() -> list[Experiment]:
    return CLASSIC + EXTENDED


@dataclass
class ReadoutResult:
    label: str
    spec: str
    hz: float
    sd: float
    lo: float
    hi: float
    ok: bool
    per_seed: list[float] = field(default_factory=list)


@dataclass
class ExperimentResult:
    name: str
    readouts: list[ReadoutResult]
    after_sps: float            # spikes/s in the whole brain 1 s after the stimulus ends
    after_note: str
    wall_s: float
    seeds: list[int]

    @property
    def ok(self) -> bool:
        return all(r.ok for r in self.readouts)

    def to_dict(self) -> dict:
        return {"name": self.name, "ok": self.ok, "after_spikes_per_s": self.after_sps, "after": self.after_note,
                "wall_s": round(self.wall_s, 2), "seeds": self.seeds,
                "readouts": [r.__dict__ for r in self.readouts]}


def in_range(hz: float, lo: float, hi: float) -> bool:
    return lo - 1e-9 <= hz <= hi + max(1.0, 0.15 * hi)


def after_note(sps: float) -> str:
    return ("calm" if sps < 1000 else "a small loop keeps firing" if sps < 50000
            else "RUNAWAY LOOP (see README, 'Limitations')")


def run_experiment(brain: FlyBrain, exp: Experiment, seeds=(0,), after_ms: float = 1000.0,
                   verbose: bool = False) -> ExperimentResult:
    """Run one experiment for each seed; readouts are averaged, the after-stimulus test uses the last."""
    per: dict[str, list[float]] = {r.label: [] for r in exp.readouts}
    t0 = time.time()
    after_sps = 0.0
    for seed in seeds:
        brain.rng = np.random.default_rng(seed)
        brain.reset()
        brain.clear_stimuli()
        for spec, hz in exp.stimulus.items():
            brain.stimulate(spec, hz)
        brain.run(exp.settle_ms)                  # let activity spread, then measure
        brain.reset_counts()
        brain.run(exp.ms - exp.settle_ms)
        for r in exp.readouts:
            per[r.label].append(brain.rate(r.spec))
        if exp.stimulus:                          # switch the stimulus off: does the brain calm down?
            brain.clear_stimuli()
            brain.run(after_ms / 2)
            brain.reset_counts()
            brain.run(after_ms / 2)
            after_sps = brain.spike_count.sum() / (after_ms / 2000.0)
    wall = time.time() - t0
    results = []
    for r in exp.readouts:
        vals = per[r.label]
        mean = float(np.mean(vals))
        results.append(ReadoutResult(r.label, r.spec, mean, float(np.std(vals)) if len(vals) > 1 else 0.0,
                                     r.lo, r.hi, in_range(mean, r.lo, r.hi), [float(v) for v in vals]))
    brain.clear_stimuli()
    brain.reset()
    res = ExperimentResult(exp.name, results, float(after_sps), after_note(after_sps) if exp.stimulus else "",
                           wall, list(seeds))
    if verbose:
        print(format_result(res))
    return res


def format_result(res: ExperimentResult) -> str:
    lines = []
    for k, r in enumerate(res.readouts):
        sd = f" ±{r.sd:4.1f}" if r.sd else ""
        flag = "ok" if r.ok else "<-- not the usual result"
        lines.append(f" {res.name if k == 0 else '':34} {r.label:32} {r.hz:6.1f}{sd:6} Hz  {r.lo:g}-{r.hi:g} Hz  {flag}")
    if res.after_note:
        lines.append(f" {'':34} {'1 s after it stops':32} {res.after_sps:8,.0f} spikes/s  {res.after_note}")
    lines.append(f" {'':34} ({res.wall_s:.1f} s wall time, seeds {res.seeds})")
    return "\n".join(lines)


def run_all(brain: FlyBrain, experiments=None, only: str | None = None, seeds=(0,), verbose=True):
    experiments = experiments if experiments is not None else all_experiments()
    if verbose:
        print("\n Experiment                         Neuron                            Rate        Expected")
        print(" " + "-" * 96)
    out = []
    for exp in experiments:
        if only and only.lower() not in exp.name.lower() and only.lower() not in " ".join(exp.tags):
            continue
        out.append(run_experiment(brain, exp, seeds=seeds, verbose=verbose))
    return out


def sweep(brain: FlyBrain, spec: str, rates, watch: list[str], ms: float = 500.0, settle_ms: float = 100.0,
          extra: dict[str, float] | None = None) -> list[dict]:
    """Dose-response: stimulate ``spec`` at each rate and report the watched populations' rates."""
    rows = []
    for hz in rates:
        brain.reset()
        brain.clear_stimuli()
        for s, h in (extra or {}).items():
            brain.stimulate(s, h)
        if hz > 0:
            brain.stimulate(spec, hz)
        brain.run(settle_ms)
        brain.reset_counts()
        brain.run(ms - settle_ms)
        rows.append({"hz": float(hz), **{w: brain.rate(w) for w in watch}})
    brain.clear_stimuli()
    brain.reset()
    return rows


def lesion_scan(brain: FlyBrain, exp: Experiment, candidates: list[str], readout: str, seed: int = 0) -> list[dict]:
    """Silence each candidate population in turn and re-measure one readout.

    Returns rows sorted by how much the readout dropped: the relays the behaviour depends on."""
    base = run_experiment(brain, exp, seeds=(seed,))
    base_hz = next(r.hz for r in base.readouts if r.spec == readout or r.label == readout)
    rows = []
    for spec in candidates:
        try:
            n = brain.silence(spec)
        except ValueError:
            continue
        res = run_experiment(brain, exp, seeds=(seed,))
        hz = next(r.hz for r in res.readouts if r.spec == readout or r.label == readout)
        brain.unsilence(spec)
        rows.append({"silenced": spec, "neurons": n, "hz": hz, "baseline": base_hz,
                     "change": (hz - base_hz) / base_hz if base_hz else 0.0})
    rows.sort(key=lambda r: r["change"])
    return rows


def save_json(results, path, brain: FlyBrain | None = None):
    payload = {"results": [r.to_dict() for r in results]}
    if brain is not None:
        payload["settings"] = brain.settings()
    with open(path, "w") as f:
        json.dump(payload, f, indent=1)
