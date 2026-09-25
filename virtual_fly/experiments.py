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
    profile: str | None = None      # the model profile the expected ranges were measured with (None = any)
    silence: tuple[str, ...] = ()   # populations whose output is blocked for the experiment (a genetic lesion)


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

# The circuits this kit added, with the ranges measured on this machine in the *game* profile
# (docs/SCIENCE.md has every number; the pure model runs away on most of these stimuli).
PROBE = "measured on the real connectome, game profile (docs/SCIENCE.md)"
EXTENDED: list[Experiment] = [
    Experiment("Smell of vinegar", ODOUR_VINEGAR, 600,
               [R("DM1_lPN", "DM1 projection neuron", 150, 500, PROBE),
                R("class:Kenyon_Cell", "Kenyon cells (sparse code)", 0.5, 5, PROBE),
                R("MBON14", "MBON14, a KC-driven MBON", 10, 90, PROBE),
                R("MBON11", "MBON11 (γ1pedc, KC input only: dopamine muted)", 1, 60, PROBE)],
               tags=("extended", "smell"), profile="game"),
    Experiment("Bitter taste → punishment dopamine", BITTER, 600,
               [R("PPL101", "PPL101 (γ1pedc dopamine)", 30, 150, PROBE), R("prefix:PAM", "PAM reward dopamine", 0, 2, PROBE)],
               tags=("extended", "learning"), profile="game"),
    Experiment("A loud sound", {"prefix:JO-B": 100}, 400,
               [R("DNp01", "giant fibre (startle)", 20, 150, PROBE)], tags=("extended", "hearing"), profile="game"),
    Experiment("Courtship command", {"prefix:pC1_": 60}, 500,
               [R("pIP10", "pIP10 song neuron", 30, 150, PROBE), R("DNp13", "DNp13", 5, 90, PROBE)],
               tags=("extended", "courtship"), profile="game"),
    Experiment("Wide-field motion, right eye", {"T4a/R,T5a/R": 100}, 400,
               [R("HSE/R,HSN/R,HSS/R", "HS cells, right", 200, 500, PROBE), R("DNp15/R", "DNp15 right (optomotor)", 50, 300, PROBE),
                R("DNa02/R", "DNa02 right (turn right)", 40, 250, PROBE), R("DNa02/L", "DNa02 left", 0, 20, PROBE)],
               tags=("extended", "vision"), profile="game"),
]


# Genetic experiments: a population defined by gene expression (the MaleCNS fruitless/doublesex
# annotation, see genetics.py) is silenced, as a fly lab does with tetanus toxin or Kir2.1 under a
# driver, and the reflexes are re-measured. Ranges measured in the game profile, seeds 0 and 1.
GENETIC = [
    Experiment("Courtship command: song motor neurons", {"prefix:pC1_": 60}, 500,
               [R("regex:^ps1", "ps1 song motor neurons", 20, 120, PROBE), R("regex:^hg", "hg1/hg2 song motor neurons", 20, 120, PROBE),
                R("regex:^DLMn", "DLMn wing power motor neurons", 8, 60, PROBE)],
               tags=("genetic", "courtship"), profile="game",
               note="the intact fly, for comparison with the two lesions below"),
    Experiment("Courtship command, fruitless neurons silenced", {"prefix:pC1_": 60}, 500,
               [R("prefix:pC1_", "pC1 (driven; mostly dsx, not silenced)", 40, 90, PROBE),
                R("pIP10", "pIP10 (fires, but its output is blocked)", 15, 80, PROBE),
                R("regex:^ps1", "ps1 song motor neurons", 0, 3, PROBE), R("regex:^hg", "hg1/hg2 song motor neurons", 0, 3, PROBE),
                R("regex:^DLMn", "DLMn wing power motor neurons", 0, 3, PROBE)],
               tags=("genetic", "courtship"), profile="game", silence=("gene:fru",),
               note="fruitless males do not sing (Demir & Dickson 2005): pIP10 and its route to the wing motor neurons are fru+"),
    Experiment("Courtship command, doublesex neurons silenced", {"prefix:pC1_": 60}, 500,
               [R("pIP10", "pIP10 song neuron", 0, 3, PROBE), R("regex:^ps1", "ps1 song motor neurons", 0, 3, PROBE),
                R("regex:^hg", "hg1/hg2 song motor neurons", 0, 3, PROBE)],
               tags=("genetic", "courtship"), profile="game", silence=("gene:dsx",),
               note="pC1 itself is dsx+: with its output blocked nothing downstream moves"),
    Experiment("Sugar on the mouthparts, fruitless neurons silenced", {"LB3b,LB3c": 120}, 500,
               [R("GNG232", "G2N-1 taste interneuron", 15, 60, PROBE), R("MN9", "MN9 proboscis motor neuron", 15, 60, PROBE)],
               tags=("genetic", "taste"), profile="game", silence=("gene:fru",),
               note="control: feeding does not run through fruitless neurons"),
    Experiment("Something looming on the right, fruitless neurons silenced", {"LC4/R,LPLC2/R": 150}, 400,
               [R("DNp01", "giant fibre (escape)", 250, 400, PROBE), R("TTMn", "TTMn jump muscle motor neuron", 40, 100, PROBE)],
               tags=("genetic", "escape"), profile="game", silence=("gene:fru",),
               note="control: the escape circuit is not fruitless-dependent"),
]


SEEDS = (0, 1, 2, 3, 4)   # a verdict on a fly: every experiment on five seeds (five runs of the same fly)


def survival(brain: FlyBrain, experiments=None, profile: str | None = None, on_progress=None,
             seeds=SEEDS) -> list[dict]:
    """Run the experiments one by one and report, per experiment, whether every readout was in its
    range (on the mean over ``seeds``): the survival report of a grown fly (see :mod:`virtual_fly.wiring`).
    ``fragile`` marks an experiment that passes on the mean although some seed on its own does not. An
    experiment whose populations do not exist in this connectome is reported as ``ok: None``.
    ``on_progress(rows)`` is called after each experiment with the rows so far."""
    experiments = experiments if experiments is not None else CLASSIC + EXTENDED
    rows: list[dict] = []
    for exp in experiments:
        if profile is not None and exp.profile is not None and exp.profile != profile:
            continue
        try:
            res = run_experiment(brain, exp, seeds=seeds)
            rows.append({"name": exp.name, "ok": res.ok, "fragile": res.fragile, "seeds": len(res.seeds),
                         "readouts": [{"label": r.label, "hz": round(r.hz, 1), "lo": r.lo, "hi": r.hi, "ok": r.ok,
                                       "per_seed": [round(v, 1) for v in r.per_seed], "seeds_out": r.seeds_out}
                                      for r in res.readouts]})
        except (ValueError, KeyError) as e:
            rows.append({"name": exp.name, "ok": None, "readouts": [], "error": str(e)})
        if on_progress is not None:
            on_progress(list(rows))
    return rows


def all_experiments() -> list[Experiment]:
    return CLASSIC + EXTENDED + GENETIC


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
    seeds_out: int = 0              # how many seeds, on their own, fall outside the range


@dataclass
class ExperimentResult:
    name: str
    readouts: list[ReadoutResult]
    after_sps: float            # spikes/s in the whole brain 1 s after the stimulus ends
    after_note: str
    wall_s: float
    seeds: list[int]
    silenced: list[str] = field(default_factory=list)   # populations whose output was blocked

    @property
    def ok(self) -> bool:
        return all(r.ok for r in self.readouts)

    @property
    def fragile(self) -> bool:
        """Passes on the mean, but some seed on its own falls outside a range."""
        return self.ok and any(r.seeds_out for r in self.readouts)

    def to_dict(self) -> dict:
        return {"name": self.name, "ok": self.ok, "after_spikes_per_s": self.after_sps, "after": self.after_note,
                "wall_s": round(self.wall_s, 2), "seeds": self.seeds,
                "readouts": [r.__dict__ for r in self.readouts], "silenced": list(self.silenced)}


def in_range(hz: float, lo: float, hi: float) -> bool:
    return lo - 1e-9 <= hz <= hi + max(1.0, 0.15 * hi)


def after_note(sps: float) -> str:
    return ("calm" if sps < 1000 else "a small loop keeps firing" if sps < 50000
            else "RUNAWAY LOOP (see README, 'Limitations')")


def kept_learning(brain: FlyBrain):
    """The brain's learned Kenyon-cell -> MBON strengths as they are now, and a function that puts them
    back. ``reset()`` clears the learning traces but not what was learned, so without this one run's
    learning would carry into the next seed and the next experiment."""
    pl = brain.plasticity
    if pl is None:
        return lambda: None
    saved = pl.scale.copy()

    def back():
        pl.scale[:] = saved
        pl.reapply(brain)
    return back


def run_experiment(brain: FlyBrain, exp: Experiment, seeds=(0,), after_ms: float = 1000.0,
                   verbose: bool = False) -> ExperimentResult:
    """Run one experiment for each seed; readouts are averaged, the after-stimulus test uses the last.

    Every seed starts from what the fly had learned when the experiment began (learning during a run
    counts, as in a real fly, but does not carry into the next seed), and the fly is left as it was."""
    per: dict[str, list[float]] = {r.label: [] for r in exp.readouts}
    t0 = time.time()
    after_sps = 0.0
    restore_learning = kept_learning(brain)
    for spec in exp.silence:                      # the lesion: like expressing tetanus toxin in those cells
        brain.silence(spec)
    try:
        for seed in seeds:
            restore_learning()
            brain.rng = np.random.default_rng(seed)
            brain.reset()
            brain.clear_stimuli()
            for spec, hz in exp.stimulus.items():
                brain.stimulate(spec, hz)
            brain.run(exp.settle_ms)              # let activity spread, then measure
            brain.reset_counts()
            brain.run(exp.ms - exp.settle_ms)
            for r in exp.readouts:
                per[r.label].append(brain.rate(r.spec))
            if exp.stimulus:                      # switch the stimulus off: does the brain calm down?
                brain.clear_stimuli()
                brain.run(after_ms / 2)
                brain.reset_counts()
                brain.run(after_ms / 2)
                after_sps = brain.spike_count.sum() / (after_ms / 2000.0)
    finally:
        for spec in exp.silence:
            brain.unsilence(spec)
        restore_learning()
    wall = time.time() - t0
    results = []
    for r in exp.readouts:
        vals = per[r.label]
        mean = float(np.mean(vals))
        results.append(ReadoutResult(r.label, r.spec, mean, float(np.std(vals)) if len(vals) > 1 else 0.0,
                                     r.lo, r.hi, in_range(mean, r.lo, r.hi), [float(v) for v in vals],
                                     seeds_out=sum(not in_range(v, r.lo, r.hi) for v in vals)))
    brain.clear_stimuli()
    brain.reset()
    res = ExperimentResult(exp.name, results, float(after_sps), after_note(after_sps) if exp.stimulus else "",
                           wall, list(seeds), silenced=list(exp.silence))
    if verbose:
        print(format_result(res))
    return res


def format_result(res: ExperimentResult) -> str:
    lines = []
    for k, r in enumerate(res.readouts):
        sd = f" ±{r.sd:4.1f}" if r.sd else ""
        flag = "ok" if r.ok else "<-- not the usual result"
        if r.ok and r.seeds_out:
            flag = f"ok on the mean, but {r.seeds_out} of {len(r.per_seed)} seeds outside"
        lines.append(f" {res.name if k == 0 else '':34} {r.label:32} {r.hz:6.1f}{sd:6} Hz  {r.lo:g}-{r.hi:g} Hz  {flag}")
    if res.silenced:
        lines.append(f" {'':34} (output blocked in {', '.join(res.silenced)})")
    if res.after_note:
        lines.append(f" {'':34} {'1 s after it stops':32} {res.after_sps:8,.0f} spikes/s  {res.after_note}")
    lines.append(f" {'':34} ({res.wall_s:.1f} s wall time, seeds {res.seeds})")
    return "\n".join(lines)


def run_all(brain: FlyBrain, experiments=None, only: str | None = None, seeds=(0,), verbose=True,
            profile: str | None = None):
    """Run experiments; those whose expected ranges were measured with another profile are skipped
    (say ``profile=None`` to run everything regardless)."""
    experiments = experiments if experiments is not None else all_experiments()
    if verbose:
        print("\n Experiment                         Neuron                            Rate        Expected")
        print(" " + "-" * 96)
    out, skipped = [], []
    for exp in experiments:
        if only and only.lower() not in exp.name.lower() and only.lower() not in " ".join(exp.tags):
            continue
        if profile is not None and exp.profile is not None and exp.profile != profile:
            skipped.append(exp)
            continue
        out.append(run_experiment(brain, exp, seeds=seeds, verbose=verbose))
    if verbose and skipped:
        print(f"\n (skipped {len(skipped)} experiments whose ranges were measured with the "
              f"'{skipped[0].profile}' profile: {', '.join(e.name for e in skipped)}; run them with --profile {skipped[0].profile})")
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
    if not any(r.spec == readout or r.label == readout for r in exp.readouts):
        raise ValueError(f"'{readout}' is not a readout of experiment '{exp.name}': "
                         f"{[r.label for r in exp.readouts]}")
    base = run_experiment(brain, exp, seeds=(seed,))
    base_hz = next(r.hz for r in base.readouts if r.spec == readout or r.label == readout)
    rows = []
    for spec in candidates:
        try:
            n = brain.silence(spec)
        except ValueError:
            continue
        try:
            res = run_experiment(brain, exp, seeds=(seed,))
            hz = next(r.hz for r in res.readouts if r.spec == readout or r.label == readout)
        finally:
            brain.unsilence(spec)                    # never leave a lesion behind, whatever happened
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
