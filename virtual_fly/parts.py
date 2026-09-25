"""
The genes as each neuron's parts list (level 2 of the genome plan; docs/SCIENCE.md section 8).

:mod:`virtual_fly.genetics` reads which genes a neuron expresses (level 1) and
:mod:`virtual_fly.wiring` grows the wiring from rules (level 3). This module is what sits between
them: the genes decide what kind of machine each neuron is. The published model (Shiu et al. 2024)
gives every neuron the same machine, a leaky integrate-and-fire unit whose only per-neuron property
is the sign of its transmitter. Two facts about the fly's parts are solid enough to build in, and
both change that model:

* **Dopamine, octopamine and serotonin have no fast receptors in the fly.** Every receptor for
  them is G-protein-coupled (Dop1R1, Dop1R2, Dop2R, DopEcR; Oamb, Octα2R, Octβ1R-3R; 5-HT1A,
  5-HT1B, 5-HT2A, 5-HT2B, 5-HT7), so a spike in one of these neurons cannot make a fast synaptic
  potential in its targets. The published model treats them as ordinary excitatory neurons. Here
  their fast synapses are removed and each spike instead adds to a slow *tone* on its targets that
  lingers for seconds and scales how strongly those targets respond to all their other inputs.
* **Many optic-lobe cell types do not spike.** Photoreceptors, the lamina monopolar cells L1-L5,
  the medulla columnar cells that feed the motion detectors, the motion detectors T4/T5 themselves
  and the HS/VS tangential cells signal with graded potentials. Here they release transmitter in
  proportion to their depolarisation instead of firing all-or-none spikes: a graded cell transmits
  below the spike threshold, saturates at it, and has no reset and no refractory period.

The third piece is a table hook, :data:`CELL_PARAMS`: per-type overrides of the spike threshold and
of the graded flag, in the same population-spec language as everything else, so that measured
values (from patch recordings or, one day, from ion-channel expression in the Fly Cell Atlas) can be
dropped in without touching the model. It ships empty apart from what the two facts above need.

Two more inputs come from Virtual Fly Brain through :mod:`virtual_fly.vfb` (v2.5):

* **Curated transmitters.** Where the fly anatomy ontology's literature-curated class of a cell type
  says its transmitter differs from the connectome's synapse-shape prediction (the DPM neuron is
  GABAergic and serotonergic, not dopaminergic; OA-ASM3 is octopaminergic, not serotonergic; Mi15
  releases dopamine as well as acetylcholine), or where the prediction is "unclear" and the curated
  class asserts one, the parts list gives the neuron the literature's machine: the right tone, its
  fast synapses kept or removed, the right sign. ``curated`` picks the policy (see
  :func:`virtual_fly.vfb.transmitter_overrides`; ``"all"`` also lets the literature overrule confident
  fast predictions and other connectomes' predictions fill "unclear" ones).
* **Receptor signs.** A modulator's effect on a target follows the receptors the target's cell type
  expresses in the adult single-cell RNA-seq atlases (:data:`RECEPTORS`): Gs- and Gq-coupled receptors
  raise the target's gain, Gi-coupled ones lower it, each weighted by the fraction of cells expressing
  it. Targets whose type has no adult cluster feel no tone (``unknown_sign``, 0; v2.7 gave them the
  modulator's one net sign, +1).

Two more tables came from chasing the one readout the parts list missed (v2.6; docs/SCIENCE.md 8.2):

* :data:`RECEPTOR_FACTS`: receptors a cell type is shown to use by direct evidence in that type, for
  types no atlas cluster covers (APL uses Dop2R, so dopamine lowers its gain; Zhou et al. 2019), or the
  measured effect of a modulator where the receptor is unknown (octopamine raises the VS cells'
  gain; Suver et al. 2012). Since v2.8 a target with neither feels no tone at all.
* :data:`LOCAL`: wide-field neurons whose release follows the activity around each target rather
  than the whole cell's. APL's activity and inhibition stay local (Amin et al. 2020); here its
  release onto a target scales with how active the Kenyon cells of that target's lobe system are
  compared with APL's mean input, never above the whole cell's.

Everything is a :class:`PartsList`; ``FlyBrain(conn, parts=True)`` uses the default one, and the
game's Genome card switches it on and off and shows the tone of each modulator.
"""

from __future__ import annotations

import gzip
import json
from dataclasses import dataclass, field, replace

import numpy as np

from .brain import THETA
from .connectome import PROJECT_DIR, Connectome

# ------------------------------------------------------------------ the tables
@dataclass(frozen=True)
class Modulator:
    """A transmitter that acts through slow receptors only."""
    nt: str                  # the MaleCNS transmitter prediction it applies to
    label: str
    genes: tuple[str, ...]   # synthesis and transport genes (symbols as in genetics.GENES)
    receptors: str           # the fly's receptors for it (all G-protein-coupled)
    tau_ms: float            # how long the tone lingers on a target
    gain: float              # change of the targets' input gain at saturation (+0.5 = respond 50 % more)
    syn_ref: float = 10.0    # one spike through this many synapses raises the target's level by one unit
    half: float = 5.0        # level at which the effect is half its maximum (a sustained 10 Hz through
                             # syn_ref synapses with tau 0.5 s sits at level 5)
    why: str = ""


@dataclass(frozen=True)
class Receptor:
    """A slow receptor: which modulator it answers to and which way its G protein pushes the gain."""
    gene: str
    fbgn: str
    modulator: str
    coupling: str            # Gs / Gq (raise the gain) or Gi (lower it)
    sign: int
    why: str = ""


@dataclass(frozen=True)
class Graded:
    """A cell type that signals with graded potentials rather than spikes."""
    spec: str
    label: str
    why: str = ""


@dataclass(frozen=True)
class Local:
    """A wide-field neuron whose activity stays local: where it releases follows the activity of the input
    groups around each of its targets rather than the whole cell's."""
    spec: str                       # the neuron(s), usually one per hemisphere
    groups: tuple[str, ...]         # the input populations whose territories are its compartments
    label: str
    why: str = ""
    tau_ms: float = 20.0            # how long a compartment's activity lingers (the membrane time constant)
    by_region: bool = True          # compartments are the brain regions where the synapses are, when the harvested
                                    # region table (:func:`region_table`) covers these neurons; else the groups


@dataclass(frozen=True)
class ReceptorFact:
    """Receptors a cell type is shown to use by direct evidence in that cell type. Used for types that no
    adult single-cell cluster resolves, and only for the modulators whose receptors it names. Where the
    effect of a modulator on the type was measured but its receptor was not identified, ``effects`` gives the
    measured sign instead, e.g. ``(("octopamine", 1.0),)``."""
    spec: str
    receptors: tuple[str, ...]      # gene symbols from :data:`RECEPTORS`
    label: str
    why: str = ""
    effects: tuple[tuple[str, float], ...] = ()     # (modulator, sign of its effect on the gain), receptor unknown

    def what(self) -> str:
        """Short text for the page: the receptors, or the measured effect."""
        return ", ".join([*self.receptors, *(f"{m} {'raises' if v > 0 else 'lowers'} its gain" for m, v in self.effects)])


@dataclass(frozen=True)
class CellParam:
    """Per-type overrides: the hook for measured cell-type parameters."""
    spec: str
    theta_mv: float | None = None      # spike threshold above rest (default 7 mV for everyone)
    graded: bool | None = None         # force the graded flag on or off
    label: str = ""
    why: str = ""


MODULATORS: tuple[Modulator, ...] = (
    Modulator("dopamine", "dopamine", ("ple", "DAT"),
              "Dop1R1, Dop1R2, Dop2R, DopEcR", tau_ms=500.0, gain=0.3,
              why="Dopamine receptors in Drosophila are all GPCRs; dopamine transients in mushroom-body compartments "
                  "last about a second (Cohn, Morantte & Ruta 2015) and set the gain of the compartment's output "
                  "as well as gating plasticity (Hige et al. 2015). The fast synapses of the 399 dopaminergic "
                  "neurons are removed; their plasticity role is unchanged."),
    Modulator("octopamine", "octopamine", ("Tdc2", "Tbh"),
              "Oamb, Octα2R, Octβ1R, Octβ2R, Octβ3R", tau_ms=1000.0, gain=0.5,
              why="Octopamine released during flight raises the gain of the lobula-plate tangential cells' motion "
                  "responses by about half (Suver, Mamiya & Dickinson 2012) and retunes T4/T5 (Arenz et al. 2017); "
                  "all its receptors are GPCRs. 165 neurons."),
    Modulator("serotonin", "serotonin", ("Trh", "SerT"),
              "5-HT1A, 5-HT1B, 5-HT2A, 5-HT2B, 5-HT7", tau_ms=2000.0, gain=0.3,
              why="Serotonin from the CSD neuron enhances antennal-lobe projection-neuron responses over seconds "
                  "(Dacks, Green, Root, Nighorn & Wang 2009); Drosophila has no ionotropic serotonin receptor. "
                  "Its sign differs between targets in the fly (5-HT1 receptors are inhibitory, 5-HT2/7 excitatory): "
                  "without receptor expression per cell type one net gain stands in. 415 neurons."),
)

RECEPTORS: tuple[Receptor, ...] = (
    Receptor("Dop1R1", "FBgn0011582", "dopamine", "Gs", +1, "cAMP up (Sugamori et al. 1995; VFB: GO:0001588)"),
    Receptor("Dop1R2", "FBgn0266137", "dopamine", "Gq", +1, "calcium up (Han, Millar & Davis 1996; Himmelreich et al. 2017)"),
    Receptor("Dop2R", "FBgn0053517", "dopamine", "Gi", -1, "cAMP down (Hearn et al. 2002; VFB: GO:0001591)"),
    Receptor("DopEcR", "FBgn0035538", "dopamine", "Gs", +1, "cAMP up, also binds ecdysone (Srivastava et al. 2005; VFB: GO:0001588)"),
    Receptor("Oamb", "FBgn0024944", "octopamine", "Gq", +1, "calcium up (Han, Millar, Grotewiel & Davis 1998)"),
    Receptor("Octα2R", "FBgn0038653", "octopamine", "Gi", -1, "cAMP down (Qi et al. 2017)"),
    Receptor("Octβ1R", "FBgn0038980", "octopamine", "Gs", +1, "cAMP up (Maqueira, Chatwin & Evans 2005)"),
    Receptor("Octβ2R", "FBgn0038063", "octopamine", "Gs", +1, "cAMP up (Maqueira, Chatwin & Evans 2005)"),
    Receptor("Octβ3R", "FBgn0250910", "octopamine", "Gs", +1, "cAMP up (Maqueira, Chatwin & Evans 2005)"),
    Receptor("5-HT1A", "FBgn0004168", "serotonin", "Gi", -1, "cAMP down (Saudou et al. 1992)"),
    Receptor("5-HT1B", "FBgn0263116", "serotonin", "Gi", -1, "cAMP down (Saudou et al. 1992)"),
    Receptor("5-HT2A", "FBgn0087012", "serotonin", "Gq", +1, "calcium up (Colas et al. 1995)"),
    Receptor("5-HT2B", "FBgn0261929", "serotonin", "Gq", +1, "calcium up (Blenau et al. 2017)"),
    Receptor("5-HT7", "FBgn0004573", "serotonin", "Gs", +1, "cAMP up (Witz, Amlaiky & Hen 1990)"),
)

GRADED: tuple[Graded, ...] = (
    Graded("prefix:R1-R6,prefix:R7,prefix:R8", "photoreceptors R1-R8",
           "graded, histaminergic (Hardie & Raghu 2001)"),
    Graded("L1,L2,L3,L4,L5", "lamina monopolar cells L1-L5",
           "graded, no spikes (Laughlin & Hardie 1978; Zheng et al. 2006)"),
    Graded("Mi1,Mi4,Mi9,Tm1,Tm2,Tm3,Tm4,Tm9", "medulla inputs to T4/T5",
           "graded voltage and calcium responses, no spikes (Behnia et al. 2014; Yang et al. 2016; Arenz et al. 2017)"),
    Graded("prefix:T4,prefix:T5", "the motion detectors T4 and T5",
           "graded, non-spiking (Gruntman, Romani & Reiser 2018)"),
    Graded("HSE,HSN,HSS,HST,prefix:VS,CT1", "lobula-plate tangential cells HS/VS and CT1",
           "graded with small spikelets (Haag & Borst 1996; Schnell et al. 2010; Meier & Borst 2019)"),
)

LOCAL: tuple[Local, ...] = (
    Local("APL", ("prefix:KCg", "prefix:KCab", "prefix:KCa'b'"), "APL, the mushroom body's GABAergic feedback neuron",
          "APL does not spike, and both its activity and its inhibition of Kenyon cells stay local, so it inhibits "
          "each mushroom-body compartment according to the Kenyon cells active there (Amin et al. 2020). Here APL "
          "keeps its spikes as the measure of its overall depolarisation, but its release onto a target follows the "
          "Kenyon cells active where APL's synapses onto that target are: by neuPrint's mushroom-body regions (calyx, "
          "pedunculus, each lobe) when the region table is installed, else by the target's Kenyon-cell lobe system "
          "(γ, α/β, α'/β'). Less where they are quieter than APL's average input, never more than the whole cell."),
)

RECEPTOR_FACTS: tuple[ReceptorFact, ...] = (
    ReceptorFact("APL", ("Dop2R",), "APL, the mushroom body's GABAergic feedback neuron",
                 "Dopamine neurons synapse onto APL and suppress it through the D2-like receptor Dop2R; knocking Dop2R "
                 "down in APL impairs aversive learning (Zhou et al. 2019). APL is one cell per side, so the "
                 "single-cell atlases have no cluster for it."),
    ReceptorFact("VS,regex:^VS[0-9]+$", (), "the VS cells, the lobula plate's vertical-motion neurons",
                 "Octopamine released during flight boosts the VS cells' response to visual motion; octopamine applied "
                 "to quiescent flies mimics the boost, and octopamine neurons are necessary and sufficient for it (Suver, "
                 "Mamiya & Dickinson 2012). The receptor was not identified, and the adult atlases have no cluster for "
                 "these few large cells. This is the measured effect the octopamine tone's gain was set from; it names the "
                 "recorded VS cells only (not the VS-like or 'twin' types, nor the HS cells).",
                 effects=(("octopamine", 1.0),)),
)

CELL_PARAMS: tuple[CellParam, ...] = ()

GRADED_RATE_HZ = 300.0     # release quanta per second of a graded cell held at the spike threshold
BIG_THRESHOLD = 1e9        # graded cells never cross it


# ------------------------------------------------------------------ the compiled parts of one connectome
REGION_FILE = PROJECT_DIR / "data" / "mb_roi_connectivity.json.gz"
_REGIONS: dict = {"table": None, "loaded": False}
REGION_LABELS = {"CA": "calyx", "PED": "pedunculus", "gL": "γ lobe", "aL": "α lobe", "a'L": "α′ lobe",
                 "bL": "β lobe", "b'L": "β′ lobe"}
GROUP_LABELS = {"prefix:KCg": "γ lobe", "prefix:KCab": "α/β lobes", "prefix:KCa'b'": "α′/β′ lobes"}


def region_table() -> dict | None:
    """Where APL's and DPM's synapses are, region by region (``data/mb_roi_connectivity.json.gz``, harvested
    from neuPrint by ``tools/harvest_neuprint_rois.py``), or None when the file is not installed."""
    if not _REGIONS["loaded"]:
        _REGIONS["loaded"] = True
        try:
            with gzip.open(REGION_FILE, "rt", encoding="utf-8") as f:
                _REGIONS["table"] = json.load(f)
        except (OSError, ValueError):
            _REGIONS["table"] = None
    return _REGIONS["table"]


def use_region_table(table: dict | None):
    """Install a region table (tests use a hand-made one; None = no table)."""
    _REGIONS.update(table=table, loaded=True, injected=True)


def injected_region_table() -> dict | None:
    """``{"table": ...}`` when :func:`use_region_table` installed one (None included), else None (the file)."""
    return {"table": _REGIONS["table"]} if _REGIONS.get("injected") else None


OUTSIDE = "outside the mushroom body"


def region_label(roi: str) -> str:
    """'gL(R)' -> 'γ lobe'; every region outside the mushroom body's named compartments -> OUTSIDE."""
    return REGION_LABELS.get(roi.split("(")[0], OUTSIDE)


@dataclass
class CompiledLocal:
    """One :class:`Local` entry on one connectome: its compartments and how each output synapse weighs them.

    A compartment is a brain region where the neuron has synapses (from the harvested region table) or, without
    the table, the territory of one input group (a Kenyon-cell lobe system)."""
    local: Local
    neurons: np.ndarray             # (A,) the local neurons
    member_pos: np.ndarray          # (n,) row of each group member in ``syn_to``, -1 for everyone else
    syn_to: np.ndarray              # (M, A, C) synapses from each group member onto each local neuron, per compartment
    sites: np.ndarray               # (A, C) input synapses of each local neuron in each compartment
    edges: np.ndarray               # (E,) the local neurons' output connections
    owner: np.ndarray               # (E,) which local neuron (0..A-1) each connection leaves from
    mix: np.ndarray                 # (E, C) the share of each output connection's synapses in each compartment
    placed: np.ndarray              # (E,) False: nothing places this connection, so release there stays global
    labels: list = field(default_factory=list)       # (C,) a name per compartment ('calyx', 'γ lobe', ...)
    mode: str = "groups"            # "regions" (from the table) or "groups"
    out_weight: np.ndarray = None   # (A, C) output synapses in each compartment
    group_sizes: list = field(default_factory=list)

    def summary(self) -> dict:
        per: dict[str, list[float]] = {}
        for c, lab in enumerate(self.labels):
            per.setdefault(lab, [0.0, 0.0])
            per[lab][0] += float(self.sites[:, c].sum())
            per[lab][1] += float(self.out_weight[:, c].sum())
        return {"spec": self.local.spec, "label": self.local.label, "neurons": int(self.neurons.size),
                "groups": [{"spec": s, "neurons": k} for s, k in zip(self.local.groups, self.group_sizes)],
                "mode": self.mode, "compartments": len(self.labels),
                "by_label": {lab: {"input_synapses": int(v[0]), "output_synapses": int(v[1])} for lab, v in per.items()},
                "outputs": int(self.edges.size), "placed": int(self.placed.sum()), "tau_ms": self.local.tau_ms,
                "why": self.local.why}


def compile_local(conn: Connectome, loc: Local, table: dict | None = None) -> CompiledLocal:
    n, neurons = conn.n, np.unique(conn.select(loc.spec)).astype(np.int64)
    group = np.full(n, -1, dtype=np.int64)
    sizes = []
    for c, spec in enumerate(loc.groups):
        idx = conn.select(spec)
        group[idx] = c
        sizes.append(int(idx.size))
    members = np.flatnonzero(group >= 0)
    member_pos = np.full(n, -1, dtype=np.int64)
    member_pos[members] = np.arange(members.size)
    A = int(neurons.size)
    edges = conn.out_edges(neurons) if A else np.zeros(0, dtype=np.int64)
    owner = np.searchsorted(neurons, conn.pre_idx[edges])
    post = conn.post_idx[edges]
    layout = _region_layout(conn, neurons, members, member_pos, edges, owner, table) if loc.by_region and table else None
    if layout is not None:
        syn_to, mix, placed, labels = layout
        mode = "regions"
    else:
        C = len(loc.groups)
        syn_to = np.zeros((members.size, A, C), dtype=np.float32)
        if A and members.size:
            into = conn.edges_between(members, neurons)
            m = member_pos[conn.pre_idx[into]]
            np.add.at(syn_to, (m, np.searchsorted(neurons, conn.post_idx[into]), group[members][m]),
                      conn.n_syn[into].astype(np.float32))
        # each target's compartment mix: a group member sits in its own; anyone else in proportion to its group input
        per_group = np.zeros((n, C), dtype=np.float64)
        from_members = conn.out_edges(members) if members.size else np.zeros(0, dtype=np.int64)
        for c in range(C):
            e = from_members[group[conn.pre_idx[from_members]] == c]
            per_group[:, c] = np.bincount(conn.post_idx[e], weights=conn.n_syn[e], minlength=n)
        per_group[members] = 0.0
        per_group[members, group[members]] = 1.0
        total = per_group[post].sum(axis=1)
        placed = total > 0
        mix = np.zeros((edges.size, C), dtype=np.float32)
        mix[placed] = per_group[post[placed]] / total[placed, None]
        labels = [GROUP_LABELS.get(g, g) for g in loc.groups]
        mode = "groups"
    sites = syn_to.sum(axis=0, dtype=np.float64)
    out_weight = np.zeros((A, len(labels)), dtype=np.float64)
    if edges.size:
        np.add.at(out_weight, owner, mix * conn.n_syn[edges][:, None])
    return CompiledLocal(loc, neurons, member_pos, syn_to, sites, edges, owner, mix, placed, labels, mode,
                         out_weight, sizes)


def _region_layout(conn, neurons, members, member_pos, edges, owner, table):
    """Compartments from the harvested region table: the mushroom body's named regions on each side (calyx,
    pedunculus, each lobe) where these neurons have synapses, plus one OUTSIDE compartment for everything else,
    which takes no group input and so keeps the whole cell's release (a few hundred synapses per region are too
    few to carry a local signal). None when the table does not cover these neurons (another connectome, a grown
    fly, or neurons it was not harvested for)."""
    if (getattr(conn, "meta", None) or {}).get("rewired"):
        return None                                    # a grown or rewired fly: the table lists the real fly's pairs
    bid = conn.body_id
    local_b = {int(b): a for a, b in enumerate(bid[neurons].tolist())}
    known = table.get("neurons", {})
    if not local_b or any(str(b) not in known for b in local_b):
        return None
    member_b = {int(b): int(member_pos[i]) for i, b in zip(members.tolist(), bid[members].tolist())}
    names = [region_label(r) for r in table["rois"]]
    comp: dict[str, int] = {}                          # one compartment per mushroom-body region, one OUTSIDE
    key = [n if n == OUTSIDE else r for r, n in zip(table["rois"], names)]
    ins, outs = [], {}
    for pre, post, r, w in table["edges"]:
        if post in local_b and pre in member_b and names[r] != OUTSIDE:
            ins.append((member_b[pre], local_b[post], comp.setdefault(key[r], len(comp)), w))
        if pre in local_b:
            outs.setdefault((pre, post), []).append((comp.setdefault(key[r], len(comp)), w))
    C = len(comp)
    syn_to = np.zeros((members.size, len(local_b), C), dtype=np.float32)
    for m, a, c, w in ins:
        syn_to[m, a, c] += w
    mix = np.zeros((edges.size, C), dtype=np.float32)
    placed = np.zeros(edges.size, dtype=bool)
    pre_b = bid[neurons][owner].tolist()
    post_b = bid[conn.post_idx[edges]].tolist()
    for e, key in enumerate(zip(pre_b, post_b)):
        rows = outs.get((int(key[0]), int(key[1])))
        if rows:
            tot = float(sum(w for _, w in rows))
            for c, w in rows:
                mix[e, c] += w / tot
            placed[e] = True
    labels = [None] * C
    for k, c in comp.items():
        labels[c] = k if k == OUTSIDE else region_label(k)
    return syn_to, mix, placed, labels


@dataclass
class CompiledParts:
    """Per-neuron arrays the brain needs, built from a :class:`PartsList` for one connectome."""
    parts: "PartsList"
    mod_kind: np.ndarray            # per neuron: index into parts.modulators, or -1
    mod_neurons: np.ndarray         # indices of the modulatory neurons
    mod_targets: np.ndarray         # sorted indices of the neurons that receive modulatory synapses
    target_pos: np.ndarray          # per neuron: position in mod_targets, or -1
    graded_idx: np.ndarray          # sorted indices of the graded neurons
    graded_mask: np.ndarray         # bool per neuron
    theta: np.ndarray               # per-neuron spike threshold (mV above rest; BIG for graded)
    kind_targets: list = field(default_factory=list)   # per modulator: positions (in mod_targets) of its own targets
    counts: dict = field(default_factory=dict)
    sign: np.ndarray = None         # per neuron: the sign of its fast synapses after the curated overrides
    keep_fast: np.ndarray = None    # per neuron: a modulator that also keeps its fast synapses (co-release)
    mod_sign: np.ndarray = None     # per modulator, per target: the receptor sign-weight of the tone (+1 = unknown)
    roles: dict = field(default_factory=dict)          # per changed neuron: what the curated data did to it
    local: list = field(default_factory=list)          # CompiledLocal per Local entry
    receptor_facts: dict = field(default_factory=dict)  # per neuron whose tone signs come from a ReceptorFact: that fact

    @property
    def n_kinds(self) -> int:
        return len(self.parts.modulators)

    def summary(self) -> dict:
        return self.counts

    def role(self, i: int) -> dict:
        """What this neuron is in the running model: its fast sign, its tone (if any) and why."""
        i = int(i)
        k = int(self.mod_kind[i])
        out = {"sign": int(self.sign[i]), "modulator": self.parts.modulators[k].nt if k >= 0 else None,
               "keep_fast": bool(self.keep_fast[i]), "graded": bool(self.graded_mask[i]),
               "theta_mv": None if self.graded_mask[i] else float(self.theta[i])}
        if i in self.roles:
            out["curated"] = self.roles[i]
        for loc in self.local:
            if i in loc.neurons:
                out["local"] = {"label": loc.local.label, "groups": list(loc.local.groups), "mode": loc.mode,
                                "compartments": sorted(set(loc.labels))}
        if i in self.receptor_facts:
            f = self.receptor_facts[i]
            out["receptor_fact"] = {"receptors": list(f.receptors), "effects": dict(f.effects), "what": f.what(),
                                    "why": f.why}
        return out


@dataclass
class PartsList:
    modulators: tuple[Modulator, ...] = MODULATORS
    graded: tuple[Graded, ...] = GRADED
    params: tuple[CellParam, ...] = CELL_PARAMS
    graded_rate_hz: float = GRADED_RATE_HZ
    theta_mv: float = THETA
    curated: str = "modulators"     # vfb.transmitter_overrides policy: off / modulators / all
    receptor_signs: bool = True     # per-target tone signs from the receptors the target type expresses
    unknown_sign: float = 0.0       # tone sign-weight on a target whose receptors are unknown (v2.7: 1.0, the one-sign rule)
    receptors: tuple[Receptor, ...] = RECEPTORS
    receptor_facts: tuple[ReceptorFact, ...] = RECEPTOR_FACTS
    local: tuple[Local, ...] = LOCAL

    def compile(self, conn: Connectome) -> CompiledParts:
        from . import vfb
        n = conn.n
        ov = vfb.transmitter_overrides(conn, self.curated, modulators=tuple(m.nt for m in self.modulators))
        kind_of = {m.nt: k for k, m in enumerate(self.modulators)}
        mod_kind = np.full(n, -1, dtype=np.int8)
        for k, m in enumerate(self.modulators):
            mod_kind[conn.nt == m.nt] = k
        changed = np.flatnonzero(ov.mod_nt != None)                       # noqa: E711  (object array)
        for i in changed.tolist():
            mod_kind[i] = kind_of.get(ov.mod_nt[i], -1)
        keep_fast = ov.keep_fast & (mod_kind >= 0)
        mods, own_targets = [], []
        for k, m in enumerate(self.modulators):
            idx = np.flatnonzero(mod_kind == k).astype(np.int64)
            edges = conn.out_edges(idx) if idx.size else np.zeros(0, dtype=np.int64)
            own_targets.append(np.unique(conn.post_idx[edges]).astype(np.int64))
            mods.append({"nt": m.nt, "label": m.label, "genes": list(m.genes), "receptors": m.receptors,
                         "tau_ms": m.tau_ms, "gain": m.gain, "neurons": int(idx.size), "synapses": int(conn.n_syn[edges].sum()),
                         "targets": int(own_targets[-1].size), "co_release": int(keep_fast[idx].sum()), "why": m.why})
        mod_neurons = np.flatnonzero(mod_kind >= 0).astype(np.int64)
        edges = conn.out_edges(mod_neurons) if mod_neurons.size else np.zeros(0, dtype=np.int64)
        mod_targets = np.unique(conn.post_idx[edges]).astype(np.int64)
        target_pos = np.full(n, -1, dtype=np.int64)
        target_pos[mod_targets] = np.arange(mod_targets.size)
        kind_targets = [target_pos[t] for t in own_targets]
        if self.receptor_signs:
            mod_sign, coverage = vfb.receptor_signs(conn, mod_targets, self.modulators, unknown=self.unknown_sign)
            fact_of, fact_rows = self._apply_receptor_facts(conn, mod_sign, target_pos)
        else:
            mod_sign, coverage = np.ones((len(self.modulators), mod_targets.size), dtype=np.float32), []
            fact_of, fact_rows = {}, []
        roles = {int(i): {"action": ov.action[i], "curated": list(ov.curated[i]), "predicted": conn.nt[i]}
                 for i in np.flatnonzero(ov.action != None).tolist()}            # noqa: E711
        graded_mask = np.zeros(n, dtype=bool)
        graded_rows = []
        for g in self.graded:
            idx = conn.select(g.spec)
            graded_mask[idx] = True
            graded_rows.append({"spec": g.spec, "label": g.label, "neurons": int(idx.size), "why": g.why})
        theta = np.full(n, self.theta_mv, dtype=np.float32)
        param_rows = []
        for p in self.params:
            idx = conn.select(p.spec)
            if p.theta_mv is not None:
                theta[idx] = np.float32(p.theta_mv)
            if p.graded is not None:
                graded_mask[idx] = p.graded
            param_rows.append({"spec": p.spec, "neurons": int(idx.size), "theta_mv": p.theta_mv, "graded": p.graded,
                               "label": p.label, "why": p.why})
        graded_mask[mod_neurons] = False                      # a modulatory neuron keeps its spikes (they are its events)
        graded_idx = np.flatnonzero(graded_mask).astype(np.int64)
        theta[graded_idx] = np.float32(BIG_THRESHOLD)
        local = [compile_local(conn, loc, region_table()) for loc in self.local]
        counts = {"modulators": mods, "modulatory_neurons": int(mod_neurons.size), "modulated_targets": int(mod_targets.size),
                  "co_release_neurons": int(keep_fast.sum()),
                  "graded": graded_rows, "graded_neurons": int(graded_idx.size), "graded_rate_hz": self.graded_rate_hz,
                  "params": param_rows, "local": [c.summary() for c in local],
                  "curated": {**ov.counts, "rows": ov.rows[:40],
                              "signs_changed": int((ov.sign != conn.sign).sum()),          # incl. "unclear" (counted +) filled with an inhibitory one
                              "confident_signs_flipped": int(((ov.sign != conn.sign) & ~np.isin(conn.nt, ["", "unclear"])).sum())},
                  "receptor_signs": {"on": self.receptor_signs, "coverage": coverage, "facts": fact_rows,
                                     "receptors": [{"gene": r.gene, "modulator": r.modulator, "coupling": r.coupling, "sign": r.sign}
                                                   for r in self.receptors]}}
        return CompiledParts(self, mod_kind, mod_neurons, mod_targets, target_pos, graded_idx, graded_mask, theta,
                             kind_targets, counts, sign=ov.sign, keep_fast=keep_fast, mod_sign=mod_sign, roles=roles,
                             local=local, receptor_facts=fact_of)

    def _apply_receptor_facts(self, conn: Connectome, mod_sign: np.ndarray, target_pos: np.ndarray):
        """Set the tone signs a :class:`ReceptorFact` names, on its types that no atlas cluster covers."""
        from . import vfb
        by_gene = {r.gene: r for r in self.receptors}
        fact_of, rows = {}, []
        for f in self.receptor_facts:
            unknown = [g for g in f.receptors if g not in by_gene]
            if unknown:
                raise ValueError(f"receptor fact for {f.spec}: unknown receptor {', '.join(unknown)} "
                                 f"(known: {', '.join(by_gene)})")
            bad = [m for m, _ in f.effects if m not in {x.nt for x in self.modulators}]
            if bad:
                raise ValueError(f"receptor fact for {f.spec}: unknown modulator {', '.join(bad)}")
            effects = dict(f.effects)
            idx = conn.select(f.spec)
            atlas = sorted({t for t in conn.types[idx].tolist() if vfb.receptors_of_type(t, conn) is not None})
            idx = idx[~np.isin(conn.types[idx], atlas)]
            signs = {}
            for k, m in enumerate(self.modulators):
                named = [by_gene[g].sign for g in f.receptors if by_gene[g].modulator == m.nt]
                if named or m.nt in effects:
                    signs[m.nt] = float(max(-1, min(1, sum(named)))) if named else float(effects[m.nt])
                    pos = target_pos[idx]
                    mod_sign[k, pos[pos >= 0]] = np.float32(signs[m.nt])
            for i in idx.tolist():
                fact_of[int(i)] = f
            rows.append({"spec": f.spec, "label": f.label, "receptors": list(f.receptors), "effects": effects,
                         "what": f.what(), "neurons": int(idx.size),
                         "targets": int((target_pos[idx] >= 0).sum()), "signs": signs,
                         "left_to_the_atlas": atlas, "why": f.why})
        return fact_of, rows

    def describe(self) -> dict:
        """The tables, for the docs and the game (no connectome needed)."""
        return {"modulators": [{"nt": m.nt, "label": m.label, "genes": list(m.genes), "receptors": m.receptors,
                                "tau_ms": m.tau_ms, "gain": m.gain, "why": m.why} for m in self.modulators],
                "graded": [{"spec": g.spec, "label": g.label, "why": g.why} for g in self.graded],
                "params": [{"spec": p.spec, "theta_mv": p.theta_mv, "graded": p.graded, "label": p.label, "why": p.why}
                           for p in self.params],
                "graded_rate_hz": self.graded_rate_hz, "curated": self.curated, "receptor_signs": self.receptor_signs,
                "unknown_sign": self.unknown_sign,
                "receptors": [{"gene": r.gene, "fbgn": r.fbgn, "modulator": r.modulator, "coupling": r.coupling,
                               "sign": r.sign, "why": r.why} for r in self.receptors],
                "receptor_facts": [{"spec": f.spec, "receptors": list(f.receptors), "effects": dict(f.effects),
                                    "what": f.what(), "label": f.label, "why": f.why}
                                   for f in self.receptor_facts],
                "local": [{"spec": x.spec, "groups": list(x.groups), "label": x.label, "tau_ms": x.tau_ms,
                           "by_region": x.by_region, "why": x.why} for x in self.local]}

    def with_params(self, specs: list[str] | tuple[str, ...]) -> "PartsList":
        """A copy with extra per-type overrides from strings such as ``"class:Kenyon_Cell:theta=10"`` or
        ``"MN9:graded=1"`` (keys ``theta`` in mV and ``graded`` 0/1, comma-separated)."""
        extra = tuple(parse_param(s) for s in specs)
        return replace(self, params=self.params + extra)


def parse_param(text: str) -> CellParam:
    spec, sep, settings = text.rpartition(":")
    if not sep or "=" not in settings:
        raise ValueError(f"a part override looks like SPEC:theta=5 or SPEC:graded=1, not '{text}'")
    theta = graded = None
    for item in settings.split(","):
        key, _, val = item.partition("=")
        key, val = key.strip().lower(), val.strip()
        if key in ("theta", "theta_mv"):
            theta = float(val)
            if not 0.5 <= theta <= 100:
                raise ValueError("theta must be between 0.5 and 100 mV")
        elif key == "graded":
            graded = val.lower() in ("1", "true", "yes", "on")
        else:
            raise ValueError(f"unknown part setting '{key}' (theta or graded)")
    return CellParam(spec.strip(), theta_mv=theta, graded=graded, label=spec.strip(), why="set by hand")


def as_parts(parts) -> PartsList | None:
    """``None``/``False`` = no parts list, ``True`` = the default, or a :class:`PartsList`."""
    if parts is None or parts is False:
        return None
    if parts is True:
        return PartsList()
    if isinstance(parts, PartsList):
        return parts
    raise TypeError("parts must be True, False or a PartsList")
