"""
What Virtual Fly Brain and the fly anatomy ontology add (v2.5).

The connectome names every neuron with a MaleCNS cell-type string such as ``"LC4"`` or ``"KCg-m"``.
`Virtual Fly Brain <https://virtualflybrain.org>`_ (VFB) keys the whole of fly neuroscience by a
different vocabulary: the classes of the FlyBase anatomy ontology (``FBbt:00003874`` = *lobula columnar
neuron LC4*), each with a curated definition, a place in an ``is_a`` tree, the transmitter the
literature has established, and pointers to every other data set that saw the same cell type. This
module joins the two vocabularies, offline, from three files in ``data/``: the first two built by
``tools/build_vfb_data.py`` from the ontology's own OBO release plus a one-time harvest of VFB's
``name_in_male-cns`` synonyms (``tools/vfb_overlay.json``), the third by ``tools/merge_vfb_harvest.py``
from a harvest of VFB's scRNA-seq expression tables:

* ``fbbt_map.json.gz``: kit type -> FBbt class(es), how the match was made, and the transmitters the
  ontology asserts for that class;
* ``fbbt_tree.json.gz``: the classes above them (labels, parents, symbols, short definitions);
* ``vfb_receptors.json.gz``: for the classes that have a single-cell RNA-seq cluster on VFB, the
  fraction of cells expressing each aminergic receptor gene.

What the kit does with them:

* a population selector ``fbbt:<class>`` (an id or a label) that takes the ontology's ``is_a``
  closure: ``fbbt:lobula columnar neuron`` is every LC type the kit has, ``fbbt:adult descending
  neuron`` every DN, ``fbbt:dopaminergic neuron`` every cell the ontology calls dopaminergic;
* a selector ``rx:<receptor>`` (``rx:Dop2R``, ``rx:5-HT1A>0.5``) for the cell types whose adult
  scRNA-seq cluster expresses a receptor;
* per-neuron facts for the page's popover (the class, its definition, the lineage, the peptides, the
  curated transmitter, the receptors, a link to VFB);
* two inputs to the parts list (:mod:`virtual_fly.parts`): *curated transmitters* (where the
  literature says a neuron's transmitter differs from the connectome's prediction, or the prediction
  is "unclear") and *receptor signs* (a modulator's effect on a target follows the receptors that
  target's cell type expresses: Gs/Gq-coupled receptors raise the gain, Gi-coupled ones lower it).

Nothing here needs the network; the files are optional and every function degrades to "nothing
known" without them. Licences: FBbt is CC-BY 4.0 (FlyBase); the scRNA-seq clusters on VFB are
CC-BY 4.0 (Fly Cell Atlas and the other data sets named in the file).
"""

from __future__ import annotations

import collections
import gzip
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .connectome import PROJECT_DIR

DATA_DIR = PROJECT_DIR / "data"
MAP_FILE = DATA_DIR / "fbbt_map.json.gz"
TREE_FILE = DATA_DIR / "fbbt_tree.json.gz"
RX_FILE = DATA_DIR / "vfb_receptors.json.gz"
VFB_REPORT = "https://virtualflybrain.org/reports/{}"        # FBbt_00003874 (underscore form)
FLYBASE = "https://flybase.org/reports/{}"

FAST_SIGN = {"acetylcholine": 1.0, "gaba": -1.0, "glutamate": -1.0, "histamine": -1.0}
MODULATOR_NTS = ("dopamine", "octopamine", "serotonin")
NOT_ANATOMY = re.compile(r"(ergic neuron|peptidergic|lineage|hemilineage|primary neuron|secondary neuron|"
                         r"-specific neuron|expressing neuron|system neuron$|^neuron$|^adult neuron$|^larval neuron$)", re.I)
LINEAGE = re.compile(r"(lineage neuron$|hemilineage)", re.I)


def _read(path: Path) -> dict | None:
    path = Path(path)
    if not path.exists():
        return None
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return json.load(f)
    return json.loads(path.read_text(encoding="utf-8"))


def norm_id(value: str) -> str:
    """``FBbt_00003874`` / ``fbbt:00003874`` -> ``FBbt:00003874``."""
    v = value.strip().replace("_", ":")
    if v.lower().startswith("fbbt:"):
        return "FBbt:" + v[5:]
    return v


def underscore(cid: str) -> str:
    return cid.replace(":", "_")


# ---------------------------------------------------------------------------------------------
# 1. The ontology slice
# ---------------------------------------------------------------------------------------------

class Ontology:
    """The kit's slice of FBbt: the mapped types, the classes above them, and the closures over them."""

    def __init__(self, map_data: dict | None = None, tree_data: dict | None = None):
        map_data = map_data or {}
        tree_data = tree_data or {}
        self.source: dict = tree_data.get("source", {})
        self.overlay_source: str | None = map_data.get("overlay_source")
        self.types: dict[str, dict] = map_data.get("types", {})
        self.classes: dict[str, dict] = tree_data.get("classes", {})
        self.roots: dict = tree_data.get("roots", {})
        self.children: dict[str, list[str]] = collections.defaultdict(list)
        for cid, c in self.classes.items():
            for p in c.get("parents", []):
                self.children[p].append(cid)
        self.types_of_class: dict[str, list[str]] = collections.defaultdict(list)
        for name, e in self.types.items():
            for cid in e.get("fbbt", []):
                self.types_of_class[cid].append(name)
        self.by_name: dict[str, list[str]] = collections.defaultdict(list)
        for cid, c in self.classes.items():
            self.by_name[c["label"].lower()].append(cid)
            if c.get("symbol"):
                self.by_name[c["symbol"].lower()].append(cid)
        self._desc: dict[str, frozenset] = {}
        self._anc: dict[str, list[str]] = {}
        self._tags: dict[str, dict] = {}
        self._counts: dict = {}

    @property
    def empty(self) -> bool:
        return not self.types

    # --- lookups
    def resolve(self, value: str) -> list[str]:
        """Class ids for an id (either form) or a label / symbol (case-insensitive, exact)."""
        v = value.strip()
        if not v:
            return []
        cid = norm_id(v)
        if cid in self.classes:
            return [cid]
        return list(self.by_name.get(v.lower(), []))

    def label(self, cid: str) -> str:
        return self.classes.get(cid, {}).get("label", cid)

    def symbol(self, cid: str) -> str | None:
        return self.classes.get(cid, {}).get("symbol")

    def definition(self, cid: str) -> str:
        return self.classes.get(cid, {}).get("def", "")

    def parents(self, cid: str) -> list[str]:
        return list(self.classes.get(cid, {}).get("parents", []))

    def url(self, cid: str) -> str:
        return VFB_REPORT.format(underscore(cid))

    def classes_of(self, type_name: str) -> list[str]:
        return list(self.types.get(type_name, {}).get("fbbt", []))

    def descendants(self, cid: str) -> frozenset:
        """``is_a`` closure below a class, the class included."""
        if cid in self._desc:
            return self._desc[cid]
        seen = {cid}
        stack = [cid]
        while stack:
            c = stack.pop()
            for ch in self.children.get(c, []):
                if ch not in seen:
                    seen.add(ch)
                    stack.append(ch)
        self._desc[cid] = frozenset(seen)
        return self._desc[cid]

    def ancestors(self, cid: str) -> list[str]:
        """Ancestors nearest first (breadth-first over ``is_a``), the class excluded."""
        if cid in self._anc:
            return self._anc[cid]
        out, seen, frontier = [], {cid}, [cid]
        while frontier:
            nxt = []
            for c in frontier:
                for p in self.parents(c):
                    if p not in seen:
                        seen.add(p)
                        out.append(p)
                        nxt.append(p)
            frontier = nxt
        self._anc[cid] = out
        return out

    def types_under(self, cid: str) -> list[str]:
        names = []
        for c in self.descendants(cid):
            names.extend(self.types_of_class.get(c, []))
        return sorted(set(names))

    # --- interpretation
    def tags(self, cid: str) -> dict:
        """What the class's ancestry says: transmitters, peptides, lineage, birth order."""
        if cid in self._tags:
            return self._tags[cid]
        chain = set(self.ancestors(cid)) | {cid}
        roots = self.roots
        nts = [nt for nt, r in roots.get("transmitters", {}).items() if r in chain]
        peptides = []
        pep_root = roots.get("peptidergic")
        if pep_root and pep_root in chain:
            for a in chain:
                lab = self.label(a)
                if pep_root in self.parents(a) and re.search(r"^[\w\-]+ neuron$", lab) and not lab.startswith(("adult ", "larval ")):
                    peptides.append(lab[:-len(" neuron")])
        lineage = sorted({self.label(a) for a in chain if LINEAGE.search(self.label(a))})
        birth = "primary" if roots.get("primary") in chain else "secondary" if roots.get("secondary") in chain else None
        self._tags[cid] = {"transmitters": nts, "peptides": sorted(set(peptides)), "lineage": lineage, "birth": birth}
        return self._tags[cid]

    def breadcrumb(self, cid: str, depth: int = 4, conn=None) -> list[dict]:
        """The anatomical parent chain, nearest first: at each step the most specific parent (fewest kit
        types below it, when a connectome is given) that is not a transmitter, lineage, peptide or
        stage class."""
        counts = self.counts(conn) if conn is not None else None
        out, c = [], cid
        for _ in range(depth):
            ps = self.parents(c)
            if not ps:
                break
            anat = [p for p in ps if not NOT_ANATOMY.search(self.label(p))] or ps
            if counts is not None:
                anat = sorted(anat, key=lambda p: counts.get(p, (0, 0))[0])
            c = anat[0]
            out.append({"fbbt": c, "label": self.label(c)})
        return out

    # --- counts over a connectome
    def counts(self, conn) -> dict[str, tuple[int, int]]:
        """Per class: (kit types under it, neurons under it), computed once per connectome."""
        key = id(conn.tables)                 # rewired (grown) copies share the table: same counts
        hit = self._counts.get(key)
        if hit is not None:
            return hit[1]
        tc = conn.type_counts()
        direct = {cid: [(t, tc.get(t, 0)) for t in ts] for cid, ts in self.types_of_class.items()}
        out = {}
        for cid in self.classes:
            names = {}
            for c in self.descendants(cid):
                for t, n in direct.get(c, []):
                    names[t] = n
            out[cid] = (len(names), int(sum(names.values())))
        if len(self._counts) >= 4:
            self._counts.pop(next(iter(self._counts)), None)
        self._counts[key] = (conn.tables, out)          # the table is kept alive, so its id cannot be reused
        return out

    def search(self, text: str, conn, limit: int = 30) -> list[dict]:
        """Classes whose label or symbol contains the text, largest first (by neurons in the kit)."""
        q = text.strip().lower()
        if not q:
            return []
        counts = self.counts(conn)
        hits = []
        for cid, c in self.classes.items():
            lab = c["label"]
            sym = c.get("symbol") or ""
            if q in lab.lower() or q == sym.lower():
                n_types, n = counts.get(cid, (0, 0))
                if n_types:
                    hits.append({"fbbt": cid, "label": lab, "symbol": sym or None, "types": n_types, "neurons": n,
                                 "spec": f"fbbt:{cid}"})
        hits.sort(key=lambda h: (-h["neurons"], h["label"]))
        return hits[:limit]

    def class_info(self, cid: str, conn, top: int = 40) -> dict | None:
        cid = norm_id(cid)
        if cid not in self.classes:
            return None
        counts = self.counts(conn)
        tc = conn.type_counts()
        kids = [{"fbbt": k, "label": self.label(k), "types": counts[k][0], "neurons": counts[k][1]}
                for k in self.children.get(cid, []) if counts.get(k, (0, 0))[0]]
        kids.sort(key=lambda k: -k["neurons"])
        types = sorted(((t, tc.get(t, 0)) for t in self.types_under(cid)), key=lambda x: -x[1])
        n_types, n = counts.get(cid, (0, 0))
        return {"fbbt": cid, "label": self.label(cid), "symbol": self.symbol(cid), "definition": self.definition(cid),
                "url": self.url(cid), "parents": [{"fbbt": p, "label": self.label(p)} for p in self.parents(cid)],
                "children": kids, "types": [{"type": t, "n": k} for t, k in types[:top]], "n_types": n_types, "neurons": n,
                "spec": f"fbbt:{cid}", "tags": self.tags(cid)}

    def describe_type(self, type_name: str, conn=None) -> dict | None:
        """Everything the ontology says about one kit type, for the popover; None if unmapped."""
        e = self.types.get(type_name)
        if not e:
            return None
        ids = e["fbbt"]
        first = ids[0]
        tags = self.tags(first)
        siblings = self.types_of_class.get(first, [])
        out = {"fbbt": ids, "label": self.label(first), "symbol": self.symbol(first), "url": self.url(first),
               "route": e.get("route"), "coarse": bool(e.get("coarse")), "definition": self.definition(first),
               "breadcrumb": self.breadcrumb(first, conn=conn), "curated_nt": e.get("nt", []), "evidence": e.get("evidence"),
               "peptides": tags["peptides"], "lineage": tags["lineage"], "birth": tags["birth"],
               "shared_by": len(siblings) if len(siblings) > 1 else 0}
        if len(ids) > 1:
            out["labels"] = [self.label(i) for i in ids]
        return out

    def summary(self, conn) -> dict:
        """For the page: coverage and where the curated transmitters differ from the predictions."""
        if self.empty:
            return {"available": False}
        tc = conn.type_counts()
        mapped_n = sum(tc.get(t, 0) for t in self.types)
        typed_n = sum(tc.values())
        agree = differ = filled = 0
        rows = []
        nt_by_type = _majority_nt(conn)
        for t, e in self.types.items():
            S = e.get("nt")
            if not S:
                continue
            k = nt_by_type.get(t)
            if k in (None, "", "unclear"):
                filled += 1
                continue
            if k in S:
                agree += 1
            else:
                differ += 1
                rows.append({"type": t, "n": tc.get(t, 0), "predicted": k, "curated": S, "evidence": e.get("evidence"),
                             "fbbt": e["fbbt"][0], "label": self.label(e["fbbt"][0])})
        rows.sort(key=lambda r: -r["n"])
        return {"available": True, "source": self.source, "overlay_source": self.overlay_source,
                "types_mapped": len(self.types), "types_total": len(tc), "neurons_mapped": mapped_n, "neurons_typed": typed_n,
                "classes": len(self.classes), "curated": {"agree": agree, "differ": differ, "unclear_with_curated": filled,
                                                            "differ_rows": rows[:12]}}


def _majority_nt(conn) -> dict[str, str]:
    """The commonest predicted transmitter per type (types are nearly always homogeneous)."""
    cache = getattr(conn, "_majority_nt_cache", None)
    if cache is not None:
        return cache
    counts: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for t, nt in zip(conn.types.tolist(), conn.nt.tolist()):
        if t:
            counts[t][nt] += 1
    out = {t: c.most_common(1)[0][0] for t, c in counts.items()}
    conn._majority_nt_cache = out
    return out


# ---------------------------------------------------------------------------------------------
# 2. Receptor expression per class (scRNA-seq clusters on VFB)
# ---------------------------------------------------------------------------------------------

class Receptors:
    """``classes[FBbt][gene] = [[cluster id, extent, level], ...]`` plus the clusters' data set and stage.

    ``extent`` is the fraction of the cluster's cells expressing the gene; VFB's tables only carry
    values of 0.2 and above, so a gene missing from a cluster means "under 20 % of its cells", which
    the averages and the sign count as 0 (an underestimate, never an invention). ``genes`` lists only
    the genes that were harvested; a modulator none of whose receptors was harvested has no sign
    (``sign`` returns None and the old one-sign rule stands). Only adult clusters are used, in the
    order of ``families`` (the male Fly Cell Atlas first)."""

    def __init__(self, data: dict | None = None):
        data = data or {}
        self.source: dict = data.get("source", {})
        self.genes: dict[str, dict] = data.get("genes", {})
        self.clusters: dict[str, dict] = data.get("clusters", {})
        self.classes: dict[str, dict] = data.get("classes", {})
        self.families: list[str] = data.get("families", [])
        self._by_class: dict[str, dict | None] = {}

    @property
    def empty(self) -> bool:
        return not self.classes

    def for_class(self, cid: str) -> dict | None:
        """``{"extent": {gene: fraction}, "cluster": ..., "family": ..., "n_clusters": k}`` from the best adult
        family with data for this class, or None."""
        if cid in self._by_class:
            return self._by_class[cid]
        rows = self.classes.get(cid)
        best = None
        if rows:
            sums: dict[str, dict[str, float]] = collections.defaultdict(lambda: collections.defaultdict(float))
            members: dict[str, set] = collections.defaultdict(set)      # the family's clusters of this class
            for gene, entries in rows.items():
                for fblc, extent, *_ in entries:
                    cl = self.clusters.get(fblc, {})
                    if cl.get("stage") != "adult":
                        continue
                    fam = cl.get("family", "?")
                    sums[fam][gene] += float(extent)
                    members[fam].add(fblc)
            for fam in self.families + sorted(set(members) - set(self.families)):
                if fam in members:
                    k = len(members[fam])               # a gene a cluster does not list counts as 0 there
                    ext = {g: round(v / k, 4) for g, v in sums[fam].items()}
                    best = {"extent": ext, "cluster": sorted(members[fam])[0], "family": fam, "n_clusters": k}
                    break
        self._by_class[cid] = best
        return best

    def for_type(self, type_name: str, ont: Ontology, conn=None, max_depth: int = 2, max_types: int = 150) -> dict | None:
        """The class's own cluster, else the nearest ancestor's within ``max_depth`` levels, skipping classes
        so broad (more than ``max_types`` kit types) that inheriting from them would say nothing."""
        cids = ont.classes_of(type_name)
        if not cids or self.empty:
            return None
        counts = ont.counts(conn) if conn is not None else None
        seen = set()
        frontier = list(cids)
        for depth in range(max_depth + 1):
            for cid in frontier:
                if cid in seen:
                    continue
                seen.add(cid)
                if depth and counts is not None and counts.get(cid, (0, 0))[0] > max_types:
                    continue
                hit = self.for_class(cid)
                if hit:
                    return {**hit, "class": cid, "label": ont.label(cid), "depth": depth}
            frontier = [p for c in frontier for p in ont.parents(c) if p not in seen]
            if not frontier:
                break
        return None

    def sign(self, extent: dict[str, float], modulator: str) -> float | None:
        """``clip(sum(coupling sign x extent))`` over the modulator's harvested receptors (a receptor the
        cluster does not list counts 0); None if none of the modulator's receptors was harvested."""
        genes = [g for g, info in self.genes.items()
                 if info.get("modulator") == modulator and info.get("sign") and info.get("harvested", True)]
        if not genes:
            return None
        s = sum(info_sign * extent.get(g, 0.0) for g, info_sign in ((g, self.genes[g]["sign"]) for g in genes))
        return float(max(-1.0, min(1.0, s)))


# ---------------------------------------------------------------------------------------------
# 3. Module singletons (the files are read once; tests inject their own)
# ---------------------------------------------------------------------------------------------

_ONT: Ontology | None = None
_RX: Receptors | None = None
_GENERATION = 0


def generation() -> int:
    """Bumped by :func:`use`; :meth:`Connectome.select` keys its cache of ``fbbt:``/``rx:`` selections on it."""
    return _GENERATION


def ontology() -> Ontology:
    global _ONT
    if _ONT is None:
        _ONT = Ontology(_read(MAP_FILE), _read(TREE_FILE))
    return _ONT


def receptors() -> Receptors:
    global _RX
    if _RX is None:
        _RX = Receptors(_read(RX_FILE))
    return _RX


def use(ont: Ontology | None = None, rx: Receptors | None = None):
    """Install (or, with None, drop back to the files) the data the module answers from."""
    global _ONT, _RX, _GENERATION
    _ONT, _RX = ont, rx
    _GENERATION += 1


# ---------------------------------------------------------------------------------------------
# 4. Selectors
# ---------------------------------------------------------------------------------------------

def _type_mask(conn, names) -> np.ndarray:
    lookup = {t: k for k, t in enumerate(conn.tables["types"])}
    ks = np.array(sorted({lookup[n] for n in names if n in lookup}), dtype=np.int64)
    if ks.size == 0:
        return np.zeros(conn.n, dtype=bool)
    return np.isin(conn.type_idx, ks)


def fbbt_mask(conn, value: str) -> np.ndarray:
    """``fbbt:<id or label>``: every neuron whose type maps to the class or to anything below it."""
    ont = ontology()
    cids = ont.resolve(value)
    if not cids:
        raise ValueError(f"no FBbt class called '{value}' among the ones the kit's cell types map to")
    names = set()
    for cid in cids:
        names.update(ont.types_under(cid))
    return _type_mask(conn, names)


RX_RE = re.compile(r"^\s*([^<>=]+?)\s*(?:(>=|>)\s*(\d*\.?\d+))?\s*$")


def rx_mask(conn, value: str) -> np.ndarray:
    """``rx:Dop2R`` (present, i.e. extent >= 0.2 in the type's adult cluster), ``rx:Dop2R>0.5``."""
    m = RX_RE.match(value)
    if not m:
        raise ValueError(f"a receptor filter looks like rx:Dop2R or rx:Dop2R>0.5, not '{value}'")
    gene, op, thr = m.group(1), m.group(2), m.group(3)
    rx, ont = receptors(), ontology()
    if rx.empty:
        raise ValueError("rx: needs the receptor expression table (data/vfb_receptors.json.gz), which is not installed")
    known = {g.lower(): g for g in rx.genes}
    if gene.lower() not in known:
        raise ValueError(f"no receptor '{gene}' in the expression table (known: {', '.join(sorted(rx.genes))})")
    gene = known[gene.lower()]
    thr = float(thr) if thr else 0.2
    names = []
    for t in ont.types:
        hit = rx.for_type(t, ont, conn)
        if hit is None:
            continue
        x = hit["extent"].get(gene, 0.0)
        if (x > thr) if op == ">" else (x >= thr):
            names.append(t)
    return _type_mask(conn, names)


# ---------------------------------------------------------------------------------------------
# 5. Inputs to the parts list
# ---------------------------------------------------------------------------------------------

POLICIES = ("off", "modulators", "all")


@dataclass
class Overrides:
    """Per neuron: the transmitter role the curated ontology gives it instead of the prediction."""
    sign: np.ndarray                       # effective sign of its fast synapses (+1/-1), conn.sign where unchanged
    mod_nt: np.ndarray                     # object: None = as predicted, "" = no modulator, or a modulator name
    keep_fast: np.ndarray                  # bool: a modulator that also keeps its fast synapses (co-release)
    action: np.ndarray                     # object: what was done to the neuron, or None
    curated: np.ndarray                    # object: the class's curated transmitter list, or None
    rows: list = field(default_factory=list)
    counts: dict = field(default_factory=dict)


def transmitter_overrides(conn, policy: str = "modulators", modulators=MODULATOR_NTS) -> Overrides:
    """Apply the ontology's curated transmitters to the connectome's predictions.

    ``off``: nothing. ``modulators`` (the default): the literature-curated classes fill "unclear"
    predictions and change *what kind of modulator* a neuron is (a co-releasing neuron keeps its fast
    synapses and gains a tone; a neuron the literature calls octopaminergic and the prediction
    cholinergic gains an octopamine tone and keeps its predicted synapses; one the prediction calls
    serotonergic and the literature cholinergic loses its tone), never the sign of a confident fast
    prediction. ``all``: the literature also wins over a confident fast prediction (its sign is flipped
    where the class asserts fast transmitters of the other sign only, and a neuron the literature calls
    purely modulatory loses its predicted fast synapses), and the connectome-derived classes
    (FBbt:2xxxxxxx: another data set's own prediction for the same type) fill "unclear" predictions.
    They never change a confident one, and coarse matches (a ``_a`` type mapped to its stem's class, a
    neuron's class read from one VFB individual) only ever annotate. ``modulators`` are the modulator
    names the caller models; a curated modulator outside them is ignored."""
    if policy not in POLICIES:
        raise ValueError(f"curated policy must be one of {POLICIES}, not '{policy}'")
    n = conn.n
    sign = conn.sign.astype(np.float32).copy()
    mod_nt = np.full(n, None, dtype=object)
    keep_fast = np.zeros(n, dtype=bool)
    action = np.full(n, None, dtype=object)
    curated = np.full(n, None, dtype=object)
    rows: list[dict] = []
    ont = ontology()
    counts = collections.Counter()
    if policy == "off" or ont.empty:
        return Overrides(sign, mod_nt, keep_fast, action, curated, rows, {"policy": policy, "types": 0, "neurons": 0, "by_action": {}})
    lookup = {t: k for k, t in enumerate(conn.tables["types"])}
    order = np.argsort(conn.type_idx, kind="stable")
    bounds = np.searchsorted(conn.type_idx[order], np.arange(len(conn.tables["types"]) + 1))
    for t, e in ont.types.items():
        S = e.get("nt")
        k_type = lookup.get(t)
        if not S or k_type is None:
            continue
        idx = order[bounds[k_type]:bounds[k_type + 1]]
        if idx.size == 0:
            continue
        if e.get("coarse"):
            continue                                         # the class of a family of types: annotation only
        literature = e.get("evidence") == "literature"
        fast = [x for x in S if x in FAST_SIGN]
        mods = [x for x in S if x in MODULATOR_NTS and x in modulators]
        fast_signs = {FAST_SIGN[x] for x in fast}
        fast_sign = fast_signs.pop() if len(fast_signs) == 1 else None
        per_action: dict[str, list[int]] = collections.defaultdict(list)
        for i in idx.tolist():
            k = conn.nt[i]
            unclear = k in ("", "unclear")
            if not literature and not (unclear and policy == "all"):
                continue                                     # another connectome's prediction: only fills "unclear", only under "all"
            if k in S:
                if mods and k in FAST_SIGN:                  # e.g. Mi15: cholinergic and dopaminergic
                    mod_nt[i], keep_fast[i] = mods[0], True
                    per_action["co-release: fast synapses kept, tone added"].append(i)
                elif k in MODULATOR_NTS and fast and fast_sign is not None:   # a modulator that also releases GABA...
                    keep_fast[i], sign[i] = True, fast_sign
                    per_action["co-release: tone and fast synapses"].append(i)
                continue
            if unclear:
                if mods and fast and fast_sign is not None:
                    mod_nt[i], keep_fast[i], sign[i] = mods[0], True, fast_sign
                    per_action["unclear filled: fast synapses and a tone"].append(i)
                elif mods and not fast:
                    mod_nt[i] = mods[0]
                    per_action["unclear filled: a modulator"].append(i)
                elif mods:                                   # a modulator plus fast transmitters of both signs
                    mod_nt[i], keep_fast[i] = mods[0], True
                    per_action["unclear filled: a tone, fast synapses left as predicted"].append(i)
                elif fast_sign is not None:
                    sign[i] = fast_sign
                    per_action["unclear filled: fast transmitter"].append(i)
                continue
            if k in MODULATOR_NTS:                           # predicted a modulator, the literature says otherwise
                if mods:
                    mod_nt[i] = mods[0]
                    if fast and fast_sign is not None:
                        keep_fast[i], sign[i] = True, fast_sign
                    per_action["modulator changed"].append(i)
                elif fast_sign is not None:
                    mod_nt[i], sign[i] = "", fast_sign
                    per_action["not a modulator: fast synapses kept"].append(i)
                continue
            # predicted a fast transmitter with confidence, the literature disagrees
            if mods:
                mod_nt[i] = mods[0]
                if fast:
                    keep_fast[i] = True
                    if fast_sign is not None and fast_sign != sign[i] and policy == "all":
                        sign[i] = fast_sign
                        per_action["sign flipped and a tone added"].append(i)
                    else:
                        per_action["tone added, predicted synapses kept"].append(i)
                else:
                    keep_fast[i] = policy != "all"
                    per_action["tone added" + (" (predicted synapses kept)" if keep_fast[i] else ", fast synapses removed")].append(i)
            elif fast_sign is not None and fast_sign != sign[i] and policy == "all":
                sign[i] = fast_sign
                per_action["sign flipped"].append(i)
        for what, ids in per_action.items():
            rows.append({"type": t, "n": len(ids), "predicted": _majority_nt(conn).get(t), "curated": S,
                         "evidence": e.get("evidence"), "action": what, "fbbt": e["fbbt"][0], "label": ont.label(e["fbbt"][0])})
            counts[what] += len(ids)
            for i in ids:
                action[i], curated[i] = what, S
    rows.sort(key=lambda r: -r["n"])
    return Overrides(sign, mod_nt, keep_fast, action, curated, rows,
                     {"policy": policy, "types": len({r["type"] for r in rows}), "neurons": int(sum(counts.values())),
                      "by_action": dict(counts)})


def receptor_signs(conn, targets: np.ndarray, modulators) -> tuple[np.ndarray, list[dict]]:
    """Per modulator and target: the sign-weight of the tone from the target type's receptors (+1 where
    nothing is known, the old one-sign rule). Also a coverage row per modulator."""
    K, T = len(modulators), int(targets.size)
    out = np.ones((K, T), dtype=np.float32)
    rx, ont = receptors(), ontology()
    coverage = [{"nt": m.nt, "targets": T, "with_data": 0, "mean_sign": 1.0, "negative": 0} for m in modulators]
    if rx.empty or ont.empty or T == 0:
        return out, coverage
    per_type: dict[str, list[float | None]] = {}
    types = conn.types[targets]
    for j, t in enumerate(types.tolist()):
        if t not in per_type:
            hit = rx.for_type(t, ont, conn)
            per_type[t] = [None] * K if hit is None else [rx.sign(hit["extent"], m.nt) for m in modulators]
        for k, s in enumerate(per_type[t]):
            if s is not None:
                out[k, j] = np.float32(s)
    for k in range(K):
        known = np.array([per_type[t][k] is not None for t in types.tolist()])
        if known.any():
            vals = out[k, known]
            coverage[k].update(with_data=int(known.sum()), mean_sign=round(float(vals.mean()), 3), negative=int((vals < 0).sum()))
    return out, coverage


def receptors_of_type(type_name: str, conn=None) -> dict | None:
    """For the popover: the receptors a type's adult cluster expresses, with the data set."""
    rx, ont = receptors(), ontology()
    hit = rx.for_type(type_name, ont, conn)
    if hit is None:
        return None
    genes = sorted(hit["extent"].items(), key=lambda kv: -kv[1])
    return {"class": hit["class"], "label": hit["label"], "depth": hit["depth"], "family": hit["family"],
            "family_label": rx.source.get("families", {}).get(hit["family"], {}).get("label", hit["family"]),
            "cluster": hit["cluster"],
            "receptors": [{"gene": g, "extent": x, "modulator": rx.genes.get(g, {}).get("modulator"),
                           "sign": rx.genes.get(g, {}).get("sign"), "flybase": FLYBASE.format(rx.genes[g]["fbgn"]) if g in rx.genes and rx.genes[g].get("fbgn") else None}
                          for g, x in genes]}


def describe_type(type_name: str, conn=None) -> dict | None:
    return ontology().describe_type(type_name, conn)
