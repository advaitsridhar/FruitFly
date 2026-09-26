"""
Genetics: the genes the connectome data can speak for, and the driver lines that reach its neurons.

The genome cannot go into the simulation directly (it is a recipe for building a fly, not a
description of the finished brain). What can go in is *gene expression per cell type*, joined to
the wiring by cell-type names, and this kit's data file already carries two kinds of it:

* **fruitless and doublesex**, the two transcription factors that make a male brain male. The
  MaleCNS team registered light-microscopy images of *fru* and *dsx* expression to the EM volume
  and marked the matching neurons as expressing the gene with high or low confidence (4,858 *fru*,
  412 *dsx*, 258 both), and compared every cell type with a female connectome to label neurons as
  male-specific or sexually dimorphic. This module exposes those labels as populations
  (``gene:fru``, ``gene:dsx``, ``gene:both``, ``dimorphism:male``, ``dimorphism:dimorphic``) that
  can be silenced, activated or watched like any other, which is how a fly lab uses these genes.
* **The transmitter each neuron makes**, predicted from the appearance of its synapses. Making
  acetylcholine, GABA, glutamate, histamine, serotonin, dopamine or octopamine is the work of a
  handful of synthesis and transport genes (ChAT/VAChT, Gad1/VGAT, VGlut, Hdc, Trh/SerT, ple/DAT,
  Tdc2/Tbh), so ``gene:VGlut`` names the glutamatergic neurons and so on. These are the genes the
  model's one physiological rule (glutamate, GABA and histamine inhibit, everything else excites)
  rests on.

Every gene links to its FlyBase report. :class:`NeuronBridge` fetches, from Janelia's public
NeuronBridge data, which split-GAL4 and GAL4 driver lines match a population's neurons and which
MaleCNS neurons a given line labels, so a real published line can be zapped or silenced here.
Results are cached on disk; the lookups need internet access and fail cleanly without it.
"""

from __future__ import annotations

import json
import re
import urllib.request
from pathlib import Path

import numpy as np

FLYBASE = "https://flybase.org/reports/{}"

# symbol, FlyBase id, name, what it marks in this data, population spec
GENES = [
    ("fru", "FBgn0004652", "fruitless", "transcription factor; annotated *fru*-expressing neurons (both confidence grades)", "gene:fru"),
    ("dsx", "FBgn0000504", "doublesex", "transcription factor; annotated *dsx*-expressing neurons (both grades)", "gene:dsx"),
    ("ChAT", "FBgn0000303", "Choline acetyltransferase", "makes acetylcholine: the cholinergic (excitatory) neurons", "nt:acetylcholine"),
    ("VAChT", "FBgn0270928", "Vesicular acetylcholine transporter", "packages acetylcholine: the same neurons", "nt:acetylcholine"),
    ("Gad1", "FBgn0004516", "Glutamic acid decarboxylase 1", "makes GABA: the GABAergic (inhibitory) neurons", "nt:gaba"),
    ("VGAT", "FBgn0033911", "Vesicular GABA transporter", "packages GABA: the same neurons", "nt:gaba"),
    ("VGlut", "FBgn0031424", "Vesicular glutamate transporter", "packages glutamate: the glutamatergic (inhibitory here) neurons", "nt:glutamate"),
    ("Hdc", "FBgn0005619", "Histidine decarboxylase", "makes histamine: the photoreceptors (inhibitory)", "nt:histamine"),
    ("Trh", "FBgn0035187", "Tryptophan hydroxylase (neuronal)", "makes serotonin: the serotonergic neurons", "nt:serotonin"),
    ("SerT", "FBgn0010414", "Serotonin transporter", "recycles serotonin: the same neurons", "nt:serotonin"),
    ("ple", "FBgn0005626", "pale (tyrosine hydroxylase)", "makes dopamine: the dopamine neurons (PAM, PPL1, ...)", "nt:dopamine"),
    ("DAT", "FBgn0034136", "Dopamine transporter", "recycles dopamine: the same neurons", "nt:dopamine"),
    ("Tdc2", "FBgn0050446", "Tyrosine decarboxylase 2", "makes tyramine, the step before octopamine: the octopaminergic neurons", "nt:octopamine"),
    ("Tbh", "FBgn0010329", "Tyramine beta-hydroxylase", "makes octopamine: the same neurons", "nt:octopamine"),
]
GENE_SPEC = {g[0].lower(): g[4] for g in GENES}
FLYBASE_ID = {g[0]: g[1] for g in GENES}

# the expression labels in the data ("fruDsx" table) grouped by gene
# MaleCNS labels carry a confidence (_high/_low); FlyWire's (the female fly) are plain fru, dsx, coexpress
FRU_LABELS = ("fru_high", "fru_low", "coexpress_high", "coexpress_low", "fru", "coexpress")
DSX_LABELS = ("dsx_high", "dsx_low", "coexpress_high", "coexpress_low", "dsx", "coexpress")
BOTH_LABELS = ("coexpress_high", "coexpress_low", "coexpress")
HIGH_LABELS = ("fru_high", "dsx_high", "coexpress_high")
GENE_GROUPS = {"fru": FRU_LABELS, "fruitless": FRU_LABELS, "dsx": DSX_LABELS, "doublesex": DSX_LABELS,
               "both": BOTH_LABELS, "fru+dsx": BOTH_LABELS, "coexpress": BOTH_LABELS}
DIMORPHISM_GROUPS = {"male": ("male-specific", "potentially male-specific"),
                     "female": ("female-specific", "potentially female-specific"),
                     "dimorphic": ("sexually dimorphic", "potentially sexually dimorphic")}
TRANSMITTER_GENES = {"acetylcholine": ["ChAT", "VAChT"], "gaba": ["Gad1", "VGAT"], "glutamate": ["VGlut"],
                     "histamine": ["Hdc"], "serotonin": ["Trh", "SerT"], "dopamine": ["ple", "DAT"],
                     "octopamine": ["Tdc2", "Tbh"]}

SOURCE = ("fruitless/doublesex labels and the male-specific/dimorphic status: the MaleCNS v1.0 annotation "
          "(light-microscopy expression images registered to the EM volume; a female connectome for the "
          "comparison). Transmitters: predicted from synapse appearance in the EM data. Gene identities "
          "and links: FlyBase.")
SOURCE_FEMALE = ("fruitless/doublesex labels and the female-specific/dimorphic status: FlyWire's annotation "
                 "(Schlegel et al. 2024; matched to published expression data and to the MaleCNS, Berg et al. 2025). "
                 "Transmitters: FlyWire's prediction from synapse appearance, as it is (it has no histamine class and "
                 "calls the Kenyon cells dopaminergic; the parts list corrects both from FlyWire's literature column, "
                 "this card does not). Gene identities and links: FlyBase.")


def gene_mask(conn, value: str) -> np.ndarray:
    """The neurons a ``gene:VALUE`` filter selects (used by :meth:`Connectome.select`)."""
    v = value.strip()
    key = v.lower()
    if key in GENE_GROUPS:
        return np.isin(conn.frudsx, GENE_GROUPS[key])
    if v in set(conn.tables.get("fruDsx", [])):          # one confidence grade, e.g. gene:fru_high
        return conn.frudsx == v
    if key in GENE_SPEC and GENE_SPEC[key].startswith("nt:"):
        return conn.nt == GENE_SPEC[key][3:]
    raise ValueError(f"unknown gene '{value}' (try fru, dsx, both, fru_high, or a transmitter gene such as "
                     f"{', '.join(g[0] for g in GENES[2:])})")


def dimorphism_mask(conn, value: str) -> np.ndarray:
    v = value.strip().lower()
    if v in DIMORPHISM_GROUPS:
        return np.isin(conn.dimorphism, DIMORPHISM_GROUPS[v])
    if v == "any":
        return np.isin(conn.dimorphism, sum(DIMORPHISM_GROUPS.values(), ()))
    return conn.dimorphism == value


def genotype(conn, idx: np.ndarray) -> dict:
    """What a population expresses, as fractions, plus short tags for the screen (>= half the cells)."""
    idx = np.asarray(idx)
    if idx.size == 0:
        return {"n": 0, "fru": 0.0, "dsx": 0.0, "male": 0.0, "female": 0.0, "dimorphic": 0.0, "nt": "", "tags": []}
    fd = conn.frudsx[idx]
    dm = conn.dimorphism[idx]
    nts = conn.nt[idx]
    vals, counts = np.unique(nts, return_counts=True)
    nt = str(vals[np.argmax(counts)]) if vals.size else ""
    out = {"n": int(idx.size),
           "fru": float(np.isin(fd, FRU_LABELS).mean()), "dsx": float(np.isin(fd, DSX_LABELS).mean()),
           "male": float(np.isin(dm, DIMORPHISM_GROUPS["male"]).mean()),
           "female": float(np.isin(dm, DIMORPHISM_GROUPS["female"]).mean()),
           "dimorphic": float(np.isin(dm, DIMORPHISM_GROUPS["dimorphic"]).mean()), "nt": nt}
    tags = [t for t, k in (("fru", "fru"), ("dsx", "dsx"), ("♂", "male"), ("♀", "female"), ("♂♀", "dimorphic"))
            if out[k] >= 0.5]
    out["tags"] = tags
    return out


def genes_of(conn, i: int) -> list[dict]:
    """The genes this data can name for one neuron, each with its FlyBase link."""
    i = int(i)
    out = []
    label = conn.frudsx[i]
    grade = (" (high confidence)" if label.endswith("_high") else " (low confidence)" if label.endswith("_low")
             else "")                                             # FlyWire's labels have no grade
    if label in FRU_LABELS:
        out.append({"symbol": "fru", "flybase": FLYBASE.format(FLYBASE_ID["fru"]), "why": f"fruitless-expressing{grade}"})
    if label in DSX_LABELS:
        out.append({"symbol": "dsx", "flybase": FLYBASE.format(FLYBASE_ID["dsx"]), "why": f"doublesex-expressing{grade}"})
    for sym in TRANSMITTER_GENES.get(conn.nt[i], []):
        out.append({"symbol": sym, "flybase": FLYBASE.format(FLYBASE_ID[sym]), "why": f"makes/handles {conn.nt[i]}"})
    return out


def summary(conn, readouts: list[dict] | None = None) -> dict:
    """Everything the Genetics panel shows: the expression populations, the transmitter groups, the
    genotype of each readout, and the source line."""
    groups = []
    graded = any(t.endswith(("_high", "_low")) for t in conn.tables.get("fruDsx", []))   # FlyWire's labels are not
    for key, label, spec, gene in (("fru", "fruitless", "gene:fru", "fru"), ("dsx", "doublesex", "gene:dsx", "dsx"),
                                   ("both", "fruitless and doublesex", "gene:both", None)):
        idx = conn.select(spec)
        high = int(np.isin(conn.frudsx[idx], HIGH_LABELS).sum()) if graded else None
        groups.append({"key": key, "label": label, "spec": spec, "n": int(idx.size), "high": high,
                       "gene": gene, "flybase": FLYBASE.format(FLYBASE_ID[gene]) if gene else None,
                       "types": int(len(set(conn.types[idx].tolist()) - {""}))})
    sex = getattr(conn, "sex", "male")
    for key, label, spec in ((sex, f"{sex}-specific", f"dimorphism:{sex}"), ("dimorphic", "sexually dimorphic", "dimorphism:dimorphic")):
        idx = conn.select(spec)
        groups.append({"key": key, "label": label, "spec": spec, "n": int(idx.size), "high": None, "gene": None,
                       "flybase": None, "types": int(len(set(conn.types[idx].tolist()) - {""}))})
    transmitters = []
    total_syn = float(conn.n_syn.sum())
    for nt, genes in TRANSMITTER_GENES.items():
        idx = conn.select(f"nt:{nt}")
        share = float(conn.n_syn[np.isin(conn.pre_idx, idx)].sum() / total_syn) if idx.size else 0.0
        transmitters.append({"nt": nt, "spec": f"nt:{nt}", "n": int(idx.size), "sign": int(conn.sign[idx[0]]) if idx.size else 0,
                             "synapse_share": share,
                             "genes": [{"symbol": g, "flybase": FLYBASE.format(FLYBASE_ID[g])} for g in genes]})
    unclear = conn.select("nt:unclear")
    # MaleCNS counts its "unclear" neurons as excitatory; FlyWire's keep the low-confidence prediction's sign
    out = {"expression": groups, "transmitters": transmitters, "unclear": int(unclear.size),
           "unclear_inhibitory": int((conn.sign[unclear] < 0).sum()),
           "source": SOURCE_FEMALE if sex == "female" else SOURCE,
           "genes": [{"symbol": s, "flybase": FLYBASE.format(fb), "name": n, "marks": m, "spec": sp} for s, fb, n, m, sp in GENES]}
    if readouts is not None:
        out["readouts"] = {r["key"]: genotype(conn, conn.select(r["spec"])) for r in readouts}
    return out


# ---------------------------------------------------------------- NeuronBridge (Janelia): driver lines <-> neurons
NB_BASE = "https://janelia-neuronbridge-data-prod.s3.amazonaws.com"
MALECNS_LIBRARIES = ("FlyEM_Male_CNS_Brain", "FlyEM_Male_CNS_VNC")


class NeuronBridgeError(RuntimeError):
    pass


def _default_fetch(url: str, timeout: float) -> bytes:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read()


def _malecns_only(conn):
    """NeuronBridge matches driver-line images to MaleCNS neurons by body id; a FlyWire root id matches nothing."""
    if getattr(conn, "sex", "male") != "male":
        raise ValueError("NeuronBridge lookups match MaleCNS neurons; they are not available for the female fly (FlyWire)")


class NeuronBridge:
    """Lookups in NeuronBridge's public data: lines that match a neuron (colour-depth search) and
    MaleCNS neurons that match a line. Every file is cached under ``cache_dir`` once fetched.

    ``fetch(url, timeout) -> bytes`` can be replaced (the tests use an offline fake)."""

    def __init__(self, cache_dir: Path | str | None = None, fetch=None, timeout: float = 25.0):
        from .connectome import DATA_DIR
        self.cache_dir = Path(cache_dir) if cache_dir is not None else DATA_DIR / "neuronbridge"
        self.fetch = fetch or _default_fetch
        self.timeout = timeout
        self._version: str | None = None
        self._mem: dict[str, object] = {}

    # ---- plumbing
    def _get(self, path: str):
        if path in self._mem:
            return self._mem[path]
        safe = re.sub(r"[^A-Za-z0-9_.-]", "_", path.strip("/"))
        cached = self.cache_dir / safe
        if cached.exists():
            data = json.loads(cached.read_text())
        else:
            try:
                raw = self.fetch(NB_BASE + path, self.timeout)
            except Exception as e:                     # no network, 403, timeout: one clear message
                raise NeuronBridgeError(f"NeuronBridge (Janelia) could not be reached: {e}. "
                                        "Driver-line lookups need internet access.") from e
            try:
                data = json.loads(raw)
            except Exception as e:
                raise NeuronBridgeError(f"NeuronBridge returned something that is not JSON for {path}") from e
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            cached.write_text(json.dumps(data))
        self._mem[path] = data
        return data

    def version(self) -> str:
        if self._version is None:
            try:
                self._version = self.fetch(NB_BASE + "/current.txt", self.timeout).decode().strip()
            except Exception as e:
                raise NeuronBridgeError(f"NeuronBridge (Janelia) could not be reached: {e}. "
                                        "Driver-line lookups need internet access.") from e
        return self._version

    def _by_body(self, body: int) -> list[dict]:
        try:
            js = self._get(f"/{self.version()}/metadata/by_body/{int(body)}.json")
        except NeuronBridgeError as e:
            if "404" in str(e) or "Not Found" in str(e):
                return []
            raise
        return [r for r in js.get("results", []) if str(r.get("libraryName", "")).startswith(MALECNS_LIBRARIES)]

    def _cds(self, name: str) -> list[dict]:
        return self._get(f"/{self.version()}/metadata/cdsresults/{name}").get("results", [])

    # ---- queries
    def lines_for(self, conn, spec: str, max_neurons: int = 4, top: int = 12) -> dict:
        """Driver lines whose expression images match the neurons of ``spec`` (a sample of at most
        ``max_neurons`` of them, spread over the population). A line's score is its best
        colour-depth-search score; lines that match several of the sampled neurons come first."""
        _malecns_only(conn)
        idx = conn.select(spec)
        if idx.size == 0:
            raise ValueError(f"no neurons match '{spec}'")
        sample = idx[np.linspace(0, idx.size - 1, min(max_neurons, idx.size)).round().astype(int)]
        lines: dict[str, dict] = {}
        unmatched = []
        for i in sample:
            body = int(conn.body_id[i])
            entries = self._by_body(body)
            if not entries:
                unmatched.append(body)
                continue
            seen_here: set[str] = set()
            for e in entries:
                cds = e.get("files", {}).get("CDSResults")
                if not cds:
                    continue
                for m in self._cds(cds):
                    im = m.get("image", {})
                    name = im.get("publishedName")
                    if not name:
                        continue
                    row = lines.setdefault(name, {"line": name, "library": im.get("libraryName", ""), "score": 0.0,
                                                  "neurons": 0, "split": "Split-GAL4" in str(im.get("libraryName", ""))})
                    row["score"] = max(row["score"], float(m.get("normalizedScore", 0.0)))
                    if name not in seen_here:
                        row["neurons"] += 1
                        seen_here.add(name)
        ranked = sorted(lines.values(), key=lambda r: (-r["neurons"], -r["score"]))[:top]
        return {"spec": spec, "n": int(idx.size), "sampled": [int(conn.body_id[i]) for i in sample],
                "unmatched": unmatched, "lines": ranked, "version": self.version()}

    def neurons_for_line(self, conn, line: str, max_images: int = 6, top: int = 40) -> dict:
        """MaleCNS neurons that match a driver line's expression images. A line can have dozens of
        images (each match file is about a megabyte), so ``max_images`` of them are searched, brain
        images first. Each neuron carries NeuronBridge's cell type and the kit's, so a disagreement
        (the two data versions differ) is visible."""
        _malecns_only(conn)
        line = line.strip()
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", line):
            raise ValueError("a line name looks like SS02385, MB112C, R85B12 or VT019730")
        try:
            js = self._get(f"/{self.version()}/metadata/by_line/{line}.json")
        except NeuronBridgeError as e:
            if "404" in str(e) or "Not Found" in str(e):
                raise ValueError(f"NeuronBridge has no line called '{line}'") from e
            raise
        images = [im for im in js.get("results", []) if im.get("files", {}).get("CDSResults")]
        images.sort(key=lambda im: 0 if im.get("anatomicalArea", "Brain") == "Brain" else 1)   # stable: brain first
        if not images:
            raise ValueError(f"NeuronBridge has no searchable images for line '{line}'")
        body_index = _body_index(conn)
        found: dict[int, dict] = {}
        for im in images[:max_images]:
            for m in self._cds(im["files"]["CDSResults"]):
                em = m.get("image", {})
                if not str(em.get("libraryName", "")).startswith(MALECNS_LIBRARIES):
                    continue
                try:
                    body = int(str(em.get("publishedName", "")).split(":")[-1])
                except ValueError:
                    continue
                score = float(m.get("normalizedScore", 0.0))
                row = found.get(body)
                if row is None or score > row["score"]:
                    k = body_index.get(body)
                    found[body] = {"body": body, "index": k, "score": score, "nb_type": em.get("neuronType") or "",
                                   "type": conn.types[k] if k is not None else "", "side": conn.side[k] if k is not None else "",
                                   "in_kit": k is not None}
        ranked = sorted(found.values(), key=lambda r: -r["score"])[:top]
        in_kit = [r for r in ranked if r["in_kit"]]
        return {"line": line, "images": len(images), "searched": min(max_images, len(images)),
                "library": images[0].get("libraryName", ""), "neurons": ranked,
                "spec": ",".join(f"body:{r['body']}" for r in in_kit[:20]), "version": self.version()}


def _body_index(conn) -> dict[int, int]:
    cache = getattr(conn, "_body_index_cache", None)
    if cache is None:
        cache = {int(b): i for i, b in enumerate(conn.body_id.tolist())}
        conn._body_index_cache = cache
    return cache
