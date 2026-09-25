"""
The wiring diagram: loading the MaleCNS v1.0 connectome and asking it questions.

This module is the only place that knows the on-disk format. Everything else works with a
:class:`Connectome`, which answers three kinds of questions:

* **Which neurons?** :meth:`Connectome.select` turns a *population spec* such as ``"LC10a/L"`` or
  ``"class:Kenyon_Cell"`` into neuron indices.
* **Who talks to whom?** :meth:`Connectome.inputs_of` / :meth:`Connectome.outputs_of` summarise a
  population's partners by cell type, and :meth:`Connectome.type_graph` collapses the 6.3 million
  connections into a cell-type-level graph that :mod:`virtual_fly.pathways` searches.
* **Where is it?** soma positions and medulla column (hex) coordinates, used by the brain map and
  by the retina model in :mod:`virtual_fly.senses.vision`.

Data credit (CC BY 4.0): Berg et al., *Cell* 2026 (MaleCNS v1.0, neuPrint ``male-cns:v1.0``). The
compact ``.flyb`` file was built from neuPrint by the fly-brain-minecraft project (MIT / CC BY 4.0)
and keeps every connection with 5 or more synapses.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
import struct
import sys
import time
import urllib.request
from pathlib import Path
from typing import Iterable

import numpy as np

# ----------------------------------------------------------------------------------------------
# 1. Getting the data
# ----------------------------------------------------------------------------------------------

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = PACKAGE_DIR.parent
DEFAULT_DATA_FILE = PROJECT_DIR / "data" / "malecns-v1.0.flyb.gz"
# FLY_DATA_FILE points the whole kit at another connectome file (no download, no checksum)
DATA_FILE = Path(os.environ.get("FLY_DATA_FILE", DEFAULT_DATA_FILE))
# Pinned to one commit so the file can never change under you.
DATA_URL = ("https://raw.githubusercontent.com/blendi-remade/fly-brain-minecraft/"
            "6cfa30175003ef25da68a237d5eda958f8047b82/src/main/resources/connectome/malecns-v1.0.flyb.gz")
DATA_SHA256 = "e33df182bed7a6f3ea279daf4790a82b05706d3d41e819a6a80c0473e8c559f3"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_connectome(path: Path | str = DATA_FILE, url: str | None = None, quiet: bool = False) -> Path:
    """Download the connectome file once (about 23 MB) and check that it is intact."""
    path = Path(path)
    url = url or os.environ.get("FLY_DATA_URL", DATA_URL)
    if path.exists() and _sha256(path) == DATA_SHA256:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    if not quiet:
        print(f"Downloading the fly connectome (23 MB) from\n  {url}", file=sys.stderr)
    tmp = path.with_suffix(".part")
    manual = (f"\nYou can also download it yourself in a browser from\n  {DATA_URL}\n"
              f"and save it as\n  {path}\nthen run this again.")
    try:
        with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as f:
            total = int(r.headers.get("Content-Length") or 0)
            done = 0
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                if total and not quiet:
                    print(f"\r  {done / 1e6:5.1f} / {total / 1e6:.1f} MB", end="", file=sys.stderr, flush=True)
    except OSError as e:                                   # no internet, firewall, certificate problems...
        if tmp.exists():
            tmp.unlink()
        raise SystemExit(f"\nCouldn't download the connectome: {e}{manual}")
    if not quiet:
        print(file=sys.stderr)
    if _sha256(tmp) != DATA_SHA256:
        tmp.unlink()
        raise SystemExit(f"The downloaded file is damaged (its checksum doesn't match). Please try again.{manual}")
    tmp.replace(path)
    return path


# ----------------------------------------------------------------------------------------------
# 2. Reading the wiring diagram
# ----------------------------------------------------------------------------------------------

class Connectome:
    """The wiring diagram: one entry per neuron, plus who connects to whom.

    Neurons are numbered ``0 .. n-1``. For neuron ``i``:

    ==================  =====================================================================
    ``body_id[i]``      its ID in neuPrint (look it up at neuprint.janelia.org, ``male-cns:v1.0``)
    ``types[i]``        its cell type, e.g. ``"MN9"`` or ``"DNa02"`` (``""`` if unannotated)
    ``sign[i]``         ``+1`` excitatory, ``-1`` inhibitory (from its predicted neurotransmitter)
    ``side[i]``         ``"L"``, ``"R"``, ``"M"`` (midline) or ``""``
    ``soma[i]``         (x, y, z) of the cell body in nm, NaN if unknown
    ``hex1[i]``, ``hex2[i]``  medulla column coordinates for columnar visual neurons, else -1
    ==================  =====================================================================

    Connections are stored *compressed sparse row* style: the outgoing connections of neuron ``i``
    are ``post_idx[row_ptr[i]:row_ptr[i+1]]`` with synapse counts ``n_syn`` at the same positions.
    :attr:`col_ptr` / :attr:`pre_of_input` give the same edges sorted by *target* (built lazily).
    """

    TABLES = ["types", "superclasses", "classes", "subclasses", "nts", "sides",
              "dimorphisms", "fruDsx", "neuromeres", "nerves"]

    def __init__(self, path: Path | str):
        path = Path(path)
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rb") as f:
            buf = f.read()
        self.path = path
        self._buf, self._pos = buf, 0
        if buf[:4] != b"FLYB":
            raise ValueError(f"{path} is not a FLYB connectome file")
        self._pos = 4
        version, n, n_edges, n_retina = self._unpack("<IIII")
        if version != 1:
            raise ValueError(f"unsupported FLYB version {version}")
        self.n, self.n_edges = int(n), int(n_edges)
        self.dataset = self._str("<H")
        self.meta = json.loads(self._str("<I"))
        self.tables = {name: self._table() for name in self.TABLES}

        a = self._array
        self.body_id = a("<i8", n)
        self.type_idx = a("<i4", n)
        self.superclass_idx = a("u1", n)
        self.class_idx = a("u1", n)
        self.subclass_idx = a("<u2", n)
        self.nt_idx = a("u1", n)
        self.sign = a("i1", n).astype(np.float32)
        self.side_idx = a("u1", n)
        self.hex1, self.hex2 = a("i1", n).astype(np.int16), a("i1", n).astype(np.int16)
        self.dimorphism_idx, self.frudsx_idx = a("u1", n), a("u1", n)
        self.neuromere_idx, self.nerve_idx = a("u1", n), a("u1", n)
        self.soma = a("<f4", 3 * n).reshape(n, 3)          # NaN where the cell body wasn't found
        self.n_pre, self.n_post = a("<i4", n), a("<i4", n)
        self.row_ptr = a("<i4", n + 1).astype(np.int64)
        self.post_idx = a("<i4", n_edges)
        self.n_syn = a("<u2", n_edges)
        del self._buf

        t = self.tables
        self.types = np.array(t["types"], dtype=object)[self.type_idx]
        self.superclass = np.array(t["superclasses"], dtype=object)[self.superclass_idx]
        self.cls = np.array(t["classes"], dtype=object)[self.class_idx]
        self.subclass = np.array(t["subclasses"], dtype=object)[self.subclass_idx]
        self.nt = np.array(t["nts"], dtype=object)[self.nt_idx]
        self.side = np.array(t["sides"], dtype=object)[self.side_idx]
        self.nerve = np.array(t["nerves"], dtype=object)[self.nerve_idx]
        self.neuromere = np.array(t["neuromeres"], dtype=object)[self.neuromere_idx]
        self.dimorphism = np.array(t["dimorphisms"], dtype=object)[self.dimorphism_idx]
        self.frudsx = np.array(t["fruDsx"], dtype=object)[self.frudsx_idx]
        # which fly: "male" (MaleCNS) or "female" (FlyWire, see flywire.py), and the kit's names for cells this
        # file calls something else (e.g. the female file's "MN9" -> "CB0701")
        self.sex = str(self.meta.get("sex", "male"))
        self.aliases: dict[str, str] = dict(self.meta.get("aliases") or {})
        self._by_type = None
        self._cache: dict[str, np.ndarray] = {}
        self._resolving: set[str] = set()
        self._col = None
        self._type_graph = None

    def rewired(self, row_ptr: np.ndarray, post_idx: np.ndarray, n_syn: np.ndarray, label: str = "rewired") -> "Connectome":
        """A copy of this connectome with new wiring (same neurons and annotations). Used by
        :mod:`virtual_fly.wiring` to grow synthetic flies. Caches that depend on the wiring are
        dropped; everything per neuron is shared with the original."""
        c = object.__new__(Connectome)
        c.__dict__.update({k: v for k, v in self.__dict__.items() if not k.startswith("_")})
        c.row_ptr = np.asarray(row_ptr, dtype=np.int64)
        c.post_idx = np.asarray(post_idx, dtype=np.int32)
        c.n_syn = np.asarray(n_syn, dtype=np.uint16)
        if c.row_ptr.size != self.n + 1 or c.post_idx.size != c.n_syn.size or int(c.row_ptr[-1]) != c.post_idx.size:
            raise ValueError("rewired(): row_ptr, post_idx and n_syn do not describe a valid CSR wiring")
        c.n_edges = int(c.post_idx.size)
        syn = c.n_syn.astype(np.int64)
        c.n_post = np.bincount(c.post_idx, weights=syn, minlength=self.n).astype(np.int32)
        c.n_pre = np.bincount(np.repeat(np.arange(self.n), np.diff(c.row_ptr)), weights=syn, minlength=self.n).astype(np.int32)
        c.dataset = f"{self.dataset} ({label})"
        c.meta = dict(self.meta, rewired=label)
        c._by_type = self._by_type            # depends only on the types: safe to share
        c._cache = dict(self._cache)          # population selections depend only on the annotations
        c._resolving = set()
        c._col = None
        c._type_graph = None
        return c

    # --- tiny binary reader helpers ---
    def _unpack(self, fmt):
        vals = struct.unpack_from(fmt, self._buf, self._pos)
        self._pos += struct.calcsize(fmt)
        return vals

    def _str(self, len_fmt):
        (length,) = self._unpack(len_fmt)
        s = self._buf[self._pos:self._pos + length].decode("utf-8")
        self._pos += length
        return s

    def _table(self):
        (count,) = self._unpack("<H")
        return [self._str("<H") for _ in range(count)]

    def _array(self, dtype, count):
        arr = np.frombuffer(self._buf, dtype=dtype, count=int(count), offset=self._pos).copy()
        self._pos += arr.nbytes
        return arr

    # ------------------------------------------------------------------ finding neurons
    def select(self, spec: str | np.ndarray | Iterable[int]) -> np.ndarray:
        """Return the (sorted, unique) indices of neurons matching a population spec.

        A spec is a comma-separated list of terms (their union). Each term is one of::

            MN9                  exact cell type
            prefix:LC10          cell types starting with this text
            contains:LC10        cell types containing this text
            regex:^LC1[0-2]      cell types matching a regular expression
            class:Kenyon_Cell    neuPrint class      (also superclass:, subclass:, nt:, nerve:,
                                                      neuromere:, dimorphism:, frudsx:)
            gene:fru             neurons annotated as expressing fruitless (gene:dsx, gene:both,
                                 gene:fru_high; or a transmitter gene: gene:VGlut, gene:Gad1, ...)
            dimorphism:male      male-specific incl. 'potentially' (dimorphism:dimorphic, :any,
                                 or an exact label such as dimorphism:sexually dimorphic)
            fbbt:<class>         an anatomy-ontology class and everything below it, by id or label
                                 (fbbt:FBbt_00003870, fbbt:lobula columnar neuron, fbbt:dopaminergic neuron)
            rx:<receptor>        cell types whose adult scRNA-seq cluster expresses a receptor
                                 (rx:Dop2R = at least 20 % of the cells; rx:5-HT1A>0.5)
            body:10783           one neuron by its neuPrint bodyId
            index:1234           one neuron by its index in this file
            hex:12:7             columnar visual neurons in medulla column (hex1, hex2)
            all                  every neuron

        End a term with ``/L``, ``/R`` or ``/M`` to keep only that side, e.g. ``"DNa02/L"``.
        Join filters with ``&`` to intersect them, e.g. ``"class:mechanosensory_tactile&nerve:ADMN"``.
        Prefix a term with ``!`` to subtract it, e.g. ``"prefix:LC10,!LC10a"``.
        A file can carry aliases (the female fly's FlyWire file answers to the kit's MaleCNS names, e.g.
        ``"MN9"`` selects FlyWire's CB0701); a real cell type of the same name always wins.
        An array of indices is returned unchanged (so functions can accept either form).
        """
        if isinstance(spec, np.ndarray):
            return spec.astype(np.int64, copy=False)
        if not isinstance(spec, str):
            return np.asarray(list(spec), dtype=np.int64)
        key = spec
        if "fbbt:" in spec or "rx:" in spec:            # these depend on vfb's data, which tests (and tools) can swap
            from . import vfb
            key = (spec, vfb.generation())
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        if self._by_type is None:
            order = np.argsort(self.type_idx, kind="stable")
            bounds = np.searchsorted(self.type_idx[order], np.arange(len(self.tables["types"]) + 1))
            self._by_type = (order, bounds, {t: k for k, t in enumerate(self.tables["types"])})
        exact = self._exact_type(spec.strip())        # 70 type names contain ',' or '&': try whole first
        if exact is None:
            exact = self._alias(spec.strip())
        if exact is not None:
            result = np.flatnonzero(exact).astype(np.int64)
            self._cache[key] = result
            return result
        keep, drop = [], []
        for term in (s.strip() for s in spec.split(",")):
            if not term:
                continue
            negate = term.startswith("!")
            term = term[1:].strip() if negate else term
            mask = self._exact_type(term)
            if mask is None:
                mask = self._alias(term)
            if mask is None:
                mask = np.ones(self.n, dtype=bool)
                for part in self._and_parts(term):
                    mask &= self._match(part)
            (drop if negate else keep).append(mask)
        if not keep:
            result = np.zeros(0, dtype=np.int64)
        else:
            mask = np.logical_or.reduce(keep)
            for d in drop:
                mask &= ~d
            result = np.flatnonzero(mask).astype(np.int64)
        self._cache[key] = result
        return result

    def _and_parts(self, term: str) -> list[str]:
        """Split a term on ``&``, except inside an ontology label (ten lineage classes are called e.g.
        "adult SLPa&l1 lineage neuron"): an ``fbbt:`` piece takes the longest run of following pieces
        that still names a class."""
        pieces = term.split("&")
        out, i = [], 0
        while i < len(pieces):
            piece = pieces[i].strip()
            if piece.startswith("fbbt:") and i + 1 < len(pieces):
                for j in range(len(pieces), i + 1, -1):
                    joined = "&".join(pieces[i:j]).strip()
                    try:
                        self._match(joined)
                    except ValueError:
                        continue
                    piece, i = joined, j - 1
                    break
            out.append(piece)
            i += 1
        return out

    def _exact_type(self, term: str) -> np.ndarray | None:
        """Mask for a term that is literally a cell-type name (optionally with a ``/L`` ``/R`` ``/M`` side),
        or None. Checked before any splitting so type names containing ',' '&' or ':' still work."""
        order, bounds, lookup = self._by_type
        side = None
        if len(term) > 2 and term[-2] == "/" and term[-1] in "LRM" and term not in lookup:
            term, side = term[:-2], term[-1]
        k = lookup.get(term)
        if k is None or term == "":
            return None
        mask = np.zeros(self.n, dtype=bool)
        mask[order[bounds[k]:bounds[k + 1]]] = True
        if side is not None:
            mask &= self.side == side
        return mask

    def _alias(self, term: str) -> np.ndarray | None:
        """Mask for a term this file knows under another name (:attr:`aliases`, optionally with a side), or None.
        A real cell type of the same name always wins: aliases are looked up only when the name is not one."""
        if not self.aliases:
            return None
        target, side = self.aliases.get(term), None
        if target is None and len(term) > 2 and term[-2] == "/" and term[-1] in "LRM":
            target, side = self.aliases.get(term[:-2]), term[-1]
        if target is None or term in self._resolving:
            return None
        self._resolving.add(term)
        try:
            mask = np.zeros(self.n, dtype=bool)
            mask[self.select(target)] = True
        finally:
            self._resolving.discard(term)
        if side is not None:
            mask &= self.side == side
        return mask

    def _match(self, part: str) -> np.ndarray:
        side = None
        if len(part) > 2 and part[-2] == "/" and part[-1] in "LRM":
            part, side = part[:-2], part[-1]
        key, _, value = part.partition(":") if ":" in part else ("type", "", part)
        if part == "all":
            key = "all"
        if key == "type":
            order, bounds, lookup = self._by_type
            mask = np.zeros(self.n, dtype=bool)
            k = lookup.get(value)
            if k is not None and value != "":
                mask[order[bounds[k]:bounds[k + 1]]] = True
        elif key == "all":
            mask = np.ones(self.n, dtype=bool)
        elif key in ("prefix", "contains", "regex"):
            names = self.tables["types"]
            if key == "regex":
                rx = re.compile(value)
                ok = np.array([bool(t) and rx.search(t) is not None for t in names])
            else:
                ok = np.array([bool(t) and (t.startswith(value) if key == "prefix" else value in t) for t in names])
            mask = ok[self.type_idx]
        elif key == "gene":
            from . import genetics
            mask = genetics.gene_mask(self, value)
        elif key == "dimorphism":
            from . import genetics
            mask = genetics.dimorphism_mask(self, value)
        elif key == "fbbt":                          # an anatomy-ontology class and everything below it (vfb.py)
            from . import vfb
            mask = vfb.fbbt_mask(self, value)
        elif key == "rx":                            # cell types whose adult scRNA-seq cluster expresses a receptor
            from . import vfb
            mask = vfb.rx_mask(self, value)
        elif key in ("class", "superclass", "subclass", "nt", "nerve", "neuromere", "frudsx"):
            col = {"class": self.cls, "superclass": self.superclass, "subclass": self.subclass,
                   "nt": self.nt, "nerve": self.nerve, "neuromere": self.neuromere, "frudsx": self.frudsx}[key]
            mask = col == value
        elif key == "body":
            mask = self.body_id == int(value)
        elif key == "index":
            mask = np.zeros(self.n, dtype=bool)
            mask[int(value)] = True
        elif key == "hex":
            h1, h2 = (int(v) for v in value.split(":"))
            mask = (self.hex1 == h1) & (self.hex2 == h2)
        else:
            raise ValueError(f"unknown filter '{key}:' in population spec")
        if side is not None:
            mask &= self.side == side
        return mask

    def count(self, spec: str) -> int:
        try:
            return int(self.select(spec).size)
        except ValueError:
            return 0

    def find_types(self, text: str, limit: int | None = None) -> list[tuple[str, int]]:
        """List ``(type, count)`` for every cell type whose name contains ``text`` (case-insensitive)."""
        text = text.lower()
        counts = np.bincount(self.type_idx, minlength=len(self.tables["types"]))
        hits = [(t, int(counts[k])) for k, t in enumerate(self.tables["types"]) if t and text in t.lower()]
        hits.sort(key=lambda tc: (not tc[0].lower().startswith(text), tc[0].lower()))
        return hits[:limit] if limit else hits

    def type_counts(self) -> dict[str, int]:
        counts = np.bincount(self.type_idx, minlength=len(self.tables["types"]))
        return {t: int(counts[k]) for k, t in enumerate(self.tables["types"]) if t}

    def describe(self, i: int) -> str:
        i = int(i)
        return (f"#{i} bodyId {self.body_id[i]}  type {self.types[i] or '?'}  side {self.side[i] or '?'}  "
                f"{self.superclass[i]}/{self.cls[i]}  {self.nt[i]} ({'+' if self.sign[i] > 0 else '-'})  "
                f"{self.row_ptr[i + 1] - self.row_ptr[i]} outgoing connections")

    def info(self, i: int) -> dict:
        """Everything known about one neuron, as a JSON-friendly dict (used by the game's API)."""
        i = int(i)
        soma = self.soma[i]
        return {
            "index": i, "body_id": int(self.body_id[i]), "body_ref": str(int(self.body_id[i])),   # exact in JavaScript too
            "type": self.types[i], "side": self.side[i],
            "superclass": self.superclass[i], "class": self.cls[i], "subclass": self.subclass[i],
            "nt": self.nt[i], "sign": int(self.sign[i]), "nerve": self.nerve[i], "neuromere": self.neuromere[i],
            "dimorphism": self.dimorphism[i], "frudsx": self.frudsx[i],
            "hex": None if self.hex1[i] < 0 else [int(self.hex1[i]), int(self.hex2[i])],
            "soma": None if np.isnan(soma[0]) else [float(v) for v in soma],
            "n_outputs": int(self.row_ptr[i + 1] - self.row_ptr[i]),
            "n_inputs": int(self.col_ptr[i + 1] - self.col_ptr[i]),
            "n_pre_synapses": int(self.n_pre[i]), "n_post_synapses": int(self.n_post[i]),
        }

    # ------------------------------------------------------------------ edges
    def out_edges(self, idx: np.ndarray) -> np.ndarray:
        """Positions (into ``post_idx``/``n_syn``) of every outgoing connection of the given neurons."""
        idx = np.asarray(idx, dtype=np.int64)
        starts = self.row_ptr[idx]
        lengths = self.row_ptr[idx + 1] - starts
        total = int(lengths.sum())
        if total == 0:
            return np.zeros(0, dtype=np.int64)
        return np.repeat(starts - np.cumsum(lengths) + lengths, lengths) + np.arange(total)

    @property
    def pre_idx(self) -> np.ndarray:
        """Presynaptic neuron of every connection (int32, built on first use, 25 MB)."""
        if getattr(self, "_pre_idx", None) is None:
            self._pre_idx = np.repeat(np.arange(self.n, dtype=np.int32), np.diff(self.row_ptr))
        return self._pre_idx

    def _build_col(self):
        order = np.argsort(self.post_idx, kind="stable").astype(np.int32)
        col_ptr = np.zeros(self.n + 1, dtype=np.int64)
        np.cumsum(np.bincount(self.post_idx, minlength=self.n), out=col_ptr[1:])
        self._col = (col_ptr, order)

    @property
    def col_ptr(self) -> np.ndarray:
        """Like ``row_ptr`` but for *incoming* connections: inputs of neuron j are the edges
        ``in_edge_order[col_ptr[j]:col_ptr[j+1]]`` (positions into ``pre_idx`` / ``n_syn``)."""
        if self._col is None:
            self._build_col()
        return self._col[0]

    @property
    def in_edge_order(self) -> np.ndarray:
        if self._col is None:
            self._build_col()
        return self._col[1]

    def in_edges(self, idx: np.ndarray) -> np.ndarray:
        """Positions of every incoming connection of the given neurons."""
        idx = np.asarray(idx, dtype=np.int64)
        col_ptr, order = self.col_ptr, self.in_edge_order
        starts = col_ptr[idx]
        lengths = col_ptr[idx + 1] - starts
        total = int(lengths.sum())
        if total == 0:
            return np.zeros(0, dtype=np.int64)
        pos = np.repeat(starts - np.cumsum(lengths) + lengths, lengths) + np.arange(total)
        return order[pos].astype(np.int64)

    def _partner_summary(self, edges: np.ndarray, partners: np.ndarray, top: int, by_side: bool):
        if edges.size == 0:
            return []
        keys = self.type_idx[partners].astype(np.int64)
        if by_side:
            keys = keys * 8 + self.side_idx[partners]
        uniq, inv = np.unique(keys, return_inverse=True)
        syn = np.bincount(inv, weights=self.n_syn[edges].astype(np.float64), minlength=uniq.size)
        n_edges = np.bincount(inv, minlength=uniq.size)
        # number of distinct partner neurons per key
        pu = np.unique(keys * (self.n + 1) + partners)
        n_cells = np.bincount(np.searchsorted(uniq, pu // (self.n + 1)), minlength=uniq.size)
        order = np.argsort(-syn)[:top]
        out = []
        types = self.tables["types"]
        for k in order:
            key = int(uniq[k])
            t_idx, s_idx = (key // 8, key % 8) if by_side else (key, None)
            # sign of the partner type: from any neuron of it in this edge set
            first = partners[np.flatnonzero(inv == k)[0]]
            out.append({
                "type": types[t_idx] or "(unannotated)",
                "side": self.tables["sides"][s_idx] if s_idx is not None else None,
                "synapses": int(syn[k]), "connections": int(n_edges[k]), "neurons": int(n_cells[k]),
                "nt": self.nt[first], "sign": int(self.sign[first]),
            })
        return out

    def inputs_of(self, spec, top: int = 20, by_side: bool = False) -> list[dict]:
        """Strongest presynaptic cell types of a population, by total synapse count."""
        idx = self.select(spec)
        edges = self.in_edges(idx)
        return self._partner_summary(edges, self.pre_idx[edges], top, by_side)

    def outputs_of(self, spec, top: int = 20, by_side: bool = False) -> list[dict]:
        """Strongest postsynaptic cell types of a population, by total synapse count."""
        idx = self.select(spec)
        edges = self.out_edges(idx)
        return self._partner_summary(edges, self.post_idx[edges], top, by_side)

    def synapses_between(self, pre_spec, post_spec) -> int:
        """Total synapses from one population onto another."""
        pre, post = self.select(pre_spec), self.select(post_spec)
        if pre.size == 0 or post.size == 0:
            return 0
        edges = self.out_edges(pre)
        mask = np.zeros(self.n, dtype=bool)
        mask[post] = True
        return int(self.n_syn[edges][mask[self.post_idx[edges]]].sum())

    def edges_between(self, pre_spec, post_spec) -> np.ndarray:
        """Edge positions of every connection from one population onto another."""
        pre, post = self.select(pre_spec), self.select(post_spec)
        if pre.size == 0 or post.size == 0:
            return np.zeros(0, dtype=np.int64)
        edges = self.out_edges(pre)
        mask = np.zeros(self.n, dtype=bool)
        mask[post] = True
        return edges[mask[self.post_idx[edges]]]

    # ------------------------------------------------------------------ the cell-type graph
    def type_graph(self):
        """Collapse the neuron-level wiring into a (cell type, side) graph.

        Returns a :class:`TypeGraph` with nodes ``"DNa02/L"`` etc. Unannotated neurons are dropped.
        Edge weight = summed synapses between the two groups; each node also knows its total input
        synapses so weights can be read as *fraction of the target's input*.
        """
        if self._type_graph is None:
            self._type_graph = TypeGraph(self)
        return self._type_graph


class TypeGraph:
    """Cell-type-level graph (nodes are ``type/side``) built from a :class:`Connectome`."""

    def __init__(self, conn: Connectome):
        self.conn = conn
        n = conn.n
        side_code = np.where(conn.side == "L", 1, np.where(conn.side == "R", 2, np.where(conn.side == "M", 3, 0)))
        node_key = conn.type_idx.astype(np.int64) * 4 + side_code
        node_key[conn.type_idx == 0] = -1                    # unannotated
        uniq = np.unique(node_key[node_key >= 0])
        self.node_of_neuron = np.searchsorted(uniq, node_key)
        self.node_of_neuron[node_key < 0] = -1
        types = conn.tables["types"]
        sides = ["", "L", "R", "M"]
        self.names = [f"{types[k // 4]}/{sides[k % 4]}" if k % 4 else types[k // 4] for k in uniq.tolist()]
        self.index = {name: i for i, name in enumerate(self.names)}
        self.n_nodes = len(self.names)
        self.size = np.bincount(self.node_of_neuron[self.node_of_neuron >= 0], minlength=self.n_nodes)
        # sign and nt per node: majority of its neurons
        sgn = np.bincount(self.node_of_neuron[self.node_of_neuron >= 0],
                          weights=conn.sign[self.node_of_neuron >= 0], minlength=self.n_nodes)
        self.sign = np.where(sgn >= 0, 1.0, -1.0).astype(np.float32)
        first = np.full(self.n_nodes, -1, dtype=np.int64)
        seen = self.node_of_neuron >= 0
        first[self.node_of_neuron[seen][::-1]] = np.flatnonzero(seen)[::-1]
        self.nt = conn.nt[first]
        self.superclass = conn.superclass[first]
        # edges
        pre_node = self.node_of_neuron[conn.pre_idx]
        post_node = self.node_of_neuron[conn.post_idx]
        ok = (pre_node >= 0) & (post_node >= 0)
        key = pre_node[ok].astype(np.int64) * self.n_nodes + post_node[ok]
        ukey, inv = np.unique(key, return_inverse=True)
        w = np.bincount(inv, weights=conn.n_syn[ok].astype(np.float64), minlength=ukey.size)
        self.src = (ukey // self.n_nodes).astype(np.int64)
        self.dst = (ukey % self.n_nodes).astype(np.int64)
        self.weight = w.astype(np.float32)
        order = np.argsort(self.src, kind="stable")
        self.src, self.dst, self.weight = self.src[order], self.dst[order], self.weight[order]
        self.row_ptr = np.zeros(self.n_nodes + 1, dtype=np.int64)
        np.cumsum(np.bincount(self.src, minlength=self.n_nodes), out=self.row_ptr[1:])
        self.in_total = np.bincount(self.dst, weights=self.weight, minlength=self.n_nodes)
        self.out_total = np.bincount(self.src, weights=self.weight, minlength=self.n_nodes)
        # incoming index
        iorder = np.argsort(self.dst, kind="stable")
        self.in_order = iorder
        self.col_ptr = np.zeros(self.n_nodes + 1, dtype=np.int64)
        np.cumsum(np.bincount(self.dst, minlength=self.n_nodes), out=self.col_ptr[1:])

    def nodes_of(self, spec) -> np.ndarray:
        """Node ids covering a population spec (any node with at least one selected neuron)."""
        idx = self.conn.select(spec)
        nodes = self.node_of_neuron[idx]
        return np.unique(nodes[nodes >= 0])

    def out(self, node: int):
        a, b = self.row_ptr[node], self.row_ptr[node + 1]
        return self.dst[a:b], self.weight[a:b]

    def inp(self, node: int):
        a, b = self.col_ptr[node], self.col_ptr[node + 1]
        e = self.in_order[a:b]
        return self.src[e], self.weight[e]


def load_connectome(path: Path | str | None = None, quiet: bool = False, female: bool = False) -> Connectome:
    """Download (first time only) and load the MaleCNS v1.0 connectome, or with ``female=True`` the female
    fly's FlyWire 783 connectome (built on first use, see :mod:`virtual_fly.flywire`).

    Only the default male file is downloaded and checksummed; a path given explicitly or through the
    ``FLY_DATA_FILE`` environment variable is loaded as it is."""
    if female:
        if path is not None:
            raise ValueError("load_connectome(): give a path or female=True, not both")
        from .flywire import ensure_female
        path = ensure_female(quiet=quiet)
    path = Path(DATA_FILE if path is None else path)
    if path == DEFAULT_DATA_FILE:
        download_connectome(path, quiet=quiet)
    t0 = time.time()
    conn = Connectome(path)
    if not quiet:
        print(f"Loaded {conn.dataset} ({conn.sex}): {conn.n:,} neurons, {conn.n_edges:,} connections "
              f"({int(conn.n_syn.sum()):,} synapses) in {time.time() - t0:.1f}s", file=sys.stderr)
    return conn
