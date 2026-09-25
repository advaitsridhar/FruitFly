"""
The female fly: FlyWire's whole-brain connectome (release 783) in the kit's FLYB format.

The published model this kit builds on (Shiu et al. 2024) was run on FlyWire, the connectome of an adult
female brain (Dorkenwald et al. 2024). The kit's default brain is the male MaleCNS; this module builds the
female one on first use from two public files, downloaded once, pinned to one commit each and checked:

* the connectivity table the published model uses, ``Connectivity_783.parquet`` in
  github.com/philshiu/Drosophila_brain_model (one row per connected pair, with its synapse count);
* FlyWire's neuron annotations, ``Supplemental_file1_neuron_annotations.tsv`` in
  github.com/flyconnectome/flywire_annotations (Schlegel et al. 2024; from version 3 the cell-type names are
  matched to the MaleCNS, Berg et al. 2025): class, cell type, side, predicted transmitter, fruitless and
  doublesex expression, sex dimorphism, soma position.

The data files are not redistributed here: they come from their sources and the female FLYB file is built on
your machine (about 130 MB to download once; the result is about 12 MB). Reading the parquet file needs
``pyarrow`` (``pip install pyarrow``). As for the male file, connections of fewer than 5 synapses are left out.

FlyWire is a brain without the ventral nerve cord, so the leg, wing and neck motor neurons of the male data do
not exist in it, and the experiments that read them are skipped. There are no medulla column coordinates in the
annotations, so the computed column-by-column motion vision is off for the female fly.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import struct
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

from .connectome import PROJECT_DIR

FEMALE_FILE = PROJECT_DIR / "data" / "flywire-v783.flyb.gz"
SOURCE_DIR = PROJECT_DIR / "data" / "flywire-src"
DATASET = "flywire:v783"
MIN_SYNAPSES = 5

SOURCES = {
    "connectivity": {
        "url": ("https://raw.githubusercontent.com/philshiu/Drosophila_brain_model/"
                "91bdd1e7dcf193f3e7ca5a8933497fcef63b7960/Connectivity_783.parquet"),
        "file": "Connectivity_783.parquet", "mb": 101,
        "sha256": "efeb23fb99098e9c390f6869969b2a121a2ee92c833cfc45ecb2c1d8e1af0347"},
    "annotations": {
        "url": ("https://raw.githubusercontent.com/flyconnectome/flywire_annotations/"
                "8587524c1748ce5ef2080822a2fc890fc03bf597/supplemental_files/Supplemental_file1_neuron_annotations.tsv"),
        "file": "Supplemental_file1_neuron_annotations.tsv", "mb": 32,
        "sha256": "9a4f8b2f843196074431ebd7cd883536afa1be86c8a4ce90970441e8be81d1be"},
}

# FlyWire's super classes in the MaleCNS vocabulary the rest of the kit uses
SUPERCLASS = {"optic": "ol_intrinsic", "central": "cb_intrinsic", "visual_projection": "visual_projection",
              "visual_centrifugal": "visual_centrifugal", "ascending": "ascending_neuron",
              "descending": "descending_neuron", "sensory_ascending": "sensory_ascending", "motor": "cb_motor",
              "endocrine": "cb_endocrine", "sensory": "cb_sensory"}
SIDE = {"left": "L", "right": "R", "center": "M"}
NT_SIGN = {"acetylcholine": 1, "glutamate": -1, "gaba": -1, "histamine": -1, "dopamine": 1, "octopamine": 1,
           "serotonin": 1, "unclear": 1, "": 1}                        # as in the male file
VOXEL_NM = (4.0, 4.0, 40.0)                                            # FlyWire's annotation voxel size


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_sources(src_dir: Path | str = SOURCE_DIR, quiet: bool = False) -> dict[str, Path]:
    """Fetch the two source files (once) and check them."""
    src_dir = Path(src_dir)
    src_dir.mkdir(parents=True, exist_ok=True)
    out = {}
    for key, s in SOURCES.items():
        path = src_dir / s["file"]
        if not path.exists() or _sha256(path) != s["sha256"]:
            if not quiet:
                print(f"Downloading the female fly's {key} ({s['mb']} MB) from\n  {s['url']}", file=sys.stderr)
            tmp = path.with_suffix(path.suffix + ".part")
            try:
                with urllib.request.urlopen(s["url"], timeout=120) as r, open(tmp, "wb") as f:
                    while chunk := r.read(1 << 20):
                        f.write(chunk)
            except OSError as e:
                raise SystemExit(f"\nCouldn't download {s['url']}: {e}\nYou can also download it in a browser "
                                 f"and put it at {path}")
            if _sha256(tmp) != s["sha256"]:
                tmp.unlink(missing_ok=True)
                raise SystemExit(f"The downloaded {key} file is damaged (its checksum doesn't match). Please try again.")
            tmp.replace(path)
        out[key] = path
    return out


def read_annotations(path: Path | str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def read_connectivity(path: Path | str):
    """(pre root ids, post root ids, synapse counts) from the published model's parquet file."""
    try:
        import pyarrow.parquet as pq
    except ImportError:
        raise SystemExit("The female fly needs pyarrow to read FlyWire's connectivity file: pip install pyarrow")
    t = pq.read_table(path, columns=["Presynaptic_ID", "Postsynaptic_ID", "Connectivity"])
    return (t.column("Presynaptic_ID").to_numpy(), t.column("Postsynaptic_ID").to_numpy(),
            t.column("Connectivity").to_numpy())


def neuron_rows(annotations: list[dict], extra_ids=()) -> list[dict]:
    """One row per neuron in the kit's vocabulary; ``extra_ids`` are connected neurons the annotations lack."""
    rows = []
    for a in annotations:
        nt = a.get("top_nt") or "unclear"
        sc = SUPERCLASS.get(a.get("super_class", ""), "")
        if sc == "cb_sensory" and a.get("cell_class") == "visual":
            sc = "ol_sensory"                                            # photoreceptors and ocelli
        soma = [float(a[k]) * v if a.get(k) not in (None, "", "nan") else np.nan
                for k, v in zip(("soma_x", "soma_y", "soma_z"), VOXEL_NM)]
        rows.append({"root": int(a["root_id"]), "type": a.get("cell_type") or a.get("hemibrain_type") or "",
                     "superclass": sc, "cls": a.get("cell_class", ""), "subclass": a.get("cell_sub_class", ""),
                     "nt": nt, "side": SIDE.get(a.get("side", ""), ""),
                     "dimorphism": "" if a.get("dimorphism") in (None, "", "isomorphic") else a["dimorphism"],
                     "frudsx": a.get("fru_dsx", ""), "neuromere": "", "nerve": a.get("nerve", ""), "soma": soma})
    for r in extra_ids:
        rows.append({"root": int(r), "type": "", "superclass": "", "cls": "", "subclass": "", "nt": "unclear",
                     "side": "", "dimorphism": "", "frudsx": "", "neuromere": "", "nerve": "",
                     "soma": [np.nan] * 3})
    return rows


def write_flyb(path: Path | str, rows: list[dict], pre_root, post_root, n_syn, meta: dict) -> Path:
    """Write neurons and connections in the kit's FLYB layout (see :class:`virtual_fly.connectome.Connectome`)."""
    n = len(rows)
    pos = {r["root"]: i for i, r in enumerate(rows)}
    pre = np.fromiter((pos[int(x)] for x in pre_root), dtype=np.int64, count=len(pre_root))
    post = np.fromiter((pos[int(x)] for x in post_root), dtype=np.int64, count=len(post_root))
    order = np.lexsort((post, pre))
    pre, post, syn = pre[order], post[order], np.minimum(np.asarray(n_syn)[order], 65535).astype("<u2")
    row_ptr = np.zeros(n + 1, dtype="<i4")
    np.add.at(row_ptr, pre + 1, 1)
    row_ptr = np.cumsum(row_ptr).astype("<i4")
    n_pre = np.bincount(pre, weights=syn, minlength=n).astype("<i4")
    n_post = np.bincount(post, weights=syn, minlength=n).astype("<i4")
    tables = {}
    for key in ("type", "superclass", "cls", "subclass", "nt", "side", "dimorphism", "frudsx", "neuromere", "nerve"):
        tables[key] = [""] + sorted({r[key] for r in rows} - {""})
    for key, width in (("superclass", 255), ("cls", 255), ("nt", 255), ("side", 255), ("dimorphism", 255),
                       ("frudsx", 255), ("neuromere", 255), ("nerve", 255), ("subclass", 65535)):
        if len(tables[key]) > width:
            raise ValueError(f"too many distinct {key} labels ({len(tables[key])}) for the FLYB format")
    index = {k: {v: i for i, v in enumerate(t)} for k, t in tables.items()}

    def col(key, dtype):
        return np.asarray([index[key][r[key]] for r in rows], dtype=dtype)

    def s16(s: str) -> bytes:
        b = s.encode("utf-8")
        return struct.pack("<H", len(b)) + b

    body = json.dumps(meta).encode()
    parts = [b"FLYB", struct.pack("<IIII", 1, n, len(post), 0), s16(DATASET), struct.pack("<I", len(body)) + body]
    for key in ("type", "superclass", "cls", "subclass", "nt", "side", "dimorphism", "frudsx", "neuromere", "nerve"):
        parts.append(struct.pack("<H", len(tables[key])) + b"".join(s16(t) for t in tables[key]))
    sign = np.asarray([NT_SIGN.get(r["nt"], 1) for r in rows], dtype="i1")
    arrays = [np.asarray([r["root"] for r in rows], dtype="<i8"), col("type", "<i4"), col("superclass", "u1"),
              col("cls", "u1"), col("subclass", "<u2"), col("nt", "u1"), sign, col("side", "u1"),
              np.full(n, -1, dtype="i1"), np.full(n, -1, dtype="i1"),          # no medulla column coordinates
              col("dimorphism", "u1"), col("frudsx", "u1"), col("neuromere", "u1"), col("nerve", "u1"),
              np.asarray([r["soma"] for r in rows], dtype="<f4").reshape(-1), n_pre, n_post, row_ptr,
              post.astype("<i4"), syn]
    parts.extend(a.tobytes() for a in arrays)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    with gzip.open(tmp, "wb", compresslevel=6) as f:
        f.write(b"".join(parts))
    tmp.replace(path)
    return path


def build_female(out: Path | str = FEMALE_FILE, src_dir: Path | str = SOURCE_DIR, quiet: bool = False,
                 min_synapses: int = MIN_SYNAPSES) -> Path:
    """Download the sources (once) and write the female FLYB file."""
    t0 = time.time()
    src = download_sources(src_dir, quiet=quiet)
    if not quiet:
        print("Building the female fly's connectome (FlyWire 783)...", file=sys.stderr)
    ann = read_annotations(src["annotations"])
    pre, post, syn = read_connectivity(src["connectivity"])
    keep = syn >= min_synapses
    pre, post, syn = pre[keep], post[keep], syn[keep]
    known = {int(a["root_id"]) for a in ann}
    extra = sorted({int(x) for x in np.unique(np.concatenate([pre, post]))} - known)
    rows = neuron_rows(ann, extra)
    meta = {"dataset": DATASET, "sex": "female", "built": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "min_weight": min_synapses, "nt_signs": NT_SIGN, "neurons": len(rows), "edges": int(pre.size),
            "synapses_in_edges": int(syn.sum()), "unannotated_connected_neurons": len(extra),
            "sources": {k: {"url": s["url"], "sha256": s["sha256"]} for k, s in SOURCES.items()},
            "credits": ("FlyWire connectome v783 (Dorkenwald et al. 2024, Nature); annotations Schlegel et al. 2024 "
                        "and Berg et al. 2025; connectivity table from Shiu et al. 2024 (Drosophila_brain_model, MIT)")}
    write_flyb(out, rows, pre, post, syn, meta)
    if not quiet:
        print(f"Built {out} ({len(rows):,} neurons, {pre.size:,} connections) in {time.time() - t0:.0f} s",
              file=sys.stderr)
    return Path(out)


def ensure_female(quiet: bool = False) -> Path:
    """The female FLYB file, built on first use."""
    return FEMALE_FILE if FEMALE_FILE.exists() else build_female(quiet=quiet)


if __name__ == "__main__":
    build_female()
