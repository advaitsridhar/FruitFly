"""
BANC: one adult female's brain AND ventral nerve cord as a third connectome (the two-flies work, docs/TWO_FLIES_PLAN.md
section 9; Bates et al. 2026, Nature, doi:10.1038/s41586-026-10735-w; the data deposit on Harvard Dataverse,
doi:10.7910/DVN/7WTH1N, CC BY 4.0). Modelled on :mod:`virtual_fly.flywire`, which builds the FlyWire female: the deposit's
metadata table, edge list and transmitter predictions are fetched once (pinned by id, size, MD5 and SHA-256, never
committed), and written as a FLYB file the rest of the kit reads like the male's.

What this fly has that FlyWire's female lacks: a whole ventral nerve cord (leg taste cells, so the male's touch reaches a
wired target in her; leg motor neurons; the abdominal ganglion). What it lacks (Bates et al. 2026): the lamina and ocelli
are not in the sample; both antennal nerves were damaged (hearing is under-represented); no fruitless or doublesex
columns; no medulla column coordinates (columnar vision stays off); fewer synaptic links with an identified neuron on
both sides (18 % against FlyWire's 42 %), so fewer synapses per cell.

Every mapping from BANC's vocabulary to the kit's is a recorded choice: the superclass table below, and the class,
subclass and nerve maps built FROM THE DATA by majority vote over the neurons BANC itself matches to the male file
(``malecns_match``), stored in the file's meta with their vote shares. Tyramine, which BANC predicts and the kit never
met, keeps its label with the sign +1 as a documented modelling choice (TYRAMINE_SIGN, a switch; decision 31). The
male's fruitless/doublesex labels are NOT copied across (decision 32): ``gene:`` selectors select nothing on her.
"""
from __future__ import annotations

import collections
import csv
import hashlib
import json
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

from . import __version__
from .connectome import DATA_DIR, data_folder

BANC_FILE = DATA_DIR / "banc-v888.flyb.gz"
BUILD = 2                             # bump when the builder changes what goes in the file: older files are rebuilt
SOURCE_DIR = DATA_DIR / "banc-src"
DATASET = "banc:v888"
MIN_SYNAPSES = 1                      # every connection, as the kit uses the other two files
DEFAULT_GAIN = 1.0                    # shipped first (plan 9.7); the sweep's choice replaces it, recorded in SCIENCE.md
TYRAMINE_SIGN = 1                     # decision 31: +1 as a modelling choice; 0 or -1 are the switch's other settings

# The deposit's files, pinned by tools/pin_banc.py on 2026-10-04 from the Dataverse listing of version 3.0 (the
# "Publication version", released 2026-07-01): id, size and MD5 as Dataverse lists them, SHA-256 as measured after the
# download. The v3 edge list and the transmitter table are byte-identical in every published version (1.0 to 3.0);
# only the metadata table was revised (decision 28, docs/TWO_FLIES_PROGRESS.md).
SOURCES = {
    "meta": {"url": "https://dataverse.harvard.edu/api/access/datafile/14033740", "file": "banc_888_meta.feather", "id": 14033740,
             "version": "3.0", "mb": 58, "bytes": 57550610, "md5": "6275eda42f98c49539d1ab513d979d09",
             "sha256": "819bbcff476e52702d6f8d8604ce1f12d1d7b11942281df2f49df2a73a6f15a5"},
    "edges_v3": {"url": "https://dataverse.harvard.edu/api/access/datafile/13918810", "file": "banc_888_edgelist_simple_v3.feather",
                 "id": 13918810, "version": "3.0", "mb": 359, "bytes": 359161658, "md5": "08542b0771db7418ed474be60dc9886c",
                 "sha256": "8c296e946f3c69a8c7222f30ad75fa8a98eeb189124fec6df829c9125f4be64b"},
    "transmitters": {"url": "https://dataverse.harvard.edu/api/access/datafile/13916450",
                     "file": "banc_888_neurotransmitter_prediction_v2.csv", "id": 13916450, "version": "3.0", "mb": 21,
                     "bytes": 21107592, "md5": "4ebbd1d6e05d4192ad0c6db27739a8e3",
                     "sha256": "bb0f4afa48a05d90008c6d801e053ff2667fb1ba68e29501c92f49514540da1d"},
}
AGENT = f"virtual-fly/{__version__}"

# BANC's super classes in the MaleCNS vocabulary the kit uses (plan 9.4, each row a recorded choice); sensory and
# motor split by region, the visceral/circulatory classes by region too (the rule is in the meta)
SUPERCLASS = {
    "optic_lobe_intrinsic": "ol_intrinsic", "central_brain_intrinsic": "cb_intrinsic",
    "ventral_nerve_cord_intrinsic": "vnc_intrinsic", "ascending": "ascending_neuron", "descending": "descending_neuron",
    "sensory_ascending": "sensory_ascending", "sensory_descending": "sensory_descending",
    "visual_projection": "visual_projection", "visual_centrifugal": "visual_centrifugal",
    "ascending_visceral_circulatory": "efferent_ascending",
}
BY_REGION = {"sensory": {"optic_lobe": "ol_sensory", "central_brain": "cb_sensory", "ventral_nerve_cord": "vnc_sensory"},
             "motor": {"optic_lobe": "cb_motor", "central_brain": "cb_motor", "ventral_nerve_cord": "vnc_motor"},
             "visceral_circulatory": {"optic_lobe": "cb_endocrine", "central_brain": "cb_endocrine", "ventral_nerve_cord": "vnc_endocrine"}}
SIDE = {"left": "L", "right": "R"}
DIMORPHISM = {"dimorphic": "sexually dimorphic", "female-specific": "female-specific", "male-specific": "male-specific"}
# as in the male and FlyWire files, plus tyramine (decision 31)
NT_SIGN = {"acetylcholine": 1, "glutamate": -1, "gaba": -1, "histamine": -1, "dopamine": 1, "octopamine": 1,
           "serotonin": 1, "tyramine": TYRAMINE_SIGN, "unclear": 1, "": 1}
NT_CONF_FALLBACK = 0.5                # a prediction scored below this is labelled "unclear" (the sign stays the prediction's)
KNOWN_NTS = ("acetylcholine", "gaba", "glutamate", "histamine", "dopamine", "octopamine", "serotonin")
DROP_STATUS = ("GLIA", "NOT_A_NEURON", "TRACHEA")

# Names the kit uses (MaleCNS cell types) -> the same cells in BANC, by hand, with where the match comes from. Most
# names need no entry (BANC spells them as the MaleCNS does, and the builder adds data-derived aliases for the rest:
# cells whose ``malecns_cell_type`` is the name); these are the ones the plan lists (9.5), checked by tools/alias_audit.py.
ALIASES = {
    "GNG232": ("CB0616", "G2N-1, FBbt_00051850 (the sugar relay), spelled CB0616 in BANC's FAFB-style names"),
    "GNG087": ("CB0219", "FBbt_20004033 (the bitter relay), spelled CB0219 in BANC's FAFB-style names"),
    "prefix:pC1_": ("prefix:pC1", "the doublesex pC1 cluster: pC1a-e in the female (the male's pC1_ types include P1)"),
    "R1-R6": ("R1-6", "the outer photoreceptors, FlyWire's spelling (not in the BANC sample: the lamina is missing)"),
    "prefix:R1-R6": ("R1-6", "the same, for the parts list's graded cells"),
    "VS": ("regex:^VS[0-9]+$", "the VS cells, typed VS1-VS8 in FAFB-style names"),
    "prefix:KCa'b'": ("prefix:KCa'b',prefix:KCapbp", "the alpha'/beta' Kenyon cells, spelled KCapbp-* in FAFB-style names"),
    # the audit's findings (tools/alias_audit.py, 2026-10-04): what BANC calls the kit's remaining names
    "AN19A018": ("prefix:AN19A018", "the brake neuron: BANC splits it into AN19A018_a to _d (12 cells by their MANC name)"),
    "regex:^DLMn": ("DLM1-4,DLM5", "the dorsal longitudinal flight muscle motor neurons, typed DLM1-4 and DLM5 in BANC"),
    "regex:^hg": ("iv1,iv2,iv3,iv4", "the hg1-hg4 wing motor neurons of MANC's names, typed iv1-iv4 in BANC"),
    "subclass:wind_gravity": ("JO-C,JO-E", "the wind and gravity Johnston's-organ cells: BANC's sub classes name the JO groups, "
                                            "its types JO-C (wind) and JO-E (gravity) carry them"),
}
# the other name columns a kit name may hide under, in the order tried, for the data-derived aliases
NAME_COLUMNS = ("malecns_cell_type", "manc_cell_type", "fafb_cell_type")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download_sources(src_dir: Path | str = SOURCE_DIR, quiet: bool = False, keys=("meta", "edges_v3")) -> dict[str, Path]:
    """Fetch the deposit's files (once) and check them: size, then SHA-256 (the MD5 Dataverse lists is in SOURCES too)."""
    src_dir = data_folder(src_dir)
    out = {}
    for key in keys:
        s = SOURCES[key]
        path = src_dir / s["file"]
        if not path.exists() or path.stat().st_size != s["bytes"] or _sha256(path) != s["sha256"]:
            if not quiet:
                print(f"Downloading BANC's {key} ({s['mb']} MB) from\n  {s['url']}", file=sys.stderr)
            tmp = path.with_suffix(path.suffix + ".part")
            try:
                req = urllib.request.Request(s["url"], headers={"User-Agent": AGENT})
                with urllib.request.urlopen(req, timeout=300) as r, open(tmp, "wb") as f:
                    while chunk := r.read(1 << 20):
                        f.write(chunk)
            except KeyboardInterrupt:
                tmp.unlink(missing_ok=True)
                raise
            except OSError as e:
                tmp.unlink(missing_ok=True)
                raise SystemExit(f"\nCouldn't download {s['url']}: {e}\nYou can also download it in a browser (the Dataverse "
                                 f"deposit doi:10.7910/DVN/7WTH1N, compiled_data/{s['file']}) and put it at {path}")
            if tmp.stat().st_size != s["bytes"] or _sha256(tmp) != s["sha256"]:
                tmp.unlink(missing_ok=True)
                raise SystemExit(f"The downloaded {key} file is damaged (its size or checksum doesn't match). Please try again.")
            tmp.replace(path)
        out[key] = path
    return out


def _feather():
    try:
        import pyarrow.feather as feather
    except ImportError:
        raise SystemExit("BANC's files are Arrow/Feather tables: pip install pyarrow (the kit's `female` extra has it)") from None
    return feather


META_COLUMNS = ("root_id", "proofread", "roughly_proofread", "status", "side", "region", "nerve", "neuromere", "super_class",
                "cell_class", "cell_sub_class", "cell_type", "fafb_cell_type", "manc_cell_type", "malecns_cell_type",
                "malecns_match", "sexually_dimorphic", "neurotransmitter_predicted", "neurotransmitter_score",
                "neurotransmitter_verified", "root_position_nm")


def read_meta(path: Path | str) -> list[dict]:
    """The metadata table's rows (the columns the builder uses), None for a missing value."""
    t = _feather().read_table(str(path), columns=[c for c in META_COLUMNS])
    cols = {c: t.column(c).to_pylist() for c in t.column_names}
    return [{c: cols[c][i] for c in cols} for i in range(t.num_rows)]


def read_edges(path: Path | str):
    """(pre, post, count) of the edge list as int64 root ids and int32 synapse counts."""
    t = _feather().read_table(str(path), columns=["pre", "post", "count"])
    pre = np.asarray(t.column("pre").to_numpy(zero_copy_only=False)).astype(np.int64)
    post = np.asarray(t.column("post").to_numpy(zero_copy_only=False)).astype(np.int64)
    return pre, post, np.asarray(t.column("count").to_numpy()).astype(np.int32)


def _true(v) -> bool:
    return str(v).strip().upper() == "TRUE"


def select_neurons(meta: list[dict]) -> tuple[list[dict], dict]:
    """Decision 30: proofread or roughly proofread neurons; no glia, trachea or non-neurons; one row per root id (the
    proofread one preferred). Returns the rows kept and every count dropped."""
    counts = collections.Counter()
    best: dict[int, dict] = {}
    for a in meta:
        counts["rows"] += 1
        status = (a.get("status") or "").upper()
        if a.get("super_class") == "glia" or any(k in status for k in DROP_STATUS):
            counts["glia_trachea_or_not_a_neuron"] += 1
            continue
        if not (_true(a.get("proofread")) or _true(a.get("roughly_proofread"))):
            counts["not_proofread"] += 1
            continue
        rid = int(a["root_id"])
        prev = best.get(rid)
        if prev is None:
            best[rid] = a
        else:
            counts["duplicate_root_ids"] += 1
            if _true(a.get("proofread")) and not _true(prev.get("proofread")):
                best[rid] = a
    counts["kept"] = len(best)
    return list(best.values()), dict(counts)


def _soma(a: dict):
    text = a.get("root_position_nm")
    if not text:
        return [np.nan] * 3
    try:
        x, y, z = (float(v) for v in str(text).split(","))
    except ValueError:
        return [np.nan] * 3
    return [x, y, z]


def _superclass(a: dict) -> str:
    sc, region = a.get("super_class") or "", a.get("region") or ""
    if sc in BY_REGION:
        return BY_REGION[sc].get(region, BY_REGION[sc]["central_brain"])
    return SUPERCLASS.get(sc, "")


def vote_maps(rows: list[dict], male) -> dict:
    """The class, subclass and nerve maps from the data: for every BANC value, the kit's value its ``malecns_match``
    neurons carry in the male file, by majority vote, with the share and the number of votes (plan 9.4)."""
    index = {int(b): i for i, b in enumerate(np.asarray(male.body_id).tolist())}
    votes = {"cls": collections.defaultdict(collections.Counter), "subclass": collections.defaultdict(collections.Counter),
             "nerve": collections.defaultdict(collections.Counter)}
    for a in rows:
        m = a.get("malecns_match")
        if not m:
            continue
        try:
            i = index.get(int(m))
        except ValueError:
            i = None
        if i is None:
            continue
        for key, banc_col, male_col in (("cls", "cell_class", male.cls), ("subclass", "cell_sub_class", male.subclass),
                                       ("nerve", "nerve", male.nerve)):
            v = a.get(banc_col)
            if v:
                votes[key][v][str(male_col[i])] += 1
    maps = {}
    for key, table in votes.items():
        maps[key] = {}
        for v, c in table.items():
            (winner, n), total = c.most_common(1)[0], sum(c.values())
            maps[key][v] = {"to": winner, "share": round(n / total, 3), "votes": total}
    return maps


def _apply_map(maps: dict, key: str, value, fallback: str = "") -> str:
    if not value:
        return ""
    entry = maps.get(key, {}).get(value)
    return entry["to"] if entry and entry["to"] else fallback


def neuron_rows(rows: list[dict], maps: dict) -> list[dict]:
    """One row per neuron in the kit's vocabulary (as flywire.neuron_rows makes them)."""
    out = []
    for a in rows:
        nt = a.get("neurotransmitter_predicted") or "unclear"
        score = a.get("neurotransmitter_score")
        try:
            label = nt if score is None or float(score) >= NT_CONF_FALLBACK else "unclear"
        except (TypeError, ValueError):
            label = nt
        name = a.get("cell_type") or a.get("fafb_cell_type") or a.get("manc_cell_type") or a.get("malecns_cell_type") or ""
        out.append({"root": int(a["root_id"]), "type": name, "superclass": _superclass(a),
                    "cls": _apply_map(maps, "cls", a.get("cell_class"), fallback=a.get("cell_class") or ""),
                    "subclass": _apply_map(maps, "subclass", a.get("cell_sub_class"), fallback=a.get("cell_sub_class") or ""),
                    "nt": label, "sign_nt": nt, "side": SIDE.get(a.get("side") or "", ""),
                    "dimorphism": DIMORPHISM.get(a.get("sexually_dimorphic") or "", ""), "frudsx": "",
                    "neuromere": a.get("neuromere") or "", "nerve": _apply_map(maps, "nerve", a.get("nerve")),
                    "soma": _soma(a)})
    return out


def known_transmitters(rows: list[dict]) -> dict[str, dict]:
    """Per cell type, the transmitters BANC's ``neurotransmitter_verified`` column gives, by the half-of-the-type rule of
    flywire.known_transmitters; nitric oxide, glycine, peptides and tyramine left out."""
    per: dict[str, list[set[str]]] = collections.defaultdict(list)
    for a in rows:
        t = a.get("cell_type") or a.get("fafb_cell_type") or a.get("manc_cell_type") or a.get("malecns_cell_type") or ""
        if not t:
            continue
        got = {x.strip() for x in (a.get("neurotransmitter_verified") or "").split(",") if x.strip() in KNOWN_NTS}
        per[t].append(got)
    out = {}
    for t, sets in per.items():
        nts = [x for x in KNOWN_NTS if sum(x in S for S in sets) * 2 >= len(sets)]
        if nts:
            out[t] = {"nt": nts, "source": "BANC neurotransmitter_verified (Bates et al. 2026)"}
    return out


def data_aliases(rows: list[dict], names) -> dict[str, str]:
    """For every kit name that is not a BANC cell type (a plain name, or ``regex:^...``), the BANC cells whose MaleCNS,
    MANC or FAFB name (``NAME_COLUMNS``, the union over the three) is that name, as a ``body:`` population spec (plan 9.5;
    ``auto:`` proposals are left out and counted by the audit). A regex name takes the cells whose other name matches it."""
    import re
    types = {r.get("cell_type") for r in rows}
    found = {}
    for n in names:
        if n in types or n.startswith(("prefix:", "class:", "superclass:", "subclass:", "nt:", "nerve:", "neuromere:", "gene:", "body:")):
            continue
        pattern = re.compile(n[len("regex:"):]) if n.startswith("regex:") else None
        ids = []
        for col in NAME_COLUMNS:
            for a in rows:
                v = a.get(col)
                if not v or str(v).startswith("auto:"):
                    continue
                if pattern is not None:
                    if any(pattern.match(part.strip()) for part in str(v).split(",")):
                        ids.append(int(a["root_id"]))
                elif str(v) == n or n in [part.strip() for part in str(v).split(",")]:
                    ids.append(int(a["root_id"]))
        if ids:
            found[n] = ",".join(f"body:{r}" for r in sorted(set(ids)))
    return found


def build_banc(out: Path | str = BANC_FILE, src_dir: Path | str = SOURCE_DIR, quiet: bool = False,
               min_synapses: int = MIN_SYNAPSES, keep_connected: bool = False, alias_names=None) -> Path:
    """Download the sources (once) and write the BANC FLYB file."""
    from .connectome import load_connectome
    from .flywire import write_flyb
    t0 = time.time()
    _feather()                                      # fail before downloading 0.4 GB, not after
    src = download_sources(src_dir, quiet=quiet)
    if not quiet:
        print("Building the female fly with a nerve cord (BANC v888)...", file=sys.stderr)
    meta = read_meta(src["meta"])
    kept, counts = select_neurons(meta)
    pre, post, syn = read_edges(src["edges_v3"])
    counts["edges"] = int(pre.size)
    keep = syn >= min_synapses
    counts["edges_below_min_synapses"] = int((~keep).sum())
    pre, post, syn = pre[keep], post[keep], syn[keep]
    ids = np.array(sorted(int(r["root_id"]) for r in kept), dtype=np.int64)
    if keep_connected:                              # the switch: every connected id, annotated or not
        extra = sorted(set(np.unique(np.concatenate([pre, post])).tolist()) - set(ids.tolist()))
        counts["unannotated_connected_neurons"] = len(extra)
        for r in extra:
            kept.append({"root_id": str(r)})
        ids = np.array(sorted(int(r["root_id"]) for r in kept), dtype=np.int64)
    known = np.isin(pre, ids) & np.isin(post, ids)
    counts["edges_to_dropped_neurons"] = int((~known).sum())
    counts["synapses_in_dropped_edges"] = int(syn[~known].sum())
    table_ids = np.array(sorted({int(a["root_id"]) for a in meta}), dtype=np.int64)
    absent = ~np.isin(pre, table_ids) | ~np.isin(post, table_ids)
    counts["edges_with_an_id_absent_from_the_table"] = int(absent.sum())
    counts["connected_ids_absent_from_the_table"] = int(np.unique(np.concatenate([pre[~np.isin(pre, table_ids)],
                                                                                   post[~np.isin(post, table_ids)]])).size)
    # the input every kept neuron receives, with the shipped rule (from kept neurons only) and with every connected id
    # kept (FlyWire's own rule, decision 30's switch): the medians the gain prior is read from (plan 9.7)
    cb = np.array(sorted(int(a["root_id"]) for a in kept if a.get("super_class") == "central_brain_intrinsic"), dtype=np.int64)
    def medians(mask):
        tot = np.bincount(np.searchsorted(ids, post[mask]), weights=syn[mask], minlength=ids.size)
        has = tot > 0
        cbm = np.isin(ids, cb) & has
        return {"all": int(np.median(tot[has])) if has.any() else None, "cb_intrinsic": int(np.median(tot[cbm])) if cbm.any() else None}
    post_kept = np.isin(post, ids)
    counts["median_input_synapses"] = {"kept_pre_only": medians(post_kept & np.isin(pre, ids)), "any_pre": medians(post_kept)}
    pre, post, syn = pre[known], post[known], syn[known]
    self_loops = pre == post
    counts["self_connections"] = int(self_loops.sum())
    pre, post, syn = pre[~self_loops], post[~self_loops], syn[~self_loops]
    male = load_connectome(quiet=True)
    maps = vote_maps(kept, male)
    rows = neuron_rows(kept, maps)
    names = list(alias_names) if alias_names is not None else []
    if alias_names is None:
        try:
            from .specs import kit_alias_candidates
            names = kit_alias_candidates()
        except Exception:                           # the audit's collector is a convenience, not a requirement
            names = []
    aliases = {k: v for k, (v, _) in ALIASES.items()}
    derived = data_aliases(kept, names)
    aliases.update({k: v for k, v in derived.items() if k not in aliases})
    counts["kept_without_super_class_by_region"] = dict(collections.Counter(
        (a.get("region") or "none") for a in kept if not a.get("super_class")))
    somas = np.array([r["soma"] for r in rows], dtype=float)
    ok = ~np.isnan(somas[:, 0])
    meta_out = {"dataset": DATASET, "sex": "female", "has_vnc": True, "default_gain": DEFAULT_GAIN,
                "built": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "build": BUILD, "kit": __version__,
                "aliases": aliases, "alias_notes": {k: v[1] for k, v in ALIASES.items()},
                "data_aliases": sorted(derived), "known_nt": known_transmitters(kept),
                "min_weight": min_synapses, "nt_signs": NT_SIGN, "tyramine_sign": TYRAMINE_SIGN,
                "nt_conf_fallback": NT_CONF_FALLBACK, "neurons": len(rows), "edges": int(pre.size),
                "synapses_in_edges": int(syn.sum()), "counts": counts, "maps": maps,
                "superclass_map": {**SUPERCLASS, **{f"{k} ({r})": v for k, t in BY_REGION.items() for r, v in t.items()}},
                "superclass_rule": "sensory, motor and visceral_circulatory split by BANC's region column; "
                                   "visceral_circulatory becomes the region's endocrine class",
                "neurons_with_soma": int(ok.sum()),
                "soma_range_nm": {ax: [float(somas[ok, i].min()), float(somas[ok, i].max())] for i, ax in enumerate("xyz")} if ok.any() else {},
                "soma_by_region_nm": {},
                "layout_axis_hint": "y",
                "sources": {k: {"url": s["url"], "id": s["id"], "version": s["version"], "bytes": s["bytes"], "md5": s["md5"],
                                "sha256": s["sha256"]} for k, s in SOURCES.items()},
                "credits": ("BANC v888 (Bates, Phelps, Kim, Yang et al. 2026, Nature, doi:10.1038/s41586-026-10735-w; the data "
                            "deposit doi:10.7910/DVN/7WTH1N, CC BY 4.0)")}
    for region in ("central_brain", "ventral_nerve_cord", "optic_lobe"):
        sel = np.array([(a.get("region") == region) for a in kept]) & ok
        if sel.any():
            meta_out["soma_by_region_nm"][region] = {ax: [round(float(somas[sel, i].min())), round(float(somas[sel, i].max()))]
                                                     for i, ax in enumerate("xyz")}
    write_flyb(out, rows, pre, post, syn, meta_out, dataset=DATASET, nt_sign=NT_SIGN)
    if not quiet:
        print(f"Built {out} ({len(rows):,} neurons, {pre.size:,} connections, {int(syn.sum()):,} synapses) in "
              f"{time.time() - t0:.0f} s", file=sys.stderr)
    return Path(out)


def built_with(path: Path | str) -> int:
    from .flywire import built_with as _built_with
    return _built_with(path)


def ensure_banc(quiet: bool = False) -> Path:
    """The BANC FLYB file, built on first use (and rebuilt when this builder puts more in it than the file has)."""
    if BANC_FILE.exists() and built_with(BANC_FILE) >= BUILD:
        return BANC_FILE
    return build_banc(quiet=quiet)


if __name__ == "__main__":
    build_banc()
