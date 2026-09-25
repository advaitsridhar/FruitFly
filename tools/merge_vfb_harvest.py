#!/usr/bin/env python3
"""
Merge a Virtual Fly Brain connector harvest (made in a Claude session; the connector cannot be scripted
from a machine) into the two files the kit's builders read:

    tools/vfb_overlay.json        kit type -> FBbt class for the names the OBO lacks, plus the OBO matches
                                  VFB's own MaleCNS-name synonyms contradict ("rejected"); an input of
                                  tools/build_vfb_data.py
    data/vfb_receptors.json.gz    per FBbt class, per aminergic receptor gene: the scRNA-seq clusters on
                                  VFB (fraction of cells expressing, level), with each cluster's data set,
                                  stage and sex; read by virtual_fly/vfb.py

The harvest directory holds, one JSON file per worker: ``overlay_<i>.json`` (search_terms +
get_term_info per unresolved type), ``verify_<i>.json`` (get_term_info on the OBO matches),
``routec_<i>.json`` (bodyId -> VFB individual -> classes), ``rx_<gene>.json`` (run_query
expressionCluster per receptor gene), ``datasets.json`` (the clusters' data sets and stages) and
``corrections_checked.json`` (alternative classes for contradicted matches, each checked by hand to
carry the type's ``name_in_male-cns`` synonym). A partial or malformed file is skipped with a warning.

    python tools/merge_vfb_harvest.py --harvest <dir> [--out data/] [--overlay tools/vfb_overlay.json]
"""
from __future__ import annotations

import argparse
import collections
import glob
import gzip
import json
import re
import sys
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

FBBT_RE = re.compile(r"^FBbt[_:]\d{8}$")
# the ASCII names the harvest used for the receptor genes -> the symbols parts.RECEPTORS uses
GENE_NAMES = {"Octalpha2R": "Octα2R", "Octbeta1R": "Octβ1R", "Octbeta2R": "Octβ2R", "Octbeta3R": "Octβ3R"}
TYRAMINE = {"Oct-TyrR", "TyrR", "TyrRII"}
# which adult data set the kit prefers for a class, when several have a cluster of it: the Fly Cell Atlas
# (young adults) first, then the brain atlases, the aging atlas and the nerve-cord atlases; male before
# mixed before female (MaleCNS is male); head or brain before whole fly, nerve cord and other tissues
STUDY_RANK = {"FCA2022": 0, "Davie2018": 1, "Ozel2021": 2, "Baker2021": 3, "Mokashi2021": 3, "AFCA2023": 4,
              "Allen2020": 5, "Saavedra2023": 5}
SEX_RANK = {"male": 0, "mixed": 1, "unknown": 1, "female": 2}
TISSUE_RANK = {"head": 0, "brain": 0, "optic lobe": 0, "whole fly (all FCA samples)": 1, "whole fly (all AFCA samples)": 1,
               "VNC": 2, "thorax (VNC)": 2, "antenna": 2, "proboscis": 2, "leg": 2, "wing": 2, "haltere": 2,
               "pars intercerebralis (insulin-producing cells)": 2, "body": 3}


def family_rank(key: str, fam: dict) -> tuple:
    """Study first; then nervous-system tissue (head, brain, optic lobe, whole fly) before the rest; then
    male before mixed before female; then head before whole fly, younger before older."""
    age = re.search(r"_D(\d+)_", key)
    fine = TISSUE_RANK.get(fam.get("tissue"), 4)
    return (STUDY_RANK.get(fam.get("study"), 9), 0 if fine <= 1 else fine, SEX_RANK.get(fam.get("sex"), 1), fine,
            int(age.group(1)) if age else 0, key)


def family_label(key: str, fam: dict) -> str:
    """ "FCA 2022, male head" from the family's study, sex and tissue."""
    study = re.sub(r"(\D)(\d{4})$", r"\1 \2", str(fam.get("study") or key))
    age = re.search(r"_D(\d+)_", key)
    bits = [b for b in ((f"day {age.group(1)}" if age else ""), fam.get("sex") if fam.get("sex") not in (None, "unknown") else "",
                        fam.get("tissue") or "") if b]
    return f"{study}, {' '.join(bits)}" if bits else study


def norm(cid: str) -> str:
    return cid.replace("_", ":")


def load_rows(pattern: str) -> list:
    rows = []
    for f in sorted(glob.glob(pattern)):
        if not re.search(r"_\d+\.json$", Path(f).name):         # overlay_12.json, not overlay_12_progress.json
            continue
        try:
            d = json.loads(Path(f).read_text())
        except json.JSONDecodeError as e:
            print(f"  skipping {f}: {e}", file=sys.stderr)
            continue
        if isinstance(d, dict) and "rows" in d and not d.get("gene"):
            d = d["rows"]
        if isinstance(d, list):
            rows.extend(r for r in d if isinstance(r, dict))
    return rows


def merge_overlay(harvest: Path) -> dict:
    types, ambiguous, rejected = {}, {}, {}
    for r in load_rows(str(harvest / "overlay_*.json")):
        t, st = r.get("type"), r.get("status")
        ids = [norm(i) for i in (r.get("fbbt") or []) if FBBT_RE.match(str(i))]
        if not t:
            continue
        if st == "resolved" and len(ids) == 1:
            types[t] = {"fbbt": ids, "route": r.get("route") or "name_in_male-cns", "label": r.get("label", ""), "source": "overlay"}
        elif st == "other_name" and len(ids) == 1:          # only another data set's name: weaker, annotation only
            types[t] = {"fbbt": ids, "route": r.get("route") or "other_name", "label": r.get("label", ""), "source": "overlay",
                        "coarse": True}
        elif st == "ambiguous" or len(ids) > 1:
            ambiguous[t] = {"fbbt": ids, "note": r.get("note", "")}
    for r in load_rows(str(harvest / "routec_*.json")):
        t = r.get("type")
        if not t or r.get("status") != "found" or t in types:
            continue
        cells = [c for c in (r.get("cell_type_classes") or []) if FBBT_RE.match(str(c.get("fbbt", "")))]
        if not cells:
            continue
        stem = re.split(r"[_/]", t)[0]
        named = [c for c in cells if t in c.get("label", "") or (len(stem) > 2 and stem in c.get("label", ""))]
        pick = named if len(named) == 1 else cells if len(cells) == 1 else []
        if pick:
            types[t] = {"fbbt": [norm(pick[0]["fbbt"])], "route": "vfb_individual", "label": pick[0].get("label", ""),
                        "source": f"MaleCNS:{r.get('body_id')} -> {r.get('vfb_id')}", "coarse": True}
        else:
            ambiguous[t] = {"fbbt": [norm(c["fbbt"]) for c in cells], "note": "several cell-type classes on the individual"}
    checked = {}
    if (harvest / "corrections_checked.json").exists():
        checked = json.loads((harvest / "corrections_checked.json").read_text()).get("types", {})
    for r in load_rows(str(harvest / "verify_*.json")):
        t, verdict = r.get("type"), r.get("verdict")
        was = r.get("fbbt") or []
        was = [norm(w) for w in ([was] if isinstance(was, str) else was)]
        if not t:
            continue
        if t in checked and norm(checked[t]["was"]) in was:          # an alternative class checked to carry the name
            c = checked[t]
            types[t] = {"fbbt": [norm(c["fbbt"])], "route": "name_in_male-cns", "label": c.get("label", ""), "override": True,
                        "was": was, "source": "verification: " + c.get("evidence", "")}
        elif verdict == "not_this_class":
            rejected[t] = {"fbbt": [], "route": "rejected", "was": was,
                           "note": r.get("note") or "the class carries MaleCNS names, none of them this one"}
    for t, c in checked.items():
        if t not in types:
            print(f"  corrections_checked: {t} has no verification row naming {c['was']}; not applied", file=sys.stderr)
    types.update(rejected)
    return {"source": "Virtual Fly Brain (virtualflybrain.org): name_in_male-cns synonyms (Berg et al. 2025) and MaleCNS "
                      "individuals' classes, read through the VFB MCP connector; FBbt is CC-BY 4.0 (FlyBase)",
            "harvested": str(date.today()), "types": types, "ambiguous": ambiguous}


def family_of(name: str, dataset: str | None, families: dict) -> str | None:
    """The data-set family of a cluster: the family whose ``match`` string the cluster name starts with (VFB
    names clusters "<data set>_seq_clustering_<cell type>"), else the family whose probe (match, name or
    key) occurs in the name or data set as a whole token (so "FCA" never matches inside "AFCA"), longest first."""
    prefixed = [k for k, f in families.items() if f.get("match") and (name or "").startswith(str(f["match"]))]
    if prefixed:
        return max(prefixed, key=lambda k: len(str(families[k]["match"])))
    text = " ".join(x for x in (name, dataset) if x).upper()
    best = None
    for key, fam in families.items():
        for probe in [str(fam.get("match", "")), str(fam.get("name", "")), key]:
            probe = probe.upper().strip()
            if len(probe) < 3:
                continue
            if re.search(r"(?<![A-Z0-9])" + re.escape(probe) + r"(?![A-Z0-9])", text) and (best is None or len(probe) > len(best[1])):
                best = (key, probe)
    return best[0] if best else None


def merge_receptors(harvest: Path) -> dict:
    from virtual_fly.parts import RECEPTORS
    ds = {}
    if (harvest / "datasets.json").exists():
        ds = json.loads((harvest / "datasets.json").read_text())
    families = dict(ds.get("families") or {})
    genes, clusters, classes = {}, {}, collections.defaultdict(lambda: collections.defaultdict(list))
    coupling = {}
    harvested = set()
    for f in sorted(glob.glob(str(harvest / "rx_*.json"))):
        try:
            d = json.loads(Path(f).read_text())
        except (json.JSONDecodeError, OSError) as e:
            print(f"  skipping {f}: {e}", file=sys.stderr)
            continue
        if not isinstance(d, dict) or not d.get("rows"):
            print(f"  skipping {f}: no rows", file=sys.stderr)
            continue
        if d.get("count_status") != "exact" or (d.get("count") not in (None, -1) and len(d["rows"]) != d["count"]):
            print(f"  skipping {f}: incomplete ({len(d['rows'])} rows of {d.get('count')}, {d.get('count_status')})", file=sys.stderr)
            continue
        gene = GENE_NAMES.get(d.get("gene", ""), d.get("gene", ""))
        harvested.add(gene)
        coupling[gene] = d.get("coupling")
        for r in d["rows"]:
            fblc, cid = r.get("fblc"), r.get("anatomy_fbbt")
            if not fblc or not cid or not FBBT_RE.match(str(cid)):
                continue
            try:
                extent, level = float(r.get("extent")), float(r.get("level") or 0.0)
            except (TypeError, ValueError):
                continue
            cid = norm(cid)
            if fblc not in clusters:
                fam = family_of(r.get("cluster") or "", r.get("dataset"), families)
                meta = families.get(fam, {}) if fam else {}
                clusters[fblc] = {"name": r.get("cluster"), "family": fam or "unknown", "stage": meta.get("stage", "unknown"),
                                  "sex": meta.get("sex", "unknown"), "anatomy": cid}
            classes[cid][gene].append([fblc, round(extent, 4), round(level, 2)])
    known = {rec.gene: rec for rec in RECEPTORS}
    for gene in sorted(harvested):
        rec = known.get(gene)
        if rec is not None:
            genes[gene] = {"fbgn": rec.fbgn, "modulator": rec.modulator, "sign": rec.sign, "coupling": rec.coupling,
                           "vfb_coupling": coupling.get(gene), "harvested": True}
        else:
            what = "tyramine receptor (not modelled)" if gene in TYRAMINE else "not in parts.RECEPTORS (not modelled)"
            if gene not in TYRAMINE:
                print(f"  {gene}: not in parts.RECEPTORS; kept without a sign", file=sys.stderr)
            genes[gene] = {"fbgn": None, "modulator": None, "sign": 0, "coupling": what, "vfb_coupling": coupling.get(gene),
                           "harvested": True}
    missing = [rec.gene for rec in RECEPTORS if rec.gene not in harvested]
    if missing:
        print(f"  not harvested (their modulators keep the one-sign rule where none of theirs is): {', '.join(missing)}", file=sys.stderr)
    used = {c["family"] for c in clusters.values()}
    fam_meta = {k: {"label": family_label(k, v), "study": v.get("study"), "stage": v.get("stage"), "sex": v.get("sex"),
                    "tissue": v.get("tissue"), "licence": v.get("licence"), "dataset": v.get("licence_source")}
                for k, v in families.items() if k in used}
    adult = sorted((k for k in used if families.get(k, {}).get("stage") == "adult"), key=lambda k: family_rank(k, families[k]))
    return {"source": {"what": "single-cell RNA-seq clusters on Virtual Fly Brain: fraction of a cluster's cells expressing "
                               "each aminergic receptor gene (VFB's expressionCluster tables, values of 0.2 and above only)",
                       "licence": "CC-BY 4.0 (the data sets named in families)", "harvested": str(date.today()),
                       "families": fam_meta, "how_to_read": ds.get("how_to_read_dataset")},
            "families": adult, "genes": genes, "clusters": clusters,
            "classes": {cid: dict(g) for cid, g in classes.items()}}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--harvest", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=HERE.parent / "data")
    ap.add_argument("--overlay", type=Path, default=HERE / "vfb_overlay.json")
    a = ap.parse_args(argv)
    ov = merge_overlay(a.harvest)
    a.overlay.parent.mkdir(parents=True, exist_ok=True)
    a.overlay.write_text(json.dumps(ov, indent=1, ensure_ascii=False) + "\n")
    routes = collections.Counter(e["route"] for e in ov["types"].values())
    print(f"overlay: {len(ov['types'])} types ({dict(routes)}), {len(ov['ambiguous'])} ambiguous -> {a.overlay}")
    rx = merge_receptors(a.harvest)
    a.out.mkdir(parents=True, exist_ok=True)
    with gzip.open(a.out / "vfb_receptors.json.gz", "wt", encoding="utf-8") as f:
        json.dump(rx, f, separators=(",", ":"), ensure_ascii=False)
    stages = collections.Counter(c["stage"] for c in rx["clusters"].values())
    fams = collections.Counter(c["family"] for c in rx["clusters"].values())
    print(f"receptors: {len(rx['genes'])} genes, {len(rx['clusters'])} clusters ({dict(stages)}; families {dict(fams)}), "
          f"{len(rx['classes'])} classes; adult families in order: {rx['families']}")


if __name__ == "__main__":
    main()
