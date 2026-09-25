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
expressionCluster per receptor gene) and ``datasets.json`` (the clusters' data sets and stages).

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
# data-set families in the order the kit prefers them (adult, male first); keys are matched against the
# cluster names / data-set names case-insensitively
FAMILY_ORDER = ["FCA_MALE", "FCA_MIXED", "FCA_FEMALE", "FCA", "DAVIE", "AFCA", "OZEL_ADM", "BAKER", "MOKASHI", "ALLEN", "SAAVEDRA",
                "KURMANGALIYEV", "HORMANN", "OZEL", "AVALOS"]


def norm(cid: str) -> str:
    return cid.replace("_", ":")


def load_rows(pattern: str) -> list:
    rows = []
    for f in sorted(glob.glob(pattern)):
        if re.search(r"(partial|progress|scope)", Path(f).name):
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
        elif st == "other_name" and len(ids) == 1:
            types[t] = {"fbbt": ids, "route": r.get("route") or "other_name", "label": r.get("label", ""), "source": "overlay"}
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
    for r in load_rows(str(harvest / "verify_*.json")):
        t, verdict = r.get("type"), r.get("verdict")
        if not t or verdict != "not_this_class":
            continue
        alts = [a for a in (r.get("alternatives") or []) if FBBT_RE.match(str(a.get("fbbt", "")))]
        if len(alts) == 1:
            types[t] = {"fbbt": [norm(alts[0]["fbbt"])], "route": "name_in_male-cns", "label": alts[0].get("label", ""),
                        "source": "verify: the OBO match carried other MaleCNS names; this class matched the name"}
        else:
            rejected[t] = {"fbbt": [], "route": "rejected", "was": [norm(i) for i in r.get("fbbt", [])],
                           "note": r.get("note", "the class carries MaleCNS names, none of them this one")}
    types.update(rejected)
    return {"source": "Virtual Fly Brain (virtualflybrain.org): name_in_male-cns synonyms (Berg et al. 2025) and MaleCNS "
                      "individuals' classes, read through the VFB MCP connector; FBbt is CC-BY 4.0 (FlyBase)",
            "harvested": str(date.today()), "types": types, "ambiguous": ambiguous}


def family_of(name: str, dataset: str | None, families: dict) -> str | None:
    text = " ".join(x for x in (name, dataset) if x).upper()
    best = None
    for key, fam in families.items():
        for probe in [key] + [str(fam.get(k, "")) for k in ("name", "match", "prefix")]:
            probe = probe.upper().replace("SCRNASEQ_", "").strip("_ ")
            if len(probe) >= 3 and probe in text.replace("SCRNASEQ_", "") and (best is None or len(probe) > len(best[1])):
                best = (key, probe)
    return best[0] if best else None


def merge_receptors(harvest: Path) -> dict:
    from virtual_fly.parts import RECEPTORS
    ds = {}
    if (harvest / "datasets.json").exists():
        ds = json.loads((harvest / "datasets.json").read_text())
    families = {k.upper(): v for k, v in (ds.get("families") or {}).items()}
    genes, clusters, classes = {}, {}, collections.defaultdict(lambda: collections.defaultdict(list))
    coupling = {}
    for f in sorted(glob.glob(str(harvest / "rx_*.json"))):
        d = json.loads(Path(f).read_text())
        if not isinstance(d, dict) or not d.get("rows"):
            print(f"  {f}: no rows", file=sys.stderr)
            continue
        gene = GENE_NAMES.get(d.get("gene", ""), d.get("gene", ""))
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
    for rec in RECEPTORS:
        genes[rec.gene] = {"fbgn": rec.fbgn, "modulator": rec.modulator, "sign": rec.sign, "coupling": rec.coupling,
                           "vfb_coupling": coupling.get(rec.gene)}
    for gene in coupling:
        genes.setdefault(gene, {"fbgn": None, "modulator": None, "sign": 0, "coupling": "tyramine receptor (not modelled)",
                                "vfb_coupling": coupling.get(gene)})
    fam_meta = {k: {"label": v.get("name", k), "stage": v.get("stage"), "sex": v.get("sex"), "licence": v.get("licence"),
                    "publication": v.get("publication")} for k, v in families.items()}
    order = [k for k in FAMILY_ORDER if k in families] + [k for k in families if k not in FAMILY_ORDER]
    adult = [k for k in order if families[k].get("stage") == "adult"]
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
