#!/usr/bin/env python3
"""
Harvest where APL's and DPM's synapses are, region by region, from neuPrint:

    data/mb_roi_connectivity.json.gz    every connection into and out of APL and DPM, split by
                                        neuPrint's primary regions (the calyx, the pedunculus, each lobe ...)

The kit's connectome file holds one synapse count per pair of neurons and no positions, so the
local-release rule for APL (``parts.LOCAL``) had to place each synapse by the Kenyon-cell lobe of the
partner. This table says where the synapses really are. The kit never needs it to run: without the file
the rule falls back to the lobe approximation.

neuPrint needs a personal token (neuprint.janelia.org, account page). The script reads it from
``NEUPRINT_APPLICATION_CREDENTIALS`` (the name neuprint-python uses) and never prints it. It runs in the
repository's GitHub Actions workflow ``neuprint-harvest.yml``, which passes the ``NEUPRINT_TOKEN`` secret,
or anywhere that can reach neuprint.janelia.org:

    pip install neuprint-python
    NEUPRINT_APPLICATION_CREDENTIALS=... python tools/harvest_neuprint_rois.py --out data/mb_roi_connectivity.json.gz

The data are the MaleCNS connectome as served by neuPrint (Janelia FlyEM; CC BY 4.0).
"""
from __future__ import annotations

import argparse
import datetime as dt
import gzip
import json
import os
import sys

SERVER = "neuprint.janelia.org"
CELLS = ("APL", "DPM")


def pick_dataset(names, wanted: str | None) -> str:
    """The MaleCNS dataset: the one asked for, else ``male-cns:v1.0`` (the kit's release), else the newest."""
    names = sorted(names)
    if wanted:
        if wanted not in names:
            raise SystemExit(f"dataset {wanted!r} is not on {SERVER}; it has: {', '.join(names)}")
        return wanted
    male = [n for n in names if n.startswith("male-cns")]
    if not male:
        raise SystemExit(f"no male-cns dataset on {SERVER}; it has: {', '.join(names)}")
    return "male-cns:v1.0" if "male-cns:v1.0" in male else male[-1]


def pack(neurons, conns, dataset: str, version: str) -> dict:
    """Compact JSON: region names once, neurons by body id, edges as [pre, post, region index, synapses].

    ``neurons``: rows (bodyId, type, instance); ``conns``: rows (bodyId_pre, bodyId_post, roi, weight),
    duplicates allowed (a connection between APL and DPM comes back from both queries)."""
    seen, edges, rois = set(), [], {}
    for pre, post, roi, w in conns:
        key = (int(pre), int(post), str(roi))
        if key in seen or int(w) <= 0:
            continue
        seen.add(key)
        edges.append([key[0], key[1], rois.setdefault(key[2], len(rois)), int(w)])
    edges.sort()
    return {"dataset": dataset, "server": SERVER, "cells": list(CELLS), "neuprint_python": version,
            "fetched": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "source": "neuPrint (Janelia FlyEM), MaleCNS; CC BY 4.0",
            "rois": list(rois),
            "neurons": {str(int(b)): [t if isinstance(t, str) else "", i if isinstance(i, str) else ""]
                        for b, t, i in neurons},
            "edges": edges}


def summary(table: dict) -> str:
    """Synapses per region into and out of each harvested cell type (no token, safe to log)."""
    types = {int(b): v[0] for b, v in table["neurons"].items()}
    out = []
    for cell in table["cells"]:
        for way, side in (("from", 0), ("onto", 1)):
            tot: dict[str, int] = {}
            for e in table["edges"]:
                if types.get(e[side]) == cell:
                    r = table["rois"][e[2]]
                    tot[r] = tot.get(r, 0) + e[3]
            top = sorted(tot.items(), key=lambda kv: -kv[1])[:8]
            label = f"{cell} outputs" if way == "from" else f"{cell} inputs"
            out.append(f"{label}: {sum(tot.values()):,} synapses; " + ", ".join(f"{r} {n:,}" for r, n in top))
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default="data/mb_roi_connectivity.json.gz")
    ap.add_argument("--dataset", default=None, help="neuPrint dataset (default: male-cns:v1.0, else the newest male-cns)")
    args = ap.parse_args(argv)
    token = os.environ.get("NEUPRINT_APPLICATION_CREDENTIALS", "").strip()
    if not token:
        raise SystemExit("set NEUPRINT_APPLICATION_CREDENTIALS to a neuPrint token (in CI: the NEUPRINT_TOKEN secret)")
    import neuprint
    from neuprint import Client, NeuronCriteria as NC, fetch_adjacencies

    import requests                                   # (a neuprint-python dependency)
    r = requests.get(f"https://{SERVER}/api/dbmeta/datasets", headers={"Authorization": f"Bearer {token}"}, timeout=60)
    r.raise_for_status()
    dataset = pick_dataset(r.json().keys(), args.dataset)
    print(f"dataset {dataset} on {SERVER} (neuprint-python {neuprint.__version__})", flush=True)
    client = Client(SERVER, dataset=dataset, token=token, progress=False)
    neurons, conns = [], []
    for cell in CELLS:
        for kw in ({"targets": NC(type=cell, client=client)}, {"sources": NC(type=cell, client=client)}):
            ndf, cdf = fetch_adjacencies(**kw, properties=["type", "instance"], weight_props=["weight"], client=client)
            neurons += list(ndf[["bodyId", "type", "instance"]].itertuples(index=False, name=None))
            conns += list(cdf[["bodyId_pre", "bodyId_post", "roi", "weight"]].itertuples(index=False, name=None))
            way = "into" if "targets" in kw else "out of"
            print(f"  {len(cdf):,} region rows {way} {cell}", flush=True)
    table = pack(neurons, conns, dataset, neuprint.__version__)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with gzip.open(args.out, "wt", encoding="utf-8") as f:
        json.dump(table, f, separators=(",", ":"))
    print(f"wrote {args.out}: {len(table['neurons']):,} neurons, {len(table['edges']):,} region rows, "
          f"{len(table['rois'])} regions", flush=True)
    print(summary(table), flush=True)


if __name__ == "__main__":
    sys.exit(main())
