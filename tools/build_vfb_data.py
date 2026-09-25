#!/usr/bin/env python3
"""
Build the Virtual Fly Brain / FlyBase-anatomy data files the kit ships in ``data/``:

    data/fbbt_map.json.gz        MaleCNS cell type -> FBbt class(es), with the route that made the match
    data/fbbt_tree.json.gz       the part of the FBbt ``is_a`` tree above the mapped classes (labels, parents,
                                 symbols, short definitions) plus the root classes the kit interprets

from three inputs:

* ``fbbt.obo`` — the Drosophila anatomy ontology (FlyBase, CC-BY 4.0), fetched from
  https://raw.githubusercontent.com/FlyBase/drosophila-anatomy-developmental-ontology/master/fbbt.obo
* the kit's connectome (for the list of cell types and their neuron counts);
* optionally an *overlay* JSON harvested from the Virtual Fly Brain connector in a Claude session
  (``{"<kit type>": {"fbbt": ["FBbt_..."], "label": "...", "route": "name_in_male-cns"}}``), for the type
  names the OBO does not carry as synonyms (VFB adds ``name_in_male-cns`` synonyms on its own server).

Matching is deliberately strict (see ``match_type``): a kit type name must equal a class label, a VFB
symbol or an EXACT synonym of a non-obsolete *neuron* class of the adult fly, in that order of preference
(RELATED synonyms are old names that collide: Tm36 is a RELATED synonym of TmY21, Li28 of Li16, so they
are never used); ties are broken by the rule "prefer the class whose label ends with the code", then
"prefer the male class", and anything still ambiguous is left unresolved rather than guessed. Comma-joined
kit types ("MNad04,MNad48") map to every member; a ``_a``/``_b`` suffix type falls back to its stem's class,
marked coarse. The ``tests/`` never need this script: they build a tiny map by hand.

    python tools/build_vfb_data.py --obo /path/to/fbbt.obo [--overlay overlay.json] [--out data/]
"""
from __future__ import annotations

import argparse
import collections
import gzip
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

NEURON = "FBbt:00005106"
ADULT = "FBbt:00047095"
LARVAL = "FBbt:00001446"
FEMALE = "FBbt:00048491"     # female-specific anatomical entity: never the class of a MaleCNS type
ROOTS = {
    "neuron": NEURON, "adult": ADULT, "larval": LARVAL,
    "peptidergic": "FBbt:00004101", "primary": "FBbt:00047097", "secondary": "FBbt:00047096",
    "transmitters": {
        "acetylcholine": "FBbt:00007173", "gaba": "FBbt:00007228", "glutamate": "FBbt:00100291",
        "dopamine": "FBbt:00005131", "octopamine": "FBbt:00007364", "serotonin": "FBbt:00005133",
        "histamine": "FBbt:00007367", "tyramine": "FBbt:00100397",
    },
}
SYN_RE = re.compile(r'^"(.*)" (EXACT|RELATED|NARROW|BROAD)(?: (\S+))? \[')
DEF_RE = re.compile(r'^"(.*)" \[')
SYMBOL_RE = re.compile(r'^IAO:0000028 "(.*)" xsd:string')
STAGE_WORDS = re.compile(r"\b(larval|larva|embryonic|embryo|immature|pupal|pupa)\b", re.I)


def parse_obo(path: Path) -> dict[str, dict]:
    """Every ``[Term]`` stanza as a dict of lists (``id``, ``name``, ``def``, ``is_a``, ``synonym``...)."""
    terms: dict[str, dict] = {}
    stanza = None
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line == "[Term]":
                stanza = {}
                continue
            if line.startswith("["):
                stanza = None
                continue
            if stanza is None:
                continue
            if not line:
                if "id" in stanza:
                    terms[stanza["id"][0]] = stanza
                stanza = None
                continue
            key, _, value = line.partition(": ")
            stanza.setdefault(key, []).append(value)
    if stanza and "id" in stanza:
        terms[stanza["id"][0]] = stanza
    return terms


def clean(terms: dict[str, dict]) -> dict[str, dict]:
    """Per term: name, def, parents (plain ``is_a`` only, no general class inclusion axioms), synonyms by
    scope/type, symbols, obsolete flag."""
    out = {}
    for tid, s in terms.items():
        parents = []
        for p in s.get("is_a", []):
            if "{" in p.split("!")[0]:          # `is_a: X {gci_relation=...}` is not a plain subclass axiom
                continue
            parents.append(p.split(" ")[0])
        syn = collections.defaultdict(list)
        for v in s.get("synonym", []):
            m = SYN_RE.match(v)
            if not m:
                continue
            text, scope, typ = m.group(1), m.group(2), m.group(3)
            if typ == "VFB_SYMBOL":
                syn["symbol"].append(text)
            elif typ in ("HARTENSTEIN_BRAIN_LINEAGE", "ITO_LEE_BRAIN_LINEAGE", "TECHNAU_BRAIN_LINEAGE",
                         "PRIMARY_VNC_LINEAGE", "SECONDARY_VNC_LINEAGE"):
                syn["lineage_name"].append(text)
            else:
                syn[scope].append(text)
        for v in s.get("property_value", []):
            m = SYMBOL_RE.match(v)
            if m and m.group(1) not in syn["symbol"]:
                syn["symbol"].append(m.group(1))
        d = ""
        if s.get("def"):
            m = DEF_RE.match(s["def"][0])
            d = m.group(1) if m else s["def"][0]
            d = d.replace('\\"', '"')
        out[tid] = {"name": s["name"][0] if s.get("name") else tid, "def": d, "parents": parents,
                    "syn": dict(syn), "obsolete": s.get("is_obsolete", ["false"])[0] == "true",
                    "comment": (s.get("comment") or [""])[0]}
    return out


def ancestors_index(t: dict[str, dict]) -> dict[str, set]:
    """``is_a`` ancestor closure per term (memoised depth-first walk)."""
    memo: dict[str, set] = {}

    def walk(tid: str) -> set:
        if tid in memo:
            return memo[tid]
        memo[tid] = set()                       # guards against cycles
        acc = set()
        for p in t.get(tid, {}).get("parents", []):
            acc.add(p)
            acc |= walk(p)
        memo[tid] = acc
        return acc

    for tid in t:
        walk(tid)
    return memo


def is_stage_ok(tid: str, t: dict, anc: dict) -> bool:
    """Adult or stage-agnostic classes only: MaleCNS is an adult connectome."""
    a = anc[tid]
    if ADULT in a:
        return True
    if LARVAL in a:
        return False
    return STAGE_WORDS.search(t[tid]["name"]) is None


def build_indices(t: dict, anc: dict) -> dict[str, dict[str, list[str]]]:
    idx = {"label": collections.defaultdict(list), "symbol": collections.defaultdict(list),
           "exact": collections.defaultdict(list), "related": collections.defaultdict(list)}
    for tid, term in t.items():
        if term["obsolete"] or NEURON not in anc[tid] or not is_stage_ok(tid, t, anc):
            continue
        if FEMALE in anc[tid] or "(female)" in term["name"]:
            continue
        idx["label"][term["name"]].append(tid)
        for s in term["syn"].get("symbol", []):
            idx["symbol"][s].append(tid)
        for s in term["syn"].get("EXACT", []):
            idx["exact"][s].append(tid)
        for s in term["syn"].get("RELATED", []):
            idx["related"][s].append(tid)
    return idx


def disambiguate(name: str, cands: list[str], t: dict, anc: dict) -> list[str]:
    """The tie-breaking rules; returns the survivors (1 = resolved, >1 = ambiguous)."""
    cands = list(dict.fromkeys(cands))
    if len(cands) <= 1:
        return cands
    ends = [c for c in cands if re.search(r"(^|[\s(])" + re.escape(name) + r"(\)|$)", t[c]["name"])]
    if len(ends) == 1:
        return ends
    if ends:
        cands = ends
    adult = [c for c in cands if ADULT in anc[c]]
    if len(adult) == 1:
        return adult
    if adult:
        cands = adult
    male = [c for c in cands if "(female)" not in t[c]["name"]]
    if len(male) == 1:
        return male
    if male:
        cands = male
    return cands


STEM_RE = re.compile(r"^(.+?)_[a-z](?:\d|[a-z])?$")


def match_type(name: str, idx: dict, t: dict, anc: dict) -> dict:
    """Resolve one kit type name. Returns ``{"fbbt": [...], "route": ..., "coarse": bool}`` or
    ``{"fbbt": [], "route": "ambiguous"|"unresolved", "candidates": [...]}``."""
    for route, key in (("obo_label", "label"), ("obo_symbol", "symbol"), ("obo_exact", "exact")):
        cands = idx[key].get(name)
        if not cands:
            continue
        keep = disambiguate(name, cands, t, anc)
        if len(keep) == 1:
            return {"fbbt": keep, "route": route, "coarse": False}
        return {"fbbt": [], "route": "ambiguous", "candidates": keep}
    if "," in name:                              # "MNad04,MNad48": every member
        parts = [p.strip() for p in name.split(",") if p.strip()]
        found, routes = [], []
        for p in parts:
            r = match_type(p, idx, t, anc)
            if r["fbbt"]:
                found.extend(r["fbbt"])
                routes.append(r["route"])
        if found:
            return {"fbbt": list(dict.fromkeys(found)), "route": "obo_split", "coarse": len(found) < len(parts),
                    "members": len(parts), "matched": len(found)}
        return {"fbbt": [], "route": "unresolved"}
    m = STEM_RE.match(name)                      # "ER3a_a" -> the class of "ER3a", marked coarse
    if m:
        r = match_type(m.group(1), idx, t, anc)
        if r["fbbt"] and r["route"] in ("obo_label", "obo_symbol", "obo_exact"):
            return {"fbbt": r["fbbt"], "route": "obo_stem", "coarse": True, "stem": m.group(1)}
    return {"fbbt": [], "route": "unresolved"}


def curated_transmitters(tid: str, anc: dict) -> list[str]:
    a = anc[tid] | {tid}
    return [nt for nt, root in ROOTS["transmitters"].items() if root in a]


def short_def(text: str, limit: int = 480) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit(". ", 1)[0]
    return (cut if len(cut) > 120 else text[:limit].rstrip()) + " …"


def build(obo: Path, kit_types: dict[str, int], overlay: dict | None = None) -> tuple[dict, dict, dict]:
    raw = parse_obo(obo)
    header_licence = "CC-BY 4.0"
    t = clean(raw)
    anc = ancestors_index(t)
    idx = build_indices(t, anc)
    overlay = overlay or {}
    fbbt_map, unresolved = {}, {}
    for name, n in kit_types.items():
        r = match_type(name, idx, t, anc)
        ov = overlay.get(name)
        was = {w.replace("_", ":") for w in (ov or {}).get("was", [])}
        if ov is not None and ov.get("route") == "rejected" and (not was or set(r["fbbt"]) <= was):
            r = {"fbbt": [], "route": "rejected"}                     # VFB's MaleCNS names say the OBO match is wrong
        # the harvest fills what the OBO cannot match, replaces a coarse match, and overrides a direct match
        # only when it was recorded as a correction of exactly that match
        corrects = bool(ov and ov.get("override") and (not was or set(r["fbbt"]) <= was))
        if ov is not None and ov.get("fbbt") and (not r["fbbt"] or r["route"] in ("obo_stem", "obo_split", "rejected") or corrects):
            ids = [i.replace("_", ":") for i in ov["fbbt"]]
            ids = [i for i in ids if i in t and not t[i]["obsolete"]]
            if ids:
                r = {"fbbt": ids, "route": ov.get("route", "connector"), "coarse": bool(ov.get("coarse", False))}
        if r["fbbt"]:
            entry = {"fbbt": r["fbbt"], "route": r["route"], "n": n}
            if r.get("coarse"):
                entry["coarse"] = True
            if r.get("stem"):
                entry["stem"] = r["stem"]
            nts = sorted({nt for i in r["fbbt"] for nt in curated_transmitters(i, anc)})
            if nts:
                entry["nt"] = nts
                # FBbt:2xxxxxxx classes are the systematic connectome-derived types (their transmitter is
                # another EM data set's prediction); everything else is literature-curated
                entry["evidence"] = "literature" if any(not i.startswith("FBbt:2") for i in r["fbbt"]) else "connectome"
            fbbt_map[name] = entry
        else:
            unresolved[name] = {"n": n, "route": r["route"], "candidates": [
                {"fbbt": c, "label": t[c]["name"]} for c in r.get("candidates", [])]}
    # the tree: mapped classes and every ancestor, restricted to neuron classes (plus the roots)
    mapped = {i for e in fbbt_map.values() for i in e["fbbt"]}
    keep = set(mapped)
    for i in mapped:
        keep |= anc[i]
    keep = {i for i in keep if i in t and (NEURON in anc[i] or i == NEURON)}
    types_of = collections.defaultdict(list)
    for name, e in fbbt_map.items():
        for i in e["fbbt"]:
            types_of[i].append(name)
    classes = {}
    for i in sorted(keep):
        term = t[i]
        entry = {"label": term["name"], "parents": [p for p in term["parents"] if p in keep]}
        if term["syn"].get("symbol"):
            entry["symbol"] = term["syn"]["symbol"][0]
        if term["syn"].get("lineage_name"):
            entry["lineage_names"] = term["syn"]["lineage_name"][:4]
        if i in mapped and term["def"]:
            entry["def"] = short_def(term["def"])
        classes[i] = entry
    tree = {"source": {"ontology": "FBbt (Drosophila anatomy ontology), FlyBase", "release": _release(obo),
                       "licence": header_licence, "url": "https://github.com/FlyBase/drosophila-anatomy-developmental-ontology"},
            "roots": ROOTS, "classes": classes}
    return fbbt_map, tree, unresolved


def _release(obo: Path) -> str:
    with open(obo, encoding="utf-8") as f:
        for _ in range(20):
            line = f.readline()
            if line.startswith("data-version:"):
                return line.split(":", 1)[1].strip()
    return "unknown"


def kit_type_counts() -> dict[str, int]:
    from virtual_fly.connectome import load_connectome
    conn = load_connectome(quiet=True)
    return conn.type_counts()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--obo", required=True, type=Path)
    ap.add_argument("--overlay", type=Path, default=None, help="connector harvest: type -> FBbt ids")
    ap.add_argument("--types", type=Path, default=None, help="JSON {type: n} instead of loading the connectome")
    ap.add_argument("--out", type=Path, default=HERE.parent / "data")
    ap.add_argument("--report", type=Path, default=None, help="write the unresolved list here (JSON)")
    a = ap.parse_args(argv)
    kit = json.loads(a.types.read_text()) if a.types else kit_type_counts()
    overlay = json.loads(a.overlay.read_text()) if a.overlay else None
    if overlay and "types" in overlay:                       # the file merge_vfb_harvest.py writes
        overlay = overlay["types"]
    fbbt_map, tree, unresolved = build(a.obo, kit, overlay)
    a.out.mkdir(parents=True, exist_ok=True)
    src = dict(tree["source"])
    with gzip.open(a.out / "fbbt_map.json.gz", "wt", encoding="utf-8") as f:
        json.dump({"source": src, "overlay_source": "Virtual Fly Brain (name_in_male-cns synonyms, Berg et al. 2025)" if overlay else None,
                   "types": fbbt_map}, f, separators=(",", ":"), ensure_ascii=False)
    with gzip.open(a.out / "fbbt_tree.json.gz", "wt", encoding="utf-8") as f:
        json.dump(tree, f, separators=(",", ":"), ensure_ascii=False)
    n_types, n_neurons = len(fbbt_map), sum(e["n"] for e in fbbt_map.values())
    total_n = sum(kit.values())
    routes = collections.Counter(e["route"] for e in fbbt_map.values())
    print(f"mapped {n_types} of {len(kit)} types, {n_neurons} of {total_n} typed neurons ({100 * n_neurons / max(total_n, 1):.1f} %)")
    print("routes:", dict(routes))
    print(f"unresolved: {len(unresolved)} types, {sum(u['n'] for u in unresolved.values())} neurons; "
          f"ambiguous {sum(1 for u in unresolved.values() if u['route'] == 'ambiguous')}")
    print(f"tree: {len(tree['classes'])} classes")
    if a.report:
        a.report.write_text(json.dumps(unresolved, indent=1))


if __name__ == "__main__":
    main()
