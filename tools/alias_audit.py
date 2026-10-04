#!/usr/bin/env python3
"""Every population spec the kit's code names, counted on every connectome file (the two-flies work, docs/TWO_FLIES_PLAN.md
9.5): the male, FlyWire's female and BANC, the female with a nerve cord. A hand-run tool.

    python tools/alias_audit.py                              # every spec of virtual_fly.specs.kit_specs(), three files
    python tools/alias_audit.py --files banc --only-zeros    # the rows where a term is empty somewhere
    python tools/alias_audit.py --markdown > audit.md        # a table for the docs
    python tools/alias_audit.py "prefix:LC10" "MN9,NoSuch"   # extra specs beside the kit's

A spec is a union of terms (``Connectome.select``); a union with one empty member hides behind the others' cells, so every
term is counted on its own (``Connectome.terms`` splits a spec the way select reads it: a type name or an alias that holds
a comma is one term). A term that a file answers through an alias says so, with the alias's spec; a term that raises (an
unknown selector kind, a bad regex) is reported, not fatal; a subtraction (``!x``) is counted as what it subtracts and never
called empty. The point on BANC: every zero must be explained, so for each term empty there the tool names the nearest
candidates: BANC cell types containing the term (case-insensitive) or starting with its first four characters, and how
many of BANC's neurons carry the term as their ``malecns_cell_type`` (the builder makes ``body:`` aliases from those,
banc.data_aliases); the last needs the deposit's metadata table in the data folder (banc.SOURCE_DIR).
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from virtual_fly import specs as kitspecs                         # noqa: E402

FILES = ("male", "flywire", "banc")
MAX_CANDIDATES = 12


# ---------------------------------------------------------------------------------------------- loading
def load_files(names) -> tuple[dict, dict[str, str]]:
    """{name: Connectome} for the files that load, and {name: why} for those that do not (a missing file is a note)."""
    from virtual_fly.connectome import Connectome, load_connectome
    conns, notes = {}, {}
    for name in names:
        try:
            if name == "male":
                conns[name] = load_connectome(quiet=True)
            elif name == "flywire":
                from virtual_fly.flywire import FEMALE_FILE
                if not FEMALE_FILE.exists():
                    notes[name] = f"no file at {FEMALE_FILE} (it is built on first use: python fly_brain.py --female)"
                    continue
                conns[name] = Connectome(FEMALE_FILE)
            elif name == "banc":
                from virtual_fly.banc import BANC_FILE
                if not BANC_FILE.exists():
                    notes[name] = f"no file at {BANC_FILE} (it is built on first use: python -m virtual_fly.banc)"
                    continue
                conns[name] = Connectome(BANC_FILE)
            else:
                notes[name] = f"unknown file name {name!r}: choose from {', '.join(FILES)}"
        except SystemExit as e:                                  # the loader's one-line refusals (a damaged file, ...)
            notes[name] = str(e)
        except Exception as e:                                   # anything else: reported, not fatal
            notes[name] = f"{type(e).__name__}: {e}"
    return conns, notes


# ---------------------------------------------------------------------------------------------- counting
def _alias_of(conn, term: str):
    """The alias spec a file would use for this term (not for a real cell type of the same name), or None."""
    aliases = getattr(conn, "aliases", None) or {}
    if not aliases:
        return None
    try:
        conn._index_types()
        if conn._exact_type(term) is not None:
            return None
    except Exception:
        pass
    if term in aliases:
        return aliases[term]
    if len(term) > 2 and term[-2] == "/" and term[-1] in "LRM" and term[:-2] in aliases:
        return aliases[term[:-2]]
    return None


def count_term(conn, term: str) -> dict:
    """One term on one file: its count, the alias it went through (if any), or the error it raised."""
    negate = term.startswith("!")
    bare = term[1:].strip() if negate else term.strip()
    row = {"term": term, "negate": negate, "count": None, "via": None, "error": None}
    if not bare:
        row["error"] = "empty term"
        return row
    try:
        row["count"] = int(conn.select(bare).size)
    except ValueError as e:
        row["error"] = str(e)
        return row
    except Exception as e:                                       # a bug rather than a bad spec: still reported, not fatal
        row["error"] = f"{type(e).__name__}: {e}"
        return row
    via = _alias_of(conn, bare)
    if via is not None:
        row["via"] = via
    return row


def count_spec(conn, spec: str) -> dict:
    """A spec on one file: the whole spec's count, every term's, and which terms are empty (subtractions never are)."""
    out = {"count": None, "error": None, "terms": [], "empty_terms": []}
    try:
        out["count"] = int(conn.select(spec).size)
    except ValueError as e:
        out["error"] = str(e)
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"
    try:
        terms = conn.terms(spec)
    except Exception:
        terms = kitspecs.terms(spec)
    for t in terms:
        r = count_term(conn, t)
        out["terms"].append(r)
        if not r["negate"] and (r["count"] == 0 or r["error"]):
            out["empty_terms"].append(r["term"])
    return out


def audit(conns: dict, spec_list) -> list[dict]:
    """The audit: ``conns`` is {file name: Connectome}; ``spec_list`` holds (where, spec) pairs or plain specs."""
    rows = []
    for item in spec_list:
        where, spec = item if isinstance(item, tuple) else ("", item)
        rows.append({"where": where, "spec": spec, "files": {name: count_spec(conn, spec) for name, conn in conns.items()}})
    return rows


def has_zero(row: dict) -> bool:
    return any(f["empty_terms"] or f["error"] for f in row["files"].values())


def summarise(rows: list[dict], names) -> dict:
    """Per file: how many specs have every term non-empty, how many have an empty term, and those terms."""
    out = {}
    for name in names:
        empty = collections.OrderedDict()
        ok = bad = 0
        for row in rows:
            f = row["files"].get(name)
            if f is None:
                continue
            if f["empty_terms"] or f["error"]:
                bad += 1
                for t in f["empty_terms"]:
                    empty.setdefault(t, []).append(row["where"] or row["spec"])
                if f["error"] and not f["empty_terms"]:
                    empty.setdefault(f"(whole spec) {row['spec']}", []).append(f["error"])
            else:
                ok += 1
        out[name] = {"specs_all_terms_present": ok, "specs_with_an_empty_term": bad, "empty_terms": dict(empty)}
    return out


# ---------------------------------------------------------------------------------------------- BANC candidates
NAME_COLUMNS = ("malecns_cell_type", "manc_cell_type", "fafb_cell_type", "cell_type")


def _banc_malecns_counts(conn):
    """{column: {value: cells in the file}} for BANC's name columns (malecns, manc, fafb, its own cell_type) from the
    deposit's metadata table, or None without it (a note); ``auto:`` proposals left out."""
    try:
        from virtual_fly.banc import SOURCES, SOURCE_DIR
        import pyarrow.feather as feather
    except ImportError:
        return None, "pyarrow is not installed"
    path = SOURCE_DIR / SOURCES["meta"]["file"]
    if not path.exists():
        return None, f"no metadata table at {path} (python -m virtual_fly.banc downloads it)"
    t = feather.read_table(str(path), columns=["root_id", *NAME_COLUMNS])
    ids = t.column("root_id").to_pylist()
    present = set(int(b) for b in conn.body_id.tolist())
    keep = []
    for i, rid in enumerate(ids):
        try:
            keep.append(int(rid) in present)
        except ValueError:
            keep.append(False)
    counts = {}
    for col in NAME_COLUMNS:
        vals = t.column(col).to_pylist()
        c = collections.Counter()
        for ok, m in zip(keep, vals):
            if ok and m and not str(m).startswith("auto:"):
                c[m] += 1
        counts[col] = c
    return counts, None


def candidates(conn, term: str, malecns_counts=None) -> dict:
    """What an empty term might have meant on this file: types containing it, types with its first four characters,
    and (BANC) the neurons whose malecns_cell_type is the term."""
    bare = term[1:].strip() if term.startswith("!") else term.strip()
    name = bare.split(":", 1)[1] if ":" in bare and not bare.startswith(("prefix:", "contains:", "regex:")) else bare
    if bare.startswith(("prefix:", "contains:", "regex:")):
        name = bare.split(":", 1)[1]
    if len(name) > 2 and name[-2] == "/" and name[-1] in "LRM":
        name = name[:-2]
    key = name.lower()
    types = [t for t in conn.tables["types"] if t]
    contains = [t for t in types if key and key in t.lower()]
    same_start = [t for t in types if len(key) >= 4 and t.lower().startswith(key[:4]) and t not in contains]
    out = {"types_containing": contains[:MAX_CANDIDATES], "types_containing_n": len(contains),
           "types_same_start": same_start[:MAX_CANDIDATES], "types_same_start_n": len(same_start)}
    if malecns_counts is not None:                           # BANC's name columns: equal to the term, or containing it
        out["malecns_cell_type_cells"] = int(malecns_counts["malecns_cell_type"].get(name, 0))
        for col in ("manc_cell_type", "fafb_cell_type"):
            out[f"{col}_cells"] = int(malecns_counts[col].get(name, 0))
            hits = sorted((v, n) for v, n in malecns_counts[col].items() if key and key in v.lower())
            out[f"{col}_containing"] = [f"{v} ({n})" for v, n in hits[:MAX_CANDIDATES]]
    return out


# ---------------------------------------------------------------------------------------------- printing
def _cell(f: dict) -> str:
    if f is None:
        return "–"
    if f["error"] and f["count"] is None:
        return f"error: {f['error'][:60]}"
    parts = []
    for r in f["terms"]:
        c = "err" if r["error"] else str(r["count"])
        mark = "!" if (not r["negate"] and (r["count"] == 0 or r["error"])) else ""
        via = f" via {r['via'][:40]}" if r["via"] else ""
        parts.append(f"{r['term']}={c}{mark}{via}")
    return f"{f['count']} ({'; '.join(parts)})" if len(f["terms"]) > 1 or f["terms"] and (f["terms"][0]["via"] or f["terms"][0]["negate"]) else f"{f['count']}"


def render_text(rows, names, summary, notes, cands) -> str:
    lines = []
    for name, why in notes.items():
        lines.append(f"[{name}] not audited: {why}")
    width = max((len(r["spec"]) for r in rows), default=10)
    for row in rows:
        cells = "  ".join(f"{name}: {_cell(row['files'].get(name))}" for name in names)
        lines.append(f"{row['spec']:{min(width, 48)}s}  {cells}    [{row['where']}]")
    lines.append("")
    for name in names:
        s = summary.get(name)
        if s is None:
            continue
        lines.append(f"== {name}: {s['specs_all_terms_present']} specs with every term present, "
                     f"{s['specs_with_an_empty_term']} with an empty term or an error ==")
        for term, where in s["empty_terms"].items():
            lines.append(f"  {term}   <- {', '.join(sorted(set(where)))[:160]}")
            c = cands.get(name, {}).get(term)
            if c:
                lines.append(f"      candidates: containing {c['types_containing']}{' …' if c['types_containing_n'] > MAX_CANDIDATES else ''}; "
                             f"same start {c['types_same_start']}{' …' if c['types_same_start_n'] > MAX_CANDIDATES else ''}"
                             + (f"; malecns_cell_type == term: {c['malecns_cell_type_cells']} cells" if "malecns_cell_type_cells" in c else "")
                             + _name_columns(c))
    return "\n".join(lines)


def _name_columns(c: dict) -> str:
    """BANC's manc and fafb name columns: equal to the term, or containing it (value (cells))."""
    bits = []
    for col in ("manc_cell_type", "fafb_cell_type"):
        if f"{col}_cells" in c:
            bits.append(f"{col} == term: {c[f'{col}_cells']} cells, containing: {c[f'{col}_containing'] or '[]'}")
    return ("; " + "; ".join(bits)) if bits else ""


def render_markdown(rows, names, summary, notes, cands) -> str:
    lines = ["# Alias audit: every spec the kit names, counted on every file", ""]
    for name, why in notes.items():
        lines.append(f"- `{name}` not audited: {why}")
    lines += ["", "| spec | where | " + " | ".join(names) + " |", "|---|---|" + "---|" * len(names)]
    for row in rows:
        cells = " | ".join(_cell(row["files"].get(name)).replace("|", "\\|") for name in names)
        lines.append(f"| `{row['spec']}` | {row['where']} | {cells} |")
    lines.append("")
    lines.append("`a=0!` marks a term that selects nothing on that file; `via` names the alias the file answered through.")
    for name in names:
        s = summary.get(name)
        if s is None:
            continue
        lines += ["", f"## {name}: {s['specs_all_terms_present']} specs with every term present, "
                      f"{s['specs_with_an_empty_term']} with an empty term or an error", ""]
        if not s["empty_terms"]:
            lines.append("No empty term.")
        for term, where in s["empty_terms"].items():
            lines.append(f"- `{term}` (used by: {', '.join(sorted(set(where)))[:200]})")
            c = cands.get(name, {}).get(term)
            if c:
                lines.append(f"  - candidates: containing `{c['types_containing']}`{' …' if c['types_containing_n'] > MAX_CANDIDATES else ''}; "
                             f"same start `{c['types_same_start']}`{' …' if c['types_same_start_n'] > MAX_CANDIDATES else ''}"
                             + (f"; `malecns_cell_type` equal to the term: {c['malecns_cell_type_cells']} cells" if "malecns_cell_type_cells" in c else "")
                             + _name_columns(c).replace("[", "`[").replace("]", "]`"))
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("specs", nargs="*", help="extra specs to count beside the kit's own")
    ap.add_argument("--files", default=",".join(FILES), help="comma list of " + ", ".join(FILES) + " (default: all three)")
    ap.add_argument("--only-zeros", action="store_true", help="only the rows where some file has an empty term or an error")
    ap.add_argument("--no-kit", action="store_true", help="only the specs given on the command line")
    ap.add_argument("--json", type=Path, help="write every row, the summary and the candidates here")
    ap.add_argument("--markdown", action="store_true", help="print a markdown table instead of text")
    args = ap.parse_args(argv)
    names = [n.strip() for n in args.files.split(",") if n.strip()]
    conns, notes = load_files(names)
    present = [n for n in names if n in conns]
    spec_list = [] if args.no_kit else list(kitspecs.kit_specs().items())
    spec_list += [("command line", s) for s in args.specs]
    rows = audit(conns, spec_list)
    summary = summarise(rows, present)
    cands: dict = {}
    for name in present:
        counts, why = (None, None)
        if name == "banc":
            counts, why = _banc_malecns_counts(conns[name])
            if why:
                notes[f"{name} candidates"] = why
        cands[name] = {term: candidates(conns[name], term, counts) for term in summary[name]["empty_terms"]
                       if not term.startswith("(whole spec)")}
    shown = [r for r in rows if has_zero(r)] if args.only_zeros else rows
    print(render_markdown(shown, present, summary, notes, cands) if args.markdown
          else render_text(shown, present, summary, notes, cands))
    if args.json:
        args.json.write_text(json.dumps({"files": present, "notes": notes, "rows": rows, "summary": summary,
                                         "candidates": cands}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
