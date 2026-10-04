"""tools/alias_audit.py: every term of a spec counted on its own, so that a union with one empty member cannot hide behind
the others' cells (docs/TWO_FLIES_PLAN.md 9.5). On the synthetic connectome."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import alias_audit as A  # noqa: E402


def test_every_term_is_counted_on_its_own(conn):
    n_mn9 = conn.select("MN9").size
    assert n_mn9 > 0
    rows = A.audit({"syn": conn}, [("a readout", "MN9"), ("a union", "MN9,NoSuchType"), ("a prefix", "prefix:GNG"),
                                   ("a subtraction", "prefix:GNG,!GNG232"), "bogus:x"])
    by = {r["spec"]: r["files"]["syn"] for r in rows}
    plain = by["MN9"]
    assert plain["count"] == n_mn9 and plain["error"] is None and plain["empty_terms"] == []
    assert [t["term"] for t in plain["terms"]] == ["MN9"] and plain["terms"][0]["count"] == n_mn9
    union = by["MN9,NoSuchType"]
    assert union["count"] == n_mn9                                  # the union's count hides the empty member ...
    assert union["empty_terms"] == ["NoSuchType"]                   # ... which the audit names
    assert [(t["term"], t["count"]) for t in union["terms"]] == [("MN9", n_mn9), ("NoSuchType", 0)]
    prefix = by["prefix:GNG"]
    assert prefix["count"] > 0 and prefix["empty_terms"] == [] and prefix["terms"][0]["via"] is None
    sub = by["prefix:GNG,!GNG232"]
    assert sub["terms"][1]["negate"] and sub["terms"][1]["count"] == conn.select("GNG232").size
    assert sub["empty_terms"] == []                                 # a subtraction is never called empty
    assert sub["count"] == prefix["count"] - conn.select("GNG232").size
    bad = by["bogus:x"]
    assert bad["count"] is None and "unknown filter" in bad["error"]   # reported, not raised
    assert bad["terms"][0]["error"] and bad["empty_terms"] == ["bogus:x"]
    assert rows[0]["where"] == "a readout" and rows[-1]["where"] == ""
    assert [A.has_zero(r) for r in rows] == [False, True, False, False, True]


def test_the_summary_and_the_candidates(conn):
    rows = A.audit({"syn": conn}, [("ok", "MN9"), ("half empty", "MN9,NoSuchType"), ("bad", "bogus:x")])
    s = A.summarise(rows, ["syn"])["syn"]
    assert s["specs_all_terms_present"] == 1 and s["specs_with_an_empty_term"] == 2
    assert set(s["empty_terms"]) == {"NoSuchType", "bogus:x"} and s["empty_terms"]["NoSuchType"] == ["half empty"]
    c = A.candidates(conn, "GNG23", None)
    assert "GNG232" in c["types_containing"] and "malecns_cell_type_cells" not in c
    counts = {"malecns_cell_type": {"NoSuchType": 3}, "manc_cell_type": {"NoSuchType_a": 2}, "fafb_cell_type": {}, "cell_type": {}}
    c = A.candidates(conn, "NoSuchType", counts)
    assert c["types_containing"] == [] and c["malecns_cell_type_cells"] == 3
    assert c["manc_cell_type_cells"] == 0 and c["manc_cell_type_containing"] == ["NoSuchType_a (2)"] and c["fafb_cell_type_containing"] == []
    text = A.render_text(rows, ["syn"], A.summarise(rows, ["syn"]), {}, {"syn": {}})
    assert "NoSuchType=0!" in text and "bogus:x" in text
    md = A.render_markdown(rows, ["syn"], A.summarise(rows, ["syn"]), {"other": "no file"}, {"syn": {}})
    assert md.startswith("# Alias audit") and "`other` not audited: no file" in md and "| `MN9` | ok |" in md


def test_an_alias_is_named(conn, tmp_path):
    """A file whose aliases map a kit name to another spec answers through it, and the audit says so."""
    import copy
    c = copy.copy(conn)
    c.aliases = {"KitName": "MN9"}
    c._cache = {}
    r = A.count_term(c, "KitName")
    assert r["count"] == conn.select("MN9").size and r["via"] == "MN9"
    assert A.count_term(c, "MN9")["via"] is None                  # a real type of the same name wins, no alias named
    assert A._alias_of(c, "KitName/L") == "MN9"
