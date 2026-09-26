"""Connectome loading, population specs, partner summaries and the cell-type graph."""

import gzip

import numpy as np
import pytest

from virtual_fly.connectome import Connectome, TypeGraph, load_connectome


# ------------------------------------------------------------------ loading
def test_loads_synthetic_file(conn, synthetic_path):
    assert conn.path == synthetic_path
    assert conn.dataset == "synthetic-fly:test"
    assert conn.meta["source"].startswith("synthetic")
    assert conn.n == len(conn.types) == conn.row_ptr.size - 1 > 500
    assert conn.n_edges == conn.post_idx.size == conn.n_syn.size == conn.row_ptr[-1]
    assert set(conn.tables) == set(Connectome.TABLES)
    assert conn.tables["types"][0] == ""                       # index 0 = unannotated
    assert conn.soma.shape == (conn.n, 3)
    assert np.isnan(conn.soma[:, 0]).any() and (~np.isnan(conn.soma[:, 0])).any()
    assert conn.n_syn.min() >= 5
    assert set(np.unique(conn.sign).tolist()) == {-1.0, 1.0}
    assert (conn.sign[conn.nt == "gaba"] == -1).all() and (conn.sign[conn.nt == "acetylcholine"] == 1).all()


def test_load_connectome_with_explicit_path_does_not_download(synthetic_path, monkeypatch, capsys):
    import virtual_fly.connectome as C
    monkeypatch.setattr(C, "download_connectome", lambda *a, **k: pytest.fail("tried to download"))
    c = load_connectome(synthetic_path, quiet=False)
    assert c.n > 0
    assert "Loaded synthetic-fly:test" in capsys.readouterr().err


def test_rejects_non_flyb_and_wrong_version(tmp_path):
    bad = tmp_path / "bad.flyb"
    bad.write_bytes(b"NOPE" + b"\0" * 32)
    with pytest.raises(ValueError, match="not a FLYB"):
        Connectome(bad)
    v2 = tmp_path / "v2.flyb.gz"
    with gzip.open(v2, "wb") as f:
        f.write(b"FLYB" + (2).to_bytes(4, "little") + b"\0" * 12)
    with pytest.raises(ValueError, match="unsupported FLYB version"):
        Connectome(v2)


# ------------------------------------------------------------------ select()
@pytest.mark.parametrize("spec, expected", [
    ("MN9", 2), ("prefix:LC1", 28), ("prefix:LC10", 20), ("contains:C10", 20), ("regex:^LC1[01]", 28),
    ("class:Kenyon_Cell", 60), ("superclass:descending_neuron", lambda c: (c.superclass == "descending_neuron").sum()),
    ("nt:gaba", lambda c: (c.nt == "gaba").sum()), ("subclass:wind_gravity", 24), ("nerve:ADMN", 12),
    ("neuromere:T2", lambda c: (c.neuromere == "T2").sum()), ("dimorphism:sexually dimorphic", 2), ("frudsx:fru_high", 2),
    ("LB3b,LB3c", 20), ("MN9/L", 1), ("GNG_M1/M", 2), ("prefix:LC10/R", 10), ("class:DAN/L", 6),
    ("class:Kenyon_Cell&subclass:gamma", 40), ("class:Kenyon_Cell&nt:gaba", 0), ("superclass:sensory&nerve:AN/L", lambda c: ((c.superclass == "sensory") & (c.nerve == "AN") & (c.side == "L")).sum()),
    ("prefix:LC1,!LC10a", 8), ("class:MBON,!nt:glutamate", 6), ("!MN9", 0), ("all", None), ("", 0), (" MN9 , , LB3b ", 12),
    ("gene:fru", 30), ("gene:dsx", 10), ("gene:both", 8), ("gene:fru_high", 2), ("gene:fruitless&class:descending", 2),
    ("gene:VGlut", lambda c: (c.nt == "glutamate").sum()), ("gene:gad1", lambda c: (c.nt == "gaba").sum()),
    ("dimorphism:male", 10), ("dimorphism:dimorphic", 2), ("dimorphism:any", 12), ("dimorphism:male-specific", 10),
])
def test_select_specs(conn, spec, expected):
    idx = conn.select(spec)
    assert idx.dtype == np.int64
    assert np.array_equal(idx, np.unique(idx))                   # sorted and unique
    if spec == "all":
        assert idx.size == conn.n
    else:
        assert idx.size == (expected(conn) if callable(expected) else expected), spec


def test_select_type_filter_matches_exactly(conn):
    lc10a = conn.select("LC10a")
    assert set(conn.types[lc10a]) == {"LC10a"}
    assert conn.select("LC10").size == 0                       # not a prefix match
    assert set(conn.side[conn.select("LC10a/L")]) == {"L"}
    assert conn.select("LC10a/L").size + conn.select("LC10a/R").size == lc10a.size


def test_select_type_names_containing_commas(conn):
    idx = conn.select("DNp51,DNpe019")                              # the whole spec is one type name
    assert idx.size == 2 and set(conn.types[idx]) == {"DNp51,DNpe019"}
    assert conn.select("DNp51,DNpe019/L").size == 1 and set(conn.side[conn.select("DNp51,DNpe019/R")]) == {"R"}
    assert conn.count("DNp51,DNpe019") == 2 and conn.select("prefix:DNp51").size == 2
    assert conn.select("DNp51").size == 0 and conn.select("DNpe019").size == 0


def test_select_body_and_index(conn):
    i = 17
    assert conn.select(f"index:{i}").tolist() == [i]
    assert conn.select(f"body:{conn.body_id[i]}").tolist() == [i]
    assert conn.select("body:1").size == 0
    for spec, message in (("body:abc", "body: needs a neuron's id"), ("body:", "body: needs"), ("body:12.5", "body: needs"),
                          ("index:x", "index: needs a neuron's number"), ("hex:12", "hex: needs two medulla column numbers"),
                          ("hex:1:2:3", "hex: needs"), (f"index:{conn.n}", f"out of range: this fly's neurons are numbered 0 to {conn.n - 1:,}"),
                          ("index:-1", "out of range")):                  # not int()'s words, nor numpy's wrap-around
        with pytest.raises(ValueError, match=message):
            conn.select(spec)
    with pytest.raises(ValueError, match="not a regular expression"):   # re.error becomes a ValueError too
        conn.select("regex:[")
    assert conn.count("regex:[") == 0


def test_select_accepts_arrays_and_iterables(conn):
    arr = np.array([3, 1, 2])
    assert conn.select(arr) is arr or np.array_equal(conn.select(arr), arr)
    assert conn.select([5, 6]).tolist() == [5, 6]
    assert conn.select(range(3)).tolist() == [0, 1, 2]


def test_unknown_gene_raises(conn):
    with pytest.raises(ValueError, match="unknown gene"):
        conn.select("gene:notagene")


def test_select_unknown_filter_raises(conn):
    with pytest.raises(ValueError, match="unknown filter"):
        conn.select("colour:red")
    assert conn.count("colour:red") == 0                       # count() swallows the error
    assert conn.count("MN9") == 2


def test_select_is_cached(conn):
    a = conn.select("class:MBON")
    assert conn.select("class:MBON") is a


def test_select_hex_column(conn):
    idx = conn.select("hex:1:1")
    assert idx.size == 16                                        # 8 columnar reference types x 2 eyes
    assert (conn.hex1[idx] == 1).all() and (conn.hex2[idx] == 1).all()


def test_hex_coordinates_form_a_lattice(conn):
    mi1 = conn.select("Mi1/L")
    cols = set(zip(conn.hex1[mi1].tolist(), conn.hex2[mi1].tolist()))
    assert cols == {(h1, h2) for h1 in range(3) for h2 in range(2)}
    assert (conn.hex1[conn.select("MN9")] == -1).all() and (conn.hex1[conn.select("T4a")] == -1).all()


# ------------------------------------------------------------------ names, counts, descriptions
def test_find_types_and_type_counts(conn):
    hits = conn.find_types("lc10")
    assert hits[0] == ("LC10a", 20)                              # prefix matches sort first
    assert ("LC11", 8) not in hits and ("LPLC2", 16) not in hits
    assert [t for t, _ in conn.find_types("MBON")] == sorted(t for t in conn.type_counts() if "MBON" in t)
    assert conn.find_types("mbon", limit=2) == conn.find_types("MBON")[:2]
    counts = conn.type_counts()
    assert "" not in counts
    assert sum(counts.values()) == int((conn.types != "").sum()) == conn.n - 20
    assert counts["KCg-m"] == 40 and counts["MN9"] == 2


def test_find_types_lists_the_aliases(conn):
    c = conn.rewired(conn.row_ptr, conn.post_idx, conn.n_syn, label="aliased")
    c.aliases = {"MN9x": "MN9", "prefix:MN9y": "MN9", "MN9": "GNG232", "NOPEx": "NOPE"}
    assert c.find_types("mn9") == [("MN9", 2), ("MN9x", 2)]       # a real type wins; spec aliases and empty ones are left out
    assert c.find_types("9x") == [("MN9x", 2)] and conn.find_types("mn9") == [("MN9", 2)]


def test_describe_and_info(conn):
    i = int(conn.select("MN9/L")[0])
    text = conn.describe(i)
    assert text.startswith(f"#{i} bodyId {conn.body_id[i]}") and "type MN9" in text and "side L" in text
    assert "cb_motor/motor" in text and "acetylcholine (+)" in text
    info = conn.info(i)
    assert info["type"] == "MN9" and info["side"] == "L" and info["sign"] == 1 and info["nt"] == "acetylcholine"
    assert info["n_outputs"] == 0 and info["n_inputs"] == conn.in_edges(np.array([i])).size
    assert info["n_post_synapses"] == int(conn.n_syn[conn.in_edges(np.array([i]))].sum())
    assert info["hex"] is None and len(info["soma"]) == 3
    j = int(conn.select("Mi1/L")[0])
    assert conn.info(j)["hex"] == [int(conn.hex1[j]), int(conn.hex2[j])]
    unannotated = int(np.flatnonzero(conn.types == "")[0])
    u = conn.info(unannotated)
    assert u["soma"] is None and u["type"] == "" and u["n_inputs"] == u["n_outputs"] == 0
    assert "type ?" in conn.describe(unannotated)


# ------------------------------------------------------------------ edges
def test_out_edges_and_in_edges_are_consistent(conn):
    for j in [int(conn.select("MN9/L")[0]), int(conn.select("GNG232/R")[0]), 0]:
        ins = conn.in_edges(np.array([j]))
        assert np.array_equal(np.sort(ins), np.flatnonzero(conn.post_idx == j))
        outs = conn.out_edges(np.array([j]))
        assert np.array_equal(outs, np.arange(conn.row_ptr[j], conn.row_ptr[j + 1]))
        assert (conn.pre_idx[outs] == j).all()
    assert conn.out_edges(np.flatnonzero(conn.types == "")).size == 0
    assert conn.in_edges(np.zeros(0, dtype=np.int64)).size == 0
    # every edge appears exactly once in the column index
    assert np.array_equal(np.sort(conn.in_edge_order), np.arange(conn.n_edges))
    assert conn.col_ptr[-1] == conn.n_edges


def test_inputs_of_and_outputs_of_synapse_sums(conn):
    rows = conn.inputs_of("MN9", top=50)
    by_type = {r["type"]: r for r in rows}
    assert set(by_type) == {"GNG232", "GNG087", "DNg67"}
    assert rows[0]["type"] == "GNG232" and rows[0]["sign"] == 1
    assert by_type["GNG087"]["sign"] == -1 and by_type["GNG087"]["nt"] == "gaba"
    assert sum(r["synapses"] for r in rows) == int(conn.n_syn[conn.in_edges(conn.select("MN9"))].sum())
    assert by_type["GNG232"]["neurons"] == 4 and by_type["GNG232"]["connections"] == 8
    assert by_type["GNG232"]["synapses"] == conn.synapses_between("GNG232", "MN9")
    sided = conn.inputs_of("MN9/L", by_side=True)
    assert {(r["type"], r["side"]) for r in sided} >= {("GNG232", "L"), ("GNG232", "R"), ("GNG087", "L")}
    assert rows[0]["side"] is None
    outs = conn.outputs_of("LB3b", top=3)
    assert outs[0]["type"] == "GNG232" and len(outs) == 1
    assert conn.inputs_of("ORN_DM1") == []                      # sensory neurons have no inputs
    assert conn.outputs_of("class:Kenyon_Cell", top=2)[0]["type"] in {"MBON11", "MBON12", "MBON01", "MBON02"}


def test_synapses_between_and_edges_between(conn):
    e = conn.edges_between("LC10a/L", "AOTU019/L")
    assert e.size == 10 * 4
    assert set(conn.types[conn.pre_idx[e]]) == {"LC10a"} and set(conn.types[conn.post_idx[e]]) == {"AOTU019"}
    assert conn.synapses_between("LC10a/L", "AOTU019/L") == int(conn.n_syn[e].sum()) > 0
    assert conn.synapses_between("LC10a/L", "AOTU019/R") == 0
    assert conn.synapses_between("MN9", "LB3b") == 0
    assert conn.edges_between("nothing_here", "MN9").size == 0
    assert conn.synapses_between("class:Kenyon_Cell", "class:MBON") == int(
        conn.n_syn[conn.edges_between(conn.select("class:Kenyon_Cell"), conn.select("class:MBON"))].sum())


# ------------------------------------------------------------------ type graph
def test_type_graph_nodes_and_edges(conn):
    tg = conn.type_graph()
    assert isinstance(tg, TypeGraph) and conn.type_graph() is tg
    assert "DNa02/L" in tg.index and "GNG_M1/M" in tg.index and "" not in tg.names
    assert tg.n_nodes == len(tg.names) == len(tg.index)
    assert (tg.node_of_neuron[conn.types == ""] == -1).all()
    assert tg.size[tg.index["LC10a/L"]] == 10 and tg.size.sum() == conn.n - 20
    assert tg.sign[tg.index["GNG087/L"]] == -1 and tg.nt[tg.index["GNG087/L"]] == "gaba"
    assert tg.superclass[tg.index["MN9/R"]] == "cb_motor"
    # summed synapses per node pair match the neuron-level wiring
    a, b = tg.index["AOTU019/L"], tg.index["DNa02/L"]
    dst, w = tg.out(a)
    assert w[dst == b][0] == conn.synapses_between("AOTU019/L", "DNa02/L") == 200
    src, w_in = tg.inp(b)
    assert set(tg.names[s] for s in src) == {"AOTU019/L", "AOTU025/L", "LC11/L"}
    assert tg.in_total[b] == pytest.approx(w_in.sum()) == 230
    assert tg.in_total.sum() == pytest.approx(tg.out_total.sum()) == pytest.approx(tg.weight.sum())
    assert tg.weight.sum() == pytest.approx(conn.n_syn.sum())    # every neuron with a type is in the graph
    assert np.array_equal(tg.nodes_of("LC10a"), np.array(sorted([tg.index["LC10a/L"], tg.index["LC10a/R"]])))
    assert tg.nodes_of("class:nothing").size == 0
    assert tg.row_ptr[-1] == tg.col_ptr[-1] == tg.src.size
