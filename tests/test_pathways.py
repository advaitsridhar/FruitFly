"""Static pathway tracing on the cell-type graph."""

import pytest

from virtual_fly.pathways import Hop, Path, relay_ranking, strongest_partners, trace


def test_trace_finds_the_designed_steering_route(conn):
    paths = trace(conn, "LC10a/L", "DNa02/L")
    assert paths and all(isinstance(p, Path) for p in paths)
    best = paths[0]
    assert best.nodes == ["LC10a/L", "AOTU019/L", "DNa02/L"]
    assert len(best.hops) == 2 and all(isinstance(h, Hop) for h in best.hops)
    h1, h2 = best.hops
    n_syn = conn.synapses_between("LC10a/L", "AOTU019/L")
    assert (h1.src, h1.dst, h1.synapses, h1.sign, h1.src_size, h1.dst_size) == ("LC10a/L", "AOTU019/L", n_syn, 1, 10, 4)
    assert h1.fraction == pytest.approx(1.0)                                # LC10a is AOTU019's only input
    assert (h2.src, h2.dst, h2.synapses) == ("AOTU019/L", "DNa02/L", 200)
    assert h2.fraction == pytest.approx(200 / 230)
    assert best.score == pytest.approx(h1.fraction * h2.fraction) and best.net_sign == 1
    assert [p.score for p in paths] == sorted((p.score for p in paths), reverse=True)
    assert paths[1].nodes == ["LC10a/L", "AOTU025/L", "DNa02/L"] and paths[1].score < 0.1
    assert f"LC10a/L -(+)-> AOTU019/L [{n_syn} syn, 100.0% of input]" in best.describe() and "net excitatory" in best.describe()
    d = best.to_dict()
    assert d["nodes"] == best.nodes and d["net_sign"] == 1 and d["hops"][0]["synapses"] == n_syn
    assert not any("DNa02/R" in p.nodes for p in trace(conn, "LC10a/L", "DNa02"))   # nothing crosses the midline


def test_trace_avoid_and_hop_limits(conn):
    around = trace(conn, "LC10a/L", "DNa02/L", avoid="AOTU019")
    assert [p.nodes for p in around] == [["LC10a/L", "AOTU025/L", "DNa02/L"]]
    assert trace(conn, "LC10a/L", "DNa02/L", avoid="AOTU019,AOTU025") == []
    assert trace(conn, "LC10a/L", "DNa02/L", max_hops=1) == []                # needs two hops
    assert trace(conn, "LC10a/L", "DNa02/L", min_fraction=0.5)[0].nodes == ["LC10a/L", "AOTU019/L", "DNa02/L"]
    assert len(trace(conn, "LC10a/L", "DNa02/L", min_fraction=0.5)) == 1     # the weak relay is filtered out
    assert len(trace(conn, "LB1a", "MN9", top=3)) == 3
    with pytest.raises(ValueError, match="no neurons match"):
        trace(conn, "nothing", "MN9")
    with pytest.raises(ValueError, match="no neurons match"):
        trace(conn, "MN9", "nothing")


def test_trace_signs_and_excitatory_only(conn):
    bitter = trace(conn, "LB1a", "MN9")
    assert bitter and all(p.net_sign == -1 for p in bitter)
    assert all("GNG087/L" in p.nodes or "GNG087/R" in p.nodes for p in bitter)
    gaba_hops = [h for p in bitter for h in p.hops if h.src.startswith("GNG087")]
    assert gaba_hops and all(h.sign == -1 for h in gaba_hops)
    assert "-(-)->" in bitter[0].describe() and "net inhibitory" in bitter[0].describe()
    assert trace(conn, "LB1a", "MN9", excitatory_only=True) == []
    sugar = trace(conn, "LB3b", "MN9", excitatory_only=True)
    assert sugar and sugar[0].nodes[1].startswith("GNG232") and all(p.net_sign == 1 for p in sugar)
    two_inhibitory = [p for p in trace(conn, "LB1a", "MN9", max_hops=5, top=60) if "GNG_M1/M" in p.nodes]
    assert two_inhibitory and all(p.net_sign == 1 for p in two_inhibitory)  # GABA -> GABA flips back


def test_relay_ranking(conn):
    paths = trace(conn, "LC10a/L", "DNa02/L")
    ranking = relay_ranking(paths)
    assert [name for name, _ in ranking] == ["AOTU019/L", "AOTU025/L"]
    assert ranking[0][1] == pytest.approx(paths[0].score) and ranking[1][1] < ranking[0][1]
    assert relay_ranking([]) == []
    bitter = relay_ranking(trace(conn, "LB1a", "MN9"))
    assert bitter[0][0].startswith("GNG087") and bitter[0][1] == pytest.approx(
        sum(p.score for p in trace(conn, "LB1a", "MN9") if bitter[0][0] in p.nodes))


def test_strongest_partners(conn):
    ins = strongest_partners(conn, "MN9/L", "in", top=10)
    assert ins[0]["type"] == "GNG232" and ins[0]["side"] in ("L", "R")
    tg = conn.type_graph()
    for r in ins:
        node = tg.index[f"{r['type']}/{r['side']}"]
        assert r["fraction"] == pytest.approx(r["synapses"] / tg.out_total[node])
    outs = strongest_partners(conn, "GNG232/L", "out", top=10)
    mn9 = next(r for r in outs if r["type"] == "MN9" and r["side"] == "L")
    assert mn9["fraction"] == pytest.approx(mn9["synapses"] / tg.in_total[tg.index["MN9/L"]])
    assert mn9["fraction"] == pytest.approx(160 / 470)
    flat = strongest_partners(conn, "GNG232", "out", top=3, by_side=False)
    assert flat[0]["side"] is None and flat[0]["type"] == "MN9"
    both = sum(tg.in_total[tg.index[n]] for n in ("MN9/L", "MN9/R") if n in tg.index)
    assert flat[0]["fraction"] == pytest.approx(flat[0]["synapses"] / both)   # summed over both sides
    assert strongest_partners(conn, "ORN_DM1", "in") == []
