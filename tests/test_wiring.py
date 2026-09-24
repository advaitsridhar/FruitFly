"""Growing synthetic flies from type-level rules, and the genome bottleneck (synthetic connectome)."""

import numpy as np
import pytest

from virtual_fly import wiring as W
from virtual_fly.brain import FlyBrain
from virtual_fly.experiments import Experiment, R, run_experiment


def _csr_is_valid(c):
    assert c.row_ptr.size == c.n + 1 and c.row_ptr[0] == 0 and c.row_ptr[-1] == c.post_idx.size == c.n_syn.size
    assert (np.diff(c.row_ptr) >= 0).all() and c.post_idx.min() >= 0 and c.post_idx.max() < c.n
    assert (c.n_syn >= W.MIN_SYN).all()
    for i in range(0, c.n, max(1, c.n // 50)):
        seg = c.post_idx[c.row_ptr[i]:c.row_ptr[i + 1]]
        assert (np.diff(seg) > 0).all(), "targets sorted and unique per row"
        assert not (seg == i).any(), "no self-connection"


def test_learn_rules_reproduces_the_type_graph(conn):
    rules = W.learn_rules(conn)
    assert rules.level == "type" and rules.n_groups == len(set(f"{t or '?'}/{s or '-'}" for t, s in zip(conn.types, conn.side)))
    assert rules.pair_edges.sum() == conn.n_edges and rules.n_edges_total == conn.n_edges
    g = rules.groups.index("GNG232/L"); h = rules.groups.index("MN9/L")
    k = np.flatnonzero((rules.pair_pre == g) & (rules.pair_post == h))
    assert k.size == 1 and rules.pair_edges[k[0]] == 2                     # both GNG232/L cells reach MN9/L
    assert np.exp(rules.pair_mu[k[0]]) == pytest.approx(80, rel=0.05)     # 80 synapses each in the synthetic data
    assert rules.size_numbers() == 3 * rules.n_pairs
    coarse = W.learn_rules(conn, "class")
    assert coarse.n_groups < rules.n_groups and coarse.pair_edges.sum() == conn.n_edges
    with pytest.raises(ValueError):
        W.learn_rules(conn, "colour")


def test_grow_keeps_the_rules_and_randomises_the_neurons(conn):
    rules = W.learn_rules(conn)
    g1 = W.grow(conn, rules, seed=1)
    _csr_is_valid(g1)
    assert g1.n == conn.n and g1 is not conn and g1.types is conn.types and "grown:type:seed1" in g1.dataset
    assert abs(g1.n_edges - conn.n_edges) < 0.1 * conn.n_edges and abs(int(g1.n_syn.sum()) - int(conn.n_syn.sum())) < 0.1 * conn.n_syn.sum()
    # type-level totals are kept: GNG232 -> MN9 as strong as before
    assert g1.synapses_between("GNG232", "MN9") == pytest.approx(conn.synapses_between("GNG232", "MN9"), rel=0.25)
    assert g1.synapses_between("LB3b,LB3c", "GNG232") == pytest.approx(conn.synapses_between("LB3b,LB3c", "GNG232"), rel=0.25)
    assert g1.synapses_between("MN9", "GNG232") == 0 == conn.synapses_between("MN9", "GNG232")    # no rule, no wire
    g2 = W.grow(conn, rules, seed=2)
    same = W.grow(conn, rules, seed=1)
    assert np.array_equal(same.post_idx, g1.post_idx) and np.array_equal(same.n_syn, g1.n_syn)       # deterministic
    assert not np.array_equal(g2.post_idx, g1.post_idx)                                             # another individual
    cmp = W.compare(conn, g1)
    assert cmp["edges_grown"] == g1.n_edges and 0 < cmp["shared_connections"] < conn.n_edges
    # the rewired copy has fresh wiring-dependent caches and shared annotations
    assert g1.in_edges(conn.select("MN9")).size == g1.n_post[conn.select("MN9")].astype(bool).sum() or g1.in_edges(conn.select("MN9")).size > 0
    assert g1.select("class:Kenyon_Cell").size == conn.select("class:Kenyon_Cell").size
    assert g1.type_graph() is not conn.type_graph()


def test_bottleneck_compresses_and_regrows(conn):
    rules = W.learn_rules(conn)
    b = W.bottleneck(rules, 8)
    assert b.level == "bottleneck:8" and b.rank == 8 and b.n_groups == rules.n_groups
    assert b.size_numbers() == 2 * rules.n_groups * 8 + 2 * rules.n_groups < rules.size_numbers() * 10
    assert 0.5 * rules.n_pairs <= b.n_pairs <= 1.5 * rules.n_pairs                 # re-thresholded to the same order
XX
    g = W.grow(conn, b, seed=1)
    _csr_is_valid(g)
    assert 0.3 * conn.n_edges < g.n_edges < 1.5 * conn.n_edges
    with pytest.raises(ValueError):
        W.bottleneck(b, 4)


def test_grow_level_and_cache(conn):
    cache = {}
    real, r = W.grow_level(conn, "real", rules_cache=cache)
    assert real is conn and r is None and cache == {}
    g, r = W.grow_level(conn, "type", seed=3, rules_cache=cache)
    assert r.level == "type" and "type" in cache and g.n_edges > 0
    g2, r2 = W.grow_level(conn, "bottleneck:4", seed=3, rules_cache=cache)
    assert r2.rank == 4 and "bottleneck:4" in cache and cache["type"] is r
    g3, r3 = W.grow_level(conn, "class", rules_cache=cache)
    assert r3.level == "class"
    with pytest.raises(ValueError):
        W.grow_level(conn, "hemilineage")


def test_a_grown_fly_still_feeds(conn):
    """The feeding reflex is wired type-to-type, so it must survive growth from the rules."""
    g = W.grow(conn, W.learn_rules(conn), seed=5)
    brain = FlyBrain(g, seed=0)
    sugar = Experiment("sugar", {"LB3b,LB3c": 120}, 400, [R("GNG232", "relay", 10, 500), R("MN9", "MN9", 10, 500)])
    res = run_experiment(brain, sugar)
    assert res.ok, [(x.label, x.hz) for x in res.readouts]
    bitter = Experiment("bitter", {"LB1a,LB1b,LB1c,LB1d": 100, "LB3b,LB3c": 120}, 400, [R("MN9", "MN9", 0, 5)])
    assert run_experiment(brain, bitter).ok
