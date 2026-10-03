"""The compiled integrator must reproduce the NumPy one spike for spike, on every mechanism."""

import numpy as np
import pytest

from brain_backends import CONFIGS, _run, assert_backends_agree       # the configurations, the run and the comparison every backend shares
from virtual_fly import fastbrain
from virtual_fly.brain import FlyBrain

pytestmark = pytest.mark.skipif(not fastbrain.available(), reason="numba is not installed")


@pytest.mark.parametrize("name", sorted(CONFIGS))
def test_numba_matches_numpy_spike_for_spike(conn, name):
    cfg = CONFIGS[name]
    a, rec_a = _run(conn, "numpy", cfg)
    b, rec_b = _run(conn, "numba", cfg)
    assert a.backend == "numpy" and b.backend == "numba"
    assert_backends_agree(conn, cfg, name, a, rec_a, b, rec_b)


def test_numba_matches_numpy_with_the_curated_parts_list(conn, mini_vfb):
    from virtual_fly.parts import PartsList
    cfg = {"parts": PartsList(curated="all"), "fatigue_mv": 0.05}
    a, rec_a = _run(conn, "numpy", cfg)
    b, rec_b = _run(conn, "numba", cfg)
    assert a.total_spikes == b.total_spikes > 100 and len(rec_a) == len(rec_b)
    for (ta, sa), (tb, sb) in zip(rec_a, rec_b):
        assert ta == tb and np.array_equal(sa, sb)
    assert np.array_equal(a._mod_gain, b._mod_gain) and np.array_equal(a._mod_level, b._mod_level)
    assert a.parts.counts["curated"]["neurons"] > 0 and (a.parts.mod_sign < 0).any()


def test_numba_matches_numpy_with_learning(conn):
    a, rec_a = _run(conn, "numpy", {"fatigue_mv": 0.05, "kenyon_gain": 1.0}, learning=True)
    b, rec_b = _run(conn, "numba", {"fatigue_mv": 0.05, "kenyon_gain": 1.0}, learning=True)
    assert a.total_spikes == b.total_spikes
    assert all(ta == tb and np.array_equal(sa, sb) for (ta, sa), (tb, sb) in zip(rec_a, rec_b))
    assert np.array_equal(a.w, b.w) and a.plasticity.depressed_fraction() == b.plasticity.depressed_fraction()


def test_numba_backend_selection_and_settings(conn):
    auto = FlyBrain(conn)
    assert auto.backend == "numba" and auto.settings()["backend"] == "numba"
    assert FlyBrain(conn, backend="numpy").backend == "numpy"
    with pytest.raises(ValueError):
        FlyBrain(conn, backend="cuda")


def test_numba_silence_modulate_and_snapshot(conn):
    b = FlyBrain(conn, backend="numba", fatigue_mv=0.05)
    b.stimulate("LB3b,LB3c", 120)
    b.run(100)
    assert b.rate("MN9") > 20
    b.silence("GNG232")
    b.reset_counts()
    b.run(100)
    assert b.rate("MN9") == 0
    b.unsilence()
    b.modulate("GNG232", 0.0)
    b.reset_counts()
    b.run(100)
    assert b.rate("MN9") == 0
    b.unmodulate()
    snap = b.snapshot()
    b.start_recording()
    b.run(80)
    first = b.stop_recording()
    b.restore(snap)
    b.start_recording()
    b.run(80)
    second = b.stop_recording()
    assert len(first) == len(second) > 0
    assert all(ta == tb and np.array_equal(sa, sb) for (ta, sa), (tb, sb) in zip(first, second))


def test_numba_kernels_handle_refractory_forced_and_empty_steps(conn):
    b = FlyBrain(conn, backend="numba")
    assert b.step().size == 0 and b.quiet                      # the quiet path never calls the kernel
    b.stimulate("LB3b", 2000)                                  # p = 1: forced every step, even while refractory
    spikes = [b.step() for _ in range(6)]
    idx = conn.select("LB3b")
    assert all(np.array_equal(s, idx) for s in spikes)
    assert b.spike_count[idx].min() == 6
    # a forced neuron that also crossed threshold is counted once
    b.clear_stimuli()
    b.stimulate("LB3b,LB3c", 120)
    for _ in range(200):
        s = b.step()
        assert np.array_equal(s, np.unique(s))
