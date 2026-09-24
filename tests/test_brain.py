"""The leaky integrate-and-fire simulator on the synthetic wiring."""

import numpy as np
import pytest

from conftest import drive
from virtual_fly.brain import THETA, FlyBrain, Monitor

SUGAR = {"LB3b,LB3c": 120}
BITTER = {"LB1a,LB1b,LB1c,LB1d": 120}


def test_silence_at_rest(brain):
    brain.run(300)
    assert brain.total_spikes == 0 and brain.spike_count.sum() == 0
    assert brain.time_ms == pytest.approx(300) and brain.t == 600
    assert brain.rate("all") == 0.0 and brain.quiet


def test_stimulated_population_fires_at_requested_poisson_rate(brain):
    brain.stimulate("LB3b,LB3c", 100)
    brain.run(1000)
    assert brain.rate("LB3b,LB3c") == pytest.approx(100, rel=0.1)           # 20 neurons x 1 s: sd ~2 %
    assert brain.active_fraction("LB3b,LB3c") == 1.0
    assert brain.rate("ORN_DM1") == 0.0
    brain.stimulate("LB3b,LB3c", 0)                                          # 0 Hz = stop
    assert brain.stim == {} and brain._stim_idx.size == 0
    with pytest.raises(ValueError, match="no neurons match"):
        brain.stimulate("not_a_type", 50)


def test_sugar_pathway_drives_mn9_and_bitter_suppresses_it(brain):
    sugar = drive(brain, SUGAR).rates(["GNG232", "MN9", "GNG087"])
    assert sugar["GNG232"] > 40 and sugar["MN9"] > 30 and sugar["GNG087"] == 0
    bitter = drive(brain, BITTER).rates({"relay": "GNG087", "mn9": "MN9", "dan": "PPL101"})
    assert bitter["relay"] > 80 and bitter["mn9"] == 0 and bitter["dan"] > 10
    both = drive(brain, {**SUGAR, **BITTER}).rates(["GNG232", "MN9"])
    assert both["MN9"] < 5 and both["GNG232"] < sugar["GNG232"] / 2


def test_silence_blocks_downstream_and_unsilence_restores(brain):
    assert brain.silence("GNG232") == 4
    assert brain.settings()["silenced"] == ["GNG232"]
    assert brain.silenced_mask().sum() == 4
    assert (brain.w[brain.conn.out_edges(brain.conn.select("GNG232"))] == 0).all()
    r = drive(brain, SUGAR).rates(["GNG232", "MN9"])
    assert r["GNG232"] > 40 and r["MN9"] == 0                                # they still fire, nobody hears them
    brain.unsilence("GNG232")
    assert brain.silenced == {} and np.array_equal(brain.w, brain._w_original)
    assert drive(brain, SUGAR).rate("MN9") > 30
    brain.silence("GNG232")
    brain.silence("MN9")
    brain.unsilence()                                                        # None = everything
    assert brain.silenced == {}
    with pytest.raises(ValueError):
        brain.silence("nope")


def test_modulate_scales_output_weights(brain):
    conn = brain.conn
    e = conn.out_edges(conn.select("LB3b,LB3c"))
    assert brain.modulate("LB3b,LB3c", 0.25) == 20
    assert np.allclose(brain.w[e], 0.25 * brain._w_original[e])
    assert brain.settings()["modulated"] == {"LB3b,LB3c": 0.25}
    weak = drive(brain, SUGAR).rate("GNG232")
    brain.modulate("LB3b,LB3c", 2.0)
    strong = drive(brain, SUGAR).rate("GNG232")
    assert strong > weak + 10
    brain.modulate("LB3b,LB3c", 1.0)                                         # factor 1 removes the entry
    assert brain.modulated == {} and np.array_equal(brain.w, brain._w_original)
    brain.modulate("MN9", 3.0)
    brain.unmodulate()
    assert brain.modulated == {}


def test_reset_clears_state_but_keeps_wiring(brain):
    brain.silence("MN9")
    brain.add_monitor("relay", "GNG232", bin_ms=10)
    drive(brain, SUGAR, ms=200)
    assert brain.total_spikes > 0 and len(brain.monitors["relay"].history) > 0
    brain.reset()
    assert brain.t == 0 and brain.total_spikes == 0 and brain.window_ms == 0
    assert not brain.v.any() and not brain.g.any() and not brain.queue.any() and not brain.spike_count.any()
    assert brain.monitors["relay"].history == [] and brain.quiet
    assert brain.silenced == {"MN9": brain.silenced["MN9"]} and brain.stim         # wiring and input kept
    assert (brain.thr == THETA).all()


def test_rates_active_fraction_and_rankings(brain):
    drive(brain, SUGAR, ms=400)
    assert brain.rates(["MN9", "LC4"]) == {"MN9": brain.rate("MN9"), "LC4": 0.0}
    assert brain.rates({"a": "GNG232"})["a"] == brain.rate("GNG232")
    assert brain.rate("nothing") == 0.0 and brain.active_fraction("nothing") == 0.0
    assert brain.active_fraction("GNG232") == 1.0 and brain.active_fraction("LC4") == 0.0
    assert 0 < brain.active_fraction("all") < 0.2
    top = brain.top_neurons(5)
    assert len(top) == 5 and all(hz > 0 for _, hz in top)
    assert [hz for _, hz in top] == sorted((hz for _, hz in top), reverse=True)
    assert top[0][1] == pytest.approx(brain.spike_count[top[0][0]] / (brain.window_ms / 1000))
    assert all(i in set(brain.conn.select("LB3b,LB3c").tolist()) for i, _ in top)
    quiet = brain.top_neurons(5, exclude_stimulated=True)
    assert quiet and all(i not in set(brain.conn.select("LB3b,LB3c").tolist()) for i, _ in quiet)
    types = brain.top_types(3)
    assert {r["type"] for r in types} <= {"LB3b", "LB3c", "GNG232", "MN9"}
    assert set(types[0]) == {"type", "side", "hz", "neurons", "active"} and types[0]["side"] in ("L", "R")
    assert types[0]["hz"] >= types[-1]["hz"]
    excl = brain.top_types(4, exclude_stimulated=True, by_side=False)
    assert {r["type"] for r in excl} == {"GNG232", "MN9"} and all(r["side"] == "" for r in excl)
    relay = next(r for r in excl if r["type"] == "GNG232")
    assert relay["neurons"] == 4 and relay["active"] == 4
    assert brain.top_types(10) != [] and FlyBrain(brain.conn).top_types() == []


def test_monitor_histories_have_correct_bins(brain):
    m = brain.add_monitor("sugar", "LB3b", bin_ms=10)
    assert isinstance(m, Monitor) and brain.monitors["sugar"] is m
    brain.run(100)                                                            # at rest: bins still advance
    assert m.history == [0.0] * 10
    brain.stimulate("LB3b", 100)
    brain.run(200)
    assert len(m.history) == 30 and any(h > 0 for h in m.history[10:])
    total = sum(m.history[10:]) * 10 * (10 / 1000)                           # Hz per neuron x neurons x bin
    assert total == pytest.approx(brain.spike_count[m.idx].sum(), abs=3.0)   # +- the last step
    brain.reset_counts()
    assert brain.spike_count.sum() == 0 and brain.window_ms == 0 and m._count == 0
    brain.run(50)
    assert len(m.history) == 35
    assert sum(m.history[30:]) * 10 * 0.01 == pytest.approx(brain.spike_count[m.idx].sum(), abs=3.0)
    brain.remove_monitor("sugar")
    assert brain.monitors == {}


def test_new_monitor_ignores_spikes_fired_before_it_existed(brain):
    brain.stimulate("LB3b", 200)
    brain.run(100)
    idx = brain.conn.select("LB3b")
    before = int(brain.spike_count[idx].sum())
    assert before > 10
    m = brain.add_monitor("late", "LB3b", bin_ms=10)
    assert m._count == before and m.history == []
    brain.run(10)
    delta = int(brain.spike_count[idx].sum()) - before
    assert m.history and m.history[0] * idx.size * 0.01 == pytest.approx(delta, abs=3.0)
    assert m.history[0] * idx.size * 0.01 < before                        # not the whole backlog


def test_snapshot_restore_rewinds_monitors(brain):
    m = brain.add_monitor("sugar", "LB3b", bin_ms=10)
    brain.stimulate("LB3b", 150)
    brain.run(100)
    snap = brain.snapshot()
    assert snap["monitors"] == {"sugar": (m._count, m._t_start, 10)}
    brain.run(100)
    first = list(m.history)
    assert len(first) == 20
    brain.restore(snap)
    assert len(m.history) == 10 and m.history == first[:10] and m._count == snap["monitors"]["sugar"][0]
    brain.run(100)
    assert m.history == first                                             # bins after the restore repeat exactly
    late = brain.add_monitor("late", "LB3c", bin_ms=10)                   # a monitor the snapshot never saw
    brain.restore(snap)
    assert late.history == [] and late._count == int(brain.spike_count[late.idx].sum())
    assert late._t_start == brain.time_ms


def test_recording(brain):
    assert brain.stop_recording() == []
    brain.stimulate("LB3b,LB3c", 120)
    brain.start_recording()
    brain.run(100)
    rec = brain.stop_recording()
    assert brain.recording is None and len(rec) > 0
    assert sum(s.size for _, s in rec) == brain.total_spikes
    t_ms, idx = brain.recording_arrays(rec)
    assert t_ms.dtype == np.float32 and idx.dtype == np.int32 and t_ms.size == idx.size == brain.total_spikes
    assert (np.diff(t_ms) >= 0).all() and t_ms.max() < 100
    assert set(brain.conn.types[idx]) <= {"LB3b", "LB3c", "GNG232", "MN9", "DNg67"}
    listed = brain.run(50, record=True)
    assert listed and all(isinstance(t, float) and s.size for t, s in listed)
    assert brain.recording_arrays([])[0].size == 0


def test_snapshot_restore_reproduces_identical_spike_trains(brain):
    brain.stimulate("LB3b,LB3c", 120)
    brain.run(100)
    snap = brain.snapshot()
    brain.start_recording()
    brain.run(100)
    rec_a = brain.stop_recording()
    counts_a = brain.spike_count.copy()
    brain.restore(snap)
    assert brain.t == snap["t"]
    brain.start_recording()
    brain.run(100)
    rec_b = brain.stop_recording()
    assert np.array_equal(counts_a, brain.spike_count)
    assert len(rec_a) == len(rec_b) and all(ta == tb and np.array_equal(sa, sb) for (ta, sa), (tb, sb) in zip(rec_a, rec_b))


def test_short_term_depression_reduces_downstream_drive(conn):
    plain = drive(FlyBrain(conn, std_u=0.0), SUGAR).rate("GNG232")
    std = FlyBrain(conn, std_u=0.3, std_tau_ms=500)
    assert std.std_x is not None and std.std_x.min() == 1.0
    depressed = drive(std, SUGAR).rate("GNG232")
    assert depressed < plain * 0.5
    assert std.std_x[conn.select("LB3b")].max() < 0.5                       # the sensory terminals ran low


def test_background_noise_makes_spikes_without_stimulus(conn):
    noisy = FlyBrain(conn, noise_hz=500, noise_mv=3.0)
    noisy.run(200)
    assert noisy.total_spikes > 0 and noisy.stim == {}
    local = FlyBrain(conn, noise_hz=500, noise_mv=3.0, noise_spec="MN9")
    local.run(200)
    assert local.total_spikes > 0 and set(np.flatnonzero(local.spike_count > 0)) <= set(conn.select("MN9"))


def test_threshold_jitter_changes_thresholds(conn):
    flat = FlyBrain(conn)
    assert (flat._theta_i == THETA).all() and (flat.thr == THETA).all()
    jit = FlyBrain(conn, threshold_jitter=1.0)
    assert jit._theta_i.std() == pytest.approx(1.0, rel=0.25) and jit._theta_i.mean() == pytest.approx(THETA, abs=0.3)
    assert jit._theta_i.min() >= 2.0 and np.array_equal(jit.thr, jit._theta_i)
    assert np.array_equal(FlyBrain(conn, threshold_jitter=1.0)._theta_i, jit._theta_i)   # seeded
    assert not np.array_equal(FlyBrain(conn, threshold_jitter=1.0, seed=5)._theta_i, jit._theta_i)


def test_fatigue_raises_threshold_after_spikes(conn):
    tired = FlyBrain(conn, fatigue_mv=0.5, fatigue_ms=2000)
    tired.stimulate("LB3b", 100)
    tired.run(200)
    lb3b = conn.select("LB3b")
    assert (tired.thr[lb3b] > THETA).all() and (tired.thr[conn.select("LC4")] == THETA).all()
    tired.clear_stimuli()
    high = tired.thr[lb3b].copy()
    tired.run(3000)                                                           # fades while asleep too
    assert (tired.thr[lb3b] < high).all() and (tired.thr[lb3b] > THETA).all()


def test_set_stimulus_arrays_per_neuron_rates(brain):
    conn = brain.conn
    i0, i1 = int(conn.select("LC4/L")[0]), int(conn.select("LC4/L")[1])
    brain.set_stimulus_arrays(np.array([i0, i1, i1 + 1]), np.array([50.0, 200.0, 0.0]), extra={"MN9": 30, "LB3b": 0})
    assert list(brain.stim) == ["MN9"]
    assert brain._stim_idx.size == 2 + 2                                      # two columnar + two MN9, 0 Hz dropped
    brain.run(1000)
    assert brain.spike_count[i0] == pytest.approx(50, rel=0.3)
    assert brain.spike_count[i1] == pytest.approx(200, rel=0.2)
    assert brain.spike_count[i1 + 1] == 0 and brain.rate("MN9") == pytest.approx(30, rel=0.3)
    brain.set_stimulus_arrays(np.zeros(0, dtype=np.int64), np.zeros(0))
    assert brain._stim_idx.size == 0
    # a neuron in two populations keeps the higher rate
    brain.set_stimuli({"LB3b": 20, "LB3b,LB3c": 80, "LC4": 0})
    assert list(brain.stim) == ["LB3b", "LB3b,LB3c"]
    assert brain._stim_idx.size == 20 and brain._stim_p.max() == pytest.approx(80 * brain.dt / 1000)
    assert brain._stim_p.min() == pytest.approx(80 * brain.dt / 1000)


def test_determinism_with_same_seed(conn):
    a = drive(FlyBrain(conn, seed=3), SUGAR, ms=300).spike_count.copy()
    b = drive(FlyBrain(conn, seed=3), SUGAR, ms=300).spike_count.copy()
    c = drive(FlyBrain(conn, seed=4), SUGAR, ms=300).spike_count.copy()
    assert np.array_equal(a, b) and not np.array_equal(a, c)


def test_run_until_quiet(brain):
    assert brain.run_until_quiet(max_ms=500) == 100.0                        # nothing happening: first check
    brain.stimulate("LB3b,LB3c", 120)
    brain.run(200)
    brain.clear_stimuli()
    done = brain.run_until_quiet(max_ms=2000)
    assert 100 <= done < 2000
    before = brain.total_spikes
    brain.run(200)
    assert brain.total_spikes == before and brain.quiet
    assert not brain.active.any() and not brain.v.any() and not brain.g.any()   # everything pruned back to rest


def test_on_spikes_callback_and_settings(brain):
    seen = []
    brain.on_spikes.append(lambda spikes, t: seen.append((t, spikes.size)))
    brain.stimulate("MN9", 100)
    brain.run(100)
    assert seen and sum(n for _, n in seen) == brain.total_spikes
    s = brain.settings()
    assert s["dt"] == 0.5 and s["gain"] == 0.65 and s["kenyon_gain"] == 0.25 and s["plasticity"] is None
    assert s["silenced"] == [] and s["modulated"] == {} and s["seed"] == 0


def test_kenyon_gain_scales_inputs_to_kenyon_cells(conn):
    kc_edges = conn.edges_between("class:ALPN", "class:Kenyon_Cell")
    other = conn.edges_between("LB3b", "GNG232")
    pure, game = FlyBrain(conn, kenyon_gain=0.25), FlyBrain(conn, kenyon_gain=1.0)
    assert np.allclose(pure._w_original[kc_edges] * 4, game._w_original[kc_edges])
    assert np.allclose(pure._w_original[other], game._w_original[other])
    assert np.allclose(game._w_original[other], conn.n_syn[other] * 0.275 * 0.65)
