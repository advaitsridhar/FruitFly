"""Dopamine-gated depression of KC->MBON synapses."""

import numpy as np
import pytest

from virtual_fly.brain import FlyBrain
from virtual_fly.plasticity import MushroomBodyPlasticity, mbon_valence

KC = "class:Kenyon_Cell"


@pytest.fixture
def learner(conn):
    brain = FlyBrain(conn, seed=0)
    pl = MushroomBodyPlasticity().attach(brain)
    return brain, pl


def strengths(pl):
    return {t: v["strength"] for t, v in pl.summary().items()}


def pair(brain, dan_spec, dan_hz=80.0, ms=1000.0, kc_hz=20.0):
    brain.reset()
    brain.clear_stimuli()
    brain.stimulate(KC, kc_hz)
    if dan_hz > 0:
        brain.stimulate(dan_spec, dan_hz)
    brain.run(ms)


def test_attach_finds_plastic_edges_and_compartments(learner):
    brain, pl = learner
    conn = brain.conn
    assert brain.plasticity is pl and pl.brain is brain and pl.enabled
    assert pl.pe.size == conn.edges_between(KC, "class:MBON").size == 240
    assert set(conn.types[conn.pre_idx[pl.pe]]) == {"KCg-m", "KCab-m"}
    assert set(conn.types[conn.post_idx[pl.pe]]) == {"MBON11", "MBON12", "MBON01", "MBON02"}
    assert np.array_equal(pl.kc[pl.pe_kc], conn.pre_idx[pl.pe]) and np.array_equal(pl.mbon[pl.pe_mbon], conn.post_idx[pl.pe])
    # DAN->MBON synapses define each MBON's compartment dopamine neurons
    comp = {}
    for k, t in enumerate(conn.types[pl.mbon]):
        comp.setdefault(t, set()).update(conn.types[pl.dan[pl.dm_dan[pl.dm_mbon == k]]].tolist())
    assert comp == {"MBON11": {"PPL101"}, "MBON12": {"PPL101", "PPL102"}, "MBON01": {"PAM01"}, "MBON02": {"PAM01"},
                    "MBON20": set()}
    assert pl.mbon_has_dan.sum() == 8 and not pl.mbon_has_dan[conn.types[pl.mbon] == "MBON20"].any()
    per_mbon = np.bincount(pl.dm_mbon, weights=pl.dm_w, minlength=pl.mbon.size)
    assert np.allclose(per_mbon[pl.mbon_has_dan], 1.0)                       # gating weights sum to 1 per MBON
    assert (pl.scale == 1.0).all() and np.allclose(brain.w[pl.pe], brain._w_original[pl.pe])
    s = pl.settings()
    assert s["plastic_synapses"] == 240 and s["dan_floor"] == 0.5 and s["enabled"]
    assert brain.settings()["plasticity"] == s


def test_punishment_depresses_only_the_ppl1_compartment(learner):
    brain, pl = learner
    pair(brain, "PPL101")
    s = strengths(pl)
    assert s["MBON11"] < 0.8 and s["MBON12"] < 0.8
    assert s["MBON01"] == 1.0 and s["MBON02"] == 1.0 and s["MBON20"] == 1.0
    assert pl.events > 0 and 0 < pl.depressed_fraction() <= 0.5
    conn = brain.conn
    onto11 = conn.types[conn.post_idx[pl.pe]] == "MBON11"
    assert (pl.scale[onto11] < 1).all() and (pl.scale[~onto11 & (conn.types[conn.post_idx[pl.pe]] == "MBON01")] == 1).all()
    assert np.allclose(brain.w[pl.pe], pl.base * pl.scale)                    # applied to the running weights
    summ = pl.summary()
    assert summ["MBON11"]["valence"] == 1 and summ["MBON01"]["valence"] == -1 and summ["MBON20"]["valence"] == 0
    assert summ["MBON11"]["dopamine"] > 0 and summ["MBON01"]["dopamine"] == 0 and summ["MBON11"]["kc_synapses"] == 60


def test_reward_depresses_only_the_pam_compartment(learner):
    brain, pl = learner
    pair(brain, "PAM01")
    s = strengths(pl)
    assert s["MBON01"] < 0.8 and s["MBON02"] < 0.8 and s["MBON11"] == 1.0 and s["MBON12"] == 1.0


def test_dan_floor_blocks_weak_dopamine_and_kcs_alone_do_nothing(learner):
    brain, pl = learner
    pair(brain, "PPL101", dan_hz=10)                                           # 10 Hz << dan_scale x dan_floor
    assert pl.depressed_fraction() == 0 and pl.events == 0 and pl.dopamine.max() == 0
    pair(brain, "PPL101", dan_hz=0)
    assert pl.depressed_fraction() == 0
    brain.reset()
    brain.clear_stimuli()
    brain.stimulate("PPL101", 80)                                            # dopamine without KC activity
    brain.run(500)
    assert pl.depressed_fraction() == 0 and pl.dan_trace.max() > 0
    nofloor = MushroomBodyPlasticity(dan_floor=0.0).attach(FlyBrain(brain.conn))
    pair(nofloor.brain, "PPL101", dan_hz=10)
    assert nofloor.depressed_fraction() > 0


def test_floor_is_respected_and_disable_stops_learning(learner):
    brain, pl = learner
    pair(brain, "PPL101", ms=6000)
    onto11 = brain.conn.types[brain.conn.post_idx[pl.pe]] == "MBON11"
    assert pl.scale.min() >= pl.floor - 1e-6 and pl.scale[onto11].max() == pytest.approx(pl.floor, abs=0.02)
    pl.reset_weights()
    pl.enabled = False
    pair(brain, "PPL101")
    assert (pl.scale == 1).all() and pl.kc_trace.max() > 0                  # traces still run, no depression


def test_reset_weights_restores_originals(learner):
    brain, pl = learner
    pair(brain, "PPL101")
    assert pl.depressed_fraction() > 0
    pl.reset_weights()
    assert (pl.scale == 1.0).all() and np.allclose(brain.w[pl.pe], brain._w_original[pl.pe])
    assert pl.depressed_fraction() == 0.0


def test_recovery_over_time(conn):
    pl = MushroomBodyPlasticity(recover_min=0.05).attach(FlyBrain(conn))
    brain = pl.brain
    pair(brain, "PAM01")
    brain.clear_stimuli()
    brain.stimulate("MN9", 5)                       # keeps the brain awake (see test_traces_freeze_while_quiet)
    brain.run(3000)                                 # traces fade (kc_tau 1.5 s, dan_tau 0.4 s)
    early = pl.scale.min()
    assert early < 0.95
    brain.run(6000)
    assert pl.scale.min() > early + 0.05
    forever = MushroomBodyPlasticity(recover_min=0.0).attach(FlyBrain(conn))
    pair(forever.brain, "PAM01")
    forever.brain.clear_stimuli()
    forever.brain.stimulate("MN9", 5)
    forever.brain.run(3000)
    low = forever.scale.min()
    forever.brain.run(3000)
    assert forever.scale.min() == low


def test_traces_and_forgetting_continue_while_the_brain_is_quiet(conn):
    pl = MushroomBodyPlasticity(recover_min=0.05).attach(FlyBrain(conn))
    brain = pl.brain
    pair(brain, "PAM01")
    trace_at_end = pl.kc_trace.max()
    brain.clear_stimuli()
    brain.run(3000)                                 # no input: the brain falls asleep after ~100 ms
    assert pl.kc_trace.max() < 0.2 * trace_at_end   # 3 s = 2 x kc_tau
    low = pl.scale.min()
    brain.run(6000)
    assert pl.scale.min() > low + 0.05


def test_silencing_kcs_is_reflected_in_base_weights(learner):
    brain, pl = learner
    brain.silence("KCg-m")
    kcg = brain.conn.types[brain.conn.pre_idx[pl.pe]] == "KCg-m"
    assert (pl.base[kcg] == 0).all() and (pl.base[~kcg] != 0).all()
    assert (brain.w[pl.pe][kcg] == 0).all()
    brain.unsilence()
    assert np.allclose(pl.base, brain._w_original[pl.pe])


def test_save_load_round_trip(learner, tmp_path):
    brain, pl = learner
    pair(brain, "PPL101")
    saved = pl.scale.copy()
    f = tmp_path / "weights.npz"
    pl.save(f)
    pl.reset_weights()
    pl.load(f)
    assert np.array_equal(pl.scale, saved) and np.allclose(brain.w[pl.pe], pl.base * saved)
    other = tmp_path / "other.npz"
    np.savez_compressed(other, edges=pl.pe[:-1], scale=saved[:-1])
    with pytest.raises(ValueError, match="different connectome"):
        pl.load(other)


def test_snapshot_restore_through_the_brain(learner):
    brain, pl = learner
    pair(brain, "PPL101", ms=500)
    snap = brain.snapshot()
    assert "plasticity" in snap and np.array_equal(snap["plasticity"]["scale"], pl.scale)
    brain.run(500)
    later = pl.scale.copy()
    assert not np.array_equal(later, snap["plasticity"]["scale"])
    brain.restore(snap)
    assert np.array_equal(pl.scale, snap["plasticity"]["scale"]) and pl.events == snap["plasticity"]["events"]
    assert np.allclose(brain.w[pl.pe], pl.base * pl.scale)
    brain.run(500)
    assert np.array_equal(pl.scale, later)                                   # identical replay


def test_mbon_valence_signs(conn):
    idx, val = mbon_valence(conn)
    assert idx.size == 10 and val.dtype == np.int8
    by_type = {t: int(v) for t, v in zip(conn.types[idx], val)}
    assert by_type == {"MBON11": 1, "MBON12": 1, "MBON01": -1, "MBON02": -1, "MBON20": 0}
    i2, v2 = mbon_valence(conn, "MBON01")
    assert i2.size == 2 and (v2 == -1).all()


def test_brain_reset_clears_traces(learner):
    brain, pl = learner
    pair(brain, "PPL101", ms=300)
    assert pl.kc_trace.max() > 0 and pl.dan_trace.max() > 0
    brain.reset()
    assert not pl.kc_trace.any() and not pl.dan_trace.any() and not pl.dopamine.any() and pl.events == 0
    assert pl.depressed_fraction() > 0                                       # weights are memory, not state
