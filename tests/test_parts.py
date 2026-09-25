"""The genes as each neuron's parts list: slow modulators, graded cells and per-type overrides
(synthetic connectome)."""

import numpy as np
import pytest

from virtual_fly import parts as P
from virtual_fly.brain import FlyBrain
from virtual_fly.experiments import Experiment, R, run_experiment
from virtual_fly.settings import build_brain


def _measure(brain, stim, ms=300, settle=100):
    brain.reset()
    brain.set_stimuli(stim)
    brain.run(settle)
    brain.reset_counts()
    brain.run(ms - settle)
    return brain


def test_compile_finds_the_parts_of_the_connectome(conn):
    cp = P.PartsList().compile(conn)
    assert cp.n_kinds == 3 and [m["nt"] for m in cp.counts["modulators"]] == ["dopamine", "octopamine", "serotonin"]
    assert cp.mod_neurons.size == conn.select("nt:dopamine,nt:octopamine,nt:serotonin").size == 16
    assert set(cp.mod_kind[conn.select("PPL101")]) == {0} and set(cp.mod_kind[conn.select("OA-VPM3")]) == {1}
    assert (cp.mod_kind[conn.select("MN9")] == -1).all()
    assert cp.mod_targets.size == cp.counts["modulated_targets"] > 0 and (cp.target_pos[cp.mod_targets] == np.arange(cp.mod_targets.size)).all()
    hse = conn.select("HSE/L")[0]
    assert cp.target_pos[hse] in cp.kind_targets[1] and cp.target_pos[hse] not in cp.kind_targets[0]
    assert cp.graded_mask[conn.select("T4a,T5b,HSE,Mi1,Tm9")].all() and not cp.graded_mask[conn.select("LC4,MN9,DNp01")].any()
    assert cp.graded_idx.size == cp.counts["graded_neurons"] == conn.select("prefix:T4,prefix:T5,HSE,HSN,HSS,Mi1,Mi4,Mi9,Tm1,Tm2,Tm4,Tm9").size
    assert (cp.theta[cp.graded_idx] == np.float32(P.BIG_THRESHOLD)).all() and (cp.theta[conn.select("MN9")] == 7.0).all()
    d = P.PartsList().describe()
    assert {m["nt"] for m in d["modulators"]} == {"dopamine", "octopamine", "serotonin"} and all("GPCR" not in m["genes"] for m in d["modulators"])
    assert all(m["receptors"] and m["why"] and m["genes"] for m in d["modulators"]) and len(d["graded"]) == 5
    assert P.as_parts(None) is None and P.as_parts(False) is None and isinstance(P.as_parts(True), P.PartsList)
    with pytest.raises(TypeError):
        P.as_parts("yes")


def test_param_overrides_and_parsing(conn):
    pl = P.PartsList().with_params(["GNG232:theta=3", "LC4:graded=1", "T4a:graded=0,theta=9"])
    cp = pl.compile(conn)
    assert (cp.theta[conn.select("GNG232")] == np.float32(3.0)).all()
    assert cp.graded_mask[conn.select("LC4")].all() and not cp.graded_mask[conn.select("T4a")].any()
    assert (cp.theta[conn.select("T4a")] == 9.0).all() and cp.counts["params"][0]["neurons"] == 4
    for bad in ("MN9", "MN9:colour=red", "MN9:theta=0.1"):
        with pytest.raises(ValueError):
            P.parse_param(bad)
    # a lower threshold on the sugar relay: a weak sugar drive now reaches the motor neuron
    lo = FlyBrain(conn, seed=0, parts=pl)
    hi = FlyBrain(conn, seed=0, parts=True)
    assert lo.settings()["parts"]["graded_neurons"] > 0
    assert _measure(lo, {"LB3b": 40}).rate("MN9") > 10 and _measure(hi, {"LB3b": 40}).rate("MN9") == 0


def test_modulators_lose_their_fast_synapses_and_leave_a_tone(conn):
    plain = FlyBrain(conn, seed=0)
    parts = FlyBrain(conn, seed=0, parts=True)
    assert parts.settings()["parts"] == {"graded_neurons": parts.parts.graded_idx.size, "modulatory_neurons": 16,
                                         "modulated_targets": parts.parts.mod_targets.size,
                                         "modulators": ["dopamine", "octopamine", "serotonin"], "graded_rate_hz": 300.0,
                                         "curated": "modulators", "receptor_signs": True, "co_release_neurons": 0,
                                         "curated_neurons": 0}
    # dopamine drives the MBONs directly in the published model; with the parts list it does not
    assert _measure(plain, {"PPL101": 300}).rate("MBON11") > 20
    assert _measure(parts, {"PPL101": 300}).rate("MBON11") == 0 and parts.rate("PPL101") > 150
    _measure(parts, {"PPL101": 80})
    st = parts.parts_status()
    assert 0.3 < st["tone"]["dopamine"]["mean"] <= 1 and st["tone"]["dopamine"]["targets_on"] == 4      # MBON11/12, both sides
    assert st["tone"]["octopamine"]["mean"] == 0 and st["modulatory"] == 16 and st["graded"] == parts.parts.graded_idx.size
    assert parts._mod_active and (parts._mod_gain > 1).sum() == 4 and parts._mod_gain.max() < 1.3
    mbon = conn.select("MBON11,MBON12")
    assert (parts._mod_gain[parts.parts.target_pos[mbon]] > 1.2).all()          # the tone raises their input gain
    parts.clear_stimuli()
    parts.run(5000)                                                 # 500 ms time constant: ten of them and the tone is gone
    assert not parts._mod_active and parts.parts_status()["tone"]["dopamine"]["mean"] == 0 and parts._mod_gain.max() == 1
    # silencing a modulatory population also removes its tone
    parts.silence("PPL101")
    _measure(parts, {"PPL101": 80})
    assert parts.parts_status()["tone"]["dopamine"]["mean"] == 0
    parts.unsilence()


def test_octopamine_and_serotonin_tone_from_a_sound(conn):
    b = FlyBrain(conn, seed=0, parts=True)
    _measure(b, {"prefix:JO-B": 100}, ms=800, settle=100)
    st = b.parts_status()["tone"]
    assert st["octopamine"]["mean"] > 0.2 and st["serotonin"]["mean"] > 0.2 and st["octopamine"]["targets_on"] >= 6
    assert b.rate("OA-VPM3") > 20 and b.rate("CSD") > 20
    # a target of both hears the tone with the modulator's own time constant: octopamine (1 s) fades before serotonin (2 s)
    b.clear_stimuli()
    b.run(1500)
    st2 = b.parts_status()["tone"]
    assert st2["octopamine"]["mean"] < st["octopamine"]["mean"] and st2["serotonin"]["mean"] < st["serotonin"]["mean"]
    assert st2["octopamine"]["mean"] / st["octopamine"]["mean"] < st2["serotonin"]["mean"] / st["serotonin"]["mean"]
    # octopamine raises the HS cells' response to the same motion
    quiet = FlyBrain(conn, seed=0, parts=True)
    motion = {"T4a/R,T5a/R": 10}                                     # below the graded cells' saturation
    base = _measure(quiet, motion, ms=500, settle=200).rate("HSE/R,HSN/R,HSS/R")
    aroused = FlyBrain(conn, seed=0, parts=True)
    aroused.set_stimuli({"prefix:JO-B": 100})
    aroused.run(600)
    aroused.set_stimuli(motion)
    aroused.run(200)
    aroused.reset_counts()
    aroused.run(300)
    assert aroused.rate("HSE/R,HSN/R,HSS/R") > base * 1.2 > 0


def test_graded_cells_release_below_threshold_without_reset_or_refractory_period(conn):
    plain = FlyBrain(conn, seed=0)
    parts = FlyBrain(conn, seed=0, parts=True)
    weak = {"Mi1/R": 30}                                            # a weak drive of one T4 input
    t4 = "T4a/R,T4b/R,T4c/R,T4d/R"
    assert _measure(plain, weak).rate(t4) == 0                     # sub-threshold in a spiking T4
    assert _measure(parts, weak).rate(t4) > 5                      # a graded T4 transmits it
    assert parts.parts_status()["graded_active"] > 0
    # no reset: the potential stays where the input holds it, and the release rate saturates at 300 Hz
    strong = _measure(parts, {"Mi1/R,Mi9/R": 400}, ms=400, settle=200)
    idx = conn.select(t4)
    assert strong.v[idx].max() > 7.0 and strong.rate(t4) == pytest.approx(300, abs=10)
    assert (strong.thr[idx] >= 1e8).all() and strong._rel.size == parts.parts.graded_idx.size
    # graded cells never appear in the refractory list
    parts.set_stimuli({"Mi1/R,Mi9/R": 400})
    for _ in range(20):
        parts.step()
        assert not parts._gmask[np.concatenate(parts._recent)].any()
    # a forced graded cell (the retina drives T4/T5 directly) still fires at the forced rate, once per step at most
    forced = _measure(FlyBrain(conn, seed=1, parts=True), {"T4a/R": 100}, ms=600, settle=100)
    assert 70 < forced.rate("T4a/R") < 130
    # the HS cells are graded too, so a modest motion signal still reaches DNp15 (in the spiking model it needs more)
    assert _measure(parts, {"T4a/R,T5a/R": 25}, ms=600, settle=100).rate("DNp15/R") > 0


def test_snapshot_restore_and_reset_carry_the_parts_state(conn):
    b = FlyBrain(conn, seed=0, parts=True)
    b.set_stimuli({"PPL101": 80, "Mi1/R": 60})
    b.run(200)
    snap = b.snapshot()
    assert snap["rel"].size == b.parts.graded_idx.size and snap["mod_active"] and snap["mod_level"].max() > 0
    b.start_recording(); b.run(100); first = b.stop_recording()
    b.restore(snap)
    b.start_recording(); b.run(100); second = b.stop_recording()
    assert len(first) == len(second) > 0 and all(t1 == t2 and np.array_equal(s1, s2) for (t1, s1), (t2, s2) in zip(first, second))
    b.reset()
    assert not b._mod_active and b._mod_level.max() == 0 and b._rel.max() == 0 and b._mod_gain.min() == 1


def test_profiles_and_experiments_with_the_parts_list(conn):
    game = build_brain(conn, "game", parts=True, seed=0)
    assert game.parts is not None and "class:DAN" not in game.silenced and "class:ALLN" in game.silenced
    assert build_brain(conn, "game", seed=0).silenced.keys() >= {"class:DAN", "class:ALLN"}
    sugar = Experiment("sugar", {"LB3b,LB3c": 120}, 400, [R("GNG232", "relay", 10, 500), R("MN9", "MN9", 10, 500)])
    assert run_experiment(game, sugar).ok
    bitter = Experiment("bitter", {"LB1a,LB1b,LB1c,LB1d": 100, "LB3b,LB3c": 120}, 400, [R("MN9", "MN9", 0, 5)])
    assert run_experiment(game, bitter).ok
    loom = Experiment("loom", {"LC4/R,LPLC2/R": 150}, 300, [R("DNp01/R", "GF", 50, 500)])
    assert run_experiment(game, loom).ok
    pure = build_brain(conn, "pure", parts=P.PartsList(graded=()), seed=0)
    assert pure.parts.graded_idx.size == 0 and pure.parts.mod_neurons.size == 16
