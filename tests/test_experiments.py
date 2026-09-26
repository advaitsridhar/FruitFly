"""Experiment protocols, dose-response sweeps and lesion scans on the synthetic brain."""

import json

import numpy as np

import pytest

from virtual_fly import experiments as E
from virtual_fly.experiments import Experiment, R, after_note, format_result, in_range, run_experiment

SUGAR_EXP = Experiment("Synthetic sugar", {"LB3b,LB3c": 120}, 400,
                       [R("MN9", "MN9 proboscis motor neuron", 20, 400, "synthetic design"),
                        R("GNG232", "sugar relay", 10, 400), R("LC4", "looming detectors (silent)", 0, 0)],
                       settle_ms=100, tags=("synthetic", "taste"), note="designed pathway")
LOOM_EXP = Experiment("Synthetic loom", {"LC4/R,LPLC2/R": 150}, 300,
                      [R("DNp01/R", "giant fibre", 20, 400), R("TTMn/R", "jump muscle MN", 5, 300)], tags=("synthetic",))


def test_in_range_and_after_note():
    assert in_range(5, 0, 5) and in_range(0, 0, 0) and in_range(0.9, 0, 0)  # tolerance max(1, 15 %)
    assert in_range(5.9, 0, 5) and not in_range(6.1, 0, 5) and not in_range(-0.1, 0, 5)
    assert in_range(114, 50, 100) and not in_range(116, 50, 100) and not in_range(49, 50, 100)
    assert after_note(10) == "calm" and after_note(999) == "calm"
    assert "small loop" in after_note(5000) and "RUNAWAY" in after_note(60000)


def test_a_pass_through_the_margin_shows_it():
    from virtual_fly.experiments import ExperimentResult, ReadoutResult, in_margin, margin
    assert margin(5) == 1.0 and margin(100) == 15.0
    mean = ReadoutResult("MN9", "MN9", 5.8, 1.6, 0, 5, True, [4.0, 4.0, 6.0, 8.0, 7.0], seeds_out=2)   # 6 Hz: in via the margin
    seed = ReadoutResult("MN9", "MN9", 4.4, 1.0, 0, 5, True, [4.0, 5.5, 3.7], seeds_out=0)
    plain = ReadoutResult("MN9", "MN9", 4.0, 0.5, 0, 5, True, [3.5, 4.5], seeds_out=0)
    assert in_margin(mean) and in_margin(seed) and not in_margin(plain)
    text = format_result(ExperimentResult("x", [mean, seed, plain], 0.0, "", 0.1, [0]))
    lines = text.splitlines()
    assert "0-5(+1) Hz  ok on the mean, but 2 of 5 seeds outside" in lines[0] and "0-5(+1) Hz  ok" in lines[1]
    assert "0-5 Hz  ok" in lines[2]


def test_run_experiment_readouts_and_flags(brain):
    res = run_experiment(brain, SUGAR_EXP, seeds=(0, 1))
    assert res.name == "Synthetic sugar" and res.seeds == [0, 1] and res.ok and res.wall_s >= 0
    by = {r.spec: r for r in res.readouts}
    assert set(by) == {"MN9", "GNG232", "LC4"}
    assert by["MN9"].hz > 20 and by["MN9"].ok and by["MN9"].label == "MN9 proboscis motor neuron"
    assert len(by["MN9"].per_seed) == 2 and by["MN9"].hz == pytest.approx(sum(by["MN9"].per_seed) / 2)
    assert by["MN9"].sd >= 0 and by["LC4"].hz == 0 and by["LC4"].sd == 0 and by["LC4"].ok
    assert (by["MN9"].lo, by["MN9"].hi) == (20, 400)
    assert res.after_note == "calm" and res.after_sps < 1000               # no loops: it calms down
    assert brain.stim == {} and brain.t == 0                               # left at rest
    d = res.to_dict()
    assert d["ok"] and d["name"] == res.name and d["readouts"][0]["label"] == res.readouts[0].label
    assert json.dumps(d)                                                   # JSON-friendly
    text = format_result(res)
    assert "Synthetic sugar" in text and "1 s after it stops" in text and "seeds [0, 1]" in text


def test_run_experiment_flags_out_of_range_readouts(brain):
    exp = Experiment("Impossible", {"LB3b,LB3c": 120}, 300, [R("MN9", "too high", 1000, 2000), R("MN9", "fine", 0, 500)])
    res = run_experiment(brain, exp)
    assert not res.ok and [r.ok for r in res.readouts] == [False, True]
    assert "not the usual result" in format_result(res)
    silent = run_experiment(brain, Experiment("Silence", {}, 200, [R("all", "whole brain", 0, 0)]))
    assert silent.ok and silent.after_note == "" and silent.after_sps == 0


def test_run_all_with_only_filter(brain, capsys):
    exps = [SUGAR_EXP, LOOM_EXP]
    out = E.run_all(brain, exps, only="loom", verbose=False)
    assert [r.name for r in out] == ["Synthetic loom"] and out[0].ok
    assert [r.name for r in E.run_all(brain, exps, only="taste", verbose=False)] == ["Synthetic sugar"]   # tag match
    assert E.run_all(brain, exps, only="nothing", verbose=False) == []
    both = E.run_all(brain, exps, verbose=True)
    assert len(both) == 2
    printed = capsys.readouterr().out
    assert "Experiment" in printed and "Synthetic loom" in printed and "Synthetic sugar" in printed
    assert E.all_experiments() == E.CLASSIC + E.EXTENDED + E.GENETIC and len(E.CLASSIC) == 6
    assert all(e.profile == "game" and "genetic" in e.tags for e in E.GENETIC) and len(E.GENETIC) == 5
    assert E.run_all(brain, E.GENETIC, verbose=False, profile="pure") == []       # measured in the game profile


def test_classic_silence_experiment_passes_on_synthetic_brain(brain):
    res = run_experiment(brain, E.CLASSIC[0])
    assert res.name.startswith("Silence") and res.ok and res.readouts[0].hz == 0


def test_sweep_dose_response(brain):
    rows = E.sweep(brain, "LB3b,LB3c", [0, 40, 120], ["MN9", "GNG232"], ms=300, settle_ms=100)
    assert [r["hz"] for r in rows] == [0.0, 40.0, 120.0]
    assert set(rows[0]) == {"hz", "MN9", "GNG232"}
    assert rows[0]["MN9"] == rows[0]["GNG232"] == 0.0
    assert rows[2]["GNG232"] > rows[1]["GNG232"] >= 0 and rows[2]["MN9"] > 20
    with_bitter = E.sweep(brain, "LB3b,LB3c", [120], ["MN9"], ms=300, extra={"LB1a,LB1b,LB1c,LB1d": 120})
    assert with_bitter[0]["MN9"] < 5
    assert brain.stim == {} and brain.total_spikes == 0                    # reset afterwards


def test_lesion_scan_orders_relays_by_effect(brain):
    rows = E.lesion_scan(brain, SUGAR_EXP, ["DNp01", "GNG232", "no_such_type", "GNG087"], "MN9")
    assert [r["silenced"] for r in rows][0] == "GNG232"                    # the relay the behaviour needs
    assert {r["silenced"] for r in rows} == {"DNp01", "GNG232", "GNG087"}  # the unknown spec is skipped
    by = {r["silenced"]: r for r in rows}
    assert by["GNG232"]["hz"] == 0 and by["GNG232"]["change"] == pytest.approx(-1.0)
    assert by["GNG232"]["neurons"] == 4 and by["GNG232"]["baseline"] > 20
    assert abs(by["DNp01"]["change"]) < 0.5 and by["DNp01"]["baseline"] == by["GNG232"]["baseline"]
    assert [r["change"] for r in rows] == sorted(r["change"] for r in rows)
    assert brain.silenced == {}                                            # every lesion undone
    by_label = E.lesion_scan(brain, SUGAR_EXP, ["GNG232"], "MN9 proboscis motor neuron")
    assert by_label[0]["hz"] == 0


def test_lesion_scan_validates_the_readout_and_always_unsilences(brain, monkeypatch):
    with pytest.raises(ValueError, match="not a readout"):
        E.lesion_scan(brain, SUGAR_EXP, ["GNG232"], "no such readout")
    assert brain.silenced == {}
    calls = []
    real = E.run_experiment

    def flaky(b, exp, seeds=(0,)):
        calls.append(1)
        if len(calls) == 2:                                               # the first lesioned run blows up
            raise RuntimeError("boom")
        return real(b, exp, seeds=seeds)

    monkeypatch.setattr(E, "run_experiment", flaky)
    with pytest.raises(RuntimeError):
        E.lesion_scan(brain, SUGAR_EXP, ["GNG232"], "MN9")
    assert brain.silenced == {}                                           # the lesion was undone anyway


def test_save_json(brain, tmp_path):
    results = [run_experiment(brain, LOOM_EXP)]
    out = tmp_path / "results.json"
    E.save_json(results, out, brain)
    data = json.loads(out.read_text())
    assert data["results"][0]["name"] == "Synthetic loom" and data["results"][0]["ok"]
    assert data["settings"]["gain"] == 0.65
    E.save_json(results, out)
    assert "settings" not in json.loads(out.read_text())


PAIRING = Experiment("Vinegar with bitter taste", {"ORN_DM1,ORN_DM4,ORN_VM7d": 120, "LB1a,LB1b,LB1c,LB1d": 120}, 600,
                     [R("class:Kenyon_Cell", "Kenyon cells", 0, 1000), R("PPL101", "punishment dopamine", 0, 1000)])


def test_learning_counts_within_a_run_but_never_carries_to_the_next(game_brain, monkeypatch):
    pl = game_brain.plasticity
    for start in (1.0, 0.5):                                   # a naive fly, then a trained one
        pl.scale[:] = start
        pl.reapply(game_brain)
        seen = []
        real_reset, real_clear = game_brain.reset, game_brain.clear_stimuli
        monkeypatch.setattr(game_brain, "reset", lambda: (seen.append(("start", pl.scale.copy())), real_reset())[1])
        monkeypatch.setattr(game_brain, "clear_stimuli", lambda: (seen.append(("end", pl.scale.copy())), real_clear())[1])
        run_experiment(game_brain, PAIRING, seeds=(0, 1, 2))
        monkeypatch.undo()
        starts = [x for k, x in seen if k == "start"]
        ends = [x for k, x in seen if k == "end"]
        assert len(starts) >= 3 and all((x == np.float32(start)).all() for x in starts)   # every seed starts from the same fly
        assert min(x.min() for x in ends) < start                  # odour plus punishment: it learned during the runs
        assert (pl.scale == np.float32(start)).all()             # and the fly is left as it was
    pl.reset_weights()


def test_fragile_readouts_and_the_survival_rows(brain):
    from virtual_fly.experiments import ExperimentResult, ReadoutResult
    steady = ReadoutResult("a", "MN9", 30.0, 0.0, 20, 40, True, [30.0, 30.0], seeds_out=0)
    shaky = ReadoutResult("b", "MN9", 21.0, 1.0, 20, 40, True, [22.0, 19.0, 22.0], seeds_out=1)
    res = ExperimentResult("x", [steady, shaky], 0.0, "calm", 0.1, [0, 1, 2])
    assert res.ok and res.fragile and "ok on the mean, but 1 of 3 seeds outside" in format_result(res)
    assert res.to_dict()["fragile"] is True and not ExperimentResult("y", [steady], 0.0, "calm", 0.1, [0, 1]).to_dict()["fragile"]
    assert not ExperimentResult("y", [steady], 0.0, "calm", 0.1, [0, 1]).fragile
    rows = E.survival(brain, [SUGAR_EXP, LOOM_EXP], seeds=(0, 1))
    assert [r["seeds"] for r in rows] == [2, 2] and all(r["ok"] and r["fragile"] is False for r in rows)
    assert len(rows[0]["readouts"][0]["per_seed"]) == 2 and rows[0]["readouts"][0]["seeds_out"] == 0
    assert E.SEEDS == (0, 1, 2, 3, 4)


def test_the_after_line_splits_graded_quanta_and_tallies_every_seed(conn):
    from virtual_fly.brain import FlyBrain
    res = E.ExperimentResult("x", [], 60000.0, after_note(60000.0), 0.1, [0, 1, 2], after_per_seed=[500.0, 60000.0, 2500.0],
                             after_graded_per_seed=[0.0, 20000.0, 0.0])
    text = format_result(res)
    assert "60,000 events/s (40,000 spikes + 20,000 graded quanta)  RUNAWAY LOOP (see README, 'Honest limitations')" in text
    assert "per seed: 500, 60.0k, 2.5k (1 calm, 1 a small loop keeps firing, 1 RUNAWAY LOOP; the worst is shown)" in text
    d = res.to_dict()
    assert d["after_graded_per_seed"] == [0.0, 20000.0, 0.0] and d["after_events_per_s"] == d["after_spikes_per_s"] == 60000.0
    # measured: one graded share per seed, never more than all the events
    parts = FlyBrain(conn, seed=0, parts=True)
    clear, n = parts.clear_stimuli, {"calls": 0}

    def keep_on():          # run_experiment clears before each seed (odd calls) and before the after period (even ones)
        n["calls"] += 1
        if n["calls"] % 2:
            clear()
    parts.clear_stimuli = keep_on                      # so the graded cells are still releasing when it counts
    live = run_experiment(parts, Experiment("motion", {"T4a/R,T5a/R": 25, "Mi1/R,Mi9/R": 400}, 300, [R("HSE", "HS", 0, 400)]),
                          seeds=(0, 1))
    assert len(live.after_graded_per_seed) == len(live.after_per_seed) == 2
    assert all(0 < g < a for g, a in zip(live.after_graded_per_seed, live.after_per_seed))   # a share, not all or none
    assert "graded quanta" in format_result(live)
    plain = run_experiment(FlyBrain(conn, seed=0), SUGAR_EXP, seeds=(0,))
    assert plain.after_graded_per_seed == [0.0]                          # no parts list: no graded cells


def test_an_experiment_that_cannot_be_done_says_so_without_claiming_a_lesion():
    res = E.ExperimentResult("Courtship, fru silenced", [E.ReadoutResult("pC1", "prefix:pC1_", 0.0, 0.0, 40, 90, None),
                                                         E.ReadoutResult("pIP10", "pIP10", 0.0, 0.0, 15, 80, None)],
                             0.0, "", 0.0, [0], silenced=["gene:fru"], missing=["pIP10"], na=True)
    text = format_result(res)
    assert "pC1" in text and "not run" in text and "not in this fly" in text and "output blocked" not in text
    assert "cannot be done: this fly has no pIP10" in text
