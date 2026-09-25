"""The background re-test: identical results with less work, in a separate low-priority process."""

import threading
import time

import numpy as np

from virtual_fly import experiments as E
from virtual_fly import retest as RT
from virtual_fly.experiments import Experiment, R, run_experiment
from virtual_fly.game import Game
from virtual_fly.settings import build_brain

SUGAR_EXP = Experiment("Synthetic sugar", {"LB3b,LB3c": 120}, 400,
                       [R("MN9", "MN9", 20, 400), R("GNG232", "sugar relay", 10, 400)], settle_ms=100)
LOOM_EXP = Experiment("Synthetic loom", {"LC4/R,LPLC2/R": 150}, 300,
                      [R("DNp01/R", "giant fibre", 20, 400), R("TTMn/R", "jump muscle MN", 5, 300)])


def _reference(brain, exp, seeds, after_ms=1000.0):
    """The v2.7 loop: every seed ran the after-stimulus period (v2.7 reported the last one's count)."""
    per, after = {r.label: [] for r in exp.readouts}, []
    for seed in seeds:
        brain.rng = np.random.default_rng(seed)
        brain.reset(); brain.clear_stimuli()
        for spec, hz in exp.stimulus.items():
            brain.stimulate(spec, hz)
        brain.run(exp.settle_ms); brain.reset_counts(); brain.run(exp.ms - exp.settle_ms)
        for r in exp.readouts:
            per[r.label].append(brain.rate(r.spec))
        brain.clear_stimuli(); brain.run(after_ms / 2); brain.reset_counts(); brain.run(after_ms / 2)
        after.append(float(brain.spike_count.sum() / (after_ms / 2000.0)))
    brain.clear_stimuli(); brain.reset()
    return per, after


def test_every_seed_runs_the_after_period_and_the_worst_is_reported(game_brain):
    for exp in (SUGAR_EXP, LOOM_EXP):
        per, after = _reference(game_brain, exp, (0, 1, 2))
        res = run_experiment(game_brain, exp, seeds=(0, 1, 2))
        assert all(r.per_seed == per[r.label] for r in res.readouts)
        assert res.after_per_seed == after and res.after_sps == max(after)
        assert res.after_not_calm == sum(v >= 1000 for v in after)
        quick = run_experiment(game_brain, exp, seeds=(0, 1, 2), after_ms=0)
        assert all(r.per_seed == per[r.label] for r in quick.readouts) and quick.after_note in ("calm", "")


def test_survival_skips_the_after_period_and_can_stop(game_brain):
    ref = [run_experiment(game_brain, e, seeds=(0, 1)) for e in (SUGAR_EXP, LOOM_EXP)]
    rows = E.survival(game_brain, [SUGAR_EXP, LOOM_EXP], seeds=(0, 1))
    assert [[x["per_seed"] for x in r["readouts"]] for r in rows] == \
        [[[round(v, 1) for v in x.per_seed] for x in r.readouts] for r in ref]
    assert E.survival(game_brain, [SUGAR_EXP, LOOM_EXP], seeds=(0, 1), should_stop=lambda: True) == []


def _wait(game, cond, secs=60):
    t0 = time.time()
    while time.time() - t0 < secs:
        game.tick()
        if cond():
            return True
        time.sleep(0.01)
    return False


def test_game_retests_in_a_process_with_the_same_results_as_a_thread(conn):
    out = {}
    for mode in ("process", "thread"):
        g = Game(build_brain(conn, "game", seed=0), autopilot=False, seed=1, retest=mode)
        assert g.action({"type": "parts", "on": True})["ok"]
        assert _wait(g, lambda: g.genome["survival"] and not g.genome["survival"]["running"], secs=120)
        sv = g.genome["survival"]
        assert sv["where"] == mode and sv["tested"] > 0 and "error" not in sv
        out[mode] = sv["results"]
    assert out["process"] == out["thread"]
    # and the child built the very brain the game builds (the test's swapped-in data tables travel with it)
    g = Game(build_brain(conn, "game", seed=0), autopilot=False, seed=1, retest="process")
    g.parts_on = True
    spec = g._retest_spec(g.conn)
    assert spec["vfb"] is not None and spec["regions"] == {"table": None}     # conftest's empty tables
    h = RT.Retest(spec)
    h.wait()
    assert h.info["fingerprint"] == RT.fingerprint(g.brain_factory(g.conn, parts=g.parts_arg(True)))


def test_a_newer_retest_terminates_the_older_process(conn):
    g = Game(build_brain(conn, "game", seed=0), autopilot=False, seed=1)
    spec = g._retest_spec(g.conn)
    assert spec is not None and spec["wiring"] is None and spec["profile"] == "game"
    h = RT.Retest(spec)
    h.cancel()
    h.proc.join(timeout=10)
    assert not h.proc.is_alive()
    # a caller with its own brain factory and no brain_kwargs re-tests in a thread
    g2 = Game(build_brain(conn, "game", seed=0), autopilot=False, seed=1,
              brain_factory=lambda c, **kw: build_brain(c, "game", **kw))
    assert g2._retest_spec(g2.conn) is None


def test_a_fly_tested_before_with_the_same_settings_comes_from_the_cache(conn):
    g = Game(build_brain(conn, "game", seed=0), autopilot=False, seed=1, retest="thread")
    done = lambda: g.genome["survival"] and not g.genome["survival"]["running"]        # noqa: E731
    rows = []
    for on in (True, False, True):
        assert g.action({"type": "parts", "on": on})["ok"]
        assert _wait(g, lambda: not g.genome["growing"] and done(), secs=120)
        rows.append((g.genome["survival"]["where"], g.genome["survival"]["results"]))
    assert rows[0][0] == "thread" and rows[2][0] == "cache" and rows[2][1] == rows[0][1]
    assert rows[1][0] == "thread"                          # parts off on this fly: not tested before


def test_a_retest_process_that_dies_falls_back_to_a_thread(conn, monkeypatch):
    class Broken:                                            # a child that could not start (e.g. an import failed)
        def __init__(self, spec):
            self.cancelled = False

        def cancel(self):
            self.cancelled = True

        def wait(self, on_progress=None):
            raise RuntimeError("the re-test process stopped (exit code 1)")
    monkeypatch.setattr(RT, "Retest", Broken)
    g = Game(build_brain(conn, "game", seed=0), autopilot=False, seed=1, retest="process")
    assert g.action({"type": "parts", "on": True})["ok"]
    assert _wait(g, lambda: g.genome["survival"] and not g.genome["survival"]["running"], secs=120)
    sv = g.genome["survival"]
    assert "error" not in sv and sv["where"] == "thread" and sv["tested"] > 0


def test_a_newer_retest_supersedes_a_running_one(conn):
    g = Game(build_brain(conn, "game", seed=0), autopilot=False, seed=1, retest="process")
    g._survival_token = t1 = object()
    th = threading.Thread(target=g._survival_worker, args=(g.conn, t1))
    th.start()
    assert _wait(g, lambda: g._retest_handle is not None, 30)
    first = g._retest_handle
    g._survival_token = t2 = object()
    g._survival_worker(g.conn, t2)                     # the newer one cancels the older process and finishes
    th.join(30)
    assert first.cancelled and not first.proc.is_alive()
    assert g.genome["survival"]["where"] == "process" and not g.genome["survival"]["running"]
