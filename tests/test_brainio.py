"""The BrainIO seam (brainio.py): a brain in its own process gives the same numbers as the brain in this process."""

import json
import os
import signal
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from virtual_fly import brainio, retest as RT
from virtual_fly.brainio import BrainTick, LocalBrain, ProcessBrain, advance_all
from virtual_fly.game import READOUTS, TICK_MS, Game
from virtual_fly.settings import build_brain

GOLDEN = json.loads((Path(__file__).with_name("golden_single_fly.json")).read_text())["hashes"]


def _game(conn, procs="on", **kw):
    return Game(build_brain(conn, "game", seed=0), autopilot=False, seed=1, brain_procs=procs, **kw)


@pytest.fixture
def pgame(conn):
    g = _game(conn)
    try:
        yield g
    finally:
        g.close()


@pytest.mark.parametrize("name", ["autopilot", "zap_mdn", "silence_and_watch"])
def test_a_brain_in_its_own_process_gives_the_golden_hash(conn, name):
    from tools.golden_hashes import run_hash
    assert run_hash(conn, name, game_kwargs={"brain_procs": "on"}) == GOLDEN[name]


def test_the_options_are_checked(conn):
    with pytest.raises(ValueError, match="brain_procs"):
        Game(build_brain(conn, "game", seed=0), brain_procs="yes")
    g = _game(conn, procs="off")
    assert isinstance(g.fly(0).io, LocalBrain) and g.brain is not None
    g = _game(conn, procs="auto")
    assert isinstance(g.fly(0).io, LocalBrain)                  # one fly: in this process
    # a brain without a connectome file cannot go to a child
    grown = conn.rewired(conn.row_ptr, conn.post_idx, conn.n_syn, label="grown")
    with pytest.raises(ValueError, match="file"):
        Game(build_brain(grown, "game", seed=0), brain_procs="on")


def test_the_process_brain_is_the_games_brain(pgame, conn):
    g = pgame
    io = g.fly(0).io
    assert isinstance(io, ProcessBrain) and g.brain is None and io.brain is None and io.proc.is_alive()
    # the child built the very brain the game would have (the test's swapped-in data tables travel with it)
    local = build_brain(conn, "game", seed=0)
    assert io.fingerprint == RT.fingerprint(local)
    assert io.settings() == local.settings()
    assert io.has_plasticity and io.parts_info() is None and io.parts_status() is None
    info = io.learning_summary()
    assert info["depressed_fraction"] == 0.0 and info["events"] == 0 and info["settings"]["plastic_synapses"] == local.plasticity.settings()["plastic_synapses"]
    rate, role = io.neuron(int(conn.select("MN9/L")[0]))
    assert rate == 0.0 and role is None


def test_commands_round_trip(pgame, conn):
    g = pgame
    io = g.fly(0).io
    n = int(conn.select("MN9").size)
    assert g.action({"type": "silence", "spec": "MN9"}) == {"ok": True, "n": n}
    assert g.action({"type": "zap", "spec": "GNG232", "hz": 200, "secs": 2.0})["ok"]
    for _ in range(12):
        g.tick()
    bt = g.fly(0).bt
    assert "MN9" in bt.silenced_specs and g.state_dict["silenced"] == ["MN9"] and g.state_dict["baseline"] == ["class:ALLN", "class:DAN"]
    assert bt.hz["MN9"] > 0 and bt.motor_hz["MN9"] == 0.0            # it fires, nothing hears it
    assert g.action({"type": "unsilence"})["ok"]
    g.tick()
    assert g.fly(0).bt.silenced_specs == ["class:ALLN", "class:DAN"]
    assert io.modulate("MN9", 1.5) == n and io.settings()["modulated"] == {"MN9": 1.5}
    io.unmodulate("MN9")
    assert io.settings()["modulated"] == {}
    assert g.action({"type": "watch", "spec": "GNG232", "key": "relay"})["ok"]
    g.tick()
    assert "relay" in g.fly(0).bt.hz and "relay" in g.state_dict["hz"] and "relay" in g.fly(0).histories
    assert g.action({"type": "unwatch", "key": "relay"})["ok"]
    g.tick()
    assert "relay" not in g.fly(0).bt.hz and "relay" not in g.fly(0).histories
    # learning on, off, forget
    assert g.action({"type": "learning", "on": False})["ok"]
    g.tick()
    assert g.state_dict["learning"]["enabled"] is False and io.learning_summary()["settings"]["enabled"] is False
    assert g.action({"type": "learning", "on": True, "forget": True})["ok"]
    g.tick()
    assert g.state_dict["learning"]["enabled"] is True and io.learning_summary()["depressed_fraction"] == 0.0
    # the spike recording lives in the child
    assert g.action({"type": "record", "on": True, "spikes": True})["ok"]
    for _ in range(3):
        g.tick()
    assert io.record("active") is True and g.state_dict["recording"] == {"frames": 3, "spikes": True, "active": True}
    assert g.action({"type": "record", "on": False})["ok"]
    g.tick()
    assert io.record("active") is False
    t_ms, idx = io.record("arrays")
    assert isinstance(t_ms, np.ndarray) and isinstance(idx, np.ndarray) and t_ms.size == idx.size > 0
    # reset: the brain back at rest, the histories cleared
    assert g.action({"type": "calm"})["ok"]
    g.tick()
    assert g.state_dict["msg"] == "Brain reset to rest." and all(len(h) == 1 for h in g.fly(0).histories.values())


def test_the_parts_list_is_switched_in_the_child(pgame, mini_vfb):
    g = pgame
    io = g.fly(0).io
    assert g.action({"type": "parts", "on": True})["ok"]
    t0 = time.time()
    while g.genome["growing"] is not None and time.time() - t0 < 120:
        g.tick()
        time.sleep(0.01)
    assert g.genome["growing"] is None and g.parts_on and "parts" in g.done
    assert io.parts_status() is not None and io.settings()["parts"] is not None and io.parts_info()[1]["modulatory_neurons"] > 0
    assert g.state_dict["genome"]["parts"]["on"] is True and g.state_dict["genome"]["parts"]["status"]["graded"] > 0
    # and it is the brain a local game builds with the parts list on
    local = Game(build_brain(g.real_conn, "game", seed=0, parts=g.parts_arg(True)), autopilot=False, seed=1)
    assert io.fingerprint == RT.fingerprint(local.brain)


def test_commands_from_many_threads_do_not_interleave(pgame):
    g = pgame
    io = g.fly(0).io
    errors, replies = [], []

    def worker():
        try:
            for _ in range(15):
                s = io.settings()
                assert s["dt"] == 0.5 and "silenced" in s
                bin_ms, hist = g.fly(0).history(["MN9", "GF"], 5)
                assert set(hist) <= {"MN9", "GF"}
                replies.append(io.learning_summary()["events"])
        except Exception as e:                    # noqa: BLE001
            errors.append(repr(e))
    threads = [threading.Thread(target=worker) for _ in range(4)]
    for th in threads:
        th.start()
    for _ in range(30):
        g.tick()
    for th in threads:
        th.join(30)
    assert not errors and len(replies) == 60 and not any(th.is_alive() for th in threads)


def test_parent_side_history_equals_the_monitors(conn):
    actions = [{"type": "zap", "spec": "GNG232", "hz": 150, "secs": 1.0}, {"type": "watch", "spec": "DNa02", "key": "turn"}]
    out = {}
    for procs in ("off", "on"):
        g = _game(conn, procs=procs)
        try:
            for a in actions:
                assert g.action(dict(a))["ok"]
            for _ in range(60):
                g.tick()
            out[procs] = g.fly(0).history(None, 400)
            bin_ms, hist = out[procs]
            assert bin_ms == TICK_MS and set(hist) == {r[0] for r in READOUTS if g.readouts[r[0]].size} | {"turn"}
            assert len(hist["MN9"]) == 60 and any(v > 0 for v in hist["G2N-1"])      # the sugar relay, zapped
        finally:
            g.close()
    assert out["on"] == out["off"]


def test_close_leaves_no_child_and_is_idempotent(conn, monkeypatch):
    """Game.close() reaches every fly's seam once per call and leaves no child. That serve() calls it only after the
    loop thread has ended is proved by tests/test_server.py::test_ctrl_c_stops_the_game_loop_before_serve_returns."""
    g = _game(conn)
    io = g.fly(0).io
    proc = io.proc
    calls = []
    real_close = io.close
    monkeypatch.setattr(io, "close", lambda: (calls.append(1), real_close()))
    loop = threading.Thread(target=g.loop, daemon=True)
    loop.start()
    time.sleep(0.3)
    assert loop.is_alive()
    g.stop_loop.set()
    loop.join(10)
    g.close()                                    # as serve does: after the loop thread has ended
    proc.join(5)
    assert calls == [1] and not proc.is_alive() and not io.proc.is_alive() and proc.exitcode == 0
    g.close()                                    # idempotent
    assert calls == [1, 1] and not io.proc.is_alive()


def test_a_child_that_dies_is_reported_not_waited_for(conn):
    g = _game(conn)
    io = g.fly(0).io
    io.proc.kill()
    io.proc.join(5)
    t0 = time.time()
    with pytest.raises(RuntimeError, match="fly 0 stopped .*exit code"):
        g.tick()
    assert time.time() - t0 < 5
    g.close()                                    # still returns, nothing left
    assert not io.proc.is_alive()


def test_advance_all_is_a_lockstep_over_seams(conn):
    """One local and one process brain, sent their input together, answer with the same readouts."""
    g_loc, g_proc = _game(conn, procs="off"), _game(conn, procs="on")
    try:
        a, b = g_loc.fly(0), g_proc.fly(0)
        dt = TICK_MS / 1000.0
        rates = {"LB3b,LB3c": 120.0}
        empty = np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float32)
        for seq in range(5):
            ticks = advance_all([a.io, b.io], [a.advance_input(dt, rates, *empty, seq), b.advance_input(dt, rates, *empty, seq)])
            assert isinstance(ticks[0], BrainTick) and isinstance(ticks[1], BrainTick)
            assert ticks[0].hz == ticks[1].hz and ticks[0].spikes_shown == ticks[1].spikes_shown and ticks[0].n_spikes == ticks[1].n_spikes
        assert ticks[0].hz["MN9"] > 0
    finally:
        g_loc.close()
        g_proc.close()


def _answers_from_another_thread(io, secs: float = 3.0) -> bool:
    """Does the seam answer a settings() call from another thread within ``secs`` (its lock is free)?"""
    done = threading.Event()
    threading.Thread(target=lambda: (io.settings(), done.set()), daemon=True).start()
    return done.wait(secs)


@pytest.mark.parametrize("when", ["before the sends", "during the receives"])
def test_a_dead_brain_in_the_lockstep_leaves_the_others_free(conn, monkeypatch, when):
    """advance_all with one failing brain: the healthy brain, sent its input but not yet received when the barrier
    fails, still gets its answer read and its lock let go, and closes cleanly afterwards."""
    ga, gb = _game(conn), _game(conn)
    try:
        a, b = ga.fly(0), gb.fly(0)
        dt = TICK_MS / 1000.0
        empty = np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float32)
        if when == "before the sends":               # the healthy brain first: sent, then the dead one's send fails
            healthy, dead, order = a, b, [a, b]
            advance_all([x.io for x in order], [x.advance_input(dt, {}, *empty, 0) for x in order])   # a healthy lockstep
            dead.io.proc.kill()
            dead.io.proc.join(5)
        else:                                        # the failing brain first: its receive fails, the healthy one is left sent
            healthy, dead, order = b, a, [a, b]
            advance_all([x.io for x in order], [x.advance_input(dt, {}, *empty, 0) for x in order])
            real = dead.io.recv_advance

            def failing():
                real()
                raise RuntimeError("the brain process of fly 0 stopped (exit code -9): test")
            monkeypatch.setattr(dead.io, "recv_advance", failing)
        assert _answers_from_another_thread(healthy.io)
        with pytest.raises(RuntimeError, match="fly 0 stopped"):
            advance_all([x.io for x in order], [x.advance_input(dt, {}, *empty, 1) for x in order])
        # the healthy brain: no lock held, and no answer left in its pipe (its next tick reads its own answer)
        assert _answers_from_another_thread(healthy.io)
        bt = healthy.io.advance(dt, {"LB3b,LB3c": 120.0}, *empty, 2, healthy.readouts)
        assert isinstance(bt, BrainTick) and bt.stims == 1
        t0 = time.time()
        healthy.io.close()
        assert time.time() - t0 < 3 and healthy.io.proc.exitcode == 0     # asked to close, not terminated
    finally:
        ga.close()
        gb.close()


def test_a_paused_game_survives_its_brain_process_dying(conn):
    g = _game(conn)
    try:
        g.paused = True
        loop = threading.Thread(target=g.loop, daemon=True)
        loop.start()
        time.sleep(0.2)
        g.fly(0).io.proc.kill()
        time.sleep(1.0)
        assert loop.is_alive() and "brain process of fly 0" in g.message     # "stopped (exit code -9)", then "is closed"
    finally:
        g.stop_loop.set()
        loop.join(5)
        g.close()


def test_a_killed_child_reports_its_real_exit_code(conn):
    g = _game(conn)
    io = g.fly(0).io
    io.proc.kill()                               # no join: the parent learns the code as it notices the death
    with pytest.raises(RuntimeError, match=r"exit code -9"):
        g.tick()
    g.close()
    assert not io.proc.is_alive()


def _learned_game(conn, procs):
    """A game whose fly has learned something (plasticity events and depressed synapses), as the odour test does."""
    g = _game(conn, procs=procs)
    hx, hy = g.body.pose.head
    assert g.action({"type": "drop", "kind": "vinegar", "x": hx, "y": hy, "food": "sugar"})["ok"]
    for _ in range(60):
        g.tick()
    assert g.state_dict["learning"]["events"] > 0 and g.state_dict["learning"]["depressed_fraction"] > 0
    return g


@pytest.mark.parametrize("procs", ["off", "on"])
def test_the_watchdog_reset_tick_publishes_the_brain_after_the_reset(conn, monkeypatch, procs):
    g = _learned_game(conn, procs)
    try:
        a = g.fly(0)
        real = a.advance

        def runaway(*args, **kw):                # this tick's answer says the brain fires far too much
            bt = real(*args, **kw)
            bt.n_spikes = 10_000_000
            return bt
        monkeypatch.setattr(a, "advance", runaway)
        a.runaway_s, a.since_input = 4.0, 1.0    # over the line at this tick's watchdog
        before = a.calms
        g.tick()
        assert a.calms == before + 1 and any(e["text"].startswith("runaway firing") for e in g.events.items)
        st = g.state_dict
        live = a.io.learning_summary()           # the brain as it is now, after its reset
        assert st["learning"]["events"] == live["events"] == 0
        assert st["learning"]["mbon"] == {t: {"strength": round(v["strength"], 3), "now": round(v["now"], 3),
                                              "valence": v["valence"], "dopamine": round(v["dopamine"], 3)}
                                          for t, v in live["mbon"].items()}
        assert all(v["dopamine"] == 0 for v in st["learning"]["mbon"].values())
        assert st["genome"]["parts"]["status"] == a.io.parts_status() and st["stims"] == a.io.status_fields()["stims"]
    finally:
        g.close()


def test_a_rebuild_that_fails_in_the_child_keeps_the_game_usable(pgame, monkeypatch):
    g = pgame
    io = g.fly(0).io
    monkeypatch.setattr(io, "swap", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no room for a second brain")))
    assert g.action({"type": "parts", "on": True})["ok"]
    t0 = time.time()
    while g.genome["growing"] is not None and time.time() - t0 < 30:
        g.tick()
        time.sleep(0.01)
    assert g.genome["growing"] is None and g.genome["error"] == "no room for a second brain"
    assert not g.parts_on and io.parts_status() is None      # the old fly stayed
    assert any("could not be rebuilt" in e["text"] for e in g.events.items)
    assert g.action({"type": "parts", "on": True})["ok"]     # the next attempt is taken, not refused


def test_a_child_that_stops_answering_is_stopped_not_waited_for(conn, monkeypatch):
    g = _game(conn)
    io = g.fly(0).io
    try:
        monkeypatch.setattr(ProcessBrain, "REQUEST_TIMEOUT", 0.3)
        os.kill(io.proc.pid, signal.SIGSTOP)                   # the child can no longer answer
        t0 = time.time()
        with pytest.raises(RuntimeError, match="did not answer within 0.3 s"):
            io.settings()
        assert time.time() - t0 < 3 and io.closed
        with pytest.raises(RuntimeError, match="closed"):      # its late answer can never reach a later request
            io.settings()
    finally:
        os.kill(io.proc.pid, signal.SIGCONT)                   # let the stop it was sent take effect
        g.close()
    assert not io.proc.is_alive()


@pytest.mark.parametrize("procs", ["off", "on"])
def test_depressed_fraction_is_the_live_number_on_both_seams(conn, procs):
    g = _learned_game(conn, procs)
    try:
        a = g.fly(0)
        d = a.depressed_fraction()
        assert d > 0 and d == a.io.learning_summary()["depressed_fraction"] == a.io.depressed_fraction()
        assert g.action({"type": "learning", "forget": True})["ok"]
        g._apply_actions()                       # the forget takes effect (a tick would learn again on the sugar)
        assert a.depressed_fraction() == 0.0 == a.io.learning_summary()["depressed_fraction"]
    finally:
        g.close()
