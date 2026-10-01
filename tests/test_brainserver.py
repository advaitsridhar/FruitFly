"""One brain process for every fly (brainio.GpuBrainServer, docs/TWO_FLIES_PLAN.md 6.5): the same numbers as a process
per brain and as the brains in this process, checked with CPU brains on the synthetic connectome (the GPU brain, when
it is there, lives in that process the same way)."""

import json
import os
import signal
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from synthetic_connectome import build_synthetic
from virtual_fly import brainio, retest as RT
from virtual_fly.brainio import BrainTick, GpuBrainHandle, GpuBrainServer, LocalBrain, ProcessBrain, advance_all
from virtual_fly.connectome import Connectome
from virtual_fly.game import TICK_MS, Game
from virtual_fly.settings import build_brain

GOLDEN = json.loads((Path(__file__).with_name("golden_single_fly.json")).read_text())["hashes"]


@pytest.fixture(scope="module")
def fconn(tmp_path_factory):
    return Connectome(build_synthetic(tmp_path_factory.mktemp("data") / "synthetic-female.flyb.gz", sex="female"))


def _game(conn, procs="server", **kw):
    return Game(build_brain(conn, "game", seed=0), autopilot=False, seed=1, brain_procs=procs, **kw)


def _pair(conn, fconn, procs="server", **kw):
    return Game(build_brain(conn, "game", seed=0), autopilot=False, seed=1, brain_procs=procs, partner={"conn": fconn}, **kw)


@pytest.fixture
def sgame(conn):
    g = _game(conn)
    try:
        yield g
    finally:
        g.close()


def _answers_from_another_thread(io, secs: float = 3.0) -> bool:
    """Does the seam answer (or refuse) a settings() call from another thread within ``secs`` (its lock is free)?"""
    done = threading.Event()

    def call():
        try:
            io.settings()
        except Exception:                        # a closed seam refuses at once: the lock was free
            pass
        done.set()
    threading.Thread(target=call, daemon=True).start()
    return done.wait(secs)


# ------------------------------------------------------------------ the same numbers
@pytest.mark.parametrize("name", ["autopilot", "zap_mdn", "silence_and_watch"])
def test_a_brain_in_the_shared_process_gives_the_golden_hash(conn, name):
    from tools.golden_hashes import run_hash
    assert run_hash(conn, name, game_kwargs={"brain_procs": "server"}) == GOLDEN[name]


def test_the_handle_is_the_games_brain(sgame, conn):
    g = sgame
    io = g.fly(0).io
    assert isinstance(io, GpuBrainHandle) and g.brain_mode == "server" and g.brain is None and io.brain is None
    assert io.proc is g.brain_server.proc and io.proc.is_alive() and io.fly == 0
    local = build_brain(conn, "game", seed=0)
    assert io.fingerprint == RT.fingerprint(local)
    assert io.settings() == local.settings()
    assert io.has_plasticity and io.parts_info() is None and io.parts_status() is None
    rate, role = io.neuron(int(conn.select("MN9/L")[0]))
    assert rate == 0.0 and role is None
    with pytest.raises(NotImplementedError):
        io.history()


def test_two_flies_in_one_process_give_the_numbers_of_a_process_per_brain_and_of_this_process(conn, fconn):
    seen = {}
    for procs in ("server", "on", "off"):
        g = _pair(conn, fconn, procs=procs)
        try:
            if procs == "server":
                assert all(isinstance(f.io, GpuBrainHandle) for f in g.flies) and g.brain_mode == "server"
                assert g.flies[0].io.proc is g.flies[1].io.proc and g.brain_server.proc.is_alive()
                assert sorted(g.brain_server.handles) == [0, 1]
            elif procs == "on":
                assert all(isinstance(f.io, ProcessBrain) for f in g.flies) and g.brain_mode == "procs"
            else:
                assert all(isinstance(f.io, LocalBrain) for f in g.flies) and g.brain_mode == "local"
            assert g.action({"type": "zap", "spec": "GNG232", "hz": 150, "secs": 1.0})["ok"]
            rows = []
            for _ in range(20):
                g.tick()
                rows.append([(f.bt.hz, f.bt.spikes_shown, f.bt.n_spikes, f.bt.gf) for f in g.flies])
            seen[procs] = rows
            assert any(r[0][2] > 0 for r in rows) and any(r[1][2] > 0 for r in rows)
        finally:
            g.close()
        if procs == "server":
            assert not g.brain_server.proc.is_alive() and g.brain_server.proc.exitcode == 0
    assert seen["server"] == seen["on"] == seen["off"]


def test_the_lockstep_is_one_message_for_every_fly(conn, fconn):
    g = _pair(conn, fconn)
    try:
        sent = []
        real = g.brain_server._send
        g.brain_server._send = lambda fly, msg: (sent.append((fly, msg[0], [m[0] for m in msg[1]] if msg[0] == "advance_all" else None)), real(fly, msg))[1]
        g.tick()
        ticks = [m for m in sent if m[1] == "advance_all"]
        assert len(ticks) == 1 and ticks[0][2] == [0, 1]                      # one message, both flies, in order
        assert isinstance(g.flies[1].bt, BrainTick)
        assert _answers_from_another_thread(g.flies[0].io)                    # the pipe is free again
    finally:
        g.close()


def test_a_local_brain_and_a_handle_advance_in_one_lockstep(conn):
    g_loc, g_srv = _game(conn, procs="off"), _game(conn, procs="server")
    try:
        a, b = g_loc.fly(0), g_srv.fly(0)
        dt = TICK_MS / 1000.0
        rates = {"LB3b,LB3c": 120.0}
        empty = np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float32)
        for seq in range(5):
            ticks = advance_all([a.io, b.io], [a.advance_input(dt, rates, *empty, seq), b.advance_input(dt, rates, *empty, seq)])
            assert ticks[0].hz == ticks[1].hz and ticks[0].spikes_shown == ticks[1].spikes_shown
        assert ticks[0].hz["MN9"] > 0
        bt = b.io.advance(dt, rates, *empty, 9, b.readouts)                   # the plain per-fly tick still works
        assert isinstance(bt, BrainTick) and bt.stims == 1
    finally:
        g_loc.close()
        g_srv.close()


# ------------------------------------------------------------------ the commands, for the second fly
def test_commands_round_trip_for_the_second_fly(conn, fconn):
    g = _pair(conn, fconn)
    try:
        io = g.flies[1].io
        n = int(fconn.select("MN9").size)
        assert g.action({"type": "silence", "spec": "MN9", "fly": 1}) == {"ok": True, "n": n}
        assert g.action({"type": "zap", "spec": "GNG232", "hz": 200, "secs": 2.0, "fly": 1})["ok"]
        for _ in range(12):
            g.tick()
        bt = g.flies[1].bt
        assert "MN9" in bt.silenced_specs and g.state_dict["flies"][1]["silenced"] == ["MN9"]
        assert bt.hz["MN9"] > 0 and bt.motor_hz["MN9"] == 0.0            # it fires, nothing hears it
        assert "MN9" not in g.flies[0].bt.silenced_specs                   # the first fly was left alone
        assert g.action({"type": "unsilence", "fly": 1})["ok"]
        g.tick()
        assert io.modulate("MN9", 1.5) == n and io.settings()["modulated"] == {"MN9": 1.5}
        io.unmodulate("MN9")
        assert io.settings()["modulated"] == {}
        assert g.action({"type": "watch", "spec": "GNG232", "key": "relay", "fly": 1})["ok"]
        g.tick()
        assert "relay" in g.flies[1].bt.hz and "relay" in g.flies[1].histories and "relay" not in g.flies[0].bt.hz
        assert g.action({"type": "unwatch", "key": "relay", "fly": 1})["ok"]
        g.tick()
        assert "relay" not in g.flies[1].bt.hz
        assert g.action({"type": "learning", "on": False, "fly": 1})["ok"]
        g.tick()
        assert io.learning_summary()["settings"]["enabled"] is False and g.flies[0].io.learning_summary()["settings"]["enabled"] is True
        assert g.action({"type": "learning", "on": True, "forget": True, "fly": 1})["ok"]
        g.tick()
        assert g.action({"type": "record", "on": True, "spikes": True})["ok"]
        for _ in range(3):
            g.tick()
        assert io.record("active") is True
        assert g.action({"type": "record", "on": False})["ok"]
        g.tick()
        t_ms, idx = io.record("arrays")
        assert isinstance(t_ms, np.ndarray) and t_ms.size == idx.size > 0
        rate, role = io.neuron(int(fconn.select("MN9/L")[0]))
        assert rate >= 0.0 and role is None
        assert io.fingerprint == RT.fingerprint(build_brain(fconn, "game", seed=1001)) and io.parts_status() is None
        assert g.action({"type": "calm", "fly": 1})["ok"]
        g.tick()
        assert g.state_dict["msg"] == "Brain reset to rest."
        assert all(len(h) == 1 for h in g.flies[1].histories.values()) and all(len(h) > 1 for h in g.flies[0].histories.values())
    finally:
        g.close()


# ------------------------------------------------------------------ the process's life
def test_close_leaves_no_child_and_is_idempotent(conn, fconn, monkeypatch):
    g = _pair(conn, fconn)
    proc = g.brain_server.proc
    calls = []
    real_close = g.brain_server.close
    monkeypatch.setattr(g.brain_server, "close", lambda: (calls.append(1), real_close()))
    loop = threading.Thread(target=g.loop, daemon=True)
    loop.start()
    time.sleep(0.3)
    assert loop.is_alive()
    g.stop_loop.set()
    loop.join(10)
    g.close()                                    # as serve does: after the loop thread has ended
    proc.join(5)
    assert calls == [1, 1, 1] and not proc.is_alive() and proc.exitcode == 0   # two handles and the game: one process stopped
    g.close()                                    # idempotent
    assert not proc.is_alive()


def test_a_child_that_dies_is_reported_not_waited_for(conn, fconn):
    g = _pair(conn, fconn)
    io = g.flies[0].io
    io.proc.kill()
    io.proc.join(5)
    t0 = time.time()
    with pytest.raises(RuntimeError, match="fly 0 stopped .*exit code"):
        g.tick()
    assert time.time() - t0 < 5
    # every lock let go: both handles refuse at once, and the game closes
    assert _answers_from_another_thread(g.flies[1].io) and io.closed and g.flies[1].io.closed
    with pytest.raises(RuntimeError, match="closed"):
        g.flies[1].io.settings()
    g.close()
    assert not io.proc.is_alive()


def test_a_paused_game_survives_its_brain_process_dying(conn):
    g = _game(conn)
    try:
        g.paused = True
        loop = threading.Thread(target=g.loop, daemon=True)
        loop.start()
        time.sleep(0.2)
        g.fly(0).io.proc.kill()
        time.sleep(1.0)
        assert loop.is_alive() and "brain process of fly 0" in g.message
    finally:
        g.stop_loop.set()
        loop.join(5)
        g.close()


def test_a_child_that_stops_answering_is_stopped_not_waited_for(conn, monkeypatch):
    g = _game(conn)
    io = g.fly(0).io
    try:
        monkeypatch.setattr(GpuBrainServer, "REQUEST_TIMEOUT", 0.3)
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


def test_a_rebuild_that_fails_in_the_child_keeps_the_game_usable(sgame, monkeypatch):
    g = sgame
    io = g.fly(0).io
    monkeypatch.setattr(io, "swap", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no room for a second brain")))
    assert g.action({"type": "parts", "on": True})["ok"]
    t0 = time.time()
    while g.genome["growing"] is not None and time.time() - t0 < 30:
        g.tick()
        time.sleep(0.01)
    assert g.genome["growing"] is None and g.genome["error"] == "no room for a second brain"
    assert not g.parts_on and io.parts_status() is None      # the old fly stayed
    assert g.action({"type": "parts", "on": True})["ok"]     # the next attempt is taken, not refused


def test_a_failed_send_in_the_lockstep_leaves_every_lock_free(conn, fconn, monkeypatch):
    """The tick's one message fails to go out: the tick raises, the flies already sent get the same failure at once
    (advance_all's finally), the pipe is free, and the next tick works."""
    g = _pair(conn, fconn)
    try:
        server = g.brain_server
        real = server._send
        state = {"fail": True}

        def failing(fly, msg):
            if state["fail"] and msg[0] == "advance_all":
                state["fail"] = False
                raise RuntimeError(f"the brain process of fly {fly} stopped (exit code -9): test")
            return real(fly, msg)
        monkeypatch.setattr(server, "_send", failing)
        with pytest.raises(RuntimeError, match="fly 0 stopped"):
            g.tick()
        assert server._tick is None and _answers_from_another_thread(g.flies[1].io) and not server.closed
        g.tick()                                                         # the next tick is a clean one
        assert isinstance(g.flies[1].bt, BrainTick) and server._tick is None
    finally:
        g.close()


def test_a_cuda_error_in_the_child_is_fatal_and_reported_once(conn, monkeypatch):
    """An error whose name says CUDA stops the child after one reply (a poisoned context would fail every tick)."""
    g = _game(conn)
    io = g.fly(0).io
    try:
        class CUDARuntimeError(RuntimeError):
            pass
        assert brainio._fatal(CUDARuntimeError("device lost")) and not brainio._fatal(ValueError("x"))
        # drive the child loop in this process with a brain whose handle raises it, through the serving function
        class Dead:
            def handle(self, cmd, *args):
                raise CUDARuntimeError("device lost")
        with pytest.raises(CUDARuntimeError):
            brainio._serve({0: Dead()}, ("settings", 0))
        with pytest.raises(ValueError, match="no brain for fly 7"):
            brainio._serve({0: Dead()}, ("settings", 7))
    finally:
        g.close()
    assert not io.proc.is_alive()


# ------------------------------------------------------------------ choosing the server
class _Recorder:
    """A stand-in for GpuBrainServer that builds the brains here (on the CPU) and records what it was asked for."""
    specs = []

    def __init__(self):
        self.proc = None
        self.closed = False
        self.handles = {}

    def attach(self, spec, fly):
        from virtual_fly.connectome import load_connectome
        _Recorder.specs.append((fly, spec))
        conn = load_connectome(spec["path"], quiet=True)
        kw = {**spec["brain_kwargs"], "backend": "auto"}
        io = LocalBrain(build_brain(conn, spec["profile"], **kw))
        self.handles[fly] = io
        return io

    def close(self):
        self.closed = True


def test_the_server_is_chosen_for_a_gpu_brain_and_refused_with_a_process_per_brain(conn, fconn, monkeypatch):
    monkeypatch.setattr(brainio, "GpuBrainServer", _Recorder)
    _Recorder.specs = []
    kw = {"backend": "cupy", "seed": 0}
    g = Game(build_brain(conn, "game", seed=0), autopilot=False, seed=1, brain_kwargs=kw, partner={"conn": fconn})
    try:
        assert g.brain_mode == "server" and isinstance(g.brain_server, _Recorder)
        assert [f for f, _ in _Recorder.specs] == [0, 1]
        assert all(s["brain_kwargs"]["backend"] == "cupy" for _, s in _Recorder.specs)   # the process builds the GPU brain
        g.tick()
    finally:
        g.close()
    assert g.brain_server.closed
    with pytest.raises(ValueError, match="brain_procs='on' cannot hold a cupy brain"):
        Game(build_brain(conn, "game", seed=0), autopilot=False, brain_kwargs=kw, brain_procs="on")
    with pytest.raises(ValueError, match="same backend"):
        Game(build_brain(conn, "game", seed=0), autopilot=False, brain_kwargs=kw, partner={"conn": fconn, "brain_kwargs": {"seed": 0}})
    with pytest.raises(ValueError, match="brain_procs"):
        Game(build_brain(conn, "game", seed=0), brain_procs="yes")
    # a GPU brain with brain_procs off stays in this process
    g = Game(build_brain(conn, "game", seed=0), autopilot=False, brain_kwargs=kw, brain_procs="off")
    assert g.brain_mode == "local" and isinstance(g.fly(0).io, LocalBrain) and g.brain_server is None
    g.close()
    assert Game._brain_mode("auto", None, None) == "local" and Game._brain_mode("auto", {}, {"conn": fconn}) == "procs"
    assert Game._brain_mode("server", None, None) == "server" and Game._brain_mode("on", {}, None) == "procs"


def test_everything_built_beside_a_gpu_brain_runs_on_the_cpu(conn, monkeypatch):
    """The re-test child, the re-test's thread fallback and the default factory never build a second GPU brain in
    the game process; the survival cache does not care which integrator ran (plan 6.5)."""
    g = _game(conn, procs="server")
    try:
        a = g.fly(0)
        key_cpu = a._survival_key(a.conn)
        real = a.io.settings
        monkeypatch.setattr(a.io, "settings", lambda: {**real(), "backend": "cupy"})
        spec = g._retest_spec(a.conn)
        assert spec["brain_kwargs"]["backend"] == "auto"
        assert a._cpu_only() == {"backend": "auto"}
        assert a._default_brain_factory(a.conn).backend != "cupy"
        assert a._survival_key(a.conn) == key_cpu
    finally:
        g.close()
    g = _game(conn, procs="off")
    try:
        a = g.fly(0)
        assert a._cpu_only() == {} and g._retest_spec(a.conn)["brain_kwargs"]["backend"] == a.io.settings()["backend"]
    finally:
        g.close()


def test_play_refuses_a_process_per_brain_with_the_gpu_before_loading(capsys, monkeypatch):
    from virtual_fly import gpubrain, play
    monkeypatch.setattr(gpubrain, "unavailable_reason", lambda: None)
    monkeypatch.setattr(play, "load_connectome", lambda **k: pytest.fail("loaded the data first"))
    with pytest.raises(SystemExit) as e:
        play.main(["--backend", "cupy", "--brain-procs", "on", "--no-browser"])
    assert e.value.code == 2
    err = capsys.readouterr().err
    assert "error: --brain-procs on cannot hold a GPU brain" in err and "Traceback" not in err
