"""
The seam between the game and a fly's brain: :class:`LocalBrain` (the brain in this process, the single fly's way)
and :class:`ProcessBrain` (the same brain in a child process of its own, one per fly when there are two: the
two-flies work, docs/TWO_FLIES_PLAN.md 5.3). Both answer the same calls with the same numbers, so the game never
looks inside a :class:`~virtual_fly.brain.FlyBrain` itself: per tick it hands the seam the sensory rates and gets a
:class:`BrainTick` back, and every other thing the page or a lab action needs (silencing, watches, the learning
summary, the spike recording, the brain's settings) is one call on the seam.

Why a process per brain rather than a thread: the compiled kernels release the interpreter lock, but the Python
around them does not, and a second brain in a thread slowed the game to about three quarters of real time
(docs/SCIENCE.md 2.3, retest.py). A child process shares nothing with the game loop. The child is started with
``spawn`` (as the re-test process is), loads the connectome file itself, builds the brain with the same settings,
and reports the brain's fingerprint (retest.fingerprint) so a test can check it built the game's brain. Messages
go over one :func:`multiprocessing.Pipe`, one request in and one reply out, under one lock per brain: the server's
HTTP threads and the game thread share the pipe, and without the lock their messages would interleave.

With the GPU backend (docs/TWO_FLIES_PLAN.md 6.5) every brain of the game lives in one child instead
(:class:`GpuBrainServer`, one :class:`GpuBrainHandle` per fly): one CUDA context for both brains, and one
``advance_all`` message per tick. The protocol is the same, with the fly's number in each message.

What crosses the pipe each tick is small: the rates dict and the columnar arrays in, a BrainTick out (readouts, a
sample of at most 2,500 spikes for the brain map, the learning summary, the parts status: a few kB). The parent
keeps each readout's history itself from the ticks (one value per 25 ms tick, the game's own monitor bin), so
``/api/history`` needs no round trip.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import signal
import threading
import time
from dataclasses import dataclass

import numpy as np

from . import retest as retestlib
from .game import HIDDEN_READOUTS, READOUTS, TICK_MS
from .plasticity import APPROACH_NTS, AVOID_NTS

SPIKES_SHOWN = 2500          # at most this many spikes are drawn on the brain map per tick (a random sample)
HISTORY_CAP, HISTORY_KEEP = 100000, 50000     # as brain.Monitor keeps its history


@dataclass
class BrainTick:
    """What the world needs from one brain after one 25 ms tick."""
    hz: dict                    # readouts from all counts (Hz per cell)
    motor_hz: dict              # readouts from live counts (silenced cells zeroed): what reaches the body
    gf: int                     # live giant-fibre spikes this tick
    n_spikes: int               # every event this tick (spikes and, with the parts list, graded quanta)
    n_graded: int               # graded quanta among them
    spikes_shown: list          # at most SPIKES_SHOWN soma-bearing neuron indices, sampled as the page has always seen them
    learned: float              # the learned part of odour steering: 1.2 * (wv - wa) from the KC->MBON synapses, or 0.0
    learn: tuple                # (plasticity.events, depressed_fraction()) for the "learn" check, or (0, 0.0)
    learning: dict | None       # {"mbon", "depressed_fraction", "events", "settings"} from the plasticity, or None
    stims: int                  # len(brain.stim), for the state's "stims"
    silenced_specs: list        # brain.silenced's specs, for the state's "baseline"
    parts_status: dict | None   # brain.parts_status(), for the genome status


def _learned(pl) -> float:
    """The learned bias odour steering reads off the KC->MBON synapses (agent.py's odour_steering, verbatim)."""
    learned = 0.0
    if pl is not None:
        trace = pl.kc_trace
        act = trace[pl.pe_kc]                       # eligibility of each plastic synapse's KC
        if act.sum() > 0:
            nt = pl.mbon_nt[pl.pe_mbon]                  # the MBONs' transmitters (plasticity.mbon_transmitters)
            depression = 1.0 - pl.scale
            app = np.isin(nt, APPROACH_NTS)
            av = np.isin(nt, AVOID_NTS)
            wa = float((act * depression)[app].sum() / max(act[app].sum(), 1e-6))
            wv = float((act * depression)[av].sum() / max(act[av].sum(), 1e-6))
            learned = 1.2 * (wv - wa)               # avoid-MBON synapses weakened -> approach more
    return learned


def _learning(pl, conn) -> dict | None:
    if pl is None:
        return None
    return {"mbon": pl.summary(conn), "depressed_fraction": pl.depressed_fraction(), "events": pl.events,
            "settings": pl.settings()}


def step_tick(brain, dt: float, rates: dict, col_idx, col_hz):
    """Drive the brain with this tick's rates and step it for TICK_MS (the single fly's code, unchanged).
    Returns (spikes, counts, live counts)."""
    b = brain
    with b.lock:
        if col_idx.size:
            b.set_stimulus_arrays(col_idx, col_hz, extra=rates)
        else:
            b.set_stimuli(rates)
        b.reset_counts()
        spikes = b.advance(int(round(TICK_MS / b.dt)))   # every spike of the tick in order (the GPU runs whole chunks)
        counts = b.spike_count
        live = np.where(b.silenced_mask(), 0, counts) if b.silenced else counts   # what reaches the body
    return spikes, counts, live


def tick_fields(brain, has_soma, readouts: dict, spikes, counts, live, dt: float, seq: int,
                want_learned: bool = False) -> BrainTick:
    """Everything the game reads from a brain after a tick, computed the way the single fly always computed it.
    ``want_learned``: the fly smells something, so odour steering will read the learned bias (a process brain
    computes it here, once, from the same arrays odour steering read; it is 0.0 whenever the fly smells nothing)."""
    hz = {k: (float(counts[i].sum()) / max(1, i.size) / dt if i.size else 0.0) for k, i in readouts.items()}
    motor_hz = {k: (float(live[i].sum()) / max(1, i.size) / dt if i.size else 0.0) for k, i in readouts.items()}
    gf = int(live[readouts["GF"]].sum())
    n_graded = int(brain._gmask[spikes].sum()) if spikes.size and brain._graded_idx.size else 0
    vis = spikes[has_soma[spikes]]
    if vis.size > SPIKES_SHOWN:
        vis = np.random.default_rng(seq).choice(vis, SPIKES_SHOWN, replace=False)
    pl = brain.plasticity
    return BrainTick(hz, motor_hz, gf, int(spikes.size), n_graded, vis.tolist(), _learned(pl) if want_learned else 0.0,
                     (pl.events, pl.depressed_fraction()) if pl is not None else (0, 0.0), _learning(pl, brain.conn),
                     len(brain.stim), sorted(brain.silenced), brain.parts_status())


def status_fields(brain) -> dict:
    """The BrainTick fields that describe the brain's state rather than its tick (the learning summary, the parts
    status, the stimulus count, the silenced specs), read from the brain as it is now: after the watchdog has reset
    it, the frame of that tick publishes these post-reset values, as the single fly always did."""
    return {"learning": _learning(brain.plasticity, brain.conn), "parts_status": brain.parts_status(),
            "stims": len(brain.stim), "silenced_specs": sorted(brain.silenced)}


class LocalBrain:
    """A fly's brain in this process: the calls go straight to the :class:`~virtual_fly.brain.FlyBrain`."""

    parent_histories = False    # the brain's own monitors keep the readout histories

    def __init__(self, brain):
        self._brain = brain
        self._has_soma = None

    @property
    def brain(self):
        return self._brain

    @brain.setter
    def brain(self, b):
        self._brain, self._has_soma = b, None

    @property
    def has_soma(self):
        if self._has_soma is None:
            self._has_soma = ~np.isnan(self._brain.conn.soma[:, 0])
        return self._has_soma

    @property
    def has_plasticity(self) -> bool:
        return self._brain.plasticity is not None

    # ------------------------------------------------------------------ per tick
    def advance(self, dt: float, rates: dict, col_idx, col_hz, seq: int, readouts: dict, want_learned: bool = False) -> BrainTick:
        spikes, counts, live = step_tick(self._brain, dt, rates, col_idx, col_hz)
        return tick_fields(self._brain, self.has_soma, readouts, spikes, counts, live, dt, seq)

    def learned(self, bt: BrainTick) -> float:
        """The learned odour bias, read off the plasticity arrays now (as the single fly always read it)."""
        return _learned(self._brain.plasticity)

    def peek(self, dt: float, seq: int, readouts: dict) -> BrainTick:
        """The same fields without stepping (the paused game keeps publishing)."""
        b = self._brain
        spikes = np.zeros(0, dtype=np.int64)
        with b.lock:
            counts = b.spike_count
            live = np.where(b.silenced_mask(), 0, counts) if b.silenced else counts
        return tick_fields(b, self.has_soma, readouts, spikes, counts, live, dt, seq)

    def send_advance(self, dt, rates, col_idx, col_hz, seq, readouts, want_learned=False):
        self._pending = (dt, rates, col_idx, col_hz, seq, readouts, want_learned)

    def recv_advance(self) -> BrainTick:
        args, self._pending = self._pending, None
        return self.advance(*args)

    # ------------------------------------------------------------------ commands
    def silence(self, spec: str) -> int:
        with self._brain.lock:
            return self._brain.silence(spec)

    def unsilence(self, spec: str | None = None):
        with self._brain.lock:
            self._brain.unsilence(spec)

    def modulate(self, spec: str, factor: float) -> int:
        with self._brain.lock:
            return self._brain.modulate(spec, factor)

    def unmodulate(self, spec: str | None = None):
        with self._brain.lock:
            self._brain.unmodulate(spec)

    def add_monitor(self, key: str, spec: str):
        self._brain.add_monitor(key, spec, bin_ms=TICK_MS)

    def remove_monitor(self, key: str):
        self._brain.remove_monitor(key)

    def history(self, keys=None, n: int = 400):
        """(bin_ms, {key: the last n values}) from the brain's monitors, as /api/history has always served them."""
        mon = self._brain.monitors
        out = {}
        for k in keys or list(mon):
            m = mon.get(k)
            if m is not None:
                out[k] = m.history[-n:]
        return (mon and next(iter(mon.values())).bin_ms), out

    def reset(self):
        with self._brain.lock:
            self._brain.reset()

    def learning(self, on: bool | None = None, forget: bool = False):
        pl = self._brain.plasticity
        if pl is None:
            return
        if forget:
            with self._brain.lock:
                pl.reset_weights()
        if on is not None:
            pl.enabled = bool(on)

    def record(self, op: str):
        b = self._brain
        if op == "start":
            b.start_recording()
        elif op == "stop":                        # the take is kept for download until the next start
            b.recording_kept = b.stop_recording()
            return len(b.recording_kept)
        elif op == "clear_kept":
            b.recording_kept = []
        elif op == "active":
            return b.recording is not None
        elif op == "arrays":
            with b.lock:                          # the game thread appends to the live recording
                rec = list(b.recording) if b.recording is not None else list(getattr(b, "recording_kept", []))
            return b.recording_arrays(rec)
        else:
            raise ValueError(f"unknown recording operation {op!r}")

    def settings(self) -> dict:
        return self._brain.settings()

    def parts_status(self) -> dict | None:
        return self._brain.parts_status()

    def parts_info(self):
        p = self._brain.parts
        return None if p is None else (p.parts, p.counts)

    def learning_summary(self) -> dict | None:
        return _learning(self._brain.plasticity, self._brain.conn)

    def depressed_fraction(self):
        """The share of plastic synapses that have been depressed (0 without plasticity): what a scenario's
        measure reads every tick, so only that number, not the whole learning summary."""
        pl = self._brain.plasticity
        return pl.depressed_fraction() if pl is not None else 0

    def status_fields(self) -> dict:
        return status_fields(self._brain)

    def neuron(self, i: int):
        b = self._brain
        return (float(b.spike_count[i]) / max(b.window_ms, 1) * 1000.0,
                b.parts.role(i) if b.parts is not None else None)

    def fingerprint(self) -> str:
        return retestlib.fingerprint(self._brain)

    def close(self):
        pass


# ---------------------------------------------------------------------------------------------- the child process
class _ChildBrain:
    """The brain and its bookkeeping inside the child process (one instance per process)."""

    def __init__(self, spec: dict):
        from . import parts, vfb
        from .connectome import load_connectome
        from .settings import build_brain
        if spec.get("vfb") is not None:                  # the game runs on data swapped in at runtime (tests do)
            vfb.use(*spec["vfb"])
        if spec.get("regions") is not None:
            parts.use_region_table(spec["regions"]["table"])
        self.build_brain = build_brain
        self.profile = spec["profile"]
        self.brain_kwargs = dict(spec["brain_kwargs"])
        self.base_conn = load_connectome(spec["path"], quiet=True)
        conn = self.base_conn
        w = spec.get("wiring")
        if w is not None:
            conn = conn.rewired(w["row_ptr"], w["post_idx"], w["n_syn"], label=w.get("label", "grown"))
        self.readout_specs: dict[str, str] = dict(spec["readouts"])
        self._install(self.build_brain(conn, self.profile, **self.brain_kwargs))

    def _install(self, brain):
        self.brain, self.conn = brain, brain.conn
        self.has_soma = ~np.isnan(self.conn.soma[:, 0])
        self.readouts = {k: self.conn.select(s) for k, s in self.readout_specs.items()}

    def info(self) -> dict:
        p = self.brain.parts
        return {"fingerprint": retestlib.fingerprint(self.brain), "has_plasticity": self.brain.plasticity is not None,
                "settings": self.brain.settings(), "parts_counts": None if p is None else p.counts}

    def handle(self, cmd: str, *args):
        b = self.brain
        if cmd == "advance":
            dt, rates, col_idx, col_hz, seq, want_learned = args
            spikes, counts, live = step_tick(b, dt, rates, col_idx, col_hz)
            return tick_fields(b, self.has_soma, self.readouts, spikes, counts, live, dt, seq, want_learned)
        if cmd == "peek":
            dt, seq = args
            spikes = np.zeros(0, dtype=np.int64)
            with b.lock:
                counts = b.spike_count
                live = np.where(b.silenced_mask(), 0, counts) if b.silenced else counts
            return tick_fields(b, self.has_soma, self.readouts, spikes, counts, live, dt, seq)
        if cmd == "silence":
            with b.lock:
                return b.silence(args[0])
        if cmd == "unsilence":
            with b.lock:
                b.unsilence(args[0])
            return None
        if cmd == "modulate":
            with b.lock:
                return b.modulate(args[0], args[1])
        if cmd == "unmodulate":
            with b.lock:
                b.unmodulate(args[0])
            return None
        if cmd == "add_monitor":
            key, spec = args
            if key not in self.readouts:
                self.readout_specs[key] = spec
                self.readouts[key] = self.conn.select(spec)
            return None
        if cmd == "remove_monitor":
            self.readout_specs.pop(args[0], None)
            self.readouts.pop(args[0], None)
            return None
        if cmd == "reset":
            with b.lock:
                b.reset()
            return None
        if cmd == "learning":
            on, forget = args
            pl = b.plasticity
            if pl is not None:
                if forget:
                    with b.lock:
                        pl.reset_weights()
                if on is not None:
                    pl.enabled = bool(on)
            return None
        if cmd == "record":
            return LocalBrain(b).record(args[0])
        if cmd == "settings":
            return b.settings()
        if cmd == "parts_status":
            return b.parts_status()
        if cmd == "parts_info":
            return None if b.parts is None else (b.parts.parts, b.parts.counts)
        if cmd == "learning_summary":
            return _learning(b.plasticity, b.conn)
        if cmd == "depressed_fraction":
            return LocalBrain(b).depressed_fraction()
        if cmd == "status_fields":
            return status_fields(b)
        if cmd == "neuron":
            return LocalBrain(b).neuron(args[0])
        if cmd == "fingerprint":
            return retestlib.fingerprint(b)
        if cmd == "swap":
            wiring, parts_arg, silenced, modulated, learning_on = args
            conn = self.base_conn
            if wiring is not None:
                conn = conn.rewired(wiring["row_ptr"], wiring["post_idx"], wiring["n_syn"], label=wiring.get("label", "grown"))
            kw = dict(self.brain_kwargs)
            kw["parts"] = parts_arg
            new = self.build_brain(conn, self.profile, **kw)
            with b.lock:                                  # nothing steps the old brain while the new one takes over
                self._install(new)
                for spec in silenced:
                    try:
                        new.silence(spec)
                    except ValueError:
                        pass
                for spec, factor in modulated.items():
                    try:
                        new.modulate(spec, factor)
                    except ValueError:
                        pass
                if new.plasticity is not None:
                    new.plasticity.enabled = learning_on
            return self.info()
        raise ValueError(f"unknown brain command {cmd!r}")


def _child(pipe, spec: dict):
    """The brain process (module level, so that ``spawn`` can import it)."""
    try:
        signal.signal(signal.SIGINT, signal.SIG_IGN)     # Ctrl+C in the game's console is the game's to handle
    except (ValueError, OSError):
        pass
    try:
        state = _ChildBrain(spec)
        pipe.send(("ready", state.info()))
    except BaseException as e:                    # report every failure, never leave the game waiting
        try:
            pipe.send(("error", f"{type(e).__name__}: {e}"))
        finally:
            return
    parent = mp.parent_process()
    while True:
        while not pipe.poll(1.0):
            if parent is not None and not parent.is_alive():      # the game has gone: stop quietly
                os._exit(0)
        try:
            msg = pipe.recv()
        except EOFError:
            os._exit(0)
        if msg[0] == "close":
            break
        try:
            reply = ("ok", state.handle(msg[0], *msg[1:]))
        except Exception as e:
            reply = ("err", type(e).__name__, str(e))
        pipe.send(reply)
    pipe.close()


_warmed = False


def warm_up_once():
    """Compile or load the numba kernels once in the parent, so spawned children load the cache instead of racing
    to compile it."""
    global _warmed
    if not _warmed:
        from . import fastbrain
        fastbrain.warm_up()
        _warmed = True


class _PipeClient:
    """The parent's end of a brain child: one request in and one reply out over one pipe, under one lock, and a
    child that dies or stops answering reported as "the brain process of fly k ..." (the fly the request was for),
    never waited for. :class:`ProcessBrain` (one brain per child) and :class:`GpuBrainServer` (every brain in one
    child) share it."""

    START_TIMEOUT = 600.0       # the child loads the connectome and builds the brain (seconds to minutes on a slow disk)
    REQUEST_TIMEOUT = 300.0     # any later request (a swap rebuilds the brain: seconds); a child slower than this is stopped
    proc = None

    def _start(self, target, args: tuple, name: str):
        self.closed = False
        self._lock = threading.Lock()
        ctx = mp.get_context("spawn")
        self._pipe, child_end = ctx.Pipe()
        self.proc = ctx.Process(target=target, args=(child_end, *args), daemon=True, name=name)
        self.proc.start()
        child_end.close()

    def _dead(self, fly: int, why: str = "") -> RuntimeError:
        self.closed = True
        self.proc.join(timeout=1.0)               # a child that has just died is reaped first, so its exit code is known
        return RuntimeError(f"the brain process of fly {fly} stopped (exit code {self.proc.exitcode}){why}")

    def _recv(self, fly: int, timeout: float | None = None):
        """The next message, or a RuntimeError naming the fly and the exit code if the child has died: never a hang.
        A child that does not answer within ``timeout`` (REQUEST_TIMEOUT by default) is stopped and the seam marked
        closed, so that its late answer can never be taken for a later request's."""
        timeout = self.REQUEST_TIMEOUT if timeout is None else timeout
        deadline = time.monotonic() + timeout
        while True:
            try:
                if self._pipe.poll(max(0.0, min(0.5, deadline - time.monotonic()))):
                    return self._pipe.recv()
            except (EOFError, OSError, BrokenPipeError):
                raise self._dead(fly, ": its pipe closed") from None
            if not self.proc.is_alive():
                if self._pipe.poll(0.0):                   # it may have answered just before it stopped
                    continue
                raise self._dead(fly)
            if time.monotonic() > deadline:
                self.closed = True
                if self.proc.is_alive():
                    self.proc.terminate()
                raise RuntimeError(f"the brain process of fly {fly} did not answer within {timeout:g} s: "
                                   "it was stopped")

    def _send(self, fly: int, msg):
        if self.closed:
            raise RuntimeError(f"the brain process of fly {fly} is closed")
        try:
            self._pipe.send(msg)
        except (OSError, BrokenPipeError, EOFError):
            raise self._dead(fly, ": its pipe closed") from None

    def _reply(self, fly: int):
        kind, *rest = self._recv(fly)
        if kind == "ok":
            return rest[0]
        if kind == "err":
            name, text = rest
            exc = {"ValueError": ValueError, "KeyError": KeyError, "TypeError": TypeError}.get(name, RuntimeError)
            raise exc(text)
        raise RuntimeError(f"the brain process of fly {fly} sent {kind!r}")

    def _request(self, fly: int, *msg):
        with self._lock:
            self._send(fly, msg)
            return self._reply(fly)

    def close(self):
        """Ask the child to exit and wait for it (terminate, then kill, if it does not); safe to call twice."""
        if self.proc is None:
            return
        if not self.closed:
            self.closed = True
            if self._lock.acquire(timeout=2.0):
                try:
                    if self.proc.is_alive():
                        try:
                            self._pipe.send(("close",))
                        except (OSError, BrokenPipeError, EOFError):
                            pass
                finally:
                    self._lock.release()
        self.proc.join(timeout=5.0)
        if self.proc.is_alive():
            self.proc.terminate()
            self.proc.join(timeout=2.0)
        if self.proc.is_alive():
            self.proc.kill()
            self.proc.join(timeout=2.0)
        try:
            self._pipe.close()
        except OSError:
            pass


class ProcessBrain(_PipeClient):
    """A fly's brain in a child process of its own; the same calls as :class:`LocalBrain`, each a request and a reply
    over one pipe under one lock. ``brain`` is None: the parent has no FlyBrain of its own."""

    parent_histories = True     # the FlyAgent keeps the readout histories from the ticks
    brain = None

    def __init__(self, spec: dict, fly: int = 0):
        warm_up_once()
        self.fly = fly
        self._start(_child, (spec,), f"fly-brain-{fly}")
        kind, info = self._recv(fly, self.START_TIMEOUT)
        if kind != "ready":
            self.close()
            raise RuntimeError(f"the brain process of fly {fly} could not start: {info}")
        self.fingerprint: str = info["fingerprint"]
        self.has_plasticity: bool = info["has_plasticity"]
        self.parts_counts = info["parts_counts"]

    # ------------------------------------------------------------------ per tick
    def advance(self, dt: float, rates: dict, col_idx, col_hz, seq: int, readouts: dict = None,
                want_learned: bool = False) -> BrainTick:
        return self._request(self.fly, "advance", dt, rates, col_idx, col_hz, seq, want_learned)

    def learned(self, bt: BrainTick) -> float:
        """The learned odour bias this tick's answer carries (the child read it from the same arrays, after its step;
        nothing changes them between the step and odour steering)."""
        return bt.learned

    def peek(self, dt: float, seq: int, readouts: dict = None) -> BrainTick:
        return self._request(self.fly, "peek", dt, seq)

    def send_advance(self, dt, rates, col_idx, col_hz, seq, readouts=None, want_learned=False):
        """The first half of :meth:`advance`, so that several brains can be sent their input before any is waited
        for (advance_all's lockstep). The lock is taken here and let go in :meth:`recv_advance`: nothing else may
        use the pipe between the two, or its reply and the tick's would cross."""
        self._lock.acquire()
        try:
            self._send(self.fly, ("advance", dt, rates, col_idx, col_hz, seq, want_learned))
        except BaseException:
            self._lock.release()
            raise

    def recv_advance(self) -> BrainTick:
        try:
            return self._reply(self.fly)
        finally:
            self._lock.release()

    # ------------------------------------------------------------------ commands
    def silence(self, spec: str) -> int:
        return self._request(self.fly, "silence", spec)

    def unsilence(self, spec: str | None = None):
        self._request(self.fly, "unsilence", spec)

    def modulate(self, spec: str, factor: float) -> int:
        return self._request(self.fly, "modulate", spec, factor)

    def unmodulate(self, spec: str | None = None):
        self._request(self.fly, "unmodulate", spec)

    def add_monitor(self, key: str, spec: str):
        self._request(self.fly, "add_monitor", key, spec)

    def remove_monitor(self, key: str):
        self._request(self.fly, "remove_monitor", key)

    def history(self, keys=None, n: int = 400):
        raise NotImplementedError("a ProcessBrain's histories live on its FlyAgent (agent.py)")

    def reset(self):
        self._request(self.fly, "reset")

    def learning(self, on: bool | None = None, forget: bool = False):
        self._request(self.fly, "learning", on, forget)

    def record(self, op: str):
        return self._request(self.fly, "record", op)

    def settings(self) -> dict:
        return self._request(self.fly, "settings")

    def parts_status(self) -> dict | None:
        return self._request(self.fly, "parts_status")

    def parts_info(self):
        return self._request(self.fly, "parts_info")

    def learning_summary(self) -> dict | None:
        return self._request(self.fly, "learning_summary")

    def depressed_fraction(self):
        return self._request(self.fly, "depressed_fraction")

    def status_fields(self) -> dict:
        return self._request(self.fly, "status_fields")

    def neuron(self, i: int):
        return self._request(self.fly, "neuron", i)

    def swap(self, wiring: dict | None, parts_arg, silenced, modulated: dict, learning_on: bool) -> dict:
        """Rebuild the brain in the child (a grown wiring, or the parts list switched) and re-apply what the player
        had done to the old one; returns the child's new info (fingerprint, settings, parts counts)."""
        info = self._request(self.fly, "swap", wiring, parts_arg, sorted(silenced), dict(modulated), bool(learning_on))
        self.fingerprint, self.has_plasticity, self.parts_counts = info["fingerprint"], info["has_plasticity"], info["parts_counts"]
        return info


# ---------------------------------------------------------------------------------- every brain in one child process
def _fatal(e: BaseException) -> bool:
    """A CUDA error poisons the process's GPU context: the child says it once and stops, so that the game reports
    "the brain process of fly k stopped" instead of a confusing error on every later tick."""
    name = type(e).__name__
    module = getattr(type(e), "__module__", "") or ""
    return "cuda" in name.lower() or module.split(".")[0] in ("cupy", "cupy_backends", "cupyx")


def _brain_of(brains: dict, fly):
    try:
        return brains[fly]
    except (KeyError, TypeError):
        raise ValueError(f"no brain for fly {fly!r} in this process") from None


def _serve(brains: dict, msg):
    """One message of the shared brain process: ("add_brain", fly, spec), ("advance_all", [(fly, dt, rates, col_idx,
    col_hz, seq, want_learned), ...]) or (cmd, fly, *args) for every command a single brain's child answers."""
    cmd = msg[0]
    if cmd == "add_brain":
        fly, spec = msg[1], msg[2]
        if fly in brains:
            raise ValueError(f"fly {fly} already has a brain in this process")
        brains[fly] = _ChildBrain(spec)
        return brains[fly].info()
    if cmd == "advance_all":                     # every brain's tick back to back (one process, one GPU context)
        return [_brain_of(brains, item[0]).handle("advance", *item[1:]) for item in msg[1]]
    return _brain_of(brains, msg[1]).handle(cmd, *msg[2:])


def _server_child(pipe, specs: list):
    """The brain process that holds every brain of the game (module level, so that ``spawn`` can import it):
    ``specs`` is a list of (fly, spec), the brains built in order before "ready" is sent."""
    try:
        signal.signal(signal.SIGINT, signal.SIG_IGN)     # Ctrl+C in the game's console is the game's to handle
    except (ValueError, OSError):
        pass
    brains: dict = {}
    try:
        for fly, spec in specs:
            brains[fly] = _ChildBrain(spec)
        pipe.send(("ready", [brains[fly].info() for fly, _ in specs]))
    except BaseException as e:                    # report every failure, never leave the game waiting
        try:
            pipe.send(("error", f"{type(e).__name__}: {e}"))
        finally:
            return
    parent = mp.parent_process()
    while True:
        while not pipe.poll(1.0):
            if parent is not None and not parent.is_alive():      # the game has gone: stop quietly
                os._exit(0)
        try:
            msg = pipe.recv()
        except EOFError:
            os._exit(0)
        if msg[0] == "close":
            break
        try:
            reply = ("ok", _serve(brains, msg))
        except Exception as e:
            reply = ("err", type(e).__name__, str(e))
            if _fatal(e):
                try:
                    pipe.send(reply)
                finally:
                    os._exit(3)
        pipe.send(reply)
    pipe.close()


class GpuBrainServer(_PipeClient):
    """One child process holding every brain of the game, each fly reaching its own through a :class:`GpuBrainHandle`
    (docs/TWO_FLIES_PLAN.md 6.5). Why one process: with the GPU backend both brains then share one CUDA context (two
    processes would time-slice the GPU between two), the game process stays light, and a CUDA error cannot take the
    web server down. The protocol is the single brain's, with the fly in each message; one ``advance_all`` message
    per tick drives every brain in lockstep. The server is backend-independent: with CPU brains it is the same as a
    process per brain, which is how the tests check it."""

    def __init__(self):
        self.proc = None
        self.closed = False
        self._lock = threading.Lock()
        self.handles: dict = {}
        self._tick = None                            # the lockstep in flight: who was sent, who was received

    def attach(self, spec: dict, fly: int) -> "GpuBrainHandle":
        """A handle for fly ``fly``, whose brain is built from ``spec`` in the child: the first attach starts the
        process, the next ones add a brain to it."""
        if fly in self.handles:
            raise ValueError(f"fly {fly} already has a brain in this process")
        if self.proc is None:
            warm_up_once()
            self._start(_server_child, ([(fly, spec)],), "fly-brains")
            kind, info = self._recv(fly, self.START_TIMEOUT)
            if kind != "ready":
                self.close()
                raise RuntimeError(f"the brain process of fly {fly} could not start: {info}")
            info = info[0]
        else:
            info = self._request(fly, "add_brain", fly, spec)
        handle = GpuBrainHandle(self, fly, info)
        self.handles[fly] = handle
        return handle

    # ------------------------------------------------------------------ the lockstep (advance_all)
    def send_advance(self, fly: int, args: tuple):
        """Buffer one fly's input for this tick; the first send takes the pipe for the whole tick, and when every
        attached fly has sent (at once with one fly) one ``advance_all`` message goes out."""
        if self._tick is None:
            self._lock.acquire()
            self._tick = {"buffered": {}, "order": [], "ticks": None, "error": None, "outstanding": set(), "sent": False}
        st = self._tick
        st["buffered"][fly] = args
        st["outstanding"].add(fly)
        if len(st["buffered"]) >= len(self.handles):
            try:
                self._flush_tick()
            except Exception:
                st["outstanding"].discard(fly)        # this fly's receive will never be called (advance_all did not record it)
                if not st["outstanding"]:
                    self._tick = None
                    self._lock.release()
                raise

    def _flush_tick(self):
        st = self._tick
        if st["sent"]:
            return
        st["sent"] = True
        st["order"] = sorted(st["buffered"])
        try:
            self._send(st["order"][0], ("advance_all", [(f, *st["buffered"][f]) for f in st["order"]]))
        except Exception as e:
            st["error"] = e
            raise

    def recv_advance(self, fly: int) -> BrainTick:
        """One fly's tick from the ``advance_all`` answer (read once, on the first receive); every later receive of
        this tick gets its tick or the same failure at once, and the last one lets the pipe go."""
        st = self._tick
        if st is None or fly not in st["outstanding"]:
            raise RuntimeError(f"the brain process of fly {fly} was not sent this tick's input")
        if not st["sent"]:                           # a receive before every fly has sent: go with what was sent
            try:
                self._flush_tick()
            except Exception:
                pass                                 # recorded in st["error"]
        if st["ticks"] is None and st["error"] is None:
            try:
                reply = self._reply(fly)
                st["ticks"] = dict(zip(st["order"], reply))
            except Exception as e:
                st["error"] = e
        st["outstanding"].discard(fly)
        try:
            if st["error"] is not None:
                raise st["error"]
            return st["ticks"][fly]
        finally:
            if not st["outstanding"]:
                self._tick = None
                self._lock.release()


class GpuBrainHandle:
    """One fly's end of a :class:`GpuBrainServer`: the same calls as :class:`ProcessBrain`, each a request to the
    shared child with this fly's number in it. ``brain`` is None: the parent has no FlyBrain of its own."""

    parent_histories = True     # the FlyAgent keeps the readout histories from the ticks
    brain = None

    def __init__(self, server: GpuBrainServer, fly: int, info: dict):
        self.server, self.fly = server, fly
        self.fingerprint: str = info["fingerprint"]
        self.has_plasticity: bool = info["has_plasticity"]
        self.parts_counts = info["parts_counts"]

    @property
    def proc(self):
        return self.server.proc

    @property
    def closed(self) -> bool:
        return self.server.closed

    def _request(self, cmd: str, *args):
        return self.server._request(self.fly, cmd, self.fly, *args)

    # ------------------------------------------------------------------ per tick
    def advance(self, dt: float, rates: dict, col_idx, col_hz, seq: int, readouts: dict = None,
                want_learned: bool = False) -> BrainTick:
        return self._request("advance", dt, rates, col_idx, col_hz, seq, want_learned)

    def learned(self, bt: BrainTick) -> float:
        return bt.learned

    def peek(self, dt: float, seq: int, readouts: dict = None) -> BrainTick:
        return self._request("peek", dt, seq)

    def send_advance(self, dt, rates, col_idx, col_hz, seq, readouts=None, want_learned=False):
        self.server.send_advance(self.fly, (dt, rates, col_idx, col_hz, seq, want_learned))

    def recv_advance(self) -> BrainTick:
        return self.server.recv_advance(self.fly)

    # ------------------------------------------------------------------ commands
    def silence(self, spec: str) -> int:
        return self._request("silence", spec)

    def unsilence(self, spec: str | None = None):
        self._request("unsilence", spec)

    def modulate(self, spec: str, factor: float) -> int:
        return self._request("modulate", spec, factor)

    def unmodulate(self, spec: str | None = None):
        self._request("unmodulate", spec)

    def add_monitor(self, key: str, spec: str):
        self._request("add_monitor", key, spec)

    def remove_monitor(self, key: str):
        self._request("remove_monitor", key)

    def history(self, keys=None, n: int = 400):
        raise NotImplementedError("a GpuBrainHandle's histories live on its FlyAgent (agent.py)")

    def reset(self):
        self._request("reset")

    def learning(self, on: bool | None = None, forget: bool = False):
        self._request("learning", on, forget)

    def record(self, op: str):
        return self._request("record", op)

    def settings(self) -> dict:
        return self._request("settings")

    def parts_status(self) -> dict | None:
        return self._request("parts_status")

    def parts_info(self):
        return self._request("parts_info")

    def learning_summary(self) -> dict | None:
        return self._request("learning_summary")

    def depressed_fraction(self):
        return self._request("depressed_fraction")

    def status_fields(self) -> dict:
        return self._request("status_fields")

    def neuron(self, i: int):
        return self._request("neuron", i)

    def swap(self, wiring: dict | None, parts_arg, silenced, modulated: dict, learning_on: bool) -> dict:
        """Rebuild this fly's brain in the shared child and re-apply what the player had done to the old one."""
        info = self._request("swap", wiring, parts_arg, sorted(silenced), dict(modulated), bool(learning_on))
        self.fingerprint, self.has_plasticity, self.parts_counts = info["fingerprint"], info["has_plasticity"], info["parts_counts"]
        return info

    def close(self):
        """Close the shared process (every handle's close reaches the same server; safe to call twice)."""
        self.server.close()


def advance_all(brains, inputs) -> list:
    """Send every brain its tick's input, then wait for every answer: the lockstep barrier of the two-flies game.
    ``inputs`` holds one (dt, rates, col_idx, col_hz, seq, readouts, want_learned) per brain.

    If one brain fails (its child died, or a send or receive raised), the others are still received: every brain
    that was sent its input gets its answer read (or its failure raised and swallowed), so no lock stays held and no
    answer stays in a pipe to be taken for the next tick's. A local brain among them steps too, so every brain that
    was sent has advanced by the same tick. The first failure is then raised."""
    sent, ticks = [], []
    try:
        for io, args in zip(brains, inputs):
            io.send_advance(*args)
            sent.append(io)
        while sent:
            ticks.append(sent.pop(0).recv_advance())
        return ticks
    finally:
        for io in sent:                           # sent but not yet received: let their answers and locks go
            try:
                io.recv_advance()
            except Exception:                     # the failure that stopped the barrier, or a second dead child
                pass
