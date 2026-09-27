"""
One fly in the dish: its brain handle, senses, decoder, body and bookkeeping (:class:`FlyAgent`).

The game (game.py) holds what every fly shares: the dish (the world), the clock, the message line, the action queue,
the event log, the scenarios and the recording. Everything that belongs to one fly lives here, so that a second
simulated fly is another :class:`FlyAgent` in the same dish, sensing the first one only through the world (the
two-flies work, docs/TWO_FLIES_PLAN.md). With one fly the game runs exactly as before: the code below was moved out
of ``Game`` unchanged, and the tick order (senses, brain, decoder, body, bookkeeping, world, events, watchdog) is the
one the single fly always had. The senses, the decoder, the walking urge and the internal state are hand-built and
labelled as such (:meth:`FlyAgent.whats_real`); only the wiring in between is the connectome's.
"""

from __future__ import annotations

import json
import math
import os
import threading
import time
from dataclasses import dataclass

import numpy as np

from .body import FlyBody, WALK_SPEED
from .brainio import BrainTick, LocalBrain, ProcessBrain
from . import genetics
from . import parts as partslib
from . import retest as retestlib
from . import vfb
from . import wiring
from .experiments import survival as survival_report
from .game import (CHECKS, COURTSHIP_HZ, COURTSHIP_SECS, COURTSHIP_SPEC, GF_BURST, HIDDEN_READOUTS,
                   PHEROMONE_GRNS, READOUTS, REWARD_HZ, REWARD_SPEC, SHOCK_HZ, SHOCK_SPEC, SOUND_HZ, SOUND_SPEC,
                   STILL_TICKS, TICK_MS, ZAP_PRESETS, InternalState, MotorDecoder)
from .scenarios import SCENARIOS
from .senses.mechano import Antennae, Bristles
from .senses.olfaction import ODOURS, Nose
from .senses.taste import WATER_GRNS, Forelegs, Mouth
from .senses.vision import Retina
from .world import ARENA_R, FLY_HALF, wrap


def _empty_tick(readouts: dict) -> BrainTick:
    """What a fly's brain has said before its first tick: nothing."""
    zero = {k: 0.0 for k in readouts}
    return BrainTick(zero, dict(zero), 0, 0, 0, [], 0.0, (0, 0.0), None, 0, [], None)


class FlyAgent:
    """One fly: everything the game keeps per fly. ``game`` holds the shared world and clock (read here through
    :attr:`world`, :attr:`t`, :attr:`events`, :attr:`actions`, :attr:`scenario`, :meth:`say`).

    The brain is reached only through :attr:`io` (brainio.py): a :class:`~virtual_fly.brainio.LocalBrain` in this
    process, or a :class:`~virtual_fly.brainio.ProcessBrain` in a child process of its own (``brain_procs``), with
    the same numbers either way. :attr:`brain` is the FlyBrain itself when it is local, else None."""

    def __init__(self, game, id: int, brain, *, sex: str, rng, seed: int, autopilot: bool = True, columnar: bool = True,
                 body: str = "drawn", stride_average: bool = False, parts_list=None, brain_factory=None,
                 brain_kwargs: dict | None = None, retest: str = "auto", brain_procs: bool = False):
        self.game, self.id, self.sex, self.rng, self.seed = game, id, sex, rng, seed
        self.conn = brain.conn
        # the genome: the real wiring, and flies grown from its rules (see wiring.py)
        self.real_conn = brain.conn
        self.brain_factory = brain_factory or self._default_brain_factory
        # how a re-test process rebuilds the brain: build_brain(conn, profile, **brain_kwargs, parts=...). Known
        # for the default factory; a caller with its own factory says so (play.py), or re-tests run in a thread.
        self._brain_kwargs = brain_kwargs if brain_kwargs is not None else (None if brain_factory else {})
        if retest not in ("auto", "process", "thread"):
            raise ValueError("retest must be 'auto', 'process' or 'thread'")
        self.retest_mode = retest
        self._retest_handle = None                      # the running re-test process, if any
        self._survival_cache: dict[tuple, list] = {}     # re-test results: the same fly and settings give the same rows
        self._retest_lock = threading.Lock()
        self._rules_cache: dict = {}
        self._survival_token = None
        self.genome: dict = {"level": "real", "seed": 0, "growing": None, "survival": None, "wiring": None, "rules": None, "error": None}
        self.parts_on = brain.parts is not None          # the genes as each neuron's parts list (parts.py)
        # the PartsList to (re)build with: the running brain's, else the one the caller chose (curated policy ...)
        self._parts_list = brain.parts.parts if brain.parts is not None else parts_list
        self._parts_counts: dict | None = None
        self._learning_on = brain.plasticity is not None
        # the brain: here, or in a child process built from this brain's settings (then this one is let go)
        if brain_procs:
            self.io = ProcessBrain(self.child_spec(brain), fly=id)
            del brain
        else:
            self.io = LocalBrain(brain)
        self.histories: dict[str, list] = {}             # a process brain's readout histories, one value per tick
        self.autopilot = autopilot
        if body == "physics":                          # optional: NeuroMechFly v2 in MuJoCo (virtual_fly/physics.py)
            from .physics import make_body
            self.body = make_body("physics", self.world, self.rng, seed=seed, stride_average=stride_average)
        else:
            self.body = FlyBody(self.world, self.rng)
        self.body_kind = body
        self.retina = Retina(self.world, self.conn, columnar=columnar)
        self.columnar_on = self.retina.columnar is not None
        self.nose = Nose(self.world)
        self.mouth = Mouth(self.world)
        # in the female file LB3a is an alias for the published model's 18 water cells (FlyWire's types do not split
        # sugar from water)
        self.water_cells = int(sum(self.conn.select(s).size for s in WATER_GRNS))
        self.forelegs = Forelegs(self.world, PHEROMONE_GRNS)
        self.antennae = Antennae(self.world)
        self.bristles = Bristles(self.world)
        self.decoder = MotorDecoder(self.conn)
        self.state = InternalState()
        self.readouts = {r[0]: self.conn.select(r[1]) for r in READOUTS}
        self.readouts.update({k: self.conn.select(v) for k, v in HIDDEN_READOUTS.items()})
        c = self.conn
        self.readout_meta = [{"key": r[0], "spec": r[1], "label": r[2], "group": r[3], "max": r[4], "colour": r[5],
                              "genes": genetics.genotype(c, c.select(r[1]))["tags"]}
                             for r in READOUTS if self.readouts[r[0]].size]      # (the female fly has no pIP10, no TTMn)
        self.genetics = genetics.summary(c, self.readout_meta)
        self.neuronbridge = genetics.NeuronBridge()
        self.custom_readouts: dict[str, str] = {}
        for r in READOUTS:
            if self.readouts[r[0]].size:                 # not for cells this fly lacks (/api/history leaves them out)
                self.add_monitor(r[0], r[1])
        self.user_silenced: set[str] = set()
        self.user_modulated: dict[str, float] = {}
        self.has_soma = ~np.isnan(self.conn.soma[:, 0])
        # build the input index and the cell-type graph now (a second or two), so that the first
        # pathway trace or neuron lookup from the browser does not stall behind the game loop
        self.conn.col_ptr
        self.conn.type_graph()
        self.done: set[str] = set()                      # the checklist: kept when the player asks for a new fly
        # what one tick leaves for the next steps of the same tick (see Game.tick)
        self.bt: BrainTick = _empty_tick(self.readouts)  # what the brain said this tick
        self.m: dict = dict(self.decoder.m)
        self._why: list = []
        self.courting = 0.0
        self.rates_now: dict = {}
        self.sps = 0.0

    # ------------------------------------------------------------------ the brain, through its seam
    @property
    def brain(self):
        """The FlyBrain when it lives in this process (LocalBrain), else None."""
        return self.io.brain

    @brain.setter
    def brain(self, b):
        if not isinstance(self.io, LocalBrain):
            raise AttributeError("this fly's brain lives in its own process; it cannot be replaced from here")
        self.io.brain = b

    @property
    def learning_on(self) -> bool:
        return self._learning_on

    @learning_on.setter
    def learning_on(self, on: bool):
        self._learning_on = bool(on)
        self.io.learning(on=self._learning_on)

    def child_spec(self, brain) -> dict:
        """What a brain process needs to build this brain (as _retest_spec says it for the re-test)."""
        path = getattr(self.real_conn, "path", None)
        if path is None or not os.path.exists(path) or self.real_conn.meta.get("rewired"):
            raise ValueError("a brain in its own process is built from the connectome's file, and this wiring is not "
                             "the file's (" + ("grown or rewired: " + str(self.real_conn.meta.get("rewired")) if path
                                               else "no file") + "); use brain_procs='off'")
        if self._brain_kwargs is None:
            raise ValueError("a brain in its own process needs the brain's settings (a custom brain_factory without "
                             "brain_kwargs); use brain_procs='off'")
        kw = {k: v for k, v in brain.settings().items() if k in self._BRAIN_KWARGS}
        kw.update(self._brain_kwargs)
        kw["parts"] = self.parts_arg(self.parts_on)
        return {"path": str(path), "wiring": None, "profile": self.profile_name, "brain_kwargs": kw,
                "readouts": [(r[0], r[1]) for r in READOUTS] + list(HIDDEN_READOUTS.items()),
                "vfb": vfb.injected(), "regions": partslib.injected_region_table()}   # data swapped in at runtime

    def add_monitor(self, key: str, spec: str):
        self.io.add_monitor(key, spec)
        if self.io.parent_histories:
            self.histories[key] = []

    def remove_monitor(self, key: str):
        self.io.remove_monitor(key)
        self.histories.pop(key, None)

    def history(self, keys=None, n: int = 400):
        """(bin_ms, {key: the last n values}): the brain's monitors, or this fly's own histories for a process brain."""
        if not self.io.parent_histories:
            return self.io.history(keys, n)
        out = {k: self.histories[k][-n:] for k in (keys or list(self.histories)) if k in self.histories}
        return (TICK_MS if self.histories else False), out

    def reset_brain(self):
        """Every neuron back at rest (a new fly, a calm, the watchdog); the histories start again, as the monitors do."""
        self.io.reset()
        for h in self.histories.values():
            h.clear()

    def depressed_fraction(self) -> float:
        """The share of plastic synapses that have been depressed (0 without plasticity), for the scenarios' measures:
        read live, as the single fly always read it, and only that number (a scenario measures every tick)."""
        return self.io.depressed_fraction() if self.io.has_plasticity else 0

    # ------------------------------------------------------------------ the shared world, read as the code always did
    @property
    def world(self):
        return self.game.world

    @property
    def t(self) -> float:
        return self.game.t

    @property
    def events(self):
        return self.game.events

    @property
    def actions(self):
        return self.game.actions

    @property
    def scenario(self):
        return self.game.scenario

    @property
    def profile_name(self) -> str:
        return self.game.profile_name

    @property
    def lock(self):
        return self.game.lock

    def say(self, text: str, secs: float = 3.0):
        self.game.say(text, secs)

    # ------------------------------------------------------------------ a new fly (the per-fly part of Game.reset_world)
    def reset_state(self):
        """Everything this fly forgets when the player asks for a new fly (its checklist is kept)."""
        self.body.reset()
        # the senses forget what they were in the middle of
        self.antennae.dust_left = 0.0
        self.antennae.hearing = 0.0
        self.bristles.touch_left = 0.0
        self.bristles.last_touch_t = -99.0
        self.nose.background.clear()
        self.retina.features = type(self.retina.features)(self.retina.eyes)
        self._prev_felt = {}
        self.zaps = []                    # [spec, hz, seconds left]
        self.mode = "idle"
        self.wander = dict(walking=True, left=2.0, yaw=0.0)
        self.runaway_s = 0.0
        self.graded_eps = 0.0                         # the graded cells' release quanta per second (parts list)
        self.since_input = 0.0
        self.calms = 0
        self.hz_shown = {k: 0.0 for k in self.readouts}
        self.senses_now: dict = {}
        self.driver = ""
        self.gf_cooldown = 0.0
        self.gf_prev = 0                   # live giant-fibre spikes in the last tick (a jump needs a burst)
        self.still_ticks = STILL_TICKS     # ticks the body has not moved (a new fly starts at rest)
        self.turn_command = 0.0            # last tick's commanded yaw (efference copy for the eyes)
        self.court_left = 0.0              # seconds of pC1 drive left after tapping the female
        self.sound_left = 0.0
        self.bitter_t = 0.0
        self.sugar_t = 0.0                 # how long sugar has been at the mouth
        self.eating = 0.0
        self.drinking = 0.0
        self.song_side = 1.0
        self.shock_left = 0.0
        self.learned_bias = 0.0
        self.smelling: str | None = None
        self.state = InternalState()
        self.decoder.m = {k: 0.0 for k in self.decoder.m}

    # ------------------------------------------------------------------ the genome: growing a fly
    _BRAIN_KWARGS = ("dt", "gain", "kenyon_gain", "fatigue_mv", "fatigue_ms", "std_u", "std_tau_ms",
                     "noise_hz", "noise_mv", "noise_spec", "threshold_jitter", "seed", "backend")

    def _default_brain_factory(self, conn, **extra):
        from .settings import build_brain
        kw = {k: v for k, v in self.io.settings().items() if k in self._BRAIN_KWARGS}
        kw.update(extra)
        return build_brain(conn, self.profile_name, **kw)

    def parts_arg(self, on: bool):
        """The ``parts=`` argument for a rebuild: the PartsList this game was started with (its curated
        policy and receptor setting), or the default one."""
        return (self._parts_list or True) if on else False

    def parts_list(self) -> "partslib.PartsList":
        """The parts list this game switches on (the default one unless it was started with another)."""
        return self._parts_list or partslib.PartsList()

    def parts_counts(self) -> dict:
        """What the parts list finds in this connectome (compiled once; the same whether it is switched on)."""
        info = self.io.parts_info()
        if info is not None:
            return info[1]
        if self._parts_counts is None:
            self._parts_counts = self.parts_list().compile(self.real_conn).counts
        return self._parts_counts

    def _start_grow(self, level: str, seed: int):
        self.genome.update(growing={"level": level, "seed": seed, "t0": time.time()}, error=None)
        self.events.add(self.t, "genome", f"growing a fly: {level} wiring, seed {seed}")
        self.say(f"Growing a fly from its {level} wiring rules…", 4.0)
        threading.Thread(target=self._grow_worker, args=(level, seed), daemon=True).start()

    def _grow_worker(self, level: str, seed: int):
        try:
            conn2, rules = wiring.grow_level(self.real_conn, level, seed, rules_cache=self._rules_cache)
            brain2 = self._build_or_ship(conn2, self.parts_arg(self.parts_on))
            cmp = wiring.compare(self.real_conn, conn2) if conn2 is not self.real_conn else None
            self.actions.put({"type": "_swap_brain", "brain": brain2, "conn": conn2, "level": level, "seed": seed,
                              "rules": rules.summary() if rules is not None else None, "wiring": cmp,
                              "parts": self.parts_on, "reason": "grow"})
        except Exception as e:                       # a bad level, or out of memory: report, keep the old fly
            self.genome.update(growing=None, error=str(e))
            print("error growing a fly", repr(e))

    def _start_rebuild(self, on: bool):
        """Rebuild the current fly's brain with the parts list on or off (same wiring), in the background."""
        self.genome.update(growing={"level": self.genome["level"], "seed": self.genome["seed"], "reason": "parts",
                                    "parts": on, "t0": time.time()}, error=None)
        self.events.add(self.t, "genome", f"rebuilding the brain with the parts list {'on' if on else 'off'}")
        self.say("Giving each neuron its parts…" if on else "Back to identical neurons…", 4.0)
        threading.Thread(target=self._rebuild_worker, args=(on,), daemon=True).start()

    def _rebuild_worker(self, on: bool):
        try:
            brain2 = self._build_or_ship(self.conn, self.parts_arg(on))
            self.actions.put({"type": "_swap_brain", "brain": brain2, "conn": self.conn, "level": self.genome["level"],
                              "seed": self.genome["seed"], "rules": self.genome["rules"], "wiring": self.genome["wiring"],
                              "parts": on, "reason": "parts"})
        except Exception as e:
            self.genome.update(growing=None, error=str(e))
            print("error rebuilding the brain", repr(e))

    def _build_or_ship(self, conn, parts_arg):
        """A local brain is built here (in the worker thread, as it always was); a process brain is rebuilt in its
        child, which needs only the wiring (for a grown fly) and the parts setting: what _swap_brain ships."""
        if isinstance(self.io, LocalBrain):
            return self.brain_factory(conn, parts=parts_arg)
        wiring_ = None
        if conn is not self.real_conn:                   # a grown fly: send its wiring, the neurons are the same
            wiring_ = {"row_ptr": conn.row_ptr, "post_idx": conn.post_idx, "n_syn": conn.n_syn, "label": "grown"}
        return {"wiring": wiring_, "parts": parts_arg}

    def _swap_brain(self, a: dict):
        if isinstance(self.io, LocalBrain):
            old = self.brain
            with old.lock:
                self.brain, self.conn = a["brain"], a["conn"]
                for r in READOUTS:
                    if self.readouts[r[0]].size:
                        self.brain.add_monitor(r[0], r[1], bin_ms=TICK_MS)
                for k, spec in self.custom_readouts.items():
                    self.brain.add_monitor(k, spec, bin_ms=TICK_MS)
                for spec in self.user_silenced:
                    try:
                        self.brain.silence(spec)
                    except ValueError:
                        pass
                for spec, factor in self.user_modulated.items():
                    try:
                        self.brain.modulate(spec, factor)
                    except ValueError:
                        pass
                if self.brain.plasticity is not None:
                    self.brain.plasticity.enabled = self.learning_on
                self.zaps = []
                self.runaway_s = 0.0
        else:                                            # the child rebuilds and re-applies the lab's changes itself
            ship = a["brain"]
            try:
                self.io.swap(ship["wiring"], ship["parts"], self.user_silenced, self.user_modulated, self.learning_on)
            except Exception as e:                       # a bad level, or out of memory, in the child: keep the old fly
                self.genome.update(growing=None, error=str(e))
                self.events.add(self.t, "genome", f"the brain could not be rebuilt: {e}")
                self.say(f"Error: {e}", 4.0)
                print("error rebuilding the brain in its process", repr(e))
                return
            self.conn = a["conn"]
            for h in self.histories.values():
                h.clear()
            self.zaps = []
            self.runaway_s = 0.0
        level, seed = a["level"], a["seed"]
        self.parts_on = bool(a.get("parts", self.parts_on))
        self.genome.update(level=level, seed=seed, growing=None, wiring=a["wiring"], rules=a["rules"], error=None,
                           survival={"running": True, "results": []})
        if a.get("reason") == "parts":
            if self.parts_on:
                c = self.parts_counts()
                self.events.add(self.t, "genome", f"parts list on: {c['modulatory_neurons']:,} modulatory neurons act through slow "
                                f"tones on {c['modulated_targets']:,} targets, {c['graded_neurons']:,} cells transmit graded signals")
                self.say("Each neuron now has its parts. Testing the reflexes…", 4.0)
                self.done.add("parts")
            else:
                self.events.add(self.t, "genome", "parts list off: every neuron is the same machine again")
                self.say("Every neuron is the same machine again. Testing the reflexes…", 4.0)
        elif level == "real":
            self.events.add(self.t, "genome", "back to the real wiring")
            self.say("The real wiring is back.", 3.0)
        else:
            w = a["wiring"] or {}
            self.events.add(self.t, "genome", f"a fly grown from its {level} wiring rules (seed {seed}): "
                            f"{w.get('edges_grown', 0):,} connections, {100 * w.get('shared_connections_fraction', 0):.0f}% shared with the real wiring")
            self.say(f"A new fly, grown from its {level} wiring rules. Testing its reflexes…", 4.0)
            self.done.add("genome")
        token = object()
        self._survival_token = token
        threading.Thread(target=self._survival_worker, args=(a["conn"], token), daemon=True).start()

    def _retest_spec(self, conn) -> dict | None:
        """What a re-test process needs to rebuild this brain, or None when it cannot (then: a thread)."""
        if self.retest_mode == "thread" or self._brain_kwargs is None:
            return None
        path = getattr(self.real_conn, "path", None)
        if path is None or not os.path.exists(path):
            return None
        kw = {k: v for k, v in self.io.settings().items() if k in self._BRAIN_KWARGS}
        kw.update(self._brain_kwargs)
        kw["parts"] = self.parts_arg(self.parts_on)
        wiring_ = None
        if conn is not self.real_conn:                   # a grown fly: send its wiring, the neurons are the same
            wiring_ = {"row_ptr": conn.row_ptr, "post_idx": conn.post_idx, "n_syn": conn.n_syn, "label": "grown"}
        return {"path": str(path), "wiring": wiring_, "profile": self.profile_name, "brain_kwargs": kw,
                "vfb": vfb.injected(), "regions": partslib.injected_region_table()}   # data swapped in at runtime

    def _survival_key(self, conn) -> tuple:
        """Everything a survival report depends on. Each seed resets the brain and starts its own random
        generator, so the same wiring, profile, brain settings and parts list always give the same rows."""
        kw = {k: v for k, v in self.io.settings().items() if k in self._BRAIN_KWARGS}
        kw.update(self._brain_kwargs or {})
        return (conn.dataset, int(conn.n_edges), self.profile_name, repr(sorted(kw.items())),
                repr(self.parts_arg(self.parts_on)))

    def _survival_worker(self, conn, token):
        """Run the validated experiments on a private copy of the new brain while the game keeps going: in a
        low-priority child process when possible (retest.py), else in this thread. A newer re-test replaces
        an older one (the older process is terminated, an older thread stops at its next experiment)."""
        with self._retest_lock:
            prev, self._retest_handle = self._retest_handle, None
        if prev is not None:
            prev.cancel()
        t0 = time.time()

        def progress(rows):
            if self._survival_token is token:
                self.genome["survival"] = {"running": True, "results": rows}
        try:
            key = self._survival_key(conn)
            cached = self._survival_cache.get(key)
            spec = None if cached is not None else self._retest_spec(conn)
            handle = None
            if cached is not None:                       # this fly was tested with these settings before
                rows, where = [dict(r) for r in cached], "cache"
            elif spec is not None:
                try:
                    handle = retestlib.Retest(spec)
                except Exception as e:                   # no child processes here: re-test in this thread
                    print("re-testing in a thread:", repr(e))
            if handle is not None:
                with self._retest_lock:
                    if self._survival_token is token:
                        self._retest_handle = handle
                    else:                                # superseded while the process was starting
                        handle.cancel()
                try:
                    rows, where = handle.wait(progress), "process"
                except RuntimeError as e:
                    if handle.cancelled or self._survival_token is not token:
                        raise
                    print("the re-test process failed, re-testing in a thread:", e)   # e.g. it could not import
                    failed = handle
                else:
                    failed = None
                finally:
                    with self._retest_lock:                  # done: let go of its queue (no leaked semaphores at exit)
                        if self._retest_handle is handle:
                            self._retest_handle = None
                if failed is not None:
                    handle = None
            if cached is None and handle is None:
                brain = self.brain_factory(conn, parts=self.parts_arg(self.parts_on))
                rows = survival_report(brain, profile=self.profile_name, on_progress=progress,
                                       should_stop=lambda: self._survival_token is not token)
                where = "thread"
            if self._survival_token is token:
                if cached is None:
                    self._survival_cache[key] = [dict(r) for r in rows]
                ok = sum(1 for r in rows if r["ok"]); tested = sum(1 for r in rows if r["ok"] is not None)
                self.genome["survival"] = {"running": False, "results": rows, "ok": ok, "tested": tested,
                                           "secs": round(time.time() - t0, 1), "where": where}
                self.events.add(self.t, "genome", f"reflex survival: {ok} of {tested} experiments pass on this wiring")
        except Exception as e:
            if self._survival_token is token:
                self.genome["survival"] = {"running": False, "results": [], "error": str(e)}

    def genome_status(self, parts_status: dict | None = None, known: bool = False) -> dict:
        """``known``: ``parts_status`` is this tick's (publish passes the BrainTick's), so the brain is not asked again."""
        g = dict(self.genome)
        if g["growing"]:
            g["growing"] = {**g["growing"], "secs": round(time.time() - g["growing"]["t0"], 1)}
        g["parts"] = {"on": self.parts_on, "status": parts_status if known else self.io.parts_status()}
        return g

    # ------------------------------------------------------------------ senses (hand-built encoders)
    def senses(self, dt: float):
        """Returns (spec -> Hz, per-neuron indices, per-neuron Hz) and fills ``self.senses_now``."""
        pose = self.body.pose
        rates: dict[str, float] = {}
        felt: dict = {}

        def add(r: dict[str, float]):
            for spec, hz in r.items():
                rates[spec] = max(rates.get(spec, 0.0), hz)

        # eyes: render the retina, derive feature-detector rates (+ columnar T4/T5 when enabled)
        vis, col_idx, col_hz = self.retina.look(pose, dt, turn_command=self.turn_command)
        add(vis)
        f = self.retina.features.felt
        if "loom" in f:
            felt["loom"] = f["loom"]
        if "small" in f:
            felt["small"] = f["small"]
        if "flow" in f:
            felt["flow"] = f["flow"]
        # taste at the mouth and the forelegs
        add(self.mouth.rates(pose, self.state.hunger, self.state.thirst, pose.proboscis))
        for kind in self.mouth.touching:
            felt["taste_" + kind] = True
        add(self.forelegs.rates(pose))
        if self.forelegs.touching_female:
            felt["pheromone"] = True
            self.court_left = COURTSHIP_SECS
        if self.court_left > 0:                      # hand-built: contact with the female arouses pC1
            rates[COURTSHIP_SPEC] = max(rates.get(COURTSHIP_SPEC, 0.0), COURTSHIP_HZ * min(1.0, self.court_left / 1.0))
            felt["courting"] = True
            self.court_left -= dt
        if self.sound_left > 0:
            rates[SOUND_SPEC] = max(rates.get(SOUND_SPEC, 0.0), SOUND_HZ)
            felt["sound"] = True
            self.sound_left -= dt
        # smell
        add(self.nose.rates(pose, dt))
        od, c = self.nose.strongest()
        self.smelling = od if c > 0.02 else None
        if self.smelling:
            felt["smell"] = self.smelling
            self.done.add("smell")
        # wind, sound, dust
        if self.world.female is not None:
            self.antennae.hearing = 0.0
        add(self.antennae.rates(pose, dt))
        if self.antennae.airspeed_l + self.antennae.airspeed_r > 0.05:
            felt["wind"] = round(math.degrees(self.antennae.wind_bearing))
        if self.antennae.dust_left > 0:
            felt["dust"] = True
        # touch and proprioception
        add(self.bristles.rates(pose, dt, self.body.bumped, self.t))
        if self.bristles.touch_left > 0:
            felt["touch"] = True
        # neuromodulatory signals the wiring does not carry (hand-built, labelled)
        if self.eating > 0.2 and "sugar" in self.mouth.touching and self.learning_on:
            rates[REWARD_SPEC] = max(rates.get(REWARD_SPEC, 0.0), REWARD_HZ * min(1.0, self.eating))
            felt["reward"] = True
        if self.shock_left > 0:
            rates[SHOCK_SPEC] = max(rates.get(SHOCK_SPEC, 0.0), SHOCK_HZ)
            felt["shock"] = True
            self.shock_left -= dt
        # zaps from the neuron panel
        for z in self.zaps:
            rates[z[0]] = max(rates.get(z[0], 0.0), z[1])
            z[2] -= dt
        self.zaps = [z for z in self.zaps if z[2] > 0]
        if self.zaps:
            felt["zap"] = ", ".join(f"{z[0]} {z[1]:g} Hz" for z in self.zaps)
        self.senses_now = felt
        return rates, col_idx, col_hz

    def sense(self, dt: float):
        """The same as :meth:`senses` (the name the plan uses for one fly's sensing step)."""
        return self.senses(dt)

    # ------------------------------------------------------------------ behaviour selection (hand-built)
    def choose_mode(self, m: dict, gf_spikes: int) -> str:
        burst, self.gf_prev = gf_spikes + self.gf_prev, gf_spikes
        if self.body.pose.jump is not None:
            return "escape"
        # the giant fibre must fire a burst: >= GF_BURST live spikes over this tick and the last (50 Hz per cell over
        # 50 ms). Walking alone (leg proprioception) makes the pair fire together now and then, at most 4 spikes over
        # two ticks in 30 min of it; a clap gives 5-9, a fast looming hand 14-30, a zap as many as its rate
        # (docs/SCIENCE.md 5.7)
        if burst >= GF_BURST and self.body.jump_lock <= 0 and self.gf_cooldown <= 0:
            away = self.world.hand if (self.world.hand is not None and self.world.tool == "hand") else None
            self.body.start_jump(away)
            self.gf_cooldown = 1.0
            if "loom" in self.senses_now:
                self.done.add("escape")
            if self.body_kind == "physics":                # NeuroMechFly has no jump model
                self.say("Giant fibre fired: the escape command (the physics body cannot jump)", 1.5)
                self.events.add(self.t, "behaviour", "escape command (giant fibre DNp01 burst; the physics body has no jump)")
            else:
                self.say("Giant fibre fired: escape jump!", 1.5)
                self.events.add(self.t, "behaviour", "escape jump (giant fibre DNp01 burst)")
            return "escape"
        # strongest command wins (a hand-built stand-in for the nerve cord's own arbitration).
        # A behaviour starts above its threshold and continues until its drive falls to half of it.
        options = [(0.0, "")]
        for name, start, weight in (("backward", 0.25, 1.15), ("feed", 0.3, 1.0), ("groom", 0.35, 0.9)):
            ongoing = self.mode == name
            if m[name] > (start / 2 if ongoing else start):
                options.append((weight * m[name] + (0.15 if ongoing else 0.0), name))
        mode = max(options)[1] or "walk"
        if mode == "walk" and (m["song"] > 0.3 or m["court"] > 0.4):
            mode = "court"
        return mode

    def odour_steering(self) -> tuple[float, str]:
        """Hand-built odour-guided steering: innate valence (a literature label) plus the *learned*
        bias read off the KC->MBON synapses of the Kenyon cells active right now.

        Returns (yaw bias, explanation). + = turn right."""
        if not self.smelling:
            self.learned_bias = 0.0
            return 0.0, ""
        odour = ODOURS[self.smelling]
        innate = {"attractive": 0.35, "aversive": -0.35}.get(odour.innate, 0.0)
        learned = self.io.learned(self.bt)          # 1.2 * (wv - wa) off the KC->MBON synapses (brainio._learned)
        self.learned_bias = learned
        valence = max(-1.0, min(1.0, innate + learned))
        if abs(valence) < 0.05:
            return 0.0, ""
        grad = self.nose.gradient().get(self.smelling, 0.0)     # + = stronger on the left antenna
        c = self.nose.felt.get(self.smelling, 0.0)
        toward = -math.copysign(1.0, grad) if abs(grad) > 0.02 * max(c, 0.05) else 0.0   # -1 = left
        # no gradient: head upwind (attractive) or downwind (aversive), like a real fly's surge
        if toward == 0.0 and self.antennae.airspeed_l + self.antennae.airspeed_r > 0.05:
            b = self.antennae.wind_bearing                        # where the wind comes from, + = left
            toward = -math.copysign(1.0, b) if abs(b) > 0.15 else 0.0
        yaw = valence * toward * 0.6
        why = (f"odour '{odour.name}': innate {innate:+.2f}" + (f", learned {learned:+.2f}" if abs(learned) > 0.02 else "")
               + " (hand-built steering)")
        return yaw, why

    # ------------------------------------------------------------------ one tick, in steps (Game.tick calls them in this order)
    def advance_input(self, dt: float, rates: dict, col_idx, col_hz, seq: int) -> tuple:
        """What the brain's seam is sent for this tick (Game.tick hands it to advance_all with the other flies')."""
        self.rates_now = rates                       # the watchdog's "since input" reads whether anything drove the brain
        return (dt, rates, col_idx, col_hz, seq, self.readouts, bool(self.smelling))   # smelling: odour steering will want the learned bias

    def advance_done(self, bt: BrainTick) -> BrainTick:
        """Keep this tick's answer (the later steps read it) and, for a process brain, the readout histories."""
        self.bt = bt
        if self.io.parent_histories:
            for k, h in self.histories.items():
                h.append(round(bt.hz.get(k, 0.0), 2))
                if len(h) > 100000:
                    del h[:50000]
        return bt

    def advance(self, dt: float, rates: dict, col_idx, col_hz, seq: int = 0) -> BrainTick:
        """Drive the brain with this tick's rates and step it for TICK_MS; the readouts it leaves."""
        return self.advance_done(self.io.advance(*self.advance_input(dt, rates, col_idx, col_hz, seq)))

    def act(self, dt: float, bt: BrainTick):
        """Decode the descending neurons, choose a mode, add the hand-built walking urge and steering, move the body,
        then the resting rule (a walk still for STILL_TICKS reads idle)."""
        motor_hz, gf = bt.motor_hz, bt.gf
        m = self.decoder.decode(motor_hz, dt)
        self.gf_cooldown -= dt
        mode = self.choose_mode(m, gf)
        why = []
        wander_yaw = 0.0
        if mode == "escape":
            why.append("brain: giant fibre DNp01 burst")
        elif mode == "backward":
            why.append("brain: MDN (moonwalker) neurons")
        elif mode == "feed":
            why.append("brain: MN9 proboscis motor neuron")
        elif mode == "groom":
            why.append("brain: aDN1/aDN2 grooming neurons")
        elif mode == "court":
            why.append("brain: pC1 → pIP10 courtship neurons" if m["song"] > 0.3 else "brain: pC1 courtship neurons")
        if mode in ("walk", "court"):
            if abs(m["yaw"]) > 0.08:
                parts = []
                if abs(motor_hz["DNa02R"] - motor_hz["DNa02L"]) > 8 or abs(motor_hz["DNg13R"] - motor_hz["DNg13L"]) > 15:
                    parts.append("DNa02/DNg13")
                if abs(motor_hz["DNp15R"] - motor_hz["DNp15L"]) > 15:
                    parts.append("optomotor T4/T5 → HS → DNp15")
                why.append("brain: steering via " + (", ".join(parts) if parts else "DNa01/DNa03"))
            if m["forward"] > 0.1:
                why.append("brain: walking DNs (DNp09, BDN1/2/4, oDN1)")
            drive_forward = m["forward"]
            if self.autopilot:                       # hand-built urge to walk: flies do this on their own,
                w = self.wander                      # but this model has no spontaneous activity
                w["left"] -= dt
                if w["left"] <= 0:
                    w["walking"] = not w["walking"] or self.rng.random() < 0.3 + 0.3 * self.state.hunger
                    w["left"] = self.rng.uniform(1.5, 4.5) if w["walking"] else self.rng.uniform(0.4, 1.5)
                w["yaw"] += -w["yaw"] * dt / 0.8 + 0.9 * math.sqrt(dt) * self.rng.gauss(0, 1)
                if w["walking"]:
                    drive_forward = max(drive_forward, 0.5 + 0.2 * self.state.hunger)
                    wander_yaw = max(-0.35, min(0.35, w["yaw"])) * (1 - min(1.0, abs(m["yaw"]) * 3))
                    why.append("autopilot: walking urge (hand-built)")
                if (self.t - self.bristles.last_touch_t < 1.5 and m["backward"] < 0.1
                        and self.t - self.bristles.last_touch_t > 0.6):
                    wander_yaw = 0.8                 # hand-built fallback if it stays stuck at the wall
                    why.append("autopilot: turning away from the wall")
            od_yaw, od_why = self.odour_steering()   # always computed (the learned bias is shown on screen)
            if od_yaw and self.autopilot:
                wander_yaw += od_yaw
                why.append(od_why)
            m = {**m, "forward": drive_forward}
        if mode not in ("walk", "court"):
            self.odour_steering()                    # keeps the learned-bias readout live while feeding etc.
        drive = {**m, "song_side": self.song_side}
        if self.world.female is not None:
            fx, fy = self.world.female.x, self.world.female.y
            bearing = wrap(math.atan2(fy - self.body.pose.y, fx - self.body.pose.x) - self.body.pose.h)
            self.song_side = 1.0 if bearing < 0 else -1.0
            d = math.hypot(fx - self.body.pose.x, fy - self.body.pose.y)
            drive["abdomen"] = 1.0 if (m["court"] > 0.5 and d < 6.0) else 0.0
        self.body.move(dt, mode, drive, wander_yaw)
        self.turn_command = wander_yaw if mode in ("walk", "court") else 0.0   # voluntary part only
        # nothing moves the legs: say "resting", not "walking". The drawn body is judged by how it moves; the physics body
        # by what it is told to do (its stepping drive, as a speed against the same 0.05 mm/s), since MuJoCo's thorax
        # jitters faster than that while the fly stands. Only once it has stayed still for STILL_TICKS, so that a
        # one-tick lull does not blink 'resting'
        if self.body_kind == "physics":
            still = max(abs(d) for d in self.body.drive_lr) * WALK_SPEED < 0.05
        else:
            still = abs(self.body.pose.v) < 0.05 and abs(self.body.pose.w) < 0.05
        self.still_ticks = self.still_ticks + 1 if still else 0
        if mode == "walk" and self.still_ticks >= STILL_TICKS:
            mode = self.body.pose.mode = "idle"
        self.mode = mode
        self._why, self.m = why, m

    def bookkeep(self, dt: float):
        """Eating, drinking and grooming bookkeeping, the checklist, the learning check and the internal state."""
        mode, m = self.mode, self.m
        # eating, drinking, grooming bookkeeping
        self.eating = self.drinking = 0.0
        if mode == "feed":
            hx, hy = self.body.pose.head
            for f in self.world.food:
                if math.hypot(hx - f.x, hy - f.y) < f.r + 0.8:
                    if f.kind == "sugar":
                        f.amount -= 14.0 * dt
                        self.eating = 1.0
                        if "feed" not in self.done:
                            self.events.add(self.t, "behaviour", "eating: sugar taste → MN9, proboscis out")
                        self.done.add("feed")
                    elif f.kind == "water":
                        f.amount -= 10.0 * dt
                        self.drinking = 1.0
            self.world.food = [f for f in self.world.food if f.amount > 3]
        if mode == "groom" and "dust" in self.senses_now:     # the "Dust it" item: grooming at a wall bump does not count
            self.done.add("groom")
        if mode == "escape" and "sound" in self.senses_now and "loom" not in self.senses_now:
            self.done.add("sound")
        if self.world.drum_speed and abs(m["yaw"]) > 0.2 and (m["yaw"] > 0) == (self.world.drum_speed < 0):
            self.done.add("optomotor")
        if mode == "backward":
            if self.t - self.bristles.last_touch_t < 1.0:
                self.done.add("wall")
            if any(z[0] == "MDN" for z in self.zaps):
                self.done.add("moonwalk")
        if "taste_bitter" in self.senses_now and m["feed"] < 0.1:
            self.bitter_t += dt
            if self.bitter_t > 0.5:
                self.done.add("bitter")
        small = self.senses_now.get("small", "")
        if small in ("L", "R") and ((small == "L" and m["yaw"] < -0.25) or (small == "R" and m["yaw"] > 0.25)) \
                and self.world.tool == "lure" and self.world.hand is not None:
            self.done.add("lure")
        if m["song"] > 0.3 and self.world.female is not None:
            if "court" not in self.done:
                self.events.add(self.t, "behaviour", "courtship song: pC1 → pIP10, one wing out")
            self.done.add("court")
        if self.io.has_plasticity and self.bt.learn[0] and "learn" not in self.done \
                and self.bt.learn[1] > 0.002:
            self.done.add("learn")
            self.events.add(self.t, "learning", "KC→MBON synapses depressed: the fly has learned something about this odour")
        courting = 1.0 if mode == "court" else 0.0
        self.state.step(dt, self.eating, self.drinking, courting)
        self.courting = courting

    def after_senses(self, dt: float, bt: BrainTick):
        """Sense events, the feeding-bout and water Why lines, the driver text and the smoothed numbers shown on screen."""
        mode, why, n_spikes, hz = self.mode, self._why, bt.n_spikes, bt.hz
        # events for salient sense changes
        self._sense_events()
        # sugar at the mouth for a while, yet MN9 itself quiet, with no bitter and nothing silenced: the bout has ended
        self.sugar_t = self.sugar_t + dt if "taste_sugar" in self.senses_now else 0.0
        if self.sugar_t > 1.0 and mode != "feed" and "taste_bitter" not in self.senses_now and not self.user_silenced \
                and self.hz_shown["MN9"] < 15:
            why.append("tastes sugar, but MN9 fires too little to feed: under steady sugar it tires within seconds, "
                       "so feeding comes in bouts")
        if "taste_water" in self.senses_now and mode != "feed":   # water never reaches MN9 here: say so, add no drive
            why.append("touches water, but this fly's data name no water cells" if not self.water_cells else
                       "touches water, but is not thirsty" if self.state.thirst <= 0.2 else
                       "tastes water (LB3a), but in this wiring the water cells do not reach MN9: it does not drink"
                       if getattr(self.real_conn, "sex", "male") != "female" else      # as her What's real says
                       "tastes water (the published model's water cells), but at the game's rates they do not reach MN9: "
                       "it does not drink")
        self.driver = "  +  ".join(why)
        if not self.driver:
            self.driver = ("nothing: the brain is quiet" if not n_spikes else
                           "the brain is busy, but no movement command is strong enough")
        for k, v in hz.items():                      # smooth the numbers shown on screen
            self.hz_shown[k] = self.hz_shown.get(k, 0.0) + (v - self.hz_shown.get(k, 0.0)) * min(1.0, dt / 0.15)

    def watchdog(self, dt: float, bt: BrainTick):
        """Graded count, events per second, and the reset after a runaway (its event carries the game's clock, which
        Game.tick has already advanced, as it always did)."""
        rates = self.rates_now
        # watchdog: this simple model can lock into runaway firing. It counts events: spikes plus the graded cells'
        # release quanta (parts list), each one a spike's worth of transmitter; the graded share is shown apart
        n_graded = bt.n_graded
        self.graded_eps = n_graded / dt
        sps = bt.n_spikes / dt
        self.since_input = 0.0 if rates else self.since_input + dt
        self.runaway_s = self.runaway_s + dt if sps > 150000 else 0.0
        if (self.runaway_s > 1.5 and self.since_input > 0.5) or self.runaway_s > 4.0:
            self.reset_brain()
            # the frame published at the end of this tick describes the brain after the reset (its learning traces
            # and events cleared, its tones gone), as it always did: refresh the fields the tick's answer carried
            for k, v in self.io.status_fields().items():
                setattr(bt, k, v)
            self.calms += 1
            self.runaway_s = 0.0
            self.say("Runaway firing (a known flaw of this simple model: the smell centre, or with the parts list the optic lobe). Brain calmed.", 4.0)
            self.events.add(self.t, "system", "runaway firing: brain reset to rest")
        self.sps = sps

    _prev_felt: dict = {}

    def _sense_events(self):
        f, p = self.senses_now, self._prev_felt
        for key, text in (("loom", "sees something looming"), ("pheromone", "tastes the female's pheromone (foreleg contact)"),
                          ("dust", "dust on the antennae"), ("shock", "electric shock"), ("reward", "sugar reward → PAM dopamine (hand-built)"),
                          ("sound", "hears a loud sound"), ("courting", "courtship arousal: pC1 driven after tapping the female (hand-built)")):
            if key in f and key not in p:
                self.events.add(self.t, "sense", text)
        if f.get("smell") and f.get("smell") != p.get("smell"):
            self.events.add(self.t, "sense", f"smells {ODOURS[f['smell']].name}")
        self._prev_felt = dict(f)

    # ------------------------------------------------------------------ what the page shows
    def learning_summary(self, bt: BrainTick | None = None) -> dict | None:
        """The learning card's numbers; ``bt`` (this tick's) saves asking the brain again (publish passes it)."""
        if not self.io.has_plasticity:
            return None
        info = bt.learning if bt is not None else self.io.learning_summary()
        if info is None:
            return None
        return {"enabled": self.learning_on, "depressed_fraction": round(info["depressed_fraction"], 4),
                "events": info["events"], "learned_bias": round(self.learned_bias, 3), "smelling": self.smelling,
                "mbon": {t: {"strength": round(v["strength"], 3), "now": round(v["now"], 3), "valence": v["valence"],
                             "dopamine": round(v["dopamine"], 3)} for t, v in info["mbon"].items()}}

    # ------------------------------------------------------------------ static data for the browser
    def _make_layout(self):
        c = self.conn
        x, y, z = c.soma[:, 0], c.soma[:, 1], c.soma[:, 2]
        ok = ~np.isnan(x)
        x0, x1 = np.nanmin(x), np.nanmax(x)
        y0, y1 = np.nanmin(y), np.nanmax(y)
        z0, z1 = np.nanmin(z), np.nanmax(z)
        scale = max(x1 - x0, z1 - z0) / 1000.0
        px = np.where(ok, (x1 - x) / scale, -1).round().astype(int)      # flipped: fly's left on the left
        py = np.where(ok, (z - z0) / scale, -1).round().astype(int)      # head at the top, nerve cord below
        pz = np.where(ok, (y - y0) / scale, -1).round().astype(int)      # depth (anterior-posterior)
        sc = c.superclass
        region = np.full(c.n, 6, dtype=int)
        region[np.isin(sc, ["ol_intrinsic", "visual_projection", "visual_centrifugal", "ol_sensory",
                            "visual_projection_tbc"])] = 0
        region[np.char.startswith(sc.astype(str), "cb_")] = 1
        region[np.char.startswith(sc.astype(str), "vnc")] = 3
        region[np.isin(sc, ["descending_neuron", "descending_neuron_tbc"])] = 2
        region[np.isin(sc, ["ascending_neuron", "sensory_ascending", "sensory_ascending_tbc"])] = 4
        region[np.isin(sc, ["cb_motor", "vnc_motor"])] = 5
        region[c.cls == "Kenyon_Cell"] = 7
        region[np.isin(c.cls, ["MBON", "DAN"])] = 8
        type_counts = c.type_counts()
        return json.dumps({
            "n": int(c.n), "w": int(round((x1 - x0) / scale)), "h": int(round((z1 - z0) / scale)),
            "d": int(round((y1 - y0) / scale)),
            "x": px.tolist(), "y": py.tolist(), "z": pz.tolist(), "region": region.tolist(),
            "regions": ["optic lobes", "central brain", "descending", "nerve cord", "ascending", "motor", "other",
                        "Kenyon cells", "MBON / DAN"],
            "arena_r": ARENA_R, "fly_half": FLY_HALF, "tick_ms": TICK_MS,
            "presets": [{"spec": s, "hz": h, "label": l} for s, h, l in ZAP_PRESETS if c.select(s).size],   # cells this fly has
            "types": sorted(type_counts, key=lambda t: -type_counts[t])[:5000],
            "edges": int(c.n_edges), "synapses": int(c.n_syn.sum()),
            "dataset": c.dataset, "sex": c.sex,
            "readouts": self.readout_meta,
            "checks": [{"id": i, "text": t} for i, t in CHECKS                    # none this fly cannot do
                       if (i not in ("court", "genetics") or self.readouts["pIP10"].size)   # no song cells, no song to lose
                       and not (c.sex == "female" and i in ("groom", "sound", "wall"))],
            "odours": [{"id": o.id, "name": o.name, "glomeruli": o.glomeruli, "innate": o.innate, "colour": o.colour,
                        "note": o.note} for o in ODOURS.values()],
            "scenarios": [{"id": k, "name": s.name, "description": s.female if c.sex == "female" and s.female else s.description}
                          for k, s in SCENARIOS.items()],
            "retina": self.retina.layout(),
            "profile": self.profile_name,
            "genetics": self.genetics,
            "genome": {"levels": [{"level": lv, "label": lb} for lv, lb in wiring.LEVELS],
                       "rules": {"type_groups": None}},
            "parts": {"tables": self.parts_list().describe(), "counts": self.parts_counts()},
            "vfb": vfb.ontology_for(c).summary(c),
            "settings": self.io.settings(),
            "decoder": self.decoder.dn_targets,
            "columnar_vision": self.columnar_on,
            "body": self.body_kind, "stride_average": bool(getattr(self.body, "stride_average", False)),
            "whats_real": self.whats_real(),
        }, separators=(",", ":")).encode()

    def _has(self, key: str) -> bool:
        """Whether this fly has the cells of a readout (the female fly has no pIP10 and no TTMn)."""
        i = self.readouts.get(key)
        return i is None or i.size > 0

    def whats_real(self) -> dict:
        male = getattr(self.real_conn, "sex", "male") != "female"
        src = "MaleCNS" if male else "FlyWire"
        return {
            "wiring": [
                "Sugar taste neurons → MN9, the proboscis motor neuron. Bitter taste keeps MN9 silent"
                + (", even on top of sugar." if male else "; on top of sugar only with the published model's settings "
                   "(fly_brain.py --female), not the game's (MN9 7-13 Hz)."),
                (("Water taste cells (LB3a, the type whose outputs match the published model's water cells) → Fudog (DNg67), "
                  "not MN9 (at no rate up to 200 Hz): a thirsty fly tastes water but does not drink." if male else
                  "Water taste cells (the published model's 18 water cells) → Fudog (DNg67); at the game's 80 Hz not MN9, "
                  "so a thirsty fly tastes water but does not drink (at 200 Hz they do reach MN9: docs/SCIENCE.md 2.2).")
                 if self.water_cells else
                 "Water: this fly's data name no water taste cells (FlyWire types its labellar sugar and water cells as "
                 "LB3 (122) and LB2d (7), without splitting them by taste), so water tastes of nothing here and she does "
                 "not drink (docs/SCIENCE.md 9.2)."),
                "Looming detectors (LC4, LPLC2) → giant fibre DNp01, the escape command.",
                "A small moving object seen on one side (LC10a, a courtship-chase cell type) → DNa02 on that same side → a turn toward it.",
                ("Head bristles → MDN, the 'moonwalker' backward-walking neurons. Antennal sensors (Johnston's organ) → "
                 "aDN1/aDN2 grooming neurons." if male else "In this female brain the head bristles reach the grooming neuron "
                 "aDN1 (86 Hz) and not MDN (0 Hz): at a wall she grooms instead of backing up (docs/SCIENCE.md 9.4)."),
                "Odour receptor neurons → projection neurons → a sparse, odour-specific Kenyon-cell code → mushroom body output neurons.",
                *(["Bitter taste → PPL1 dopamine neurons (the punishment signal for learning)."] if male else []),
                "Which Kenyon-cell synapses are plastic and which dopamine neurons gate each MBON: read from the wiring (DAN→MBON synapses).",
                *(["pC1 courtship neurons → pIP10 → wing motor neurons (song), and → DNp13; a female seen as a small moving object → LC10a → DNa02 (the chase)."] if male else []),
                ("A loud sound → Johnston's organ B neurons → the giant fibre (a startle jump), and wind on the antennae → grooming and backing neurons."
                 if male else "In this female brain Johnston's organ reaches the giant fibre and the grooming neurons only weakly (5-9 Hz): "
                 "a clap does not startle her and dust does not make her groom (docs/SCIENCE.md 9.4)."),
                "Wide-field motion → T4/T5 (" + ("driven column by column from the retina" if self.columnar_on else
                                                 "driven as whole populations from the retina's motion signal") +
                ") → HS cells → DNa02 and DNp15 on the same side: the optomotor reflex.",
                ("Which neurons express fruitless and doublesex, and which are male-specific or dimorphic: the MaleCNS annotation, read from the data. Silencing the fruitless neurons stops the song (pIP10 and its route to the wing motor neurons are fru+) and leaves feeding and escape alone."
                 if male else "Which neurons express fruitless and doublesex, and which are female-specific or dimorphic: FlyWire's annotation (Schlegel et al. 2024), read from the data. This female brain has no pIP10 and no nerve cord, so no song."),
                *(["A grown fly (Genome card) keeps the connectome's cell-type wiring rules and nothing else: 9 of the 11 validated reflexes survive on type-level rules, none on class-level rules."] if male else []),
                f"The parts list (Genome card): which neurons make dopamine, octopamine or serotonin is the {src} transmitter prediction"
                + ("" if male else ", corrected from FlyWire's literature column (known_nt)") +
                "; that these act only through slow receptors, and that photoreceptors, L1-L5, the medulla inputs to T4/T5, T4/T5 and HS/VS signal without spikes, is the literature (parts.py cites it).",
                "Which anatomy-ontology class each cell type is (the fbbt: selector, the ontology line in a neuron's popover, the VFB links): the FlyBase anatomy ontology and Virtual Fly Brain's MaleCNS name synonyms, joined offline by name"
                + ("" if male else " (for FlyWire's own type names, the classes FlyWire's annotation gives them)") + ". Where the literature-curated class says a neuron's transmitter differs from the prediction, or fills an 'unclear' one, the parts list follows the literature; the receptors each cell type expresses come from the adult single-cell RNA-seq atlases on VFB and set which way a tone pushes that target.",
            ],
            "hand_built": [
                "The retina (which facet sees what) and the feature computations that turn retinal images into LC4/LPLC2/LC10a/T4/T5 rates.",
                "How smells, wind, touch and dust become firing rates, and which sensory types they drive.",
                ("How descending-neuron firing becomes movement: the decoder's weights, what wins when commands compete, and that "
                 "a giant-fibre burst (5 spikes in 50 ms, not one) is the escape command. "
                 "Speeds and turn rates come from leg physics (NeuroMechFly v2 in MuJoCo; its stepping rhythm, recorded steps and "
                 "left/right drive are flygym's)." if getattr(self, "body_kind", "drawn") == "physics" else
                 "How descending-neuron firing becomes movement: speeds, turn rates, the jump (it needs a giant-fibre burst, "
                 "5 spikes in 50 ms, not one), and what wins when commands compete."),
                "The walking urge, hunger and thirst, odour-guided steering (innate valence + the learned KC→MBON bias), the female's behaviour.",
                "Sugar reward → PAM dopamine except PAM-γ3 (the wiring's taste-to-PAM routes give the best-connected PAM-α1 cells about a third of the drive they need; SCIENCE.md 4.5); the 'shock' tool → PPL1.",
                ("Courtship arousal: tapping the female fires the tarsal taste neurons (wiring), but their route to pC1 is ~8x too weak in this model, so contact also drives pC1 directly."
                 if male else "Courtship arousal: contact drives pC1 directly (this brain has no tarsal taste neurons)."),
                "Wind on Johnston's organ is kept weak: at the rates real wind would give, the same neurons drive grooming in this model; there is no wind-steering route, so heading upwind is hand-built.",
                "Efference copy: the eyes' motion signal is damped while the fly turns on purpose, as in real flies.",
                "The learning rule's constants (rate, time windows, floor, forgetting).",
                f"Fixes for runaway loops: mild neuron fatigue and blocking the output of the {self.real_conn.select('class:ALLN').size:,} antennal-lobe local neurons.",
                "With the parts list on, where APL releases: that its inhibition stays local to the busy part of the mushroom body is the literature (Amin et al. 2020), the rule that turns each lobe's Kenyon-cell activity into APL's release there is the kit's. That dopamine turns APL down through Dop2R is the literature (Zhou et al. 2019).",
            ],
            "not_modelled": [
                "Real neuron shapes and individual properties, hormones, electrical synapses, most neuromodulation, development.",
                "Genes beyond two transcription factors' expression labels, the transmitter each neuron makes and (with the parts list on) the aminergic receptors its cell type expresses where an adult scRNA-seq cluster exists. Still no ion-channel differences, no peptide signalling (the ontology only names the peptidergic types), no development from the genome.",
                "Absolute firing rates shouldn't be trusted, only which neurons respond. Nothing here is conscious.",
                *(["The escape jump with the physics body: the giant fibre fires, but the fly stays on its feet (NeuroMechFly "
                   "has no jump model)."] if getattr(self, "body_kind", "drawn") == "physics" else []),
            ],
        }
