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
from .game import (BUILTIN_KEYS, CHECKS, COURTSHIP_HZ, COURTSHIP_SECS, COURTSHIP_SPEC, FEMALE_READOUTS, GF_BURST, HIDDEN_READOUTS,
                   PAIR_CHECKS_FEMALE, PAIR_CHECKS_MALE, PAIR_CHECKS_SAME,
                   PHEROMONE_GRNS, READOUTS, REWARD_HZ, REWARD_SPEC, SHOCK_HZ, SHOCK_SPEC, SOUND_HZ, SOUND_SPEC,
                   STILL_TICKS, TICK_MS, ZAP_PRESETS, InternalState, MotorDecoder)
from .scenarios import PAIR_SCENARIOS, SCENARIOS, pair_available
from .senses.mechano import Antennae, Bristles
from .senses.mechano import SOUND
from .senses.olfaction import ODOURS, Nose
from .senses import social
from .senses.taste import WATER_GRNS, Forelegs, Mouth
from .senses.vision import Retina
from .world import ARENA_R, FLY_HALF, wrap


def _empty_tick(readouts: dict) -> BrainTick:
    """What a fly's brain has said before its first tick: nothing."""
    zero = {k: 0.0 for k in readouts}
    return BrainTick(zero, dict(zero), 0, 0, 0, [], 0.0, (0, 0.0), None, 0, [], None)


HOME = (0.0, -12.0, math.pi / 2)             # where a new fly stands (FlyBody.reset's own default)
PARTNER_HOME = (0.0, 12.0, -math.pi / 2)     # a partner starts across the dish, facing the protagonist
PAIR_SEEN_MM = 30.0                          # a pair "seen" item needs the other fly this near, in the eye that saw
                                             # something small move (hand-built, provisional: FlyAgent._other_in_view)


@dataclass
class PoseView:
    """One fly as the other flies see it at the start of a tick (docs/TWO_FLIES_PLAN.md 5.4, D4): every fly senses
    the same snapshot, whatever order the flies are ticked in. Positions are real millimetres (the dish is in
    real mm); ``song`` and ``court`` are the fly's last decoded drives (0..1)."""
    id: int
    sex: str
    x: float
    y: float
    h: float
    v: float
    wing_l: float
    wing_r: float
    song: float
    court: float
    jump: bool
    body_kind: str


class FlyAgent:
    """One fly: everything the game keeps per fly. ``game`` holds the shared world and clock (read here through
    :attr:`world`, :attr:`t`, :attr:`events`, :attr:`actions`, :attr:`scenario`, :meth:`say`).

    The brain is reached only through :attr:`io` (brainio.py): a :class:`~virtual_fly.brainio.LocalBrain` in this
    process, or a :class:`~virtual_fly.brainio.ProcessBrain` in a child process of its own (``brain_procs``), with
    the same numbers either way. :attr:`brain` is the FlyBrain itself when it is local, else None."""

    def __init__(self, game, id: int, brain=None, *, conn=None, sex: str, rng, seed: int, autopilot: bool = True,
                 columnar: bool = True, body: str = "drawn", stride_average: bool = False, parts_list=None,
                 brain_factory=None, brain_kwargs: dict | None = None, retest: str = "auto", brain_procs: bool = False,
                 parts=False, pair: bool = False, home: tuple | None = None, io_factory=None):
        """``brain``: a built FlyBrain (the protagonist's, as play.py builds it); or ``brain=None`` with ``conn``: a fly
        whose brain is built from ``brain_kwargs`` and ``parts`` (False, True or a PartsList), in its own process when
        ``brain_procs`` is set, else here (a partner, docs/TWO_FLIES_PLAN.md 5.4). ``pair``: this fly shares the dish
        with another simulated fly (a female then shows her decision neurons, 5.6). ``home``: where a new fly stands.
        ``io_factory``: with ``brain_procs``, ``io_factory(spec, fly)`` makes the seam instead of a ProcessBrain of its
        own (the game's shared brain process, brainio.GpuBrainServer.attach, 6.5)."""
        self.game, self.id, self.sex, self.rng, self.seed = game, id, sex, rng, seed
        self.pair = pair
        self.home = tuple(home) if home is not None else HOME
        if brain is None:
            if conn is None:
                raise ValueError("a fly needs a brain, or a connectome to build one from")
            if brain_procs and brain_kwargs is None:
                raise ValueError("a brain in its own process needs the brain's settings (brain_kwargs)")
            self.conn = self.real_conn = conn
            self.brain_factory = brain_factory or self._default_brain_factory
            self._brain_kwargs = dict(brain_kwargs or {})
            self.parts_on = bool(parts)
            self._parts_list = parts if isinstance(parts, partslib.PartsList) else parts_list
        else:
            self.conn = brain.conn
            # the genome: the real wiring, and flies grown from its rules (see wiring.py)
            self.real_conn = brain.conn
            self.brain_factory = brain_factory or self._default_brain_factory
            # how a re-test process rebuilds the brain: build_brain(conn, profile, **brain_kwargs, parts=...). Known
            # for the default factory; a caller with its own factory says so (play.py), or re-tests run in a thread.
            self._brain_kwargs = brain_kwargs if brain_kwargs is not None else (None if brain_factory else {})
            self.parts_on = brain.parts is not None          # the genes as each neuron's parts list (parts.py)
            # the PartsList to (re)build with: the running brain's, else the one the caller chose (curated policy ...)
            self._parts_list = brain.parts.parts if brain.parts is not None else parts_list
        if retest not in ("auto", "process", "thread"):
            raise ValueError("retest must be 'auto', 'process' or 'thread'")
        self.retest_mode = retest
        self._retest_handle = None                      # the running re-test process, if any
        self._survival_cache: dict[tuple, list] = {}     # re-test results: the same fly and settings give the same rows
        self._retest_lock = threading.Lock()
        self._rules_cache: dict = {}
        self._survival_token = None
        self.genome: dict = {"level": "real", "seed": 0, "growing": None, "survival": None, "wiring": None, "rules": None, "error": None}
        self._parts_counts: dict | None = None
        # the readouts this fly reports each tick: the kit's, the decoder's hidden ones and, for a female with a
        # partner, her decision neurons (FEMALE_READOUTS); the same list, in the same order, in a brain process
        self._readout_specs: list = [(r[0], r[1]) for r in READOUTS] + list(HIDDEN_READOUTS.items())
        if pair and sex == "female":                     # her decision neurons, watched (5.6; never a verdict)
            self._readout_specs += [(r[0], r[1]) for r in FEMALE_READOUTS]
        # the brain: here, or in a child process built from this brain's settings (then this one is let go)
        make_io = io_factory or (lambda spec, fly: ProcessBrain(spec, fly=fly))
        if brain is None:
            if brain_procs:
                self.io = make_io(self.child_spec(None), id)
            else:
                from .settings import build_brain
                self.io = LocalBrain(build_brain(self.conn, self.profile_name, **self._brain_kwargs,
                                                 parts=self.parts_arg(self.parts_on)))
            self._learning_on = self.io.has_plasticity
        else:
            self._learning_on = brain.plasticity is not None
            if brain_procs:
                self.io = make_io(self.child_spec(brain), id)
                del brain
            else:
                self.io = LocalBrain(brain)
        self.histories: dict[str, list] = {}             # a process brain's readout histories, one value per tick
        self.autopilot = autopilot
        if isinstance(body, str):
            if body == "physics":                      # optional: NeuroMechFly v2 in MuJoCo (virtual_fly/physics.py)
                from .physics import make_body
                self.body = make_body("physics", self.world, self.rng, seed=seed, stride_average=stride_average)
            else:
                self.body = FlyBody(self.world, self.rng)
            self.body_kind = body
        else:                                          # a body the game made: this fly's body in the shared MuJoCo world
            self.body = body                           # of a physics pair (physics_pair.PairPhysicsBody; docs/TWO_FLIES_PLAN.md 8.3)
            self.body.world, self.body.rng = self.world, self.rng
            self.body_kind = body.kind
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
        self.readouts = {k: self.conn.select(spec) for k, spec in self._readout_specs}
        c = self.conn
        shown = READOUTS + (FEMALE_READOUTS if pair and sex == "female" else [])   # the bars the page shows
        self._shown = shown                              # (a rebuilt brain gets monitors for the same bars)
        # the keys a 'watch' may never take: the decoder's and the page's built-in readouts, hers included
        self.builtin_keys = BUILTIN_KEYS | ({r[0] for r in FEMALE_READOUTS} if pair and sex == "female" else set())
        self.readout_meta = [{"key": r[0], "spec": r[1], "label": r[2], "group": r[3], "max": r[4], "colour": r[5],
                              "genes": genetics.genotype(c, c.select(r[1]))["tags"]}
                             for r in shown if self.readouts[r[0]].size]      # (the female fly has no pIP10, no TTMn)
        self.genetics = genetics.summary(c, self.readout_meta)
        self.neuronbridge = genetics.NeuronBridge()
        self.custom_readouts: dict[str, str] = {}
        for r in shown:
            if self.readouts[r[0]].size:                 # not for cells this fly lacks (/api/history leaves them out)
                self.add_monitor(r[0], r[1])
        self.user_silenced: set[str] = set()
        self.user_modulated: dict[str, float] = {}
        self.has_soma = ~np.isnan(self.conn.soma[:, 0])
        # the social channels (senses/social.py): which cells this fly has for each, looked up once
        self.has_leg_taste = any(self.conn.select(spec).size for spec in PHEROMONE_GRNS)
        self.has_sound_cells = self.conn.select(SOUND).size > 0
        self.has_cva_cells = self.conn.select("ORN_DA1").size > 0
        self.has_sag_cells = self.conn.select(social.SAG_SPEC).size > 0
        self._said_no_leg_taste = False
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
        self._others = ()                                # the other flies' start-of-tick poses this tick (act keeps them)

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

    def child_spec(self, brain=None) -> dict:
        """What a brain process needs to build this brain (as _retest_spec says it for the re-test): from a built
        brain's settings, or (``brain=None``, a partner) from this fly's brain_kwargs alone."""
        path = getattr(self.real_conn, "path", None)
        if path is None or not os.path.exists(path) or self.real_conn.meta.get("rewired"):
            raise ValueError("a brain in its own process is built from the connectome's file, and this wiring is not "
                             "the file's (" + ("grown or rewired: " + str(self.real_conn.meta.get("rewired")) if path
                                               else "no file") + "); use brain_procs='off'")
        if self._brain_kwargs is None:
            raise ValueError("a brain in its own process needs the brain's settings (a custom brain_factory without "
                             "brain_kwargs); use brain_procs='off'")
        kw = {k: v for k, v in brain.settings().items() if k in self._BRAIN_KWARGS} if brain is not None else {}
        kw.update(self._brain_kwargs)
        kw["parts"] = self.parts_arg(self.parts_on)
        return {"path": str(path), "wiring": None, "profile": self.profile_name, "brain_kwargs": kw,
                "readouts": list(self._readout_specs),
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

    def event(self, kind: str, text: str, **extra):
        """Log an event of this fly's; with more than one fly in the dish it carries ``fly`` (its id), so the page can
        tell whose it is; with one fly the event is exactly what it always was."""
        if len(self.game.flies) > 1:
            extra.setdefault("fly", self.id)
        self.game.events.add(self.game.t, kind, text, **extra)

    # ------------------------------------------------------------------ a new fly (the per-fly part of Game.reset_world)
    def reset_state(self):
        """Everything this fly forgets when the player asks for a new fly (its checklist is kept)."""
        self.body.reset(*self.home)
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
        if kw.get("backend") == "cupy" and not isinstance(self.io, LocalBrain):
            kw["backend"] = "auto"                   # a GPU brain lives in the brain process only (6.5): anything built
        kw.update(extra)                             # here beside it (the re-test's thread fallback) runs on the CPU
        return build_brain(conn, self.profile_name, **kw)

    def _cpu_only(self) -> dict:
        """The keyword that keeps a brain built in this process off the GPU when the seam's brain is a GPU brain in
        the brain process (plan 6.5: the re-test always uses the CPU, whose spikes are the same)."""
        try:
            backend = self.io.settings().get("backend")
        except Exception:                            # a dead brain process: the caller reports it
            backend = None
        return {"backend": "auto"} if backend == "cupy" else {}

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
        self.event("genome", f"growing a fly: {level} wiring, seed {seed}")
        self.say(f"Growing a fly from its {level} wiring rules…", 4.0)
        threading.Thread(target=self._grow_worker, args=(level, seed), daemon=True).start()

    def _grow_worker(self, level: str, seed: int):
        try:
            conn2, rules = wiring.grow_level(self.real_conn, level, seed, rules_cache=self._rules_cache)
            brain2 = self._build_or_ship(conn2, self.parts_arg(self.parts_on))
            cmp = wiring.compare(self.real_conn, conn2) if conn2 is not self.real_conn else None
            self.actions.put({"type": "_swap_brain", "fly": self.id, "brain": brain2, "conn": conn2, "level": level, "seed": seed,
                              "rules": rules.summary() if rules is not None else None, "wiring": cmp,
                              "parts": self.parts_on, "reason": "grow"})
        except Exception as e:                       # a bad level, or out of memory: report, keep the old fly
            self.genome.update(growing=None, error=str(e))
            print("error growing a fly", repr(e))

    def _start_rebuild(self, on: bool):
        """Rebuild the current fly's brain with the parts list on or off (same wiring), in the background."""
        self.genome.update(growing={"level": self.genome["level"], "seed": self.genome["seed"], "reason": "parts",
                                    "parts": on, "t0": time.time()}, error=None)
        self.event("genome", f"rebuilding the brain with the parts list {'on' if on else 'off'}")
        self.say("Giving each neuron its parts…" if on else "Back to identical neurons…", 4.0)
        threading.Thread(target=self._rebuild_worker, args=(on,), daemon=True).start()

    def _rebuild_worker(self, on: bool):
        try:
            brain2 = self._build_or_ship(self.conn, self.parts_arg(on))
            self.actions.put({"type": "_swap_brain", "fly": self.id, "brain": brain2, "conn": self.conn, "level": self.genome["level"],
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
                for r in self._shown:                    # the same bars as at the start (hers included)
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
                self.event("genome", f"the brain could not be rebuilt: {e}")
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
                self.event("genome", f"parts list on: {c['modulatory_neurons']:,} modulatory neurons act through slow "
                                f"tones on {c['modulated_targets']:,} targets, {c['graded_neurons']:,} cells transmit graded signals")
                self.say("Each neuron now has its parts. Testing the reflexes…", 4.0)
                self.done.add("parts")
            else:
                self.event("genome", "parts list off: every neuron is the same machine again")
                self.say("Every neuron is the same machine again. Testing the reflexes…", 4.0)
        elif level == "real":
            self.event("genome", "back to the real wiring")
            self.say("The real wiring is back.", 3.0)
        else:
            w = a["wiring"] or {}
            self.event("genome", f"a fly grown from its {level} wiring rules (seed {seed}): "
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
        if kw.get("backend") == "cupy":
            kw["backend"] = "auto"                   # the re-test child always uses the CPU (plan 6.5): same spikes, no second GPU context
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
        kw.pop("backend", None)                      # every integrator gives the same spikes: the report does not depend on it
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
                brain = self.brain_factory(conn, parts=self.parts_arg(self.parts_on), **self._cpu_only())
                rows = survival_report(brain, profile=self.profile_name, on_progress=progress,
                                       should_stop=lambda: self._survival_token is not token)
                where = "thread"
            if self._survival_token is token:
                if cached is None:
                    self._survival_cache[key] = [dict(r) for r in rows]
                ok = sum(1 for r in rows if r["ok"]); tested = sum(1 for r in rows if r["ok"] is not None)
                self.genome["survival"] = {"running": False, "results": rows, "ok": ok, "tested": tested,
                                           "secs": round(time.time() - t0, 1), "where": where}
                self.event("genome", f"reflex survival: {ok} of {tested} experiments pass on this wiring")
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

    # ------------------------------------------------------------------ as the other flies see this one
    def pose_view(self) -> PoseView:
        """This fly at the start of a tick, for the other flies' senses (a snapshot: D4)."""
        p = self.body.pose
        return PoseView(self.id, self.sex, p.x, p.y, p.h, p.v, p.wing_l, p.wing_r, float(self.m.get("song", 0.0)),
                        float(self.m.get("court", 0.0)), p.jump is not None, self.body_kind)

    @property
    def can_court(self) -> bool:
        """Whether the decoder may put this fly in court mode (the wing gesture, the "court" label): the protagonist
        always (as the single fly always could); a partner only when it has song cells (pIP10): a female partner's
        pC1 shows as a readout instead (docs/TWO_FLIES_PLAN.md D12)."""
        return self.id == 0 or self.readouts["pIP10"].size > 0

    @property
    def contact_pc1(self) -> bool:
        """Whether tapping a simulated partner drives this fly's pC1 directly (the kit's hand-built arousal): by the
        social config, else on for a male and off for a female toucher (decision 14). The scripted-female path of
        single-fly play does not read this (D7)."""
        return self.game.social.contact_pc1_for(self.sex)

    # ------------------------------------------------------------------ senses (hand-built encoders)
    def senses(self, dt: float, others=()):
        """Returns (spec -> Hz, per-neuron indices, per-neuron Hz) and fills ``self.senses_now``. ``others``: the
        other flies' start-of-tick poses (PoseView), which the social encoders read (senses/social.py); with none
        (the single fly) nothing here changes."""
        pose = self.body.pose
        rates: dict[str, float] = {}
        felt: dict = {}

        def add(r: dict[str, float]):
            for spec, hz in r.items():
                rates[spec] = max(rates.get(spec, 0.0), hz)

        cfg = self.game.social
        # eyes: render the retina, derive feature-detector rates (+ columnar T4/T5 when enabled); another simulated
        # fly is a small dark object to it, as the scripted female is (channel 1)
        vis, col_idx, col_hz = self.retina.look(pose, dt, turn_command=self.turn_command,
                                                others=others if (others and cfg.seen) else ())
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
        if others:                                   # the other simulated flies (senses/social.py; hand-built encoders)
            self._social_senses(dt, others, rates, felt)
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
        # touch and proprioception (a bump into another fly counts only with the touch channel on: channel 9)
        bumped = self.body.bumped or bool(others and cfg.touch and getattr(self.body, "bumped_fly", False))
        add(self.bristles.rates(pose, dt, bumped, self.t))
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

    def _social_senses(self, dt: float, others, rates: dict, felt: dict):
        """What the other simulated flies do to this fly's sensory neurons (senses/social.py, docs/TWO_FLIES_PLAN.md
        5.5): hand-built encoders, each with a switch; the rates merge with the fly's own by maximum, as every sense
        does. Only called when there are other flies."""
        cfg, pose = self.game.social, self.body.pose
        if cfg.contact:                                  # channel 3: a foreleg tip on the other fly
            if hasattr(self.body, "touching_other"):     # a physics pair: the tap is MuJoCo's contact of this fly's head or
                tap = self.body.touching_other           # forelegs on the other's body, with its force (8.4, decision 23)
                touched = next((o for o in others if o.id == tap[0]), None) if tap is not None else None
            else:
                tap = None
                touched = social.touching(pose, others)  # the drawn rule: a foreleg tip within 3.4 mm of the other's centre
            if touched is not None:
                felt["touches_fly"] = touched.id
                if tap is not None:
                    felt["tap_force"] = round(tap[1], 1)   # MuJoCo's units (the fly weighs about 9,800)
                if touched.sex == "female":              # these leg taste cells answer female pheromone (hand-built rule)
                    if self.has_leg_taste:
                        for spec, hz in PHEROMONE_GRNS.items():
                            rates[spec] = max(rates.get(spec, 0.0), hz)
                        felt["pheromone"] = True
                    elif not self._said_no_leg_taste:    # said once, not every tick
                        self._said_no_leg_taste = True
                        self.event("sense", "taps the other fly, but leg taste is not wired in this brain "
                                   "(no LgLG1a/LgLG1b cells)")
                    if self.contact_pc1:                 # the kit's hand-built pC1 arousal, as for the scripted female
                        self.court_left = COURTSHIP_SECS
        if cfg.song and self.has_sound_cells:            # channel 2: the other fly's song on Johnston's organ
            hz = social.song_rate(pose, others)
            if hz > 0.0:
                rates[SOUND] = max(rates.get(SOUND, 0.0), hz)
                felt["hears_song"] = round(hz, 1)
        if cfg.cva and self.has_cva_cells:               # channel 5 (off by default): a male's cVA on ORN_DA1
            hz = social.cva_rate(pose, others)
            if hz > 0.0:
                rates["ORN_DA1"] = max(rates.get("ORN_DA1", 0.0), hz)
                felt["smells_cva"] = round(hz, 1)
        if cfg.mating == "virgin" and self.sex == "female" and self.has_sag_cells:   # channel 6 (off): "virgin" as a tonic SAG drive
            rates[social.SAG_SPEC] = max(rates.get(social.SAG_SPEC, 0.0), social.SAG_HZ)
            felt["virgin_drive"] = True

    def sense(self, dt: float, others=()):
        """The same as :meth:`senses` (the name the plan uses for one fly's sensing step)."""
        return self.senses(dt, others)

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
                self.event("behaviour", "escape command (giant fibre DNp01 burst; the physics body has no jump)")
            else:
                self.say("Giant fibre fired: escape jump!", 1.5)
                self.event("behaviour", "escape jump (giant fibre DNp01 burst)")
            return "escape"
        # strongest command wins (a hand-built stand-in for the nerve cord's own arbitration).
        # A behaviour starts above its threshold and continues until its drive falls to half of it.
        options = [(0.0, "")]
        for name, start, weight in (("backward", 0.25, 1.15), ("feed", 0.3, 1.0), ("groom", 0.35, 0.9)):
            ongoing = self.mode == name
            if m[name] > (start / 2 if ongoing else start):
                options.append((weight * m[name] + (0.15 if ongoing else 0.0), name))
        mode = max(options)[1] or "walk"
        if mode == "walk" and (m["song"] > 0.3 or m["court"] > 0.4) and self.can_court:
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

    def act(self, dt: float, bt: BrainTick, others=()):
        """Decode the descending neurons, choose a mode, add the hand-built walking urge and steering, move the body,
        then the resting rule (a walk still for STILL_TICKS reads idle). ``others``: the other flies' start-of-tick
        poses, which the song gesture faces and the body must not walk into (senses/social.py)."""
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
        elif others:                                 # the same hand-built gestures, toward the nearest simulated fly
            o = social.nearest(self.body.pose, others)
            bearing = wrap(math.atan2(o.y - self.body.pose.y, o.x - self.body.pose.x) - self.body.pose.h)
            self.song_side = 1.0 if bearing < 0 else -1.0
            # the abdominal bend is a male courtship gesture: a partner that may not court (no pIP10, D12) keeps it
            drive["abdomen"] = 1.0 if (m["court"] > 0.5 and self.can_court and social.distance(self.body.pose, o) < 6.0) else 0.0
        self._others = others
        capsules = social.capsules_of(others) if (others and self.game.social.collide) else ()   # channel 4
        if capsules:
            self.body.move(dt, mode, drive, wander_yaw, others=capsules)
        else:
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
                            self.event("behaviour", "eating: sugar taste → MN9, proboscis out")
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
                self.event("behaviour", "courtship song: pC1 → pIP10, one wing out")
            self.done.add("court")
        elif m["song"] > 0.3 and any(o.sex == "female" for o in self._others):   # sung at a simulated female (D9: no scripted one)
            if "court" not in self.done:
                self.event("behaviour", "courtship song: pC1 → pIP10, one wing out")
            self.done.add("court")
        if self._others:
            self._pair_checks()
        if self.io.has_plasticity and self.bt.learn[0] and "learn" not in self.done \
                and self.bt.learn[1] > 0.002:
            self.done.add("learn")
            self.event("learning", "KC→MBON synapses depressed: the fly has learned something about this odour")
        courting = 1.0 if mode == "court" else 0.0
        self.state.step(dt, self.eating, self.drinking, courting)
        self.courting = courting

    def pair_checks(self) -> list[tuple[str, str]]:
        """The checklist items about the other fly this fly can do (docs/TWO_FLIES_PLAN.md 5.9 item 5): a male's
        with a female in the dish (seen, sang if it has song cells, tapped if it has leg taste cells), a female's
        with a male in the dish (heard if she has sound cells, seen him, touched), and with a partner of its own
        sex only the eyes' item (a tap tastes nothing, there is no song to hear). An item whose channel is off
        (--social) is left out, since it could never tick. Empty for a single fly."""
        others = [f for f in self.game.flies if f is not self]
        if not others:
            return []
        cfg = self.game.social
        needs = {"pair:seen": cfg.seen, "pair:seen_him": cfg.seen, "pair:sang": cfg.song, "pair:heard": cfg.song,
                 "pair:tapped": cfg.contact, "pair:touched": cfg.contact}
        if self.sex == "male" and any(f.sex == "female" for f in others):
            items = [(i, text) for i, text in PAIR_CHECKS_MALE
                     if (i != "pair:sang" or self.readouts["pIP10"].size) and (i != "pair:tapped" or self.has_leg_taste)]
        elif self.sex == "female" and any(f.sex == "male" for f in others):
            items = [(i, text) for i, text in PAIR_CHECKS_FEMALE if i != "pair:heard" or self.has_sound_cells]
        else:
            items = list(PAIR_CHECKS_SAME.get(self.sex, []))
        return [(i, text) for i, text in items if needs.get(i, True)]

    def _pair_checks(self):
        """Tick the pair checks from this tick's senses (hand-built rules on the labelled encoders' outputs): only
        the items this fly's checklist lists (:meth:`pair_checks`: its sex, its cells, the channels that are on)."""
        felt, m, pose = self.senses_now, self.m, self.body.pose
        listed = {i for i, _ in self.pair_checks()}
        if not listed:
            return
        # the eyes: a small moving object on the side the other fly is, with the other fly near enough to be it and
        # no hand or lure in the dish (the retina does not say which object moved: a sugar drop or a post crossing
        # the eye with the other fly behind, or on the other side, does not count)
        seen_id = next((i for i in ("pair:seen", "pair:seen_him") if i in listed), None)
        if seen_id is not None and "small" in felt and self.world.hand is None and self._other_in_view(felt["small"]):
            self.done.add(seen_id)
        # the song: sung at a female the song channel can reach (its item says "within 15 mm of her")
        if "pair:sang" in listed and m["song"] > 0.3 and any(
                o.sex == "female" and social.distance(pose, o) <= social.SONG_FAR_MM for o in self._others):
            self.done.add("pair:sang")
        if "pair:tapped" in listed and "touches_fly" in felt and "pheromone" in felt:
            self.done.add("pair:tapped")
        if "pair:heard" in listed and "hears_song" in felt:
            self.done.add("pair:heard")
        if "pair:touched" in listed and any(f.senses_now.get("touches_fly") == self.id
                                            for f in self.game.flies if f is not self):
            self.done.add("pair:touched")

    def _other_in_view(self, sides: str) -> bool:
        """Whether another fly stands within PAIR_SEEN_MM in the field of an eye that reports a small moving object
        (``sides``: "L", "R" or "LR"): from 8 degrees across the midline to 165 degrees to that side, as the eyes of
        senses/vision.py see. A hand-built rule for the checklist only (the brain's LC10a rates are the retina's,
        whatever moved); provisional."""
        for o in self._others:
            if social.distance(self.body.pose, o) > PAIR_SEEN_MM:
                continue
            rel = math.degrees(wrap(math.atan2(o.y - self.body.pose.y, o.x - self.body.pose.x) - self.body.pose.h))   # + = left
            if ("L" in sides and -8.0 <= rel <= 165.0) or ("R" in sides and -165.0 <= rel <= 8.0):
                return True
        return False

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
            self.event("system", "runaway firing: brain reset to rest")
        self.sps = sps

    _prev_felt: dict = {}

    def _sense_events(self):
        f, p = self.senses_now, self._prev_felt
        for key, text in (("loom", "sees something looming"), ("pheromone", "tastes the female's pheromone (foreleg contact)"),
                          ("dust", "dust on the antennae"), ("shock", "electric shock"), ("reward", "sugar reward → PAM dopamine (hand-built)"),
                          ("sound", "hears a loud sound"), ("courting", "courtship arousal: pC1 driven after tapping the female (hand-built)")):
            if key in f and key not in p:
                self.event("sense", text)
        if f.get("smell") and f.get("smell") != p.get("smell"):
            self.event("sense", f"smells {ODOURS[f['smell']].name}")
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

    # ------------------------------------------------------------------ what the page shows of this fly
    def state_entry(self, bt: BrainTick) -> dict:
        """This fly's entry in the state's ``flies`` list (two or more flies; docs/TWO_FLIES_PLAN.md 5.7): the per-fly
        fields the single fly publishes at the top level, from this tick's BrainTick. ``mode`` takes the same values
        as the top level, ``idle`` included; the learning summary comes with the tick for every fly."""
        return {"id": self.id, "sex": self.sex, "dataset": self.conn.dataset,
                "fly": self.body.to_dict(), "mode": self.mode, "autopilot": bool(self.autopilot), "driver": self.driver,
                "senses": self.senses_now,
                "retina": self.retina.images_b64(),
                "hz": {k: round(v, 1) for k, v in self.hz_shown.items() if self._has(k)},
                "motor": {k: round(v, 3) for k, v in self.decoder.m.items()},
                "spikes": bt.spikes_shown, "sps": int(self.sps), "graded_eps": int(self.graded_eps), "stims": bt.stims,
                "calms": self.calms, "state": self.state.to_dict(), "learning": self.learning_summary(bt),
                "silenced": sorted(self.user_silenced), "baseline": sorted(set(bt.silenced_specs) - self.user_silenced),
                "modulated": self.user_modulated, "custom": self.custom_readouts,
                "genome": self.genome_status(bt.parts_status, known=True), "done": sorted(self.done)}

    def layout_json(self) -> bytes:
        """This fly's layout (``GET /api/layout?fly=k``); fly 0's is the game's own ``layout_json``, made once."""
        return self._make_layout()

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
            "checks": [{"id": i, "text": self._check_text(i, t)} for i, t in CHECKS   # none this fly cannot do
                       if (i not in ("court", "genetics") or self.readouts["pIP10"].size)   # no song cells, no song to lose
                       and not (c.sex == "female" and i in ("groom", "sound", "wall"))
                       and not (i == "court" and self._partner_sexes() == {"male"})]    # a male partner: no one to court, no female to add (D9)
                      + [{"id": i, "text": t} for i, t in self.pair_checks()],        # only with a partner in the dish
            "odours": [{"id": o.id, "name": o.name, "glomeruli": o.glomeruli, "innate": o.innate, "colour": o.colour,
                        "note": o.note} for o in ODOURS.values()],
            "scenarios": [{"id": k, "name": s.name, "description": self._scenario_text(s)} for k, s in SCENARIOS.items()]
                         + [{"id": k, "name": s.name, "description": s.description} for k, s in PAIR_SCENARIOS.items()
                            if pair_available(self.game, k) is None],                 # the two-fly scenarios this dish can run
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

    def _scenario_text(self, s) -> str:
        """A scenario's description for this fly: its two-fly text with a partner in the dish (the scripted female
        cannot enter then, D9; the text is chosen by the protagonist's sex and the partner's), the female fly's own
        where hers differs, else the plain one. Single-fly texts unchanged."""
        flies = self.game.flies
        if len(flies) > 1 and s.pair:
            return s.pair(flies[0].sex, flies[1].sex) if callable(s.pair) else s.pair
        return s.female if self.conn.sex == "female" and s.female else s.description

    def _partner_sexes(self) -> set:
        """The sexes of the other simulated flies in the dish (empty for a single fly)."""
        return {f.sex for f in self.game.flies if f is not self}

    def _check_text(self, check_id: str, text: str) -> str:
        """A checklist item's text; with a simulated partner in the dish the two courtship items name the partner,
        since no scripted female can be added then (D9): a female partner is courted, a male one is not (the item
        is left out of the list), and the fruitless item says what is lost. Single-fly texts are untouched."""
        if check_id in ("court", "genetics") and len(self.game.flies) > 1:
            if "female" in self._partner_sexes():
                if check_id == "court":
                    return text.replace("Add a female → it chases her", "The female partner: it chases her")
                return text.replace("then add a female:", "with the female partner in the dish:")
            if check_id == "genetics":                   # a male partner: nothing to court, no female to add
                return text.replace("then add a female: no chase, no song, as in fruitless mutants",
                                    "as in fruitless mutants (pIP10 and its route to the wings are fru+; with a male "
                                    "partner in the dish there is no song to lose)")
        return text

    def _has(self, key: str) -> bool:
        """Whether this fly has the cells of a readout (the female fly has no pIP10 and no TTMn)."""
        i = self.readouts.get(key)
        return i is None or i.size > 0

    def whats_real(self) -> dict:
        out = self._whats_real_one()
        if len(self.game.flies) > 1:
            self._whats_real_pair(out)
        return out

    def _whats_real_pair(self, out: dict):
        """What the two-flies work adds, each line a hand-built encoder with a switch (senses/social.py); with one fly
        nothing here is shown, so the single fly's text is unchanged."""
        cfg = self.game.social
        hb = out["hand_built"]
        for i, line in enumerate(hb):                     # the scripted female is single-fly play's; here the partner has a brain
            if "the female's behaviour" in line:
                hb[i] = line.replace("the female's behaviour", "the scripted female's behaviour (single-fly play only: here the "
                                     "other fly has a brain of its own)")
        hb.append("Two simulated flies reach each other only through the world, never synapse to synapse; each channel below "
                  "is a hand-built encoder with a switch (--social), and what its sensory neurons do next is the wiring.")
        if cfg.seen:
            hb.append("Seen: the other fly is a small dark object to the retina (radius 1.6 mm, height 2.2 mm at the drawn scale, "
                      "as the scripted female was); the detectors that turn it into LC10a/LC11/LC4/LPLC2 rates are the kit's.")
        if cfg.song:
            hb.append(f"Song: the singer's decoded song drives the hearer's Johnston's organ (JO-A/JO-B) at up to {social.SONG_MAX_HZ:g} Hz "
                      f"per cell, full within {social.SONG_NEAR_MM:g} mm and fading to nothing at {social.SONG_FAR_MM:g} mm (real "
                      f"centre-to-centre mm; provisional, no source). {social.SONG_MAX_HZ:g} Hz is a hand-built calibration taken "
                      "from the male connectome and the burst rule: the loudest steady drive at which fewer than 1 % of 50 ms "
                      "windows make his giant fibre burst (measured over five seeds, parts list off and on; docs/SCIENCE.md).")
        if cfg.contact:
            hb.append(f"Contact: a foreleg tip within {social.CONTACT_MM:g} mm (drawn scale) of the other fly tastes it: a toucher's leg "
                      "taste cells (LgLG1a/LgLG1b, when its file has them) fire when the touched fly is female, and a male toucher's "
                      "pC1 is also driven directly (the kit's hand-built arousal, switchable: contact_pc1); a female toucher's is not.")
        if cfg.collide:
            hb.append(f"Collide: each drawn body is a capsule (half-length {social.FLY_CAPSULE_HALF:g} mm, radius "
                      f"{social.FLY_CAPSULE_R:g} mm, drawn scale) the other cannot walk into; any overlap is pushed apart equally.")
        if cfg.cva:
            hb.append(f"cVA: a male's pheromone reaches the other fly's ORN_DA1 at up to {social.CVA_MAX_HZ:g} Hz within "
                      f"{social.CVA_MM:g} mm (real centre-to-centre mm, Taisz et al. 2023); the rate is provisional.")
        if cfg.mating != "none":
            hb.append(f"Mating status: {cfg.mating}: " + (f"a steady {social.SAG_HZ:g} Hz drive on her SAG neurons (AN_SMP_2), provisional"
                                                          if cfg.mating == "virgin" else "her SAG neurons (AN_SMP_2) silenced") + ".")
        if cfg.touch:
            hb.append("Touch: bumping into the other fly reaches the head bristles as a wall bump does (it needs the collide "
                      "channel, which is switched on with it).")
        if cfg.cues and any(f.sex == "female" for f in self.game.flies):
            hb.append("Her cues: the marks drawn at the tip of a female's abdomen are readout displays of her DNp37 (vaginal "
                      "plate opening command) and DNp13 (ovipositor extrusion command) rates, thresholds provisional; "
                      "hand-built, and never a verdict on the male (docs/TWO_FLIES_PLAN.md 3.2).")
        if any(f.sex == "female" for f in self.game.flies):
            hb.append("Her walking is the hand-built walking urge; her brain changes it only through the decoder's thresholds "
                      "(her walking descending neurons stay near 0 Hz in this data: docs/TWO_FLIES_PLAN.md D5).")
        out["not_modelled"].append("Between the two flies: female song, the male pheromone 7-T, kicking and wing flicking, "
                                   "copulation; her decision neurons (DNp37, DNp13) are shown as readouts and never read as a verdict.")

    def _whats_real_one(self) -> dict:
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
                # the 3-D view (docs/TWO_FLIES_PLAN.md 7.3, decision 19): what it shows is an animation of the drawn body's
                # poses, said on its badge and here; the four lines hold whatever the body is, the view is part of the page
                "When the 3-D view is on, the legs are an animation, not physics: the gait phase the drawn body walks with picks "
                "a pose out of a gait atlas (NeuroMechFly's recorded stride through the kit's own leg controller); the wings, the "
                "abdomen and the proboscis are hand-built rotations of the meshes about hinges; a jump lifts the body by a hand-built amount.",
                "When the 3-D view is on, both flies, the male included, wear NeuroMechFly's body, a model built from a female fly "
                "(micro-CT), decimated for the browser; its colours are chosen by hand (red eyes, a tan body with a banded abdomen, "
                "clear wings, darker legs; the female a shade lighter): nothing in the data says what colour a part is.",
                "When the 3-D view is on, the legs replay NeuroMechFly's recorded stride (flygym 1.2.1), never this fly's own leg "
                "commands: the brain's descending neurons move the drawn body, and the meshes follow it.",
                "When the 3-D view is on, the flies are drawn at the drawn scale, about three times real size (as the 2-D dish and "
                "the senses; docs/TWO_FLIES_PLAN.md decision 19); two physics flies in one world are drawn at real size, as replays will be.",
                # the physics pair (docs/TWO_FLIES_PLAN.md 8.3-8.5, decisions 22-24; physics_pair.py)
                *(["Both flies are NeuroMechFly bodies in ONE MuJoCo world (physics_pair.py): the contact between them is physical, "
                   "through explicit contact pairs (each fly's head and forelegs against the other's body, and body to body; which "
                   "parts may touch is a hand-built choice, the last tarsal segment never, since it carries the adhesion), and the "
                   "tap that arouses pC1 is MuJoCo's contact force on those parts, not the drawn 3.4 mm rule. The other fly is "
                   "seen at real size (a 0.7 mm cylinder 1.1 mm tall, from the model's measured 2.8 x 1.0 x 1.1 mm), and both "
                   "flies are drawn at real size, about a third of the drawn fly.",
                   "The male wears NeuroMechFly's body model, built from a female fly (micro-CT): flygym 1.2.1 has no male body, "
                   "and scaling this one would be hand-built."] if hasattr(self.body, "pair") else []),
            ],
            "not_modelled": [
                "Real neuron shapes and individual properties, hormones, electrical synapses, most neuromodulation, development.",
                "Genes beyond two transcription factors' expression labels, the transmitter each neuron makes and (with the parts list on) the aminergic receptors its cell type expresses where an adult scRNA-seq cluster exists. Still no ion-channel differences, no peptide signalling (the ontology only names the peptidergic types), no development from the genome.",
                "Absolute firing rates shouldn't be trusted, only which neurons respond. Nothing here is conscious.",
                *(["The escape jump with the physics body: the giant fibre fires, but the fly stays on its feet (NeuroMechFly "
                   "has no jump model)."] if getattr(self, "body_kind", "drawn") == "physics" else []),
            ],
        }
