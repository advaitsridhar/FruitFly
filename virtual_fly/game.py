"""
The sensorimotor loop: arena -> senses -> connectome -> descending neurons -> body -> arena.

    arena --(senses, hand-built)--> sensory neurons --(the connectome)--> descending neurons
      ^                                                                          |
      +---------------(body physics, hand-built)<--(decoder, hand-built)--------+

The middle arrow is the only part nobody designed: which descending neurons light up for a given
input comes purely from the wiring diagram, plus what the mushroom body has learned. Everything
hand-built is labelled as such, here and on screen (see :meth:`Game.whats_real`).

Compared with the starter kit this loop adds: computed vision (a retina), smell with turbulent
plumes and associative learning, wind and hearing, a second fly and courtship, internal state
(hunger, thirst), an event log, scripted scenarios (conditioning protocols), recording, and a
data-driven decoder whose DN-to-motor-pool map is measured from the connectome at start-up.
"""

from __future__ import annotations

import json
import math
import os
import queue
import random
import threading
import time
from dataclasses import dataclass, field

import numpy as np

from .body import FlyBody, JUMP_TIME
from .brain import FlyBrain
from .plasticity import APPROACH_NTS, AVOID_NTS
from . import genetics
from . import parts as partslib
from . import retest as retestlib
from . import vfb
from . import wiring
from .experiments import survival as survival_report
from .scenarios import SCENARIOS, ScenarioRunner
from .senses.mechano import Antennae, Bristles
from .senses.olfaction import ODOURS, Nose
from .senses.taste import Forelegs, Mouth
from .senses.vision import Retina
from .world import ARENA_R, FLY_HALF, World, wrap

TICK_MS = 25.0          # brain time simulated per world update

# ----------------------------------------------------------------------------------------------
# Readouts: neurons we listen to (rates in Hz each tick). Sent to the browser as the panel layout.
# ----------------------------------------------------------------------------------------------
READOUTS = [
    # key, spec, label, group, bar max, colour
    ("G2N-1", "GNG232", "sugar relay", "Taste → feeding", 120, "#f2b134"),
    ("Scapula", "GNG087", "bitter relay", "Taste → feeding", 300, "#a77bff"),
    ("MN9", "MN9", "proboscis", "Taste → feeding", 100, "#f2b134"),
    ("LC4", "LC4", "looming", "Eyes → escape", 200, "#ff8a8a"),
    ("LPLC2", "LPLC2", "looming", "Eyes → escape", 200, "#ff8a8a"),
    ("GF", "DNp01", "giant fibre", "Eyes → escape", 350, "#ff5d5d"),
    ("LC10aL", "LC10a/L", "small object, left eye", "Eyes → steering", 100, "#4de2c5"),
    ("LC10aR", "LC10a/R", "small object, right eye", "Eyes → steering", 100, "#4de2c5"),
    ("DNa02L", "DNa02/L", "turn left", "Eyes → steering", 120, "#4de2c5"),
    ("DNa02R", "DNa02/R", "turn right", "Eyes → steering", 120, "#4de2c5"),
    ("HS", "HSE,HSN,HSS", "wide-field motion", "Eyes → steering", 100, "#7fd0b8"),
    ("ORN", "class:olfactory", "receptor neurons", "Nose → learning", 60, "#9be15d"),
    ("KC", "class:Kenyon_Cell", "Kenyon cells", "Nose → learning", 5, "#9be15d"),
    ("MBONapp", "MBON11,MBON12,MBON14,MBON09", "approach MBONs (KC-driven)", "Nose → learning", 40, "#9be15d"),
    ("MBONav", "MBON01,MBON02,MBON05,MBON06,MBON07", "avoidance MBONs (KC-driven)", "Nose → learning", 40, "#ff9f6b"),
    ("PPL1", "prefix:PPL1", "punishment dopamine", "Nose → learning", 100, "#ff5d8f"),
    ("PAM", "prefix:PAM", "reward dopamine", "Nose → learning", 60, "#ffc857"),
    ("pC1", "prefix:pC1_", "courtship command", "Courtship", 80, "#ff9ad5"),
    ("pIP10", "pIP10", "song neuron", "Courtship", 150, "#ff9ad5"),
    ("JOA", "prefix:JO-A,prefix:JO-B", "hearing", "Courtship", 100, "#ff9ad5"),
    ("WindL", "prefix:JO-C/L,prefix:JO-E/L", "wind, left antenna", "Wind and touch", 110, "#8fb8ff"),
    ("WindR", "prefix:JO-C/R,prefix:JO-E/R", "wind, right antenna", "Wind and touch", 110, "#8fb8ff"),
    ("MDN", "MDN", "walk backward", "Body", 100, "#ffa24d"),
    ("DNp09", "DNp09", "walk forward", "Body", 100, "#8b98a9"),
    ("BDN2", "DNg100", "walk forward", "Body", 100, "#8b98a9"),
    ("web", "DNg74_a,DNg74_b", "slow down", "Body", 200, "#8b98a9"),
    ("aDN1", "DNg62", "groom", "Body", 200, "#7cc6ff"),
    ("aDN2", "DNge078", "groom", "Body", 200, "#7cc6ff"),
    ("TTMn", "TTMn", "jump muscle", "Body", 100, "#ff5d5d"),
]
HIDDEN_READOUTS = {   # used by the decoder but not shown as bars
    "DNg13L": "DNg13/L", "DNg13R": "DNg13/R", "DNa01L": "DNa01/L", "DNa01R": "DNa01/R",
    "BDN1": "DNge053", "BDN4": "DNge050", "oDN1": "DNg97", "bluebell": "DNg60", "Brake": "AN19A018",
    "DNa03L": "DNa03/L", "DNa03R": "DNa03/R", "DNp15L": "DNp15/L", "DNp15R": "DNp15/R",
}
BUILTIN_KEYS = {r[0] for r in READOUTS} | set(HIDDEN_READOUTS)   # a 'watch' may never shadow these

# The antennal-lobe local neurons are silenced in the game (outputs blocked, like tetanus toxin).
BASELINE_SILENCED = "class:ALLN"

# Tarsal taste neurons fired when the male taps the female (the leg chemosensory types with any route
# to pC1 in this data). In this model that route is far too weak to fire pC1 (LgLG1a at 400 Hz gives
# pC1 0.2 Hz; it would need ~8x more input gain), so contact ALSO drives the pC1 courtship neurons
# directly (COURTSHIP_DRIVE, hand-built). Everything downstream of pC1 (pIP10, wing motor neurons,
# DNp13) is wiring. See docs/SCIENCE.md.
PHEROMONE_GRNS = {"LgLG1a,LgLG1b": 60.0}
COURTSHIP_SPEC, COURTSHIP_HZ, COURTSHIP_SECS = "prefix:pC1_", 60.0, 2.5
# A sound (a clap) vibrates the antennae: Johnston's organ A/B neurons. Through the wiring they reach
# the giant fibre, so a loud sound can make the fly jump (as real flies do).
SOUND_SPEC, SOUND_HZ = "prefix:JO-B", 100.0        # JO-B: the louder-sound channel; JO-A+B together reach the GF less

# Reward: sugar does not reach the PAM dopamine neurons in this model (docs/SCIENCE.md 4.5: the wiring's sugar
# routes end on PAM-a1 and g5, whose best-connected cells get about a third of the depolarisation they need, and
# the known sweet-taste route runs through octopamine, which the model cannot turn into firing), so eating sugar
# drives them directly. Hand-built. PAM12 (PAM-g3) is left out: sugar suppresses its ongoing activity and
# activating it teaches aversion (Yamagata et al. 2016, doi:10.1371/journal.pbio.1002586). The neuron lab's
# "prefix:PAM" zap stays the en-masse activation, which in real flies also teaches reward.
REWARD_SPEC, REWARD_HZ = "prefix:PAM,!PAM12", 40.0
# Punishment: bitter taste reaches the PPL1 dopamine neurons through the wiring, no injection needed.
# The "shock" tool (the classic conditioning stimulus, an electric shock) drives PPL1 directly. Hand-built.
SHOCK_SPEC, SHOCK_HZ = "prefix:PPL1", 80.0

ZAP_PRESETS = [
    ("MDN", 60, "MDN: the 'moonwalker' command neurons"),
    ("DNp01", 150, "Giant fibre: the escape command"),
    ("DNa02/L", 80, "DNa02 left: steering"),
    ("DNa02/R", 80, "DNa02 right: steering"),
    ("MN9", 80, "MN9: proboscis motor neuron"),
    ("LC10a/L", 60, "LC10a left: 'small moving thing on my left'"),
    ("prefix:BM_InOm", 100, "Head bristles: 'something touched my head'"),
    ("prefix:pC1_", 80, "pC1: courtship command neurons"),
    ("pIP10", 60, "pIP10: song"),
    ("prefix:PPL1", 80, "PPL1: punishment dopamine (pairs with what it smells now)"),
    ("prefix:PAM", 60, "PAM: reward dopamine"),
    ("prefix:JO-A,prefix:JO-B", 100, "Johnston's organ: a loud sound (reaches the giant fibre)"),
    ("T4a/R,T5a/R", 60, "T4a/T5a right: front-to-back motion on the right eye (optomotor)"),
]

CHECKS = [
    ("feed", "Feed it: drop sugar in its path → MN9 fires, the proboscis comes out"),
    ("bitter", "Offer bitter food → the bitter pathway keeps MN9 silent"),
    ("lure", "Lure it: wiggle the decoy beside it → LC10a → DNa02 → it turns"),
    ("escape", "Scare it: swoop the hand at it fast → retina → LC4/LPLC2 → giant fibre → jump"),
    ("groom", "Dust it → aDN1/aDN2 → it grooms"),
    ("wall", "Watch it bump a wall → head bristles → MDN → it backs up"),
    ("smell", "Drop an odour → its receptor neurons → a sparse Kenyon-cell code"),
    ("learn", "Pair an odour with sugar or bitter → the KC→MBON synapses change"),
    ("court", "Add a female → it chases her (LC10a) and taps her → pC1 → pIP10 → song"),
    ("wind", "Turn the wind on → Johnston's organ senses it from the side it comes from"),
    ("sound", "Clap → Johnston's organ → the giant fibre: a startle jump"),
    ("optomotor", "Paint stripes on the wall and spin them → T4/T5 → HS → the fly turns with them"),
    ("moonwalk", "Zap MDN → it walks backward"),
    ("silence", "Silence MN9, then offer sugar: it can taste, but can't eat"),
    ("genetics", "Silence the fruitless neurons (Genetics), then add a female: no chase, no song, as in fruitless mutants"),
    ("genome", "Grow a fly from its wiring rules (Genome): same neurons, new wiring; see which reflexes survive"),
    ("parts", "Switch the parts list on (Genome): modulators become slow tones, the optic lobe's graded cells transmit below threshold; watch the tones"),
]


def n01(r: float, top: float) -> float:
    return min(1.0, max(0.0, r) / top)


@dataclass
class InternalState:
    """Hand-built drives. Hunger rises with time and falls when the fly eats; it sharpens the sugar
    sense (as dopamine/NPF do in real flies) and makes the fly explore more. Thirst likewise for
    water. Arousal rises with courtship activity."""
    hunger: float = 0.7
    thirst: float = 0.3
    arousal: float = 0.0

    def step(self, dt: float, eating: float, drinking: float, courting: float):
        self.hunger = max(0.0, min(1.0, self.hunger + dt / 240.0 - eating * dt / 12.0))
        self.thirst = max(0.0, min(1.0, self.thirst + dt / 600.0 - drinking * dt / 10.0))
        self.arousal = max(0.0, min(1.0, self.arousal + courting * dt / 4.0 - dt / 30.0))

    def to_dict(self):
        return {"hunger": round(self.hunger, 3), "thirst": round(self.thirst, 3), "arousal": round(self.arousal, 3)}


class EventLog:
    def __init__(self, cap: int = 200):
        self.items: list[dict] = []
        self.cap = cap
        self.seq = 0

    def add(self, t: float, kind: str, text: str, **extra):
        self.seq += 1
        self.items.append({"id": self.seq, "t": round(t, 2), "kind": kind, "text": text, **extra})
        if len(self.items) > self.cap:
            del self.items[: len(self.items) - self.cap]

    def since(self, last_id: int) -> list[dict]:
        return [e for e in self.items if e["id"] > last_id]


class MotorDecoder:
    """Descending-neuron rates -> motor drives (hand-built weights on real DN types).

    At start-up it also measures, from the wiring, which leg/wing/abdominal motor pools each of its
    DNs reaches (``self.dn_targets``), so the on-screen explanation can say *why* a DN is read as
    "walk forward" or "turn". The weights themselves are chosen by hand (see docs/SCIENCE.md).
    """

    TAUS = dict(forward=0.15, yaw=0.10, backward=0.15, halt=0.25, feed=0.12, groom=0.30, song=0.25, court=0.4)

    def __init__(self, conn):
        self.conn = conn
        self.m = dict(forward=0.0, yaw=0.0, backward=0.0, halt=0.0, feed=0.0, groom=0.0, song=0.0, court=0.0)
        self.dn_targets = self._measure_targets()

    def _measure_targets(self) -> dict:
        """Synapses from each decoder DN onto motor neurons, by neuromere (T1-T3 legs, wing/neck)."""
        conn = self.conn
        out = {}
        motor = conn.select("superclass:vnc_motor,superclass:cb_motor")
        motor_mask = np.zeros(conn.n, dtype=bool)
        motor_mask[motor] = True
        for spec in ("DNp09", "DNg100", "DNge053", "DNge050", "DNg97", "DNa02", "DNg13", "DNa01", "MDN",
                     "DNg60", "DNg74_a,DNg74_b", "AN19A018", "DNp01", "DNg62", "DNge078", "pIP10", "MN9"):
            idx = conn.select(spec)
            if idx.size == 0:
                continue
            # two hops: DN -> premotor -> motor (direct DN->motor synapses are rare)
            e1 = conn.out_edges(idx)
            post1 = conn.post_idx[e1]
            direct = int(conn.n_syn[e1][motor_mask[post1]].sum())
            inter = np.unique(post1[conn.sign[post1] > 0])
            e2 = conn.out_edges(inter[:4000])
            post2 = conn.post_idx[e2]
            hit = motor_mask[post2]
            by_neuromere = {}
            for nm, syn in zip(conn.neuromere[post2[hit]], conn.n_syn[e2][hit]):
                by_neuromere[nm or "?"] = by_neuromere.get(nm or "?", 0) + int(syn)
            out[spec] = {"direct_motor_synapses": direct, "two_hop_motor_synapses_by_neuromere":
                         dict(sorted(by_neuromere.items(), key=lambda kv: -kv[1])[:6])}
        return out

    def decode(self, hz: dict[str, float], dt: float) -> dict[str, float]:
        raw = dict(
            forward=0.3 * n01(hz["DNp09"], 50) + 0.25 * n01(hz["BDN2"], 50) + 0.15 * n01(hz["BDN1"], 50)
                    + 0.15 * n01(hz["BDN4"], 50) + 0.15 * n01(hz["oDN1"], 50),
            yaw=max(-1.0, min(1.0, (hz["DNa02R"] - hz["DNa02L"]) / 50 + 0.5 * (hz["DNg13R"] - hz["DNg13L"]) / 100
                               + 0.25 * (hz["DNa01R"] - hz["DNa01L"]) / 50 + 0.2 * (hz["DNa03R"] - hz["DNa03L"]) / 50
                               + 0.35 * (hz["DNp15R"] - hz["DNp15L"]) / 60)),     # DNp15: HS-driven optomotor steering
            backward=n01(hz["MDN"], 60),
            halt=min(1.0, 0.6 * n01(hz["bluebell"], 50) + 0.4 * n01(hz["web"], 50) + n01(hz["Brake"], 50)),
            feed=n01(hz["MN9"], 50),
            groom=n01(max(hz["aDN1"], hz["aDN2"]), 100),
            song=n01(hz["pIP10"], 40),
            court=n01(hz["pC1"], 30),
        )
        for k, v in raw.items():                  # smooth, like muscles would
            self.m[k] += (v - self.m[k]) * min(1.0, dt / self.TAUS[k])
        return self.m


class Game:
    def __init__(self, brain: FlyBrain, autopilot: bool = True, seed: int = 0, columnar: bool = True,
                 profile_name: str = "game", brain_factory=None, parts_list=None, brain_kwargs: dict | None = None,
                 retest: str = "auto", body: str = "drawn", stride_average: bool = False):
        self.brain, self.conn = brain, brain.conn
        self.profile_name = profile_name
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
        self.rng = random.Random(seed)
        self.seed = seed
        self.actions: queue.Queue = queue.Queue()
        self.autopilot = autopilot
        self.paused = False
        self.speed = 1.0
        self.world = World(seed)
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
        self.forelegs = Forelegs(self.world, PHEROMONE_GRNS)
        self.antennae = Antennae(self.world)
        self.bristles = Bristles(self.world)
        self.decoder = MotorDecoder(self.conn)
        self.state = InternalState()
        self.events = EventLog()
        self.scenario = ScenarioRunner(self)
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
            self.brain.add_monitor(r[0], r[1], bin_ms=TICK_MS)
        self.user_silenced: set[str] = set()
        self.user_modulated: dict[str, float] = {}
        self.has_soma = ~np.isnan(self.conn.soma[:, 0])
        # build the input index and the cell-type graph now (a second or two), so that the first
        # pathway trace or neuron lookup from the browser does not stall behind the game loop
        self.conn.col_ptr
        self.conn.type_graph()
        self.layout_json = self._make_layout()
        self.state_json = b"{}"
        self.state_dict: dict = {}
        self.seq = 0
        self.rtf = 1.0
        self.recording: list | None = None
        self.record_active = False
        self.record_spikes = False
        self.learning_on = brain.plasticity is not None
        self.lock = threading.RLock()
        self.subscribers: list[queue.Queue] = []
        self.reset_world(first=True)

    # ------------------------------------------------------------------ world
    def reset_world(self, first: bool = False):
        self.brain.reset()
        self.world.clear("all")
        self.world.hand = None
        self.world.set_stripes(0, 0.0)
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
        self.since_input = 0.0
        self.calms = 0
        self.message, self.message_left = "", 0.0
        self.done: set[str] = set()
        self.hz_shown = {k: 0.0 for k in self.readouts}
        self.t = 0.0
        self.senses_now: dict = {}
        self.driver = ""
        self.gf_cooldown = 0.0
        self.turn_command = 0.0            # last tick's commanded yaw (efference copy for the eyes)
        self.court_left = 0.0              # seconds of pC1 drive left after tapping the female
        self.sound_left = 0.0
        self.bitter_t = 0.0
        self.eating = 0.0
        self.drinking = 0.0
        self.song_side = 1.0
        self.shock_left = 0.0
        self.learned_bias = 0.0
        self.smelling: str | None = None
        self.state = InternalState()
        self.decoder.m = {k: 0.0 for k in self.decoder.m}
        self.scenario.stop(silent=True)
        if not first:
            self.events.add(self.t, "system", "New fly, fresh brain (learned synapses kept; use 'forget' to reset them).")

    # ------------------------------------------------------------------ the genome: growing a fly
    _BRAIN_KWARGS = ("dt", "gain", "kenyon_gain", "fatigue_mv", "fatigue_ms", "std_u", "std_tau_ms",
                     "noise_hz", "noise_mv", "noise_spec", "threshold_jitter", "seed", "backend")

    def _default_brain_factory(self, conn, **extra):
        from .settings import build_brain
        kw = {k: v for k, v in self.brain.settings().items() if k in self._BRAIN_KWARGS}
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
        if self.brain.parts is not None:
            return self.brain.parts.counts
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
            brain2 = self.brain_factory(conn2, parts=self.parts_arg(self.parts_on))
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
            brain2 = self.brain_factory(self.conn, parts=self.parts_arg(on))
            self.actions.put({"type": "_swap_brain", "brain": brain2, "conn": self.conn, "level": self.genome["level"],
                              "seed": self.genome["seed"], "rules": self.genome["rules"], "wiring": self.genome["wiring"],
                              "parts": on, "reason": "parts"})
        except Exception as e:
            self.genome.update(growing=None, error=str(e))
            print("error rebuilding the brain", repr(e))

    def _swap_brain(self, a: dict):
        old = self.brain
        with old.lock:
            self.brain, self.conn = a["brain"], a["conn"]
            for r in READOUTS:
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
        level, seed = a["level"], a["seed"]
        self.parts_on = bool(a.get("parts", self.parts_on))
        self.genome.update(level=level, seed=seed, growing=None, wiring=a["wiring"], rules=a["rules"], error=None,
                           survival={"running": True, "results": []})
        if a.get("reason") == "parts":
            if self.parts_on:
                c = self.brain.parts.counts
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
        kw = {k: v for k, v in self.brain.settings().items() if k in self._BRAIN_KWARGS}
        kw.update(self._brain_kwargs)
        kw["parts"] = self.parts_arg(self.parts_on)
        wiring_ = None
        if conn is not self.real_conn:                   # a grown fly: send its wiring, the neurons are the same
            wiring_ = {"row_ptr": conn.row_ptr, "post_idx": conn.post_idx, "n_syn": conn.n_syn, "label": "grown"}
        return {"path": str(path), "wiring": wiring_, "profile": self.profile_name, "brain_kwargs": kw}

    def _survival_key(self, conn) -> tuple:
        """Everything a survival report depends on. Each seed resets the brain and starts its own random
        generator, so the same wiring, profile, brain settings and parts list always give the same rows."""
        kw = {k: v for k, v in self.brain.settings().items() if k in self._BRAIN_KWARGS}
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
                    rows = handle.wait(progress)
                finally:
                    with self._retest_lock:                  # done: let go of its queue (no leaked semaphores at exit)
                        if self._retest_handle is handle:
                            self._retest_handle = None
                where = "process"
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

    def genome_status(self) -> dict:
        g = dict(self.genome)
        if g["growing"]:
            g["growing"] = {**g["growing"], "secs": round(time.time() - g["growing"]["t0"], 1)}
        g["parts"] = {"on": self.parts_on, "status": self.brain.parts_status()}
        return g

    def say(self, text: str, secs: float = 3.0):
        self.message, self.message_left = text, secs

    # ------------------------------------------------------------------ input from the browser
    def action(self, a: dict) -> dict:
        kind = a.get("type")
        if kind == "hand":                                   # pointer moves: just remember it
            self.world.hand = (float(a["x"]), float(a["y"]))
            return {"ok": True}
        if kind == "hand_off":
            self.world.hand = None
            return {"ok": True}
        if kind in ("zap", "silence", "modulate", "watch"):
            spec = str(a.get("spec", "")).strip()
            try:
                n = len(self.conn.select(spec))
            except (ValueError, KeyError) as e:
                return {"ok": False, "error": str(e)}
            if n == 0:
                return {"ok": False, "error": f"No neurons match '{spec}'. Try a type like MDN or prefix:LC10."}
            a["n"] = n
            a["spec"] = spec
        if kind == "watch":
            explicit = str(a.get("key") or "").strip()
            key = (explicit or a["spec"])[:24]
            if key in BUILTIN_KEYS:                          # the decoder reads these; a rewire would steer the body
                if explicit:
                    return {"ok": False, "error": f"'{key}' is a built-in readout; pick another name for the watch."}
                key = f"watch:{a['spec']}"[:24]
            a["key"] = key
        if kind == "unwatch" and a.get("key") not in self.custom_readouts:
            return {"ok": False, "error": f"'{a.get('key')}' is not a custom watch."}
        if kind == "scenario" and a.get("id") and a["id"] not in SCENARIOS:
            return {"ok": False, "error": f"unknown scenario {a['id']}"}
        if kind == "grow":
            level = str(a.get("level", "type")).strip().lower()
            if not wiring.valid_level(level):
                return {"ok": False, "error": "level must be real, type, class or bottleneck:K (K = 1 to 2048)"}
            if self.genome["growing"]:
                return {"ok": False, "error": "a fly is already being grown; wait for it"}
            try:
                a["level"], a["seed"] = level, int(a.get("seed", 1))
            except (TypeError, ValueError):
                return {"ok": False, "error": "seed must be a whole number"}
            self.genome.update(growing={"level": level, "seed": a["seed"], "t0": time.time()}, error=None)   # claimed now, one at a time
        if kind == "parts":
            if not isinstance(a.get("on"), bool):
                return {"ok": False, "error": "'on' must be true or false"}
            if self.genome["growing"]:
                return {"ok": False, "error": "the brain is being rebuilt; wait for it"}
            if a["on"] == self.parts_on:
                return {"ok": False, "error": f"the parts list is already {'on' if a['on'] else 'off'}"}
            self.genome.update(growing={"level": self.genome["level"], "seed": self.genome["seed"], "reason": "parts",
                                        "parts": a["on"], "t0": time.time()}, error=None)                 # claimed now
        if kind == "_swap_brain":
            return {"ok": False, "error": "internal"}
        self.actions.put(a)
        return {"ok": True, **({"n": a["n"]} if "n" in a else {})}

    def _apply_actions(self):
        while not self.actions.empty():
            a = self.actions.get()
            try:
                self._apply(a)
            except Exception as e:                 # a bad action must not kill the loop
                self.say(f"Error: {e}", 4.0)
                print("error in action", a, repr(e))

    def _apply(self, a: dict):
        kind = a.get("type")
        w = self.world
        if kind == "tool":
            w.tool = a.get("tool", "lure")
        elif kind == "drop":
            what = a.get("kind")
            x, y = float(a["x"]), float(a["y"])
            if what in ("sugar", "bitter", "water"):
                f = w.add_food(what, x, y)
                if f:
                    self.events.add(self.t, "world", f"{what} dropped at ({x:.0f}, {y:.0f})")
            elif what in ODOURS:
                src = w.add_odour(what, x, y, food=a.get("food"))
                if src:
                    self.events.add(self.t, "world", f"odour '{ODOURS[what].name}' placed" +
                                    (f" with {a['food']}" if a.get("food") else ""))
            elif what == "post":
                w.add_obstacle(x, y, float(a.get("r", 4.0)))
        elif kind == "remove":
            w.remove(int(a["id"]))
        elif kind == "dust":
            x, y = float(a.get("x", self.body.pose.x)), float(a.get("y", self.body.pose.y))
            if math.hypot(x - self.body.pose.x, y - self.body.pose.y) < 20:
                self.antennae.puff_dust(1.5)
                self.say("Dust on the antennae!", 1.5)
        elif kind == "sound":
            self.sound_left = float(a.get("secs", 0.3))
            self.events.add(self.t, "world", "a loud sound (Johnston's organ A/B neurons)")
            self.say("Clap! Johnston's organ hears it.", 1.5)
        elif kind == "shock":
            self.shock_left = float(a.get("secs", 1.0))
            self.events.add(self.t, "learning", "Electric shock: PPL1 punishment dopamine driven directly (hand-built)")
            self.say("Shock! Punishment dopamine pairs with whatever it smells now.", 2.5)
        elif kind == "zap":
            self.zaps = [z for z in self.zaps if z[0] != a["spec"]]
            self.zaps.append([a["spec"], float(a.get("hz", 60)), float(a.get("secs", 2.0))])
            self.done.add("zap")
            self.say(f"Zapping {a['spec']} ({a['n']} neurons) at {float(a.get('hz', 60)):g} Hz", 2.0)
            self.events.add(self.t, "lab", f"zap {a['spec']} at {float(a.get('hz', 60)):g} Hz for {float(a.get('secs', 2.0)):g} s")
        elif kind == "silence":
            with self.brain.lock:
                n = self.brain.silence(a["spec"])
            self.user_silenced.add(a["spec"])
            self.done.add("silence")
            if a["spec"].startswith(("gene:", "dimorphism:")):
                self.done.add("genetics")
            self.say(f"Silenced {a['spec']} ({n} neurons): they still fire, but nothing hears them.", 3.0)
            self.events.add(self.t, "lab", f"silenced {a['spec']} ({n} neurons)")
        elif kind == "unsilence":
            with self.brain.lock:
                for spec in list(self.user_silenced):
                    if a.get("spec") in (None, "", spec):
                        self.brain.unsilence(spec)
                        self.user_silenced.discard(spec)
            self.say("Silencing removed.", 2.0)
        elif kind == "modulate":
            factor = float(a.get("factor", 1.0))
            with self.brain.lock:
                n = self.brain.modulate(a["spec"], factor)
            if abs(factor - 1.0) < 1e-6:
                self.user_modulated.pop(a["spec"], None)
            else:
                self.user_modulated[a["spec"]] = factor
            self.say(f"Output of {a['spec']} ({n} neurons) scaled x{factor:g}.", 3.0)
            self.events.add(self.t, "lab", f"modulate {a['spec']} x{factor:g}")
        elif kind == "watch":
            key = a["key"]                                   # validated in action()
            if key in BUILTIN_KEYS:
                raise ValueError(f"'{key}' is a built-in readout")
            if key in self.custom_readouts:
                self.brain.remove_monitor(key)
            self.custom_readouts[key] = a["spec"]
            self.readouts[key] = self.conn.select(a["spec"])
            self.hz_shown[key] = 0.0
            self.brain.add_monitor(key, a["spec"], bin_ms=TICK_MS)
        elif kind == "unwatch":
            key = a.get("key")
            if key in self.custom_readouts:
                del self.custom_readouts[key]
                self.readouts.pop(key, None)
                self.hz_shown.pop(key, None)
                self.brain.remove_monitor(key)
        elif kind == "clear":
            w.clear(a.get("what", "all"))
        elif kind == "reset":
            self.reset_world()
            self.say("New fly, fresh brain.", 2.0)
        elif kind == "autopilot":
            self.autopilot = bool(a.get("on", True))
        elif kind == "pause":
            self.paused = bool(a.get("on", False))
        elif kind == "speed":
            self.speed = max(0.1, min(3.0, float(a.get("value", 1.0))))
        elif kind == "calm":
            with self.brain.lock:
                self.brain.reset()
            self.say("Brain reset to rest.", 2.0)
        elif kind == "wind":
            w.set_wind(a.get("angle"), a.get("speed"))
            if w.wind.speed > 0:
                self.done.add("wind")
            self.events.add(self.t, "world", f"wind {w.wind.speed:.0f} mm/s toward {math.degrees(w.wind.angle):.0f}°")
        elif kind == "stripes":
            w.set_stripes(a.get("count"), a.get("drum_speed"))
            self.events.add(self.t, "world", f"wall stripes: {w.stripes}, drum {w.drum_speed:+.2f} rad/s")
        elif kind == "female":
            w.toggle_female(bool(a.get("on", True)), a.get("x"), a.get("y"))
            self.events.add(self.t, "world", "a female fly enters" if w.female else "the female leaves")
        elif kind == "learning":
            if self.brain.plasticity is not None:
                if a.get("forget"):
                    with self.brain.lock:
                        self.brain.plasticity.reset_weights()
                    self.events.add(self.t, "learning", "all KC→MBON synapses reset to their original strength")
                    self.say("Memories erased.", 2.0)
                if "on" in a:
                    self.learning_on = bool(a["on"])
                    self.brain.plasticity.enabled = self.learning_on
        elif kind == "scenario":
            if a.get("id"):
                self.scenario.start(a["id"])
            else:
                self.scenario.stop()
        elif kind == "record":
            if a.get("on", True):
                if self.brain.recording is not None:     # a restart while spikes were being recorded
                    self.brain.stop_recording()
                self.brain.recording_kept = []           # the previous take is gone once a new one starts
                self.recording = []
                self.record_active = True
                self.record_spikes = bool(a.get("spikes", False))
                if self.record_spikes:
                    self.brain.start_recording()
                self.events.add(self.t, "system", "recording started")
            elif self.record_active:
                self.record_active = False               # frames are kept for download until the next start
                if self.brain.recording is not None:
                    self.brain.recording_kept = self.brain.stop_recording()
                self.events.add(self.t, "system", f"recording stopped ({len(self.recording or [])} frames)")
        elif kind == "grow":
            self._start_grow(a["level"], a["seed"])
        elif kind == "parts":
            self._start_rebuild(a["on"])
        elif kind == "_swap_brain":
            self._swap_brain(a)
        elif kind == "state":
            for k in ("hunger", "thirst"):
                if k in a:
                    setattr(self.state, k, max(0.0, min(1.0, float(a[k]))))
        elif kind == "place_fly":
            self.body.reset(float(a["x"]), float(a["y"]), float(a.get("h", self.body.pose.h)))

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

    # ------------------------------------------------------------------ behaviour selection (hand-built)
    def choose_mode(self, m: dict, gf_spikes: int) -> str:
        if self.body.pose.jump is not None:
            return "escape"
        # the giant fibre must fire at least twice in this tick (>= 40 Hz across the pair): a lone spike
        # from a weak, indirect route (e.g. local motion through T4/T5 -> LPLC2) is not an attack
        if gf_spikes >= 2 and self.body.jump_lock <= 0 and self.gf_cooldown <= 0:
            away = self.world.hand if (self.world.hand is not None and self.world.tool == "hand") else None
            self.body.start_jump(away)
            self.gf_cooldown = 1.0
            if "loom" in self.senses_now:
                self.done.add("escape")
            self.say("Giant fibre fired: escape jump!", 1.5)
            self.events.add(self.t, "behaviour", "escape jump (giant fibre DNp01 spiked)")
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
        learned = 0.0
        pl = self.brain.plasticity
        if pl is not None:
            trace = pl.kc_trace
            act = trace[pl.pe_kc]                       # eligibility of each plastic synapse's KC
            if act.sum() > 0:
                nt = self.conn.nt[pl.mbon][pl.pe_mbon]
                depression = 1.0 - pl.scale
                app = np.isin(nt, APPROACH_NTS)
                av = np.isin(nt, AVOID_NTS)
                wa = float((act * depression)[app].sum() / max(act[app].sum(), 1e-6))
                wv = float((act * depression)[av].sum() / max(act[av].sum(), 1e-6))
                learned = 1.2 * (wv - wa)               # avoid-MBON synapses weakened -> approach more
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

    # ------------------------------------------------------------------ the loop
    def tick(self):
        dt = TICK_MS / 1000.0
        self._apply_actions()
        self.scenario.step(dt)
        rates, col_idx, col_hz = self.senses(dt)
        b = self.brain
        with b.lock:
            if col_idx.size:
                b.set_stimulus_arrays(col_idx, col_hz, extra=rates)
            else:
                b.set_stimuli(rates)
            b.reset_counts()
            chunks = []
            for _ in range(int(round(TICK_MS / b.dt))):
                s = b.step()
                if s.size:
                    chunks.append(s)
            spikes = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.int64)
            counts = b.spike_count
            live = np.where(b.silenced_mask(), 0, counts) if b.silenced else counts   # what reaches the body
        hz = {k: (float(counts[i].sum()) / max(1, i.size) / dt if i.size else 0.0) for k, i in self.readouts.items()}
        motor_hz = {k: (float(live[i].sum()) / max(1, i.size) / dt if i.size else 0.0) for k, i in self.readouts.items()}
        gf = int(live[self.readouts["GF"]].sum())
        m = self.decoder.decode(motor_hz, dt)
        self.gf_cooldown -= dt
        mode = self.choose_mode(m, gf)
        why = []
        wander_yaw = 0.0
        if mode == "escape":
            why.append("brain: giant fibre DNp01 spiked")
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
        self.mode = mode
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
        if mode == "groom":
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
        if self.brain.plasticity is not None and self.brain.plasticity.events and "learn" not in self.done \
                and self.brain.plasticity.depressed_fraction() > 0.002:
            self.done.add("learn")
            self.events.add(self.t, "learning", "KC→MBON synapses depressed: the fly has learned something about this odour")
        courting = 1.0 if mode == "court" else 0.0
        self.state.step(dt, self.eating, self.drinking, courting)
        self.world.step(dt, (self.body.pose.x, self.body.pose.y), fly_singing=m["song"], courted=courting > 0)
        # events for salient sense changes
        self._sense_events()
        self.driver = "  +  ".join(why)
        if not self.driver:
            self.driver = ("nothing: the brain is quiet" if not spikes.size else
                           "the brain is busy, but no movement command is strong enough")
        for k, v in hz.items():                      # smooth the numbers shown on screen
            self.hz_shown[k] = self.hz_shown.get(k, 0.0) + (v - self.hz_shown.get(k, 0.0)) * min(1.0, dt / 0.15)
        self.t += dt
        self.message_left -= dt
        # watchdog: this simple model can lock into runaway firing (mostly the smell centre)
        sps = spikes.size / dt
        self.since_input = 0.0 if rates else self.since_input + dt
        self.runaway_s = self.runaway_s + dt if sps > 150000 else 0.0
        if (self.runaway_s > 1.5 and self.since_input > 0.5) or self.runaway_s > 4.0:
            with b.lock:
                b.reset()
            self.calms += 1
            self.runaway_s = 0.0
            self.say("Runaway firing (a known flaw of this simple model, mostly in the smell centre). Brain calmed.", 4.0)
            self.events.add(self.t, "system", "runaway firing: brain reset to rest")
        if self.recording is not None and self.record_active:
            self.recording.append({"t": round(self.t, 3), "fly": self.body.to_dict(), "mode": mode,
                                   "hz": {k: round(v, 1) for k, v in hz.items()}, "senses": self.senses_now,
                                   "sps": int(sps)})
        self.publish(spikes, self.hz_shown, sps)

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

    # ------------------------------------------------------------------ publishing
    def learning_summary(self) -> dict | None:
        pl = self.brain.plasticity
        if pl is None:
            return None
        summ = pl.summary(self.conn)
        return {"enabled": self.learning_on, "depressed_fraction": round(pl.depressed_fraction(), 4),
                "events": pl.events, "learned_bias": round(self.learned_bias, 3), "smelling": self.smelling,
                "mbon": {t: {"strength": round(v["strength"], 3), "now": round(v["now"], 3), "valence": v["valence"],
                             "dopamine": round(v["dopamine"], 3)} for t, v in summ.items()}}

    def publish(self, spikes, hz, sps):
        vis = spikes[self.has_soma[spikes]]
        if vis.size > 2500:
            vis = np.random.default_rng(self.seq).choice(vis, 2500, replace=False)
        w = self.world.to_dict()
        if len(w["puffs"]) > 300:
            w["puffs"] = w["puffs"][-300:]
        state = {
            "seq": self.seq + 1, "t": round(self.t, 3), "rtf": round(self.rtf, 2), "speed": self.speed,
            "fly": self.body.to_dict(),
            "world": w,
            "autopilot": self.autopilot, "paused": self.paused,
            "senses": self.senses_now,
            "retina": self.retina.images_b64(),
            "hz": {k: round(v, 1) for k, v in hz.items()},
            "motor": {k: round(v, 3) for k, v in self.decoder.m.items()},
            "driver": self.driver, "mode": self.mode,
            "spikes": vis.tolist(), "sps": int(sps), "stims": len(self.brain.stim),
            "calms": self.calms, "msg": self.message if self.message_left > 0 else "",
            "silenced": sorted(self.user_silenced),
            "baseline": sorted(set(self.brain.silenced) - self.user_silenced),
            "modulated": self.user_modulated,
            "custom": self.custom_readouts,
            "done": sorted(self.done),
            "state": self.state.to_dict(),
            "learning": self.learning_summary(),
            "events": self.events.items[-12:],
            "event_seq": self.events.seq,
            "scenario": self.scenario.status(),
            "genome": self.genome_status(),
            "recording": None if self.recording is None else {"frames": len(self.recording), "spikes": self.record_spikes,
                                                              "active": self.record_active},
        }
        self.seq += 1
        self.state_dict = state
        self.state_json = json.dumps(state, separators=(",", ":")).encode()
        for q in list(self.subscribers):
            try:
                q.put_nowait(self.state_json)
            except queue.Full:
                pass

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=4)
        self.subscribers.append(q)
        return q

    def unsubscribe(self, q):
        if q in self.subscribers:
            self.subscribers.remove(q)

    def loop(self):
        while True:
            t0 = time.perf_counter()
            if self.paused:
                self._apply_actions()
                self.publish(np.zeros(0, dtype=np.int64), self.hz_shown, 0)   # keep the page in sync
                time.sleep(0.05)
                continue
            try:
                self.tick()
            except Exception as e:                    # keep the game alive, show the problem
                self.say(f"Error: {e}", 5.0)
                import traceback
                traceback.print_exc()
                time.sleep(0.2)
            used = time.perf_counter() - t0
            budget = TICK_MS / 1000.0 / self.speed
            self.rtf += (min(1.0, budget / max(used, 1e-6)) - self.rtf) * 0.1
            if used < budget:
                time.sleep(budget - used)

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
            "presets": [{"spec": s, "hz": h, "label": l} for s, h, l in ZAP_PRESETS],
            "types": sorted(type_counts, key=lambda t: -type_counts[t])[:5000],
            "edges": int(c.n_edges), "synapses": int(c.n_syn.sum()),
            "dataset": c.dataset, "sex": c.sex,
            "readouts": self.readout_meta,
            "checks": [{"id": i, "text": t} for i, t in CHECKS],
            "odours": [{"id": o.id, "name": o.name, "glomeruli": o.glomeruli, "innate": o.innate, "colour": o.colour,
                        "note": o.note} for o in ODOURS.values()],
            "scenarios": [{"id": k, "name": s.name, "description": s.description} for k, s in SCENARIOS.items()],
            "retina": self.retina.layout(),
            "profile": self.profile_name,
            "genetics": self.genetics,
            "genome": {"levels": [{"level": lv, "label": lb} for lv, lb in wiring.LEVELS],
                       "rules": {"type_groups": None}},
            "parts": {"tables": self.parts_list().describe(), "counts": self.parts_counts()},
            "vfb": vfb.ontology().summary(c),
            "settings": self.brain.settings(),
            "decoder": self.decoder.dn_targets,
            "columnar_vision": self.columnar_on,
            "whats_real": self.whats_real(),
        }, separators=(",", ":")).encode()

    def whats_real(self) -> dict:
        return {
            "wiring": [
                "Sugar taste neurons → MN9, the proboscis motor neuron. Bitter taste keeps MN9 silent, even on top of sugar.",
                "Looming detectors (LC4, LPLC2) → giant fibre DNp01, the escape command.",
                "A small moving object seen on one side (LC10a, a courtship-chase cell type) → DNa02 on that same side → a turn toward it.",
                "Head bristles → MDN, the 'moonwalker' backward-walking neurons. Antennal sensors (Johnston's organ) → aDN1/aDN2 grooming neurons.",
                "Odour receptor neurons → projection neurons → a sparse, odour-specific Kenyon-cell code → mushroom body output neurons.",
                "Bitter taste → PPL1 dopamine neurons (the punishment signal for learning).",
                "Which Kenyon-cell synapses are plastic and which dopamine neurons gate each MBON: read from the wiring (DAN→MBON synapses).",
                "pC1 courtship neurons → pIP10 → wing motor neurons (song), and → DNp13; a female seen as a small moving object → LC10a → DNa02 (the chase).",
                "A loud sound → Johnston's organ A/B neurons → the giant fibre (a startle jump), and wind on the antennae → grooming and backing neurons.",
                "Wide-field motion → T4/T5 (driven column by column from the retina) → HS cells → DNa02 and DNp15 on the same side: the optomotor reflex.",
                "Which neurons express fruitless and doublesex, and which are male-specific or dimorphic: the MaleCNS annotation, read from the data. Silencing the fruitless neurons stops the song (pIP10 and its route to the wing motor neurons are fru+) and leaves feeding and escape alone.",
                "A grown fly (Genome card) keeps the connectome's cell-type wiring rules and nothing else: 9 of the 11 validated reflexes survive on type-level rules, none on class-level rules.",
                "The parts list (Genome card): which neurons make dopamine, octopamine or serotonin is the MaleCNS transmitter prediction; that these act only through slow receptors, and that photoreceptors, L1-L5, the medulla inputs to T4/T5, T4/T5 and HS/VS signal without spikes, is the literature (parts.py cites it).",
                "Which anatomy-ontology class each cell type is (the fbbt: selector, the ontology line in a neuron's popover, the VFB links): the FlyBase anatomy ontology and Virtual Fly Brain's MaleCNS name synonyms, joined offline by name. Where the literature-curated class says a neuron's transmitter differs from the prediction, or fills an 'unclear' one, the parts list follows the literature; the receptors each cell type expresses come from the adult single-cell RNA-seq atlases on VFB and set which way a tone pushes that target.",
            ],
            "hand_built": [
                "The retina (which facet sees what) and the feature computations that turn retinal images into LC4/LPLC2/LC10a/T4/T5 rates.",
                "How smells, wind, touch and dust become firing rates, and which sensory types they drive.",
                ("How descending-neuron firing becomes movement: the decoder's weights, the jump and what wins when commands compete. "
                 "Speeds and turn rates come from leg physics (NeuroMechFly v2 in MuJoCo; its stepping rhythm, recorded steps and "
                 "left/right drive are flygym's)." if getattr(self, "body_kind", "drawn") == "physics" else
                 "How descending-neuron firing becomes movement: speeds, turn rates, the jump, and what wins when commands compete."),
                "The walking urge, hunger and thirst, odour-guided steering (innate valence + the learned KC→MBON bias), the female's behaviour.",
                "Sugar reward → PAM dopamine except PAM-γ3 (the wiring's taste-to-PAM routes give the best-connected PAM-α1 cells about a third of the drive they need; SCIENCE.md 4.5); the 'shock' tool → PPL1.",
                "Courtship arousal: tapping the female fires the tarsal taste neurons (wiring), but their route to pC1 is ~8x too weak in this model, so contact also drives pC1 directly.",
                "Wind on Johnston's organ is kept weak: at the rates real wind would give, the same neurons drive grooming in this model; there is no wind-steering route, so heading upwind is hand-built.",
                "Efference copy: the eyes' motion signal is damped while the fly turns on purpose, as in real flies.",
                "The learning rule's constants (rate, time windows, floor, forgetting).",
                "Fixes for runaway loops: mild neuron fatigue and blocking the output of the 420 antennal-lobe local neurons.",
                "With the parts list on, where APL releases: that its inhibition stays local to the busy part of the mushroom body is the literature (Amin et al. 2020), the rule that turns each lobe's Kenyon-cell activity into APL's release there is the kit's. That dopamine turns APL down through Dop2R is the literature (Zhou et al. 2019).",
            ],
            "not_modelled": [
                "Real neuron shapes and individual properties, hormones, electrical synapses, most neuromodulation, development.",
                "Genes beyond two transcription factors' expression labels, the transmitter each neuron makes and (with the parts list on) the aminergic receptors its cell type expresses where an adult scRNA-seq cluster exists. Still no ion-channel differences, no peptide signalling (the ontology only names the peptidergic types), no development from the genome.",
                "Absolute firing rates shouldn't be trusted, only which neurons respond. Nothing here is conscious.",
            ],
        }
