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
import queue
import random
import threading
import time
from dataclasses import dataclass

import numpy as np

from .brain import FlyBrain
from . import genetics
from . import parts as partslib
from . import wiring
from .scenarios import PAIR_SCENARIOS, SCENARIOS, ScenarioRunner, pair_available
from .senses.social import SONG_FAR_MM, resolve_overlaps
from .senses.olfaction import ODOURS
from .world import World

TICK_MS = 25.0          # brain time simulated per world update
GF_BURST = 5            # live giant-fibre spikes over two ticks that start an escape jump (hand-built, see choose_mode)
STILL_TICKS = 8         # a walk still for this many ticks (0.2 s) is shown as resting

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
# The female's decision neurons, watched for a female fly that shares the dish with another simulated fly (the
# two-flies work, docs/TWO_FLIES_PLAN.md 5.6; FlyWire's own type names, so her file needs no rebuild; the names in
# the papers come from FlyWire's annotation columns: DNp37 is vpoDN, AN_SMP_2 is SAG, the seven pC2l types carry
# "Nojima 2021: pC2l" except one unlabelled SIP200f cell, left out by its body id). Readouts, never a verdict: song
# drives both the plate-opening and the extrusion command, and what extrusion means depends on a mating status this
# model does not have (3.2). A single female fly (male or female play without a partner) does not get them.
FEMALE_READOUTS = [
    ("vpoEN", "vpoEN", "song-tuned input to vpoDN", "Her decisions", 60, "#f4a6d7"),
    ("pC2l", "AVLP567,AVLP568,AVLP569,AVLP570,CL313,SIP200f,SIP201f,!body:720575940610359758",
     "pulse-song detectors (pC2l)", "Her decisions", 60, "#f4a6d7"),
    ("DNp37", "DNp37", "vaginal plate opening command (vpoDN)", "Her decisions", 60, "#ff7fc8"),
    ("DNp13", "DNp13", "ovipositor extrusion command", "Her decisions", 60, "#ff7fc8"),
    ("DNp55", "DNp55", "a strong vpoEN target (role unknown)", "Her decisions", 60, "#f4a6d7"),
    ("oviDN", "oviDNa_a,oviDNa_b,oviDNb", "egg laying", "Her decisions", 60, "#d9a3ff"),
    ("SAG", "AN_SMP_2", "mating status (SAG)", "Her decisions", 60, "#d9a3ff"),
]

# The antennal-lobe local neurons are silenced in the game (outputs blocked, like tetanus toxin).
BASELINE_SILENCED = "class:ALLN"

# Tarsal taste neurons fired when the male taps the female (the leg chemosensory types with any route
# to pC1 in this data). In this model that route is far too weak to fire pC1 (LgLG1a at 400 Hz gives
# pC1 0.2 Hz; it would need ~8x more input gain), so contact ALSO drives the pC1 courtship neurons
# directly (COURTSHIP_DRIVE, hand-built). Everything downstream of pC1 (pIP10, wing motor neurons,
# DNp13) is wiring. See docs/SCIENCE.md.
PHEROMONE_GRNS = {"LgLG1a,LgLG1b": 60.0}
COURTSHIP_SPEC, COURTSHIP_HZ, COURTSHIP_SECS = "prefix:pC1_", 60.0, 2.5
# A sound (a clap) vibrates the antennae: Johnston's organ B neurons. Through the wiring they reach
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
    ("prefix:JO-A,prefix:JO-B", 100, "Johnston's organ A+B: all its sound neurons (the clap drives B alone, which reaches the giant fibre more)"),
    ("T4a/R,T5a/R", 60, "T4a/T5a right: front-to-back motion on the right eye (optomotor)"),
]

# The action types the API takes (docs/API.md), each with the numbers and switches it reads: {field: (int, float or
# bool, required)}. Game.action converts the numbers and checks that a switch is a JSON true or false before the
# action is queued, so a bad or missing field gets {"ok": false, "error": ...}.
ACTIONS = {
    "hand": {"x": (float, True), "y": (float, True)}, "hand_off": {}, "tool": {},
    "drop": {"x": (float, True), "y": (float, True), "r": (float, False)}, "remove": {"id": (int, True)},
    "dust": {"x": (float, False), "y": (float, False)}, "shock": {"secs": (float, False)}, "sound": {"secs": (float, False)},
    "stripes": {"count": (int, False), "drum_speed": (float, False)}, "zap": {"hz": (float, False), "secs": (float, False)},
    "silence": {}, "unsilence": {}, "modulate": {"factor": (float, False)}, "watch": {}, "unwatch": {}, "grow": {},
    "parts": {}, "clear": {}, "reset": {}, "calm": {}, "autopilot": {"on": (bool, False)}, "pause": {"on": (bool, False)},
    "speed": {"value": (float, False)}, "wind": {"angle": (float, False), "speed": (float, False)},
    "female": {"x": (float, False), "y": (float, False), "on": (bool, False)},
    "learning": {"on": (bool, False), "forget": (bool, False)}, "scenario": {},
    "record": {"on": (bool, False), "spikes": (bool, False)}, "state": {"hunger": (float, False), "thirst": (float, False)},
    "capture": {"on": (bool, False)},                     # save a replay to disk (recording.py; the server checks its origin)
    "place_fly": {"x": (float, True), "y": (float, True), "h": (float, False)},
}
# the tools the page offers (and "none", which scenarios use): the "tool" action takes these and the odour ids
TOOLS = ("lure", "hand", "sugar", "bitter", "water", "dust", "shock", "post", "none")

# The pair checks (docs/TWO_FLIES_PLAN.md 5.9 item 5): shown only when a simulated partner is in the dish, each in the
# checklist (and the `done` set) of the fly it describes; a male's need a female other, a female's a male other.
PAIR_CHECKS_MALE = [
    ("pair:seen", "His eyes pick her out: LC10a fires when she crosses his view"),
    ("pair:sang", f"He sings at her: pC1 → pIP10, one wing out, within {SONG_FAR_MM:g} mm of her"),   # the song's reach
    ("pair:tapped", "He taps her: a foreleg lands, his leg taste cells fire"),
]
PAIR_CHECKS_FEMALE = [
    ("pair:heard", "She hears his song: his song reaches her Johnston's organ"),
    ("pair:seen_him", "She sees him: LC10a fires when he crosses her view"),
    ("pair:touched", "She was tapped (her file has no leg taste cells; nothing fires in her)"),
]
PAIR_CHECKS_SAME = {   # two flies of one sex: only the eyes have anything to say (a tap tastes nothing: the leg taste
    "male": [("pair:seen", "His eyes pick the other male out: LC10a fires when he crosses his view")],   # cells and the
    "female": [("pair:seen", "Her eyes pick the other female out: LC10a fires when she crosses her view")],   # kit's arousal answer a female; no song)
}
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
    sense (as dopamine/NPF do in real flies) and makes the fly explore more. Thirst sets how strongly
    the water cells fire and falls as it drinks, but in this wiring the fly never drinks by itself: water
    alone does not reach MN9 (senses/taste.py; zapping MN9 on a water drop does make it drink). Arousal
    rises with courtship activity."""
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
                 retest: str = "auto", body: str = "drawn", stride_average: bool = False, brain_procs: str = "auto",
                 partner: dict | None = None, social=None, physics_levers=None):
        """``partner``: a second simulated fly in the dish (docs/TWO_FLIES_PLAN.md 5.4): a dict with its ``conn``
        (loaded here, for the API and the layout; its brain is built from the file in a process of its own, or here
        with brain_procs="off"), and optionally ``brain_kwargs`` (default: the protagonist's overrides), ``parts``
        (False, True or a PartsList), ``autopilot`` (default on: her walking urge, decision 13). ``social``: a
        senses.social.SocialConfig, or its comma list (default "seen,song,contact,collide")."""
        from .agent import HOME, PARTNER_HOME, FlyAgent    # (agent.py imports this module's constants)
        from .senses.social import SocialConfig
        if brain_procs not in ("auto", "on", "off", "server"):
            raise ValueError("brain_procs must be 'auto', 'on', 'off' or 'server'")
        self.brain_procs = brain_procs                   # auto: in this process for one fly, a process per brain with more,
        #                                                  one process for every brain with a GPU brain (6.5); server: that one
        self.social = social if isinstance(social, SocialConfig) else SocialConfig.from_list(social)
        self.profile_name = profile_name
        self.rng = random.Random(seed)
        self.seed = seed
        self.actions: queue.Queue = queue.Queue()
        self.paused = False
        self.speed = 1.0
        self.world = World(seed)
        self.events = EventLog()
        self.scenario = ScenarioRunner(self)
        from .physics import PAIR_DEFAULT_LEVERS, parse_levers
        # speed levers of the physics body (physics.LEVERS): none for a single fly; the pair's adopted default when none are
        # named (docs/SCIENCE.md 13.5); "none" switches the pair's off
        if physics_levers is None:
            physics_levers = PAIR_DEFAULT_LEVERS if (partner is not None and body == "physics") else ()
        self.physics_levers = parse_levers(physics_levers)
        self.pair_world = None                           # a partner with the physics body: one MuJoCo world for both flies
        if partner is not None and body == "physics":    # (physics_pair.py; docs/TWO_FLIES_PLAN.md 8.3-8.5)
            from .physics_pair import PairWorld, available, unavailable_reason
            if not available():
                raise RuntimeError(unavailable_reason())
            if stride_average:
                raise ValueError("--stride-average is the single physics fly's: in a pair each fly's senses see its body as it is")
            self.pair_world = PairWorld([HOME, PARTNER_HOME], seed=seed, world=self.world, levers=self.physics_levers)
        # where the brains run: "local" (this process), "procs" (a process per brain), "server" (one process for every
        # brain: the GPU's way, docs/TWO_FLIES_PLAN.md 6.5, chosen by auto when a fly's brain_kwargs say backend cupy)
        self.brain_mode = self._brain_mode(brain_procs, brain_kwargs, partner)
        self.brain_server = None
        io_factory = None
        if self.brain_mode == "server":
            from .brainio import GpuBrainServer          # (brainio.py imports this module's constants)
            self.brain_server = GpuBrainServer()
            io_factory = self.brain_server.attach
        procs = self.brain_mode != "local"
        # the flies: fly 0 keeps the game's own random stream and seed; fly k gets random.Random(f"{seed}:fly{k}") and
        # brain seed + 1000 k, and never draws from World.rng (docs/TWO_FLIES_PLAN.md D11)
        self.flies = [FlyAgent(self, 0, brain, sex=getattr(brain.conn, "sex", "male"), rng=self.rng, seed=seed,
                               autopilot=autopilot, columnar=columnar, stride_average=stride_average,
                               body=self.pair_world.bodies[0] if self.pair_world is not None else body,
                               physics_levers=self.physics_levers,
                               parts_list=parts_list, brain_factory=brain_factory, brain_kwargs=brain_kwargs, retest=retest,
                               brain_procs=procs, pair=partner is not None, io_factory=io_factory)]
        del brain                                        # a process brain has been built from it: let it go
        if partner is not None:
            k = len(self.flies)
            pconn = partner["conn"]
            kw = dict(partner.get("brain_kwargs") if partner.get("brain_kwargs") is not None
                      else (self.flies[0]._brain_kwargs or {}))
            kw["seed"] = seed + 1000 * k
            self.flies.append(FlyAgent(self, k, None, conn=pconn, sex=getattr(pconn, "sex", "male"),
                                       rng=random.Random(f"{seed}:fly{k}"), seed=seed + 1000 * k,
                                       autopilot=partner.get("autopilot", True), columnar=columnar,
                                       body=self.pair_world.bodies[k] if self.pair_world is not None else "drawn",
                                       parts_list=parts_list, brain_kwargs=kw, retest=retest, brain_procs=procs,
                                       parts=partner.get("parts", False), pair=True, home=PARTNER_HOME, io_factory=io_factory))
            if self.social.mating == "mated":                # channel 6 (off by default): a mated female's SAG is silent
                from .senses.social import SAG_SPEC
                for f in self.flies:
                    if f.sex == "female" and f.conn.count(SAG_SPEC):
                        f.io.silence(SAG_SPEC)
                        f.user_silenced.add(SAG_SPEC)       # shown as silenced, as the lab's own silencing is
        self.layout_json = self._make_layout()
        self.state_json = b"{}"
        self.state_dict: dict = {}
        self.seq = 0
        self.rtf = 1.0
        self.recording: list | None = None
        self.record_active = False
        self.record_spikes = False
        self.capture = None                              # a replay being saved to disk (recording.Capture)
        self.lock = threading.RLock()
        self.stop_loop = threading.Event()               # set: loop() returns after the tick it is in
        self.subscribers: list[queue.Queue] = []
        self.reset_world(first=True)

    def fly(self, k: int = 0) -> "FlyAgent":
        """The k-th fly in the dish (fly 0: the protagonist)."""
        return self.flies[k]

    @staticmethod
    def _brain_mode(brain_procs: str, brain_kwargs: dict | None, partner: dict | None) -> str:
        """``local``, ``procs`` or ``server`` from the brain_procs setting and the flies' backends: a GPU brain (backend
        ``cupy`` in a fly's brain_kwargs) only ever lives in the one shared brain process, or in this process with
        brain_procs off (docs/TWO_FLIES_PLAN.md 6.5); the two flies must use the same backend."""
        kw0 = brain_kwargs or {}
        gpu0 = kw0.get("backend") == "cupy"
        gpu = gpu0
        if partner is not None:
            pkw = partner.get("brain_kwargs") if partner.get("brain_kwargs") is not None else kw0
            gpu1 = pkw.get("backend") == "cupy"
            if gpu0 != gpu1:
                raise ValueError("the two flies' brains must use the same backend, and fly 0's is "
                                 f"{kw0.get('backend', 'auto')} while the partner's is {pkw.get('backend', 'auto')}")
            gpu = gpu0 or gpu1
        if brain_procs == "on" and gpu:
            raise ValueError("brain_procs='on' cannot hold a cupy brain: one brain process serves every fly on the GPU "
                             "(brain_procs='auto' or 'server'), or brain_procs='off' keeps it in this process")
        if brain_procs == "auto":
            return "server" if gpu else ("procs" if partner is not None else "local")
        return {"on": "procs", "off": "local", "server": "server"}[brain_procs]

    def close(self):
        """Stop the loop (it returns after the tick it is in) and let every fly's brain process go. server.serve calls
        this after Ctrl+C, once the loop thread has been joined, so no child is closed while a tick is still using it."""
        self.stop_loop.set()
        if self.capture is not None:                 # a replay being saved is finished first (its files, poses, header)
            c, self.capture = self.capture, None
            c.stop()
        for a in self.flies:
            a.io.close()
        if self.brain_server is not None:            # every handle's close reached it already; once more is harmless
            self.brain_server.close()
        if self.pair_world is not None:              # the shared MuJoCo world, once no tick can be using it (8.11)
            self.pair_world.close()

    # ------------------------------------------------------------------ world
    def reset_world(self, first: bool = False):
        a = self.flies[0]
        a.reset_brain()
        self.world.clear("all")
        self.world.hand = None
        self.world.set_stripes(0, 0.0)
        a.reset_state()
        for b in self.flies[1:]:                         # a partner starts afresh too, at its own place
            b.reset_brain()
            b.reset_state()
        self.message, self.message_left = "", 0.0
        self.t = 0.0
        self.scenario.stop(silent=True)
        if not first:
            self.events.add(self.t, "system", "New fly, fresh brain (learned synapses kept; use 'forget' to reset them).")

    def say(self, text: str, secs: float = 3.0):
        self.message, self.message_left = text, secs

    # ------------------------------------------------------------------ input from the browser
    # the actions that act on one fly (docs/TWO_FLIES_PLAN.md 5.7): they take an optional "fly": k (default 0)
    PER_FLY = ("zap", "silence", "unsilence", "modulate", "watch", "unwatch", "learning", "grow", "parts", "state",
               "place_fly", "calm", "autopilot", "dust", "shock", "sound")

    def action(self, a: dict) -> dict:
        kind = a.get("type")
        fields = ACTIONS.get(kind) if isinstance(kind, str) else None
        if fields is None:
            return {"ok": False, "error": "the action needs a 'type'" if kind is None else f"unknown action type {kind!r}"}
        # "fly": which fly an action is for. Checked before the numeric fields and unlike them: only a JSON whole number
        # (not text, not a fraction, not true/false), since it names a fly and nothing else
        raw_fly = a.get("fly")
        if raw_fly is None:
            a.pop("fly", None)                               # null: fly 0
            fly = self.flies[0]
        elif type(raw_fly) is not int:
            return {"ok": False, "error": f"'fly' must be a whole number, not {raw_fly!r}"}
        elif not 0 <= raw_fly < len(self.flies):
            return {"ok": False, "error": f"no fly {raw_fly}"}
        else:
            fly = self.flies[raw_fly]
        for f, (num, required) in fields.items():            # checked here, so a bad field is refused, not queued
            if a.get(f) is None:
                if required:
                    return {"ok": False, "error": f"'{kind}' needs '{f}' (a number)"}
                a.pop(f, None)                               # null: the default
                continue
            if num is bool:                                  # a switch: true or false, not "false" (which bool() reads as on)
                if not isinstance(a[f], bool):
                    return {"ok": False, "error": f"'{f}' must be true or false, not {a[f]!r}"}
                continue
            if isinstance(a[f], float) and not math.isfinite(a[f]):   # before int(), which cannot take Infinity (or 1e400)
                return {"ok": False, "error": f"'{f}' must be a finite number"}
            try:
                a[f] = num(a[f])
                finite = math.isfinite(a[f])
            except (TypeError, ValueError):
                return {"ok": False, "error": f"'{f}' must be {'a whole number' if num is int else 'a number'}, not {a[f]!r}"}
            except OverflowError:                            # a JSON integer too big for a float
                finite = False
            if not finite:
                return {"ok": False, "error": f"'{f}' must be a finite number"}
            positive = f == "secs" or (f == "r" and a.get("kind") == "post")       # a post needs a size
            if (positive and a[f] <= 0) or (f in ("hz", "factor", "r") and a[f] < 0):
                return {"ok": False, "error": f"'{f}' must be {'more than 0' if positive else '0 or more'}"}
        if kind == "hand":                                   # pointer moves: just remember it
            self.world.hand = (a["x"], a["y"])
            return {"ok": True}
        if kind == "tool":
            if a.get("tool") is None:
                a.pop("tool", None)                          # null: the default (the lure)
            elif not isinstance(a["tool"], str) or (a["tool"] not in TOOLS and a["tool"] not in ODOURS):
                return {"ok": False, "error": f"unknown tool {a['tool']!r}: {', '.join(TOOLS)} or an odour ({', '.join(ODOURS)})"}
        if kind == "drop":
            what = a.get("kind")
            if not isinstance(what, str) or (what not in ("sugar", "bitter", "water", "post") and what not in ODOURS):
                return {"ok": False, "error": f"unknown drop kind {what!r}: sugar, bitter, water, post or an odour "
                                              f"({', '.join(ODOURS)})"}
            if what in ODOURS and a.get("food") not in (None, "", "sugar", "bitter", "water"):
                return {"ok": False, "error": f"an odour's food must be sugar, bitter or water, not {a['food']!r}"}
            margin = a.get("r", 4.0) + 1.0 if what == "post" else 2.0     # as World.add_food, add_odour, add_obstacle
            if not self.world.inside(a["x"], a["y"], margin):
                self.say(f"Too close to the wall: drop it at least {margin:g} mm inside the dish.", 2.5)
                return {"ok": False, "error": f"({a['x']:g}, {a['y']:g}) is outside the dish or within {margin:g} mm of its wall"}
        if kind == "remove" and not any(o.id == a["id"] for o in (*self.world.food, *self.world.obstacles, *self.world.odours)):
            return {"ok": False, "error": f"nothing in the dish has id {a['id']}"}
        if kind == "clear" and a.get("what", "all") not in ("all", "food", "odours", "obstacles"):
            return {"ok": False, "error": "'what' must be all, food, odours or obstacles"}
        if kind == "hand_off":
            self.world.hand = None
            return {"ok": True}
        if kind == "female" and len(self.flies) > 1 and a.get("on", True):
            return {"ok": False, "error": "a simulated partner is in the dish; the scripted female is for single-fly play "
                                          "(start the game without --partner to use her)"}
        if kind in ("zap", "silence", "modulate", "watch"):
            spec = str(a.get("spec", "")).strip()
            try:
                n = len(fly.conn.select(spec))                # that fly's cells (her pIP10: none)
            except (ValueError, KeyError) as e:
                return {"ok": False, "error": str(e)}
            if n == 0:
                return {"ok": False, "error": f"No neurons match '{spec}'. Try a type like MDN or prefix:LC10."}
            a["n"] = n
            a["spec"] = spec
        if kind == "watch":
            explicit = str(a.get("key") or "").strip()
            key = (explicit or a["spec"])[:24]
            if key in fly.builtin_keys:                      # the decoder and the page read these; a rewire would steer the body
                if explicit:
                    return {"ok": False, "error": f"'{key}' is a built-in readout; pick another name for the watch."}
                key = f"watch:{a['spec']}"[:24]
            a["key"] = key
        if kind == "unwatch" and (a.get("key") in fly.builtin_keys or a.get("key") not in fly.custom_readouts):
            return {"ok": False, "error": f"'{a.get('key')}' is not a custom watch."}
        if kind == "scenario" and a.get("id") is not None and not isinstance(a["id"], str):
            return {"ok": False, "error": f"'id' must be a scenario id ({', '.join(SCENARIOS)}), not {a['id']!r}"}
        if kind == "scenario" and a.get("id") and a["id"] not in SCENARIOS:
            if a["id"] not in PAIR_SCENARIOS:
                return {"ok": False, "error": f"unknown scenario {a['id']}"}
            why = pair_available(self, a["id"])              # a two-fly scenario (docs/TWO_FLIES_PLAN.md 5.9 item 3)
            if why is not None:
                return {"ok": False, "error": why}
        if kind == "capture":                                # refused here, with the reason, before it is queued (8.8)
            from . import recording
            if a.get("on", True):
                why = recording.cannot_start(self)
                if why:
                    return {"ok": False, "error": why}
            elif self.capture is None:
                return {"ok": False, "error": "no replay is being saved"}
        if kind == "grow":
            level = str(a.get("level", "type")).strip().lower()
            if not wiring.valid_level(level):
                return {"ok": False, "error": wiring.LEVEL_ERROR}
            if fly.genome["growing"]:
                return {"ok": False, "error": "a fly is already being grown; wait for it"}
            try:
                a["level"], a["seed"] = level, 1 if a.get("seed") is None else int(a["seed"])    # null: the default
            except (TypeError, ValueError):
                return {"ok": False, "error": "seed must be a whole number"}
            fly.genome.update(growing={"level": level, "seed": a["seed"], "t0": time.time()}, error=None)   # claimed now, one at a time
        if kind == "parts":
            if not isinstance(a.get("on"), bool):
                return {"ok": False, "error": "'on' must be true or false"}
            if fly.genome["growing"]:
                return {"ok": False, "error": "the brain is being rebuilt; wait for it"}
            if a["on"] == fly.parts_on:
                return {"ok": False, "error": f"the parts list is already {'on' if a['on'] else 'off'}"}
            fly.genome.update(growing={"level": fly.genome["level"], "seed": fly.genome["seed"], "reason": "parts",
                                       "parts": a["on"], "t0": time.time()}, error=None)                 # claimed now
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
        f = self.flies[a.get("fly", 0)]                  # the fly a per-fly action is for (checked in action(); default 0)
        many = len(self.flies) > 1

        def ev(kind_, text):                             # a per-fly event names its fly when there is more than one
            self.events.add(self.t, kind_, text, **({"fly": f.id} if many else {}))
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
            x, y = float(a.get("x", f.body.pose.x)), float(a.get("y", f.body.pose.y))
            if math.hypot(x - f.body.pose.x, y - f.body.pose.y) < 20:
                f.antennae.puff_dust(1.5)
                self.say("Dust on the antennae!", 1.5)
        elif kind == "sound":
            f.sound_left = float(a.get("secs", 0.3))
            ev("world", "a loud sound (Johnston's organ B neurons)")
            self.say("Clap! Johnston's organ hears it.", 1.5)
        elif kind == "shock":
            f.shock_left = float(a.get("secs", 1.0))
            ev("learning", "Electric shock: PPL1 punishment dopamine driven directly (hand-built)")
            self.say("Shock! Punishment dopamine pairs with whatever it smells now.", 2.5)
        elif kind == "zap":
            f.zaps = [z for z in f.zaps if z[0] != a["spec"]]
            f.zaps.append([a["spec"], float(a.get("hz", 60)), float(a.get("secs", 2.0))])
            f.done.add("zap")
            self.say(f"Zapping {a['spec']} ({a['n']} neurons) at {float(a.get('hz', 60)):g} Hz", 2.0)
            ev("lab", f"zap {a['spec']} at {float(a.get('hz', 60)):g} Hz for {float(a.get('secs', 2.0)):g} s")
        elif kind == "silence":
            n = f.io.silence(a["spec"])
            f.user_silenced.add(a["spec"])
            f.done.add("silence")
            if a["spec"].startswith(("gene:", "dimorphism:")):
                f.done.add("genetics")
            self.say(f"Silenced {a['spec']} ({n} neurons): they still fire, but nothing hears them.", 3.0)
            ev("lab", f"silenced {a['spec']} ({n} neurons)")
        elif kind == "unsilence":
            for spec in list(f.user_silenced):
                if a.get("spec") in (None, "", spec):
                    f.io.unsilence(spec)
                    f.user_silenced.discard(spec)
            self.say("Silencing removed.", 2.0)
        elif kind == "modulate":
            factor = float(a.get("factor", 1.0))
            n = f.io.modulate(a["spec"], factor)
            if abs(factor - 1.0) < 1e-6:
                f.user_modulated.pop(a["spec"], None)
            else:
                f.user_modulated[a["spec"]] = factor
            self.say(f"Output of {a['spec']} ({n} neurons) scaled x{factor:g}.", 3.0)
            ev("lab", f"modulate {a['spec']} x{factor:g}")
        elif kind == "watch":
            key = a["key"]                                   # validated in action()
            if key in f.builtin_keys:
                raise ValueError(f"'{key}' is a built-in readout")
            if key in f.custom_readouts:
                f.remove_monitor(key)
            f.custom_readouts[key] = a["spec"]
            f.readouts[key] = f.conn.select(a["spec"])
            f.hz_shown[key] = 0.0
            f.add_monitor(key, a["spec"])
        elif kind == "unwatch":
            key = a.get("key")
            if key in f.custom_readouts:
                del f.custom_readouts[key]
                f.readouts.pop(key, None)
                f.hz_shown.pop(key, None)
                f.remove_monitor(key)
        elif kind == "clear":
            w.clear(a.get("what", "all"))
        elif kind == "reset":
            self.reset_world()
            self.say("New fly, fresh brain.", 2.0)
        elif kind == "autopilot":
            f.autopilot = bool(a.get("on", True))
        elif kind == "pause":
            self.paused = bool(a.get("on", False))
        elif kind == "speed":
            self.speed = max(0.1, min(3.0, float(a.get("value", 1.0))))
        elif kind == "calm":
            f.reset_brain()
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
            io = f.io
            if io.has_plasticity:
                if a.get("forget"):
                    io.learning(forget=True)
                    ev("learning", "all KC→MBON synapses reset to their original strength")
                    self.say("Memories erased.", 2.0)
                if "on" in a:
                    f.learning_on = bool(a["on"])            # (the setter tells the brain)
        elif kind == "scenario":
            if a.get("id"):
                self.scenario.start(a["id"])
            else:
                self.scenario.stop()
        elif kind == "record":
            ios = [b.io for b in self.flies]             # every fly's brain records its spikes (/api/spikes?fly=k reads fly k's)
            if a.get("on", True):
                for io in ios:
                    if io.record("active"):              # a restart while spikes were being recorded
                        io.record("stop")
                    io.record("clear_kept")              # the previous take is gone once a new one starts
                self.recording = []
                self.record_active = True
                self.record_spikes = bool(a.get("spikes", False))
                if self.record_spikes:
                    for io in ios:
                        io.record("start")
                self.events.add(self.t, "system", "recording started")
            elif self.record_active:
                self.record_active = False               # frames are kept for download until the next start
                for io in ios:
                    if io.record("active"):
                        io.record("stop")
                self.events.add(self.t, "system", f"recording stopped ({len(self.recording or [])} frames)")
        elif kind == "capture":
            from . import recording
            if a.get("on", True):
                if self.capture is None:
                    self.capture = recording.Capture.start(self)
                    self.events.add(self.t, "system", f"saving a replay: recordings/{self.capture.id}")
                    self.say(f"Saving a replay to recordings/{self.capture.id}", 3.0)
            elif self.capture is not None:
                c, self.capture = self.capture, None
                c.stop()
                self.events.add(self.t, "system", f"replay saved: recordings/{c.id} ({c.ticks} ticks)")
                self.say(f"Replay saved: recordings/{c.id} ({c.ticks} ticks)", 3.0)
        elif kind == "grow":
            f._start_grow(a["level"], a["seed"])
        elif kind == "parts":
            f._start_rebuild(a["on"])
        elif kind == "_swap_brain":
            f._swap_brain(a)                                 # (the worker that made it put its fly's id in)
        elif kind == "state":
            for k in ("hunger", "thirst"):
                if k in a:
                    setattr(f.state, k, max(0.0, min(1.0, float(a[k]))))
        elif kind == "place_fly":
            f.body.reset(float(a["x"]), float(a["y"]), float(a.get("h", f.body.pose.h)))

    # ------------------------------------------------------------------ the loop
    def tick_paused(self):
        """A paused loop's turn: the queued actions, then the page kept in sync with every brain's state without a
        step. No brain stepped, so every fly's rates read 0, as the top level's always did (with two or more flies
        the ``flies`` entries mirror it)."""
        self._apply_actions()
        for f in self.flies:
            f.graded_eps = 0.0
            f.sps = 0.0
        peeks = [f.io.peek(TICK_MS / 1000.0, self.seq, f.readouts) for f in self.flies]
        self.publish(peeks[0], self.hz_shown, 0, peeks)

    def tick(self):
        """One 25 ms tick of every fly in the dish (docs/TWO_FLIES_PLAN.md 5.4): all flies sense the same start-of-tick
        snapshot of the others, all brains advance (in lockstep when each has its own process), then all bodies
        move, then the bookkeeping; with one fly this is exactly the single fly's order."""
        dt = TICK_MS / 1000.0
        self._apply_actions()
        self.scenario.step(dt)
        flies = self.flies
        a = flies[0]
        if len(flies) == 1:
            rates, col_idx, col_hz = a.senses(dt)
            bts = [a.advance(dt, rates, col_idx, col_hz, self.seq)]   # seq: the brain-map sample is drawn as publish will number it
            others = [()]
        else:
            from .brainio import advance_all                 # (brainio.py imports this module's constants)
            snap = [f.pose_view() for f in flies]
            others = [[s for s in snap if s.id != f.id] for f in flies]
            inputs = []
            for f, o in zip(flies, others):
                rates, col_idx, col_hz = f.senses(dt, o)
                inputs.append(f.advance_input(dt, rates, col_idx, col_hz, self.seq))
            bts = [f.advance_done(bt) for f, bt in zip(flies, advance_all([f.io for f in flies], inputs))]
        for f, bt, o in zip(flies, bts, others):
            f.act(dt, bt, o)
        if self.pair_world is not None:                  # both drives set: the shared physics world steps once (8.5)
            self.pair_world.advance_tick(dt)
        if len(flies) > 1 and self.social.collide:
            resolve_overlaps(flies)                      # both pushed apart equally (D4; senses/social.py)
        for f in flies:
            f.bookkeep(dt)
        self.world.step(dt, (a.body.pose.x, a.body.pose.y), fly_singing=a.m["song"], courted=a.courting > 0)
        for f, bt in zip(flies, bts):
            f.after_senses(dt, bt)
        self.t += dt
        self.message_left -= dt
        for f, bt in zip(flies, bts):
            f.watchdog(dt, bt)
        bt = bts[0]
        if self.recording is not None and self.record_active:
            frame = {"t": round(self.t, 3), "fly": a.body.to_dict(), "mode": a.mode,
                     "hz": {k: round(v, 1) for k, v in bt.hz.items() if a._has(k)}, "senses": a.senses_now,
                     "sps": int(a.sps)}
            if len(flies) > 1:                           # with two flies: every fly, and a small snapshot of the dish
                frame["flies"] = [{"id": f.id, "fly": f.body.to_dict(), "mode": f.mode,
                                   "hz": {k: round(v, 1) for k, v in b.hz.items() if f._has(k)}, "senses": f.senses_now,
                                   "sps": int(f.sps)} for f, b in zip(flies, bts)]
                w = self.world.to_dict()
                frame["world"] = {k: w[k] for k in ("food", "obstacles", "odours", "wind", "stripes", "tool")}
            self.recording.append(frame)
        if self.capture is not None:                     # a replay saved to disk: its own frame (recording.py)
            c = self.capture
            try:
                c.add(self)
            except Exception as e:                       # a full disk must not kill the loop: the replay stops here
                c.error = f"{e!r}"
                c.stop()
            if c.error and self.capture is c:            # stopped on its own (the cap, a rebuilt world, an error)
                self.capture = None
                self.events.add(self.t, "system", f"replay recordings/{c.id} stopped: {c.error}")
                self.say(f"Replay stopped: {c.error}", 5.0)
        self.publish(bt, a.hz_shown, a.sps, bts)

    # ------------------------------------------------------------------ publishing
    def publish(self, bt, hz, sps, bts=None):
        """The state the page reads, from this tick's BrainTick ``bt`` (its spike sample, stimulus count, silenced
        specs, learning summary and parts status were taken from the brain after it stepped). ``bts``: every fly's
        BrainTick; with two or more flies the state gains a ``flies`` list (docs/TWO_FLIES_PLAN.md 5.7), the top level
        keeps mirroring fly 0, and with one fly nothing is added."""
        a = self.flies[0]
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
            "hz": {k: round(v, 1) for k, v in hz.items() if self._has(k)},     # not the cells this fly lacks
            "motor": {k: round(v, 3) for k, v in self.decoder.m.items()},
            "driver": self.driver, "mode": self.mode,
            "spikes": bt.spikes_shown, "sps": int(sps), "graded_eps": int(self.graded_eps),
            "stims": bt.stims,
            "calms": self.calms, "msg": self.message if self.message_left > 0 else "",
            "silenced": sorted(self.user_silenced),
            "baseline": sorted(set(bt.silenced_specs) - self.user_silenced),
            "modulated": self.user_modulated,
            "custom": self.custom_readouts,
            "done": sorted(self.done),
            "state": self.state.to_dict(),
            "learning": a.learning_summary(bt),
            "events": self.events.items[-12:],
            "event_seq": self.events.seq,
            "scenario": self.scenario.status(),
            "genome": a.genome_status(bt.parts_status, known=True),
            "recording": None if self.recording is None else {"frames": len(self.recording), "spikes": self.record_spikes,
                                                              "active": self.record_active},
            # "capture" only while a replay is being saved: the single fly's frames stay byte for byte what they were (the
            # golden hashes cover every state frame), and the page reads a missing key as "not saving"
            **({"capture": self.capture.status()} if self.capture is not None else {}),
        }
        if len(self.flies) > 1 and bts is not None:
            state["flies"] = [f.state_entry(b) for f, b in zip(self.flies, bts)]
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
        """Tick in real time (scaled by ``speed``) until :attr:`stop_loop` is set (server.serve sets it after Ctrl+C)."""
        while not self.stop_loop.is_set():
            t0 = time.perf_counter()
            try:
                if self.paused:
                    self.tick_paused()
                    time.sleep(0.05)
                    continue
                self.tick()
            except Exception as e:                    # keep the game alive, show the problem (a paused game too: its
                self.say(f"Error: {e}", 5.0)          # brain process can die while it waits)
                import traceback
                traceback.print_exc()
                time.sleep(0.2)
                if self.paused:
                    continue
            used = time.perf_counter() - t0
            budget = TICK_MS / 1000.0 / self.speed
            self.rtf += (min(1.0, budget / max(used, 1e-6)) - self.rtf) * 0.1
            if used < budget:
                time.sleep(budget - used)

    # ------------------------------------------------------------------ fly 0, as the game's own attributes
    # One fly's state lives on its FlyAgent (agent.py). These names keep reading and writing fly 0's, so that the
    # server, the scenarios and every caller written for one fly (game.brain, game.body, game.mode, ...) still work.
    _FLY_ATTRS = ("brain", "conn", "real_conn", "brain_factory", "_brain_kwargs", "retest_mode", "_retest_handle",
                  "_survival_cache", "_retest_lock", "_rules_cache", "_survival_token", "genome", "parts_on", "_parts_list",
                  "_parts_counts", "autopilot", "body", "body_kind", "retina", "columnar_on", "nose", "mouth", "water_cells",
                  "forelegs", "antennae", "bristles", "decoder", "state", "readouts", "readout_meta", "genetics", "neuronbridge",
                  "custom_readouts", "user_silenced", "user_modulated", "has_soma", "learning_on", "done", "zaps", "mode",
                  "wander", "runaway_s", "graded_eps", "since_input", "calms", "hz_shown", "senses_now", "_prev_felt", "driver",
                  "gf_cooldown", "gf_prev", "still_ticks", "turn_command", "court_left", "sound_left", "bitter_t", "sugar_t",
                  "eating", "drinking", "song_side", "shock_left", "learned_bias", "smelling")

    def senses(self, dt: float):
        return self.flies[0].senses(dt)

    def choose_mode(self, m: dict, gf_spikes: int) -> str:
        return self.flies[0].choose_mode(m, gf_spikes)

    def odour_steering(self) -> tuple[float, str]:
        return self.flies[0].odour_steering()

    def _sense_events(self):
        return self.flies[0]._sense_events()

    def learning_summary(self) -> dict | None:
        return self.flies[0].learning_summary()

    def genome_status(self) -> dict:
        return self.flies[0].genome_status()

    def parts_arg(self, on: bool):
        return self.flies[0].parts_arg(on)

    def parts_list(self) -> "partslib.PartsList":
        return self.flies[0].parts_list()

    def parts_counts(self) -> dict:
        return self.flies[0].parts_counts()

    def _retest_spec(self, conn) -> dict | None:
        return self.flies[0]._retest_spec(conn)

    def _survival_key(self, conn) -> tuple:
        return self.flies[0]._survival_key(conn)

    def _survival_worker(self, conn, token):
        return self.flies[0]._survival_worker(conn, token)

    def _start_grow(self, level: str, seed: int):
        return self.flies[0]._start_grow(level, seed)

    def _grow_worker(self, level: str, seed: int):
        return self.flies[0]._grow_worker(level, seed)

    def _start_rebuild(self, on: bool):
        return self.flies[0]._start_rebuild(on)

    def _rebuild_worker(self, on: bool):
        return self.flies[0]._rebuild_worker(on)

    def _swap_brain(self, a: dict):
        return self.flies[0]._swap_brain(a)

    def _default_brain_factory(self, conn, **extra):
        return self.flies[0]._default_brain_factory(conn, **extra)

    def _has(self, key: str) -> bool:
        return self.flies[0]._has(key)

    def _make_layout(self):
        return self.flies[0]._make_layout()

    def whats_real(self) -> dict:
        return self.flies[0].whats_real()


def _fly_shim(name: str) -> property:
    return property(lambda self: getattr(self.flies[0], name), lambda self, v: setattr(self.flies[0], name, v),
                    doc=f"fly 0's {name} (one fly's state lives on its FlyAgent, agent.py)")


for _name in Game._FLY_ATTRS:
    setattr(Game, _name, _fly_shim(_name))
