"""
Scripted scenarios: classic fly experiments as timed protocols the game can run for you.

A scenario is a list of steps; each step has a caption for the screen and an action (a function
of the game) or a wait, and optionally a *measure* that is evaluated during the step (e.g. the
fly's preference between two odours). Scenarios only place things in the world and inject the
labelled hand-built signals (reward, shock); what the fly does is up to its brain.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class Step:
    caption: str
    secs: float = 0.0
    action: Callable | None = None       # game -> None, run once when the step starts
    measure: Callable | None = None      # game -> dict, evaluated every tick and shown


@dataclass
class Scenario:
    id: str
    name: str
    description: str
    steps: list[Step] = field(default_factory=list)
    female: str = ""                     # the description for the female fly, where hers differs
    pair: str | Callable = ""            # the description with a simulated partner in the dish, where it differs: a
                                         # string, or a function of (the protagonist's sex, the partner's sex)


def _place_two_odours(game, a="vinegar", b="banana", with_food=None, on="a"):
    w = game.world
    w.clear("odours")
    w.clear("food")
    w.add_odour(a, -24.0, 14.0, food=with_food if on == "a" else None)
    w.add_odour(b, 24.0, 14.0, food=with_food if on == "b" else None)
    game.body.reset(0.0, -18.0, math.pi / 2)


def _preference(game, a="vinegar", b="banana"):
    """Time-weighted preference index between the two odour sources, -1 (all B) .. +1 (all A)."""
    w = game.world
    srcs = {s.odour: s for s in w.odours}
    if a not in srcs or b not in srcs:
        return {}
    p = game.body.pose
    da = math.hypot(p.x - srcs[a].x, p.y - srcs[a].y)
    db = math.hypot(p.x - srcs[b].x, p.y - srcs[b].y)
    st = game.scenario.store
    st["pair"] = (a, b)
    st["near_a"] = st.get("near_a", 0.0) + (1.0 if da < 14 else 0.0) * 0.025
    st["near_b"] = st.get("near_b", 0.0) + (1.0 if db < 14 else 0.0) * 0.025
    tot = st["near_a"] + st["near_b"]
    pi = (st["near_a"] - st["near_b"]) / tot if tot > 0 else 0.0
    return {"preference_index": round(pi, 2), "near_a_s": round(st["near_a"], 1), "near_b_s": round(st["near_b"], 1)}


def _reset_store(game):
    game.scenario.store.clear()


SCENARIOS: dict[str, Scenario] = {}


def _add(sc: Scenario):
    SCENARIOS[sc.id] = sc


_add(Scenario(
    "appetitive", "Appetitive conditioning (odour + sugar)",
    "Two odours. One is paired with sugar while the fly is hungry; then both are tested without food. "
    "Real flies come to prefer the paired odour (Tempel et al. 1983). Here the pairing depresses the "
    "Kenyon-cell synapses onto the avoidance-promoting MBONs in the PAM compartments.",
    [
        Step("Baseline: two odours, no food. Where does the fly spend its time?", 40.0,
             lambda g: (_reset_store(g), _place_two_odours(g), setattr(g.state, "hunger", 1.0),
                        g.world.set_wind(math.pi / 2, 0.0)),
             lambda g: _preference(g)),
        Step("Training: vinegar now comes with sugar. Eating it drives the PAM reward neurons (hand-built).", 60.0,
             lambda g: (_reset_store(g), _place_two_odours(g, with_food="sugar", on="a")),
             lambda g: {**_preference(g), "depressed": g.flies[0].depressed_fraction()}),
        Step("Test: food removed. Has the preference for vinegar grown?", 40.0,
             lambda g: (_reset_store(g), _place_two_odours(g)),
             lambda g: _preference(g)),
        Step("Done. Compare the baseline and test preference indices in the event log.", 0.0,
             lambda g: g.events.add(g.t, "scenario", f"appetitive conditioning finished: test preference {_preference(g).get('preference_index', 0):+.2f}")),
    ]))

_add(Scenario(
    "aversive", "Aversive conditioning (odour + shock)",
    "Odour A is paired with electric shock (PPL1 punishment dopamine, driven directly); odour B is not. "
    "Then both are tested. Real flies avoid the shocked odour (Tully & Quinn 1985).",
    [
        Step("Baseline: two odours, nothing else.", 40.0,
             lambda g: (_reset_store(g), _place_two_odours(g, "banana", "yeast"), g.world.set_wind(math.pi / 2, 0.0)),
             lambda g: _preference(g, "banana", "yeast")),
        Step("Training: whenever the fly smells banana, it gets a shock.", 60.0,
             lambda g: (_reset_store(g), _place_two_odours(g, "banana", "yeast"), g.scenario.store.__setitem__("shock_odour", "banana")),
             lambda g: {**_preference(g, "banana", "yeast"), "depressed": g.flies[0].depressed_fraction()}),
        Step("Test: no more shocks. Does it now avoid banana?", 40.0,
             lambda g: (_reset_store(g), _place_two_odours(g, "banana", "yeast"), g.scenario.store.pop("shock_odour", None)),
             lambda g: _preference(g, "banana", "yeast")),
        Step("Done. Compare the baseline and test preference indices in the event log.", 0.0,
             lambda g: g.events.add(g.t, "scenario", f"aversive conditioning finished: test preference {_preference(g, 'banana', 'yeast').get('preference_index', 0):+.2f}")),
    ]))

def _female_enters(game):
    """The courtship scenario's female: the scripted one in single-fly play; with a simulated partner in the dish
    (docs/TWO_FLIES_PLAN.md 5.4, D9) the partner is placed where she would have stood, facing wherever its own dice
    say, and no scripted female is made."""
    if len(game.flies) > 1:
        f = game.flies[1]
        f.body.reset(14.0, 10.0, f.rng.uniform(-math.pi, math.pi))
    else:
        game.world.toggle_female(True, 14.0, 10.0)


def _courtship_pair_text(me: str, other: str) -> str:
    """The courtship scenario's text with a simulated partner in the dish (D9: no scripted female can enter), by the
    protagonist's sex (``me``, fly 0) and the partner's (``other``, fly 1): the partner is placed where the scripted
    female would have stood."""
    if me == "male" and other == "female":
        return ("The simulated female is placed ahead of him and both flies are left to their brains for 90 s. Nothing links "
                "the two brains but the world: he sees her as a small moving object (LC10a → DNa02, chase), taps her (his leg "
                "taste cells, and the kit's hand-built pC1 arousal) and sings (pIP10, one wing out); she sees him, hears his "
                "song on her Johnston's organ and is bumped by him. The measure shows his pC1 and song and their distance.")
    if me == "female" and other == "male":
        return ("The simulated male is placed ahead of her and both flies are left to their brains for 90 s. Nothing links "
                "the two brains but the world: she sees him as a small moving object (LC10a → DNa02, chase) and is bumped by "
                "him; he sees her, taps her (his leg taste cells, and the kit's hand-built pC1 arousal) and sings (pIP10, one "
                "wing out), which reaches her Johnston's organ. The measure shows her pC1 (this brain has no pIP10, so no "
                "song) and their distance.")
    if me == "male":
        return ("A second simulated male is placed ahead of him and both flies are left to their brains for 90 s. Nothing "
                "links the two brains but the world: each sees the other as a small moving object (LC10a → DNa02, chase) and "
                "bumps into him; a foreleg on another male tastes nothing and starts no courtship arousal (both answer the "
                "pheromone of the other sex), so no song is expected. The measure shows his pC1 and song and their distance.")
    return ("A second simulated female is placed ahead of her and both flies are left to their brains for 90 s. Nothing links "
            "the two brains but the world: each sees the other as a small moving object (LC10a → DNa02, chase) and bumps into "
            "the other; a foreleg on another female finds no leg taste cells in this file and, for a female toucher, starts "
            "no courtship arousal unless contact_pc1 says so (decision 14); neither sings (no pIP10). The measure shows this "
            "fly's pC1 and their distance.")


_add(Scenario(
    "courtship", "Courtship",
    "A female enters the dish. The male sees her as a small moving object (LC10a → DNa02, chase), "
    "taps her with a foreleg (pheromone taste → pC1) and sings (pIP10, one wing out).",
    [
        Step("A female enters.", 90.0,
             lambda g: (g.world.clear("all"), _female_enters(g), g.body.reset(-10.0, -8.0, 0.3)),
             lambda g: {"pC1_hz": round(g.hz_shown.get("pC1", 0), 1), "song": round(g.decoder.m["song"], 2),
                        **({"partner_distance": round(math.hypot(g.flies[1].body.pose.x - g.body.pose.x,
                                                                  g.flies[1].body.pose.y - g.body.pose.y), 1)}
                           if len(g.flies) > 1 else
                           {"female_receptive": round(g.world.female.receptive, 2) if g.world.female else 0})}),
        Step("Done.", 0.0, lambda g: g.events.add(g.t, "scenario", "courtship scenario finished")),
    ],
    female="A second female enters the dish. The fly sees her as a small moving object (LC10a → DNa02, chase); touching "
           "her drives the fly's pC1 neurons directly (hand-built: this brain has no tarsal taste neurons). The fly does "
           "not sing: this female brain has no pIP10 and no nerve cord.",
    pair=_courtship_pair_text))

_add(Scenario(
    "plume", "Following a plume upwind",
    "Wind blows from the right; vinegar is released upwind. The fly's antennae report wind direction "
    "(Johnston's organ) and the plume flickers past it. Odour-guided steering is hand-built; wind sensing is wired.",
    [
        Step("Wind on, vinegar upwind.", 90.0,
             lambda g: (g.world.clear("all"), g.world.set_wind(math.pi, 18.0), g.world.add_odour("vinegar", 34.0, 0.0),
                        g.body.reset(-20.0, 0.0, -math.pi / 2)),
             lambda g: {"distance_to_source": round(math.hypot(g.body.pose.x - 34.0, g.body.pose.y), 1),
                        "wind_from_deg": g.senses_now.get("wind")}),
        Step("Done.", 0.0, lambda g: g.events.add(g.t, "scenario", "plume scenario finished")),
    ]))

_add(Scenario(
    "escape", "Looming escape",
    "A hand swoops at the fly three times, from different sides. Watch the retina, LC4/LPLC2 and the giant fibre.",
    [
        Step("Swoop 1 (from the right).", 3.0, lambda g: g.scenario.store.update(swoop=(1, 0.0))),
        Step("Swoop 2 (from the front).", 3.0, lambda g: g.scenario.store.update(swoop=(2, 0.0))),
        Step("Swoop 3 (slowly: a slow hand should not scare it).", 5.0, lambda g: g.scenario.store.update(swoop=(3, 0.0))),
        Step("Done.", 0.0, lambda g: (g.scenario.store.pop("swoop", None), setattr(g.world, "hand", None))),
    ]))


# ---------------------------------------------------------------------------------------------- two flies
# Scenarios that need a simulated partner in the dish (docs/TWO_FLIES_PLAN.md 5.9 item 3). They are kept apart from
# SCENARIOS, so the single fly's layout and the ids Game.action accepts without a partner do not change (D3).
PAIR_SCENARIOS: dict[str, Scenario] = {}
PAIR_NEAR_MM = 15.0                      # "near": real centre-to-centre mm, as the plan's measurements count it


def _pair_roles(game):
    """The male and the female of a two-fly dish, whichever index each has (fly 0 is the female with
    ``--female --partner male``), or None when the dish has no such pair (one fly, two males, two females)."""
    males = [f for f in game.flies if f.sex == "male"]
    females = [f for f in game.flies if f.sex == "female"]
    if not males or not females:
        return None
    return males[0], females[0]


def pair_available(game, sid: str) -> str | None:
    """Why a two-fly scenario cannot run in this dish (the text of the refusal), or None when it can: every one needs
    a simulated partner, and ``pair_courtship`` a male and a female (whichever index each has). Game.action and the
    layout read this, so a scenario is offered only where it can run."""
    if len(game.flies) < 2:
        return (f"{sid} is a two-fly scenario: it needs a simulated partner in the dish "
                "(start the game with --partner female)")
    if sid == "pair_courtship" and _pair_roles(game) is None:
        sexes = {f.sex for f in game.flies}
        both = f"both flies are {sexes.pop()}" if len(sexes) == 1 else "neither fly has a sex"
        return (f"the pair courtship scenario needs a male and a female in the dish, and here {both} "
                "(start the game with --partner female, or with --female --partner male)")
    return None


def _pair_start(game):
    """Place the male and the female as the courtship scenario does (he behind and to her left, she ahead facing
    wherever her own dice say) and start the counts afresh. The flies' senses this tick are the tick before's
    (Game.tick runs the actions, then the measure), so the measure skips them (``fresh``)."""
    roles = _pair_roles(game)
    if roles is None:
        raise ValueError(pair_available(game, "pair_courtship"))
    male, female = roles
    _reset_store(game)
    game.world.clear("all")
    female.body.reset(14.0, 10.0, female.rng.uniform(-math.pi, math.pi))
    male.body.reset(-10.0, -8.0, 0.3)
    game.scenario.store.update(fresh=True, ticks=0, near=0, sang=0, taps=0, touching=False, hear_sum=0.0, hear_n=0,
                               dnp37_sum=0.0, dnp13_sum=0.0, vpoen_sum=0.0, v_sang=[0.0, 0], v_quiet=[0.0, 0],
                               bursts=[0, 0], gf_prev=[0, 0])


def _pair_measure(game):
    """What the two flies are doing, this tick and so far (numbers only: none of them is a verdict). Runs at the
    start of each tick, so it reads the last completed tick."""
    from .game import GF_BURST
    roles = _pair_roles(game)
    if roles is None:
        return {}
    m, f = roles                                           # the male and the female, whichever index each has
    st = game.scenario.store
    if "ticks" not in st:                                  # started by hand, without the step's action
        _pair_start(game)
    d = math.hypot(f.body.pose.x - m.body.pose.x, f.body.pose.y - m.body.pose.y)
    if st.pop("fresh", False):                             # the tick the flies were placed in: its senses are stale
        return {"distance_mm": round(d, 1), "he_sings": False, "she_hears_hz": 0.0, "her_DNp37_hz": 0.0,
                "her_DNp13_hz": 0.0, "her_vpoEN_hz": 0.0, "her_speed_mm_s": 0.0, "taps": 0, "bursts_male": 0,
                "bursts_female": 0}
    st["ticks"] += 1
    sings = m.m["song"] > 0.3
    st["near"] += d < PAIR_NEAR_MM
    st["sang"] += sings
    touching = "touches_fly" in m.senses_now
    st["taps"] += touching and not st["touching"]
    st["touching"] = touching
    hear = f.senses_now.get("hears_song", 0.0)
    if hear:
        st["hear_sum"] += hear
        st["hear_n"] += 1
    hz = f.bt.hz
    st["dnp37_sum"] += hz.get("DNp37", 0.0)
    st["dnp13_sum"] += hz.get("DNp13", 0.0)
    st["vpoen_sum"] += hz.get("vpoEN", 0.0)
    if f.body.pose.jump is None:                           # a jump's flight is not walking
        acc = st["v_sang"] if sings else st["v_quiet"]
        acc[0] += abs(f.body.pose.v)
        acc[1] += 1
    for k, a in enumerate((m, f)):                         # the game's own escape rule: a burst over two ticks
        if a.bt.gf + st["gf_prev"][k] >= GF_BURST:
            st["bursts"][k] += 1
        st["gf_prev"][k] = a.bt.gf
    return {"distance_mm": round(d, 1), "he_sings": sings, "she_hears_hz": round(hear, 1),
            "her_DNp37_hz": round(hz.get("DNp37", 0.0), 1), "her_DNp13_hz": round(hz.get("DNp13", 0.0), 1),
            "her_vpoEN_hz": round(hz.get("vpoEN", 0.0), 1), "her_speed_mm_s": round(abs(f.body.pose.v), 1),
            "taps": st["taps"], "bursts_male": st["bursts"][0], "bursts_female": st["bursts"][1]}


def _pair_summary(game):
    """The numbers of the 90 s, into the event log (docs/SCIENCE.md 10.5 reports them over five seeds)."""
    from .game import TICK_MS
    st = game.scenario.store
    n = max(1, st.get("ticks", 0))
    secs = n * TICK_MS / 1000.0
    mean = lambda acc: acc[0] / acc[1] if acc[1] else 0.0
    game.events.add(game.t, "scenario",
                    f"two-fly courtship finished: within {PAIR_NEAR_MM:g} mm {100 * st.get('near', 0) / n:.0f} % of {secs:.0f} s; "
                    f"he sang {st.get('sang', 0) * TICK_MS / 1000:.1f} s and tapped her {st.get('taps', 0)} times "
                    f"({st.get('taps', 0) / secs * 60:.1f}/min); she heard {st.get('hear_sum', 0.0) / max(1, st.get('hear_n', 0)):.1f} Hz "
                    f"on her Johnston's organ for {100 * st.get('hear_n', 0) / n:.0f} % of the time; her DNp37 {st.get('dnp37_sum', 0.0) / n:.1f} Hz, "
                    f"DNp13 {st.get('dnp13_sum', 0.0) / n:.1f} Hz, vpoEN {st.get('vpoen_sum', 0.0) / n:.1f} Hz; her speed "
                    f"{mean(st.get('v_sang', [0, 0])):.1f} mm/s while he sang, {mean(st.get('v_quiet', [0, 0])):.1f} while he did not; "
                    f"giant-fibre bursts: his {st.get('bursts', [0, 0])[0]}, hers {st.get('bursts', [0, 0])[1]}")


PAIR_SCENARIOS["pair_courtship"] = Scenario(
    "pair_courtship", "Courtship, two brains",
    "The male and the female (whichever of the two is the partner), left to their brains for 90 s: he is placed behind "
    "and to her left, she ahead with a heading of her own. The measure shows their distance, whether he sings, what she hears, her decision "
    "neurons (vpoEN, DNp37, DNp13: readouts, not verdicts), her speed, his taps and each fly's giant-fibre bursts; the "
    "end logs the totals. Her walking is the hand-built walking urge, so compare with the same run with it off "
    "(docs/SCIENCE.md 10.5).",
    [
        Step("Two brains, one dish: 90 s.", 90.0, _pair_start, _pair_measure),
        Step("Done.", 0.0, _pair_summary),
    ])


class ScenarioRunner:
    def __init__(self, game):
        self.game = game
        self.current: Scenario | None = None
        self.step_i = 0
        self.left = 0.0
        self.store: dict = {}
        self.measure: dict = {}
        self.saved_tool: str | None = None

    def start(self, sid: str):
        if self.current is not None:
            self.stop(silent=True)
        sc = SCENARIOS.get(sid)
        if sc is None and sid in PAIR_SCENARIOS:           # Game.action refuses these first; this is for callers in code
            why = pair_available(self.game, sid)
            if why is not None:
                raise KeyError(why)
            sc = PAIR_SCENARIOS[sid]
        if sc is None:
            raise KeyError(f"unknown scenario {sid}")
        self.current = sc
        self.step_i = -1
        self.store = {}
        self.saved_tool = self.game.world.tool
        self.game.world.tool = "none"
        self.game.events.add(self.game.t, "scenario", f"scenario started: {self.current.name}")
        self._next()

    def stop(self, silent: bool = False):
        if self.current is not None and not silent:
            self.game.events.add(self.game.t, "scenario", f"scenario stopped: {self.current.name}")
        self._finish()

    def _finish(self):
        """Give the player back their tool and take the scripted hand away (whether stopped or run to the end)."""
        self.current = None
        self.store = {}
        self.measure = {}
        w = self.game.world
        if w.tool == "hand" or w.tool == "none":          # only what the scenario set; keep a tool the player chose since
            w.tool = self.saved_tool if self.saved_tool not in (None, "none") else "lure"
        w.hand = None
        self.saved_tool = None

    def _next(self):
        if self.current is not None and self.step_i >= 0 and self.current.steps[self.step_i].measure is not None \
                and "preference_index" in self.measure:          # log each period's preference before the next wipes it
            m, (a, b) = self.measure, self.store.get("pair", ("A", "B"))
            label = self.current.steps[self.step_i].caption.split(":")[0]
            self.game.events.add(self.game.t, "scenario", f"{label}: preference index {m['preference_index']:+.2f} "
                                 f"({a} {m['near_a_s']:g} s, {b} {m['near_b_s']:g} s)")
        self.step_i += 1
        self.measure = {}
        if self.current is None or self.step_i >= len(self.current.steps):
            if self.current is not None:
                self.game.events.add(self.game.t, "scenario", f"scenario finished: {self.current.name}")
            self._finish()
            return
        st = self.current.steps[self.step_i]
        self.left = st.secs
        self.game.say(st.caption, min(6.0, max(2.0, st.secs)))
        self.game.events.add(self.game.t, "scenario", st.caption)
        if st.action is not None:
            st.action(self.game)
        if st.secs <= 0:
            self._next()

    def step(self, dt: float):
        if self.current is None:
            return
        st = self.current.steps[self.step_i]
        g = self.game
        # scenario-driven stimuli
        shock_odour = self.store.get("shock_odour")
        if shock_odour and g.smelling == shock_odour and g.shock_left <= 0:
            g.shock_left = 0.5
        swoop = self.store.get("swoop")
        if swoop:
            k, age = swoop
            age += dt
            self.store["swoop"] = (k, age)
            p = g.body.pose
            g.world.tool = "hand"
            if k in (1, 2):
                speed = 140.0
                ang = p.h - math.pi / 2 if k == 1 else p.h
                d = max(2.0, 38.0 - speed * age)
            else:
                speed = 10.0
                ang = p.h + math.pi / 2
                d = max(2.0, 38.0 - speed * age)
            g.world.hand = (p.x + d * math.cos(ang), p.y + d * math.sin(ang))
            if d <= 2.0:
                g.world.hand = None
        if st.measure is not None:
            try:
                self.measure = st.measure(g)
            except Exception as e:
                self.measure = {"error": str(e)}
        self.left -= dt
        if self.left <= 0:
            self._next()

    def status(self):
        if self.current is None:
            return None
        st = self.current.steps[self.step_i]
        return {"id": self.current.id, "name": self.current.name, "step": self.step_i + 1,
                "steps": len(self.current.steps), "caption": st.caption, "left": round(max(0.0, self.left), 1),
                "measure": self.measure}
