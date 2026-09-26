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
             lambda g: {**_preference(g), "depressed": g.brain.plasticity.depressed_fraction() if g.brain.plasticity else 0}),
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
             lambda g: {**_preference(g, "banana", "yeast"), "depressed": g.brain.plasticity.depressed_fraction() if g.brain.plasticity else 0}),
        Step("Test: no more shocks. Does it now avoid banana?", 40.0,
             lambda g: (_reset_store(g), _place_two_odours(g, "banana", "yeast"), g.scenario.store.pop("shock_odour", None)),
             lambda g: _preference(g, "banana", "yeast")),
        Step("Done. Compare the baseline and test preference indices in the event log.", 0.0,
             lambda g: g.events.add(g.t, "scenario", f"aversive conditioning finished: test preference {_preference(g, 'banana', 'yeast').get('preference_index', 0):+.2f}")),
    ]))

_add(Scenario(
    "courtship", "Courtship",
    "A female enters the dish. The male sees her as a small moving object (LC10a → DNa02, chase), "
    "taps her with a foreleg (pheromone taste → pC1) and sings (pIP10, one wing out).",
    [
        Step("A female enters.", 90.0,
             lambda g: (g.world.clear("all"), g.world.toggle_female(True, 14.0, 10.0), g.body.reset(-10.0, -8.0, 0.3)),
             lambda g: {"pC1_hz": round(g.hz_shown.get("pC1", 0), 1), "song": round(g.decoder.m["song"], 2),
                        "female_receptive": round(g.world.female.receptive, 2) if g.world.female else 0}),
        Step("Done.", 0.0, lambda g: g.events.add(g.t, "scenario", "courtship scenario finished")),
    ]))

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
        self.current = SCENARIOS[sid]
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
