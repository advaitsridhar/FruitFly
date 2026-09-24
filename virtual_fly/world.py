"""
The arena and everything in it that is not the fly's nervous system (all hand-built).

* a round arena with optional obstacles (posts the fly can bump into and see),
* food drops (sugar, bitter, water) that shrink as they are eaten,
* odour sources that release **puffs** carried by the wind: a lightweight filament model of a
  turbulent plume (Farrell et al. 2002 style), so what the fly smells flickers the way a real
  plume does instead of being a smooth gradient,
* wind: a direction and speed that also deflect the fly's antennae,
* a second fly (a scripted female) for courtship,
* the player's pointer, which is a lure, a hand, or a dropper depending on the tool.

Coordinates are millimetres with the arena centre at the origin; angles are radians,
counter-clockwise from +x. Nothing here is simulated at the neuron level.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

ARENA_R = 50.0          # mm, radius of the round arena (a big petri dish)
FLY_HALF = 3.6          # mm from the fly's centre to its mouthparts (drawn ~3x real size so you can see it)


def wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


@dataclass
class Food:
    kind: str                       # sugar | bitter | water
    x: float
    y: float
    amount: float = 100.0
    id: int = 0

    @property
    def r(self) -> float:
        return 1.2 + 2.3 * math.sqrt(max(self.amount, 0.0) / 100.0)


@dataclass
class Obstacle:
    x: float
    y: float
    r: float = 4.0
    id: int = 0


@dataclass
class OdourSource:
    """A drop of something smelly. ``odour`` names an entry of :data:`virtual_fly.senses.olfaction.ODOURS`."""
    odour: str
    x: float
    y: float
    strength: float = 1.0
    id: int = 0
    with_food: int | None = None    # id of a food drop sitting on top of it (odour + taste together)


@dataclass
class Puff:
    """One filament of odour: a slowly growing blob drifting with the wind."""
    x: float
    y: float
    odour: str
    r: float = 1.5
    c: float = 1.0                  # concentration at its centre (relative)
    age: float = 0.0


@dataclass
class Wind:
    angle: float = 0.0              # direction the wind blows TOWARD (rad)
    speed: float = 0.0              # mm/s

    @property
    def vec(self) -> tuple[float, float]:
        return self.speed * math.cos(self.angle), self.speed * math.sin(self.angle)


class Female:
    """A scripted second fly: walks about, pauses, and moves away when the male gets too close.
    Entirely hand-built; she has no brain. Her role is to be *seen* (a small moving object), *touched*
    (foreleg contact = pheromone taste) and *courted* by the male's real circuits."""

    def __init__(self, rng: random.Random, x=15.0, y=10.0):
        self.rng = rng
        self.x, self.y, self.h = x, y, rng.uniform(-math.pi, math.pi)
        self.speed = 0.0
        self.leg_phase = 0.0
        self.walk_left = 1.0
        self.walking = True
        self.turn = 0.0
        self.receptive = 0.0        # rises while courted, decays otherwise (hand-built "interest")

    def step(self, dt: float, male_xy: tuple[float, float], male_singing: float, courted: bool):
        d = math.hypot(self.x - male_xy[0], self.y - male_xy[1])
        self.receptive += (min(1.0, male_singing) * 0.25 - 0.06 * (1 - male_singing)) * dt
        self.receptive = max(0.0, min(1.0, self.receptive))
        self.walk_left -= dt
        if self.walk_left <= 0:
            self.walking = not self.walking or self.rng.random() < 0.4
            self.walk_left = self.rng.uniform(0.8, 2.5) if self.walking else self.rng.uniform(0.5, 2.0)
            self.turn = self.rng.gauss(0, 1.2)
        target = 0.0
        if self.walking:
            target = 9.0
        if d < 9.0 and self.receptive < 0.6:          # too close, not yet receptive: move away
            away = math.atan2(self.y - male_xy[1], self.x - male_xy[0])
            self.h += wrap(away - self.h) * min(1.0, dt * 6)
            target = 16.0
            self.walking = True
        elif self.walking:
            self.h += self.turn * dt
        self.speed += (target - self.speed) * min(1.0, dt / 0.15)
        nx = self.x + self.speed * math.cos(self.h) * dt
        ny = self.y + self.speed * math.sin(self.h) * dt
        if math.hypot(nx, ny) < ARENA_R - 4.0:
            self.x, self.y = nx, ny
        else:                                         # at the wall: turn toward the centre
            self.h += wrap(math.atan2(-self.y, -self.x) - self.h) * min(1.0, dt * 4)
        self.h = wrap(self.h)
        self.leg_phase += abs(self.speed) * dt * 1.4

    def to_dict(self):
        return {"x": round(self.x, 3), "y": round(self.y, 3), "h": round(self.h, 4),
                "legs": round(self.leg_phase, 3), "receptive": round(self.receptive, 3)}


class World:
    """Everything in the dish, and the physics of odour plumes and wind."""

    def __init__(self, seed: int = 0, arena_r: float = ARENA_R):
        self.rng = random.Random(seed)
        self.arena_r = arena_r
        self.food: list[Food] = []
        self.obstacles: list[Obstacle] = []
        self.odours: list[OdourSource] = []
        self.puffs: list[Puff] = []
        self.wind = Wind(angle=math.pi, speed=0.0)
        self.stripes = 0                # vertical stripes painted on the wall (0 = plain grey wall)
        self.stripe_phase = 0.0         # rotation of the stripe pattern (rad)
        self.drum_speed = 0.0           # rad/s: the stripe "drum" rotates (the classic optomotor stimulus)
        self.female: Female | None = None
        self.hand: tuple[float, float] | None = None
        self.tool = "lure"
        self._next_id = 1
        self.t = 0.0

    def next_id(self) -> int:
        self._next_id += 1
        return self._next_id

    # ------------------------------------------------------------------ placing things
    def inside(self, x: float, y: float, margin: float = 2.0) -> bool:
        return math.hypot(x, y) < self.arena_r - margin

    def add_food(self, kind: str, x: float, y: float, amount: float = 100.0) -> Food | None:
        if not self.inside(x, y):
            return None
        f = Food(kind, x, y, amount, self.next_id())
        self.food.append(f)
        return f

    def add_obstacle(self, x: float, y: float, r: float = 4.0) -> Obstacle | None:
        if not self.inside(x, y, r + 1):
            return None
        o = Obstacle(x, y, r, self.next_id())
        self.obstacles.append(o)
        return o

    def add_odour(self, odour: str, x: float, y: float, strength: float = 1.0, food: str | None = None) -> OdourSource | None:
        if not self.inside(x, y):
            return None
        src = OdourSource(odour, x, y, strength, self.next_id())
        if food:
            f = self.add_food(food, x, y)
            src.with_food = f.id if f else None
        self.odours.append(src)
        return src

    def remove(self, obj_id: int) -> bool:
        for lst in (self.food, self.obstacles, self.odours):
            for k, o in enumerate(lst):
                if o.id == obj_id:
                    del lst[k]
                    return True
        return False

    def clear(self, what: str = "all"):
        if what in ("all", "food"):
            self.food = []
        if what in ("all", "obstacles"):
            self.obstacles = []
        if what in ("all", "odours"):
            self.odours = []
            self.puffs = []
        if what == "all":
            self.female = None

    def set_wind(self, angle: float | None = None, speed: float | None = None):
        if angle is not None:
            self.wind.angle = wrap(float(angle))
        if speed is not None:
            self.wind.speed = max(0.0, min(60.0, float(speed)))

    def set_stripes(self, count: int | None = None, drum_speed: float | None = None):
        if count is not None:
            self.stripes = max(0, min(48, int(count)))
        if drum_speed is not None:
            self.drum_speed = max(-6.0, min(6.0, float(drum_speed)))

    def toggle_female(self, on: bool, x: float | None = None, y: float | None = None):
        if on and self.female is None:
            if x is None:
                a = self.rng.uniform(-math.pi, math.pi)
                x, y = 18 * math.cos(a), 18 * math.sin(a)
            self.female = Female(self.rng, x, y)
        elif not on:
            self.female = None

    # ------------------------------------------------------------------ physics
    def step(self, dt: float, fly_xy: tuple[float, float], fly_singing: float = 0.0, courted: bool = False):
        self.t += dt
        if self.drum_speed:
            self.stripe_phase = wrap(self.stripe_phase + self.drum_speed * dt)
        wx, wy = self.wind.vec
        # odour sources release puffs; the release rate rises with wind (more turbulent mixing)
        for src in self.odours:
            rate = 6.0 + 0.2 * self.wind.speed                 # puffs per second
            if self.rng.random() < rate * dt and len(self.puffs) < 600:
                self.puffs.append(Puff(src.x + self.rng.gauss(0, 0.5), src.y + self.rng.gauss(0, 0.5),
                                       src.odour, r=1.2, c=src.strength))
        # puffs drift with the wind, wander (turbulence), grow and thin out
        keep = []
        for p in self.puffs:
            p.age += dt
            p.x += (wx + self.rng.gauss(0, 6.0 + 0.3 * self.wind.speed)) * dt
            p.y += (wy + self.rng.gauss(0, 6.0 + 0.3 * self.wind.speed)) * dt
            p.r += (0.9 + 0.03 * self.wind.speed) * dt
            p.c *= math.exp(-dt / 6.0)
            if p.c > 0.02 and p.age < 40 and math.hypot(p.x, p.y) < self.arena_r + 6:
                keep.append(p)
        self.puffs = keep
        # food drops on odour sources: keep them linked
        food_ids = {f.id for f in self.food}
        for src in self.odours:
            if src.with_food is not None and src.with_food not in food_ids:
                src.with_food = None
        if self.female is not None:
            self.female.step(dt, fly_xy, fly_singing, courted)

    def concentration(self, x: float, y: float) -> dict[str, float]:
        """Odour concentration at a point, per odour (sum of Gaussian puffs, plus a faint halo
        around each source so a fly standing on it always smells it)."""
        out: dict[str, float] = {}
        for p in self.puffs:
            d2 = (p.x - x) ** 2 + (p.y - y) ** 2
            if d2 < (3 * p.r) ** 2:
                out[p.odour] = out.get(p.odour, 0.0) + p.c * math.exp(-d2 / (2 * p.r * p.r))
        for src in self.odours:
            d = math.hypot(src.x - x, src.y - y)
            if d < 6.0:
                out[src.odour] = out.get(src.odour, 0.0) + src.strength * 0.8 * (1 - d / 6.0)
        return out

    def blocked(self, x: float, y: float, half: float = FLY_HALF) -> bool:
        """Would a fly centred here overlap the wall or an obstacle?"""
        if math.hypot(x, y) > self.arena_r - 0.3 - half * 0.4:
            return True
        for o in self.obstacles:
            if math.hypot(o.x - x, o.y - y) < o.r + half * 0.5:
                return True
        return False

    def to_dict(self) -> dict:
        return {
            "food": [{"id": f.id, "kind": f.kind, "x": round(f.x, 2), "y": round(f.y, 2), "r": round(f.r, 3),
                      "amount": round(f.amount, 1)} for f in self.food],
            "obstacles": [{"id": o.id, "x": o.x, "y": o.y, "r": o.r} for o in self.obstacles],
            "odours": [{"id": s.id, "odour": s.odour, "x": round(s.x, 2), "y": round(s.y, 2),
                        "strength": s.strength, "food": s.with_food} for s in self.odours],
            "puffs": [[round(p.x, 1), round(p.y, 1), round(p.r, 2), round(p.c, 2), p.odour] for p in self.puffs],
            "wind": {"angle": round(self.wind.angle, 3), "speed": round(self.wind.speed, 1)},
            "stripes": {"count": self.stripes, "phase": round(self.stripe_phase, 3), "drum_speed": round(self.drum_speed, 3)},
            "female": None if self.female is None else self.female.to_dict(),
            "hand": None if self.hand is None else [round(self.hand[0], 2), round(self.hand[1], 2)],
            "tool": self.tool,
        }
