"""
Taste: what is under the mouthparts and the forelegs -> gustatory receptor neurons (GRNs).

* **Sugar** drives the labellar sugar cells (``LB3b``, ``LB3c``), the pharyngeal sugar cells
  (``PhG1a-c``, active once the proboscis is out and the fly is pumping) and leg sugar cells
  (``LgLG3``), at the rates of Shiu et al. 2024 scaled by concentration and by *hunger*: a starved
  fly's sugar GRNs respond more strongly (dopamine and NPF act on them; Inagaki et al. 2012).
* **Bitter** drives the labellar bitter cells ``LB1a-d``.
* **Water** drives the labellar water cells, ``LB3a``, only when the fly is thirsty. The MaleCNS data do
  not say which cells sense water; LB3a is the type whose outputs match the published model's water cells
  (Shiu et al. 2024, which FlyWire types as LB3 with the sugar cells), and like them at 80 Hz it drives
  Fudog (DNg67) and not MN9, so in this wiring a thirsty fly tastes water but does not drink. ``LB2a-c``, the
  earlier choice, are FlyWire's low-salt cells (the published model's Ir94e list). docs/SCIENCE.md 2.2.
* **Pheromones**: a male's foreleg tarsi carry contact chemoreceptors that detect the female's
  cuticular hydrocarbons when he taps her. Which leg GRN types carry that signal into the courtship
  circuit was measured from the wiring (see :mod:`virtual_fly.game`, ``PHEROMONE_GRNS``).
"""

from __future__ import annotations

import math

SUGAR_GRNS = {"LB3b,LB3c": 120.0, "PhG1a,PhG1b,PhG1c": 100.0, "LgLG3": 80.0}   # labellar, pharyngeal, leg
BITTER_GRNS = {"LB1a,LB1b,LB1c,LB1d": 120.0}
WATER_GRNS = {"LB3a": 80.0}


class Mouth:
    """Detects food drops touching the head, scaled by internal state."""

    def __init__(self, world):
        self.world = world
        self.touching: dict[str, float] = {}     # kind -> amount touched this tick

    def rates(self, pose, hunger: float, thirst: float, proboscis: float) -> dict[str, float]:
        hx, hy = pose.head
        out: dict[str, float] = {}
        self.touching = {}
        for f in self.world.food:
            if math.hypot(hx - f.x, hy - f.y) < f.r + 0.8:
                self.touching[f.kind] = f.amount
                if f.kind == "sugar":
                    gain = 0.6 + 0.6 * hunger                   # a full fly tastes sugar half as keenly
                    for spec, hz in SUGAR_GRNS.items():
                        if spec.startswith("PhG") and proboscis < 0.3:
                            continue                            # pharyngeal cells need the proboscis out
                        out[spec] = max(out.get(spec, 0.0), hz * gain)
                elif f.kind == "bitter":
                    for spec, hz in BITTER_GRNS.items():
                        out[spec] = max(out.get(spec, 0.0), hz)
                elif f.kind == "water" and thirst > 0.2:
                    for spec, hz in WATER_GRNS.items():
                        out[spec] = max(out.get(spec, 0.0), hz * thirst)
        return out


class Forelegs:
    """Contact chemosensation on the front tarsi: pheromones on another fly, and food on the floor."""

    def __init__(self, world, pheromone_specs: dict[str, float]):
        self.world = world
        self.pheromone_specs = pheromone_specs
        self.touching_female = False

    def rates(self, pose) -> dict[str, float]:
        out: dict[str, float] = {}
        self.touching_female = False
        fem = self.world.female
        if fem is not None and self.pheromone_specs:
            for (lx, ly) in pose.forelegs:
                if math.hypot(lx - fem.x, ly - fem.y) < 3.4:
                    self.touching_female = True
            if self.touching_female:
                for spec, hz in self.pheromone_specs.items():
                    out[spec] = max(out.get(spec, 0.0), hz)
        return out
