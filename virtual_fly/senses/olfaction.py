"""
Smell: odour concentration at the antennae -> olfactory receptor neurons (ORNs).

Each of the fly's ~50 glomeruli collects the ORNs that express one receptor, and every odour
activates its own combination of glomeruli. The combinations below are hand-chosen from the
literature (DoOR database, Hallem & Carlson 2006; Stensmyr et al. 2012 for geosmin; Suh et al. 2004
for CO2) and deliberately coarse: four to five glomeruli per odour so that the Kenyon-cell code is
sparse but not empty (a single glomerulus barely reaches the mushroom body in this model).
The ORN types are the ``ORN_<glomerulus>`` cell types of neuPrint.

Innate valence is only used to *label* the odour on screen; what the fly does about it comes out
of the wiring plus whatever it has learned.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

ORN_MAX_HZ = 90.0       # ORN rate at saturating concentration (real ORNs: up to ~200 Hz; the
                        # model's antennal lobe is calibrated for ~80 Hz)


@dataclass(frozen=True)
class Odour:
    id: str
    name: str
    glomeruli: tuple[str, ...]
    innate: str            # "attractive" | "aversive" | "neutral" (literature label, not used by the model)
    colour: str
    note: str


ODOURS: dict[str, Odour] = {
    "vinegar": Odour("vinegar", "apple cider vinegar", ("DM1", "DM4", "VM7d", "DP1m", "VA2"), "attractive", "#f2b134",
                     "fruit esters and acetic acid: DM1 (Or42b), DM4 (Or59b), VM7d (Or42a), DP1m (Ir64a), VA2 (Or92a)"),
    "banana": Odour("banana", "banana / isoamyl acetate", ("DM2", "DM3", "VM2", "DC2", "VA6"), "attractive", "#e8d44d",
                    "fruit esters: DM2 (Or22a), DM3 (Or47a), VM2 (Or43b), DC2 (Or13a), VA6 (Or82a)"),
    "geosmin": Odour("geosmin", "geosmin (mould) + CO2", ("DA2", "V", "DL4", "DL5", "DC4"), "aversive", "#a77bff",
                     "DA2 (Or56a, geosmin: innately aversive), V (Gr21a/Gr63a, CO2), DL4 (Or49a/Or85f), DL5 (Or7a), DC4 (Ir64a acid)"),
    "yeast": Odour("yeast", "yeast / fermentation", ("DM5", "VC1", "VA1v", "DL1", "VM5d"), "attractive", "#9be15d",
                   "DM5 (Or85a), VC1 (Or33c/Or85e), VA1v (Or47b), DL1 (Or10a), VM5d (Or85b)"),
}


def sat(s: float, half: float = 0.25, n: float = 1.5) -> float:
    """Hill-type saturation, 0..1: half response at ``half``."""
    s = max(0.0, s)
    return s ** n / (half ** n + s ** n)


class Nose:
    """Samples the plume at the two antennae and returns ORN rates per glomerulus type.

    Adaptation: ORNs respond most to *changes* in concentration; a steady background fades
    (Nagel & Wilson 2011). Implemented as a slow high-pass on each antenna's signal.
    """

    def __init__(self, world, adapt_tau: float = 3.0):
        self.world = world
        self.adapt_tau = adapt_tau
        self.background: dict[str, float] = {}
        self.felt: dict[str, float] = {}       # last concentrations per odour (for the UI)
        self.left: dict[str, float] = {}
        self.right: dict[str, float] = {}

    def rates(self, pose, dt: float) -> dict[str, float]:
        # antennae sit on the head, 0.8 mm apart
        hx, hy = pose.head
        out: dict[str, float] = {}
        self.left = self.world.concentration(hx - 0.8 * math.sin(pose.h), hy + 0.8 * math.cos(pose.h))
        self.right = self.world.concentration(hx + 0.8 * math.sin(pose.h), hy - 0.8 * math.cos(pose.h))
        felt = {}
        for od in set(self.left) | set(self.right):
            c = 0.5 * (self.left.get(od, 0.0) + self.right.get(od, 0.0))
            bg = self.background.get(od, 0.0)
            self.background[od] = bg + (c - bg) * min(1.0, dt / self.adapt_tau)
            eff = max(0.0, c - 0.6 * bg)                       # adapted signal
            felt[od] = c
            odour = ODOURS.get(od)
            if odour is None or eff <= 0.01:
                continue
            hz = ORN_MAX_HZ * sat(eff)
            spec = ",".join(f"ORN_{g}" for g in odour.glomeruli)
            out[spec] = max(out.get(spec, 0.0), hz)
        for od in list(self.background):
            if od not in felt:
                self.background[od] *= math.exp(-dt / self.adapt_tau)
        self.felt = felt
        return out

    def strongest(self) -> tuple[str | None, float]:
        if not self.felt:
            return None, 0.0
        od = max(self.felt, key=self.felt.get)
        return od, self.felt[od]

    def gradient(self) -> dict[str, float]:
        """Left-minus-right concentration per odour (the fly can compare its two antennae)."""
        return {od: self.left.get(od, 0.0) - self.right.get(od, 0.0) for od in set(self.left) | set(self.right)}
