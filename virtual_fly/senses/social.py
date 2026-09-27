"""
How one simulated fly reaches another: the social encoders (docs/TWO_FLIES_PLAN.md 5.5).

Two brains in one dish are coupled only through the world, never synapse to synapse: a fly is *seen* by the other's
retina, *heard* through its Johnston's organ, *tasted* by a foreleg, and its body is *in the way*. Every channel
here is hand-built (an encoder someone chose, with constants chosen by hand) and switchable; what happens
downstream of the sensory neurons it drives is the connectome's. Each channel does nothing when there are no
other flies, so single-fly play is unchanged.

Units matter, because the drawn fly is about three times real size while the dish and the speeds are in real
millimetres (world.py): every distance constant below says which scale it is in.

  seen      the other fly is a small dark object to the retina (senses/vision.py), like the scripted female
  song      the singer's decoded song drives the hearer's JO-A/JO-B at up to SONG_MAX_HZ, fading with distance
  contact   a foreleg tip within CONTACT_MM of the other fly: the toucher's leg taste cells (when its file has
            them) and, for a male toucher, the kit's hand-built pC1 arousal (game.py COURTSHIP_*)
  collide   the drawn bodies are capsules that cannot walk through each other
  cva       (off) the male's pheromone cVA reaches the female's ORN_DA1 within CVA_MM
  mating    (off) "virgin" drives her SAG neurons steadily, "mated" silences them
  touch     (off) bumping into the other fly reaches the head bristles, as a wall bump does
  cues      whether her decision neurons' rates are drawn on her abdomen (the page; readout displays, never a
            verdict: docs/TWO_FLIES_PLAN.md 3.2)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from ..body import FLY_CAPSULE_HALF, FLY_CAPSULE_R, capsule_overlap, capsule_points

# ---------------------------------------------------------------------------------------------- the constants
# Song. Hand-built calibration taken from the male connectome and the burst rule: the loudest steady JO-A/JO-B
# drive at which fewer than 1 % of 50 ms windows hold GF_BURST or more DNp01 spikes (docs/TWO_FLIES_PLAN.md 5.9
# item 1, decision 6). Measured with tools/song_startle.py on 2026-09-27 (game profile, seeds 0-4, 5 s per rate,
# 10-100 Hz in steps of 10): 70 Hz gives 0.50 % of windows a burst with the parts list off and 0.80 % with it on
# (every burst window exactly 5 spikes); 80 Hz gives 1.21 % and 2.21 %. The female's giant fibre never bursts under
# the same drive (largest window 4). See docs/SCIENCE.md, the two-flies section.
SONG_MAX_HZ = 70.0
# The song's reach, in REAL centre-to-centre millimetres: full within SONG_NEAR_MM, fading to nothing at
# SONG_FAR_MM (hand-built, provisional, no source: decision 8).
SONG_NEAR_MM, SONG_FAR_MM = 6.0, 15.0
# Contact: a foreleg tip within this distance of the other fly's centre, at the DRAWN scale (the single fly's
# rule for the scripted female, senses/taste.py Forelegs).
CONTACT_MM = 3.4
# cVA (off by default): ORNs respond reliably within about 5 mm of a male (Taisz et al. 2023), a REAL
# centre-to-centre distance; the rate is a provisional placeholder.
CVA_MM, CVA_MAX_HZ = 5.0, 40.0
# Mating status (off by default): the tonic drive on the SAG neurons (AN_SMP_2) that stands for "virgin";
# a provisional placeholder rate until a sweep chooses it (docs/TWO_FLIES_PLAN.md 5.5 channel 6).
SAG_SPEC, SAG_HZ = "AN_SMP_2", 30.0
# The capsule each drawn fly is for collisions lives in body.py (FLY_CAPSULE_HALF, FLY_CAPSULE_R: drawn scale).

CHANNELS = ("seen", "song", "contact", "collide", "cva", "mating", "touch", "cues")
DEFAULT_CHANNELS = "seen,song,contact,collide"


@dataclass
class SocialConfig:
    """One switch per channel. ``contact_pc1``: whether tapping a simulated partner also drives the toucher's pC1
    directly (the kit's hand-built arousal, game.py COURTSHIP_*): None = by sex, on for a male toucher and off for
    a female toucher (decision 14); the scripted-female path of single-fly play is untouched either way (D7)."""
    seen: bool = True
    song: bool = True
    contact: bool = True
    collide: bool = True
    cva: bool = False
    mating: str = "none"            # "none", "virgin" or "mated"
    touch: bool = False
    cues: bool = True
    contact_pc1: bool | None = None

    @classmethod
    def from_list(cls, text: str | None) -> "SocialConfig":
        """The ``--social`` list: the channels that are on (``mating`` as ``mating:virgin`` or ``mating:mated``).
        A misspelt member is refused by name."""
        if text is None:
            text = DEFAULT_CHANNELS
        on: dict = {c: False for c in CHANNELS if c != "mating"}
        mating = "none"
        for raw in str(text).split(","):
            name = raw.strip()
            if not name:
                continue
            key, _, value = name.partition(":")
            if key == "mating":
                if value not in ("virgin", "mated"):
                    raise ValueError(f"unknown social channel '{name}': mating takes mating:virgin or mating:mated")
                mating = value
            elif key in on and not value:
                on[key] = True
            else:
                raise ValueError(f"unknown social channel '{name}'; the channels are {', '.join(CHANNELS)}")
        return cls(mating=mating, **on)

    def names(self) -> list[str]:
        """The channels that are on, as a comma list would say them."""
        out = [c for c in CHANNELS if c != "mating" and getattr(self, c)]
        if self.mating != "none":
            out.append(f"mating:{self.mating}")
        return out

    def contact_pc1_for(self, sex: str) -> bool:
        return self.contact_pc1 if self.contact_pc1 is not None else sex != "female"


def falloff(d: float, near: float = SONG_NEAR_MM, far: float = SONG_FAR_MM) -> float:
    """1 within ``near`` mm, straight down to 0 at ``far`` mm (real centre-to-centre mm; hand-built, provisional)."""
    if d <= near:
        return 1.0
    if d >= far:
        return 0.0
    return (far - d) / (far - near)


def distance(a, b) -> float:
    """Real centre-to-centre millimetres between two poses (anything with x and y)."""
    return math.hypot(a.x - b.x, a.y - b.y)


# ---------------------------------------------------------------------------------------------- the encoders
def song_rate(pose, others) -> float:
    """Channel 2, what this fly hears: the loudest other fly's song (its decoded song, 0..1) times the falloff with
    the real centre-to-centre distance, times SONG_MAX_HZ, as a steady Poisson rate on every JO-A/JO-B cell
    (decision 7: a steady drive while it sings). A fly never hears itself (it is not among ``others``).
    Hand-built."""
    best = 0.0
    for o in others:
        if o.song > 0.0:
            best = max(best, min(1.0, o.song) * falloff(distance(pose, o)) * SONG_MAX_HZ)
    return best


def touching(pose, others, reach: float = CONTACT_MM):
    """Channel 3: the first other fly whose centre is within ``reach`` (drawn scale) of one of this fly's foreleg
    tips (body.Pose.forelegs, the single fly's rule for the scripted female), or None. Hand-built."""
    for o in others:
        for (lx, ly) in pose.forelegs:
            if math.hypot(lx - o.x, ly - o.y) < reach:
                return o
    return None


def cva_rate(pose, others) -> float:
    """Channel 5 (off by default): the male pheromone cVA reaching this fly's ORN_DA1 from the nearest male within
    CVA_MM (real centre-to-centre mm), fading straight to nothing at that distance. Hand-built, provisional."""
    best = 0.0
    for o in others:
        if o.sex != "female":
            d = distance(pose, o)
            if d < CVA_MM:
                best = max(best, CVA_MAX_HZ * (1.0 - d / CVA_MM))
    return best


def nearest(pose, others):
    """The other fly closest to this one, or None."""
    return min(others, key=lambda o: distance(pose, o)) if others else None


# ---------------------------------------------------------------------------------------------- collisions (channel 4)
def resolve_overlaps(flies):
    """After every fly has moved: any two drawn bodies still overlapping (each refused to step into the other's
    START-OF-TICK capsule, but both may have stepped into the same gap) are pushed apart symmetrically, each by
    half the overlap along the line between the closest points of their capsules, then kept inside the dish.
    Symmetric, so the flies' order in the list does not matter (D4). Hand-built world geometry, drawn scale."""
    drawn = [f for f in flies if getattr(f, "body_kind", "drawn") == "drawn"]
    for i in range(len(drawn)):
        for j in range(i + 1, len(drawn)):
            pa, pb = drawn[i].body.pose, drawn[j].body.pose
            gap, (ax, ay), (bx, by) = capsule_overlap((pa.x, pa.y, pa.h), (pb.x, pb.y, pb.h))
            if gap >= 0:
                continue
            dx, dy = bx - ax, by - ay
            d = math.hypot(dx, dy)
            if d < 1e-9:                                  # the axes cross: push along the line between the centres
                dx, dy = pb.x - pa.x, pb.y - pa.y
                d = math.hypot(dx, dy)
                if d < 1e-9:
                    dx, dy, d = math.cos(pa.h + math.pi / 2), math.sin(pa.h + math.pi / 2), 1.0
            ux, uy = dx / d, dy / d
            half = -gap / 2.0
            pa.x, pa.y = pa.x - ux * half, pa.y - uy * half
            pb.x, pb.y = pb.x + ux * half, pb.y + uy * half
            drawn[i].body._keep_inside()
            drawn[j].body._keep_inside()


def capsules_of(others) -> list:
    """The other flies' collision capsules (x, y, h) from their start-of-tick poses (drawn bodies only)."""
    return [(o.x, o.y, o.h) for o in others if getattr(o, "body_kind", "drawn") == "drawn"]


__all__ = ["SocialConfig", "CHANNELS", "DEFAULT_CHANNELS", "SONG_MAX_HZ", "SONG_NEAR_MM", "SONG_FAR_MM", "CONTACT_MM",
           "CVA_MM", "CVA_MAX_HZ", "SAG_SPEC", "SAG_HZ", "FLY_CAPSULE_HALF", "FLY_CAPSULE_R", "falloff", "distance",
           "song_rate", "touching", "cva_rate", "nearest", "resolve_overlaps", "capsules_of", "capsule_points"]
