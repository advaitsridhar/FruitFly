"""
The fly's body (hand-built): a kinematic model with inertia, a tripod gait, and the appendages the
brain can move. It takes *motor drives* from :class:`virtual_fly.game.MotorDecoder` and produces a
pose. No muscles or legs are simulated as physics; speeds and turn rates are chosen by hand to look
like a walking fly (real flies walk at ~10-20 mm/s and turn at up to ~400 deg/s).

What is *not* hand-built is which descending neurons drive which motor field: that mapping is
measured from the connectome by the decoder (which DN reaches which motor pools) and reported in
the game's "What's real here?" panel.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .world import FLY_HALF, wrap

WALK_SPEED = 14.0       # mm/s at full forward drive
BACK_SPEED = 8.0        # mm/s when MDN drives backward walking
TURN_RATE = math.radians(300)   # rad/s at full steering drive
JUMP_DIST, JUMP_TIME = 22.0, 0.16   # an escape hop: mm and seconds
ACCEL_TAU = 0.12        # s, how fast speed follows the drive (inertia + muscle dynamics)
TURN_TAU = 0.08


@dataclass
class Pose:
    x: float = 0.0
    y: float = -12.0
    h: float = math.pi / 2
    v: float = 0.0           # forward speed mm/s (negative = backward)
    w: float = 0.0           # yaw rate rad/s
    proboscis: float = 0.0   # 0..1 extension
    wing_l: float = 0.0      # 0..1 extension (song = one wing out, vibrating)
    wing_r: float = 0.0
    leg_phase: float = 0.0   # gait cycle
    groom_phase: float = 0.0
    abdomen: float = 0.0     # 0..1 abdominal bend (courtship attempt)
    jump: list | None = None # [dx, dy, time left] during an escape hop
    mode: str = "idle"

    @property
    def head(self) -> tuple[float, float]:
        return self.x + FLY_HALF * math.cos(self.h), self.y + FLY_HALF * math.sin(self.h)

    @property
    def tail(self) -> tuple[float, float]:
        return self.x - FLY_HALF * math.cos(self.h), self.y - FLY_HALF * math.sin(self.h)

    @property
    def forelegs(self) -> tuple[tuple[float, float], tuple[float, float]]:
        """Where the front tarsi touch down (used for contact chemosensation)."""
        out = []
        for side in (1, -1):
            a = self.h + side * math.radians(35)
            out.append((self.x + 3.2 * math.cos(a), self.y + 3.2 * math.sin(a)))
        return out[0], out[1]


class FlyBody:
    """Turns motor drives into movement, with a bit of inertia and a gait cycle."""

    def __init__(self, world, rng):
        self.world, self.rng = world, rng
        self.pose = Pose()
        self.jump_lock = 0.0
        self.distance = 0.0            # total path length walked (mm)
        self.bumped = False            # did the last step hit the wall or an obstacle?

    def reset(self, x=0.0, y=-12.0, h=math.pi / 2):
        self.pose = Pose(x=x, y=y, h=h)
        self.jump_lock = 0.0
        self.distance = 0.0

    def start_jump(self, away_from: tuple[float, float] | None):
        p = self.pose
        if away_from is not None:
            away = math.atan2(p.y - away_from[1], p.x - away_from[0])
        else:
            away = p.h + math.pi
        away += math.radians(self.rng.uniform(-25, 25))
        p.jump = [JUMP_DIST * math.cos(away), JUMP_DIST * math.sin(away), JUMP_TIME]
        p.h = away
        p.v, p.w = 0.0, 0.0
        self.jump_lock = 0.8

    def move(self, dt: float, mode: str, drive: dict, wander_yaw: float = 0.0):
        """Advance the body. ``drive`` has forward, yaw (+ = right), backward, halt, feed, groom,
        song (0..1). ``mode`` is the winning behaviour chosen by the decoder."""
        p = self.pose
        self.jump_lock -= dt
        self.bumped = False
        p.mode = mode
        if p.jump is not None:
            f = dt / JUMP_TIME
            p.x += p.jump[0] * f
            p.y += p.jump[1] * f
            p.jump[2] -= dt
            if p.jump[2] <= 0:
                p.jump = None
            self._keep_inside()
            self._appendages(dt, mode, drive)
            return
        target_v, target_w = 0.0, 0.0
        if mode == "backward":
            target_v = -BACK_SPEED * min(1.0, drive["backward"])
            target_w = -TURN_RATE * 0.5 * drive["yaw"]
        elif mode in ("walk", "idle", "court"):
            steer = drive["yaw"] + wander_yaw
            steer = 0.0 if abs(steer) < 0.05 else max(-1.0, min(1.0, steer))
            target_v = WALK_SPEED * drive["forward"] * (1 - 0.7 * drive["halt"])
            target_w = -TURN_RATE * steer * (0.6 + 0.4 * (1 - min(1.0, abs(target_v) / WALK_SPEED)))
        # inertia: speed and turn rate follow their targets with a short lag
        p.v += (target_v - p.v) * min(1.0, dt / ACCEL_TAU)
        p.w += (target_w - p.w) * min(1.0, dt / TURN_TAU)
        p.h = wrap(p.h + p.w * dt)
        nx = p.x + p.v * math.cos(p.h) * dt
        ny = p.y + p.v * math.sin(p.h) * dt
        hx, hy = nx + FLY_HALF * math.cos(p.h), ny + FLY_HALF * math.sin(p.h)
        tx, ty = nx - FLY_HALF * math.cos(p.h), ny - FLY_HALF * math.sin(p.h)
        if not (self.world.blocked(hx, hy, 0) or self.world.blocked(tx, ty, 0) or self.world.blocked(nx, ny)):
            self.distance += math.hypot(nx - p.x, ny - p.y)
            p.x, p.y = nx, ny
        else:
            self.bumped = True
            p.v *= 0.2
        p.leg_phase += abs(p.v) * dt * 1.4 + abs(p.w) * dt * 2.0
        self._keep_inside()
        self._appendages(dt, mode, drive)

    def _keep_inside(self):
        p = self.pose
        r = math.hypot(p.x, p.y)
        lim = self.world.arena_r - FLY_HALF - 0.4
        if r > lim:
            k = lim / r
            p.x, p.y = p.x * k, p.y * k
        for o in self.world.obstacles:
            d = math.hypot(p.x - o.x, p.y - o.y)
            lim = o.r + FLY_HALF * 0.5
            if 0 < d < lim:
                p.x = o.x + (p.x - o.x) * lim / d
                p.y = o.y + (p.y - o.y) * lim / d

    def _appendages(self, dt, mode, drive):
        p = self.pose
        target = min(1.0, drive["feed"] * 1.3) if mode == "feed" else 0.0
        p.proboscis += (target - p.proboscis) * min(1.0, dt / 0.08)
        song = drive.get("song", 0.0)
        # courtship song: the wing nearer the female is extended (hand-built choice: alternate)
        side = drive.get("song_side", 1.0)
        wl = song if side < 0 else 0.0
        wr = song if side >= 0 else 0.0
        if mode == "escape" or p.jump is not None:
            wl = wr = 1.0
        p.wing_l += (wl - p.wing_l) * min(1.0, dt / 0.1)
        p.wing_r += (wr - p.wing_r) * min(1.0, dt / 0.1)
        if mode == "groom":
            p.groom_phase += dt * 9.0
        target_ab = drive.get("abdomen", 0.0)
        p.abdomen += (target_ab - p.abdomen) * min(1.0, dt / 0.2)

    def to_dict(self) -> dict:
        p = self.pose
        hx, hy = p.head
        return {"x": round(p.x, 3), "y": round(p.y, 3), "h": round(p.h, 4), "v": round(p.v, 2), "w": round(p.w, 3),
                "mode": p.mode, "prob": round(p.proboscis, 3), "legs": round(p.leg_phase, 3),
                "groom": round(p.groom_phase, 3), "wingL": round(p.wing_l, 3), "wingR": round(p.wing_r, 3),
                "abdomen": round(p.abdomen, 3),
                "jump": None if p.jump is None else round(1 - p.jump[2] / JUMP_TIME, 3),
                "hx": round(hx, 3), "hy": round(hy, 3), "dist": round(self.distance, 1)}
