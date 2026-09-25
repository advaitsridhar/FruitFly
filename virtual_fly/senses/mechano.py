"""
Mechanosensation: wind, sound, touch and dust -> Johnston's organ and bristle neurons.

* **Wind** deflects the antennae. Johnston's organ C/E neurons (``JO-C*``, ``JO-E*``) encode
  sustained deflection; wind from the fly's left pushes the left antenna more, so the two sides
  differ (Yorozu et al. 2009; Suver et al. 2019). Encoded here as rates on the left and right JO-CE
  populations proportional to airspeed at each antenna, with the difference carrying direction.
* **Sound** (courtship song, a wing tone) vibrates the antennae: ``JO-A`` / ``JO-B`` neurons.
* **Touch**: bumping the head into a wall or an obstacle fires the head bristle neurons
  (``BM_InOm``); a puff of dust fires the antennal wind/gravity and grooming sensilla
  (``subclass:wind_gravity``, ``subclass:grooming``), the stimulus of Shiu et al. 2024.
* **Self-motion**: leg proprioceptors (``mechanosensory_proprioceptive``, chordotonal organs)
  fire while the fly walks, so the nerve cord knows the legs are moving.
"""

from __future__ import annotations

import math

WIND_LEFT = "prefix:JO-C/L,prefix:JO-E/L"
WIND_RIGHT = "prefix:JO-C/R,prefix:JO-E/R"
SOUND = "prefix:JO-A,prefix:JO-B"
HEAD_BRISTLES = "prefix:BM_InOm"
DUST_SENSORS = {"subclass:wind_gravity,subclass:grooming": 150.0}
LEG_PROPRIO = "class:mechanosensory_proprioceptive&superclass:vnc_sensory"
WIND_MAX_HZ = 20.0       # above ~30 Hz the same neurons trigger grooming in this model


class Antennae:
    """Wind and sound at the head, from the world's wind field and the fly's own motion."""

    def __init__(self, world):
        self.world = world
        self.airspeed_l = 0.0
        self.airspeed_r = 0.0
        self.wind_bearing = 0.0       # where the wind comes FROM, relative to the fly (+ = left)
        self.dust_left = 0.0
        self.hearing = 0.0

    def puff_dust(self, secs: float = 1.5):
        self.dust_left = secs

    def rates(self, pose, dt: float) -> dict[str, float]:
        out: dict[str, float] = {}
        w = self.world.wind
        wx, wy = w.vec
        # airflow relative to a moving fly = wind minus the fly's own velocity
        ax = wx - pose.v * math.cos(pose.h)
        ay = wy - pose.v * math.sin(pose.h)
        speed = math.hypot(ax, ay)
        if speed > 0.5:
            # the wind comes FROM the opposite direction of the airflow vector
            from_dir = math.atan2(-ay, -ax)
            self.wind_bearing = (from_dir - pose.h + math.pi) % (2 * math.pi) - math.pi
            # each antenna is deflected most by air arriving from its own side and from the front
            lat = math.sin(self.wind_bearing)                     # +1 from the left, -1 from the right
            front = max(0.0, math.cos(self.wind_bearing))
            base = min(1.0, speed / 25.0)
            self.airspeed_l = min(1.0, base * (0.5 * front + 0.5 * max(0.0, lat) + 0.2))
            self.airspeed_r = min(1.0, base * (0.5 * front + 0.5 * max(0.0, -lat) + 0.2))
            # kept well below the 150 Hz "dust" stimulus: strong antennal deflection means grooming
            # (Hampel et al. 2015), steady wind is a weaker, sustained signal
            out[WIND_LEFT] = WIND_MAX_HZ * self.airspeed_l
            out[WIND_RIGHT] = WIND_MAX_HZ * self.airspeed_r
        else:
            self.airspeed_l = self.airspeed_r = 0.0
        if self.hearing > 0.05:
            out[SOUND] = 100.0 * min(1.0, self.hearing)
        if self.dust_left > 0:
            out.update(DUST_SENSORS)
            self.dust_left -= dt
        return out


class Bristles:
    """Touch on the head, and proprioception from walking legs.

    Bristle mechanoreceptors adapt quickly: they fire when a bristle is deflected, not for as long
    as it stays bent. So a touch fires the head bristles for 0.2 s, and a head held still against
    the wall or a post does not fire them again until it moves (0.3 mm or 0.15 rad) or ``REARM_S``
    have passed. Without this the touch drives the grooming neurons, grooming stops the fly with
    its head on the wall, the touch never ends, and the fly grooms there for ever."""

    REARM_S = 2.5

    def __init__(self, world):
        self.world = world
        self.touch_left = 0.0
        self.last_touch_t = -99.0
        self._contact = None                  # (head x, head y, heading) at the touch that last fired

    def rates(self, pose, dt: float, bumped: bool, t: float) -> dict[str, float]:
        out: dict[str, float] = {}
        hx, hy = pose.head
        near_wall = math.hypot(hx, hy) > self.world.arena_r - 0.6
        near_post = any(math.hypot(hx - o.x, hy - o.y) < o.r + 0.6 for o in self.world.obstacles)
        if bumped or near_wall or near_post:
            c = self._contact
            moved = (c is None or math.hypot(hx - c[0], hy - c[1]) > 0.3
                     or abs(math.atan2(math.sin(pose.h - c[2]), math.cos(pose.h - c[2]))) > 0.15)
            if moved or t - self.last_touch_t > self.REARM_S:
                self.touch_left = 0.2
                self.last_touch_t = t
                self._contact = (hx, hy, pose.h)
        else:
            self._contact = None
        if self.touch_left > 0:
            out[HEAD_BRISTLES] = 100.0
            self.touch_left -= dt
        if abs(pose.v) > 2.0:
            out[LEG_PROPRIO] = 30.0 * min(1.0, abs(pose.v) / 12.0)
        return out
