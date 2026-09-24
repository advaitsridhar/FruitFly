"""The arena (world.py) and the fly's body (body.py): no neurons involved."""

import math
import random

import pytest

from virtual_fly.body import ACCEL_TAU, JUMP_DIST, JUMP_TIME, FlyBody, Pose
from virtual_fly.world import ARENA_R, FLY_HALF, Female, Food, Puff, World, wrap


def drive(**kw):
    d = dict(forward=0.0, yaw=0.0, backward=0.0, halt=0.0, feed=0.0, groom=0.0, song=0.0, court=0.0)
    d.update(kw)
    return d


def test_wrap():
    assert wrap(0) == 0 and wrap(math.pi + 0.1) == pytest.approx(-math.pi + 0.1)
    assert wrap(-3 * math.pi) == pytest.approx(math.pi) or wrap(-3 * math.pi) == pytest.approx(-math.pi)
    assert all(-math.pi <= wrap(a) <= math.pi for a in (-10, -1, 0, 1, 10, 100))


def test_world_add_remove_clear():
    w = World(seed=1)
    f = w.add_food("sugar", 5, 5)
    o = w.add_obstacle(-10, 0, r=3)
    s = w.add_odour("vinegar", 0, 20, strength=0.5, food="bitter")
    assert isinstance(f, Food) and f.kind == "sugar" and f.amount == 100 and f.r == pytest.approx(3.5)
    assert o.r == 3 and s.odour == "vinegar" and s.strength == 0.5
    assert s.with_food is not None and len(w.food) == 2 and w.food[1].id == s.with_food
    assert len({f.id, o.id, s.id, w.food[1].id}) == 4                     # unique ids
    assert w.add_food("sugar", ARENA_R, 0) is None and w.add_obstacle(ARENA_R - 3, 0, r=4) is None
    assert w.add_odour("banana", 0, -ARENA_R) is None
    assert w.remove(o.id) and w.obstacles == [] and not w.remove(o.id) and not w.remove(-1)
    d = w.to_dict()
    assert {x["kind"] for x in d["food"]} == {"sugar", "bitter"} and d["odours"][0]["food"] == w.food[1].id
    assert d["wind"] == {"angle": round(math.pi, 3), "speed": 0.0} and d["female"] is None and d["hand"] is None
    w.remove(w.food[1].id)
    w.step(0.05, (0, 0))
    assert s.with_food is None                                              # the eaten drop is unlinked
    w.clear("food")
    assert w.food == [] and len(w.odours) == 1
    w.toggle_female(True)
    w.clear("all")
    assert w.odours == [] and w.puffs == [] and w.female is None


def test_wind_and_female_toggle():
    w = World()
    w.set_wind(angle=4.0, speed=999)
    assert w.wind.speed == 60 and -math.pi <= w.wind.angle <= math.pi
    w.set_wind(speed=-3)
    assert w.wind.speed == 0 and w.wind.angle == pytest.approx(wrap(4.0))
    w.set_wind(angle=0, speed=10)
    assert w.wind.vec == pytest.approx((10, 0))
    w.toggle_female(True, 3, 4)
    assert (w.female.x, w.female.y) == (3, 4)
    first = w.female
    w.toggle_female(True)
    assert w.female is first                                                # already there
    w.toggle_female(False)
    assert w.female is None
    w.toggle_female(True)
    assert math.hypot(w.female.x, w.female.y) == pytest.approx(18)


def test_puffs_are_released_and_drift_with_wind():
    w = World(seed=3)
    src = w.add_odour("vinegar", -15, 0)
    w.set_wind(angle=0, speed=20)                                          # blows toward +x
    for _ in range(60):
        w.step(0.05, (0, 0))
    assert 5 < len(w.puffs) <= 600 and all(isinstance(p, Puff) and p.odour == "vinegar" for p in w.puffs)
    assert sum(p.x for p in w.puffs) / len(w.puffs) > src.x + 5
    assert all(p.r > 1.2 and 0.02 < p.c <= 1.0 and p.age > 0 for p in w.puffs)
    old = max(w.puffs, key=lambda p: p.age)
    assert old.r > 2 and old.c < 1.0
    calm = World(seed=3)
    calm.add_odour("banana", 0, 0)
    for _ in range(60):
        calm.step(0.05, (0, 0))
    assert abs(sum(p.x for p in calm.puffs) / len(calm.puffs)) < 8         # no wind: puffs stay around
    assert w.t == pytest.approx(3.0)


def test_concentration_decays_with_distance():
    w = World()
    w.add_odour("vinegar", 0, 0, strength=1.0)
    assert w.concentration(0, 0) == {"vinegar": pytest.approx(0.8)}
    assert w.concentration(3, 0)["vinegar"] == pytest.approx(0.4)
    assert w.concentration(7, 0) == {}
    w.puffs.append(Puff(10, 10, "banana", r=2.0, c=0.5))
    c = w.concentration(10, 10)
    assert c["banana"] == pytest.approx(0.5) and "vinegar" not in c
    assert w.concentration(12, 10)["banana"] < 0.5 and "banana" not in w.concentration(20, 10)
    assert w.concentration(1, 0)["vinegar"] > w.concentration(2, 0)["vinegar"]


def test_blocked():
    w = World()
    w.add_obstacle(10, 0, r=4)
    assert not w.blocked(0, 0) and w.blocked(ARENA_R - 0.5, 0) and w.blocked(10, 0)
    assert w.blocked(10, 4 + FLY_HALF * 0.5 - 0.1) and not w.blocked(10, 4 + FLY_HALF * 0.5 + 0.1)
    assert not w.blocked(0, ARENA_R - 1, half=0)                            # a point right at the wall
    assert not w.blocked(10, 5, half=0)


def test_female_walks_and_stays_in_the_arena():
    fem = Female(random.Random(5), 30, 30)
    xs = []
    for k in range(600):
        fem.step(0.05, (-40.0, -40.0), male_singing=0.0, courted=False)
        assert math.hypot(fem.x, fem.y) < ARENA_R - 3.9
        assert -math.pi <= fem.h <= math.pi
        xs.append((fem.x, fem.y))
    assert len(set(xs)) > 100 and fem.leg_phase > 0                         # she moved
    d = fem.to_dict()
    assert set(d) == {"x", "y", "h", "legs", "receptive"} and d["receptive"] == 0
    # she moves away from a male that gets too close while not receptive, and warms up to song
    fem2 = Female(random.Random(1), 0, 0)
    for _ in range(20):
        fem2.step(0.05, (3.0, 0.0), 0.0, False)
    assert fem2.x < -1.0
    for _ in range(40):
        fem2.step(0.1, (40.0, 40.0), 1.0, True)
    assert fem2.receptive > 0.9


def test_pose_geometry():
    p = Pose(x=1, y=2, h=0)
    assert p.head == pytest.approx((1 + FLY_HALF, 2)) and p.tail == pytest.approx((1 - FLY_HALF, 2))
    (lx, ly), (rx, ry) = p.forelegs
    assert lx == rx == pytest.approx(1 + 3.2 * math.cos(math.radians(35))) and ly > 2 > ry


def test_body_walks_forward_and_turns():
    w = World()
    body = FlyBody(w, random.Random(0))
    body.reset(0, 0, 0)
    for _ in range(40):
        body.move(0.05, "walk", drive(forward=1.0))
    p = body.pose
    assert p.x > 15 and abs(p.y) < 1e-6 and p.v == pytest.approx(14, abs=0.5) and body.distance == pytest.approx(p.x)
    assert p.leg_phase > 0 and p.mode == "walk" and not body.bumped
    d = body.to_dict()
    assert d["dist"] == round(body.distance, 1) and d["hx"] == pytest.approx(p.x + FLY_HALF, abs=1e-3) and d["jump"] is None
    body.reset(0, 0, 0)
    for _ in range(6):
        body.move(0.05, "walk", drive(forward=0.5, yaw=1.0))                # + yaw = turn right (clockwise)
    assert -2.5 < body.pose.h < -0.5 and body.pose.y < 0 and body.pose.w < 0
    body.reset(0, 0, 0)
    for _ in range(20):
        body.move(0.05, "backward", drive(backward=1.0))
    assert body.pose.x < -3 and body.pose.v < 0
    body.reset(0, 0, 0)
    for _ in range(20):
        body.move(0.05, "walk", drive(forward=1.0, halt=1.0))
    assert 0 < body.pose.x < 5                                               # halting slows it down


def test_body_respects_walls_and_obstacles():
    w = World()
    body = FlyBody(w, random.Random(0))
    body.reset(0, ARENA_R - 8, math.pi / 2)                                  # facing the wall
    bumped = False
    for _ in range(80):
        body.move(0.05, "walk", drive(forward=1.0))
        bumped = bumped or body.bumped
        assert math.hypot(body.pose.x, body.pose.y) <= ARENA_R - FLY_HALF - 0.4 + 1e-6
    assert bumped and body.pose.v < 3
    w.add_obstacle(0, 6, r=4)
    body.reset(0, -6, math.pi / 2)
    bumps = 0
    for _ in range(80):
        body.move(0.05, "walk", drive(forward=1.0))
        bumps += body.bumped
    assert bumps > 10 and body.pose.y < 6 - 4 - FLY_HALF * 0.5 + 1e-6 and body.pose.y > -6
    assert math.hypot(body.pose.x, body.pose.y - 6) >= 4 + FLY_HALF * 0.5 - 1e-6   # never inside the post


def test_jump_keeps_the_fly_inside_and_spreads_the_wings():
    w = World()
    body = FlyBody(w, random.Random(0))
    body.reset(ARENA_R - 10, 0, 0)
    body.start_jump(away_from=(0.0, 0.0))                                   # would land outside the dish
    assert body.pose.jump is not None and body.jump_lock == 0.8
    assert abs(wrap(body.pose.h)) < math.radians(26)                        # heading away from the origin
    for _ in range(int(JUMP_TIME / 0.02) + 2):
        body.move(0.02, "escape", drive())
        assert math.hypot(body.pose.x, body.pose.y) <= ARENA_R - FLY_HALF - 0.4 + 1e-6
    assert body.pose.jump is None and body.pose.wing_l > 0.8 and body.pose.wing_r > 0.8
    body.reset(0, 0, 0)
    body.start_jump(None)
    assert abs(wrap(body.pose.h - math.pi)) < math.radians(26)              # no threat: jump backwards
    dx, dy, left = body.pose.jump
    assert math.hypot(dx, dy) == pytest.approx(JUMP_DIST) and left == JUMP_TIME


def test_appendages():
    w = World()
    body = FlyBody(w, random.Random(0))
    for _ in range(20):
        body.move(0.05, "feed", drive(feed=1.0))
    assert body.pose.proboscis > 0.95 and body.pose.v == pytest.approx(0, abs=1e-6)
    for _ in range(20):
        body.move(0.05, "walk", drive())
    assert body.pose.proboscis < 0.05
    for _ in range(20):
        body.move(0.05, "court", drive(song=1.0, song_side=-1.0, abdomen=1.0))
    assert body.pose.wing_l > 0.9 and body.pose.wing_r < 0.05 and body.pose.abdomen > 0.9
    for _ in range(20):
        body.move(0.05, "court", drive(song=1.0, song_side=1.0))
    assert body.pose.wing_r > 0.9 and body.pose.wing_l < 0.05
    g0 = body.pose.groom_phase
    body.move(0.05, "groom", drive(groom=1.0))
    assert body.pose.groom_phase > g0
