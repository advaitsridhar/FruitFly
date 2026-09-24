"""Hand-built sensory encoders: retina and feature detectors, nose, mouth, forelegs, antennae, bristles."""

import math

import numpy as np
import pytest

from virtual_fly.body import Pose
from virtual_fly.senses.mechano import (DUST_SENSORS, HEAD_BRISTLES, LEG_PROPRIO, SOUND, WIND_LEFT, WIND_MAX_HZ,
                                        WIND_RIGHT, Antennae, Bristles)
from virtual_fly.senses.olfaction import ODOURS, ORN_MAX_HZ, Nose, sat as hill
from virtual_fly.senses.taste import BITTER_GRNS, SUGAR_GRNS, WATER_GRNS, Forelegs, Mouth
from virtual_fly.senses.vision import (COLUMNAR_AXES, WALL_HEIGHT, ColumnarMotion, Eye, FeatureDetectors, Retina,
                                       VisibleObject, sat)
from virtual_fly.world import ARENA_R, FLY_HALF, World

DT = 0.025
VINEGAR_SPEC = ",".join(f"ORN_{g}" for g in ODOURS["vinegar"].glomeruli)


# ------------------------------------------------------------------ eyes
def test_eye_geometry_and_wall():
    eye = Eye("L")
    assert eye.n == 30 * 16 and eye.grid(eye.image).shape == (30, 16)
    assert eye.az.max() > math.radians(160) and eye.az.min() < 0 and eye.facet_deg > 0
    assert Eye("R").az.min() < math.radians(-160) and Eye("R").facet_deg == pytest.approx(eye.facet_deg)
    img = eye.render(0, 0, 0, [], ARENA_R)
    assert img is eye.image and img.shape == (eye.n,)
    wall = img == 0.55                                                       # grey wall band at the horizon
    assert wall.sum() % eye.n_az == 0 and 0 < wall.sum() < eye.n and ((img == 1.0) | wall).all()
    assert eye.el[wall].min() >= math.atan2(-1.0, 50.0) and eye.el[wall].max() <= math.atan2(WALL_HEIGHT - 1, 50.0)
    assert eye.dark_mask().sum() == 0 and not eye.object_mask.any()
    eye.render(0, 0, 0, [], ARENA_R, stripes=8, stripe_phase=0.0)
    img = eye.image
    assert np.allclose(np.unique(img), [0.25, 0.85, 1.0])                     # a striped drum on the wall
    assert (img == 0.25).any() and (img == 0.85).any() and not eye.object_mask.any()
    old = eye.image
    eye.render(0, 0, 0, [], ARENA_R)
    assert eye.prev is old


def test_eye_renders_dark_objects_on_the_correct_side():
    left, right = Eye("L"), Eye("R")
    post = VisibleObject(0, 10, 4.0, 15.0, "post", 0.1)                     # 10 mm to the left of a fly facing +x
    left.render(0, 0, 0, [post], ARENA_R)
    right.render(0, 0, 0, [post], ARENA_R)
    assert left.dark_mask().sum() > 4 and right.dark_mask().sum() == 0
    assert left.image.min() == pytest.approx(0.1)
    az_dark = left.az[left.dark_mask()]
    assert math.radians(60) < az_dark.min() and az_dark.max() < math.radians(120)
    front = VisibleObject(10, 0, 4.0, 15.0, "post")
    for eye in (left, right):
        eye.render(0, 0, 0, [front], ARENA_R)
        assert eye.dark_mask().sum() > 0 and abs(eye.az[eye.dark_mask()]).max() < math.radians(25)
    # too close (on top of the fly) is skipped; the far wall stays grey, not dark
    left.render(0, 0, 0, [VisibleObject(0.1, 0.1, 4.0, 15.0, "post")], ARENA_R)
    assert left.dark_mask().sum() == 0
    # a small low object (food) only darkens facets below the horizon
    left.render(0, 0, 0, [VisibleObject(6, 0, 2.0, 0.6, "food", 0.5)], ARENA_R)
    assert (left.image == 0.5).sum() > 0 and left.el[left.image == 0.5].max() < 0


def test_sat_shapes():
    assert sat(0) == 0 and sat(1) == pytest.approx(1) and 0 < sat(0.2) < 0.6 and sat(2) == sat(1)
    assert hill(0) == 0 and hill(0.25) == pytest.approx(0.5) and hill(100) > 0.99


@pytest.fixture
def scene():
    world = World(seed=0)
    retina = Retina(world)
    pose = Pose(x=0.0, y=0.0, h=math.pi / 2)                                 # facing +y, in the middle
    return world, retina, pose


def swoop(world, retina, pose, speed, frames, side=0.0):
    """Move the hand toward the fly at ``speed`` mm/s from bearing ``side`` (rad, 0 = straight ahead)."""
    world.tool = "hand"
    out = []
    for k in range(frames):
        d = max(2.0, 38.0 - speed * k * DT)
        ang = pose.h + side
        world.hand = (pose.x + d * math.cos(ang), pose.y + d * math.sin(ang))
        rates, _, _ = retina.look(pose, DT)
        out.append((rates, dict(retina.features.felt)))
    return out


def test_fast_expansion_drives_looming_detectors(scene):
    world, retina, pose = scene
    frames = swoop(world, retina, pose, speed=140.0, frames=12, side=-math.pi / 2)   # from the right
    lc4 = [r.get("LC4/R", 0.0) for r, _ in frames]
    assert max(lc4) > 60 and all(r.get("LC4/L", 0.0) == 0 for r, _ in frames)
    k = int(np.argmax(lc4))
    rates, felt = frames[k]
    assert rates["LPLC2/R"] > 0 and rates["LPLC2/R"] <= rates["LC4/R"] and felt["loom"] == "R"
    assert not any(key.startswith("T4") for key in rates)                 # no self-motion: no wide-field flow
    assert retina.features.expansion["R"] > FeatureDetectors.LOOM_THRESHOLD_DEG
    b64 = retina.images_b64()
    assert set(b64) == {"L", "R"} and len(b64["R"]) > 100
    lay = retina.layout()
    assert lay["L"]["n_az"] == 30 and len(lay["R"]["az"]) == 30 * 16


def test_slow_approach_does_not_loom(scene):
    world, retina, pose = scene
    frames = swoop(world, retina, pose, speed=10.0, frames=40)
    assert max(r.get("LC4/L", 0.0) + r.get("LC4/R", 0.0) for r, _ in frames) < 20
    assert not any("loom" in f and len(f["loom"]) == 2 for _, f in frames)


def test_small_moving_object_drives_lc10a_on_the_seeing_eye(scene):
    world, retina, pose = scene
    world.toggle_female(True, 0.0, 0.0)
    for sign, eye in ((1, "L"), (-1, "R")):
        retina.features.small_pos = {s: None for s in retina.eyes}
        retina.features.small_speed = {s: 0.0 for s in retina.eyes}
        seen = []
        for k in range(20):
            ang = pose.h + sign * (0.9 - 0.03 * k)                            # sweeps toward the front
            world.female.x, world.female.y = pose.x + 12 * math.cos(ang), pose.y + 12 * math.sin(ang)
            rates, _, _ = retina.look(pose, DT)
            seen.append((rates, dict(retina.features.felt)))
        other = "R" if eye == "L" else "L"
        hits = [r for r, _ in seen if f"LC10a/{eye}" in r]
        assert len(hits) >= 5 and all(f"LC10a/{other}" not in r for r, _ in seen)
        assert all(0 < r[f"LC10a/{eye}"] <= 70 and r[f"LC11/{eye}"] == pytest.approx(0.6 * r[f"LC10a/{eye}"]) for r in hits)
        assert all(f["small"] == eye for _, f in seen if "small" in f)
        assert not any("LC4" in key for r, _ in seen for key in r)          # a small thing never looms
    # a stationary small object stops driving LC10a
    for _ in range(15):
        rates, _, _ = retina.look(pose, DT)
    assert not any(key.startswith("LC10a") for key in rates)


def turn(world, retina, pose, frames=10, w=4.0):
    pose.w = w
    keys, felt = set(), []
    for _ in range(frames):
        pose.h += pose.w * DT
        rates, _, _ = retina.look(pose, DT)
        keys |= set(rates)
        felt.append(dict(retina.features.felt))
    return keys, felt


def test_turning_produces_wide_field_flow_but_no_looming(scene):
    world, retina, pose = scene
    world.add_obstacle(0, 15, r=4)                                           # a post straight ahead
    keys, felt = turn(world, retina, pose)
    flow_keys = {k for k in keys if k.startswith("T4")}
    assert flow_keys and all(k.split(",")[1].startswith("T5") for k in flow_keys)
    assert any("flow" in f for f in felt) and not any("loom" in f for f in felt)
    assert not any(k.startswith("LC4") or k.startswith("LPLC2") for k in keys)
    assert max(abs(v) for v in retina.features.flow.values()) > 8


def test_object_entering_the_field_during_a_turn_does_not_loom(scene):
    world, retina, pose = scene
    world.add_obstacle(-12, 8, r=4)                                          # left-front; a left turn sweeps it right
    keys, felt = turn(world, retina, pose, frames=12)
    assert not any(k.startswith("LC4") for k in keys) and not any("loom" in f for f in felt)


def test_columnar_motion_maps_facets_to_the_hex_lattice(conn, scene):
    world, _, pose = scene
    retina = Retina(world, conn, columnar=True)
    cm = retina.columnar
    assert isinstance(cm, ColumnarMotion) and cm.eyes is retina.eyes and cm.axes is COLUMNAR_AXES
    assert Retina(world).columnar is None and Retina(world, conn).columnar is None
    for side in "LR":
        cols = cm.cols[side]
        assert cols["n_cols"] == 6 and cols["facet_col"].shape == (retina.eyes[side].n,)
        assert set(cols["facet_col"].tolist()) == set(range(6))              # every column gets facets
        assert sorted(cols["hex"].tolist()) == sorted(h1 * 1000 + h2 for h1 in range(3) for h2 in range(2))
        for name, (idx, col) in cols["neurons"].items():
            assert idx.size == 6 and set(conn.types[idx]) == {name} and set(conn.side[idx]) == {side}
            assert sorted(col.tolist()) == list(range(6))                    # home column = its inputs' column
            home = conn.hex1[idx] * 0                                        # T4/T5 have no hex of their own
            assert (conn.hex1[idx] == -1).all() and home.size == 6
    idx, hz = cm.rates(DT)
    assert idx.size == hz.size == 0                                          # nothing has moved yet
    world.add_obstacle(6, 12, r=4)
    fired = set()
    for k in range(12):
        pose.h += 0.08
        rates, idx, hz = retina.look(pose, DT)
        assert idx.size == hz.size and ((hz > 2.0).all() if hz.size else True)
        assert np.array_equal(idx, cm.last_idx) and not any(key.startswith("T4") for key in rates)
        fired |= set(idx.tolist())
    t45 = set(conn.select("prefix:T4,prefix:T5").tolist())
    assert len(fired) > 10 and fired <= t45
    # efference copy: a commanded turn damps the columnar motion signal
    retina.look(pose, DT, turn_command=1.0)
    assert cm.efference_gain == pytest.approx(0.15)
    retina.look(pose, DT, turn_command=0.0)
    assert cm.efference_gain == 1.0


# ------------------------------------------------------------------ nose
def test_nose_orn_specs_near_a_source_and_adaptation():
    world = World()
    world.add_odour("vinegar", 0.0, 0.0)
    nose = Nose(world)
    pose = Pose(x=-FLY_HALF, y=0.0, h=0.0)                                   # head exactly on the source
    rates = nose.rates(pose, DT)
    c = 0.8 * (1 - 0.8 / 6)                                                  # antennae 0.8 mm off the source
    assert list(rates) == [VINEGAR_SPEC] and rates[VINEGAR_SPEC] == pytest.approx(ORN_MAX_HZ * hill(c), rel=1e-3)
    assert nose.felt == {"vinegar": pytest.approx(c)} and nose.strongest() == ("vinegar", pytest.approx(c))
    assert nose.gradient()["vinegar"] == pytest.approx(0.0, abs=1e-9)
    hz = [rates[VINEGAR_SPEC]]
    for _ in range(8):
        hz.append(nose.rates(pose, 1.0)[VINEGAR_SPEC])
    assert all(a > b for a, b in zip(hz, hz[1:])) and hz[-1] > 0.3 * hz[0]  # adapts, never to zero
    assert nose.background["vinegar"] > 0.5 * c
    far = Pose(x=30.0, y=30.0, h=0.0)
    assert nose.rates(far, 1.0) == {} and nose.felt == {} and nose.strongest() == (None, 0.0)
    for _ in range(20):
        nose.rates(far, 1.0)
    assert nose.background["vinegar"] < 0.01 * c                             # the background fades away
    world.add_odour("perfume", 30.0, 30.0)                                   # unknown odour: felt, no ORN spec
    assert nose.rates(far, DT) == {} and nose.felt["perfume"] > 0


def test_nose_gradient_between_the_antennae():
    world = World()
    world.add_odour("banana", 0.0, 3.0)
    nose = Nose(world)
    nose.rates(Pose(x=-FLY_HALF, y=0.0, h=0.0), DT)
    assert nose.gradient()["banana"] > 0.1                                   # stronger on the left antenna
    assert nose.left["banana"] > nose.right["banana"] > 0


# ------------------------------------------------------------------ mouth and forelegs
def test_mouth_sugar_bitter_water_and_hunger():
    world = World()
    pose = Pose()
    hx, hy = pose.head
    mouth = Mouth(world)
    assert mouth.rates(pose, 1.0, 1.0, 1.0) == {} and mouth.touching == {}
    world.add_food("sugar", hx, hy)
    hungry = mouth.rates(pose, hunger=1.0, thirst=0.0, proboscis=0.0)
    assert hungry == {"LB3b,LB3c": pytest.approx(144.0), "LgLG3": pytest.approx(96.0)} and mouth.touching == {"sugar": 100}
    full = mouth.rates(pose, hunger=0.0, thirst=0.0, proboscis=0.0)
    assert full["LB3b,LB3c"] == pytest.approx(72.0)                          # half as keen
    out = mouth.rates(pose, hunger=1.0, thirst=0.0, proboscis=0.5)
    assert out["PhG1a,PhG1b,PhG1c"] == pytest.approx(120.0) and set(out) == set(SUGAR_GRNS)
    world.clear("food")
    world.add_food("bitter", hx + 0.5, hy)
    assert mouth.rates(pose, 1.0, 1.0, 1.0) == dict(BITTER_GRNS)
    world.clear("food")
    world.add_food("water", hx, hy)
    assert mouth.rates(pose, 1.0, thirst=0.1, proboscis=0.0) == {} and mouth.touching == {"water": 100}
    assert mouth.rates(pose, 1.0, thirst=0.5, proboscis=0.0) == {k: pytest.approx(0.5 * v) for k, v in WATER_GRNS.items()}
    world.add_food("sugar", hx + 10, hy)                                      # out of reach
    assert "LB3b,LB3c" not in mouth.rates(pose, 1.0, 0.0, 0.0)


def test_forelegs_taste_the_female():
    world = World()
    legs = Forelegs(world, {"LgLG4": 100.0, "LgLG1a,LgLG1b": 60.0})
    pose = Pose(x=0.0, y=0.0, h=0.0)
    assert legs.rates(pose) == {} and not legs.touching_female
    world.toggle_female(True, 20.0, 20.0)
    assert legs.rates(pose) == {} and not legs.touching_female
    lx, ly = pose.forelegs[0]
    world.female.x, world.female.y = lx + 1.0, ly
    assert legs.rates(pose) == {"LgLG4": 100.0, "LgLG1a,LgLG1b": 60.0} and legs.touching_female
    assert Forelegs(world, {}).rates(pose) == {}


# ------------------------------------------------------------------ antennae and bristles
def test_antennae_wind_asymmetry_and_dust():
    world = World()
    ant = Antennae(world)
    pose = Pose(x=0.0, y=0.0, h=math.pi / 2)                                 # facing +y: its left is -x
    assert ant.rates(pose, DT) == {} and ant.airspeed_l == ant.airspeed_r == 0
    world.set_wind(angle=0.0, speed=20.0)                                    # blows toward +x: comes from the left
    r = ant.rates(pose, DT)
    assert r[WIND_LEFT] > r[WIND_RIGHT] > 0 and r[WIND_LEFT] <= WIND_MAX_HZ
    assert math.degrees(ant.wind_bearing) == pytest.approx(90)
    world.set_wind(angle=math.pi, speed=20.0)                                # from the right
    r = ant.rates(pose, DT)
    assert r[WIND_RIGHT] > r[WIND_LEFT] and math.degrees(ant.wind_bearing) == pytest.approx(-90)
    world.set_wind(angle=-math.pi / 2, speed=20.0)                           # head wind
    r = ant.rates(pose, DT)
    assert r[WIND_LEFT] == pytest.approx(r[WIND_RIGHT]) and abs(ant.wind_bearing) < 1e-9
    world.set_wind(speed=0.0)
    pose.v = 10.0                                                            # walking makes its own head wind
    r = ant.rates(pose, DT)
    assert r[WIND_LEFT] == pytest.approx(r[WIND_RIGHT]) and r[WIND_LEFT] > 0 and abs(ant.wind_bearing) < 1e-9
    pose.v = 0.0
    ant.puff_dust(0.05)
    r = ant.rates(pose, 0.05)
    assert r == dict(DUST_SENSORS) and ant.dust_left == 0
    assert ant.rates(pose, 0.05) == {}
    ant.hearing = 0.5
    assert ant.rates(pose, DT) == {SOUND: 50.0}


def test_bristles_touch_and_proprioception():
    world = World()
    br = Bristles(world)
    pose = Pose(x=0.0, y=0.0, h=0.0)
    assert br.rates(pose, DT, bumped=False, t=1.0) == {} and br.last_touch_t == -99.0
    assert br.rates(pose, DT, bumped=True, t=1.0) == {HEAD_BRISTLES: 100.0} and br.last_touch_t == 1.0
    n = sum(HEAD_BRISTLES in br.rates(pose, 0.05, False, 1.0 + 0.05 * k) for k in range(1, 10))
    assert 2 <= n <= 4                                                       # about 0.2 s of touch
    wall = Pose(x=ARENA_R - FLY_HALF - 0.4, y=0.0, h=0.0)
    assert HEAD_BRISTLES in br.rates(wall, DT, False, 5.0)
    world.add_obstacle(10, 0, r=4)
    assert HEAD_BRISTLES in br.rates(Pose(x=10 - 4 - FLY_HALF - 0.3, y=0.0, h=0.0), DT, False, 9.0)
    br = Bristles(world)                                                      # fresh: no touch pending
    pose.v = 6.0
    assert br.rates(pose, DT, False, 20.0) == {LEG_PROPRIO: pytest.approx(15.0)}
    pose.v = -12.0
    assert br.rates(pose, DT, False, 30.0)[LEG_PROPRIO] == 30.0
    pose.v = 1.0
    assert LEG_PROPRIO not in br.rates(pose, DT, False, 40.0)
