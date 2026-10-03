"""Two physics bodies in one MuJoCo world (virtual_fly/physics_pair.py; docs/TWO_FLIES_PLAN.md 8.3-8.5). The world tests need
flygym (skipped without it, like tests/test_physics.py); the contact-set and the install-hint tests run everywhere."""
import math
import random

import pytest

from virtual_fly import physics
from virtual_fly import physics_pair as pp
from virtual_fly.senses.vision import SEEN_FLY
from virtual_fly.world import FLY_HALF, World

needs_flygym = pytest.mark.skipif(not physics.available(), reason=f"physics body unavailable (optional): {physics._IMPORT_ERROR!r}")
TICK = 0.025


def drive(**kw):
    d = dict(forward=0.0, yaw=0.0, backward=0.0, halt=0.0, feed=0.0, groom=0.0, song=0.0, court=0.0)
    d.update(kw)
    return d


def test_contact_sets_never_include_the_adhesion_segment_and_agree_with_the_retina():
    for mine, theirs, both in pp.CONTACT_SETS.values():
        assert not any(g.endswith("Tarsus5") for g in (*mine, *theirs, *both))
    mine, theirs, both = pp.CONTACT_SETS["forelegs"]
    assert set(mine) == {"Head"} | {f"{leg}{seg}" for leg in ("LF", "RF") for seg in ("Tibia", "Tarsus1", "Tarsus2", "Tarsus3", "Tarsus4")}
    assert "LWing" in theirs and "A6" in theirs and "Thorax" in both and "Head" in both
    assert pp.CONTACT_SETS["none"] == ((), (), ())
    assert SEEN_FLY["physics"] == pp.SEEN_FLY_MM                    # the retina's real-size other is the model's size
    assert pp.DRAWN_SCALE == pytest.approx(pp.REAL_FLY_MM[0] / (2 * FLY_HALF), abs=1e-3)


def test_unavailable_pair_world_names_what_failed(monkeypatch):
    monkeypatch.setattr(physics, "_IMPORT_ERROR", ImportError("No module named 'flygym'"))
    assert not pp.available()
    with pytest.raises(RuntimeError, match="flygym"):
        pp.PairWorld([(0.0, 0.0, 0.0), (5.0, 0.0, math.pi)])


def _world(contact_set="forelegs", seed=0, poses=((-3.0, 0.0, 0.0), (3.0, 0.0, math.pi))):
    return pp.PairWorld(list(poses), seed=seed, world=World(seed), rngs=[random.Random(0), random.Random(1)], contact_set=contact_set)


def _stand(w, ticks=8):
    for _ in range(ticks):
        for b in w.bodies:
            b.move(TICK, "idle", drive())
        w.advance_tick(TICK)


def _distance(w):
    a, b = w.bodies[0].pose, w.bodies[1].pose
    return math.hypot(a.x - b.x, a.y - b.y)


@needs_flygym
def test_the_world_places_both_flies_where_the_kit_says_and_moves_them_without_a_rebuild():
    w = _world()
    try:
        assert [f.name for f in w.flies] == ["fly0", "fly1"]
        assert len(w.pairs) == 233 and len(set(w.pairs)) == 233       # decision 22's set, no duplicates
        for k, (x, y, h) in enumerate([(-3.0, 0.0, 0.0), (3.0, 0.0, math.pi)]):
            pos, hd = w.thorax(k)
            assert pos[0] == pytest.approx(x, abs=1e-6) and pos[1] == pytest.approx(y, abs=1e-6)
            assert abs(math.sin(hd - h)) < 1e-6 and math.cos(hd - h) > 0
        sim = w.sim
        w.bodies[0].reset(10.0, 10.0, 1.0)                            # placing is a joint write, not a rebuild
        pos, hd = w.thorax(0)
        assert pos[0] == pytest.approx(10.0, abs=1e-6) and pos[1] == pytest.approx(10.0, abs=1e-6) and hd == pytest.approx(1.0, abs=1e-6)
        assert w.sim is sim and w.bodies[0].pose.x == 10.0 and w.bodies[0].distance == 0.0
        _stand(w)                                                     # and it stands on its legs afterwards
        pos, _ = w.thorax(0)
        assert 0.8 < pos[2] < 1.4 and abs(pos[0] - 10.0) < 0.5
    finally:
        w.close()


@needs_flygym
def test_two_flies_meet_within_half_a_second_feel_the_tap_and_do_not_stick():
    w = _world()
    try:
        _stand(w)
        assert w.contacts_between(0, 1) == 0 and w.bodies[0].touching_other is None
        first, taps = None, []
        for i in range(40):                                           # walking at each other, 3 mm apart, until they touch
            for b in w.bodies:
                b.move(TICK, "walk", drive(forward=1.0))
            w.advance_tick(TICK)
            if w.bodies[0].touching_other is not None:
                taps.append(w.bodies[0].touching_other)
            if w.contacts_between(0, 1):
                first = (i + 1) * TICK
                break
        assert first is not None and first <= 0.5
        for _ in range(2):                                            # two more ticks pressing: the tap registers with a force
            for b in w.bodies:
                b.move(TICK, "walk", drive(forward=1.0))
            w.advance_tick(TICK)
            if w.bodies[0].touching_other is not None:
                taps.append(w.bodies[0].touching_other)
        assert taps and all(j == 1 for j, _ in taps) and max(f for _, f in taps) > 0.0
        # no sticking: fly 0 backs away for 1 s from the fly it touched (face to face still: the first contact is the heads');
        # nothing holds it, the contact ends. (Two flies walking on through each other slide past, deflected by their
        # rounded heads, which is why the backing starts here and not after a second of pushing.)
        d0 = _distance(w)
        for _ in range(40):
            w.bodies[0].move(TICK, "backward", drive(backward=1.0))
            w.bodies[1].move(TICK, "idle", drive())
            w.advance_tick(TICK)
        assert _distance(w) > d0 + 2.0 and w.contacts_between(0, 1) == 0
        d = w.bodies[0].to_dict()
        assert d["physics"]["pair"] is True and d["scale"] == pp.DRAWN_SCALE and len(d["physics"]["tarsi"]) == 6
    finally:
        w.close()


@needs_flygym
def test_each_fly_feels_only_its_own_wall():
    w = _world(poses=((48.0, 0.0, 0.0), (0.0, 0.0, 0.0)))
    try:
        for _ in range(20):
            w.bodies[0].move(TICK, "walk", drive(forward=1.0))
            w.bodies[1].move(TICK, "idle", drive())
            w.advance_tick(TICK)
        assert w.bodies[0].bumped and not w.bodies[1].bumped
        assert w.touching_wall(0) and not w.touching_wall(1)
    finally:
        w.close()


@needs_flygym
def test_the_pair_world_is_deterministic():
    def run():
        w = _world(seed=3)
        try:
            for _ in range(40):
                w.bodies[0].move(TICK, "walk", drive(forward=0.8, yaw=0.3))
                w.bodies[1].move(TICK, "walk", drive(forward=0.6))
                w.advance_tick(TICK)
            return [(b.pose.x, b.pose.y, b.pose.h, b.distance) for b in w.bodies]
        finally:
            w.close()
    assert run() == run()


@needs_flygym
def test_a_game_runs_two_physics_flies_in_one_world(conn):
    from virtual_fly.game import Game
    from virtual_fly.settings import build_brain
    g = Game(build_brain(conn, "game", seed=0), seed=1, body="physics", brain_procs="off",
             partner={"conn": conn, "brain_kwargs": {"seed": 1000}, "parts": False})
    try:
        assert g.pair_world is not None and [f.body_kind for f in g.flies] == ["physics", "physics"]
        assert g.flies[0].body is g.pair_world.bodies[0] and g.flies[1].body is g.pair_world.bodies[1]
        for _ in range(4):
            g.tick()
        s = g.state_dict
        assert s["fly"]["physics"]["pair"] is True and s["flies"][1]["fly"]["physics"]["pair"] is True
        assert s["fly"]["scale"] == pp.DRAWN_SCALE and g.pair_world.wall_s > 0
        real = g.whats_real()
        assert any("ONE MuJoCo world" in line for line in real["hand_built"])
        assert any("built from a female fly" in line for line in real["hand_built"])
        with pytest.raises(ValueError, match="stride-average"):
            Game(build_brain(conn, "game", seed=0), seed=1, body="physics", stride_average=True, brain_procs="off",
                 partner={"conn": conn, "brain_kwargs": {"seed": 1000}, "parts": False})
    finally:
        g.close()
    assert g.pair_world.sim is None                                   # closed with the game
