"""The optional physics body (virtual_fly/physics.py). The drive mapping is tested everywhere; the MuJoCo body
only where flygym is installed (it is an optional dependency)."""

import math
import random
import types

import pytest

from virtual_fly import physics
from virtual_fly.physics import available, descending_drive


def drive(**kw):
    d = dict(forward=0.0, yaw=0.0, backward=0.0, halt=0.0, feed=0.0, groom=0.0, song=0.0, court=0.0)
    d.update(kw)
    return d


def test_drive_mapping():
    assert descending_drive("walk", drive(forward=0.6)) == (0.6, 0.6)
    left, right = descending_drive("walk", drive(forward=1.0, yaw=1.0))          # right turn: right legs inside
    assert (left, right) == pytest.approx((1.2, 0.4))                                # flygym's own turning pair
    assert descending_drive("walk", drive(forward=1.0, yaw=-1.0)) == pytest.approx((0.4, 1.2))
    assert descending_drive("walk", drive(forward=1.0, yaw=0.03)) == (1.0, 1.0)    # the drawn body's dead band
    assert descending_drive("walk", drive(forward=0.0, yaw=1.0)) == (0.0, 0.0)     # no stepping, no steering
    assert descending_drive("walk", drive(forward=1.0, halt=1.0)) == pytest.approx((0.3, 0.3))
    b = descending_drive("backward", drive(backward=0.8))
    assert b == pytest.approx((-0.8, -0.8))
    for mode in ("feed", "groom", "escape"):
        assert descending_drive(mode, drive(forward=1.0, yaw=1.0)) == (0.0, 0.0)


def test_unavailable_body_names_what_failed(monkeypatch):
    from virtual_fly import play
    from virtual_fly.world import World
    monkeypatch.setattr(physics, "_IMPORT_ERROR", ModuleNotFoundError("No module named 'mujoco'"))
    monkeypatch.setattr(physics, "sys", types.SimpleNamespace(version_info=(3, 13, 1)))
    reason = physics.unavailable_reason()
    assert "Python 3.12-3.14" in reason and "flygym==2.1.0" in reason and "No module named 'mujoco'" in reason
    with pytest.raises(RuntimeError, match="No module named 'mujoco'"):
        physics.make_body("physics", World(seed=1), random.Random(0))
    with pytest.raises(SystemExit, match="No module named 'mujoco'"):         # before any data is loaded
        play.main(["--body", "physics", "--no-browser"])
    monkeypatch.setattr(physics, "sys", types.SimpleNamespace(version_info=(3, 11, 9)))
    assert "this is Python 3.11" in physics.unavailable_reason() and "Python 3.12-3.14" in physics.unavailable_reason()
    monkeypatch.setattr(physics, "sys", types.SimpleNamespace(version_info=(3, 15, 0)))
    assert "this is Python 3.15" in physics.unavailable_reason()


needs_flygym = pytest.mark.skipif(not available(), reason=f"physics body unavailable (optional): {physics._IMPORT_ERROR!r}")


@needs_flygym
def test_physics_body_walks_and_stands():
    from virtual_fly.physics import PhysicsBody
    from virtual_fly.world import World
    body = PhysicsBody(World(seed=1), random.Random(0))
    y0 = body.pose.y
    for _ in range(12):                                   # 0.3 s of forward drive
        body.move(0.025, "walk", drive(forward=1.0))
    assert body.pose.y - y0 > 1.5 and abs(math.degrees(body.pose.h) - 90) < 20      # it walked where it faced
    assert body.walker.wall_pairs == 48 * len(physics.WALL_TOUCHERS) and body.walker._m.npair == 55 + body.walker.wall_pairs
    x, y = body.pose.x, body.pose.y
    for _ in range(8):
        body.move(0.025, "idle", drive())
    for _ in range(8):
        body.move(0.025, "idle", drive())
    assert math.hypot(body.pose.x - x, body.pose.y - y) < 1.0                        # and stops
    d = body.to_dict()
    assert len(d["physics"]["tarsi"]) == 6 and 0.3 < d["physics"]["z"] < 2.0


@needs_flygym
def test_game_ticks_with_physics_body(conn):
    from virtual_fly.game import Game
    from virtual_fly.settings import build_brain
    g = Game(build_brain(conn, "game", seed=0), autopilot=True, seed=1, body="physics")
    assert g.physics_levers == physics.DEFAULT_LEVERS == () and g.body.levers == ()   # nothing to switch on flygym 2.1 (SCIENCE.md 13.5)
    assert Game(build_brain(conn, "game", seed=0), seed=1).physics_levers == ()  # the drawn body has none
    for _ in range(4):
        g.tick()
    assert g.body.kind == "physics" and "physics" in g.state_dict["fly"]
    assert any("leg physics" in s for s in g.whats_real()["hand_built"])


# ---------------------------------------------------------------------------------------------- speed levers (plan 8.6)
def test_levers_are_named_checked_and_off_by_default():
    assert physics.parse_levers(None) == () and physics.parse_levers("") == () and physics.parse_levers(()) == ()
    assert physics.parse_levers("solver100, dedupe") == ("dedupe", "solver100")     # LEVERS order, whatever the input's
    assert physics.parse_levers(["dt2"]) == ("dt2",) and physics.parse_levers("none") == ()
    assert physics.DEFAULT_LEVERS == ()                       # flygym 2.1's model already is what dedupe, solver100, noslip5, noself ask for
    assert set(physics.NO_OP_LEVERS) == {"dedupe", "solver100", "noslip5", "noself"} and all(l in physics.LEVERS for l in physics.NO_OP_LEVERS)
    with pytest.raises(ValueError, match="unknown physics lever nope"):
        physics.parse_levers("dedupe,nope")
    with pytest.raises(ValueError, match="exclude each other"):
        physics.parse_levers("noslip5,noslip0")
    assert physics.lever_fly_kwargs(()) == {} and physics.lever_fly_kwargs(("noself",)) == {}
    assert physics.lever_fly_kwargs(("simple",)) == {"geom_fitting_option": "all_to_capsules"}
    assert physics.lever_timestep(()) == physics.TIMESTEP and physics.lever_timestep(("dt2",)) == 2e-4
    assert set(physics.LEVERS) == {"dedupe", "solver100", "noslip5", "noslip0", "noself", "simple", "dt2"}


@needs_flygym
def test_levers_change_the_model_as_they_say():
    from virtual_fly.physics import PhysicsBody, Walker
    from virtual_fly.world import World
    import mujoco
    plain = Walker()
    m0 = plain._m
    assert plain.levers == () and plain.pairs_dropped == 0 and plain.wall_pairs == 0
    assert (m0.opt.iterations, m0.opt.noslip_iterations, m0.opt.timestep) == (100, 5, pytest.approx(1e-4))   # flygym 2.1's own
    assert m0.npair == 55 and m0.opt.tolerance == pytest.approx(1e-8)        # 55 ground pairs, no self-collision pairs, none twice
    deduped = Walker(levers="dedupe,solver100,noslip5,noself")                # the four no-ops: the same model
    assert deduped.pairs_dropped == 0 and deduped._m.npair == 55 and (deduped._m.opt.iterations, deduped._m.opt.noslip_iterations) == (100, 5)
    fast = Walker(levers="noslip0")
    assert fast._m.opt.iterations == 100 and fast._m.opt.noslip_iterations == 0
    simple = Walker(levers="simple")
    assert (simple._m.geom_type[1:] == mujoco.mjtGeom.mjGEOM_CAPSULE).all()        # every part a capsule (the ground is geom 0)
    for w in (plain, deduped, fast, simple):
        w.close()
    body = PhysicsBody(World(seed=1), random.Random(0), levers="dt2")
    assert body.timestep == 2e-4 and body.walker._m.opt.timestep == pytest.approx(2e-4)
    y0 = body.pose.y
    for _ in range(12):                                   # 0.3 s of forward drive at the coarser step: it still walks
        body.move(0.025, "walk", drive(forward=1.0))
    assert body.pose.y - y0 > 1.5
    assert body.to_dict()["physics"]["z"] > 0.3
