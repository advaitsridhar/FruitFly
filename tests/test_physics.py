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
    monkeypatch.setattr(physics, "_IMPORT_ERROR", ModuleNotFoundError("No module named 'numba'"))
    monkeypatch.setattr(physics, "sys", types.SimpleNamespace(version_info=(3, 11, 9)))
    reason = physics.unavailable_reason()
    assert "Python 3.10-3.12" in reason and "numba &&" in reason and "No module named 'numba'" in reason
    with pytest.raises(RuntimeError, match="No module named 'numba'"):
        physics.make_body("physics", World(seed=1), random.Random(0))
    with pytest.raises(SystemExit, match="No module named 'numba'"):         # before any data is loaded
        play.main(["--body", "physics", "--no-browser"])
    monkeypatch.setattr(physics, "sys", types.SimpleNamespace(version_info=(3, 13, 1)))
    assert "this is Python 3.13" in physics.unavailable_reason()


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
    for _ in range(4):
        g.tick()
    assert g.body.kind == "physics" and "physics" in g.state_dict["fly"]
    assert any("leg physics" in s for s in g.whats_real()["hand_built"])


# ---------------------------------------------------------------------------------------------- speed levers (plan 8.6)
def test_levers_are_named_checked_and_off_by_default():
    assert physics.parse_levers(None) == () and physics.parse_levers("") == () and physics.parse_levers(()) == ()
    assert physics.parse_levers("solver100, dedupe") == ("dedupe", "solver100")     # LEVERS order, whatever the input's
    assert physics.parse_levers(["dt2"]) == ("dt2",)
    with pytest.raises(ValueError, match="unknown physics lever nope"):
        physics.parse_levers("dedupe,nope")
    with pytest.raises(ValueError, match="exclude each other"):
        physics.parse_levers("noslip5,noslip0")
    assert physics.lever_fly_kwargs(()) == {} and physics.lever_fly_kwargs(("noself",)) == {"self_collisions": "none"}
    assert physics.lever_fly_kwargs(("simple",)) == {"xml_variant": "seqik_simple", "floor_collisions": "tarsi"}
    assert physics.lever_timestep(()) == physics.TIMESTEP and physics.lever_timestep(("dt2",)) == 2e-4
    assert set(physics.LEVERS) == {"dedupe", "solver100", "noslip5", "noslip0", "noself", "simple", "dt2"}


@needs_flygym
def test_levers_change_the_model_as_they_say():
    from virtual_fly.physics import PhysicsBody, Walker
    from virtual_fly.world import World
    plain = Walker()
    m0 = plain.sim.physics.model.ptr
    assert plain.levers == () and plain.pairs_dropped == 0
    assert (m0.opt.iterations, m0.opt.noslip_iterations, m0.opt.timestep) == (1000, 100, pytest.approx(1e-4))
    deduped = Walker(levers="dedupe")
    assert deduped.pairs_dropped == 1086 and deduped.sim.physics.model.ptr.npair == m0.npair - 1086
    fast = Walker(levers="solver100,noslip0")
    m = fast.sim.physics.model.ptr
    assert m.opt.iterations == 100 and m.opt.tolerance == pytest.approx(1e-8) and m.opt.noslip_iterations == 0
    assert Walker(levers="noslip5").sim.physics.model.ptr.opt.noslip_iterations == 5
    assert Walker(levers="noself").sim.physics.model.ptr.npair == m0.npair - 2172
    body = PhysicsBody(World(seed=1), random.Random(0), levers="dt2")
    assert body.timestep == 2e-4 and body.walker.sim.physics.model.ptr.opt.timestep == pytest.approx(2e-4)
    y0 = body.pose.y
    for _ in range(12):                                   # 0.3 s of forward drive at the coarser step: it still walks
        body.move(0.025, "walk", drive(forward=1.0))
    assert body.pose.y - y0 > 1.5
    assert body.to_dict()["physics"]["z"] > 0.3
