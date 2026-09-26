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
