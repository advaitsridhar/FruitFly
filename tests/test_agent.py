"""The fly-0 shims on Game (docs/TWO_FLIES_PLAN.md 5.2, step 1): one fly's state lives on its FlyAgent (agent.py), and
every name in ``Game._FLY_ATTRS`` reads and writes fly 0's, with one fly in the dish and with two."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from synthetic_connectome import build_synthetic  # noqa: E402

from virtual_fly.connectome import Connectome
from virtual_fly.game import Game
from virtual_fly.settings import build_brain


@pytest.fixture(scope="module")
def games(conn, tmp_path_factory):
    """A single synthetic male, and the same male with a synthetic female partner (both brains in this process)."""
    fconn = Connectome(build_synthetic(tmp_path_factory.mktemp("data") / "synthetic-female.flyb.gz", sex="female"))
    one = Game(build_brain(conn, "game", seed=0), autopilot=False, seed=1)
    two = Game(build_brain(conn, "game", seed=0), autopilot=False, seed=1, brain_procs="off", partner={"conn": fconn})
    for g in (one, two):
        g.tick()
    return one, two


def test_every_shim_is_a_property_with_a_setter():
    for name in Game._FLY_ATTRS:
        prop = Game.__dict__[name]
        assert isinstance(prop, property) and prop.fget is not None and prop.fset is not None, name
        assert "fly 0" in (prop.__doc__ or ""), name


def test_every_shim_reads_fly_0(games):
    for g in games:
        f0 = g.flies[0]
        for name in Game._FLY_ATTRS:
            mine, theirs = getattr(g, name), getattr(f0, name)
            assert mine is theirs or (not isinstance(mine, np.ndarray) and mine == theirs), name
    one, two = games
    # with two flies the shims are fly 0's and never the partner's
    f0, f1 = two.flies
    assert two.conn is f0.conn and two.conn is not f1.conn and two.body is f0.body and two.body is not f1.body
    f1.graded_eps, f1.mode = 12345.0, "groom"
    assert two.graded_eps == f0.graded_eps != 12345.0 and two.mode == f0.mode
    assert two.autopilot is False and f1.autopilot is True                 # the fixture's: his off, hers on


def test_every_shim_writes_fly_0(games):
    for g in games:
        f0 = g.flies[0]
        g.autopilot = True
        assert f0.autopilot is True
        g.autopilot = False
        assert f0.autopilot is False
        g.mode = "walk"
        assert f0.mode == "walk"
        g.graded_eps = 7.0
        assert f0.graded_eps == 7.0
        g.court_left = 1.5
        assert f0.court_left == 1.5
        g.done.add("feed")
        assert "feed" in f0.done
        g.hz_shown = {"MDN": 3.0}
        assert f0.hz_shown == {"MDN": 3.0}
    one, two = games
    f0, f1 = two.flies
    hers = f1.autopilot
    two.autopilot = not f0.autopilot
    assert f1.autopilot is hers                                           # the partner is never written through the shim
    f1.graded_eps = 99.0
    two.graded_eps = 1.0
    assert f0.graded_eps == 1.0 and f1.graded_eps == 99.0
