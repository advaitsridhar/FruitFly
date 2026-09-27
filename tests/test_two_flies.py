"""Two simulated flies in one dish (docs/TWO_FLIES_PLAN.md, Phase 1): a synthetic male protagonist and a synthetic
female partner, each with a brain of its own, sensing each other only through the world."""

import gzip
import hashlib
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from synthetic_connectome import build_synthetic  # noqa: E402

from virtual_fly.agent import HOME, PARTNER_HOME, PoseView
from virtual_fly.brainio import LocalBrain, ProcessBrain
from virtual_fly.connectome import Connectome
from virtual_fly.game import STILL_TICKS, TICK_MS, Game
from virtual_fly.settings import build_brain

# the male file's bytes before the builder learnt to write a female (tests/synthetic_connectome.py): they must not move
MALE_FILE_SHA1 = "f2305629294321bd53a932361ddfbc949d1bc443"


@pytest.fixture(scope="session")
def female_path(tmp_path_factory) -> Path:
    return build_synthetic(tmp_path_factory.mktemp("data") / "synthetic-female.flyb.gz", sex="female")


@pytest.fixture(scope="session")
def fconn(female_path):
    return Connectome(female_path)


def pair(conn, fconn, procs="off", seed=1, **kw):
    """A male protagonist (fly 0) with a female partner (fly 1)."""
    partner = {"conn": fconn, **kw.pop("partner", {})}
    return Game(build_brain(conn, "game", seed=0), autopilot=False, seed=seed, brain_procs=procs, partner=partner, **kw)


def run_hash(game, n):
    h = hashlib.sha1()
    for _ in range(n):
        game.tick()
        h.update(game.state_json)
    return h.hexdigest()


# ------------------------------------------------------------------ the synthetic female
def test_the_default_synthetic_file_is_byte_for_byte_what_it_was(synthetic_path, tmp_path):
    assert hashlib.sha1(gzip.open(synthetic_path, "rb").read()).hexdigest() == MALE_FILE_SHA1
    assert hashlib.sha1(gzip.open(build_synthetic(tmp_path / "again.flyb.gz"), "rb").read()).hexdigest() == MALE_FILE_SHA1


def test_the_synthetic_female_lacks_the_male_only_cells(conn, fconn):
    assert fconn.sex == "female" and conn.sex == "male"
    for spec in ("pIP10", "TTMn", "LgLG1a,LgLG1b", "LgLG4"):
        assert conn.count(spec) > 0 and fconn.count(spec) == 0, spec
    for spec in ("MDN", "DNp01", "prefix:pC1_", "prefix:JO-A,prefix:JO-B", "LC10a", "MN9"):
        assert fconn.count(spec) > 0, spec
    with pytest.raises(ValueError):
        build_synthetic("x.flyb.gz", sex="other")


# ------------------------------------------------------------------ two flies in the dish
def test_the_partner_is_its_own_fly(conn, fconn):
    g = pair(conn, fconn)
    f0, f1 = g.flies
    assert g.fly(1) is f1 and f1.id == 1 and f1.sex == "female" and f0.sex == "male" and f1.pair and f0.pair
    assert f1.rng is not g.rng and f1.rng is not f0.rng and f0.rng is g.rng               # D11: its own dice
    assert f1.seed == g.seed + 1000 and f0.seed == g.seed
    assert f1.io.settings()["seed"] == g.seed + 1000 and f0.io.settings()["seed"] == 0    # (fly 0's brain: build_brain's seed)
    assert f1.decoder is not f0.decoder and f1.readouts is not f0.readouts and f1.body is not f0.body
    assert f1.conn is fconn and f0.conn is conn
    assert (f0.body.pose.x, f0.body.pose.y, f0.body.pose.h) == HOME
    assert (f1.body.pose.x, f1.body.pose.y, f1.body.pose.h) == PARTNER_HOME
    assert f1.readouts["pIP10"].size == 0 and not f1.can_court and f0.can_court           # D12
    assert isinstance(f0.io, LocalBrain) and isinstance(f1.io, LocalBrain)                # brain_procs off
    assert f1.autopilot and g.autopilot is False                                           # her walking urge, on (decision 13)
    assert set(g.state_dict) == set()                                                      # nothing published yet
    g.tick()
    assert "flies" not in g.state_dict                                                     # (the state's flies list: step 6)


def test_the_partner_walks_on_its_own_and_the_protagonist_sees_it_only_through_the_world(conn, fconn):
    g = pair(conn, fconn)
    x0, y0 = g.flies[1].body.pose.x, g.flies[1].body.pose.y
    for _ in range(80):
        g.tick()
    f1 = g.flies[1]
    assert math.hypot(f1.body.pose.x - x0, f1.body.pose.y - y0) > 2.0                    # her walking urge moves her
    assert g.world.female is None                                                          # no scripted female (D9)
    assert g.state_dict["fly"]["x"] == round(g.flies[0].body.pose.x, 3)                   # the state is fly 0's
    assert f1.mode in ("walk", "idle", "court", "backward", "feed", "groom", "escape")


def test_two_runs_give_the_same_bytes_and_a_process_brain_the_same_as_a_local_one(conn, fconn):
    a = run_hash(pair(conn, fconn), 40)
    assert a == run_hash(pair(conn, fconn), 40)
    g = pair(conn, fconn, procs="auto")
    try:
        assert all(isinstance(f.io, ProcessBrain) for f in g.flies)                       # auto: a process per brain with two
        assert all(f.io.proc.is_alive() for f in g.flies)
        assert run_hash(g, 40) == a
    finally:
        g.close()
    assert all(not f.io.proc.is_alive() for f in g.flies)


def test_the_scripted_female_is_refused_while_a_partner_is_in_the_dish(conn, fconn):
    g = pair(conn, fconn)
    r = g.action({"type": "female", "on": True, "x": 10, "y": 10})
    assert r["ok"] is False and "partner" in r["error"] and g.actions.empty()
    g.tick()
    assert g.world.female is None
    assert g.action({"type": "female", "on": False})["ok"] is True                        # switching her off is harmless


def test_the_courtship_scenario_places_the_partner_and_makes_no_scripted_female(conn, fconn):
    g = pair(conn, fconn)
    assert g.action({"type": "scenario", "id": "courtship"})["ok"] is True
    g.tick()
    f1 = g.flies[1]
    assert g.world.female is None and (round(f1.body.pose.x, 1), round(f1.body.pose.y, 1)) == (14.0, 10.0)
    assert -math.pi <= f1.body.pose.h <= math.pi
    assert "partner_distance" in g.scenario.measure and "female_receptive" not in g.scenario.measure
    single = Game(build_brain(conn, "game", seed=0), autopilot=False, seed=1)
    single.action({"type": "scenario", "id": "courtship"})
    single.tick()
    assert single.world.female is not None and "female_receptive" in single.scenario.measure


def test_a_partner_without_song_cells_never_enters_court_mode(conn, fconn):
    g = pair(conn, fconn)
    f0, f1 = g.flies
    m = {k: 0.0 for k in f0.decoder.m}
    m["court"] = 0.9                                                                       # pC1 alone
    assert f1.choose_mode(dict(m), 0) == "walk" and f0.choose_mode(dict(m), 0) == "court"
    m["song"] = 0.9
    assert f1.choose_mode(dict(m), 0) == "walk"


def test_each_fly_keeps_its_own_resting_count(conn, fconn):
    g = pair(conn, fconn)
    f0, f1 = g.flies
    f0.autopilot = f1.autopilot = False
    for _ in range(STILL_TICKS + 2):
        g.tick()
    assert f0.mode == "idle" and f1.mode == "idle" and f0.still_ticks >= STILL_TICKS and f1.still_ticks >= STILL_TICKS
    f1.autopilot = True
    for _ in range(40):
        g.tick()
    assert f1.still_ticks < f0.still_ticks and f0.mode == "idle"                            # only she moved


def test_a_new_fly_resets_both_flies(conn, fconn):
    g = pair(conn, fconn)
    for _ in range(30):
        g.tick()
    f1 = g.flies[1]
    assert (f1.body.pose.x, f1.body.pose.y) != PARTNER_HOME[:2]
    g.action({"type": "reset"})
    g.tick()
    assert (round(f1.body.pose.x, 3), round(f1.body.pose.y, 3)) == (PARTNER_HOME[0], PARTNER_HOME[1] + 0.0) or \
        math.hypot(f1.body.pose.x - PARTNER_HOME[0], f1.body.pose.y - PARTNER_HOME[1]) < 0.5   # back home (one tick of walking)


def test_pose_view_is_the_start_of_tick_snapshot(conn, fconn):
    g = pair(conn, fconn)
    v = g.flies[1].pose_view()
    assert isinstance(v, PoseView) and (v.id, v.sex, v.body_kind, v.jump) == (1, "female", "drawn", False)
    assert (v.x, v.y, v.h) == PARTNER_HOME and v.song == 0.0 and v.court == 0.0


def test_the_options_are_checked(conn, fconn):
    with pytest.raises(ValueError, match="Phase 4"):
        Game(build_brain(conn, "game", seed=0), body="physics", partner={"conn": fconn})
    with pytest.raises(ValueError, match="social channel"):
        Game(build_brain(conn, "game", seed=0), partner={"conn": fconn}, social="seen,sogn")
    assert TICK_MS == 25.0
