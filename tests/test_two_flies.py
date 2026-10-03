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
from virtual_fly.game import GF_BURST, STILL_TICKS, TICK_MS, Game
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
    assert [e["id"] for e in g.state_dict["flies"]] == [0, 1]                              # the state's flies list (5.7)


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
    from virtual_fly import physics
    if physics.available():                      # a partner with the physics body: one MuJoCo world (tests/test_physics_pair.py)
        with pytest.raises(ValueError, match="stride-average"):
            Game(build_brain(conn, "game", seed=0), body="physics", stride_average=True, brain_procs="off",
                 partner={"conn": fconn, "brain_kwargs": {"seed": 1000}, "parts": False})
    else:                                        # without flygym: the install hint, before any brain process starts
        with pytest.raises(RuntimeError, match="flygym"):
            Game(build_brain(conn, "game", seed=0), body="physics", partner={"conn": fconn})
    with pytest.raises(ValueError, match="social channel"):
        Game(build_brain(conn, "game", seed=0), partner={"conn": fconn}, social="seen,sogn")
    assert TICK_MS == 25.0


# ------------------------------------------------------------------ the social encoders (senses/social.py)
from virtual_fly.body import FLY_CAPSULE_HALF, FLY_CAPSULE_R, capsule_overlap  # noqa: E402
from virtual_fly.game import COURTSHIP_SPEC, PHEROMONE_GRNS  # noqa: E402
from virtual_fly.senses import social  # noqa: E402
from virtual_fly.senses.mechano import SOUND  # noqa: E402
from virtual_fly.senses.social import SocialConfig  # noqa: E402

GOLDEN = __import__("json").loads((Path(__file__).with_name("golden_single_fly.json")).read_text())["hashes"]
PHEROMONE_SPEC = next(iter(PHEROMONE_GRNS))


def place(fly, x, y, h):
    fly.body.reset(x, y, h)


def test_the_social_config_reads_the_list():
    cfg = SocialConfig.from_list(None)
    assert cfg.names() == ["seen", "song", "contact", "collide"] and cfg.mating == "none" and not cfg.cva
    cfg = SocialConfig.from_list("song, cva,mating:virgin")
    assert cfg.names() == ["song", "cva", "mating:virgin"] and not cfg.seen and not cfg.collide
    for bad in ("sogn", "mating", "mating:maybe", "seen:on"):
        with pytest.raises(ValueError, match="social channel"):
            SocialConfig.from_list(bad)
    assert SocialConfig().contact_pc1_for("male") and not SocialConfig().contact_pc1_for("female")
    assert SocialConfig(contact_pc1=True).contact_pc1_for("female")
    assert social.falloff(3.0) == 1.0 and social.falloff(10.5) == pytest.approx(0.5) and social.falloff(16.0) == 0.0


def test_each_fly_sees_the_other_and_not_itself(conn, fconn):
    g = pair(conn, fconn)
    f0, f1 = g.flies
    place(f0, 0.0, 0.0, 0.0)
    place(f1, 10.0, 0.0, math.pi)
    g.tick()
    flies_seen = [o for o in f0.retina.objects(f0.body.pose, [f1.pose_view()]) if o.kind == "fly"]
    assert len(flies_seen) == 1 and (flies_seen[0].x, flies_seen[0].y, flies_seen[0].r, flies_seen[0].h) == (f1.body.pose.x, f1.body.pose.y, 1.6, 2.2)
    assert [o for o in f1.retina.objects(f1.body.pose, [f0.pose_view()]) if o.kind == "fly"][0].x == f0.body.pose.x
    assert [o for o in f0.retina.objects(f0.body.pose) if o.kind == "fly"] == []          # nothing with no others
    # the switch is proven by what reaches the brain (test_the_seen_channel_reaches_the_brain_only_when_on), not by a flag
    # the retina sets only for a blob that moves across the eye


def test_the_contact_channel_tastes_a_female_and_arouses_a_male_toucher(conn, fconn):
    g = pair(conn, fconn)
    f0, f1 = g.flies
    place(f0, 0.0, 0.0, 0.0)                    # nose to nose, 5 mm apart: both flies' foreleg tips reach the other's centre
    place(f1, 5.0, 0.0, math.pi)
    g.tick()
    assert f0.senses_now["touches_fly"] == 1 and f0.senses_now["pheromone"] is True and f0.senses_now["courting"] is True
    assert f0.rates_now[PHEROMONE_SPEC] == PHEROMONE_GRNS[PHEROMONE_SPEC] and f0.rates_now[COURTSHIP_SPEC] > 0
    assert f0.court_left > 0 and f0.contact_pc1                                          # a male toucher: the kit's arousal
    assert f1.senses_now["touches_fly"] == 0 and "pheromone" not in f1.senses_now and "courting" not in f1.senses_now
    assert PHEROMONE_SPEC not in f1.rates_now and COURTSHIP_SPEC not in f1.rates_now   # she touched a male, and has no leg taste
    assert f1.court_left == 0 and not f1.contact_pc1
    # the arousal stands for her pheromone: touching a male never gives it, whatever the switch says
    g2 = pair(conn, fconn, social=SocialConfig(contact_pc1=True))
    place(g2.flies[0], 0.0, 0.0, 0.0)
    place(g2.flies[1], 5.0, 0.0, math.pi)
    g2.tick()
    assert g2.flies[1].court_left == 0 and COURTSHIP_SPEC not in g2.flies[1].rates_now and g2.flies[0].court_left > 0
    # the switch: a female touching a female may be given the arousal
    ff2 = Game(build_brain(fconn, "game", seed=0), autopilot=False, seed=1, brain_procs="off", partner={"conn": fconn},
               social=SocialConfig(contact_pc1=True))
    place(ff2.flies[0], 0.0, 0.0, 0.0)
    place(ff2.flies[1], 5.0, 0.0, math.pi)
    ff2.tick()
    assert ff2.flies[0].court_left > 0 and ff2.flies[1].court_left > 0 and COURTSHIP_SPEC in ff2.flies[1].rates_now
    # a female touching a female: no leg taste cells in her file, said once in the event log
    ff = Game(build_brain(fconn, "game", seed=0), autopilot=False, seed=1, brain_procs="off", partner={"conn": fconn})
    place(ff.flies[0], 0.0, 0.0, 0.0)
    place(ff.flies[1], 5.0, 0.0, math.pi)
    for _ in range(3):
        ff.tick()
    said = [e for e in ff.events.items if "not wired in this brain" in e["text"]]
    assert len(said) == 2 and {e["fly"] for e in said} == {0, 1} and PHEROMONE_SPEC not in ff.flies[0].rates_now
    assert ff.flies[0].court_left == 0                                                    # a female toucher: no arousal by default
    # out of reach: nothing
    g3 = pair(conn, fconn)
    place(g3.flies[0], 0.0, 0.0, 0.0)
    place(g3.flies[1], 9.0, 0.0, math.pi)
    g3.tick()
    assert "touches_fly" not in g3.flies[0].senses_now and PHEROMONE_SPEC not in g3.flies[0].rates_now


def test_the_song_channel_fades_with_distance_and_never_reaches_the_singer(conn, fconn):
    for d, share in ((3.0, 1.0), (10.5, 0.5), (16.0, 0.0)):
        g = pair(conn, fconn, social=SocialConfig(collide=False))
        f0, f1 = g.flies
        place(f0, 0.0, 0.0, 0.0)
        place(f1, d, 0.0, math.pi)
        f0.m["song"] = 0.8                          # the male sang last tick (his decoded song, as the snapshot carries it)
        g.tick()
        want = 0.8 * social.SONG_MAX_HZ * share
        if want:
            assert f1.rates_now[SOUND] == pytest.approx(want) and f1.senses_now["hears_song"] == round(want, 1)
        else:
            assert SOUND not in f1.rates_now and "hears_song" not in f1.senses_now
        assert SOUND not in f0.rates_now and "hears_song" not in f0.senses_now         # he does not hear himself
    g = pair(conn, fconn, social=SocialConfig(song=False))
    g.flies[0].m["song"] = 1.0
    place(g.flies[0], 0.0, 0.0, 0.0)
    place(g.flies[1], 3.0, 0.0, math.pi)
    g.tick()
    assert SOUND not in g.flies[1].rates_now


def test_every_channel_off_leaves_the_brains_input_free_of_social_terms(conn, fconn):
    cfg = SocialConfig(seen=False, song=False, contact=False, collide=False)
    g = pair(conn, fconn, social=cfg)
    f0, f1 = g.flies
    f0.m["song"] = 1.0
    for _ in range(3):
        place(f0, 0.0, 0.0, 0.0)
        place(f1, 3.0, 0.0, math.pi)
        g.tick()
    for f in (f0, f1):
        assert not ({SOUND, PHEROMONE_SPEC, COURTSHIP_SPEC, "ORN_DA1", social.SAG_SPEC} & set(f.rates_now))
        assert not ({"touches_fly", "hears_song", "pheromone", "courting", "smells_cva", "virgin_drive"} & set(f.senses_now))
        assert not f.body.bumped_fly
    assert [o for o in f0.retina.objects(f0.body.pose, [f1.pose_view()]) if o.kind == "fly"]   # objects() itself lists them ...
    for k in range(12):                                                                    # ... but the senses never ask for them:
        place(f1, 10.0, -6.0 + k, math.pi)                                                 # she crosses his view, 1 mm a tick
        g.tick()
        assert not any(k2.startswith(("LC10a/", "LC11/")) for k2 in f0.rates_now), f0.rates_now


def test_the_bodies_do_not_overlap_yet_a_head_on_tap_still_lands(conn, fconn, monkeypatch):
    g = pair(conn, fconn)
    f0, f1 = g.flies
    f0.autopilot = f1.autopilot = False
    for f in (f0, f1):
        m = f.decoder.m
        monkeypatch.setattr(f.decoder, "decode", lambda hz, dt, m=m: {**{k: 0.0 for k in m}, "forward": 1.0})
    place(f0, -7.0, 0.0, 0.0)
    place(f1, 7.0, 0.0, math.pi)
    tapped, bumped, gaps = [], [], []
    for _ in range(80):
        g.tick()
        gap = capsule_overlap((f0.body.pose.x, f0.body.pose.y, f0.body.pose.h), (f1.body.pose.x, f1.body.pose.y, f1.body.pose.h))[0]
        gaps.append(gap)
        tapped.append("touches_fly" in f0.senses_now and "touches_fly" in f1.senses_now)
        bumped.append((f0.body.bumped_fly, f0.body.bumped, f1.body.bumped_fly, f1.body.bumped))
    assert min(gaps) >= -1e-6, min(gaps)                                                # never inside each other
    assert min(gaps) < 0.5                                                              # they came right up to each other
    assert any(tapped), "a head-on foreleg tap must still be reachable through the capsules"
    assert any(b[0] and not b[1] for b in bumped) and any(b[2] and not b[3] for b in bumped)   # a fly bump, not a wall bump
    d = math.hypot(f0.body.pose.x - f1.body.pose.x, f0.body.pose.y - f1.body.pose.y)
    assert 2 * (FLY_CAPSULE_HALF + FLY_CAPSULE_R) - 1e-6 <= d < 5.5                    # stopped nose to nose, within a tap's reach


def test_pushing_overlapping_flies_apart_is_the_same_whichever_comes_first(conn, fconn):
    outcomes = []
    for order in ((0, 1), (1, 0)):
        g = pair(conn, fconn)
        f0, f1 = g.flies
        place(f0, 0.0, 0.0, 0.2)
        place(f1, 1.0, 0.6, -0.9)                                                          # well inside each other
        before = capsule_overlap((0.0, 0.0, 0.2), (1.0, 0.6, -0.9))[0]
        assert before < 0
        social.resolve_overlaps([g.flies[i] for i in order])
        after = capsule_overlap((f0.body.pose.x, f0.body.pose.y, f0.body.pose.h), (f1.body.pose.x, f1.body.pose.y, f1.body.pose.h))[0]
        assert after >= -1e-6
        outcomes.append(((f0.body.pose.x, f0.body.pose.y), (f1.body.pose.x, f1.body.pose.y)))
        # symmetric: both moved by the same amount
        assert math.hypot(f0.body.pose.x, f0.body.pose.y) == pytest.approx(math.hypot(f1.body.pose.x - 1.0, f1.body.pose.y - 0.6))
    assert outcomes[0] == outcomes[1]
    # far apart: untouched
    g = pair(conn, fconn)
    social.resolve_overlaps(g.flies)
    assert (g.flies[0].body.pose.x, g.flies[0].body.pose.y) == HOME[:2] and (g.flies[1].body.pose.x, g.flies[1].body.pose.y) == PARTNER_HOME[:2]


def test_the_song_gesture_faces_the_partner(conn, fconn):
    g = pair(conn, fconn)
    f0, f1 = g.flies
    place(f0, 0.0, -12.0, math.pi / 2)
    place(f1, 10.0, -12.0, math.pi / 2)                                                   # to his right
    g.tick()
    assert f0.song_side == 1.0
    place(f1, -10.0, -12.0, math.pi / 2)                                                  # to his left
    g.tick()
    assert f0.song_side == -1.0


def test_contact_pc1_changes_nothing_on_the_single_fly_path(conn, fconn):
    from tools.golden_hashes import run_hash
    for flag in (True, False):
        assert run_hash(conn, "courtship_scenario", game_kwargs={"social": SocialConfig(contact_pc1=flag)}) == GOLDEN["courtship_scenario"]
    # a female protagonist tapping the scripted female gets the arousal whatever the switch says (D7)
    for flag in (True, False):
        g = Game(build_brain(fconn, "game", seed=0), autopilot=False, seed=1, social=SocialConfig(contact_pc1=flag))
        g.action({"type": "female", "on": True, "x": 0.0, "y": -7.0})                     # right in front of her forelegs
        g.tick()
        assert g.flies[0].court_left > 0 and COURTSHIP_SPEC in g.flies[0].rates_now


def test_whats_real_lists_the_channels_only_with_a_partner(conn, fconn):
    single = Game(build_brain(conn, "game", seed=0), autopilot=False, seed=1).whats_real()
    both = pair(conn, fconn).whats_real()
    joined = " ".join(both["hand_built"] + both["not_modelled"])
    for word in ("Seen:", "Song:", "Contact:", "Collide:", "hand-built", "drawn scale", "real centre-to-centre mm", "provisional",
                 "walking urge"):
        assert word in joined, word
    assert not any(w in joined.lower() for w in ("acceptance", "rejection"))
    assert not any(l.startswith(("Seen:", "Song:", "Contact:", "Collide:")) for l in single["hand_built"])
    assert any("the female's behaviour" in l for l in single["hand_built"])
    assert any("scripted female's behaviour" in l for l in both["hand_built"])
    assert "Two simulated flies" in joined
    off = pair(conn, fconn, social=SocialConfig(song=False, cva=True, mating="virgin", touch=True)).whats_real()
    j = " ".join(off["hand_built"])
    assert "Song:" not in j and "cVA:" in j and "virgin" in j and "Touch:" in j


# ------------------------------------------------------------------ her decision neurons (game.FEMALE_READOUTS)
from virtual_fly.game import FEMALE_READOUTS, READOUTS  # noqa: E402

FEMALE_KEYS = [r[0] for r in FEMALE_READOUTS]


def test_a_female_with_a_partner_watches_her_decision_neurons(conn, fconn):
    g = pair(conn, fconn)
    f0, f1 = g.flies
    assert set(FEMALE_KEYS) <= set(f1.readouts) and not (set(FEMALE_KEYS) & set(f0.readouts))
    present = [k for k in FEMALE_KEYS if f1.readouts[k].size]
    assert set(present) == {"vpoEN", "DNp37", "DNp13", "SAG"}                             # what the synthetic female has
    meta = {m["key"]: m for m in f1.readout_meta}
    assert set(present) <= set(meta) and all(meta[k]["group"] == "Her decisions" for k in present)
    assert [m["key"] for m in f1.readout_meta if m["key"] not in FEMALE_KEYS] == [r[0] for r in READOUTS if f1.readouts[r[0]].size]
    assert not (set(FEMALE_KEYS) & {m["key"] for m in f0.readout_meta})
    assert set(present) <= set(f1.brain.monitors) and not (set(FEMALE_KEYS) & set(f0.brain.monitors))
    g.tick()
    assert set(present) <= set(f1.bt.hz) and "DNp37" in f1.hz_shown and "DNp37" not in f0.bt.hz
    text = " ".join(r[2] + r[3] for r in FEMALE_READOUTS).lower()
    assert "acceptance" not in text and "rejection" not in text


def test_a_female_without_a_partner_does_not_get_them(conn, fconn):
    single = Game(build_brain(fconn, "game", seed=0), autopilot=False, seed=1)             # the female protagonist alone
    assert not (set(FEMALE_KEYS) & set(single.readouts)) and not (set(FEMALE_KEYS) & {m["key"] for m in single.readout_meta})
    male_alone = Game(build_brain(conn, "game", seed=0), autopilot=False, seed=1)
    assert not (set(FEMALE_KEYS) & set(male_alone.readouts))
    # a female protagonist with a male partner does get them; her male partner does not
    g = Game(build_brain(fconn, "game", seed=0), autopilot=False, seed=1, brain_procs="off", partner={"conn": conn})
    assert set(FEMALE_KEYS) <= set(g.flies[0].readouts) and g.flies[1].sex == "male" and not (set(FEMALE_KEYS) & set(g.flies[1].readouts))
    assert g.flies[1].can_court and g.flies[0].can_court                                 # D12 is a partner's rule: fly 0 always may


def test_her_decision_neurons_travel_to_her_brain_process(conn, fconn):
    g = pair(conn, fconn, procs="auto")
    try:
        g.tick()
        f1 = g.flies[1]
        assert isinstance(f1.io, ProcessBrain) and {"vpoEN", "DNp37", "DNp13", "SAG"} <= set(f1.bt.hz)
        assert "DNp37" in f1.histories and len(f1.histories["DNp37"]) == 1
    finally:
        g.close()


# ------------------------------------------------------------------ what the second review found (docs/TWO_FLIES_PROGRESS.md)
import json  # noqa: E402

from virtual_fly.game import CHECKS  # noqa: E402


def _force_decode(fly, monkeypatch, **drives):
    m = fly.decoder.m
    monkeypatch.setattr(fly.decoder, "decode", lambda hz, dt, m=m: {**{k: 0.0 for k in m}, **drives})


def test_the_seen_channel_reaches_the_brain_only_when_on(conn, fconn):
    """The other fly is a small moving object to the retina: LC10a/LC11 rates when she crosses his view, none with the
    channel off (the retina's own 'small' flag needs a blob that moves, so a still partner proves nothing)."""
    seen = {}
    for on in (True, False):
        g = pair(conn, fconn, social=SocialConfig(seen=on, collide=False))
        f0, f1 = g.flies
        f0.autopilot = f1.autopilot = False
        keys = set()
        for k in range(12):
            place(f0, 0.0, 0.0, 0.0)
            place(f1, 10.0, -6.0 + k, math.pi)                                             # across his view, 1 mm a tick
            g.tick()
            keys |= {k2 for k2 in f0.rates_now if k2.startswith(("LC10a/", "LC11/"))}
        seen[on] = keys
    assert seen[True] >= {"LC10a/L", "LC11/L"} or seen[True] >= {"LC10a/R", "LC11/R"}, seen[True]
    assert seen[False] == set()


def test_her_decision_monitors_survive_a_brain_swap_on_the_local_seam(conn, fconn, monkeypatch):
    """Finding 1/7: a rebuilt local brain got monitors for the kit's readouts only; hers must come back too."""
    g = Game(build_brain(fconn, "game", seed=0), autopilot=False, seed=1, brain_procs="off", partner={"conn": conn})
    f = g.flies[0]
    monkeypatch.setattr(f, "_survival_worker", lambda conn, token: None)                 # no re-test child in this test
    before = [k for k in FEMALE_KEYS if k in f.brain.monitors]
    assert before == [k for k in FEMALE_KEYS if f.readouts[k].size] and before
    f._swap_brain({"brain": build_brain(fconn, "game", seed=0), "conn": fconn, "level": "real", "seed": 0, "rules": None,
                   "wiring": None, "parts": False, "reason": "parts"})
    assert [k for k in FEMALE_KEYS if k in f.brain.monitors] == before
    g.tick()
    assert set(before) <= set(f.history()[1]) and "MDN" in f.history()[1]


def test_a_male_singing_at_a_simulated_female_ticks_the_court_check(conn, fconn, monkeypatch):
    """Finding 2/10: the song event and the 'court' item were keyed on the scripted female, absent with a partner."""
    g = pair(conn, fconn)
    f0, f1 = g.flies
    f0.autopilot = f1.autopilot = False
    _force_decode(f0, monkeypatch, song=0.9, court=0.9)
    place(f0, 0.0, 0.0, 0.0)
    place(f1, 5.0, 0.0, math.pi)
    for _ in range(4):
        g.tick()
    assert f0.mode == "court" and "court" in f0.done and "court" not in f1.done
    song = [e for e in g.events.items if e["text"].startswith("courtship song")]
    assert len(song) == 1 and song[0]["fly"] == 0
    # the checklist text names the partner now, and the single fly's text is what it always was
    checks = {c["id"]: c["text"] for c in json.loads(f0._make_layout())["checks"]}
    assert "female partner" in checks["court"] and "Add a female" not in checks["court"]
    assert "female partner" in checks["genetics"] and "add a female" not in checks["genetics"]
    single = {c["id"]: c["text"] for c in json.loads(Game(build_brain(conn, "game", seed=0), autopilot=False, seed=1)._make_layout())["checks"]}
    assert single == dict(CHECKS)


def test_her_cues_are_a_display_not_a_channel():
    """Finding 3: 'cues' is on by default and is not a --social list member."""
    assert SocialConfig.from_list(None).cues is True and SocialConfig.from_list("seen").cues is True
    assert "cues" not in SocialConfig().names() and "cues" not in social.CHANNELS
    with pytest.raises(ValueError, match="social channel"):
        SocialConfig.from_list("cues")


def test_touch_switches_collide_on_with_it(conn, fconn):
    """Finding 4: a bump is only noticed when the bodies collide."""
    assert SocialConfig.from_list("touch").collide is True and SocialConfig.from_list("touch").touch is True
    assert SocialConfig(touch=True, collide=False).collide is True
    assert SocialConfig(collide=False).collide is False
    j = " ".join(pair(conn, fconn, social=SocialConfig(touch=True)).whats_real()["hand_built"])
    assert "Touch:" in j and "collide" in j
    j = " ".join(pair(conn, fconn).whats_real()["hand_built"])
    assert "Her cues:" in j and "readout displays" in j


def test_a_partner_that_may_not_court_keeps_its_abdomen_straight(conn, fconn, monkeypatch):
    """Finding 5: the abdominal bend is the male's courtship gesture; a female partner's pC1 must not bend hers."""
    g = pair(conn, fconn, social=SocialConfig(collide=False))
    f0, f1 = g.flies
    f0.autopilot = f1.autopilot = False
    for f in (f0, f1):
        _force_decode(f, monkeypatch, court=0.9)
    place(f0, 0.0, 0.0, 0.0)
    place(f1, 4.0, 0.0, math.pi)                                                           # within 6 mm, both facing
    for _ in range(12):
        g.tick()
    assert f0.body.pose.abdomen > 0.5 and f1.body.pose.abdomen == 0.0


@pytest.mark.parametrize("procs", ["off", "auto"])
def test_a_watch_cannot_take_or_remove_her_built_in_readouts(conn, fconn, procs):
    """Finding 6: her decision keys are built-in for her; a watch is renamed, an unwatch of them refused."""
    g = Game(build_brain(fconn, "game", seed=0), autopilot=False, seed=1, brain_procs=procs, partner={"conn": conn})
    try:
        f = g.flies[0]
        assert {"DNp37", "SAG", "vpoEN"} <= f.builtin_keys and not ({"DNp37", "SAG"} & g.flies[1].builtin_keys)
        r = g.action({"type": "watch", "spec": "DNp37"})
        assert r["ok"] is True and r["n"] == 2
        assert g.action({"type": "watch", "spec": "MDN", "key": "SAG"})["ok"] is False
        assert g.action({"type": "unwatch", "key": "DNp37"})["ok"] is False
        g.tick()
        assert "watch:DNp37" in f.custom_readouts and "DNp37" in f.readouts and "DNp37" in f.bt.hz and "DNp37" in g.state_dict["hz"]
        assert g.action({"type": "unwatch", "key": "watch:DNp37"})["ok"] is True
        g.tick()
        assert "watch:DNp37" not in f.custom_readouts and "DNp37" in f.bt.hz and f.readouts["DNp37"].size == 2
        # a single fly keeps the kit's set of built-in keys exactly
        assert Game(build_brain(conn, "game", seed=0), autopilot=False, seed=1).flies[0].builtin_keys == __import__("virtual_fly.game", fromlist=["BUILTIN_KEYS"]).BUILTIN_KEYS
    finally:
        g.close()


def test_each_fly_has_its_own_checklist_and_new_fly_keeps_both(conn, fconn):
    """Plan 5.10: the synthetic female's layout lacks groom, sound, wall, court and genetics; New fly keeps each fly's done."""
    g = pair(conn, fconn)
    f0, f1 = g.flies
    ids0 = [c["id"] for c in json.loads(f0._make_layout())["checks"]]
    ids1 = [c["id"] for c in json.loads(f1._make_layout())["checks"]]
    n = len(CHECKS)
    assert ids0[:n] == [c[0] for c in CHECKS] and ids0[n:] == ["pair:seen", "pair:sang", "pair:tapped"]   # then his pair checks
    kit1, pair1 = [i for i in ids1 if not i.startswith("pair:")], [i for i in ids1 if i.startswith("pair:")]
    assert set(kit1) == set(ids0[:n]) - {"groom", "sound", "wall", "court", "genetics"} and len(kit1) == 12
    assert pair1 == ["pair:heard", "pair:seen_him", "pair:touched"]
    assert json.loads(f1._make_layout())["sex"] == "female"
    f0.done.add("smell")
    f1.done.add("feed")
    assert g.action({"type": "reset"})["ok"] is True
    g.tick()
    assert "smell" in f0.done and "feed" in f1.done and "feed" not in f0.done


# ------------------------------------------------------------------ the per-fly API and state (plan 5.7)
import threading  # noqa: E402
import urllib.error  # noqa: E402
import urllib.request  # noqa: E402
from http.server import ThreadingHTTPServer  # noqa: E402

from virtual_fly import server as S  # noqa: E402

FLY_ENTRY_KEYS = {"id", "sex", "dataset", "fly", "mode", "autopilot", "driver", "senses", "retina", "hz", "motor", "spikes", "sps", "graded_eps",
                  "stims", "calms", "state", "learning", "silenced", "baseline", "modulated", "custom", "genome", "done"}
STATE_KEYS_ONE = {"seq", "t", "rtf", "speed", "fly", "world", "autopilot", "paused", "senses", "retina", "hz", "motor", "driver", "mode",
                  "spikes", "sps", "stims", "calms", "msg", "silenced", "baseline", "modulated", "custom", "done", "state", "learning",
                  "events", "event_seq", "scenario", "recording", "genome", "graded_eps"}


@pytest.fixture
def served_pair(conn, fconn):
    """(game, base url): the synthetic pair behind a live server."""
    g = pair(conn, fconn)
    for _ in range(3):
        g.tick()
    srv = ThreadingHTTPServer(("127.0.0.1", 0), S.make_handler(g))
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        yield g, f"http://127.0.0.1:{srv.server_address[1]}"
    finally:
        srv.shutdown()
        srv.server_close()


def _get(base, path):
    req = urllib.request.Request(base + path)
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def _post(base, payload):
    data = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
    req = urllib.request.Request(base + "/api/action", data=data, method="POST", headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read())


def test_the_state_lists_both_flies_and_the_single_fly_is_unchanged(conn, fconn):
    g = pair(conn, fconn)
    g.tick()
    st = g.state_dict
    assert set(st) == STATE_KEYS_ONE | {"flies"} and len(st["flies"]) == 2
    for e in st["flies"]:
        assert set(e) == FLY_ENTRY_KEYS
    f0, f1 = st["flies"]
    assert (f0["id"], f0["sex"], f1["id"], f1["sex"]) == (0, "male", 1, "female") and f1["dataset"] == fconn.dataset
    assert f0["fly"] == st["fly"] and f0["mode"] == st["mode"] and f0["hz"] == st["hz"] and f0["done"] == st["done"]
    assert f1["mode"] in ("idle", "walk", "court", "backward", "feed", "groom", "escape") and "DNp37" in f1["hz"] and "DNp37" not in f0["hz"]
    assert f1["retina"].keys() == {"L", "R"} and f1["learning"] is not None and f1["genome"]["level"] == "real"
    assert json.loads(g.state_json)["flies"][1]["id"] == 1
    single = Game(build_brain(conn, "game", seed=0), autopilot=False, seed=1)
    single.tick()
    assert set(single.state_dict) == STATE_KEYS_ONE                                       # no new key with one fly


def test_recording_frames_carry_every_fly_and_the_dish(conn, fconn):
    g = pair(conn, fconn)
    assert g.action({"type": "record", "on": True})["ok"] is True
    for _ in range(3):
        g.tick()
    frame = g.recording[-1]
    assert set(frame) == {"t", "fly", "mode", "hz", "senses", "sps", "flies", "world"}
    assert [f["id"] for f in frame["flies"]] == [0, 1] and set(frame["flies"][1]) == {"id", "fly", "mode", "hz", "senses", "sps"}
    assert set(frame["world"]) == {"food", "obstacles", "odours", "wind", "stripes", "tool"}
    single = Game(build_brain(conn, "game", seed=0), autopilot=False, seed=1)
    single.action({"type": "record", "on": True})
    single.tick()
    assert set(single.recording[-1]) == {"t", "fly", "mode", "hz", "senses", "sps"}


def test_per_fly_actions_reach_their_fly(conn, fconn):
    g = pair(conn, fconn)
    f0, f1 = g.flies
    r = g.action({"type": "zap", "spec": "MDN", "hz": 80, "secs": 5, "fly": 1})
    assert r == {"ok": True, "n": 4}
    for _ in range(6):
        g.tick()
    assert f1.hz_shown["MDN"] > 20 and f0.hz_shown["MDN"] < 5 and f1.zaps and not f0.zaps
    her = [e for e in g.state_dict["flies"] if e["id"] == 1][0]
    assert her["hz"]["MDN"] > 20 and g.state_dict["hz"]["MDN"] < 5
    zap_events = [e for e in g.events.items if e["text"].startswith("zap MDN")]
    assert zap_events and zap_events[-1]["fly"] == 1
    assert g.action({"type": "silence", "spec": "MN9", "fly": 1})["ok"] is True
    assert g.action({"type": "watch", "spec": "DNa02", "key": "hers", "fly": 1})["ok"] is True
    assert g.action({"type": "autopilot", "on": False, "fly": 1})["ok"] is True
    assert g.action({"type": "state", "hunger": 0.1, "fly": 1})["ok"] is True
    assert g.action({"type": "place_fly", "x": 3.0, "y": 4.0, "fly": 1})["ok"] is True
    g.tick()
    assert "MN9" in f1.user_silenced and "MN9" not in f0.user_silenced and "hers" in f1.custom_readouts and "hers" not in f0.custom_readouts
    assert f1.autopilot is False and f1.state.hunger == pytest.approx(0.1, abs=0.01) and f0.state.hunger > 0.5
    assert abs(f1.body.pose.x - 3.0) < 1.0 and abs(f1.body.pose.y - 4.0) < 1.0
    assert g.action({"type": "unwatch", "key": "hers"})["ok"] is False                     # fly 0 has no such watch
    assert g.action({"type": "unwatch", "key": "hers", "fly": 1})["ok"] is True
    # a per-fly spec is checked against that fly's connectome: the female has no pIP10
    assert g.action({"type": "zap", "spec": "pIP10", "fly": 1})["ok"] is False
    assert g.action({"type": "zap", "spec": "pIP10", "fly": 0})["ok"] is True and g.action({"type": "zap", "spec": "pIP10"})["ok"] is True


def test_a_bad_fly_field_is_refused_and_queues_nothing(served_pair):
    g, base = served_pair
    while not g.actions.empty():
        g.actions.get()
    for bad in ('"1"', "1.5", "true", "Infinity", "1e400", "7", "-1", '"a"', "[1]"):
        r = _post(base, f'{{"type": "zap", "spec": "MDN", "fly": {bad}}}'.encode())
        assert r["ok"] is False and ("whole number" in r["error"] or "no fly" in r["error"]), (bad, r)
    assert g.actions.empty()
    assert _post(base, {"type": "zap", "spec": "MDN", "fly": None})["ok"] is True          # null: fly 0
    assert _post(base, {"type": "zap", "spec": "pIP10", "fly": 1})["ok"] is False           # not in her connectome
    assert _post(base, {"type": "tool", "tool": "hand", "fly": 1})["ok"] is True            # a world action: fly ignored


def test_every_per_fly_endpoint_takes_fly(served_pair, fconn):
    g, base = served_pair
    code, lay = _get(base, "/api/layout?fly=1")
    assert code == 200 and lay["sex"] == "female" and lay["dataset"] == fconn.dataset
    groups = {r["group"] for r in lay["readouts"]}
    assert "Her decisions" in groups and {r["id"] for r in lay["checks"]} == {r["id"] for r in lay["checks"]} - {"groom", "sound", "wall", "court", "genetics"}
    code, lay0 = _get(base, "/api/layout")
    assert code == 200 and lay0["sex"] == "male" and lay0 == json.loads(g.layout_json)
    code, h = _get(base, "/api/history?fly=1&keys=DNp37,MDN")
    assert code == 200 and set(h["history"]) == {"DNp37", "MDN"} and len(h["history"]["DNp37"]) == 3
    code, h0 = _get(base, "/api/history?keys=DNp37")
    assert code == 200 and h0["history"] == {}                                             # he has no such readout
    code, t = _get(base, "/api/types?fly=1&q=pIP")
    assert code == 200 and t["types"] == []                                                # not in her connectome
    code, t = _get(base, "/api/types?q=pIP")
    assert code == 200 and t["types"] and t["types"][0]["type"] == "pIP10"
    code, l = _get(base, "/api/learning?fly=1")
    assert code == 200 and l["learning"] is not None
    code, pr = _get(base, "/api/parts?fly=1")
    assert code == 200 and pr["on"] is False and "counts" in pr
    code, d = _get(base, "/api/decoder?fly=1")
    assert code == 200 and "MDN" in d["targets"] and "pIP10" not in d["targets"]
    code, gn = _get(base, "/api/genome?fly=1")
    assert code == 200 and gn["level"] == "real"
    code, ge = _get(base, "/api/genes?fly=1")
    assert code == 200 and "expression" in ge
    i = int(fconn.select("MDN")[0])
    code, n = _get(base, f"/api/neuron?fly=1&index={i}")
    assert code == 200 and n["neuron"]["type"] == "MDN"
    code, pa = _get(base, "/api/partners?fly=1&spec=MN9")
    assert code == 200 and pa["n"] == fconn.count("MN9")
    code, tr = _get(base, "/api/trace?fly=1&from=LC10a/L&to=DNa02/L&hops=3")
    assert code == 200 and "paths" in tr
    code, rec = _get(base, "/api/recording?fly=1")
    assert code == 200 and rec["settings"]["seed"] == g.seed + 1000
    for path in ("/api/layout?fly=2", "/api/history?fly=x", "/api/neuron?fly=-1&index=0", "/api/types?fly=1.0&q=a"):
        code, err = _get(base, path)
        assert code == 404 and err["ok"] is False and "no fly" in err["error"], path
    assert _get(base, "/api/state")[1]["flies"][1]["sex"] == "female"


# ------------------------------------------------------------------ the pair scenario and the pair checks (plan 5.9 items 3, 5)
from virtual_fly.scenarios import PAIR_SCENARIOS, SCENARIOS  # noqa: E402


def test_the_pair_scenario_needs_a_partner(conn):
    """pair_courtship lives in PAIR_SCENARIOS, not SCENARIOS: a single fly's layout, ids and refusals are what they were."""
    single = Game(build_brain(conn, "game", seed=0), autopilot=False, seed=1)
    r = single.action({"type": "scenario", "id": "pair_courtship"})
    assert r["ok"] is False and "two-fly scenario" in r["error"] and "--partner female" in r["error"] and single.actions.empty()
    assert single.action({"type": "scenario", "id": "nope"}) == {"ok": False, "error": "unknown scenario nope"}
    lay = json.loads(single._make_layout())
    assert [s["id"] for s in lay["scenarios"]] == list(SCENARIOS) and "pair_courtship" not in {s["id"] for s in lay["scenarios"]}
    assert {s["id"]: s["description"] for s in lay["scenarios"]}["courtship"] == SCENARIOS["courtship"].description
    assert [c["id"] for c in lay["checks"]] == [c[0] for c in CHECKS]
    with pytest.raises(KeyError, match="two-fly scenario"):
        single.scenario.start("pair_courtship")
    assert "pair_courtship" in PAIR_SCENARIOS and "pair_courtship" not in SCENARIOS


def test_the_pair_scenario_places_both_flies_measures_and_logs_its_totals(conn, fconn):
    g = pair(conn, fconn)
    f0, f1 = g.flies
    assert g.action({"type": "scenario", "id": "pair_courtship"}) == {"ok": True}
    g.tick()
    assert g.world.female is None
    assert (f0.body.pose.x, f0.body.pose.y) == pytest.approx((-10.0, -8.0), abs=0.5) and f0.body.pose.h == pytest.approx(0.3, abs=0.1)
    assert (f1.body.pose.x, f1.body.pose.y) == pytest.approx((14.0, 10.0), abs=0.5) and -math.pi <= f1.body.pose.h <= math.pi
    st = g.scenario.status()
    assert st["id"] == "pair_courtship" and st["name"] == "Courtship, two brains" and (st["step"], st["steps"]) == (1, 2)
    for _ in range(3):
        g.tick()
    m = g.scenario.measure
    assert set(m) == {"distance_mm", "he_sings", "she_hears_hz", "her_DNp37_hz", "her_DNp13_hz", "her_vpoEN_hz",
                      "her_speed_mm_s", "taps", "bursts_male", "bursts_female"}
    assert m["distance_mm"] == pytest.approx(math.hypot(f1.body.pose.x - f0.body.pose.x, f1.body.pose.y - f0.body.pose.y), abs=0.5)
    assert m["he_sings"] is False and m["taps"] == 0 and g.scenario.store["ticks"] == 3      # the placing tick is not measured
    while g.scenario.current is not None:                                    # run it to its end
        g.scenario._next()
    done = [e for e in g.events.items if e["text"].startswith("two-fly courtship finished")]
    assert len(done) == 1
    text = done[0]["text"]
    for part in ("within 15 mm", "he sang", "tapped her", "she heard", "her DNp37", "DNp13", "vpoEN", "her speed", "giant-fibre bursts"):
        assert part in text, part
    assert "acceptance" not in text.lower() and "rejection" not in text.lower()
    assert any(e["text"] == "scenario finished: Courtship, two brains" for e in g.events.items)
    assert g.scenario.store == {} and g.scenario.measure == {} and g.world.tool == "lure"
    # the pair layouts list it after the single-fly scenarios, and the courtship scenario has its two-fly text
    for f in (f0, f1):
        lay = json.loads(f._make_layout())
        ids = [s["id"] for s in lay["scenarios"]]
        assert ids[:len(SCENARIOS)] == list(SCENARIOS) and ids[len(SCENARIOS):] == list(PAIR_SCENARIOS)
        texts = {s["id"]: s["description"] for s in lay["scenarios"]}
        assert texts["courtship"] == SCENARIOS["courtship"].pair("male", "female") and "Nothing links the two brains" in texts["courtship"]
        assert texts["pair_courtship"] == PAIR_SCENARIOS["pair_courtship"].description
        assert not any(w in json.dumps(lay["scenarios"]).lower() for w in ("acceptance", "rejection"))


def test_the_pair_checks_belong_to_the_fly_they_describe(conn, fconn, monkeypatch):
    """Plan 5.9 item 5: pair checks shown only with a partner, ticked from that fly's own senses, kept across New fly."""
    g = pair(conn, fconn)
    f0, f1 = g.flies
    f0.autopilot = f1.autopilot = False
    texts0 = {c["id"]: c["text"] for c in json.loads(f0._make_layout())["checks"]}
    texts1 = {c["id"]: c["text"] for c in json.loads(f1._make_layout())["checks"]}
    assert set(texts0) >= {"pair:seen", "pair:sang", "pair:tapped"} and not {"pair:heard", "pair:seen_him", "pair:touched"} & set(texts0)
    assert set(texts1) >= {"pair:heard", "pair:seen_him", "pair:touched"} and not {"pair:seen", "pair:sang", "pair:tapped"} & set(texts1)
    assert "leg taste cells" in texts1["pair:touched"] and "nothing fires in her" in texts1["pair:touched"]
    assert not any(i.startswith("pair:") for i in f0.done | f1.done)
    # a tap, nose to nose: his leg taste cells fire (his pair:tapped); she was tapped (her pair:touched)
    place(f0, 0.0, 0.0, 0.0)
    place(f1, 5.0, 0.0, math.pi)
    g.tick()
    assert "pair:tapped" in f0.done and "pair:touched" in f1.done
    assert "pair:touched" not in f0.done and "pair:tapped" not in f1.done and "pair:heard" not in f1.done
    # he sings at her (his pair:sang) and she hears it (her pair:heard)
    _force_decode(f0, monkeypatch, song=0.9, court=0.9)
    for _ in range(3):
        g.tick()
    assert "pair:sang" in f0.done and "pair:heard" in f1.done and "pair:sang" not in f1.done and "pair:heard" not in f0.done
    # she crosses his view (his pair:seen), then he crosses hers (her pair:seen_him)
    assert "pair:seen" not in f0.done
    for k in range(12):
        place(f0, 0.0, 0.0, 0.0)
        place(f1, 10.0, -6.0 + k, math.pi)
        g.tick()
    assert "pair:seen" in f0.done                       # (she faced him while crossing, so she may have seen him too)
    for k in range(12):
        place(f1, 0.0, 0.0, 0.0)
        place(f0, 10.0, -6.0 + k, math.pi)
        g.tick()
    assert "pair:seen_him" in f1.done
    # New fly keeps every fly's checklist, the pair items included
    assert g.action({"type": "reset"})["ok"] is True
    g.tick()
    assert {"pair:seen", "pair:sang", "pair:tapped"} <= f0.done and {"pair:heard", "pair:seen_him", "pair:touched"} <= f1.done
    # the state's flies entries carry them (each fly its own)
    st = g.state_dict
    assert "pair:sang" in st["flies"][0]["done"] and "pair:heard" in st["flies"][1]["done"] and "pair:heard" not in st["flies"][0]["done"]


def test_the_pair_checks_are_off_the_single_fly_and_two_males_get_the_eyes_item_only(conn, fconn):
    single = Game(build_brain(conn, "game", seed=0), autopilot=False, seed=1)
    assert single.flies[0].pair_checks() == [] and not any(c["id"].startswith("pair:") for c in json.loads(single._make_layout())["checks"])
    # two males: no female in the dish, so neither fly gets the male's items, only the eyes' (a tap on a male tastes
    # nothing, there is no song to hear); a female with a male partner gets hers
    mm = Game(build_brain(conn, "game", seed=0), autopilot=False, seed=1, brain_procs="off", partner={"conn": conn})
    assert [i for i, _ in mm.flies[0].pair_checks()] == ["pair:seen"] == [i for i, _ in mm.flies[1].pair_checks()]
    assert "other male" in dict(mm.flies[0].pair_checks())["pair:seen"]
    fm = Game(build_brain(fconn, "game", seed=0), autopilot=False, seed=1, brain_procs="off", partner={"conn": conn})
    assert [i for i, _ in fm.flies[0].pair_checks()] == ["pair:heard", "pair:seen_him", "pair:touched"]
    assert [i for i, _ in fm.flies[1].pair_checks()] == ["pair:seen", "pair:sang", "pair:tapped"]


# ------------------------------------------------------------------ what the third review found (docs/TWO_FLIES_PROGRESS.md)
import io  # noqa: E402
import time  # noqa: E402

from virtual_fly.agent import PAIR_SEEN_MM  # noqa: E402
from virtual_fly.scenarios import pair_available  # noqa: E402


def test_each_flys_entry_says_whether_its_walking_urge_is_on(conn, fconn):
    """Finding 1: autopilot is a per-fly action, so each flies entry carries its own (the page's W key and checkbox
    read the focused fly's, not the top level's, which is fly 0's)."""
    g = pair(conn, fconn)                                                     # the fixture: his urge off, hers on
    g.tick()
    f0, f1 = g.state_dict["flies"]
    assert f0["autopilot"] is False and f1["autopilot"] is True and g.state_dict["autopilot"] is False
    assert g.action({"type": "autopilot", "on": True})["ok"] is True
    assert g.action({"type": "autopilot", "on": False, "fly": 1})["ok"] is True
    g.tick()
    f0, f1 = g.state_dict["flies"]
    assert f0["autopilot"] is True and f1["autopilot"] is False and g.state_dict["autopilot"] is True
    assert g.flies[0].autopilot is True and g.flies[1].autopilot is False
    single = Game(build_brain(conn, "game", seed=0), autopilot=False, seed=1)
    single.tick()
    assert set(single.state_dict) == STATE_KEYS_ONE                           # no new key with one fly


def test_every_flys_brain_records_its_spikes(served_pair):
    """Finding 2: the record action started spike recording on fly 0's brain only, so /api/spikes?fly=1 was an
    empty file; now every fly's brain records and ?fly=k downloads fly k's."""
    g, base = served_pair
    assert _post(base, {"type": "record", "on": True, "spikes": True})["ok"] is True
    for _ in range(20):
        g.tick()
    assert g.state_dict["recording"] == {"frames": 20, "spikes": True, "active": True}
    sizes = {}
    for k in (0, 1):
        with urllib.request.urlopen(urllib.request.Request(f"{base}/api/spikes?fly={k}"), timeout=10) as r:
            assert r.status == 200 and r.headers["Content-Type"] == "application/octet-stream"
            z = np.load(io.BytesIO(r.read()))
        sizes[k] = int(z["neuron"].size)
        assert z["time_ms"].size == sizes[k] == z["body_id"].size
        assert g.flies[k].io.record("active") is True
    assert sizes[0] > 0 and sizes[1] > 0
    assert _post(base, {"type": "record", "on": False})["ok"] is True
    g.tick()
    assert g.flies[0].io.record("active") is False and g.flies[1].io.record("active") is False
    with urllib.request.urlopen(urllib.request.Request(f"{base}/api/spikes?fly=1"), timeout=10) as r:   # the kept take
        assert np.load(io.BytesIO(r.read()))["neuron"].size == sizes[1]
    assert _post(base, {"type": "record", "on": True})["ok"] is True                        # a new take without spikes
    g.tick()
    assert g.flies[1].io.record("active") is False
    with urllib.request.urlopen(urllib.request.Request(f"{base}/api/spikes?fly=1"), timeout=10) as r:
        assert np.load(io.BytesIO(r.read()))["neuron"].size == 0
    _post(base, {"type": "record", "on": False})
    g.tick()


def test_a_paused_game_zeroes_every_flys_rates(conn, fconn):
    """Finding 3: paused, the top level published sps 0 while flies[k] kept the last tick's sps and graded_eps."""
    g = pair(conn, fconn)
    for _ in range(24):
        g.tick()
    assert g.state_dict["sps"] > 0 and all(e["sps"] > 0 for e in g.state_dict["flies"])
    g.paused = True
    g.tick_paused()
    st = g.state_dict
    assert st["sps"] == 0 and st["graded_eps"] == 0 and len(st["flies"]) == 2
    assert [(e["sps"], e["graded_eps"]) for e in st["flies"]] == [(0, 0), (0, 0)]
    th = threading.Thread(target=g.loop, daemon=True)                       # the loop itself, paused
    seq = g.seq
    th.start()
    deadline = time.time() + 10
    while g.seq == seq and time.time() < deadline:
        time.sleep(0.01)
    g.stop_loop.set()
    th.join(timeout=10)
    assert not th.is_alive() and g.seq > seq
    assert g.state_dict["sps"] == 0 and [e["sps"] for e in g.state_dict["flies"]] == [0, 0]
    g.paused = False
    g.stop_loop.clear()
    g.tick()
    assert g.state_dict["sps"] > 0 and g.state_dict["flies"][0]["sps"] == g.state_dict["sps"] > 0


def test_the_pair_scenario_finds_the_male_and_the_female_by_sex(conn, fconn, monkeypatch):
    """Finding 9: with --female --partner male the female is fly 0 and the male fly 1; the placing, the measure and
    the summary follow the sexes, not the indices."""
    g = Game(build_brain(fconn, "game", seed=0), autopilot=False, seed=1, brain_procs="off",
             partner={"conn": conn, "autopilot": False})
    her, him = g.flies
    assert (her.sex, him.sex) == ("female", "male") and pair_available(g, "pair_courtship") is None
    assert "pair_courtship" in {s["id"] for s in json.loads(her._make_layout())["scenarios"]}
    assert g.action({"type": "scenario", "id": "pair_courtship"}) == {"ok": True}
    g.tick()
    assert (him.body.pose.x, him.body.pose.y) == pytest.approx((-10.0, -8.0), abs=0.5) and him.body.pose.h == pytest.approx(0.3, abs=0.1)
    assert (her.body.pose.x, her.body.pose.y) == pytest.approx((14.0, 10.0), abs=0.5)
    assert g.scenario.measure["distance_mm"] == pytest.approx(30.0, abs=0.5) and g.world.female is None
    # he sings next to her, taps her, and his giant fibre bursts: every number lands on the right fly
    _force_decode(him, monkeypatch, song=0.9, court=0.9)
    place(him, 0.0, 0.0, 0.0)
    place(her, 5.0, 0.0, math.pi)
    for _ in range(3):
        g.tick()
    m = g.scenario.measure
    assert m["he_sings"] is True and m["she_hears_hz"] > 0 and m["taps"] >= 1 and him.mode == "court"
    st = g.scenario.store
    assert st["bursts"] == [m["bursts_male"], m["bursts_female"]]
    for his_gf, her_gf in ((5, 0), (0, 5)):                                   # read by the next tick's measure, with the
        before, prev = list(st["bursts"]), list(st["gf_prev"])              # game's own two-tick rule (her brain may burst too)
        him.bt.gf, her.bt.gf = his_gf, her_gf
        g.tick()
        assert st["bursts"][0] == before[0] + (his_gf + prev[0] >= GF_BURST) and st["gf_prev"][0] == his_gf
        assert st["bursts"][1] == before[1] + (her_gf + prev[1] >= GF_BURST) and st["gf_prev"][1] == her_gf
        assert (g.scenario.measure["bursts_male"], g.scenario.measure["bursts_female"]) == tuple(st["bursts"])
    assert st["bursts"][0] >= 1 and st["bursts"][1] >= 1
    sang, (his, hers) = st["sang"], st["bursts"]
    while g.scenario.current is not None:
        g.scenario._next()
    text = [e for e in g.events.items if e["text"].startswith("two-fly courtship finished")][0]["text"]
    assert f"he sang {sang * TICK_MS / 1000:.1f} s" in text and sang >= 4 and f"giant-fibre bursts: his {his}, hers {hers}" in text
    # the pair's courtship text is from her side, and the two-fly scenario names no index
    texts = {s["id"]: s["description"] for s in json.loads(her._make_layout())["scenarios"]}
    assert texts["courtship"].startswith("The simulated male is placed ahead of her") and "whichever" in texts["pair_courtship"]


def test_two_males_cannot_run_the_pair_scenario_and_are_not_told_to_add_a_female(conn, fconn):
    """Findings 9 and 12: pair_courtship is for a male and a female; with --partner male it is neither listed nor
    accepted, the courtship texts speak of a male partner, and no checklist item asks for a female (D9)."""
    mm = Game(build_brain(conn, "game", seed=0), autopilot=False, seed=1, brain_procs="off",
              partner={"conn": conn, "autopilot": False})
    f0, f1 = mm.flies
    lay0, lay1 = (json.loads(f._make_layout()) for f in mm.flies)
    assert "pair_courtship" not in {s["id"] for s in lay0["scenarios"]} | {s["id"] for s in lay1["scenarios"]}
    assert [s["id"] for s in lay0["scenarios"]] == list(SCENARIOS)
    r = mm.action({"type": "scenario", "id": "pair_courtship"})
    assert r["ok"] is False and "male and a female" in r["error"] and "both flies are male" in r["error"] and mm.actions.empty()
    with pytest.raises(KeyError, match="male and a female"):
        mm.scenario.start("pair_courtship")
    assert mm.scenario.current is None
    text = {s["id"]: s["description"] for s in lay0["scenarios"]}["courtship"]
    assert text.startswith("A second simulated male") and not any(w in text for w in ("female", " her ", " she "))
    checks = {c["id"]: c["text"] for c in lay0["checks"]}
    assert "court" not in checks and "fruitless" in checks["genetics"] and "no song to lose" in checks["genetics"]
    assert "add a female" not in json.dumps(lay0["checks"]).lower()
    assert [i for i in checks if i.startswith("pair:")] == ["pair:seen"]
    # the eyes' item ticks when the other male crosses his view, and nothing else of the pair's ever does
    for k in range(12):
        place(f0, 0.0, 0.0, 0.0)
        place(f1, 10.0, -6.0 + k, math.pi)
        mm.tick()
    assert "pair:seen" in f0.done and {i for i in f0.done | f1.done if i.startswith("pair:")} <= {"pair:seen"}
    assert {i for i in f0.done if i.startswith("pair:")} <= {i for i, _ in f0.pair_checks()}
    # the plain courtship scenario still places the partner where the scripted female would have stood
    assert mm.action({"type": "scenario", "id": "courtship"})["ok"] is True
    mm.tick()
    assert mm.world.female is None and (f1.body.pose.x, f1.body.pose.y) == pytest.approx((14.0, 10.0), abs=0.5)
    # two females: the same rule, from her side
    ff = Game(build_brain(fconn, "game", seed=0), autopilot=False, seed=1, brain_procs="off", partner={"conn": fconn})
    assert "both flies are female" in pair_available(ff, "pair_courtship")
    assert [i for i, _ in ff.flies[0].pair_checks()] == ["pair:seen"] and "other female" in dict(ff.flies[0].pair_checks())["pair:seen"]
    assert {s["id"] for s in json.loads(ff.flies[0]._make_layout())["scenarios"]} == set(SCENARIOS)


def test_a_sugar_drop_crossing_his_eye_does_not_tick_pair_seen(conn, fconn, monkeypatch):
    """Finding 10: the retina's small-object flag does not say what moved, so the 'seen' items need the other fly
    within PAIR_SEEN_MM in the field of the eye that saw something small move."""
    for what in ("sugar", "post"):
        g = pair(conn, fconn)
        f0, f1 = g.flies
        f1.autopilot = False
        _force_decode(f0, monkeypatch, forward=1.0)
        place(f0, -10.0, 0.0, 0.0)
        place(f1, -36.0, 0.0, 0.0)                                            # 26 mm straight behind him: no eye looks there
        if what == "sugar":
            g.world.add_food("sugar", 2.0, 4.0)
        else:
            g.world.add_obstacle(20.0, 14.0, 4.0)
        small = set()
        for _ in range(80):
            g.tick()
            small |= set(f0.senses_now.get("small", ""))
        assert small, what                                                   # it did cross his eye as a small moving object
        assert "pair:seen" not in f0.done and f0.body.pose.x > -8.0, what      # and he walked past it
    assert PAIR_SEEN_MM >= 20.0
    g = pair(conn, fconn)                                                     # the partner crossing his view: ticked, as before
    f0, f1 = g.flies
    f1.autopilot = False
    for k in range(12):
        place(f0, 0.0, 0.0, 0.0)
        place(f1, 10.0, -6.0 + k, math.pi)
        g.tick()
    assert "pair:seen" in f0.done


def test_pair_sang_needs_her_within_song_range(conn, fconn, monkeypatch):
    """Finding 11: the item says 'within 15 mm of her'; a song she cannot hear does not tick it."""
    g = pair(conn, fconn)
    f0, f1 = g.flies
    f0.autopilot = f1.autopilot = False
    _force_decode(f0, monkeypatch, song=0.9, court=0.9)
    place(f0, -20.0, 0.0, 0.0)
    place(f1, 20.0, 0.0, math.pi)                                             # 40 mm: beyond SONG_FAR_MM
    for _ in range(3):
        g.tick()
    assert f0.mode == "court" and "court" in f0.done and "pair:sang" not in f0.done and "hears_song" not in f1.senses_now
    assert 10.0 < social.SONG_FAR_MM < 40.0
    place(f0, 0.0, 0.0, 0.0)
    place(f1, 10.0, 0.0, math.pi)
    g.tick()
    assert "pair:sang" in f0.done and "hears_song" in f1.senses_now


def test_fly_takes_a_few_ascii_digits_only(served_pair):
    """Review (api lens): str.isdigit took '²' and the like, which int() refused with a 500; a 5,000-digit string
    tripped int()'s conversion limit. Both answer the documented 404 now."""
    g, base = served_pair
    for q in ("%C2%B2", "1" * 5000, "%EF%BC%91", "1.0", "+1", "%201", "0x1", "-0"):
        code, err = _get(base, f"/api/layout?fly={q}")
        assert code == 404 and err["ok"] is False and err["error"].startswith("no fly") and len(err["error"]) < 40, q
    assert _get(base, "/api/layout?fly=1")[0] == 200 and _get(base, "/api/layout?fly=001")[0] == 200
    assert _get(base, "/api/layout?fly=7")[0] == 404 and _get(base, "/api/layout")[0] == 200


def test_the_pair_measure_skips_the_senses_of_the_tick_it_started_in(conn, fconn):
    """Review (scenario lens): Game.tick runs the actions (which place the flies) and then the measure, whose senses
    are the tick before's; a tap from before the start is not counted."""
    g = pair(conn, fconn)
    f0, f1 = g.flies
    f0.autopilot = f1.autopilot = False
    place(f0, 0.0, 0.0, 0.0)
    place(f1, 5.0, 0.0, math.pi)
    g.tick()
    assert f0.senses_now.get("touches_fly") == 1                              # nose to nose: a tap before the scenario
    assert g.action({"type": "scenario", "id": "pair_courtship"}) == {"ok": True}
    g.tick()
    m = g.scenario.measure
    assert m["taps"] == 0 and m["distance_mm"] == pytest.approx(30.0, abs=0.5) and g.scenario.store["ticks"] == 0
    for _ in range(40):
        g.tick()
    assert g.scenario.store["taps"] == 0 and g.scenario.measure["taps"] == 0 and g.scenario.store["ticks"] == 40
    assert "fresh" not in g.scenario.store


def test_pair_checks_are_listed_only_for_channels_that_are_on(conn, fconn, monkeypatch):
    """Review (scenario lens): an item whose channel is off could never tick, so it is not listed."""
    g = pair(conn, fconn, social=SocialConfig(seen=False, song=False, contact=False))
    assert g.flies[0].pair_checks() == [] and g.flies[1].pair_checks() == []
    assert not any(c["id"].startswith("pair:") for f in g.flies for c in json.loads(f._make_layout())["checks"])
    g = pair(conn, fconn, social=SocialConfig(song=False))
    f0, f1 = g.flies
    assert [i for i, _ in f0.pair_checks()] == ["pair:seen", "pair:tapped"]
    assert [i for i, _ in f1.pair_checks()] == ["pair:seen_him", "pair:touched"]
    f0.autopilot = f1.autopilot = False
    _force_decode(f0, monkeypatch, song=0.9, court=0.9)                        # he sings next to her: no song channel, no item
    place(f0, 0.0, 0.0, 0.0)
    place(f1, 8.0, 0.0, math.pi)
    for _ in range(3):
        g.tick()
    assert "court" in f0.done and "pair:sang" not in f0.done and "pair:heard" not in f1.done
    g = pair(conn, fconn)
    assert [i for i, _ in g.flies[0].pair_checks()] == ["pair:seen", "pair:sang", "pair:tapped"]
    assert [i for i, _ in g.flies[1].pair_checks()] == ["pair:heard", "pair:seen_him", "pair:touched"]
