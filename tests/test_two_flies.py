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
    off = pair(conn, fconn, social=SocialConfig(seen=False))
    off.tick()
    assert "small" not in off.flies[0].senses_now or off.flies[0].senses_now.get("small") == ""


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
    g.tick()
    assert "small" not in f0.senses_now                                                   # ... but the senses never ask for them


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
