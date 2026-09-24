"""The sensorimotor loop (game.py) driven by the synthetic brain, without a server."""

import base64
import json
import math
import time

import numpy as np
import pytest

from virtual_fly.body import JUMP_TIME
from virtual_fly.game import (CHECKS, HIDDEN_READOUTS, READOUTS, TICK_MS, EventLog, Game, InternalState, MotorDecoder)
from virtual_fly.scenarios import SCENARIOS
from virtual_fly.settings import build_brain

STATE_KEYS = {"seq", "t", "rtf", "speed", "fly", "world", "autopilot", "paused", "senses", "retina", "hz", "motor",
              "driver", "mode", "spikes", "sps", "stims", "calms", "msg", "silenced", "baseline", "modulated", "custom",
              "done", "state", "learning", "events", "event_seq", "scenario", "recording", "genome"}
LAYOUT_KEYS = {"n", "w", "h", "d", "x", "y", "z", "region", "regions", "arena_r", "fly_half", "tick_ms", "presets",
               "types", "edges", "synapses", "readouts", "checks", "odours", "scenarios", "retina", "profile", "settings",
               "decoder", "columnar_vision", "whats_real", "genetics", "genome", "parts"}
FLY_KEYS = {"x", "y", "h", "v", "w", "mode", "prob", "legs", "groom", "wingL", "wingR", "abdomen", "jump", "hx", "hy", "dist"}


@pytest.fixture
def game(conn):
    return Game(build_brain(conn, "game", seed=0), autopilot=False, seed=1)


def ticks(game, n):
    for _ in range(n):
        game.tick()
    return game.state_dict


def head_xy(game):
    return game.body.pose.head


# ------------------------------------------------------------------ construction and publishing
def test_layout_matches_the_documented_schema(game, conn):
    lay = json.loads(game.layout_json)
    assert set(lay) == LAYOUT_KEYS
    assert lay["n"] == conn.n and len(lay["x"]) == len(lay["y"]) == len(lay["z"]) == len(lay["region"]) == conn.n
    no_soma = np.isnan(conn.soma[:, 0])
    assert all(lay["x"][i] == -1 for i in np.flatnonzero(no_soma)) and max(lay["x"]) <= max(lay["w"], lay["h"]) + 1
    assert len(lay["regions"]) == 9 and set(lay["region"]) <= set(range(9))
    kc = int(conn.select("class:Kenyon_Cell")[0])
    assert lay["region"][kc] == 7 and lay["region"][int(conn.select("MBON11")[0])] == 8
    assert lay["region"][int(conn.select("DNp01")[0])] == 2 and lay["region"][int(conn.select("TTMn")[0])] == 5
    assert lay["arena_r"] == 50 and lay["tick_ms"] == TICK_MS and lay["profile"] == "game"
    assert [r["key"] for r in lay["readouts"]] == [r[0] for r in READOUTS]
    assert set(lay["readouts"][0]) == {"key", "spec", "label", "group", "max", "colour", "genes"}
    assert lay["types"][0] == "KCg-m" and len(lay["types"]) == len(conn.type_counts())
    assert lay["edges"] == conn.n_edges and lay["synapses"] == int(conn.n_syn.sum())
    assert [c["id"] for c in lay["checks"]] == [c[0] for c in CHECKS]
    assert {o["id"] for o in lay["odours"]} == {"vinegar", "banana", "geosmin", "yeast"}
    assert {s["id"] for s in lay["scenarios"]} == set(SCENARIOS)
    assert lay["settings"]["silenced"] == ["class:ALLN", "class:DAN"] and lay["columnar_vision"] is True
    assert set(lay["whats_real"]) == {"wiring", "hand_built", "not_modelled"}
    assert lay["presets"][0]["spec"] == "MDN" and lay["retina"]["L"]["n_az"] == game.retina.eyes["L"].n_az
    dec = lay["decoder"]
    assert dec["DNp01"]["direct_motor_synapses"] > 0 and dec["MN9"]["direct_motor_synapses"] == 0
    assert set(dec["DNa02"]["two_hop_motor_synapses_by_neuromere"]) == {"T1", "T2", "T3"}
    assert set(game.readouts) >= set(HIDDEN_READOUTS) | {r[0] for r in READOUTS}
    assert set(game.brain.monitors) == {r[0] for r in READOUTS}


def test_tick_publishes_a_documented_state(game, conn):
    assert game.state_json == b"{}" and game.seq == 0
    state = ticks(game, 1)
    assert set(state) == STATE_KEYS and json.loads(game.state_json) == state
    assert state["seq"] == 1 and state["t"] == pytest.approx(TICK_MS / 1000) and game.seq == 1
    assert set(state["fly"]) == FLY_KEYS and state["mode"] == "walk" and state["fly"]["mode"] == "walk"
    assert set(state["world"]) == {"food", "obstacles", "odours", "puffs", "wind", "stripes", "female", "hand", "tool"}
    assert state["world"]["stripes"] == {"count": 0, "phase": 0.0, "drum_speed": 0.0}
    assert set(state["hz"]) == set(game.readouts) and all(v == 0 for v in state["hz"].values())
    assert set(state["motor"]) == {"forward", "yaw", "backward", "halt", "feed", "groom", "song", "court"}
    assert len(base64.b64decode(state["retina"]["L"])) == game.retina.eyes["L"].n
    assert state["baseline"] == ["class:ALLN", "class:DAN"] and state["silenced"] == [] and state["modulated"] == {}
    assert state["driver"] == "nothing: the brain is quiet" and state["spikes"] == [] and state["sps"] == 0
    assert state["state"] == {"hunger": 0.7, "thirst": 0.3, "arousal": 0.0} and state["done"] == []
    learn = state["learning"]
    assert set(learn) == {"enabled", "depressed_fraction", "events", "learned_bias", "smelling", "mbon"}
    assert learn["enabled"] and learn["mbon"]["MBON11"] == {"strength": 1.0, "now": 1.0, "valence": 1, "dopamine": 0.0}
    assert learn["mbon"]["MBON01"]["valence"] == -1
    assert state["scenario"] is None and state["recording"] is None and state["custom"] == {}
    assert state["events"] == [] and state["event_seq"] == 0 and not state["paused"] and not state["autopilot"]
    s2 = ticks(game, 1)
    assert s2["seq"] == 2 and s2["t"] == pytest.approx(0.05)


def test_subscribers_receive_each_published_state(game):
    q = game.subscribe()
    ticks(game, 2)
    assert q.qsize() == 2 and q.get_nowait() != q.get_nowait()
    game.unsubscribe(q)
    ticks(game, 1)
    assert q.empty()
    game.unsubscribe(q)                                                       # idempotent


# ------------------------------------------------------------------ behaviours through the wiring
def test_sugar_at_the_mouth_makes_the_fly_feed(game):
    hx, hy = head_xy(game)
    assert game.action({"type": "drop", "kind": "sugar", "x": hx, "y": hy}) == {"ok": True}
    state = ticks(game, 40)
    assert state["mode"] == "feed" and state["fly"]["prob"] > 0.5 and "feed" in state["done"]
    assert state["senses"].get("taste_sugar") and state["senses"].get("reward")
    assert state["hz"]["MN9"] > 20 and state["hz"]["G2N-1"] > 20 and state["hz"]["Scapula"] == 0
    assert "MN9 proboscis" in state["driver"]
    assert state["world"]["food"][0]["amount"] < 100 and game.state.hunger < 0.7
    assert any("eating" in e["text"] for e in game.events.items) and any(e["kind"] == "world" for e in game.events.items)
    assert state["stims"] >= 2 and state["sps"] > 0 and len(state["spikes"]) > 0


def test_bitter_at_the_mouth_keeps_mn9_silent(game):
    hx, hy = head_xy(game)
    game.action({"type": "drop", "kind": "bitter", "x": hx, "y": hy})
    state = ticks(game, 40)
    assert state["hz"]["Scapula"] > 50 and state["hz"]["MN9"] < 5 and state["mode"] != "feed"
    assert "bitter" in state["done"] and state["senses"].get("taste_bitter") and state["fly"]["prob"] < 0.1


def test_zap_mdn_walks_backward(game):
    assert game.action({"type": "zap", "spec": "MDN", "hz": 60, "secs": 1.0}) == {"ok": True, "n": 4}
    state = ticks(game, 30)
    assert state["mode"] == "backward" and state["fly"]["v"] < -2 and "MDN" in state["senses"]["zap"]
    assert {"zap", "moonwalk"} <= set(state["done"]) and state["hz"]["MDN"] > 30 and "moonwalker" in state["driver"]
    ticks(game, 20)
    assert game.zaps == [] and "zap" not in game.senses_now                     # 1 s has passed
    assert game.action({"type": "zap", "spec": "no_such_type"})["ok"] is False
    assert "unknown filter" in game.action({"type": "zap", "spec": "colour:red"})["error"]


def test_dust_makes_the_fly_groom(game):
    p = game.body.pose
    game.action({"type": "dust", "x": p.x, "y": p.y})
    state = ticks(game, 40)
    assert "groom" in state["done"] and state["senses"].get("dust") and state["hz"]["aDN1"] > 20
    assert any(e["text"] == "dust on the antennae" for e in game.events.items)
    far = Game(game.brain, autopilot=False, seed=1)
    far.action({"type": "dust", "x": p.x + 30, "y": p.y})
    far.tick()
    assert "dust" not in far.senses_now                                        # too far from the fly


def test_giant_fibre_spike_triggers_an_escape_jump(game):
    game.action({"type": "zap", "spec": "DNp01", "hz": 150, "secs": 0.2})
    state = ticks(game, 2)
    assert state["mode"] == "escape" and state["fly"]["jump"] is not None and game.body.pose.jump is not None
    assert "giant fibre" in state["driver"] and any("escape jump" in e["text"] for e in game.events.items)
    assert "escape" not in state["done"]                                       # no looming was seen
    assert state["fly"]["wingL"] > 0.3 and state["fly"]["wingR"] > 0.3 and game.gf_cooldown > 0
    state = ticks(game, int(JUMP_TIME / (TICK_MS / 1000)) + 2)
    assert state["fly"]["jump"] is None and game.body.pose.jump is None


def test_lure_on_the_left_turns_the_fly_left(game):
    game.action({"type": "tool", "tool": "lure"})
    p = game.body.pose
    yaws = []
    for k in range(20):                                                       # sweeps from 52 to 19 degrees left
        ang = 0.9 - k * 0.03
        game.action({"type": "hand", "x": p.x + 12 * math.cos(p.h + ang), "y": p.y + 12 * math.sin(p.h + ang)})
        game.tick()
        yaws.append(game.decoder.m["yaw"])
    state = game.state_dict
    assert "lure" in state["done"] and min(yaws) < -0.25 and state["world"]["tool"] == "lure"
    assert state["hz"]["LC10aL"] > 10 and state["hz"]["DNa02L"] > 0 and state["hz"]["DNa02R"] == 0
    game.action({"type": "hand_off"})
    game.tick()
    assert game.world.hand is None and game.state_dict["world"]["hand"] is None


def test_odour_drives_receptor_neurons_and_reward_teaches(game):
    hx, hy = head_xy(game)
    game.action({"type": "drop", "kind": "vinegar", "x": hx, "y": hy, "food": "sugar"})
    state = ticks(game, 60)
    assert state["senses"].get("smell") == "vinegar" and "smell" in state["done"] and state["hz"]["ORN"] > 10
    assert any(e["text"] == "smells apple cider vinegar" for e in game.events.items)
    assert state["world"]["odours"][0]["food"] == state["world"]["food"][0]["id"]
    learn = state["learning"]
    assert learn["smelling"] == "vinegar" and "learn" in state["done"] and learn["depressed_fraction"] > 0
    assert learn["mbon"]["MBON01"]["strength"] < 1.0 and learn["mbon"]["MBON11"]["strength"] == 1.0
    assert state["hz"]["PAM"] > 10 and state["senses"].get("reward")


# ------------------------------------------------------------------ lab actions
def test_silence_modulate_and_watch_actions(game):
    assert game.action({"type": "silence", "spec": "MN9"}) == {"ok": True, "n": 2}
    hx, hy = head_xy(game)
    game.action({"type": "drop", "kind": "sugar", "x": hx, "y": hy})
    state = ticks(game, 40)
    assert state["silenced"] == ["MN9"] and state["baseline"] == ["class:ALLN", "class:DAN"]
    assert state["hz"]["MN9"] > 20 and state["mode"] != "feed" and "silence" in state["done"]   # tastes, can't eat
    game.action({"type": "unsilence", "spec": "MN9"})
    state = ticks(game, 1)
    assert state["silenced"] == [] and game.brain.silenced.keys() == {"class:ALLN", "class:DAN"}
    game.action({"type": "modulate", "spec": "MN9", "factor": 1.5})
    assert ticks(game, 1)["modulated"] == {"MN9": 1.5} and game.brain.modulated["MN9"][1] == 1.5
    game.action({"type": "modulate", "spec": "MN9", "factor": 1.0})
    assert ticks(game, 1)["modulated"] == {} and game.brain.modulated == {}
    assert game.action({"type": "watch", "spec": "GNG232", "key": "relay"}) == {"ok": True, "n": 4}
    state = ticks(game, 1)
    assert state["custom"] == {"relay": "GNG232"} and "relay" in state["hz"] and "relay" in game.brain.monitors
    game.action({"type": "unwatch", "key": "relay"})
    state = ticks(game, 1)
    assert state["custom"] == {} and "relay" not in game.readouts and "relay" not in game.brain.monitors
    assert game.action({"type": "watch", "spec": ""})["ok"] is False


def test_watch_cannot_shadow_a_built_in_readout(game):
    before = {k: v.copy() for k, v in game.readouts.items()}
    for key in ("MN9", "DNp15L", "GF"):                                # shown and hidden readouts alike
        reply = game.action({"type": "watch", "spec": "GNG232", "key": key})
        assert reply["ok"] is False and "built-in" in reply["error"]
    # without a name, a spec that happens to be a readout key gets a distinct one
    assert game.action({"type": "watch", "spec": "MN9"}) == {"ok": True, "n": 2}
    state = ticks(game, 1)
    assert state["custom"] == {"watch:MN9": "MN9"} and "watch:MN9" in state["hz"]
    for key in ("MN9", "DNp15L", "nothing"):
        assert game.action({"type": "unwatch", "key": key})["ok"] is False
    ticks(game, 1)
    assert all(np.array_equal(game.readouts[k], v) for k, v in before.items())
    assert "MN9" in game.hz_shown and "DNp15L" in game.readouts
    # re-watching the same name replaces the spec and keeps one monitor
    assert game.action({"type": "watch", "spec": "GNG087", "key": "watch:MN9"})["ok"]
    state = ticks(game, 1)
    assert state["custom"] == {"watch:MN9": "GNG087"} and game.brain.monitors["watch:MN9"].spec == "GNG087"


def test_unwatch_removes_the_key_from_published_rates(game):
    game.action({"type": "watch", "spec": "GNG232", "key": "relay"})
    ticks(game, 1)
    game.action({"type": "unwatch", "key": "relay"})
    assert "relay" not in ticks(game, 1)["hz"]


def test_world_actions(game):
    game.action({"type": "wind", "angle": 0.0, "speed": 20.0})
    state = ticks(game, 2)
    assert state["world"]["wind"] == {"angle": 0.0, "speed": 20.0} and "wind" in state["done"]
    assert isinstance(state["senses"]["wind"], int) and state["hz"]["WindL"] > 0
    game.action({"type": "female", "on": True, "x": 10.0, "y": 10.0})
    state = ticks(game, 1)
    assert state["world"]["female"]["x"] == pytest.approx(10, abs=1) and any("female fly enters" in e["text"] for e in game.events.items)
    game.action({"type": "female", "on": False})
    assert ticks(game, 1)["world"]["female"] is None
    game.action({"type": "drop", "kind": "post", "x": 20.0, "y": 0.0, "r": 3.0})
    game.action({"type": "drop", "kind": "water", "x": -20.0, "y": 0.0})
    state = ticks(game, 1)
    assert state["world"]["obstacles"] == [{"id": game.world.obstacles[0].id, "x": 20.0, "y": 0.0, "r": 3.0}]
    assert state["world"]["food"][0]["kind"] == "water"
    game.action({"type": "remove", "id": game.world.obstacles[0].id})
    game.action({"type": "clear", "what": "food"})
    state = ticks(game, 1)
    assert state["world"]["obstacles"] == [] and state["world"]["food"] == []
    game.action({"type": "tool", "tool": "hand"})
    game.action({"type": "state", "hunger": 0.1, "thirst": 5})
    game.action({"type": "place_fly", "x": 5.0, "y": 6.0, "h": 0.0})
    game.action({"type": "speed", "value": 9})
    game.action({"type": "autopilot", "on": True})
    state = ticks(game, 1)
    assert state["world"]["tool"] == "hand" and state["state"]["hunger"] == 0.1 and state["state"]["thirst"] == 1.0
    assert state["speed"] == 3.0 and state["autopilot"] and abs(state["fly"]["x"] - 5.0) < 1
    assert "autopilot" in state["driver"] or state["fly"]["v"] != 0 or state["mode"] == "walk"
    game.action({"type": "pause", "on": True})
    game.action({"type": "bogus"})                                             # unknown types are ignored
    assert ticks(game, 1)["paused"]


def test_reset_calm_and_learning_actions(game):
    pl = game.brain.plasticity
    pl.scale[:] = 0.5
    hx, hy = head_xy(game)
    game.action({"type": "drop", "kind": "sugar", "x": hx, "y": hy})
    ticks(game, 5)
    assert game.world.food and game.brain.t > 0
    game.action({"type": "reset"})
    state = ticks(game, 1)
    assert state["world"]["food"] == [] and state["done"] == [] and state["t"] == pytest.approx(0.025)
    assert any(e["text"].startswith("New fly, fresh brain") for e in game.events.items)
    assert np.allclose(pl.scale, 0.5, atol=1e-3)                               # learned synapses are kept
    game.action({"type": "learning", "forget": True})
    state = ticks(game, 1)
    assert (pl.scale == 1.0).all() and state["msg"] == "Memories erased." and state["learning"]["enabled"]
    game.action({"type": "learning", "on": False})
    state = ticks(game, 1)
    assert not pl.enabled and not state["learning"]["enabled"] and not game.learning_on
    game.brain.stimulate("MN9", 100)
    game.brain.run(50)
    game.action({"type": "calm"})
    game.tick()
    assert game.brain.t == int(TICK_MS / game.brain.dt) and game.state_dict["msg"] == "Brain reset to rest."


def test_record_action(game):
    hx, hy = head_xy(game)
    game.action({"type": "drop", "kind": "sugar", "x": hx, "y": hy})           # something to record
    game.action({"type": "record", "on": True, "spikes": True})
    state = ticks(game, 3)
    assert state["recording"] == {"frames": 3, "spikes": True, "active": True} and game.brain.recording is not None
    frame = game.recording[-1]
    assert set(frame) == {"t", "fly", "mode", "hz", "senses", "sps"} and frame["t"] == pytest.approx(0.075)
    game.action({"type": "record", "on": False})
    state = ticks(game, 2)
    # stopping keeps the frames for download but stops adding to them
    assert game.brain.recording is None and any(e["text"].startswith("recording stopped") for e in game.events.items)
    assert state["recording"] == {"frames": 3, "spikes": True, "active": False}
    assert len(game.brain.recording_kept) > 0                               # the spikes of the take, for download
    # a new take discards the kept spikes and starts fresh frames; a restart mid-take never leaks a live recording
    game.action({"type": "record", "on": True, "spikes": False})
    state = ticks(game, 1)
    assert game.brain.recording_kept == [] and game.brain.recording is None and state["recording"]["frames"] == 1
    game.action({"type": "record", "on": True, "spikes": True})
    ticks(game, 1)
    game.action({"type": "record", "on": True, "spikes": True})           # restart while spikes are being recorded
    state = ticks(game, 2)
    assert game.brain.recording is not None and state["recording"] == {"frames": 2, "spikes": True, "active": True}
    game.action({"type": "record", "on": False})
    ticks(game, 1)
    assert game.brain.recording is None and game.brain.recording_kept and len(game.brain.recording_kept) > 0


def test_scenario_gives_the_tool_back_when_it_ends(game):
    game.world.tool = "lure"
    game.action({"type": "scenario", "id": "escape"})
    ticks(game, 2)
    assert game.world.tool == "hand" and game.world.hand is not None
    game.action({"type": "scenario"})                                     # stopped by the player
    ticks(game, 1)
    assert game.world.tool == "lure" and game.world.hand is None and game.scenario.saved_tool is None
    game.world.tool = "none"
    game.action({"type": "scenario", "id": "escape"})
    ticks(game, 2)
    while game.scenario.current is not None:                              # run it to its natural end
        game.scenario._next()
    assert game.world.tool == "lure" and game.world.hand is None
    assert any(e["text"] == "scenario finished: Looming escape" for e in game.events.items)
    assert game.scenario.store == {} and game.scenario.measure == {}


def test_scenario_runner_advances_steps(game):
    assert game.action({"type": "scenario", "id": "unicorn"})["ok"] is False
    assert game.action({"type": "scenario", "id": "escape"}) == {"ok": True}
    state = ticks(game, 1)
    sc = state["scenario"]
    assert sc["id"] == "escape" and sc["step"] == 1 and sc["steps"] == 4 and sc["caption"].startswith("Swoop 1")
    assert 2.9 <= sc["left"] <= 3.0 and game.world.tool == "hand" and game.world.hand is not None
    assert state["msg"].startswith("Swoop 1") and any(e["text"] == "scenario started: Looming escape" for e in game.events.items)
    state = ticks(game, 125)                                                   # 3.1 s: step 1 is 3 s long
    assert state["scenario"]["step"] == 2
    game.action({"type": "scenario"})
    state = ticks(game, 1)
    assert state["scenario"] is None and any(e["text"].startswith("scenario stopped") for e in game.events.items)
    game.action({"type": "scenario", "id": "appetitive"})
    state = ticks(game, 2)
    sc = state["scenario"]
    assert sc["step"] == 1 and {o["odour"] for o in state["world"]["odours"]} == {"vinegar", "banana"}
    assert game.state.hunger == 1.0 and "preference_index" in sc["measure"]


def test_choose_mode_arbitration(game):
    m = dict(forward=0.0, yaw=0.0, backward=0.0, halt=0.0, feed=0.0, groom=0.0, song=0.0, court=0.0)
    assert game.choose_mode(m, 0) == "walk"
    assert game.choose_mode({**m, "feed": 0.5}, 0) == "feed"
    assert game.choose_mode({**m, "feed": 0.5, "backward": 0.5}, 0) == "backward"        # weighted 1.15
    assert game.choose_mode({**m, "groom": 0.3}, 0) == "walk"                            # below its start threshold
    game.mode = "groom"
    assert game.choose_mode({**m, "groom": 0.2}, 0) == "groom"                           # hysteresis: keeps going
    game.mode = "idle"
    assert game.choose_mode({**m, "song": 0.5}, 0) == "court" and game.choose_mode({**m, "court": 0.5}, 0) == "court"
    assert game.choose_mode(m, 1) == "walk" and game.body.pose.jump is None       # a lone GF spike is ignored
    assert game.choose_mode(m, 3) == "escape" and game.body.pose.jump is not None and game.gf_cooldown == 1.0
    assert game.choose_mode(m, 0) == "escape"                                            # still in the air


def test_odour_steering_uses_innate_valence_and_learned_bias(game):
    assert game.odour_steering() == (0.0, "")
    game.smelling = "vinegar"
    game.nose.left, game.nose.right, game.nose.felt = {"vinegar": 0.5}, {"vinegar": 0.3}, {"vinegar": 0.4}
    yaw, why = game.odour_steering()
    assert yaw == pytest.approx(-0.35 * 0.6) and "innate +0.35" in why and game.learned_bias == 0
    pl = game.brain.plasticity
    conn = game.conn
    pl.kc_trace[:] = 5.0
    avoid = np.isin(conn.nt[pl.mbon][pl.pe_mbon], ["glutamate"])
    pl.scale[avoid] = 0.5                                                      # avoidance MBON inputs weakened
    yaw2, why2 = game.odour_steering()
    assert game.learned_bias == pytest.approx(0.6) and yaw2 < yaw and "learned +0.60" in why2
    game.smelling = "geosmin"
    game.nose.left, game.nose.right, game.nose.felt = {"geosmin": 0.3}, {"geosmin": 0.5}, {"geosmin": 0.4}
    pl.kc_trace[:] = 0.0
    yaw3, _ = game.odour_steering()
    assert yaw3 == pytest.approx(-0.35 * 0.6)                                  # aversive, stronger on the right: turn left


# ------------------------------------------------------------------ small hand-built parts
def test_event_log_cap_and_since():
    log = EventLog(cap=5)
    for k in range(8):
        log.add(0.1 * k, "kind", f"event {k}", extra=k)
    assert len(log.items) == 5 and [e["id"] for e in log.items] == [4, 5, 6, 7, 8] and log.seq == 8
    assert log.items[0]["extra"] == 3 and log.items[0]["t"] == pytest.approx(0.3)
    assert [e["id"] for e in log.since(6)] == [7, 8] and log.since(8) == [] and len(log.since(0)) == 5


def test_internal_state_bounds():
    s = InternalState()
    s.step(1000.0, eating=0.0, drinking=0.0, courting=0.0)
    assert s.hunger == 1.0 and s.thirst == 1.0 and s.arousal == 0.0
    s.step(1000.0, eating=1.0, drinking=1.0, courting=1.0)
    assert s.hunger == 0.0 and s.thirst == 0.0 and s.arousal == 1.0
    s = InternalState(hunger=0.5)
    s.step(0.025, 1.0, 0.0, 0.0)
    assert 0.497 < s.hunger < 0.5 and s.to_dict() == {"hunger": round(s.hunger, 3), "thirst": 0.3, "arousal": 0.0}


def test_motor_decoder(conn):
    dec = MotorDecoder(conn)
    zero = {k: 0.0 for k in list(HIDDEN_READOUTS) + [r[0] for r in READOUTS]}
    assert all(v == 0 for v in dec.decode(zero, 0.025).values())
    for _ in range(40):
        m = dec.decode({**zero, "MDN": 60, "DNa02R": 50, "DNp09": 50, "MN9": 25, "pIP10": 40}, 0.025)
    assert m["backward"] == pytest.approx(1.0, abs=0.01) and m["yaw"] == pytest.approx(1.0, abs=0.01)
    assert m["forward"] == pytest.approx(0.3, abs=0.01) and m["feed"] == pytest.approx(0.5, abs=0.01)
    assert m["song"] == pytest.approx(1.0, abs=0.03) and m["court"] == 0
    m = dec.decode({**zero, "DNa02L": 100, "DNa02R": 0}, 10.0)
    assert m["yaw"] == -1.0
    assert set(dec.dn_targets) >= {"DNp01", "DNa02", "MDN", "MN9", "pIP10"}
    assert dec.dn_targets["DNp01"]["direct_motor_synapses"] == conn.synapses_between("DNp01", "TTMn")


def test_checks_and_whats_real(game):
    ids = {c[0] for c in CHECKS}
    assert ids == {"feed", "bitter", "lure", "escape", "groom", "wall", "smell", "learn", "court", "wind", "sound",
                   "optomotor", "moonwalk", "silence", "genetics", "genome", "parts"}
    real = game.whats_real()
    assert len(real["wiring"]) >= 5 and len(real["hand_built"]) >= 5 and len(real["not_modelled"]) >= 1


def test_genetics_in_the_layout_and_the_check(game, conn):
    lay = json.loads(game.layout_json)
    g = lay["genetics"]
    assert {e["key"] for e in g["expression"]} == {"fru", "dsx", "both", "male", "dimorphic"}
    assert g["readouts"]["pIP10"]["tags"] == ["fru", "♂"] and g["readouts"]["MN9"]["tags"] == []
    by_key = {r["key"]: r for r in lay["readouts"]}
    assert by_key["pIP10"]["genes"] == ["fru", "♂"] and by_key["pC1"]["genes"] == ["fru", "dsx", "♂"] and by_key["MN9"]["genes"] == []
    assert game.action({"type": "silence", "spec": "gene:fru"}) == {"ok": True, "n": 30}
    state = ticks(game, 1)
    assert "genetics" in state["done"] and "gene:fru" in state["silenced"]
    game.action({"type": "unsilence", "spec": "gene:fru"})
    assert "gene:fru" not in ticks(game, 1)["silenced"]


def _wait(game, cond, secs=60):
    t0 = time.time()
    while time.time() - t0 < secs:
        game.tick()
        if cond():
            return True
        time.sleep(0.01)
    return False


def test_grow_a_fly_swaps_the_wiring_and_tests_its_reflexes(game, conn):
    assert game.action({"type": "grow", "level": "hemilineage"})["ok"] is False
    assert game.action({"type": "grow", "level": "type", "seed": "x"})["ok"] is False
    game.action({"type": "watch", "spec": "GNG232", "key": "relay"})
    game.action({"type": "silence", "spec": "MN9"})
    ticks(game, 1)
    real_brain = game.brain
    assert game.action({"type": "grow", "level": "type", "seed": 3}) == {"ok": True}
    assert game.action({"type": "grow", "level": "type", "seed": 4})["ok"] is False        # one at a time
    assert _wait(game, lambda: game.genome["growing"] is None and game.brain is not real_brain)
    assert game.conn is not game.real_conn and game.brain.conn is game.conn and game.genome["level"] == "type"
    assert game.genome["seed"] == 3 and game.genome["wiring"]["edges_grown"] > 0 and game.genome["rules"]["level"] == "type"
    assert set(game.brain.monitors) >= {"MN9", "GF", "relay"} and "MN9" in game.brain.silenced   # watches and lesions carried over
    state = ticks(game, 1)
    assert state["genome"]["level"] == "type" and "genome" in state["done"]
    assert any(e["text"].startswith("a fly grown from its type wiring rules") for e in game.events.items)
    assert _wait(game, lambda: game.genome["survival"] and not game.genome["survival"]["running"], secs=120)
    sv = game.genome["survival"]
    assert sv["results"] and all(r["ok"] in (True, False, None) for r in sv["results"]) and sv["tested"] <= len(sv["results"])
    assert sv["ok"] <= sv["tested"]
    # back to the real wiring
    assert game.action({"type": "grow", "level": "real"})["ok"]
    assert _wait(game, lambda: game.genome["growing"] is None and game.conn is game.real_conn)
    assert game.genome["level"] == "real" and game.brain.conn is game.real_conn
    lay = json.loads(game.layout_json)
    assert [l["level"] for l in lay["genome"]["levels"]][:2] == ["real", "type"]


def test_columnar_vision_shares_the_rendered_eyes_and_fires_t4_t5(conn):
    g = Game(build_brain(conn, "game"), autopilot=False, seed=1)                 # columnar vision is on by default
    assert g.columnar_on and json.loads(g.layout_json)["columnar_vision"] is True
    assert g.retina.columnar.eyes["L"] is g.retina.eyes["L"] and g.retina.columnar.conn is conn
    g.action({"type": "drop", "kind": "post", "x": 6.0, "y": -2.0})
    g.action({"type": "zap", "spec": "DNa02/L", "hz": 80, "secs": 3})
    fired = set()
    for _ in range(40):
        g.tick()
        fired |= set(g.retina.columnar.last_idx.tolist())
    assert fired and fired <= set(conn.select("prefix:T4,prefix:T5").tolist())
    assert not any(k.startswith("T4") for k in g.brain.stim)                    # no hand-built T4 stand-in
    off = Game(build_brain(conn, "game"), autopilot=False, seed=1, columnar=False)
    assert not off.columnar_on and off.retina.columnar is None


def test_sound_action_reaches_the_giant_fibre(game):
    game.action({"type": "sound", "secs": 0.3})
    state = ticks(game, 4)
    assert state["senses"].get("sound") and state["hz"]["JOA"] > 10 and state["hz"]["GF"] > 0
    assert any(e["text"] == "hears a loud sound" for e in game.events.items)
    assert state["mode"] == "escape" and "sound" in state["done"] and "loom" not in state["senses"]
    ticks(game, 12)
    assert game.sound_left <= 0 and "sound" not in game.senses_now


def test_stripes_action_spins_the_drum(game):
    game.action({"type": "stripes", "count": 12, "drum_speed": 2.0})
    state = ticks(game, 1)
    assert state["world"]["stripes"]["count"] == 12 and state["world"]["stripes"]["drum_speed"] == 2.0
    assert any(e["text"].startswith("wall stripes: 12") for e in game.events.items)
    phase = state["world"]["stripes"]["phase"]
    fired = set()
    for _ in range(10):
        game.tick()
        fired |= set(game.retina.columnar.last_idx.tolist())
    assert game.state_dict["world"]["stripes"]["phase"] != phase
    assert fired and fired <= set(game.conn.select("prefix:T4,prefix:T5").tolist())    # the drum drives T4/T5
    game.action({"type": "stripes", "count": 0, "drum_speed": 0.0})
    assert ticks(game, 1)["world"]["stripes"]["count"] == 0


def test_parts_list_toggle_rebuilds_the_brain_and_retests_the_reflexes(game):
    assert game.parts_on is False and game.brain.parts is None
    assert game.action({"type": "parts", "on": "yes"})["ok"] is False
    assert game.action({"type": "parts", "on": False})["ok"] is False                 # already off
    lay = json.loads(game.layout_json)
    assert lay["parts"]["counts"]["modulatory_neurons"] == 16 and lay["parts"]["counts"]["graded_neurons"] > 100
    assert [m["nt"] for m in lay["parts"]["tables"]["modulators"]] == ["dopamine", "octopamine", "serotonin"]
    old_brain = game.brain
    game.action({"type": "silence", "spec": "MN9"})
    assert game.action({"type": "parts", "on": True})["ok"]
    assert game.action({"type": "parts", "on": False})["ok"] is False                # one rebuild at a time
    assert game.genome["growing"]["reason"] == "parts"
    assert _wait(game, lambda: game.genome["growing"] is None and game.brain is not old_brain)
    assert game.parts_on and game.brain.parts is not None and game.conn is game.real_conn and "parts" in game.done
    assert "MN9" in game.brain.silenced and "class:DAN" not in game.brain.silenced and "class:ALLN" in game.brain.silenced
    st = game.genome_status()["parts"]
    assert st["on"] and set(st["status"]["tone"]) == {"dopamine", "octopamine", "serotonin"} and st["status"]["graded"] > 100
    assert _wait(game, lambda: game.genome["survival"] and not game.genome["survival"]["running"], secs=120)
    assert game.genome["survival"]["tested"] > 0
    state = json.loads(game.state_json)
    assert state["genome"]["parts"]["on"] and "tone" in state["genome"]["parts"]["status"]
    # a grown fly keeps the parts list, and switching it off keeps the grown wiring
    game.action({"type": "grow", "level": "type", "seed": 2})
    assert _wait(game, lambda: game.genome["growing"] is None and game.conn is not game.real_conn)
    assert game.brain.parts is not None and game.parts_on and game.genome["level"] == "type"
    grown = game.conn
    assert game.action({"type": "parts", "on": False})["ok"]
    assert _wait(game, lambda: game.genome["growing"] is None and game.brain.parts is None)
    assert not game.parts_on and game.conn is grown and game.genome["level"] == "type" and "class:DAN" in game.brain.silenced
    assert json.loads(game.state_json)["genome"]["parts"] == {"on": False, "status": None}
