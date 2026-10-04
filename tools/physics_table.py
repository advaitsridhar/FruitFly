#!/usr/bin/env python3
"""docs/SCIENCE.md 6.7, measured again: the physics-body table reproduced on the kit as it is, so that every speed lever of
the two-flies work can be checked against it (docs/TWO_FLIES_PLAN.md 8.2 and 8.6, decision 25). A hand-run tool.

    nice -n 10 .venv/bin/python tools/physics_table.py --json ../runs/p4-table.json --jobs 6
    nice -n 10 .venv/bin/python tools/physics_table.py --bodies drawn --seeds 0 1 --only lure_left mdn
    .venv/bin/python tools/physics_table.py --compare ../runs/p4-table.json      # the report again, from a saved run

The conditions are the table's: the real male connectome, the game profile, the parts list on, seeds 0-4, one fly in an
empty dish, 3 s unless stated. Every run is a fresh game (a fresh brain, built from the seed) in a child process of its own,
so that no run sees another's random draws and the peak memory is that run's. The rows:

  speed       the body alone, no brain: forward drive 0.3, 0.6 and 1.0 for 3 s. The net displacement of the thorax over the
              3 s divided by 3 s (not the path length, which with the physics body adds up the sway within each stride)
  turn        the body alone: the same drives with full steering (yaw 1): the heading change over 3 s divided by 3 s
  lure_left   the lure 20 mm from the fly at 70 deg to the left of its heading, placed before the first tick, the walking
  lure_right  urge on, 3 s: the heading change; whether and when the fly faces the lure (within 15 deg)
  lure_still  the lure 15 mm away at 70 deg left, the walking urge off, 3 s: the heading change (turning in place), and where
              the lure still is at the end
  The lure wiggles (--wiggle, --wiggle-hz): it swings sideways across its bearing by that amplitude at that rate, as the
  page's own lure check expects ("wiggle the decoy beside it": the small-object cells LC10a answer a blob that moves on the
  retina, and a standing fly sees a still lure stand still). With --wiggle 0 the lure is still.
  mdn         MDN zapped at 60 Hz for the whole 3 s: the net displacement and its backward part, the heading drift, and HS
              and the brain's events/s over the ticks spent backing
  quiet       10 s, nothing in the dish, the walking urge on: mean HS, mean events/s, a high state (more than 60,000 events/s
              held for 1 s or longer: when it starts), backing with no wall touch in the last 1.5 s

Every game row also counts the giant-fibre bursts, the escape events (the ticks on which the mode turns to escape) and the
ticks spent escaping: the table was measured under v2.8.0's jump rule (two spikes in one tick), and v2.8.1's burst rule
(GF_BURST live spikes over two ticks) fires far less often, so a row that differs from the published one with a different
escape count differs for that reason first (the drawn-body entries the table marks "v2.8.0 jump rule" are not expected to
reproduce). Distances are world millimetres: the physics body is real size, but the lure distances are the same world
millimetres for both bodies, since the senses use the drawn geometry for both (docs/SCIENCE.md 6.7).

The report puts every measured number beside the published one. Decision 25's verdicts: the speeds and turn rates within
10 % of the published value, the "faces the lure" count within 1 of it; everything else is shown for the eye, with the
escape counts.
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import queue
import resource
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

TICK_S = 0.025
DRIVES = (0.3, 0.6, 1.0)
LURE_FAR_MM, LURE_NEAR_MM, LURE_DEG, FACING_DEG = 20.0, 15.0, 70.0, 15.0
MDN_HZ = 60.0
HIGH_EPS, HIGH_HOLD_S = 60_000.0, 1.0           # a high state: events/s above this, held this long
NO_TOUCH_S = 1.5                                 # backing with no wall touch in the last 1.5 s
SCENARIOS = ("lure_left", "lure_right", "lure_still", "mdn", "quiet")
WIGGLE_MM, WIGGLE_HZ = 1.5, 3.0                 # the lure's sideways swing (amplitude, rate): see --wiggle
VARIANTS = {"drawn": ("drawn", False), "physics": ("physics", False), "physics_sa": ("physics", True)}
TOL_SPEED, TOL_FACES = 0.10, 1                   # decision 25

# docs/SCIENCE.md 6.7 as published with v2.8.0 (2026-09-26), per seed 0-4 where the table gives seeds. "jump rule": an
# entry the table marks "(v2.8.0 jump rule)", not expected to reproduce under v2.8.1's burst rule.
PUBLISHED = {
    "drawn": {
        "speed": {"0.3": 4.2, "0.6": 8.4, "1": 13.9}, "speed_note": "chosen",
        "turn": {"0.3": 264, "0.6": 228, "1": 180},
        "lure_left": [97, 79, 253, 94, 257], "lure_left_note": "seeds 2 and 4 jump and circle (jump rule)",
        "lure_right": [-94, -102, -89, -100, 82], "lure_right_note": "seed 4 jumps (jump rule)",
        "faces": (10, "0.95-1.33 s"),
        "lure_still": "turns 62-73 deg in place (66), faces it at 0.48-0.8 s",
        "mdn": [18.4, 19.7, 18.6, 17.9, 19.7], "mdn_note": "",
        "mdn_hs": "0-0.6 Hz / 26-30k events/s",
        "quiet_hs": [5.5, 10.1, 7.9, 72.6, 108.2], "quiet_eps": [25.9e3, 30.4e3, 24.9e3, 78.0e3, 72.3e3],
        "quiet_high": "seeds 3 and 4, from 3.9 and 6.2 s, with 7 and 21 escape ticks (jump rule)",
        "quiet_back": "none",
        "rtf": "0.90-1.23", "memory": "876 MB",
    },
    "physics": {
        "speed": {"0.3": 4.1, "0.6": 8.9, "1": 14.9}, "speed_note": "measured",
        "turn": {"0.3": 28, "0.6": 86, "1": 174},
        "lure_left": [102, 103, 105, 94, 94], "lure_left_note": "mean 99",
        "lure_right": [-87, -106, -100, -90, -125], "lure_right_note": "mean -102",
        "faces": (8, "1.65-2.55 s"),
        "lure_still": "turns 0.4 deg (-0.5 to 2.2); lure still 69-71 deg off",
        "mdn": [20.6, 19.5, 21.6, 21.0, 19.5], "mdn_note": "mean 20.4; heading drifts 3-10 deg",
        "mdn_hs": "57-84 Hz / 68-103k events/s",
        "quiet_hs": [24.8, 13.7, 14.4, 18.5, 19.0], "quiet_eps": [56.2e3, 46.7e3, 43.1e3, 40.5e3, 58.8e3],
        "quiet_high": "seeds 0 and 2, from 8.9 s, around the first wall touch (8.9-9.5 s)",
        "quiet_back": "none",
        "rtf": "0.086-0.113 (MuJoCo alone 0.095-0.130)", "memory": "1.32-1.35 GB",
    },
    "physics_sa": {
        "speed": None, "turn": None,
        "lure_left": [99, 90, 113, 101, 126], "lure_left_note": "mean 106",
        "lure_right": [-62, -79, -118, -85, -113], "lure_right_note": "mean -92",
        "faces": (6, "1.83-2.83 s"),
        "lure_still": "turns 0.3 deg (-0.4 to 1.2)",
        "mdn": [19.9, 19.2, 20.1, 19.2, 21.4], "mdn_note": "mean 20.0; heading drifts 2-9 deg",
        "mdn_hs": "0-0.4 Hz / 27-31k events/s",
        "quiet_hs": [57.7, 11.6, 8.8, 6.9, 9.0], "quiet_eps": [56.8e3, 38.7e3, 27.1e3, 45.5e3, 48.4e3],
        "quiet_high": "seeds 0 and 3, from 8.1 and 8.6 s",
        "quiet_back": "seed 4, 21 ticks from 3.8 s",
        "rtf": "0.089-0.116 (MuJoCo alone 0.098-0.132)", "memory": "1.32-1.35 GB",
    },
}


def drive(**kw) -> dict:
    d = dict(forward=0.0, yaw=0.0, backward=0.0, halt=0.0, feed=0.0, groom=0.0, song=0.0, court=0.0)
    d.update(kw)
    return d


def peak_mb() -> float:
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 0)      # Linux: kB


def mean(xs, nd: int = 2):
    return round(statistics.fmean(xs), nd) if xs else None


def variant_of(row: dict) -> str:
    return row["body"] + ("_sa" if row.get("stride_average") else "")


# ---------------------------------------------------------------------------------------------- the jobs (child processes)
def body_job(kind: str, seed: int, seconds: float, levers=None) -> dict:
    """The body alone, no brain: the speed at the three forward drives and the turn rate at full steering."""
    import random
    from virtual_fly.physics import make_body
    from virtual_fly.world import World, wrap
    from virtual_fly.physics import DEFAULT_LEVERS
    levers = DEFAULT_LEVERS if (levers is None and kind == "physics") else tuple(levers or ())   # None: the game's default
    out = {"job": "body", "body": kind, "stride_average": False, "seed": seed, "seconds": seconds, "levers": list(levers),
           "speed": {}, "path_speed": {}, "turn": {}, "turn_net_mm": {}, "rtf": {}}
    n = int(round(seconds / TICK_S))
    for f in DRIVES:
        for steer in (0.0, 1.0):
            body = make_body(kind, World(seed=seed), random.Random(seed), seed=seed, levers=levers)
            body.reset()
            x0, y0, prev = body.pose.x, body.pose.y, body.pose.h
            h_acc = 0.0
            t0 = time.perf_counter()
            for _ in range(n):
                body.move(TICK_S, "walk", drive(forward=f, yaw=steer))
                h_acc += wrap(body.pose.h - prev)
                prev = body.pose.h
            wall = time.perf_counter() - t0
            net = math.hypot(body.pose.x - x0, body.pose.y - y0)
            key = f"{f:g}"
            if steer == 0.0:
                out["speed"][key] = round(net / seconds, 2)
                try:
                    out["path_speed"][key] = round(float(body.to_dict()["dist"]) / seconds, 2)
                except Exception:
                    out["path_speed"][key] = None
            else:
                out["turn"][key] = round(abs(math.degrees(h_acc)) / seconds, 1)
                out["turn_net_mm"][key] = round(net, 1)
            out["rtf"][f"{key}:{steer:g}"] = round(seconds / wall, 3)
    out["peak_mb"] = peak_mb()
    return out


def game_job(kind: str, stride_average: bool, seed: int, scenario: str, seconds: float,
             wiggle: tuple = (WIGGLE_MM, WIGGLE_HZ), levers=None) -> dict:
    """One scenario on a fresh game: the fly and its brain from the seed, the parts list on, the dish empty."""
    from virtual_fly import load_connectome
    from virtual_fly.game import GF_BURST, Game
    from virtual_fly.parts import PartsList
    from virtual_fly.settings import build_brain
    from virtual_fly.world import wrap
    t_build = time.perf_counter()
    conn = load_connectome(quiet=True)
    parts_list = PartsList()
    kw = {"seed": seed, "dt": 0.5}
    brain = build_brain(conn, "game", **kw, parts=parts_list)
    game = Game(brain, seed=seed, body=kind, stride_average=stride_average, brain_kwargs=dict(kw), parts_list=parts_list,
                brain_procs="auto", autopilot=scenario != "lure_still", physics_levers=levers)
    del brain
    build_s = time.perf_counter() - t_build
    try:
        f = game.flies[0]
        p = f.body.pose
        x0, y0, h0 = p.x, p.y, p.h
        lure = None
        if scenario.startswith("lure"):
            dist = LURE_NEAR_MM if scenario == "lure_still" else LURE_FAR_MM
            side = -1.0 if scenario == "lure_right" else 1.0
            ang = h0 + side * math.radians(LURE_DEG)
            lure = (x0 + dist * math.cos(ang), y0 + dist * math.sin(ang))
            across = (-math.sin(ang), math.cos(ang))             # the wiggle swings across the lure's bearing
            for a in ({"type": "tool", "tool": "lure"}, {"type": "hand", "x": lure[0], "y": lure[1]}):
                r = game.action(a)
                if not r.get("ok"):
                    raise RuntimeError(f"{a}: {r}")
        elif scenario == "mdn":
            r = game.action({"type": "zap", "spec": "MDN", "hz": MDN_HZ, "secs": seconds})
            if not r.get("ok"):
                raise RuntimeError(f"zap: {r}")
        n = int(round(seconds / TICK_S))
        h_acc, prev = 0.0, h0
        faced_at = None
        hs, eps, hs_back, eps_back, ncon, sps, graded = [], [], [], [], [], [], []
        modes: dict = {}
        bursts = escapes = escape_ticks = back_ticks = back_no_touch = 0
        prev_gf, escaping = 0, False
        first_back_no_touch = high_from = high_t = None
        high_run = 0.0
        wall_s0 = float(getattr(f.body, "wall_s", 0.0))
        t0 = time.perf_counter()
        for i in range(n):
            if lure is not None and wiggle[0] > 0:
                sw = wiggle[0] * math.sin(2 * math.pi * wiggle[1] * i * TICK_S)
                game.action({"type": "hand", "x": lure[0] + sw * across[0], "y": lure[1] + sw * across[1]})
            game.tick()
            p = f.body.pose
            h_acc += wrap(p.h - prev)
            prev = p.h
            t = (i + 1) * TICK_S
            if lure is not None and faced_at is None:
                if abs(wrap(math.atan2(lure[1] - p.y, lure[0] - p.x) - p.h)) < math.radians(FACING_DEG):
                    faced_at = round(t, 3)
            bt = f.bt
            hz_hs = float(bt.hz.get("HS", 0.0))
            e = float(f.sps + f.graded_eps)
            hs.append(hz_hs)
            eps.append(e)
            sps.append(float(f.sps))
            graded.append(float(f.graded_eps))
            modes[f.mode] = modes.get(f.mode, 0) + 1
            if f.mode == "backward":
                hs_back.append(hz_hs)
                eps_back.append(e)
                back_ticks += 1
                if game.t - f.bristles.last_touch_t > NO_TOUCH_S:
                    back_no_touch += 1
                    if first_back_no_touch is None:
                        first_back_no_touch = round(t, 3)
            if bt.gf + prev_gf >= GF_BURST:
                bursts += 1
            prev_gf = bt.gf
            esc = f.mode == "escape"
            escapes += esc and not escaping
            escaping = esc
            escape_ticks += esc
            if e > HIGH_EPS:
                if high_from is None:
                    high_from = t - TICK_S
                high_run += TICK_S
                if high_t is None and high_run >= HIGH_HOLD_S - 1e-9:
                    high_t = round(high_from, 3)
            else:
                high_run, high_from = 0.0, None
            if kind == "physics":
                ncon.append(int(f.body.walker._d.ncon))
        wall = time.perf_counter() - t0
        p = f.body.pose
        dx, dy = p.x - x0, p.y - y0
        row = {"job": "game", "body": kind, "stride_average": stride_average, "seed": seed, "scenario": scenario,
               "seconds": seconds, "autopilot": scenario != "lure_still", "levers": list(game.physics_levers),
               "heading_change_deg": round(math.degrees(h_acc), 1), "net_mm": round(math.hypot(dx, dy), 2),
               "backward_mm": round(-(dx * math.cos(h0) + dy * math.sin(h0)), 2),
               "faced_at_s": faced_at,
               "lure_bearing_end_deg": None if lure is None else round(math.degrees(wrap(math.atan2(lure[1] - p.y, lure[0] - p.x) - p.h)), 1),
               "wiggle_mm": wiggle[0], "wiggle_hz": wiggle[1],
               "hs_hz": mean(hs), "events_per_s": mean(eps, 0), "spikes_per_s": mean(sps, 0), "graded_per_s": mean(graded, 0),
               "hs_hz_backing": mean(hs_back),
               "events_per_s_backing": mean(eps_back, 0), "back_ticks": back_ticks, "back_ticks_no_touch": back_no_touch,
               "back_no_touch_from_s": first_back_no_touch, "high_state_from_s": high_t,
               "bursts": bursts, "escape_events": escapes, "escape_ticks": escape_ticks, "modes": modes,
               "rtf": round(seconds / wall, 3), "build_s": round(build_s, 1), "peak_mb": peak_mb(),
               "wall_s": round(wall, 1), "load1": round(os.getloadavg()[0], 2)}
        if kind == "physics":
            mj = float(f.body.wall_s) - wall_s0
            row["mujoco_rtf"] = round(seconds / mj, 3) if mj > 0 else None
            row["ncon_mean"] = mean(ncon, 1)
        return row
    finally:
        game.close()


def _child(q, idx: int, fn, kwargs: dict):
    try:
        q.put((idx, fn(**kwargs)))
    except BaseException as e:                       # report it, so that the parent never waits for a row that cannot come
        import traceback
        q.put((idx, {"error": f"{fn.__name__}({kwargs}): {e!r}", "trace": traceback.format_exc()}))
        raise


def run_jobs(jobs: list[tuple], n_jobs: int, describe) -> list[dict]:
    """Each job (function, kwargs) in a fresh child process, at most ``n_jobs`` at a time; the rows in job order."""
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    results: list = [None] * len(jobs)
    pending = list(enumerate(jobs))
    running: dict = {}
    started: dict = {}
    t_start = time.monotonic()
    done = 0
    while pending or running:
        while pending and len(running) < n_jobs:
            idx, (fn, kw) = pending.pop(0)
            p = ctx.Process(target=_child, args=(q, idx, fn, kw))
            p.start()
            running[idx], started[idx] = p, time.monotonic()
        try:
            idx, row = q.get(timeout=10)
        except queue.Empty:
            for idx, p in list(running.items()):
                if p.exitcode not in (None, 0) and results[idx] is None:      # died without a row (killed, out of memory)
                    results[idx] = {"error": f"the child for {describe(jobs[idx])} died with exit code {p.exitcode}"}
                    p.join()
                    del running[idx]
                elif time.monotonic() - started[idx] > 3600:
                    p.terminate()
                    p.join(10)
                    results[idx] = {"error": f"{describe(jobs[idx])}: no row after an hour"}
                    del running[idx]
            continue
        results[idx] = row
        p = running.pop(idx)
        p.join(timeout=120)
        done += 1
        print(f"[{done}/{len(jobs)} {time.monotonic() - t_start:6.0f} s] {describe(jobs[idx])}: {brief(row)}", flush=True)
    return results


def brief(row: dict) -> str:
    if "error" in row:
        return "ERROR " + row["error"]
    if row["job"] == "body":
        return (f"speed {row['speed']} mm/s, turn {row['turn']} deg/s, rtf {min(row['rtf'].values())}-{max(row['rtf'].values())}, "
                f"peak {row['peak_mb']:.0f} MB")
    bits = [f"heading {row['heading_change_deg']:+.1f} deg", f"net {row['net_mm']} mm"]
    if row["faced_at_s"] is not None:
        bits.append(f"faces at {row['faced_at_s']} s")
    if row["scenario"] == "mdn":
        bits.append(f"back {row['backward_mm']} mm, HS {row['hs_hz_backing']} Hz, {row['events_per_s_backing']} ev/s while backing")
    if row["scenario"] == "quiet":
        bits.append(f"HS {row['hs_hz']} Hz, {row['events_per_s']} ev/s, high from {row['high_state_from_s']}")
    bits.append(f"escapes {row['escape_events']} ({row['escape_ticks']} ticks), bursts {row['bursts']}")
    bits.append(f"rtf {row['rtf']}" + (f" (MuJoCo {row['mujoco_rtf']})" if row.get("mujoco_rtf") else "") + f", peak {row['peak_mb']:.0f} MB")
    return ", ".join(bits)


# ---------------------------------------------------------------------------------------------- the report
def fmt_list(xs, nd=1):
    return ", ".join("–" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x)) for x in xs)


def rng_str(xs, nd=2):
    xs = [x for x in xs if x is not None]
    return "–" if not xs else (f"{min(xs):.{nd}f}-{max(xs):.{nd}f}" if min(xs) != max(xs) else f"{min(xs):.{nd}f}")


def within(measured, published, tol) -> str:
    if measured is None or published in (None, 0):
        return "–"
    return "yes" if abs(measured - published) <= tol * abs(published) else f"NO ({(measured - published) / published:+.0%})"


def summarise(rows: list[dict]) -> dict:
    """A saved run's rows in PUBLISHED's shape, so that a lever run can be read against the tool's own baseline run on the
    unchanged body (the gate the levers are judged by, decision 25)."""
    by: dict = {}
    for r in rows:
        if "error" not in r:
            by.setdefault(variant_of(r), {}).setdefault(r.get("scenario", "body"), []).append(r)
    out = {}
    for variant, groups in by.items():
        t: dict = {"speed": None, "turn": None, "speed_note": "baseline"}
        body = sorted(groups.get("body", []), key=lambda r: r["seed"])
        if body:
            t["speed"] = {d: round(statistics.fmean(r["speed"][d] for r in body), 2) for d in ("0.3", "0.6", "1")}
            t["turn"] = {d: round(statistics.fmean(r["turn"][d] for r in body), 1) for d in ("0.3", "0.6", "1")}
        faces_n = faces_total = 0
        faces_t = []
        for scen in ("lure_left", "lure_right"):
            g = sorted(groups.get(scen, []), key=lambda r: r["seed"])
            t[scen] = [round(r["heading_change_deg"]) for r in g]
            t[scen + "_note"] = f"escapes {fmt_list([r['escape_events'] for r in g])}" if g else ""
            faces_n += sum(1 for r in g if r["faced_at_s"] is not None)
            faces_total += len(g)
            faces_t += [r["faced_at_s"] for r in g if r["faced_at_s"] is not None]
        t["faces"] = (faces_n * 10 // max(faces_total, 1) if faces_total else 0, f"{rng_str(faces_t)} s")
        g = sorted(groups.get("lure_still", []), key=lambda r: r["seed"])
        t["lure_still"] = (f"turns {rng_str([r['heading_change_deg'] for r in g], 1)} deg; faces at {fmt_list([r['faced_at_s'] for r in g], 2)} s; "
                           f"lure at the end {fmt_list([r['lure_bearing_end_deg'] for r in g], 0)} deg") if g else "–"
        g = sorted(groups.get("mdn", []), key=lambda r: r["seed"])
        t["mdn"] = [r["net_mm"] for r in g]
        t["mdn_note"] = f"heading drift {rng_str([abs(r['heading_change_deg']) for r in g], 0)} deg" if g else ""
        t["mdn_hs"] = (f"{rng_str([r['hs_hz_backing'] for r in g], 1)} Hz / "
                       f"{rng_str([r['events_per_s_backing'] / 1000 for r in g if r['events_per_s_backing'] is not None], 0)}k events/s") if g else "–"
        g = sorted(groups.get("quiet", []), key=lambda r: r["seed"])
        t["quiet_hs"] = [r["hs_hz"] for r in g]
        t["quiet_eps"] = [r["events_per_s"] for r in g]
        high = [f"seed {r['seed']} from {r['high_state_from_s']} s" for r in g if r["high_state_from_s"] is not None]
        t["quiet_high"] = ", ".join(high) if high else ("none" if g else "–")
        back = [f"seed {r['seed']}: {r['back_ticks_no_touch']} ticks" for r in g if r["back_ticks_no_touch"]]
        t["quiet_back"] = ", ".join(back) if back else ("none" if g else "–")
        game_rows = [r for s_, rs in groups.items() if s_ != "body" for r in rs]
        t["rtf"] = (rng_str([r["rtf"] for r in game_rows], 3) + (f" (MuJoCo alone {rng_str([r['mujoco_rtf'] for r in game_rows if r.get('mujoco_rtf')], 3)})"
                                                                 if any(r.get("mujoco_rtf") for r in game_rows) else "")) if game_rows else "–"
        t["memory"] = f"{rng_str([r['peak_mb'] for r in game_rows], 0)} MB" if game_rows else "–"
        out[variant] = t
    return out


def report(rows: list[dict], target: dict | None = None, label: str = "published") -> list[str]:
    """The measured numbers beside the target's: the published table (the default) or a baseline run's summary."""
    target = PUBLISHED if target is None else target
    out, verdicts = [], []
    by: dict = {}
    for r in rows:
        if "error" in r:
            out.append("ERROR: " + r["error"])
            continue
        by.setdefault(variant_of(r), {}).setdefault(r.get("scenario", "body"), []).append(r)
    for variant, groups in by.items():
        pub = target.get(variant)
        levers = next((r.get("levers") for rs in groups.values() for r in rs if r.get("levers")), None)
        out.append("")
        out.append(f"== {variant}{' with levers ' + ', '.join(levers) if levers else ''} (against the {label} numbers) ==")
        if pub is None:
            out.append(f"    (the {label} numbers have no {variant} column)")
            continue
        body = sorted(groups.get("body", []), key=lambda r: r["seed"])
        if body:
            for key, title in (("speed", "speed at forward drive 0.3 / 0.6 / 1.0 (mm/s, net displacement over 3 s)"),
                               ("turn", "turn rate at full steering, same drives (deg/s)")):
                means = {d: mean([r[key][d] for r in body], 2) for d in ("0.3", "0.6", "1")}
                out.append(f"{title}:")
                out.append("    measured (mean of seeds " + ", ".join(str(r["seed"]) for r in body) + "): "
                           + " / ".join(f"{means[d]:.1f}" for d in means) + "   per seed: "
                           + "; ".join(f"{d}: {fmt_list([r[key][d] for r in body])}" for d in means))
                if pub.get(key):
                    ver = [within(means[d], pub[key][d], TOL_SPEED) for d in means]
                    out.append(f"    {label}: " + " / ".join(str(pub[key][d]) for d in means)
                               + (f" ({pub.get('speed_note', '')})" if key == "speed" else "") + f"   within {TOL_SPEED:.0%}: " + ", ".join(ver))
                    verdicts.append((f"{variant} {key}", all(v == "yes" for v in ver)))
            if any(r["path_speed"].get("1") for r in body):
                out.append("    (path length over 3 s, for the eye: " + " / ".join(
                    rng_str([r["path_speed"][d] for r in body], 1) for d in ("0.3", "0.6", "1")) + " mm/s)")
            out.append("    body-only real-time factor: " + rng_str([v for r in body for v in r["rtf"].values()], 3)
                       + f"; peak memory {rng_str([r['peak_mb'] for r in body], 0)} MB")
        faces_n, faces_t, faces_total = 0, [], 0
        for scen in ("lure_left", "lure_right"):
            g = sorted(groups.get(scen, []), key=lambda r: r["seed"])
            if not g:
                continue
            out.append(f"{scen.replace('_', ' ')}, 20 mm at 70 deg (wiggle {g[0].get('wiggle_mm', 0):g} mm at {g[0].get('wiggle_hz', 0):g} Hz), "
                       f"walking urge on: heading change (deg) per seed {[r['seed'] for r in g]}:")
            out.append(f"    measured:  {fmt_list([r['heading_change_deg'] for r in g], 0)}   (mean {mean([r['heading_change_deg'] for r in g], 0)}); "
                       f"escapes {fmt_list([r['escape_events'] for r in g])} (ticks {fmt_list([r['escape_ticks'] for r in g])}); "
                       f"faces at {fmt_list([r['faced_at_s'] for r in g], 2)} s")
            out.append(f"    {label}: {fmt_list(pub.get(scen) or [], 0)}   ({pub.get(scen + '_note', '')})")
            faces_n += sum(1 for r in g if r["faced_at_s"] is not None)
            faces_t += [r["faced_at_s"] for r in g if r["faced_at_s"] is not None]
            faces_total += len(g)
        if faces_total:
            pn, pt = pub["faces"]
            scaled = pn * faces_total / 10.0
            ok = abs(faces_n - scaled) <= TOL_FACES
            out.append(f"faces the lure (within 15 deg): measured {faces_n} of {faces_total} runs, at {rng_str(faces_t)} s; "
                       f"{label} {pn} of 10, at {pt}; within {TOL_FACES}: {'yes' if ok else 'NO'}"
                       + ("" if faces_total == 10 else f" ({label} scaled to {scaled:.1f} of {faces_total})"))
            verdicts.append((f"{variant} faces the lure", ok))
        g = sorted(groups.get("lure_still", []), key=lambda r: r["seed"])
        if g:
            out.append("lure 15 mm at 70 deg left, walking urge off:")
            out.append(f"    measured:  turns {fmt_list([r['heading_change_deg'] for r in g], 1)} deg (mean {mean([r['heading_change_deg'] for r in g], 1)}), "
                       f"net {fmt_list([r['net_mm'] for r in g], 1)} mm; faces at {fmt_list([r['faced_at_s'] for r in g], 2)} s; "
                       f"lure at the end {fmt_list([r['lure_bearing_end_deg'] for r in g], 0)} deg; escapes {fmt_list([r['escape_events'] for r in g])}")
            out.append(f"    {label}: {pub.get('lure_still', '–')}")
        g = sorted(groups.get("mdn", []), key=lambda r: r["seed"])
        if g:
            out.append("MDN at 60 Hz for 3 s:")
            out.append(f"    measured:  net {fmt_list([r['net_mm'] for r in g], 1)} mm (mean {mean([r['net_mm'] for r in g], 1)}); "
                       f"backward part {fmt_list([r['backward_mm'] for r in g], 1)} mm; heading drift {fmt_list([abs(r['heading_change_deg']) for r in g], 0)} deg; "
                       f"ticks backing {fmt_list([r['back_ticks'] for r in g])}; escapes {fmt_list([r['escape_events'] for r in g])}")
            out.append(f"    {label}: {fmt_list(pub.get('mdn') or [], 1)} mm ({pub.get('mdn_note', '')})")
            out.append(f"    HS / brain while backing: measured {rng_str([r['hs_hz_backing'] for r in g], 1)} Hz / "
                       f"{rng_str([r['events_per_s_backing'] / 1000 for r in g if r['events_per_s_backing'] is not None], 0)}k events/s; "
                       f"{label} {pub.get('mdn_hs', '–')}")
        g = sorted(groups.get("quiet", []), key=lambda r: r["seed"])
        if g:
            out.append(f"{g[0]['seconds']:g} s in a quiet arena, walking urge on:")
            out.append(f"    HS (Hz):       measured {fmt_list([r['hs_hz'] for r in g], 1)};  {label} {fmt_list(pub.get('quiet_hs') or [], 1)}")
            out.append(f"    brain (ev/s):  measured {fmt_list([round(r['events_per_s'] / 1000, 1) for r in g], 1)}k;  "
                       f"{label} {fmt_list([x / 1000 for x in (pub.get('quiet_eps') or [])], 1)}k")
            if all(r.get("spikes_per_s") is not None for r in g):
                out.append(f"      (of which spikes {fmt_list([round(r['spikes_per_s'] / 1000, 1) for r in g], 1)}k, "
                           f"graded quanta {fmt_list([round(r['graded_per_s'] / 1000, 1) for r in g], 1)}k)")
            high = [f"seed {r['seed']} from {r['high_state_from_s']} s" for r in g if r["high_state_from_s"] is not None]
            out.append(f"    high state (>60k events/s held >= 1 s): measured {', '.join(high) if high else 'none'}; "
                       f"escapes {fmt_list([r['escape_events'] for r in g])} (ticks {fmt_list([r['escape_ticks'] for r in g])}); "
                       f"{label} {pub.get('quiet_high', '–')}")
            back = [f"seed {r['seed']}: {r['back_ticks_no_touch']} ticks from {r['back_no_touch_from_s']} s" for r in g if r["back_ticks_no_touch"]]
            out.append(f"    backing with no wall touch in the last 1.5 s: measured {', '.join(back) if back else 'none'}; {label} {pub.get('quiet_back', '–')}")
        game_rows = [r for s, rs in groups.items() if s != "body" for r in rs]
        if game_rows:
            rt = f"real-time factor: measured {rng_str([r['rtf'] for r in game_rows], 3)}"
            if any(r.get("mujoco_rtf") for r in game_rows):
                rt += f" (MuJoCo alone {rng_str([r['mujoco_rtf'] for r in game_rows if r.get('mujoco_rtf')], 3)}; "
                rt += f"contacts per step {rng_str([r['ncon_mean'] for r in game_rows if r.get('ncon_mean') is not None], 1)})"
            rt += f"; {label} {pub.get('rtf', '–')}   [1-minute load while running: {rng_str([r['load1'] for r in game_rows], 1)}]"
            out.append(rt)
            out.append(f"peak memory: measured {rng_str([r['peak_mb'] for r in game_rows], 0)} MB (brain build "
                       f"{rng_str([r['build_s'] for r in game_rows], 0)} s); {label} {pub.get('memory', '–')}")
    if verdicts:
        out.append("")
        out.append("decision 25 verdicts: " + "; ".join(f"{k}: {'within tolerance' if ok else 'OUTSIDE'}" for k, ok in verdicts))
    return out


# ---------------------------------------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bodies", nargs="+", default=list(VARIANTS), choices=list(VARIANTS),
                    help="which columns: drawn, physics (raw pose), physics_sa (--stride-average)")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--only", nargs="+", default=["body", *SCENARIOS], choices=["body", *SCENARIOS],
                    help="which rows: body (speed and turn, no brain) and the game scenarios")
    ap.add_argument("--seconds", type=float, default=3.0, help="the 3 s rows (shorter for a smoke test)")
    ap.add_argument("--quiet-seconds", type=float, default=10.0)
    ap.add_argument("--jobs", type=int, default=1, help="child processes at a time (each physics run holds about 1.6 GB)")
    ap.add_argument("--wiggle", type=float, default=WIGGLE_MM, help="the lure's sideways swing in mm (0: a still lure)")
    ap.add_argument("--wiggle-hz", type=float, default=WIGGLE_HZ, help="the swing's rate")
    ap.add_argument("--json", type=Path, help="write every run's row and the report here")
    ap.add_argument("--compare", type=Path, help="print the report from a saved --json file and exit")
    ap.add_argument("--baseline", type=Path, help="read the numbers against this saved run (the tool's own run on the unchanged "
                                                  "body: the levers' gate) instead of the published table")
    ap.add_argument("--levers", default=None, help="physics speed levers to switch on, a comma list of " + ", ".join(
        __import__("virtual_fly.physics", fromlist=["LEVERS"]).LEVERS) + "; unset: the game's default (dedupe, which changes no "
        "number); 'none': the body with no lever")
    args = ap.parse_args(argv)
    from virtual_fly.physics import parse_levers
    try:
        levers = None if args.levers is None else parse_levers(args.levers)     # None: the game's default
    except ValueError as e:
        ap.error(str(e))
    target, label = None, "published"
    if args.baseline:
        target, label = summarise(json.loads(args.baseline.read_text())["rows"]), f"baseline ({args.baseline.name})"
    if args.compare:
        saved = json.loads(args.compare.read_text())
        print("\n".join(report(saved["rows"], target, label)))
        return 0
    jobs: list[tuple] = []
    for variant in args.bodies:
        kind, sa = VARIANTS[variant]
        for seed in args.seeds:
            if "body" in args.only and not sa:
                jobs.append((body_job, {"kind": kind, "seed": seed, "seconds": args.seconds, "levers": levers}))
            for scen in SCENARIOS:
                if scen in args.only:
                    secs = args.quiet_seconds if scen == "quiet" else args.seconds
                    jobs.append((game_job, {"kind": kind, "stride_average": sa, "seed": seed, "scenario": scen, "seconds": secs,
                                            "wiggle": (args.wiggle, args.wiggle_hz), "levers": levers}))

    def describe(job):
        fn, kw = job
        v = kw["kind"] + ("_sa" if kw.get("stride_average") else "")
        return f"{v} seed {kw['seed']} {kw.get('scenario', 'body')}"

    print(f"{len(jobs)} runs, {args.jobs} at a time; load now {os.getloadavg()[0]:.1f}"
          + (f"; levers {', '.join(levers)}" if levers else "; the game's default levers" if levers is None else "; no lever"), flush=True)
    t0 = time.monotonic()
    rows = run_jobs(jobs, max(1, args.jobs), describe)
    lines = report(rows, target, label)
    print("\n".join(lines))
    print(f"\n{len(jobs)} runs in {(time.monotonic() - t0) / 60:.1f} min")
    if args.json:
        from virtual_fly import __version__
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({"tool": "tools/physics_table.py", "kit": __version__, "date": time.strftime("%Y-%m-%d %H:%M"),
                                         "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
                                         "rows": rows, "report": lines}, indent=1))
        print(f"written {args.json}")
    return 1 if any("error" in r for r in rows) else 0


if __name__ == "__main__":
    sys.exit(main())
