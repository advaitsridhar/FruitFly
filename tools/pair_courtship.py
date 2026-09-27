#!/usr/bin/env python3
"""The pair scenario, measured: the male and the simulated FlyWire female in one dish for 90 s, five seeds, with the three
controls the plan asks for (a hand-run tool for the two-flies work; docs/TWO_FLIES_PLAN.md 5.9 item 3).

    nice -n 10 .venv/bin/python tools/pair_courtship.py --json ../runs/p1-pair-courtship.json
    nice -n 10 .venv/bin/python tools/pair_courtship.py --conditions full --seeds 0 --seconds 10

Conditions (each seed starts a fresh game, both brains in their own processes, placed as the courtship scenario places them:
the male at (-10, -8) heading 0.3 rad, the female at (14, 10) with a heading from her own dice):

  full       the four default channels (seen, song, contact, collide), her walking urge on
  no_urge    the same, her walking urge (the hand-built autopilot) off: what she does without it
  no_song    the song channel off, the male still singing: what his song adds
  no_social  a partner placed but every channel off: what is left when the flies cannot sense each other

Her walking descending neurons stay near 0 Hz (docs/SCIENCE.md 9.4), so her speed, heading and orientation come mostly
from the hand-built walking urge; an effect is claimed only when it exceeds the spread across seeds against these
controls. Escape jumps are counted for each fly as ticks with GF_BURST or more live giant-fibre spikes over two ticks,
the game's own rule; jump ticks are left out of the speed numbers.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from virtual_fly.game import GF_BURST, TICK_MS
from virtual_fly.senses.mechano import SOUND
from virtual_fly.senses.social import SocialConfig
from virtual_fly.world import wrap

CONDITIONS = ("full", "no_urge", "no_song", "no_social")
NEAR_MM, FACING_DEG = 15.0, 30.0           # real centre-to-centre mm; degrees off his heading


def social_for(condition: str) -> SocialConfig:
    if condition == "no_social":
        return SocialConfig(seen=False, song=False, contact=False, collide=False)
    if condition == "no_song":
        return SocialConfig(song=False)
    return SocialConfig()


def mean(xs):
    return round(statistics.fmean(xs), 3) if xs else None


def run(seed: int, condition: str, seconds: float, parts: bool) -> dict:
    from virtual_fly import load_connectome
    from virtual_fly.game import Game
    from virtual_fly.parts import PartsList
    from virtual_fly.settings import build_brain
    conn, fconn = load_connectome(quiet=True), load_connectome(female=True, quiet=True)
    parts_list = PartsList() if parts else None
    kw = {"seed": seed, "dt": 0.5}
    brain = build_brain(conn, "game", **kw, **({"parts": parts_list} if parts else {}))
    game = Game(brain, seed=seed, brain_kwargs=dict(kw), parts_list=parts_list, brain_procs="auto", social=social_for(condition),
                partner={"conn": fconn, "brain_kwargs": dict(kw), "parts": parts_list if parts else False,
                         "autopilot": condition != "no_urge"})
    del brain
    try:
        m, f = game.flies
        m.body.reset(-10.0, -8.0, 0.3)
        f.body.reset(14.0, 10.0, f.rng.uniform(-math.pi, math.pi))
        ticks = int(round(seconds * 1000.0 / TICK_MS))
        near = facing = singing = 0
        contacts = 0
        touching_before = False
        hear_hz, her_v_song, her_v_quiet, his_v = [], [], [], []
        her = {"vpoEN": [], "DNp37": [], "DNp13": [], "DNp01": [], "pC1": []}
        his = {"pC1": [], "pIP10": [], "LC10aL": [], "LC10aR": [], "DNp01": []}
        bursts = {"male": 0, "female": 0}
        prev_gf = {"male": 0, "female": 0}
        modes = {"male": {}, "female": {}}
        min_d = 1e9
        for _ in range(ticks):
            game.tick()
            mp, fp = m.body.pose, f.body.pose
            d = math.hypot(fp.x - mp.x, fp.y - mp.y)
            min_d = min(min_d, d)
            sings = m.m["song"] > 0.3
            singing += sings
            if d < NEAR_MM:
                near += 1
                if abs(wrap(math.atan2(fp.y - mp.y, fp.x - mp.x) - mp.h)) < math.radians(FACING_DEG):
                    facing += 1
            touching = "touches_fly" in m.senses_now
            contacts += touching and not touching_before
            touching_before = touching
            hz = f.rates_now.get(SOUND, 0.0) if f.rates_now else 0.0
            if hz > 0:
                hear_hz.append(hz)
            for k in her:
                if k in f.bt.hz:
                    her[k].append(f.bt.hz[k])
            for k in his:
                if k in m.bt.hz:
                    his[k].append(m.bt.hz[k])
            if fp.jump is None:
                (her_v_song if sings else her_v_quiet).append(abs(fp.v))
            if mp.jump is None:
                his_v.append(abs(mp.v))
            for name, a in (("male", m), ("female", f)):
                if a.bt.gf + prev_gf[name] >= GF_BURST:
                    bursts[name] += 1
                prev_gf[name] = a.bt.gf
                modes[name][a.mode] = modes[name].get(a.mode, 0) + 1
        escapes = {"male": sum(1 for e in game.events.items if "escape" in e["text"] and e.get("fly", 0) == 0),
                   "female": sum(1 for e in game.events.items if "escape" in e["text"] and e.get("fly") == 1)}
        return {"seed": seed, "condition": condition, "parts": parts, "seconds": seconds, "ticks": ticks,
                "channels": game.social.names(),
                "time_near_s": round(near * TICK_MS / 1000, 1), "facing_when_near": round(facing / near, 3) if near else None,
                "min_distance_mm": round(min_d, 1), "time_singing_s": round(singing * TICK_MS / 1000, 1),
                "contacts_per_min": round(contacts / seconds * 60, 2),
                "her_hearing_hz_mean": mean(hear_hz), "her_hearing_share": round(len(hear_hz) / ticks, 3),
                "her_hz": {k: mean(v) for k, v in her.items()}, "his_hz": {k: mean(v) for k, v in his.items()},
                "her_speed_while_he_sings": mean(her_v_song), "her_speed_while_quiet": mean(her_v_quiet),
                "his_speed": mean(his_v), "bursts": bursts, "escape_events": escapes,
                "modes": {k: {mm: round(n / ticks, 3) for mm, n in v.items()} for k, v in modes.items()}}
    finally:
        game.close()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--conditions", default=",".join(CONDITIONS))
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--seconds", type=float, default=90.0)
    ap.add_argument("--parts", action="store_true")
    ap.add_argument("--json", metavar="FILE")
    args = ap.parse_args(argv)
    conds = [c.strip() for c in args.conditions.split(",") if c.strip()]
    bad = [c for c in conds if c not in CONDITIONS]
    if bad or not conds:
        ap.error(f"--conditions: unknown {', '.join(bad) or '(empty)'}; choose from {', '.join(CONDITIONS)}")
    try:
        seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    except ValueError:
        ap.error("--seeds takes a comma list of whole numbers")
    if args.seconds <= 0 or not seeds:
        ap.error("--seconds must be more than 0 and --seeds not empty")
    if args.json and not Path(args.json).resolve().parent.is_dir():
        ap.error(f"--json: the folder of {args.json} does not exist")
    from virtual_fly.brainio import warm_up_once
    warm_up_once()
    rows = []
    t0 = time.time()
    for condition in conds:
        for seed in seeds:
            t1 = time.time()
            row = run(seed, condition, args.seconds, args.parts)
            rows.append(row)
            print(f"{condition:9s} seed {seed}: near {row['time_near_s']:5.1f} s, facing {row['facing_when_near']}, sang "
                  f"{row['time_singing_s']:5.1f} s, contacts/min {row['contacts_per_min']}, she hears {row['her_hearing_hz_mean']} Hz "
                  f"({100 * row['her_hearing_share']:.0f} % of ticks), her v sing/quiet {row['her_speed_while_he_sings']}/"
                  f"{row['her_speed_while_quiet']} mm/s, DNp37 {row['her_hz']['DNp37']}, DNp13 {row['her_hz']['DNp13']}, "
                  f"bursts m/f {row['bursts']['male']}/{row['bursts']['female']}  ({time.time() - t1:.0f} s)", flush=True)
    if args.json:
        Path(args.json).write_text(json.dumps({"when": time.strftime("%Y-%m-%d %H:%M:%S"), "seconds": args.seconds,
                                               "parts": args.parts, "rows": rows, "wall_s": round(time.time() - t0)}, indent=1))
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
