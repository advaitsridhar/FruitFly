#!/usr/bin/env python3
"""How loud can the male's song be before it startles the listener? (a hand-run tool for the two-flies work)

    nice -n 10 .venv/bin/python tools/song_startle.py --json ../runs/p1-startle-male.json
    nice -n 10 .venv/bin/python tools/song_startle.py --female --json ../runs/p1-startle-female.json

The song reaches the other fly as a rate on its Johnston's-organ cells (``prefix:JO-A,prefix:JO-B``, the game's ``SOUND``
spec). In this model those cells also reach the giant fibre, and since v2.8.1 the game starts an escape jump when the giant
fibre (``DNp01``, two cells) fires a burst: ``game.GF_BURST`` (5) or more live spikes over two consecutive 25 ms ticks
(docs/SCIENCE.md 5.7). Mean rates hide bursts, so this tool counts them per window: for each swept rate it drives the sound
cells at that rate for ``--seconds`` of brain time on each seed, bins the giant fibre's spikes into 25 ms ticks (a monitor
with ``bin_ms`` = one tick), adds each tick to the one before (the window slides by one tick, as the game counts it), and
reports the share of windows holding ``GF_BURST`` or more spikes and the largest window. The plan's rule (docs/TWO_FLIES_PLAN.md
5.9 item 1, decision 6) makes ``SONG_MAX_HZ`` the highest swept rate at which fewer than 1 % of windows hold a burst, the windows pooled
over five seeds (995 per rate and parts setting), parts list off and on: a hand-built calibration taken from the male
connectome and the burst rule. (Applied seed by seed the rule would give 60 Hz: docs/SCIENCE.md 10.3.)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from virtual_fly.game import GF_BURST, TICK_MS          # never write 5 here: the rule is the game's

SOUND = "prefix:JO-A,prefix:JO-B"
GF = "DNp01"


def windows(history: list[float], n_cells: int, tick_ms: float) -> list[int]:
    """Spikes per two-tick window from a monitor's Hz-per-cell history (one value per tick)."""
    per_tick = [int(round(hz * n_cells * tick_ms / 1000.0)) for hz in history]
    return [a + b for a, b in zip(per_tick[1:], per_tick[:-1])]


def measure(conn, parts: bool, seed: int, rates: list[float], seconds: float, settle_ms: float) -> list[dict]:
    from virtual_fly.settings import build_brain
    brain = build_brain(conn, "game", seed=seed, **({"parts": True} if parts else {}))
    n_gf = int(conn.select(GF).size)
    rows = []
    for hz in rates:
        brain.reset()
        brain.clear_stimuli()
        brain.stimulate(SOUND, hz)
        brain.run(settle_ms)                                       # the first ticks after onset are the onset response
        mon = brain.add_monitor("gf", GF, bin_ms=TICK_MS)
        brain.reset_counts()
        brain.run(seconds * 1000.0)
        w = windows(mon.history, n_gf, TICK_MS)
        brain.remove_monitor("gf")
        bursts = sum(1 for x in w if x >= GF_BURST)
        rows.append({"hz": hz, "seed": seed, "parts": parts, "windows": len(w), "bursts": bursts,
                     "burst_share": bursts / max(1, len(w)), "max_window": max(w) if w else 0,
                     "gf_hz_per_cell": round(sum(mon.history) / max(1, len(mon.history)), 2)})
        print(f"  {'parts on ' if parts else 'parts off'} seed {seed} {hz:5.1f} Hz: {bursts}/{len(w)} windows with a burst "
              f"({100 * bursts / max(1, len(w)):.2f} %), largest {max(w) if w else 0}, GF {rows[-1]['gf_hz_per_cell']} Hz/cell",
              flush=True)
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--female", action="store_true", help="the FlyWire female as the listener (the rule itself is set on the male)")
    ap.add_argument("--rates", default="10,20,30,40,50,60,70,80,90,100", help="comma list of JO-A/B rates in Hz")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--seconds", type=float, default=5.0, help="brain seconds counted per rate and seed")
    ap.add_argument("--settle-ms", type=float, default=100.0, help="brain time after onset left out of the count")
    ap.add_argument("--parts", default="off,on", help="off, on or off,on")
    ap.add_argument("--json", metavar="FILE")
    args = ap.parse_args(argv)
    try:
        rates = [float(x) for x in args.rates.split(",") if x.strip()]
        seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    except ValueError as e:
        ap.error(f"--rates and --seeds take comma lists of numbers ({e})")
    parts_modes = [p.strip() for p in args.parts.split(",") if p.strip()]
    if not rates or not seeds or any(r <= 0 for r in rates) or any(s < 0 for s in seeds):
        ap.error("--rates must be positive, --seeds 0 or more, and neither empty")
    if any(p not in ("off", "on") for p in parts_modes) or not parts_modes:
        ap.error("--parts takes off, on or off,on")
    if args.seconds <= 0 or args.settle_ms < 0:
        ap.error("--seconds must be more than 0 and --settle-ms 0 or more")
    if args.json and not Path(args.json).resolve().parent.is_dir():
        ap.error(f"--json: the folder of {args.json} does not exist")
    from virtual_fly import load_connectome
    conn = load_connectome(female=args.female, quiet=True)
    print(f"{conn.dataset}: {conn.select(SOUND).size} sound cells, {conn.select(GF).size} giant-fibre cells; a burst is "
          f"{GF_BURST} or more spikes over two {TICK_MS:g} ms ticks; {args.seconds:g} s per rate and seed, seeds {seeds}")
    t0 = time.time()
    rows = []
    for parts in [p == "on" for p in parts_modes]:
        for seed in seeds:
            rows.extend(measure(conn, parts, seed, rates, args.seconds, args.settle_ms))
    summary = []
    for parts in [p == "on" for p in parts_modes]:
        for hz in rates:
            sel = [r for r in rows if r["parts"] == parts and r["hz"] == hz]
            share = sum(r["bursts"] for r in sel) / max(1, sum(r["windows"] for r in sel))
            summary.append({"parts": parts, "hz": hz, "burst_share": share, "max_window": max(r["max_window"] for r in sel),
                            "gf_hz_per_cell": round(sum(r["gf_hz_per_cell"] for r in sel) / len(sel), 2)})
    print("\nrate  parts  share of 50 ms windows with a burst   largest window   GF Hz per cell")
    for s in summary:
        print(f"{s['hz']:5.1f}  {'on ' if s['parts'] else 'off'}    {100 * s['burst_share']:6.2f} %"
              f"{'':26}{s['max_window']:3d}{'':13}{s['gf_hz_per_cell']}")
    ok_rates = [hz for hz in rates if all(s["burst_share"] < 0.01 for s in summary if s["hz"] == hz)]
    chosen = max(ok_rates) if ok_rates else None
    print(f"\nSONG_MAX_HZ by the rule (under 1 % of windows with a burst, pooled over the seeds, parts off and on): "
          f"{chosen if chosen is not None else 'none of the swept rates'}")
    if args.json:
        Path(args.json).write_text(json.dumps({"dataset": conn.dataset, "female": args.female, "gf_burst": GF_BURST,
                                               "tick_ms": TICK_MS, "seconds": args.seconds, "settle_ms": args.settle_ms,
                                               "seeds": seeds, "rates": rates, "rows": rows, "summary": summary,
                                               "song_max_hz": chosen, "wall_s": round(time.time() - t0, 1)}, indent=1))
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
