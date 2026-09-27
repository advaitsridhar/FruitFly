#!/usr/bin/env python3
"""Baseline speeds for the two-flies work (hand-run tool; nothing in the game uses it).

Real-time factor (RTF) = simulated time / wall time; 1.0 = real time.

    nice -n 10 .venv/bin/python tools/bench_two_flies.py --json ../runs/p0-bench.json
    nice -n 10 .venv/bin/python tools/bench_two_flies.py --only brain,pair --seconds 5

  brain    one brain at a time: male and female, parts list off and on, busy input
  pair     the male and the female brain at the same time, one process each, busy input
  game     game ticks with the drawn body and a scripted female (male, then female protagonist)
  physics  male game ticks with the physics body (needs flygym)

Every row is measured in a fresh child process (``spawn``), so its peak memory is its own: a process's high-water mark
never falls, and on Linux a child inherits the parent's ``ru_maxrss``, so rows measured one after another in one process
would all report the largest so far. ``peak_rss_mb`` reads ``VmHWM`` from ``/proc/self/status`` where it exists.
"""
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import platform
import queue
import statistics
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

KINDS = ("brain", "pair", "game", "physics")     # the rows --only can choose

# Busy input (the names resolve on both flies; the female reaches LB3b,LB3c through her aliases)
BUSY = {"LB3b,LB3c": 120.0, "LC4/R,LPLC2/R": 150.0,
        "ORN_DM1,ORN_DM4,ORN_VM7d,ORN_DP1m": 80.0, "prefix:JO-B": 100.0}


def peak_rss_mb() -> float | None:
    """This process's peak resident memory in MB (its own: not inherited from the parent, see the docstring)."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmHWM:"):
                    return round(int(line.split()[1]) / 1024.0, 1)             # kB
    except (OSError, ValueError, IndexError):
        pass
    try:
        import resource
        return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1)     # Linux: kB
    except (ImportError, AttributeError):
        return None


def brain_rtf(female: bool, parts: bool, seconds: float = 3.0, backend: str = "auto", barrier=None) -> dict:
    from virtual_fly import load_connectome
    from virtual_fly.settings import build_brain
    conn = load_connectome(female=female, quiet=True)
    t0 = time.perf_counter()
    brain = build_brain(conn, "game", seed=0, backend=backend, **({"parts": True} if parts else {}))
    build_s = time.perf_counter() - t0
    brain.set_stimuli({k: v for k, v in BUSY.items() if conn.select(k).size})
    brain.run(300.0)                      # warm-up: kernel cache, activity settles
    if barrier is not None:
        barrier.wait(timeout=600)         # both brains start timing together; gives up if the other one died
    brain.reset_counts()
    t0 = time.perf_counter()
    brain.run(seconds * 1000.0)
    wall = time.perf_counter() - t0
    steps = seconds * 1000.0 / brain.dt
    return {"fly": "female" if female else "male", "parts": parts, "backend": brain.backend,
            "sim_s": seconds, "wall_s": round(wall, 3), "rtf": round(seconds / wall, 3),
            "ms_per_step": round(wall * 1000.0 / steps, 4),
            "events_per_s": round(float(brain.spike_count.sum()) / seconds),
            "build_s": round(build_s, 2), "peak_rss_mb": peak_rss_mb()}


def _child(q, barrier, fn, kwargs):
    """A row measured in its own process (module level, so that ``spawn`` can import it)."""
    try:
        if barrier is not None:
            kwargs = {**kwargs, "barrier": barrier}
        q.put(fn(**kwargs))
    except BaseException as e:            # report it, so the parent does not wait for a row that never comes
        q.put({"error": f"{fn.__name__}({kwargs}): {e!r}"})
        raise


def in_children(jobs: list[tuple], together: bool = False) -> list[dict]:
    """Run each (function, kwargs) job in a fresh child process and return their rows in arrival order. With
    ``together`` the children share a barrier, so they start timing at the same moment (the pair rows)."""
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    barrier = ctx.Barrier(len(jobs)) if together and len(jobs) > 1 else None
    procs = [ctx.Process(target=_child, args=(q, barrier, fn, kw)) for fn, kw in jobs]
    for p in procs:
        p.start()
    rows, deadline = [], time.monotonic() + 1800
    try:
        while len(rows) < len(procs):
            if time.monotonic() > deadline:
                raise RuntimeError("benchmark: no result after 30 minutes")
            try:
                row = q.get(timeout=30)
            except queue.Empty:
                if any(p.exitcode not in (None, 0) for p in procs):   # killed (for example out of memory)
                    raise RuntimeError(f"a benchmark process died: exit codes {[p.exitcode for p in procs]}")
                continue
            if "error" in row:
                raise RuntimeError(row["error"])
            rows.append(row)
    finally:
        for p in procs:                   # stop any survivor through its own handle (its PID), never by pattern
            if p.is_alive():
                p.terminate()
            p.join(timeout=30)
    return rows


def pair_rtf(parts: bool, seconds: float) -> list[dict]:
    """The male and the female brain at the same time, one process each, timed from a shared barrier."""
    rows = in_children([(brain_rtf, {"female": female, "parts": parts, "seconds": seconds}) for female in (False, True)],
                       together=True)
    return sorted(rows, key=lambda r: r["fly"] != "male")      # male first, whichever finished first


def game_rtf(body: str, ticks: int, female: bool = False, parts: bool = False) -> dict:
    from virtual_fly import load_connectome
    from virtual_fly.game import TICK_MS, Game
    from virtual_fly.settings import build_brain
    conn = load_connectome(female=female, quiet=True)
    brain = build_brain(conn, "game", seed=0, **({"parts": True} if parts else {}))
    t0 = time.perf_counter()
    game = Game(brain, seed=0, body=body)
    setup_s = time.perf_counter() - t0
    game.world.toggle_female(True, 14.0, 10.0)         # something to look at and chase
    for _ in range(20):
        game.tick()
    per_tick = []
    for _ in range(ticks):
        t1 = time.perf_counter()
        game.tick()
        per_tick.append(time.perf_counter() - t1)
    wall = sum(per_tick)
    q = statistics.quantiles(per_tick, n=100)
    return {"fly": "female" if female else "male", "body": body, "parts": parts, "ticks": ticks,
            "rtf": round(ticks * TICK_MS / 1000.0 / wall, 3),
            "tick_ms_p50": round(q[49] * 1000, 2), "tick_ms_p99": round(q[98] * 1000, 2),
            "setup_s": round(setup_s, 1), "peak_rss_mb": peak_rss_mb()}


def machine() -> dict:
    import numpy
    info = {"python": sys.version.split()[0], "platform": platform.platform(), "cpus": os.cpu_count(),
            "numpy": numpy.__version__}
    try:
        import numba
        info["numba"] = numba.__version__
    except ImportError:
        info["numba"] = None
    try:
        info["load_avg"] = os.getloadavg()
    except (AttributeError, OSError):
        pass
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                              "--format=csv,noheader"], capture_output=True, text=True, timeout=10)
        info["gpu"] = out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        info["gpu"] = None
    return info


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", default=",".join(KINDS), help="comma list of " + ", ".join(KINDS) + " (default: all)")
    ap.add_argument("--seconds", type=float, default=3.0, help="simulated seconds per brain measurement")
    ap.add_argument("--json", metavar="FILE")
    args = ap.parse_args()
    # refuse bad values before anything is loaded (the kit's rule for every option: one line, exit code 2)
    want = {s.strip() for s in args.only.split(",") if s.strip()}
    unknown = sorted(want - set(KINDS))
    if unknown or not want:
        ap.error(f"--only: unknown kind{'s' if len(unknown) > 1 else ''} {', '.join(unknown) or '(empty)'}; choose from {', '.join(KINDS)}")
    if not math.isfinite(args.seconds) or args.seconds <= 0:
        ap.error(f"--seconds must be more than 0, not {args.seconds:g}")
    if args.json:
        if Path(args.json).is_dir():
            ap.error(f"--json {args.json} is a folder; give a file name")
        folder = Path(args.json).resolve().parent
        if not folder.is_dir():
            ap.error(f"--json: the folder {folder} does not exist; make it first, so the results are not lost at the end")
    from virtual_fly import fastbrain
    fastbrain.warm_up()                  # compile or load the kernels once, before any child starts
    out = {"when": time.strftime("%Y-%m-%d %H:%M:%S"), "machine": machine(), "rows": []}

    def add(kind, row):
        out["rows"].append({"kind": kind, **row})
        print(kind, row, flush=True)

    if "brain" in want:
        for female in (False, True):
            for parts in (False, True):
                add("brain", in_children([(brain_rtf, {"female": female, "parts": parts, "seconds": args.seconds})])[0])
    if "pair" in want:
        for parts in (False, True):
            for row in pair_rtf(parts, args.seconds):
                add("pair", row)
    if "game" in want:
        for female in (False, True):
            add("game", in_children([(game_rtf, {"body": "drawn", "ticks": 400, "female": female})])[0])
    if "physics" in want:
        from virtual_fly import physics
        if physics.available():
            add("game", in_children([(game_rtf, {"body": "physics", "ticks": 80})])[0])
        else:
            print("physics: flygym is not installed, skipped")
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
