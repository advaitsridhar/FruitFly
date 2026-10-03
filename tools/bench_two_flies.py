#!/usr/bin/env python3
"""Baseline speeds for the two-flies work (hand-run tool; nothing in the game uses it).

Real-time factor (RTF) = simulated time / wall time; 1.0 = real time.

    nice -n 10 .venv/bin/python tools/bench_two_flies.py --json ../runs/p0-bench.json
    nice -n 10 .venv/bin/python tools/bench_two_flies.py --only brain,pair --seconds 5

  brain    one brain at a time: male and female, parts list off and on, busy input
  pair     the male and the female brain at the same time, one process each, busy input
  game     game ticks with the drawn body and a scripted female (male, then female protagonist)
  physics  male game ticks with the physics body (needs flygym)
  pair-game  the two-fly game (male protagonist, FlyWire female partner, both brains in their own processes,
           the social channels on), parts list off and on, dt 0.5 and 1.0: ticks, and the peak memory of the
  pair-physics  the same game with both flies as NeuroMechFly bodies in one MuJoCo world (physics_pair.py; parts on,
             80 ticks), one row per --contact-sets entry: the real-time factor and MuJoCo's share, pairs and contacts
           parent and of each brain child (Phase 1, docs/TWO_FLIES_PLAN.md 5.9 item 4)

``--backend cupy`` (Phase 2, docs/TWO_FLIES_PLAN.md 6.7 item 3) runs the brains of every row on the GPU: the brain and pair rows
inline, the game rows with the brain server (one child process for every brain); the pair-game row then also samples the GPU's
utilisation and memory through nvidia-smi while it ticks (where nvidia-smi answers: on WSL2 it reports utilisation but no
per-process memory).

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

KINDS = ("brain", "pair", "game", "physics", "pair-game", "pair-physics")     # the rows --only can choose

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


def pair_rtf(parts: bool, seconds: float, backend: str = "auto") -> list[dict]:
    """The male and the female brain at the same time, one process each, timed from a shared barrier."""
    rows = in_children([(brain_rtf, {"female": female, "parts": parts, "seconds": seconds, "backend": backend})
                        for female in (False, True)], together=True)
    return sorted(rows, key=lambda r: r["fly"] != "male")      # male first, whichever finished first


def game_rtf(body: str, ticks: int, female: bool = False, parts: bool = False, backend: str = "auto") -> dict:
    from virtual_fly import load_connectome
    from virtual_fly.game import TICK_MS, Game
    from virtual_fly.settings import build_brain
    conn = load_connectome(female=female, quiet=True)
    kw = {"seed": 0, "backend": backend}
    brain = build_brain(conn, "game", **kw, **({"parts": True} if parts else {}))
    t0 = time.perf_counter()
    game = Game(brain, seed=0, body=body, brain_kwargs=dict(kw))
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


def _vmhwm_mb(pid: int) -> float | None:
    """Another process's peak resident memory in MB, from /proc (Linux); None elsewhere."""
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("VmHWM:"):
                    return round(int(line.split()[1]) / 1024.0, 1)
    except (OSError, ValueError, IndexError):
        pass
    return None


class _GpuSampler:
    """Samples the GPU's utilisation and memory through nvidia-smi every half second in a thread (plan 6.7 item 3);
    silent where nvidia-smi is missing or answers nothing."""

    def __init__(self):
        import threading
        self.samples, self._stop = [], threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            try:
                out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
                                     capture_output=True, text=True, timeout=5)
                util, mem = (float(x) for x in out.stdout.strip().split(",")[:2])
                self.samples.append((util, mem))
            except (OSError, ValueError, subprocess.SubprocessError):
                pass
            self._stop.wait(0.5)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=10)

    def summary(self) -> dict | None:
        if not self.samples:
            return None
        return {"samples": len(self.samples), "util_percent_mean": round(statistics.fmean(u for u, _ in self.samples), 1),
                "util_percent_max": max(u for u, _ in self.samples), "memory_used_mb_max": max(m for _, m in self.samples)}


def pair_game_rtf(parts: bool, dt: float, ticks: int, backend: str = "auto", body: str = "drawn", contact_set: str = "forelegs") -> dict:
    """The two-fly game: the male with a simulated FlyWire female, each brain in its own process (or, with the cupy
    backend, both in the brain server's process on the GPU), lockstepped."""
    import contextlib
    from virtual_fly import load_connectome
    from virtual_fly.game import TICK_MS, Game
    from virtual_fly.parts import PartsList
    from virtual_fly.settings import build_brain
    conn, fconn = load_connectome(quiet=True), load_connectome(female=True, quiet=True)
    parts_list = PartsList() if parts else None
    kw = {"seed": 0, "dt": dt, "backend": backend}
    # the game process builds the first brain on the CPU to describe it to its child (as fly_game.py does); the child
    # builds the brain the kwargs ask for
    brain = build_brain(conn, "game", **{**kw, "backend": "auto" if backend == "cupy" else backend},
                        **({"parts": parts_list} if parts else {}))
    t0 = time.perf_counter()
    game = Game(brain, seed=0, brain_kwargs=dict(kw), parts_list=parts_list, brain_procs="auto", body=body,
                partner={"conn": fconn, "brain_kwargs": dict(kw), "parts": parts_list if parts else False})
    del brain
    if game.pair_world is not None and contact_set != game.pair_world.contact_set:    # the pair's contact set, measured
        from virtual_fly.agent import HOME, PARTNER_HOME
        from virtual_fly.physics_pair import PairWorld
        game.pair_world.close()
        game.pair_world = PairWorld([HOME, PARTNER_HOME], seed=0, world=game.world, contact_set=contact_set)
        for f, b in zip(game.flies, game.pair_world.bodies):
            b.world, b.rng, f.body = f.world, f.rng, b
        game.reset_world()
    setup_s = time.perf_counter() - t0
    try:
        for _ in range(20):
            game.tick()
        per_tick, ncon = [], []
        import resource
        cpu0 = resource.getrusage(resource.RUSAGE_SELF)
        mj0 = game.pair_world.wall_s if game.pair_world is not None else 0.0
        with (_GpuSampler() if backend == "cupy" else contextlib.nullcontext()) as gpu:
            for _ in range(ticks):
                t1 = time.perf_counter()
                game.tick()
                per_tick.append(time.perf_counter() - t1)
                if game.pair_world is not None:
                    ncon.append(game.pair_world.ncon)
        cpu1 = resource.getrusage(resource.RUSAGE_SELF)
        wall = sum(per_tick)
        physics_row = None
        if game.pair_world is not None:           # the shared MuJoCo world's share (docs/TWO_FLIES_PLAN.md 8.2, 8.6)
            mj = game.pair_world.wall_s - mj0
            physics_row = {"contact_set": game.pair_world.contact_set, "fly_to_fly_pairs": len(game.pair_world.pairs),
                           "model_pairs": int(game.pair_world._m.npair), "mujoco_s": round(mj, 2),
                           "mujoco_share": round(mj / wall, 3), "mujoco_rtf": round(ticks * TICK_MS / 1000.0 / mj, 3) if mj else None,
                           "ms_per_step": round(mj / (ticks * TICK_MS / 1000.0 / game.pair_world.timestep) * 1000, 4),
                           "ncon_mean": round(statistics.fmean(ncon), 1), "ncon_max": max(ncon),
                           "min_distance_mm": None}
        q = statistics.quantiles(per_tick, n=100)
        children, seen = {}, set()
        for a in game.flies:                      # a process per brain, or the one brain server for every brain (6.5)
            server = getattr(a.io, "server", None)
            proc = getattr(a.io, "proc", None)
            if proc is not None and proc.pid not in seen:
                seen.add(proc.pid)
                children["brain_server" if server is not None else f"fly{a.id}_{a.sex}"] = _vmhwm_mb(proc.pid)
        return {"fly": "male+female", "body": body, "parts": parts, "dt": dt, "ticks": ticks, "flies": len(game.flies),
                "physics": physics_row,
                "backend": game.flies[0].io.settings().get("backend"),
                "channels": sorted(game.social.names()) if hasattr(game.social, "names") else None,
                "rtf": round(ticks * TICK_MS / 1000.0 / wall, 3),
                "tick_ms_p50": round(q[49] * 1000, 2), "tick_ms_p99": round(q[98] * 1000, 2),
                "game_process_cpu_percent": round(100.0 * ((cpu1.ru_utime - cpu0.ru_utime) + (cpu1.ru_stime - cpu0.ru_stime)) / wall, 1),
                "gpu": gpu.summary() if backend == "cupy" else None,
                "setup_s": round(setup_s, 1), "peak_rss_mb": peak_rss_mb(), "children_peak_rss_mb": children}
    finally:
        game.close()


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
        import cupy
        info["cupy"] = cupy.__version__
        info["cuda_runtime"] = cupy.cuda.runtime.runtimeGetVersion()
        info["cuda_driver"] = cupy.cuda.runtime.driverGetVersion()
    except Exception:
        info["cupy"] = None
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
    ap.add_argument("--contact-sets", default="forelegs", help="pair-physics: the fly-to-fly contact sets to measure, a comma list "
                                                               "of forelegs, full, none (physics_pair.CONTACT_SETS)")
    ap.add_argument("--backend", choices=("auto", "numpy", "numba", "cupy"), default="auto",
                    help="the brains' integrator for every row (cupy: the GPU, Phase 2; checked before anything loads)")
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
    if args.backend == "cupy":           # one line before anything loads, as fly_brain.py and fly_game.py say it
        from virtual_fly import gpubrain
        reason = gpubrain.unavailable_reason()
        if reason is not None:
            ap.error(f"--backend cupy: {reason}")
    from virtual_fly import fastbrain
    fastbrain.warm_up()                  # compile or load the kernels once, before any child starts
    out = {"when": time.strftime("%Y-%m-%d %H:%M:%S"), "machine": machine(), "rows": []}

    def add(kind, row):
        out["rows"].append({"kind": kind, **row})
        print(kind, row, flush=True)

    if "brain" in want:
        for female in (False, True):
            for parts in (False, True):
                add("brain", in_children([(brain_rtf, {"female": female, "parts": parts, "seconds": args.seconds,
                                                        "backend": args.backend})])[0])
    if "pair" in want:
        for parts in (False, True):
            for row in pair_rtf(parts, args.seconds, args.backend):
                add("pair", row)
    if "game" in want:
        for female in (False, True):
            add("game", in_children([(game_rtf, {"body": "drawn", "ticks": 400, "female": female, "backend": args.backend})])[0])
    if "physics" in want:
        from virtual_fly import physics
        if physics.available():
            add("game", in_children([(game_rtf, {"body": "physics", "ticks": 80, "backend": args.backend})])[0])
        else:
            print("physics: flygym is not installed, skipped")
    if "pair-physics" in want:                # both flies NeuroMechFly bodies in one MuJoCo world (8.3): parts on, per contact set
        for contact_set in args.contact_sets.split(","):
            add("pair-physics", in_children([(pair_game_rtf, {"parts": True, "dt": 0.5, "ticks": 80, "backend": args.backend,
                                                               "body": "physics", "contact_set": contact_set.strip()})])[0])
    if "pair-game" in want:
        for parts in (False, True):
            for dt in (0.5, 1.0):
                add("pair-game", in_children([(pair_game_rtf, {"parts": parts, "dt": dt, "ticks": 400, "backend": args.backend})])[0])
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
