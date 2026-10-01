#!/usr/bin/env python3
"""The GPU brain measured (hand-run tool; nothing in the game uses it; docs/TWO_FLIES_PLAN.md 6.7).

    nice -n 10 .venv/bin/python tools/bench_gpu.py --json ../runs/p2-bench-gpu.json
    nice -n 10 .venv/bin/python tools/bench_gpu.py --only environment

  environment  the driver, the device, its memory and a measured device-to-device bandwidth, CuPy's configuration
  steps        microseconds per step of the whole GPU step on the male, the female and both, at the busy input of
               tools/bench_two_flies.py, as one launch per step (eager) and as captured chunks (graphs); the CPU
               (numba) alongside for the same input (plan 6.7 item 2)
  pair-game    the two-fly game on the GPU, ticks and GPU use: tools/bench_two_flies.py --only pair-game --backend cupy

Every number is a measurement on this machine under the stated conditions, never a promise. Timing uses CUDA events on
the device for kernel work and perf_counter on the host for whole steps (which include the host's replay).
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

KINDS = ("environment", "steps", "pair-game")
BUSY = {"LB3b,LB3c": 120.0, "LC4/R,LPLC2/R": 150.0, "ORN_DM1": 100.0, "prefix:JO-B": 100.0}   # as bench_two_flies.py's busy input


def environment() -> dict:
    """What the GPU is and how fast its memory is (a 256 MB device-to-device copy, the mean of ten)."""
    import cupy as cp
    info = {"python": sys.version.split()[0], "platform": platform.platform(), "cupy": cp.__version__,
            "cuda_runtime": cp.cuda.runtime.runtimeGetVersion(), "cuda_driver": cp.cuda.runtime.driverGetVersion()}
    props = cp.cuda.runtime.getDeviceProperties(0)
    info["device"] = props["name"].decode() if isinstance(props["name"], bytes) else props["name"]
    info["compute_capability"] = f"{props['major']}.{props['minor']}"
    info["multiprocessors"] = props["multiProcessorCount"]
    info["memory_mb"] = round(props["totalGlobalMem"] / 2**20)
    info["memory_clock_khz"] = props.get("memoryClockRate")
    info["memory_bus_bits"] = props.get("memoryBusWidth")
    free, total = cp.cuda.runtime.memGetInfo()
    info["memory_free_mb"] = round(free / 2**20)
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=10)
        info["nvidia_smi"] = out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        info["nvidia_smi"] = None
    n = 64 * 2**20                                                   # 64 M float32 = 256 MB
    a = cp.ones(n, dtype=cp.float32)
    b = cp.empty_like(a)
    cp.copyto(b, a); cp.cuda.Device().synchronize()
    start, end = cp.cuda.Event(), cp.cuda.Event()
    times = []
    for _ in range(10):
        start.record(); cp.copyto(b, a); end.record(); end.synchronize()
        times.append(cp.cuda.get_elapsed_time(start, end) / 1000.0)
    info["copy_bandwidth_gb_s"] = round(2 * n * 4 / statistics.median(times) / 1e9, 1)   # read + write
    del a, b
    cp.get_default_memory_pool().free_all_blocks()
    return info


def steps(female: bool, parts: bool, seconds: float, backend: str) -> dict:
    """Microseconds per step of a brain under the busy input: the whole step as the kit runs it (advance_steps in
    chunks, i.e. graphs on the GPU), after 300 ms of warm-up."""
    from virtual_fly import load_connectome
    from virtual_fly.settings import build_brain
    conn = load_connectome(female=female, quiet=True)
    t0 = time.perf_counter()
    brain = build_brain(conn, "game", seed=0, backend=backend, **({"parts": True} if parts else {}))
    build_s = time.perf_counter() - t0
    brain.set_stimuli({k: v for k, v in BUSY.items() if conn.select(k).size})
    brain.run(300.0)
    n_steps = int(round(seconds * 1000.0 / brain.dt))
    chunk = 50                                                        # one game tick
    per_chunk = []
    brain.reset_counts()
    for _ in range(n_steps // chunk):
        t1 = time.perf_counter()
        brain.advance(chunk)
        per_chunk.append((time.perf_counter() - t1) / chunk)
    q = statistics.quantiles(per_chunk, n=100)
    eager = []
    for _ in range(200):                                              # one launch per step, the API's step()
        t1 = time.perf_counter()
        brain.step()
        eager.append(time.perf_counter() - t1)
    qe = statistics.quantiles(eager, n=100)
    return {"fly": "female" if female else "male", "parts": parts, "backend": brain.backend, "build_s": round(build_s, 2),
            "steps": n_steps, "events_per_s": round(float(brain.spike_count.sum()) / seconds),
            "us_per_step_chunked_p50": round(q[49] * 1e6, 1), "us_per_step_chunked_p99": round(q[98] * 1e6, 1),
            "us_per_step_eager_p50": round(qe[49] * 1e6, 1), "us_per_step_eager_p99": round(qe[98] * 1e6, 1),
            "rtf_chunked": round(brain.dt / 1000.0 / statistics.median(per_chunk), 2)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", default=",".join(KINDS), help="comma list of " + ", ".join(KINDS))
    ap.add_argument("--seconds", type=float, default=2.0, help="simulated seconds per steps row")
    ap.add_argument("--json", metavar="FILE")
    args = ap.parse_args()
    want = {s.strip() for s in args.only.split(",") if s.strip()}
    unknown = sorted(want - set(KINDS))
    if unknown or not want:
        ap.error(f"--only: unknown kind{'s' if len(unknown) > 1 else ''} {', '.join(unknown) or '(empty)'}; choose from {', '.join(KINDS)}")
    if args.json and Path(args.json).is_dir():
        ap.error(f"--json {args.json} is a folder; give a file name")
    from virtual_fly import gpubrain
    reason = gpubrain.unavailable_reason()
    if reason is not None:
        raise SystemExit(f"bench_gpu.py: {reason}")
    out = {"when": time.strftime("%Y-%m-%d %H:%M:%S"), "load_avg": list(os.getloadavg()), "rows": []}

    def add(kind, row):
        out["rows"].append({"kind": kind, **row})
        print(kind, row, flush=True)

    if "environment" in want:
        add("environment", environment())
    if "steps" in want:
        for female in (False, True):
            for parts in (False, True):
                for backend in ("numba", "cupy"):
                    add("steps", steps(female, parts, args.seconds, backend))
    if "pair-game" in want:
        cmd = [sys.executable, str(Path(__file__).with_name("bench_two_flies.py")), "--only", "pair-game", "--backend", "cupy"]
        print("running:", " ".join(cmd), flush=True)
        res = subprocess.run(cmd, capture_output=True, text=True)
        for line in res.stdout.splitlines():
            if line.startswith("pair-game"):
                print(line, flush=True)
                out["rows"].append({"kind": "pair-game", "line": line})
        if res.returncode:
            print(res.stderr[-2000:], file=sys.stderr)
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
