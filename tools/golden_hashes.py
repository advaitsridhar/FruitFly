#!/usr/bin/env python3
"""Golden single-fly hashes: what one fly does, byte for byte (a hand-run tool for the real connectomes).

    .venv/bin/python tools/golden_hashes.py --save ../runs/p0-golden-real.json        # before any change (Phase 0)
    .venv/bin/python tools/golden_hashes.py --compare ../runs/p0-golden-real.json     # at the end of every phase

For the male and the female fly, each configuration in CONFIGS builds ``Game(build_brain(conn, "game", seed=0),
seed=0, ...)``, sends its actions, runs 400 ticks (80 with the physics body, which needs flygym) and takes one SHA-1
over every ``state_json`` frame in order. Two runs of the same code must give the same hash; a refactor that keeps
single-fly behaviour must too. ``tests/test_golden_single_fly.py`` does the same on the synthetic connectome (in CI,
which has no data); this tool is the real-data version and records the Python, NumPy and numba versions next to the
hashes, since a library change is the one thing besides a code change that can move them (docs/TWO_FLIES_PLAN.md, 4.9).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# name: (Game keyword arguments, actions before tick 0, actions at tick 100). Every reply must be {"ok": true, ...}:
# since v2.8.1 a refused action answers {"ok": false, "error": ...} and does nothing, so a typo here fails loudly
# instead of pinning a run without it. Switches are Python True/False (1 or "true" is refused).
CONFIGS = {
    "autopilot":          ({}, [], []),
    "no_autopilot":       ({"autopilot": False}, [], []),
    "scripted_female":    ({}, [{"type": "female", "on": True, "x": 14, "y": 10}], []),
    "odour_and_sugar":    ({}, [{"type": "drop", "kind": "vinegar", "x": 10, "y": 5, "food": "sugar"},
                                {"type": "drop", "kind": "sugar", "x": -8, "y": 12}], []),
    "drum":               ({}, [{"type": "stripes", "count": 12, "drum_speed": 1.0}], []),
    "zap_mdn":            ({}, [], [{"type": "zap", "spec": "MDN", "hz": 60, "secs": 1.0}]),
    "silence_and_watch":  ({}, [{"type": "silence", "spec": "MN9"}, {"type": "watch", "spec": "DNa02"}], []),
    "courtship_scenario": ({}, [{"type": "scenario", "id": "courtship"}], []),
    "physics_body":       ({"body": "physics"}, [], []),     # 80 ticks, not 400; needs flygym
}
TICKS, PHYSICS_TICKS, ACTION_TICK = 400, 80, 100


def needs_physics(name: str) -> bool:
    return CONFIGS[name][0].get("body") == "physics"


def ticks_for(name: str) -> int:
    return PHYSICS_TICKS if needs_physics(name) else TICKS


def versions() -> dict:
    import numpy
    try:
        import numba
        nb = numba.__version__
    except ImportError:
        nb = None
    return {"python": platform.python_version(), "numpy": numpy.__version__, "numba": nb, "platform": sys.platform}


def run_hash(conn, name: str, ticks: int | None = None, game_kwargs: dict | None = None) -> str:
    """One configuration on ``conn``: the SHA-1 over every state_json frame, in order. ``game_kwargs`` are added to
    the Game call (a test runs the same configuration with the brain in its own process, brain_procs="on")."""
    from virtual_fly.game import Game
    from virtual_fly.settings import build_brain
    kwargs, before, at_100 = CONFIGS[name]
    game = Game(build_brain(conn, "game", seed=0), seed=0, **{**kwargs, **(game_kwargs or {})})
    n = ticks_for(name) if ticks is None else ticks
    h = hashlib.sha1()
    for tick in range(n):
        for a in (before if tick == 0 else at_100 if tick == ACTION_TICK else ()):
            reply = game.action(dict(a))
            if reply.get("ok") is not True:              # zap, silence, modulate and watch also return "n": check "ok" only
                raise RuntimeError(f"{name}: action {a} was refused: {reply}")
        game.tick()
        h.update(game.state_json)
    game.close()
    return h.hexdigest()


def all_hashes(flies=("male", "female"), only=None, quiet=False) -> dict[str, str]:
    from virtual_fly import load_connectome, physics
    out: dict[str, str] = {}
    for fly in flies:
        conn = load_connectome(female=(fly == "female"), quiet=True)
        for name in CONFIGS:
            if only and name not in only:
                continue
            if needs_physics(name) and not physics.available():
                if not quiet:
                    print(f"{fly}/{name}: skipped ({physics.unavailable_reason()})", flush=True)
                continue
            t0 = time.perf_counter()
            out[f"{fly}/{name}"] = run_hash(conn, name)
            if not quiet:
                print(f"{fly}/{name} {out[f'{fly}/{name}']}  ({time.perf_counter() - t0:.1f} s)", flush=True)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--save", metavar="FILE", help="run everything and write the hashes and versions to FILE")
    g.add_argument("--compare", metavar="FILE", help="run everything and compare with the hashes in FILE")
    ap.add_argument("--flies", default="male,female", help="which flies (default: male,female)")
    ap.add_argument("--only", help="comma list of configuration names (default: all)")
    args = ap.parse_args(argv)
    flies = tuple(f.strip() for f in args.flies.split(",") if f.strip())
    bad = [f for f in flies if f not in ("male", "female")]
    if bad:
        ap.error(f"--flies takes male and/or female, not {', '.join(bad)}")
    only = {s.strip() for s in args.only.split(",")} if args.only else None
    if only and not only <= set(CONFIGS):
        ap.error(f"unknown configuration(s) {', '.join(sorted(only - set(CONFIGS)))}; choose from {', '.join(CONFIGS)}")
    from virtual_fly import fastbrain
    fastbrain.warm_up()
    hashes = all_hashes(flies, only)
    if args.save:
        Path(args.save).write_text(json.dumps({"when": time.strftime("%Y-%m-%d %H:%M:%S"), "versions": versions(),
                                               "hashes": hashes}, indent=1, sort_keys=True) + "\n")
        print(f"wrote {args.save} ({len(hashes)} hashes)")
        return 0
    saved = json.loads(Path(args.compare).read_text())
    old, differ, missing = saved.get("hashes", {}), [], []
    for key, h in hashes.items():
        if key not in old:
            missing.append(key)
        elif old[key] != h:
            differ.append(key)
    same = [k for k in hashes if k in old and old[k] == hashes[k]]
    not_run = [k for k in old if k not in hashes]        # saved, but this run made no hash for it (skipped, or narrowed)
    narrowed = bool(args.only) or set(flies) != {"male", "female"}
    print(f"{len(same)} unchanged, {len(differ)} changed, {len(missing)} not in {args.compare}, "
          f"{len(not_run)} saved but not run{' (the run was narrowed with --flies or --only)' if narrowed and not_run else ''}")
    for k in differ:
        print(f"  CHANGED  {k}: {old[k]} -> {hashes[k]}")
    for k in missing:
        print(f"  MISSING  {k}: {hashes[k]} (not in the saved file)")
    for k in not_run:
        print(f"  NOT RUN  {k}: {old[k]} (saved, but not checked by this run"
              f"{'' if narrowed else ': a skipped configuration, for example the physics body without flygym'})")
    if saved.get("versions") != versions():
        print(f"  versions differ: saved {saved.get('versions')} / now {versions()}")
    # a full run that could not check a saved hash is a failure too: the safety net has a hole in it
    return 1 if differ or missing or (not_run and not narrowed) else 0


if __name__ == "__main__":
    sys.exit(main())
