#!/usr/bin/env python3
"""Compare two ``fly_brain.py --json`` result files (a hand-run tool; nothing in the game uses it).

    .venv/bin/python tools/compare_experiments.py ../runs/p0-male-game.json NEW.json

The four Phase 0 runs (male and female, parts list off and on; docs/TWO_FLIES_PLAN.md, 4.7) are the baseline: after
every later phase the same commands are run again and their numbers must be identical. This prints every readout whose
mean or per-seed rates differ, every experiment whose ok, fragile or n/a state differs, and any after-stimulus
activity that differs; it ignores ``wall_s``. Exit code 0 when nothing differs, 1 when something does, 2 on a usage
error (a missing or unreadable file).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

IGNORED = ("wall_s",)                                    # wall time differs from run to run and means nothing


def _load(path: str) -> dict:
    try:
        with open(path) as f:
            data = json.load(f)
    except (OSError, ValueError) as e:
        _usage_error(f"can't read {path}: {e}")
    if not isinstance(data, dict) or not isinstance(data.get("results"), list):
        _usage_error(f"{path} is not a fly_brain.py --json file (no 'results' list)")
    return data


def _usage_error(message: str):
    print(f"compare_experiments.py: error: {message}", file=sys.stderr)
    sys.exit(2)


def _by_name(results: list[dict]) -> dict[str, dict]:
    return {r.get("name", f"#{i}"): r for i, r in enumerate(results)}


def compare(old: dict, new: dict) -> list[str]:
    """The differences between two result files, one line each (empty when the numbers are identical)."""
    diffs: list[str] = []
    a, b = _by_name(old["results"]), _by_name(new["results"])
    for name in a.keys() - b.keys():
        diffs.append(f"{name}: only in the first file")
    for name in b.keys() - a.keys():
        diffs.append(f"{name}: only in the second file")
    for name in (n for n in a if n in b):
        x, y = a[name], b[name]
        for key in ("ok", "fragile", "after_events_per_s", "after_per_seed", "after_graded_per_seed", "silenced",
                    "missing", "seeds", "error"):
            if key in IGNORED:
                continue
            if x.get(key) != y.get(key):
                diffs.append(f"{name}: {key} {x.get(key)!r} -> {y.get(key)!r}")
        rx = {r.get("label", r.get("spec")): r for r in x.get("readouts", [])}
        ry = {r.get("label", r.get("spec")): r for r in y.get("readouts", [])}
        for label in rx.keys() ^ ry.keys():
            diffs.append(f"{name} / {label}: only in the {'first' if label in rx else 'second'} file")
        for label in (k for k in rx if k in ry):
            u, v = rx[label], ry[label]
            for key in ("hz", "per_seed", "ok", "seeds_out", "lo", "hi", "sd"):
                if u.get(key) != v.get(key):
                    diffs.append(f"{name} / {label}: {key} {u.get(key)!r} -> {v.get(key)!r}")
    if old.get("settings") != new.get("settings"):
        diffs.append(f"settings differ: {old.get('settings')!r} -> {new.get('settings')!r}")
    return diffs


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 2:
        print(__doc__.strip(), file=sys.stderr)
        return 2
    old, new = _load(argv[0]), _load(argv[1])
    diffs = compare(old, new)
    n_old, n_new = len(old["results"]), len(new["results"])
    if diffs:
        print(f"{len(diffs)} difference(s) between {Path(argv[0]).name} ({n_old} experiments) and "
              f"{Path(argv[1]).name} ({n_new} experiments):")
        for d in diffs:
            print("  " + d)
        return 1
    print(f"identical: {n_new} experiments, every readout's mean and per-seed rates, ok and fragile flags, and "
          f"after-stimulus activity match (wall time ignored)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
