"""Golden single-fly hashes: what one fly does, byte for byte, pinned before the two-flies work.

Each configuration in ``tools/golden_hashes.py::CONFIGS`` builds ``Game(build_brain(conn, "game", seed=0), seed=0, ...)``
on the synthetic connectome, sends its actions, runs 400 ticks (80 with the physics body, which is skipped without
flygym) and takes one SHA-1 over every ``state_json`` frame in order. The hashes live in ``golden_single_fly.json``
next to the Python, NumPy and numba versions they were made with. A refactor that keeps single-fly behaviour must keep
every hash (docs/TWO_FLIES_PLAN.md, 4.9); ``tools/golden_hashes.py`` does the same on the real male and female data.

Regenerate the JSON only on purpose, before a code change, or for a behaviour change that is intended, documented
and approved by the owner:

    VF_UPDATE_GOLDEN=1 python -m pytest tests/test_golden_single_fly.py
"""

import json
import os
from pathlib import Path

import pytest

from tools.golden_hashes import CONFIGS, needs_physics, run_hash, versions

GOLDEN = Path(__file__).with_name("golden_single_fly.json")
UPDATE = os.environ.get("VF_UPDATE_GOLDEN") == "1"


def _golden() -> dict:
    if not GOLDEN.exists():
        return {"versions": {}, "hashes": {}}
    return json.loads(GOLDEN.read_text())


@pytest.mark.parametrize("name", list(CONFIGS))
def test_a_single_fly_gives_its_golden_hash(conn, name):
    if needs_physics(name):
        from virtual_fly import physics
        if not physics.available():
            pytest.skip(f"physics body unavailable (optional): {physics.unavailable_reason()}")
    h = run_hash(conn, name)
    assert len(h) == 40
    if UPDATE:
        data = _golden()
        data["versions"] = versions()
        data["hashes"][name] = h
        GOLDEN.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n")
        return
    data = _golden()
    assert name in data["hashes"], f"no golden hash for {name!r}: run once with VF_UPDATE_GOLDEN=1 before any change"
    assert h == data["hashes"][name], (
        f"the single fly's {name!r} run changed (single-fly behaviour must stay identical when the new features are "
        f"off; regenerate only for an intended, documented change). Golden versions {data['versions']}, "
        f"these versions {versions()}")


def test_the_golden_file_names_every_configuration_and_its_versions():
    if UPDATE:
        pytest.skip("regenerating")
    data = _golden()
    assert set(data["hashes"]) == set(CONFIGS), "golden_single_fly.json and CONFIGS disagree"
    assert set(data["versions"]) >= {"python", "numpy", "numba"}


def test_two_runs_of_one_configuration_agree(conn):
    """The hashes are only a safety net if a run is reproducible: the same configuration twice, the same bytes."""
    assert run_hash(conn, "autopilot", ticks=60) == run_hash(conn, "autopilot", ticks=60)
