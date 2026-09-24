"""
Shared fixtures. The whole suite runs on a small synthetic connectome (see ``synthetic_connectome.py``)
written once per session into a temporary directory, so no download and no 23 MB file is needed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))          # for `import synthetic_connectome`

from synthetic_connectome import build_synthetic  # noqa: E402


@pytest.fixture(scope="session")
def synthetic_path(tmp_path_factory) -> Path:
    """Path of the gzipped synthetic FLYB file."""
    return build_synthetic(tmp_path_factory.mktemp("data") / "synthetic.flyb.gz")


@pytest.fixture(scope="session")
def conn(synthetic_path):
    """The synthetic connectome, loaded once (it is never mutated by the code under test)."""
    from virtual_fly.connectome import Connectome
    return Connectome(synthetic_path)


@pytest.fixture
def brain(conn):
    """A fresh 'pure' FlyBrain (the published model, nothing added)."""
    from virtual_fly.brain import FlyBrain
    return FlyBrain(conn, seed=0)


@pytest.fixture
def game_brain(conn):
    """A fresh brain built with the 'game' profile (fatigue, ALLN/DAN silenced, plasticity on)."""
    from virtual_fly.settings import build_brain
    return build_brain(conn, "game", seed=0)


def drive(brain, stimulus: dict[str, float], ms: float = 500.0, settle_ms: float = 100.0):
    """Reset, stimulate, settle, then measure for the remaining time (mirrors experiments.run_experiment)."""
    brain.reset()
    brain.clear_stimuli()
    for spec, hz in stimulus.items():
        brain.stimulate(spec, hz)
    brain.run(settle_ms)
    brain.reset_counts()
    brain.run(ms - settle_ms)
    return brain
