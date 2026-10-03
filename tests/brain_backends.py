"""What every integrator of :class:`virtual_fly.brain.FlyBrain` must reproduce spike for spike, and the comparison.

Shared by ``tests/test_fastbrain.py`` (the numba kernels against the NumPy reference, in CI) and
``tests/test_gpubrain.py`` (the CuPy kernels against both, wherever a GPU is): one list of configurations, one run,
one set of assertions, so that an assertion added here is checked for every backend at once.
"""

import numpy as np

from virtual_fly.brain import FlyBrain
from virtual_fly.plasticity import MushroomBodyPlasticity

CONFIGS = {
    "pure": {},
    "fatigue": {"fatigue_mv": 0.05},
    "depression": {"std_u": 0.1, "std_tau_ms": 100.0},
    "jitter": {"threshold_jitter": 0.5},
    "noise": {"noise_hz": 5.0, "noise_mv": 1.0},
    "everything": {"fatigue_mv": 0.05, "std_u": 0.1, "threshold_jitter": 0.5, "noise_hz": 5.0, "kenyon_gain": 1.0},
    "fast": {"dt": 1.0, "fatigue_mv": 0.05},
    "parts": {"parts": True, "fatigue_mv": 0.05},
    "parts_mb": {"parts": True, "fatigue_mv": 0.05, "kenyon_gain": 1.0},     # Kenyon cells fire: APL releases locally
}


def _run(conn, backend, cfg, learning=False, seed=3):
    brain = FlyBrain(conn, seed=seed, backend=backend, **cfg)
    if learning:
        MushroomBodyPlasticity().attach(brain)
    brain.stimulate("LB3b,LB3c", 120)                           # sugar, then bitter on top, then silence
    brain.stimulate("LC4/R,LPLC2/R", 150)
    rec = brain.run(120, record=True)
    brain.stimulate("LB1a,LB1b", 100)
    if learning or cfg.get("parts"):
        brain.stimulate("ORN_DM1", 100)
        brain.stimulate("PPL101", 60)
    if cfg.get("parts"):                                        # graded cells and every modulator in play
        brain.stimulate("Mi1/R,Mi9/R", 80)
        brain.stimulate("T4a/R", 60)
        brain.stimulate("prefix:JO-B", 100)
        if cfg.get("kenyon_gain"):                              # enough glomeruli for the Kenyon cells to fire
            brain.stimulate("ORN_DM4,ORN_VM7d,ORN_DP1m", 120)
    rec += brain.run(120, record=True)
    brain.clear_stimuli()
    rec += brain.run(300, record=True)                          # runs down to rest (quiet path)
    return brain, rec


def assert_backends_agree(conn, cfg, name, a, rec_a, b, rec_b):
    """Everything ``test_fastbrain`` has always asserted between two backends, ``a`` the reference."""
    assert a.total_spikes == b.total_spikes > 100
    assert len(rec_a) == len(rec_b)
    for (ta, sa), (tb, sb) in zip(rec_a, rec_b):
        assert ta == tb and np.array_equal(sa, sb)
    assert np.array_equal(a.spike_count, b.spike_count)
    assert np.array_equal(a.v, b.v) and np.array_equal(a.g, b.g) and np.array_equal(a.thr, b.thr)
    assert a.t == b.t and a.quiet == b.quiet
    if a.std_x is not None:
        assert np.array_equal(a.std_x, b.std_x) and np.array_equal(a.std_t, b.std_t)
    if cfg.get("parts"):
        assert np.array_equal(a._rel, b._rel) and np.array_equal(a._mod_level, b._mod_level) and a._mod_active == b._mod_active
        assert a.parts_status() == b.parts_status() and a.parts_status()["tone"]["octopamine"]["mean"] > 0
        assert a.spike_count[conn.select("T4b/R")].sum() > 0                  # graded events happened
        assert np.array_equal(a.w, b.w)
        assert all(np.array_equal(x, y) for x, y in zip(a._local_drive + a._local_p, b._local_drive + b._local_p))
    if name == "parts_mb":
        assert a.spike_count[conn.select("class:Kenyon_Cell")].sum() > 0 and a.local_status() == b.local_status()
