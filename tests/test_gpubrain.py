"""The CuPy integrator must reproduce the NumPy one (and numba's) spike for spike and bit for bit, on every mechanism.

Skipped without CuPy or a GPU (CI has neither); run it where there is one and paste the result into the PR."""

import hashlib
import math

import numpy as np
import pytest

from brain_backends import CONFIGS, _run, assert_backends_agree
from virtual_fly import fastbrain, gpubrain
from virtual_fly.brain import FlyBrain
from virtual_fly.plasticity import MushroomBodyPlasticity

pytestmark = pytest.mark.skipif(not gpubrain.available(), reason=f"CuPy or a GPU is missing: {gpubrain.unavailable_reason()}")

STATE = ("v", "g", "thr", "spike_count", "std_x", "std_t", "_rel", "_mod_level", "_mod_gain", "queue")


def _bits(brain, names=STATE):
    """The bytes of the state arrays (so -0.0 and +0.0 count as different, which np.array_equal hides)."""
    h = hashlib.sha1()
    for name in names:
        x = getattr(brain, name)
        if x is not None:
            h.update(np.ascontiguousarray(x).tobytes())
    return h.hexdigest()


def _raster(steps):
    h = hashlib.sha1()
    for t, s in steps:
        h.update(np.int64(t).tobytes())
        h.update(np.asarray(s, dtype=np.int64).tobytes())
    return h.hexdigest()


def _assert_bits_equal(a, b):
    for name in STATE:
        x, y = getattr(a, name), getattr(b, name)
        if x is None or y is None:
            assert x is None and y is None, name
            continue
        assert x.shape == y.shape and np.ascontiguousarray(x).tobytes() == np.ascontiguousarray(y).tobytes(), name


@pytest.mark.parametrize("name", sorted(CONFIGS))
def test_cupy_matches_numpy_spike_for_spike(conn, name):
    cfg = CONFIGS[name]
    a, rec_a = _run(conn, "numpy", cfg)
    b, rec_b = _run(conn, "cupy", cfg)
    assert a.backend == "numpy" and b.backend == "cupy" and isinstance(b, gpubrain.GpuFlyBrain)
    assert_backends_agree(conn, cfg, name, a, rec_a, b, rec_b)
    _assert_bits_equal(a, b)
    assert np.array_equal(a._pending, b._pending) and len(a._recent) == len(b._recent)
    assert all(np.array_equal(x, y) for x, y in zip(a._recent, b._recent))


@pytest.mark.skipif(not fastbrain.available(), reason="numba is not installed")
@pytest.mark.parametrize("name", sorted(CONFIGS))
def test_cupy_matches_numba_spike_for_spike(conn, name):
    cfg = CONFIGS[name]
    a, rec_a = _run(conn, "numba", cfg)
    b, rec_b = _run(conn, "cupy", cfg)
    assert_backends_agree(conn, cfg, name, a, rec_a, b, rec_b)
    _assert_bits_equal(a, b)


def test_cupy_matches_numpy_with_the_curated_parts_list(conn, mini_vfb):
    from virtual_fly.parts import PartsList
    cfg = {"parts": PartsList(curated="all"), "fatigue_mv": 0.05}
    a, rec_a = _run(conn, "numpy", cfg)
    b, rec_b = _run(conn, "cupy", cfg)
    assert a.total_spikes == b.total_spikes > 100 and len(rec_a) == len(rec_b)
    for (ta, sa), (tb, sb) in zip(rec_a, rec_b):
        assert ta == tb and np.array_equal(sa, sb)
    assert np.array_equal(a._mod_gain, b._mod_gain) and np.array_equal(a._mod_level, b._mod_level)
    assert a.parts.counts["curated"]["neurons"] > 0 and (a.parts.mod_sign < 0).any()
    _assert_bits_equal(a, b)


def test_cupy_matches_numpy_with_learning(conn):
    a, rec_a = _run(conn, "numpy", {"fatigue_mv": 0.05, "kenyon_gain": 1.0}, learning=True)
    b, rec_b = _run(conn, "cupy", {"fatigue_mv": 0.05, "kenyon_gain": 1.0}, learning=True)
    assert a.total_spikes == b.total_spikes
    assert all(ta == tb and np.array_equal(sa, sb) for (ta, sa), (tb, sb) in zip(rec_a, rec_b))
    assert np.array_equal(a.w, b.w) and a.plasticity.depressed_fraction() == b.plasticity.depressed_fraction()
    assert a.plasticity.events == b.plasticity.events


def test_cupy_backend_selection_and_settings(conn):
    b = FlyBrain(conn, backend="cupy")
    assert b.backend == "cupy" and b.settings()["backend"] == "cupy"
    assert FlyBrain(conn).backend != "cupy"                     # auto never picks the GPU
    with pytest.raises(ValueError):
        FlyBrain(conn, backend="cuda")


def test_cupy_silence_modulate_and_snapshot(conn):
    b = FlyBrain(conn, backend="cupy", fatigue_mv=0.05)
    b.stimulate("LB3b,LB3c", 120)
    b.run(100)
    assert b.rate("MN9") > 20
    b.silence("GNG232")
    b.reset_counts()
    b.run(100)
    assert b.rate("MN9") == 0
    b.unsilence()
    b.modulate("GNG232", 0.0)
    b.reset_counts()
    b.run(100)
    assert b.rate("MN9") == 0
    b.unmodulate()
    snap = b.snapshot()
    b.start_recording()
    b.run(80)
    first = b.stop_recording()
    end_first = _bits(b)
    b.restore(snap)
    b.start_recording()
    b.run(80)
    second = b.stop_recording()
    assert len(first) == len(second) > 0
    assert all(ta == tb and np.array_equal(sa, sb) for (ta, sa), (tb, sb) in zip(first, second))
    assert _bits(b) == end_first


def test_a_snapshot_mid_run_is_the_cpus_field_by_field(conn):
    """The CPU representation (the queue, the pending flags, the reset lists, the generator) comes back the same."""
    cfg = CONFIGS["everything"]
    a = FlyBrain(conn, seed=3, backend="numpy", **cfg)
    b = FlyBrain(conn, seed=3, backend="cupy", **cfg)
    for br in (a, b):
        br.stimulate("LB3b,LB3c", 120)
        br.run(63.5)                                             # 127 steps: not a chunk boundary
    sa, sb = a.snapshot(), b.snapshot()
    for key in ("t", "v", "g", "queue", "pending", "thr", "std_x", "std_t", "spike_count", "rel", "mod_level", "mod_gain",
                "mod_active", "window_ms", "total_spikes", "quiet", "quiet_since"):
        x, y = sa[key], sb[key]
        if isinstance(x, np.ndarray):
            assert x.shape == y.shape and x.tobytes() == y.tobytes(), key
        else:
            assert x == y, key
    assert len(sa["recent"]) == len(sb["recent"]) and all(np.array_equal(x, y) for x, y in zip(sa["recent"], sb["recent"]))
    assert sa["rng"] == sb["rng"]
    # and the GPU brain restored from the CPU's snapshot goes on exactly as the CPU does
    b.restore(sa)
    ra, rb = a.advance_steps(97), b.advance_steps(97)
    assert len(ra) == len(rb) and all(ta == tb and np.array_equal(x, y) for (ta, x), (tb, y) in zip(ra, rb))
    _assert_bits_equal(a, b)


def test_cupy_kernels_handle_refractory_forced_and_empty_steps(conn):
    b = FlyBrain(conn, backend="cupy")
    launches = b._gpu.launches
    assert b.step().size == 0 and b.quiet and b._gpu.launches == launches     # the quiet path never launches
    b.stimulate("LB3b", 2000)                                   # p = 1: forced every step, even while refractory
    spikes = [b.step() for _ in range(6)]
    idx = conn.select("LB3b")
    assert all(np.array_equal(s, idx) for s in spikes)
    assert b.spike_count[idx].min() == 6
    b.clear_stimuli()
    b.stimulate("LB3b,LB3c", 120)
    for _ in range(200):
        s = b.step()
        assert s.dtype == np.int64 and np.array_equal(s, np.unique(s))


@pytest.mark.parametrize("name", ["pure", "everything", "parts_mb"])
def test_advance_steps_equals_stepping_on_the_gpu_and_the_cpu(conn, name):
    """Chunks (with the graph) give what single steps give, from any step, across blocks and the quiet check."""
    cfg = CONFIGS[name]
    plan = [(3, True), (7, False), (10, False), (23, True), (50, False), (230, False), (4, True), (200, False)]
    ref = FlyBrain(conn, seed=5, backend="numpy", **cfg)
    stepped = FlyBrain(conn, seed=5, backend="cupy", **cfg)
    chunked = FlyBrain(conn, seed=5, backend="cupy", **cfg)
    brains = (ref, stepped, chunked)
    stims = [("LB3b,LB3c", 120), ("LC4/R,LPLC2/R", 150), ("LB1a,LB1b", 100), ("prefix:JO-B", 90)]
    for i, (n, change) in enumerate(plan):
        if change:
            spec, hz = stims[i % len(stims)]
            for br in brains:
                br.stimulate(spec, hz)
        if i == 5:
            for br in brains:
                br.clear_stimuli()                               # the long stretch runs down to rest
        for _ in range(n):
            ref.step()
        out_step = []
        for _ in range(n):
            t = stepped.t
            s = stepped.step()
            if s.size:
                out_step.append((t, s))
        out_chunk = chunked.advance_steps(n)
        assert len(out_step) == len(out_chunk)
        assert all(ta == tb and np.array_equal(sa, sb) for (ta, sa), (tb, sb) in zip(out_step, out_chunk))
        _assert_bits_equal(stepped, chunked)
        _assert_bits_equal(ref, chunked)
        assert ref.t == stepped.t == chunked.t and ref.quiet == stepped.quiet == chunked.quiet
    assert chunked.t == sum(n for n, _ in plan)                  # (a stimulus is on at the end: the brain is awake)


@pytest.mark.parametrize("name", ["everything", "parts_mb"])
def test_ten_repeated_runs_give_identical_rasters_and_state(conn, name):
    cfg = CONFIGS[name]
    rasters, states = set(), set()
    for _ in range(5):                                           # fresh brains
        b, rec = _run(conn, "cupy", cfg)
        rasters.add(_raster(rec)); states.add(_bits(b))
    b = FlyBrain(conn, seed=3, backend="cupy", **cfg)
    for _ in range(5):                                           # the same brain, reset between runs
        b.rng = np.random.default_rng(3)
        b.reset()
        b.stimulate("LB3b,LB3c", 120)
        b.stimulate("LC4/R,LPLC2/R", 150)
        rec = b.run(120, record=True)
        b.stimulate("LB1a,LB1b", 100)
        if cfg.get("parts"):
            b.stimulate("ORN_DM1", 100); b.stimulate("PPL101", 60); b.stimulate("Mi1/R,Mi9/R", 80); b.stimulate("T4a/R", 60)
            b.stimulate("prefix:JO-B", 100)
            if cfg.get("kenyon_gain"):
                b.stimulate("ORN_DM4,ORN_VM7d,ORN_DP1m", 120)
        rec += b.run(120, record=True)
        b.clear_stimuli()
        rec += b.run(300, record=True)
        rasters.add(_raster(rec)); states.add(_bits(b))
    assert len(rasters) == 1 and len(states) == 1
    a, rec_a = _run(conn, "numpy", cfg)
    assert _raster(rec_a) in rasters and _bits(a) in states


def test_the_depression_table_covers_every_wait_and_matches_both_cpu_paths(conn):
    """Beyond its last entry the recovery term vanishes in double for every float32 x, so the entry stands in for all
    later steps; and ``math.exp`` (numba's) equals ``np.exp`` (NumPy's) on every argument the table holds."""
    for cfg in (CONFIGS["depression"], {"std_u": 0.1}):
        b = FlyBrain(conn, backend="cupy", **cfg)
        eng = b._gpu
        table = eng.std_table
        assert eng.std_table_mismatch is None
        assert table[-1] <= 2.0 ** -54 and table[-2] > 2.0 ** -54
        assert table[0] == 1.0 and np.all(np.diff(table) < 0)
        rec = table[-1]
        xs = np.linspace(0.0, 1.0, 1 << 20, dtype=np.float32)   # a dense sample of the float32 resource values
        d = (np.float32(1.0) - xs).astype(np.float64)
        assert np.all(1.0 - d * rec == 1.0) and 1.0 - 1.0 * rec == 1.0   # the largest difference, 1.0, is the bound
        k = np.arange(table.size, dtype=np.float64)
        assert np.array_equal(table, np.exp(-(k * b.dt) / b.std_tau_ms))
        assert all(table[i] == math.exp(-((i * b.dt) / b.std_tau_ms)) for i in range(0, table.size, 97))


def test_the_gpu_brain_refuses_a_wiring_with_unordered_targets(conn):
    """The ordered pull needs every row's targets ascending with one connection per pair: anything else is refused."""
    post = np.asarray(conn.post_idx).copy()
    i = int(np.flatnonzero(np.diff(conn.row_ptr) >= 2)[0])
    a = int(conn.row_ptr[i])
    post[a], post[a + 1] = post[a + 1], post[a]
    bad = conn.rewired(conn.row_ptr, post, conn.n_syn, label="swapped")
    with pytest.raises(ValueError, match="ascending order"):
        FlyBrain(bad, backend="cupy")


def test_host_copies_are_read_only_and_assignment_uploads(conn):
    b = FlyBrain(conn, backend="cupy", fatigue_mv=0.05)
    b.stimulate("LB3b,LB3c", 120)
    b.run(20)
    v = b.v
    with pytest.raises(ValueError):
        v[0] = 1.0                                               # a host write the device would never see
    b.thr = b.thr + np.float32(1.0)                              # assigning uploads: the brain fires less
    before = b.total_spikes
    b.run(20)
    assert b.total_spikes > before
    assert b.thr.min() > np.float32(7.5)                      # the uploaded thresholds are what the device runs with


@pytest.mark.parametrize("bin_ms", [3.0, 10.0, 25.0])
def test_monitors_recording_and_callbacks_match_the_cpu_across_chunks(conn, bin_ms):
    """A monitor whose bin closes inside a chunk (3 ms), at a chunk end (10 ms) or at a tick end (25 ms), the spike
    recording and the callbacks all see the per-step numbers the CPU gives, through chunked advance_steps."""
    a = FlyBrain(conn, seed=4, backend="numpy", fatigue_mv=0.05)
    b = FlyBrain(conn, seed=4, backend="cupy", fatigue_mv=0.05)
    seen = {"a": [], "b": []}
    for key, br in (("a", a), ("b", b)):
        br.add_monitor("mn9", "MN9", bin_ms=bin_ms)
        br.add_monitor("gng", "GNG232", bin_ms=bin_ms)
        br.stimulate("LB3b,LB3c", 120)
        br.start_recording()
        br.on_spikes.append(lambda s, t, key=key: seen[key].append((t, s.copy())))
    a.advance_steps(50); a.advance_steps(23); a.advance_steps(50)
    b.advance_steps(50); b.advance_steps(23); b.advance_steps(50)
    assert a.t == b.t == 123
    for name in ("mn9", "gng"):
        assert a.monitors[name].history == b.monitors[name].history and len(a.monitors[name].history) > 0
        assert a.monitors[name]._count == b.monitors[name]._count and a.monitors[name]._t_start == b.monitors[name]._t_start
    ra, rb = a.stop_recording(), b.stop_recording()
    assert len(ra) == len(rb) > 0 and all(x == y and np.array_equal(sa, sb) for (x, sa), (y, sb) in zip(ra, rb))
    assert len(seen["a"]) == len(seen["b"]) > 0 and all(x == y and np.array_equal(sa, sb) for (x, sa), (y, sb) in zip(seen["a"], seen["b"]))
    assert np.array_equal(a.spike_count, b.spike_count) and a.total_spikes == b.total_spikes
    _assert_bits_equal(a, b)
