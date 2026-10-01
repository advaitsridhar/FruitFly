"""
A GPU backend for :class:`virtual_fly.brain.FlyBrain` (optional; needs ``cupy`` and an NVIDIA GPU).

``FlyBrain(conn, backend="cupy")`` builds :class:`GpuFlyBrain`, a subclass whose step runs as hand-written CUDA
kernels compiled at run time by CuPy (NVRTC, no fused multiply-add), bit-identical to the NumPy and numba steps:
the same float32 operations in the same order, every random draw still made on the host by the brain's own
generator, and every target's synaptic kicks added in ascending presynaptic order from the queue slot's value
(docs/TWO_FLIES_PLAN.md section 6). ``available()`` says whether CuPy and a GPU are there; ``unavailable_reason()``
says in one line what is missing. Nothing here is imported unless the backend is asked for, so a machine without
CUDA never pays for it.
"""
from __future__ import annotations

from .brain import FlyBrain


def unavailable_reason() -> str | None:
    """None when CuPy imports and sees an NVIDIA GPU; otherwise one line naming what failed."""
    try:
        import cupy as cp
    except Exception as e:                      # pragma: no cover - depends on the machine
        return f"the cupy backend needs the cupy package (pip install cupy-cuda13x, or cupy-cuda12x for an older driver): {type(e).__name__}: {e}"
    try:
        n = cp.cuda.runtime.getDeviceCount()
    except Exception as e:                      # pragma: no cover - depends on the machine
        return f"the cupy backend needs an NVIDIA GPU and its driver: {type(e).__name__}: {e}"
    if n < 1:                                   # pragma: no cover
        return "the cupy backend needs an NVIDIA GPU: CUDA reports no device"
    return None


def available() -> bool:
    return unavailable_reason() is None


class GpuFlyBrain(FlyBrain):
    """The brain with its dynamical state on the GPU (built by ``FlyBrain(conn, backend="cupy")``)."""

    def __init__(self, *args, **kwargs):
        reason = unavailable_reason()
        if reason is not None:
            raise RuntimeError(reason)
        super().__init__(*args, **kwargs)

    def _step_gpu(self, slot: int):
        raise NotImplementedError("the cupy backend's kernels are being built (docs/TWO_FLIES_PLAN.md 6.4)")
