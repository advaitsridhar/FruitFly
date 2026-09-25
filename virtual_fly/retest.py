"""
Re-testing a fly in the background: the survival report (:func:`virtual_fly.experiments.survival`)
in a separate, low-priority process.

When the game's brain changes (the parts list switched, a fly grown) it re-runs the validated
experiments on a private copy of the new brain. In a thread of the game's own process that re-test
competes with the game loop for the interpreter lock as well as for the CPU (the compiled kernels
release the lock, the Python around them does not), and the game slowed to 0.75 of real time. In a
child process at low priority it shares nothing with the game loop: the game kept its speed and the
re-test ran as fast as on its own.

The child is started with ``spawn`` (the only start method on Windows, and the default on macOS),
so it imports the package afresh, loads the connectome file itself (about a second) and, for a grown
fly, receives the grown wiring (three arrays, about 40 MB). A newer re-test cancels an older one by
terminating its process; a child whose game has gone away exits at its next progress report.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import queue
import signal
import sys
import time


def lower_priority() -> str:
    """Put this process below the game: ``nice 19`` on Linux and macOS, BELOW_NORMAL_PRIORITY_CLASS on
    Windows (through ctypes, no extra package). Returns what was done; never raises."""
    try:
        if sys.platform == "win32":
            import ctypes
            k32 = ctypes.windll.kernel32
            k32.GetCurrentProcess.restype = ctypes.c_void_p
            k32.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
            ok = k32.SetPriorityClass(k32.GetCurrentProcess(), 0x00004000)     # BELOW_NORMAL_PRIORITY_CLASS
            return "below normal" if ok else "priority unchanged"
        return f"nice {os.nice(19)}"
    except Exception as e:                        # a re-test at normal priority is still a re-test
        return f"priority unchanged ({e!r})"


def fingerprint(brain) -> str:
    """A hash of what makes two brains behave the same: the synaptic weights and, with the parts list, every
    neuron's sign and tone signs. The re-test process reports it, so a test can check it built the game's brain."""
    import hashlib
    h = hashlib.sha1(brain._w_original.tobytes())
    p = getattr(brain, "parts", None)
    if p is not None:
        for a in (p.sign, p.mod_sign, p.theta):
            h.update(a.tobytes())
    return h.hexdigest()


def _child(q, spec: dict):
    """The re-test process (module level, so that ``spawn`` can import it)."""
    try:
        signal.signal(signal.SIGINT, signal.SIG_IGN)     # Ctrl+C in the game's console is the game's to handle
    except (ValueError, OSError):
        pass
    t0 = time.time()
    try:
        prio = lower_priority()
        from . import parts, vfb
        from .connectome import load_connectome
        from .experiments import survival
        from .settings import build_brain
        if spec.get("vfb") is not None:                  # the game runs on data swapped in at runtime (tests do)
            vfb.use(*spec["vfb"])
        if spec.get("regions") is not None:
            parts.use_region_table(spec["regions"]["table"])
        conn = load_connectome(spec["path"], quiet=True)
        w = spec.get("wiring")
        if w is not None:
            conn = conn.rewired(w["row_ptr"], w["post_idx"], w["n_syn"], label=w.get("label", "grown"))
        brain = build_brain(conn, spec["profile"], **spec["brain_kwargs"])
        t_ready = time.time() - t0
        parent = mp.parent_process()

        def progress(rows):
            if parent is not None and not parent.is_alive():      # the game has gone: stop quietly
                os._exit(0)
            q.put(("progress", rows))
        rows = survival(brain, profile=spec["profile"], on_progress=progress)
        q.put(("done", rows, {"ready_s": round(t_ready, 2), "total_s": round(time.time() - t0, 2), "priority": prio,
                              "fingerprint": fingerprint(brain)}))
    except BaseException as e:                    # report every failure, never leave the game waiting
        q.put(("error", f"{type(e).__name__}: {e}"))


class Retest:
    """One re-test running in a child process."""

    def __init__(self, spec: dict):
        ctx = mp.get_context("spawn")
        self.q = ctx.Queue()
        self.proc = ctx.Process(target=_child, args=(self.q, spec), daemon=True, name="fly-retest")
        self.proc.start()
        self.cancelled = False
        self.info: dict = {}

    def cancel(self):
        self.cancelled = True
        if self.proc.is_alive():
            self.proc.terminate()

    def wait(self, on_progress=None) -> list[dict]:
        """Block until the child is done, passing progress on. Raises RuntimeError on failure or cancel."""
        try:
            while True:
                if self.cancelled:
                    raise RuntimeError("cancelled")
                try:
                    msg = self.q.get(timeout=0.5)
                except queue.Empty:
                    if not self.proc.is_alive():
                        try:                                  # it may have finished between the two checks
                            msg = self.q.get(timeout=1.0)
                        except queue.Empty:
                            raise RuntimeError(f"the re-test process stopped (exit code {self.proc.exitcode})") from None
                    else:
                        continue
                if msg[0] == "progress":
                    if on_progress is not None:
                        on_progress(msg[1])
                elif msg[0] == "done":
                    self.info = msg[2]
                    return msg[1]
                else:
                    raise RuntimeError(msg[1])
        finally:
            self.proc.join(timeout=2)
            if self.proc.is_alive():
                self.proc.terminate()
            self.q.close()
