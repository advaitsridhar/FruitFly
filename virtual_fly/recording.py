"""
Record-and-replay for the dish (the two-flies work, docs/TWO_FLIES_PLAN.md 8.7-8.8): a recording is one folder,
``recordings/<YYYYmmdd-HHMMSS>/`` in a checkout (the repository ignores ``recordings/``) or ``<data folder>/recordings/``
in an installed copy, never inside site-packages. It holds what the page needs to play a run back smoothly at any speed,
however slowly the run itself went (a physics pair runs at about a tenth of real time), and what an offline render needs
(``tools/render_replay.py``):

    header.json        format 1; the kit's version and git commit; tick_ms; every fly (id, sex, dataset, body kind, brain
                       settings); the social channels; the physics settings; the model file; at the stop: ticks, seconds, bytes
    frames.jsonl.gz    one line per tick: {seq, t, flies: [{id, sex, fly, mode, hz, senses, sps, graded_eps, driver}], world}
                       (fly: the body's to_dict, the state's "fly"; world: the dish's food, posts, odours, wind, stripes, tool,
                       the pointer and the scripted female)
    qpos.npy, t.npy    physics runs: MuJoCo's qpos after every tick (float64, ticks x nq) and the game's clock
    model/             physics runs: the MuJoCo model as compiled, with its assets (dm_control's export_with_assets)
    poses.f32          physics runs, written at the stop: per tick, per fly, per geom, position and quaternion in that fly's
                       thorax frame (float32 little-endian, ticks x flies x geoms x 7: x y z qw qx qy qz, mm), from qpos through
                       mj_kinematics on the exported model, so the browser plays the legs with no kinematics of its own;
                       header.json's "poses" gives the geom order (models/nmf_gait.json's, which the 3-D view already uses)

A drawn-body run has no qpos, model or poses. A size cap on the whole recordings folder (SIZE_CAP, 2 GB): when it is
reached, a recording refuses to start, and one in progress stops; nothing is deleted for you. The existing in-memory
Record button (frames kept in the game, ``GET /api/recording``) is untouched.

The recording action is the one API call that writes to disk, so the server takes it only from the page's own origin
(server.py: a request with a foreign Origin header, or a Host that is not localhost or 127.0.0.1, is refused) and the
replay endpoints carry no wildcard CORS header. Every ``.npy`` is loaded with allow_pickle=False.
"""
from __future__ import annotations

import gzip
import json
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np

from . import __version__

FORMAT = 1
SIZE_CAP = 2 * 1024 ** 3                   # bytes, the whole recordings folder
ID_RE = re.compile(r"[0-9]{8}-[0-9]{6}(-[0-9]{1,3})?")
WORLD_KEYS = ("food", "obstacles", "odours", "wind", "stripes", "tool", "hand", "female")
POSES_LAYOUT = ("float32 little-endian (ticks, flies, geoms, 7): x, y, z in mm in that fly's thorax frame, then the quaternion "
                "qw, qx, qy, qz of the geom in the thorax frame; the geom order is this header's poses.geoms")


def recordings_dir() -> Path:
    """Where recordings go: the checkout's ``recordings/`` folder, or the data folder's in an installed copy."""
    from . import connectome
    return (connectome.DATA_DIR if connectome.INSTALLED else connectome.PROJECT_DIR) / "recordings"


def folder_bytes(folder: Path) -> int:
    total = 0
    if folder.is_dir():
        for p in folder.rglob("*"):
            try:
                if p.is_file():
                    total += p.stat().st_size
            except OSError:
                pass
    return total


def _writable(folder: Path) -> str | None:
    """None when the folder exists (made here if need be) and a file can be written in it, else why not (one line, as
    connectome.data_folder says it, but returned rather than raised: this runs in the server's threads)."""
    try:
        folder.mkdir(parents=True, exist_ok=True)
        tempfile.TemporaryFile(dir=folder).close()
    except OSError as e:
        why = ("a file of that name is in the way" if isinstance(e, FileExistsError) or (folder.exists() and not folder.is_dir())
               else "part of that path is a file" if isinstance(e, NotADirectoryError) else e.strerror or str(e))
        return f"can't write to the recordings folder {folder} ({why})"
    return None


def cannot_start(game) -> str | None:
    """Why a recording cannot start now, or None: one at a time, a writable folder, the size cap."""
    if getattr(game, "capture", None) is not None:
        return "a replay is being saved already; stop it first"
    folder = recordings_dir()
    why = _writable(folder)
    if why:
        return why
    used = folder_bytes(folder)
    if used >= SIZE_CAP:
        return (f"the recordings folder {folder} holds {used / 1024 ** 3:.2f} GB, at the cap of {SIZE_CAP / 1024 ** 3:.0f} GB: "
                "move or delete old recordings first (nothing is deleted for you)")
    return None


def folder_of(rid) -> Path | None:
    """The folder of a recording id, or None when the id is not a plain recording name or no such recording exists."""
    if not isinstance(rid, str) or not ID_RE.fullmatch(rid):
        return None
    folder = recordings_dir() / rid
    return folder if (folder / "header.json").is_file() else None


def read_header(folder: Path) -> dict:
    return json.loads((folder / "header.json").read_text())


def list_recordings() -> list[dict]:
    """Every recording, newest first: what the page's list shows."""
    out = []
    base = recordings_dir()
    if not base.is_dir():
        return out
    for folder in sorted(base.iterdir(), reverse=True):
        if not folder.is_dir() or not ID_RE.fullmatch(folder.name) or not (folder / "header.json").is_file():
            continue
        try:
            h = read_header(folder)
        except (OSError, ValueError):
            continue
        out.append({"id": folder.name, "started": h.get("started"), "active": bool(h.get("active")),
                    "ticks": h.get("ticks"), "seconds": h.get("seconds"), "bytes": h.get("bytes", folder_bytes(folder)),
                    "flies": [{"id": f.get("id"), "sex": f.get("sex"), "body": f.get("body")} for f in h.get("flies", [])],
                    "physics": h.get("physics") is not None, "poses": (folder / "poses.f32").is_file()})
    return out


def _git_commit() -> str | None:
    from . import connectome
    if connectome.INSTALLED:
        return None
    try:
        r = subprocess.run(["git", "-C", str(connectome.PROJECT_DIR), "rev-parse", "--short", "HEAD"],
                           capture_output=True, text=True, timeout=3)
        return r.stdout.strip() or None if r.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def _physics_source(game):
    """(model ptr, data ptr, the MJCF root, the flies' model names, the model file name, settings) of the game's MuJoCo
    world, or None for drawn bodies."""
    pw = getattr(game, "pair_world", None)
    if pw is not None:
        return (pw._m, pw._d, pw.sim.arena.root_element, [f.name for f in pw.flies], "two_flies.xml",
                {"timestep": pw.timestep, "contact_set": pw.contact_set, "pairs": len(pw.pairs), "hulls": pw.hulls,
                 "native_ccd": pw.native_ccd, "pair_solref": pw.pair_solref, "pair_solimp": pw.pair_solimp, "wall": pw.wall})
    walker = getattr(game.flies[0].body, "walker", None)
    if walker is not None:
        return (walker._m, walker._d, walker.sim.arena.root_element, [walker.fly.name], "one_fly.xml",
                {"timestep": walker.timestep, "stride_average": bool(getattr(game.flies[0].body, "stride_average", False))})
    return None


def atlas_geoms() -> list[str]:
    """The geom order the 3-D view uses (models/nmf_gait.json): poses.f32 follows it."""
    from . import connectome
    path = Path(connectome.PACKAGE_DIR) / "web" / "models" / "nmf_gait.json"
    return list(json.loads(path.read_text())["geoms"])


class Capture:
    """One recording in progress. ``start`` opens the folder and the files (and exports the MuJoCo model of a physics run);
    ``add`` is called by the game after every tick; ``stop`` closes the files, converts qpos, writes poses.f32 and the final
    header. The game owns it (``game.capture``)."""

    def __init__(self, game):
        self.game = game
        self.id = self.folder = None
        self.ticks = 0
        self.bytes = 0
        self.physics = None
        self._frames = self._qpos = None
        self._data = None
        self._stopped = False
        self.error = None

    @classmethod
    def start(cls, game) -> "Capture":
        c = cls(game)
        c._open()
        return c

    def _open(self):
        from .game import TICK_MS
        base = recordings_dir()
        base.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        folder, i = base / stamp, 1
        while folder.exists():
            i += 1
            folder = base / f"{stamp}-{i}"
        folder.mkdir()
        self.id, self.folder = folder.name, folder
        game = self.game
        src = _physics_source(game)
        header = {"format": FORMAT, "kit": __version__, "git": _git_commit(), "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
                  "tick_ms": TICK_MS, "seed": game.seed, "profile": game.profile_name,
                  "flies": [{"id": f.id, "sex": f.sex, "dataset": getattr(f.conn, "dataset", None), "body": f.body_kind,
                             "settings": f.io.settings()} for f in game.flies],
                  "social": game.social.names() if hasattr(game.social, "names") else None,
                  "physics": None, "poses": None, "active": True}
        if src is not None:
            m, d, root, names, model_file, settings = src
            from dm_control import mjcf
            mjcf.export_with_assets(root, str(folder / "model"), out_file_name=model_file)
            header["physics"] = {**settings, "model": model_file, "fly_names": names, "nq": int(m.nq)}
            header["poses"] = {"geoms": atlas_geoms(), "fly_names": names, "layout": POSES_LAYOUT}
            self.physics = (m, d, names, model_file)
            self._data = d
            self._qpos = open(folder / "qpos.f64", "wb")
            self._t = []
        self.header = header
        (folder / "header.json").write_text(json.dumps(header, indent=1))
        self._frames = gzip.open(folder / "frames.jsonl.gz", "wt", compresslevel=3)

    def add(self, game):
        """After a tick: one frame line, and the qpos row of a physics run."""
        if self._stopped:
            return
        flies = []
        for f in game.flies:
            bt = getattr(f, "bt", None)
            flies.append({"id": f.id, "sex": f.sex, "fly": f.body.to_dict(), "mode": f.mode,
                          "hz": {k: round(v, 1) for k, v in bt.hz.items() if f._has(k)} if bt is not None else {},
                          "senses": f.senses_now, "sps": int(f.sps), "graded_eps": int(f.graded_eps), "driver": f.driver})
        w = game.world.to_dict()
        frame = {"seq": game.seq + 1, "t": round(game.t, 3), "flies": flies, "world": {k: w.get(k) for k in WORLD_KEYS}}
        self._frames.write(json.dumps(frame, separators=(",", ":")) + "\n")
        if self._qpos is not None:
            src = _physics_source(game)
            if src is None or src[1] is not self._data:        # the world was rebuilt (a new single fly): the replay stops here
                self.error = "the physics world was rebuilt (a new fly or a move): the replay stops here"
                self.stop()
                return
            self._qpos.write(np.ascontiguousarray(self._data.qpos, dtype=np.float64).tobytes())
            self._t.append(game.t)
        self.ticks += 1
        if self.ticks % 400 == 0:
            self._frames.flush()
            if self._qpos is not None:
                self._qpos.flush()
            self.bytes = folder_bytes(recordings_dir())
            if self.bytes >= SIZE_CAP:
                self.error = f"the recordings folder reached its cap of {SIZE_CAP / 1024 ** 3:.0f} GB: the replay stops here"
                self.stop()

    def status(self) -> dict:
        return {"id": self.id, "ticks": self.ticks, "physics": self.physics is not None, "active": not self._stopped,
                "error": self.error}

    def stop(self):
        """Close the files, convert qpos, write poses.f32 and the final header. Safe to call twice."""
        if self._stopped:
            return
        self._stopped = True
        try:
            self._frames.close()
        except Exception:
            pass
        if self._qpos is not None:
            self._qpos.close()
            raw = self.folder / "qpos.f64"
            m = self.physics[0]
            q = np.fromfile(raw, dtype=np.float64)
            q = q.reshape(-1, int(m.nq)) if int(m.nq) else q.reshape(0, 0)
            np.save(self.folder / "qpos.npy", q, allow_pickle=False)
            np.save(self.folder / "t.npy", np.asarray(self._t, dtype=np.float64), allow_pickle=False)
            raw.unlink()
            try:
                write_poses(self.folder, self.header)
            except Exception as e:                       # the frames and qpos are kept; the browser then animates the legs
                self.header["poses_error"] = repr(e)
        self.header.update(active=False, ticks=self.ticks, seconds=round(self.ticks * self.header["tick_ms"] / 1000.0, 3),
                           stopped=time.strftime("%Y-%m-%dT%H:%M:%S"), error=self.error)
        self.header["bytes"] = folder_bytes(self.folder)
        (self.folder / "header.json").write_text(json.dumps(self.header, indent=1))


# ---------------------------------------------------------------------------------------------- poses from qpos
def _mat2quat(R: np.ndarray) -> np.ndarray:
    """(N, 3, 3) rotation matrices -> (N, 4) unit quaternions (w, x, y, z), vectorised (Shepperd's method)."""
    m00, m01, m02 = R[:, 0, 0], R[:, 0, 1], R[:, 0, 2]
    m10, m11, m12 = R[:, 1, 0], R[:, 1, 1], R[:, 1, 2]
    m20, m21, m22 = R[:, 2, 0], R[:, 2, 1], R[:, 2, 2]
    tr = m00 + m11 + m22
    q = np.empty((R.shape[0], 4))
    c0 = tr > 0
    c1 = ~c0 & (m00 >= m11) & (m00 >= m22)
    c2 = ~c0 & ~c1 & (m11 >= m22)
    c3 = ~c0 & ~c1 & ~c2
    s = np.sqrt(np.maximum(tr[c0] + 1.0, 1e-12)) * 2
    q[c0] = np.stack([0.25 * s, (m21[c0] - m12[c0]) / s, (m02[c0] - m20[c0]) / s, (m10[c0] - m01[c0]) / s], 1)
    s = np.sqrt(np.maximum(1.0 + m00[c1] - m11[c1] - m22[c1], 1e-12)) * 2
    q[c1] = np.stack([(m21[c1] - m12[c1]) / s, 0.25 * s, (m01[c1] + m10[c1]) / s, (m02[c1] + m20[c1]) / s], 1)
    s = np.sqrt(np.maximum(1.0 + m11[c2] - m00[c2] - m22[c2], 1e-12)) * 2
    q[c2] = np.stack([(m02[c2] - m20[c2]) / s, (m01[c2] + m10[c2]) / s, 0.25 * s, (m12[c2] + m21[c2]) / s], 1)
    s = np.sqrt(np.maximum(1.0 + m22[c3] - m00[c3] - m11[c3], 1e-12)) * 2
    q[c3] = np.stack([(m10[c3] - m01[c3]) / s, (m02[c3] + m20[c3]) / s, (m12[c3] + m21[c3]) / s, 0.25 * s], 1)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    return q


def load_model(folder: Path, header: dict | None = None):
    """The recording's exported MuJoCo model (physics runs)."""
    import mujoco
    header = header or read_header(folder)
    return mujoco.MjModel.from_xml_path(str(folder / "model" / header["physics"]["model"]))


def geom_poses(m, d, fly_names: list[str], geoms: list[str]) -> np.ndarray:
    """Every fly's geoms in its thorax frame at the model's current kinematics: (flies, geoms, 7)."""
    out = np.empty((len(fly_names), len(geoms), 7), dtype=np.float32)
    for k, fly in enumerate(fly_names):
        gids = np.array([m.geom(f"{fly}/{n}").id for n in geoms])
        t = m.body(f"{fly}/Thorax").id
        Rt = d.xmat[t].reshape(3, 3)
        out[k, :, :3] = (d.geom_xpos[gids] - d.xpos[t]) @ Rt           # R_t^T (p - p_t)
        Rg = d.geom_xmat[gids].reshape(-1, 3, 3)
        out[k, :, 3:] = _mat2quat(np.einsum("ji,njk->nik", Rt, Rg))     # R_t^T R_g
    return out


def write_poses(folder: Path, header: dict | None = None) -> Path:
    """qpos.npy replayed through mj_kinematics on the exported model -> poses.f32 (the browser's replay of the legs)."""
    import mujoco
    header = header or read_header(folder)
    m = load_model(folder, header)
    d = mujoco.MjData(m)
    q = np.load(folder / "qpos.npy", allow_pickle=False)
    names, geoms = header["poses"]["fly_names"], header["poses"]["geoms"]
    out = folder / "poses.f32"
    with open(out, "wb") as f:
        for row in q:
            d.qpos[:] = row
            mujoco.mj_kinematics(m, d)
            f.write(geom_poses(m, d, names, geoms).astype("<f4").tobytes())
    return out
