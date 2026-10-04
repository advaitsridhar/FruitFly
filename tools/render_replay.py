#!/usr/bin/env python3
"""Render a saved physics recording to an MP4 with MuJoCo's own renderer (the two-flies work, docs/TWO_FLIES_PLAN.md 8.9).
A hand-run tool. The recording is a folder made by the game's "capture" action (virtual_fly/recording.py): its exported
MuJoCo model and `qpos.npy` are replayed through MuJoCo's kinematics at the video's frame rate, however fast or slowly the
run itself went, so the video plays at real speed.

    python tools/render_replay.py recordings/20261003-210246                       # overhead, 1080p, 30 fps, 1x
    python tools/render_replay.py recordings/20261003-210246 --camera follow --follow 0 --out /tmp/male.mp4
    python tools/render_replay.py recordings/20261003-210246 --camera side --speed 0.5 --start 2 --end 5
    python tools/render_replay.py recordings/20261003-210246 --check /tmp/frames   # three PNGs (first, middle, last), no video

Cameras (all hand-built choices, as the page's are): `overhead` looks straight down at the flies (their midpoint, or fly K
with --follow K; `--follow none` frames the whole dish), zoomed to frame them with --margin mm around; `follow` tracks fly K
from behind and above, turning with its heading, at --distance mm; `side` is a low oblique view of the flies' midpoint.
The camera's aim is smoothed over about a quarter of a second of recording time, so a stride's wobble does not shake the
picture. Between two recorded ticks the joints are interpolated linearly (the root quaternions normalised after the
blend). The second fly's body is tinted lighter so the two can be told apart (hand-built; the model's own colours are
flygym's grey). A small label gives the recording, the time and the speed (--no-hud leaves it out).

MUJOCO_GL picks the renderer: this file sets it to `egl` unless it is exported already (as virtual_fly/physics.py sets
`disable` for physics without a screen); on Linux `egl` and `osmesa` work without a display, `glfw` needs one. The MP4 is
written by OpenCV (opencv-python-headless comes with the physics extra) with the `mp4v` codec; nothing else is needed.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")       # before mujoco is imported (an exported value wins)

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from virtual_fly import physics, recording        # noqa: E402

CAMERAS = ("overhead", "follow", "side")
SMOOTH_S = 0.25                                   # the camera's aim follows the flies with this time constant (recording s)
TINT = 0.45                                       # the second fly's colours blended this far toward white (hand-built)


def _resolve(folder: str) -> Path:
    p = Path(folder)
    if (p / "header.json").is_file():
        return p
    found = recording.folder_of(folder)
    if found is None:
        raise SystemExit(f"{folder}: not a recording folder (no header.json), and no recording of that id under "
                         f"{recording.recordings_dir()}")
    return found


def _thorax_id(m, fly: str) -> int:
    """The fly's thorax body: flygym 2.1's name (c_thorax), or 1.2.1's (Thorax) in a recording made before v3.0."""
    for name in (physics.THORAX, "Thorax"):
        try:
            return m.body(f"{fly}/{name}").id
        except KeyError:
            continue
    raise SystemExit(f"{fly}: no thorax body in the recording's model")


def _free_joints(m) -> list[int]:
    import mujoco
    return [int(m.jnt_qposadr[j]) for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]


def interpolate(q: np.ndarray, t: np.ndarray, tau: float, free_adrs: list[int]) -> np.ndarray:
    """qpos at recording time tau: the two nearest ticks blended linearly, each root quaternion put in the same hemisphere
    as its partner first and normalised after."""
    n = len(t)
    if n == 1 or tau <= t[0]:
        return q[0].copy()
    if tau >= t[-1]:
        return q[-1].copy()
    k = int(np.searchsorted(t, tau, side="right") - 1)
    k = min(max(k, 0), n - 2)
    span = t[k + 1] - t[k]
    w = 0.0 if span <= 0 else min(1.0, max(0.0, (tau - t[k]) / span))
    a, b = q[k], q[k + 1].copy()
    for adr in free_adrs:
        if np.dot(a[adr + 3:adr + 7], b[adr + 3:adr + 7]) < 0:
            b[adr + 3:adr + 7] *= -1.0
    out = (1.0 - w) * a + w * b
    for adr in free_adrs:
        nrm = np.linalg.norm(out[adr + 3:adr + 7])
        if nrm > 0:
            out[adr + 3:adr + 7] /= nrm
    return out


class _Aim:
    """Exponential smoothing of the camera's aim across frames (a point, a distance and an unwrapped angle)."""

    def __init__(self, dt_frame: float):
        self.alpha = 1.0 - math.exp(-dt_frame / SMOOTH_S) if SMOOTH_S > 0 else 1.0
        self.point = None
        self.distance = None
        self.angle = None

    def update(self, point: np.ndarray, distance: float, angle: float):
        if self.point is None:
            self.point, self.distance, self.angle = np.array(point, dtype=float), float(distance), float(angle)
        else:
            self.point += (point - self.point) * self.alpha
            self.distance += (distance - self.distance) * self.alpha
            d = (angle - self.angle + 180.0) % 360.0 - 180.0          # the short way round
            self.angle += d * self.alpha
        return self.point, self.distance, self.angle


def tint_second_fly(m, fly_names: list[str]):
    """The second fly's geoms blended toward white (hand-built, so that the two flies can be told apart)."""
    import mujoco
    if len(fly_names) < 2:
        return 0
    prefix = fly_names[1] + "/"
    n = 0
    for g in range(m.ngeom):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        if name.startswith(prefix):
            m.geom_rgba[g, :3] = m.geom_rgba[g, :3] * (1.0 - TINT) + TINT
            n += 1
    return n


def render(folder: Path, out: Path | None = None, fps: float = 30.0, width: int = 1920, height: int = 1080,
           camera: str = "overhead", follow: str | None = None, speed: float = 1.0, start: float | None = None,
           end: float | None = None, check: Path | None = None, margin: float = 3.0, distance: float | None = None,
           hud: bool = True, tint: bool = True, quiet: bool = False) -> dict:
    import cv2
    import mujoco
    header = recording.read_header(folder)
    if not header.get("physics"):
        raise SystemExit(f"{folder}: a drawn-body recording has no MuJoCo model to render (physics runs only)")
    m = recording.load_model(folder, header)
    q = np.load(folder / "qpos.npy", allow_pickle=False)
    t = np.load(folder / "t.npy", allow_pickle=False)
    if q.ndim != 2 or q.shape[0] != t.shape[0] or q.shape[1] != m.nq:
        raise SystemExit(f"{folder}: qpos.npy is {q.shape} and t.npy {t.shape}, the model has nq {m.nq}")
    if len(t) == 0:
        raise SystemExit(f"{folder}: an empty recording")
    fly_names = header["physics"]["fly_names"]
    thorax = [_thorax_id(m, fly) for fly in fly_names]
    free_adrs = _free_joints(m)
    arena_r = 50.0
    if tint:
        tint_second_fly(m, fly_names)
    m.vis.global_.offwidth = max(int(m.vis.global_.offwidth), width)      # the offscreen buffer must hold the frame
    m.vis.global_.offheight = max(int(m.vis.global_.offheight), height)
    d = mujoco.MjData(m)
    t0_rec, t1_rec = float(t[0]), float(t[-1])
    start = t0_rec if start is None else max(t0_rec, float(start))
    end = t1_rec if end is None else min(t1_rec, float(end))
    if end < start:
        raise SystemExit(f"--end {end} is before --start {start}")
    if speed <= 0 or fps <= 0:
        raise SystemExit("--speed and --fps must be more than 0")
    dt_frame = speed / fps                                                  # recording seconds per video frame
    n = int(math.floor((end - start) / dt_frame + 1e-9)) + 1
    which = set(range(n)) if check is None else {0, n // 2, n - 1}
    if camera not in CAMERAS:
        raise SystemExit(f"--camera must be one of {', '.join(CAMERAS)}")
    follow_k = None                                                         # None: the flies' midpoint; -1: the dish centre
    if follow is not None and str(follow).lower() == "none":
        follow_k = -1
    elif follow is not None:
        follow_k = int(follow)
        if not 0 <= follow_k < len(fly_names):
            raise SystemExit(f"--follow {follow}: the recording has flies 0-{len(fly_names) - 1}")
    elif camera == "follow":
        follow_k = 0
    if camera == "follow" and follow_k == -1:
        raise SystemExit("--camera follow needs a fly to follow (--follow 0, 1, ...), not 'none'")
    fovy = float(m.vis.global_.fovy)
    tan_half = math.tan(math.radians(fovy) / 2)
    aspect = width / height
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    aim = _Aim(dt_frame)
    renderer = mujoco.Renderer(m, height, width)
    writer = None
    if check is None:
        out = out or (folder / f"{camera}.mp4")
        out.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (width, height))
        if not writer.isOpened():
            raise SystemExit(f"OpenCV could not open {out} for writing (the mp4v codec)")
    else:
        check.mkdir(parents=True, exist_ok=True)
    rid = folder.name
    wall0 = time.perf_counter()
    rendered = 0
    try:
        for i in range(n):
            tau = min(end, start + i * dt_frame)
            d.qpos[:] = interpolate(q, t, tau, free_adrs)
            mujoco.mj_kinematics(m, d)
            pos = np.array([d.xpos[b] for b in thorax])                   # every fly's thorax
            if follow_k is None:
                target = pos.mean(axis=0)
            elif follow_k == -1:
                target = np.array([0.0, 0.0, 0.0])
            else:
                target = pos[follow_k].copy()
            if camera == "overhead":
                if follow_k == -1:
                    half_y, half_x = arena_r + margin, arena_r + margin
                else:
                    half_y = float(np.abs(pos[:, 1] - target[1]).max()) + margin
                    half_x = float(np.abs(pos[:, 0] - target[0]).max()) + margin
                dist = max(12.0, half_y / tan_half, half_x / (aspect * tan_half))
                angle = 90.0                                                # +x to the right, +y up, as the 2-D dish
                elevation = -89.9
                target = np.array([target[0], target[1], 0.0])
            elif camera == "follow":
                xm = d.xmat[thorax[follow_k]]
                angle = math.degrees(math.atan2(xm[3], xm[0]))               # the fly's heading: the camera looks along it
                dist = 9.0 if distance is None else distance
                elevation = -30.0
            else:                                                           # side: low and oblique
                if follow_k == -1:
                    span = arena_r
                else:
                    span = float(np.linalg.norm(pos.max(axis=0) - pos.min(axis=0))) / 2 + margin
                dist = max(10.0, span / tan_half) if distance is None else distance
                angle = 35.0
                elevation = -12.0
            point, dist_s, angle_s = aim.update(target, dist, angle)
            if i not in which:
                continue
            cam.lookat[:] = point
            cam.distance = dist_s
            cam.azimuth = angle_s
            cam.elevation = elevation
            renderer.update_scene(d, camera=cam)
            img = renderer.render()
            frame = cv2.cvtColor(np.ascontiguousarray(img), cv2.COLOR_RGB2BGR)
            if hud:
                label = f"replay {rid}   t {tau:6.2f} s   {speed:g}x   {camera}" + (f" fly {follow_k}" if follow_k not in (None, -1) else "")
                scale = max(0.4, height / 1080 * 0.8)
                cv2.putText(frame, label, (int(12 * scale / 0.8), height - int(14 * scale / 0.8)), cv2.FONT_HERSHEY_SIMPLEX,
                            scale, (20, 20, 20), max(1, int(3 * scale)), cv2.LINE_AA)
                cv2.putText(frame, label, (int(12 * scale / 0.8), height - int(14 * scale / 0.8)), cv2.FONT_HERSHEY_SIMPLEX,
                            scale, (235, 235, 235), max(1, int(scale)), cv2.LINE_AA)
            if writer is not None:
                writer.write(frame)
            else:
                name = {0: "check-first.png", n // 2: "check-middle.png", n - 1: "check-last.png"}
                for idx, fname in name.items():                            # with one or two frames, one frame serves twice
                    if idx == i:
                        cv2.imwrite(str(check / fname), frame)
            rendered += 1
    finally:
        if writer is not None:
            writer.release()
        renderer.close()
    wall = time.perf_counter() - wall0
    summary = {"frames": rendered, "video_frames": n if writer is not None else 0, "seconds": round(end - start, 3),
               "fps": fps, "speed": speed, "width": width, "height": height, "camera": camera, "wall_s": round(wall, 2),
               "render_fps": round(rendered / wall, 2) if wall > 0 else None, "gl": os.environ.get("MUJOCO_GL"),
               "out": str(out) if writer is not None else str(check), "bytes": out.stat().st_size if writer is not None else None}
    if not quiet:
        if writer is not None:
            print(f"{rendered} frames ({end - start:.3f} s of the recording at {speed:g}x, {fps:g} fps, {width}x{height}, {camera}) "
                  f"in {wall:.1f} s: {summary['render_fps']} frames per second rendered; MUJOCO_GL={summary['gl']}; "
                  f"written {out} ({summary['bytes'] / 1e6:.1f} MB)")
        else:
            print(f"{rendered} check frames of {n} ({width}x{height}, {camera}) in {wall:.1f} s under {check}; MUJOCO_GL={summary['gl']}")
    return summary


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("recording", help="a recording folder, or its id under the recordings folder")
    ap.add_argument("--out", type=Path, help="the MP4 (default: <camera>.mp4 inside the recording's folder)")
    ap.add_argument("--fps", type=float, default=30.0, help="the video's frame rate (default 30)")
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--camera", choices=CAMERAS, default="overhead")
    ap.add_argument("--follow", default=None, help="which fly the camera centres on (0, 1, ...); 'none' for the dish centre; "
                                                   "the default is the flies' midpoint (follow: fly 0)")
    ap.add_argument("--speed", type=float, default=1.0, help="playback speed: 1 is real time, 0.25 slow motion")
    ap.add_argument("--start", type=float, default=None, help="recording seconds to start at")
    ap.add_argument("--end", type=float, default=None, help="recording seconds to end at")
    ap.add_argument("--check", type=Path, default=None, help="write three PNG frames (first, middle, last) here instead of a video")
    ap.add_argument("--margin", type=float, default=3.0, help="mm around the flies in the overhead and side views")
    ap.add_argument("--distance", type=float, default=None, help="the follow and side cameras' distance in mm")
    ap.add_argument("--no-hud", action="store_true", help="no label on the frames")
    ap.add_argument("--no-tint", action="store_true", help="keep the model's own colours for the second fly")
    args = ap.parse_args(argv)
    folder = _resolve(args.recording)
    return render(folder, out=args.out, fps=args.fps, width=args.width, height=args.height, camera=args.camera,
                  follow=args.follow, speed=args.speed, start=args.start, end=args.end, check=args.check, margin=args.margin,
                  distance=args.distance, hud=not args.no_hud, tint=not args.no_tint)


if __name__ == "__main__":
    main()
