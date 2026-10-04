"""tools/render_replay.py: a saved physics recording rendered to an MP4 with MuJoCo's renderer (docs/TWO_FLIES_PLAN.md 8.9).
Needs flygym (the recording comes from a physics pair, as in tests/test_recording.py) and a MuJoCo renderer. MuJoCo fixes
its GL backend when it is first imported, and virtual_fly.physics asks for none (MUJOCO_GL=disable) unless the variable is
exported, so this file sets MUJOCO_GL=egl before anything imports mujoco; when another test module imported it first (the
whole suite), the renderer fixture finds no renderer and skips. On its own: ``python -m pytest -q tests/test_render_replay.py``;
in the whole suite: ``MUJOCO_GL=egl python -m pytest -q`` (``osmesa`` works too on Linux)."""
import importlib.util
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from virtual_fly import physics, recording  # noqa: E402
from virtual_fly.game import Game  # noqa: E402
from virtual_fly.settings import build_brain  # noqa: E402

needs_flygym = pytest.mark.skipif(not physics.available(), reason=f"physics body unavailable (optional): {physics._IMPORT_ERROR!r}")
ROOT = Path(__file__).resolve().parents[1]


def _tool():
    spec = importlib.util.spec_from_file_location("render_replay", ROOT / "tools" / "render_replay.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["render_replay"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def renderer_ok():
    import mujoco
    try:
        m = mujoco.MjModel.from_xml_string("<mujoco><worldbody><light pos='0 0 1'/><geom size='.1'/></worldbody></mujoco>")
        r = mujoco.Renderer(m, 32, 32)
        r.close()
    except Exception as e:
        pytest.skip(f"no MuJoCo renderer in this process (MUJOCO_GL={os.environ.get('MUJOCO_GL')}): {e!r}")
    return True


@pytest.fixture
def rec_dir(tmp_path, monkeypatch):
    d = tmp_path / "recordings"
    monkeypatch.setattr(recording, "recordings_dir", lambda: d)
    return d


def _pair_recording(conn, ticks: int = 6) -> Path:
    g = Game(build_brain(conn, "game", seed=0), seed=1, body="physics", brain_procs="off",
             partner={"conn": conn, "brain_kwargs": {"seed": 1000}, "parts": False})
    try:
        assert g.action({"type": "capture", "on": True})["ok"]
        for _ in range(ticks):
            g.tick()
        rid = g.capture.id
        assert g.action({"type": "capture", "on": False})["ok"]
        g.tick()
    finally:
        g.close()
    return recording.folder_of(rid)


def test_interpolation_blends_ticks_and_keeps_the_root_quaternions_unit():
    tool = _tool()
    t = np.array([0.0, 1.0, 2.0])
    q = np.zeros((3, 8))
    q[:, 3:7] = [[1, 0, 0, 0], [0, 1, 0, 0], [-0.6, -0.8, 0, 0]]     # the quaternion lives at 3..7 (a free joint at 0)
    q[:, 7] = [0.0, 10.0, 20.0]
    assert tool.interpolate(q, t, -1.0, [0]).tolist() == q[0].tolist() and tool.interpolate(q, t, 5.0, [0]).tolist() == q[2].tolist()
    mid = tool.interpolate(q, t, 0.5, [0])
    assert mid[7] == 5.0 and abs(np.linalg.norm(mid[3:7]) - 1.0) < 1e-12 and mid[3] == pytest.approx(mid[4])
    late = tool.interpolate(q, t, 1.5, [0])        # the second tick's quaternion sits in the other hemisphere (dot -0.8): flipped
    assert abs(np.linalg.norm(late[3:7]) - 1.0) < 1e-12 and late[7] == 15.0
    assert late[3] > 0 and late[4] > 0 and late[3] == pytest.approx(0.3 / math.hypot(0.3, 0.9))


@needs_flygym
def test_a_recording_renders_to_an_mp4_and_to_check_frames(conn, rec_dir, tmp_path, renderer_ok):
    import cv2
    tool = _tool()
    folder = _pair_recording(conn)
    out = tmp_path / "out.mp4"
    summary = tool.main([str(folder), "--out", str(out), "--fps", "10", "--width", "160", "--height", "120"])
    assert out.is_file() and out.stat().st_size > 1024 and summary["out"] == str(out) and summary["gl"] in ("egl", "osmesa", "glfw")
    assert summary["frames"] == summary["video_frames"] == 2                 # 0.125 s of ticks at 10 fps: frames at 0 and 0.1 s
    cap = cv2.VideoCapture(str(out))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 2 and cap.get(cv2.CAP_PROP_FPS) == pytest.approx(10.0)
    ok, frame = cap.read()
    cap.release()
    assert ok and frame.shape == (120, 160, 3) and frame.std() > 1.0        # not a blank frame
    chk = tmp_path / "check"
    summary = tool.main([str(folder), "--check", str(chk), "--width", "160", "--height", "120", "--camera", "follow"])
    assert sorted(p.name for p in chk.iterdir()) == ["check-first.png", "check-last.png", "check-middle.png"]
    assert summary["frames"] >= 1 and all(p.stat().st_size > 200 for p in chk.iterdir())
    with pytest.raises(SystemExit, match="flies 0-1"):
        tool.main([str(folder), "--check", str(chk), "--follow", "5", "--width", "64", "--height", "48"])
    with pytest.raises(SystemExit, match="not a recording folder"):
        tool.main([str(tmp_path / "nowhere")])
