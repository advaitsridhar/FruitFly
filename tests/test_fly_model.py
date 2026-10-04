"""The 3-D fly model shipped with the kit (docs/TWO_FLIES_PLAN.md 7.2, 7.3): the GLB, the gait atlas and its header, the
licence and the notice are present and agree with each other, parsed with plain Python (no MuJoCo, no trimesh)."""

import importlib.util
import json
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "virtual_fly" / "web" / "models"
FILES = ("nmf_fly.glb", "nmf_gait.json", "nmf_gait.bin", "LICENSE-NeuroMechFly.txt", "NOTICE-NeuroMechFly.txt")
WHEEL_SHA = "5db9bb89b7f57e2fda8d716fd8205b0ba7ac9a46e7c194ea6e752e38964f390d"


def glb_json(path: Path) -> dict:
    b = path.read_bytes()
    magic, version, length = struct.unpack("<III", b[:12])
    assert magic == 0x46546C67 and version == 2 and length == len(b), "not a glTF 2 binary"
    clen, ctype = struct.unpack("<II", b[12:20])
    assert ctype == 0x4E4F534A
    return json.loads(b[20:20 + clen])


def node_transform(node: dict) -> np.ndarray:
    if "matrix" in node:
        return np.array(node["matrix"], dtype=np.float64).reshape(4, 4).T
    T = np.eye(4)
    if "translation" in node:
        T[:3, 3] = node["translation"]
    return T


def test_the_model_files_are_shipped():
    for name in FILES:
        assert (MODELS / name).is_file(), name
    glb = (MODELS / "nmf_fly.glb").stat().st_size
    assert 100_000 < glb <= 3_500_000, f"the GLB is {glb:,} bytes"
    licence = (MODELS / "LICENSE-NeuroMechFly.txt").read_text()
    assert "Apache License" in licence and "NeuroMechFly" in licence and "Version 2.0" in licence
    notice = (MODELS / "NOTICE-NeuroMechFly.txt").read_text()
    assert WHEEL_SHA in notice and "decimated" in notice and "doi:10.1038/s41592-024-02497-y" in notice


def test_the_atlas_and_the_model_agree():
    header = json.loads((MODELS / "nmf_gait.json").read_text())
    assert header["version"] == 1 and header["unit"] == "mm" and header["axes"].startswith("mujoco")
    names = header["geoms"]
    G = len(names)
    assert G == 69 and len(set(names)) == G and {"Thorax", "Head", "LFCoxa", "RHTarsus5", "LArista"} <= set(names)
    assert header["phases"] == 64 and header["standing"] == 64 and header["frames"] == 65
    assert abs(header["stride_ms"] - 1000.0 / 12.0) < 0.01                   # the CPG's 12 Hz
    raw = (MODELS / "nmf_gait.bin").read_bytes()
    assert len(raw) == 65 * G * 7 * 4
    atlas = np.frombuffer(raw, dtype="<f4").reshape(65, G, 7)
    assert np.isfinite(atlas).all()
    assert np.allclose(np.linalg.norm(atlas[:, :, 3:], axis=2), 1.0, atol=1e-4)   # unit quaternions
    assert np.abs(atlas[:, :, :3]).max() < 6.0                                # a fly is a few millimetres long
    moving = np.abs(atlas[:64, :, :3] - atlas[64, :, :3]).max(axis=(0, 2))    # the legs move over the stride ...
    thorax = names.index("Thorax")
    assert moving[names.index("LFTarsus5")] > 0.1 and moving[thorax] < 1e-6    # ... the thorax never does
    # every geom in exactly one part group
    parts = header["parts"]
    grouped = [n for key, g in parts.items() for n in (sum(g.values(), []) if isinstance(g, dict) else g)]
    assert sorted(grouped) == sorted(names)
    assert set(parts["legs"]) == {"LF", "LM", "LH", "RF", "RM", "RH"} and all(len(v) == 8 for v in parts["legs"].values())
    for key in ("wing_left", "wing_right", "abdomen", "proboscis"):
        h = header["hinges"][key]
        assert len(h) == 3 and all(np.isfinite(h)) and max(abs(x) for x in h) < 3.0
    assert header["hinges"]["wing_left"][1] > 0 > header["hinges"]["wing_right"][1]   # left is +y (MuJoCo's x forward, z up)
    # the GLB's nodes are the header's geoms, placed at the standing frame
    j = glb_json(MODELS / "nmf_fly.glb")
    nodes = {n["name"]: n for n in j["nodes"] if "name" in n}
    assert set(nodes) == set(names)
    assert all("mesh" in nodes[n] for n in names)
    for p in j["meshes"]:
        assert all({"POSITION", "NORMAL"} <= set(prim["attributes"]) for prim in p["primitives"])
    standing = atlas[64].astype(np.float64)
    for k, n in enumerate(names):
        T = node_transform(nodes[n])
        assert np.abs(T[:3, 3] - standing[k, :3]).max() < 1e-4, n
    assert header["triangles"] < 100_000 and header["triangles_before"] > header["triangles"]
    assert header["source"]["wheel_sha256"] == WHEEL_SHA and header["source"]["licence"] == "Apache-2.0"


def test_the_stride_fits_flygym_2_1_and_rebuilds_the_atlas():
    """The shipped stride (web/models/nmf_stride.npz, tools/refit_stride.py): flygym 2.1's joint order, the recorded step's swing
    windows, and the whole gait atlas rebuilt from the 2.1 skeleton through the file's geom offsets (needs the physics extra)."""
    from virtual_fly import physics
    z = np.load(ROOT / "virtual_fly" / "web" / "models" / "nmf_stride.npz", allow_pickle=False)
    assert z["angles"].shape == (65, 42) and int(z["phases"]) == 64 and int(z["standing"]) == 64 and float(z["stride_ms"]) == pytest.approx(1000 / 12, abs=0.01)
    assert len(z["dof_names"]) == 42 and str(z["dof_names"][0]) == "c_thorax-lf_coxa-pitch" and str(z["dof_names"][41]) == "rh_tibia-rh_tarsus1-pitch"
    assert list(z["geom_names"]) == json.loads((ROOT / "virtual_fly" / "web" / "models" / "nmf_gait.json").read_text())["geoms"]
    assert set(z["geom_bodies"]) >= {"c_thorax", "lf_coxa", "rh_tarsus5"} and float(z["atlas_pos_error_mm"]) < 0.02 and float(z["atlas_angle_error_deg"]) < 1.0
    assert (z["swing_start"] == 0).all() and (z["swing_end"] > 1.9).all() and (z["swing_end"] < 2.4).all()
    assert "Apache-2.0" in str(z["source"]) and "NOTICE-NeuroMechFly" in str(z["source"])
    if not physics.available():
        pytest.skip("the physics extra is not installed")
    spec = importlib.util.spec_from_file_location("refit_stride", ROOT / "tools" / "refit_stride.py")
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    assert tool.check(ROOT / "virtual_fly" / "web" / "models") == []


def test_the_tool_checks_its_own_files():
    """The build tool's --check on the shipped files (needs the physics extra and trimesh; skipped without them)."""
    from virtual_fly import physics
    if not physics.available():
        pytest.skip("the physics extra is not installed")
    pytest.importorskip("trimesh")
    out = subprocess.run([sys.executable, str(ROOT / "tools" / "build_fly_model.py"), "--check"], capture_output=True, text=True,
                         timeout=600, cwd=str(ROOT))
    assert out.returncode == 0 and "ok:" in out.stdout, out.stdout + out.stderr
