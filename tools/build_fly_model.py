#!/usr/bin/env python3
"""Build the 3-D fly model the page shows (docs/TWO_FLIES_PLAN.md 7.2 and 7.3; a hand-run tool: nothing in the game runs it).

    nice -n 10 .venv/bin/python tools/build_fly_model.py             # writes virtual_fly/web/models/*
    nice -n 10 .venv/bin/python tools/build_fly_model.py --check     # loads the shipped files back and verifies them

What it makes, from the NeuroMechFly v2 body of flygym 1.2.1 (Apache-2.0) as MuJoCo compiles it for the kit's physics body:

  nmf_fly.glb    one glTF binary: one node per mesh geom, named by the MJCF geom name (the fly's name prefix removed; the
                 geom names are the body names), the mesh in that geom's compiled frame. MuJoCo recentres every mesh at its
                 centroid and aligns it with its principal axes, so the raw STL files are not in the frames MuJoCo poses: the
                 compiled vertices and faces are exported instead, placed from the compiled poses, as flygym's own viewer does.
                 Decimated, with normals, in millimetres. Each node's transform is the geom's pose in the standing pose
                 relative to the thorax, so the loaded file is a standing fly at the origin, x forward, z up (MuJoCo's axes).
  nmf_gait.bin   the gait atlas: float32, little-endian, shape (65, G, 7): 64 phases of one stride (frames 0-63, the phase
                 f/64 of the stride) and the standing pose (frame 64); per geom x, y, z (mm, in the thorax frame) then the
                 quaternion w, x, y, z (MuJoCo's order). The stride is NeuroMechFly's recorded single step replayed, which is
                 what the kit's physics fly walks with: the 42 leg joints are set to the targets the kit's Walker
                 (virtual_fly/physics.py) commands at that phase of its CPG, the tripod phase offsets included and the
                 stepping amplitude at its full value, and the forward kinematics evaluated (no physics, no contacts).
  nmf_gait.json  the header: the geom names in node order, the part groups, the hinge points for the page's hand-built wing,
                 abdomen and proboscis rotations (the kit drives no joints there), the stride duration, sizes, the source.
  LICENSE-NeuroMechFly.txt, NOTICE-NeuroMechFly.txt   the licence and what was changed (plan 1.2, 7.2).

Hand-built choices, labelled: "standing" is the Walker's neutral pose (the stance the kit's physics fly holds at zero drive),
not flygym's "stretch" spawn pose; the atlas phase is the phase of leg LF, the phase the physics body reports; the triangle
budgets per part are what reads well at the drawn scale (the aristae and halteres go very low). Nothing here is fitted to data.

Needs the physics extra (mujoco, flygym) and trimesh with fast-simplification:
``pip install -c ../runs/constraints.txt "trimesh==5.1.0" "fast-simplification==0.2.0"``.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import struct
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("MUJOCO_GL", "disable")             # kinematics only: nothing is rendered here
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

MODELS = Path(__file__).resolve().parents[1] / "virtual_fly" / "web" / "models"
GLB, ATLAS_BIN, ATLAS_JSON = "nmf_fly.glb", "nmf_gait.bin", "nmf_gait.json"
LICENCE, NOTICE = "LICENSE-NeuroMechFly.txt", "NOTICE-NeuroMechFly.txt"
PHASES = 64                                               # frames of one stride; frame PHASES is the standing pose
WHEEL = {"file": "flygym-1.2.1-py3-none-any.whl", "bytes": 23345853,
         "sha256": "5db9bb89b7f57e2fda8d716fd8205b0ba7ac9a46e7c194ea6e752e38964f390d"}      # PyPI, docs/TWO_FLIES_PLAN.md 7.2
CITATION = ("Wang-Chen S, Stimpfling VA, Lam TKC, Özdil PG, Genoud L, Hurtak F, Ramdya P (2024). NeuroMechFly v2: simulating "
            "embodied sensorimotor control in adult Drosophila. Nature Methods 21(12):2353-2362. doi:10.1038/s41592-024-02497-y")
LEGS = ("LF", "LM", "LH", "RF", "RM", "RH")
SEGMENTS = ("Coxa", "Femur", "Tibia", "Tarsus1", "Tarsus2", "Tarsus3", "Tarsus4", "Tarsus5")
# triangle budgets per part (hand-built: what reads well at the drawn scale; a part with fewer triangles is kept whole)
BUDGET = {"Head": 8000, "LEye": 2500, "REye": 2500, "Thorax": 4000, "A1A2": 1500, "A3": 1500, "A4": 1500, "A5": 1500, "A6": 1500,
          "LWing": 1500, "RWing": 1500, "LHaltere": 300, "RHaltere": 300, "Rostrum": 1000, "Haustellum": 1000,
          "LPedicel": 500, "RPedicel": 500, "LFuniculus": 500, "RFuniculus": 500, "LArista": 300, "RArista": 300}
LEG_BUDGET = {"Coxa": 900, "Femur": 900, "Tibia": 700, "Tarsus1": 600, "Tarsus2": 300, "Tarsus3": 300, "Tarsus4": 300, "Tarsus5": 300}
HINGE_BODIES = {"wing_left": "LWing", "wing_right": "RWing", "abdomen": "A1A2", "proboscis": "Rostrum"}


def _need():
    """The physics extra and trimesh, or one line saying what is missing."""
    from virtual_fly import physics
    if not physics.available():
        raise SystemExit("build_fly_model.py needs the physics extra: " + physics.unavailable_reason().splitlines()[0])
    try:
        import trimesh                                    # noqa: F401
        import fast_simplification                        # noqa: F401
    except Exception as e:
        raise SystemExit('build_fly_model.py needs trimesh with fast-simplification: pip install -c ../runs/constraints.txt '
                         f'"trimesh==5.1.0" "fast-simplification==0.2.0" (the import failed with: {e!r})')


def budget_for(name: str) -> int:
    if name in BUDGET:
        return BUDGET[name]
    for leg in LEGS:
        if name.startswith(leg):
            return LEG_BUDGET[name[len(leg):]]
    raise KeyError(f"no triangle budget for the part {name!r}")


def part_groups(names: list) -> dict:
    """Every geom in exactly one group, from the MJCF body names (the geom names)."""
    groups = {"head": [], "thorax": [], "abdomen": [], "wing_left": [], "wing_right": [], "proboscis": [], "antennae": [],
              "legs": {leg: [] for leg in LEGS}}
    for n in names:
        if n in ("Head", "LEye", "REye"):
            groups["head"].append(n)
        elif n in ("Thorax", "LHaltere", "RHaltere"):
            groups["thorax"].append(n)
        elif n.startswith("A") and n[1:2].isdigit():
            groups["abdomen"].append(n)
        elif n == "LWing":
            groups["wing_left"].append(n)
        elif n == "RWing":
            groups["wing_right"].append(n)
        elif n in ("Rostrum", "Haustellum"):
            groups["proboscis"].append(n)
        elif n.endswith(("Pedicel", "Funiculus", "Arista")):
            groups["antennae"].append(n)
        elif n[:2] in LEGS and n[2:] in SEGMENTS:
            groups["legs"][n[:2]].append(n)
        else:
            raise KeyError(f"the geom {n!r} fits no part group")
    return groups


class FlyModel:
    """The kit's physics fly, compiled, with the poses the atlas needs."""

    def __init__(self):
        import mujoco
        from virtual_fly import physics
        self.mujoco = mujoco
        self.walker = physics.Walker(seed=0)              # flygym's FlatTerrain, the kit's CPG and recorded step tables
        self.m, self.d = self.walker._m, self.walker._d
        prefix = f"{self.walker.fly.name}/"
        self.mesh_geoms = [g for g in range(self.m.ngeom) if self.m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH]
        self.names = [mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_GEOM, g).removeprefix(prefix) for g in self.mesh_geoms]
        self.thorax = self.walker._thorax
        self.body_id = {mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_BODY, b).removeprefix(prefix): b for b in range(self.m.nbody)}
        acts = np.asarray(self.walker._act)               # the 42 leg position actuators, in the Walker's leg-by-joint order
        self.qadr = np.array([self.m.jnt_qposadr[self.m.actuator_trnid[a, 0]] for a in acts])
        # the tripod: at the CPG's fixed point every leg's phase is leg LF's plus its bias (physics.py, flygym's controller)
        self.bias = np.asarray(physics._tripod_phase_biases[0], dtype=np.float64)
        self.stride_ms = 1000.0 / float(abs(self.walker.cpg.intrinsic_freqs[0]) or 12.0)   # the CPG's 12 Hz

    def joint_targets(self, phase: float, amplitude: float) -> np.ndarray:
        """The Walker's leg joint targets at stride phase ``phase`` (0-1, leg LF's) and stepping amplitude ``amplitude``:
        ``neutral + amplitude * step(phase + bias)`` per leg, the recorded step interpolated as physics.py does."""
        w = self.walker
        ph = (2 * np.pi * phase + self.bias) % (2 * np.pi)
        u = ph / (2 * np.pi) * w._n
        i0 = u.astype(np.int64)
        fr = (u - i0)[:, None]
        legs = np.arange(6)
        off = w._offs[legs, :, i0] * (1 - fr) + w._offs[legs, :, i0 + 1] * fr
        return (w._neutral + amplitude * off).ravel()

    def pose(self, phase: float | None) -> np.ndarray:
        """Every mesh geom's position and quaternion relative to the thorax at a stride phase, or standing (None): (G, 7)."""
        m, d, mujoco = self.m, self.d, self.mujoco
        q = self.joint_targets(0.0, 0.0) if phase is None else self.joint_targets(phase, 1.0)
        d.qpos[self.qadr] = q
        mujoco.mj_kinematics(m, d)
        Rt = d.xmat[self.thorax].reshape(3, 3)
        pt = d.xpos[self.thorax]
        out = np.empty((len(self.mesh_geoms), 7), dtype=np.float64)
        quat = np.empty(4)
        for k, g in enumerate(self.mesh_geoms):
            out[k, :3] = Rt.T @ (d.geom_xpos[g] - pt)
            R = Rt.T @ d.geom_xmat[g].reshape(3, 3)
            mujoco.mju_mat2Quat(quat, np.ascontiguousarray(R).ravel())
            out[k, 3:] = quat
        return out

    def hinges(self) -> dict:
        """Where the page may rotate the wings, the abdomen and the proboscis: those bodies' origins in the thorax frame."""
        self.pose(None)
        Rt, pt = self.d.xmat[self.thorax].reshape(3, 3), self.d.xpos[self.thorax]
        return {k: [round(float(x), 5) for x in Rt.T @ (self.d.xpos[self.body_id[b]] - pt)] for k, b in HINGE_BODIES.items()}

    def mesh(self, k: int):
        """The compiled mesh of mesh geom number ``k`` as a trimesh, in the geom's frame."""
        import trimesh
        m = self.m
        mid = m.geom_dataid[self.mesh_geoms[k]]
        va, vn, fa, fn = m.mesh_vertadr[mid], m.mesh_vertnum[mid], m.mesh_faceadr[mid], m.mesh_facenum[mid]
        return trimesh.Trimesh(m.mesh_vert[va:va + vn].copy(), m.mesh_face[fa:fa + fn].copy(), process=False)


def pose_matrix(row: np.ndarray) -> np.ndarray:
    """A (7,) position-and-quaternion row to a 4x4 matrix."""
    w, x, y, z = row[3:]
    R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                  [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                  [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = row[:3]
    return T


def signed_volume(mesh) -> float:
    v, f = np.asarray(mesh.vertices), np.asarray(mesh.faces)
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)


def build(out: Path) -> dict:
    import trimesh
    t0 = time.perf_counter()
    fly = FlyModel()
    names = fly.names
    groups = part_groups(names)
    standing = fly.pose(None)
    atlas = np.empty((PHASES + 1, len(names), 7), dtype=np.float32)
    for f in range(PHASES):
        atlas[f] = fly.pose(f / PHASES)
    atlas[PHASES] = standing
    # the meshes, decimated to their budgets, placed at their standing poses
    scene = trimesh.Scene()
    before, after, kept_whole, flipped = {}, {}, [], []
    for k, name in enumerate(names):
        mesh = fly.mesh(k)
        before[name] = len(mesh.faces)
        want = budget_for(name)
        if len(mesh.faces) > want:
            try:
                mesh = mesh.simplify_quadric_decimation(face_count=want)
            except Exception as e:                        # a mesh the decimator refuses stays whole: said in the summary
                kept_whole.append(f"{name} ({e!r})")
        if signed_volume(mesh) < 0:                       # keep every surface facing outward (the mirrored side included)
            mesh.invert()
            flipped.append(name)
        after[name] = len(mesh.faces)
        scene.add_geometry(mesh, node_name=name, geom_name=name, transform=pose_matrix(standing[k]))
    out.mkdir(parents=True, exist_ok=True)
    glb = scene.export(file_type="glb", include_normals=True)
    (out / GLB).write_bytes(glb)
    (out / ATLAS_BIN).write_bytes(atlas.astype("<f4").tobytes())
    header = {
        "version": 1, "unit": "mm", "axes": "mujoco: x forward, z up",
        "names": "the MJCF geom names of flygym 1.2.1's neuromechfly_seqik_kinorder_ypr.xml (the fly's name prefix removed; "
                 "they are the body names), one node per geom in this order",
        "geoms": names, "phases": PHASES, "standing": PHASES, "frames": PHASES + 1,
        "stride_ms": round(fly.stride_ms, 4), "thorax": "Thorax",
        "parts": groups, "hinges": fly.hinges(),
        "hinges_note": "the origins of the LWing, RWing, A1A2 and Rostrum bodies in the thorax frame, where the page's "
                       "hand-built wing extension, abdomen bend and proboscis extension rotate those parts (the kit drives "
                       "no joints there); every other leg and body joint is in the atlas",
        "stride": "NeuroMechFly's recorded single step replayed: the leg joints at the targets the kit's CPG walker commands at "
                  "each phase (tripod offsets, full stepping amplitude), forward kinematics only; frame 64 is the walker's "
                  "neutral pose, the stance the physics fly holds at zero drive",
        "phase_of": "leg LF, as the physics body reports it (phase 0 = frame 0)",
        "frame_layout": "float32 little-endian (frames, geoms, 7): x, y, z in mm in the thorax frame, then qw, qx, qy, qz",
        "source": {"package": "flygym", "version": "1.2.1", "wheel": WHEEL["file"], "wheel_bytes": WHEEL["bytes"],
                   "wheel_sha256": WHEEL["sha256"], "licence": "Apache-2.0", "citation": CITATION},
        "triangles": int(sum(after.values())), "triangles_before": int(sum(before.values())),
        "triangles_per_part": after, "glb_bytes": len(glb), "atlas_bytes": atlas.nbytes,
        "built": time.strftime("%Y-%m-%d"), "tool": "tools/build_fly_model.py",
    }
    (out / ATLAS_JSON).write_text(json.dumps(header, indent=1) + "\n")
    licence_text = _flygym_licence()
    (out / LICENCE).write_text("NeuroMechFly v2 / flygym 1.2.1, Copyright 2023-2026 The NeuroMechFly v2 Authors (NeLy-EPFL), "
                               "Apache License 2.0. The licence text, as shipped in the flygym 1.2.1 wheel:\n\n" + licence_text)
    (out / NOTICE).write_text(
        "NeuroMechFly v2 body meshes, as shipped with Virtual Fly's 3-D view (nmf_fly.glb, nmf_gait.bin, nmf_gait.json)\n"
        "\n"
        f"Taken from: the mesh files of the flygym package, version 1.2.1 ({WHEEL['file']}, {WHEEL['bytes']:,} bytes from PyPI,\n"
        f"SHA-256 {WHEEL['sha256']}), NeLy-EPFL, licensed under the Apache License 2.0 (LICENSE-NeuroMechFly.txt next to this file).\n"
        "\n"
        "Changes made (Apache License 2.0, section 4(b)): the 69 meshes of the model neuromechfly_seqik_kinorder_ypr.xml were\n"
        "exported from MuJoCo's compiled model in the frames MuJoCo poses them (recentred, principal axes), scaled to millimetres,\n"
        "converted to glTF binary with vertex normals, and decimated from "
        f"{header['triangles_before']:,} to {header['triangles']:,} triangles in all (per part: {json.dumps(after)});\n"
        "a gait atlas (nmf_gait.bin) of the model's poses over one stride of its recorded single step was computed with MuJoCo's\n"
        "forward kinematics. No mesh was redrawn; no vertex was edited by hand.\n"
        "\n"
        f"Please cite: {CITATION}\n"
        "\n"
        f"Made on {time.strftime('%Y-%m-%d')} by tools/build_fly_model.py of Virtual Fly.\n")
    summary = {"triangles_before": header["triangles_before"], "triangles": header["triangles"], "per_part": after,
               "glb_bytes": len(glb), "atlas_shape": list(atlas.shape), "atlas_bytes": atlas.nbytes, "unit": "mm",
               "stride_ms": header["stride_ms"], "kept_whole": kept_whole, "flipped": flipped, "hinges": header["hinges"],
               "seconds": round(time.perf_counter() - t0, 1), "fly_height_mm": round(float(fly.d.xpos[fly.thorax][2]), 3)}
    return summary


def _flygym_licence() -> str:
    import importlib.metadata as md
    dist = md.distribution("flygym")
    for f in dist.files or []:
        if f.name.upper().startswith("LICENSE"):
            return f.read_text()
    raise SystemExit("the installed flygym carries no LICENSE file: nothing to ship next to the meshes")


# ---------------------------------------------------------------------------------------------------------- the check
def read_glb_nodes(path: Path) -> dict:
    """The named nodes of a GLB and their transforms, parsed from the JSON chunk with plain Python (no trimesh)."""
    b = path.read_bytes()
    magic, version, length = struct.unpack("<III", b[:12])
    if magic != 0x46546C67 or version != 2 or length != len(b):
        raise ValueError(f"{path} is not a glTF 2 binary")
    clen, ctype = struct.unpack("<II", b[12:20])
    if ctype != 0x4E4F534A:
        raise ValueError(f"{path}: the first chunk is not JSON")
    j = json.loads(b[20:20 + clen])
    out = {}
    for n in j["nodes"]:
        name = n.get("name")
        if name is None:
            continue
        if "matrix" in n:
            T = np.array(n["matrix"], dtype=np.float64).reshape(4, 4).T     # glTF matrices are column-major
        else:
            T = np.eye(4)
            if "translation" in n:
                T[:3, 3] = n["translation"]
            if "rotation" in n:                           # glTF x, y, z, w
                x, y, z, w = n["rotation"]
                T[:3, :3] = pose_matrix(np.array([0, 0, 0, w, x, y, z]))[:3, :3]
        out[name] = {"matrix": T, "mesh": n.get("mesh")}
    return out, j


def check(models: Path) -> list:
    """What is wrong with the shipped files (empty when nothing is): node names, the standing frame, the quaternions."""
    problems = []
    header = json.loads((models / ATLAS_JSON).read_text())
    names = header["geoms"]
    G = len(names)
    raw = (models / ATLAS_BIN).read_bytes()
    if len(raw) != header["frames"] * G * 7 * 4:
        problems.append(f"the atlas holds {len(raw)} bytes, not {header['frames']} x {G} x 7 x 4")
        return problems
    atlas = np.frombuffer(raw, dtype="<f4").reshape(header["frames"], G, 7)
    norms = np.linalg.norm(atlas[:, :, 3:], axis=2)
    if not np.allclose(norms, 1.0, atol=1e-4):
        problems.append(f"quaternions off the unit sphere: max |norm - 1| = {np.abs(norms - 1).max():.2e}")
    nodes, j = read_glb_nodes(models / GLB)
    if set(nodes) != set(names):
        problems.append(f"node names differ from the header's geoms: {sorted(set(nodes) ^ set(names))[:10]}")
    standing = atlas[header["standing"]]
    for k, name in enumerate(names):
        if name not in nodes:
            continue
        T = nodes[name]["matrix"]
        want = pose_matrix(standing[k].astype(np.float64))
        if np.abs(T[:3, 3] - want[:3, 3]).max() > 1e-4 or np.abs(T[:3, :3] - want[:3, :3]).max() > 1e-4:
            problems.append(f"{name}: the GLB's transform differs from the standing frame")
    grouped = [n for key, g in header["parts"].items() for n in (sum(g.values(), []) if isinstance(g, dict) else g)]
    if sorted(grouped) != sorted(names):
        problems.append("the part groups do not cover every geom exactly once")
    try:
        import trimesh
        scene = trimesh.load(models / GLB, force="scene")
        tri = sum(len(g.faces) for g in scene.geometry.values())
        if tri != header["triangles"]:
            problems.append(f"the GLB holds {tri} triangles, the header says {header['triangles']}")
        if set(scene.graph.nodes_geometry) != set(names):
            problems.append("trimesh sees different node names than the header's geoms")
    except ImportError:
        pass
    return problems


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(MODELS), help="the folder to write (default: the package's web/models)")
    ap.add_argument("--check", action="store_true", help="load the shipped files back and verify them instead of building")
    args = ap.parse_args(argv)
    out = Path(args.out)
    if args.check:
        _need()
        problems = check(out)
        for p in problems:
            print("problem:", p)
        print("ok: the model, the atlas and the header agree" if not problems else f"{len(problems)} problem(s)")
        return 1 if problems else 0
    _need()
    s = build(out)
    print(f"model built in {s['seconds']} s into {out} (unit mm; the thorax stands {s['fly_height_mm']} mm above the floor)")
    print(f"  triangles: {s['triangles_before']:,} -> {s['triangles']:,}; GLB {s['glb_bytes']:,} bytes; "
          f"atlas {s['atlas_shape']} float32 = {s['atlas_bytes']:,} bytes; stride {s['stride_ms']} ms")
    print("  per part:", json.dumps(s["per_part"]))
    print("  hinges (mm, thorax frame):", json.dumps(s["hinges"]))
    if s["kept_whole"]:
        print("  kept whole (the decimator refused):", ", ".join(s["kept_whole"]))
    if s["flipped"]:
        print("  faces turned outward:", ", ".join(s["flipped"]))
    problems = check(out)
    for p in problems:
        print("problem:", p)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
