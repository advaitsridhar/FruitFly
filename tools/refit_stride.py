#!/usr/bin/env python3
"""Refit the kit's recorded stride onto flygym 2.1's NeuroMechFly skeleton (the two-flies work, the real-time route of
docs/TWO_FLIES_PROGRESS.md; a developer tool, run once; its output ships as virtual_fly/web/models/nmf_stride.npz).

The 3-D view's gait atlas (web/models/nmf_gait.bin, built from flygym 1.2.1 by tools/build_fly_model.py) holds, for 64 phases
of one stride plus the standing pose, every geom's pose in the thorax frame: the leg joints at the targets the kit's walker
commanded on 1.2.1 (NeuroMechFly's recorded single step, the tripod's phase offsets included). flygym 2.1 rebuilt the skeleton
with its own joint conventions (every neutral angle differs; the thorax frames and the coxa roots coincide to 0.00 mm), so
the step cannot be copied joint by joint. This tool fits it: per atlas frame and per leg, the seven actuated angles of
2.1's skeleton that put the leg's eight bodies where 1.2.1 had them (L-BFGS-B on MuJoCo's forward kinematics), then
re-indexes the frames so that frame k is EVERY leg at its own step phase 2 pi k / 64 (the atlas is a whole-body tripod
cycle: the legs RF, LM and RH sit half a cycle on from LF; the walker drives each leg by its own oscillator's phase, as
flygym's PreprogrammedSteps did). It also calibrates, on the standing frame, a fixed offset of every 1.2.1 geom in the
2.1 body that carries it (leg segments their own, everything else the thorax), so that the browser's meshes, built from
1.2.1, can follow a 2.1 body in replays (virtual_fly/recording.py); the whole atlas is then rebuilt from the 2.1 skeleton
and compared with itself (0.003 mm and 0.4 deg at most on 2026-10-04).

The 1.2.1 inputs that 2.1 cannot regenerate travel inside the output and are read back from it on the next run: the geoms'
poses within their 1.2.1 bodies, and the recorded step's swing windows (flygym 1.2.1's PreprogrammedSteps.swing_period).

    python tools/refit_stride.py                  # rebuild web/models/nmf_stride.npz from the shipped atlas (a few minutes)
    python tools/refit_stride.py --check          # load the shipped file back and verify it against the atlas on the 2.1 skeleton

The stride is NeuroMechFly v2's (Wang-Chen et al. 2024, Nature Methods, doi:10.1038/s41592-024-02497-y; flygym, Apache-2.0,
web/models/LICENSE-NeuroMechFly.txt and NOTICE-NeuroMechFly.txt).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "virtual_fly" / "web" / "models"
OUT = "nmf_stride.npz"
LEGS_121 = ("LF", "LM", "LH", "RF", "RM", "RH")
SEGS_121 = ("Coxa", "Femur", "Tibia", "Tarsus1", "Tarsus2", "Tarsus3", "Tarsus4", "Tarsus5")
SECOND_TRIPOD = ("RF", "LM", "RH")          # half a cycle on from LF in the atlas (flygym's tripod phase biases)
POS_TOL_MM, ANG_TOL_DEG = 0.02, 1.0         # the check's tolerances on the rebuilt atlas (measured 0.003 mm, 0.4 deg)


def quat_to_mat(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def mat_to_quat(R):
    import mujoco
    q = np.empty(4)
    mujoco.mju_mat2Quat(q, np.ascontiguousarray(R, dtype=np.float64).reshape(9))
    return q


def read_atlas(models: Path):
    meta = json.loads((models / "nmf_gait.json").read_text())
    raw = np.fromfile(models / "nmf_gait.bin", dtype="<f4")
    atlas = raw.reshape(meta["frames"], len(meta["geoms"]), 7).astype(np.float64)
    return meta, atlas


class Skeleton21:
    """flygym 2.1's NeuroMechFly legs, compiled, the thorax pinned at the origin: forward kinematics for the fit."""

    def __init__(self):
        import mujoco
        from virtual_fly import physics
        if not physics.available():
            raise SystemExit("refit_stride.py needs the physics extra: " + physics.unavailable_reason().splitlines()[0])
        from flygym.compose.world.flat_ground import FlatGroundWorld
        from flygym.simulation import Simulation
        from flygym.utils.api1to2 import BODY_NAMES_OLD2NEW
        from flygym.utils.math import Rotation3D
        self.mujoco = mujoco
        fly, dof_names = physics.build_fly("f")
        world = FlatGroundWorld()
        world.add_fly(fly, (0.0, 0.0, 0.0), Rotation3D("quat", (1.0, 0.0, 0.0, 0.0)), add_ground_contact_sensors=False)
        sim = Simulation(world)
        self.m, self.d = sim.mj_model, sim.mj_data
        self.dof_names = dof_names
        act = sim._intern_actuatorids_by_type_by_fly[physics.ActuatorType.POSITION]["f"]
        self.qadr = np.array([int(self.m.jnt_qposadr[self.m.actuator_trnid[a][0]]) for a in act])
        free = int(self.m.jnt_qposadr[0])
        self.d.qpos[free:free + 7] = (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
        self.thorax = self.m.body("f/c_thorax").id
        self.old2new = BODY_NAMES_OLD2NEW
        self.leg_cols = {leg: [i for i, n in enumerate(dof_names) if n.split("-")[1].startswith(leg.lower() + "_")] for leg in LEGS_121}
        self.leg_bodies = {leg: [self.m.body("f/" + BODY_NAMES_OLD2NEW[leg + s]).id for s in SEGS_121] for leg in LEGS_121}
        assert all(len(c) == 7 for c in self.leg_cols.values())

    def body_of_geom(self, old_name: str) -> str:
        """The 2.1 body that carries a 1.2.1 geom: a leg segment's own body, the thorax for everything else (2.1's
        legs-only skeleton has no head, abdomen, wing or antenna joints)."""
        if old_name[:2] in LEGS_121 and old_name[2:] in SEGS_121:
            return self.old2new[old_name]
        return "c_thorax"

    def set_angles(self, angles42):
        self.d.qpos[self.qadr] = angles42
        self.mujoco.mj_kinematics(self.m, self.d)

    def body_pose_in_thorax(self, body: int):
        d = self.d
        Rt = d.xmat[self.thorax].reshape(3, 3)
        return Rt.T @ (d.xpos[body] - d.xpos[self.thorax]), Rt.T @ d.xmat[body].reshape(3, 3)


def old_body_pose(atlas_row, geom_pos121, geom_quat121):
    """A 1.2.1 body's pose in the thorax frame from its geom's atlas pose and the geom's pose within the body."""
    pg, Rg = atlas_row[:3], quat_to_mat(atlas_row[3:])
    Ro = quat_to_mat(geom_quat121)
    Rb = Rg @ Ro.T
    return pg - Rb @ geom_pos121, Rb


def fit(sk: Skeleton21, meta, atlas, g121, quiet=False):
    """Every atlas frame refitted: angles (frames, 42) in the actuator order, and the residuals (frames, 6, 2)."""
    import scipy.optimize
    geoms = meta["geoms"]
    gi = {n: i for i, n in enumerate(geoms)}
    frames, standing = meta["frames"], meta["standing"]
    angles = np.zeros((frames, 42))
    errs = np.zeros((frames, 6, 2))
    key_q = sk.d.qpos[sk.qadr].copy()                        # 2.1's own neutral pose: the first fit starts there
    prev = {leg: key_q[sk.leg_cols[leg]].copy() for leg in LEGS_121}
    order = [standing] + [f for f in range(frames) if f != standing]
    t0 = time.time()
    for n, frame in enumerate(order):
        for li, leg in enumerate(LEGS_121):
            targets = [old_body_pose(atlas[frame, gi[leg + s]], g121["pos"][gi[leg + s]], g121["quat"][gi[leg + s]]) for s in SEGS_121]
            tp = np.array([t[0] for t in targets])
            tq = np.array([mat_to_quat(t[1]) for t in targets])
            cols, bodies = sk.leg_cols[leg], sk.leg_bodies[leg]
            qadr = sk.qadr[cols]

            def cost(x):
                sk.d.qpos[qadr] = x
                sk.mujoco.mj_kinematics(sk.m, sk.d)
                c = 0.0
                for k, b in enumerate(bodies):
                    dp = sk.d.xpos[b] - tp[k]
                    c += 100.0 * float(dp @ dp)                    # mm^2, weighted up: a 0.1 mm error counts about 1
                    dq = abs(float(np.dot(sk.d.xquat[b], tq[k])))
                    c += 1.0 - min(1.0, dq) ** 2
                return c

            r = scipy.optimize.minimize(cost, prev[leg], method="L-BFGS-B", options={"maxiter": 300, "ftol": 1e-10, "gtol": 1e-8})
            sk.d.qpos[qadr] = r.x
            sk.mujoco.mj_kinematics(sk.m, sk.d)
            errs[frame, li, 0] = max(float(np.linalg.norm(sk.d.xpos[b] - tp[k])) for k, b in enumerate(bodies))
            errs[frame, li, 1] = float(np.linalg.norm(sk.d.xpos[bodies[-1]] - tp[-1]))
            angles[frame, cols] = r.x
            prev[leg] = r.x
        if not quiet and n % 8 == 0:
            print(f"  frame {frame}: worst body error {errs[frame, :, 0].max():.3f} mm ({time.time() - t0:.0f} s)", flush=True)
    return angles, errs


def reindex_per_leg(angles, sk: Skeleton21, phases: int):
    """The atlas is a whole-body cycle (frame k: leg LF at phase k, the other tripod half a cycle on): per leg, roll the
    frames so that frame k is that leg at its own phase k."""
    out = angles.copy()
    for leg in SECOND_TRIPOD:
        cols = sk.leg_cols[leg]
        out[:phases, cols] = np.roll(angles[:phases, cols], -phases // 2, axis=0)
    return out


def calibrate_offsets(sk: Skeleton21, meta, atlas, angles_atlas):
    """Every 1.2.1 geom's fixed pose within the 2.1 body that carries it, from the standing frame."""
    geoms = meta["geoms"]
    sk.set_angles(angles_atlas[meta["standing"]])
    bodies, pos, quat = [], np.zeros((len(geoms), 3)), np.zeros((len(geoms), 4))
    for i, g in enumerate(geoms):
        bname = sk.body_of_geom(g)
        pb, Rb = sk.body_pose_in_thorax(sk.m.body("f/" + bname).id)
        pg, Rg = atlas[meta["standing"], i, :3], quat_to_mat(atlas[meta["standing"], i, 3:])
        bodies.append(bname)
        pos[i] = Rb.T @ (pg - pb)
        quat[i] = mat_to_quat(Rb.T @ Rg)
    return bodies, pos, quat


def rebuild_atlas(sk: Skeleton21, meta, angles_atlas, bodies, pos, quat):
    """The atlas as the 2.1 skeleton gives it through the fixed offsets: (frames, geoms, 7)."""
    geoms = meta["geoms"]
    out = np.zeros((meta["frames"], len(geoms), 7))
    bid = [sk.m.body("f/" + b).id for b in bodies]
    Roff = [quat_to_mat(q) for q in quat]
    for k in range(meta["frames"]):
        sk.set_angles(angles_atlas[k])
        poses = {b: sk.body_pose_in_thorax(b) for b in set(bid)}
        for i in range(len(geoms)):
            pb, Rb = poses[bid[i]]
            out[k, i, :3] = pb + Rb @ pos[i]
            out[k, i, 3:] = mat_to_quat(Rb @ Roff[i])
    return out


def compare(a, b):
    """Position (mm) and orientation (deg) errors between two atlases, per frame and geom."""
    pos = np.linalg.norm(a[..., :3] - b[..., :3], axis=-1)
    dots = np.clip(np.abs(np.sum(a[..., 3:] * b[..., 3:], axis=-1)), 0.0, 1.0)
    ang = np.degrees(2 * np.arccos(dots))
    return pos, ang


def atlas_order(angles_per_leg, sk: Skeleton21, phases: int):
    """The inverse of reindex_per_leg: per-leg-phase frames back to the atlas's whole-body frames."""
    out = angles_per_leg.copy()
    for leg in SECOND_TRIPOD:
        cols = sk.leg_cols[leg]
        out[:phases, cols] = np.roll(angles_per_leg[:phases, cols], phases // 2, axis=0)
    return out


def build(models: Path, geoms_from: Path, steps_from: Path, quiet=False) -> dict:
    meta, atlas = read_atlas(models)
    g = np.load(geoms_from, allow_pickle=False)
    g121 = {"names": [str(s) for s in (g["geom_names"] if "geom_names" in g.files else g["names"])],
            "pos": g["geom121_pos"] if "geom121_pos" in g.files else g["pos"],
            "quat": g["geom121_quat"] if "geom121_quat" in g.files else g["quat"],
            "bodies": [str(s) for s in (g["geom121_bodies"] if "geom121_bodies" in g.files else g["bodies"])]}
    if g121["names"] != list(meta["geoms"]):
        raise SystemExit(f"{geoms_from}: its geom names are not the atlas's")
    s = np.load(steps_from, allow_pickle=False)
    swing_start, swing_end = np.asarray(s["swing_start"], dtype=np.float64), np.asarray(s["swing_end"], dtype=np.float64)
    sk = Skeleton21()
    t0 = time.time()
    if not quiet:
        print(f"fitting {meta['frames']} frames x 6 legs on flygym 2.1's skeleton ...", flush=True)
    angles_atlas, errs = fit(sk, meta, atlas, g121, quiet=quiet)
    phases = int(meta["phases"])
    angles = reindex_per_leg(angles_atlas, sk, phases)
    bodies, pos, quat = calibrate_offsets(sk, meta, atlas, angles_atlas)
    rebuilt = rebuild_atlas(sk, meta, angles_atlas, bodies, pos, quat)
    perr, aerr = compare(rebuilt, atlas)
    out = models / OUT
    np.savez_compressed(
        out, angles=angles, dof_names=np.array(sk.dof_names), legs=np.array([l.lower() for l in LEGS_121]),
        phases=phases, standing=int(meta["standing"]), stride_ms=float(meta["stride_ms"]),
        swing_start=swing_start, swing_end=swing_end,
        geom_names=np.array(meta["geoms"]), geom_bodies=np.array(bodies), geom_pos=pos, geom_quat=quat,
        geom121_bodies=np.array(g121["bodies"]), geom121_pos=np.asarray(g121["pos"]), geom121_quat=np.asarray(g121["quat"]),
        body_error_mm=errs[:, :, 0], tarsus5_error_mm=errs[:, :, 1],
        atlas_pos_error_mm=float(perr.max()), atlas_angle_error_deg=float(aerr.max()),
        source=np.array("NeuroMechFly v2's recorded single step (flygym 1.2.1's PreprogrammedSteps, the kit's gait atlas nmf_gait.bin) "
                        "refitted onto flygym 2.1's NeuroMechFly skeleton by tools/refit_stride.py; frame k is every leg at its own "
                        "step phase 2*pi*k/64 (the swing starts at 0), frame 64 the standing pose; the 42 angles are in the order of "
                        "dof_names (2.1's joint conventions); geom_* place the 1.2.1 geoms on the 2.1 bodies. Apache-2.0, see "
                        "NOTICE-NeuroMechFly.txt."),
        built=np.array(time.strftime("%Y-%m-%d")))
    summary = {"frames": meta["frames"], "phases": phases, "seconds": round(time.time() - t0, 1),
               "body_error_mm_max": round(float(errs[:, :, 0].max()), 4), "body_error_mm_median": round(float(np.median(errs[:, :, 0])), 4),
               "tarsus5_error_mm_max": round(float(errs[:, :, 1].max()), 4),
               "atlas_pos_error_mm": round(float(perr.max()), 4), "atlas_angle_error_deg": round(float(aerr.max()), 3),
               "bytes": out.stat().st_size, "out": str(out)}
    return summary


def check(models: Path) -> list:
    """What is wrong with the shipped stride file (empty when nothing is): its names against the 2.1 skeleton's actuators,
    its shape, the swing windows, and the atlas rebuilt from it within POS_TOL_MM and ANG_TOL_DEG."""
    problems = []
    path = models / OUT
    if not path.is_file():
        return [f"{path} is missing"]
    z = np.load(path, allow_pickle=False)
    meta, atlas = read_atlas(models)
    sk = Skeleton21()
    if [str(s) for s in z["dof_names"]] != sk.dof_names:
        problems.append("dof_names differ from flygym 2.1's actuated leg joints in actuator order")
    phases, standing = int(z["phases"]), int(z["standing"])
    if z["angles"].shape != (meta["frames"], 42) or phases != meta["phases"] or standing != meta["standing"]:
        problems.append(f"angles {z['angles'].shape}, phases {phases}, standing {standing}: not the atlas's {meta['frames']} frames")
        return problems
    if not (np.all(z["swing_start"] == 0.0) and np.all((z["swing_end"] > 1.5) & (z["swing_end"] < 3.0))):
        problems.append("the swing windows are not the recorded step's (start 0, end 1.9-2.4 rad)")
    if [str(s) for s in z["geom_names"]] != list(meta["geoms"]):
        problems.append("geom_names differ from the atlas's geoms")
        return problems
    for leg in LEGS_121:                                   # per-leg frames: the two tripods must agree leg for leg with the mirror
        pass
    angles_atlas = atlas_order(z["angles"], sk, phases)
    rebuilt = rebuild_atlas(sk, meta, angles_atlas, [str(b) for b in z["geom_bodies"]], z["geom_pos"], z["geom_quat"])
    perr, aerr = compare(rebuilt, atlas)
    if perr.max() > POS_TOL_MM or aerr.max() > ANG_TOL_DEG:
        problems.append(f"the atlas rebuilt from the 2.1 skeleton is off by {perr.max():.4f} mm and {aerr.max():.3f} deg "
                        f"(tolerance {POS_TOL_MM} mm, {ANG_TOL_DEG} deg)")
    # the stride is left-right symmetric once each leg is at its own phase: the mirror legs' feet travel alike
    bid = {leg: sk.leg_bodies[leg][-1] for leg in LEGS_121}
    tips = {leg: [] for leg in LEGS_121}
    for k in range(phases):
        sk.set_angles(z["angles"][k])
        for leg in LEGS_121:
            tips[leg].append(sk.body_pose_in_thorax(bid[leg])[0])
    for l, r in (("LF", "RF"), ("LM", "RM"), ("LH", "RH")):
        a, b = np.array(tips[l]), np.array(tips[r]) * np.array([1.0, -1.0, 1.0])
        if np.abs(a - b).max() > 0.05:
            problems.append(f"{l} and {r} do not mirror each other once re-indexed (max {np.abs(a - b).max():.3f} mm)")
    return problems


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", type=Path, default=MODELS, help="the folder with nmf_gait.json/.bin, where the output goes")
    ap.add_argument("--geoms-from", type=Path, default=None, help="where the 1.2.1 geoms' poses within their bodies come from "
                                                                   "(default: the shipped nmf_stride.npz)")
    ap.add_argument("--steps-from", type=Path, default=None, help="where the recorded step's swing windows come from (default: the same)")
    ap.add_argument("--check", action="store_true", help="verify the shipped file against the atlas on the 2.1 skeleton instead of building")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    if args.check:
        problems = check(args.models)
        for p in problems:
            print("problem:", p)
        print("ok: the stride file, the atlas and flygym 2.1's skeleton agree" if not problems else f"{len(problems)} problem(s)")
        return 1 if problems else 0
    shipped = args.models / OUT
    s = build(args.models, args.geoms_from or shipped, args.steps_from or args.geoms_from or shipped, quiet=args.quiet)
    print(f"stride refitted in {s['seconds']} s: body error median {s['body_error_mm_median']} mm, max {s['body_error_mm_max']}; "
          f"tarsus5 max {s['tarsus5_error_mm_max']}; the atlas rebuilt from 2.1 within {s['atlas_pos_error_mm']} mm and "
          f"{s['atlas_angle_error_deg']} deg; {s['bytes']:,} bytes -> {s['out']}")
    problems = check(args.models)
    for p in problems:
        print("problem:", p)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
