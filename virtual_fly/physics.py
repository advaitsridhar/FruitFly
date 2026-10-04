"""
A physics body (optional): NeuroMechFly v2 in MuJoCo, through the flygym package (2.1), driven by the kit's
motor decoder. A drop-in for :class:`virtual_fly.body.FlyBody` (the drawn, kinematic body).

    decoder drives (forward, yaw, backward, halt)  --descending_drive()-->  (left, right) stepping drive
    --> six coupled oscillators (12 Hz tripod) --> the recorded single step per leg --> 42 leg joints (position servos)
    + tarsal adhesion --> MuJoCo contacts with the floor --> where the thorax goes

The interface is flygym's own two-sided descending drive (Wang-Chen et al. 2024, Nat Methods,
doi:10.1038/s41592-024-02497-y): one number per body side scales the amplitude of that side's stepping
(its sign is the stepping direction). Steering uses flygym's published constants for connectome-driven
following (flygym 1.2.1's examples/vision/follow_fly_closed_loop.py): the inner side is attenuated by 0.6|s|
(to 0.4 at most) and the outer side lengthened by 0.2|s| (to 1.2 at most). These are the two steering
gestures of Yang et al. 2024 (Cell, doi:10.1016/j.cell.2024.08.033): DNa02 shortens ipsilateral strides,
DNg13 lengthens contralateral ones. Steering therefore modulates an ongoing rhythm: a fly that is not
stepping does not turn (the drawn body turns in place).

The body (v3.0, the real-time route of docs/TWO_FLIES_PROGRESS.md): flygym 2.1's NeuroMechFly on MuJoCo 3.9, composed
here from flygym's parts (its legs-only skeleton: 66 hinges, 42 of them actuated, the four passive tarsal joints per
leg; position actuators; tarsal adhesion; its flat ground) plus the kit's wall. One fly in the game runs at real time on a
laptop and two at half (measured: 1.04 and 0.49), where flygym 1.2.1 gave 0.16 and 0.085. What carries over from the body
the kit validated on 1.2.1
(docs/SCIENCE.md 6.7): the joints' spring and damping (0.05 / 0.06 on the actuated joints, 7.5 / 0.01 on the passive
tarsi), the servos (kp 45, force +-65) and the adhesion (40), because 2.1's own stiffer joints (10 / 0.5) hold the
recorded stride back (measured: 10.3 against 14.3 mm/s at full drive); the stride itself, NeuroMechFly's recorded
single step, refitted onto 2.1's skeleton (web/models/nmf_stride.npz, tools/refit_stride.py: every leg body within
0.014 mm of where 1.2.1 put it); and the oscillators, flygym 1.2.1's CPGNetwork equations (Apache-2.0, carried in
:class:`CPG`, credited in web/models/NOTICE-NeuroMechFly.txt), since flygym 2.x ships no controller.

Nothing here is fitted to the kit. What stays hand-built: the decoder's weights (game.py), the mapping's
forward term (the drawn body's), and the proboscis, wings and abdomen (still drawn: the skeleton has no joints
there). Not modelled: the escape jump (the giant fibre still fires; the game then reports the escape command).

Install (Python 3.12-3.14; flygym 2.1.0 brings MuJoCo 3.9, scipy and numba; opencv is for tools/render_replay.py)::

    pip install -e ".[physics]"          # or: pip install "flygym==2.1.0" "opencv-python-headless>=4"

No OpenGL is needed: rendering is switched off (MUJOCO_GL=disable).
"""

from __future__ import annotations

import math
import os
import sys
import time
from pathlib import Path

import numpy as np

from .body import FlyBody, Pose
from .world import ARENA_R, wrap

os.environ.setdefault("MUJOCO_GL", "disable")          # physics only; set MUJOCO_GL=egl/glfw yourself to render

try:
    import mujoco
    from flygym.anatomy import ActuatedDOFPreset, JointPreset, Skeleton
    from flygym.compose.fly.base_fly import ActuatorType, GeomFittingOption
    from flygym.compose.fly.neuromechfly import NeuroMechFly
    from flygym.compose.physics import ContactParams
    from flygym.compose.pose import KinematicPosePreset
    from flygym.compose.world.flat_ground import FlatGroundWorld
    from flygym.simulation import Simulation
    from flygym.utils.api1to2 import BODY_NAMES_OLD2NEW
    from flygym.utils.math import Rotation3D
    _IMPORT_ERROR: Exception | None = None
except Exception as e:                                   # pragma: no cover - exercised where flygym is missing
    _IMPORT_ERROR = e

INSTALL_HINT = ('the physics body needs Python 3.12-3.14 with flygym 2.1 and MuJoCo 3.9: pip install -e ".[physics]" '
                '(or pip install "flygym==2.1.0" "opencv-python-headless>=4")')

INNER_ATTENUATION = 0.6      # flygym follow_fly_closed_loop.py: inner = max(0.4, 1 - 0.6 |s|)
OUTER_BOOST = 0.2            # flygym follow_fly_closed_loop.py: outer = min(1.2, 1 + 0.2 |s|)
TIMESTEP = 1e-4              # s, flygym's default (its contacts have a 2e-4 s time constant)
CONTROL_MS = 0.5             # the leg controller runs every 0.5 ms (5 steps at 0.1 ms, 2 at 0.2: 170 updates per 83 ms stride):
#                              the same motion as every step (speed, wobble and turn within 1 %, measured), a fifth of the Python
SWING_EXTENSION = math.pi / 4   # flygym HybridTurningController._init_phasic_gain: adhesion stays off pi/4 longer

# the body's settings: flygym 1.2.1's Fly defaults, which the body the kit validated had (docs/SCIENCE.md 6.7)
JOINT_SPRING = (0.05, 0.06)  # stiffness, damping of the 42 actuated leg joints
TARSUS_SPRING = (7.5, 0.01)  # of the 24 passive tarsal joints (tarsus2-5)
KP, FORCERANGE, ADHESION = 45.0, 65.0, 40.0     # position servos' gain and force limit; adhesion force per foot (the fly weighs ~10)
SPAWN_Z = 0.0                # the spawn site's height: flygym's neutral keyframe then stands the fly on the ground
FLY_NAME = "nmf"             # the single body's name in its model (the pair's are fly0, fly1, ...)
THORAX = "c_thorax"          # flygym 2.1's name for the thorax body, the skeleton's root (1.2.1: Thorax)
LEGS = ("lf", "lm", "lh", "rf", "rm", "rh")           # flygym 2.1's leg order; 1.2.1 wrote them LF, LM, LH, RF, RM, RH
LEGS_121 = ("LF", "LM", "LH", "RF", "RM", "RH")
WALL_SEGMENTS = 48
# the wall pairs' sliding friction, hand-set (flygym's ground has 1.0): at 1.0 the 2.1 body's forelegs climb the 3 mm wall
# and the fly falls on its side within half a second of walking into it; at 0 the contact solver blows up; at 0.3 the fly
# slides along the wall on its feet (measured on four approaches, docs/SCIENCE.md 6.7). The drawn body has no wall physics.
WALL_FRICTION = 0.3
STRIDE_FILE = Path(__file__).resolve().parent / "web" / "models" / "nmf_stride.npz"

# flygym 1.2.1's tripod (examples/locomotion/turning_controller.py): legs 0, 2, 4 (LF, LH, RM) step together, the others
# half a cycle on; every coupled pair weighs 10
TRIPOD_PHASE_BIASES = math.pi * np.array([[0, 1, 0, 1, 0, 1], [1, 0, 1, 0, 1, 0], [0, 1, 0, 1, 0, 1],
                                          [1, 0, 1, 0, 1, 0], [0, 1, 0, 1, 0, 1], [1, 0, 1, 0, 1, 0]], dtype=np.float64)
TRIPOD_COUPLING = (TRIPOD_PHASE_BIASES > 0) * 10.0
CPG_FREQ_HZ = 12.0
CPG_CONVERGENCE = 20.0


def available() -> bool:
    return _IMPORT_ERROR is None


def unavailable_reason() -> str:
    """Why the physics body cannot start: the install hint and the import that failed (say, mujoco missing)."""
    major, minor = sys.version_info[:2]
    if (major, minor) < (3, 12) or (major, minor) >= (3, 15):
        return (f"the physics body needs Python 3.12-3.14 (flygym 2.1 installs on no other), and this is Python {major}.{minor}: "
                f"make a virtual environment with Python 3.12 for it (docs/SCIENCE.md section 6.7)")
    return f"{INSTALL_HINT}\n(the import failed with: {_IMPORT_ERROR!r})"


def descending_drive(mode: str, drive: dict, wander_yaw: float = 0.0) -> tuple[float, float]:
    """The decoder's drives -> flygym's (left, right) stepping drive. Pure function (no flygym needed).

    The stepping drive f uses the drawn body's expression for its target speed; the steering s (+ = right)
    the drawn body's dead band. Feeding, grooming and the escape leave the legs standing."""
    if mode == "backward":
        f, s = -min(1.0, drive["backward"]), 0.5 * drive["yaw"]
    elif mode in ("walk", "idle", "court"):
        s = drive["yaw"] + wander_yaw
        s = 0.0 if abs(s) < 0.05 else max(-1.0, min(1.0, s))
        f = drive["forward"] * (1 - 0.7 * drive["halt"])
    else:
        return 0.0, 0.0
    inner = f * max(0.4, 1 - INNER_ATTENUATION * abs(s))
    outer = f * min(1.2, 1 + OUTER_BOOST * abs(s))
    if f < 0:                                  # backing up: turn the same way the drawn body does
        inner, outer = outer, inner
    if s > 0:
        return outer, inner                    # right turn: the right legs are on the inside
    if s < 0:
        return inner, outer
    return f, f


# the kit names body parts as flygym 1.2.1 did (the 3-D view's atlas and meshes carry those names); part_geom maps a
# name to the geom (and body) flygym 2.1 gives the same part
WALL_TOUCHERS = ("Head", "Thorax") + tuple(f"{leg}{seg}" for leg in LEGS_121 for seg in ("Tibia", "Tarsus1"))


def part_geom(part: str) -> str:
    """A 1.2.1 part name (Head, Thorax, LFTibia, ...) -> flygym 2.1's name for its geom and body (c_head, lf_tibia, ...)."""
    return BODY_NAMES_OLD2NEW[part]


# ---------------------------------------------------------------------------------------------- speed levers
# Each lever changes the physics, so each is a switch that is OFF unless asked for (fly_game.py --physics-levers, or
# tools/physics_table.py --levers): measured against docs/SCIENCE.md 6.7's table with tools/physics_table.py before it
# is ever adopted, and adopted only within the agreed tolerance (docs/TWO_FLIES_PLAN.md 8.6, decision 25). On flygym 2.1
# three of them describe what the model already is (docs/SCIENCE.md 13.5): they are accepted and change nothing.
LEVERS = {
    "dedupe": "keep one of each self-collision pair (flygym 1.2.1 added every pair twice; 2.1 adds none: nothing to drop)",
    "solver100": "the Newton solver at 100 iterations and tolerance 1e-8 (flygym 2.1's own setting: no change)",
    "noslip5": "5 noslip iterations (flygym 2.1's own setting: no change)",
    "noslip0": "noslip iterations 5 -> 0 (adhesion and leg slip depend on them)",
    "noself": "no self-collision pairs (flygym 2.1's legs-only skeleton has none: no change; legs may pass through each other)",
    "simple": "every body part a capsule instead of a mesh (flygym 2.1's ALL_TO_CAPSULES fitting; 1.2.1: the seqik_simple model)",
    "dt2": "a 0.2 ms time step instead of 0.1 ms",
}
NO_OP_LEVERS = ("dedupe", "solver100", "noslip5", "noself")   # what flygym 2.1's model already is

DEFAULT_LEVERS = ()   # nothing: the 1.2.1 default `dedupe` (docs/SCIENCE.md 13.5) has nothing to drop on 2.1


def parse_levers(levers) -> tuple[str, ...]:
    """A comma list (or any sequence) of lever names -> a tuple in LEVERS order, each name checked; nothing, or the word
    "none", -> ()."""
    if not levers:
        return ()
    names = [s.strip() for s in (levers.split(",") if isinstance(levers, str) else levers) if s and str(s).strip()]
    if names == ["none"]:
        return ()
    bad = [n for n in names if n not in LEVERS]
    if bad:
        raise ValueError(f"unknown physics lever{'s' if len(bad) > 1 else ''} {', '.join(bad)}: choose from {', '.join(LEVERS)}")
    if "noslip5" in names and "noslip0" in names:
        raise ValueError("noslip5 and noslip0 exclude each other")
    return tuple(n for n in LEVERS if n in names)


def lever_timestep(levers, timestep: float = TIMESTEP) -> float:
    return 2e-4 if "dt2" in levers else timestep


def lever_fly_kwargs(levers) -> dict:
    """What a lever changes in flygym's NeuroMechFly(...) call."""
    kw = {}
    if "simple" in levers:
        kw["geom_fitting_option"] = "all_to_capsules"
    return kw


def apply_levers(model, levers) -> None:
    """The levers that act on the compiled model's options (model: a mujoco.MjModel)."""
    if "solver100" in levers:
        model.opt.iterations, model.opt.tolerance = 100, 1e-8
    if "noslip5" in levers:
        model.opt.noslip_iterations = 5
    if "noslip0" in levers:
        model.opt.noslip_iterations = 0


# ---------------------------------------------------------------------------------------------- the stride and the oscillators
class Stride:
    """NeuroMechFly's recorded single step on flygym 2.1's skeleton (web/models/nmf_stride.npz, tools/refit_stride.py):
    ``angles`` (65, 42), frame k every leg at its own step phase 2 pi k / 64, frame 64 the standing pose; ``neutral`` (6, 7)
    and ``offsets`` (64, 6, 7) per leg in flygym 2.1's leg and joint order (coxa pitch, roll, yaw, trochanterfemur pitch,
    roll, tibia pitch, tarsus1 pitch); the swing windows (the swing starts at phase 0); and where each of the 3-D view's
    geoms sits on the 2.1 bodies (``geom_bodies``, ``geom_pos``, ``geom_quat``), for the replays."""

    def __init__(self, path: Path = STRIDE_FILE):
        z = np.load(path, allow_pickle=False)
        self.path = Path(path)
        self.angles = np.asarray(z["angles"], dtype=np.float64)
        self.n = int(z["phases"])
        self.standing = int(z["standing"])
        self.dof_names = [str(s) for s in z["dof_names"]]
        self.neutral = self.angles[self.standing].reshape(6, 7)
        self.offsets = (self.angles[:self.n] - self.angles[self.standing]).reshape(self.n, 6, 7)
        self.swing_start = np.asarray(z["swing_start"], dtype=np.float64)
        self.swing_end = np.asarray(z["swing_end"], dtype=np.float64)
        self.stride_ms = float(z["stride_ms"])
        self.geoms = [str(s) for s in z["geom_names"]]
        self.geom_bodies = [str(s) for s in z["geom_bodies"]]
        self.geom_pos = np.asarray(z["geom_pos"], dtype=np.float64)
        self.geom_quat = np.asarray(z["geom_quat"], dtype=np.float64)
        self.source = str(z["source"])


_STRIDE: Stride | None = None


def stride() -> Stride:
    """The shipped stride, loaded once."""
    global _STRIDE
    if _STRIDE is None:
        _STRIDE = Stride()
    return _STRIDE


class CPG:
    """flygym 1.2.1's CPGNetwork (examples/locomotion/cpg_controller.py, Apache-2.0, Wang-Chen et al. 2024), carried here
    because flygym 2.x ships no controller. N phase oscillators with phases theta and magnitudes r::

        dtheta_i/dt = 2 pi nu_i + sum_j r_j w_ij sin(theta_j - theta_i - phi_ij)      dr_i/dt = alpha_i (R_i - r_i)

    stepped by Euler's method; at a reset the phases are random (the seeded stream) and the magnitudes zero. The attribute
    names are flygym's (intrinsic_freqs, intrinsic_amps, curr_phases, curr_magnitudes)."""

    def __init__(self, timestep: float, intrinsic_freqs, intrinsic_amps, coupling_weights, phase_biases, convergence_coefs, seed: int = 0):
        self.timestep = float(timestep)
        self.intrinsic_freqs = np.asarray(intrinsic_freqs, dtype=np.float64)
        self.intrinsic_amps = np.asarray(intrinsic_amps, dtype=np.float64)
        self.coupling_weights = np.asarray(coupling_weights, dtype=np.float64)
        self.phase_biases = np.asarray(phase_biases, dtype=np.float64)
        self.convergence_coefs = np.asarray(convergence_coefs, dtype=np.float64)
        self.num_cpgs = self.intrinsic_freqs.size
        self.random_state = np.random.RandomState(seed)
        self.reset(None, None)

    def reset(self, init_phases=None, init_magnitudes=None):
        self.curr_phases = (self.random_state.random(self.num_cpgs) * 2 * np.pi if init_phases is None
                            else np.array(init_phases, dtype=np.float64))
        self.curr_magnitudes = np.zeros(self.num_cpgs) if init_magnitudes is None else np.array(init_magnitudes, dtype=np.float64)

    def step(self):
        theta, r = self.curr_phases, self.curr_magnitudes
        phase_diff = theta[np.newaxis, :] - theta[:, np.newaxis]
        dtheta_dt = 2 * np.pi * self.intrinsic_freqs + (r * self.coupling_weights * np.sin(phase_diff - self.phase_biases)).sum(axis=1)
        dr_dt = self.convergence_coefs * (self.intrinsic_amps - r)
        self.curr_phases += dtheta_dt * self.timestep
        self.curr_magnitudes += dr_dt * self.timestep


# ---------------------------------------------------------------------------------------------- building the world
def build_fly(name: str, levers=()):
    """flygym 2.1's NeuroMechFly with the kit's body settings: the legs-only skeleton at flygym's neutral pose, the joint
    springs (the actuated joints JOINT_SPRING, the passive tarsi TARSUS_SPRING), 42 position servos and six adhesion
    actuators. Returns (fly, the actuated joints' names in actuator order)."""
    kw = lever_fly_kwargs(levers)
    if "geom_fitting_option" in kw:
        kw["geom_fitting_option"] = GeomFittingOption(kw["geom_fitting_option"])
    fly = NeuroMechFly(name=name, **kw)
    skeleton = Skeleton(axis_order=fly.AXIS_ORDER_CLASS.DONTCARE, joint_preset=JointPreset.LEGS_ONLY)
    joints = fly.add_joints(skeleton, neutral_pose=KinematicPosePreset.NEUTRAL)
    dofs = skeleton.get_actuated_dofs_from_preset(ActuatedDOFPreset.LEGS_ACTIVE_ONLY)
    active = set(dofs)
    for dof, joint in joints.items():
        k, c = JOINT_SPRING if dof in active else TARSUS_SPRING
        joint.stiffness = np.array([k, 0.0, 0.0])
        joint.damping = np.array([c, 0.0, 0.0])
    fly.add_actuators(dofs, ActuatorType.POSITION, neutral_input=KinematicPosePreset.NEUTRAL, kp=KP, forcerange=(-FORCERANGE, FORCERANGE))
    fly.add_leg_adhesion(gain=ADHESION)
    return fly, [dof.name for dof in dofs]


def add_wall(spec, center=(0.0, 0.0), r: float = ARENA_R, n: int = WALL_SEGMENTS) -> None:
    """The kit's round arena wall on a world's MjSpec: n box segments, 3 mm high, that collide with nothing until pairs
    name them (contype and conaffinity 0, as flygym gives every fly geom)."""
    seg = 2 * math.pi * r / n
    for i in range(n):
        a = 2 * math.pi * i / n
        spec.worldbody.add_geom(name=f"wall_{i}", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.5, seg / 2 + 0.2, 1.5],
                                pos=[center[0] + (r + 0.5) * math.cos(a), center[1] + (r + 0.5) * math.sin(a), 1.5],
                                quat=[math.cos(a / 2), 0.0, 0.0, math.sin(a / 2)], contype=0, conaffinity=0,
                                rgba=[0.6, 0.6, 0.6, 1.0])


def add_wall_pairs(spec, fly_name: str, n: int = WALL_SEGMENTS) -> int:
    """Contact pairs between one fly's head, thorax, tibiae and first tarsi and every wall segment, with flygym 2.1's
    ground contact parameters (as its own ground pairs; explicit pairs, like flygym's floor contacts: making the fly's
    geoms collidable instead would change every contact)."""
    cp = ContactParams(sliding_friction=WALL_FRICTION)
    count = 0
    for i in range(n):
        for part in WALL_TOUCHERS:
            spec.add_pair(geomname1=f"{fly_name}/{part_geom(part)}", geomname2=f"wall_{i}", name=f"wall_{i}_{fly_name}_{part}",
                          friction=cp.get_friction_tuple(), solref=cp.get_solref_tuple(), solimp=cp.get_solimp_tuple(),
                          margin=cp.margin)
            count += 1
    return count


def spawn_rotation(heading: float):
    """A yaw (rad, counter-clockwise from +x) as flygym's spawn rotation."""
    return Rotation3D("quat", (math.cos(heading / 2), 0.0, 0.0, math.sin(heading / 2)))


def _qpos_size(m, j: int) -> int:
    return {mujoco.mjtJoint.mjJNT_FREE: 7, mujoco.mjtJoint.mjJNT_BALL: 4}.get(mujoco.mjtJoint(m.jnt_type[j]), 1)


def _dof_size(m, j: int) -> int:
    return {mujoco.mjtJoint.mjJNT_FREE: 6, mujoco.mjtJoint.mjJNT_BALL: 3}.get(mujoco.mjtJoint(m.jnt_type[j]), 1)


class FlyRig:
    """One fly in a compiled model: its oscillators, its actuator and adhesion ids (checked against the stride's joint
    order), its thorax and tarsus bodies, its geoms, its block of qpos and qvel (the free joint first), and the leg
    controller: per step the oscillators advance, each leg's targets are ``neutral + magnitude x step(phase)`` and its
    adhesion is on except during the swing plus SWING_EXTENSION."""

    def __init__(self, sim, name: str, seed: int, st: Stride):
        m = sim.mj_model
        self.name, self.seed, self.stride = name, seed, st
        self.control_steps = max(1, int(round(CONTROL_MS / 1000.0 / m.opt.timestep)))   # physics steps per controller update
        self._i = 0
        self.act = np.asarray(sim._intern_actuatorids_by_type_by_fly[ActuatorType.POSITION][name], dtype=np.int64)
        names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, int(a)) for a in self.act]
        want = [f"{name}/{d}-position" for d in st.dof_names]
        if names != want:
            raise RuntimeError(f"fly {name}: its actuators are not in the stride file's joint order ({names[:3]} ...)")
        self.adh = np.asarray(sim._intern_adhesionactuatorids_by_fly[name], dtype=np.int64)
        adh_names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, int(a)) for a in self.adh]
        if adh_names != [f"{name}/{leg}_tarsus5-adhesion" for leg in LEGS]:
            raise RuntimeError(f"fly {name}: its adhesion actuators are not in leg order ({adh_names})")
        self.thorax = m.body(f"{name}/{THORAX}").id
        self.tarsi = [m.body(f"{name}/{leg}_tarsus5").id for leg in LEGS]
        self.root = int(m.body_rootid[self.thorax])
        self.geoms = frozenset(g for g in range(m.ngeom) if m.body_rootid[m.geom_bodyid[g]] == self.root)
        joints = [j for j in range(m.njnt) if m.body_rootid[m.jnt_bodyid[j]] == self.root]
        free = int(m.body_jntadr[self.root])
        if not joints or joints[0] != free or m.jnt_type[free] != mujoco.mjtJoint.mjJNT_FREE \
                or joints != list(range(joints[0], joints[-1] + 1)):
            raise RuntimeError(f"fly {name}: its joints are not one block starting with its free joint")
        last = joints[-1]
        self.qpos = slice(int(m.jnt_qposadr[free]), int(m.jnt_qposadr[last]) + _qpos_size(m, last))
        self.qvel = slice(int(m.jnt_dofadr[free]), int(m.jnt_dofadr[last]) + _dof_size(m, last))
        self.cpg = CPG(m.opt.timestep * self.control_steps, np.ones(6) * CPG_FREQ_HZ, np.zeros(6), TRIPOD_COUPLING,
                       TRIPOD_PHASE_BIASES, np.ones(6) * CPG_CONVERGENCE, seed=seed)
        self._sw0 = st.swing_start
        self._sw1 = st.swing_end + SWING_EXTENSION
        self._six = np.arange(6)
        self._neutral_flat = st.neutral.ravel()

    def reset_control(self, d) -> None:
        """The oscillators back to their seeded start, the legs at the standing pose, adhesion on."""
        self.cpg.random_state = np.random.RandomState(self.seed)
        self.cpg.reset(None, None)
        self.cpg.intrinsic_amps = np.zeros(6)
        self.cpg.intrinsic_freqs = np.ones(6) * CPG_FREQ_HZ
        self._i = 0
        d.ctrl[self.act] = self._neutral_flat
        d.ctrl[self.adh] = 1.0

    def set_drive(self, left: float, right: float) -> None:
        self.cpg.intrinsic_amps = np.repeat(np.abs([left, right]), 3)
        self.cpg.intrinsic_freqs = np.repeat([CPG_FREQ_HZ if left > 0 else -CPG_FREQ_HZ, CPG_FREQ_HZ if right > 0 else -CPG_FREQ_HZ], 3)

    def write_controls(self, ctrl) -> None:
        """Called before every physics step: every ``control_steps`` steps the oscillators advance by that much and the 42
        targets and the 6 adhesion states are written; in between the servos hold their targets."""
        self._i += 1
        if self._i < self.control_steps:
            return
        self._i = 0
        cpg, st = self.cpg, self.stride
        cpg.step()
        ph = cpg.curr_phases % (2 * np.pi)
        u = ph / (2 * np.pi) * st.n
        i0 = u.astype(np.int64) % st.n
        i1 = (i0 + 1) % st.n
        fr = (u - np.floor(u))[:, None]
        off = st.offsets[i0, self._six] * (1 - fr) + st.offsets[i1, self._six] * fr
        ctrl[self.act] = (st.neutral + cpg.curr_magnitudes[:, None] * off).ravel()
        ctrl[self.adh] = ~((self._sw0 < ph) & (ph < self._sw1))


class Walker:
    """One NeuroMechFly on flygym's flat ground (with the kit's wall when a centre is given), stepping MuJoCo directly:
    flygym 1.2.1's CPG controller (six phase oscillators with tripod coupling, the recorded single step per leg,
    adhesion on during stance, the swing's adhesion-off window pi/4 longer, as flygym's HybridTurningController had it),
    without that controller's stumbling and retraction rules (they act on rough terrain and need a full observation
    every 0.1 ms). The drive's 0.6/0.2 steering constants are the same controller's."""

    def __init__(self, timestep: float = TIMESTEP, seed: int = 0, wall_center=None, levers=()):
        if _IMPORT_ERROR is not None:
            raise RuntimeError(unavailable_reason()) from _IMPORT_ERROR
        self.levers = levers = parse_levers(levers)
        st = stride()
        self.fly, _dofs = build_fly(FLY_NAME, levers)
        self.world = FlatGroundWorld()
        if wall_center is not None:
            add_wall(self.world.mjcf_root, center=wall_center)
        self.world.add_fly(self.fly, (0.0, 0.0, SPAWN_Z), spawn_rotation(0.0), add_ground_contact_sensors=False)
        self.wall_pairs = add_wall_pairs(self.world.mjcf_root, FLY_NAME) if wall_center is not None else 0
        self.sim = Simulation(self.world, timestep=lever_timestep(levers, timestep))
        self._m, self._d = self.sim.mj_model, self.sim.mj_data
        apply_levers(self._m, levers)
        self.pairs_dropped = 0                 # flygym 2.1 adds each contact pair once: `dedupe` finds nothing to drop
        self.timestep, self.seed = float(self._m.opt.timestep), seed
        self.rig = FlyRig(self.sim, FLY_NAME, seed, st)
        self.cpg = self.rig.cpg
        self._act, self._adh, self._thorax, self._tarsi = self.rig.act, self.rig.adh, self.rig.thorax, self.rig.tarsi
        names = [mujoco.mj_id2name(self._m, mujoco.mjtObj.mjOBJ_GEOM, i) or "" for i in range(self._m.ngeom)]
        self._wall_geoms = {i for i, nm in enumerate(names) if nm.startswith("wall_")}
        self.reset()

    def reset(self):
        self.sim.reset()                       # flygym's neutral keyframe: standing at the spawn site
        self.rig.reset_control(self._d)
        mujoco.mj_forward(self._m, self._d)
        self.set_drive(0.0, 0.0)

    def set_drive(self, left: float, right: float):
        self.rig.set_drive(left, right)

    def advance(self, n_steps: int):
        d, m, rig, ctrl = self._d, self._m, self.rig, self._d.ctrl
        for _ in range(n_steps):
            rig.write_controls(ctrl)
            mujoco.mj_step(m, d)

    def thorax(self):
        """(x, y, z) of the thorax in mm and its heading (rad, counter-clockwise from +x)."""
        d = self._d
        xm = d.xmat[self._thorax]
        return np.array(d.xpos[self._thorax]), math.atan2(xm[3], xm[0])

    def tarsi_xy(self) -> np.ndarray:
        return np.array(self._d.xpos[self._tarsi])[:, :2]

    def touching_wall(self) -> bool:
        d = self._d
        return any(c.geom1 in self._wall_geoms or c.geom2 in self._wall_geoms for c in d.contact[:d.ncon])

    def close(self):
        sim, self.sim = self.sim, None
        if sim is not None:
            try:
                sim.close()
            except Exception:
                pass


class PhysicsBody:
    """Same interface as :class:`virtual_fly.body.FlyBody` (pose, move, reset, start_jump, to_dict), so the
    game can use either. The physical fly is real size; the kit draws it three times larger, and the senses
    keep the drawn geometry (mouth and forelegs ahead of the thorax)."""

    kind = "physics"

    def __init__(self, world, rng, timestep: float = TIMESTEP, seed: int = 0, wall: bool = True,
                 stride_average: bool = False, levers=()):
        self.world, self.rng = world, rng
        self.levers = parse_levers(levers)             # speed levers (LEVERS): none unless asked for
        timestep = lever_timestep(self.levers, timestep)
        # stride_average (off by default): the senses see the thorax pose averaged over the last stride (1/12 s,
        # the CPG's period) instead of the stride-by-stride body yaw wobble; a hand-built stand-in for gaze
        # stabilisation during walking (Cruz et al. 2021, doi:10.1016/j.cub.2021.08.041). On one seed it removed
        # the quiet-arena high state; on five it did not (docs/SCIENCE.md 6.7), so the senses see the body as it is.
        self.stride_average = stride_average
        self._hist: list = []
        self.timestep = timestep
        self._seed, self._wall = seed, wall
        self.walker = None
        self.pose = Pose()
        self.jump_lock = 0.0
        self.distance = 0.0
        self.bumped = False
        self.drive_lr = (0.0, 0.0)
        self.wall_s = 0.0                          # wall-clock seconds spent in MuJoCo
        self.reset()

    _spawn = {}            # the thorax's spawn pose in MuJoCo (x, y, heading) per lever set: the same in every arena, measured once

    def reset(self, x=0.0, y=-12.0, h=math.pi / 2):
        # the physics world is built around the fly's start: the arena centre is where the kit says it is. The
        # thorax spawns a little off MuJoCo's origin (about 0.5 mm with flygym's model), so the wall is placed from
        # the measured spawn pose: kit = start + R(h - spawn heading) (p - spawn position), solved for kit = (0, 0).
        if self.levers not in PhysicsBody._spawn:
            first = Walker(self.timestep, self._seed, wall_center=None, levers=self.levers)
            pos, hd = first.thorax()
            PhysicsBody._spawn[self.levers] = (float(pos[0]), float(pos[1]), float(hd))
            if not self._wall:
                self.walker = first
            else:
                first.close()
        sx, sy, sh = PhysicsBody._spawn[self.levers]
        r = -(h - sh)
        c = (round(sx - x * math.cos(r) + y * math.sin(r), 9), round(sy - x * math.sin(r) - y * math.cos(r), 9))
        if self.walker is None or (self._wall and getattr(self, "_center", None) != c):
            if self.walker is not None:
                self.walker.close()
            self.walker = Walker(self.timestep, self._seed, wall_center=c if self._wall else None, levers=self.levers)
            self._center = c
        else:
            self.walker.reset()
        self.pose = Pose(x=x, y=y, h=h)
        self.jump_lock, self.distance = 0.0, 0.0
        self._hist = []
        pos, hd = self.walker.thorax()                 # the thorax's spawn pose is the kit's (x, y, h)
        self._p0, self._rot = pos[:2].copy(), h - hd
        self._origin = (x, y, h)

    def _to_kit(self, xy):
        x0, y0, _ = self._origin
        c, s = math.cos(self._rot), math.sin(self._rot)
        dx, dy = xy[0] - self._p0[0], xy[1] - self._p0[1]
        return x0 + c * dx - s * dy, y0 + s * dx + c * dy

    def start_jump(self, away_from):
        self.jump_lock = 0.8                       # no physical jump: NeuroMechFly has no jump model

    def move(self, dt: float, mode: str, drive: dict, wander_yaw: float = 0.0):
        p = self.pose
        self.jump_lock -= dt
        p.mode = mode
        self.drive_lr = descending_drive(mode, drive, wander_yaw)
        self.walker.set_drive(*self.drive_lr)
        t0 = time.perf_counter()
        n = int(round(dt / self.timestep))
        if self.stride_average:
            k = max(1, int(round(0.0025 / self.timestep)))           # sample the thorax every 2.5 ms
            for i in range(0, n, k):
                self.walker.advance(min(k, n - i))
                pos, hd = self.walker.thorax()
                self._hist.append((pos.copy(), hd))
            keep = max(1, int(round((1 / CPG_FREQ_HZ) / (k * self.timestep))))
            self._hist = self._hist[-keep:]
            hs = np.unwrap([q[1] for q in self._hist])
            pos = np.mean([q[0] for q in self._hist], axis=0)
            hd = float(np.mean(hs))
        else:
            self.walker.advance(n)
            pos, hd = self.walker.thorax()
        self.wall_s += time.perf_counter() - t0
        x, y = self._to_kit(pos[:2])
        h = wrap(hd + self._rot)
        dx, dy = x - p.x, y - p.y
        p.v = (dx * math.cos(p.h) + dy * math.sin(p.h)) / dt
        p.w = wrap(h - p.h) / dt
        self.distance += math.hypot(dx, dy)
        p.x, p.y, p.h = x, y, h
        p.leg_phase = float(self.walker.cpg.curr_phases[0] % (2 * math.pi)) / (2 * math.pi)
        self.bumped = self._wall and self.walker.touching_wall()
        self._z = float(pos[2])
        FlyBody._appendages(self, dt, mode, drive)           # proboscis, wings, abdomen: still drawn

    def to_dict(self) -> dict:
        d = FlyBody.to_dict(self)
        d["physics"] = {"left": round(self.drive_lr[0], 3), "right": round(self.drive_lr[1], 3),
                        "z": round(getattr(self, "_z", 0.0), 3),
                        "tarsi": [[round(a, 2), round(b, 2)] for a, b in (self._to_kit(t) for t in self.walker.tarsi_xy())]}
        return d


def make_body(kind: str, world, rng, seed: int = 0, stride_average: bool = False, levers=()):
    """``drawn`` (the default kinematic body) or ``physics`` (this module; needs flygym). ``levers``: speed levers (LEVERS),
    none unless asked for."""
    if kind == "drawn":
        return FlyBody(world, rng)
    if kind == "physics":
        if not available():
            raise RuntimeError(unavailable_reason()) from _IMPORT_ERROR
        return PhysicsBody(world, rng, seed=seed, stride_average=stride_average, levers=levers)
    raise ValueError("body must be 'drawn' or 'physics'")
