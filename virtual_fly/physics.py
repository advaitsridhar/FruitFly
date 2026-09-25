"""
A physics body (optional): NeuroMechFly v2 in MuJoCo, through the flygym package, driven by the kit's
motor decoder. A drop-in for :class:`virtual_fly.body.FlyBody` (the drawn, kinematic body).

    decoder drives (forward, yaw, backward, halt)  --descending_drive()-->  (left, right) stepping drive
    --> six coupled oscillators (12 Hz tripod) --> recorded single steps --> 42 leg joints (position servos)
    + tarsal adhesion --> MuJoCo contacts with the floor --> where the thorax goes

The interface is flygym's own two-sided descending drive (Wang-Chen et al. 2024, Nat Methods,
doi:10.1038/s41592-024-02497-y): one number per body side scales the amplitude of that side's stepping
(its sign is the stepping direction). Steering uses flygym's published constants for connectome-driven
following (flygym/examples/vision/follow_fly_closed_loop.py): the inner side is attenuated by 0.6|s|
(to 0.4 at most) and the outer side lengthened by 0.2|s| (to 1.2 at most). These are the two steering
gestures of Yang et al. 2024 (Cell, doi:10.1016/j.cell.2024.08.033): DNa02 shortens ipsilateral strides,
DNg13 lengthens contralateral ones. Steering therefore modulates an ongoing rhythm: a fly that is not
stepping does not turn (the drawn body turns in place).

Nothing here is fitted to the kit. What stays hand-built: the decoder's weights (game.py), the mapping's
forward term (the drawn body's), the proboscis, wings and abdomen (still drawn: the NeuroMechFly model
has no joints there), and the escape jump (not modelled physically).

Install (the kit's other dependencies are unchanged; flygym's own requirement list pins numba 0.60 and
pulls in Jupyter, so install it without its dependencies)::

    pip install "mujoco==3.2.7" "dm_control==1.0.27" "dm_tree==0.1.8" gymnasium scipy \
                "opencv-python-headless>=4" imageio matplotlib pyyaml tqdm networkx
    pip install --no-deps flygym==1.2.1

Python 3.10-3.12 on Linux, macOS and Windows (wheels exist for all); not 3.13 (flygym, dm_tree 0.1.8).
No OpenGL is needed: rendering is switched off (MUJOCO_GL=disable).
"""

from __future__ import annotations

import math
import os
import time

import numpy as np

from .body import FlyBody, Pose
from .world import ARENA_R, wrap

os.environ.setdefault("MUJOCO_GL", "disable")          # physics only; set MUJOCO_GL=egl/glfw yourself to render

try:
    import mujoco
    from flygym import Fly, SingleFlySimulation
    from flygym.arena import FlatTerrain
    from flygym.examples.locomotion import CPGNetwork, PreprogrammedSteps
    from flygym.examples.locomotion.turning_controller import _tripod_coupling_weights, _tripod_phase_biases
    _IMPORT_ERROR: Exception | None = None
except Exception as e:                                   # pragma: no cover - exercised where flygym is missing
    _IMPORT_ERROR = e

INSTALL_HINT = ('the physics body needs flygym and MuJoCo: pip install "mujoco==3.2.7" "dm_control==1.0.27" '
                '"dm_tree==0.1.8" gymnasium scipy "opencv-python-headless>=4" imageio matplotlib pyyaml tqdm networkx '
                '&& pip install --no-deps flygym==1.2.1')

INNER_ATTENUATION = 0.6      # flygym follow_fly_closed_loop.py: inner = max(0.4, 1 - 0.6 |s|)
OUTER_BOOST = 0.2            # flygym follow_fly_closed_loop.py: outer = min(1.2, 1 + 0.2 |s|)
TIMESTEP = 1e-4              # s, flygym's default; its contacts (solref 2e-4 s) are unstable at 5e-4
SWING_EXTENSION = math.pi / 4   # flygym HybridTurningController._init_phasic_gain: adhesion stays off pi/4 longer


def available() -> bool:
    return _IMPORT_ERROR is None


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


WALL_TOUCHERS = ("Head", "Thorax") + tuple(f"{leg}{seg}" for leg in ("LF", "LM", "LH", "RF", "RM", "RH")
                                           for seg in ("Tibia", "Tarsus1"))

if _IMPORT_ERROR is None:
    class _WalledFloor(FlatTerrain):
        """flygym's flat floor plus the kit's round arena wall (n box segments, 3 mm high)."""

        def __init__(self, center=(0.0, 0.0), r=ARENA_R, n=48):
            super().__init__()
            seg = 2 * math.pi * r / n
            for i in range(n):
                a = 2 * math.pi * i / n
                self.root_element.worldbody.add(
                    "geom", type="box", name=f"wall_{i}", size=(0.5, seg / 2 + 0.2, 1.5),
                    pos=(center[0] + (r + 0.5) * math.cos(a), center[1] + (r + 0.5) * math.sin(a), 1.5),
                    euler=(0, 0, a), contype=0, conaffinity=0, rgba=(0.6, 0.6, 0.6, 1))

    class _Fly(Fly):
        """flygym's fly; the wall touches its head, thorax, tibiae and first tarsi through explicit contact
        pairs, like flygym's own floor contacts. (Making the fly's geoms collidable instead would make MuJoCo
        build convex hulls for all 69 meshes: 1.6x faster, but straight walking 14% faster too.)"""

        def init_floor_contacts(self, arena):
            walls = [g for g in arena.root_element.find_all("geom") if (g.name or "").startswith("wall_")]
            for g in walls:                    # flygym treats every arena geom as floor unless named *sensor*
                g.name = "sensor_" + g.name
            super().init_floor_contacts(arena)
            for g in walls:
                g.name = g.name[len("sensor_"):]
                for part in WALL_TOUCHERS:
                    arena.root_element.contact.add("pair", name=f"{g.name}_{self.name}_{part}", geom1=f"{self.name}/{part}",
                                                   geom2=g.name, solref=self.contact_solref, solimp=self.contact_solimp,
                                                   margin=0.0, friction=(1.0, 1.0, 0.005, 0.0001, 0.0001))


class Walker:
    """flygym's CPG controller on flat ground, stepping MuJoCo directly: six phase oscillators with tripod
    coupling, recorded single steps per leg, adhesion on during stance. The equations and constants of
    flygym's HybridTurningController, whose 0.6/0.2 steering constants the drive uses, including its
    adhesion timing (the swing, adhesion off, lasts pi/4 longer), but without its stumbling and retraction
    rules (these act on rough terrain and need a full observation every 0.1 ms, 3.5x slower). Turn rates
    then match the controller's (about 180 deg/s at the steering limit)."""

    def __init__(self, timestep: float = TIMESTEP, seed: int = 0, wall_center=None):
        if _IMPORT_ERROR is not None:
            raise RuntimeError(INSTALL_HINT) from _IMPORT_ERROR
        arena = _WalledFloor(center=wall_center) if wall_center is not None else FlatTerrain()
        self.fly = (_Fly if wall_center is not None else Fly)(enable_adhesion=True, draw_adhesion=False,
                                                                  spawn_pos=(0.0, 0.0, 0.2))
        self.sim = SingleFlySimulation(fly=self.fly, cameras=[], timestep=timestep, arena=arena)
        self.timestep, self.seed = timestep, seed
        steps = PreprogrammedSteps()
        self.cpg = CPGNetwork(timestep=timestep, intrinsic_freqs=np.ones(6) * 12, intrinsic_amps=np.zeros(6),
                              coupling_weights=_tripod_coupling_weights, phase_biases=_tripod_phase_biases,
                              convergence_coefs=np.ones(6) * 20, seed=seed)
        self._n = 2048                         # the recorded step, tabulated over its phase
        grid = np.linspace(0, 2 * np.pi, self._n + 1)
        self._neutral = np.stack([steps.neutral_pos[leg][:, 0] for leg in steps.legs])
        self._offs = np.stack([steps._psi_funcs[leg](grid) - steps.neutral_pos[leg] for leg in steps.legs])
        self._sw0 = np.array([steps.swing_period[leg][0] for leg in steps.legs])
        self._sw1 = np.array([steps.swing_period[leg][1] for leg in steps.legs]) + SWING_EXTENSION
        self.reset()

    def reset(self):
        self.sim.reset(seed=self.seed)
        self.cpg.random_state = np.random.RandomState(self.seed)
        self.cpg.reset(None, None)
        p = self.sim.physics
        self._m, self._d = p.model.ptr, p.data.ptr
        self._act = np.asarray(p.bind(self.fly.actuators).element_id)
        self._adh = np.asarray(p.bind(self.fly.adhesion_actuators).element_id)
        self._thorax = p.model.name2id(self.fly.name + "/Thorax", "body")
        self._tarsi = [p.model.name2id(f"{self.fly.name}/{leg}Tarsus5", "body") for leg in
                       ("LF", "LM", "LH", "RF", "RM", "RH")]
        names = [mujoco.mj_id2name(self._m, mujoco.mjtObj.mjOBJ_GEOM, i) or "" for i in range(self._m.ngeom)]
        self._wall_geoms = {i for i, nm in enumerate(names) if nm.startswith("wall_")}
        self.set_drive(0.0, 0.0)

    def set_drive(self, left: float, right: float):
        self.cpg.intrinsic_amps = np.repeat(np.abs([left, right]), 3)
        self.cpg.intrinsic_freqs = np.repeat([12.0 if left > 0 else -12.0, 12.0 if right > 0 else -12.0], 3)

    def advance(self, n_steps: int):
        d, m, cpg, ctrl = self._d, self._m, self.cpg, self._d.ctrl
        legs = np.arange(6)
        for _ in range(n_steps):
            cpg.step()
            ph = cpg.curr_phases % (2 * np.pi)
            u = ph / (2 * np.pi) * self._n
            i0 = u.astype(np.int64)
            fr = (u - i0)[:, None]
            off = self._offs[legs, :, i0] * (1 - fr) + self._offs[legs, :, i0 + 1] * fr
            ctrl[self._act] = (self._neutral + cpg.curr_magnitudes[:, None] * off).ravel()
            ctrl[self._adh] = ~((self._sw0 < ph) & (ph < self._sw1))
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


class PhysicsBody:
    """Same interface as :class:`virtual_fly.body.FlyBody` (pose, move, reset, start_jump, to_dict), so the
    game can use either. The physical fly is real size; the kit draws it three times larger, and the senses
    keep the drawn geometry (mouth and forelegs ahead of the thorax)."""

    kind = "physics"

    def __init__(self, world, rng, timestep: float = TIMESTEP, seed: int = 0, wall: bool = True,
                 stride_average: bool = False):
        self.world, self.rng = world, rng
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

    def reset(self, x=0.0, y=-12.0, h=math.pi / 2):
        # the physics world is built around the fly's start: the arena centre is where the kit says it is
        c = (-(x * math.cos(-h) - y * math.sin(-h)), -(x * math.sin(-h) + y * math.cos(-h)))
        if self.walker is None or (self._wall and getattr(self, "_center", None) != c):
            self.walker = Walker(self.timestep, self._seed, wall_center=c if self._wall else None)
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
            keep = max(1, int(round((1 / 12.0) / (k * self.timestep))))
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
        p.leg_phase = float(self.walker.cpg.curr_phases[0]) / (2 * math.pi)
        self.bumped = self._wall and self.walker.touching_wall()
        self._z = float(pos[2])
        FlyBody._appendages(self, dt, mode, drive)           # proboscis, wings, abdomen: still drawn

    def to_dict(self) -> dict:
        d = FlyBody.to_dict(self)
        d["physics"] = {"left": round(self.drive_lr[0], 3), "right": round(self.drive_lr[1], 3),
                        "z": round(getattr(self, "_z", 0.0), 3),
                        "tarsi": [[round(a, 2), round(b, 2)] for a, b in (self._to_kit(t) for t in self.walker.tarsi_xy())]}
        return d


def make_body(kind: str, world, rng, seed: int = 0, stride_average: bool = False):
    """``drawn`` (the default kinematic body) or ``physics`` (this module; needs flygym)."""
    if kind == "drawn":
        return FlyBody(world, rng)
    if kind == "physics":
        if not available():
            raise RuntimeError(INSTALL_HINT)
        return PhysicsBody(world, rng, seed=seed, stride_average=stride_average)
    raise ValueError("body must be 'drawn' or 'physics'")
