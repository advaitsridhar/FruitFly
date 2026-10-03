"""
Two (or more) NeuroMechFly bodies in ONE MuJoCo world, each driven by its own brain, able to touch each other: the physics
pair of the two-flies work (docs/TWO_FLIES_PLAN.md section 8, decisions 20-24). Optional, like :mod:`virtual_fly.physics`,
whose single-fly body this module leaves exactly as it is: it reuses that module's drive mapping, its walled floor and its
fly class, and builds flygym's multi-fly ``Simulation`` instead of its ``SingleFlySimulation``.

    brain k --decoder--> drives --descending_drive()--> (left, right) stepping drive for fly k   (physics.py, unchanged)
    both drives set --> PairWorld.advance_tick(): per 0.1 ms step each fly's CPG and control writes, then ONE mj_step
    --> each PairPhysicsBody reads its thorax pose, its own wall contact and its tap on the other fly

What is physical here: the bodies, the floor and the wall, the contact between the flies (explicit MuJoCo contact pairs:
every pair is checked on every step, so the set is kept small, ``CONTACT_SETS``), the tap (a foreleg or the head on the
other fly's body, with MuJoCo's contact force). What stays hand-built: everything the single physics body lists (the
decoder's weights, the forward term, the drawn proboscis, wings and abdomen), plus the size the other fly's retina gives a
physics fly (``SEEN_FLY_MM``, taken from the model's measured size) and the choice of contact set. Not modelled: the
escape jump (NeuroMechFly has no jump model; a giant-fibre burst is the escape command with the legs standing). The
NeuroMechFly body was built from a micro-CT scan of a female fly: the male wears it too, and says so (decision 24).

Frames: MuJoCo's frame IS the kit's frame here (the wall is centred at the origin), unlike the single-fly body, which
builds its world around the fly's start. Positions in mm, headings counter-clockwise from +x, as everywhere in the kit.
"""
from __future__ import annotations

import math
import time

import numpy as np

from . import physics
from .body import FlyBody, Pose
from .world import FLY_HALF, wrap

TIMESTEP = physics.TIMESTEP
LEGS = ("LF", "LM", "LH", "RF", "RM", "RH")
# the geoms of one fly that touch the other (decision 22): its head and forelegs (tibia and the first four tarsal
# segments) against the other's body. Never Tarsus5: it carries the adhesion actuator, and MuJoCo's adhesion acts on every
# contact of that body, so a stance foot would glue itself to the other fly.
MY_TOUCHERS = ("Head",) + tuple(f"{leg}{seg}" for leg in ("LF", "RF") for seg in ("Tibia", "Tarsus1", "Tarsus2", "Tarsus3", "Tarsus4"))
THEIR_BODY = ("Thorax", "A1A2", "A3", "A4", "A5", "A6", "Head", "LWing", "RWing")
BODY_TO_BODY = ("Thorax", "Head", "A1A2", "A3", "A4", "A5", "A6")
ALL_TOUCHERS = tuple(f"{leg}{seg}" for leg in LEGS for seg in ("Tibia", "Tarsus1", "Tarsus2", "Tarsus3", "Tarsus4")) + THEIR_BODY
CONTACT_SETS = {
    "forelegs": (MY_TOUCHERS, THEIR_BODY, BODY_TO_BODY),   # the default: 233 distinct pairs
    "full": (ALL_TOUCHERS, ALL_TOUCHERS, ()),               # the research's 45 x 45 = 2,025 pairs (dearer, measured)
    "none": ((), (), ()),                                   # flygym's own default: the flies pass through each other
}
# the standing body (head, thorax, abdomen) of flygym 1.2.1's model, measured on the compiled model: length, width, height
REAL_FLY_MM = (2.8, 1.0, 1.1)
# how another fly's retina sees a physics fly: a dark cylinder of this radius and height (hand-built, from REAL_FLY_MM; a
# drawn fly is seen as 1.6 and 2.2, senses/vision.py), and how the dish draws it: real size over the drawn fly's length
SEEN_FLY_MM = (0.7, 1.1)
DRAWN_SCALE = round(REAL_FLY_MM[0] / (2 * FLY_HALF), 3)      # 0.389: a physics fly drawn at real size (decision 23)


def available() -> bool:
    return physics.available()


def unavailable_reason() -> str:
    return physics.unavailable_reason()


if physics.available():
    class _PairFly(physics._Fly):
        """flygym's fly with the kit's wall pairs (physics._Fly) and, on one fly of the world, the explicit fly-to-fly
        contact pairs (``pairs``: (this fly's geom, the other fly's geom)); init_floor_contacts runs after every fly has
        been spawned and before the model is compiled, so the other fly's geoms exist by then."""

        pairs: tuple = ()
        other: str = ""
        pair_solref: tuple | None = None       # the fly-to-fly pairs' own solref/solimp (None: the fly's, flygym's)
        pair_solimp: tuple | None = None
        hulls: bool = False                    # make MuJoCo build each mesh's convex hull (see PairWorld; a measured option)

        def init_floor_contacts(self, arena):
            if self.hulls:
                # flygym gives every fly geom contype 0 and conaffinity 0 (only explicit pairs collide), and MuJoCo's
                # compiler then builds no convex hull for the meshes, so a mesh-against-mesh pair finds its contact late
                # and shallow and the bodies pass through each other. A contype bit that no conaffinity in the model
                # answers (the floor and the wall have conaffinity 0 too) adds no bitmask collision but makes the compiler
                # build the hulls, and the pairs then collide as solids. Measured in docs/SCIENCE.md's two-body section.
                for g in self.model.find_all("geom"):
                    if g.dclass is not None and g.dclass.dclass == "nmf" or (g.name or "").startswith(("LF", "LM", "LH", "RF", "RM", "RH", "Thorax", "Head", "A", "LWing", "RWing")):
                        g.contype, g.conaffinity = 1, 0
            super().init_floor_contacts(arena)
            for g1, g2 in self.pairs:
                arena.root_element.contact.add("pair", name=f"{self.name}_{g1}__{self.other}_{g2}",
                                               geom1=f"{self.name}/{g1}", geom2=f"{self.other}/{g2}",
                                               solref=self.pair_solref or self.contact_solref,
                                               solimp=self.pair_solimp or self.contact_solimp, margin=0.0)


class _Legs:
    """One fly's leg controller state in the shared world: its CPG, its actuator and adhesion ids, its bodies and geoms."""

    def __init__(self, world: "PairWorld", k: int, fly, seed: int):
        import mujoco
        from flygym.examples.locomotion import CPGNetwork
        from flygym.examples.locomotion.turning_controller import _tripod_coupling_weights, _tripod_phase_biases
        p = world.sim.physics
        m = world._m
        self.k, self.fly, self.seed = k, fly, seed
        self.cpg = CPGNetwork(timestep=world.timestep, intrinsic_freqs=np.ones(6) * 12, intrinsic_amps=np.zeros(6),
                              coupling_weights=_tripod_coupling_weights, phase_biases=_tripod_phase_biases,
                              convergence_coefs=np.ones(6) * 20, seed=seed)
        self.act = np.asarray(p.bind(fly.actuators).element_id)
        self.adh = np.asarray(p.bind(fly.adhesion_actuators).element_id)
        self.thorax = p.model.name2id(f"{fly.name}/Thorax", "body")
        self.tarsi = [p.model.name2id(f"{fly.name}/{leg}Tarsus5", "body") for leg in LEGS]
        self.root = p.model.name2id(f"{fly.name}/", "body")
        self.geoms = frozenset(g for g in range(m.ngeom) if m.body_rootid[m.geom_bodyid[g]] == self.root)
        name_id = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g): g for g in self.geoms}
        mine, theirs, _ = CONTACT_SETS[world.contact_set]
        self.touchers = frozenset(name_id[f"{fly.name}/{n}"] for n in mine)
        self.body_geoms = frozenset(name_id[f"{fly.name}/{n}"] for n in theirs)
        # the fly's joints: one slice of qpos and one of qvel (flygym attaches the flies in order, each with its free joint
        # first), checked here rather than assumed
        joints = [j for j in range(m.njnt) if m.body_rootid[m.jnt_bodyid[j]] == self.root]
        free = m.body_jntadr[self.root]
        if not joints or joints[0] != free or m.jnt_type[free] != mujoco.mjtJoint.mjJNT_FREE \
                or joints != list(range(joints[0], joints[-1] + 1)):
            raise RuntimeError(f"fly {fly.name}: its joints are not one block starting with its free joint")
        last = joints[-1]
        self.qpos = slice(int(m.jnt_qposadr[free]), int(m.jnt_qposadr[last]) + _qpos_size(m, last))
        self.qvel = slice(int(m.jnt_dofadr[free]), int(m.jnt_dofadr[last]) + _dof_size(m, last))
        self.qpos0 = None                     # the spawn pose of every joint (set by the world after the build)
        self.spawn_z = 0.0
        self.offset = (0.0, 0.0)              # the thorax ahead of the root, in the root's frame (measured after the build)


def _qpos_size(m, j: int) -> int:
    import mujoco
    return {mujoco.mjtJoint.mjJNT_FREE: 7, mujoco.mjtJoint.mjJNT_BALL: 4}.get(mujoco.mjtJoint(m.jnt_type[j]), 1)


def _dof_size(m, j: int) -> int:
    import mujoco
    return {mujoco.mjtJoint.mjJNT_FREE: 6, mujoco.mjtJoint.mjJNT_BALL: 3}.get(mujoco.mjtJoint(m.jnt_type[j]), 1)


class PairWorld:
    """One MuJoCo world holding every physics fly of a game. ``poses``: each fly's (x, y, heading) in the kit's frame;
    ``world`` and ``rngs``: the kit's World and each fly's random stream, for the bodies. The flies are ``fly0``,
    ``fly1``, ... in flygym's model; fly k's CPG is seeded ``seed + 1000 k``, as its brain is."""

    def __init__(self, poses, seed: int = 0, world=None, rngs=None, timestep: float = TIMESTEP,
                 contact_set: str = "forelegs", wall: bool = True, pair_solref=None, pair_solimp=None, hulls: bool = False,
                 native_ccd: bool = False, levers=()):
        if not physics.available():
            raise RuntimeError(unavailable_reason()) from physics._IMPORT_ERROR
        if contact_set not in CONTACT_SETS:
            raise ValueError(f"contact_set must be one of {', '.join(CONTACT_SETS)}")
        import mujoco
        from flygym import Simulation
        from flygym.arena import FlatTerrain
        from flygym.examples.locomotion import PreprogrammedSteps
        poses = [tuple(float(v) for v in p) for p in poses]
        if len(poses) < 1:
            raise ValueError("a pair world needs at least one fly")
        self.levers = physics.parse_levers(levers)             # speed levers (physics.LEVERS): none unless asked for
        timestep = physics.lever_timestep(self.levers, timestep)
        self.timestep, self.seed, self.contact_set, self.wall = timestep, seed, contact_set, wall
        self.pair_solref, self.pair_solimp, self.hulls, self.native_ccd = pair_solref, pair_solimp, hulls, native_ccd
        names = [f"fly{k}" for k in range(len(poses))]
        mine, theirs, both = CONTACT_SETS[contact_set]
        pairs = {(a, b) for a in mine for b in theirs} | {(b, a) for a in mine for b in theirs} | {(a, b) for a in both for b in both}
        self.pairs = tuple(sorted(pairs)) if len(poses) > 1 else ()
        self.flies = []
        for k, (x, y, h) in enumerate(poses):
            fly = _PairFly(name=names[k], enable_adhesion=True, draw_adhesion=False, spawn_pos=(x, y, 0.2),
                           spawn_orientation=(0.0, 0.0, h), **physics.lever_fly_kwargs(self.levers))
            if "dedupe" in self.levers:
                physics.dedupe_self_pairs(fly)
            fly.hulls = hulls
            if k == 0 and len(poses) > 1:
                fly.pairs, fly.other = self.pairs, names[1]
                fly.pair_solref, fly.pair_solimp = pair_solref, pair_solimp
            self.flies.append(fly)
        arena = physics._WalledFloor(center=(0.0, 0.0)) if wall else FlatTerrain()
        self.sim = Simulation(flies=self.flies, cameras=[], arena=arena, timestep=timestep)
        p = self.sim.physics
        self._m, self._d = p.model.ptr, p.data.ptr
        m, d = self._m, self._d
        if native_ccd:                                # MuJoCo's own convex collider instead of libccd's (a measured option)
            m.opt.enableflags |= mujoco.mjtEnableBit.mjENBL_NATIVECCD
        physics.apply_levers(m, self.levers)
        steps = PreprogrammedSteps()                  # the recorded step, tabulated over its phase (as physics.Walker)
        self._n = 2048
        grid = np.linspace(0, 2 * np.pi, self._n + 1)
        self._neutral = np.stack([steps.neutral_pos[leg][:, 0] for leg in steps.legs])
        self._offs = np.stack([steps._psi_funcs[leg](grid) - steps.neutral_pos[leg] for leg in steps.legs])
        self._sw0 = np.array([steps.swing_period[leg][0] for leg in steps.legs])
        self._sw1 = np.array([steps.swing_period[leg][1] for leg in steps.legs]) + physics.SWING_EXTENSION
        names_all = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) or "" for i in range(m.ngeom)]
        self._wall_geoms = frozenset(i for i, nm in enumerate(names_all) if nm.startswith("wall_"))
        self.legs = [_Legs(self, k, fly, seed + 1000 * k) for k, fly in enumerate(self.flies)]
        mujoco.mj_forward(m, d)
        for L in self.legs:                           # where the thorax sits relative to the root, in the root's frame
            L.qpos0 = d.qpos[L.qpos].copy()
            L.spawn_z = float(d.qpos[L.qpos.start + 2])
            rot = d.xmat[L.root].reshape(3, 3)
            rel = rot.T @ (d.xpos[L.thorax] - d.xpos[L.root])
            L.offset = (float(rel[0]), float(rel[1]))
        self.wall_s = 0.0                             # wall-clock seconds spent stepping MuJoCo (both flies)
        self._f6 = np.zeros(6)
        for k, (x, y, h) in enumerate(poses):
            self.place(k, x, y, h)
        self.bodies = [PairPhysicsBody(self, k, world, rngs[k] if rngs is not None else None) for k in range(len(poses))]
        for b, (x, y, h) in zip(self.bodies, poses):
            b.pose = Pose(x=x, y=y, h=h)

    # ------------------------------------------------------------------ placing and driving
    def place(self, k: int, x: float, y: float, h: float):
        """Put fly k's thorax at (x, y) heading h without rebuilding the model (a rebuild costs seconds): its joints back
        to the spawn pose, its root free joint written, its velocities zeroed, its CPG reset, then mj_forward."""
        import mujoco
        L, d = self.legs[k], self._d
        d.qpos[L.qpos] = L.qpos0
        c, s = math.cos(h), math.sin(h)
        ox, oy = L.offset
        q = L.qpos.start
        d.qpos[q:q + 3] = (x - (c * ox - s * oy), y - (s * ox + c * oy), L.spawn_z)
        d.qpos[q + 3:q + 7] = (math.cos(h / 2), 0.0, 0.0, math.sin(h / 2))
        d.qvel[L.qvel] = 0.0
        d.qacc_warmstart[L.qvel] = 0.0
        d.ctrl[L.act] = self._neutral.ravel()
        d.ctrl[L.adh] = 1.0
        L.cpg.random_state = np.random.RandomState(L.seed)
        L.cpg.reset(None, None)
        L.cpg.intrinsic_amps = np.zeros(6)
        L.cpg.intrinsic_freqs = np.ones(6) * 12.0
        mujoco.mj_forward(self._m, d)

    def set_drive(self, k: int, left: float, right: float):
        cpg = self.legs[k].cpg
        cpg.intrinsic_amps = np.repeat(np.abs([left, right]), 3)
        cpg.intrinsic_freqs = np.repeat([12.0 if left > 0 else -12.0, 12.0 if right > 0 else -12.0], 3)

    def advance(self, n_steps: int):
        """``n_steps`` steps of the whole world: every fly's CPG step and control writes, then one mj_step."""
        import mujoco
        d, m, ctrl, legs = self._d, self._m, self._d.ctrl, self.legs
        six = np.arange(6)
        t0 = time.perf_counter()
        for _ in range(n_steps):
            for L in legs:
                cpg = L.cpg
                cpg.step()
                ph = cpg.curr_phases % (2 * np.pi)
                u = ph / (2 * np.pi) * self._n
                i0 = u.astype(np.int64)
                fr = (u - i0)[:, None]
                off = self._offs[six, :, i0] * (1 - fr) + self._offs[six, :, i0 + 1] * fr
                ctrl[L.act] = (self._neutral + cpg.curr_magnitudes[:, None] * off).ravel()
                ctrl[L.adh] = ~((self._sw0 < ph) & (ph < self._sw1))
            mujoco.mj_step(m, d)
        self.wall_s += time.perf_counter() - t0

    def advance_tick(self, dt: float):
        """One game tick for every fly: the world steps dt (both drives already set), then each body reads its pose."""
        self.advance(int(round(dt / self.timestep)))
        for b in self.bodies:
            b.settle()

    # ------------------------------------------------------------------ reading the world
    def thorax(self, k: int):
        """(x, y, z) of fly k's thorax in mm and its heading (rad, counter-clockwise from +x)."""
        d, L = self._d, self.legs[k]
        xm = d.xmat[L.thorax]
        return np.array(d.xpos[L.thorax]), math.atan2(xm[3], xm[0])

    def tarsi_xy(self, k: int) -> np.ndarray:
        return np.array(self._d.xpos[self.legs[k].tarsi])[:, :2]

    def touching_wall(self, k: int) -> bool:
        """Fly k's own contact with the wall (not the other fly's)."""
        d, own, wall = self._d, self.legs[k].geoms, self._wall_geoms
        for c in d.contact[:d.ncon]:
            if (c.geom1 in wall and c.geom2 in own) or (c.geom2 in wall and c.geom1 in own):
                return True
        return False

    def touching(self, k: int):
        """Fly k's tap on another fly: (that fly's index, the summed normal force) from the contacts between fly k's head
        or foreleg geoms and the other's body geoms this step, or None. The force is in the model's units (mm, mg: 1 is
        1 nN; the fly weighs about 9,800), summed over those contacts."""
        import mujoco
        d, m, mine = self._d, self._m, self.legs[k].touchers
        if not mine:
            return None
        best, force = None, 0.0
        for i in range(d.ncon):
            c = d.contact[i]
            for j, L in enumerate(self.legs):
                if j == k:
                    continue
                if (c.geom1 in mine and c.geom2 in L.body_geoms) or (c.geom2 in mine and c.geom1 in L.body_geoms):
                    mujoco.mj_contactForce(m, d, i, self._f6)
                    force += abs(float(self._f6[0]))
                    best = j if best is None else best
        return None if best is None else (best, force)

    def contacts_between(self, k: int, j: int) -> int:
        """How many contacts this step join any geom of fly k with any geom of fly j."""
        d, a, b = self._d, self.legs[k].geoms, self.legs[j].geoms
        return sum(1 for c in d.contact[:d.ncon] if (c.geom1 in a and c.geom2 in b) or (c.geom2 in a and c.geom1 in b))

    @property
    def ncon(self) -> int:
        return int(self._d.ncon)

    def close(self):
        """Let the world go (call it once no tick can be using it: the game's loop thread has stopped)."""
        sim, self.sim = self.sim, None
        if sim is not None:
            try:
                sim.close()
            except Exception:
                pass


class PairPhysicsBody:
    """One fly's body in a :class:`PairWorld`: the same interface as :class:`virtual_fly.physics.PhysicsBody` (pose, move,
    reset, start_jump, to_dict, bumped, jump_lock, distance, drive_lr), except that ``move`` only sets the drive: the world
    steps once per tick for every fly (``PairWorld.advance_tick``), after which ``settle`` reads the pose. ``touching_other``
    is the physical tap, (the other fly's index, the force) or None, read by the contact channel instead of the drawn 3.4 mm
    rule (decision 23). Drawn at real size (``scale``)."""

    kind = "physics"
    stride_average = False

    def __init__(self, pair: PairWorld, k: int, world=None, rng=None):
        self.pair, self.k, self.world, self.rng = pair, k, world, rng
        self.timestep = pair.timestep
        self.pose = Pose()
        self.jump_lock = 0.0
        self.distance = 0.0
        self.bumped = False
        self.bumped_fly = False
        self.drive_lr = (0.0, 0.0)
        self.touching_other = None
        self._pending = None
        self._z = 0.0

    @property
    def wall_s(self) -> float:
        return self.pair.wall_s                    # the shared world's seconds in MuJoCo (every fly's steps)

    def reset(self, x=0.0, y=-12.0, h=math.pi / 2):
        self.pair.place(self.k, x, y, h)
        self.pose = Pose(x=x, y=y, h=h)
        self.jump_lock, self.distance = 0.0, 0.0
        self.bumped, self.touching_other, self._pending = False, None, None
        self.drive_lr = (0.0, 0.0)

    def start_jump(self, away_from):
        self.jump_lock = 0.8                       # no physical jump: NeuroMechFly has no jump model

    def move(self, dt: float, mode: str, drive: dict, wander_yaw: float = 0.0, others=()):
        p = self.pose
        self.jump_lock -= dt
        p.mode = mode
        self.drive_lr = physics.descending_drive(mode, drive, wander_yaw)
        self.pair.set_drive(self.k, *self.drive_lr)
        self._pending = (dt, mode, drive)

    def settle(self):
        """After the world's tick: the pose from the thorax, the path, the wall bump, the tap, the drawn appendages."""
        if self._pending is None:
            return
        dt, mode, drive = self._pending
        self._pending = None
        pos, hd = self.pair.thorax(self.k)
        p = self.pose
        x, y, h = float(pos[0]), float(pos[1]), wrap(hd)
        dx, dy = x - p.x, y - p.y
        p.v = (dx * math.cos(p.h) + dy * math.sin(p.h)) / dt
        p.w = wrap(h - p.h) / dt
        self.distance += math.hypot(dx, dy)
        p.x, p.y, p.h = x, y, h
        p.leg_phase = float(self.pair.legs[self.k].cpg.curr_phases[0]) / (2 * math.pi)
        self.bumped = self.pair.wall and self.pair.touching_wall(self.k)
        self.touching_other = self.pair.touching(self.k)
        self._z = float(pos[2])
        FlyBody._appendages(self, dt, mode, drive)

    def to_dict(self) -> dict:
        d = FlyBody.to_dict(self)
        tap = self.touching_other
        d["physics"] = {"left": round(self.drive_lr[0], 3), "right": round(self.drive_lr[1], 3), "z": round(self._z, 3),
                        "tarsi": [[round(float(a), 2), round(float(b), 2)] for a, b in self.pair.tarsi_xy(self.k)],
                        "tap": None if tap is None else [tap[0], round(tap[1], 1)], "pair": True}
        d["scale"] = DRAWN_SCALE
        return d
