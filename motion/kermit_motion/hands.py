"""Swappable end effectors that play the puppet head.

Each profile attaches itself to the xArm6 flange, adds a ``head`` site (the puppet's neck pivot,
x = facing direction, z = up), optionally hides itself inside the felt head (costume.py), maps mouth
openness (0 = closed, 1 = wide) to its own joints, and drives the felt jaw from its kinematics. Adding a new hand means writing one more subclass and
registering it in ``PROFILES``.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from . import costume
from .xarm6 import ASSETS

ADAPTER = (0.12, 0.12, 0.13, 1)

# head site frame expressed in the "palm" frame used by both hands below
# (palm z = fingers / jaws pointing out of the flange, palm +y = top of the head)
HEAD_FROM_PALM = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], float)  # columns: head x, y, z


def _mat2quat(R: np.ndarray) -> list[float]:
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.ascontiguousarray(R).ravel())
    return q.tolist()


def _geom(body, name, gtype, size, pos, rgba, quat=(1, 0, 0, 0), material=None):
    g = body.add_geom(name=name, type=gtype, size=list(size) + [0] * (3 - len(size)), pos=list(pos),
                      quat=list(quat), rgba=list(rgba), contype=0, conaffinity=0, group=1, density=0)
    if material:
        g.material = material
    return g


@dataclass
class HandProfile:
    name: str = "base"
    joint_names: list[str] = field(default_factory=list)   # joints this profile drives
    vel_max: np.ndarray = field(default_factory=lambda: np.zeros(0))  # per joint (rad/s or m/s)
    vel_note: str = ""
    lip_lead_s: float = 0.03  # open the mouth slightly before the sound, as puppeteers do

    def attach(self, spec: mujoco.MjSpec, flange: mujoco.MjsSite, shell: bool) -> None:
        raise NotImplementedError

    def prepare(self, model: mujoco.MjModel) -> None:
        """Called once with the compiled model (precompute mouth tables, look up joint ids)."""
        self.qadr = np.array([model.jnt_qposadr[model.joint(n).id] for n in self.joint_names])
        self.range = np.array([model.jnt_range[model.joint(n).id] for n in self.joint_names])

    def mouth_qpos(self, openness: np.ndarray) -> np.ndarray:
        """(N,) openness 0..1 -> (N, len(joint_names)) joint positions."""
        raise NotImplementedError

    def dress_pose(self, model: mujoco.MjModel, data: mujoco.MjData) -> bool:
        """Optionally pose cosmetic costume joints from the current kinematics; True if anything moved."""
        return False


class ShadowHand(HandProfile):
    """Shadow Dexterous Hand (mujoco_menagerie) as a stand-in for the real dexterous hand.

    Puppet mapping: four fingers together are the upper jaw, the opposed thumb is the lower jaw.
    Wrist joints WRJ1/WRJ2 are locked at 0: the real candidate hands (Inspire, Ability) have no
    wrist, so all head motion comes from the xArm.
    """

    FINGERS = ["FF", "MF", "RF", "LF"]

    def __init__(self):
        names = []
        for f in self.FINGERS:
            names += [f"hand/rh_{f}J4", f"hand/rh_{f}J3", f"hand/rh_{f}J2", f"hand/rh_{f}J1"]
        names += ["hand/rh_LFJ5"] + [f"hand/rh_THJ{i}" for i in (5, 4, 3, 2, 1)]
        names += ["hand/rh_WRJ2", "hand/rh_WRJ1"]
        # Speeds of the most likely real hand (Inspire RH56DFX: fingers ~260 deg/s, thumb 70-107 deg/s),
        # applied per Shadow joint as a stand-in until the real hand is confirmed.
        vel = np.array([np.deg2rad(90.0) if "TH" in n else np.deg2rad(260.0) for n in names])
        vel[-2:] = 0.0  # wrist locked
        super().__init__(name="shadow", joint_names=names, vel_max=vel,
                         vel_note="Inspire RH56DFX-like: fingers 260 deg/s, thumb 90 deg/s (stand-in)")

    # back of the mouth (the knuckle line, just under the palm's mid-plane) in the palm frame
    MOUTH_HINGE = (0.0, -0.012, 0.086)

    def attach(self, spec, flange, shell):
        hs = mujoco.MjSpec.from_file(str(ASSETS / "shadow_hand/right_hand.xml"))
        hs.meshdir = str(ASSETS / "shadow_hand/assets")
        hs.option.impratio, hs.option.cone = spec.option.impratio, spec.option.cone  # no attach conflict
        mount = flange.parent
        _geom(mount, "adapter", mujoco.mjtGeom.mjGEOM_CYLINDER, [0.037, 0.012], [0, 0, 0.012], ADAPTER)
        wrist = flange.attach_body(hs.body("rh_wrist"), "hand/", "")
        wrist.pos = [0, 0, 0.032]
        wrist.quat = [1, 0, 0, 0]  # palm frame == flange frame (palm z out of the flange)
        palm = spec.body("hand/rh_palm")
        # Neck pivot at the wrist. "Forward" is the closed beak (fingertips meeting the thumb at about
        # (0, -0.05, 0.115) in the palm frame), so a neutral head aims its mouth at the audience.
        fwd = np.array([0, -0.05, 0.115]) / np.linalg.norm([0, -0.05, 0.115])
        up = np.array([0, fwd[2], -fwd[1]])
        R = np.column_stack([fwd, np.cross(up, fwd), up])
        palm.add_site(name="head", pos=[0.0, 0.0, 0.0], quat=_mat2quat(R), size=[0.005] * 3, group=5)
        spec.body("hand/rh_thdistal").add_site(name="thumb_tip", pos=[0, -0.008, 0.022], size=[0.003] * 3,
                                               group=5)
        for f in ["ff", "mf", "rf"]:
            spec.body(f"hand/rh_{f}distal").add_site(name=f"{f}_pad", pos=[0, -0.007, 0.018],
                                                     size=[0.003] * 3, group=5)
        if shell:
            # the whole hand goes inside the felt head: hide its geoms (group 4; --xray shows them)
            for geom in spec.geoms:
                body = geom.parent
                if body is not None and body.name.startswith("hand/"):
                    geom.group = 4
            # puppet frame: same orientation as the head site, origin at the back of the mouth
            costume.add_head(spec, palm, self.MOUTH_HINGE, _mat2quat(R))

    # ---- mouth model -------------------------------------------------------------------------------
    # Fingers: closed = gently curled snout; open = upper jaw lifted (MCP extended).
    # Fingers: closed = curled "beak" whose tips meet the thumb; open = upper jaw lifted (MCP extended).
    FINGER_CLOSED = dict(J3=0.95, J2=0.65, J1=0.45)
    FINGER_OPEN = dict(J3=0.5, J2=0.35, J1=0.25)
    # knuckle adduction so the four fingers close up into one jaw (FF/MF axes are -y, RF/LF +y)
    ADDUCT = dict(FF=0.12, MF=0.03, RF=0.03, LF=0.12)

    def prepare(self, model):
        super().prepare(model)
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "puppet_skull")
        self._skull = model.jnt_qposadr[jid] if jid >= 0 else None
        self._jaw = model.jnt_qposadr[model.joint("puppet_jaw").id] if jid >= 0 else None
        self._ffj3 = model.jnt_qposadr[model.joint("hand/rh_FFJ3").id]
        self._palm = model.body("hand/rh_palm").id
        self._tip = model.site("thumb_tip").id
        idx = {n: i for i, n in enumerate(self.joint_names)}
        levels = np.linspace(0, 1, 21)
        table = np.zeros((len(levels), len(self.joint_names)))
        d = mujoco.MjData(model)
        th_names = [f"hand/rh_THJ{i}" for i in (5, 4, 3, 2, 1)]
        th_adr = np.array([model.jnt_qposadr[model.joint(n).id] for n in th_names])
        th_dof = np.array([model.jnt_dofadr[model.joint(n).id] for n in th_names])
        th_rng = np.array([model.jnt_range[model.joint(n).id] for n in th_names])
        tip = model.site("thumb_tip").id
        palm = model.body("hand/rh_palm").id
        q_th = np.array([1.0, 1.0, 0.0, 0.1, -0.2])
        for k, o in enumerate(levels):
            s = o  # finger lift follows openness linearly; the curve shaping happens upstream
            for f in self.FINGERS:
                table[k, idx[f"hand/rh_{f}J4"]] = self.ADDUCT[f]
                for j in ("J3", "J2", "J1"):
                    table[k, idx[f"hand/rh_{f}{j}"]] = (1 - s) * self.FINGER_CLOSED[j] + s * self.FINGER_OPEN[j]
            # little finger cups in slightly so the jaw edge reads as one line
            table[k, idx["hand/rh_LFJ5"]] = 0.1
            d.qpos[:] = 0
            d.qpos[self.qadr] = table[k]
            mujoco.mj_kinematics(model, d)
            # lower-jaw target in palm frame: the closed-beak meeting point under the ff/mf pads, dropping
            # down (palm -y) and back as the mouth opens. Unreachable targets resolve to the closest pose.
            Rp = d.xmat[palm].reshape(3, 3)
            if k == 0:
                pads = np.mean([d.site(f"{f}_pad").xpos for f in ("ff", "mf")], axis=0)
                closed_p = Rp.T @ (pads - d.xpos[palm]) + np.array([0.0, -0.006, -0.004])
            target_p = closed_p + np.array([0.0, -0.055 * o, -0.028 * o])
            target = d.xpos[palm] + Rp @ target_p
            for _ in range(60):
                d.qpos[th_adr] = q_th
                mujoco.mj_kinematics(model, d)
                mujoco.mj_comPos(model, d)
                err = target - d.site_xpos[tip]
                if np.linalg.norm(err) < 2e-4:
                    break
                jacp = np.zeros((3, model.nv))
                mujoco.mj_jacSite(model, d, jacp, None, tip)
                J = jacp[:, th_dof]
                # damped least squares with a weak pull toward a natural opposed thumb
                dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(3), err)
                q_th = np.clip(q_th + dq + 0.02 * (np.array([0.8, 1.0, 0.0, 0.1, 0.0]) - q_th), th_rng[:, 0],
                               th_rng[:, 1])
            for n, v in zip(th_names, q_th):
                table[k, idx[n]] = v
        self.levels, self.table = levels, table
        # thumb-tip angle at closed / wide-open mouth, for mapping onto the felt jaw
        angles = []
        for row in (table[0], table[-1]):
            d.qpos[:] = 0
            d.qpos[self.qadr] = row
            mujoco.mj_kinematics(model, d)
            angles.append(self._thumb_angle(d))
        self._th0, self._th1 = angles

    def mouth_qpos(self, openness):
        o = np.clip(openness, 0, 1)
        return np.stack([np.interp(o, self.levels, self.table[:, j]) for j in range(self.table.shape[1])], axis=1)

    def _thumb_angle(self, data):
        """Thumb-tip angle below the palm plane, seen from the mouth hinge (palm x axis)."""
        Rp = data.xmat[self._palm].reshape(3, 3)
        tip = Rp.T @ (data.site_xpos[self._tip] - data.xpos[self._palm]) - np.asarray(self.MOUTH_HINGE)
        return np.arctan2(-tip[1], tip[2])

    def dress_pose(self, model, data):
        """After kinematics: felt skull follows the fingers, felt jaw follows the thumb."""
        if self._skull is None:
            return False
        ff = data.qpos[self._ffj3]
        up = (self.FINGER_CLOSED["J3"] - ff) / (self.FINGER_CLOSED["J3"] - self.FINGER_OPEN["J3"])
        low = (self._thumb_angle(data) - self._th0) / (self._th1 - self._th0)
        data.qpos[self._skull] = costume.UPPER_MAX * np.clip(up, -0.2, 1.3)
        data.qpos[self._jaw] = costume.LOWER_MAX * np.clip(low, -0.05, 1.3)
        return True


class DHGripper(HandProfile):
    """DH Robotics PGC-140-50 parallel gripper (the hardware we know exists) as the puppet mouth.

    The jaws open vertically: the upper jaw carries the skull, the lower jaw the felt chin; the wrist
    is the neck. Stroke 2 x 25 mm; full open/close takes 0.6-0.75 s (datasheets differ, we use 0.75 s),
    so syllable-rate lip sync is heavily slew-limited: the mouth flaps per word, not per syllable.
    """

    MOUTH_HINGE = (0.0, 0.0, 0.075)  # in the gripper base frame (z out of the flange, y = jaw travel)

    def __init__(self):
        super().__init__(name="dh_pgc140", joint_names=["jaw_upper", "jaw_lower"],
                         vel_max=np.full(2, 0.025 / 0.75),
                         vel_note="PGC-140-50: 25 mm per jaw in 0.75 s (conservative of 0.6/0.75 s datasheets)")

    def attach(self, spec, flange, shell):
        d = ASSETS / "dh_pgc140/meshes"
        spec.add_mesh(name="pgc_base", file=str(d / "base_link.STL"))
        spec.add_mesh(name="pgc_finger1", file=str(d / "finger1_Link.STL"))
        spec.add_mesh(name="pgc_finger2", file=str(d / "finger2_Link.STL"))
        spec.add_material(name="pgc_body", rgba=(0.2, 0.22, 0.25, 1), specular=0.5)
        spec.add_material(name="pgc_finger", rgba=(0.55, 0.57, 0.6, 1), specular=0.5)
        link6 = flange.parent
        hidden = 4 if shell else 0
        base = link6.add_body(name="pgc", pos=[0, 0, 0.0084])
        base.add_geom(name="pgc_base_vis", type=mujoco.mjtGeom.mjGEOM_MESH, meshname="pgc_base",
                      material="pgc_body", contype=0, conaffinity=0, group=hidden)
        # base y axis = jaw travel = the puppet's up axis
        base.add_site(name="head", pos=[0, 0, 0.0], quat=_mat2quat(HEAD_FROM_PALM), size=[0.005] * 3, group=5)
        # URDF: finger1 at +y slides along -y as q grows (q = 0 open, 0.025 closed)
        for name, mesh, pos, yaw in [("jaw_upper", "pgc_finger1", (0.0132, 0.0265, 0.093), -np.pi / 2),
                                     ("jaw_lower", "pgc_finger2", (-0.0132, -0.0265, 0.093), np.pi / 2)]:
            b = base.add_body(name=name, pos=list(pos), quat=[np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])
            b.add_joint(name=name, type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[1, 0, 0], range=[0, 0.025],
                        limited=mujoco.mjtLimited.mjLIMITED_TRUE)
            b.add_geom(name=f"{name}_vis", type=mujoco.mjtGeom.mjGEOM_MESH, meshname=mesh, material="pgc_finger",
                       contype=0, conaffinity=0, group=hidden)
        if shell:
            costume.add_head(spec, base, self.MOUTH_HINGE, _mat2quat(HEAD_FROM_PALM))

    def prepare(self, model):
        super().prepare(model)
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "puppet_skull")
        self._skull = model.jnt_qposadr[jid] if jid >= 0 else None
        self._jaw = model.jnt_qposadr[model.joint("puppet_jaw").id] if jid >= 0 else None

    def mouth_qpos(self, openness):
        q = 0.025 * (1 - np.clip(openness, 0, 1))
        return np.stack([q, q], axis=1)

    def dress_pose(self, model, data):
        if self._skull is None:
            return False
        up = 1 - data.qpos[self.qadr[0]] / 0.025
        low = 1 - data.qpos[self.qadr[1]] / 0.025
        data.qpos[self._skull] = costume.UPPER_MAX * up
        data.qpos[self._jaw] = costume.LOWER_MAX * low
        return True


PROFILES = {"shadow": ShadowHand, "dh_pgc140": DHGripper}
