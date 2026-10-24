"""Articulated people for the stage: presenter and judges.

A stylised but detailed human (~1.75 m) built from primitives, posed kinematically every frame:
mocap root (floor point between the feet, x = facing direction) plus ~20 hinge joints. The front
and back read clearly: face, shirt opening, tie and buttons in front, hair and plain jacket at
the back, shoes pointing forward. The presenter can hold a handheld microphone (placed by IK).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

ELL = mujoco.mjtGeom.mjGEOM_ELLIPSOID
CAP = mujoco.mjtGeom.mjGEOM_CAPSULE
SPH = mujoco.mjtGeom.mjGEOM_SPHERE
BOX = mujoco.mjtGeom.mjGEOM_BOX
CYL = mujoco.mjtGeom.mjGEOM_CYLINDER

# body dimensions (m), standing, root at the floor
HIP_Z, HIP_Y = 0.92, 0.09
THIGH, SHIN, ANKLE_Z = 0.43, 0.42, 0.07
LUMBAR_Z = 1.0
SHOULDER_Z, SHOULDER_Y = 1.43 - LUMBAR_Z, 0.185   # relative to the lumbar joint
NECK_Z = 1.49 - LUMBAR_Z
HEAD_Z = 0.13                                     # head centre above the neck joint
UPPER_ARM, FOREARM = 0.29, 0.25

STYLES = {
    "presenter": dict(skin=(0.87, 0.68, 0.54), hair=(0.22, 0.14, 0.08), jacket=(0.14, 0.2, 0.36),
                      shirt=(0.95, 0.95, 0.97), tie=(0.7, 0.1, 0.12), pants=(0.25, 0.25, 0.28),
                      shoes=(0.12, 0.08, 0.05), hair_style="short"),
    "judge_a": dict(skin=(0.62, 0.44, 0.32), hair=(0.06, 0.05, 0.05), jacket=(0.55, 0.15, 0.2),
                    shirt=(0.15, 0.15, 0.17), tie=None, pants=(0.12, 0.12, 0.14), shoes=(0.08, 0.08, 0.08),
                    hair_style="long", glasses=True),
    "judge_b": dict(skin=(0.93, 0.78, 0.66), hair=(0.75, 0.6, 0.35), jacket=(0.3, 0.42, 0.3),
                    shirt=(0.9, 0.86, 0.75), tie=None, pants=(0.36, 0.3, 0.25), shoes=(0.3, 0.18, 0.1),
                    hair_style="short"),
    "judge_c": dict(skin=(0.45, 0.3, 0.22), hair=(0.12, 0.1, 0.1), jacket=(0.85, 0.65, 0.2),
                    shirt=(0.97, 0.97, 0.97), tie=None, pants=(0.2, 0.22, 0.35), shoes=(0.9, 0.9, 0.9),
                    hair_style="bald", glasses=False),
}

JOINTS = ["lumbar_bend", "lumbar_side", "lumbar_twist", "neck_yaw", "neck_pitch", "neck_roll",
          "sh_flex_l", "sh_abd_l", "sh_rot_l", "elbow_l", "wrist_l",
          "sh_flex_r", "sh_abd_r", "sh_rot_r", "elbow_r", "wrist_r",
          "hip_flex_l", "hip_abd_l", "knee_l", "hip_flex_r", "hip_abd_r", "knee_r"]


def _q(R):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.ascontiguousarray(R, float).ravel())
    return q.tolist()


@dataclass
class PersonModel:
    name: str
    root_mocap: int = -1
    qadr: dict = field(default_factory=dict)   # joint -> qpos address
    mouth_geom: int = -1
    mouth_size: np.ndarray | None = None
    mic_site: int = -1
    mouth_site: int = -1


def add_person(spec: mujoco.MjSpec, name: str, style: str, mic: bool = False) -> None:
    st = STYLES[style]
    rgba = {k: (*v, 1) for k, v in st.items() if isinstance(v, tuple)}

    def g(body, gname, gtype, size, pos, color, quat=(1, 0, 0, 0), **kw):
        body.add_geom(name=f"{name}_{gname}", type=gtype, size=list(size) + [0] * (3 - len(size)), pos=list(pos),
                      quat=list(quat), rgba=list(color), contype=0, conaffinity=0, group=1, density=0, **kw)

    def joint(body, jname, axis):
        body.add_joint(name=f"{name}_{jname}", type=mujoco.mjtJoint.mjJNT_HINGE, axis=list(axis), damping=1.0,
                       armature=0.01)

    def link(parent, bname, pos):
        return parent.add_body(name=f"{name}_{bname}", pos=list(pos), mass=0.5, inertia=[0.005] * 3,
                               explicitinertial=True)

    root = spec.worldbody.add_body(name=f"{name}_root", mocap=True)
    pelvis = link(root, "pelvis", [0, 0, HIP_Z])
    g(pelvis, "hips", ELL, [0.11, 0.165, 0.11], [0, 0, 0.04], rgba["pants"])
    g(pelvis, "belt", ELL, [0.112, 0.168, 0.025], [0, 0, 0.1], (0.1, 0.08, 0.06, 1))

    # ---- legs
    for s, side in [(1, "l"), (-1, "r")]:
        thigh = link(pelvis, f"thigh_{side}", [0, s * HIP_Y, 0])
        joint(thigh, f"hip_flex_{side}", [0, -1, 0])
        joint(thigh, f"hip_abd_{side}", [s, 0, 0])
        g(thigh, f"thigh_{side}", CAP, [0.07], [0, 0, 0], rgba["pants"], fromto=[0, 0, 0, 0, 0, -THIGH])
        shin = link(thigh, f"shin_{side}", [0, 0, -THIGH])
        joint(shin, f"knee_{side}", [0, 1, 0])
        g(shin, f"shin_{side}", CAP, [0.055], [0, 0, 0], rgba["pants"], fromto=[0, 0, 0, 0, 0, -SHIN + 0.02])
        g(shin, f"shoe_{side}", ELL, [0.13, 0.05, 0.045], [0.06, 0, -SHIN - 0.025], rgba["shoes"])
        g(shin, f"sole_{side}", ELL, [0.135, 0.052, 0.015], [0.06, 0, -SHIN - 0.06], (0.05, 0.05, 0.05, 1))

    # ---- torso
    chest = link(pelvis, "chest", [0, 0, LUMBAR_Z - HIP_Z])
    joint(chest, "lumbar_bend", [0, 1, 0])
    joint(chest, "lumbar_side", [1, 0, 0])
    joint(chest, "lumbar_twist", [0, 0, 1])
    g(chest, "torso", ELL, [0.115, 0.175, 0.25], [0, 0, 0.22], rgba["jacket"])
    g(chest, "shoulders", CAP, [0.075], [0, 0, 0], rgba["jacket"],
      fromto=[0, -SHOULDER_Y + 0.03, SHOULDER_Z - 0.02, 0, SHOULDER_Y - 0.03, SHOULDER_Z - 0.02])
    # front: shirt opening (V), tie, buttons, lapels; the back stays plain jacket
    g(chest, "shirt", ELL, [0.03, 0.065, 0.16], [0.09, 0, 0.3], rgba["shirt"])
    if st.get("tie"):
        g(chest, "tie_knot", ELL, [0.02, 0.022, 0.02], [0.12, 0, 0.42], rgba["tie"])
        g(chest, "tie", ELL, [0.012, 0.03, 0.13], [0.117, 0, 0.29], rgba["tie"])
    for k, z in enumerate((0.12, 0.04)):
        g(chest, f"button{k}", SPH, [0.011], [0.113, -0.03, z], (0.08, 0.08, 0.08, 1))
    for s, side in [(1, "l"), (-1, "r")]:
        R = np.array([[1, 0, 0], [0, np.cos(0.35 * s), -np.sin(0.35 * s)], [0, np.sin(0.35 * s), np.cos(0.35 * s)]])
        g(chest, f"lapel_{side}", ELL, [0.015, 0.035, 0.12], [0.105, s * 0.06, 0.3], rgba["jacket"], quat=_q(R))
    g(chest, "collar", CAP, [0.055], [0, 0, 0], rgba["shirt"], fromto=[0, 0, NECK_Z - 0.04, 0, 0, NECK_Z - 0.01])

    # ---- neck and head
    neck = link(chest, "neck", [0, 0, NECK_Z])
    joint(neck, "neck_yaw", [0, 0, 1])
    joint(neck, "neck_pitch", [0, 1, 0])
    joint(neck, "neck_roll", [1, 0, 0])
    g(neck, "neck", CAP, [0.045], [0, 0, 0], rgba["skin"], fromto=[0, 0, -0.02, 0, 0, 0.07])
    hz = HEAD_Z
    g(neck, "head", ELL, [0.1, 0.082, 0.12], [0.01, 0, hz], rgba["skin"])
    g(neck, "jaw", ELL, [0.075, 0.065, 0.06], [0.03, 0, hz - 0.06], rgba["skin"])
    g(neck, "nose", ELL, [0.022, 0.014, 0.024], [0.105, 0, hz - 0.005], rgba["skin"])
    for s, side in [(1, "l"), (-1, "r")]:
        g(neck, f"ear_{side}", ELL, [0.015, 0.012, 0.028], [0.0, s * 0.078, hz], rgba["skin"])
        g(neck, f"eye_{side}", SPH, [0.0125], [0.088, s * 0.032, hz + 0.022], (0.97, 0.97, 0.95, 1))
        g(neck, f"iris_{side}", SPH, [0.0072], [0.0985, s * 0.032, hz + 0.022], (0.15, 0.1, 0.06, 1))
        g(neck, f"brow_{side}", CAP, [0.005], [0, 0, 0], rgba["hair"],
          fromto=[0.094, s * 0.016, hz + 0.045, 0.09, s * 0.05, hz + 0.042])
        if st.get("glasses"):
            g(neck, f"lens_{side}", CYL, [0.021, 0.002], [0.106, s * 0.032, hz + 0.022], (0.05, 0.05, 0.05, 1),
              quat=_q(np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])))
            g(neck, f"lens_glass_{side}", CYL, [0.018, 0.0025], [0.107, s * 0.032, hz + 0.022],
              (0.7, 0.8, 0.9, 1), quat=_q(np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])))
    g(neck, "mouth", ELL, [0.008, 0.026, 0.004], [0.088, 0, hz - 0.045], (0.35, 0.08, 0.08, 1))
    neck.add_site(name=f"{name}_mouth", pos=[0.1, 0, hz - 0.045], size=[0.005] * 3, group=5)
    neck.add_site(name=f"{name}_face", pos=[0.08, 0, hz + 0.01], size=[0.005] * 3, group=5)
    hs = st.get("hair_style", "short")
    if hs == "short":
        g(neck, "hair", ELL, [0.1, 0.083, 0.08], [-0.005, 0, hz + 0.05], rgba["hair"])
        g(neck, "hair_back", ELL, [0.07, 0.08, 0.09], [-0.035, 0, hz + 0.0], rgba["hair"])
    elif hs == "long":
        g(neck, "hair", ELL, [0.1, 0.085, 0.085], [-0.005, 0, hz + 0.045], rgba["hair"])
        g(neck, "hair_back", ELL, [0.07, 0.1, 0.17], [-0.045, 0, hz - 0.06], rgba["hair"])
    else:
        g(neck, "hair_back", ELL, [0.05, 0.08, 0.05], [-0.055, 0, hz - 0.01], rgba["hair"])

    # ---- arms
    for s, side in [(1, "l"), (-1, "r")]:
        upper = link(chest, f"upper_{side}", [0, s * SHOULDER_Y, SHOULDER_Z])
        joint(upper, f"sh_flex_{side}", [0, -1, 0])
        joint(upper, f"sh_abd_{side}", [s, 0, 0])
        joint(upper, f"sh_rot_{side}", [0, 0, s])
        g(upper, f"upper_{side}", CAP, [0.048], [0, 0, 0], rgba["jacket"], fromto=[0, 0, 0, 0, 0, -UPPER_ARM])
        fore = link(upper, f"fore_{side}", [0, 0, -UPPER_ARM])
        joint(fore, f"elbow_{side}", [0, -1, 0])
        g(fore, f"fore_{side}", CAP, [0.04], [0, 0, 0], rgba["jacket"], fromto=[0, 0, 0, 0, 0, -FOREARM + 0.02])
        g(fore, f"cuff_{side}", CYL, [0.036, 0.008], [0, 0, -FOREARM + 0.01], rgba["shirt"])
        hand = link(fore, f"hand_{side}", [0, 0, -FOREARM])
        joint(hand, f"wrist_{side}", [0, -1, 0])
        g(hand, f"palm_{side}", ELL, [0.02, 0.042, 0.05], [0.005, 0, -0.05], rgba["skin"])
        g(hand, f"fingers_{side}", ELL, [0.018, 0.04, 0.04], [0.015, 0, -0.11], rgba["skin"])
        g(hand, f"thumb_{side}", CAP, [0.011], [0, 0, 0], rgba["skin"],
          fromto=[0.02, -s * 0.03, -0.03, 0.045, -s * 0.035, -0.075])
        if mic and side == "r":
            # handheld mic gripped in the fist, coming out of its thumb side
            fist = np.array([0.012, 0.0, -0.09])
            ax = np.array([0.25, -s * 1.0, 0.45])
            ax /= np.linalg.norm(ax)
            Rm = _axis_frame(ax)
            g(hand, "mic_handle", CYL, [0.016, 0.085], fist + ax * 0.03, (0.08, 0.08, 0.09, 1), quat=_q(Rm))
            g(hand, "mic_ring", CYL, [0.019, 0.008], fist + ax * 0.115, (0.5, 0.5, 0.52, 1), quat=_q(Rm))
            g(hand, "mic_head", SPH, [0.028], fist + ax * 0.14, (0.35, 0.35, 0.37, 1))
            hand.add_site(name=f"{name}_mic", pos=list(fist + ax * 0.16), size=[0.005] * 3, group=5)


def _axis_frame(z):
    z = np.asarray(z, float) / np.linalg.norm(z)
    x = np.cross([0, 0, 1.0], z) if abs(z[2]) < 0.95 else np.array([1.0, 0, 0])
    x /= np.linalg.norm(x)
    return np.column_stack([x, np.cross(z, x), z])


def add_chair(spec: mujoco.MjSpec, name: str, pos, yaw: float, floor_z: float):
    b = spec.worldbody.add_body(name=f"{name}_chair", pos=[pos[0], pos[1], floor_z],
                                quat=[np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])
    wood = (0.25, 0.25, 0.28, 1)
    kw = dict(contype=0, conaffinity=0, group=1, density=0)
    b.add_geom(name=f"{name}_seat", type=BOX, size=[0.22, 0.22, 0.03], pos=[0.02, 0, 0.45], rgba=wood, **kw)
    b.add_geom(name=f"{name}_back", type=BOX, size=[0.03, 0.22, 0.25], pos=[-0.2, 0, 0.73], rgba=wood, **kw)
    for i, (dx, dy) in enumerate([(0.2, 0.2), (0.2, -0.2), (-0.18, 0.2), (-0.18, -0.2)]):
        b.add_geom(name=f"{name}_leg{i}", type=CYL, size=[0.018, 0.22], pos=[dx, dy, 0.22],
                   rgba=(0.6, 0.6, 0.62, 1), **kw)


def bind(model: mujoco.MjModel, name: str) -> PersonModel:
    pm = PersonModel(name=name)
    pm.root_mocap = model.body_mocapid[model.body(f"{name}_root").id]
    pm.qadr = {j: model.jnt_qposadr[model.joint(f"{name}_{j}").id] for j in JOINTS}
    pm.mouth_geom = model.geom(f"{name}_mouth").id
    pm.mouth_size = model.geom_size[pm.mouth_geom].copy()
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{name}_mic")
    pm.mic_site = sid
    pm.mouth_site = model.site(f"{name}_mouth").id
    return pm


def apply(model, data, pm: PersonModel, root_pos, yaw, q: dict, mouth_open: float = 0.0):
    data.mocap_pos[pm.root_mocap] = root_pos
    data.mocap_quat[pm.root_mocap] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
    for j, adr in pm.qadr.items():
        data.qpos[adr] = q.get(j, 0.0)
    model.geom_size[pm.mouth_geom] = pm.mouth_size * np.array([1, 1 - 0.15 * mouth_open, 1 + 3.5 * mouth_open])


SEATED = dict(hip_flex_l=1.5, hip_flex_r=1.5, knee_l=1.45, knee_r=1.45, hip_abd_l=0.08, hip_abd_r=0.08,
              sh_flex_l=0.35, sh_flex_r=0.35, elbow_l=1.2, elbow_r=1.2, sh_abd_l=0.12, sh_abd_r=0.12,
              lumbar_bend=0.05)
STANDING = dict(sh_abd_l=0.1, sh_abd_r=0.1, elbow_l=0.15, elbow_r=0.15, hip_abd_l=0.04, hip_abd_r=0.04)
SEAT_DROP = -(HIP_Z - 0.5)  # root z offset so the pelvis sits on a 0.48 m seat


ARM_JOINTS = ("sh_flex", "sh_abd", "sh_rot", "elbow", "wrist")


def solve_mic_hold(model, pm: PersonModel, base: dict, side: str = "r", offset=(0.05, -0.01, -0.035)) -> dict:
    """Arm pose (added to ``base``) that puts the mic head just in front of and below the mouth."""
    d = mujoco.MjData(model)
    names = [f"{j}_{side}" for j in ARM_JOINTS]
    adr = np.array([pm.qadr[n] for n in names])
    dof = np.array([model.jnt_dofadr[model.joint(f"{pm.name}_{n}").id] for n in names])
    d.mocap_pos[pm.root_mocap] = [0, 0, 0]
    d.mocap_quat[pm.root_mocap] = [1, 0, 0, 0]
    for j, a in pm.qadr.items():
        d.qpos[a] = base.get(j, 0.0)
    q = np.array([0.45, -0.05, 0.3, 2.1, 0.0])  # natural mic hold: elbow down at the side, forearm up
    rest = q.copy()
    jac = np.zeros((3, model.nv))
    for _ in range(200):
        d.qpos[adr] = q
        mujoco.mj_kinematics(model, d)
        mujoco.mj_comPos(model, d)
        target = d.site_xpos[pm.mouth_site] + np.asarray(offset)
        err = target - d.site_xpos[pm.mic_site]
        if err @ err < 1e-8:
            break
        mujoco.mj_jacSite(model, d, jac, None, pm.mic_site)
        J = jac[:, dof]
        # damped least squares with a null-space pull toward the natural pose
        Jp = J.T @ np.linalg.inv(J @ J.T + 1e-3 * np.eye(3))
        q = q + Jp @ err + 0.1 * (np.eye(5) - Jp @ J) @ (rest - q)
    out = dict(base)
    out.update(zip(names, q.tolist()))
    return out
