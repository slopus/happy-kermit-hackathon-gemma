"""Stage scene: xArm6 on a small riser, a puppet hand on the flange, curtain backdrop, 3/4 camera."""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from . import costume, human
from .hands import PROFILES, HandProfile
from .xarm6 import add_xarm6

# Neutral head pose in world (robot base frame): facing +x, toward the audience / camera.
HEAD_REST_POS = np.array([0.45, 0.0, 0.25])
# head rotations pivot about a neck point this far behind the head site (like a puppeteer's wrist)
NECK_BACK = 0.10
# 3/4 front camera relative to the rest head position: in front, to the puppet's left, a bit high.
STAGE_Z = -0.32
# 3/4 front camera: ~25 deg off the puppet's facing direction, toward its left, a little high
CAM_OFFSET = 1.2 * np.array([np.cos(np.deg2rad(25)), np.sin(np.deg2rad(25)), 0.14])
CAM_TARGET_OFFSET = np.array([0.0, 0.0, 0.0])
CAM_FOVY = 33


@dataclass
class Scene:
    model: mujoco.MjModel
    hand: HandProfile
    arm_qadr: np.ndarray
    arm_dof: np.ndarray
    has_body: bool = False
    floor_z: float = STAGE_Z


def _lookat_xyaxes(eye, target):
    z = eye - target
    z /= np.linalg.norm(z)
    x = np.cross([0, 0, 1], z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.r_[x, y]


def _add_person(spec: mujoco.MjSpec):
    """Person stand-in (mocap): capsule body plus head sphere, origin at the face point."""
    p = spec.worldbody.add_body(name="person", mocap=True, pos=[2.0, 0, 0.5])
    skin = (0.86, 0.68, 0.55, 1)
    shirt = (0.22, 0.38, 0.62, 1)
    p.add_geom(name="person_head", type=mujoco.mjtGeom.mjGEOM_ELLIPSOID, size=[0.085, 0.075, 0.105],
               pos=[0.02, 0, 0], rgba=skin, contype=0, conaffinity=0)
    p.add_geom(name="person_hair", type=mujoco.mjtGeom.mjGEOM_ELLIPSOID, size=[0.09, 0.08, 0.07],
               pos=[0.04, 0, 0.05], rgba=(0.18, 0.12, 0.08, 1), contype=0, conaffinity=0)
    p.add_geom(name="person_torso", type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=[0.16, 0, 0],
               fromto=[0.03, 0, -0.32, 0.03, 0, -0.62], rgba=shirt, contype=0, conaffinity=0)
    p.add_geom(name="person_legs", type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=[0.12, 0, 0],
               fromto=[0.03, 0, -0.8, 0.03, 0, -1.45], rgba=(0.15, 0.15, 0.2, 1), contype=0, conaffinity=0)
    # the person faces the robot: their body frame x points away from it (set per frame by the renderer)


def _add_tracking_camera(spec: mujoco.MjSpec):
    from .follow import CAMERA_POS
    b = spec.body("link_base")
    b.add_geom(name="trackcam_box", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.02, 0.045, 0.02],
               pos=list(CAMERA_POS + np.array([0.04, 0, 0])), rgba=(0.08, 0.08, 0.09, 1), contype=0, conaffinity=0)
    # the perception camera itself (CONTRACT §4 default pose, optical axis = base +x), ~78 deg HFOV at 4:3
    b.add_camera(name="base_cam", pos=list(CAMERA_POS + np.array([0.07, 0, 0])), xyaxes=[0, -1, 0, 0, 0, 1],
                 fovy=float(np.degrees(2 * np.arctan(np.tan(np.radians(39)) * 3 / 4))))
    b.add_geom(name="trackcam_lens", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.011, 0.006, 0],
               pos=list(CAMERA_POS + np.array([0.064, 0, 0])), quat=[0.7071, 0, 0.7071, 0],
               rgba=(0.15, 0.25, 0.45, 1), contype=0, conaffinity=0)


def _add_ptz(spec: mujoco.MjSpec, mount, floor_z: float):
    """Insta360 Link 2 on a post beside the arm: a side table up to the base height, a ~0.5 m post, and the
    gimbal as two mocap bodies, ``ptz_yoke`` (pan) and ``ptz_head`` (pan + tilt) carrying ``ptz_cam``.
    The head's origin is the optical centre at ``mount``; LiveWorld poses both bodies every step."""
    mount = np.asarray(mount, float)
    w = spec.worldbody
    dark, grey = (0.07, 0.07, 0.08, 1), (0.16, 0.16, 0.17, 1)
    nc = dict(contype=0, conaffinity=0)
    w.add_geom(name="ptz_table", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.09, 0.09, -floor_z / 2],
               pos=[mount[0], mount[1], floor_z / 2], rgba=(0.12, 0.12, 0.13, 1), **nc)
    w.add_geom(name="ptz_post_foot", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.05, 0.006, 0],
               pos=[mount[0], mount[1], 0.006], rgba=grey, **nc)
    w.add_geom(name="ptz_post", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.012, 0, 0],
               fromto=[mount[0], mount[1], 0.0, mount[0], mount[1], mount[2] - 0.062], rgba=grey, **nc)
    w.add_geom(name="ptz_base", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.026, 0.007, 0],
               pos=[mount[0], mount[1], mount[2] - 0.055], rgba=dark, **nc)
    yoke = w.add_body(name="ptz_yoke", mocap=True, pos=list(mount))
    for side in (-1, 1):
        yoke.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.012, 0.004, 0.028], pos=[0, side * 0.029, -0.02],
                      rgba=dark, **nc)
    yoke.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.014, 0.033, 0.004], pos=[0, 0, -0.047], rgba=dark, **nc)
    head = w.add_body(name="ptz_head", mocap=True, pos=list(mount))
    head.add_geom(name="ptz_shell", type=mujoco.mjtGeom.mjGEOM_ELLIPSOID, size=[0.024, 0.025, 0.024],
                  rgba=(0.10, 0.10, 0.11, 1), **nc)
    head.add_geom(name="ptz_lens", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.011, 0.003, 0],
                  pos=[0.022, 0, 0.002], quat=[0.7071, 0, 0.7071, 0], rgba=(0.03, 0.05, 0.1, 1), **nc)
    head.add_geom(name="ptz_led", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.0018, 0, 0], pos=[0.021, 0, -0.016],
                  rgba=(0.2, 0.6, 1.0, 1), **nc)
    head.add_camera(name="ptz_cam", pos=[0.026, 0, 0.002], xyaxes=[0, -1, 0, 0, 0, 1], fovy=44.4)


def build(hand: str = "shadow", shell: bool = True, width: int = 1280, height: int = 720,
          person: bool = False, people: list[tuple] = (), floor_z: float = STAGE_Z, ptz_mount=None) -> Scene:
    """``people``: [(name, style, mic, chair_pose or None)] articulated humans (human.py).
    ``floor_z``: stage floor below the robot base; people scenes put the robot on a table (-0.8).
    ``ptz_mount``: optical-centre position (base frame) of a PTZ camera on a post beside the arm."""
    spec = mujoco.MjSpec()
    spec.compiler.degree = False
    spec.modelname = f"kermit_{hand}"
    spec.visual.global_.offwidth = max(1920, width)
    spec.visual.global_.offheight = max(1080, height)
    spec.visual.quality.shadowsize = 4096
    spec.visual.headlight.ambient = [0.25, 0.25, 0.25]
    spec.visual.headlight.diffuse = [0.35, 0.35, 0.35]
    spec.visual.headlight.specular = [0.05, 0.05, 0.05]
    spec.visual.map.znear = 0.01
    # the robot is kinematic (replayed); only the costume body is simulated, with no contacts
    spec.option.timestep = 1 / 480
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT

    spec.add_texture(name="sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX, builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                     rgb1=[0.10, 0.08, 0.10], rgb2=[0.02, 0.02, 0.03], width=256, height=1536)
    spec.add_texture(name="boards", type=mujoco.mjtTexture.mjTEXTURE_2D, builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                     rgb1=[0.42, 0.27, 0.15], rgb2=[0.47, 0.31, 0.18], width=512, height=512)
    spec.add_material(name="stage", textures=["", "boards"], texrepeat=[6, 1], reflectance=0.12)
    spec.add_texture(name="curtain", type=mujoco.mjtTexture.mjTEXTURE_2D, builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                     rgb1=[0.62, 0.05, 0.08], rgb2=[0.25, 0.01, 0.03], width=64, height=64)
    spec.add_material(name="curtain", textures=["", "curtain"], texrepeat=[14, 1], specular=0.1)

    w = spec.worldbody
    # key, fill and rim lights
    w.add_light(name="key", pos=[1.6, 1.0, 1.9], dir=[-0.7, -0.45, -0.6], diffuse=[0.75, 0.72, 0.65],
                specular=[0.3, 0.3, 0.3], castshadow=True, cutoff=40, exponent=4)
    w.add_light(name="fill", pos=[1.4, -1.2, 1.0], dir=[-0.6, 0.6, -0.3], diffuse=[0.3, 0.32, 0.38],
                castshadow=False)
    w.add_light(name="rim", pos=[-0.4, 0.3, 1.4], dir=[0.6, -0.1, -0.5], diffuse=[0.35, 0.35, 0.3], castshadow=False)

    w.add_geom(name="stage", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[4, 4, 0.1], material="stage", pos=[0, 0, floor_z])
    # pleated curtain: a row of thin vertical capsules behind the robot
    for i in range(60):
        y = -2.4 + i * 0.08
        w.add_geom(name=f"pleat{i}", type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=[0.05, 1.3, 0],
                   pos=[-0.75 + 0.02 * (i % 2), y, floor_z + 1.3], rgba=[0.55 - 0.08 * (i % 2), 0.04, 0.07, 1],
                   contype=0, conaffinity=0)
    # riser the arm stands on
    w.add_geom(name="riser", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.16, 0.16, -floor_z / 2], pos=[0, 0, floor_z / 2],
               rgba=[0.12, 0.12, 0.13, 1], contype=0, conaffinity=0)

    flange = add_xarm6(spec, w)
    profile = PROFILES[hand]()
    profile.attach(spec, flange, shell)
    body = costume.add_body(spec) if shell else None
    if person:
        _add_person(spec)
    if person or people:
        _add_tracking_camera(spec)
    for name, style, mic, chair in people:
        human.add_person(spec, name, style, mic=mic)
        if chair is not None:
            human.add_chair(spec, name, chair[:2], chair[2], floor_z)
    if ptz_mount is not None:
        _add_ptz(spec, ptz_mount, floor_z)

    # Aim the cameras at the dressed puppet in its rest pose: compile once, solve the rest IK,
    # measure the puppet's bounding box, then add the cameras and compile again.
    from .ik import SEED, HeadIK, euler_head  # local import: ik depends on Scene
    model = spec.compile()
    tmp = Scene(model=model, hand=profile, arm_qadr=model.jnt_qposadr[[model.joint(f"joint{i + 1}").id for i in range(6)]],
                arm_dof=model.jnt_dofadr[[model.joint(f"joint{i + 1}").id for i in range(6)]])
    ik = HeadIK(tmp)
    ik.solve(HEAD_REST_POS, euler_head(0, 0, 0), SEED, iters=200)
    mujoco.mj_kinematics(model, ik.d)
    hand_bodies = [b for b in range(model.nbody) if model.body(b).name.startswith(("hand/", "pgc", "jaw_"))]
    pts = ik.d.xpos[hand_bodies]
    center = 0.5 * (pts.min(axis=0) + pts.max(axis=0))
    if body:  # frame the whole puppet: head plus the body hanging below it
        center = ik.d.site_xpos[model.site("head").id] + np.array([0.07, 0.05, -0.02])
    center = center + CAM_TARGET_OFFSET

    eye = center + CAM_OFFSET
    w.add_camera(name="front34", pos=eye, xyaxes=_lookat_xyaxes(eye, center), fovy=CAM_FOVY)
    eye2 = center + np.array([0.0, 0.85, 0.03])
    w.add_camera(name="side", pos=eye2, xyaxes=_lookat_xyaxes(eye2, center), fovy=32)
    eye3 = center + np.array([0.8, 0.0, 0.05])
    w.add_camera(name="front", pos=eye3, xyaxes=_lookat_xyaxes(eye3, center), fovy=30)
    # wide shot for person following: high, behind the person's path, looking over their heads at the
    # puppet, so the person crosses the frame and the puppet's face stays visible
    eye5 = np.array([2.75, -1.25, 1.4])
    w.add_camera(name="follow", pos=eye5, xyaxes=_lookat_xyaxes(eye5, np.array([0.95, 0.05, 0.1])), fovy=40)
    eye4 = np.array([1.5, 1.1, 0.95])
    w.add_camera(name="wide", pos=eye4, xyaxes=_lookat_xyaxes(eye4, np.array([0.15, 0.0, 0.3])), fovy=40)

    model = spec.compile()
    profile.prepare(model)
    arm = [model.joint(f"joint{i + 1}").id for i in range(6)]
    return Scene(model=model, hand=profile, arm_qadr=model.jnt_qposadr[arm], arm_dof=model.jnt_dofadr[arm],
                 has_body=body is not None, floor_z=floor_z)
