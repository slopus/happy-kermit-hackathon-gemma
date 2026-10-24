"""UFactory xArm6 model (MjSpec) and its hardware limits.

Kinematics, inertias and meshes come from the official xarm_description package (copied into
``assets/xarm_description``); the joint frames match ``Tasks/task/.../sim/xarm_sim.py``.
"""
from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import yaml
from scipy.spatial.transform import Rotation

ASSETS = Path(__file__).resolve().parent.parent / "assets"
XARM = ASSETS / "xarm_description"

JOINT_XYZ = [(0, 0, .267), (0, 0, 0), (.0535, -.2845, 0),
             (.0775, .3425, 0), (0, 0, 0), (.076, .097, 0)]
JOINT_RPY = [(0, 0, 0), (-np.pi / 2, 0, 0), (0, 0, 0),
             (-np.pi / 2, 0, 0), (np.pi / 2, 0, 0), (-np.pi / 2, 0, 0)]

# Hardware limits used by the safety checker (limits.py). Position ranges are the conservative
# real-session ranges from the calibration handoff (narrower than the datasheet on j2, j3, j5).
# Speed/acceleration are the xArm6 datasheet / SDK maxima: 180 deg/s per joint, 1145 deg/s^2.
JOINT_RANGE = np.deg2rad([[-360, 360], [-117, 116], [-219, 10],
                          [-360, 360], [-97, 180], [-360, 360]])
JOINT_VEL_MAX = np.deg2rad([180.0] * 6)
JOINT_ACC_MAX = np.deg2rad([1145.0] * 6)
TCP_VEL_MAX = 1.0  # m/s

SILVER = (0.86, 0.87, 0.89, 1)
DARK = (0.30, 0.31, 0.34, 1)


def _quat(rpy) -> list[float]:
    q = Rotation.from_euler("xyz", rpy).as_quat()
    return [q[3], q[0], q[1], q[2]]


def add_xarm6(spec: mujoco.MjSpec, parent: mujoco.MjsBody, pos=(0, 0, 0)) -> mujoco.MjsSite:
    """Add the arm under ``parent``; returns the ``flange`` site on link6 (z out of the flange)."""
    inertias = yaml.safe_load((XARM / "config/link_inertial/xarm6_type6_HT_BR.yaml").read_text())
    names = ["link_base"] + [f"link{i}" for i in range(1, 7)]
    for name in names:
        spec.add_mesh(name=f"xarm_{name}", file=str(XARM / f"meshes/xarm6/visual/{name}.stl"))
    mat_silver = spec.add_material(name="xarm_silver", rgba=SILVER, specular=0.6, shininess=0.6)
    mat_dark = spec.add_material(name="xarm_dark", rgba=DARK, specular=0.4, shininess=0.4)

    def mesh_geom(body, name, material):
        body.add_geom(name=f"{name}_vis", type=mujoco.mjtGeom.mjGEOM_MESH, meshname=f"xarm_{name}",
                      material=material.name, contype=0, conaffinity=0, group=2)

    body = parent.add_body(name="link_base", pos=pos)
    mesh_geom(body, "link_base", mat_silver)
    for i in range(6):
        name = f"link{i + 1}"
        body = body.add_body(name=name, pos=JOINT_XYZ[i], quat=_quat(JOINT_RPY[i]))
        p = inertias[name]
        inertia = p["inertia"]
        body.mass = p["mass"]
        body.ipos = [p["origin"][k] for k in "xyz"]
        body.fullinertia = [inertia[k] for k in ["ixx", "iyy", "izz", "ixy", "ixz", "iyz"]]
        body.explicitinertial = True
        body.add_joint(name=f"joint{i + 1}", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0, 0, 1],
                       range=JOINT_RANGE[i], limited=mujoco.mjtLimited.mjLIMITED_TRUE)
        mesh_geom(body, name, mat_dark if i == 5 else mat_silver)
    return body.add_site(name="flange", pos=[0, 0, 0], size=[0.004] * 3, group=5)
