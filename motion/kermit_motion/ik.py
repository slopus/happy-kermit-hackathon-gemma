"""Head-pose IK: place the puppet's ``head`` site at a world pose with the xArm6's six joints."""
from __future__ import annotations

import mujoco
import numpy as np

from .scene import Scene

# Rest pose found by a sweep over head positions and IK branches: the forearm (link4) hangs almost
# vertically and the wrist is bent ~90 deg, like a hand puppet worn from above. Then joint4 is the
# head's yaw, joint5 its pitch and joint6 its roll, so head turns don't need large wrist
# counter-rotations.
SEED = np.deg2rad([0.0, 4.0, -25.0, 180.0, 92.0, 90.0])


def euler_head(yaw: float, pitch: float, roll: float) -> np.ndarray:
    """World rotation of the head: yaw about z (+ = look left), pitch about y (+ = nose down),
    roll about x (+ = tilt toward its right shoulder). Neutral faces +x with z up."""
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    return Rz @ Ry @ Rx


class HeadIK:
    def __init__(self, scene: Scene):
        self.s = scene
        self.m = scene.model
        self.d = mujoco.MjData(self.m)
        self.site = self.m.site("head").id
        self.jacp = np.zeros((3, self.m.nv))
        self.jacr = np.zeros((3, self.m.nv))
        lo, hi = self.m.jnt_range[[self.m.joint(f"joint{i + 1}").id for i in range(6)]].T
        self.lo, self.hi = lo, hi

    def fk(self, q: np.ndarray):
        self.d.qpos[self.s.arm_qadr] = q
        mujoco.mj_kinematics(self.m, self.d)
        mujoco.mj_comPos(self.m, self.d)
        return self.d.site_xpos[self.site].copy(), self.d.site_xmat[self.site].reshape(3, 3).copy()

    def solve(self, pos: np.ndarray, R: np.ndarray, q0: np.ndarray, iters: int = 50, tol: float = 1e-6):
        q = q0.copy()
        for _ in range(iters):
            p, Rc = self.fk(q)
            ep = pos - p
            er = np.zeros(3)
            # rotation error as a world-frame rotation vector of R * Rc^T
            qe, qc, qt = np.zeros(4), np.zeros(4), np.zeros(4)
            mujoco.mju_mat2Quat(qc, Rc.ravel())
            mujoco.mju_mat2Quat(qt, np.ascontiguousarray(R).ravel())
            mujoco.mju_negQuat(qe, qc)
            mujoco.mju_mulQuat(qe, qt, qe)
            mujoco.mju_quat2Vel(er, qe, 1.0)
            err = np.r_[ep, er]
            if err @ err < tol * tol:
                break
            mujoco.mj_jacSite(self.m, self.d, self.jacp, self.jacr, self.site)
            J = np.r_[self.jacp[:, self.s.arm_dof], self.jacr[:, self.s.arm_dof]]
            dq = J.T @ np.linalg.solve(J @ J.T + 1e-6 * np.eye(6), err)
            q = np.clip(q + dq, self.lo, self.hi)
        return q, float(np.sqrt(err @ err))
