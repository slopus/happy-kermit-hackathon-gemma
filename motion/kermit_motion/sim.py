"""Playback: replay the robot trajectory kinematically and simulate the puppet's dangling body."""
from __future__ import annotations

import mujoco
import numpy as np

from .scene import Scene

# neck anchor (top of the collar) relative to the head site, in the head's yaw-only frame
NECK_OFFSET = np.array([0.045, 0.0, -0.045])
BODY_YAW_FOLLOW = 0.6  # the body turns with the head, but less


class PuppetSim:
    def __init__(self, scene: Scene):
        self.s = scene
        self.m = scene.model
        self.d = mujoco.MjData(self.m)
        self.hand = scene.hand
        self.head = self.m.site("head").id
        self.body = scene.has_body
        if self.body:
            self.mocap = self.m.body_mocapid[self.m.body("puppet_anchor").id]
            j = self.m.joint("puppet_torso")
            self.torso_q = self.m.jnt_qposadr[j.id]
        self.robot_q = np.r_[scene.arm_qadr, scene.hand.qadr]

    def _pose_robot(self, arm, hand):
        self.d.qpos[self.s.arm_qadr] = arm
        self.d.qpos[self.hand.qadr] = hand
        mujoco.mj_kinematics(self.m, self.d)
        if self.hand.dress_pose(self.m, self.d):
            mujoco.mj_kinematics(self.m, self.d)

    def _anchor(self):
        p = self.d.site_xpos[self.head]
        R = self.d.site_xmat[self.head].reshape(3, 3)
        yaw = np.arctan2(R[1, 0], R[0, 0])
        c, s = np.cos(yaw), np.sin(yaw)
        pos = p + np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]) @ NECK_OFFSET
        by = BODY_YAW_FOLLOW * yaw
        return pos, np.array([np.cos(by / 2), 0, 0, np.sin(by / 2)])

    def reset(self, arm, hand, settle: float = 1.5):
        mujoco.mj_resetData(self.m, self.d)
        self._pose_robot(arm, hand)
        if self.body:
            pos, quat = self._anchor()
            self.d.mocap_pos[self.mocap], self.d.mocap_quat[self.mocap] = pos, quat
            self.d.qpos[self.torso_q:self.torso_q + 3] = pos
            self.d.qpos[self.torso_q + 3:self.torso_q + 7] = quat
            for _ in range(int(settle / self.m.opt.timestep)):
                self._step(arm, hand)

    def _step(self, arm, hand):
        self._pose_robot(arm, hand)
        if self.body:
            pos, quat = self._anchor()
            self.d.mocap_pos[self.mocap], self.d.mocap_quat[self.mocap] = pos, quat
        self.d.qvel[self.s.arm_dof] = 0
        mujoco.mj_step(self.m, self.d)
        # the robot is replayed, not simulated: put it back exactly where the trajectory says
        self._pose_robot(arm, hand)

    def advance(self, arm0, arm1, hand0, hand1, dt: float):
        """Move from one frame to the next over ``dt`` seconds, sub-stepping the body physics."""
        n = max(1, int(round(dt / self.m.opt.timestep)))
        for k in range(1, n + 1):
            a = k / n
            self._step((1 - a) * arm0 + a * arm1, (1 - a) * hand0 + a * hand1)
        mujoco.mj_forward(self.m, self.d)
