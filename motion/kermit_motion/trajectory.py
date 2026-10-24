"""Animation channels -> joint trajectories (arm via head IK, hand via the profile's mouth map)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import xarm6
from .animation import Animation, lowpass, slew_limit
from .ik import SEED, HeadIK, euler_head
from .scene import HEAD_REST_POS, NECK_BACK, Scene


@dataclass
class Trajectory:
    t: np.ndarray
    arm: np.ndarray          # (N, 6) rad
    hand: np.ndarray         # (N, nh) hand joint positions (profile.joint_names)
    mouth_cmd: np.ndarray    # (N,) mouth openness before hand speed limiting
    ik_err: np.ndarray       # (N,) IK residual
    governor_hz: float | None = None  # arm low-pass cutoff the safety governor had to apply (None = untouched)


def _arm_ratio(arm: np.ndarray, dt: float) -> float:
    v = np.gradient(arm, dt, axis=0)
    a = np.gradient(v, dt, axis=0)
    return max(float((np.abs(v) / xarm6.JOINT_VEL_MAX).max()), float((np.abs(a) / xarm6.JOINT_ACC_MAX).max()))


def govern(arm: np.ndarray, dt: float, budget: float = 0.95):
    """Safety net: if the keyed motion exceeds the arm's speed/acceleration limits, low-pass the joint
    trajectories (zero phase, so no A/V lag) with the highest cutoff that fits. Gestures are designed
    not to need this; the cutoff is reported so a violating gesture can be re-keyed."""
    if _arm_ratio(arm, dt) <= budget:
        return arm, None
    for hz in (8.0, 6.0, 5.0, 4.0, 3.0, 2.5, 2.0, 1.5, 1.0):
        out = np.stack([lowpass(arm[:, j], 1 / dt, hz) for j in range(6)], axis=1)
        if _arm_ratio(out, dt) <= budget:
            return out, hz
    return out, hz


def solve(scene: Scene, anim: Animation, hw_limited_hand: bool = True) -> Trajectory:
    ik = HeadIK(scene)
    c = anim.ch
    n = len(anim.t)
    q = SEED.copy()
    q, _ = ik.solve(HEAD_REST_POS, euler_head(0, 0, 0), q, iters=200)
    arm = np.zeros((n, 6))
    err = np.zeros(n)
    for i in range(n):
        # rotate about a neck point behind the face, then translate
        R = euler_head(c["yaw"][i], c["pitch"][i], c["roll"][i])
        neck = HEAD_REST_POS - np.array([NECK_BACK, 0, 0])
        pos = neck + R @ np.array([NECK_BACK, 0, 0]) + np.array([c["dx"][i], c["dy"][i], c["dz"][i]])
        # person-following bias: the whole puppet pose turns about the robot's base axis
        b = c.get("base_yaw", np.zeros(n))[i]
        Rb = euler_head(b, 0, 0)
        pos, R = Rb @ pos, Rb @ R
        q, err[i] = ik.solve(pos, R, q, iters=30)
        arm[i] = q
    dt = anim.t[1] - anim.t[0]
    arm, gov = govern(arm, dt)
    hand = scene.hand.mouth_qpos(c["mouth"])
    if hw_limited_hand:
        hand = np.stack([slew_limit(hand[:, j], dt, 0.97 * scene.hand.vel_max[j]) if scene.hand.vel_max[j] > 0
                         else hand[:, j] for j in range(hand.shape[1])], axis=1)
    return Trajectory(t=anim.t, arm=arm, hand=hand, mouth_cmd=c["mouth"], ik_err=err, governor_hz=gov)
