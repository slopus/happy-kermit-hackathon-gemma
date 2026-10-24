"""Hardware feasibility checks for a trajectory against the real xArm6 and hand limits."""
from __future__ import annotations

import mujoco
import numpy as np

from . import xarm6
from .scene import Scene
from .trajectory import Trajectory

MARGIN = 0.8  # flag anything above 80 % of a hardware limit as a warning


def _derivs(x: np.ndarray, dt: float):
    v = np.gradient(x, dt, axis=0)
    a = np.gradient(v, dt, axis=0)
    return v, a


def check(scene: Scene, traj: Trajectory) -> dict:
    dt = float(traj.t[1] - traj.t[0])
    m = scene.model
    d = mujoco.MjData(m)
    v, a = _derivs(traj.arm, dt)
    report: dict = {"dt": dt, "frames": len(traj.t), "arm": [], "hand": [], "tcp": {}, "ok": True, "warnings": []}

    def judge(peak, limit):
        r = peak / limit
        return "FAIL" if r > 1.0 else ("WARN" if r > MARGIN else "ok"), round(float(r), 3)

    for j in range(6):
        lo, hi = xarm6.JOINT_RANGE[j]
        pos_margin = float(min(traj.arm[:, j].min() - lo, hi - traj.arm[:, j].max()))
        vs, vr = judge(np.abs(v[:, j]).max(), xarm6.JOINT_VEL_MAX[j])
        as_, ar = judge(np.abs(a[:, j]).max(), xarm6.JOINT_ACC_MAX[j])
        entry = dict(joint=f"joint{j + 1}", min_deg=round(float(np.rad2deg(traj.arm[:, j].min())), 1),
                     max_deg=round(float(np.rad2deg(traj.arm[:, j].max())), 1),
                     range_margin_deg=round(float(np.rad2deg(pos_margin)), 1),
                     vel_peak_dps=round(float(np.rad2deg(np.abs(v[:, j]).max())), 1), vel_ratio=vr, vel=vs,
                     acc_peak_dps2=round(float(np.rad2deg(np.abs(a[:, j]).max())), 0), acc_ratio=ar, acc=as_)
        if pos_margin < 0:
            entry["range"] = "FAIL"
        report["arm"].append(entry)
        for k in ("vel", "acc", "range"):
            if entry.get(k) == "FAIL":
                report["ok"] = False
            if entry.get(k) in ("FAIL", "WARN"):
                report["warnings"].append(f"{entry['joint']} {k} {entry.get(k)}")

    # TCP (head site) speed
    site = m.site("head").id
    pos = np.zeros((len(traj.t), 3))
    contacts = 0
    min_floor = np.inf
    for i in range(len(traj.t)):
        d.qpos[:] = 0
        d.qpos[scene.arm_qadr] = traj.arm[i]
        d.qpos[scene.hand.qadr] = traj.hand[i]
        mujoco.mj_kinematics(m, d)
        pos[i] = d.site_xpos[site]
        min_floor = min(min_floor, float(d.xpos[1:, 2].min()))
    tv = np.linalg.norm(np.gradient(pos, dt, axis=0), axis=1)
    s, r = judge(tv.max(), xarm6.TCP_VEL_MAX)
    report["tcp"] = dict(vel_peak_mps=round(float(tv.max()), 3), vel_ratio=r, vel=s,
                         min_body_height_m=round(min_floor, 3))
    if s == "FAIL":
        report["ok"] = False

    hv = np.abs(np.gradient(traj.hand, dt, axis=0)).max(axis=0)
    for j, name in enumerate(scene.hand.joint_names):
        lo, hi = scene.hand.range[j]
        s, r = judge(hv[j], scene.hand.vel_max[j]) if scene.hand.vel_max[j] > 0 else ("ok", 0.0)
        in_range = bool(traj.hand[:, j].min() >= lo - 1e-6 and traj.hand[:, j].max() <= hi + 1e-6)
        if hv[j] > 1e-6:
            report["hand"].append(dict(joint=name, vel_peak=round(float(hv[j]), 3), vel_ratio=r, vel=s,
                                       in_range=in_range))
        if s == "FAIL" or not in_range:
            report["ok"] = False
            report["warnings"].append(f"{name} vel {s} in_range={in_range}")
    report["hand_vel_note"] = scene.hand.vel_note
    report["ik_err_max"] = float(traj.ik_err.max())
    report["governor_hz"] = traj.governor_hz
    if traj.governor_hz:
        report["warnings"].append(f"arm motion low-passed at {traj.governor_hz} Hz to fit limits")
    if traj.ik_err.max() > 1e-3:
        report["warnings"].append(f"IK residual {traj.ik_err.max():.2e}")
    return report


def summary(report: dict) -> str:
    lines = [f"hardware check: {'OK' if report['ok'] else 'FAIL'}  (IK err max {report['ik_err_max']:.1e})"]
    for e in report["arm"]:
        lines.append(f"  {e['joint']}: [{e['min_deg']:7.1f}, {e['max_deg']:7.1f}] deg  margin {e['range_margin_deg']:6.1f}"
                     f"  vel {e['vel_peak_dps']:6.1f} deg/s ({e['vel_ratio'] * 100:3.0f}%)"
                     f"  acc {e['acc_peak_dps2']:6.0f} deg/s2 ({e['acc_ratio'] * 100:3.0f}%)")
    t = report["tcp"]
    lines.append(f"  head speed peak {t['vel_peak_mps']:.2f} m/s ({t['vel_ratio'] * 100:.0f}%)")
    if report["hand"]:
        worst = max(report["hand"], key=lambda h: h["vel_ratio"])
        lines.append(f"  hand worst joint {worst['joint']}: {worst['vel_peak']:.2f} ({worst['vel_ratio'] * 100:.0f}% of limit)")
    if report["warnings"]:
        lines.append("  warnings: " + "; ".join(report["warnings"]))
    return "\n".join(lines)
