"""Render a multi-person stage script (presenter / judges + puppet) to MP4 with camera cuts.

    /var/lib/happy-agent/IsaacLab/.venv/bin/python render_stage.py samples/stage/turnaround.stage.json \
        -o videos/turnaround.mp4

The puppet's gaze follows CONTRACT §5 ``speaker_id`` (simulated from the script, with hysteresis);
its own lines play the VOICE timelines with the usual gesture/lip-sync pipeline (render.py).
Also writes ``out.perception.ndjson`` (the synthetic §5 log), ``out.check.json`` and ``out.traj.npz``.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import warnings
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
warnings.filterwarnings("ignore", message="Attach conflict")

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kermit_motion import follow, limits, scene as scene_mod, stage as stage_mod  # noqa: E402
from kermit_motion.animation import animate  # noqa: E402
from kermit_motion.sim import PuppetSim  # noqa: E402
from kermit_motion.trajectory import solve  # noqa: E402
from render import Hud  # noqa: E402

HEAD = scene_mod.HEAD_REST_POS


def _clear_azimuth(tgt, want, standing, dist, clearance=0.42):
    """Nudge a camera azimuth around ``tgt`` until no standing person blocks the line of sight."""
    for da in np.radians([0, 8, -8, 16, -16, 24, -24, 32, -32, 45, -45]):
        a = want + da
        seg = dist * np.array([np.cos(a), np.sin(a)])
        if all(np.linalg.norm(tgt[:2] + np.clip((P - tgt[:2]) @ seg / (seg @ seg), 0, 1) * seg - P) >= clearance
               or (P - tgt[:2]) @ seg <= 0.05 * (seg @ seg) for P in standing):
            return a
    return want


def kermit_cu(gaze_yaw: float, standing) -> tuple:
    """Puppet 3/4 close-up (the hero.mp4 look) ~25 deg off where it is looking, on the audience side."""
    look = 0.6 * gaze_yaw  # the head turns ~60 % of the way to its target (the base adds the rest)
    want = look - np.radians(25) * np.sign(look) if abs(look) > np.radians(12) else np.radians(20)
    k = HEAD + np.array([0.06 * np.cos(look), 0.06 * np.sin(look), -0.03])
    a = _clear_azimuth(k, want, standing, 0.95)
    return (k + 0.95 * np.array([np.cos(a), np.sin(a), 0.14]), k, 33)


def default_cameras(st) -> dict:
    """name -> (eye, target, fovy). ``kermit_cu@<deg>`` close-ups are made on demand (see camera_for)."""
    presenters = [p for p in st.people if p.role == "presenter"]
    judges = [p for p in st.people if p.role == "judge"]
    standing = [p.pos for p in st.people if not p.seated]
    cams = {"kermit_cu": kermit_cu(np.arctan2(*(presenters[0].pos - HEAD[:2])[::-1]) if presenters else 0.0,
                                   standing)}
    for pres in presenters:
        P = np.array([pres.pos[0], pres.pos[1], st.floor_z + 1.3])
        K = HEAD + np.array([0.0, 0, -0.05])
        line = P - K
        perp = np.array([-line[1], line[0], 0.0])
        perp /= np.linalg.norm(perp)
        if perp[0] < 0:  # the audience side of the puppet-presenter line
            perp = -perp
        tgt = K + 0.45 * line + np.array([0, 0, 0.08])
        others = [p.pos for p in presenters if p is not pres]
        side = np.sign(pres.pos[1] - HEAD[1]) or 1.0
        for rot in np.radians([0, 10, 20, 30, 40]):  # swing toward the presenter's side until the other one is out
            c, s_ = np.cos(rot * side), np.sin(rot * side)
            pr = np.array([c * perp[0] - s_ * perp[1], s_ * perp[0] + c * perp[1], 0.0])
            eye = tgt + 1.75 * pr + np.array([0, 0, 0.24])
            view = np.arctan2(tgt[1] - eye[1], tgt[0] - eye[0])
            if all(abs((np.arctan2(o[1] - eye[1], o[0] - eye[0]) - view + np.pi) % (2 * np.pi) - np.pi)
                   > np.radians(48) for o in others):
                break
        cams[f"two_{pres.name}"] = (eye, tgt, 50)
        if pres is presenters[0]:
            cams["two_shot"] = cams[f"two_{pres.name}"]
    if len(presenters) >= 2:  # both presenters and the puppet, from the audience
        cams["three_shot"] = (np.array([2.6, 0.25, 0.75]), np.array([0.55, 0.0, 0.1]), 42)
    cams["stage_wide"] = (np.array([3.3, 1.6, 1.25]), np.array([0.9, -0.15, -0.05]), 44)
    if judges:
        jc = np.mean([p.pos for p in judges], axis=0)
        eye = np.array([jc[0] + 1.6, jc[1] + 0.55, 1.35])
        cams["qa_wide"] = (eye, HEAD + np.array([0.35, 0.0, -0.18]), 40)
    cams["_standing"] = standing
    return cams


def camera_for(cams: dict, name: str, standing=None) -> tuple:
    """Close-ups are built on first use, avoiding wherever the standing people are at that moment."""
    if name not in cams and name.startswith("kermit_cu@"):
        cams[name] = kermit_cu(np.radians(float(name.split("@")[1])), standing or cams["_standing"])
    return cams.get(name, cams["kermit_cu"])


def render_stage(script: Path, out: Path, hand: str = "shadow", width: int = 1280, height: int = 720,
                 fps: int = 60, hud: bool = True):
    t0 = time.time()
    st = stage_mod.build_stage(script, fps=fps)
    sc = scene_mod.build(hand, width=width, height=height, people=st.people_spec(), floor_z=st.floor_z)
    m = sc.model
    st.bind(m)
    sim = PuppetSim(sc)
    d = sim.d

    # 1. what PERCEPTION would see: people posed on the scene clock, puppet at rest
    stage_mod.perceive(st, m, d, lambda i: st.pose_people(m, d, i, HEAD))
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out.with_suffix(".perception.ndjson"), "w") as f:
        for row in st.perception[:: max(1, int(fps // 15))]:
            f.write(json.dumps(row) + "\n")

    # 2. puppet animation with the speaker_id gaze bias
    tl = st.timeline

    def follow_fn(t):
        w = follow.gaze_weight(tl.events, "speaker", t, fps)
        return stage_mod.speaker_bias(st, HEAD, w)

    anim = animate(tl, fps=fps, lip_lead=sc.hand.lip_lead_s, follow=follow_fn, tail=0.6)
    traj = solve(sc, anim)
    report = limits.check(sc, traj)
    report.update(script=str(script), hand_profile=hand, video=str(out))
    out.with_suffix(".check.json").write_text(json.dumps(report, indent=1))
    np.savez_compressed(out.with_suffix(".traj.npz"), t=traj.t, arm=traj.arm, hand=traj.hand,
                        hand_joints=np.array(sc.hand.joint_names), mouth=traj.mouth_cmd)
    print(limits.summary(report))

    cams = default_cameras(st)
    renderer = mujoco.Renderer(m, height, width)
    hand_label = {"shadow": "xArm6 + Shadow hand (stand-in)", "dh_pgc140": "xArm6 + DH PGC-140-50"}[hand]
    overlay = Hud(tl, anim, width, height, hand_label, report["ok"]) if hud else None
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
           "-r", str(fps), "-i", "-", "-i", str(st.audio_path),
           "-af", f"adelay={int(round(st.audio_delay * 1000))}:all=1", "-c:a", "aac", "-b:a", "160k",
           "-c:v", "libx264", "-preset", "medium", "-crf", "19", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
           str(out)]
    ff = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    head_site = m.site("head").id
    cam = mujoco.MjvCamera()
    n = min(len(traj.t), len(st.t))
    try:
        sim.reset(traj.arm[0], traj.hand[0])
        for i in range(n):
            if i:
                sim.advance(traj.arm[i - 1], traj.arm[i], traj.hand[i - 1], traj.hand[i], 1 / fps)
            st.pose_people(m, d, i, d.site_xpos[head_site].copy())
            mujoco.mj_forward(m, d)
            eye, tgt, fov = camera_for(cams, st.camera(i), [p.pos_t[i] for p in st.people if not p.seated])
            v = np.asarray(tgt) - np.asarray(eye)
            cam.lookat[:] = tgt
            cam.distance = np.linalg.norm(v)
            cam.azimuth = np.degrees(np.arctan2(v[1], v[0]))
            cam.elevation = np.degrees(np.arcsin(v[2] / np.linalg.norm(v)))
            m.vis.global_.fovy = fov
            renderer.update_scene(d, cam)
            img = renderer.render()
            if overlay:
                img = overlay.draw(img, traj.t[i])
            ff.stdin.write(np.ascontiguousarray(img).tobytes())
    finally:
        ff.stdin.close()
        ff.wait()
        renderer.close()
    print(f"wrote {out} ({n} frames, {time.time() - t0:.1f}s), shots: {[(round(a, 1), b) for a, b in st.shots]}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("script", type=Path)
    ap.add_argument("-o", "--out", type=Path, required=True)
    ap.add_argument("--hand", default="shadow", choices=list(scene_mod.PROFILES))
    ap.add_argument("--size", default="1280x720")
    ap.add_argument("--fps", type=int, default=60)
    ap.add_argument("--no-hud", action="store_true")
    a = ap.parse_args()
    w, h = map(int, a.size.split("x"))
    render_stage(a.script, a.out, a.hand, w, h, a.fps, not a.no_hud)


if __name__ == "__main__":
    main()
