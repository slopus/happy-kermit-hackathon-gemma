"""Render a VOICE performance timeline as a hand-puppet video with audio.

    /var/lib/happy-agent/IsaacLab/.venv/bin/python render.py TIMELINE.json -o out.mp4 [--hand shadow|dh_pgc140]

Writes ``out.mp4`` (H.264 + AAC), ``out.check.json`` (hardware limit report) and ``out.traj.npz``
(joint trajectories at the render rate, ready for a later hardware replay).
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
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from kermit_motion import limits, scene as scene_mod  # noqa: E402
from kermit_motion.animation import Timeline, animate  # noqa: E402
from kermit_motion.sim import PuppetSim  # noqa: E402
from kermit_motion import follow as follow_mod  # noqa: E402
from kermit_motion.trajectory import solve  # noqa: E402

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT_REG = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"


class Hud:
    def __init__(self, tl: Timeline, anim, width, height, title, check_ok):
        self.tl, self.anim, self.w, self.h = tl, anim, width, height
        s = height / 720
        self.f_sub = ImageFont.truetype(FONT, int(30 * s))
        self.f_tag = ImageFont.truetype(FONT, int(24 * s))
        self.f_small = ImageFont.truetype(FONT_REG, int(16 * s))
        self.title, self.check_ok, self.s = title, check_ok, s

    def draw(self, img: np.ndarray, t: float) -> np.ndarray:
        im = Image.fromarray(img)
        d = ImageDraw.Draw(im, "RGBA")
        s = self.s
        # subtitle: the current phrase with the spoken word highlighted
        words = self.tl.words
        cur = [i for i, w in enumerate(words) if w["t0"] - 0.05 <= t <= w["t1"] + 0.35]
        cap = [c for c in self.tl.captions if c[0] - 0.05 <= t <= c[1] + 0.3]
        if cap and not cur:
            _, _, who, text = cap[-1]
            label = f"{who}: "
            # wrap to at most two lines that fit 86 % of the frame
            maxw = 0.86 * self.w
            lines, line = [], label
            for word in text.split():
                if d.textlength(line + word, font=self.f_sub) > maxw and line.strip():
                    lines.append(line.rstrip())
                    line = ""
                line += word + " "
            lines.append(line.rstrip())
            lines = lines[-2:] if len(lines) > 2 else lines
            lh = 40 * s
            y0 = self.h - 70 * s - lh * (len(lines) - 1)
            tw = max(d.textlength(ln, font=self.f_sub) for ln in lines)
            x0 = (self.w - tw) / 2
            d.rounded_rectangle([x0 - 16 * s, y0 - 10 * s, x0 + tw + 16 * s, y0 + lh * (len(lines) - 1) + 44 * s],
                                12 * s, fill=(0, 0, 0, 150))
            for k, ln in enumerate(lines):
                y = y0 + k * lh
                x = (self.w - d.textlength(ln, font=self.f_sub)) / 2
                if ln.startswith(label):
                    d.text((x, y), label, font=self.f_sub, fill=(120, 200, 255, 255))
                    x += d.textlength(label, font=self.f_sub)
                    ln = ln[len(label):]
                d.text((x, y), ln, font=self.f_sub, fill=(225, 235, 245, 255))
        if cur:
            i = cur[-1]
            lo = max(0, i - 4)
            hi = min(len(words), lo + 9)
            # keep the subtitle inside one sentence
            while lo < i and words[lo - 1 if lo else 0]["w"][-1:] not in ".!?\u2014" and lo > 0:
                lo -= 1
            for j in range(lo, i):
                if words[j]["w"][-1:] in ".!?\u2014":
                    lo = j + 1
            hi = min(len(words), lo + 9)
            for j in range(max(i, lo), hi):
                if words[j]["w"][-1:] in ".!?\u2014":
                    hi = j + 1
                    break
            pieces = [(w["w"], j == i and t <= words[j]["t1"] + 0.05) for j, w in enumerate(words[lo:hi], lo)]
            text = " ".join(p for p, _ in pieces)
            tw = d.textlength(text, font=self.f_sub)
            x = (self.w - tw) / 2
            y = self.h - 70 * s
            d.rounded_rectangle([x - 16 * s, y - 10 * s, x + tw + 16 * s, y + 44 * s], 12 * s, fill=(0, 0, 0, 150))
            for p, hl in pieces:
                d.text((x, y), p, font=self.f_sub, fill=(255, 225, 90, 255) if hl else (240, 240, 240, 255))
                x += d.textlength(p + " ", font=self.f_sub)
        # active gesture tag
        active = [lab for (a, b, lab) in self.anim.labels if a <= t <= b]
        if active:
            text = " + ".join(dict.fromkeys(active))
            d.rounded_rectangle([20 * s, 20 * s, 40 * s + d.textlength(text, font=self.f_tag), 62 * s], 10 * s,
                                fill=(40, 140, 40, 200))
            d.text((30 * s, 26 * s), text, font=self.f_tag, fill=(255, 255, 255, 255))
        emotion = self.tl.emotion
        for e in self.tl.events:
            if e["type"] == "emotion" and e["t"] <= t:
                emotion = e["value"]
        badge = f"{self.title} · emotion: {emotion} · hw limits {'OK' if self.check_ok else 'EXCEEDED'}"
        bw = d.textlength(badge, font=self.f_small)
        d.text((self.w - bw - 20 * s, 24 * s), badge, font=self.f_small, fill=(230, 230, 230, 220))
        return np.asarray(im)


def render(timeline: Path, out: Path, hand: str = "shadow", width: int = 1280, height: int = 720, fps: int = 60,
           camera: str = "front34", shell: bool = True, hud: bool = True, hw_hand: bool = True,
           xray: bool = False, preroll: float = 0.5, track_path: Path | None = None) -> dict:
    t_start = time.time()
    tl = Timeline.load(timeline).shifted(preroll)
    sc = scene_mod.build(hand, shell=shell, width=width, height=height, person=track_path is not None)
    track = None
    follow_fn = None
    if track_path is not None:
        def follow_fn(t):
            nonlocal track
            track = follow_mod.load_track(track_path, t)
            w = follow_mod.gaze_weight(tl.events, tl.gaze, t, fps)
            return follow_mod.bias(track, scene_mod.HEAD_REST_POS, w, fps)
        tl.duration = max(tl.duration, _track_len(track_path) - 0.6)
    anim = animate(tl, fps=fps, lip_lead=sc.hand.lip_lead_s, follow=follow_fn)
    traj = solve(sc, anim, hw_limited_hand=hw_hand)
    report = limits.check(sc, traj)
    report.update(timeline=str(timeline), hand_profile=hand, video=str(out))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".check.json").write_text(json.dumps(report, indent=1))
    np.savez_compressed(out.with_suffix(".traj.npz"), t=traj.t, arm=traj.arm, hand=traj.hand,
                        hand_joints=np.array(sc.hand.joint_names), mouth=traj.mouth_cmd)
    print(limits.summary(report))

    m = sc.model
    sim = PuppetSim(sc)
    d = sim.d
    renderer = mujoco.Renderer(m, height, width)
    opt = mujoco.MjvOption()
    if xray:  # see the robot hand inside the felt head
        opt.geomgroup[4] = 1
        for name in ("felt", "mouth", "eye"):
            m.mat_rgba[m.material(name).id, 3] = 0.3
    hand_label = {"shadow": "xArm6 + Shadow hand (stand-in)", "dh_pgc140": "xArm6 + DH PGC-140-50"}[hand]
    overlay = Hud(tl, anim, width, height, hand_label, report["ok"]) if hud else None

    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
           "-r", str(fps), "-i", "-"]
    if tl.audio and tl.audio.exists():
        cmd += ["-i", str(tl.audio), "-af", f"adelay={int(round(preroll * 1000))}:all=1", "-c:a", "aac", "-b:a", "160k"]
    cmd += ["-c:v", "libx264", "-preset", "medium", "-crf", "19", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            str(out)]
    ff = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    try:
        sim.reset(traj.arm[0], traj.hand[0])
        for i, t in enumerate(traj.t):
            if i:
                sim.advance(traj.arm[i - 1], traj.arm[i], traj.hand[i - 1], traj.hand[i], traj.t[i] - traj.t[i - 1])
            if track is not None:
                mid = m.body_mocapid[m.body("person").id]
                face = track.face[i]
                d.mocap_pos[mid] = face
                yaw = np.arctan2(face[1], face[0])  # body x points away from the robot
                d.mocap_quat[mid] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
                mujoco.mj_forward(m, d)
            renderer.update_scene(d, camera, scene_option=opt)
            img = renderer.render()
            if overlay:
                img = overlay.draw(img, t)
            ff.stdin.write(np.ascontiguousarray(img).tobytes())
    finally:
        ff.stdin.close()
        ff.wait()
        renderer.close()
    print(f"wrote {out} ({len(traj.t)} frames, {time.time() - t_start:.1f}s)")
    return report


def _track_len(path: Path) -> float:
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    return rows[-1]["t"] - rows[0]["t"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("timeline", type=Path)
    ap.add_argument("-o", "--out", type=Path, required=True)
    ap.add_argument("--hand", default="shadow", choices=list(scene_mod.PROFILES))
    ap.add_argument("--camera", default="front34", help="front34 | side | front | wide")
    ap.add_argument("--size", default="1280x720")
    ap.add_argument("--fps", type=int, default=60)
    ap.add_argument("--no-shell", action="store_true", help="render the bare robot hand, no Kermit costume")
    ap.add_argument("--no-hud", action="store_true")
    ap.add_argument("--ideal-hand", action="store_true", help="skip hand speed limiting (show the ideal mouth)")
    ap.add_argument("--xray", action="store_true", help="semi-transparent felt, show the robot hand inside")
    ap.add_argument("--preroll", type=float, default=0.5, help="seconds of silence before the audio (breath)")
    ap.add_argument("--follow", type=Path, help="PERCEPTION NDJSON trace: follow this person (CONTRACT section 4)")
    a = ap.parse_args()
    w, h = map(int, a.size.split("x"))
    render(a.timeline, a.out, a.hand, w, h, a.fps, a.camera, not a.no_shell, not a.no_hud, not a.ideal_hand, a.xray, a.preroll, a.follow)


if __name__ == "__main__":
    main()
