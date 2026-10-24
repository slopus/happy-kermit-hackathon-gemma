"""Closed-loop sim of the PTZ aim policy in MOTION's LiveWorld (no hardware).

    /var/lib/happy-agent/IsaacLab/.venv/bin/python sim_demo.py [--out videos/ptz_aim_demo.mp4]

Scripted §6 beats (Kirill, Steve, Kermit's replies, the hand-off to the judges). PERCEPTION is replaced by
an FOV-limited stand-in that only reports people the PTZ can actually see (inside the frame, not occluded),
turns their pixels into bearings through the camera pose (ptz.pixel_to_base, the contract), and keeps
speaker_id on the last speaker it saw when the talker is off-frame, so the policy has to find them.
Output: the PTZ feed (what PERCEPTION gets) beside a stage view and a top-down map, plus aim.ndjson and
ptz_pose.ndjson next to the video.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "motion")]

import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from aim import PTZDirector  # noqa: E402
from live import LiveWorld  # noqa: E402
from ptz import PTZ, SimBackend, fov, pixel_to_base, rotation  # noqa: E402

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
POS = {"kirill": (0.75, 0.75), "steve": (0.75, -0.75), "judge_a": (3.3, -0.9), "judge_b": (3.5, 0.0), "judge_c": (3.3, 0.9)}
STYLE = {"kirill": "presenter", "steve": "judge_b", "judge_a": "judge_a", "judge_b": "judge_c", "judge_c": "judge_a"}
IDS = {n: i + 1 for i, n in enumerate(POS)}
# (t0, t1, who): who talks when ("kermit" = Kermit's loudspeaker; the camera must not chase it)
SCRIPT = [(1.0, 7.5, "kirill"), (7.8, 10.5, "kermit"), (10.8, 17.0, "steve"), (17.3, 20.5, "kermit"),
          (20.8, 24.5, "kirill"), (24.8, 27.0, "kermit"), (28.0, 32.5, "judge_c"), (32.8, 35.0, "kermit"),
          (35.5, 39.5, "judge_a"), (39.8, 42.0, "kermit")]
EVENTS = [(27.2, {"event": "camera_mode", "mode": "qa"})]   # BRAIN: Q&A has started (hand-off agreed)
DURATION = 43.0


def talker(t):
    return next((w for a, b, w in SCRIPT if a <= t < b), None)


class FakePerception:
    """Sees only what is in the PTZ frame; bearings go pixel -> pose -> base frame, as the contract says."""

    def __init__(self, w: LiveWorld, size):
        self.w, self.W, self.H = w, *size
        self.m, self.d = w.k.m, w.k.d
        self.speaker = None
        self.gbody = self.m.geom_bodyid
        self.hist = []
        self.rng = np.random.default_rng(7)   # Light-ASD logit spread measured by PERCEPTION on real video

    def visible(self, name, pose):
        m, d = self.m, self.d
        p = d.geom_xpos[m.geom(f"{name}_head").id]
        cam = d.cam_xpos[m.camera("ptz_cam").id]
        R = rotation(pose.pan, pose.tilt, pose.mount_yaw)
        x, y, z = R.T @ (p - cam)
        if x <= 0.1:
            return None
        hfov, vfov = fov(pose.zoom, self.W / self.H)
        fx = (self.W / 2) / math.tan(math.radians(hfov) / 2)
        fy = (self.H / 2) / math.tan(math.radians(vfov) / 2)
        u, v = self.W / 2 - fx * y / x, self.H / 2 - fy * z / x
        if not (0.03 * self.W < u < 0.97 * self.W and 0.02 * self.H < v < 0.98 * self.H):
            return None
        ray = p - cam
        dist = float(np.linalg.norm(ray))
        gid = np.zeros(1, np.int32)
        hit = mujoco.mj_ray(m, d, cam, ray / dist, None, 1, m.body("ptz_head").id, gid)
        if 0 <= hit < dist - 0.12 and not m.body(self.gbody[gid[0]]).name.startswith(name):
            return None
        return u, v, dist

    def __call__(self, t, pose):
        who = talker(t)
        people = []
        for n in POS:
            seen = self.visible(n, pose)
            if seen is None:
                continue
            u, v, depth = seen
            yaw_c, pitch_c = pixel_to_base(u, v, pose, self.W, self.H)   # direction from the camera
            dvec = np.array([math.cos(math.radians(pitch_c)) * math.cos(math.radians(yaw_c)),
                             math.cos(math.radians(pitch_c)) * math.sin(math.radians(yaw_c)),
                             math.sin(math.radians(pitch_c))])
            tb = self.d.cam_xpos[self.m.camera("ptz_cam").id] + depth * dvec   # face point, base frame
            people.append({"id": IDS[n], "yaw_deg": round(math.degrees(math.atan2(tb[1], tb[0])), 1),
                           "pitch_deg": round(math.degrees(math.atan2(tb[2], math.hypot(tb[0], tb[1]))), 1),
                           "dist_m": round(float(np.linalg.norm(tb)), 2), "target_base": [round(float(x), 3) for x in tb],
                           "facing": "toward", "speaking_p": 0.9 if n == who else 0.05,
                           "face": True, "asd": round(float(self.rng.normal(-0.32, 0.74) if n == who
                                                            else self.rng.normal(-1.52, 0.76)), 2),
                           "role": "presenter" if np.hypot(*POS[n]) < 2 else "judge", "conf": 0.9})
        ids = {p["id"] for p in people}
        if who in IDS and IDS[who] in ids:
            self.speaker = IDS[who]       # speaker_id hysteresis: otherwise keep the last one seen
        speech = who not in (None, "kermit")
        # PERCEPTION's definition: over 1.5 s, voice >= 60 % of the time and the best visible lip logit's mean < -1.0
        self.hist = [h for h in self.hist if t - h[0] <= 1.5] + [(t, speech, max([p["asd"] for p in people], default=None))]
        best = [b for _, _, b in self.hist if b is not None]
        unseen = np.mean([v for _, v, _ in self.hist]) >= 0.6 and (not best or np.mean(best) < -1.0)
        return {"t": t, "speaker_id": self.speaker, "speech_active": speech, "people": people,
                "unseen_speech": bool(unseen)}


def topdown(size, people, pose, who, speaker_id):
    s = size
    img = Image.new("RGB", (s, s), (18, 18, 22))
    dr = ImageDraw.Draw(img)
    sc, ox, oy = s / 4.4, s / 2, s - 14          # base at the bottom middle, +x up, +y left

    def px(x, y):
        return ox - y * sc, oy - x * sc
    mx, my = pose.mount[0], pose.mount[1]
    hfov = fov(pose.zoom)[0]
    for a in (pose.pan - hfov / 2, pose.pan + hfov / 2):
        r = math.radians(a)
        dr.line([px(mx, my), px(mx + 4 * math.cos(r), my + 4 * math.sin(r))], fill=(90, 160, 255), width=1)
    r = math.radians(pose.pan)
    dr.line([px(mx, my), px(mx + 1.2 * math.cos(r), my + 1.2 * math.sin(r))], fill=(140, 200, 255), width=2)
    dr.rectangle([*px(0.08, 0.08), *px(-0.08, -0.08)], outline=(200, 200, 200))
    x, y = px(0.45, 0.0)
    dr.ellipse([x - 4, y - 4, x + 4, y + 4], fill=(60, 200, 60))
    x, y = px(mx, my)
    dr.ellipse([x - 3, y - 3, x + 3, y + 3], fill=(140, 200, 255))
    for n, (pxw, pyw) in POS.items():
        x, y = px(pxw, pyw)
        col = (255, 210, 60) if n == who else (170, 170, 170)
        dr.ellipse([x - 6, y - 6, x + 6, y + 6], fill=col, outline=(255, 80, 80) if IDS[n] == speaker_id else None, width=2)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "videos/ptz_aim_demo.mp4"))
    ap.add_argument("--fps", type=float, default=30)
    ap.add_argument("--no-speech-signal", action="store_true", help="speech_active always false (dead VAD)")
    ap.add_argument("--no-qa-event", action="store_true", help="BRAIN never says Q&A started: auto only")
    ap.add_argument("--no-unseen", action="store_true", help="PERCEPTION without the unseen_speech field")
    ap.add_argument("--fusion-lock", action="store_true",
                    help="PERCEPTION's 'alone in view + room speech' prior: the only visible person gets "
                         "speaking_p 0.8 and keeps speaker_id; only the raw lip score (asd) tells the truth")
    a = ap.parse_args()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    W, H = 640, 360
    w = LiveWorld(people=[{"name": n, "style": STYLE[n], "seated": n.startswith("judge"), "pos_base": list(POS[n])}
                          for n in POS], size=(W, H), view_size=(W, H), ptz=True)
    for n, xy in POS.items():
        w.set_person(n, xy, math.degrees(math.atan2(-xy[1], -xy[0])))
    cam = PTZ(SimBackend(w), mount=w.ptz["mount"], pose_log=out.with_suffix(".ptz_pose.ndjson"))
    t = 0.0
    director = PTZDirector(cam, log=out.with_suffix(".aim.ndjson"), clock=lambda: t)
    perc = FakePerception(w, (W, H))
    # stage view: high, from the audience side, over the judges' heads (sees the post and both presenters)
    m = w.k.m
    eye, target = np.array([2.3, -0.35, 1.55]), np.array([0.35, 0.05, 0.25])
    zax = (eye - target) / np.linalg.norm(eye - target)
    xax = np.cross([0, 0, 1], zax)
    xax /= np.linalg.norm(xax)
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.column_stack([xax, np.cross(zax, xax), zax]).ravel())
    cid = m.camera("wide").id
    m.cam_pos[cid], m.cam_quat[cid], m.cam_fovy[cid] = eye, q, 38
    f_big, f_small = ImageFont.truetype(FONT, 20), ImageFont.truetype(FONT, 15)
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{2 * W}x{H + 90}",
                           "-r", str(a.fps), "-i", "-", "-c:v", "libx264", "-crf", "21", "-pix_fmt", "yuv420p", str(out)],
                          stdin=subprocess.PIPE)
    events = [] if a.no_qa_event else list(EVENTS)
    rng = np.random.default_rng(3)
    t, n_frames = 0.0, int(DURATION * a.fps)
    for i in range(n_frames):
        t = i / a.fps
        who = talker(t)
        for n, xy in POS.items():
            w.set_person(n, xy, math.degrees(math.atan2(-xy[1], -xy[0])), mouth=float(rng.uniform(0.2, 0.8)) if n == who else 0.0,
                         talking=n == who, gesture="raise_hand" if n.startswith("judge") and talker(t + 1.0) == n and who != n else None)
        while events and events[0][0] <= t:
            director.on_kermit_event(events.pop(0)[1])
        director._talk_until = next((b for a0, b, wh in SCRIPT if wh == "kermit" and a0 <= t < b), -1.0)
        w.step(t)
        img = w.render("arm_base")
        pose = cam.get() if cam.history else w.ptz_pose()
        latest = perc(t, pose)
        if a.no_speech_signal:
            latest["speech_active"] = latest["unseen_speech"] = False
        if a.no_unseen:
            latest.pop("unseen_speech")
        if a.fusion_lock and len(latest["people"]) == 1 and latest["speech_active"]:
            latest["people"][0]["speaking_p"] = 0.8
            latest["speaker_id"] = perc.speaker = latest["people"][0]["id"]
        director.on_perception(latest)
        w.kermit_event(dict(latest, event="perception"))
        pose = director.tick(t)
        stage = w.render("wide")
        frame = Image.new("RGB", (2 * W, H + 90), (12, 12, 14))
        frame.paste(Image.fromarray(img), (0, 0))
        st = Image.fromarray(stage)
        st.paste(topdown(150, POS, pose, who, latest["speaker_id"]), (W - 150, H - 150))
        frame.paste(st, (W, 0))
        dr = ImageDraw.Draw(frame)
        dr.text((10, 8), "PTZ feed  (what PERCEPTION sees)", font=f_small, fill=(230, 230, 230))
        dr.text((W + 10, 8), "stage", font=f_small, fill=(230, 230, 230))
        pol = director.policy
        names = {v: k for k, v in IDS.items()}
        dr.text((12, H + 8), f"pan {pose.pan:+6.1f}   tilt {pose.tilt:+5.1f}   zoom {pose.zoom:.2f}x"
                f"{'   moving' if pose.moving else ''}", font=f_big, fill=(140, 200, 255))
        dr.text((12, H + 36), f"policy: {pol.phase} | {pol.why}", font=f_big, fill=(240, 240, 240))
        dr.text((12, H + 64), f"talking: {who or '-'}    perception speaker_id: "
                f"{names.get(latest['speaker_id'], '-')}    in view: {', '.join(names[p['id']] for p in latest['people']) or '-'}",
                font=f_small, fill=(255, 210, 60))
        dr.text((W + 70, 8), "built-in AI tracking: off    t = %5.1f s" % t, font=f_small, fill=(230, 230, 230))
        ff.stdin.write(np.asarray(frame).tobytes())
    ff.stdin.close()
    ff.wait()
    w.close()
    print("wrote", out, "| ptz errors:", cam.errors or "none", "| commands ignored while parked:", w.ptz_ignored)


if __name__ == "__main__":
    main()
