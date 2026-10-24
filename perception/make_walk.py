"""Synthetic CONTRACT §4 traces for MOTION.
walk.ndjson: person walks -40° → +40° at ~1.5 m over 15 s, 30 Hz, with detector-like noise and a 1.5 s
dropout in the middle (held for 1.0 s as in track.py, then present:false for 0.5 s).
walk_tracked.ndjson: the same walk (clipped to the camera FOV) rendered as composited face images on the
real camera background and run through track.Tracker, so it carries the real detector's noise."""
import json, math, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import track

CAM = np.array([0.05, 0.0, 0.10]); FACE_Z = 0.25  # face height above base (seated table, standing ~0.5)
rng = np.random.default_rng(7)

def pose(t, T=15.0, y0=-40, y1=40, d=1.5):
    yaw = y0 + (y1 - y0) * (0.5 - 0.5 * math.cos(math.pi * min(t / T, 1)))  # ease in/out
    p = CAM + np.array([d * math.cos(math.radians(yaw)), d * math.sin(math.radians(yaw)), FACE_Z - CAM[2]])
    return p

def msg(t, p, conf, nf=1):
    yaw, pitch = track.bearing(p)
    return {"t": round(t, 3), "present": True, "yaw_deg": round(yaw, 1), "pitch_deg": round(pitch, 1), "dist_m": round(float(np.linalg.norm(p)), 2),
            "conf": round(conf, 2), "target_base": [round(float(v), 3) for v in p], "n_faces": nf}

out = open(os.path.join(HERE, "samples", "walk.ndjson"), "w")
drop = (7.0, 8.5); last = None
for i in range(int(15 * 30) + 1):
    t = i / 30
    if drop[0] <= t < drop[1]:
        m = dict(last, t=round(t, 3), conf=round(last["conf"] * 0.5, 2), n_faces=0) if t - drop[0] <= 1.0 else {"t": round(t, 3), "present": False, "n_faces": 0}
    else:
        p = pose(t) * (1 + rng.normal(0, 0.03))             # face-size depth noise ~3 %
        p[1] += rng.normal(0, 0.005); p[2] += rng.normal(0, 0.005)
        m = last = msg(t, p, float(np.clip(rng.normal(0.9, 0.03), 0, 1)))
    out.write(json.dumps(m, separators=(",", ":")) + "\n")
out.close()

# real-detector version on the camera background
import cv2
sys.path.insert(0, os.path.join(HERE, "test")); import composite_test as ct  # reuses paste/project (runs its tests on import)
args = track.parse_args([]); tr = track.Tracker(args)
out = open(os.path.join(HERE, "samples", "walk_tracked.ndjson"), "w")
vw = cv2.VideoWriter(os.path.join(HERE, "test", "walk_tracked.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 30, (ct.W, ct.H))
for i in range(int(15 * 30) + 1):
    t = i / 30; img = ct.bg.copy(); p = pose(t, y0=-33, y1=33)
    if not (drop[0] <= t < drop[1]):
        u, v, w = ct.project(p); img = ct.paste(img, "lena.jpg", u, v, w)
    m = tr.step(img, t=t); out.write(json.dumps(m, separators=(",", ":")) + "\n")
    if m["present"]:
        u = ct.cam.cx - math.tan(math.radians(m["yaw_deg"])) * ct.cam.fx
        cv2.line(img, (int(u), 0), (int(u), ct.H), (0, 255, 0), 1)
    cv2.putText(img, f"t={t:4.1f} " + (f"yaw={m['yaw_deg']:+.1f} d={m['dist_m']:.2f}" if m["present"] else "present:false"), (8, 22), 0, 0.6, (0, 255, 0), 2)
    vw.write(img)
vw.release(); out.close()
