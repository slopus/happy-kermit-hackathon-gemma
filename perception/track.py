"""Person tracking for Happy Kermit: camera → main face → bearing in the arm base frame (CONTRACT §4).

  python track.py --source 0                                   # OpenCV camera index (AVFoundation on the Mac)
  python track.py --source "http://127.0.0.1:8090/snapshot.jpg?width=640"   # poll an HTTP snapshot
  python track.py --source clip.mp4 | --source 'frames/*.jpg'  # file / image sequence (tests)
Output: one NDJSON line per frame on stdout (or --out file), optionally the latest value written
atomically to --latest PATH and/or sent as UDP datagrams to --udp HOST:PORT.
  {"t":…, "present":true, "yaw_deg":…, "pitch_deg":…, "dist_m":…, "conf":…, "target_base":[x,y,z]}

Detector: OpenCV YuNet (face_detection_yunet_2023mar.onnx, 230 kB), CPU, no cloud.
Geometry: pinhole camera from --hfov (or --fx/--fy/--cx/--cy), depth from face width (0.16 m),
camera pose in the base frame from --cam-pos / --cam-rpy. Base frame: +x forward, +y left, +z up.
"""
import argparse, glob, json, math, os, socket, sys, time, urllib.request
import numpy as np, cv2

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(HERE, "models", "face_detection_yunet_2023mar.onnx")


# ---------- geometry ----------
def rpy_matrix(roll, pitch, yaw):
    """Base-frame rotation of the camera body (deg): yaw about z (left +), pitch about y (down +), roll about x."""
    r, p, y = np.radians([roll, pitch, yaw])
    Rx = np.array([[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]])
    Ry = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]])
    Rz = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


class Camera:
    def __init__(self, w, h, hfov=78.0, fx=None, fy=None, cx=None, cy=None, pos=(0.05, 0, 0.10), rpy=(0, 0, 0), face_w=0.16):
        self.fx = fx or (w / 2) / math.tan(math.radians(hfov) / 2)
        self.fy = fy or self.fx
        self.cx, self.cy = (cx if cx is not None else w / 2), (cy if cy is not None else h / 2)
        self.pos, self.R, self.face_w = np.array(pos, float), rpy_matrix(*rpy), face_w

    def face_to_base(self, u, v, w_px):
        """Face centre (px) and width (px) → point in base frame (m). Optical axis = body +x."""
        z = self.fx * self.face_w / max(w_px, 1.0)                   # depth along the optical axis
        p_body = np.array([z, -(u - self.cx) / self.fx * z, -(v - self.cy) / self.fy * z])  # fwd, left, up
        return self.pos + self.R @ p_body


def bearing(p):
    return math.degrees(math.atan2(p[1], p[0])), math.degrees(math.atan2(p[2], math.hypot(p[0], p[1])))


# ---------- main-person selection ----------
class MainPerson:
    """Largest face wins, but switching away from the current one needs a face >1.3× larger for >0.5 s.
    'Current' is followed frame to frame by nearest centre (faces move; indices don't persist)."""

    def __init__(self, ratio=1.3, hold_s=0.5, lost_s=1.0):
        self.ratio, self.hold_s, self.lost_s = ratio, hold_s, lost_s
        self.cur = None        # (cx, cy, w)
        self.t_seen = -1e9
        self.challenger_since = None

    def update(self, faces, t):
        """faces: list of (cx, cy, w, h, score). Returns the chosen face or None."""
        if not faces:
            self.challenger_since = None
            if t - self.t_seen > self.lost_s: self.cur = None
            return None
        if self.cur is None:
            pick = max(faces, key=lambda f: f[2])
        else:
            cx, cy, cw = self.cur
            near = min(faces, key=lambda f: math.hypot(f[0] - cx, f[1] - cy))
            if math.hypot(near[0] - cx, near[1] - cy) > 1.5 * max(cw, near[2]):
                near = None    # current face vanished this frame
            big = max(faces, key=lambda f: f[2])
            if near is None:
                pick = big if t - self.t_seen > self.lost_s else None
                if pick is None: return None
            elif big is not near and big[2] > self.ratio * near[2]:
                self.challenger_since = self.challenger_since or t
                pick = big if t - self.challenger_since >= self.hold_s else near
            else:
                self.challenger_since = None; pick = near
            if pick is big and pick is not near: self.challenger_since = None
        self.cur, self.t_seen = (pick[0], pick[1], pick[2]), t
        return pick


# ---------- sources ----------
def frames(src, fps_cap):
    """Yields BGR frames from a camera index, video file, image glob or HTTP snapshot URL."""
    if src.startswith("http"):
        while True:
            t = time.perf_counter()
            try:
                data = urllib.request.urlopen(src, timeout=2).read()
                yield cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            except OSError as e:
                print(f"snapshot error: {e}", file=sys.stderr); yield None
            time.sleep(max(0.0, 1 / fps_cap - (time.perf_counter() - t)))
    elif any(c in src for c in "*?") or os.path.isdir(src):
        for f in sorted(glob.glob(os.path.join(src, "*") if os.path.isdir(src) else src)):
            yield cv2.imread(f)
    else:
        idx = int(src) if src.isdigit() else src
        api = cv2.CAP_AVFOUNDATION if sys.platform == "darwin" and isinstance(idx, int) else cv2.CAP_ANY
        cap = cv2.VideoCapture(idx, api)
        if isinstance(idx, int):
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720); cap.set(cv2.CAP_PROP_FPS, 30)
        if not cap.isOpened(): sys.exit(f"cannot open camera {src} (macOS: grant Camera permission to the terminal)")
        while True:
            ok, f = cap.read()
            if not ok: return
            yield f


class Tracker:
    def __init__(self, a):
        self.a = a
        cv2.setNumThreads(a.threads)  # leave cores for STT/TTS in the same process
        self.det = None; self.cam = None; self.main = MainPerson(a.switch_ratio, a.switch_hold, a.lost_after)
        self.t0 = time.perf_counter(); self.last = None

    def _setup(self, w, h):
        a = self.a
        self.scale = min(1.0, a.det_width / w)                       # detect on a downscaled copy
        self.det = cv2.FaceDetectorYN.create(MODEL, "", (int(w * self.scale), int(h * self.scale)), a.score, 0.3, 50)
        self.cam = Camera(w, h, a.hfov, a.fx, a.fy, a.cx, a.cy, a.cam_pos, a.cam_rpy, a.face_width)

    def step(self, img, t=None):
        t = (time.perf_counter() - self.t0) if t is None else t
        faces = []
        if img is not None:
            h, w = img.shape[:2]
            if self.det is None or (w, h) != self.size: self._setup(w, h); self.size = (w, h)
            small = cv2.resize(img, None, fx=self.scale, fy=self.scale) if self.scale < 1 else img
            _, det = self.det.detect(small)
            for d in (det if det is not None else []):
                x, y, bw, bh = d[:4] / self.scale
                faces.append((x + bw / 2, y + bh / 2, bw, bh, float(d[14])))
        pick = self.main.update(faces, t)
        if pick is not None:
            p = self.cam.face_to_base(pick[0], pick[1], pick[2])
            yaw, pitch = bearing(p)
            self.last = {"t": round(t, 3), "present": True, "yaw_deg": round(yaw, 1), "pitch_deg": round(pitch, 1),
                         "dist_m": round(float(np.linalg.norm(p)), 2), "conf": round(pick[4], 2),
                         "target_base": [round(float(v), 3) for v in p], "n_faces": len(faces)}
            return self.last
        if self.last is not None and t - self.main.t_seen <= self.a.lost_after:
            return dict(self.last, t=round(t, 3), conf=round(self.last["conf"] * 0.5, 2), n_faces=len(faces))  # brief miss: hold
        return {"t": round(t, 3), "present": False, "n_faces": len(faces)}


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="0")
    ap.add_argument("--hfov", type=float, default=78.0, help="horizontal field of view, deg (C920/C922: ~70.4)")
    for k in ("fx", "fy", "cx", "cy"): ap.add_argument("--" + k, type=float)
    ap.add_argument("--cam-pos", type=float, nargs=3, default=[0.05, 0.0, 0.10], help="camera in base frame, m")
    ap.add_argument("--cam-rpy", type=float, nargs=3, default=[0.0, 0.0, 0.0], help="camera roll pitch yaw in base frame, deg")
    ap.add_argument("--face-width", type=float, default=0.16)
    ap.add_argument("--score", type=float, default=0.7, help="YuNet score threshold")
    ap.add_argument("--det-width", type=int, default=960, help="detector input width (frames are downscaled)")
    ap.add_argument("--switch-ratio", type=float, default=1.3); ap.add_argument("--switch-hold", type=float, default=0.5)
    ap.add_argument("--lost-after", type=float, default=1.0)
    ap.add_argument("--fps", type=float, default=30.0, help="cap for snapshot polling")
    ap.add_argument("--out", help="NDJSON file instead of stdout")
    ap.add_argument("--latest", help="also write the latest value atomically to this JSON file")
    ap.add_argument("--udp", help="also send each line as a UDP datagram to HOST:PORT")
    ap.add_argument("--threads", type=int, default=2, help="OpenCV threads")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--stats", action="store_true", help="print fps / detector ms to stderr every 2 s")
    return ap.parse_args(argv)


def main():
    a = parse_args()
    tr = Tracker(a)
    out = open(a.out, "w") if a.out else sys.stdout
    udp = None
    if a.udp:
        host, port = a.udp.rsplit(":", 1); udp = (socket.socket(socket.AF_INET, socket.SOCK_DGRAM), (host, int(port)))
    n, n_total, t_stat, det_ms = 0, 0, time.perf_counter(), []
    for img in frames(a.source, a.fps):
        t = time.perf_counter(); msg = tr.step(img); det_ms.append((time.perf_counter() - t) * 1000)
        line = json.dumps(msg, separators=(",", ":"))
        out.write(line + "\n"); out.flush()
        if a.latest:
            tmp = a.latest + ".tmp"; open(tmp, "w").write(line); os.replace(tmp, a.latest)
        if udp: udp[0].sendto(line.encode(), udp[1])
        n += 1; n_total += 1
        if a.stats and time.perf_counter() - t_stat > 2:
            print(f"{n / (time.perf_counter() - t_stat):.1f} fps, step {np.median(det_ms):.1f} ms median", file=sys.stderr)
            n, t_stat, det_ms = 0, time.perf_counter(), []
        if a.max_frames and n_total >= a.max_frames: break


if __name__ == "__main__":
    main()
