"""PERCEPTION v2 (CONTRACT §5): everyone in view, who is speaking, who is facing Kermit.

  python people.py --source 0 --latest /tmp/people.json --events /tmp/events.ndjson   # live camera, no audio
  python people.py --source clip.mp4 --render out.mp4 --log out.ndjson                # offline: audio + VAD from the file
In run_kermit.py (same process, one perf_counter clock):
  pt = people.PeopleThread(people.parse_args([...]), on_event=cb); pt.start()
  pt.feed_audio(t_end, x16k); pt.feed_vad(t, active, rms_db); pt.set_self_speaking(b)
  pt.latest (§5 dict) · pt.gaze (§4 dict aimed at speaker_id) · pt.context() (BRAIN string)

Pipeline per frame:
  YOLO11-pose + ByteTrack → bodies with 17 keypoints and track IDs → stable IDs (re-ID across track breaks by
  position + torso colour) · YuNet faces → assigned to bodies · facing from face + nose/eyes/ears/shoulders, with
  hysteresis → turned_away / turned_back · Light-ASD on face crops + mic audio (5 Hz) · YOLO-World "microphone"
  (3 Hz) → has_mic · fused speaking_p (ASD, mic, VAD, continuity) → speaker_id with hysteresis.
"""
import argparse, collections, json, math, os, subprocess, sys, threading, time
import numpy as np, cv2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("YOLO_CONFIG_DIR", os.path.join(HERE, ".ultralytics"))
os.environ.setdefault("YOLO_VERBOSE", "False")   # ultralytics must not print to stdout (MOTION reads NDJSON there)
import track  # v1: Camera geometry, frame sources, YuNet model path

M = lambda f: os.path.join(HERE, "models", f)
NOSE, LEYE, REYE, LEAR, REAR, LSHO, RSHO, LWRI, RWRI, LHIP, RHIP = 0, 1, 2, 3, 4, 5, 6, 9, 10, 11, 12
sig = lambda x: 1.0 / (1.0 + math.exp(-x))


# ---------- audio ----------
class AudioRing:
    """16 kHz mono samples indexed by absolute time (perf_counter live, video time offline)."""

    def __init__(self, seconds=6.0, sr=16000):
        self.sr, self.n = sr, int(seconds * sr)
        self.buf = np.zeros(self.n, np.float32); self.t_end = None; self.lock = threading.Lock()

    def feed(self, t_end, x):
        x = np.asarray(x, np.float32).ravel()[-self.n:]
        with self.lock:
            self.buf = np.roll(self.buf, -len(x)); self.buf[-len(x):] = x; self.t_end = t_end

    def get(self, t0, t1):
        with self.lock:
            if self.t_end is None: return None
            i1 = self.n - int(round((self.t_end - t1) * self.sr)); i0 = i1 - int(round((t1 - t0) * self.sr))
            if i0 < 0 or i1 > self.n + int(0.05 * self.sr): return None
            return self.buf[i0:min(i1, self.n)].copy()


class FileAudio:
    """Whole-file audio (offline eval): same get() interface."""

    def __init__(self, x, sr=16000): self.x, self.sr = x, sr

    def get(self, t0, t1):
        i0, i1 = int(t0 * self.sr), int(t1 * self.sr)
        if i0 < 0 or i1 > len(self.x): return None
        return self.x[i0:i1]


def load_audio(path):
    raw = subprocess.run(["ffmpeg", "-v", "quiet", "-i", path, "-ac", "1", "-ar", "16000", "-f", "f32le", "-"],
                         capture_output=True).stdout
    return np.frombuffer(raw, np.float32)


def silero_segments(x, sr=16000):
    """Offline VAD for eval: Silero via sherpa-onnx (same model/thresholds as voice/stt.py). → [(t0, t1)]."""
    import sherpa_onnx
    cfg = sherpa_onnx.VadModelConfig()
    cfg.silero_vad.model = os.path.join(HERE, "..", "voice", "models", "silero_vad.onnx")
    cfg.silero_vad.min_silence_duration = 0.25; cfg.silero_vad.min_speech_duration = 0.1; cfg.silero_vad.threshold = 0.5
    cfg.sample_rate = sr
    vad = sherpa_onnx.VoiceActivityDetector(cfg, buffer_size_in_seconds=len(x) / sr + 10)
    segs = []
    for i in range(0, len(x) - 512, 512):
        vad.accept_waveform(x[i:i + 512])
        while not vad.empty():
            s = vad.front; segs.append((s.start / sr, (s.start + len(s.samples)) / sr)); vad.pop()
    vad.flush()
    while not vad.empty():
        s = vad.front; segs.append((s.start / sr, (s.start + len(s.samples)) / sr)); vad.pop()
    return segs


# ---------- per-person state ----------
class Person:
    def __init__(self, pid, t):
        self.id, self.t_first, self.t_seen = pid, t, t
        self.box = None; self.kp = None; self.face = None; self.hist = None
        self.confirmed = False
        self.facing, self.facing_since = "toward", t
        self.raw = collections.deque(maxlen=60)          # (t, raw facing label)
        self.hands = collections.deque(maxlen=60); self.hand_up = False
        self.in_roi = True; self.attended = False
        self.look = collections.deque(maxlen=30)         # (t, -1 head turned to image left / 0 / +1 image right)
        self.crops = collections.deque(maxlen=60)        # (t, 112×112 gray)
        self.asd_logit, self.asd_t = None, -1e9
        self.mic_ema, self.has_mic = 0.0, False
        self.p = 0.0                                      # fused speaking_p (smoothed)
        self.talk_s = 0.0                                 # cumulative time as speaker (→ role)
        self.yaw = self.pitch = 0.0; self.dist = 2.0; self.target = [2.0, 0.0, 0.3]; self.conf = 0.0


def torso_hist(img, box, kp):
    x0, y0, x1, y1 = box
    h = y1 - y0
    ys, ye = int(y0 + 0.2 * h), int(y0 + 0.6 * h); xs, xe = int(x0 + 0.2 * (x1 - x0)), int(x1 - 0.2 * (x1 - x0))
    roi = img[max(0, ys):max(ys + 1, ye), max(0, xs):max(xs + 1, xe)]
    if roi.size < 30: return None
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256])
    return cv2.normalize(hist, hist).ravel()


def raw_facing(kp, face, box):
    """One frame → (label, frontness 0..1). kp: 17×3 (x, y, conf); face: dict or None."""
    c = kp[:, 2]
    eyes, nose, ears = min(c[LEYE], c[REYE]), c[NOSE], (c[LEAR], c[REAR])
    head_feat = max(c[NOSE], c[LEYE], c[REYE])
    body = max(c[LSHO], c[RSHO]) > 0.4 or max(ears) > 0.4
    if face is not None:
        # YuNet also fires on back-3/4 heads (score 0.7-0.9, garbage landmarks), so the pose keypoints must agree:
        # COCO-pose only gives a confident nose/eye when they're actually visible
        r = face["yaw_r"]
        if abs(r) < 0.45 and face["score"] > 0.6 and nose > 0.5 and eyes > 0.3: return "toward", 1.0
        if max(nose, c[LEYE], c[REYE]) < 0.55: return "away", 0.2
        return "side", 0.6
    # no detected face: keypoints decide. COCO-pose labels the nose/eyes only when visible, so their confidence
    # collapses when the back of the head faces the camera.
    sw = kp[LSHO, 0] - kp[RSHO, 0]                    # >0: the person's left shoulder is on image right → facing us
    bw = max(1.0, box[2] - box[0])
    sh_ok = min(c[LSHO], c[RSHO]) > 0.5 and abs(sw) > 0.25 * bw
    if nose > 0.6 and eyes > 0.5 and not (sh_ok and sw < 0): return ("toward" if abs(c[LEYE] - c[REYE]) < 0.3 else "side"), 0.8
    if head_feat > 0.5: return "side", 0.5
    if not body: return None, 0.5                      # nothing to judge (cropped/occluded)
    return "away", 0.0 if (sh_ok and sw < 0) or head_feat < 0.25 else 0.2


def look_dir(kp, face):
    """Which way the head is turned, in image terms: -1 toward image left, +1 toward image right, 0 frontal/unknown."""
    if face is not None:
        r = face["yaw_r"]                                  # nose right of the eye midpoint → looking to image right
        return 0 if abs(r) < 0.25 else (1 if r > 0 else -1)
    c = kp[:, 2]
    if c[LEAR] > 0.5 and c[REAR] < 0.3 and c[NOSE] > 0.3: return -1   # their left ear shows: head turned to their right = image left
    if c[REAR] > 0.5 and c[LEAR] < 0.3 and c[NOSE] > 0.3: return 1
    return 0


# ---------- main tracker ----------
class People:
    def __init__(self, a, audio=None, vad_segments=None):
        self.a = a
        cv2.setNumThreads(a.threads)
        from ultralytics import YOLO
        import lightasd
        self.device = lightasd.pick_device(a.device)
        self.pose = YOLO(M(a.pose_model), task="pose")
        self.asd = lightasd.LightASD(self.device) if a.asd else None
        self.mic = None
        if a.mic_model:
            from ultralytics import YOLOWorld
            self.mic = YOLOWorld(M(a.mic_model))
            if "-mic" not in a.mic_model: self.mic.set_classes(a.mic_prompts)   # needs CLIP; *-mic.pt has them baked in
        self.face_det = None; self.cam = None; self.size = None
        self.audio = audio or AudioRing()
        self.vad_segments = vad_segments                 # offline: [(t0, t1)]
        self.vad_state = (None, False, -99.0)            # live: (t, active, rms_db)
        self.self_speaking = False
        self.people, self.next_id, self.tid2pid = {}, 1, {}
        self.speaker_id, self.challenger = None, (None, 0.0)
        self.t_asd = self.t_mic = -1e9
        self.events = []
        self.last_speech_t = -1e9
        self.timing = collections.defaultdict(list)
        self.mic_boxes = []

    # ----- inputs -----
    def feed_audio(self, t, x): self.audio.feed(t, x)
    def feed_vad(self, t, active, rms_db=-99.0): self.vad_state = (t, bool(active), rms_db)
    def set_self_speaking(self, b): self.self_speaking = bool(b)

    def speech_active(self, t):
        if self.self_speaking: return False
        if self.vad_segments is not None:
            return any(s0 <= t <= s1 + 0.15 for s0, s1 in self.vad_segments)
        tv, act, _ = self.vad_state
        if tv is not None and t - tv < 0.5: return act
        x = self.audio.get(t - 0.3, t) if hasattr(self.audio, "get") else None   # fallback: energy
        return x is not None and len(x) and 20 * math.log10(float(np.sqrt(np.mean(x * x))) + 1e-9) > self.a.energy_db

    # ----- setup -----
    def _setup(self, w, h):
        a = self.a
        self.scale = min(1.0, a.det_width / w)
        self.face_det = cv2.FaceDetectorYN.create(track.MODEL, "", (int(w * self.scale), int(h * self.scale)), a.face_score, 0.3, 50)
        self.cam = track.Camera(w, h, a.hfov, a.fx, a.fy, a.cx, a.cy, a.cam_pos, a.cam_rpy, a.face_width)
        self.size = (w, h)

    def _faces(self, img):
        small = cv2.resize(img, None, fx=self.scale, fy=self.scale) if self.scale < 1 else img
        _, det = self.face_det.detect(small)
        out = []
        for d in (det if det is not None else []):
            d = d.copy(); d[:14] /= self.scale
            x, y, bw, bh = d[:4]
            rex, rey, lex, ley, nx, ny = d[4:10]                  # YuNet: right eye, left eye, nose, mouth R, mouth L
            ed = max(1.0, math.hypot(lex - rex, ley - rey))
            out.append({"cx": x + bw / 2, "cy": y + bh / 2, "w": bw, "h": bh, "score": float(d[14]),
                        "yaw_r": float((nx - (lex + rex) / 2) / ed), "box": (x, y, x + bw, y + bh)})
        return out

    # ----- step -----
    def step(self, img, t=None):
        t = time.perf_counter() if t is None else t
        a = self.a; ev = []
        if img is None: return self._output(t, ev)
        h, w = img.shape[:2]
        if self.size != (w, h): self._setup(w, h); self._fx1, self._fy1 = self.cam.fx, self.cam.fy
        ptz_speed = self._apply_ptz(t, w, h)
        t0 = time.perf_counter()
        r = self.pose.track(img, persist=True, tracker=M("kermit_bytetrack.yaml"), imgsz=a.imgsz, conf=a.person_conf,
                            device=self.device, verbose=False)[0]
        t1 = time.perf_counter(); self.timing["pose_ms"].append((t1 - t0) * 1000)
        faces = self._faces(img)
        t2 = time.perf_counter(); self.timing["face_ms"].append((t2 - t1) * 1000)
        dets = []
        if r.boxes is not None and r.boxes.id is not None:
            B = r.boxes.xyxy.cpu().numpy(); I = r.boxes.id.cpu().numpy().astype(int); K = r.keypoints.data.cpu().numpy()
            dets = [(int(I[i]), B[i], K[i]) for i in range(len(I))]
        seen = set()
        for tid, box, kp in dets:
            pid = self._stable_id(tid, box, kp, img, t)
            P = self.people[pid]; seen.add(pid)
            P.box, P.kp, P.t_seen, P.face = box, kp, t, None
            if kp[[NOSE, LEYE, REYE, LEAR, REAR], 2].max() > 0.5: P.head_hits = getattr(P, "head_hits", 0) + 1
            # a real person shows a head (face, or ears/back of head) at some point; the robot arm doesn't
            if not P.confirmed and t - P.t_first >= a.arrive_s and getattr(P, "head_hits", 0) >= 3:
                P.confirmed = True
        self._assign_faces(faces, seen)
        gray = None
        for pid in seen:
            P = self.people[pid]
            lab, _ = raw_facing(P.kp, P.face, P.box)
            if lab is not None: P.raw.append((t, lab))
            P.look.append((t, look_dir(P.kp, P.face)))
            if P.face is not None:
                if not hasattr(P, "turns"): P.turns = collections.deque(maxlen=30)
                P.turns.append((t, P.face["yaw_r"]))
            ev += self._update_facing(P, t)
            if P.face is not None and self.asd is not None and ptz_speed <= a.ptz_blur_dps:   # motion blur: no lip crops
                if gray is None: gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                f = P.face
                P.crops.append((t, __import__("lightasd").crop_face(gray, f["cx"], f["cy"], max(f["w"], f["h"]))))
            self._geometry(P)
            ev += self._hands(P, t)
            # attention: front rows only (distance / image ROI), unless they raise a hand or hold a mic
            cx, cy = (P.box[0] + P.box[2]) / 2 / w, (P.box[1] + P.box[3]) / 2 / h
            x0, y0, x1, y1 = a.roi
            P.in_roi = P.dist <= a.max_dist and x0 <= cx <= x1 and y0 <= cy <= y1
            P.attended = P.confirmed and (P.in_roi or P.hand_up or P.has_mic)
            if P.attended and not getattr(P, "announced", False):
                P.announced = True; ev.append({"event": "person_arrived", "id": pid})
        # people who left
        for pid in list(self.people):
            P = self.people[pid]
            if pid in seen or t - P.t_seen <= a.lost_s: continue
            if getattr(P, "announced", False):              # left: announce once, re-announce if re-identified later
                ev.append({"event": "person_left", "id": pid}); P.announced = False
            if not P.confirmed or t - P.t_seen > self._reid_s():
                del self.people[pid]
                for k in [k for k, v in self.tid2pid.items() if v == pid]: del self.tid2pid[k]
        speech = self.speech_active(t)
        if speech: self.last_speech_t = t
        if self.asd is not None and t - self.t_asd >= 1.0 / a.asd_hz:
            self.t_asd = t; t3 = time.perf_counter(); self._run_asd(t, speech); self.timing["asd_ms"].append((time.perf_counter() - t3) * 1000)
        if self.mic is not None and t - self.t_mic >= 1.0 / a.mic_hz:
            self.t_mic = t; self._t_now = t; t3 = time.perf_counter(); self._run_mic(img, seen); self.timing["mic_ms"].append((time.perf_counter() - t3) * 1000)
        ev += self._fuse(t, speech)
        for e in ev: e["t"] = round(t, 3)
        out = self._output(t, ev)
        self.timing["total_ms"].append((time.perf_counter() - t0) * 1000)
        return out

    def _hands(self, P, t):
        """hand_up: a confident wrist above the top of the head (eyes/ears/nose, else box top), held ≥ hand_hold s."""
        kp, c = P.kp, P.kp[:, 2]; a = self.a
        head = [kp[i, 1] for i in (NOSE, LEYE, REYE, LEAR, REAR) if c[i] > 0.3]
        top = min(head) - 0.5 * abs(kp[LSHO, 0] - kp[RSHO, 0]) * 0.5 if head else P.box[1] + 0.12 * (P.box[3] - P.box[1])
        up = any(c[i] > 0.5 and kp[i, 1] < top for i in (LWRI, RWRI))
        P.hands.append((t, up))
        xs = [u for tt, u in P.hands if t - tt <= a.hand_hold]
        f = sum(xs) / len(xs) if xs else 0.0
        if not P.hand_up and len(xs) >= 3 and f >= 0.8 and t - P.hands[0][0] >= a.hand_hold * 0.8:
            P.hand_up = True
            if P.confirmed: return [{"event": "hand_raised", "id": P.id}]
        elif P.hand_up and f < 0.2:
            P.hand_up = False
        return []

    def set_ptz(self, pan_deg, tilt_deg, zoom=1.0):
        """Fixed PTZ angles (pan + = left, tilt + = up, base frame). Prefer set_ptz_source() for a moving head."""
        self.ptz_fn = lambda t, d={"pan_deg": pan_deg, "tilt_deg": tilt_deg, "zoom": zoom}: d

    def set_ptz_source(self, pose_at):
        """pose_at(t) → camera pose at frame time t (kermit/camera ptz.PTZ.pose_at, a Pose or its to_json() dict)."""
        self.ptz_fn = pose_at

    def _ptz_pose(self, t, aspect=16 / 9):
        fn = getattr(self, "ptz_fn", None)
        if fn is None and self.a.ptz_pose: fn = self.ptz_fn = PosePoller(self.a.ptz_pose).pose_at
        if fn is None: return None
        q = fn(t)
        if q is None: return None
        if hasattr(q, "to_json"): q = q.to_json(aspect)   # ptz.Pose: fov for this frame's aspect
        g = lambda k, d=None: q.get(k, d)
        return {"pan": float(g("pan_deg", 0.0)), "tilt": float(g("tilt_deg", 0.0)), "zoom": float(g("zoom", 1.0) or 1.0),
                "hfov": g("hfov_deg"), "vfov": g("vfov_deg"), "speed": float(g("speed_dps", 0.0) or 0.0),
                "mount": g("mount"), "mount_yaw": float(g("mount_yaw_deg", 0.0) or 0.0)}

    def _apply_ptz(self, t, w, h):
        """Per-frame camera pose from the PTZ (MOTION's contract): pos = mount, rpy = (0, -tilt, mount_yaw + pan),
        fx from the pose's hfov (zoom included). Returns the pan/tilt speed (deg/s)."""
        q = self._ptz_pose(t, w / h)
        if q is None or self.cam is None: return 0.0
        cam = self.cam
        if q["mount"] is not None: cam.pos = np.array(q["mount"], float)
        cam.R = track.rpy_matrix(0.0, -q["tilt"], q["mount_yaw"] + q["pan"])
        if q["hfov"]:
            cam.fx = (w / 2) / math.tan(math.radians(float(q["hfov"])) / 2)
            cam.fy = (h / 2) / math.tan(math.radians(float(q["vfov"])) / 2) if q["vfov"] else cam.fx
        elif q["zoom"] != 1.0:
            cam.fx = self._fx1 * q["zoom"]; cam.fy = self._fy1 * q["zoom"]
        self.ptz_now = q
        return q["speed"]

    def _stable_id(self, tid, box, kp, img, t):
        if tid in self.tid2pid and self.tid2pid[tid] in self.people:
            pid = self.tid2pid[tid]; P = self.people[pid]
            if P.hist is None or np.random.rand() < 0.1:
                hst = torso_hist(img, box, kp)
                if hst is not None: P.hist = hst if P.hist is None else 0.8 * P.hist + 0.2 * hst
            if P.confirmed or P.box is None: return pid
            # still unconfirmed: keep trying to re-attach to a confirmed person who just dropped out
            old = self._reattach(box, img, kp, t, exclude=pid, confirmed_only=True)
            if old is None: return pid
            O = self.people[old]; O.raw.extend(P.raw); O.crops.extend(P.crops)
            del self.people[pid]
            self.tid2pid = {k: v for k, v in self.tid2pid.items() if v != old}; self.tid2pid[tid] = old
            return old
        # new ByteTrack ID: re-attach to a recently lost person if position and torso colour agree
        best = self._reattach(box, img, kp, t)
        if best is None:
            best = self.next_id; self.next_id += 1
            self.people[best] = Person(best, t); self.people[best].hist = torso_hist(img, box, kp)
        self.tid2pid = {k: v for k, v in self.tid2pid.items() if v != best}
        self.tid2pid[tid] = best
        return best

    def _reid_s(self):
        """How long a lost person is remembered: longer with a PTZ, whose head turns people out of frame."""
        return max(self.a.reid_s, self.a.reid_ptz_s) if getattr(self, "ptz_now", None) else self.a.reid_s

    def _reattach(self, box, img, kp, t, exclude=None, confirmed_only=False):
        hst = torso_hist(img, box, kp); best, best_s = None, 0.0
        cx, cy, bw = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2, box[2] - box[0]
        yaw = None
        if getattr(self, "ptz_now", None) is not None:    # PTZ: pixels move with the head, compare base-frame bearings
            yaw, _ = track.bearing(self.cam.face_to_base(cx, box[1] + 0.1 * (box[3] - box[1]), self.cam.fx * self.cam.face_w / 2.0))
        for pid, P in self.people.items():
            if pid == exclude or P.box is None or t - P.t_seen < 0.1 or t - P.t_seen > self._reid_s(): continue
            if confirmed_only and not P.confirmed: continue
            if yaw is not None and P.yaw is not None:
                d = abs((yaw - P.yaw + 180) % 360 - 180) / 15.0
            else:
                px, py, pw = (P.box[0] + P.box[2]) / 2, (P.box[1] + P.box[3]) / 2, P.box[2] - P.box[0]
                d = math.hypot(cx - px, cy - py) / max(bw, pw, 1)
            sim = float(cv2.compareHist(hst.astype(np.float32), P.hist.astype(np.float32), cv2.HISTCMP_CORREL)) if hst is not None and P.hist is not None else 0.3
            s = sim - 0.3 * d
            if d < 1.5 and sim > 0.5 and s > best_s: best, best_s = pid, s
        return best

    def _assign_faces(self, faces, seen):
        pairs = []
        for fi, f in enumerate(faces):
            for pid in seen:
                P = self.people[pid]; x0, y0, x1, y1 = P.box
                if not (x0 - 0.1 * f["w"] <= f["cx"] <= x1 + 0.1 * f["w"] and y0 - 0.3 * f["h"] <= f["cy"] <= y0 + 0.6 * (y1 - y0)): continue
                nose = P.kp[NOSE]
                hx, hy = (nose[0], nose[1]) if nose[2] > 0.3 else ((x0 + x1) / 2, y0 + 0.1 * (y1 - y0))
                pairs.append((math.hypot(f["cx"] - hx, f["cy"] - hy) / max(f["w"], 1), fi, pid))
        used_f, used_p = set(), set()
        for d, fi, pid in sorted(pairs):
            if fi in used_f or pid in used_p or d > 2.0: continue
            self.people[pid].face = faces[fi]; used_f.add(fi); used_p.add(pid)

    def _update_facing(self, P, t):
        """Asymmetric hysteresis, tuned on real video where a turned back is often seen as 'side' (profile/back-3/4)
        but a face turning back to us is reliably 'toward':
          enter away: ≥ away_frac of raw labels in the last away_win s are NOT toward, and ≥ 1 of them is 'away'
          leave away: ≥ back_frac of raw labels in the last back_win s are 'toward'"""
        a = self.a
        def frac(lab_set, win):
            xs = [l for tt, l in P.raw if t - tt <= win]
            return (sum(l in lab_set for l in xs) / len(xs), len(xs)) if xs else (0.0, 0)
        ev = []
        if P.facing != "away":
            f, n = frac({"away", "side"}, a.away_win)
            fa, _ = frac({"away"}, a.away_win)
            if n >= 3 and f >= a.away_frac and fa > 0:
                P.facing, P.facing_since = "away", t
                if P.confirmed and t - getattr(P, "t_away_ev", -1e9) >= a.event_refractory:
                    P.t_away_ev = t; P.away_announced = True
                    ev.append({"event": "turned_away", "id": P.id})
                else:
                    P.away_announced = False
            else:
                ft, nt = frac({"toward"}, 0.3)
                P.facing = "toward" if nt and ft >= 0.5 else "side"
        else:
            f, n = frac({"toward"}, a.back_win)
            if n >= 3 and f >= a.back_frac:
                P.facing, P.facing_since = "toward", t
                if getattr(P, "away_announced", False): ev.append({"event": "turned_back", "id": P.id})
                P.away_announced = False
        return ev

    def _geometry(self, P):
        cam = self.cam
        if P.face is not None:
            f = P.face; p = cam.face_to_base(f["cx"], f["cy"], f["w"]); P.conf = f["score"]
        else:
            kp, c = P.kp, P.kp[:, 2]
            head = [i for i in (NOSE, LEYE, REYE, LEAR, REAR) if c[i] > 0.3]
            if head: u, v = float(np.mean(kp[head, 0])), float(np.mean(kp[head, 1]))
            elif min(c[LSHO], c[RSHO]) > 0.3: u, v = (kp[LSHO, 0] + kp[RSHO, 0]) / 2, (kp[LSHO, 1] + kp[RSHO, 1]) / 2 - 0.25 * abs(kp[LSHO, 0] - kp[RSHO, 0])
            else: u, v = (P.box[0] + P.box[2]) / 2, P.box[1] + 0.1 * (P.box[3] - P.box[1])
            sw = abs(kp[LSHO, 0] - kp[RSHO, 0]) if min(c[LSHO], c[RSHO]) > 0.5 else 0
            z = cam.fx * 0.38 / sw if sw > 0.15 * (P.box[2] - P.box[0]) else P.dist
            z = min(max(z, 0.4), 8.0)
            p = cam.face_to_base(u, v, cam.fx * cam.face_w / z); P.conf = 0.6
        P.yaw, P.pitch = track.bearing(p); P.dist = float(np.linalg.norm(p)); P.target = [round(float(x), 3) for x in p]

    def _run_asd(self, t, speech):
        import lightasd
        W = self.a.asd_win; T = int(round(W * 25))
        batch, pids = [], []
        for pid, P in self.people.items():
            if t - P.t_seen > 0.05 or not P.crops or P.crops[-1][0] < t - 0.2: continue
            ts = np.array([c[0] for c in P.crops])
            if ts[0] > t - 0.6 * W: continue                       # need ≥ 60% of the window
            grid = t - W + (np.arange(T) + 0.5) / 25
            idx = np.clip(np.searchsorted(ts, grid), 0, len(ts) - 1)
            batch.append(np.stack([P.crops[i][1] for i in idx])); pids.append(pid)
        if not batch: return
        x = self.audio.get(t - W, t)
        if x is None or len(x) < int(W * 16000) - 160: return
        mf = lightasd.mfcc(x)[:4 * T]
        if len(mf) < 4 * T: mf = np.pad(mf, ((0, 4 * T - len(mf)), (0, 0)), mode="edge")
        logits = self.asd(np.stack(batch), mf)
        for pid, lg in zip(pids, logits):
            P = self.people[pid]; P.asd_logit = float(np.mean(lg[-self.a.asd_tail:])); P.asd_t = t

    def _run_mic(self, img, seen):
        r = self.mic.predict(img, imgsz=self.a.mic_imgsz, conf=self.a.mic_conf, device=self.device, verbose=False)[0]
        boxes = [(b.xyxy[0].cpu().numpy(), float(b.conf)) for b in r.boxes] if r.boxes is not None else []
        # mics on stands/tables don't move: drop a detection if one sat at the same spot for most of the last few s
        t, W = self._t_now, img.shape[1]
        hist = getattr(self, "mic_hist", collections.deque(maxlen=int(8 * self.a.mic_hz)))
        self.mic_hist = hist
        n_old = sum(1 for th, _ in hist if t - self.a.mic_static_s <= th <= t - 1.0)
        moving = []
        for b, cf in boxes:
            c = ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
            same = sum(1 for th, cs in hist if t - self.a.mic_static_s <= th <= t - 1.0
                       and any(math.hypot(c[0] - q[0], c[1] - q[1]) < self.a.mic_static_px * W for q in cs))
            if n_old >= 3 and same >= 0.6 * n_old: continue
            moving.append((b, cf))
        hist.append((t, [((b[0] + b[2]) / 2, (b[1] + b[3]) / 2) for b, _ in boxes]))
        self.mic_boxes = moving
        hit = {}
        for (x0, y0, x1, y1), cf in moving:
            mx, my, ms = (x0 + x1) / 2, (y0 + y1) / 2, max(x1 - x0, y1 - y0)
            best, bd = None, 1e9
            for pid in seen:
                P = self.people[pid]; c = P.kp[:, 2]
                pts = [P.kp[i, :2] for i in (LWRI, RWRI) if c[i] > 0.3]      # hand-held: near a wrist (table/stand mics are not)
                if not pts: continue
                scale = max(P.box[2] - P.box[0], 1)
                d = min(math.hypot(mx - p[0], my - p[1]) for p in pts) / scale
                if d < bd: best, bd = pid, d
            if best is not None and bd < self.a.mic_reach:
                hit[best] = max(hit.get(best, 0), cf)
        for pid in seen:
            P = self.people[pid]
            P.mic_ema = 0.6 * P.mic_ema + 0.4 * (1.0 if pid in hit else 0.0)
            if not P.has_mic and P.mic_ema > 0.55: P.has_mic = True
            elif P.has_mic and P.mic_ema < 0.2: P.has_mic = False

    def _loudness(self, t):
        """(rms_db now, noise floor). The floor is the 20th percentile over the last ~20 s, so steady crowd babble
        that keeps the VAD on all the time doesn't count as someone speaking up."""
        x = self.audio.get(t - 0.3, t) if self.audio is not None else None
        if x is None or not len(x):
            tv, _, r = self.vad_state
            if tv is None or t - tv > 0.5: return None, None
        else:
            r = 20 * math.log10(float(np.sqrt(np.mean(np.square(x)))) + 1e-9)
        h = getattr(self, "_rms_hist", None)
        if h is None: h = self._rms_hist = collections.deque(maxlen=400)
        if not h or t - h[-1][0] >= 0.05: h.append((t, r))
        while h and t - h[0][0] > 20: h.popleft()
        return r, float(np.percentile([v for _, v in h], 20))

    def _fuse(self, t, speech):
        a = self.a; ev = []
        present = [P for P in self.people.values() if t - P.t_seen <= a.lost_s and P.confirmed and getattr(P, "attended", True)]
        recent = t - self.last_speech_t < a.vad_hold
        r, floor = self._loudness(t)
        loud = r is not None and r > floor + a.loud_db
        fresh = {P.id: P.asd_logit is not None and t - P.asd_t < 0.6 for P in present}
        # explaining away: voice is clearly up, yet every face we can lip-read looks silent → the talker is someone
        # we can't lip-read (far judge, back turned, face out of view)
        seen_silent = all(P.asd_logit < a.explain_logit for P in present if fresh[P.id]) and any(fresh.values())
        n_mic = sum(1 for P in present if P.has_mic)
        # narrow view (PTZ): being the only one in frame says nothing, and when the voice comes from nobody we can
        # see (unseen_speech) the visible person must not keep a high speaking_p (MOTION searches on that)
        narrow = a.narrow_fov or getattr(self, "ptz_now", None) is not None
        unseen = narrow and getattr(self, "_unseen_now", False)
        for P in present:
            L = a.bias
            L += a.w_asd * (float(np.clip(P.asd_logit, -4, 4)) if fresh[P.id] else a.asd_prior)   # no face → weak "silent" prior
            if not fresh[P.id] and loud and seen_silent: L += a.w_explain
            if P.has_mic:
                # presenters hold handhelds all along (weak cue, shared); the audience mic only reaches whoever is asking
                L += a.w_mic if self.role(P) == "judge" else 0.0 if unseen else a.w_mic / max(2 if narrow else 1, n_mic)
            if P.id == self.speaker_id and not unseen: L += a.w_cont
            if len(present) == 1 and not narrow: L += a.w_alone
            if self.role(P) == "presenter" and not unseen: L += a.w_presenter      # §6: the near presenters do most of the talking
            if not (speech or recent): L -= 4.0
            k = 1 - math.exp(-(t - getattr(P, "_tp", t)) / a.p_tau) if hasattr(P, "_tp") else 1.0
            P.L = L if not hasattr(P, "L") else P.L + k * (L - P.L); P._tp = t     # smooth in logit space
            P.p = sig(P.L)
        # speaker selection (hysteresis); the speaker keeps the floor in silence until they leave
        cur = self.people.get(self.speaker_id)
        if cur is not None and t - cur.t_seen > a.lost_s: cur = None
        best = max(present, key=lambda P: P.L, default=None)
        new = self.speaker_id if cur is not None else None
        if best is not None and (speech or recent) and not self.self_speaking:   # Kermit talking: hold the floor
            if cur is None:
                if best.p > 0.5: new = best.id
            elif best.id != cur.id and best.p > 0.5 and best.L > cur.L + a.switch_margin:   # margin in logits
                cid, since = self.challenger
                if cid != best.id: self.challenger = (best.id, t)
                elif t - since >= a.switch_hold: new = best.id
            else:
                self.challenger = (None, t)
        if new != self.speaker_id:
            if new is not None:
                ev.append({"event": "speaker_changed", "from": self.speaker_id, "to": new})
            self.speaker_id = new; self.challenger = (None, t)
        sp = self.people.get(self.speaker_id)
        if sp is not None and (speech or recent): sp.talk_s += getattr(self, "_dt", 0.0)
        self._dt = t - getattr(self, "_tlast", t); self._tlast = t
        return ev

    def role(self, P):
        """§6 layout: presenters stand ~1 m from Kermit, judges sit 3–4 m back."""
        if getattr(P, "role_truth", None): return P.role_truth
        d = getattr(P, "role_d", None)                         # smoothed distance with ±10% hysteresis
        P.role_d = P.dist if d is None else 0.9 * d + 0.1 * P.dist
        cur = getattr(P, "role_cur", None)
        if cur is None or (cur == "judge" and P.role_d < 0.9 * self.a.presenter_dist) or \
                (cur == "presenter" and P.role_d > 1.1 * self.a.presenter_dist):
            P.role_cur = "presenter" if P.role_d < self.a.presenter_dist else "judge"
        return P.role_cur

    def addressing(self, P, t):
        """Soft addressee cue: 'kermit' when facing us, the id of the person they have turned their head toward,
        'audience' when turned away or toward nobody."""
        if getattr(P, "look_truth", None) is not None:      # oracle mode passes the sim's truth through
            return P.look_truth
        if P.facing == "away": return "audience"
        ht = self.head_turn(P, t)
        if ht is not None:
            d = 0 if abs(ht) < self.a.look_thr else int(np.sign(ht))
        else:
            if P.facing == "toward": return "kermit"
            xs = [d for tt, d in P.look if t - tt <= 0.5]
            d = 0 if not xs else int(np.sign(np.mean(xs))) if abs(np.mean(xs)) > 0.3 else 0
        if d == 0: return "kermit"
        others = [Q for Q in self.people.values() if Q is not P and t - Q.t_seen <= self.a.lost_s]
        # +yaw = image left; looking to image right (d=+1) means toward people with smaller yaw
        cand = [Q for Q in others if (Q.yaw - P.yaw) * -d > 5]
        return min(cand, key=lambda Q: abs(Q.yaw - P.yaw)).id if cand else "audience"

    def head_turn(self, P, t):
        """Mean YuNet head-turn ratio over 0.5 s (nose offset from the eye midpoint / eye distance; + = toward image
        right). None without a visible face."""
        xs = [r for tt, r in getattr(P, "turns", ()) if t - tt <= 0.5]
        return float(np.mean(xs)) if xs else None

    def _output(self, t, ev):
        speech = self.speech_active(t)
        people = []
        for P in sorted(self.people.values(), key=lambda q: q.id):
            if P.box is None or t - P.t_seen > self.a.lost_s or not P.confirmed or not getattr(P, "attended", True): continue
            people.append({"id": P.id, "yaw_deg": round(P.yaw, 1), "pitch_deg": round(P.pitch, 1), "dist_m": round(P.dist, 2),
                           "facing": P.facing, "addressing": self.addressing(P, t),
                           "head_turn": None if self.head_turn(P, t) is None else round(self.head_turn(P, t), 2), "speaking_p": round(P.p, 2), "has_mic": P.has_mic, "hand_up": P.hand_up, "role": self.role(P),
                           "conf": round(P.conf, 2), "face": P.face is not None,
                           "asd": None if P.asd_logit is None else round(P.asd_logit, 2),
                           "box": [int(v) for v in P.box],
                           "face_box": [int(v) for v in P.face["box"]] if P.face is not None else None})
        for e in ev:
            P = self.people.get(e.get("id", e.get("to")))
            if P is not None: e.setdefault("role", self.role(P)); e.setdefault("yaw_deg", round(P.yaw, 1))
            if e["event"] == "turned_away":
                # only if THIS person holds the floor right now: speaker_id, still the top fused score, p ≥ 0.6,
                # and voice in the last second (someone else starting to talk while they turn ≠ them talking)
                Q = self.people.get(e["id"])
                top = max((q for q in self.people.values() if hasattr(q, "L") and q.confirmed), key=lambda q: q.L, default=None)
                e["while_speaking"] = bool((speech or t - self.last_speech_t < 1.0) and Q is not None and
                                           e["id"] == self.speaker_id and top is Q and Q.p >= 0.6)
        self.events = ev
        # someone is talking but no visible face moves its lips (talker off-frame): over the last 1.5 s, voice ≥ 60 %
        # of the time and the best visible ASD logit averages < -1.0 (room sim: 49 % of off-frame talk, 7 % of on-frame)
        if not hasattr(self, "_unseen"): self._unseen = collections.deque()
        asd = [p["asd"] for p in people if p["asd"] is not None]
        self._unseen.append((t, speech, max(asd) if asd else -9.0))
        while self._unseen and self._unseen[0][0] < t - 1.5: self._unseen.popleft()
        W = self._unseen
        unseen = bool(speech and sum(w[1] for w in W) >= 0.6 * len(W) and sum(w[2] for w in W) / len(W) < -1.0)
        self._unseen_now = unseen
        return {"t": round(t, 3), "speaker_id": self.speaker_id, "speech_active": speech, "unseen_speech": unseen, "people": people}

    def gaze(self, out):
        """§4-compatible line aimed at speaker_id (else the closest person)."""
        sp = [p for p in out["people"] if p["id"] == out["speaker_id"]] or sorted(out["people"], key=lambda p: p["dist_m"])[:1]
        if not sp: return {"t": out["t"], "present": False, "n_faces": 0, "speaker_id": None}
        p, P = sp[0], self.people[sp[0]["id"]]
        return {"t": out["t"], "present": True, "yaw_deg": p["yaw_deg"], "pitch_deg": p["pitch_deg"], "dist_m": p["dist_m"],
                "conf": p["conf"], "target_base": P.target, "n_faces": len(out["people"]), "speaker_id": out["speaker_id"],
                "id": p["id"], "facing": p["facing"], "addressing": p["addressing"]}


def context(out):
    """BRAIN context string from a §5 frame."""
    ps = out.get("people", [])
    if not ps: return "no one is in front of you right now"
    side = lambda y: "to your left" if y > 12 else "to your right" if y < -12 else "right in front of you"
    n = f"{len(ps)} {'person' if len(ps) == 1 else 'people'} in front of you"
    sp = next((p for p in ps if p["id"] == out.get("speaker_id")), None)
    hands = [p for p in ps if p.get("hand_up") and p["id"] != out.get("speaker_id")]
    if hands:
        h = hands[0]; n += f"; someone {side(h['yaw_deg'])} is raising a hand"
    if sp is None: return n
    bits = [f"the {sp['role']} {'talking' if out.get('speech_active') else 'you are talking with'} is {side(sp['yaw_deg'])}, about {sp['dist_m']:.1f} metres away"]
    if sp["has_mic"]: bits.append("holding a microphone")
    if sp["facing"] == "away": bits.append("with their back turned to you")
    elif sp.get("addressing") == "kermit": bits.append("talking to you")
    elif isinstance(sp.get("addressing"), int): bits.append("talking to the other person, not you")
    return n + "; " + ", ".join(bits)


class PosePoller:
    """Tails MOTION's ptz_pose.ndjson (or re-reads ptz_pose.json) and returns the pose interpolated at time t."""

    def __init__(self, path):
        self.path, self.fh, self.buf, self.poses = path, None, "", collections.deque(maxlen=600)

    def _read(self):
        if self.path.endswith(".json"):
            try: self.poses.append(json.load(open(self.path)))
            except (OSError, ValueError): pass
            return
        if self.fh is None:
            if not os.path.exists(self.path): return
            self.fh = open(self.path)
        self.buf += self.fh.read()
        *lines, self.buf = self.buf.split("\n")
        for l in lines:
            try: self.poses.append(json.loads(l))
            except ValueError: pass

    def pose_at(self, t):
        self._read()
        if not self.poses: return None
        ps = [p for p in self.poses if "t" in p]
        if not ps: return self.poses[-1]
        after = next((i for i, p in enumerate(ps) if p["t"] >= t), None)
        if after is None or after == 0: return ps[-1] if after is None else ps[0]
        p0, p1 = ps[after - 1], ps[after]; k = (t - p0["t"]) / max(1e-6, p1["t"] - p0["t"])
        out = dict(p1)
        for key in ("pan_deg", "tilt_deg", "zoom", "hfov_deg", "vfov_deg", "speed_dps"):
            if key in p0 and key in p1: out[key] = p0[key] + k * (p1[key] - p0[key])
        return out


def http_frames(url, fps_cap):
    """HTTP snapshot polling that also returns the frame's capture time (X-Frame-T header, perf_counter clock)."""
    import urllib.request
    while True:
        t0 = time.perf_counter()
        try:
            r = urllib.request.urlopen(url, timeout=2)
            data = r.read(); ft = r.headers.get("X-Frame-T")
            yield cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR), (float(ft) if ft else time.perf_counter())
        except OSError as e:
            print(f"snapshot error: {e}", file=sys.stderr); yield None, time.perf_counter()
        time.sleep(max(0.0, 1 / fps_cap - (time.perf_counter() - t0)))


class PeopleThread(threading.Thread):
    """Camera thread for run_kermit.py: reads frames, steps People, publishes latest/log/events."""

    def __init__(self, a, on_event=None):
        super().__init__(daemon=True)
        self.a, self.on_event = a, on_event
        self.pp = None; self.ready = threading.Event()
        self.latest = {"t": 0.0, "speaker_id": None, "speech_active": False, "people": []}
        self.gaze = {"present": False, "n_faces": 0, "speaker_id": None}
        self._pending_audio = AudioRing(); self._self = False; self._vad = None

    def feed_audio(self, t, x): (self.pp.audio if self.pp else self._pending_audio).feed(t, x)

    def feed_vad(self, t, active, rms_db=-99.0):
        if self.pp: self.pp.feed_vad(t, active, rms_db)
        else: self._vad = (t, active, rms_db)

    def set_self_speaking(self, b):
        self._self = bool(b)
        if self.pp: self.pp.set_self_speaking(b)

    def context(self): return context(self.latest)

    def set_ptz(self, pan_deg, tilt_deg, zoom=1.0):
        self._ptz = ("set_ptz", (pan_deg, tilt_deg, zoom))
        if self.pp is not None and hasattr(self.pp, "set_ptz"): self.pp.set_ptz(pan_deg, tilt_deg, zoom)

    def set_ptz_source(self, pose_at):
        """In-process PTZ (kermit/camera ptz.PTZ.pose_at): the camera pose at each frame's capture time."""
        self._ptz = ("set_ptz_source", (pose_at,))
        if self.pp is not None and hasattr(self.pp, "set_ptz_source"): self.pp.set_ptz_source(pose_at)

    def run(self):
        try:
            self._run()
        except Exception:
            import traceback; traceback.print_exc(file=sys.stderr)
            print("people: perception thread died; gaze stays at the last value", file=sys.stderr)

    def _run(self):
        if self.a.oracle:
            import oracle
            self.pp = oracle.Oracle(self.a, self.a.oracle, self.a.error_profile); self.pp.audio = self._pending_audio
        else:
            self.pp = People(self.a, audio=self._pending_audio)
        self.pp.self_speaking = self._self
        if getattr(self, "_ptz", None) and hasattr(self.pp, self._ptz[0]): getattr(self.pp, self._ptz[0])(*self._ptz[1])
        if self._vad: self.pp.feed_vad(*self._vad)
        self.ready.set()
        pub = Publisher(self.a)
        if self.a.oracle: src = ((None, None) for _ in iter(lambda: time.sleep(1 / 30), 1))
        elif self.a.source.startswith("http"): src = http_frames(self.a.source, self.a.fps)
        else: src = ((img, None) for img in track.frames(self.a.source, self.a.fps))
        n, t_stat = 0, time.perf_counter()
        for img, ft in src:
            out = self.pp.step(img, ft)
            n += 1
            if self.a.stats and time.perf_counter() - t_stat > 5:
                tm = {k: round(float(np.median(v[-50:])), 1) for k, v in getattr(self.pp, "timing", {}).items() if v}
                print(f"people: {n / (time.perf_counter() - t_stat):.1f} Hz {tm}", file=sys.stderr); n, t_stat = 0, time.perf_counter()
            # the room's voice state rides on every gaze line too (MOTION's director reads the hub's gaze stream)
            self.latest = out
            self.gaze = {**self.pp.gaze(out), "speech_active": out["speech_active"], "unseen_speech": out.get("unseen_speech", False)}
            pub.write(out, self.pp.events)
            for e in self.pp.events:
                if self.on_event:
                    try: self.on_event(dict(e))
                    except Exception as ex: print(f"people: on_event failed: {ex}", file=sys.stderr)


class Publisher:
    def __init__(self, a):
        self.latest = a.latest; self.log = open(a.log, "w") if a.log else None
        self.ev = open(a.events, "w") if a.events else None

    def write(self, out, events):
        line = json.dumps(out, separators=(",", ":"))
        if self.latest:
            open(self.latest + ".tmp", "w").write(line); os.replace(self.latest + ".tmp", self.latest)
        if self.log: self.log.write(line + "\n"); self.log.flush()
        if self.ev:
            for e in events: self.ev.write(json.dumps(e, separators=(",", ":")) + "\n")
            if events: self.ev.flush()


# ---------- offline rendering ----------
COLORS = [(66, 135, 245), (245, 66, 93), (66, 245, 141), (245, 197, 66), (176, 66, 245), (66, 230, 245), (245, 130, 66)]


def draw(img, out, pp, ev_log):
    W = img.shape[1]; k = W / 1280; fs, th = 0.6 * k, max(1, int(round(2 * k)))
    for p in out["people"]:
        x0, y0, x1, y1 = p["box"]; col = COLORS[p["id"] % len(COLORS)]
        spk = p["id"] == out["speaker_id"]
        cv2.rectangle(img, (x0, y0), (x1, y1), col, 3 * th if spk else th)
        P = pp.people.get(p["id"])
        if P is not None and P.face is not None and "cx" in P.face:
            fx0, fy0, fx1, fy1 = map(int, P.face["box"]); cv2.rectangle(img, (fx0, fy0), (fx1, fy1), (255, 255, 255), 1)
        ad = p.get("addressing"); ad = "" if ad is None else " >Kermit" if ad == "kermit" else " >aud" if ad == "audience" else f" >#{ad}"
        lab = (f"#{p['id']} {p['role'][:5]} {p['facing']}{ad}" + (" MIC" if p["has_mic"] else "") + (" HAND" if p.get("hand_up") else "")
               + (" SPEAKER" if spk else ""))
        org = (x0 + 4, max(int(18 * k), y0 + int(20 * k)))
        cv2.putText(img, lab, org, cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), 3 * th)
        cv2.putText(img, lab, org, cv2.FONT_HERSHEY_SIMPLEX, fs, col, th)
        bh = int((y1 - y0) * p["speaking_p"])                                  # speaking bar on the box's left edge
        cv2.rectangle(img, (x0 - int(10 * k), y1 - bh), (x0 - int(4 * k), y1), (0, 255, 0) if spk else (0, 180, 255), -1)
    for (bx0, by0, bx1, by1), cf in pp.mic_boxes:
        cv2.rectangle(img, (int(bx0), int(by0)), (int(bx1), int(by1)), (0, 255, 255), th)
    h = img.shape[0]
    txt = f"t={out['t']:.1f}s  voice={'ON' if out['speech_active'] else 'off'}  speaker={out['speaker_id']}"
    cv2.putText(img, txt, (10, h - int(12 * k)), cv2.FONT_HERSHEY_SIMPLEX, fs * 1.1, (0, 0, 0), 3 * th)
    cv2.putText(img, txt, (10, h - int(12 * k)), cv2.FONT_HERSHEY_SIMPLEX, fs * 1.1, (255, 255, 255), th)
    for i, (te, e) in enumerate(ev_log[-4:]):
        if out["t"] - te < 4:
            txt = json.dumps({kk: v for kk, v in e.items() if kk not in ("t", "yaw_deg")})
            y = int((24 + 24 * i) * k)
            cv2.putText(img, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX, fs * 0.95, (0, 0, 0), 3 * th)
            cv2.putText(img, txt, (10, y), cv2.FONT_HERSHEY_SIMPLEX, fs * 0.95, (0, 255, 255), th)
    return img


def run_file(a):
    """Offline: video file with its own audio track; t = frame time."""
    x = load_audio(a.source)
    segs = silero_segments(x) if len(x) else []
    pp = People(a, audio=FileAudio(x), vad_segments=segs)
    cap = cv2.VideoCapture(a.source); fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    if a.start: cap.set(cv2.CAP_PROP_POS_MSEC, a.start * 1000)
    step_every = max(1, int(round(fps / a.proc_fps))) if a.proc_fps else 1
    pub = Publisher(a); vw = None; ev_log = []; i = 0
    while True:
        ok, img = cap.read()
        if not ok: break
        t = a.start + i / fps; i += 1
        if a.duration and t > a.start + a.duration: break
        if (i - 1) % step_every: continue
        out = pp.step(img, t)
        pub.write(out, pp.events); ev_log += [(t, e) for e in pp.events]
        for e in pp.events: print(json.dumps(e), file=sys.stderr)
        if a.render:
            if vw is None:
                vw = cv2.VideoWriter(a.render + ".tmp.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps / step_every, (img.shape[1], img.shape[0]))
            vw.write(draw(img, out, pp, ev_log))
    if vw is not None:
        vw.release()
        subprocess.run(["ffmpeg", "-y", "-v", "quiet", "-i", a.render + ".tmp.mp4", "-ss", str(a.start), "-t", str(t - a.start), "-i", a.source,
                        "-map", "0:v", "-map", "1:a?", "-c:v", "libx264", "-crf", "26", "-preset", "veryfast", "-pix_fmt", "yuv420p",
                        "-c:a", "aac", "-shortest", a.render])
        os.remove(a.render + ".tmp.mp4")
    st = {k: round(float(np.median(v)), 1) for k, v in pp.timing.items()}
    print(json.dumps({"timing_median_ms": st, "frames": i}), file=sys.stderr)
    return pp


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="0")
    ap.add_argument("--hfov", type=float, default=78.0)
    for k in ("fx", "fy", "cx", "cy"): ap.add_argument("--" + k, type=float)
    ap.add_argument("--cam-pos", type=float, nargs=3, default=[0.05, 0.0, 0.10])
    ap.add_argument("--cam-rpy", type=float, nargs=3, default=[0.0, 0.0, 0.0])
    ap.add_argument("--face-width", type=float, default=0.16)
    ap.add_argument("--device", default=None, help="cuda | mps | cpu (default: best available)")
    ap.add_argument("--pose-model", default="yolo11n-pose.pt")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--person-conf", type=float, default=0.4)
    ap.add_argument("--det-width", type=int, default=960, help="YuNet input width")
    ap.add_argument("--face-score", type=float, default=0.6)
    ap.add_argument("--no-asd", dest="asd", action="store_false")
    ap.add_argument("--asd-hz", type=float, default=5.0); ap.add_argument("--asd-win", type=float, default=1.2)
    ap.add_argument("--asd-tail", type=int, default=8, help="average the last N 25-fps frame logits of the window")
    ap.add_argument("--mic-model", default="yolov8m-worldv2-mic.pt", help="'' disables mic detection; *-mic.pt = classes baked in (no CLIP at runtime)")
    ap.add_argument("--mic-prompts", nargs="+", default=["microphone", "handheld microphone"])
    ap.add_argument("--mic-static-s", type=float, default=4.0, help="a mic seen at the same spot over this long = on a stand")
    ap.add_argument("--mic-static-px", type=float, default=0.012, help="'same spot' radius, fraction of image width")
    ap.add_argument("--mic-conf", type=float, default=0.4); ap.add_argument("--mic-imgsz", type=int, default=640)
    ap.add_argument("--mic-hz", type=float, default=3.0); ap.add_argument("--mic-reach", type=float, default=0.3)
    ap.add_argument("--away-win", type=float, default=1.0); ap.add_argument("--away-frac", type=float, default=0.85)
    ap.add_argument("--back-win", type=float, default=0.5); ap.add_argument("--back-frac", type=float, default=0.6)
    ap.add_argument("--event-refractory", type=float, default=3.0, help="min s between turned_away events per person")
    ap.add_argument("--w-asd", type=float, default=1.5); ap.add_argument("--w-mic", type=float, default=3.0)
    ap.add_argument("--w-cont", type=float, default=0.5); ap.add_argument("--w-alone", type=float, default=2.5)
    ap.add_argument("--bias", type=float, default=1.0); ap.add_argument("--asd-prior", type=float, default=-1.0)
    ap.add_argument("--w-presenter", type=float, default=0.75, help="prior bonus for people within --presenter-dist")
    ap.add_argument("--w-explain", type=float, default=2.0); ap.add_argument("--explain-logit", type=float, default=-1.5)
    ap.add_argument("--loud-db", type=float, default=6.0, help="'voice clearly up' = this far above the 20 s noise floor")
    ap.add_argument("--max-dist", type=float, default=5.0, help="ignore people farther than this (m) unless hand up / mic")
    ap.add_argument("--roi", type=float, nargs=4, default=[0, 0, 1, 1], help="x0 y0 x1 y1 (0-1 image coords) of the box centre")
    ap.add_argument("--hand-hold", type=float, default=0.5, help="wrist above head this long = hand_up")
    ap.add_argument("--narrow-fov", action="store_true", help="camera sees only part of the room (PTZ; automatic with a PTZ pose): no 'alone' prior")
    ap.add_argument("--reid-ptz-s", type=float, default=30.0, help="with a PTZ: remember people who left the frame this long (s)")
    ap.add_argument("--ptz-pose", help="MOTION's ptz_pose.ndjson (interpolated at frame time) or ptz_pose.json (latest)")
    ap.add_argument("--ptz-blur-dps", type=float, default=30.0, help="skip lip-ASD crops while the PTZ turns faster than this")
    ap.add_argument("--look-thr", type=float, default=0.25, help="|head_turn| above this = looking at the person on that side")
    ap.add_argument("--presenter-dist", type=float, default=2.0, help="closer than this = presenter, else judge")
    ap.add_argument("--p-tau", type=float, default=0.25); ap.add_argument("--vad-hold", type=float, default=0.6)
    ap.add_argument("--switch-margin", type=float, default=0.5); ap.add_argument("--switch-hold", type=float, default=0.3)
    ap.add_argument("--energy-db", type=float, default=-38.0, help="fallback VAD when no feed_vad()")
    ap.add_argument("--arrive-s", type=float, default=0.5); ap.add_argument("--lost-s", type=float, default=2.0)
    ap.add_argument("--reid-s", type=float, default=4.0)
    ap.add_argument("--fps", type=float, default=30.0, help="snapshot polling cap")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--stats", action="store_true", help="print loop rate and per-stage ms to stderr every 5 s")
    ap.add_argument("--latest"); ap.add_argument("--log"); ap.add_argument("--events")
    ap.add_argument("--oracle", help="room-sim truth.ndjson: degraded-oracle mode (no camera, see oracle.py)")
    ap.add_argument("--error-profile", default=os.path.join(HERE, "eval", "error_profile.json"))
    ap.add_argument("--render", help="offline: write an annotated MP4")
    ap.add_argument("--start", type=float, default=0.0); ap.add_argument("--duration", type=float, default=0.0)
    ap.add_argument("--proc-fps", type=float, default=0.0, help="offline: process at this rate (0 = every frame)")
    return ap.parse_args(argv)


def main():
    a = parse_args()
    if os.path.isfile(a.source) and not a.oracle:
        run_file(a); return
    th = PeopleThread(a, on_event=lambda e: print(json.dumps(e), file=sys.stderr)); th.start()
    while th.is_alive():
        time.sleep(0.5)


if __name__ == "__main__":
    main()
