"""The live room: wall-clock audio engine + scripted humans + noise + ground truth (CONTRACT §6).

  .venv/bin/python server.py --run run/<name> --noise normal [--script ../brain/demo_script/v3/script.json]

Real time, 20 ms blocks at 16 kHz. Every block:
  dry sources (humans' scheduled lines, Kermit's loudspeaker as written by run_kermit through sd.py,
  crowd babble + events, HVAC, laptop fan) ─► streaming partitioned convolution with precomputed RIRs
  ─► Kermit's mic (sent to sd.InputStream clients) + an audience-seat listener (for the video mix).
Nothing is pre-mixed: when a human speaks is decided live by the driver, from what Kermit actually does
(the room hears his loudspeaker). Kermit's run_kermit process connects over a unix socket (sd.py).

Outputs in --run: mic.wav, audience.wav, kermit_dry.wav, mic_{human,echo,noise}.wav (components at the mic),
truth.ndjson (people at 30 Hz + utterance/word/event ground truth), levels.ndjson (10 Hz), people.ndjson
(pose commands for MOTION's LiveWorld), meta.json (t0 = perf_counter of sample 0).
"""
import argparse, json, math, os, socket, struct, sys, threading, time
import numpy as np, soundfile as sf
from scipy.signal import butter, lfilter, lfilter_zi

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import acoustics as A
from acoustics import FS, BLOCK
from humans import line_key, VOICES
import noise as N

_HDR = struct.Struct("<dI")
SPK_SPL_0DBFS = 86.0          # Kermit's loudspeaker: a 0 dBFS-rms signal = 86 dB SPL at 1 m (speech ≈ 68 dB @1 m)
NOISE_LEVELS = {"quiet": 45.0, "normal": 60.0, "loud": 70.0}
CAM_POS = np.array([0.10, 0.12, 0.55])   # sim camera: raised post (CONTRACT §6 update), nudged so both presenters are in view
CAM_HFOV = 120.0


class Clock:
    def __init__(self):
        self.t0 = None

    def t(self, n):            # engine sample index → perf_counter
        return self.t0 + n / FS

    def n(self, t):
        return int(round((t - self.t0) * FS))


# ------------------------------------------------------------------ dry tracks
class Track:
    """Scheduled clips for one dry source. Clip = dict(n0, sig, gain, cut_n, meta)."""

    def __init__(self):
        self.clips, self.lock = [], threading.Lock()

    def add(self, n0, sig, gain, meta=None):
        c = dict(n0=n0, sig=sig, gain=gain, cut_n=None, meta=meta or {})
        with self.lock: self.clips.append(c)
        return c

    def cut(self, c, at_n, fade=0.08):
        c["cut_n"] = at_n; c["fade"] = int(fade * FS)

    def render(self, n):
        y = np.zeros(BLOCK, np.float32)
        with self.lock:
            keep = []
            for c in self.clips:
                L = len(c["sig"]) if c["cut_n"] is None else min(len(c["sig"]), c["cut_n"] - c["n0"] + c["fade"])
                a, b = n - c["n0"], n + BLOCK - c["n0"]
                if a >= L: continue
                keep.append(c)
                if b <= 0: continue
                i0, i1 = max(a, 0), min(b, L)
                seg = c["sig"][i0:i1] * c["gain"]
                if c["cut_n"] is not None:          # fade out after the cut point
                    k = np.arange(i0, i1) - (c["cut_n"] - c["n0"])
                    seg = seg * np.clip(1 - k / max(c["fade"], 1), 0, 1)
                y[i0 - a:i0 - a + len(seg)] += seg
            self.clips = keep
        return y

    def active(self, n):
        with self.lock:
            return any(c["n0"] <= n < c["n0"] + len(c["sig"]) and (c["cut_n"] is None or n < c["cut_n"]) for c in self.clips)


class SpeakerRing:
    """Kermit's loudspeaker feed: 24 kHz samples placed at their scheduled play time (+ room output latency
    and slow jitter), consumed by the engine 480 at a time and resampled 24k→16k with a stateful filter."""

    def __init__(self, clock, out_latency=0.012, jitter_ms=4.0, seed=3):
        self.clock, self.lat, self.jit = clock, out_latency, jitter_ms / 1000
        self.size = 24000 * 20
        self.buf = np.zeros(self.size, np.float32); self.lock = threading.Lock()
        self.rng = np.random.default_rng(seed); self.j = 0.0
        self.late_blocks = 0; self.writes = 0
        self.consumed24 = 0
        # 24k → 48k (zero-stuff x2) → LPF → keep every 3rd → 16k
        self.b, self.a = butter(8, 7600 / 24000)
        self.zi = lfilter_zi(self.b, self.a) * 0
        # loudspeaker: small driver, highpass 180 Hz; soft clip at the top
        self.hb, self.ha = butter(2, 180 / 8000, "high"); self.hzi = lfilter_zi(self.hb, self.ha) * 0

    def write(self, t_play, x):
        self.j = 0.98 * self.j + 0.2 * self.jit * self.rng.standard_normal()
        k = int(round((t_play + self.lat + self.j - self.clock.t0) * 24000))
        with self.lock:
            self.writes += 1
            if k < self.consumed24:                       # arrived after the engine passed that time
                self.late_blocks += 1
                k = self.consumed24
            idx = np.arange(k, k + len(x)) % self.size
            self.buf[idx] += x

    def take16(self, n16):
        k0 = n16 * 3 // 2
        with self.lock:
            idx = np.arange(k0, k0 + BLOCK * 3 // 2) % self.size
            x24 = self.buf[idx].copy(); self.buf[idx] = 0
            self.consumed24 = k0 + BLOCK * 3 // 2
        up = np.zeros(len(x24) * 2, np.float32); up[::2] = x24 * 2
        y, self.zi = lfilter(self.b, self.a, up, zi=self.zi)
        dry = y[::3].astype(np.float32)
        return dry


# ------------------------------------------------------------------ the room
class Room:
    def __init__(self, a):
        self.a = a
        self.clock = Clock()
        rirs, self.rir_meta = A.compute_rirs(rt60=a.rt60)
        self.S = A.sources(); self.snames = list(self.S); self.lnames = list(A.LISTENERS)
        def rir(s, l):   # a handheld moves with its holder's head: facing the audience doesn't change its direct path
            if l == "hh_" + s.split("@")[0] and s.endswith("@audience"): s = s.replace("@audience", "@kermit")
            return rirs[f"{s}|{l}"]
        self.conv = A.StreamConv([[rir(s, l) for l in self.lnames] for s in self.snames])
        self.judge_mic = None             # config B: which judge holds the passed Q&A mic
        self.groups = {g: np.array([self.S[s]["group"] == g for s in self.snames]) for g in ("human", "echo", "noise")}
        lib = N.Library(seed=a.seed)
        noise_src = [s for s in self.snames if self.S[s]["group"] == "noise"]
        self.noise = N.NoiseField(lib, N.calibrate(lib, rirs, noise_src), noise_src, seed=a.seed + 1)
        self.tracks = {p: Track() for p in A.PEOPLE}
        self.spk = SpeakerRing(self.clock)
        self.facing_w = {p: 0.0 for p in A.PEOPLE}     # 0 = facing Kermit, 1 = facing the audience (presenters)
        self.facing_target = dict(self.facing_w)
        self.n = 0
        self.mic_clients, self.cl_lock = [], threading.Lock()
        self.kermit_last_sound_n = -10 ** 9; self.kermit_on = False; self.kermit_onsets = []
        self.base_dba = NOISE_LEVELS.get(a.noise, 60.0)
        self.level_fn = lambda t: self.base_dba
        self.hz, self.hzi = butter(2, 180 / 8000, "high"), None
        self.spk_hp_zi = lfilter_zi(self.spk.hb, self.spk.ha) * 0
        self.rng = np.random.default_rng(a.seed + 7)
        self.overruns = 0; self.max_proc_ms = 0.0; self.proc_ms = []
        os.makedirs(a.run, exist_ok=True)
        w = lambda name: sf.SoundFile(os.path.join(a.run, name), "w", FS, 1, "FLOAT")
        self.wav = {k: w(k + ".wav") for k in ("mic", "mic_table", "audience", "kermit_dry", "mic_human", "mic_echo", "mic_noise")}
        self.truth = open(os.path.join(a.run, "truth.ndjson"), "w", buffering=1)
        self.levels = open(os.path.join(a.run, "levels.ndjson"), "w", buffering=1)
        self.tlock = threading.Lock()
        self.lv_acc = []
        self.stop = threading.Event()

    def log_truth(self, obj):
        with self.tlock: self.truth.write(json.dumps(obj, separators=(",", ":")) + "\n")

    def now_n(self):
        return self.clock.n(time.perf_counter())

    # --- engine ---------------------------------------------------------
    def engine(self):
        t_next = self.clock.t0 + BLOCK / FS
        hp_zi = None
        while not self.stop.is_set():
            dt = t_next - time.perf_counter()
            if dt > 0: time.sleep(dt)
            elif dt < -0.1: self.overruns += 1
            tp = time.perf_counter()
            n = self.n
            t_mid = self.clock.t(n + BLOCK // 2)
            base = self.level_fn(t_mid)
            nz = self.noise.block(base)
            X = np.zeros((len(self.snames), BLOCK), np.float32)
            lomb = 10 ** ((0.3 * max(0.0, base - 55.0)) / 20)       # Lombard: people speak up in noise
            for p in A.PEOPLE:
                d = self.tracks[p].render(n) * lomb
                if A.PEOPLE[p]["role"] == "presenter":
                    f = self.facing_w[p] = self.facing_w[p] + (self.facing_target[p] - self.facing_w[p]) * 0.12
                    X[self.snames.index(p + "@kermit")] = d * math.cos(f * math.pi / 2)
                    X[self.snames.index(p + "@audience")] = d * math.sin(f * math.pi / 2)
                else:
                    X[self.snames.index(p)] = d
            spk = self.spk.take16(n)
            self.wav["kermit_dry"].write(spk)
            if np.max(np.abs(spk)) > 1e-3:
                if not self.kermit_on:
                    i = int(np.argmax(np.abs(spk) > 1e-3))
                    self.kermit_on = True; self.kermit_onsets.append(n + i)
                    self.log_truth({"t": round(self.clock.t(n + i), 4), "event": "kermit_audio_on"})
                self.kermit_last_sound_n = n + BLOCK - 1 - int(np.argmax(np.abs(spk[::-1]) > 1e-3))
            elif self.kermit_on and n - self.kermit_last_sound_n > int(0.25 * FS):
                self.kermit_on = False
                self.log_truth({"t": round(self.clock.t(self.kermit_last_sound_n), 4), "event": "kermit_audio_off"})
            if hp_zi is None: hp_zi = lfilter_zi(self.spk.hb, self.spk.ha) * 0
            spk_hp, hp_zi = lfilter(self.spk.hb, self.spk.ha, spk, zi=hp_zi)
            spk_out = np.tanh(1.3 * spk_hp) / 1.3                     # mild driver compression/clipping
            X[self.snames.index("kermit_spk")] = spk_out * A.spl_to_dry_gain(SPK_SPL_0DBFS)
            for s, blk in nz.items():
                X[self.snames.index(s)] = blk
            hum_gain = A.spl_to_dry_gain(self.a.speech_spl) / 1.0
            for p in A.PEOPLE:   # human dry units: speech-active rms 0.1 in the line files → x10
                for s in ([p + "@kermit", p + "@audience"] if A.PEOPLE[p]["role"] == "presenter" else [p]):
                    X[self.snames.index(s)] *= hum_gain * 10
            sel = [0, 1]
            if self.a.input == "B":
                live = ["hh_kirill", "hh_steve"] + (["hh_" + self.judge_mic] if self.judge_mic else [])
                sel += [self.lnames.index(l) for l in live]
            Y = self.conv.process(X * A.PA_TO_DIGITAL, sel)             # pascals → mic digital, [S, len(sel), B]
            table = np.clip(Y[:, 0].sum(0) + self.rng.standard_normal(BLOCK).astype(np.float32) * 10 ** (-88 / 20), -1, 1).astype(np.float32)
            src = Y[:, 0:1] if self.a.input == "A" else Y[:, 2:] * (A.HH_TO_DIGITAL / A.PA_TO_DIGITAL)   # the mixer feed
            mic_parts = {g: src[self.groups[g]].sum((0, 1)) for g in self.groups}
            mic = sum(mic_parts.values()) + self.rng.standard_normal(BLOCK).astype(np.float32) * 10 ** (-88 / 20)
            mic = np.clip(mic, -1, 1).astype(np.float32)
            aud = np.clip(Y[:, 1].sum(0), -1, 1).astype(np.float32)
            self.wav["mic_table"].write(table)
            self.n = n + BLOCK
            t_end = self.clock.t(self.n)
            msg = _HDR.pack(t_end, BLOCK) + mic.tobytes()
            with self.cl_lock:
                for c in list(self.mic_clients):
                    try: c.sendall(msg)
                    except OSError: self.mic_clients.remove(c)
            self.wav["mic"].write(mic); self.wav["audience"].write(aud)
            for g in mic_parts: self.wav["mic_" + g].write(mic_parts[g])
            self.lv_acc.append((mic, mic_parts))
            if len(self.lv_acc) == 5:
                self._log_levels(t_end, base)
            pm = (time.perf_counter() - tp) * 1000
            self.max_proc_ms = max(self.max_proc_ms, pm); self.proc_ms.append(pm)
            t_next += BLOCK / FS

    def _log_levels(self, t, base):
        mic = np.concatenate([m for m, _ in self.lv_acc])
        parts = {g: np.concatenate([p[g] for _, p in self.lv_acc]) for g in self.groups}
        self.lv_acc = []
        db = lambda x: round(float(10 * np.log10(np.mean(x ** 2) + 1e-20)), 1)
        cal = (A.MIC_DBFS_AT_94 - A.HH_DBFS_AT_94) if self.a.input == "B" else 0.0   # dBFS → SPL for the channel Kermit gets
        self.levels.write(json.dumps({"t": round(t, 3), "venue_dba": round(base, 1),
                                      "mic_dbfs": db(mic), "mic_dba_spl": round(A.digital_to_spl(A.a_weight_db(mic)) + cal, 1),
                                      "human_dbfs": db(parts["human"]), "echo_dbfs": db(parts["echo"]),
                                      "noise_dbfs": db(parts["noise"]), "noise_dba_spl": round(A.digital_to_spl(A.a_weight_db(parts["noise"])) + cal, 1)}) + "\n")

    # --- socket ---------------------------------------------------------
    def serve(self, path):
        if ":" in path:      # tcp host:port
            h, p = path.rsplit(":", 1)
            srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM); srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            srv.bind((h, int(p)))
        else:
            if os.path.exists(path): os.unlink(path)
            srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM); srv.bind(path)
        srv.listen(8)
        while not self.stop.is_set():
            c, _ = srv.accept()
            threading.Thread(target=self._client, args=(c,), daemon=True).start()

    def _client(self, c):
        f = c.makefile("rb")
        hello = json.loads(f.readline())
        if hello["role"] == "mic":
            c.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1) if c.family == socket.AF_INET else None
            with self.cl_lock: self.mic_clients.append(c)
            self.log_truth({"t": round(time.perf_counter(), 4), "event": "client", "role": "mic"})
            return
        self.log_truth({"t": round(time.perf_counter(), 4), "event": "client", "role": "spk"})
        try:
            while True:
                h = f.read(_HDR.size)
                if len(h) < _HDR.size: break
                t_play, n = _HDR.unpack(h)
                x = np.frombuffer(f.read(4 * n), np.float32)
                self.spk.write(t_play, x)
        except OSError:
            pass

    def close(self):
        self.stop.set(); time.sleep(0.1)
        for w in self.wav.values(): w.close()
        self.truth.close(); self.levels.close()


# ------------------------------------------------------------------ people (poses, facing, truth)
class Person:
    def __init__(self, name):
        self.name = name; p = A.PEOPLE[name]
        self.role = p["role"]; self.pos = np.array(p["pos"], float)
        self.addressing = "kermit"
        self.body_yaw = self._body_default()
        self.head_target = "kermit"           # kermit | audience | <person>
        self.talking = False; self.gesture = None
        self.line = None                      # (clip, words_abs[(w, n0, n1)])

    def _body_default(self):
        if self.name == "kirill": return -100.0
        if self.name == "steve": return 100.0
        return 180.0

    def head_yaw(self, people):
        if self.head_target == "audience": return 0.0 if self.role == "presenter" else 180.0
        if self.head_target == "kermit": return A.head_yaw_to(self.pos, A.MIC_POS)
        return A.head_yaw_to(self.pos, people[self.head_target].pos)


def bearing(pos, origin=np.zeros(3)):
    d = np.asarray(pos, float) - origin
    yaw = math.degrees(math.atan2(d[1], d[0])); pitch = math.degrees(math.atan2(d[2], math.hypot(d[0], d[1])))
    return yaw, pitch, float(np.linalg.norm(d))


def facing_label(person_pos, head_yaw):
    to_cam = A.head_yaw_to(person_pos, CAM_POS)
    diff = abs((head_yaw - to_cam + 180) % 360 - 180)
    return ("toward" if diff < 45 else "side" if diff < 100 else "away"), round(diff, 1)


# ------------------------------------------------------------------ the driver (script)
REACTIONS = {  # crowd reaction after Kermit finishes a beat (claps at the ends of beats)
    "wake": ("applause", 8.0, 2.5), "why_here": ("laughter", 6.0, 0), "back_turned": ("laughter", 9.0, 0),
    "respect": ("applause", 6.0, 2.0), "steve_gestures": ("laughter", 5.0, 0), "steve_robotics": ("applause", 8.0, 3.0),
    "handoff": ("applause", 10.0, 3.0), "qa_nod": ("laughter", 4.0, 0), "qa_hard": ("applause", 10.0, 3.5),
}


class Driver:
    def __init__(self, room, a):
        self.room, self.a = room, a
        self.people = {p: Person(p) for p in A.PEOPLE}
        self.lines_dir = os.path.join(HERE, "assets", "lines")
        s = json.load(open(a.script)); self.steps = s["steps"]
        self.people_log = open(os.path.join(a.run, "people.ndjson"), "w", buffering=1)
        self.rng = np.random.default_rng(a.seed + 11)
        self.floor = None
        self.results = []
        self.ptz_pose = os.path.join(a.run, "ptz_pose_sim.json") if a.ptz else None   # the simulated motors
        self._ptz_cache = (0, None)

    def _ptz(self):
        try:
            m = os.path.getmtime(self.ptz_pose)
            if m != self._ptz_cache[0]: self._ptz_cache = (m, json.load(open(self.ptz_pose)))
        except (OSError, ValueError):
            pass
        return self._ptz_cache[1]

    def load_line(self, who, text):
        k = line_key(who, text)
        x, _ = sf.read(os.path.join(self.lines_dir, k + ".wav"), dtype="float32")
        return x, json.load(open(os.path.join(self.lines_dir, k + ".json")))

    # --- kermit state as heard in the room ---
    def kermit_silent_for(self):
        r = self.room
        return (r.now_n() - r.kermit_last_sound_n) / FS if not r.kermit_on else 0.0

    def wait(self, cond, timeout, poll=0.01):
        t_end = time.perf_counter() + timeout
        while time.perf_counter() < t_end:
            if cond(): return True
            time.sleep(poll)
        return False

    def sleep(self, s): time.sleep(max(0.0, s))

    # --- poses → truth + MOTION ---
    def state_loop(self):
        r = self.room
        while not r.stop.is_set():
            t = time.perf_counter(); n = r.clock.n(t)
            ppl, cmds = [], []
            for name, P in self.people.items():
                hy = P.head_yaw(self.people)
                fl, fdiff = facing_label(P.pos, hy)
                yaw, pitch, dist = bearing(P.pos)
                cy, cp, cd = bearing(P.pos, CAM_POS)
                in_fov = abs(cy) < CAM_HFOV / 2
                if self.ptz_pose:   # PTZ: visibility from the camera's current pan/tilt/zoom
                    pp = self._ptz()
                    if pp:
                        py, ppitch, _ = bearing(P.pos, np.array(pp["mount"]))
                        dy = (py - pp["pan_deg"] + 180) % 360 - 180; dp = ppitch - pp["tilt_deg"]
                        in_fov = abs(dy) < pp["hfov_deg"] / 2 - 2 and abs(dp) < pp["vfov_deg"] / 2
                        cy = dy
                speaking, mouth = False, 0.0
                if P.line is not None:
                    clip, words = P.line
                    if clip["n0"] <= n < clip["n0"] + len(clip["sig"]) and (clip["cut_n"] is None or n < clip["cut_n"]):
                        i = n - clip["n0"]
                        seg = clip["sig"][max(0, i - 480):i]
                        mouth = float(min(1.0, np.sqrt(np.mean(seg ** 2) + 1e-12) / 0.1 * 0.6)) if len(seg) else 0.0
                        speaking = any(n0 - 0.05 * FS <= n <= n1 + 0.25 * FS for _, n0, n1 in words)
                ppl.append({"id": name, "role": P.role, "yaw_deg": round(yaw, 2), "pitch_deg": round(pitch, 2), "dist_m": round(dist, 3),
                            "target_base": [round(float(v), 3) for v in P.pos], "pos_base": [round(float(v), 3) for v in P.pos],
                            "facing": fl, "facing_diff_deg": fdiff, "speaking": speaking,
                            "has_mic": self.a.input == "B" and (P.role == "presenter" or self.room.judge_mic == name),
                            "addressing": P.addressing, "body_yaw_deg": P.body_yaw, "head_yaw_deg": round(hy, 1),
                            "mouth": round(mouth, 3), "hand_up": P.gesture == "raise_hand", "in_fov": bool(in_fov), "cam_yaw_deg": round(cy, 1)})
                cmds.append({"name": name, "pos_base": [float(P.pos[0]), float(P.pos[1])], "yaw_deg": P.body_yaw,
                             "head_yaw_deg": round(((hy - P.body_yaw + 180) % 360) - 180, 1), "mouth": round(mouth, 3),
                             "talking": speaking, "gesture": P.gesture})
            r.log_truth({"t": round(t, 4), "floor": self.floor, "people": ppl})
            self.people_log.write(json.dumps({"t": round(t, 4), "people": cmds}) + "\n")
            time.sleep(max(0.0, 1 / 30 - (time.perf_counter() - t)))

    # --- one human line ---
    def say(self, who, text, step, staging=(), barge_in_expected=False):
        r = self.room
        x, meta = self.load_line(who, text)
        n0 = r.now_n() + int(0.04 * FS)
        clip = r.tracks[who].add(n0, x, 1.0, {"step": step})
        words = [(w["w"], n0 + int(w["t0"] * FS), n0 + int(w["t1"] * FS)) for w in meta["words"]]
        P = self.people[who]; P.line = (clip, words)
        self.floor = who
        t_on = r.clock.t(n0 + int(meta["words"][0]["t0"] * FS)) if meta["words"] else r.clock.t(n0)
        r.log_truth({"t": round(t_on, 4), "event": "utt_on", "who": who, "step": step, "text": text,
                     "words": [{"w": w, "t0": round(r.clock.t(a0), 4), "t1": round(r.clock.t(a1), 4)} for w, a0, a1 in words],
                     "addressing": P.addressing, "turn_end": True})
        acts = sorted(staging, key=lambda s: s["at_word"])
        cut_at, kermit_cut_in = None, None
        k_on0 = len(r.kermit_onsets)
        while True:
            n = r.now_n()
            for s in list(acts):   # staging at word onsets
                wi = min(s["at_word"], len(words) - 1)
                if n >= words[wi][1]:
                    self.stage(s); acts.remove(s)
            if len(r.kermit_onsets) > k_on0 and kermit_cut_in is None:
                kermit_cut_in = r.kermit_onsets[k_on0]
                r.log_truth({"t": round(r.clock.t(kermit_cut_in), 4), "event": "kermit_over_human", "who": who, "step": step,
                             "expected": barge_in_expected})
                if barge_in_expected:   # trail off: finish the current word, then stop
                    nxt = [w for w in words if w[2] >= n]
                    cut_at = (nxt[0][2] if nxt else n) + int(0.05 * FS)
                    r.tracks[who].cut(clip, cut_at, fade=0.12)
            end = clip["n0"] + len(x) if cut_at is None else cut_at
            if n >= end: break
            time.sleep(0.01)
        spoken = [w for w in words if cut_at is None or w[1] < cut_at]
        t_off = r.clock.t(min(spoken[-1][2], cut_at or 10 ** 12)) if spoken else r.clock.t(end)
        P.line = None if cut_at is None else P.line
        r.log_truth({"t": round(t_off, 4), "event": "utt_off", "who": who, "step": step, "cut": cut_at is not None,
                     "n_words": len(spoken)})
        self.floor = None
        return t_off, kermit_cut_in

    def stage(self, s):
        who = s.get("who"); act = s.get("action", "")
        P = self.people.get(who)
        if not P: return
        if "toward the audience" in act or "turns away" in act:
            P.head_target, P.addressing = "audience", "audience"
            P.body_yaw = -30.0 if who == "kirill" else 30.0
            self.room.facing_target[who] = 1.0
        elif "turns back" in act or "facing him" in act:
            P.head_target, P.addressing = "kermit", "kermit"
            P.body_yaw = P._body_default(); self.room.facing_target[who] = 0.0
        self.room.log_truth({"t": round(time.perf_counter(), 4), "event": "staging", "who": who, "action": act})

    def look_all(self, target):
        for name, P in self.people.items():
            if P.addressing == "audience": continue
            if name == target: continue
            P.head_target = target if target in self.people else "kermit"

    # --- the show ---
    def run(self):
        r = self.room
        threading.Thread(target=self.state_loop, daemon=True).start()
        self.sleep(self.a.lead_in)
        prev = None
        for st in self.steps:
            i, who, text = st["step"], st["speaker"], st["line"]
            k = st.get("kermit") or None
            res = {"step": i, "who": who, "beat": (k or {}).get("beat"), "expected": bool(k)}
            # ---- wait for the floor
            if prev is not None:
                pk = prev["kermit"]; pres = self.results[-1]
                if pk and pk.get("beat") == "interject":
                    # Kirill does not pause: continue right after the interjection (or after a beat if none came)
                    if pres.get("kermit_cut_in") is not None:
                        self.wait(lambda: self.kermit_silent_for() >= 0.25, 15)
                        self.sleep(0.15)
                    else:
                        self.sleep(0.4)
                elif pk:
                    got = self.wait(lambda: len(r.kermit_onsets) > pres["_k_on"], self.a.reply_timeout)
                    if not got:
                        pres["missed"] = True
                        r.log_truth({"t": round(time.perf_counter(), 4), "event": "missed_turn", "step": prev["step"]})
                    else:
                        pres["reply_onset_t"] = r.clock.t(r.kermit_onsets[pres["_k_on"]])
                        self.look_all("kermit")
                        self.wait(lambda: self.kermit_silent_for() >= self.a.floor_gap, 60)
                        pres["reply_end_t"] = r.clock.t(r.kermit_last_sound_n)
                        react = REACTIONS.get(pk.get("beat"))
                        if react:
                            kind, rel, dur = react
                            r.log_truth({"t": round(time.perf_counter(), 4), "event": "crowd", "kind": kind, "rel_db": rel})
                            (r.noise.applause(rel, dur) if kind == "applause" else r.noise.laughter(rel))
                            self.sleep((dur or 1.8) + 0.3)
                    self.sleep(self.rng.uniform(0.2, 0.5))
                else:
                    # no reply expected (presenter to presenter): a normal human gap; if Kermit talks anyway, wait
                    self.sleep(self.rng.uniform(0.5, 0.8))
                    if self.kermit_silent_for() < 0.3 or len(r.kermit_onsets) > pres["_k_on"]:
                        self.wait(lambda: self.kermit_silent_for() >= self.a.floor_gap, 30)
            # ---- the line
            if who in ("judge1", "judge2", "judge3"):
                if self.room.judge_mic != who:   # the runner brings the Q&A mic over
                    self.room.judge_mic = who
                    r.log_truth({"t": round(time.perf_counter(), 4), "event": "judge_mic", "who": who})
                    self.sleep(0.8)
                for n_, P in self.people.items():
                    if P.role == "presenter": P.head_target = who
                self.people[who].gesture = "raise_hand"; self.sleep(0.6); self.people[who].gesture = None
            if who == "steve" and i == 8 + 1:
                self.people["kirill"].head_target = "steve"
            if i == 8:
                self.people["kirill"].addressing = "steve"; self.people["kirill"].head_target = "steve"
            elif who == "kirill" and self.people["kirill"].head_target == "steve":
                self.people["kirill"].addressing = "kermit"; self.people["kirill"].head_target = "kermit"
            if who == "steve":
                self.people["steve"].addressing = "kermit"; self.people["steve"].head_target = "kermit"
            self.look_all(who)
            self.people[who].head_target = "audience" if self.people[who].addressing == "audience" else ("steve" if i == 8 else "kermit")
            # people don't start a line over Kermit: if he is (still) talking, wait for him to finish
            cont = prev is not None and (prev.get("kermit") or {}).get("beat") == "interject"   # "does not pause"
            if not cont and self.kermit_silent_for() < 0.5:
                self.wait(lambda: self.kermit_silent_for() >= self.a.floor_gap, 30)
                self.sleep(self.rng.uniform(0.2, 0.4))
            res["_k_on"] = len(r.kermit_onsets)
            barge = bool(k) and k.get("beat") == "interject"
            res["line_on_t"] = time.perf_counter()
            t_off, cut_in = self.say(who, text, i, st.get("staging", []), barge)
            res["line_off_t"] = t_off
            res["kermit_cut_in"] = r.clock.t(cut_in) if cut_in is not None else None
            if barge and cut_in is None:
                # no interjection during the line: count as missed; Kermit may still answer after it
                res["missed"] = True
            self.results.append(res)
            prev = st
        # final reply
        pres = self.results[-1]
        if prev.get("kermit"):
            if self.wait(lambda: len(r.kermit_onsets) > pres["_k_on"], self.a.reply_timeout):
                pres["reply_onset_t"] = r.clock.t(r.kermit_onsets[pres["_k_on"]])
                self.wait(lambda: self.kermit_silent_for() >= self.a.floor_gap, 60)
                react = REACTIONS.get(prev["kermit"].get("beat"))
                if react: r.noise.applause(react[1], react[2]) if react[0] == "applause" else r.noise.laughter(react[1])
            else:
                pres["missed"] = True
        self.sleep(self.a.tail)
        json.dump([{k_: v for k_, v in x.items() if not k_.startswith("_")} for x in self.results],
                  open(os.path.join(self.a.run, "driver_results.json"), "w"), indent=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--script", default=os.path.join(HERE, "..", "brain", "demo_script", "v3", "script.json"))
    ap.add_argument("--noise", default="normal", help="quiet | normal | loud | <dBA>")
    ap.add_argument("--input", default="A", choices=["A", "B"], help="Kermit's input: A = table mic at the arm base, "
                    "B = close-talk handhelds (presenters' + one passed judge mic) mixed")
    ap.add_argument("--drift-db", type=float, default=3.0, help="slow venue level drift amplitude")
    ap.add_argument("--rt60", type=float, default=A.RT60)
    ap.add_argument("--speech-spl", type=float, default=68.0, help="human voice, dB SPL @1 m before Lombard")
    ap.add_argument("--sock", default=os.environ.get("KERMIT_ROOM_SOCK", "127.0.0.1:18766"), help="host:port or unix socket path")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lead-in", type=float, default=6.0, help="room noise only, before the first line (let run_kermit connect)")
    ap.add_argument("--wait-clients", type=float, default=120.0, help="wait this long for run_kermit's mic to connect")
    ap.add_argument("--reply-timeout", type=float, default=8.0)
    ap.add_argument("--floor-gap", type=float, default=0.7, help="Kermit silent this long = he is done")
    ap.add_argument("--tail", type=float, default=4.0)
    ap.add_argument("--steps", default="", help="e.g. 1-7 to run a subset")
    ap.add_argument("--ptz", action="store_true", help="truth in_fov from the PTZ camera's pose (run/ptz_pose.json)")
    a = ap.parse_args()
    if a.noise not in NOISE_LEVELS: NOISE_LEVELS[a.noise] = float(a.noise)
    room = Room(a)
    room.clock.t0 = time.perf_counter() + 0.05
    base = NOISE_LEVELS[a.noise]
    ph = room.rng.uniform(0, 6.28)
    room.level_fn = lambda t: base + a.drift_db * math.sin(2 * math.pi * (t - room.clock.t0) / 47.0 + ph)
    threading.Thread(target=room.serve, args=(a.sock,), daemon=True).start()
    threading.Thread(target=room.engine, daemon=True).start()
    json.dump({"t0": room.clock.t0, "fs": FS, "noise": a.noise, "base_dba": base, "rt60": a.rt60, "speech_spl": a.speech_spl,
               "spk_spl_0dbfs": SPK_SPL_0DBFS, "cam_pos": CAM_POS.tolist(), "cam_hfov": CAM_HFOV, "rir": room.rir_meta,
               "noise_calib": room.noise.calib, "input": a.input, "levels_cal": "per-channel", "voices": VOICES, "script": a.script},
              open(os.path.join(a.run, "meta.json"), "w"), indent=1)
    print(f"room up: {a.run} noise={a.noise} ({base} dBA); waiting for the mic client", file=sys.stderr, flush=True)
    t_w = time.perf_counter()
    while not room.mic_clients and time.perf_counter() - t_w < a.wait_clients: time.sleep(0.1)
    drv = Driver(room, a)
    if a.steps:
        lo, hi = map(int, a.steps.split("-")); drv.steps = [s for s in drv.steps if lo <= s["step"] <= hi]
    try:
        drv.run()
    finally:
        pm = np.array(room.proc_ms or [0])
        json.dump({"overruns": room.overruns, "proc_ms_mean": round(float(pm.mean()), 2), "proc_ms_p99": round(float(np.percentile(pm, 99)), 2),
                   "proc_ms_max": round(float(pm.max()), 2), "spk_late_blocks": room.spk.late_blocks, "spk_writes": room.spk.writes,
                   "noise_events": room.noise.log, "duration_s": room.n / FS},
                  open(os.path.join(a.run, "engine.json"), "w"), indent=1)
        room.close()
        print("room done", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
