"""Degraded-oracle perception for the live room sim (people.py --oracle truth.ndjson --error-profile real.json).

The sim's rendered people are found by YOLO-pose, but their primitive faces mostly aren't by YuNet, so lip-ASD can't
run on them. Instead, the room sim's ground truth (truth.ndjson, perf_counter clock) is corrupted with error rates
measured on REAL video (eval/error_profile.json), then pushed through the SAME facing hysteresis, fusion and
speaker-selection code as people.py. The result is the same §5 stream and the same events.

Truth lines (anything without these keys is ignored):
  {"t": 12.3, "people": [{"id": "kirill", "role": "presenter", "yaw_deg": 30, "pitch_deg": 5, "dist_m": 1.0,
                         "pos_base": [x, y, z], "facing": "toward|side|away", "speaking": true, "has_mic": false}]}
Each person may also appear as its own line {"t", "id", ...}; all people with the same t are grouped together.
Mic VAD still comes from the real (simulated) room mic via feed_vad / feed_audio.
"""
import json, math, os, random, time
import numpy as np

import people as P_

DEFAULT_PROFILE = {
    "source": "measured on real video, see eval/REPORT.md; eval/error_profile.json overrides",
    # causal Light-ASD logits through the full live pipeline on the Columbia panel (static camera, frontal faces, like
    # the stage): 46k labelled face-frames, AUC 0.87. (AVA movies: -0.31±1.37 / -2.29±0.83.) AR(1) from AVA at 5 Hz.
    "asd_logit_speaking": [-0.32, 0.74], "asd_logit_silent": [-1.52, 0.76], "asd_ar": 0.8,   # AR calibrated: oracle on Columbia truth = real end-to-end speaker acc (eval/calib_oracle.py)
    "asd_max_dist_m": 4.0,           # 1080p capture, 78°: a face at 4 m is ~47 px (Columbia-like); 640 px capture → use 2.2
    # facing: lecture clip, 120 s hand-labelled at 1 Hz (rows = truth, cols = per-frame raw label)
    "facing_confusion": {"toward": {"toward": 0.97, "side": 0.03, "away": 0.0},
                         "side": {"toward": 0.0, "side": 0.84, "away": 0.16},
                         "away": {"toward": 0.0, "side": 0.48, "away": 0.52}},
    # the lecture's "side" frames were mostly back-3/4 (75-110°); a clean profile (< 75°) keeps both eyes/nose/ear
    # landmarks and is not confused with a back (lecture frames with the head at ~45-70°: 0 of 23 called away)
    "facing_confusion_side_clean": {"toward": 0.06, "side": 0.94, "away": 0.0},
    "side_clean_below_deg": 75,
    "facing_block_s": 0.5,           # raw-label errors are correlated over ~0.5 s (same pose, same mistake)
    "yaw_sigma_deg": 1.0, "dist_rel_sigma": 0.15,
    "dropouts_per_min": 0.5, "dropout_s": [0.2, 1.0], "id_switch_after_dropout": 0.1,
    "mic_miss": 0.0, "mic_false_pos": 0.0,   # person-level, single frame, on the town-hall set: R 1.0; FPs only on fixed table mics (filtered)
    "respect_in_fov": True,          # people outside the camera FOV are not seen at all
}


class Oracle(P_.People):
    def __init__(self, a, truth_path, profile_path=None):
        # no models: People.__init__ is skipped on purpose; only the state the shared methods need is set up
        self.a = a
        self.prof = dict(DEFAULT_PROFILE)
        if profile_path and os.path.exists(profile_path): self.prof.update(json.load(open(profile_path)))
        self.truth_path = truth_path; self.fh = None; self.buf = ""; self.frames = []; self.last_truth = None
        self.audio = P_.AudioRing(); self.vad_segments = None; self.vad_state = (None, False, -99.0)
        self.self_speaking = False
        self.people, self.next_id, self.name2pid = {}, 1, {}
        self.speaker_id, self.challenger = None, (None, 0.0)
        self.events = []; self.last_speech_t = -1e9; self.mic_boxes = []
        self.timing = {}
        self.rng = random.Random(7)
        self.noise = {}                 # name → dict(asd_state, drop_until, facing_obs, facing_t, mic_t)
        self.cam = None
        self.t_asd = self.t_mic = -1e9

    # ----- truth input -----
    def _read_truth(self):
        if self.fh is None:
            if not os.path.exists(self.truth_path): return
            self.fh = open(self.truth_path)
        chunk = self.fh.read()
        if not chunk: return
        self.buf += chunk
        *lines, self.buf = self.buf.split("\n")
        for l in lines:
            try: o = json.loads(l)
            except ValueError: continue
            if "people" in o and "t" in o: self.frames.append((o["t"], o["people"]))
            elif "t" in o and "id" in o and "facing" in o:
                if self.frames and abs(self.frames[-1][0] - o["t"]) < 1e-4: self.frames[-1][1].append(o)
                else: self.frames.append((o["t"], [o]))
        self.frames = self.frames[-200:]

    def truth_at(self, t):
        self._read_truth()
        best = None
        for tt, ps in self.frames:
            if tt <= t + 0.02: best = (tt, ps)
        return best

    # ----- corruption -----
    def _nz(self, name):
        if name not in self.noise:
            self.noise[name] = {"asd": 0.0, "drop_until": -1, "fobs": None, "ft": -1e9, "mic": False, "mic_t": -1e9}
        return self.noise[name]

    def _pid(self, name, t, new=False):
        if new or name not in self.name2pid:
            self.name2pid[name] = self.next_id; self.next_id += 1
        return self.name2pid[name]

    def step(self, img=None, t=None):
        t = time.perf_counter() if t is None else t
        a, pr, ev = self.a, self.prof, []
        tr = self.truth_at(t)
        seen = set()
        dt = t - getattr(self, "_t_prev", t); self._t_prev = t
        for g in (tr[1] if tr else []):
            name = str(g.get("id", g.get("name"))); nz = self._nz(name)
            if pr.get("respect_in_fov", True) and g.get("in_fov") is False: continue
            # dropouts (track lost), with an optional ID switch afterwards
            if t < nz["drop_until"]: continue
            if nz["drop_until"] > 0 and t >= nz["drop_until"]:
                nz["drop_until"] = -1
                if self.rng.random() < pr["id_switch_after_dropout"]: self._pid(name, t, new=True)
            if self.rng.random() < pr["dropouts_per_min"] / 60 * dt:
                nz["drop_until"] = t + self.rng.uniform(*pr["dropout_s"]); continue
            pid = self._pid(name, t)
            if pid not in self.people: self.people[pid] = P_.Person(pid, t)
            P = self.people[pid]; seen.add(pid)
            P.t_seen = t; P.box = [0, 0, 1, 1]; P.name = name; P.role_truth = g.get("role")
            P.attended = True; P.announced = True
            if bool(g.get("hand_up")) and not P.hand_up and P.confirmed: ev.append({"event": "hand_raised", "id": pid})
            P.hand_up = bool(g.get("hand_up"))
            if not P.confirmed and t - P.t_first >= a.arrive_s:
                P.confirmed = True; ev.append({"event": "person_arrived", "id": pid})
            # geometry
            P.yaw = float(g.get("yaw_deg", 0)) + self.rng.gauss(0, pr["yaw_sigma_deg"])
            P.pitch = float(g.get("pitch_deg", 0)) + self.rng.gauss(0, pr["yaw_sigma_deg"])
            P.dist = max(0.3, float(g.get("dist_m", 2)) * (1 + self.rng.gauss(0, pr["dist_rel_sigma"])))
            yr, pr_ = math.radians(P.yaw), math.radians(P.pitch)
            P.target = [round(P.dist * math.cos(pr_) * math.cos(yr), 3), round(P.dist * math.cos(pr_) * math.sin(yr), 3),
                        round(P.dist * math.sin(pr_), 3)]
            P.conf = 0.8
            # facing: observed label resampled per block from the real-video confusion matrix → shared hysteresis
            truth_f = g.get("facing", "toward")
            if t - nz["ft"] >= pr["facing_block_s"]:
                row = pr["facing_confusion"].get(truth_f, {truth_f: 1.0})
                ang = g.get("facing_diff_deg")
                if truth_f == "side" and ang is not None and ang < pr.get("side_clean_below_deg", 75):
                    row = pr["facing_confusion_side_clean"]     # a clean profile is never mistaken for a back
                r, acc = self.rng.random(), 0.0
                for lab, p in row.items():
                    acc += p
                    if r <= acc: nz["fobs"] = lab; break
                else: nz["fobs"] = truth_f
                nz["ft"] = t
            P.raw.append((t, nz["fobs"]))
            ev += self._update_facing(P, t)
            P.face = {"truth_facing": truth_f, "box": (0, 0, 1, 1)} if truth_f != "away" else None
            P.look_truth = g.get("addressing")
            P.truth_speaking = bool(g.get("speaking", False)); P.truth_mic = bool(g.get("has_mic", False))
        # ASD (5 Hz): AR(1) noise around the real-video logit distributions, only with a visible face in range
        if t - self.t_asd >= 1.0 / a.asd_hz:
            self.t_asd = t
            for pid in seen:
                P = self.people[pid]; nz = self._nz(P.name)
                if P.face is None or P.dist > pr["asd_max_dist_m"]: P.asd_logit = None; continue
                mu, sd = pr["asd_logit_speaking"] if P.truth_speaking else pr["asd_logit_silent"]
                rho = pr["asd_ar"]; nz["asd"] = rho * nz["asd"] + math.sqrt(1 - rho * rho) * self.rng.gauss(0, 1)
                P.asd_logit, P.asd_t = mu + sd * nz["asd"], t
        if t - self.t_mic >= 1.0 / a.mic_hz:
            self.t_mic = t
            for pid in seen:
                P = self.people[pid]
                det = (P.truth_mic and self.rng.random() > pr["mic_miss"]) or (not P.truth_mic and self.rng.random() < pr["mic_false_pos"])
                P.mic_ema = 0.6 * P.mic_ema + 0.4 * float(det)
                if not P.has_mic and P.mic_ema > 0.55: P.has_mic = True
                elif P.has_mic and P.mic_ema < 0.2: P.has_mic = False
        for pid in list(self.people):
            P = self.people[pid]
            if pid not in seen and t - P.t_seen > a.lost_s:
                if P.confirmed: ev.append({"event": "person_left", "id": pid})
                del self.people[pid]
        speech = self.speech_active(t)
        if speech: self.last_speech_t = t
        ev += self._fuse(t, speech)
        for e in ev: e["t"] = round(t, 3)
        out = self._output(t, ev)
        for p in out["people"]:
            p.pop("box", None); p.pop("face_box", None)
            p["name_truth"] = getattr(self.people[p["id"]], "name", None)
        return out

    def gaze(self, out):
        sp = [p for p in out["people"] if p["id"] == out["speaker_id"]] or sorted(out["people"], key=lambda p: p["dist_m"])[:1]
        if not sp: return {"t": out["t"], "present": False, "n_faces": 0, "speaker_id": None}
        p = sp[0]; P = self.people[p["id"]]
        return {"t": out["t"], "present": True, "yaw_deg": p["yaw_deg"], "pitch_deg": p["pitch_deg"], "dist_m": p["dist_m"],
                "conf": p["conf"], "target_base": P.target, "n_faces": len(out["people"]), "speaker_id": out["speaker_id"],
                "id": p["id"], "facing": p["facing"], "addressing": p.get("addressing")}

    def addressing(self, P, t):
        a = getattr(P, "look_truth", None)
        if a is None:                                   # no truth addressee: derive from (corrupted) facing
            return "kermit" if P.facing == "toward" else "audience" if P.facing == "away" else None
        return self.name2pid.get(a, a)                  # person names → our ids; kermit/audience pass through
