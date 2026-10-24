"""Multi-person staged scenes: presenter, judges and the puppet, with speaker-following gaze.

A *stage script* (JSON, MOTION-side) lists the people and a sequence of spoken lines:

    {"floor_z": -0.8,
     "people": [{"id": 2, "name": "presenter", "style": "presenter", "role": "presenter", "mic": true,
                 "pos": [1.0, -0.55], "seated": false}, ...],
     "lines": [{"who": "presenter", "audio": "x.wav", "text": "...", "gap": 0.4,
                "actions": [{"at": 0.6, "do": "turn_away"}]},           # at = fraction of the line
               {"who": "puppet", "timeline": "turn.timeline.json", "gap": 0.5}, ...],
     "shots": "auto"}

``build_stage`` turns it into: one WAV, the puppet's CONTRACT §2 timeline (its lines placed on the
scene clock), per-person pose curves, a synthetic CONTRACT §5 perception log (who is speaking,
bearings, facing), a gaze bias that follows ``speaker_id``, and a camera cut list.
"""
from __future__ import annotations

import json
import wave
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

from . import human as H
from .animation import Timeline, hermite, lowpass, smooth_noise, smoothstep
from .follow import CAMERA_POS, PITCH_LIMIT, YAW_LIMIT

SR = 24000


def _wav(path: Path) -> np.ndarray:
    with wave.open(str(path)) as w:
        x = np.frombuffer(w.readframes(w.getnframes()), np.int16)
        if w.getframerate() != SR:
            n = int(len(x) * SR / w.getframerate())
            x = np.interp(np.linspace(0, len(x) - 1, n), np.arange(len(x)), x).astype(np.int16)
    return x


def _envelope(pcm: np.ndarray, hz: float = 60) -> np.ndarray:
    hop = int(SR / hz)
    n = len(pcm) // hop
    x = pcm[: n * hop].astype(float) / 32768
    rms = np.sqrt(np.mean((x * x).reshape(n, hop), axis=1))
    rms = rms / (np.percentile(rms, 95) + 1e-6)
    return np.clip((rms - 0.08) / 0.8, 0, 1)


@dataclass
class Person:
    id: int
    name: str
    style: str
    role: str
    mic: bool
    pos: np.ndarray
    seated: bool
    yaw0: float = 0.0
    model: H.PersonModel | None = None
    speak: np.ndarray | None = None      # (N,) mouth/voice envelope on the scene clock
    talking: np.ndarray | None = None    # (N,) 0/1 smoothed: this person holds the floor
    yaw: np.ndarray | None = None        # (N,) root yaw
    hand_up: np.ndarray | None = None    # (N,) judges raise a hand before asking
    pos_t: np.ndarray | None = None      # (N, 2) floor position over time (move_to actions)
    mic_hold: dict | None = None


@dataclass
class Stage:
    floor_z: float
    people: list[Person]
    timeline: Timeline                   # the puppet's merged CONTRACT §2 timeline
    audio_path: Path
    captions: list
    t: np.ndarray
    fps: float
    puppet_talking: np.ndarray
    shots: list = field(default_factory=list)  # [(t0, camera)]
    audio_delay: float = 0.0                   # seconds of silence to add before the soundtrack
    perception: list = field(default_factory=list)
    speaker_face: np.ndarray | None = None
    speaker_present: np.ndarray | None = None

    # ---- scene construction --------------------------------------------------------------------
    def people_spec(self):
        out = []
        for p in self.people:
            chair = (p.pos[0] - 0.02 * np.cos(p.yaw0), p.pos[1] - 0.02 * np.sin(p.yaw0), p.yaw0) if p.seated else None
            out.append((p.name, p.style, p.mic, chair))
        return out

    def bind(self, model):
        for p in self.people:
            p.model = H.bind(model, p.name)
            if p.mic:
                p.mic_hold = H.solve_mic_hold(model, p.model, dict(H.STANDING))

    # ---- per-frame posing ----------------------------------------------------------------------
    def pose_people(self, model, data, i: int, puppet_head: np.ndarray):
        t = self.t[i]
        speaker = self._floor_holder(i)
        for p in self.people:
            base = dict(H.SEATED if p.seated else H.STANDING)
            talk = p.talking[i]
            env = p.speak[i]
            # attention: look at whoever holds the floor (the puppet by default)
            look_at = puppet_head
            if speaker is not None and speaker is not p:
                look_at = speaker.face_point
            xy = p.pos_t[i] if p.pos_t is not None else p.pos
            root = np.array([xy[0], xy[1], self.floor_z + (H.SEAT_DROP if p.seated else 0.0)])
            yaw = p.yaw[i]
            face = root + np.array([0, 0, 1.55 if p.seated else 1.62])
            if np.linalg.norm(root[:2]) > 6:  # parked offstage
                continue
            d = look_at - face
            rel = np.arctan2(d[1], d[0]) - yaw
            rel = (rel + np.pi) % (2 * np.pi) - np.pi
            facing_away = abs(rel) > np.deg2rad(100)
            ny = np.clip(rel, -1.1, 1.1) if not facing_away else 0.15 * np.sin(0.7 * t)
            npitch = -np.arctan2(d[2], np.hypot(d[0], d[1]))
            base["neck_yaw"] = 0.6 * ny
            base["lumbar_twist"] = 0.25 * ny
            base["neck_pitch"] = np.clip(npitch, -0.4, 0.5) + 0.03 * smooth_noise(np.array([t]), p.id)[0]
            # idle sway and breathing
            base["lumbar_side"] = 0.02 * smooth_noise(np.array([t]), p.id + 10, (0.21, 0.33))[0]
            base["lumbar_bend"] = base.get("lumbar_bend", 0) + 0.01 * np.sin(2 * np.pi * 0.22 * t + p.id)
            # talking: head nods on stresses, free-hand beat gestures
            if talk > 0.01:
                beat = lowpass_at(p.speak, i, self.fps)
                base["neck_pitch"] += 0.08 * talk * (env - 0.4)
                base["neck_roll"] = 0.05 * talk * smooth_noise(np.array([t]), p.id + 20, (0.4, 0.7))[0]
                g = talk * (0.5 + 0.5 * smooth_noise(np.array([t]), p.id + 30, (0.23, 0.37))[0])
                base["sh_flex_l"] = base.get("sh_flex_l", 0) + 0.35 * g + 0.25 * beat * talk
                base["elbow_l"] = base.get("elbow_l", 0) + 1.0 * g + 0.3 * beat * talk
                base["sh_abd_l"] = base.get("sh_abd_l", 0) + 0.15 * g
                base["sh_rot_l"] = -0.3 * g
            if p.mic and p.mic_hold is not None:
                hold = smoothstep(p.talking[i] * 1.6)  # mic comes up to the mouth to speak
                rest = dict(sh_flex_r=0.15, sh_abd_r=0.08, sh_rot_r=0.1, elbow_r=0.9, wrist_r=0.0)
                for j in ("sh_flex_r", "sh_abd_r", "sh_rot_r", "elbow_r", "wrist_r"):
                    base[j] = (1 - hold) * rest[j] + hold * p.mic_hold[j]
            if p.hand_up is not None and p.hand_up[i] > 0.01:
                u = p.hand_up[i]
                base["sh_flex_r"] = (1 - u) * base.get("sh_flex_r", 0) + u * 2.5
                base["sh_abd_r"] = (1 - u) * base.get("sh_abd_r", 0) + u * 0.25
                base["elbow_r"] = (1 - u) * base.get("elbow_r", 0) + u * 0.5
            # stepping while turning or walking (standing people only)
            if not p.seated:
                j0, j1 = max(i - 1, 0), min(i + 1, len(self.t) - 1)
                dyaw = abs(p.yaw[j1] - p.yaw[j0]) * self.fps / 2
                if p.pos_t is not None:
                    dyaw += 3.0 * np.linalg.norm(p.pos_t[j1] - p.pos_t[j0]) * self.fps / 2
                if dyaw > 0.05:
                    ph = np.sin(2 * np.pi * 1.8 * t)
                    base["hip_flex_l"] = max(0, ph) * 0.35 * min(1, dyaw)
                    base["knee_l"] = max(0, ph) * 0.6 * min(1, dyaw)
                    base["hip_flex_r"] = max(0, -ph) * 0.35 * min(1, dyaw)
                    base["knee_r"] = max(0, -ph) * 0.6 * min(1, dyaw)
            H.apply(model, data, p.model, root, yaw, base, mouth_open=float(env * talk))
            p.face_point = face

    def _floor_holder(self, i):
        best = max(self.people, key=lambda p: p.talking[i])
        return best if best.talking[i] > 0.3 else None

    def camera(self, i: int) -> str:
        t = self.t[i]
        cam = self.shots[0][1]
        for t0, c in self.shots:
            if t0 <= t:
                cam = c
        return cam


def lowpass_at(x, i, fps):
    lo = max(0, i - int(fps * 0.15))
    return float(np.max(x[lo:i + 1]) if i >= lo else 0)


# -------------------------------------------------------------------------------------------------

def build_stage(script_path: Path, fps: float = 60.0, preroll: float = 0.8, tail: float = 1.0) -> Stage:
    script_path = Path(script_path)
    S = json.loads(script_path.read_text())
    root = script_path.parent
    people = []
    for pj in S["people"]:
        pos = np.asarray(pj["pos"], float)
        yaw0 = pj.get("yaw", float(np.arctan2(-pos[1], -pos[0] + 0.45)))  # face the puppet by default
        people.append(Person(id=pj["id"], name=pj["name"], style=pj["style"], role=pj.get("role", "judge"),
                             mic=pj.get("mic", False), pos=pos, seated=pj.get("seated", False), yaw0=yaw0))
        people[-1].start = np.asarray(pj["start"], float) if "start" in pj else None
    by_name = {p.name: p for p in people}

    if "voice_scene" in S:
        S["_offset"] = preroll  # VOICE's mix has no pre-roll; the renderer delays it
        laid = _from_voice_scene(S, root, preroll)
    else:
        laid = _lay_out_lines(S, root, script_path, preroll)
    words, mouth, events, captions, spans, actions, audio_path, t, emotion0 = laid
    total = t + tail
    mouth += [0.0] * (int(np.ceil(total * 60)) - len(mouth))
    tl = Timeline(duration=total - 0.6, audio=audio_path, words=words, mouth=np.asarray(mouth), mouth_hz=60,
                  events=sorted(events, key=lambda e: e["t"]), emotion=emotion0 or "neutral", gaze="speaker",
                  captions=captions)
    st = _finish(S, people, by_name, tl, spans, actions, fps)
    st.audio_delay = S.get("_offset", 0.0)
    return st


def _from_voice_scene(S, root, preroll):
    """VOICE's scene.json: absolute clip times on one mixed soundtrack (barge-in overlaps allowed)."""
    vs_path = (root / S["voice_scene"]).resolve()
    vs = json.loads(vs_path.read_text())
    names = S.get("speakers", {"human": "presenter"})
    off = preroll
    words, events, captions, spans, actions = [], [], [], [], []
    emotion0 = None
    for c in vs["clips"]:
        t0, t1 = c["t"] + off, c["t"] + c["duration"] + off
        if c["who"] == "kermit":
            tl = json.loads((vs_path.parent / c["timeline"]).read_text())
            emotion0 = emotion0 or tl["emotion"]
            words += [dict(w, t0=round(w["t0"] + t0, 3), t1=round(w["t1"] + t0, 3)) for w in tl["words"]]
            events.append({"t": round(t0 - 0.3, 3), "type": "emotion", "value": tl["emotion"]})
            events += [dict(e, t=round(e["t"] + t0, 3)) for e in tl["events"]]
            spans.append(("puppet", t0, t1, None))
        else:
            who = names.get(f'{c["who"]}_{c["speaker_id"]}' if "speaker_id" in c else c["who"], c["who"])
            pcm = _wav(vs_path.parent / c["audio"])
            captions.append([round(t0, 3), round(t1, 3), S.get("labels", {}).get(who, who.capitalize()), c["text"]])
            spans.append((who, t0, t1, _envelope(pcm)))
    for a in S.get("actions", []):
        actions.append(dict(a, t=a["t"] + off))
    mouth = [0.0] * int(round(off * 60)) + list(vs["mouth"]["values"])
    audio_path = vs_path.parent / vs["audio"]
    return words, mouth, events, captions, spans, actions, audio_path, vs["duration"] + off, emotion0


def _lay_out_lines(S, root, script_path, preroll):
    """MOTION stage script: lines played back to back with gaps; writes the mixed WAV."""
    audio = [np.zeros(int(preroll * SR), np.int16)]
    t = preroll
    words, mouth, events, captions = [], [], [], []
    spans = []  # (who, t0, t1, envelope)
    actions = []
    emotion0 = None
    for line in S["lines"]:
        who = line["who"]
        if who == "puppet":
            tl = json.loads((root / line["timeline"]).read_text())
            pcm = _wav((root / line["timeline"]).parent / tl["audio"])
            emotion0 = emotion0 or tl["emotion"]
            events.append({"t": round(t - 0.3, 3), "type": "emotion", "value": tl["emotion"]})
            events += [dict(e, t=round(e["t"] + t, 3)) for e in tl["events"]]
            for e in line.get("events", []):  # scene-level extra puppet events, at = fraction of the line
                events.append(dict({k: v for k, v in e.items() if k != "at"},
                                   t=round(t + e["at"] * len(pcm) / SR, 3)))
            words += [dict(w, t0=round(w["t0"] + t, 3), t1=round(w["t1"] + t, 3)) for w in tl["words"]]
            mouth += [0.0] * (int(round(t * 60)) - len(mouth))
            mouth += tl["mouth"]["values"]
            spans.append(("puppet", t, t + len(pcm) / SR, None))
        else:
            pcm = _wav(root / line["audio"])
            captions.append([round(t, 3), round(t + len(pcm) / SR, 3), line.get("label", who.capitalize()),
                             line["text"]])
            spans.append((who, t, t + len(pcm) / SR, _envelope(pcm)))
        dur = len(pcm) / SR
        for a in line.get("actions", []):
            at = a["at"] * dur if a["at"] >= 0 else a["at"]  # negative: seconds before the line
            actions.append(dict(a, t=t + at, who=a.get("who", who)))
        audio += [pcm, np.zeros(int(line.get("gap", 0.4) * SR), np.int16)]
        t += dur + line.get("gap", 0.4)
    pcm = np.concatenate(audio)
    audio_path = root / (script_path.stem + ".wav")
    with wave.open(str(audio_path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())
    return words, mouth, events, captions, spans, actions, audio_path, t, emotion0


def _finish(S, people, by_name, tl, spans, actions, fps) -> "Stage":

    # ---- per-person curves on the render clock
    n = int(np.ceil((tl.duration + 0.6) * fps))
    tt = np.arange(n) / fps
    puppet_talking = np.zeros(n)
    for p in people:
        p.speak, p.talking, p.yaw = np.zeros(n), np.zeros(n), np.full(n, p.yaw0)
        p.pos_t = np.tile(p.pos if getattr(p, "start", None) is None else p.start, (n, 1))
    for who, t0, t1, env in spans:
        m = (tt >= t0) & (tt <= t1)
        if who == "puppet":
            puppet_talking[m] = 1
            continue
        p = by_name[who]
        p.talking[(tt >= t0 - 0.25) & (tt <= t1 + 0.3)] = 1
        if env is not None:
            p.speak[m] = np.interp(tt[m] - t0, np.arange(len(env)) / 60, env)
    for p in people:
        p.talking = np.clip(lowpass(p.talking, fps, 1.5), 0, 1)
        p.speak = np.clip(lowpass(p.speak, fps, 8.0), 0, 1)
    for a in sorted(actions, key=lambda a: a["t"]):
        p = by_name[a["who"]]
        m = tt >= max(a["t"], 0)
        u = tt[m] - a["t"]
        if a["do"] in ("turn_away", "turn_back", "turn_to"):
            if a["do"] == "turn_away":
                target = p.yaw0 + np.deg2rad(a.get("deg", 165))
            elif a["do"] == "turn_back":
                target = p.yaw0
            else:
                target = np.deg2rad(a["deg"])
            cur = p.yaw[m][0]
            dur = a.get("dur", 0.9)
            p.yaw[m] = cur + (target - cur) * smoothstep(u / dur)
        elif a["do"] == "move_to":
            # walk from wherever the person is at that moment, facing the direction of travel, then
            # turn to ``deg`` (default: the original facing)
            start = p.pos_t[m][0].copy()
            goal = np.asarray(a["pos"], float)
            dur = a.get("dur", 1.5)
            s_ = smoothstep(u / dur)[:, None]
            p.pos_t[m] = start + (goal - start) * s_
            walk_yaw = np.arctan2(*(goal - start)[::-1])
            end_yaw = np.deg2rad(a["deg"]) if "deg" in a else p.yaw0
            cur = p.yaw[m][0]
            turn_in = smoothstep(u / 0.4)
            turn_out = smoothstep((u - dur + 0.3) / 0.6)
            p.yaw[m] = (cur + (walk_yaw - cur) * turn_in) * (1 - turn_out) + end_yaw * turn_out
        elif a["do"] == "raise_hand":
            if p.hand_up is None:
                p.hand_up = np.zeros(n)
            hold = a.get("hold", 1.2)
            p.hand_up[m] = np.maximum(p.hand_up[m], hermite([(0, 0), (0.35, 1.0), (hold, 1.0), (hold + 0.4, 0)], u))

    st = Stage(floor_z=S.get("floor_z", -0.8), people=people, timeline=tl, audio_path=tl.audio,
               captions=tl.captions, t=tt, fps=fps, puppet_talking=puppet_talking)
    st.shots = _shots(S, st, spans)
    return st


def _shots(S, st: Stage, spans) -> list:
    """Auto edit: establishing wide, then cut to whoever talks (puppet close-up, presenter two-shot,
    judges wide), holding every shot >= 2 s. Script ``shots`` ([[t, camera], ...]) overrides."""
    if isinstance(S.get("shots"), list):  # times on the script's clock
        return [(float(a) + (S.get("_offset", 0.0) if a > 0 else 0.0), b) for a, b in S["shots"]]
    by_name = {p.name: p for p in st.people}
    has_pres = any(p.role == "presenter" for p in st.people)
    shots = [(0.0, "stage_wide")]
    k = 0
    for who, t0, t1, _ in spans:
        if who == "puppet":
            k += 1
            cam = ("kermit_cu" if k % 2 else "two_shot") if has_pres else "kermit_cu"
        elif by_name[who].role == "presenter":
            cam = "two_shot"
        else:
            cam = "qa_wide"
        tc = max(t0 - 0.2, shots[-1][0] + 2.0, 1.8)  # keep the establishing wide short
        if cam != shots[-1][1] and tc < t1:
            shots.append((tc, cam))
    return shots


# -------------------------------------------------------------------------------------------------
# synthetic PERCEPTION v2 (CONTRACT §5) and the gaze bias that follows speaker_id

def perceive(st: Stage, model, data, pose_fn) -> None:
    """Run the scene's people kinematically and log what the base camera would report (§5)."""
    hyst_on, hold = 0.25, 0.4
    cur, cand, cand_t = None, None, 0.0
    faces = np.zeros((len(st.t), 3))
    present = np.zeros(len(st.t), bool)
    for i, t in enumerate(st.t):
        pose_fn(i)
        mujoco.mj_kinematics(model, data)
        ppl = []
        for p in st.people:
            face = data.site_xpos[model.site(f"{p.name}_face").id].copy()
            rel = face - CAMERA_POS
            head_fwd = data.xmat[model.body(f"{p.name}_neck").id].reshape(3, 3)[:, 0]
            to_cam = -rel / np.linalg.norm(rel)
            ang = np.degrees(np.arccos(np.clip(head_fwd @ to_cam, -1, 1)))
            facing = "toward" if ang < 60 else ("side" if ang < 100 else "away")
            sp = 0.05 + 0.9 * p.talking[i] * (0.6 + 0.4 * (facing != "away") + 0.0)
            ppl.append(dict(id=p.id, yaw_deg=round(float(np.degrees(np.arctan2(rel[1], rel[0]))), 2),
                            pitch_deg=round(float(np.degrees(np.arctan2(rel[2], np.hypot(rel[0], rel[1])))), 2),
                            dist_m=round(float(np.linalg.norm(rel)), 3), facing=facing,
                            speaking_p=round(float(min(sp, 0.99)), 3), has_mic=p.mic, role=p.role, conf=0.9,
                            target_base=[round(float(x), 3) for x in face]))
        # active speaker with hysteresis: switch only if clearly louder for `hold` seconds
        best = max(ppl, key=lambda q: q["speaking_p"])
        cur_p = next((q for q in ppl if q["id"] == cur), None)
        if cur is None:
            if best["speaking_p"] > 0.5:
                cur = best["id"]
        elif best["id"] != cur and best["speaking_p"] > (cur_p["speaking_p"] if cur_p else 0) + hyst_on:
            if cand != best["id"]:
                cand, cand_t = best["id"], t
            elif t - cand_t >= hold:
                cur, cand = best["id"], None
        else:
            cand = None
        speech = any(q["speaking_p"] > 0.5 for q in ppl)
        st.perception.append(dict(t=round(float(t), 3), speaker_id=cur, speech_active=speech, people=ppl))
        if cur is not None:
            faces[i] = next(q for q in ppl if q["id"] == cur)["target_base"]
            present[i] = True
    st.speaker_face, st.speaker_present = faces, present


def speaker_bias(st: Stage, head_pos: np.ndarray, weight: np.ndarray):
    """Gaze bias toward ``speaker_id``: eyes/head lead (tau 0.25 s), the body follows (tau 0.6 s)."""
    d = st.speaker_face - head_pos
    yaw = np.where(st.speaker_present, np.clip(np.arctan2(d[:, 1], d[:, 0]), -YAW_LIMIT, YAW_LIMIT), 0.0)
    pitch = np.where(st.speaker_present,
                     np.clip(-np.arctan2(d[:, 2], np.hypot(d[:, 0], d[:, 1])), -PITCH_LIMIT, PITCH_LIMIT), 0.0)
    dt = 1 / st.fps

    def crit(x, tau):
        y, v, out = x[0], 0.0, np.zeros_like(x)
        w = 1 / tau
        for i, g in enumerate(x):
            v += (w * w * (g - y) - 2 * w * v) * dt
            y += v * dt
            out[i] = y
        return out

    base = 0.6 * crit(yaw, 0.6) * weight
    head = crit(yaw, 0.25) * weight - base
    return base, head, crit(pitch, 0.3) * weight
