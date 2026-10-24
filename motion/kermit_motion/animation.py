"""Performance timeline -> puppet animation channels.

Channels are sampled on a fixed clock (``fps``) and live in the head's rest frame:

    yaw, pitch, roll   rad   (+yaw looks left, +pitch nods down, +roll tilts to its right)
    dx, dy, dz         m     (forward, left, up)
    mouth              0..1

The result is a sum of layers: emotion posture + gaze + idle life + talking accents + gesture clips.
Gesture clips are hand-keyed curves (degrees / centimetres, scaled by intensity) with explicit
anticipation, overshoot and settle. They are timed to fit the xArm6's 20 rad/s^2 joint limit, which
means the arm can only add about 1 deg of head motion at syllable rate. Syllable-level lip sync is
the hand's job; the head gets word- and phrase-level accents.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

CHANNELS = ("yaw", "pitch", "roll", "dx", "dy", "dz", "mouth")
ANGLE = {"yaw", "pitch", "roll"}
LENGTH = {"dx", "dy", "dz"}


# ---------------------------------------------------------------------------------------------------
# curves

def hermite(keys: list[tuple[float, float]], t: np.ndarray) -> np.ndarray:
    """Catmull-Rom spline through (time, value) keys with zero slope at both ends; 0 outside."""
    k = np.asarray(keys, float)
    kt, kv = k[:, 0], k[:, 1]
    n = len(kt)
    m = np.zeros(n)
    for i in range(1, n - 1):
        m[i] = (kv[i + 1] - kv[i - 1]) / (kt[i + 1] - kt[i - 1])
        # keep extremes as true extremes (no extra overshoot beyond the keyed value)
        if (kv[i] - kv[i - 1]) * (kv[i + 1] - kv[i]) <= 0:
            m[i] = 0.0
    out = np.zeros_like(t, dtype=float)
    inside = (t >= kt[0]) & (t <= kt[-1])
    ti = t[inside]
    j = np.clip(np.searchsorted(kt, ti, side="right") - 1, 0, n - 2)
    h = kt[j + 1] - kt[j]
    s = (ti - kt[j]) / h
    h00, h10 = 2 * s**3 - 3 * s**2 + 1, s**3 - 2 * s**2 + s
    h01, h11 = -2 * s**3 + 3 * s**2, s**3 - s**2
    out[inside] = h00 * kv[j] + h10 * h * m[j] + h01 * kv[j + 1] + h11 * h * m[j + 1]
    after = t > kt[-1]
    out[after] = kv[-1]
    return out


def smoothstep(x):
    x = np.clip(x, 0, 1)
    return x * x * x * (x * (6 * x - 15) + 10)


def lowpass(x: np.ndarray, fps: float, cutoff: float) -> np.ndarray:
    """Zero-phase 2nd-order critically damped low-pass (forward-backward one-pole pairs)."""
    a = np.exp(-2 * np.pi * cutoff / fps)
    y = x.astype(float).copy()
    for _ in range(2):
        for rng in (range(1, len(y)), range(len(y) - 2, -1, -1)):
            step = 1 if rng.step == 1 else -1
            for i in rng:
                y[i] = a * y[i - step] + (1 - a) * y[i]
    return y


def slew_limit(x: np.ndarray, dt: float, vmax: float) -> np.ndarray:
    """Velocity-limited tracking, averaged forward and backward so it adds no net lag."""
    def one(seq):
        y = np.empty_like(seq)
        y[0] = seq[0]
        dmax = vmax * dt
        for i in range(1, len(seq)):
            y[i] = y[i - 1] + np.clip(seq[i] - y[i - 1], -dmax, dmax)
        return y
    return 0.5 * (one(x) + one(x[::-1])[::-1])


def smooth_noise(t: np.ndarray, seed: int, freqs=(0.13, 0.29, 0.47)) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = np.zeros_like(t)
    for f in freqs:
        out += np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi)) / len(freqs)
    return out


# ---------------------------------------------------------------------------------------------------
# emotion postures (deg / cm), energy scales gesture and talking amplitude

EMOTIONS = {
    "neutral":     dict(pitch=0, energy=1.0, mouth=0.9),
    "happy":       dict(pitch=-3, dz=0.5, energy=1.05, mouth=1.0),
    "excited":     dict(pitch=-5, dz=1.0, dx=0.5, energy=1.2, mouth=1.0),
    "sad":         dict(pitch=12, dz=-2.0, dx=-1.0, roll=-3, energy=0.6, mouth=0.75),
    "surprised":   dict(pitch=-3, dx=-1.5, energy=1.1, mouth=1.0),
    "thinking":    dict(pitch=-5, roll=7, yaw=6, energy=0.8, mouth=0.85),
    "embarrassed": dict(pitch=9, yaw=-12, roll=-5, dz=-1.0, energy=0.8, mouth=0.8),
    "annoyed":     dict(pitch=4, roll=-4, energy=0.9, mouth=0.9),
}

# gaze targets: (yaw, pitch) in degrees relative to facing the speaker
# The camera sits ~25 deg to the puppet's left, so "speaker" and "audience" lean toward it and
# left/right stay within a 3/4 turn of it (both eyes always visible, never full profile).
GAZE = {
    "speaker": (8, 0), "audience": (22, -2), "down": (0, 18), "up": (0, -14),
    "left": (34, 0), "right": (-16, 0), "center": (0, 0), "arm": (-42, 12), "robot_arm": (-42, 12), "laptop": (-18, 16), "camera": (20, -2),
}


def gaze_angles(name: str | None) -> tuple[float, float]:
    if not name:
        return GAZE["speaker"]
    name = name.removeprefix("target:")
    return GAZE.get(name, GAZE["speaker"])


# ---------------------------------------------------------------------------------------------------
# gesture clips: functions of local time u (s) returning additive channel curves (deg / cm / 0..1).
# "mouth" in a clip is a floor: the final mouth is max(speech, clip).

def clip_nod(u, k):
    return {
        # anticipation up, big nod down, rebound past neutral, smaller second nod, settle
        "pitch": hermite([(0, 0), (0.18, -5), (0.56, 15), (0.88, -3), (1.16, 5.5), (1.42, -0.5), (1.65, 0)], u) * k,
        "dz": hermite([(0, 0), (0.18, 0.5), (0.56, -1.2), (0.88, 0.2), (1.16, -0.4), (1.65, 0)], u) * k,
        "dx": hermite([(0, 0), (0.56, 1.0), (1.16, 0.3), (1.65, 0)], u) * k,
    }


def clip_shake(u, k):
    return {
        # small, quick yaw oscillations that decay
        "yaw": hermite([(0, 0), (0.12, -5.5), (0.36, 5.5), (0.6, -4), (0.84, 2.5), (1.08, -1), (1.3, 0)], u) * k,
        "pitch": hermite([(0, 0), (0.3, 2), (1.0, 1.5), (1.3, 0)], u) * k,
    }


def clip_surprise(u, k):
    return {
        # squash forward (anticipation), snap back and up, double-take shake, settle into a held lean
        "pitch": hermite([(0, 0), (0.12, 4), (0.46, -13), (0.62, -15), (1.1, -11), (1.6, -6), (2.4, 0)], u) * k,
        "dx": hermite([(0, 0), (0.12, 0.7), (0.46, -5.0), (0.64, -5.5), (1.6, -2.5), (2.4, 0)], u) * k,
        "dz": hermite([(0, 0), (0.12, -0.7), (0.46, 1.8), (0.64, 2.0), (1.6, 0.7), (2.4, 0)], u) * k,
        "yaw": hermite([(0, 0), (0.6, 0), (0.84, 4), (1.1, -3.5), (1.36, 2), (1.62, 0)], u) * k,
        "roll": hermite([(0, 0), (0.5, -4), (1.0, -3), (1.8, 0)], u) * k,
        "mouth": hermite([(0, 0), (0.12, 0), (0.3, 1.0), (0.9, 0.95), (1.5, 0)], u) * min(1.0, 0.6 + 0.4 * k),
    }


def clip_head_bob(u, k):
    return {
        "dz": hermite([(0, 0), (0.1, -0.6), (0.3, 1.6), (0.52, -0.4), (0.72, 0.8), (1.0, 0)], u) * k,
        "pitch": hermite([(0, 0), (0.1, 2), (0.3, -6), (0.52, 1.5), (0.72, -2.5), (1.0, 0)], u) * k,
        "roll": hermite([(0, 0), (0.3, 3), (0.72, -1.5), (1.0, 0)], u) * k,
    }


def clip_tilt(u, k, side):
    return {
        "roll": hermite([(0, 0), (0.12, -2 * side), (0.5, 14 * side), (0.68, 12 * side), (1.6, 11 * side),
                         (2.1, 0)], u) * k,
        "pitch": hermite([(0, 0), (0.5, -3), (1.6, -2), (2.1, 0)], u) * k,
    }


def clip_look_around(u, k):
    return {
        "yaw": hermite([(0, 0), (0.6, 24), (1.2, 22), (2.0, -24), (2.6, -22), (3.2, 0)], u) * k,
        "pitch": hermite([(0, 0), (0.6, -4), (1.3, 2), (2.0, -4), (3.2, 0)], u) * k,
        "roll": hermite([(0, 0), (0.6, 4), (2.0, -4), (3.2, 0)], u) * k,
    }


def clip_laugh(u, k):
    bob = np.where(u < 1.4, np.sin(2 * np.pi * 2.2 * u) * smoothstep(u / 0.2) * smoothstep((1.4 - u) / 0.3), 0)
    return {
        "dz": bob * 1.0 * k,
        "pitch": hermite([(0, 0), (0.2, -10), (1.2, -8), (1.7, 0)], u) * k + bob * 1.5 * k,
        "mouth": np.where(u < 1.4, 0.55 + 0.45 * np.abs(np.sin(2 * np.pi * 2.2 * u)), 0) * smoothstep(u / 0.1)
        * smoothstep((1.5 - u) / 0.2) * k,
    }


def clip_sigh(u, k):
    return {
        "pitch": hermite([(0, 0), (0.6, -5), (1.6, 16), (2.6, 12), (3.4, 0)], u) * k,
        "dz": hermite([(0, 0), (0.6, 1.0), (1.6, -2.5), (2.6, -2.0), (3.4, 0)], u) * k,
        "mouth": hermite([(0, 0), (0.6, 0.25), (1.6, 0.1), (2.0, 0)], u) * k,
    }


def clip_shrug(u, k):
    return {
        "dz": hermite([(0, 0), (0.1, -0.5), (0.4, 2.2), (0.9, 2.0), (1.3, 0)], u) * k,
        "roll": hermite([(0, 0), (0.4, 8), (0.9, 7), (1.3, 0)], u) * k,
        "pitch": hermite([(0, 0), (0.4, -4), (0.9, -3), (1.3, 0)], u) * k,
    }


def clip_lean(u, k, sign):
    return {
        "dx": hermite([(0, 0), (0.15, -0.6 * sign), (0.6, 4.5 * sign), (0.8, 4.0 * sign), (2.0, 3.8 * sign),
                       (2.6, 0)], u) * k,
        "pitch": hermite([(0, 0), (0.6, 5 * sign), (2.0, 4 * sign), (2.6, 0)], u) * k,
    }


def clip_flail(u, k):
    # Kermit-style excitement: bouncy wobble plus fast mouth flaps (the hand has no arms to wave)
    w = smoothstep(u / 0.25) * smoothstep((1.6 - u) / 0.35) * (u < 1.6)
    f = 1.6
    return {
        "yaw": np.sin(2 * np.pi * f * u) * 3.5 * w * k,
        "roll": np.sin(2 * np.pi * f * u + 1.3) * 4.0 * w * k,
        "dz": np.sin(np.pi * f * u) ** 2 * 1.4 * w * k,
        "pitch": -3.0 * w * k,
        "mouth": (0.5 + 0.5 * np.sin(2 * np.pi * 3.0 * u)) * w * k,
    }


def clip_cover_face(u, k):
    return {
        "pitch": hermite([(0, 0), (0.15, -3), (0.6, 24), (2.0, 22), (2.6, 0)], u) * k,
        "yaw": hermite([(0, 0), (0.6, -22), (2.0, -20), (2.6, 0)], u) * k,
        "dz": hermite([(0, 0), (0.6, -2), (2.0, -2), (2.6, 0)], u) * k,
    }


CLIPS = {
    "nod": clip_nod, "shake": clip_shake, "surprise": clip_surprise, "whoa": clip_surprise,
    "head_bob": clip_head_bob, "tilt_left": lambda u, k: clip_tilt(u, k, -1),
    "tilt_right": lambda u, k: clip_tilt(u, k, 1), "look_around": clip_look_around, "laugh": clip_laugh,
    "sigh": clip_sigh, "shrug": clip_shrug, "lean_in": lambda u, k: clip_lean(u, k, 1),
    "lean_back": lambda u, k: clip_lean(u, k, -1), "flail": clip_flail, "cover_face": clip_cover_face,
}
CLIP_LEN = {"nod": 1.65, "shake": 1.3, "surprise": 2.4, "whoa": 2.4, "head_bob": 1.0, "tilt_left": 2.1,
            "tilt_right": 2.1, "look_around": 3.2, "laugh": 1.7, "sigh": 3.4, "shrug": 1.3, "lean_in": 2.6,
            "lean_back": 2.6, "flail": 1.6, "cover_face": 2.6}
# events that start this much before their timestamp so the accent lands on the word (anticipation)
CLIP_LEAD = {"nod": 0.2, "surprise": 0.12, "whoa": 0.12, "head_bob": 0.1, "shake": 0.1}


def sentence_starts(words: list[dict], gap: float = 0.3) -> list[float]:
    """Onsets of sentences / phrases after a pause: where a puppeteer would take a breath."""
    starts = []
    prev_end, prev_word = -1e9, "."
    for w in words:
        if w["t0"] - prev_end > gap or prev_word.rstrip('"\')').endswith((".", "!", "?")):
            starts.append(float(w["t0"]))
        prev_end, prev_word = w["t1"], w["w"]
    return starts


def gaze_step(u, d_yaw, d_pitch):
    """One head turn as an additive step (0 before u = 0): anticipation, ease, overshoot, settle, with an
    arc (nose dips through the turn) and a lead tilt toward the new direction, part of which is kept.
    Steps sum, so a new look_at that starts before the last one settled blends in continuously."""
    span = max(abs(d_yaw), abs(d_pitch))
    dur = 0.6 + 0.016 * span  # bigger turns take longer (the wrist tops out at 180 deg/s)
    s = hermite([(0, 0), (0.18 * dur, -0.05), (0.64 * dur, 1.06), (0.82 * dur, 0.985), (dur, 1.0)], u)
    arc = np.sin(np.pi * np.clip(u / (0.7 * dur), 0, 1)) ** 2
    yaw = d_yaw * s
    pitch = d_pitch * s + 0.08 * span * arc
    roll = 0.12 * d_yaw * arc + 0.05 * d_yaw * smoothstep(u / dur)
    return yaw, pitch, roll


# ---------------------------------------------------------------------------------------------------

@dataclass
class Timeline:
    duration: float
    audio: Path | None
    words: list[dict]
    mouth: np.ndarray
    mouth_hz: float
    events: list[dict]
    emotion: str = "neutral"
    gaze: str = "speaker"
    captions: list = field(default_factory=list)  # [t0, t1, who, text] lines without word timings

    @classmethod
    def load(cls, path: str | Path) -> "Timeline":
        path = Path(path)
        j = json.loads(path.read_text())
        audio = (path.parent / j["audio"]) if j.get("audio") else None
        mouth = np.asarray(j.get("mouth", {}).get("values", []), float)
        hz = float(j.get("mouth", {}).get("hz", 60))
        dur = float(j.get("duration") or 0)
        if audio and audio.exists() and not dur:
            import wave
            with wave.open(str(audio)) as w:
                dur = w.getnframes() / w.getframerate()
        dur = max(dur, len(mouth) / hz, max([e["t"] for e in j.get("events", [])] + [0]) + 1.0)
        return cls(duration=dur, audio=audio, words=j.get("words", []), mouth=mouth, mouth_hz=hz,
                   events=sorted(j.get("events", []), key=lambda e: e["t"]), emotion=j.get("emotion", "neutral"),
                   gaze=j.get("gaze", "speaker"), captions=j.get("captions", []))


    def shifted(self, dt: float) -> "Timeline":
        """Delay everything by ``dt`` seconds (pre-roll for the breath / anticipation before t = 0)."""
        pad = np.zeros(int(round(dt * self.mouth_hz)))
        return Timeline(duration=self.duration + dt, audio=self.audio,
                        words=[dict(w, t0=w["t0"] + dt, t1=w["t1"] + dt) for w in self.words],
                        mouth=np.r_[pad, self.mouth], mouth_hz=self.mouth_hz,
                        events=[dict(e, t=e["t"] + dt) for e in self.events], emotion=self.emotion, gaze=self.gaze,
                        captions=[[c[0] + dt, c[1] + dt, *c[2:]] for c in self.captions])


@dataclass
class Animation:
    t: np.ndarray
    ch: dict[str, np.ndarray]           # head channels (radians / metres / 0..1)
    labels: list[tuple[float, float, str]] = field(default_factory=list)  # (t0, t1, gesture label)


def animate(tl: Timeline, fps: float = 60.0, tail: float = 0.6, lip_lead: float = 0.03, seed: int = 7,
            follow=None, t_start: float = 0.0, mouth_t0: float = 0.0) -> Animation:
    """``follow``: optional callable (t) -> (base_yaw, head_yaw, head_pitch) person-tracking bias (rad).
    ``t_start`` / ``mouth_t0``: animate only [t_start, duration + tail] on the absolute clock, with the mouth
    curve starting at ``mouth_t0`` (live mode animates a sliding window this way)."""
    n = int(np.ceil((tl.duration + tail - t_start) * fps))
    t = t_start + np.arange(n) / fps
    deg = {c: np.zeros(n) for c in CHANNELS}  # angles in deg, lengths in cm while composing
    # 1. emotion posture; optional {"type": "emotion", "value": ...} events crossfade to a new one
    emo_t = [(-1e9, tl.emotion)] + [(e["t"], e["value"]) for e in tl.events if e["type"] == "emotion"]
    emo_curves = {c: np.zeros(n) for c in ("yaw", "pitch", "roll", "dx", "dy", "dz", "energy", "mouth")}
    for i, (t0, name) in enumerate(emo_t):
        p = EMOTIONS.get(name, EMOTIONS["neutral"])
        w = smoothstep((t - t0) / 0.6) if i else np.ones(n)
        for c in emo_curves:
            emo_curves[c] = (1 - w) * emo_curves[c] + w * p.get(c, 1.0 if c in ("energy", "mouth") else 0.0)
    energy = emo_curves["energy"]
    for c in ("yaw", "pitch", "roll", "dx", "dy", "dz"):
        deg[c] += emo_curves[c]
    emo = {"mouth": emo_curves["mouth"]}

    # 2. gaze: initial target plus look_at / gaze events as eased, additive turns
    g = gaze_angles(tl.gaze)
    yaw_g, pitch_g, roll_g = np.full(n, g[0], float), np.full(n, g[1], float), np.zeros(n)
    cur = g
    labels = []
    for e in tl.events:
        if e["type"] not in ("look_at", "gaze"):
            continue
        to = gaze_angles(e.get("target", "speaker"))
        k = max(float(e.get("intensity", 1.0)), 0.6)
        to = (cur[0] + (to[0] - cur[0]) * k, cur[1] + (to[1] - cur[1]) * k)
        t0 = e["t"] - 0.1
        m = t >= t0
        y, p, r = gaze_step(t[m] - t0, to[0] - cur[0], to[1] - cur[1])
        yaw_g[m] += y
        pitch_g[m] += p
        roll_g[m] += r
        labels.append((t0, t0 + 0.9, f"look_at {e.get('target', '')}".strip()))
        cur = to
    deg["yaw"] += yaw_g
    deg["pitch"] += pitch_g
    deg["roll"] += roll_g

    # 3. idle life: slow breathing and drift, never fully still
    # never dead: ~1-2 deg drift and sway at 0.3-0.5 Hz plus a slow breathing rise
    idle_f = (0.31, 0.43, 0.52)
    deg["dz"] += 0.35 * np.sin(2 * np.pi * 0.25 * t)
    deg["yaw"] += 2.2 * energy * smooth_noise(t, seed, idle_f)
    deg["roll"] += 1.8 * energy * smooth_noise(t, seed + 1, idle_f)
    deg["pitch"] += 1.4 * smooth_noise(t, seed + 2, idle_f)

    # 4. speech: mouth from VOICE (led slightly), head accents from the low-passed envelope
    mouth = np.zeros(n)
    if len(tl.mouth):
        src_t = mouth_t0 + np.arange(len(tl.mouth)) / tl.mouth_hz
        mouth = np.interp(t + lip_lead, src_t, tl.mouth, left=0, right=0)
    mouth *= emo.get("mouth", 1.0)
    # Puppetry rule: the jaw does the syllables, the head only phrases. No per-word head bob, just a
    # slow lift and lean into the line while the puppet is talking.
    phrase_env = lowpass(mouth, fps, 0.35)     # is the puppet talking right now?
    deg["pitch"] += -2.0 * energy * phrase_env
    deg["dx"] += 0.6 * phrase_env
    deg["yaw"] += 4.0 * energy * phrase_env * smooth_noise(t, seed + 3, freqs=(0.37, 0.61))
    deg["roll"] += 3.0 * energy * phrase_env * smooth_noise(t, seed + 4, freqs=(0.31, 0.53))

    # 5. gesture clips (plus a breath before every sentence)
    mouth_floor = np.zeros(n)
    for t0 in sentence_starts(tl.words):
        m = (t >= t0 - 0.5) & (t <= t0 + 0.4)
        u = t[m] - t0
        deg["pitch"][m] += hermite([(-0.5, 0), (-0.15, -3.0), (0.4, 0)], u)
        deg["dz"][m] += hermite([(-0.5, 0), (-0.15, 0.6), (0.4, 0)], u)
        mouth_floor[m] = np.maximum(mouth_floor[m], hermite([(-0.45, 0), (-0.15, 0.2), (0.0, 0.12), (0.12, 0)], u))
    for e in tl.events:
        typ = e["type"]
        if typ not in CLIPS:
            continue
        t0 = e["t"] - CLIP_LEAD.get(typ, 0.0)
        k = float(e.get("intensity", 1.0)) * (0.75 + 0.25 * float(np.interp(e["t"], t, energy)))
        m = (t >= t0) & (t <= t0 + CLIP_LEN[typ] + 0.05)
        curves = CLIPS[typ](t[m] - t0, k)
        for c, v in curves.items():
            if c == "mouth":
                mouth_floor[m] = np.maximum(mouth_floor[m], v)
            else:
                deg[c][m] += v
        labels.append((t0, t0 + CLIP_LEN[typ], typ))

    ch = {}
    for c in CHANNELS:
        if c in ANGLE:
            ch[c] = np.deg2rad(deg[c])
        elif c in LENGTH:
            ch[c] = deg[c] / 100.0
    ch["mouth"] = np.clip(np.maximum(mouth, mouth_floor), 0, 1)
    ch["base_yaw"] = np.zeros(n)
    if follow is not None:
        base_yaw, head_yaw, head_pitch = follow(t)
        ch["base_yaw"] += base_yaw
        ch["yaw"] += head_yaw
        ch["pitch"] += head_pitch
    return Animation(t=t, ch=ch, labels=sorted(labels))
