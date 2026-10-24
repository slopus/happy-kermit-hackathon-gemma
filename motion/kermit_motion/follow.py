"""Person-following gaze bias (CONTRACT section 4).

PERCEPTION streams the main person's face in the arm base frame. MOTION turns it into a *bias* on the
whole pose: joint1 turns the puppet's body toward the person and the head aims the rest of the way.
Gestures and look_at play on top, relative to that direction.

    bias  = clamp(direction from the puppet's head to the face, +-60 deg yaw, +-20 deg pitch)
    home  = critically damped follow of the bias (tau ~0.4 s); no face -> eases back to 0 over ~1 s
    used  = home * weight(gaze)   speaker 1.0 | audience 0.4 | explicit other target 0 (override)
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

YAW_LIMIT = np.deg2rad(60)
PITCH_LIMIT = np.deg2rad(20)
TAU_FOLLOW = 0.4    # s, while the person is visible
TAU_LOST = 0.3      # s, critically damped: ~1 s to settle back to neutral
BASE_SHARE = 0.6    # fraction of the yaw bias done by joint1 (whole puppet turns), rest by the head
WEIGHT = {"speaker": 1.0, "audience": 0.4}
CAMERA_POS = np.array([0.05, 0.0, 0.10])  # perception camera in the arm base frame (CONTRACT default)


@dataclass
class Track:
    t: np.ndarray            # (N,) s, on the render clock
    present: np.ndarray      # (N,) bool, what the tracker reports
    face: np.ndarray         # (N, 3) face point in base frame, interpolated through dropouts (for the stand-in)


def load_track(path: str | Path, t: np.ndarray, t_offset: float = 0.0) -> Track:
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    rt = np.array([r["t"] for r in rows], float)
    rt -= rt[0]
    rt += t_offset
    present = np.array([bool(r.get("present", False)) for r in rows])
    face = np.full((len(rows), 3), np.nan)
    for i, r in enumerate(rows):
        if r.get("target_base") is not None:
            face[i] = r["target_base"]
        elif r.get("yaw_deg") is not None:
            y, p, dist = np.deg2rad(r["yaw_deg"]), np.deg2rad(r.get("pitch_deg", 0)), r.get("dist_m", 1.5)
            face[i] = CAMERA_POS + dist * np.array([np.cos(p) * np.cos(y), np.cos(p) * np.sin(y), np.sin(p)])
    ok = ~np.isnan(face[:, 0])
    out = np.stack([np.interp(t, rt[ok], face[ok, k]) for k in range(3)], axis=1)
    pres = np.interp(t, rt, present.astype(float), left=0, right=0) > 0.5
    return Track(t=t, present=pres, face=out)


def _critical(target: np.ndarray, present: np.ndarray, dt: float) -> np.ndarray:
    """Critically damped 2nd-order follower (no overshoot), time constant switches when the face is lost."""
    x, v = 0.0, 0.0
    out = np.zeros_like(target)
    for i, (g, p) in enumerate(zip(target, present)):
        tau = TAU_FOLLOW if p else TAU_LOST
        w = 1.0 / tau
        a = w * w * (g - x) - 2 * w * v
        v += a * dt
        x += v * dt
        out[i] = x
    return out


def gaze_weight(events: list[dict], gaze: str, t: np.ndarray, fps: float) -> np.ndarray:
    """How much the person bias applies over time, from the turn's gaze and look_at events."""
    def w_of(target):
        target = (target or "speaker").removeprefix("target:")
        return WEIGHT.get(target, 0.0)
    w = np.full(len(t), w_of(gaze))
    for e in events:
        if e["type"] in ("look_at", "gaze"):
            w[t >= e["t"] - 0.1] = w_of(e.get("target"))
    from .animation import lowpass
    return np.clip(lowpass(w, fps, 1.0), 0, 1)


def bias(track: Track, head_pos: np.ndarray, weight: np.ndarray, fps: float):
    """Returns (base_yaw, head_yaw, head_pitch) in rad for the animation layer."""
    d = track.face - head_pos
    yaw = np.clip(np.arctan2(d[:, 1], d[:, 0]), -YAW_LIMIT, YAW_LIMIT)
    pitch = np.clip(-np.arctan2(d[:, 2], np.hypot(d[:, 0], d[:, 1])), -PITCH_LIMIT, PITCH_LIMIT)  # + = down
    yaw = np.where(track.present, yaw, 0.0)
    pitch = np.where(track.present, pitch, 0.0)
    dt = 1.0 / fps
    yaw = _critical(yaw, track.present, dt) * weight
    pitch = _critical(pitch, track.present, dt) * weight
    return BASE_SHARE * yaw, (1 - BASE_SHARE) * yaw, pitch
