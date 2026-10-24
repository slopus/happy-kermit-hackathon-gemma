"""Real-time puppet driver for the LIVE ROOM SIM (CONTRACT §6).

    from kermit_motion.live import LiveKermit
    k = LiveKermit(people=[("kirill", "presenter", True, None), ...], floor_z=-0.8)
    k.feed(event)                       # run_kermit stdout events: segment / play / stop / done / header
    k.set_perception(latest_json)       # CONTRACT §5 latest value (speaker_id + people bearings)
    k.set_person("kirill", root_xyz, yaw, joints={...}, mouth=0.4)   # script driver poses people
    k.step()                            # advance to wall clock (or k.step(dt) for a fixed step)
    rgb = k.render_base_camera()        # what the arm-base camera sees -> PERCEPTION
    k.save_log("run.npz")               # exact replay for the offline video (render_log.py)

Each step animates a sliding window of the scheduled speech (same gesture / lip-sync code as the
offline renderer, on the absolute clock), adds the speaker-following bias causally, solves the head
IK, and passes the joints through a causal velocity / acceleration limiter at 95 % of the xArm6
limits before posing the robot. The costume body is simulated in step with the wall clock.
"""
from __future__ import annotations

import json
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

from . import human as H
from . import scene as scene_mod
from . import xarm6
from .animation import GAZE, Timeline, animate
from .follow import CAMERA_POS, PITCH_LIMIT, YAW_LIMIT, WEIGHT
from .ik import SEED, HeadIK, euler_head
from .sim import PuppetSim

WINDOW_PAST = 4.0     # s of history animated each step (longest gesture clip is 3.4 s)
WINDOW_FUTURE = 1.0   # s of already-scheduled future (breath / anticipation lead into it)
HZ = 60
PERSON_HOLD = 10.0    # s a look_at person_<id> stays locked (unless a newer gaze event or a new speaker)
TRACK_LOST = 1.5      # s without that track in perception = gone: fall back to the speaker


def _person_id(name) -> str | None:
    """'person_3' / 'target:person_3' -> '3' (PERCEPTION's stable people[].id); anything else -> None."""
    name = str(name or "").removeprefix("target:")
    return name.removeprefix("person_") if name.startswith("person_") and len(name) > 7 else None


@dataclass
class _Segment:
    turn: str
    index: int
    kind: str
    start: float                 # scene time the segment's audio starts
    t_offset: float              # segment start within its turn
    mouth: np.ndarray            # 60 Hz, relative to the segment start
    words: list
    events: list                 # times relative to the turn start (as VOICE prints them)
    emotion: str | None = None
    gaze: str | None = None
    cut: float = np.inf          # scene time playback was stopped


@dataclass
class _Follower:
    """Critically damped 2nd-order follower (causal), as in follow.py."""
    tau: float
    x: float = 0.0
    v: float = 0.0

    def __call__(self, goal: float, dt: float) -> float:
        w = 1.0 / self.tau
        self.v += (w * w * (goal - self.x) - 2 * w * self.v) * dt
        self.x += self.v * dt
        return self.x


class LiveKermit:
    def __init__(self, hand: str = "shadow", people: list[tuple] = (), floor_z: float = -0.8, fps: float = 30.0,
                 base_cam_size=(640, 480), view_size=(1280, 720), segment_dir: str | Path | None = None,
                 clock=time.perf_counter, ptz_mount=None):
        self.sc = scene_mod.build(hand, width=view_size[0], height=view_size[1], people=list(people), floor_z=floor_z,
                                  ptz_mount=ptz_mount)
        self.m = self.sc.model
        self.sim = PuppetSim(self.sc)
        self.d = self.sim.d
        self.hand = self.sc.hand
        self.ik = HeadIK(self.sc)
        self.fps = fps
        self.clock = clock
        self.segment_dir = Path(segment_dir) if segment_dir else None
        self.people = {name: H.bind(self.m, name) for name, *_ in people}
        self.people_state = {name: None for name in self.people}

        q, _ = self.ik.solve(scene_mod.HEAD_REST_POS, euler_head(0, 0, 0), SEED, iters=200)
        self.q, self.qd = q.copy(), np.zeros(6)
        self.hq = self.hand.mouth_qpos(np.array([0.0]))[0]
        self.sim.reset(self.q, self.hq, settle=1.0)
        self.t = 0.0
        self.t_wall0 = None
        self.segments: dict[tuple, _Segment] = {}
        self.pending: dict[tuple, dict] = {}
        self.turn_meta: dict[str, dict] = {}
        self.emotion, self.gaze = "neutral", "speaker"
        self.target = None           # (yaw, pitch) of the gaze target in the base frame, rad
        self.tracks: dict[str, tuple] = {}   # str(people[].id) -> (yaw, pitch, t seen), for person_<id>
        self.speaker_id, self.speaker_t = None, -np.inf
        self.person_lock = None      # (id, t set, speaker_id then) after a look_at person_<id>
        self._last_gaze_ev = None
        self.yaw_base, self.yaw_head, self.pitch = _Follower(0.45), _Follower(0.25), _Follower(0.3)
        self.events_log: list = []
        self.frames: dict[str, list] = {k: [] for k in ("t", "qpos", "mocap_pos", "mocap_quat", "geom_size")}
        self._r_base = mujoco.Renderer(self.m, base_cam_size[1], base_cam_size[0])
        self._r_view = mujoco.Renderer(self.m, view_size[1], view_size[0])
        self._mouth_geoms = [pm.mouth_geom for pm in self.people.values()]

    # ---- inputs -----------------------------------------------------------------------------------
    def feed(self, ev: dict):
        """One run_kermit stdout event (segment / play / stop / done / header / perception / heard)."""
        self.events_log.append(dict(ev, _t=self.t))
        kind = ev.get("event")
        if kind == "header":
            self.emotion, self.gaze = ev.get("emotion", self.emotion), ev.get("gaze", self.gaze)
        elif kind == "segment":
            self.pending[(ev["turn"], ev["index"])] = ev
        elif kind == "play":
            seg = self.pending.pop((ev["turn"], ev["index"]), None)
            if seg is None:
                return
            # anchor mouth[0] to when the sentence's sound starts (t_audio; older streams: t_wall)
            t_anchor = ev.get("t_audio", ev.get("t_wall"))
            lag = self.clock() - t_anchor if t_anchor is not None else 0.0
            self._schedule(seg, self.t - float(np.clip(lag, -1.0, 0.5)))
        elif kind == "stop":
            for (turn, _), s in self.segments.items():
                if turn == ev["turn"]:
                    s.cut = min(s.cut, self.t)
        elif kind == "perception":
            self.set_perception(ev)

    def _schedule(self, seg: dict, start: float):
        mouth = seg.get("mouth", {}).get("values") if isinstance(seg.get("mouth"), dict) else seg.get("mouth")
        if mouth is None:
            audio = seg.get("audio")
            if audio and seg.get("out_dir"):
                audio = str(Path(seg["out_dir"]) / audio)
            mouth = self._mouth_from_wav(audio)
        self.segments[(seg["turn"], seg["index"])] = _Segment(
            turn=seg["turn"], index=seg["index"], kind=seg.get("kind", "reply"), start=start,
            t_offset=float(seg.get("t_offset", 0.0)), mouth=np.asarray(mouth, float), words=seg.get("words", []),
            events=seg.get("events", []), emotion=seg.get("emotion"), gaze=seg.get("gaze"))

    def _mouth_from_wav(self, audio: str | None) -> np.ndarray:
        if not audio:
            return np.zeros(1)
        p = Path(audio)
        if not p.is_absolute() and self.segment_dir:
            p = self.segment_dir / p
        try:
            with wave.open(str(p)) as w:
                sr = w.getframerate()
                x = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(float) / 32768
        except (FileNotFoundError, wave.Error):
            return np.zeros(1)
        hop = sr // HZ
        n = len(x) // hop
        rms = np.sqrt(np.mean((x[: n * hop] ** 2).reshape(n, hop), axis=1))
        return np.clip((rms / (np.percentile(rms, 95) + 1e-6) - 0.08) / 0.8, 0, 1)

    def set_perception(self, latest: dict):
        """CONTRACT §5 latest value: aim at ``speaker_id`` (or the §4 single-person fields). Every track's
        bearing is kept too, for look_at ``person_<id>``."""
        sid = latest.get("speaker_id")
        if sid is not None and sid != self.speaker_id:
            self.speaker_t = self.t
        if sid is not None or "people" in latest:
            self.speaker_id = sid
        person = None
        for p in latest.get("people", []):
            if "yaw_deg" in p or "target_base" in p:
                self.tracks[str(p.get("id"))] = (*self._bearing(p), self.t)
            if p.get("id") == sid:
                person = p
        if person is None and "yaw_deg" in latest and latest.get("present", True):
            person = latest
        self.target = None if person is None else self._bearing(person)

    @staticmethod
    def _bearing(person: dict) -> tuple[float, float]:
        if "target_base" in person:
            d = np.asarray(person["target_base"], float) - scene_mod.HEAD_REST_POS
            return np.arctan2(d[1], d[0]), -np.arctan2(d[2], np.hypot(d[0], d[1]))
        return np.radians(person["yaw_deg"]), -np.radians(person.get("pitch_deg", 0.0))

    def _person_target(self, t: float, events: list, gaze: str | None):
        """look_at / gaze ``person_<id>``: that track's live bearing, or the speaker's if the id is gone.
        A look_at locks on until a newer gaze event, a different speaker, or PERSON_HOLD seconds; a
        segment-level ``target:person_<id>`` gaze holds while that segment's gaze applies.
        Returns (yaw, pitch) or None when no person target is active."""
        past = [e for e in events if e["type"] in ("look_at", "gaze") and e["t"] <= t]
        if past and (past[-1]["t"], past[-1].get("target")) != self._last_gaze_ev:
            e = past[-1]
            self._last_gaze_ev = (e["t"], e.get("target"))
            pid = _person_id(e.get("target"))
            self.person_lock = (pid, e["t"], self.speaker_id) if pid else None
        pid = None
        if self.person_lock:
            lock_pid, t_set, sid0 = self.person_lock
            if t - t_set > PERSON_HOLD or (self.speaker_id != sid0 and self.speaker_t > t_set):
                self.person_lock = None
            else:
                pid = lock_pid
        pid = pid or _person_id(gaze)
        if pid is None:
            return None
        tr = self.tracks.get(pid)
        if tr is not None and self.t - tr[2] <= TRACK_LOST:
            return tr[:2]
        return self.target if self.target is not None else (0.0, 0.0)

    def set_gaze_target(self, yaw_deg: float | None, pitch_deg: float = 0.0):
        self.target = None if yaw_deg is None else (np.radians(yaw_deg), -np.radians(pitch_deg))

    def set_person(self, name: str, root_pos, yaw: float, joints: dict | None = None, mouth: float = 0.0):
        self.people_state[name] = (np.asarray(root_pos, float), float(yaw), dict(joints or {}), float(mouth))

    # ---- stepping ---------------------------------------------------------------------------------
    def step(self, dt: float | None = None):
        if dt is None:
            now = self.clock()
            if self.t_wall0 is None:
                self.t_wall0 = now - self.t
            dt = max(1e-3, (now - self.t_wall0) - self.t)
        t1 = self.t + dt
        ch = self._channels(t1, dt)
        R = euler_head(ch["yaw"], ch["pitch"], ch["roll"])
        neck = scene_mod.HEAD_REST_POS - np.array([scene_mod.NECK_BACK, 0, 0])
        pos = neck + R @ np.array([scene_mod.NECK_BACK, 0, 0]) + np.array([ch["dx"], ch["dy"], ch["dz"]])
        Rb = euler_head(ch["base_yaw"], 0, 0)
        q_goal, _ = self.ik.solve(Rb @ pos, Rb @ R, self.q, iters=30)
        q_new, qd_new = self._limit(q_goal, dt)
        h_goal = self.hand.mouth_qpos(np.array([ch["mouth"]]))[0]
        step = self.hand.vel_max * 0.97 * dt
        h_new = np.where(step > 0, self.hq + np.clip(h_goal - self.hq, -step, step), h_goal)
        self._pose_people()
        self.sim.advance(self.q, q_new, self.hq, h_new, dt)
        self._pose_people()
        mujoco.mj_forward(self.m, self.d)
        self.q, self.qd, self.hq, self.t = q_new, qd_new, h_new, t1
        self._log()

    def _limit(self, q_goal, dt):
        """Causal per-joint limiter: velocity <= 95 % and acceleration <= 95 % of the xArm6 maxima."""
        vmax, amax = 0.95 * xarm6.JOINT_VEL_MAX, 0.95 * xarm6.JOINT_ACC_MAX
        v_goal = (q_goal - self.q) / dt
        # brake in time: never faster than what can still stop at the goal
        v_stop = np.sqrt(2 * amax * np.abs(q_goal - self.q)) * np.sign(q_goal - self.q)
        v_goal = np.where(np.abs(v_goal) > np.abs(v_stop), v_stop, v_goal)
        v = np.clip(np.clip(v_goal, self.qd - amax * dt, self.qd + amax * dt), -vmax, vmax)
        q = np.clip(self.q + v * dt, xarm6.JOINT_RANGE[:, 0], xarm6.JOINT_RANGE[:, 1])
        return q, v

    def _channels(self, t: float, dt: float) -> dict:
        lo, hi = t - WINDOW_PAST, t + WINDOW_FUTURE
        words, events, emotion, gaze = [], [], self.emotion, self.gaze
        n = int(np.ceil((hi - lo) * HZ)) + 1
        mouth = np.zeros(n)
        for s in sorted(self.segments.values(), key=lambda s: s.start):
            seg_end = min(s.start + len(s.mouth) / HZ, s.cut)
            turn0 = s.start - s.t_offset  # VOICE prints words/events relative to the turn start
            if s.start <= t and s.emotion:
                emotion = s.emotion
            if s.start <= t and s.gaze:
                gaze = s.gaze
            if seg_end < lo or s.start > hi:
                continue
            i0 = int(round((s.start - lo) * HZ))
            vals = s.mouth[: max(0, int((seg_end - s.start) * HZ))]
            a, b = max(0, i0), min(n, i0 + len(vals))
            if b > a:
                mouth[a:b] = np.maximum(mouth[a:b], vals[a - i0:b - i0])
            words += [dict(w, t0=w["t0"] + turn0, t1=w["t1"] + turn0) for w in s.words if w["t0"] + turn0 < s.cut]
            events += [dict(e, t=e["t"] + turn0) for e in s.events if e["t"] + turn0 < s.cut]
        events = [e for e in events if lo <= e["t"] <= hi]
        tl = Timeline(duration=hi, audio=None, words=words, mouth=mouth, mouth_hz=HZ,
                      events=sorted(events, key=lambda e: e["t"]), emotion=emotion, gaze=gaze)
        an = animate(tl, fps=HZ, tail=0.0, lip_lead=self.hand.lip_lead_s, t_start=lo, mouth_t0=lo)
        k = min(len(an.t) - 1, max(0, int(round((t - lo) * HZ))))
        ch = {c: float(v[k]) for c, v in an.ch.items()}
        # speaker-following bias (causal); explicit look_at targets other than speaker/audience override
        w = WEIGHT.get((gaze or "speaker").removeprefix("target:"), 0.0 if gaze in GAZE else 1.0)
        # explicit look_at targets override the speaker bias for ~3 s, then speaker-follow resumes
        recent = [e for e in events if e["type"] in ("look_at", "gaze") and t - 3.0 <= e["t"] <= t]
        if recent:
            w = WEIGHT.get(recent[-1].get("target", "speaker"), 0.0)
        target = self.target
        person = self._person_target(t, events, gaze)
        if person is not None:  # BRAIN's look_at person_<id>: full weight on that track (or the speaker)
            target, w = person, 1.0
        if target is not None:
            yaw_t = float(np.clip(target[0], -YAW_LIMIT, YAW_LIMIT)) * w
            pitch_t = float(np.clip(target[1], -PITCH_LIMIT, PITCH_LIMIT)) * w
        else:
            yaw_t = pitch_t = 0.0
        base = 0.6 * self.yaw_base(yaw_t, dt)
        ch["base_yaw"] = base
        ch["yaw"] += self.yaw_head(yaw_t, dt) - base
        ch["pitch"] += self.pitch(pitch_t, dt)
        return ch

    def _pose_people(self):
        for name, st in self.people_state.items():
            if st is not None:
                root, yaw, joints, mouth = st
                H.apply(self.m, self.d, self.people[name], root, yaw, joints, mouth)

    def _log(self):
        f = self.frames
        f["t"].append(self.t)
        f["qpos"].append(self.d.qpos.copy())
        f["mocap_pos"].append(self.d.mocap_pos.copy())
        f["mocap_quat"].append(self.d.mocap_quat.copy())
        f["geom_size"].append(self.m.geom_size[self._mouth_geoms].copy() if self._mouth_geoms else np.zeros((0, 3)))

    # ---- outputs ----------------------------------------------------------------------------------
    def render_base_camera(self) -> np.ndarray:
        self._r_base.update_scene(self.d, "base_cam")
        return self._r_base.render()

    def render(self, camera: str = "front34") -> np.ndarray:
        self._r_view.update_scene(self.d, camera)
        return self._r_view.render()

    def save_log(self, path: str | Path, meta: dict | None = None):
        path = Path(path)
        np.savez_compressed(path, **{k: np.asarray(v) for k, v in self.frames.items()},
                            mouth_geoms=np.asarray(self._mouth_geoms, int))
        info = dict(meta or {}, hand=self.hand.name, people=list(self.people), events=self.events_log)
        path.with_suffix(".events.json").write_text(json.dumps(info, default=float))

    def close(self):
        self._r_base.close()
        self._r_view.close()
