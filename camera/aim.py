"""Aim policy: PERCEPTION decides who is speaking (CONTRACT §5); the PTZ frames them. Built-in AI tracking
stays off.

    from aim import AimPolicy, PTZDirector
    d = PTZDirector(PTZ(make_backend("mac")))        # or PTZ(SimBackend(world)) in the room sim
    d.on_perception(latest)                          # every §5 latest value (base-frame bearings)
    d.on_kermit_event(ev)                            # run_kermit stdout: play / stop / done (Kermit talking)
    pose = d.tick()                                  # >= 30 Hz; returns the Pose to stamp frames with

Framing rules (all thresholds are AimPolicy arguments):
  presentation  frame the current speaker, face at ~2/3 of the frame height, zoom 1 (perception wants
                context: hands, mic, the other presenter when they're close).
                * switch to a new speaker_id only after it has held SWITCH_HOLD (0.6 s) and the current
                  shot has lasted MIN_DWELL (1.5 s)
                * same person: re-aim only past a deadband (18 % of the HFOV at once, or 8 % held 1.5 s),
                  so small sways never move the camera
                * off-frame speaker: PERCEPTION's `unseen_speech` held 0.5 s (voice present, no visible lips
                  moving; the main trigger), or the person in frame is visibly not speaking (their own lip activity:
                  1 s mean of `asd` < -0.9 when the face is visible; else speaking_p < 0.25) while
                  the room talks (speech_active, or the hub's STT heard someone) for 1 s, or for 4 s when
                  there is no live speech signal at all -> visit the other known presenters, then the
                  audience, then come back; a fresh speaker_id on the way ends the search at once
  qa            judges seated far back: one held group shot over the audience zone and every known judge
                (zoom up to 2x so far faces get more pixels); re-framed only when a judge leaves its inner 85 %. A
                judge becoming speaker_id enters qa; a presenter speaking for 3 s leaves it
  survey        at start-up, and whenever the room talks but no presenter is known, look around the
                presenter zones (+45, -60 deg) and the audience, 1.2 s each after the slew, to fill the people map
  hold / home   freeze the current goal / go to the home pose (BRAIN or the operator can force these)
People are remembered off-frame (last base-frame point, FORGET_S = 30 s): the camera only sees part of
the stage, so a person out of view isn't gone.
"""
from __future__ import annotations

import json
import math
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ptz import LINK2, PTZ, Pose, fov


@dataclass
class Person:
    id: str
    point: np.ndarray                # face point, base frame
    role: str = "unknown"
    seen: float = -math.inf          # last time PERCEPTION reported them
    speaking_p: float = 0.0
    spoke: float = -math.inf         # last time they were speaker_id
    asd: list = field(default_factory=list)   # recent (t, Light-ASD logit) while the face is visible

    # Light-ASD logits on real video: -0.32 +- 0.74 talking, -1.52 +- 0.76 silent (PERCEPTION, Columbia).
    # A single frame clears 0 for a talker only ~1/3 of the time, so decide on the ~1 s mean vs the midpoint.
    ASD_MID, ASD_WIN, ASD_MIN_N = -0.9, 1.0, 5

    ever_asd: bool = False                    # PERCEPTION reports lip scores for this person at all

    def lips(self, quiet_p: float = 0.25):
        """True = visibly talking, False = visibly not, None = can't tell yet (face just appeared)."""
        if not self.ever_asd:                 # no lip scores from PERCEPTION: fall back to the fused value
            return True if self.speaking_p >= 0.5 else (False if self.speaking_p < quiet_p else None)
        if len(self.asd) < self.ASD_MIN_N or self.seen - self.asd[-1][0] > 0.3:
            return None
        return float(np.mean([a for _, a in self.asd])) > self.ASD_MID

    def lips_quiet(self, quiet_p: float) -> bool:
        """Visibly not talking. Prefer the face's own lip activity: the fused speaking_p carries priors
        (alone in view + room VAD -> high) that would keep a narrow camera locked on a silent person."""
        return self.lips(quiet_p) is False

    def lips_talking(self) -> bool:
        return self.lips() is True


@dataclass
class AimPolicy:
    mount: tuple = (0.3, 0.3, 0.5)
    mount_yaw: float = 0.0
    aspect: float = 16 / 9
    home: tuple = (0.0, 5.0, 1.0)            # pan, tilt, zoom: toward the audience
    face_at: float = 0.66                     # face height in the frame (0 = bottom, 1 = top)
    single_zoom: float = 1.0
    qa_zoom_max: float = 2.0
    qa_margin_deg: float = 10.0
    audience_pan: tuple = (-22.0, 12.0)       # pan span (deg) where the judges sit; set it at the venue
    audience_tilt: float = -2.0               # their face elevation from the mount
    switch_hold: float = 0.6
    min_dwell: float = 1.5
    reaim_frac: float = 0.18
    drift_frac: float = 0.08
    drift_hold: float = 1.5
    quiet_p: float = 0.25
    search_after: float = 1.0
    patrol_after: float = 4.0                 # no usable speech signal: search after this long of a quiet frame
    signal_window: float = 60.0               # speech_active counts as alive if it was true this recently
    search_backoff: float = 6.0               # after a fruitless search, wait before the next one
    survey_idle_s: float = 8.0                # nobody known: survey again this often
    qa_exit_s: float = 3.0
    forget_s: float = 30.0
    survey: tuple = ((45.0, 12.0, 1.0), (-60.0, 12.0, 1.0), (0.0, -6.0, 1.0))
    survey_hold: float = 1.2
    mode: str = "auto"                        # auto | presentation | qa | hold | home
    people: dict = field(default_factory=dict)
    speaker_id: str | None = None
    speech_active: bool = False
    kermit_talking: bool = False
    phase: str = "presentation"               # the effective mode when mode == auto
    focus: str | None = None
    goal: tuple | None = None
    why: str = "home"
    _cand: tuple = (None, -math.inf)          # (speaker id waiting to be framed, since)
    _t_shot: float = -math.inf
    _drift_since: float | None = None
    _quiet_since: float | None = None
    _plan: dict | None = None                 # off-frame speaker search in progress
    _backoff_until: float = -math.inf
    _unseen_since: float | None = None
    _t_view: float = -math.inf                # when the current shot's view is fully in PERCEPTION's window
    unseen_hold: float = 0.5                  # PERCEPTION's unseen_speech held this long -> go look
    _speech_seen_t: float = -math.inf         # last time speech_active was true (is the signal alive?)
    room_speech_until: float = -math.inf      # hub evidence of human speech (STT turn_candidate / heard)
    _presenter_since: float | None = None
    _qa_out_since: float | None = None
    _survey_i: int | None = 0                 # next survey pose; None = not surveying
    _survey_t: float = -math.inf
    _survey_dur: float = 0.0
    _survey_list: list | None = None
    _unheard_since: float | None = None
    _sid_t: float = -math.inf                 # when speaker_id last changed

    # ---- inputs ----
    def observe(self, latest: dict, now: float):
        """A CONTRACT §5 latest value. Bearings are base-frame (PERCEPTION applied the camera pose)."""
        for p in latest.get("people", []):
            pid = str(p.get("id"))
            point = self._point(p)
            if point is None:
                continue
            per = self.people.get(pid) or Person(pid, point)
            per.point, per.seen = point, now
            per.role = p.get("role", per.role)
            per.speaking_p = float(p.get("speaking_p", per.speaking_p))
            asd = p.get("asd")
            if asd is not None and p.get("face", True):
                per.asd.append((now, float(asd)))
                per.ever_asd = True
            per.asd = [x for x in per.asd if now - x[0] <= Person.ASD_WIN]
            self.people[pid] = per
        sid = latest.get("speaker_id")
        sid = None if sid is None else str(sid)
        if sid != self.speaker_id:
            self._sid_t = now
        self.speaker_id = sid
        if self.speaker_id in self.people:
            self.people[self.speaker_id].spoke = now
        self.speech_active = bool(latest.get("speech_active", self.speaker_id is not None))
        if self.speech_active:
            self._speech_seen_t = now
        unseen = bool(latest.get("unseen_speech", False))   # PERCEPTION: voice, but no visible lips moving
        self._unseen_since = (now if self._unseen_since is None else self._unseen_since) if unseen else None
        for pid in [k for k, v in self.people.items() if now - v.seen > self.forget_s]:
            del self.people[pid]

    @staticmethod
    def _point(p: dict):
        if "target_base" in p:
            return np.asarray(p["target_base"], float)
        if "yaw_deg" not in p:
            return None
        y, e, r = math.radians(p["yaw_deg"]), math.radians(p.get("pitch_deg", 0.0)), float(p.get("dist_m", 1.5))
        return r * np.array([math.cos(e) * math.cos(y), math.cos(e) * math.sin(y), math.sin(e)])

    # ---- geometry ----
    def aim_at(self, point, zoom: float) -> tuple[float, float]:
        d = np.asarray(point, float) - np.asarray(self.mount, float)
        pan = math.degrees(math.atan2(d[1], d[0])) - self.mount_yaw
        elev = math.degrees(math.atan2(d[2], math.hypot(d[0], d[1])))
        return pan, elev - (self.face_at - 0.5) * fov(zoom, self.aspect)[1]

    def _group_shot(self, ids, zoom_max, extra_pans=(), extra_tilt=None):
        pans, tilts = list(extra_pans), [] if extra_tilt is None else [extra_tilt]
        for pid in ids:
            pan, _ = self.aim_at(self.people[pid].point, 1.0)
            d = self.people[pid].point - np.asarray(self.mount)
            pans.append(pan)
            tilts.append(math.degrees(math.atan2(d[2], math.hypot(d[0], d[1]))))
        span = max(pans) - min(pans) + 2 * self.qa_margin_deg
        h1 = fov(1.0, self.aspect)[0]
        zoom = float(np.clip(math.tan(math.radians(h1 / 2)) / math.tan(math.radians(max(span, 1.0) / 2)), 1.0, zoom_max))
        tilt = float(np.mean(tilts)) - (self.face_at - 0.5) * fov(zoom, self.aspect)[1]
        return (0.5 * (max(pans) + min(pans)), tilt, zoom)

    # ---- decision ----
    def step(self, now: float) -> tuple[tuple, str]:
        """Returns ((pan, tilt, zoom) goal, reason)."""
        if self.mode == "home":
            return self._set(self.home, "home", now)
        if self.mode == "hold" and self.goal is not None:
            return self.goal, "hold"
        if self._surveying(now):
            return self.goal, self.why
        if not self.people:
            return self._set(self.home, "nobody known", now)
        self._update_phase(now)
        if self.phase == "qa":
            return self._qa(now)
        return self._presentation(now)

    def _live_speaker(self, now):
        """speaker_id if it is a fresh call: in view and visibly talking, or just changed."""
        sp = self.people.get(self.speaker_id)
        if sp is None or now - sp.seen > 0.5:
            return None
        return sp if sp.speaking_p >= 0.5 or now - self._sid_t < 2.0 else None

    def _start_survey(self, skip_near=False):
        h = fov(self.goal[2] if self.goal else 1.0, self.aspect)[0]
        cur = self.goal[0] if self.goal else 0.0
        self._survey_list = [s for s in self.survey if not skip_near or abs(s[0] - cur) > 0.4 * h]
        self._survey_i = 0

    def _surveying(self, now) -> bool:
        presenters = [p for p in self.people.values() if p.role != "judge"]
        if self._survey_i is None and self.speech_active and not self.kermit_talking and not presenters \
                and self.speaker_id not in self.people and self.mode != "qa":
            self._unheard_since = self._unheard_since or now
            if now - self._unheard_since >= 1.0:
                self._start_survey()
                self._unheard_since = None
        elif presenters or not self.speech_active:
            self._unheard_since = None
        if self._survey_i is None and not self.people and now - self._t_shot >= self.survey_idle_s \
                and self.mode not in ("qa", "hold", "home"):
            self._start_survey()              # nobody seen yet (e.g. the camera came up before the room)
        if self._survey_i is None:
            return False
        poses = self._survey_list if self._survey_list is not None else self.survey
        if self._survey_i > 0 and self._live_speaker(now) is not None:
            self._survey_i = None             # found who is talking: stop looking around
            return False
        if self._survey_i == 0 or now - self._survey_t >= self._survey_dur:
            if self._survey_i >= len(poses):
                self._survey_i = None
                return False
            prev = self.goal[0] if self.goal else 0.0
            self._survey_dur = self.survey_hold + abs(poses[self._survey_i][0] - prev) / 60.0  # + slew time
            self._set(poses[self._survey_i], f"survey {self._survey_i + 1}/{len(poses)}", now)
            self._survey_i += 1
            self._survey_t = now
        return True

    def _update_phase(self, now):
        if self.mode in ("presentation", "qa"):
            self.phase = self.mode
            return
        sp = self.people.get(self.speaker_id)
        if sp is not None and sp.role == "judge" and sp.lips_talking():
            self.phase = "qa"                 # only on visible evidence: a fused prior alone mustn't flip the show
        if self.phase == "qa":
            if sp is not None and sp.role == "presenter":
                self._presenter_since = self._presenter_since or now
                if now - self._presenter_since >= self.qa_exit_s:
                    self.phase, self._presenter_since = "presentation", None
            else:
                self._presenter_since = None

    def _qa(self, now):
        self._plan = None
        judges = [k for k, v in self.people.items() if v.role == "judge"]
        shot = self._group_shot(judges, self.qa_zoom_max, self.audience_pan, self.audience_tilt)
        if self.goal is None or self.why != "qa group":
            return self._set(shot, "qa group", now)
        half = 0.85 * fov(self.goal[2], self.aspect)[0] / 2
        outside = any(abs(self.aim_at(self.people[j].point, 1.0)[0] - self.goal[0]) > half for j in judges)
        if outside:
            self._qa_out_since = self._qa_out_since or now
            if now - self._qa_out_since >= 1.0:
                self._qa_out_since = None
                self.goal = shot
                return self.goal, "qa group"
        else:
            self._qa_out_since = None
        return self.goal, "qa group"

    def _presentation(self, now):
        sid = self.speaker_id if self.speaker_id in self.people else None
        if sid is not None and not self.people[sid].lips_talking() and now - self._sid_t > 2.0:
            sid = None   # PERCEPTION is holding a stale speaker_id (seen, visibly not talking): not a reason to cut
        if self.focus not in self.people:
            self.focus = None
        if sid is not None and self.focus is not None and sid != self.focus and self._same_place(sid, self.focus):
            self.focus = sid                  # a re-ID gave the framed person a new id: same shot, not a new speaker
        # 1. a new speaker, held long enough, after the current shot has had its time
        if sid is not None and sid != self.focus:
            if self._cand[0] != sid:
                self._cand = (sid, now)
            if self.focus is None or self._plan is not None or \
                    (now - self._cand[1] >= self.switch_hold and now - self._t_shot >= self.min_dwell):
                self._plan = None
                return self._frame(sid, "speaker", now)
        else:
            self._cand = (None, -math.inf)
        # 2. the speaker is off-frame: the person in frame is visibly not talking while the room is (or, with no
        #    usable speech signal at all, for a longer while): visit the other presenters, then the audience
        f = self.people.get(self.focus)
        in_view = f is not None and now - f.seen < 0.5
        if self._plan is not None:
            if in_view and (f.lips_talking() or (self._unseen_since is None and f.speaking_p >= 0.5)):
                self._plan = None             # the one we just turned to is talking: found
            else:
                return self._search_step(now)
        talking = self.speech_active or now < self.room_speech_until
        signal_ok = now - self._speech_seen_t < self.signal_window
        if not self.kermit_talking and self._unseen_since is not None \
                and now - max(self._unseen_since, self._t_view) >= self.unseen_hold \
                and now - self._t_shot >= self.min_dwell and now >= self._backoff_until:
            self._unseen_since = None         # PERCEPTION's own "someone talks whom I can't see": the main trigger
            return self._start_search(now, "unseen speech")
        if not self.kermit_talking and in_view and f.lips_quiet(self.quiet_p) and sid in (None, self.focus):
            self._quiet_since = self._quiet_since or now
            wait = self.search_after if talking else (math.inf if signal_ok else self.patrol_after)
            if now - self._quiet_since >= wait and now - self._t_shot >= self.min_dwell and now >= self._backoff_until:
                self._quiet_since = None
                return self._start_search(now, "search" if talking else "patrol")
        else:
            self._quiet_since = None
        if self.focus is None:
            return (self.goal, self.why) if self.goal is not None else self._set(self.home, "home", now)
        # 3. same person: deadband with a slow drift correction
        want = (*self.aim_at(self.people[self.focus].point, self.single_zoom), self.single_zoom)
        err = max(abs(want[0] - self.goal[0]), abs(want[1] - self.goal[1]))
        h = fov(self.goal[2], self.aspect)[0]
        if err > self.reaim_frac * h:
            self._drift_since = None
            self.goal = want
        elif err > self.drift_frac * h:
            self._drift_since = self._drift_since or now
            if now - self._drift_since >= self.drift_hold:
                self._drift_since, self.goal = None, want
        else:
            self._drift_since = None
        return self.goal, self.why

    def _same_place(self, a, b, deg=8.0):
        pa, pb = self.people[a].point - np.asarray(self.mount), self.people[b].point - np.asarray(self.mount)
        cos = float(pa @ pb / (np.linalg.norm(pa) * np.linalg.norm(pb) + 1e-9))
        return math.degrees(math.acos(min(1.0, max(-1.0, cos)))) < deg

    def _start_search(self, now, kind):
        others = sorted((p for k, p in self.people.items() if k != self.focus and p.role != "judge"),
                        key=lambda p: (p.spoke, p.seen), reverse=True)
        steps = [("person", p.id) for p in others]
        if not steps:                         # nobody else known yet: the survey poses away from here
            h = fov(self.goal[2] if self.goal else 1.0, self.aspect)[0]
            cur = self.goal[0] if self.goal else 0.0
            steps = [("pose", sp) for sp in self.survey[:2] if abs(sp[0] - cur) > 0.4 * h]
        steps.append(("audience", None))
        if self.focus is not None:
            steps.append(("person", self.focus))   # nothing found: back to where we were
        self._plan = {"kind": kind, "steps": steps, "i": 0, "t": -math.inf, "hold": 0.0}
        return self._search_step(now)

    def _search_step(self, now):
        pl = self._plan
        if now - pl["t"] < pl["hold"]:
            return self.goal, self.why
        while pl["i"] < len(pl["steps"]):
            what, arg = pl["steps"][pl["i"]]
            pl["i"] += 1
            prev = self.goal[0] if self.goal else 0.0
            if what == "person" and arg in self.people:
                goal, label = (*self.aim_at(self.people[arg].point, self.single_zoom), self.single_zoom), f"person {arg}"
                self.focus = arg
            elif what == "pose":
                goal, label, self.focus = arg, "survey pose", None
            elif what == "audience":
                judges = [k for k, v in self.people.items() if v.role == "judge"]
                goal, label, self.focus = self._group_shot(judges, self.qa_zoom_max, self.audience_pan, self.audience_tilt), "audience", None
            else:
                continue
            last = pl["i"] == len(pl["steps"])
            # dwell: the slew, then long enough for PERCEPTION to see lips move and switch speaker_id
            pl["t"], pl["hold"] = now, abs(goal[0] - prev) / 60.0 + self.search_after + self.switch_hold
            self._drift_since = self._quiet_since = None
            self._set(goal, f"{pl['kind']}: {label}" + (" (nothing found, back)" if last and what == "person" else ""), now)
            return self.goal, self.why
        self._plan = None
        self._backoff_until = now + self.search_backoff
        return self.goal, self.why

    def _frame(self, pid, why, now):
        self.focus = pid
        self._drift_since = self._quiet_since = None
        z = self.single_zoom
        return self._set((*self.aim_at(self.people[pid].point, z), z), f"{why} {pid}", now)

    def _set(self, goal, why, now):
        if goal != self.goal:
            self._t_shot = now
            # PERCEPTION's unseen_speech looks back 1.5 s: until the slew is over and that window holds only
            # the new view, it still describes the old one
            self._t_view = now + abs(goal[0] - (self.goal[0] if self.goal else 0.0)) / 60.0 + 1.5
        self.goal, self.why = tuple(float(x) for x in goal), why
        return self.goal, why


class PTZDirector:
    """AimPolicy + PTZ, fed from the live hub. Writes one NDJSON line per decision change."""

    def __init__(self, ptz: PTZ, policy: AimPolicy | None = None, log: str | Path | None = None,
                 clock=time.perf_counter, reassert_s: float = 10.0):
        self.ptz, self.clock = ptz, clock
        self.policy = policy or AimPolicy(mount=ptz.mount, mount_yaw=ptz.mount_yaw, aspect=ptz.aspect)
        self.log = open(log, "a", buffering=1) if log else None
        self._talk_until = -math.inf
        self._last_why = None
        self.reassert_s = reassert_s
        self._t_reassert = clock()
        try:
            ptz.tracking(False)   # built-in AI tracking OFF: we own the gimbal
        except Exception:  # noqa: BLE001 - logged in ptz.errors; the dry/sim backends have nothing to switch
            pass

    def _reassert(self):
        """A raised palm can switch the Link 2's tracking on by gesture: keep switching it off."""
        try:
            self.ptz.b.tracking(False)
        except Exception as e:  # noqa: BLE001
            self.ptz.errors.append(f"tracking off: {e}")

    def on_vad(self, active: bool):
        """Optional: the hub's mic VAD (self-voice excluded), if PERCEPTION's speech_active isn't wired."""
        if active:
            self.policy.room_speech_until = self.clock() + 0.8

    def on_perception(self, latest: dict):
        self.policy.observe(latest, self.clock())

    def on_kermit_event(self, ev: dict):
        kind, now = ev.get("event"), self.clock()
        if kind == "segment":
            m = ev.get("mouth")
            n = len(m.get("values", [])) / float(m.get("hz", 60)) if isinstance(m, dict) else 0.0
            self._seg_len = max(n, 0.5)
        elif kind == "play":
            self._talk_until = max(self._talk_until, now + getattr(self, "_seg_len", 3.0) + 0.3)
        elif kind in ("stop", "done"):
            self._talk_until = now if kind == "stop" else self._talk_until
        elif kind == "turn_candidate" or (kind == "heard" and not ev.get("echo")):
            self.policy.room_speech_until = now + 3.0    # the hub's STT heard a human: the room is talking
        elif kind == "camera_mode":           # {"event": "camera_mode", "mode": "qa" | "hold" | ...}
            self.policy.mode = ev.get("mode", "auto")

    def tick(self, now: float | None = None) -> Pose:
        now = self.clock() if now is None else now
        self.policy.kermit_talking = now < self._talk_until
        goal, why = self.policy.step(now)
        if self.reassert_s and now - self._t_reassert >= self.reassert_s and self.ptz.b.name in ("mac", "v4l2"):
            self._t_reassert = now
            threading.Thread(target=self._reassert, daemon=True).start()
        self.ptz.set(*goal)
        pose = self.ptz.update(now)
        if self.log and why != self._last_why:
            self.log.write(json.dumps({"t": round(now, 3), "why": why, "goal": [round(x, 1) for x in goal],
                                       "phase": self.policy.phase}) + "\n")
        self._last_why = why
        return pose


if __name__ == "__main__":  # standalone: poll PERCEPTION's latest-value file and drive the camera
    import argparse
    from ptz import make_backend
    ap = argparse.ArgumentParser(description="Aim the Link 2 at PERCEPTION's speaker (AI tracking off)")
    ap.add_argument("--backend", default="dry", choices=["mac", "v4l2", "dry"])
    ap.add_argument("--perception", required=True, help="PERCEPTION's §5 latest-value JSON (people.py --latest)")
    ap.add_argument("--out", default="ptz_out", help="ptz_pose.json (latest) + ptz_pose.ndjson + aim.ndjson")
    ap.add_argument("--mount", type=float, nargs=3, default=[0.3, 0.3, 0.5])
    ap.add_argument("--hz", type=float, default=30)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    d = PTZDirector(PTZ(make_backend(a.backend), mount=a.mount, pose_log=out / "ptz_pose.ndjson",
                        latest=out / "ptz_pose.json"), log=out / "aim.ndjson")
    last = None
    while True:
        try:
            txt = Path(a.perception).read_text()
            if txt != last:
                last = txt
                d.on_perception(json.loads(txt))
        except (OSError, ValueError):
            pass
        d.tick()
        time.sleep(1 / a.hz)
