"""Insta360 Link 2 pan / tilt / zoom control for Kermit's perception camera.

    from ptz import PTZ, make_backend
    cam = PTZ(make_backend("mac"))          # "v4l2" (Linux) | "mac" (linkcam CLI) | "dry" | SimBackend(world)
    cam.tracking(False)                     # the built-in AI tracking stays OFF: we aim it
    cam.set(pan=30, tilt=10, zoom=1.0)      # goal, degrees / zoom factor; clamped to the Link 2 limits
    while True:
        pose = cam.update()                 # call at >= 30 Hz: smooths, rate-limits, sends, publishes the pose
        ...
    cam.pose_at(frame_t)                    # pose at a frame's capture time (perf_counter clock)

Conventions (same frame as CONTRACT §4/§5, the arm base frame):
  pan  = yaw of the optical axis about +z, + = left (toward base +y), 0 = the mount's forward direction
  tilt = elevation of the optical axis, + = up
  zoom = digital zoom factor 1..4; the field of view scales as tan(fov/2) / zoom
Camera rotation in the base frame: R = Rz(mount_yaw + pan) @ Ry(-tilt), the optical axis is the body +x;
for PERCEPTION's track.Camera that is rpy = (0, -tilt, mount_yaw + pan) and fx = fx_1x * zoom.

The gimbal reads back the last *commanded* value over UVC, not where the motors are, so the pose
published for real hardware is a model of the motors (Gimbal) fed with the commands actually sent.
The sim backend reports the simulated motors directly. Timestamps are time.perf_counter() (system-wide
monotonic on macOS and Linux, the same clock as the room sim's X-Frame-T).

Hardware facts this is built on (no Link 2 here, so the real backends are dry-tested only):
  * Linux: standard UVC controls via v4l2: pan_absolute / tilt_absolute in arc-seconds (3600 = 1 deg,
    the gimbal moves in 1-deg steps), zoom_absolute 100..400. The descriptor claims +-180 deg; measured
    travel is pan +-145, tilt -90..+100 (battistaar/insta360_link_control, tested on 2e1a:4c04).
  * macOS: fmontes/linkcam (CoreMediaIO for pan/tilt/zoom, IOKit UVC extension unit for AI tracking:
    unit 9, selector 0x1b). `linkcam set <pan> <tilt>`, `linkcam zoom <x>`, `linkcam ai off`,
    `linkcam status --json`.
  * The gimbal only moves while some app is streaming video from it; otherwise it parks (~-85 deg tilt)
    and ignores writes. So the controller re-sends its setpoint every `resend_s`.
  * UVC says positive pan is clockwise seen from above (= to the right), the opposite of our + = left,
    hence pan_sign = -1 by default. The runbook's smoke test checks this on the real camera.
"""
from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np


# ---- the camera ---------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Limits:
    pan: tuple = (-145.0, 145.0)
    tilt: tuple = (-90.0, 100.0)
    zoom: tuple = (1.0, 4.0)
    step_deg: float = 1.0            # gimbal resolution
    diag_fov_deg: float = 79.5       # at zoom 1, 16:9 (Insta360 spec)


LINK2 = Limits()


def fov(zoom: float = 1.0, aspect: float = 16 / 9, limits: Limits = LINK2) -> tuple[float, float]:
    """(hfov, vfov) in degrees at a zoom factor for an output aspect ratio (the diagonal is the spec)."""
    t = math.tan(math.radians(limits.diag_fov_deg) / 2) / max(zoom, 1e-6)
    diag = math.hypot(aspect, 1.0)
    return (math.degrees(2 * math.atan(t * aspect / diag)), math.degrees(2 * math.atan(t / diag)))


def rotation(pan: float, tilt: float, mount_yaw: float = 0.0) -> np.ndarray:
    """Camera body rotation in the base frame (columns: optical axis, image left, image up)."""
    y, p = math.radians(mount_yaw + pan), math.radians(tilt)
    Rz = np.array([[math.cos(y), -math.sin(y), 0], [math.sin(y), math.cos(y), 0], [0, 0, 1]])
    Ry = np.array([[math.cos(p), 0, -math.sin(p)], [0, 1, 0], [math.sin(p), 0, math.cos(p)]])  # Ry(-tilt)
    return Rz @ Ry


def pixel_to_base(u: float, v: float, pose: "Pose", width: int, height: int) -> tuple[float, float]:
    """Exact bearing (yaw, pitch) in degrees, base frame, of image pixel (u, v) at a camera pose.
    The contract's shortcut yaw = pan + bearing_in_image is exact at tilt 0 and within ~1.6 deg for
    |tilt| <= 20 deg and |bearing_in_image| <= 30 deg."""
    hfov, vfov = fov(pose.zoom, width / height)
    fx = (width / 2) / math.tan(math.radians(hfov) / 2)
    fy = (height / 2) / math.tan(math.radians(vfov) / 2)
    ray = rotation(pose.pan, pose.tilt, pose.mount_yaw) @ np.array([1.0, -(u - width / 2) / fx, -(v - height / 2) / fy])
    return math.degrees(math.atan2(ray[1], ray[0])), math.degrees(math.atan2(ray[2], math.hypot(ray[0], ray[1])))


@dataclass
class Pose:
    t: float                         # perf_counter seconds
    pan: float
    tilt: float
    zoom: float
    moving: bool = False
    speed: float = 0.0               # deg/s, pan/tilt combined: PERCEPTION can down-weight frames above ~30
    mount: tuple = (0.3, 0.3, 0.5)  # base frame, metres (the post beside the arm)
    mount_yaw: float = 0.0
    src: str = "model"               # "model" (motor model fed with sent commands) | "sim" | "readback"
    tracking: bool = False           # built-in AI tracking state as last commanded

    def to_json(self, aspect: float = 16 / 9) -> dict:
        hfov, vfov = fov(self.zoom, aspect)
        return {"t": round(self.t, 4), "t_unix": round(time.time(), 3), "pan_deg": round(self.pan, 2),
                "tilt_deg": round(self.tilt, 2), "zoom": round(self.zoom, 3), "hfov_deg": round(hfov, 2),
                "vfov_deg": round(vfov, 2), "moving": self.moving, "speed_dps": round(self.speed, 1),
                "mount": [round(x, 3) for x in self.mount], "mount_yaw_deg": self.mount_yaw, "src": self.src,
                "tracking": self.tracking}


@dataclass
class Gimbal:
    """Motor model: each axis approaches its commanded absolute target with a speed and acceleration
    limit; zoom ramps at a fixed rate. The Link 2's real speeds are not published; these are guesses
    to replace with the smoke test's measurement (README: "calibrate")."""
    vmax: tuple = (120.0, 90.0)      # deg/s pan, tilt
    amax: tuple = (500.0, 400.0)     # deg/s^2
    zoom_rate: float = 2.0           # zoom factor per s
    limits: Limits = LINK2
    pan: float = 0.0
    tilt: float = 0.0
    zoom: float = 1.0
    v: list = field(default_factory=lambda: [0.0, 0.0])
    target: tuple = (0.0, 0.0, 1.0)

    def command(self, pan: float, tilt: float, zoom: float):
        L = self.limits
        q = L.step_deg
        self.target = (round(float(np.clip(pan, *L.pan)) / q) * q, round(float(np.clip(tilt, *L.tilt)) / q) * q,
                       float(np.clip(zoom, *L.zoom)))

    def step(self, dt: float):
        for i, name in enumerate(("pan", "tilt")):
            x, goal, v = getattr(self, name), self.target[i], self.v[i]
            err = goal - x
            v_stop = math.copysign(math.sqrt(2 * self.amax[i] * abs(err)), err)  # brake in time
            v_goal = float(np.clip(err / max(dt, 1e-6), -abs(v_stop), abs(v_stop)))
            v = float(np.clip(np.clip(v_goal, v - self.amax[i] * dt, v + self.amax[i] * dt), -self.vmax[i], self.vmax[i]))
            x += v * dt
            if abs(goal - x) < 1e-3 and abs(v) < 1.0:
                x, v = goal, 0.0
            setattr(self, name, x)
            self.v[i] = v
        dz = self.target[2] - self.zoom
        self.zoom += float(np.clip(dz, -self.zoom_rate * dt, self.zoom_rate * dt))

    @property
    def moving(self) -> bool:
        return any(abs(v) > 0.5 for v in self.v) or abs(self.target[2] - self.zoom) > 1e-3

    @property
    def speed(self) -> float:
        return math.hypot(*self.v)


# ---- backends -----------------------------------------------------------------------------------------

class Backend:
    """write() / read() / tracking(); subprocess backends run write() on a worker thread (see PTZ)."""
    name = "base"
    min_period = 0.05                # s between writes the device / tool can sustain
    reports_actual = False           # read() returns where the motors are (sim) rather than the command
    blocking = False                 # write() is slow (spawns a process): run it off the control loop

    def write(self, pan: float, tilt: float, zoom: float, zoom_changed: bool = True): ...
    def read(self): return None
    def tracking(self, on: bool): ...
    def close(self): ...


# our + pan = left; UVC's + pan = clockwise from above = right. Flip with KERMIT_PTZ_PAN_SIGN=1 if the smoke
# test shows the camera turning the wrong way.
PAN_SIGN = int(os.environ.get("KERMIT_PTZ_PAN_SIGN", "-1"))


def _run(cmd: list[str], timeout: float = 2.0) -> str:
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if r.returncode:
        raise RuntimeError(f"{' '.join(cmd)}: {r.stderr.strip() or r.stdout.strip()}")
    return r.stdout


class V4L2Backend(Backend):
    """Linux: `v4l2-ctl` (package v4l-utils) for pan/tilt/zoom, `linkctl` (battistaar) for AI tracking."""
    name = "v4l2"
    min_period = 0.05
    blocking = True

    def __init__(self, device: str = "/dev/video0", pan_sign: int = PAN_SIGN, linkctl: str | None = None):
        self.device, self.pan_sign = device, pan_sign
        self.v4l2 = shutil.which("v4l2-ctl")
        self.linkctl = linkctl or shutil.which("linkctl")
        if not self.v4l2:
            raise RuntimeError("v4l2-ctl not found: sudo apt install v4l-utils")

    def write(self, pan, tilt, zoom, zoom_changed=True):
        ctrls = f"pan_absolute={int(round(self.pan_sign * pan)) * 3600},tilt_absolute={int(round(tilt)) * 3600}"
        if zoom_changed:
            ctrls += f",zoom_absolute={int(round(zoom * 100))}"
        _run([self.v4l2, "-d", self.device, f"--set-ctrl={ctrls}"])

    def read(self):
        out = _run([self.v4l2, "-d", self.device, "--get-ctrl=pan_absolute,tilt_absolute,zoom_absolute"])
        v = dict(line.split(":") for line in out.split("\n") if ":" in line)
        return (self.pan_sign * int(v["pan_absolute"]) / 3600, int(v["tilt_absolute"]) / 3600,
                int(v["zoom_absolute"]) / 100)

    def tracking(self, on):
        if not self.linkctl:
            raise RuntimeError("AI tracking is a vendor extension-unit control: install linkctl "
                               "(github.com/battistaar/insta360_link_control) or turn it off once in the Insta360 app")
        _run([self.linkctl, "track", "on" if on else "off"])


class LinkcamBackend(Backend):
    """macOS: fmontes/linkcam. Chosen over the lucasoskorep HTTP server (127.0.0.1:8360/api/ptz) because
    it is a documented, Link 2-tested single binary with no daemon to keep alive; that server's docs could
    not be fetched, so its API is unverified. Each call spawns a process, hence the slow rate."""
    name = "mac"
    min_period = 0.25
    blocking = True

    def __init__(self, binary: str | None = None, pan_sign: int = PAN_SIGN):
        self.bin = binary or shutil.which("linkcam") or "/usr/local/bin/linkcam"
        self.pan_sign = pan_sign
        if not Path(self.bin).exists():
            raise RuntimeError("linkcam not found: see camera/README.md (Mac runbook)")

    def write(self, pan, tilt, zoom, zoom_changed=True):
        _run([self.bin, "set", f"{self.pan_sign * pan:.0f}", f"{tilt:.0f}"])
        if zoom_changed:
            _run([self.bin, "zoom", f"{zoom:.2f}"])

    def read(self):
        st = json.loads(_run([self.bin, "status", "--json"]))
        vals = {}

        def walk(d):
            for k, v in (d.items() if isinstance(d, dict) else []):
                if isinstance(v, dict):
                    walk(v)
                elif isinstance(v, (int, float)):
                    for key in ("pan", "tilt", "zoom"):
                        if key in k.lower() and key not in vals:
                            vals[key] = float(v)
        walk(st)
        if not {"pan", "tilt"} <= vals.keys():
            return None
        return self.pan_sign * vals["pan"], vals["tilt"], vals.get("zoom", 1.0)

    def tracking(self, on):
        _run([self.bin, "ai", "on" if on else "off"])


class DryRunBackend(Backend):
    """No hardware: records every command (and optionally appends them to an NDJSON file)."""
    name = "dry"
    min_period = 0.05

    def __init__(self, log: str | Path | None = None, min_period: float | None = None):
        self.log = open(log, "a", buffering=1) if log else None
        self.min_period = self.min_period if min_period is None else min_period
        self.sent: list = []
        self.last = None

    def write(self, pan, tilt, zoom, zoom_changed=True):
        self.last = (pan, tilt, zoom)
        rec = {"t": round(time.perf_counter(), 4), "cmd": "set", "pan": pan, "tilt": tilt, "zoom": zoom}
        self.sent.append(rec)
        if self.log:
            self.log.write(json.dumps(rec) + "\n")

    def read(self):
        return self.last

    def tracking(self, on):
        rec = {"t": round(time.perf_counter(), 4), "cmd": "tracking", "on": on}
        self.sent.append(rec)
        if self.log:
            self.log.write(json.dumps(rec) + "\n")

    def close(self):
        if self.log:
            self.log.close()


class SimBackend(Backend):
    """MOTION's LiveWorld PTZ camera (motion/live.py, LiveWorld(ptz=...)): commands go to the simulated
    gimbal, read() returns the simulated motors."""
    name = "sim"
    min_period = 0.0
    reports_actual = True

    def __init__(self, world):
        self.world = world

    def write(self, pan, tilt, zoom, zoom_changed=True):
        self.world.ptz_command(pan, tilt, zoom)

    def read(self):
        p = self.world.ptz_pose()
        return p.pan, p.tilt, p.zoom

    def tracking(self, on):
        if on:
            raise RuntimeError("the sim has no built-in tracking (and we keep it off on the real camera)")


def make_backend(kind: str, **kw) -> Backend:
    if kind == "v4l2":
        return V4L2Backend(**kw)
    if kind == "mac":
        return LinkcamBackend(**kw)
    if kind == "dry":
        return DryRunBackend(**kw)
    raise ValueError(f"unknown PTZ backend {kind!r} (v4l2 | mac | dry; sim: SimBackend(world))")


# ---- the controller -----------------------------------------------------------------------------------

class PTZ:
    """Goal -> smoothed setpoint (speed and acceleration capped) -> quantized, rate-limited writes ->
    pose estimate, logged per update and kept for pose_at()."""

    def __init__(self, backend: Backend, limits: Limits = LINK2, vmax: float = 90.0, amax: float = 240.0,
                 zoom_rate: float = 1.0, resend_s: float = 1.0, mount=(0.3, 0.3, 0.5), mount_yaw: float = 0.0,
                 gimbal: Gimbal | None = None, pose_log: str | Path | None = None, latest: str | Path | None = None,
                 aspect: float = 16 / 9, clock=time.perf_counter, home=(0.0, 0.0, 1.0)):
        self.b, self.L, self.vmax, self.amax, self.zoom_rate = backend, limits, vmax, amax, zoom_rate
        self.resend_s, self.mount, self.mount_yaw, self.aspect, self.clock = resend_s, tuple(mount), mount_yaw, aspect, clock
        self.gimbal = gimbal or Gimbal(limits=limits)
        self.goal = self._clamp(*home)
        self.sp = list(self.goal)            # smoothed setpoint
        self.sp_v = [0.0, 0.0]
        self.sent, self.t_sent, self.t_last = None, -math.inf, None
        self.track_on = False
        self.history: deque = deque(maxlen=600)
        self.errors: list = []
        self.pose_log = open(pose_log, "a", buffering=1) if pose_log else None
        self.latest = Path(latest) if latest else None
        self._job, self._busy = None, threading.Event()
        self._lock = threading.Lock()
        if backend.blocking:
            self._worker = threading.Thread(target=self._work, daemon=True)
            self._wake = threading.Event()
            self._worker.start()
        self.gimbal.pan, self.gimbal.tilt, self.gimbal.zoom = self.goal
        self.gimbal.command(*self.goal)

    # -- API --
    def set(self, pan: float | None = None, tilt: float | None = None, zoom: float | None = None):
        g = self.goal
        self.goal = self._clamp(g[0] if pan is None else pan, g[1] if tilt is None else tilt, g[2] if zoom is None else zoom)

    def get(self) -> Pose:
        return self.history[-1] if self.history else self._pose(self.clock())

    def tracking(self, on: bool):
        """Built-in AI tracking. Our aim policy keeps it OFF; while on, update() stops writing."""
        try:
            self.b.tracking(on)
            self.track_on = on
        except Exception as e:  # noqa: BLE001 - report, keep running: a camera glitch must not stop Kermit
            self.errors.append(f"tracking({on}): {e}")
            raise

    def update(self, now: float | None = None) -> Pose:
        now = self.clock() if now is None else now
        dt = 0.0 if self.t_last is None else min(max(now - self.t_last, 0.0), 0.2)
        self.t_last = now
        self._smooth(dt)
        cmd = self._quantize(self._clamp(*self.sp))
        due = now - self.t_sent >= self.b.min_period
        changed = cmd != self.sent
        stale = now - self.t_sent >= self.resend_s   # writes are dropped while nothing streams: repeat them
        if not self.track_on and due and (changed or stale) and not self._busy.is_set():
            zoom_changed = self.sent is None or cmd[2] != self.sent[2] or stale
            self._send(cmd, zoom_changed, now)
        self.gimbal.step(dt)
        pose = self._pose(now)
        self.history.append(pose)
        if self.pose_log or self.latest:
            j = pose.to_json(self.aspect)
            if self.pose_log:
                self.pose_log.write(json.dumps(j) + "\n")
            if self.latest:
                tmp = self.latest.with_suffix(".tmp")
                tmp.write_text(json.dumps(j))
                os.replace(tmp, self.latest)
        return pose

    def pose_at(self, t: float) -> Pose:
        """Pose at time t (a frame's capture time), linearly interpolated over the update history."""
        h = self.history
        if not h:
            return self._pose(t)
        if t <= h[0].t:
            return h[0]
        if t >= h[-1].t:
            return h[-1]
        lo, hi = 0, len(h) - 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            lo, hi = (mid, hi) if h[mid].t <= t else (lo, mid)
        a, b = h[lo], h[hi]
        s = (t - a.t) / max(b.t - a.t, 1e-9)
        return Pose(t=t, pan=a.pan + s * (b.pan - a.pan), tilt=a.tilt + s * (b.tilt - a.tilt),
                    zoom=a.zoom + s * (b.zoom - a.zoom), moving=a.moving or b.moving, speed=max(a.speed, b.speed),
                    mount=a.mount, mount_yaw=a.mount_yaw, src=a.src, tracking=a.tracking)

    def close(self):
        if self.b.blocking:
            self._job = None
            self._wake.set()
        if self.pose_log:
            self.pose_log.close()
        self.b.close()

    # -- internals --
    def _clamp(self, pan, tilt, zoom):
        L = self.L
        return (float(np.clip(pan, *L.pan)), float(np.clip(tilt, *L.tilt)), float(np.clip(zoom, *L.zoom)))

    def _smooth(self, dt):
        """Per axis: accelerate at <= amax up to vmax, brake in time to stop on the goal (no overshoot,
        exact arrival); zoom ramps at zoom_rate. No step jumps reach the gimbal."""
        if dt <= 0:
            return
        for i in range(2):
            err = self.goal[i] - self.sp[i]
            v_stop = math.copysign(math.sqrt(2 * self.amax * abs(err)), err)
            v_goal = float(np.clip(err / dt, -abs(v_stop), abs(v_stop)))
            v = self.sp_v[i]
            v = float(np.clip(np.clip(v_goal, v - self.amax * dt, v + self.amax * dt), -self.vmax, self.vmax))
            self.sp[i] += v * dt
            self.sp_v[i] = v
        self.sp[2] += float(np.clip(self.goal[2] - self.sp[2], -self.zoom_rate * dt, self.zoom_rate * dt))

    def _quantize(self, q):
        step = self.L.step_deg
        return (round(q[0] / step) * step, round(q[1] / step) * step, round(q[2] * 50) / 50)  # zoom in 0.02 steps

    def _send(self, cmd, zoom_changed, now):
        if self.b.blocking:
            self._busy.set()
            self._job = (cmd, zoom_changed)
            self._wake.set()
        else:
            self._write(cmd, zoom_changed)
        self.sent, self.t_sent = cmd, now

    def _write(self, cmd, zoom_changed):
        try:
            self.b.write(*cmd, zoom_changed=zoom_changed)
            if not self.b.reports_actual:
                with self._lock:
                    self.gimbal.command(*cmd)
        except Exception as e:  # noqa: BLE001
            self.errors.append(f"write{cmd}: {e}")

    def _work(self):
        while True:
            self._wake.wait()
            self._wake.clear()
            job = self._job
            if job is None:
                return
            self._write(*job)
            self._busy.clear()

    def _pose(self, now) -> Pose:
        if self.b.reports_actual:
            try:
                pan, tilt, zoom = self.b.read()
                prev = self.history[-1] if self.history else None
                dt = (now - prev.t) if prev else 0.0
                speed = math.hypot(pan - prev.pan, tilt - prev.tilt) / dt if prev and dt > 0 else 0.0
                return Pose(now, pan, tilt, zoom, speed > 0.5, speed, self.mount, self.mount_yaw, "sim", self.track_on)
            except Exception as e:  # noqa: BLE001
                self.errors.append(f"read: {e}")
        g = self.gimbal
        return Pose(now, g.pan, g.tilt, g.zoom, g.moving, g.speed, self.mount, self.mount_yaw, "model", self.track_on)


# ---- 1-minute smoke test (Mac runbook) ------------------------------------------------------------------

def smoke(backend: str = "mac", out: str = "ptz_smoke", camera: int = 0, stream: bool = True):
    """Streams the camera (the gimbal only moves while streaming), turns AI tracking off, visits a few
    poses saving a frame at each (so the pan direction can be checked by eye), then measures the real pan
    speed from the video of one fast swing."""
    import cv2  # noqa: PLC0415 - only the smoke test needs OpenCV
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(camera) if stream else None
    if cap is not None and not cap.isOpened():
        raise SystemExit("could not open the camera: grant Camera access to this terminal (README, step 1)")
    if cap is not None:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cam = PTZ(make_backend(backend), pose_log=out / "pose.ndjson")
    frame = [None]
    stop = threading.Event()

    def grab():
        while not stop.is_set() and cap is not None:
            ok, f = cap.read()
            if ok:
                frame[0] = (time.perf_counter(), f)
    threading.Thread(target=grab, daemon=True).start()
    try:
        cam.tracking(False)
        print("AI tracking: off")
    except Exception as e:  # noqa: BLE001
        print("AI tracking: could not switch off:", e)
    steps = [("center", 0, 0, 1.0), ("left_40", 40, 0, 1.0), ("right_40", -40, 0, 1.0), ("up_20", 0, 20, 1.0),
             ("down_20", 0, -20, 1.0), ("zoom_2x", 0, 0, 2.0), ("center", 0, 0, 1.0)]
    print(f"{'step':10s} {'goal':>16s} {'readback':>22s} {'settle_s':>8s}")
    for name, pan, tilt, zoom in steps:
        cam.set(pan, tilt, zoom)
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < 4.0:
            p = cam.update()
            if abs(p.pan - pan) < 0.5 and abs(p.tilt - tilt) < 0.5 and abs(p.zoom - zoom) < 0.05 and time.perf_counter() - t0 > 0.5:
                break
            time.sleep(1 / 30)
        settle = time.perf_counter() - t0
        t1 = time.perf_counter()
        while time.perf_counter() - t1 < 0.8:  # no position feedback over UVC: give the motors time, then look
            cam.update()
            time.sleep(1 / 30)
        try:
            rb = cam.b.read()
        except Exception as e:  # noqa: BLE001
            rb = f"read failed: {e}"[:22]
        rb_s = "(%.0f, %.0f, %.2f)" % rb if isinstance(rb, tuple) else str(rb)
        print(f"{name:10s} {f'({pan}, {tilt}, {zoom})':>16s} {rb_s:>22s} {settle:8.2f}")
        if frame[0] is not None:
            cv2.imwrite(str(out / f"{len(list(out.glob('*.jpg'))):02d}_{name}.jpg"), frame[0][1])
    if frame[0] is not None:
        speed = _measure_speed(cam, frame, out)
        if speed:
            print(f"measured pan speed: peak {speed[0]:.0f} deg/s, 120 deg swing settles in {speed[1]:.2f} s "
                  f"(the model assumes {cam.gimbal.vmax[0]:.0f} deg/s: pass Gimbal(vmax=...) if far off)")
    stop.set()
    cam.close()
    if cap is not None:
        cap.release()
    print("errors:", cam.errors or "none")
    print(f"frames in {out}/: in 01_left_40.jpg the view should have swung to the camera's LEFT "
          "(things on your right as you face the camera). If it went the other way: export KERMIT_PTZ_PAN_SIGN=1")


def _measure_speed(cam: "PTZ", frame: list, out: Path):
    """One raw 120-deg swing (no smoothing): image shift between frames (phase correlation) -> deg/s."""
    import cv2  # noqa: PLC0415
    cam.b.write(-60, 0, 1.0)
    time.sleep(2.5)
    small = lambda f: cv2.cvtColor(cv2.resize(f, (320, 180)), cv2.COLOR_BGR2GRAY).astype(np.float32)  # noqa: E731
    cv2.phaseCorrelate(small(frame[0][1]), small(frame[0][1]))   # OpenCV's first call is slow: warm it up
    cam.b.write(60, 0, 1.0)
    t0 = time.perf_counter()
    grabbed, last = [], None
    while time.perf_counter() - t0 < 3.0:                         # collect now, correlate afterwards
        f = frame[0]
        if f is not None and f[0] != last:
            last = f[0]
            grabbed.append((f[0] - t0, small(f[1])))
        time.sleep(0.004)
    samples = [(t1, cv2.phaseCorrelate(g0, g1)[0][0], t1 - t0_) for (t0_, g0), (t1, g1) in zip(grabbed, grabbed[1:])]
    cam.b.write(0, 0, 1.0)
    if len(samples) < 10:
        return None
    fx = 160 / math.tan(math.radians(fov(1.0)[0]) / 2)
    deg = [(t, math.degrees(math.atan(abs(dx) / fx)) / max(dt, 1e-3)) for t, dx, dt in samples]
    peak = max(v for _, v in deg)
    moving = [t for t, v in deg if v > 0.1 * peak]
    with open(out / "speed.json", "w") as fh:
        json.dump({"samples": deg}, fh)
    return peak, (moving[-1] if moving else 0.0)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Insta360 Link 2 PTZ smoke test")
    ap.add_argument("cmd", choices=["smoke", "set", "status", "ai-off"])
    ap.add_argument("--backend", default="mac", choices=["mac", "v4l2", "dry"])
    ap.add_argument("--camera", type=int, default=0, help="OpenCV index of the Link 2 (smoke streams it)")
    ap.add_argument("--no-stream", action="store_true", help="smoke: another app is already streaming the camera")
    ap.add_argument("--out", default="ptz_smoke")
    ap.add_argument("values", nargs="*", type=float, help="set: pan tilt [zoom]")
    a = ap.parse_args()
    if a.cmd == "smoke":
        smoke(a.backend, a.out, a.camera, not a.no_stream)
    else:
        b = make_backend(a.backend)
        if a.cmd == "set":
            b.write(a.values[0], a.values[1], a.values[2] if len(a.values) > 2 else 1.0)
        elif a.cmd == "ai-off":
            b.tracking(False)
        print(b.read())
