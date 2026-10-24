"""Live stage for the LIVE ROOM SIM (CONTRACT §6): import this from the room-sim process.

    import sys; sys.path.insert(0, "/var/lib/happy-agent/happy/Bots/bot_2/kermit/motion")
    from live import LiveWorld
    w = LiveWorld(people=[{"name": "kirill", "style": "presenter", "mic": False},
                          {"name": "steve", "style": "judge_b"},
                          {"name": "judge_a", "style": "judge_a", "seated": True}], size=(640, 360), hfov=78)
    w.set_person("kirill", pos_base=[0.75, 0.75], yaw_deg=-135, head_yaw_deg=0, mouth=0.3, talking=True)
    w.kermit_event(json.loads(line))   # every run_kermit stdout line (segment / play / stop / gaze / perception ...)
    w.step(time.perf_counter())        # advance puppet + body physics to that wall time
    rgb = w.render("arm_base")         # (360, 640, 3) uint8 from the arm-base camera
    log.write(json.dumps(w.state()))   # per frame, for the offline re-render

PTZ camera (Insta360 Link 2 model, kermit/camera/ptz.py): ``LiveWorld(..., ptz={"mount": [0.3, 0.3, 0.5]})`` puts it
on a post beside the arm; then render("arm_base") is the PTZ view (what PERCEPTION gets) and the static
camera is render("base_cam"). Drive it with ptz.PTZ(ptz.SimBackend(w)) or w.ptz_command(pan, tilt, zoom);
w.ptz_pose() is the simulated motor pose at the last step, and state()["ptz"] logs it per frame. Like the
real camera it only moves while frames are being rendered from it, parking (tilt -85) otherwise.

Coordinates are the arm base frame: +x forward (toward the audience), +y the robot's left, metres;
the floor is ``floor_z`` (default -0.8: the robot stands on a table). Body yaw 0 faces +x, so a person
facing Kermit from +x has yaw 180. Run with /var/lib/happy-agent/IsaacLab/.venv/bin/python (MuJoCo 3.12, EGL).
"""
from __future__ import annotations

import os
import sys
import time
import warnings
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
warnings.filterwarnings("ignore", message="Attach conflict")
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(1, str(Path(__file__).resolve().parent.parent / "camera"))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from kermit_motion import human as H  # noqa: E402
from kermit_motion.animation import smooth_noise  # noqa: E402
from kermit_motion.live import LiveKermit  # noqa: E402
import ptz as P  # noqa: E402  (kermit/camera/ptz.py)

DEFAULT_LAYOUT = {  # CONTRACT §6 positions, base frame
    "kirill": (0.75, 0.75), "steve": (0.75, -0.75),
    "judge_a": (3.3, -0.9), "judge_b": (3.5, 0.0), "judge_c": (3.3, 0.9),
}

# optical centre, base frame: a ~0.5 m post beside the arm, forward of the base. Directly beside the base
# (x ~ 0) the arm's elbow (link3) blocks the far-side presenter; from here all five §6 faces are clear.
PTZ_DEFAULT = {"mount": [0.3, 0.3, 0.5],
               "mount_yaw": 0.0,            # pan 0 looks along base +x
               "size": None,                # PTZ render size (default: ``size``)
               "park_when_idle": True,      # like the real one: moves only while streaming
               "gimbal": {}}                # ptz.Gimbal overrides (vmax, amax, zoom_rate)


class LiveWorld:
    def __init__(self, people: list[dict], size=(640, 360), hfov: float = 78.0, floor_z: float = -0.8,
                 hand: str = "shadow", view_size=(1280, 720), segment_dir: str | None = None,
                 cam_pitch_deg: float = 0.0, cam_pos=None, ptz: dict | bool | None = None):
        self.people = {p["name"]: dict(p) for p in people}
        ptz = dict(PTZ_DEFAULT, **(ptz if isinstance(ptz, dict) else {})) if ptz else None
        self._config = dict(people=[dict(p, pos_base=list(p["pos_base"])) if "pos_base" in p else dict(p) for p in people],
                            size=list(size), hfov=hfov, floor_z=floor_z, hand=hand, cam_pos=cam_pos,
                            cam_pitch_deg=cam_pitch_deg, ptz=ptz)
        spec = [(p["name"], p.get("style", "presenter"), bool(p.get("mic", False)),
                 self._chair(p) if p.get("seated") else None) for p in people]
        self.k = LiveKermit(hand=hand, people=spec, floor_z=floor_z, base_cam_size=size, view_size=view_size,
                            segment_dir=segment_dir, ptz_mount=ptz["mount"] if ptz else None)
        m = self.k.m
        self.ptz = ptz
        if ptz:
            self._init_ptz(ptz, size)
        cam = m.camera("base_cam").id
        m.cam_fovy[cam] = np.degrees(2 * np.arctan(np.tan(np.radians(hfov / 2)) * size[1] / size[0]))
        # optional upward tilt (+ = up) so standing presenters close to the robot are in view
        a = np.radians(cam_pitch_deg)
        R = np.column_stack([[0, -1, 0], [-np.sin(a), 0, np.cos(a)], [-np.cos(a), 0, -np.sin(a)]])
        q = np.zeros(4)
        mujoco.mju_mat2Quat(q, R.ravel())
        m.cam_quat[cam] = q
        if cam_pos is not None:  # base frame; the camera is a child of link_base at the origin
            m.cam_pos[cam] = np.asarray(cam_pos, float)
        self.floor_z = floor_z
        self.t0 = None
        self.last_t = None
        self.pose: dict[str, dict] = {}
        for p in people:
            xy = p.get("pos_base", DEFAULT_LAYOUT.get(p["name"], (1.5, 0.0)))
            self.set_person(p["name"], pos_base=xy, yaw_deg=np.degrees(np.arctan2(-xy[1], -xy[0])))

    def _init_ptz(self, cfg, size):
        m = self.k.m
        self.ptz_size = tuple(cfg.get("size") or size)
        self.ptz_aspect = self.ptz_size[0] / self.ptz_size[1]
        self.gimbal = P.Gimbal(**{k: tuple(v) if isinstance(v, list) else v for k, v in cfg.get("gimbal", {}).items()})
        self.gimbal.tilt = -85.0 if cfg["park_when_idle"] else 0.0   # powered up, nothing streaming: parked
        self.gimbal.command(self.gimbal.pan, self.gimbal.tilt, 1.0)
        self.ptz_accepted = (0.0, 0.0, 1.0)     # last command the gimbal took (it wakes up to it)
        self.ptz_ignored = 0                    # commands dropped while parked
        self._ptz_render_t = -np.inf
        self._ptz_streaming = False
        self._ptz_ids = [m.body_mocapid[m.body(n).id] for n in ("ptz_yoke", "ptz_head")]
        self._ptz_cam = m.camera("ptz_cam").id
        self._r_ptz = mujoco.Renderer(m, self.ptz_size[1], self.ptz_size[0])
        self._pose_ptz()

    def ptz_command(self, pan: float, tilt: float, zoom: float = 1.0):
        """Absolute pan / tilt (deg, + = left / up) and zoom (1..4), as a UVC write: clamped and quantized to
        1 deg; ignored while nothing is streaming from the camera (park_when_idle)."""
        if self.ptz["park_when_idle"] and not self._ptz_streaming:
            self.ptz_ignored += 1
            return
        self.ptz_accepted = (pan, tilt, zoom)
        self.gimbal.command(pan, tilt, zoom)

    def ptz_pose(self) -> "P.Pose":
        g = self.gimbal
        return P.Pose(t=self.last_t or 0.0, pan=g.pan, tilt=g.tilt, zoom=g.zoom, moving=g.moving, speed=g.speed,
                      mount=tuple(self.ptz["mount"]), mount_yaw=self.ptz["mount_yaw"], src="sim")

    def _step_ptz(self, dt):
        if self.ptz["park_when_idle"]:
            streaming = self.last_t - self._ptz_render_t < 1.0
            if streaming and not self._ptz_streaming:
                self.gimbal.command(*self.ptz_accepted)       # wakes up to its last position
            elif not streaming and self._ptz_streaming:
                self.gimbal.command(self.gimbal.pan, -85.0, self.gimbal.zoom)
            self._ptz_streaming = streaming
        self.gimbal.step(dt)
        self._pose_ptz()

    def _pose_ptz(self):
        g, d, m = self.gimbal, self.k.d, self.k.m
        for i, R in zip(self._ptz_ids, (P.rotation(g.pan, 0.0, self.ptz["mount_yaw"]),
                                        P.rotation(g.pan, g.tilt, self.ptz["mount_yaw"]))):
            q = np.zeros(4)
            mujoco.mju_mat2Quat(q, R.ravel())
            d.mocap_pos[i], d.mocap_quat[i] = self.ptz["mount"], q
        m.cam_fovy[self._ptz_cam] = P.fov(g.zoom, self.ptz_aspect)[1]

    @staticmethod
    def _chair(p):
        xy = np.asarray(p.get("pos_base", DEFAULT_LAYOUT.get(p["name"], (3.4, 0.0))), float)
        yaw = np.arctan2(-xy[1], -xy[0])
        return (xy[0] - 0.02 * np.cos(yaw), xy[1] - 0.02 * np.sin(yaw), yaw)

    # ---- inputs -----------------------------------------------------------------------------------
    def set_person(self, name, pos_base, yaw_deg, head_yaw_deg=0.0, mouth=0.0, talking=False, gesture=None,
                   head_pitch_deg=0.0):
        self.pose[name] = dict(pos=np.asarray(pos_base, float)[:2], yaw=np.radians(yaw_deg),
                               head_yaw=np.radians(head_yaw_deg), head_pitch=np.radians(head_pitch_deg),
                               mouth=float(mouth), talking=bool(talking), gesture=gesture)

    def kermit_event(self, ev: dict):
        kind = ev.get("event")
        if kind in ("gaze", "perception") and ("people" in ev or "yaw_deg" in ev):
            self.k.set_perception(ev)
        elif kind == "gaze" and "target_yaw_deg" in ev:
            self.k.set_gaze_target(ev["target_yaw_deg"], ev.get("target_pitch_deg", 0.0))
        else:
            self.k.feed(ev)

    # ---- stepping ---------------------------------------------------------------------------------
    def step(self, t: float | None = None):
        t = time.perf_counter() if t is None else t
        if self.t0 is None:
            self.t0 = self.last_t = t
            self.k.clock = lambda: self.last_t  # play.t_wall is compared on the caller's clock
        dt = min(max(t - self.last_t, 1e-3), 0.2)  # a stalled caller can't make the physics explode
        self.last_t = t
        self._apply_people(t - self.t0)
        if self.ptz:
            self._step_ptz(dt)   # before the puppet step, whose mj_forward places the mocap bodies
        self.k.step(dt)

    def _apply_people(self, t):
        for name, p in self.pose.items():
            info = self.people[name]
            seated = bool(info.get("seated"))
            q = dict(H.SEATED if seated else H.STANDING)
            q["neck_yaw"] = 0.65 * p["head_yaw"]
            q["lumbar_twist"] = 0.35 * p["head_yaw"]
            q["neck_pitch"] = p["head_pitch"] + 0.03 * smooth_noise(np.array([t]), hash(name) % 97)[0]
            q["lumbar_side"] = 0.02 * smooth_noise(np.array([t]), hash(name) % 89, (0.21, 0.33))[0]
            if p["talking"]:
                g = 0.5 + 0.5 * smooth_noise(np.array([t]), hash(name) % 83, (0.23, 0.37))[0]
                q["neck_pitch"] += 0.08 * (p["mouth"] - 0.4)
                q["sh_flex_l"] = q.get("sh_flex_l", 0) + 0.35 * g + 0.2 * p["mouth"]
                q["elbow_l"] = q.get("elbow_l", 0) + 1.0 * g + 0.25 * p["mouth"]
                q["sh_rot_l"] = -0.3 * g
            if p["gesture"] == "raise_hand":
                q.update(sh_flex_r=2.5, sh_abd_r=0.25, elbow_r=0.5)
            root = [p["pos"][0], p["pos"][1], self.floor_z + (H.SEAT_DROP if seated else 0.0)]
            self.k.set_person(name, root, p["yaw"], q, p["mouth"] if p["talking"] else 0.0)

    # ---- outputs ----------------------------------------------------------------------------------
    def render(self, camera: str = "arm_base") -> np.ndarray:
        """"arm_base": PERCEPTION's camera (the PTZ when there is one); "base_cam": the static §4 camera."""
        if self.ptz and camera in ("arm_base", "ptz"):
            self._ptz_render_t = self.last_t if self.last_t is not None else -np.inf
            self._r_ptz.update_scene(self.k.d, "ptz_cam")
            return self._r_ptz.render()
        if camera in ("arm_base", "base_cam"):
            return self.k.render_base_camera()
        return self.k.render(camera)

    def config(self) -> dict:
        """Constructor arguments. Log once as the first line of world_state.ndjson: {"config": w.config()}."""
        return dict(self._config)

    def state(self) -> dict:
        d = self.k.d
        return {"t": self.last_t, "arm": [round(float(x), 5) for x in self.k.q],
                "mocap_pos": np.round(d.mocap_pos, 4).tolist(), "mocap_quat": np.round(d.mocap_quat, 5).tolist(),
                "mouths": {n: float(p["mouth"]) if p["talking"] else 0.0 for n, p in self.pose.items()},
                "hand": [round(float(x), 5) for x in self.k.hq],
                "qpos": [round(float(x), 5) for x in d.qpos],
                "people": {n: {"pos": [float(p["pos"][0]), float(p["pos"][1])], "yaw_deg": float(np.degrees(p["yaw"])),
                               "head_yaw_deg": float(np.degrees(p["head_yaw"])), "mouth": p["mouth"],
                               "talking": p["talking"], "gesture": p["gesture"]} for n, p in self.pose.items()},
                **({"ptz": self.ptz_pose().to_json(self.ptz_aspect)} if self.ptz else {})}

    def close(self):
        if self.ptz:
            self._r_ptz.close()
        self.k.close()


if __name__ == "__main__":  # smoke test + timing + arm-cam sample frame
    import imageio
    w = LiveWorld(people=[{"name": "kirill", "style": "presenter", "mic": True},
                          {"name": "steve", "style": "judge_b"},
                          {"name": "judge_a", "style": "judge_a", "seated": True},
                          {"name": "judge_b", "style": "judge_c", "seated": True},
                          {"name": "judge_c", "style": "judge_a", "seated": True}])
    w.set_person("kirill", [0.75, 0.75], -135, head_yaw_deg=-20, mouth=0.5, talking=True)
    w.set_person("steve", [0.75, -0.75], 135, head_yaw_deg=20)
    w.kermit_event({"event": "gaze", "speaker_id": 1, "people": [{"id": 1, "target_base": [0.75, 0.75, 0.85]}]})
    t = time.perf_counter()
    times = []
    for i in range(120):
        t0 = time.perf_counter()
        w.step(t + i / 30)
        rgb = w.render("arm_base")
        times.append(time.perf_counter() - t0)
    out = Path(__file__).resolve().parent / "videos/live"
    out.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(out / "armcam_sample.png", rgb)
    imageio.imwrite(out / "view_sample.png", w.render("front34"))
    print(f"step+render: median {1000 * np.median(times):.1f} ms, p95 {1000 * np.percentile(times, 95):.1f} ms "
          f"-> {1 / np.median(times):.0f} fps; arm {np.degrees(w.k.q).round(1)}")
    w.close()
