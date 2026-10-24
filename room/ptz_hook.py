"""run_kermit.py --ptz <backend>: the Link 2 PTZ hub hook (kermit/camera/README.md "Live wiring").

  python run_kermit.py --camera 0 --ptz mac  [--ptz-mount 0.3 0.3 0.5] [--ptz-dir out/]   # Mac, real Link 2
  python run_kermit.py --camera http://… --ptz room --ptz-dir room/run/<name>             # live room sim

What it wires (all in the hub process, one perf_counter clock):
  PERCEPTION v2  pt.set_ptz_source(ptz.pose_at)  → each frame is interpreted with the pose at its capture time
  §5 latest      PTZDirector.on_perception()     ← polled from the PeopleThread at the tick rate
  stdout events  PTZDirector.on_kermit_event()   ← play / stop / done / segment / camera_mode, plus turn_candidate /
                                                    heard as "the room is talking" (Kermit's lines never move the camera)
  30 Hz          PTZDirector.tick()              → aim policy → rate-limited setpoint → backend
Backends: mac (linkcam CLI), v4l2, dry (log only), and room: the sim camera process (room/camera.py --ptz-external)
owns the simulated gimbal; commands go through <ptz-dir>/ptz_cmd.json and the actual simulated pose comes back
through <ptz-dir>/ptz_pose_sim.json.
"""
import json, os, sys, threading, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "camera"))
import ptz as P  # noqa: E402
from aim import PTZDirector  # noqa: E402


class RoomBackend(P.Backend):
    """The live room sim's PTZ, which lives in room/camera.py (MuJoCo) — a file-based command/readback pair."""
    name = "room"
    min_period = 0.0
    reports_actual = True

    def __init__(self, run_dir):
        self.cmd = os.path.join(run_dir, "ptz_cmd.json"); self.pose = os.path.join(run_dir, "ptz_pose_sim.json")
        self._last = (0.0, 0.0, 1.0)

    def write(self, pan, tilt, zoom, zoom_changed=True):
        tmp = self.cmd + ".tmp"
        open(tmp, "w").write(json.dumps({"t": time.perf_counter(), "pan": pan, "tilt": tilt, "zoom": zoom}))
        os.replace(tmp, self.cmd)

    def read(self):
        try:
            p = json.load(open(self.pose)); self._last = (p["pan_deg"], p["tilt_deg"], p["zoom"])
        except (OSError, ValueError, KeyError):
            pass
        return self._last

    def tracking(self, on):
        if on: raise RuntimeError("the sim has no built-in tracking")


def attach(hub, emit_taps, hz=30.0):
    a = hub.a
    d = a.ptz_dir or a.out
    os.makedirs(d, exist_ok=True)
    backend = RoomBackend(d) if a.ptz == "room" else P.make_backend(a.ptz)
    cam = P.PTZ(backend, mount=tuple(a.ptz_mount), pose_log=os.path.join(d, "ptz_pose.ndjson"),
                latest=os.path.join(d, "ptz_pose.json"))
    director = PTZDirector(cam, log=os.path.join(d, "aim.ndjson"))
    v2 = getattr(hub.percep, "v2", None)
    if v2 is not None and hasattr(v2, "set_ptz_source"):
        v2.set_ptz_source(cam.pose_at)
    lock = threading.Lock()

    def tap(obj):
        if isinstance(obj, str):
            try: obj = json.loads(obj)
            except ValueError: return
        if isinstance(obj, dict) and obj.get("event") in ("play", "stop", "done", "segment", "camera_mode",
                                                                "turn_candidate", "heard"):
            with lock: director.on_kermit_event(obj)
    emit_taps.append(tap)

    def loop():
        last = None
        while True:
            t = time.perf_counter()
            latest = v2.latest if v2 is not None else None
            with lock:
                if latest is not None and latest is not last:
                    director.on_perception(latest); last = latest
                try:
                    director.tick()
                except Exception as e:  # noqa: BLE001 - a camera hiccup must never take the hub down
                    print(f"ptz: {e!r}"[:200], file=sys.stderr)
            time.sleep(max(0.0, 1.0 / hz - (time.perf_counter() - t)))
    threading.Thread(target=loop, daemon=True).start()
    print(f"ptz: {a.ptz} backend, mount {a.ptz_mount}, logs in {d}", file=sys.stderr)
    return director
