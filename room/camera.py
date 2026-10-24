"""Live sim camera for the room: MOTION's LiveWorld stepped in wall-clock time, closed loop.

  /var/lib/happy-agent/IsaacLab/.venv/bin/python camera.py --run run/<name> [--fps 30] [--port 8091]

Inputs (tailed live from the run dir):  people.ndjson  (the room driver's pose commands, 30 Hz)
                                        kermit.ndjson  (run_kermit's stdout: gaze / segment / play / stop …)
So the puppet in the sim moves as the live loop decided (gaze from perception, mouth/gestures from what is
actually playing), and the people as the script drives them. Outputs:
  http://127.0.0.1:<port>/snapshot.jpg   latest arm-base frame (header X-Frame-T: perf_counter of the step)
  world_state.ndjson                     LiveWorld.state() per frame (first line: {"config": …}) for MOTION's re-render
  armcam.mp4 + armcam_times.json         the arm-cam stream as rendered live (frame times on perf_counter)
"""
import argparse, io, json, os, signal, subprocess, sys, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "motion"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "camera"))
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np
from PIL import Image
from live import LiveWorld

STYLES = {"kirill": ("presenter", False), "steve": ("judge_b", False), "judge1": ("judge_c", True),
          "judge2": ("judge_a", True), "judge3": ("judge_c", True)}
POS = {"kirill": (0.75, 0.75), "steve": (0.75, -0.75), "judge1": (3.3, 0.9), "judge2": (3.5, 0.0), "judge3": (3.3, -0.9)}


class Tail:
    def __init__(self, path):
        self.path, self.f, self.buf = path, None, ""

    def lines(self):
        if self.f is None:
            if not os.path.exists(self.path): return []
            self.f = open(self.path)
        self.buf += self.f.read()
        if "\n" not in self.buf: return []
        *done, self.buf = self.buf.split("\n")
        out = []
        for l in done:
            try: out.append(json.loads(l))
            except ValueError: pass
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--port", type=int, default=8091)
    ap.add_argument("--size", default="640x360")
    ap.add_argument("--hfov", type=float, default=120.0)
    ap.add_argument("--cam-pos", type=float, nargs=3, default=[0.10, 0.12, 0.55])
    ap.add_argument("--cam-pitch", type=float, default=-3.0)
    ap.add_argument("--ptz-external", action="store_true", help="PTZ driven by run_kermit --ptz room (ptz_hook.py): apply "
                    "<run>/ptz_cmd.json, publish the simulated pose to <run>/ptz_pose_sim.json")
    ap.add_argument("--ptz", action="store_true", help="Link 2 PTZ on its post (camera/), aimed by PTZDirector from "
                    "PERCEPTION's latest value (run/<name>/people_latest.json); arm_base render = the PTZ view")
    a = ap.parse_args()
    W, H = map(int, a.size.split("x"))
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))   # run the finally: close the mp4
    w = LiveWorld(people=[{"name": n, "style": s, "mic": False, "seated": seated, "pos_base": POS[n]} for n, (s, seated) in STYLES.items()],
                  size=(W, H), hfov=a.hfov, cam_pos=a.cam_pos, cam_pitch_deg=a.cam_pitch, ptz=True if (a.ptz or a.ptz_external) else None)
    cmd_path, cmd_m, pose_sim = os.path.join(a.run, "ptz_cmd.json"), [0.0], os.path.join(a.run, "ptz_pose_sim.json")
    director, plat = None, None
    if a.ptz:
        from ptz import PTZ, SimBackend
        from aim import PTZDirector
        director = PTZDirector(PTZ(SimBackend(w), pose_log=os.path.join(a.run, "ptz_pose.ndjson"),
                                   latest=os.path.join(a.run, "ptz_pose.json")), log=os.path.join(a.run, "aim.ndjson"))
        plat = os.path.join(a.run, "people_latest.json"); plat_m = [0.0]
    latest = {"jpg": None, "t": 0.0}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if latest["jpg"] is None:
                self.send_response(503); self.end_headers(); return
            self.send_response(200); self.send_header("Content-Type", "image/jpeg")
            self.send_header("X-Frame-T", f"{latest['t']:.4f}"); self.send_header("Content-Length", str(len(latest["jpg"])))
            self.end_headers(); self.wfile.write(latest["jpg"])

        def log_message(self, *x): pass

    srv = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    people, kerm = Tail(os.path.join(a.run, "people.ndjson")), Tail(os.path.join(a.run, "kermit.ndjson"))
    ws = open(os.path.join(a.run, "world_state.ndjson"), "w", buffering=1)
    ws.write(json.dumps({"config": w.config()}) + "\n")
    ff = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(a.fps),
                           "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
                           os.path.join(a.run, "armcam.mp4")], stdin=subprocess.PIPE)
    times, step_ms = [], []
    period = 1.0 / a.fps
    t_next = time.perf_counter()
    done_flag = os.path.join(a.run, "engine.json")
    print("camera up", flush=True)
    try:
        while not os.path.exists(done_flag):
            t_next += period
            for fr in people.lines():
                for p in fr.get("people", []):
                    w.set_person(p["name"], p["pos_base"], p["yaw_deg"], head_yaw_deg=p["head_yaw_deg"], mouth=p["mouth"],
                                 talking=p["talking"], gesture=p.get("gesture"))
            for ev in kerm.lines():
                if "event" in ev:
                    try: w.kermit_event(ev)
                    except Exception as e: print("kermit_event error", repr(e)[:200], flush=True)
                    if director:
                        director.on_kermit_event(ev)
            if director:
                try:
                    m = os.path.getmtime(plat) if os.path.exists(plat) else 0
                    if m != plat_m[0]:
                        plat_m[0] = m; director.on_perception(json.load(open(plat)))
                except (ValueError, OSError):
                    pass
                director.tick()
            if a.ptz_external:
                try:
                    m = os.path.getmtime(cmd_path) if os.path.exists(cmd_path) else 0
                    if m != cmd_m[0]:
                        cmd_m[0] = m; c = json.load(open(cmd_path)); w.ptz_command(c["pan"], c["tilt"], c["zoom"])
                except (ValueError, OSError, KeyError):
                    pass
            t = time.perf_counter(); w.step(t); img = w.render("arm_base")
            if a.ptz or a.ptz_external:
                open(pose_sim + ".tmp", "w").write(json.dumps(w.ptz_pose().to_json())); os.replace(pose_sim + ".tmp", pose_sim)
            step_ms.append((time.perf_counter() - t) * 1000)
            b = io.BytesIO(); Image.fromarray(img).save(b, "JPEG", quality=85)
            latest["jpg"], latest["t"] = b.getvalue(), t
            ff.stdin.write(np.ascontiguousarray(img).tobytes()); times.append(round(t, 4))
            st = w.state(); ws.write(json.dumps(st, separators=(",", ":")) + "\n")
            dt = t_next - time.perf_counter()
            if dt > 0: time.sleep(dt)
            else: t_next = time.perf_counter()
    finally:
        ff.stdin.close(); ff.wait()
        sm = np.array(step_ms or [0])
        json.dump({"fps_target": a.fps, "frames": len(times), "times": times,
                   "step_render_ms_mean": round(float(sm.mean()), 2), "step_render_ms_p95": round(float(np.percentile(sm, 95)), 2),
                   "effective_fps": round(len(times) / max(1e-6, times[-1] - times[0]), 2) if len(times) > 1 else 0},
                  open(os.path.join(a.run, "armcam_times.json"), "w"))
        print("camera done", flush=True)


if __name__ == "__main__":
    main()
