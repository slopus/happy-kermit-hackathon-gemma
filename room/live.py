"""One closed-loop, wall-clock run of the live room (CONTRACT §6).

  python3 live.py --name normal_1 --noise normal [--steps 1-7] [--camera] [--brain-url http://127.0.0.1:18765]

Starts, in this order:
  1. room/server.py      (room venv)      the room: humans, noise, RIRs, ground truth; the wall-clock master
  2. room/camera.py      (MuJoCo venv)    optional: MOTION's LiveWorld, arm-cam snapshots on :8091, closed-loop puppet
  3. run_kermit.py       (voice venv)     UNCHANGED, with `sounddevice` resolved to the room (room/shim):
                                          mic = the room's mic, speaker = Kermit's loudspeaker in the room,
                                          perception = people.py (camera, or --oracle truth.ndjson)
BRAIN (llama-server + brain.py serve) must already be running; it is reset at the start of each run.
Every stdout line of run_kermit is stamped with the perf_counter time it arrived → run/<name>/kermit.ndjson.
"""
import argparse, json, os, signal, subprocess, sys, threading, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
K = os.path.dirname(HERE)
PY_ROOM = os.path.join(HERE, ".venv", "bin", "python")
PY_VOICE = os.path.join(K, "voice", ".venvs", "kokoro", "bin", "python")
PY_MUJOCO = os.environ.get("KERMIT_MUJOCO_PY", "/var/lib/happy-agent/IsaacLab/.venv/bin/python")  # any python with mujoco 3.x + EGL


def post(url, obj):
    req = urllib.request.Request(url, data=json.dumps(obj).encode(), headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=5).read()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True)
    ap.add_argument("--noise", default="normal")
    ap.add_argument("--input", default="A", choices=["A", "B"], help="A = table mic, B = close-talk handhelds")
    ap.add_argument("--steps", default="")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--camera", action="store_true", help="run MOTION's live sim camera (closed-loop puppet)")
    ap.add_argument("--perception", default="oracle", choices=["oracle", "camera", "none"])
    ap.add_argument("--brain-url", default="http://127.0.0.1:18765")
    ap.add_argument("--gpu", default="0")
    ap.add_argument("--ptz", choices=["hub", "camera"], help="Link 2 PTZ aimed at the speaker instead of the static 120° camera: "
                    "hub = run_kermit --ptz room (the real wiring), camera = director inside room/camera.py")
    ap.add_argument("--taskset", help="pin run_kermit to these CPUs, e.g. 0-7")
    ap.add_argument("--kermit-args", nargs=argparse.REMAINDER, default=[])
    a = ap.parse_args()
    run = os.path.join(HERE, "run", a.name)
    if os.path.exists(run):
        import shutil; shutil.rmtree(run)
    os.makedirs(run)
    procs = []

    def start(cmd, log, env=None):
        p = subprocess.Popen(cmd, stdout=open(os.path.join(run, log), "w"), stderr=subprocess.STDOUT,
                             env={**os.environ, **(env or {})}, cwd=HERE)
        procs.append(p); return p

    post(a.brain_url + "/reset", {})
    # which code this run used (the other teams edit live; attribute each iteration)
    files = ["run_kermit.py", "turn/endpoint.py", "perception/people.py", "perception/oracle.py", "brain/brain.py",
             "voice/kermit_voice.py", "motion/live.py", "room/server.py"]
    json.dump({"files": {f: time.strftime("%H:%M:%S", time.gmtime(os.path.getmtime(os.path.join(K, f))))
                         for f in files if os.path.exists(os.path.join(K, f))},
               "env": {k: v for k, v in os.environ.items() if k.startswith("KERMIT_")}, "taskset": a.taskset,
               "load": os.getloadavg()}, open(os.path.join(run, "code_versions.json"), "w"), indent=1)
    # 1. room
    room = start([PY_ROOM, "server.py", "--run", run, "--noise", a.noise, "--input", a.input, "--seed", str(a.seed)] + (["--steps", a.steps] if a.steps else []) + (["--ptz"] if a.ptz else []), "room.log")
    for _ in range(300):
        if os.path.exists(os.path.join(run, "truth.ndjson")): break
        time.sleep(0.1)
    time.sleep(0.5)
    kout = os.path.join(run, "kermit.ndjson")
    open(kout, "w").close()
    # 2. camera
    cam_url = None
    if a.camera:
        start([PY_MUJOCO, "camera.py", "--run", run] + ({"hub": ["--ptz-external"], "camera": ["--ptz"]}.get(a.ptz, [])), "camera.log", env={"MUJOCO_GL": "egl", "CUDA_VISIBLE_DEVICES": a.gpu})
        cam_url = "http://127.0.0.1:8091/snapshot.jpg"
    # 3. run_kermit, unchanged; sounddevice → the room
    kargs = (["taskset", "-c", a.taskset] if a.taskset else []) + [PY_VOICE, "-u", os.path.join(K, "run_kermit.py"), "--brain-url", a.brain_url, "--out", os.path.join(run, "kermit_out"),
             "--voice", "am_fenrir", "--tts-backend", "torch", "--device", "cuda"]
    if a.perception != "none":
        kargs += ["--camera", cam_url or "oracle", "--hfov", "120"]
    if a.ptz == "hub":
        kargs += ["--ptz", "room", "--ptz-dir", run]
    kargs += a.kermit_args
    if a.perception == "oracle":
        kargs += ["--perception-args", "--oracle", os.path.join(run, "truth.ndjson"), "--cam-pos", "0.10", "0.12", "0.55",
                  "--latest", os.path.join(run, "people_latest.json")]
    env = {"PYTHONPATH": os.path.join(HERE, "shim"), "CUDA_VISIBLE_DEVICES": a.gpu, "HF_HUB_OFFLINE": "1",
           "HF_HOME": os.path.join(K, "voice", ".hf"), "KERMIT_ROOM_SOCK": "127.0.0.1:18766"}
    kp = subprocess.Popen(kargs, stdout=subprocess.PIPE, stderr=open(os.path.join(run, "kermit.stderr"), "w"),
                          env={**os.environ, **env}, cwd=K, text=True, bufsize=1)
    procs.append(kp)

    def pump():
        with open(kout, "a", buffering=1) as f:
            for line in kp.stdout:
                t = time.perf_counter()
                line = line.strip()
                if not line: continue
                try:
                    ev = json.loads(line); ev["t_rx"] = round(t, 4)
                    f.write(json.dumps(ev, separators=(",", ":")) + "\n")
                except ValueError:
                    f.write(json.dumps({"t_rx": round(t, 4), "raw": line[:500]}) + "\n")
    threading.Thread(target=pump, daemon=True).start()
    print(f"run {a.name}: room pid {room.pid}, run_kermit pid {kp.pid}", flush=True)
    try:
        room.wait()
    finally:
        time.sleep(1.0)
        for p in procs[1:-1]:          # camera: let it close its mp4 (it exits when engine.json appears)
            try: p.wait(15)
            except subprocess.TimeoutExpired: pass
        for p in procs[::-1]:
            if p.poll() is None:
                p.send_signal(signal.SIGTERM)
        time.sleep(2)
        for p in procs:
            if p.poll() is None: p.kill()
    print(f"run {a.name} done → {run}", flush=True)
    if os.path.exists(os.path.join(HERE, "score.py")):
        subprocess.run([PY_ROOM, os.path.join(HERE, "score.py"), run])


if __name__ == "__main__":
    main()
