"""No-hardware checks for ptz.py / aim.py:  python test_ptz.py   (any Python 3.10+ with numpy)

Fake `v4l2-ctl`, `linkctl` and `linkcam` executables record their arguments, so the real backends' command
lines, rate limits and threading are exercised exactly as they would run against a camera.
"""
import json
import math
import os
import stat
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aim import AimPolicy  # noqa: E402
from ptz import PTZ, DryRunBackend, LinkcamBackend, Pose, V4L2Backend, fov, pixel_to_base  # noqa: E402

FAKE = """#!/bin/sh
echo "$(basename $0) $*" >> {log}
case "$*" in
  *get-ctrl*) printf 'pan_absolute: -108000\\ntilt_absolute: 36000\\nzoom_absolute: 150\\n' ;;
  *status*) echo '{{"pan": 30, "tilt": 10, "zoom": 1.5, "ai": false}}' ;;
esac
"""


def fake_bins(tmp: Path) -> Path:
    log = tmp / "calls.log"
    for name in ("v4l2-ctl", "linkctl", "linkcam"):
        p = tmp / name
        p.write_text(FAKE.format(log=log))
        p.chmod(p.stat().st_mode | stat.S_IEXEC)
    os.environ["PATH"] = f"{tmp}:{os.environ['PATH']}"
    return log


def run(cam: PTZ, seconds: float, hz: float = 60):
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        cam.update()
        time.sleep(1 / hz)


def main():
    # field of view: 79.5 deg diagonal at 16:9 -> ~72 x 44 deg; zoom 2 halves tan(fov/2)
    h, v = fov(1.0)
    assert abs(h - 71.9) < 0.3 and abs(v - 44.4) < 0.3, (h, v)
    assert abs(math.tan(math.radians(fov(2.0)[0] / 2)) * 2 - math.tan(math.radians(h / 2))) < 1e-9
    # the contract: yaw = pan + bearing_in_image, exact at tilt 0
    p0 = Pose(0, pan=30, tilt=0, zoom=1)
    yaw, _ = pixel_to_base(160, 180, p0, 640, 360)
    b_img = math.degrees(math.atan((320 - 160) / (320 / math.tan(math.radians(h / 2)))))
    assert abs(yaw - (30 + b_img)) < 1e-6, (yaw, b_img)
    yaw20, pitch20 = pixel_to_base(160, 180, Pose(0, pan=30, tilt=20, zoom=1), 640, 360)
    print(f"contract: tilt 0 exact; tilt 20 -> yaw {yaw20:.2f} vs pan+bearing {30 + b_img:.2f}, pitch {pitch20:.2f}")

    # dry run: limits, quantization, smoothing, rate limit, pose_at
    b = DryRunBackend()
    cam = PTZ(b)
    cam.set(200, -120, 9)
    assert cam.goal == (145.0, -90.0, 4.0), cam.goal
    cam.set(40.4, 10.0, 1.0)
    run(cam, 1.6)
    sets = [r for r in b.sent if r["cmd"] == "set"]
    assert all(float(r["pan"]).is_integer() for r in sets), "1-deg quantization"
    gaps = [b2["t"] - a2["t"] for a2, b2 in zip(sets, sets[1:])]
    assert min(gaps) >= 0.049, f"rate limit {min(gaps)}"
    steps = [abs(b2["pan"] - a2["pan"]) for a2, b2 in zip(sets, sets[1:])]
    assert max(steps) <= 6, f"no jumps reach the gimbal: max step {max(steps)}"
    assert sets[-1]["pan"] == 40 and abs(cam.get().pan - 40) < 0.5, (sets[-1], cam.get())
    mid = cam.history[len(cam.history) // 3]
    assert abs(cam.pose_at(mid.t).pan - mid.pan) < 1e-9
    n = len(sets)
    run(cam, 1.2)
    assert len([r for r in b.sent if r["cmd"] == "set"]) == n + 1, "one keep-alive resend after 1 s idle"
    print(f"dry: {n} writes for a 40 deg move, max step {max(steps):.0f} deg, min gap {min(gaps) * 1000:.0f} ms, keep-alive ok")

    with tempfile.TemporaryDirectory() as tmp:
        log = fake_bins(Path(tmp))
        # linux v4l2: arc-seconds, UVC pan sign flipped, zoom x100; tracking via linkctl
        v4 = V4L2Backend("/dev/video2")
        cam = PTZ(v4, pose_log=Path(tmp) / "pose.ndjson", latest=Path(tmp) / "pose.json")
        cam.tracking(False)
        cam.set(30, 10, 1.5)
        run(cam, 1.5)
        cam.close()
        calls = log.read_text().splitlines()
        assert calls[0] == "linkctl track off", calls[0]
        last = [c for c in calls if "set-ctrl" in c][-1]
        assert "pan_absolute=-108000,tilt_absolute=36000" in last and "-d /dev/video2" in last, last
        assert any("zoom_absolute=150" in c for c in calls)
        assert v4.read() == (30.0, 10.0, 1.5)
        j = json.loads((Path(tmp) / "pose.json").read_text())
        assert {"t", "pan_deg", "tilt_deg", "zoom", "hfov_deg", "moving", "mount"} <= j.keys()
        print("v4l2:", last, "| readback", v4.read())
        log.write_text("")
        # mac linkcam: degrees, ai off, zoom only when it changes, slow rate (process per call)
        lc = LinkcamBackend(str(Path(tmp) / "linkcam"))
        cam = PTZ(lc)
        cam.tracking(False)
        cam.set(-50, 5, 1.0)
        run(cam, 2.0)
        cam.close()
        calls = log.read_text().splitlines()
        sets = [c for c in calls if c.startswith("linkcam set")]
        assert calls[0] == "linkcam ai off" and sets[-1] == "linkcam set 50 5", (calls[0], sets[-1])
        assert len(sets) <= 2.0 / lc.min_period + 1, len(sets)
        assert lc.read() == (-30.0, 10.0, 1.5)
        print(f"linkcam: {len(sets)} set calls in 2 s, last '{sets[-1]}' (pan_sign -1), status parsed {lc.read()}")

    # aim policy: hysteresis and the stale-speaker guard
    pol = AimPolicy()
    pol._survey_i = None
    k = {"id": 1, "target_base": [0.75, 0.75, 0.85], "speaking_p": 0.9, "role": "presenter"}
    s = {"id": 2, "target_base": [0.75, -0.75, 0.85], "speaking_p": 0.05, "role": "presenter"}
    pol.observe({"speaker_id": 1, "speech_active": True, "people": [k, s]}, 0.0)
    g0, why = pol.step(0.0)
    assert why == "speaker 1", why
    pol.observe({"speaker_id": 2, "speech_active": True, "people": [dict(k, speaking_p=0.1), dict(s, speaking_p=0.9)]}, 0.3)
    assert pol.step(0.3)[0] == g0, "a new speaker must hold 0.6 s and the shot must last 1.5 s"
    pol.observe({"speaker_id": 2, "speech_active": True, "people": [dict(k, speaking_p=0.1), dict(s, speaking_p=0.9)]}, 1.6)
    assert pol.step(1.6)[1] == "speaker 2"
    k2 = dict(k, target_base=[0.75, 0.80, 0.85])  # small sway: inside the deadband, no re-aim
    pol.observe({"speaker_id": 2, "speech_active": True, "people": [k2, dict(s, target_base=[0.78, -0.73, 0.86])]}, 2.0)
    g1 = pol.step(2.0)[0]
    pol.observe({"speaker_id": 2, "speech_active": True, "people": [k2, dict(s, target_base=[0.80, -0.70, 0.87])]}, 2.2)
    assert pol.step(2.2)[0] == g1, "deadband"
    print("aim: switch hysteresis, min dwell and deadband ok")
    print("all PTZ checks passed")


if __name__ == "__main__":
    main()
