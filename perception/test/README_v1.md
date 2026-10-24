# Happy Kermit: PERCEPTION (who to look at)

`track.py`: camera → OpenCV **YuNet** face detector (230 kB ONNX, CPU, no cloud) → main person → bearing in the
arm base frame, streamed as CONTRACT §4 NDJSON.

```
{"t":12.345,"present":true,"yaw_deg":-18.2,"pitch_deg":6.5,"dist_m":1.4,"conf":0.92,"target_base":[1.30,-0.43,0.25],"n_faces":1}
```

- **Geometry:** a pinhole camera built from `--hfov` (default 78°; a C920/C922 is about 70°) or from `--fx --fy --cx --cy`.
  Depth comes from the face-box width, assuming 0.16 m. The face point is moved into the base frame using `--cam-pos`
  (default 0.05 0 0.10 m) and `--cam-rpy` (default 0 0 0 deg). Base frame: +x forward, +y left, +z up, so +yaw is the
  robot's left, which is the **left side of the image**.
- **Main person:** the largest face. The tracker follows it frame to frame by nearest center, and switches only when
  another face is more than 1.3× wider for 0.5 s.
- **Dropouts:** a brief miss holds the last value with conf halved. After 1 s with no face it sends `{"present": false}`.
- **Sources:** `--source 0` (camera index, AVFoundation on the Mac), an HTTP snapshot URL, a video file, or an image glob.
- **Outputs:** stdout NDJSON, `--out file`, `--latest file.json` (atomic latest value), `--udp host:port`.
- **Speed:** the detector runs on a copy downscaled to `--det-width` (default 960) with 2 OpenCV threads.

`../run_kermit.py` runs everything in one process: a camera thread plus the voice loop. Its stdout is a single NDJSON stream
of `{"event":"gaze",…}` at about 20 Hz, interleaved with the voice `segment` and `done` events. The tracker also writes
BRAIN's `context` ("one person to your left, about 1.4 metres away"), and each timeline gets a `person` field.

## What was validated (Linux dev box)
| test | result |
|---|---|
| Live arm camera (`127.0.0.1:8090` snapshot, 640×360, **empty room**) | 120 frames at 30 fps (polling cap), 8 ms per step, **0 false positives**. No person was in view, so **live detection of a real person has not been validated.** |
| Real face photos (Lena, Messi) composited into the live camera frame at known 3D positions: 0.7–3 m, yaw ±35°, pitch 8° | detected 38/42. Misses: one photo at 3 m (face about 22 px wide at 640 px). Yaw error ≤0.4°, pitch ≤1.5°, distance median 3%, max 27% (small faces). |
| Main-person hysteresis (A at 1.5 m; B approaching from 3 m to 1 m) | switched to B 0.74 s after B first exceeded 1.3× A's width (spec: ≥0.5 s) |
| Dropout | faces removed at t=4.5 s → `present:false` at 5.53 s |
| `run_kermit.py` with the live camera and with a composite (person on the right) | gaze at about 20 Hz. BRAIN context became "one person to your right, about 1.6 metres away", and the reply included a look_at. |
| Speed, 1280×720 frame on EPYC 7B13 | det-width 640: 8 ms (4 threads) / 15 ms (1 thread); 960: 17 / 31 ms; 1280: 28 / 53 ms |

Caveats:
- The composite test checks the detector and the code against the same pinhole model. Real-world bearing error is set
  by how well `--hfov` and the camera mounting match reality, plus lens distortion near the edges. Set `--hfov` to the
  real camera's value, and check once by standing at a known angle.
- Distance comes from face size. It's only good to ±25% at 2–3 m and depends on head size and pose. That's fine for
  biasing direction, not for metric reaching.
- The camera FOV is about ±39° at 78°, and a face at ±40° is half out of frame.

## Samples for MOTION
- `samples/walk.ndjson`: synthetic. 15 s at 30 Hz, yaw from -40° to +40° at about 1.5 m. Dropout from 7.0 to 8.5 s:
  the value is held to 8.0 s, then `present:false`.
- `samples/walk_tracked.ndjson`: the same walk, clipped to ±33° and produced by the real tracker on composited frames.
  The video is `test/walk_tracked.mp4`.

## Mac (M5 Max)
```bash
cd kermit/voice && ./setup_mac.sh      # also installs opencv-python and downloads YuNet into perception/models
.venv/bin/python ../perception/track.py --source 0 --hfov 70 --stats      # first run: macOS asks for Camera permission
#   grant it in System Settings › Privacy & Security › Camera for your terminal app (Terminal/iTerm/VS Code), then rerun
bash ../brain/mac/run.sh &                                                  # BRAIN on :8765
.venv/bin/python ../run_kermit.py --camera 0 --hfov 70 --ptt --tts-backend mlx --play
```
- Camera index: 0 is usually the built-in FaceTime camera. A USB webcam is often index 1, and an iPhone Continuity Camera may take an index too.
- Expected speed on the M5 Max (estimate): YuNet at det-width 960 takes about 5–10 ms per frame, so the 30 fps camera is the
  limit. Output is about 20 Hz in `run_kermit.py` (`--gaze-hz`).
- If the built-in camera is used rather than one at the arm base, pass its pose with `--cam-pos x y z` and `--cam-rpy r p y`.
