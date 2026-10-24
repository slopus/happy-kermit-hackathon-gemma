# Kermit's PTZ camera: Insta360 Link 2

PERCEPTION decides who is speaking (CONTRACT §5) and the Link 2 turns to frame them. The camera's
built-in AI tracking stays **off**. Everything was built and tested without hardware: in the MuJoCo sim,
against a dry-run backend, and against fake `v4l2-ctl`/`linkcam` binaries. **Nothing here has driven a
real Link 2 yet.** The Mac runbook below is that first test.

| file | what it is |
|---|---|
| `ptz.py` | controller: `PTZ(backend)` with `set(pan, tilt, zoom)`, `get()`, `tracking(on)`, `update()` and `pose_at(t)`. Backends: `v4l2` (Linux), `mac` (linkcam CLI), `dry`, and `SimBackend(world)`. Also `fov()`, `pixel_to_base()`, the `Gimbal` motor model, and the `smoke` test |
| `aim.py` | `AimPolicy` (speaker → framing, with hysteresis) and `PTZDirector`, which glues the policy to the hub. Can also run as a standalone daemon that polls PERCEPTION's latest-value file |
| `sim_demo.py` | closed-loop sim of §6 beats in LiveWorld, with an FOV-limited perception stand-in. Output: `videos/ptz_aim_demo.mp4` |
| `test_ptz.py` | no-hardware checks: limits, quantization, rate limit, keep-alive, both backends' command lines, policy hysteresis |
| `../motion/live.py` | `LiveWorld(..., ptz=True)`: the Link 2 on a post in the room sim (see "Sim") |

## Conventions and the pose stream

All angles are in the arm base frame, as in §4/§5:
- **pan:** + = left (toward base +y); 0 = the mount's forward direction (base +x).
- **tilt:** + = up.
- **zoom:** 1–4×.

The camera is limited to the Link 2's measured travel: pan ±145°, tilt −90…+100°, in 1° steps.

The FOV is 79.5° diagonal, which gives **71.9° × 44.4° at 16:9 and zoom 1**. tan(fov/2) scales as 1/zoom.

Camera rotation: `R = Rz(mount_yaw + pan) · Ry(−tilt)`, with the optical axis on body +x. In
`perception/track.py` terms that is `Camera(pos=mount, rpy=(0, -tilt, mount_yaw + pan))` with `fx = fx_1x · zoom`.

Every `update()` (≥30 Hz) publishes one pose:
- `ptz_pose.json`: latest value, atomically replaced
- `ptz_pose.ndjson`: log
- `PTZ.pose_at(t)`: in-process lookup

```json
{"t": 189491.12, "t_unix": 1792842437.77, "pan_deg": -65.0, "tilt_deg": -20.0, "zoom": 1.0,
 "hfov_deg": 71.88, "vfov_deg": 44.37, "moving": false, "speed_dps": 0.0,
 "mount": [0.3, 0.3, 0.5], "mount_yaw_deg": 0.0, "src": "model", "tracking": false}
```

- **`t`:** `time.perf_counter()`. It is system-wide monotonic on macOS and Linux, and it is the same clock as the room sim's `X-Frame-T`. Stamp each frame with its capture time and use `pose_at(frame_t)` or interpolate the NDJSON. Don't use the latest pose: the camera may have moved since.
- **`src`:** where the pose comes from.
  - `"model"`: for real hardware. Over UVC the Link 2 reads back the last *command*, not where the motors are, so the pose comes from a motor model (`Gimbal`) fed with the commands actually sent.
  - `"sim"`: the simulated motors.
- **`moving` / `speed_dps`:** frames taken while the camera turns faster than ~30°/s are blurred and their pose is less certain. Lower their confidence or skip ASD on them.

### Contract for PERCEPTION

For a face at horizontal angle `bearing_in_image` from the image centre (+ = left), use
`bearing_base = pan + bearing_in_image`, timestamped with the frame time.
- This is exact at tilt 0 and within ~1.6° for |tilt| ≤ 20°.
- The exact version is `ptz.pixel_to_base(u, v, pose, W, H)`. The same thing is `Camera.face_to_base` with the per-frame `R`.
- This bearing is measured **from the camera's optical centre at `mount`**, not from the arm base. With the post 0.4 m away, the parallax for presenters 1 m away is large. So still compute `target_base = mount + R · p_cam` from face size, as `track.Camera` does now, and report §5 `yaw_deg`/`pitch_deg` from that point.

## Aim policy (`aim.py`)

**presentation**
- Frames the current `speaker_id` with the face at about ⅔ of the frame height, at zoom 1. Zoom stays at 1 so perception keeps the hands, the mic and its context.
- **Hysteresis.** A new speaker must hold for 0.6 s, and the current shot must have lasted 1.5 s, before the camera cuts.
- **Deadband.** For the same person, the camera re-aims only when the error exceeds 18% of the HFOV, or 8% held for 1.5 s.
- **Stale speaker_id.** If speaker_id points at someone who is in view but visibly not talking (`speaking_p < 0.5` and no change for 2 s), it is not a reason to cut.

**Off-frame speaker.** Kermit isn't talking, and one of these holds:
1. **PERCEPTION's `unseen_speech` for 0.5 s.** This is the main trigger. It means voice was present while no visible face's lips moved. It only counts once the slew is over and 1.5 s more have passed, because PERCEPTION's window looks back over the previous view.
2. **The person in frame is visibly not talking.** The policy uses that face's own lip score: the ~1 s mean of `asd` below −0.9, the midpoint of PERCEPTION's measured talking/silent logit distributions (−0.32±0.74 / −1.52±0.76). `speaking_p < 0.25` is only the fallback when `asd` is missing. This holds for 1 s while the room talks: `speech_active`, the hub's STT `turn_candidate`/`heard`, or `on_vad()`.
3. **Patrol:** 2 holds for 4 s with no live speech signal in the last 60 s.

The fused `speaking_p` is not trusted for this decision. On a narrow view it carried an "alone in view + voice" prior, which locked the live run onto one person.

It then visits the other known presenters, then the audience group shot, then returns. A fresh speaker_id on the way ends the search, and so does a face whose 1 s lip mean clears −0.9 (or `speaking_p ≥ 0.5` once `unseen_speech` has cleared). A re-ID of the framed person, meaning a new id at the same bearing (±8°), counts as the same shot. A judge only switches the show to qa when their lips visibly move. After a fruitless search it waits 6 s before searching again.

**survey.** Runs at start-up, every 8 s while nobody is known (for example, the camera came up before the room), and whenever someone talks but no presenter is known. The camera visits +45°, −60° and the audience, each for 1.2 s after the slew, and stops as soon as it finds the speaker.

**qa.** One held group shot over the audience zone (`audience_pan`, set it at the venue) and every known judge, at up to 2× zoom so far faces get more pixels.
- The shot is re-framed only if a judge leaves its inner 85%.
- A judge becoming `speaker_id` enters qa; BRAIN can also send `{"event": "camera_mode", "mode": "qa"}`.
- A presenter talking for 3 s leaves qa.
- `mode` also accepts `hold` and `home`.

**Memory.** People stay remembered off-frame for 30 s, at their last base-frame point.

**Built-in AI tracking:** switched off at start. `PTZDirector` switches it off again every 10 s, because a raised palm can turn it on by gesture. While it is on, `PTZ.update()` stops writing.

**Smoothing.** Goal → setpoint, limited to 90°/s and 240°/s² with braking, so it arrives exactly with no overshoot. The setpoint is quantized to 1° and written at the backend's rate:
- v4l2: 20 Hz
- linkcam: 4 Hz, since every call spawns a process
- sim: every update

The setpoint is re-sent every 1 s, because the Link 2 ignores writes while nothing is streaming.

**Sim result** (`videos/ptz_aim_demo.mp4`, 43 s; `.aim.ndjson` holds the decisions):

| when | what happens |
|---|---|
| start | the survey finds Kirill talking in 1.2 s |
| Steve speaks off-frame | found 1.8 s after he starts |
| Kirill again | 0.7 s |
| Q&A | one 1.4× group shot of the three judges, held through both questions |
| Kermit's lines | never move the camera |

**Degraded sim runs** (`videos/ptz_aim_demo_<flag>.mp4`, `sim_demo.py --<flag>`):

| run | how it's degraded | Steve found | Kirill again | Q&A |
|---|---|---|---|---|
| `nospeechsignal` | no `speech_active`, no BRAIN qa event | ~3.7 s | ~3.7 s | the patrol's audience step |
| `fusionlock` | no `unseen_speech`; the only visible person gets speaking_p 0.8 and speaker_id | 1.8 s | 0.7 s | entered |

In `fusionlock` it briefly framed a judge at start-up before the lip check corrected it.

**Live room runs, earlier policy, for the record:**
- **it3_normal_B_ptz:** stuck on Kirill.
- **it5_normal_B_ptzhub:** stuck on Steve, with speaker in frame 23% of the time against 100% for the static 120° camera.
- **Cause:** the narrow view plus the fused prior made the framed person never look quiet. Also, a re-ID's new id counted as a new speaker.

The current triggers (unseen_speech, lip score, re-ID as the same shot) target exactly that. They have **not been re-run in the room**. ROOM SIM's recommendation for the venue stands: **the Link 2 fixed and wide, PTZ off**, until a live run shows otherwise. To use it as a fixed camera, run `PTZ(...).set(0, 5, 1.0)` once with tracking off, or just leave it centred with AI tracking off.

## Live wiring (hub side, not done yet)

`run_kermit.py` runs perception in-process. The hook looks like this:
```python
sys.path.insert(0, "camera"); from ptz import PTZ, make_backend; from aim import PTZDirector
ptz = PTZDirector(PTZ(make_backend("mac"), mount=(0.3, 0.3, 0.5), pose_log="run/ptz_pose.ndjson"), log="run/aim.ndjson")
# per camera frame:  pose = ptz.ptz.pose_at(frame_t)  -> perception uses R(pose), fx·zoom for this frame
# per §5 output:     ptz.on_perception(latest)
# per stdout event:  ptz.on_kermit_event(ev)     (play / stop / done / segment / camera_mode / turn_candidate / heard)
# >= 30 Hz:          ptz.tick()
```
In the room sim, the same director runs in `room/camera.py` with `PTZ(SimBackend(w))` and
`LiveWorld(..., ptz=True)`. The snapshot should then carry the frame's pose, for example a header
`X-PTZ: json(w.ptz_pose().to_json())`.

## Sim (`LiveWorld(..., ptz={...})`)

**Defaults:** `{"mount": [0.3, 0.3, 0.5], "mount_yaw": 0, "size": None, "park_when_idle": True, "gimbal": {}}`. This puts the Link 2 on a ~0.5 m post on a side table beside the arm.

**Mount position.** Directly beside the base (x ≈ 0), the arm's elbow (link3) blocks the far-side presenter. A ray scan over the §6 faces, with Kermit looking at each person in turn, found that from 0.3 m forward / 0.3 m left all five faces are clear. Looking at Steve, Kermit's head is at the edge of the frame.

**Rendering.** `render("arm_base")` is the PTZ view, and `render("base_cam")` is the old static camera.

**Driving and logging.**
- Drive it with `w.ptz_command(pan, tilt, zoom)` or through `PTZ(SimBackend(w))`.
- `w.ptz_pose()` returns the motors' pose.
- `state()["ptz"]` logs that pose every frame, and `render_log.py` rebuilds the camera from `config()`.

**Fidelity.**
- Slew is limited by `Gimbal`: 120/90°/s, 500/400°/s². These are guesses; the smoke test measures the real speed.
- Commands are clamped and quantized to 1°.
- Like the real camera, the sim gimbal parks at −85° tilt and drops writes when no frame has been rendered from it for 1 s.

**Cost.** 17 ms per step + render at 640×360.

## Linux (v4l2)

```bash
sudo apt install v4l-utils             # v4l2-ctl
v4l2-ctl --list-devices                # find the Link 2's /dev/videoN
# linkctl (AI tracking off): copy the stdlib-Python `linkctl` from github.com/battistaar/insta360_link_control onto PATH
python ptz.py smoke --backend v4l2     # (smoke needs opencv-python; set the device with V4L2Backend(device=...))
```
v4l2 writes `pan_absolute`/`tilt_absolute` in arc-seconds (degrees × 3600) and `zoom_absolute` as 100–400.

## Mac runbook (Kirill): install, permissions, 1-minute smoke test

**Backend choice: `linkcam` (github.com/fmontes/linkcam).**
- It is a single prebuilt universal binary, documented and tested on the Link 2.
- It uses CoreMediaIO for pan/tilt/zoom and the UVC extension unit for AI tracking.
- There is no daemon to keep alive.

The alternative HTTP server (lucasoskorep, `127.0.0.1:8360/api/ptz`) could not be checked: its docs didn't load, so its API is unverified, and it is not implemented here.

1. **Insta360 Link Controller app.** Open it once and turn **Gesture Control off**. Otherwise a raised palm turns tracking on and an "L" hand sign zooms, which is likely with judges raising hands. Then **quit it** (menu bar → Quit). If left running, it fights for the camera.
2. **Install linkcam.** Download the repository from github.com/fmontes/linkcam (Code → Download ZIP), unzip it, and in that folder run `./install.sh`. If the repo README has newer steps, follow those.
   - Check the install: `which linkcam && linkcam status --json` should print the pan/tilt/zoom state.
   - If macOS says the developer can't be verified, run `xattr -d com.apple.quarantine "$(which linkcam)"`, or allow it under System Settings → Privacy & Security.
3. **Camera permission.** The gimbal only moves while something streams video, so the smoke test opens the camera. The terminal app (Terminal, iTerm or VS Code) needs **System Settings → Privacy & Security → Camera** switched on. The first run triggers the prompt; after allowing it, restart the terminal.
4. **Find the Link 2's index.** Run `ffmpeg -f avfoundation -list_devices true -i ""`. The video devices are listed in order; the default is 0.
5. **Smoke test (about 1 minute).** Plug the camera straight into the Mac, then:
   ```bash
   cd kermit/camera && ../voice/.venv/bin/python ptz.py smoke --backend mac --camera 0
   ```
   The test does the following:
   - opens the stream
   - turns AI tracking off
   - visits centre → pan +40 (left) → −40 (right) → tilt +20 → −20 → 2× zoom → centre, saving a frame at each in `ptz_smoke/`
   - does one fast 120° swing and measures the real pan speed from the video

   **Pass:**
   - the camera visibly moves at every step
   - the readback column matches the goal
   - `errors: none`
   - in `ptz_smoke/01_left_40.jpg` the view has turned to the **camera's left**, which is your right as you face it

   **If it fails:**
   - **turned the wrong way:** `export KERMIT_PTZ_PAN_SIGN=1` and run again
   - **readback matches but nothing moves:** the stream isn't open (camera permission, wrong `--camera`) or the Insta360 app is still running
   - **measured speed far from 120°/s:** tell MOTION, so the motor model gets updated
6. **Optional.** With perception running and writing its latest-value file:
   ```bash
   ../voice/.venv/bin/python aim.py --backend mac --perception <latest.json> --out ptz_out
   ```
   Walk around and talk; `ptz_out/aim.ndjson` logs each decision.

Never run the arm from this folder: everything here talks only to the camera.
