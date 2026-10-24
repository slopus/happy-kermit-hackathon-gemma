# Happy Kermit: PERCEPTION v2 (who is talking, who is facing Kermit)

`people.py` tracks everyone in front of the camera. For each person it reports whether they are speaking, whether they
are facing Kermit, whom they are addressing, whether they hold a mic, and whether a hand is up. Output is CONTRACT §5
plus events for BRAIN. Everything runs on-device; no cloud is used at inference. Numbers are in [eval/REPORT.md](eval/REPORT.md).

```
camera ─▶ YOLO11n-pose + ByteTrack ─▶ stable IDs (re-ID by position + torso colour) ─▶ facing / addressing / hand_up
       ─▶ YuNet faces (assigned to bodies) ─▶ Light-ASD (lip–audio sync, 1.2 s causal window, 5 Hz)
       ─▶ YOLO-World-m "microphone" (3 Hz, hand-held only: near a wrist, not static)
mic  ─▶ Silero VAD (from run_kermit) + loudness over the noise floor
fusion per person: logit = bias + 1.5·ASD + mic/n_mics + 0.5·continuity + alone + explain-away ─▶ speaking_p
speaker_id = argmax with hysteresis (logit margin 0.5 for 0.3 s; frozen while Kermit talks)
```

## Output (latest JSON `--latest`, NDJSON `--log`)
```json
{"t":12.3,"speaker_id":2,"speech_active":true,"unseen_speech":false,"people":[
 {"id":2,"yaw_deg":12.0,"pitch_deg":6.0,"dist_m":1.2,"facing":"away","addressing":"audience","head_turn":null,
  "speaking_p":0.91,"has_mic":true,"hand_up":false,"role":"presenter","conf":0.8,"face":false,"asd":null,
  "box":[x0,y0,x1,y1],"face_box":null}]}
```
- `speech_active`: mic VAD (Silero after AEC, fed by run_kermit); false while Kermit's loudspeaker is on.
  `unseen_speech`: someone is talking but no visible face is (over 1.5 s), i.e. the talker is off-frame. Look for them.
- `facing`: toward | side | away. With asymmetric hysteresis, "away" means *not facing Kermit* for ≥1 s with at least
  one back-of-head frame. Leaving "away" needs a frontal face for 0.5 s.
- `addressing`: `kermit` | the other person's `id` | `audience`. It comes from the YuNet head-turn ratio (`head_turn`)
  and is a SOFT cue (§6).
- `role`: presenter if the smoothed distance is < `--presenter-dist` (2 m), else judge.
- `asd`: the raw Light-ASD logit (>0 means the lips move with the audio). `null` when no face is visible.
- Only people inside the attention region are listed: `--max-dist` (default 5 m) and `--roi x0 y0 x1 y1` (image
  fraction of the box centre). People outside it are listed only if they raise a hand or hold a mic.

**Events** (`on_event` callback / `--events` NDJSON). Each carries `t`, `role` and `yaw_deg`:
`person_arrived`, `person_left`, `speaker_changed {from,to}`, `turned_away {id, while_speaking}`, `turned_back`,
`hand_raised`.
- `while_speaking` is true only if this person is `speaker_id`, still has the top fused score, has `speaking_p ≥ 0.6`,
  and there was voice in the last second. Someone else starting a question while the presenter turns to look at them
  gives false.
- `turned_away` has a 3 s refractory per person, and `turned_back` fires only after an announced `turned_away`.

**§4 compatibility:** `PeopleThread.gaze` is the §4 dict (`present`, `yaw_deg`, `pitch_deg`, `dist_m`, `conf`,
`target_base`, `n_faces`) aimed at `speaker_id`, falling back to the closest person, plus `speaker_id`, `id`, `facing`,
`addressing`, `speech_active` and `unseen_speech`. MOTION needs no change.

## In run_kermit.py (already wired by VOICE)
```python
pt = people.PeopleThread(people.parse_args(["--source", cam, "--hfov", h, ...]), on_event=hub.on_event); pt.start()
pt.feed_audio(t, x16k); pt.feed_vad(t, active, rms_db)   # every 512-sample mic block, t = perf_counter at block end
pt.set_self_speaking(True/False)                           # Kermit's loudspeaker on/off: speaker_id is frozen meanwhile
pt.set_ptz_source(ptz.PTZ.pose_at)                         # Insta360 Link 2: camera pose at each frame's capture time
                                                           # (or pt.set_ptz(pan, tilt, zoom), or --ptz-pose ptz_pose.ndjson)
pt.latest  /  pt.gaze  /  pt.context()                     # §5 frame, §4 gaze, BRAIN context string
```
Pass extra options as `run_kermit.py --camera 0 --hfov 78 ... --perception-args --max-dist 5 --mic-model ''` (this
must come last). `context()` reads like: *"3 people in front of you; the presenter talking is to your left, about
1.1 metres away, holding a microphone, with their back turned to you; someone to your right is raising a hand"*.

## Degraded-oracle mode (live room sim)
`--oracle room/run/<run>/truth.ndjson` (see `oracle.py`). The sim's primitive faces defeat YuNet and lip-ASD, so the
sim's ground truth is corrupted with error rates **measured on real video** (`eval/error_profile.json`): ASD logit
distributions, facing confusion, dropouts, ID switches and bearing noise. The result then goes through the same
hysteresis, fusion and events as the camera path. Replay a run offline with
`.venv/bin/python eval/replay_room.py ../room/run/<run> --cam-pos … --hfov …`.

## Offline / eval
```
.venv/bin/python people.py --source clip.mp4 [--start S --duration D] --log out.ndjson --events ev.ndjson --render annotated.mp4
```
This uses the file's own audio, with Silero VAD offline. Eval scripts live in `eval/`: `eval_ava.py`,
`eval_columbia.py`, `replay_fusion.py`, `eval_mic.py`, `eval_facing.py`, `eval_addressing.py`, `replay_room.py`.
The demo reel is `demo/perception_v2_demo.mp4`.

## Mac (M5 Max) runbook
```bash
cd kermit/voice && ./setup_mac.sh                 # voice venv first (as before)
cd ../perception && ./setup_mac.sh                # + ultralytics/torch into voice/.venv, ~95 MB of models (online once)
../voice/.venv/bin/python people.py --source 0 --hfov 78 --device mps --stats   # smoke test: prints Hz + per-stage ms
```
- Models in `perception/models/`, about 95 MB: `yolo11n-pose.pt` (6 MB), `face_detection_yunet_2023mar.onnx`
  (0.2 MB), `pretrain_AVA_CVPR.model` (Light-ASD, 4 MB), `yolov8m-worldv2-mic.pt` (57 MB, prompts baked in, no CLIP
  at runtime), `kermit_bytetrack.yaml`. Simplest: copy `perception/models/` from the Linux box. Otherwise
  `setup_mac.sh` downloads them, plus CLIP once (~340 MB) to bake the mic prompts.
- **Permissions:** Camera, and Microphone for run_kermit. macOS prompts on first use; grant them to your terminal
  app in System Settings › Privacy & Security, then rerun.
- Set `HF_HUB_OFFLINE=1` and `YOLO_OFFLINE=1` at the venue. Ultralytics never needs the network once the `.pt` files
  are present.
- **Speed estimate on M5 Max** (measured here: RTX 4000 + EPYC = 15–20 Hz, CPU-bound on YuNet): pose (MPS) ~8 ms,
  YuNet (CPU, 960 px) ~8 ms, ASD ~5 ms per 5 Hz tick, mic ~30 ms per 3 Hz tick. That gives an estimated **≥20 Hz**
  with Gemma sharing the GPU. If the `--stats` rate is <10 Hz: `--det-width 640 --mic-hz 1 --imgsz 480`, or
  `--mic-model ''`.
- **Camera placement** (from the sim and real-geometry checks): a camera at the §4 base height cannot see the
  presenters' faces at ~1 m and ±45°. Mount it at Kermit head height (~0.5 m), use ≥110° HFOV, and capture at
  1080p. Then a judge's face at 3.5 m is ~45 px and lip-ASD still works. Set `--hfov` and
  `--cam-pos/--cam-rpy` to match, and give it the PTZ pose if the Insta360 moves (`set_ptz_source` or `--ptz-pose`).
- **PTZ:** bearings are in the base frame (mount + pan/tilt/zoom at the frame's capture time, `X-Frame-T` header for
  HTTP snapshots). Lip crops are skipped while the head turns faster than 30 deg/s. People who leave the frame keep
  their `id` for 30 s (`--reid-ptz-s`, matched by base-frame bearing + clothing colour); `person_left` fires after
  2 s and `person_arrived` again on return.
- Table/gooseneck mics fool the mic cue (see the report). Use `--mic-model ''` if the venue has them.

---
## v1 (still here, unchanged): `track.py`
Single main face → §4 bearing with YuNet. It is the fallback when `import people` fails (`run_kermit.py --perception
v1`). The validation notes are in `test/README_v1.md`; samples are in `samples/`.
