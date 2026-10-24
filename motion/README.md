# Happy Kermit: MOTION (simulation only)

This turns a VOICE performance timeline (CONTRACT §2) into a hand-puppet performance on a simulated xArm6, then renders it to MP4 with the audio muxed in. Every render is also checked against the real arm and hand limits. Nothing here talks to hardware.

```bash
cd kermit/motion
PY=/var/lib/happy-agent/IsaacLab/.venv/bin/python   # mujoco 3.12 EGL; ~50 fps incl. body physics

$PY render.py ../voice/samples/open_1.timeline.json -o videos/talk.mp4      # dexterous hand (default)
$PY render.py samples/whoa.timeline.json --hand dh_pgc140 -o videos/dh_pgc140/whoa.mp4  # gripper fallback
$PY render.py samples/follow_talk.timeline.json --follow ../perception/samples/walk_tracked.ndjson \
    --camera follow -o videos/gaze_follow.mp4                                 # person following (§4)
```

Options:
- `--camera front34|side|front|wide|follow`
- `--size 1920x1080`
- `--fps 30`
- `--xray`: semi-transparent felt, so you can see the robot hand inside
- `--no-shell`: bare robot, no costume
- `--no-hud`
- `--ideal-hand`: skip the hand speed limits
- `--preroll 0.5`: silence before the audio, so the puppet can breathe first

Every render writes three files:
- `out.mp4`: H.264 plus AAC.
- `out.check.json`: the hardware-limit report. A summary is also printed.
- `out.traj.npz`: arm and hand joint trajectories at the render rate (`t`, `arm` (N×6 rad), `hand`, `hand_joints`, `mouth`). This is what a future hardware replay would consume.

## Videos (`videos/`)
| file | content |
|---|---|
| `talk.mp4` | lip sync only (open_1) |
| `nod.mp4` | nods on "talk" and "work" (open_3) |
| `whoa.mp4` | surprise, then a look at the arm, then a lean in (open_2) |
| `turn.mp4` | looks left, then right, then back to the speaker (open_3) |
| `performance.mp4` | about 15 s: open_1 + open_2 + open_3 + think_1 joined, with emotion crossfades |
| `hero.mp4` | 44 s hero scene: 4 human lines, each followed by a puppet reply (BRAIN demo_script) |
| `gaze_follow.mp4` | talks while following a walking person (tracked trace with dropout) |
| `dh_pgc140/*.mp4` | the same shots with the DH PGC-140-50 gripper as the mouth |

Timelines are rebuilt from VOICE/BRAIN outputs by `tools/make_demos.py` and `tools/make_hero.py`.

## Multi-person stage scenes (`render_stage.py`)
```bash
$PY tools/make_show.py && $PY render_stage.py samples/stage/show.stage.json -o videos/stage/show.mp4  # v3 show, 155 s
# older v2 cuts (wrong order per CONTRACT §6) are in videos/stage/old_v2/
```
**Stage script**
- Lists the people (`human.py`):
  - articulated humans, about 1.75 m, with 22 joints
  - face on the front, plain jacket on the back
  - optional handheld mic, positioned by IK
  - seated or standing
- Gives the dialogue in one of two forms:
  - `voice_scene`: VOICE's `scene.json`, with absolute clip times on one mixed soundtrack, so barge-in overlaps work.
  - `lines`: played back to back.
- Adds staging `actions` (`turn_away`, `turn_back`, `raise_hand`) and an optional `shots` list.

**People's behaviour**
- People look at whoever holds the floor (otherwise at the puppet).
- When they talk, the mic comes up, the head beats, and the free hand gestures.
- They shuffle their feet when they turn.

**Puppet gaze.** A synthetic CONTRACT §5 log (`out.perception.ndjson`) produces `speaker_id` with hysteresis; the switch needs a +0.25 margin held for 0.4 s. The puppet's gaze follows that speaker: the head leads (τ = 0.25 s) and the base follows (τ = 0.6 s).

**Cameras**
- `stage_wide`: establishing shot only.
- `two_shot`: from the side of the puppet–presenter line, so the presenter's back reads when he turns away.
- `kermit_cu`: the reference 3/4 close-up.
- `qa_wide`: from behind the judges' row, so the head swing between judges reads.

## Live mode for the LIVE ROOM SIM (`live.py`, CONTRACT §6)
`LiveWorld` is a library the room-sim process imports.
- `set_person(...)` at 30 Hz poses people.
- `kermit_event(ev)` takes every run_kermit stdout line: §5 gaze, segment, play (anchored on `t_audio`), stop.
- `step(perf_counter_t)` advances the scene.
- `render("arm_base")` gives the perception camera frame.
- `state()` and `config()` are what to log.

Step plus render takes about 5–8 ms. The arm goes through a causal limiter at 95% of the xArm6 limits. Run `python live.py` for a smoke test, which also writes the camera samples to `videos/live/`.

`render_log.py room/run/<run>/ -o out.mp4` re-renders a logged run (world_state, kermit, people, mix.wav) in the hero style. The poses are replayed exactly; only the cameras are chosen offline. `tools/mock_room_run.py` makes a fake run to test with.

**Arm camera finding:** at the CONTRACT §4 pose (z = 0.10 m, 78° HFOV), presenters at about 1 m and ±45° are out of frame. A raised post, about (0.10, 0.25, 0.55), or a ≥110° lens fixes it. See `videos/live/armcam_{A,B,C}.png`.

**Gaze target `person_<id>`:** BRAIN can send `look_at` with target `person_<id>`, or a segment gaze of `target:person_<id>`, where `<id>` is PERCEPTION's `people[].id`.
- Kermit aims at that track's live bearing.
- A `look_at` holds until a newer gaze event, a different `speaker_id`, or 10 s.
- If the track hasn't been seen for more than 1.5 s, he falls back to the speaker.

**PTZ camera:** `LiveWorld(..., ptz=True)` adds the Insta360 Link 2 model on a post at (0.3, 0.3, 0.5). `render("arm_base")` then becomes the PTZ view. The controller, aim policy and Mac runbook are in `../camera/README.md`.

## How it works
```
timeline.json ─► animation.py ─► channels @60 Hz: yaw pitch roll dx dy dz mouth (+ base_yaw)
                                  │ layers: emotion posture, gaze turns, idle, breath, phrase accents,
                                  │ gesture clips, person-follow bias (follow.py)
                 trajectory.py ─► head IK (ik.py) → 6 arm joints; mouth → hand joints (hands.py)
                                  │ hand speed slew-limit; arm safety governor (zero-phase low-pass)
                 limits.py     ─► check vs xArm6 / hand limits → .check.json
                 sim.py        ─► kinematic robot replay + simulated dangling body (MuJoCo, 480 Hz)
                 render.py     ─► EGL frames + HUD → ffmpeg (audio delayed by the pre-roll)
```

**Rig**
- `xarm6.py`: xArm6 from the official xarm_description meshes, inertias and joint frames.
- `hands.py`: swappable hand profiles: `shadow` (menagerie Shadow Hand, a stand-in for the unconfirmed ~$10k dexterous hand) and `dh_pgc140`.
- `costume.py`: the puppet costume.

**Swapping in the real hand.** Write a `HandProfile` subclass with:
- `attach()`: mount the hand on the flange, add a `head` site, and call `costume.add_head`.
- `mouth_qpos(openness)`
- `dress_pose()`: drive the felt jaw from the hand's kinematics.
- `vel_max`: per-joint speed limits.

Then register it in `PROFILES`.

**Puppet mapping (hand-rod puppet)**
- The hand sits fully inside the felt head.
- The four fingers form the upper jaw and skull; they move at most 7°.
- The thumb is the lower jaw, opening up to 40°.
- The wrist is the neck.
- The arm carries the head pose.
- The body (torso, noodle arms and legs, webbed hands and feet) is a passive MuJoCo body softly welded to the neck. It swings and follows through on its own.

**Arm rest pose.** Forearm vertical, wrist bent 90°. At this pose joint4 acts as the head's yaw, joint5 as its pitch and joint6 as its roll, which keeps head turns cheap. The pose was found with an IK seed sweep.

## Animation rules
- **Lip sync is the hand's job.** The xArm6's 20 rad/s² joint-acceleration limit allows only about 1° of head motion at syllable rate (~5 Hz). The mouth follows VOICE's curve, led by 30 ms. The head moves only for phrasing: a slow lift and lean while talking, never a bob per word.
- **Breath:** before every sentence the head lifts about 3°, the body rises 6 mm, and the mouth opens to 0.2, starting about 150 ms early.
- **Never dead:** idle drift of 1–2° at 0.3–0.5 Hz, plus breathing.
- **Gestures** are hand-keyed Catmull-Rom curves with anticipation, overshoot and settle, scaled by intensity and emotion energy (`animation.py:CLIPS`). The full v0 vocabulary is implemented: nod, shake, surprise/whoa, head_bob, tilt_left/right, look_at, look_around, laugh, sigh, shrug, lean_in/back, flail, cover_face. Emotions set a baseline posture; mid-turn `{"type":"emotion"}` events crossfade between them.
- **Gaze:** a turn is an eased step with anticipation, overshoot, an arc (the nose dips through the turn) and a lead tilt. Turns are capped at 3/4 toward the camera, so both eyes stay visible. Unknown targets map to `speaker`.
- **Person following** (CONTRACT §4): the bias is computed from the head.
  - Joint1 takes 60% of it and the head 40%.
  - Smoothing is critically damped with τ = 0.4 s.
  - Clamped to ±60° yaw and ±20° pitch.
  - Weight is 1.0 for speaker, 0.4 for audience, and 0 for any other look_at target.
  - When the person is lost, it eases back to neutral in about 1 s.

## Hardware limits (`limits.py`, sources in `docs/hw_limits.md`)
- **xArm6**
  - Ranges: J1/J4/J6 ±360°, J2 −117…116°, J3 −219…10°, J5 −97…180°.
  - Joint limits: 180°/s and 1145°/s².
  - TCP: 1 m/s.
  - Anything above 80% of a limit is reported as WARN.
- **Hand stand-in:** Inspire RH56DFX-like speeds, fingers 260°/s and thumb 90°/s. These apply per Shadow joint until the real hand is known. Mouth commands are slew-limited to these speeds before rendering, so the video shows what the hardware could do. Lip sync: r = 0.99 against VOICE's curve.
- **DH PGC-140-50:** 25 mm per jaw in 0.75 s. Mouth motion shrinks to about 60% of the commanded range (r = 0.73); in practice the mouth flaps once per word, not per syllable.
- **Governor:** if keyed motion ever exceeds the arm limits, `trajectory.govern` low-passes the joints (zero phase, so there's no A/V lag) at the highest cutoff that fits, and reports it in the check. Current demos need at most 8 Hz.
- **All demos pass.** For live hardware later, use xArm servo_j / mode 1 at 100–250 Hz, with the `.traj.npz` resampled.

## Not done / next
- The real dexterous hand model is unconfirmed (Shadow is the stand-in).
- Optional rod-animated puppet arm (the arms are passive now).
- "Scrunch" for embarrassed/annoyed.
- A live real-time player: stream the timeline, read the PERCEPTION latest-value file, and output servo_j.
