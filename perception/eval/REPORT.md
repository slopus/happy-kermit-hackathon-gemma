# PERCEPTION v2: evaluation report

Every number here comes from the full `people.py` pipeline on real video unless the row says otherwise:
- YOLO11n-pose + ByteTrack, YuNet faces
- causal Light-ASD and Silero VAD
- the mic detector
- fusion, hysteresis and events

Measurements were made on the Linux box (RTX 4000 + EPYC). The scripts and result JSONs are in this folder. Dataset
videos are deleted after evaluation; small clips and logs are kept.

## Summary

| What | Data | Result |
|---|---|---|
| **Active speaker (`speaker_id`)**, final code, live causal | Columbia panel, 5 windows, 5 speakers, 20,568 labelled frames | **92.4 %** correct; median switch latency **1.07 s** |
| Per-face speaking F1 | same | fused **74.9**, ASD alone 48.2 |
| Light-ASD alone, causal 1.2 s window | AVA-ActiveSpeaker val, 5 videos, 41,083 face-frames | AP 66.6 (offline full-clip 84.1), accuracy 91.2 % |
| **Turned away** (presenter turns their back) | MIT lecture, 120 s, labelled at 1 Hz | not-toward runs ≥ 2 s: **R 0.67, P 1.0**; strict "back" runs: R 0.6, P 0.27 |
| Facing, 3-class per frame | same | 0.675 (toward recognised as toward 0.87) |
| Turned away in the room sim (degraded oracle) | 20 runs, 103 presenter turn-aways | **101/103 detected**, median **1.11 s**, p90 2.09 s; 32 false events (~1.6 per run) |
| **Mic in hand** | town hall + lecture, 58 frames, 285 person-frames | **P 1.0, R 1.0** (10 positives) |
| Mic, table/gooseneck mics | Columbia panel | 44 false positives → use `--mic-model ''` at such venues |
| **Addressing** ("talking to Kermit?") | Oscars two-host clip, 114 samples | 3-class 0.64; talking-to-Kermit **P 0.66, R 0.93** |
| **Hand raised** | GEMMA VISION's stage-perspective clips, 39 labelled hands | **P 100 %, R 64 %** (25 TP / 0 FP / 14 FN) |
| Speed | live C922 snapshot, 1280 px | **15–16 Hz** (pose 15 ms GPU, YuNet 31 ms CPU, mic 15 ms per tick); M5 Max estimate ≥ 20 Hz |

## 1. Active speaker

**Columbia** (`eval_columbia.py`, logs in `col3/`, results in `columbia_results_final.json`): the final `people.py`
ran on 5 windows of the panel video with `--mic-model ''`, offline and causal, using the file's own audio.

Fusion is a per-person logit:
- ASD × 1.5
- a continuity bonus
- an "alone" bonus
- a presenter prior
- explain-away when the voice is loud but no visible lips move
- −4 when nobody is speaking

The speaker switches only after a 0.5-logit margin has held for 0.3 s.

- speaker_id is correct on **92.4 %** of frames with exactly one labelled speaker. It is never None while someone
  speaks.
- Switch latency median is 1.07 s (7 switches: 0, 0, 0.4, 1.07, 2.5, 4.7, 9.2 s). The long tail comes from
  short interjections by a panelist whose face is small.
- The mic detector *on* this venue drops speaker_id to 57–81 %, because Columbia's table mics look like held mics.
  The older logs in `col/` and `col2/` were made with that setting and an earlier fusion, so don't use them.

**AVA-ActiveSpeaker** (`eval_ava.py`, `ava_results.json`; movies, GT face boxes):

| | Causal 1.2 s (live) | Offline whole clip |
|---|---|---|
| Light-ASD AP | 66.6 | 84.1 |
| Accuracy | 91.2 % | 94.2 % |

Per video, causal AP ranges from 51.6 to 83.3. Lip-ASD is a weak cue on its own, which is why the fusion does most of
the work.

## 2. Turned away / turned back

**Lecture** (`eval_facing.py`, `facing_results_lect_shm.json`): a teacher at a board, labelled every second
(T/S/A).

Raw per-frame confusion, rows are the truth:

| Truth | → toward | → side | → away |
|---|---|---|---|
| toward | 0.87 | 0.05 | 0.08 |
| side | 0.06 | 0.41 | 0.53 |
| away | 0.00 | 0.10 | 0.87 |

The events use asymmetric hysteresis:
- enter: ≥ 85 % not-toward in 1 s, with at least one "away"
- leave: ≥ 60 % toward in 0.5 s
- 3 s refractory per person

Against "not facing us for ≥ 2 s" the events score R 0.67 and P 1.0. Against strict "back fully turned" they score
R 0.6 and P 0.27, because a 3/4-back profile also triggers `turned_away`. That is the right behaviour for "the
presenter isn't looking at Kermit".

**Room sim** (`replay_room.py`, `room_summary.json`, `room_<run>.json`): ROOM SIM's 20 recorded runs replayed through
the degraded oracle with final defaults:
- 101 of 103 presenter turn-aways detected
- latency median 1.11 s, p90 2.09 s
- 32 false presenter `turned_away` across 20 runs, about 1.6 per 5-min run, mostly profile views

## 3. Speaker in the room sim (degraded oracle, `room_summary.json`)

The sim's primitive faces defeat YuNet and lip-ASD. For those runs, the truth is corrupted with the error rates
measured above (`error_profile.json`) and then sent through the same fusion and hysteresis. The ASD noise
autocorrelation was calibrated so that the oracle on Columbia truth reproduces the real 91.8 %.

| Runs | n | speaker_id | presenters | judges |
|---|---|---|---|---|
| B (handheld mic) | 14 | **0.824** | 0.838 | 0.748 |
| A (no mic) | 6 | 0.644 | 0.752 | 0.129 |

Judges at 3–4 m without a mic are the weak case: lip-ASD at that distance is near chance, and the audio has no
direction. With a handheld mic (B runs) judges reach 0.75. **Recommendation: pass a mic to Q&A speakers.**

## 4. Mic in hand (`eval_mic.py`, `mic_results.json`)

The detector is YOLO-World-m with baked "microphone" prompts, run at 3 Hz with conf 0.4. A mic counts for a person
only near a wrist (reach 0.3 × body width) or at the mouth. Mics that stay in the same place for 4 s are dropped as
static.

- On the town hall and lecture frames it catches every held mic with no false positives: P 1.0, R 1.0 on 10
  positives out of 285 person-frames.
- It cannot tell a table/gooseneck mic in front of a seated panelist from a held one (44 FP on Columbia).

## 5. Addressing (`eval_addressing.py`, `addressing_results.json`)

Head turn comes from the YuNet nose offset relative to the eye midpoint, with threshold 0.25. On a real two-person
clip (Oscars hosts, 114 samples):
- "talking to Kermit" (facing camera): R 0.93, P 0.66
- "talking to the other person": often called "Kermit" when the head turn is small (12 of 29)

Treat it as a soft cue for BRAIN.

## 6. Hands and front-row ROI

`hand_up` is set when a wrist (conf > 0.5) is above the top of the head for ≥ 80 % of 0.5 s, and it fires
`hand_raised`.

On GEMMA VISION's stage-perspective clips (`kermit/vision/pp/`, `labels.json`) it scores **P 100 %, R 64 %**. The
misses are wrists above the frame top and hands behind other heads. Separately, the classroom demo clip produced
5 `hand_raised` events.

People beyond `--max-dist` (5 m) or outside `--roi` are kept only while they raise a hand or hold a mic.

**Wide crowd shots:** at `--imgsz 640 --person-conf 0.4` the detector finds only 0–3 people. Use
`--imgsz 1280 --person-conf 0.25` if the back rows matter. GEMMA VISION runs its own 1 Hz 1280-px pass.

## 7. Speed

Live, C922 snapshot at 1280 px, `--stats`: **15–16 Hz**.

| Stage | Time |
|---|---|
| Pose (GPU) | 15 ms |
| YuNet (CPU, 960 px) | 31 ms |
| ASD, per 5 Hz tick | 8–23 ms |
| Mic, per 3 Hz tick | 15 ms |

On the M5 Max we estimate ≥ 20 Hz, since YuNet is the CPU bottleneck. If `--stats` shows < 10 Hz there, use
`--det-width 640 --mic-hz 1`.

## 8. PTZ (Insta360 Link 2)

These features are tested, but not yet against real PTZ footage:
- **Bearings:** each frame uses MOTION's pose at its capture time: mount position,
  rpy = (0, −tilt, mount_yaw + pan), and fx/fy from the zoomed FOV. Bearings matched `ptz.pixel_to_base` on a
  pan 30 / tilt 10 test.
- **Motion blur:** lip crops are skipped above 30 deg/s.
- **Leaving the frame:** with a PTZ, people who leave the frame keep their id for 30 s, re-matched by base-frame
  bearing ± 22° plus torso colour. In a test where everyone was blacked out for 6 s, ids 1 and 2 were restored;
  without a PTZ they came back as new ids 3 and 4.
- **Narrow FOV:** with the 72° Link 2 view, the only visible person used to keep a high `speaking_p` (and stay
  `speaker_id`) while someone off-frame talked. With a PTZ pose (or `--narrow-fov`) the "alone" prior is off. When
  `unseen_speech` holds (voice, but no visible lips moving over 1.5 s), the continuity, presenter and presenter-mic
  bonuses are dropped. In ROOM SIM's it3_normal_B_ptz replay:
  - `unseen_speech` fires in 47 % of off-frame-talk frames, 12 % of in-frame talk and 0 % of silence.
  - Off-frame-talk frames with every visible `speaking_p` < 0.5 went from 8 % to 48 %.
  - In-frame speaker accuracy stays at 97 %.
  - The static camera is unaffected.
  - PTZ is off for the demo (coordinator's call).

## Caveats

- Only one real clip each was used for facing (the lecture) and addressing (the Oscars), so the confidence
  intervals are wide.
- The room-sim numbers rest on the measured error profile, not on pixels.
- No real footage from the venue camera, at Kermit head height and ~1 m, has been evaluated yet. The first minutes
  at the venue should include a `--stats --render` check.
- Lip-ASD needs faces of about 40 px or more. Judges at 3–4 m need 1080p capture and ≥ 110° HFOV mounted at about
  0.5 m.
