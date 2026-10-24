# Happy Kermit: LIVE ROOM SIM (CONTRACT §6)

This directory is a closed-loop, wall-clock test bed. Everyone shares one simulated noisy room, and Kermit's real live
loop (`../run_kermit.py`, **unchanged**) runs inside it. Nothing is pre-mixed, and the video is rendered from the log afterwards.

```
                    ┌──────────── room/server.py (wall-clock master, 20 ms blocks @16 kHz) ────────────┐
script driver ──►   │ humans' lines (Kokoro, pre-rendered)  ─┐                                        │
 (v3 script,        │ crowd babble/claps/laughs/steps/HVAC   ─┼─► streaming partitioned RIR conv ─► mic ├──► sd.InputStream ─► run_kermit
  waits for Kermit  │ Kermit's loudspeaker ◄── sd.OutputStream◄┘   (pyroomacoustics, RT60 0.86 s)      │      (EOU/AEC, STT, BRAIN, TTS)
  in the room)      │                                              └─► audience seat (video mix)      │                │
                    └── truth.ndjson (who speaks, facing, words) ──► perception oracle (people.py) ◄───┘   stdout NDJSON
room/camera.py: MOTION LiveWorld at 30 fps ◄── people.ndjson (poses) + kermit.ndjson (gaze/segments/play) → armcam.mp4, :8091
```

## Run
```bash
# BRAIN (once): llama-server E4B + Kermit LoRA on GPU0 :18090, brain.py serve :18765 (see logs/)
python3 live.py --name normal_B --noise normal --input B --camera     # one full v3 run (~4 min), then score.py
./sweep.sh s1                                                         # quiet/normal/loud × A/B
.venv/bin/python aggregate.py run/s1_*                                # markdown table
.venv/bin/python splitscreen.py run/<name>                            # arm-cam + mic waveform + live transcript
```
Sockets and CUDA need full access, outside the sandbox. `sounddevice` resolves to `shim/sounddevice.py` → `sd.py` (TCP
127.0.0.1:18766), so run_kermit runs as-is. TURN also added `KERMIT_AUDIO=roomsim`.

## Model
| | |
|---|---|
| room | 12×8×4 m shoebox, ISM + air absorption, measured RT60 0.86 s. Arm base on a table 0.75 m high, 1.5 m from the front wall |
| Kermit's table mic (A) | arm base (0.08, 0, 0.08), next to the camera. 94 dB SPL = −6 dBFS |
| handhelds (B) | cardioid, 6 cm in front of the mouth: Kirill's and Steve's, plus one Q&A mic passed between judges, summed at a mixer. 94 dB SPL = −26 dBFS |
| loudspeaker | laptop, 0.32 m from the table mic. 0 dBFS = 86 dB SPL @1 m, 180 Hz HP + soft clip, 40 ms device buffer + 12 ± 4 ms |
| people | Kirill (0.75, 0.75) / Steve (0.75, −0.75), mouth at 1.6 m, ±45°. Presenters have two directivity RIRs (to Kermit / to audience), crossfaded as they turn. Judges seated at 3.3–3.5 m. Speech 68 dB @1 m + Lombard 0.3 dB/dB above 55 dBA |
| noise | DEMAND PCAFETER babble at 6 decorrelated crowd positions. ESC-50 clapping (applause = layered), laughing, footsteps, creaks (chairs), coughs, typing. Synthetic HVAC + fan hum. Venue level in dBA at the mic: quiet 45 / normal 60 / loud 70, ±3 dB drift. Applause/laughter at the ends of beats |
| driver | a line starts when Kermit has been silent ≥ 0.7 s in the room (+0.2–0.5 s), or after an 8 s timeout (= missed turn). Staging cues at word onsets. Kirill trails off if Kermit cuts in during the interjection line |
| camera | MOTION LiveWorld, raised post (0.10, 0.12, 0.55), HFOV 120°, 640×360 at 30 fps. Perception = PERCEPTION's degraded oracle on truth (sim faces defeat YuNet) |

## Outputs per run (`run/<name>/`, one perf_counter clock; wav sample 0 = meta.json `t0`)
`mic.wav` (what Kermit got), `mic_table.wav`, `mic_{human,echo,noise}.wav` (components of mic.wav), `kermit_dry.wav`
(loudspeaker feed at room time), `audience.wav` (room mixdown, front row), `truth.ndjson`, `levels.ndjson`,
`people.ndjson`, `kermit.ndjson` (run_kermit stdout + t_rx), `world_state.ndjson`, `armcam.mp4`, `score.json`, `split.mp4`.

## Scoring (`score.py`)
See its docstring. Latency = end of the human's last word → first sample of Kermit's reply in the room.

## PTZ (`live.py --ptz hub|camera`)
`run_kermit.py --ptz {mac,v4l2,dry,room}` (hook: `ptz_hook.py`) runs camera/aim.py's PTZDirector in the hub. In the
sim, `room` drives the MuJoCo Link 2 in camera.py through `ptz_cmd.json` / `ptz_pose_sim.json`, and the truth
`in_fov` follows the camera's actual pan/tilt/zoom. Result: the aim policy locks onto the one framed person
(speaker-in-frame 23–54% vs 100% static; speaker-ID 0.23–0.53 vs 0.89). **Venue recommendation: fixed, wide, PTZ off.**

## Results (Oct 24, ~03:30–05:35 PT; full v3 script, 17 steps / 16 expected replies; per-run score.json)
Configs: A = table mic at the arm base, B = close-talk handhelds. Latency is measured in the room, from the end of
the human's speech to the first sample of Kermit's reply. The dev box was shared at load 13–25, so treat latency
as pessimistic. Run-to-run p95 spread is about ±0.5 s.

| level | input | run | beats | missed | false cut-ins | self barge-in | lat mean / p95 s | WER | speaker-ID |
|---|---|---|---|---|---|---|---|---|---|
| 45 | A | quiet_1 (before AEC fix) | 1/17 | 5 | 2 | 13 | 2.36 / 3.37 | 1.38 | 0.67 |
| 45 | A | s1_quiet_A | 15/17 | 1 | 0 | 0 | 1.71 / 2.52 | 0.36 | 0.81 |
| 60 | A | it1_normal_A | 10/17 | 6 | 2 | 1 | 1.59 / 2.19 | 0.61 | 0.79 |
| 70 | A | s1_loud_A | 3/17 | 12 | 1 | 1 | 3.07 / 4.89 | 1.26 | 0.55 |
| 45 | B | it3_quiet_B | 16/17 | 1 | 0 | 0 | 1.63 / 2.39 | 0.30 | 0.81 |
| 60 | B | it3_normal_B | 16/17 | 1 | 0 | 0 | 1.65 / 2.46 | 0.64 | 0.74 |
| 60 | B | it4_normal_B_static | 16/17 | 1 | 0 | 0 | 1.89 / 3.22 | 0.38 | 0.89 |
| 70 | B | it3_loud_B | 15/17 | 1 | 2 | 0 | 1.67 / 2.71 | 0.42 | 0.91 |
| 60 | B | **final2_normal_B** (shipped state, 06:15 PT) | **17/17** | 0 | 0 | 0 | 1.52 / 2.47 | 0.23 | 0.89 |
| 45 | B | final2_quiet_B (shipped state) | 15/17 | 1 | 1 | 0 | 1.45 / 2.04 | 0.28 | 0.83 |

PTZ final attempt (it6_normal_B_ptzhub, aim.py 12:37 UTC): speaker in frame 48%, speaker-ID 0.46 vs static 100% / 0.89 → PTZ off.
| 45 | B | **final3_quiet_B** (ship state: TURN 13:13 UTC + barge-in merge, BRAIN 13:13) | **17/17** | 0 | 0 | 0 | 1.41 / 2.18 | 0.22 | 0.84 |
| 60 | B | **final3_normal_B** (ship state) | **17/17** | 0 | 0 | 0 | 1.57 / 2.50 | 0.32 | 0.82 |
