# Gemma Hardware Hackathon — Plan

Sat Oct 24, 2026 · AGI House, 1868 Floribunda Ave, Hillsborough · ~10am–5pm
Rule: Gemma-based model, 100% on-device inference. Teams 3–4. Prize: $60K cash + $5K+ hardware.

## Direction (draft)
Track 2 (Physical AI). Robot arm.
- Gemma (multimodal, on-device) = high-level planner, ~0.5–1 Hz, picks skills via function calls, checks result, replans.
- Low-level skill library: reach / grasp / place / push / home. Learned (SuperMimic-style world-model training in sim, or LeRobot ACT from teleop) with scripted IK fallback.
- Demo: spoken instruction → multi-step task → recovers from disturbance. No cloud.

## Local rig (this node)
- xArm6 + DH-PGC-140-50 gripper on a table, Logitech C922 (1080p) camera with calibration (K, rvec, tvec, table frame).
- Owned by another task: /var/lib/happy-agent/happy/Tasks/task (arm.py, kin.py, calib.json, vla_trial.py, recording.py).
- Read-only monitor server: pid 1037, :8090. Endpoints: /snapshot.jpg?width=640, /stream.mjpg, /state, /ws. It holds /dev/video0, so get frames through HTTP.
- Real frames: .context/gemma-bench/frames/

## Benchmark: RTX 4000 Ada, uncontended (gdjq…)
Results are in .context/gemma-bench/ (models 22 GB, llama.cpp build, planner_bench.py, results_*.json).
| Model | decode tok/s | camera→plan loop | JSON valid |
|---|---|---|---|
| E2B Q4_K_M | 163 | 0.55 s | 1–5/10 free, 10/10 with schema |
| E2B Q8_0 | 109 | 0.78 s | 10/10 |
| E4B Q4_K_M | 89 | 1.27 s | 10/10 |
| E4B Q8_0 | 57 | 1.90 s | 10/10 |
- A 640×360 image is about 106 tokens and encodes in ~85 ms. Audio input works (E4B transcription close to exact).
- E4B gives better plans and localization; E2B is weak at both. Both models output coordinates on a 0–1000 normalized grid, with y compressed by about 0.8.
- Jetson estimates for the E2B Q4 loop: AGX Orin ~1.2–1.4 s, Orin Nano ~3.5–4.5 s.
- Ways to speed it up: emit only the next tool call, use a compact GBNF grammar, use --reasoning off, try the MTP drafter.
- ggml-org repos have no Q4_K_M, so use the unsloth GGUF repos.

## Precision: BF16 vs Q8 vs Q4 (gdjq…)
- Loop time: E2B 0.55 / 0.78 / 1.23 s; E4B 1.27 / 1.90 / 3.12 s (Q4 / Q8 / BF16). BF16 VRAM: E2B 6.0 GB, E4B 10.7 GB.
- Localization doesn't improve with BF16. The y-offset is the same at every precision (−33…−46 px), so it is not caused by quantization.
- A square crop flips the offset to +36…+47 px. The cause is preprocessing or the coordinate convention. Next step: run the HF transformers processor as a reference.
- Decision: E4B Q4 for the planner. If we use E2B, use Q8 or BF16, never Q4. Calibrate y for one fixed camera geometry.
- The scene has changed: the apple was moved next to the can. *_live scores are against stale ground truth.

## STT: Parakeet (gdjq…), files in .context/gemma-bench/parakeet/
- Pick: parakeet-tdt-0.6b-v2 fp16 on GPU via sherpa-onnx 1.13.8 (CUDA wheel). ~22 ms per command, 0 command WER, 2.35 GB VRAM. Trigger it with VAD or push-to-talk.
- Live captions: nemotron-3.5 streaming 160 ms int8 on CPU, RTF 0.6, final text ~0.3–0.4 s after speech ends. Needs ~0.3 s of leading silence.
- Avoid: int8 ONNX on CUDA (falls back to CPU), parakeet-unified streaming (slower than real time).
- Gemma audio dropped for commands: 5–10× slower, gets object nouns wrong ("blue container").
- Jetson (estimated): TDT fp16 ~0.1–0.25 s on AGX, ~0.3–0.6 s on Nano. Use fp16 or TensorRT there, not int8.
- voice_to_plan.py: WAV → Parakeet → snapshot → planner tested end to end (346 ms plan). --mic path untested.
- Clean audio only so far: test in a noisy room.

## Idea: Happy Kermit, puppet show (Kirill, Oct 23)
- Puppet on the gripper. Gripper open/close = mouth; wrist and arm = head (nod, tilt, look around); follows the speaker using the camera.
- Pipeline (all on-device): Parakeet STT → Gemma E4B (in-character reply + timed gesture track) → local TTS (Kokoro/Piper; NOT ElevenLabs/Google cloud) → mouth from the TTS audio envelope, gestures from Gemma's track, gaze from face tracking.
- Gemma fine-tune: LoRA on E2B/E4B, text+emotion → gesture JSON, synthetic data. Generating the data with a cloud model is OK because it isn't inference.
- Rules risks: no cloud in the loop, so the Happy voice agent's cloud LLM can't drive the live demo. Ask Rocky/Jessica whether a Mac laptop counts as on-device.
- Hardware: need an arm at the venue. Is the xArm6 portable, or does there need to be a loaner arm?

## Workstreams (Happy Kermit), coordinated here. Interface spec: kermit/CONTRACT.md
| Stream | Agent | Model | Folder | Status |
|---|---|---|---|---|
| BRAIN: Gemma persona + gesture JSON, LoRA, Mac runbook | vcbc3vjw73kp4ivxz92h814x | Opus 5.5 high | kermit/brain | started |
| VOICE: local TTS, lip-sync mouth curve, Parakeet, e2e | ntpb5a4a7fqdnyqpilrfqmu3 | Opus 5.5 high | kermit/voice | started |
| MOTION: sim-only puppet hand video (talk, nod, whoa, head turn) | tmrqb7nhcpv4r8xi6if68vt9 | Opus 5.5 high | kermit/motion | started |
| Session finder (hand model, Isaac, Steve's lip-sync, Happy voice) | g1t2xkxr53132mzqomikhldf | external | — | asked once; don't ping again |
- Busy work: the tree is capped at 5 collaborators, and archive_agent fails on the old research agents. Busy work goes to the shared idle agent gdjqzjdw6w4n8b17sh0f7maf or is done inline.
- BRAIN format: Gemma emits inline markup `[emotion|gaze] {gesture i} text`, and kml.py converts it to CONTRACT JSON (spec in kermit/brain/FORMAT.md). It streams per-sentence NDJSON to VOICE (shape is being negotiated; add it as CONTRACT §1b).
- Scope decision (Kirill): no real arm. Simulated hand video is enough.
- Finder results (forwarded to all 3 streams):
  - Arm: xArm6 fw 2.7.1 + DH PGC-140-50 gripper, controlled by jmvg732d2003kai4me72rf4s (Tasks/task).
  - Dexterous ~$10k hand: NOT FOUND. MOTION builds both a stand-in hand and a gripper-as-mouth fallback.
  - Isaac work is owned by zmvkf78oa004n5d3ngffmvqt (Tasks/task_5): a scripted xArm6 + PGC pick sim with fixed assets. No RL beyond a Cartpole smoke test.
  - Steve's lip-sync: NOT FOUND, no Happy Bit project. Ask Kirill or Steve. Possible lead: Research agent cmvj0lb9j0anc2n7oy881owt.
  - Happy voice agent: cloud GPT-Live plus a gpt-6.1-sol controller, so it's branding only, not in the demo loop.
- happy-robotics project (kwgn91beovozhla6bz678xwa): Steve's Rust station (xArm6 + DH gripper, camera, CUDA perception models, arm-joint-policy crate). Read-only reference.

## BRAIN baseline (prompt-only), 20 lines × 3, RTX 4000
- E4B Q4 + KML + GBNF grammar: 60/60 valid, 146/146 gestures on the right word, first sentence in 325 ms, full turn 626 ms. ← current pick
- Direct JSON: a third of word indices wrong and 1.3 s. Rejected.
- Without a grammar, E4B invents tags → the Mac runtime is llama.cpp Metal + GBNF (MLX only if the LoRA fixes this).
- E2B Q8: 429 ms total, slightly weaker.
- Persona weak spots: gets defensive on "are you Kermit?" and the humor is generic → the LoRA will target these (gdjq is generating ~1k examples).
- Stream: POST :8765/turn, NDJSON header → sentence… → turn.

## VOICE TTS pick
- Kokoro-82M + "frogify" DSP (slow render → pitch up 2–5 st → nasal EQ; emotion → speed/pitch). TTFA 75–110 ms on GPU, WER 1.3%, exact word timings (mlx-audio has the same).
- CPU fallback: 450–660 ms per line. Estimate on M5 Max: <150 ms MLX GPU, 200–400 ms CPU (not measured).
- Qwen3 VoiceDesign is too slow for live (RTF 2). Possible use: pre-rendered intro. Breeze dropped (non-commercial license).
- MLX can't run on this box (CUDA 13 JIT), so the Mac runbook includes a smoke test.
- Waiting for a human to pick the voice by ear: fenrir_frog (agent default), puck_frog, echo_frog, michael_frog, qwen3.

## MOTION v1 (Oct 24, ~07:41 UTC)
- videos/{talk,nod,whoa,turn,performance}.mp4: xArm6 + Shadow hand with a green shell, stage and curtains, 17 s performance, all hw limits OK.
- Kirill: redesign as a classic hand-rod puppet (green felt head, ping-pong eyes on top, wide red mouth, petal collar, long dangling body and limbs). Sent to MOTION with puppetry rules: thumb = lower jaw, one flap per syllable without head-bobbing, 3/4 turns never full profile, breath before lines, constant idle, anticipation on whoa.

- MOTION v2 (redesign done):
  - Felt head with the hand hidden inside, ping-pong eyes, red mouth, 11-petal collar, damped dangling body and limbs.
  - Jaw = thumb ≤40°, skull = fingers ≤7°. Breath 150 ms early, idle drift, 3/4 turns, whoa anticipation, decaying shake.
  - Outputs: still_whoa.png, {whoa,talk,nod,turn}.mp4. --xray shows the hand inside.
  - Next: hero scene → gaze_follow → gripper fallback.

- HERO VIDEO: kermit/motion/videos/hero.mp4 (44 s, 4 exchanges, listening behaviour, all gestures, hw OK). Rebuild: tools/make_hero.py + render.py (~45 s).
- GAZE FOLLOW: videos/gaze_follow.mp4. YuNet trace with real jitter and a dropout. joint1 takes 60% of the yaw and the head 40%. Critically damped, τ = 0.4 s. Live transport = latest-value JSON file (--latest).
- Open: the name "Kermit" in the dialogue ("I'm Happy Kermit"). Kirill's pitch uses it; decide whether to keep or rename.

- MOTION FINISHED (kermit/motion/README.md).
  - Gripper fallback in videos/dh_pgc140/: the 0.75 s stroke means one flap per word (r = 0.73 vs 0.99 for the hand).
  - Missing: a LIVE real-time player (renders are offline only), rod arms, and scrunch.
  - New hand = one HandProfile subclass in hands.py.

## BRAIN demo script + LoRA
- kermit/brain/demo_script/turn_01..04.json: wake up / tired of AI / friendly face of Happy ("less dystopia, more frog") / Gemma on this laptop, no cloud.
- LoRA pilot (280 examples): validity without grammar 43→59/60, funnier and less defensive, first sentence 289 ms. Full run once ~1k examples are ready.

## VOICE e2e done
- voice/kermit_voice.py: --wav/--mic [--ptt]/--turn/--heard. Parakeet → BRAIN stream → Kokoro per sentence → NDJSON segments + turn wav/timeline.
- Speech end → first audio: 0.67–0.83 s (GPU TTS) and 1.0–1.25 s (all CPU). Mac estimate 0.8–1.2 s. Use PTT in the noisy room (VAD adds about 0.5 s).
- Lip-sync: envelope-based, 1 DOF, plot at voice/bench/mouth_final.png.
- Hero-scene audio: voice/samples/demo_script/turn_01..04 (already sent to MOTION).
- Issues sent to BRAIN: emoji in text, odd gesture choices.

- VOICE FINISHED. Start with kermit/voice/README.md (includes the Mac runbook). Run setup_mac.sh, then mac_smoke.py (~2 min, prints which backend to use). Voice is set with KERMIT_VOICE (default am_fenrir).
- Qwen VoiceDesign (~11 GB) is still on disk. Delete it if we don't use pre-rendered Qwen lines.

## PERCEPTION: person tracking (Kirill, Oct 24). Spec: CONTRACT §4
- Static camera at the arm base, looking forward. Face detection (YuNet/MediaPipe) → bearing yaw/pitch/dist in the base frame → MOTION biases the whole arm pose toward the person (smoothed, clamped ±60°/±20°), with gestures layered on top.
- Owners: ntpb… (VOICE agent, now also PERCEPTION) builds track.py plus the synthetic walk trace. tmrq… (MOTION) adds the gaze bias and gaze_follow.mp4 after the puppet redesign.

- PERCEPTION DONE: kermit/perception/track.py (YuNet, 8–17 ms per frame on CPU) + kermit/run_kermit.py (camera + voice loop, one NDJSON stream with gaze/segment/done, and it feeds the person's position to BRAIN's context).
  - Validated on composites: yaw ≤0.4°, pitch ≤1.5°, distance median 3%. NOT validated on a real person in the room (it was empty). Set --hfov to the real camera.
  - Mac: perception/README.md (Camera permission for the terminal app; setup_mac.sh fetches YuNet).

## BRAIN FINISHED (kermit/brain/README.md, MAC_RUNBOOK.md)
- Pick: Gemma 4 E4B Q4 base + runtime LoRA (70 MB kermit-E4B-lora.gguf) + per-turn GBNF.
  - 60/60 valid, 58/60 expectations met, first sentence 347 ms here. Mac estimate ~0.25–0.35 s.
- Persona A/B: LoRA wins 48–8–4. The judge also wrote the data, so it may be biased.
- Merged Q4 is ~50 ms faster but weaker ("fast" mode).
- Mac:
  - `mac/setup.sh` builds llama.cpp Metal and pulls unsloth E4B @bfc15c38.
  - Copy the adapter to the Mac by hand.
  - Then `mac/check.sh` and `mac/run.sh [fast|base]` (:8080 model, :8765 brain).
  - Prompt mode must match the model: persona ↔ LoRA, full ↔ base. A mismatch produced the emoji.
- Can be deleted: .context/brain-train (21 GB) if we won't retrain.

## Round 2 (Kirill, Oct 24 morning)
- SPEAKER TRACKING subtask ghrwd9nw0u3babdg3b8n31jd (Opus high), CONTRACT §5:
  - multi-person tracking with re-ID
  - active speaker from lips + mic-in-hand + VAD + continuity
  - turned_away / turned_back events
  - validated on REAL videos (AVA-ActiveSpeaker, Columbia, etc.)
  - deadline ~8:30am PT
- BRAIN: proactive interjection on turned_away, the scripted turn-around beat ("Oh, like..." / "You, my friend, should learn how to speak to people face to face." / Kirill: "Well, Kermit, I respect you…"), judge Q&A mode with accurate project facts.
- VOICE: barge-in (Kermit speaks while the human is still talking), render the turn-around audio, support the PERCEPTION v2 subtask.
- MOTION: realistic human model (turns around, holds a mic), presenter + judges, gaze switches to the active speaker, new turn-around scene.
- Demo structure: (1) Presentation, scripted: 4 opening turns → turn-around beat → hand-off line. (2) Q&A afterwards, open: judges ask, Kermit turns to the speaker. BRAIN has a phase switch for presentation vs qa.
- Kirill likes "the 26-second video". No file is 26 s (hero 38.5, gaze_follow 15, performance 16). Asked which one he means.

- GEMMA MODEL SWEEP subtask r0ql38cv6673444c67nasi8x (Opus high), kermit/sweep/:
  - SOTA list of Gemma models
  - size sweep from E2B to the large Gemma 4 variants on 1–2 GPUs
  - mixed one-pass vs split text→gestures
  - M5 Max recommendation; target ~7am PT
- TURN-TAKING subtask i69bae09jlr2gyesokp7nkkx (Opus high), kermit/turn/:
  - streaming Parakeet plus an end-of-turn model, benchmarked against silence thresholds (Smart Turn, LiveKit, parakeet EOU, tiny-Gemma judge, fusion)
  - speculative Gemma start
  - barge-in plus echo handling
  - fallback: VAD 0.5 s + PTT; target ~8am PT
- PERCEPTION v2 (ghrw…), people.py:
  - YOLO11n-pose + ByteTrack + re-ID, YuNet, facing, Light-ASD (causal), YOLO-World mic detector, fusion. ~50 ms per frame (~20 Hz).
  - AVA: ASD AP 89.4% offline, 70.2% causal.
  - Columbia speaker_id 45%: table mics trigger has_mic → fix is "mic counts only near wrist + mouth".
  - Sim risk: primitive humans may break face/ASD → decision is degraded oracle perception in the sim (GT + real-measured error profile), with real perception validated on real video.
- TURN SOTA (kermit/turn/), LiveKit eot-bench EN, cut@300 / lat@5%:
  - Silence only: 55.6% / 1600 ms
  - Smart Turn v3.2: 35.2% / 1051 ms
  - p99lab turn-1-mini: 22.6% / 849 ms (5M params, 1.3 ms per step)
  - LiveKit v1-mini (audio): 27.8%
  - LiveKit text detector: 396 MB
  - parakeet_realtime_eou_120m
  - Gemma as judge: ~95 ms per call
  - No open model is great → building a fusion (audio + text on streaming partials + silence). Benchmark ETA ~05:00 PT.
  - TURN now owns Mic / reply_loop / echo in run_kermit.py.
- MOTION stage scenes: videos/stage/presentation.mp4 (70 s, turn-around) and qa_draft.mp4 (26 s, placeholder audio). VOICE is making the real Q&A audio with live Gemma answers.
- VOICE barge-in DONE: run_kermit.py is full-duplex. A perception event → BRAIN POST /event with heard_partial → the interjection plays mid-utterance (Gemma 555 ms; canned fallback ~110 ms). The mic never pauses, and the full utterance is answered afterwards. New events for MOTION: play/stop/perception/heard.
  - Bug: Gemma answered the partial instead of reacting → BRAIN to make the presentation beat scripted/deterministic and the Q&A interjection short.
  - VAD 0.5 s splits the monologue at a pause → TURN-TAKING owns that.
- SWEEP lineup (kermit/sweep/SOTA.md), Apache 2.0:
  - Gemma 4 E2B, E4B, 12B Unified (dense, adds audio, Q4 7.1 GB), 26B-A4B MoE (3.8B active, Q4 ~15 GB), 31B dense (Q4 ~18 GB).
  - Every size has an MTP drafter for speculative decoding (up to 3× decode).
  - Sweep is running.
- SWEEP size sweep (kermit/sweep/), 20×3, GBNF, blind persona 1–5:
  - E2B Q8: 84%, 2.87, 294 ms
  - E4B prompt: 87%, 3.37, 440 ms
  - **E4B+LoRA: 98%, 4.23, 443 ms** (best per ms)
  - 12B: 99%, 3.82, 636 ms
  - 26B-A4B MoE: 87%, 3.93, 268 ms, 86 tok/s (fastest)
  - 31B (2 GPUs): 100%, 4.20, 1293 ms
  - Big models need --swa-full.
  - Split text→gestures: +330–430 ms per sentence and worse persona (2.88 vs 3.37) → one pass wins.
- gdjq: 255 E_ examples (judge Q&A + turn events) for the BRAIN top-up. 28 dupes with D_.
- Disk 28 GB free (93%). GPUs are shared by BRAIN, SWEEP, SPEAKER and TURN.

## Live stack (final wiring)
1. `bash kermit/brain/mac/run.sh [fast|base]`: model on :8080 and brain on :8765. This is the only place the model is selected. Default: base + LoRA, persona prompt.
2. `python kermit/run_kermit.py --mic --ptt`: camera + Parakeet + Kokoro, calling the brain at :8765.
- Measured here: first audio 560–794 ms (median ~705), full turn 1.05–1.38 s.

## Helper (gdjq) done
- Persona data: 1021 examples (1241 turns, all valid) in kermit/brain/data/raw/D_01–D_11.jsonl. Sources are gen/b*.txt; build.py overwrites D_*.
- HW limits: kermit/motion/.context/hw_limits.{md,json} (xArm6 + servo_j at 250 Hz, PGC-140, Inspire/Ability/Shadow). Hand and gripper latency unknown → measure on hardware.

## Node state (Oct 23)
- Isaac Lab v3.0.0 GA (latest; Newton 1.6.1, Warp 1.17), /var/lib/happy-agent/IsaacLab. Only a Cartpole smoke test on newton_mjwarp, no Isaac Sim rendering.
- Tasks/task_5 has xarm6_apple_sim.py (xArm6 sim).

## Open questions
- [ ] Which arms / boards are on site (Resource Hub)? Loaner availability?
- [ ] SuperMimic code released? Twitter reproductions — any for arms?
- [ ] Best Gemma variant + runtime on Jetson Orin / Pi 5 / Qualcomm; function-calling latency.
- [ ] Team?

## Research agents
- c4ex9f2gwwhkn4d4s7rp5o0w — Hub & on-site hardware (Sonnet 5.5)
- m64krkzrus5zcgnrfxbau1qt — SuperMimic code & arm applicability (Opus 5.5)
- nzcofezn76vs6x46g2sd2y8p — Gemma on-device planner stack (Sonnet 5.5)
- gdjqzjdw6w4n8b17sh0f7maf — Gemma 4 GPU benchmark on host 2× RTX 4000 Ada (Opus 5.5), work dir .context/gemma-bench

## Findings: Gemma planner stack (nzco…, unverified numbers)
- Model: Gemma 4 E2B (speed) / E4B (quality). Text+image+audio in, native tool use.
- Primary: Jetson Orin Nano 8GB, MAXN SUPER, llama.cpp CUDA (jetson-containers). E2B Q4 ~22–27 tok/s, E4B ~11–14. Keep ctx small (swap kills it).
- Backup: Pi 5 + LiteRT-LM, E2B ~8 tok/s — too slow for tight replanning. Qualcomm IQ8 fast but preview/risky.
- Audio via llama.cpp mmproj uncertain → whisper.cpp base.en fallback.
- Image encode latency unknown (~1–3 s guess) — measure.
- No existing Gemma+LeRobot/ROS2 example → write own thin loop.
- Downloads: ggml-org/gemma-4-E4B-it-GGUF (Q4_K_M + mmproj f16, ~6.5 GB); E2B equivalent (~3.5 GB, verify repo name); litert-community/gemma-4-E2B-it-litert-lm (only the .litertlm, 2.6 GB); jetson-containers llama_cpp image (docker save); whisper.cpp + ggml-base.en.bin; pip download robot deps.

## Findings: event / hub (c4ex…)
- Builder Resource Hub not public → need link from Kirill's acceptance email.
- Luma (luma.com/6ccl0jw5): 120 builders, teams 3–4, doors ~9:30, start 10am; end 5pm or 10pm (conflicting). Judges rotate science-fair style; top 6 demo on stage.
- Organizer examples: "robot that learns a chore from watching once" — fits arm idea.
- No public info on on-site hardware, arms, judging criteria, per-track prizes.
- Useful: HF blog "Gemma 4 VLA demo on Jetson Orin Nano Super" (huggingface.co/blog/nvidia/gemma4); Arduino Ventuno Q LiteRT starter (github.com/heath0xFF/google-litert-hackathon).

## Findings: SuperMimic (m64k…)
- No official code. Reproductions (G1 humanoid only): zita-ch/supermimic_vibed (MIT, 1.47 GPU-h), CuboiLeo/SuperMimic-Reproduced (mjlab). None for arms.
- Not a fit for arm grasping in 2 days: speedup comes from amortizing 130k motions; contact-rich grasping + object state breaks BPTT through WM; cheap-servo sim2real weak. → related-work / stretch only.
- Skills plan: (1) IK + scripted primitives (floor), (2) LeRobot ACT per skill from ~50 teleop demos, rented GPU 1–3 h each (TensorRT ACT runs on Orin Nano), (3) SmolVLA only on AGX Orin 32GB+.

## Decision (draft)
Jetson Orin + Gemma 4 E2B/E4B (llama.cpp) planner → JSON tool calls → IK/scripted skills, ACT upgrade. Voice: NVIDIA Parakeet (Steve: no whisper). Gemma native audio is the backup. Benchmark pending (gdjq…). Camera + AprilTags/color for object poses.

## Tasks (later, for iteration)
- Planner loop (Gemma + function-call schema, mocked skills)
- Arm skills training (sim) / LeRobot setup
- Day-of integration & demo

## Log
- 2026-10-22: plan created; research agents spawned.
