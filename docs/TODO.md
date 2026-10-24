# Happy Kermit: task list (coordinator-maintained)

## v3 show (Kirill, ~03:00 PT), CONTRACT §6
- [x] Reference look = hero.mp4 (kermit-demo-latest.mp4 is the same file)
- [x] BRAIN: v3 script, 17 steps (brain/demo_script/v3/script.json)
- [x] VOICE: v3 show audio, 152 s (voice/samples/show/). Live BRAIN drove all steps; Q&A 14–17 is live Gemma.
- [ ] VOICE: re-roll Q&A step 17 (best of 3 live takes), then send to MOTION
- [x] MOTION: v3 show video, videos/stage/show.mp4 (155 s, hero look, hw OK). Re-render takes ~3 min. Old v2 renders are in videos/stage/old_v2/.
- [x] MOTION: live.py (LiveWorld, ~120 fps) + render_log.py for room-sim runs (tested on a mock run only)
- [ ] Kirill/Steve: review the v3 script wording (especially Steve's lines 9–12, which BRAIN wrote)
- [ ] LIVE ROOM SIM (b2ncpfm7ctuvp6zex3mojzpb): one shared noisy room, Kermit's speaker in the room, live sim camera, wall-clock closed-loop run at 3 noise levels, video rendered from the log. Honest result by ~7:30am PT.
- [x] TURN: end-of-turn fusion model (silence + turn-1-mini + Smart Turn + LiveKit text), default `--endpoint eou`.
  - Cuts 4.6% of pauses on the Kermit set (Silero 0.5 s cuts 80%); median delay 433 ms live.
  - Speculative start saves ~0.36 s.
  - AEC3: 0% false barge-ins, ~350 ms to detect.
- [ ] TURN: non-blocking commit (Nemotron lagged >1 s on the loaded box), babble robustness (adaptive floor, max utterance), barge-in numbers in the room
- [ ] SPEAKER: near presenters (one each side) as the priority; facing is a soft cue; run on live sim frames
- [ ] MOTION: v3 layout (presenters close at ±45°, judges 3–4 m back), real-time live mode for the room sim

## Before doors (9:30am PT)
- [ ] Kirill: choose a voice (fenrir / puck / echo / michael / qwen)
- [ ] Kirill: choose a name ("Happy Kermit" / "Happy" / "It's Happy, actually!" correction)
- [ ] Kirill: send real lines for presentation beats 5–6 (back turned); placeholders are in voice/make_scene.py HUMAN_OVERRIDE
- [ ] Kirill: say which video is "the 26-second one"
- [ ] Kirill: AUDIO INPUT. In the room sim, a table mic 1.3 m away gets speech SNR ≈ −2 dB at a 60 dBA hackathon, and the first run landed only 1/4 beats. Plan: feed Kermit from the presenters' handheld mic (USB wireless receiver or audio interface into the Mac) or lav/headset mics; for Q&A, the passed-around judge mic. Check what AGI House's AV offers (PA mixer aux out?).
- [ ] Kirill: camera mount. At the arm base the presenters' faces are cropped and the puppet blocks the view; a raised side post (~25 cm to the side, ~55 cm up) shows faces. Option C, or a wide-angle camera.
- [ ] Kirill: stage format (pre-rendered video + live voice, or live sim player)
- [ ] Mac: copy kermit/, run brain/mac/setup.sh, voice/setup_mac.sh, mac_smoke.py, brain/mac/check.sh
- [ ] Mac: test tracking with a real person in front of the camera; set --hfov
- [ ] Ask Rocky/Jessica whether the Mac counts as on-device
- [x] BRAIN: LoRA v3 (judge Q&A + turn events) is the default; v2 is the fallback. Presentation/qa phase, deterministic turn-around beat.
  - 20-line eval: first sentence 348 ms, 57–59/60 expectations met, 60/60 valid
  - judge Q&A 20/20; off-script (OOD) 78/80
- [x] VOICE: Q&A 14–17 re-run on v3 with live takes, best of 3 (step 16 needed 6). All takes logged.
- [x] BRAIN: opinion + "no guessing numbers" lines in prompts.py.
  - robotics 16/16; opinion 20/20; head count/time 30/30; judge Q&A 20/20; interjections 10/10; OOD 79/80
- [x] VOICE: step 16 re-rolled on the new prompts.py. 3/3 takes confident; kept "It's about to explode! Small smart brains like mine will fit everywhere…". Show is now 153.6 s; MOTION told to re-render from 133 s.
- [x] SPEAKER TRACKING done (perception/eval/REPORT.md): speaker 92.4% on Columbia; turned-away detected 101/103 in the room sim; handheld mic P/R 1.0; 15–16 Hz. Judges WITHOUT a mic only 0.13 → hand judges a mic.
- [ ] TURN-TAKING: end-of-turn better than 0.5 s VAD (it currently splits sentences at "..."); barge-in + echo
- [ ] SWEEP: size sweep + mixed vs split → M5 Max model recommendation
- [x] MOTION: turn-around presentation video (videos/stage/presentation.mp4, 70 s), Q&A draft (qa_draft.mp4, 26 s), camera cuts
- [x] VOICE: Q&A audio with LIVE Gemma answers (voice/samples/qa/, 31 s). Re-run make_qa_scene.py once the top-up LoRA lands.
- [x] MOTION: Q&A video with the real judge audio and live Gemma answers (videos/stage/qa.mp4, 32 s); re-render the presentation with Kirill's real beat 5–6 lines
- [ ] Optional: re-render hero with live Gemma replies (honest "this is what Gemma said")
- [ ] Optional: live real-time sim player (~1–2 h)

## Later (after the demo)
- [ ] Fine-tune Kermit on the happy.engineering marketing + docs site, so he knows what Happy is, how it works and why it's cool. Scrape → Q&A pairs in persona → LoRA top-up → eval on product questions.
- [ ] Real dexterous hand model (find the model, then a HandProfile subclass)
- [ ] Real-arm replay of .traj.npz (needs explicit approval)

## Speed-ups (opt-in)
- [x] BRAIN: MTP drafter, `MTP=1 bash brain/mac/run.sh`. First sentence 422→339 ms, full turn 951→710 ms, same quality (Linux). Untested on Metal: keep it only if brain/mac/check.sh passes on the Mac.

## Perception v2 status (03:55)
- [x] Columbia panel: speaker_id 91.7%, fused F1 76, switch latency ~1.07 s. AVA ASD causal AP 66.6.
- [x] Mic detector: perfect on handhelds; false fires on table goosenecks → use `--mic-model ''` wherever there are table mics.
- [x] Room oracle: speaker_id 87% excluding out-of-FOV. Camera moved to HFOV 120°; re-scoring.
- [x] Demo MP4 (perception/demo/perception_v2_demo.mp4), Mac runbook, run_kermit integration
- [ ] VENUE: camera at Kermit head height (~0.5 m), ≥110° HFOV, 1080p capture

## Sweep: mixed vs split (03:58)
- [x] Mixed (one pass) wins: first sentence with gestures 348 ms vs 547–741 ms for split; full turn 645 ms vs 961–1241 ms; persona 3.33 vs 2.98–3.08. Gesture placement is a wash.
- [x] Mixed + MTP: 229 ms first sentence, 377 ms full turn (base E4B, clean GPU).
- [x] M5 Max recommendation (sweep/REPORT.md): keep E4B + LoRA, add MTP. Estimated first sentence ~205 ms on M5. 26B/31B not worth switching for the demo.

## Noise hill-climb + PTZ camera (Kirill, ~04:15)
- [ ] ROOM SIM + TURN: after the sweep, hill-climb the worst failure one change at a time. Stop after 2 non-improving iterations or at 8:00 PT. Multi-mic "loudest wins" dropped.
- [x] MOTION: Insta360 Link 2 PTZ (kermit/camera/: ptz.py v4l2/mac(linkcam)/sim/dry, aim.py director, LiveWorld ptz=True, mount (0.3,0.3,0.5)). Sim + dry-run only. Mac smoke test: camera/README.md. Demo: camera/videos/ptz_aim_demo.mp4
- [ ] ROOM SIM: add run_kermit `--ptz` hub hook (snippet in camera/README.md), then one normal_B run with PTZ vs the static 120° camera
- [ ] Kirill (Mac): turn off Gesture Control in the Insta360 app, install linkcam, run `ptz.py smoke --backend mac` (checks the pan sign)
- [x] PERCEPTION: PTZ contract implemented (pose_at / ndjson interpolation, ASD skipped while panning >30°/s, 30 s re-ID)
- Note: Link 2 can't localize sound. Voice/DOA tracking needs the Insta360 Wave speakerphone (8 mics), and no public DOA API is known.

## Gemma vision (Kirill, ~04:20)
- [ ] GEMMA VISION (zyhfzvxn2ghiuxzd90q64nc9): real stage-perspective test set (concerts, crowds, panel/town-hall Q&A, drinks in hand); Gemma 4 E4B vision vs specialist pipeline for speaker / raised-hand pick + describe / spatial commands; hybrid (specialists per frame + Gemma semantic layer at ~1 Hz); LoRA only if it clearly helps. Baseline ~6:30, integration ~8:30.
- [ ] PERCEPTION: front-row focus (ROI/depth limit), `hand_up` field + `hand_raised` event
- [x] BRAIN: spatial commands (12/12 open with the correct look; Kermit's own left/right), hand_pick → "Yes, you with the drink!", look_command (waits ≤1.5 s for VISION, plays it off if unresolved). Quirk: "look at me" sometimes greets Kirill by name.

## Room sweep s1 results (~04:30)
- Handheld mic (B): beats landed 15/17 quiet, 15/17 normal, 17/17 loud. Table mic (A) collapses: 10/17 at 60 dBA, 3/17 at 70 dBA. → HANDHELD MIC INTO THE MAC IS REQUIRED.
- AEC fix: self-echo cut-offs went from 13 to 0.
- Room latency mean 1.6–2.5 s, p95 2.5–4.1 s. The endpointer is 1.2–1.7 s of that, inflated by a CPU-saturated box (streaming ASR fell behind). Expect better on the Mac.
- Remaining: interjection beat (barge-in fix + oracle facing timing), 2 false cut-ins in normal_B.
- Videos: motion/videos/live/s1_normal_B.mp4 (honest log render), room/run/s1_normal_B/split.mp4 (mic waveform + live transcript)
- [ ] Hill-climb iteration 1 running
- [x] BRAIN fixes: "look at me" uses the current speaker (no guessed names); no-guessing rule rewritten (29/30 dodges, leakage 1/80); Q&A no longer interjects when a presenter turns toward a judge (only the current speaker, a non-presenter, triggers it). OOD 77/80, judge Q&A 19/20.
- [x] GEMMA VISION baseline: set-of-marks (numbered tracks → Gemma picks an id).
  - hand pick 87% (better recall than the specialists); describe ~91% usable (205 ms); described-look commands 83% (281 ms) vs 15% for rules
  - speaker: Gemma 65% vs Light-ASD 89% → ASD keeps it
  - no box offset at native aspect (never letterbox)
  - fine-tune probably not needed
- [x] GEMMA VISION done (kermit/vision/README.md): hybrid hand pick 95.7%, look commands 83% in 0.15–0.45 s, crowd fallback. `--vision` flag in run_kermit; second llama-server on :8081 (mac/run_vision.sh). Untested on the Mac.
- [ ] VENUE: give Q&A judges a mic (without one, judge speaker-ID is 0.13 in the sim); first minutes at the venue: `people.py --stats --render` check; `--mic-model ''` if there are table mics

## ROOM SIM final (~06:00)
- [x] Real time confirmed: handheld mics land 16/17 at 45 and 60 dBA, 15/17 at 70 dBA; room latency mean ~1.65 s, p95 2.4–2.7 s; echo triggers 0. Table mic: 10/17 at 60 dBA, 3/17 at 70 dBA.
- [x] Hill-climb stopped after iteration 3 (no ≥1 beat / ≥10% p95 gain). The biggest win was iteration 1 (interjection exempt from barge-in, ASR restart per turn).
- [x] PTZ verdict: worse than static (speaker in frame 23–54% vs 100%; speaker-ID 0.23–0.53 vs 0.89) → VENUE: Link 2 FIXED + WIDE, PTZ OFF.
- Latency knob = endpointer (turn_end 1.2–1.5 s). The box was CPU-saturated, so expect better on the Mac.
- [x] Final honest live render: motion/videos/live/it4_normal_B_static.mp4 (210 s, 16/17 beats, speaker-ID 0.89, interjection lands)
- [x] TURN: barge-in no longer drops human words; interjections are exempt (VOICE verified the repro: transcript complete, 0 non-JSON lines)
- [x] ROOM SIM: final PTZ run it6 failed (in frame 48%, speaker-ID 0.46; judges ~0%) → PTZ OFF FOR GOOD. Venue: Link 2 fixed + wide, AI tracking off.
- [x] ROOM SIM confirmation on the shipped code: normal_B 17/17 beats (0 missed / 0 false cut-ins / 0 self-triggers), latency 1.52 / 2.47 s, speaker-ID 0.89. quiet_B 15/17.
- [x] TURN: step 7 barge-in merge shipped (KERMIT_BARGEIN_MERGE_S=0 reverts). Didn't trigger in final3, so untested in the room.
- [x] BRAIN: step 6 fixed (a speaker_changed event cleared the "away" flag mid-presentation). Only turned_back clears it now, and the next line after the interjection always gets the back-turned reply. Confirmed in the room (final3).
- [x] FINAL (final3, shipped code, handheld mics): 45 dBA 17/17 beats, latency 1.41 / 2.18 s; 60 dBA 17/17, latency 1.57 / 2.50 s. 0 missed, 0 false cut-ins, 0 self-triggers in both.

## On-the-day fallbacks (if live misbehaves)
- End of turn: default `--endpoint eou` → `--endpoint vad` (Silero 0.5 s) → `--ptt` (push-to-talk)
- Echo: `--no-aec` only if echo cancellation eats words; barge-in `--bargein off`; step-7 merge `KERMIT_BARGEIN_MERGE_S=0`
- Brain: default base + LoRA v3; `run.sh fast` (merged); drop MTP if check.sh fails
- Whole show: play kermit/motion/videos/stage/show.mp4
