# Happy Kermit: VOICE

Everything audio, on-device: hear the user (Parakeet), speak (Kokoro with a frog DSP pass), and produce the
puppet's mouth curve and gesture times (CONTRACT §2 timeline).

```
mic ─VAD/PTT─► Parakeet TDT 0.6b v2 (sherpa-onnx int8, CPU) ─► BRAIN :8765 NDJSON (header, sentence…, turn)
      sentence k ─► Kokoro-82M ─► frogify (pitch+formant shift, nasal EQ) ─► trim ─► words + mouth + events
      ─► stdout NDJSON "segment" per sentence  ─►  turn_NNNN.wav + turn_NNNN.timeline.json
```

## Files
| file | what |
|---|---|
| `kermit_voice.py` | end to end: `--wav`, `--mic [--ptt]`, `--heard "…"`, `--turn turn.json`, `--text "…"` → `out/turn_NNNN.{wav,timeline.json}` |
| `tts.py` | `KokoroTTS(voice, backend="torch"\|"mlx", device)` `.synth(text, emotion) → (audio24k, words)` plus emotion→voice table and `frogify` |
| `lipsync.py` | `mouth_curve(audio, sr, words, emotion) → 60 Hz openness` |
| `timeline.py` | `align_words` (char-level alignment of text words to TTS/ASR words) and `build_timeline` |
| `stt.py` | Parakeet recognizer, Silero VAD / push-to-talk `Listener`, `words_from_audio` (ASR as forced aligner) |
| `mac_smoke.py`, `setup_mac.sh`, `requirements.txt` | Mac setup and check |
| `make_samples.py` | re-renders `samples/` |
| `samples/` | 6 demo lines plus `demo_script/turn_01..04` (hero scene) and `heard_01..04` (human lines) |
| `bench/` | TTS candidate outputs plus results (`kokoro/`, `frog/`, `qwen3/`, `breeze/`), mouth plots |

## TTS decision

**Primary: Kokoro-82M with a "frogify" DSP pass.** Runs through mlx-audio on the Mac GPU, or the pip `kokoro` package on CPU/MPS.
**Fallback:** the same model on CPU with 8 threads, about 0.3 s per sentence.
**Optional:** pre-rendered Qwen3-TTS VoiceDesign frog clips for the scripted intro, if they sound much better to a human ear. The voice was designed from text, so no clone of real Kermit audio was used.

Measured on the Linux dev box (RTX 4000 Ada / EPYC 7B13). Pitch is the median F0, and range is p5–p95 in semitones. WER is a Parakeet round trip over the 6 demo lines.

| engine | first audio / line | RTF | stream | word times | emotion ctl | pitch, range | WER | verdict |
|---|---|---|---|---|---|---|---|---|
| **Kokoro am_fenrir + frog** | **75–120 ms** GPU, 0.3–0.6 s CPU | 0.015–0.03 GPU, 0.23 CPU | per sentence | **exact** (duration predictor) | speed/pitch/level (ours) | 177 Hz, 13.7 st | 1.3% | **primary** |
| Kokoro am_puck + frog | same | same | same | exact | same | 146 Hz, 10.5 st | 1.3% | alt voice |
| Qwen3-TTS 1.7B VoiceDesign | 7–17 s (whole line) | 2.07 | no (qwen-tts 0.1.1) | no | text instruct | 378 Hz, 15.6 st | 2.7% | pre-render only |
| Qwen3-TTS 1.7B / 0.6B Base clone | 7–17 s | 2.05 | no | no | none | 312 Hz, 16.4 st | 0–2.7% | out |
| Breeze TTS 2 (voice direction) | 0.67–0.74 s | 3.24 | yes | no | text instruct | 335 Hz, 14 st | 2.7% | out: slower than real time, **non-commercial outputs** |
| Chatterbox | not run (out of collaborator slots and disk) | | no | no | exaggeration | | | untested |

Why Kokoro: it is the only candidate that streams sentence by sentence well under real time and gives
exact word timings for free. Character comes from DSP, which costs nothing in latency:

- render at `speed/f`, then resample by `f = 2^(st/12)`, so pitch **and formants** go up together (a smaller
  vocal tract gives a cartoon creature) while the speaking rate returns to the target;
- EQ: -6 dB low shelf at 180 Hz, +6 dB at 1.3 kHz (nasal, reedy), +3 dB at 3.2 kHz;
- emotion maps to (speed, semitones, level): excited 1.22×/+4.5, surprised 1.15×/+5, happy 1.12×/+3.5,
  neutral 1.08×/+3, embarrassed 1.05×/+3.5, thinking 0.98×/+2.5, annoyed 1.1×/+2, sad 0.92×/+2.

**Nobody has listened yet. I (the agent) can't hear audio.** Pick the voice by ear:
`bench/frog/{am_fenrir,am_puck,am_echo,am_michael}_{frog,raw}/` and `bench/qwen3/design/`. Switch with
`KERMIT_VOICE=am_puck`. The pitch shift and EQ live in `tts.EMOTION_VOICE` and `tts.EQ`.

## Mouth curve
Audio envelope, not visemes. The puppet has one degree of freedom, and real puppeteers flap once per syllable.
`lipsync.mouth_curve` works like this:

1. take the 250–3500 Hz RMS in dB at 60 Hz;
2. normalize to the sentence's own p90 level over a 28 dB range;
3. split into a phrase level (220 ms EMA) and the syllable excess, and boost the excess so the mouth closes between syllables;
4. open fast (10 ms) and close slower (45 ms);
5. soft-clip at the emotion's maximum (excited and surprised 1.0, sad 0.6, thinking 0.7);
6. outside words, scale to 25%, then zero anything below 0.04;
7. dip to 20% for about 60 ms at m/b/p word onsets.

`lead_ms` shifts the whole curve earlier if the actuator lags. See `bench/mouth_final.png`.

Steve's on-device lip-sync prototype ("Happy Bit") wasn't found on this node. If it turns up, it only needs
to produce the same `mouth.values` array.

## Timeline (CONTRACT §2) and streaming
`turn_NNNN.timeline.json` contains `audio`, `sample_rate`, `duration`, `words[{w,t0,t1}]` (exactly the
whitespace words of `text`), `mouth{hz,values}`, `events[{t,type,intensity,word,target?}]`, `emotion`, `gaze`,
plus extra `text` and `latency` fields.

Live mode prints one NDJSON line per sentence as soon as its audio exists:
`{"event":"segment","index":k,"audio":"turn_NNNN_sK.wav","t_offset":s,"duration":…,"words":[…],"events":[…]}`
with times on the turn clock and 120 ms gaps between sentences. The last line is `{"event":"done",…,"latency":{…}}`.
Gesture word indices are global, as in BRAIN's stream (CONTRACT §1b). Emoji and markup are stripped before TTS.
If such a "word" is left with no audio, it still gets a time interpolated from its neighbors.

## Latency (speech end → first audio)
Measured here on synthesized questions in a warm process, with llama-server on CUDA:

| config | STT | BRAIN 1st sentence (after STT) | TTS 1st sentence | **first audio** | full turn ready |
|---|---|---|---|---|---|
| **default: BRAIN server (`brain/mac/run.sh`): base E4B Q4 + Kermit LoRA, persona prompt**, over HTTP :8765, GPU TTS, CPU STT, 8 turns | 156–225 ms | +240–540 ms | 67–82 ms | **560–794 ms (median ≈705)** | 1.05–1.38 s |
| `run.sh fast` (merged Kermit GGUF), in-process brain, GPU TTS, 8 turns | 150–353 ms | +130–590 ms | 66–90 ms | 532–936 ms (median 690) | 0.92–1.23 s |
| merged GGUF, CPU TTS (8 thr) | 158–222 ms | +120–400 ms | 543–703 ms | **977–1230 ms** | 2.4–2.9 s |
| *earlier: base Gemma E4B Q4 + full prompt:* | | | | | |
| GPU TTS, CPU int8 STT | 150–234 ms | +400–520 ms | 76–83 ms | **666–834 ms** | 0.97–1.23 s |
| CPU TTS (8 thr), CPU STT | 186–276 ms | +430–660 ms | 306–420 ms | **1.03–1.25 s** | 2.0–2.2 s |

Budget on the Mac (estimate, not measured; check with `mac_smoke.py`):

| stage | budget |
|---|---|
| end-of-speech detection | PTT 0 ms; VAD = `min_silence` 500 ms (350 ms is OK in a quiet room) |
| STT (Parakeet int8, CPU) | 100–200 ms |
| BRAIN header + 1st short sentence (Gemma E4B, Metal) | 400–600 ms |
| TTS 1st sentence (Kokoro MLX / CPU) | 50–150 ms / 200–400 ms |
| audio device start | about 20 ms |
| **total** | **PTT about 0.6–1.0 s; VAD about 1.1–1.5 s.** The 1.5 s target is met; use PTT in a noisy room. |

BRAIN is the biggest share, and it's driven by the length of the first sentence. BRAIN was asked for short openers ("Whoa!", "Hi-ho!").

Gotcha: uncapped torch CPU threads next to onnxruntime took **13 s** per sentence. `tts.py` caps them at 8 (`KERMIT_TORCH_THREADS`).

## STT
Parakeet TDT 0.6b v2 through **sherpa-onnx int8 on CPU**: the same code and model tested here, 0 WER on commands,
and the same package provides Silero VAD and word timestamps. Here it takes 150–280 ms per 2–4 s utterance on EPYC
with 4 threads, and about 20 ms with fp16 CUDA. The Mac should be faster per core. Alternative if STT ever matters:
`parakeet-mlx` (`mlx-community/parakeet-tdt-0.6b-v2`) on the Mac GPU. Not integrated.
CoreML EP isn't worth it for transducers.

End-of-speech: `--ptt` (press Enter to start and to stop, zero hangover; the recommended demo mode) or Silero VAD with 0.5 s silence.

## Mac runbook (M5 Max)
Person tracking and the all-in-one loop (`../run_kermit.py`) are documented in `../perception/README.md`.

```bash
# 0. copy kermit/ to the Mac (brain/ is needed for the live loop), then:
cd kermit/voice
brew install uv            # plus llama.cpp for BRAIN: brew install llama.cpp
./setup_mac.sh             # venv, Parakeet int8, silero VAD, Kokoro (torch + MLX) weights, about 1.5 GB
export HF_HUB_OFFLINE=1 PYTORCH_ENABLE_MPS_FALLBACK=1

# 1. smoke test (1–2 min): every TTS backend, STT timing, prints which backend to use
.venv/bin/python mac_smoke.py          # → "USE: --tts-backend mlx" (expected); listen to out/smoke/*.wav

# 2. offline render: no LLM needed
.venv/bin/python kermit_voice.py --turn ../brain/demo_script/turn_01.json --tts-backend mlx --play
.venv/bin/python kermit_voice.py --text "Hi-ho! Whoa, that is a big robot arm." --emotion excited --play

# 3. live loop. BRAIN owns the model/prompt pairing: start it with its own script (see brain/mac/)
bash ../brain/mac/run.sh &        # default: base Q4 + Kermit LoRA | `run.sh fast`: merged GGUF | `run.sh base`: no adapter
#   optional: `MTP=1 bash ../brain/mac/run.sh` (MTP speculative decoding, ~80 ms faster 1st sentence on CUDA; untested on Metal, compare with brain/mac/check.sh first)
.venv/bin/python kermit_voice.py --mic --ptt --tts-backend mlx --play      # talks to BRAIN at :8765; drop --ptt for VAD
.venv/bin/python ../run_kermit.py --camera 0 --ptt --tts-backend mlx --play # same, plus person tracking
#   no BRAIN server? --brain-url inproc --llm http://127.0.0.1:8080 --brain-prompt persona3|full (must match the model; BRAIN's run.sh knows)
#   outputs: out/turn_NNNN.wav + .timeline.json; per-sentence NDJSON on stdout for MOTION
```
macOS asks for microphone permission for the terminal on the first `--mic` run. If `--tts-backend mlx` fails,
use `--device cpu` (the tested fallback) or `--device mps`.

Env knobs: `KERMIT_BRAIN_URL` (http://127.0.0.1:8765 or `inproc`), `KERMIT_BRAIN_PROMPT` (persona3, inproc only), `KERMIT_VOICE` (am_fenrir), `KERMIT_TTS_BACKEND`, `KERMIT_TORCH_THREADS` (8), `KERMIT_STT_PROVIDER` (cpu),
`KERMIT_LLM` (inproc only), `KERMIT_MODELS`.

## Full duplex and barge-in (`../run_kermit.py`)
`run_kermit.py` is the live hub. It runs mic capture (it never pauses), the camera, BRAIN, TTS and one output stream.
- **Interjections:** a perception event (CONTRACT §5) goes to BRAIN `POST /event`, together with `heard_partial`
  (Parakeet on the human's utterance so far). The reply plays immediately, even mid-utterance, and preempts any
  queued reply (`Speaker.interrupt()`). If BRAIN is unreachable, a canned line plays. All events are forwarded.
- **Silence:** BRAIN may answer `/turn` or `/event` with `{"event":"skip"}`, and Kermit then stays quiet.
- **Extra stdout events:** `play` (`t_wall` when a sentence really starts, the A/V sync point), `stop`
  (cut off), `perception`, `heard`, `skip`.
- **Headless test:** `--mic-wav` + `--events-script` + `--no-speaker`. The whole 8-beat presentation was replayed this
  way against `brain.py serve --prompt persona3 --phase presentation`, and all 8 beats fired in order. "Oh, like..."
  came 155 ms after `turned_away`; scripted replies took 0.9–1.0 s after speech end, including the 0.6 s VAD hangover.
- **Owners:** the reply path (Mic, endpointing, echo, human barge-in) belongs to TURN-TAKING (`kermit/turn/`), and
  perception v2 (`perception/people.py`) plugs into `Perception`. VOICE owns `Speaker`, `event_loop` and `voice/*`.

## Hero-video scene
`make_scene.py` renders BRAIN's `presentation_script.json` (8 beats) into `samples/scene/`:
- `scene.wav`: the 68 s mix, with the human in voice am_michael;
- `scene.json`: clips with absolute times, plus a scene-wide mouth track and events;
- per-turn `kermit_NN.{wav,timeline.json}` and `human_NN.wav`.

Kirill's turned-away words (turns 5–6) are placeholders in `HUMAN_OVERRIDE`.
TTS turns "..." into a comma pause, because Kokoro read "like..." as "light".

## v3 show and Q&A renders
- `make_show.py` runs BRAIN's v3 script (`brain/demo_script/v3/script.json`, 17 steps) through the **live** BRAIN
  server, using the same `/turn`, `/event` and `speaker_changed` calls as the hub. Output goes to `samples/show/`
  (152 s, presenters kirill and steve plus judges 1–3).
- `make_show.py --reroll N [--takes 3] [--pick K]` re-asks one live Q&A step (fresh sessions, never the fallback lines), logs every take in `brain_log[].rerolls`, keeps the best or picked take, and re-mixes the scene in place, shifting later clips.
- `make_qa_scene.py` produces three judge questions with live Gemma answers in `samples/qa/`.
- Both write `scene.wav` and a `scene.json` with `people`, `clips` (absolute times, speaker ids, staging, gaze targets,
  `source`: scripted_beat or live_gemma), a scene-wide mouth track and events, and a `brain_log` of BRAIN's raw output.
- Pronunciation: `tts.RESPELL` maps LoRA → "Lora", because Kokoro said "low raw". Check new words with a Parakeet round trip.

## Not verified
- **No run on Apple Silicon.** mlx[cuda] installs here, but its kernel JIT fails on CUDA 13 fp4 headers, so the
  mlx-audio code path was only checked by reading the source. mlx-audio 0.6.1's `KokoroPipeline` has the same
  `(lang_code, model, repo_id)` constructor and `tokens[].start_ts/end_ts` as pip kokoro. `mac_smoke.py` checks it in a minute.
- Voice quality hasn't been judged by a human.
- Chatterbox wasn't benchmarked.
