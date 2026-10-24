# Happy Kermit: TURN-TAKING

This module answers two questions: when has the human finished, and when are they talking over Kermit? It replaces Silero's fixed 0.5 s silence as the reply trigger. Push-to-talk and `--endpoint vad` still work as fallbacks.

```
mic 16 kHz ─► AEC3 (WebRTC via livekit.rtc; the reference is Speaker's output on a perf_counter timeline)
           ─► Silero VAD, adaptive floor ─► speech / pause, trailing silence
           ─► turn-1-mini (streaming, ~1 ms per 160 ms) ─► P(end of turn) from prosody
           ─► Nemotron-3.5 streaming ASR (own thread) ─► partial transcript every 160 ms
in a pause, every 100 ms: Smart Turn v3.2 (once, at 0.2 s) + LiveKit text model (partial + Kermit's last line)
           + text form ─► gradient-boosted fusion (models/fusion.json) ─► policy:
           candidate (p ≥ 0.7)  → speculative BRAIN + TTS; audio is held
           commit (silence ≥ 0.4 s and p ≥ 0.92, or silence ≥ 2.0 s) → held audio plays; the text sent to BRAIN is
                  Parakeet TDT, run on the utterance at 0.2 s of silence
           speech resumes → cancel (held audio dropped; if Kermit already started, he stops)
Kermit audible + human speech (p ≥ 0.6 for ≥ 200 ms, ≥ 6 dB over the learned echo residual) → barge-in → stop
```

## Files
| file | what |
|---|---|
| `endpoint.py` | `Endpointer(on_turn_end, on_candidate, on_cancel, on_bargein)`, `feed(t, block)`, `far(x, sr, t)`, and the model wrappers (Silero, Turn1Mini, SmartTurn, TextTurn, StreamingASR, EchoCanceller, Fusion) |
| `../run_kermit.py` | `EouMic` + hub hooks `on_turn_end/candidate/cancel/bargein`. `--endpoint eou` is the default and falls back to `vad` if models are missing |
| `setup_models.sh` | fetches the models (~560 MB). `models/fusion.json` and `models/smart-turn/mel80.npy` come with this folder |
| `bench/` | benchmark: `scores_*.py` (per-method scores on the eot-bench grid), `eval.py`, `fit_fusion.py`, `fit_live.py`, `replay.py` (live endpointer offline), `bargein.py`, `room_offline.py`, `room_turns.py` (ROOM SIM runs) |

`run_kermit.py` flags:
- `--endpoint eou|vad`
- `--no-speculative`
- `--no-aec` (headphones). `--no-speaker` turns the AEC off automatically.
- `--bargein stop|off`
- `--bargein-ms 200`
- `--final-asr tdt|stream`

Env switches:
- `KERMIT_TURN_TDT=0`: no TDT at the pause
- `KERMIT_ASR_MAX_BEHIND=1.5`: streaming ASR jumps ahead when starved
- `KERMIT_AUDIO=roomsim`: ROOM SIM's audio shim

## Results (all CPU, causal)
**LiveKit eot-bench EN** (400 real turns, 850 mid-turn pauses; eot-bench's own policy sweep). cut@X = share of mid-turn pauses cut into at ≤ X ms mean dead air; lat@5% = dead air when cut-ins are held to ≤ 5%.

| method | cut@300 | cut@600 | lat@5% |
|---|---|---|---|
| silence only (ideal VAD) | 55.6% | 21.7% | 2000 ms |
| Silero 0.5 s (the old trigger) | 37% of pauses cut | | 442 ms |
| Smart Turn v3.2 | 34.8% | 15.2% | 1031 ms |
| turn-1-mini | 22.7% | 9.6% | 849 ms |
| parakeet_realtime_eou_120m | – | 21.1% | 2000 ms |
| LiveKit text on Nemotron partials | 41.4% | 13.3% | 1040 ms |
| Gemma E2B yes/no judge | 73.5% | 16.2% | 1184 ms |
| **fusion (shipped)**, grouped CV | **20.3%** | **6.1%** | **662 ms** |

**Kermit demo set** (150 Kokoro turns with 0.35–1.6 s hesitations):
- silence only: 99% / 80% / 2000 ms
- fusion: 33.5% / 14.6% / 923 ms

**Shipped policy:**
- eot-bench: 4.8% of pauses cut, 818 ms mean delay
- Kermit set: 4.6% cut, 923 ms
- live replay through `endpoint.py` (40 Kermit turns): 3.1% of pauses cut, median 467 ms after the last word

**Speculative start at p ≥ 0.7:**
- saves 260–400 ms (mean) per turn
- 0.15–0.3 wasted BRAIN starts per turn (they are cancelled; no audio plays)

**Streaming ASR, WER on eot-bench / Kermit:**
| ASR | eot-bench | Kermit | notes |
|---|---|---|---|
| Nemotron 160 ms | 11.9% | 3.7% | last word ~0.3 s late |
| Parakeet TDT offline | 12.4% | 0.7% | |
| parakeet EOU | 37% | 38% | dropped |

**Barge-in / echo:**
- Offline leakage sim:
  - without AEC: false barge-ins on 100% of Kermit's lines
  - with AEC3 + gate: 0% false barge-ins, 79–100% detected, ~300–400 ms after onset
- ROOM SIM (loudspeaker 0.32 m, RT60 0.86 s):
  - self barge-ins: 13 → 0 once the reference was time-aligned
  - interjections are exempt from human barge-in

**ROOM SIM, handheld mic (config B):**
- 15–17/17 beats landed
- p95 room latency 2.4–2.7 s on a CPU-saturated shared box
- TDT-at-pause A/B: p95 3.68 → 2.46 s

## Known limits
- Table mic in 60 dBA babble (speech SNR around 0 dB): Silero and both ASRs miss the human. No endpointing fixes that, so use the handheld mic.
- AEC3's suppressor can drop near-end words in double talk while Kermit is speaking. Outside Kermit's speech the raw mic is used.
- The audio models were trained on human speech. Reverberant TTS humans (ROOM SIM) push turn-1-mini low, so commits there lean on silence + Smart Turn + text, at ~1.2–1.5 s endpoint.
