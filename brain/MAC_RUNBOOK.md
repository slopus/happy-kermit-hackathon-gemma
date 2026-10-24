# BRAIN on the M5 Max: runbook

Runtime: **llama.cpp with Metal** (`llama-server`) + `brain.py` (stdlib Python, works with macOS's `/usr/bin/python3`).
Everything runs on the laptop. After setup, switch Wi-Fi off and it still works.

## 0. What to copy to the Mac
The whole `kermit/brain/` folder is small, *except* `artifacts/`. You need:

| file | size | needed for |
|---|---|---|
| `kermit/brain/*.py`, `eval/`, `mac/`, `demo_script/` | <1 MB | always |
| `artifacts/kermit-E4B-lora.gguf` → `models/` | 70 MB | default mode (base + LoRA v3) |
| `artifacts/kermit-E4B-lora-v2.gguf` → `models/` | 70 MB | `v2` fallback mode |
| `artifacts/kermit-E4B-Q4_K_M.gguf` → `models/` | 5.3 GB | optional `fast` mode (merged) |

The base model (5.0 GB) is downloaded from Hugging Face by `setup.sh` at the exact revision the LoRA was trained against.
Verify the copies with `shasum -a 256 -c mac/SHA256SUMS`, run from `models/`.

## 1. One-time setup (needs internet, ~5 min)
```bash
cd kermit/brain
bash mac/setup.sh
```
This does the following:
- Clones llama.cpp and pins it to `970b51ce7c9b` (the commit tested here). It needs a recent commit for Gemma 4, `--reasoning off` and GBNF.
- Builds with `cmake -B build -DGGML_METAL=ON` and links `llama-server`/`llama-bench` into `models/`.
- Downloads `unsloth/gemma-4-E4B-it-GGUF@bfc15c38… / gemma-4-E4B-it-Q4_K_M.gguf`.

Then copy `kermit-E4B-lora.gguf` (and optionally `kermit-E4B-Q4_K_M.gguf`) into `kermit/brain/models/`.

## 2. Run
```bash
bash mac/run.sh          # default: base Q4_K_M + --lora kermit-E4B-lora.gguf (v3), prompt=persona3, phase=presentation
bash mac/run.sh v2       # previous adapter (fallback), prompt=persona3
bash mac/run.sh fast     # merged v2 kermit-E4B-Q4_K_M.gguf, prompt=persona3 (~50 ms faster)
bash mac/run.sh base     # no fine-tune: base model, prompt=full3 (rules + few-shots + facts)
PHASE=qa bash mac/run.sh # start in open conversation instead of the scripted show
MTP=1 bash mac/run.sh    # + MTP drafter (speculative decoding), combinable with any mode above
```
- llama-server runs on `127.0.0.1:8080`, and the brain on `127.0.0.1:8765`. Brain startup does a warm-up turn, so the first real turn is already fast.
- The exact server line is:
  `llama-server -m gemma-4-E4B-it-Q4_K_M.gguf --lora kermit-E4B-lora.gguf -ngl 99 -fa on -c 8192 -np 1 --reasoning off --reasoning-budget 0 --port 8080`
- **The prompt mode must match the model**: `persona3` with a fine-tune, `full3` with the base. The base model with a persona prompt produces emoji and odd gestures.
- Show control (CONTRACT §6):
  - Sessions start in the scripted `presentation` phase and switch to `qa` after the hand-off line.
  - `curl -s -X POST localhost:8765/phase -d '{"phase":"presentation","session_id":"kermit"}'` restarts the show.
  - `curl -sN -X POST localhost:8765/beat -d '{"id":"handoff","session_id":"kermit"}'` forces a beat.
  - Perception events go to `POST /event`.

API (CONTRACT §1 / §1b):
```bash
curl -sN localhost:8765/turn -d '{"heard":"hey kermit wake up","context":"one person in front of you. visible: audience, laptop","stream":true,"session_id":"demo"}'
# -> {"event":"header",...}  {"event":"sentence",...}...  {"event":"turn", <CONTRACT §1 JSON>}
curl -s localhost:8765/turn -d '{"heard":"..."}'          # non-streamed: plain turn JSON
curl -s -X POST localhost:8765/reset                        # forget conversation history
curl -s 'localhost:8765/state?session_id=kermit'           # phase, script cursor, next beat, speaker
```
In-process: `from brain import Brain; b = Brain(prompt="persona"); for ev in b.stream(heard, context): ...`

## 3. Check (2 min, in a second terminal)
```bash
bash mac/check.sh        # same mode argument as run.sh
```
It prints:
- `llama-bench` decode speed.
- One streamed turn with per-event timestamps.
- The 20-line eval. The targets are `valid 20/20`, `ms_header_med < 150` and `ms_first_sentence_med < 400`.

Finally, turn Wi-Fi off and run `check.sh` again.

## 4. Expected latency (after STT finishes)
Measured on an RTX 4000 Ada (CUDA, E4B Q4_K_M at 89 tok/s decode), median of 60 turns:

| mode | header (emotion) | first sentence (TTS can start) | full turn |
|---|---|---|---|
| base + LoRA (default) | 135 ms | 347 ms | 771 ms |
| merged (`fast`) | 106 ms | 278 ms | 645 ms |
| base, prompting only | 128 ms | 334 ms | 619 ms |

No M5 Max llama.cpp number for Gemma 4 E4B has been published. Closest data points:
- MLX on M5 Max: about 127 tok/s for E4B 4-bit.
- llama.cpp Metal on M3 Max: 73 tok/s for E4B Q4_K_M.

**Estimate for the M5 Max: 90–120 tok/s decode, so the numbers above or up to about 25% better.** That means about 0.1 s to the emotion header and **about 0.25–0.35 s to the first sentence**, both well under 1 s.

Per-turn prefill is about 30 new tokens. The system prompt prefix is cached, so prefill cost is negligible. Check the `llama-bench` tg128 number: if it's ≥ 89 t/s, you'll match or beat the table.

**MTP speculative decoding** (`MTP=1`) uses the official 100 MB drafter
`ggml-org/gemma-4-E4B-it-GGUF/mtp-gemma-4-E4B-it-Q8_0.gguf`, which setup.sh downloads:
`--spec-type draft-mtp -md … -ngld 99 --spec-draft-n-max 2`.
- On CUDA with LoRA v3: same quality (59/60 vs 60/60 expectations met). Back-to-back on a shared GPU, first sentence
  422 → 339 ms and full turn 951 → 710 ms.
- Draft acceptance is only 25–45% with the LoRA, because the drafter predicts base-model tokens, so keep n-max at 2.
- SWEEP measured base E4B going from 58 to 96 tok/s.
- It's untested on Metal. Run `MTP=1 bash mac/run.sh` + `bash mac/check.sh` and keep it only if valid stays 20/20 and
  the latency drops.

Latency levers, if needed:
- Use `fast` mode.
- Keep sentence 1 short. The fine-tune already opens with a median of 2–3 words.
- VOICE starts TTS on the first `sentence` event.

## 5. Fallbacks, in order
1. Use `bash mac/run.sh fast` if the LoRA fails to load. The merged file needs no `--lora`.
2. Use `bash mac/run.sh base` if the fine-tune misbehaves. It's the prompted baseline and is still 60/60 valid with grammar.
3. **Scripted opening scene**: `demo_script/turn_01..04.json` are pre-made CONTRACT turns, which VOICE/MOTION can play without any LLM.
4. MLX, **untested, not recommended**. `mlx_lm.server` has no GBNF, so the persona LoRA isn't converted to MLX.
   - Commands: `pip install -U mlx-lm` → `mlx_lm.server --model mlx-community/gemma-4-e4b-it-4bit --port 8080` → `python3 brain.py serve --prompt full --no-grammar --model mlx-community/gemma-4-e4b-it-4bit`.
   - Without grammar, the base model's lines are strictly valid only 41/60 times. kml.py repairs the rest, but expect stray tags or odd emotions.
   - mlx-lm issue #1242 reports Gemma 4 E4B breaking on mlx 0.31.2 / mlx-lm 0.31.3. Pin mlx 0.31.1 / mlx-lm 0.31.2 if you hit it.

## 6. Gotchas
- The `--reasoning off --reasoning-budget 0` flags are required. Otherwise Gemma 4 may spend tokens on thinking.
- Don't run two llama-servers. Memory is fine at 128 GB, but they compete for GPU time.
- `context` is free text, and only items in a `visible: a, b` list become legal look_at/gaze targets (spaces become `_`). The grammar enforces this.
- Explicit stage commands in the heard text ("nod if you agree", "look to your left", "shake your head", "look around", "do your surprised face") force the reply to open with that gesture (`kml.forced_gesture`).
