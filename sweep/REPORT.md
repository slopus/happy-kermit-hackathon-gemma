# Gemma model sweep for Happy Kermit (2026-10-24)

Question: are we using the best model or just the smallest, and does size buy quality?
Answer: **keep E4B + Kermit LoRA, and add the MTP drafter.** Base models gain persona with size, and only the base 31B ties the
LoRA. On the M5 Max the 31B would take about 1 s to the first sentence, against about 0.2 s for E4B. The
26B-A4B MoE is the only big model fast enough for real time, but prompt-only it scores below the LoRA.

Setup:
- Same KML eval as `brain/eval`: 20 lines, T=0.7, per-turn GBNF.
- llama-server on CUDA, RTX 4000 Ada (360 GB/s), frozen copy of the brain code in `snap/`.
- Base models use `--prompt full` (rules + 8 few-shots). The LoRA uses `--prompt persona`.
- Persona: blind 1–5 scoring of all configs side by side, shuffled and unlabeled (`blind.py`, `ab/`). I judged it myself.
  Caveat: I had seen a few raw outputs in the logs.

## 1. Gemma lineup → [SOTA.md](SOTA.md)
| model | params | modalities | context | license | on-device files |
|---|---|---|---|---|---|
| Gemma 4 E2B | 2.3B effective (5.1B with embeddings) | text, image, audio | 128K | Apache 2.0 | GGUF + MLX |
| Gemma 4 E4B | 4.5B effective (8B with embeddings) | text, image, audio | 128K | Apache 2.0 | GGUF + MLX |
| Gemma 4 12B "Unified" | 11.95B dense | text, image, audio | 256K | Apache 2.0 | GGUF 7.1 GB, MLX |
| Gemma 4 26B-A4B (MoE) | 25.2B total, 3.8B active | text, image | 256K | Apache 2.0 | GGUF 14.4–17 GB, MLX |
| Gemma 4 31B | 30.7B dense | text, image | 256K | Apache 2.0 | GGUF 17.7–18.3 GB, MLX |

Also available: MTP "-assistant" drafters for every size (ggml-org `mtp-*.gguf`); Gemma 3n E2B/E4B (32K context, Gemma license);
FunctionGemma 270M. There are no persona fine-tunes better suited than our LoRA.

## 2. Size sweep (60 turns per config; latency from the clean pass where available)
| config | expectations met | persona (1–5) | vs LoRA W/T/L | 1st sentence | total | tok/s |
|---|---|---|---|---|---|---|
| E2B Q8, prompt | 84/100 | 2.87 | 3/12/45 | 239 ms | 438 ms | 99 |
| E4B Q4, prompt | 85/100 | 3.37 | 5/18/37 | 348 ms | 645 ms | 79 |
| **E4B Q4 + LoRA (current)** | **97/100** | **4.23** | ref | 361 ms | 770 ms | 67 |
| 12B Q4_K_M, prompt | 79/80 | 3.82 | 11/18/31 | 636 ms* | 1478 ms* | 33* |
| 26B-A4B Q4, prompt | 124/140 | 3.93 | 9/24/27 | 300 ms | 626 ms | 76 |
| 31B Q4_K_M, prompt (2 GPUs, `-ts 3,2`) | 60/60 | 4.20 | 21/10/29 | 1293 ms | 3279 ms | 15 |

\* The 12B ran under heavy CPU load (load 15–30). Its llama-bench numbers are tg 35 / pp512 2073 tok/s.

Files: `results/{E2B_Q8_full,E4B_Q4_full,E4B_Q4_lora,12B_Q4_full,26B_QAT_full_q,31B_Q4_full}.json` (quality) and `results/C_*.json`
(clean latency, GPU1 exclusive). The persona column comes from the first set.

Takeaways:
- **Size helps prompt-only quality monotonically:**
  - E2B 2.9, E4B 3.4, 12B 3.8, 26B MoE 3.9, 31B 4.2 on persona.
  - The 31B met 60/60 expectations and is the only base model that ties the LoRA (4.20 vs 4.23; it has more wins, 21, but also more losses).
- **The LoRA is worth about a 7× bigger model.** A 70 MB adapter on E4B matches the base 31B.
- **26B-A4B MoE** decodes as fast as E4B (3.8B active) but needs 14.5 GB. Prompt-only it beats base E4B and loses to the LoRA.
- **Gotcha: 12B, 26B and 31B need `--swa-full`.** Without it, llama-server re-processes the whole prompt every turn (first token
  850 ms instead of 70 ms). E2B and E4B are unaffected.

## 3. Mixed (one KML pass) vs split (text, then gesture annotation) — clean, same GPU, base E4B
| config | text 1st sentence | 1st sentence with gestures | total | expectations met | gestures/turn | trigger hit | persona |
|---|---|---|---|---|---|---|---|
| **mixed KML** | – | **348 ms** | **645 ms** | 33/40 | 2.30 | 81% | **3.33** |
| split, E4B → E4B annotator (2 slots) | 366 | 741 | 1241 | 33/40 | 1.85 | 70% | 2.98 |
| split, E4B → E2B Q8 annotator | 237 | 547 | 961 | 34/40 | 2.00 | 86% | 3.08 |

How split works here: it's pipelined (sentence k is annotated while the text model writes k+1). The annotator's grammar forces the
exact words, so it can only add tags (`run_sweep.py --mode split`).

**Mixed wins, confirming Kirill's prior:**
- The annotation pass costs 290–350 ms per sentence (it has to re-emit the words), so sentence 1's gestures arrive 200–400 ms later.
- Persona is lower with split (blind W/T/L for mixed: 18/17/5 and 16/16/8): fewer gestures, and the text can't adapt to a gesture.
- Placement accuracy is a wash.
- Split only reaches the first *text* sooner (237 ms), and that pays off only if TTS starts before the gestures are known.

## 4. M5 Max recommendation (ESTIMATES)
How the estimates are built:
- M5 tok/s = measured RTX server tok/s × (M5 raw ÷ RTX raw decode). This carries the grammar, LoRA and MTP effects over.
- Prefill is roughly at parity: M5 Max pp512 is 3220 tok/s vs 3428 on this RTX (Llama-2-7B Q4_0).
- Script: `m5_estimate.py`.

M5 Max 40-core llama.cpp raw decode used:
- E4B ~125 (mlx-vlm 127)
- E2B ~150 (oMLX 8-bit 144)
- 12B ~55 (oMLX QAT 46)
- 26B-A4B ~75 (llama.cpp via Ollama, measured; MLX does 100–116 but has no grammar)
- 31B ~25 (MLX 4-bit 26.5)
- Calibration anchor: Llama-2-7B Q4_0 at tg 120 on the M5 Max vs 59 on this RTX.

| config | persona | expectations met | **M5 1st sentence (est.)** | M5 total (est.) | switch? |
|---|---|---|---|---|---|
| **E4B + LoRA (current)** | **4.23** | 97% | **~230 ms** | ~480 ms | keep |
| **E4B + LoRA + MTP (n-max 2)** | 4.23 (same distribution) | 95% | **~205 ms** | ~380 ms | **yes, opt-in in BRAIN's runbook** |
| E4B merged LoRA + MTP | ~4.1 (merged lost 1–3 expectations in BRAIN's runs) | 97% | ~165 ms | ~305 ms | fast fallback |
| E4B base + MTP | 3.37 | 85% | ~155 ms | ~245 ms | no (persona) |
| 26B-A4B + MTP, prompt | 3.93 | 89% | ~280 ms (MLX: ~250, no grammar) | ~575 ms | no: slower *and* worse than the LoRA; 14.5 GB |
| 12B, prompt | 3.82 | 99% | ~430 ms | ~950 ms | no |
| 31B, prompt | 4.20 | 100% | ~970 ms | ~2.4 s | no: ties on persona at 4–5× the latency; a 1 s pause kills the puppet illusion |

Bottom line:
- "Big base model, prompt only" never beats "E4B + LoRA" on persona.
- The only base model that ties it (31B) is about 1 s slower to start speaking on the Mac.
- For the hardware flex, say what's true: E4B + LoRA + MTP speculative decoding, about 0.2 s to first speech.
- An upgrade worth trying after the hackathon: put the same LoRA on the 26B-A4B (MoE, same speed class as E4B; it
  starts higher on persona prompt-only). Not feasible today: training the 26B needs more than one 20 GB GPU.

## Gotchas found
- `--swa-full` is required for 12B, 26B and 31B (see above).
- MTP (`--spec-type draft-mtp -md mtp-gemma-4-E4B-it-Q8_0.gguf -ngld 99`):
  - Draft acceptance: 46% on base E4B, 25–35% with the runtime LoRA, 35% on the merged model.
  - Use `--spec-draft-n-max 2` with the LoRA.
  - Works with GBNF.
  - Verified by BRAIN on LoRA v3: 59/60 vs 60/60 expectations met, first sentence 422 → 339 ms.
- The 26B (ggml-org Q4_0, 14.6 GB) plus MTP needs `-c 4096` on a 20 GB GPU.
- This box: decode on small models is partly CPU-bound (sampling over a 262K vocab under load 10–30), so E2B and E4B
  show similar tok/s in loaded runs. The clean pass (`C_*`) is the latency reference.

## Files
- `run_sweep.py`: mixed and split eval
- `serve.sh`, `sweep_gpu0.sh`, `big_window.sh`, `clean_pass.sh`: runners
- `analyze.py`: summary table with trigger-hit
- `blind.py`, `ab/`: blind persona scoring (`blind.md` + `scores.json`; `blind_split.md` + `scores_split.json`)
- `m5_estimate.py` → `results/m5_estimates.md`
- `logs/`: server and eval logs, plus llama-bench output
