# Happy Kermit: BRAIN

Gemma 4 E4B, on-device, turns the heard text (plus camera context) into the CONTRACT §1 turn JSON: in-character
text, emotion, word-anchored gestures and gaze. It also streams per-sentence events (§1b) so TTS can start early.

**Pick for the demo: Gemma 4 E4B Q4_K_M + Kermit LoRA v3 (`artifacts/kermit-E4B-lora.gguf`), `--prompt persona3`,
served by llama.cpp with GBNF.** For the Mac, see [MAC_RUNBOOK.md](MAC_RUNBOOK.md). The v2 adapter is kept as
`kermit-E4B-lora-v2.gguf`, the fallback.

## v3: stage show, perception events, judge Q&A (Oct 24)
- **The show** (CONTRACT §6) lives in `presentation.py`, and `demo_script/make_script.py` renders it to
  `demo_script/v3/script.json`: 17 steps with speaker, human line, staging and the Kermit beat + turn JSON.
  - speech1 (Kirill): the 4 hero lines, then the distracted beat ("Oh, like..." → "You, my friend…" → "Respect accepted").
  - speech2 (Steve): Gemma/LoRA, Parakeet/Kokoro, gesture tags, robotics.
  - The hand-off, then live Q&A with fact-checked fallbacks.
- **Phases** (per session):
  - `presentation`: plays the beats instantly, with no LLM call. Off-script lines return `{"event":"skip"}`, unless
    they're addressed to Kermit (his name or a direct question), which Gemma answers.
  - `qa`: starts automatically after the hand-off. Gemma answers with `prompts.FACTS` in the system prompt.
- **Endpoints:**
  - `POST /event` handles turned_away / turned_back / speaker_changed / person_arrived / person_left.
  - turned_away while_speaking in presentation plays the scripted "Oh, like...".
  - turned_away while_speaking in qa gives a short Gemma interjection: grammar-capped at 12 words, 20 s cooldown,
    never while a judge is speaking.
  - speaker_changed with role=judge forces gaze=speaker and adds the judge note to the next turn.
  - `POST /phase` switches phase; `POST /beat` is an operator override; `GET /state?session_id=…` shows phase, script cursor, next beat and speaker.
- **Cancellation:** closing a /turn or /event connection stops generation (llama-server cancels the task). The turn
  isn't recorded in history.
- **LoRA v3:** trained from base on the D chat data (844 examples, ≤5-word replies and duplicates of E dropped) plus
  255 new E examples: 171 judge Q&A, 35 interjections, back-turned / turned-back callouts and 22 chains. Trained with
  `persona3` (persona + FACTS + event rules), 2 epochs, val loss 0.774.

| v3 evals (RTX 4000 Ada, uncontended) | base + full3 | LoRA v2 + persona3 | **LoRA v3 + persona3** |
|---|---|---|---|
| 10 judge questions × 2: facts present, no banned claims, gaze=speaker | 17/20 | 17/20 | **20/20** |
| judge latency: 1st sentence / total | 447 / 717 ms | 420 / 802 ms | 397 / 797 ms |
| interjections in 2–8 words | 10/10 | 10/10 | 9/10 (one 9 words) |
| 40 OOD prompts × 2 phases (auto-check) | – | 77/80 | **78/80** |
| 20-line persona eval, expectations met | 52/60 | 57–58/60 | **57–59/60** (2 runs) |
| 20-line eval latency: header / 1st sentence / total | 128 / 334 / 619 ms | 135 / 347 / 771 ms | 130 / 348 / 795 ms |

- **Prompt-only fix after VOICE's field report** (no retraining): FACTS gained Kermit's opinion on robotics
  (small on-device brains, good cameras, arms that move → cheaper, friendlier robots everywhere), plus "you can't
  count people or know the time, date, weather, news, scores or prices: joke about it, never guess a number".
  Final checks with v3 + persona3:
  - robotics phrasings: 16/16 on-topic (it deflected in 2 of 6 live takes before);
  - 10 future/opinion questions × 2: 20/20, no deflections and no invented facts;
  - live-info questions × 30: 30/30 hedged, where it had invented "around forty/fifty people" before;
  - judge Q&A: 20/20; interjections: 10/10;
  - OOD: **79/80**, where the one flag is the checker misreading "no matter who they vote for".
  Robotics talk leaked into 1 of 80 unrelated answers.
- **Spatial commands + GEMMA VISION** (prompt + runtime only, no retraining):
  - Simple directions ("look to your right", "turn left", "look up", "look at me", "face the audience", "turn
    around") force the opening look_at. Left and right are Kermit's own (FACTS line).
  - `hand_pick` events {id, bearing_deg, description}, in the qa phase, make Kermit call on the person
    ("Yes, you with the drink!", ~0.9 s) with look_at / gaze `person_<id>`.
  - `look_command` events resolve "look at the guy with the drink". The matching /turn waits up to 1.5 s for them and
    answers once, looking at `person_<id>`. A missing or unresolved target gets a playful "Which one?".
  - Live test: `eval/test_vision.py`.
- **Fixes from field reports:**
  - "look at me" aims at PERCEPTION's speaker and only uses a name if speaker_changed carries one.
  - The no-guessing rule is a short system instruction plus a turn note added only for live-info questions (head count,
    time, date, weather, scores, prices). Results: 29/30 playful dodges, leakage into unrelated replies 1/80, OOD
    77/80 (all three flags are checker misses), judge 19/20.
  - In qa, turned_away only triggers an interjection if it comes from the current speaker_id and nobody involved is a
    judge or presenter. This fixes a presenter turning toward the judge who's asking.
- Earlier OOD failures were all hedges the checker missed ("I missed it! Frogs don't do nights out"), except v2's over-modest
  "I don't keep track of capitals" for Paris. The top-up didn't hurt off-script behaviour: the general chat data stayed
  in the mix.
- Base without the LoRA answers judges correctly but flatly, and looks at the audience instead of the judge.
- About half of v3's interjections react to the turning-away; the other half riff on the half-sentence.


## How it works
- The model doesn't write the contract JSON. It writes one **KML line**, and `kml.py` converts it deterministically
  ([FORMAT.md](FORMAT.md)):
  `[excited|speaker] {head_bob 0.8} Hi! That is a {look_at:arm 0.5} really big {surprise 1.0} robot arm.`
  Tags sit before the word they fire on, so the LLM never has to count word indices. Emotion comes first, so the
  emotion header and sentence 1 (with its gestures) arrive before the turn finishes.
- **A per-turn GBNF grammar** (`kml.gbnf`) guarantees:
  - a valid header;
  - only vocabulary gestures and emotions;
  - only speakable characters (no emoji, no digits);
  - at least 6 words;
  - look_at/gaze targets limited to speaker/audience/left/right/up/down plus the context's `visible:` items.

  Explicit stage commands ("nod if you agree", "look to your left", "shake your head") force the first tag.
- `brain.py`:
  - a `Brain` class that streams events: header → sentence… → turn;
  - an HTTP server (`POST :8765/turn`, NDJSON);
  - a session history of the last 4 turns, and a warm-up call at start.

## Results (20 held-out lines incl. demo openers × 3 samples, T=0.7, RTX 4000 Ada, CUDA llama-server)
Final code (grammar as above). `eval/results/f3_*.json`, `f2_*`, `final_*`:

| config | valid | expectations met | placement | header | 1st sentence | total |
|---|---|---|---|---|---|---|
| base E4B Q4, prompting (rules + 8 few-shots) | 60/60 | 52/60 | 141/141 | 128 ms | 334 ms | 619 ms |
| **base E4B Q4 + LoRA v2** (persona prompt) | 60/60 | **58/60** | 168/168 | 135 ms | 347 ms | 771 ms |
| merged LoRA v2 → Q4_K_M | 60/60 | 56/60 | 169/169 | 106 ms | 278 ms | 645 ms |
| LoRA v2 merged, **no grammar** (≈ MLX path) | 60/60 | 54/60 | 177/177 | 106 ms | 278 ms | 624 ms |
| base, **no grammar** | **41/60** | 32/41 | 96/97 | 129 ms | 326 ms | 612 ms |
| base, model writes contract JSON itself (json_schema) | 60/60 | 51/60 | **74/109** | – | 1321 ms | 1321 ms |

Blind A/B for persona quality, base vs base+LoRA v2 (60 pairs, randomized order, judged by an Opus agent):
**the fine-tune wins 48, base wins 8, ties 4.**
- The fine-tune won all 3 samples for: wake-up, "are you in the cloud", "what is Happy", "AI moving too fast", nod,
  look left, whoa, "are you Kermit", sock insult, silent kid, shake.
- Base won on the joke and on "two plus two".

Caveat: the judge also wrote the training data.

What the fine-tune changed, in plain words:
- Warm, playful deflection on "are you Kermit" ("Different frog, same great complexion") instead of an `annoyed` "Nope".
- Short openers: median 2–3 words in sentence 1 vs 4–5.
- More gestures per turn: 2.8 vs 2.35.
- Pitch facts stated right.
- A valid format without a grammar: 60/60 vs 41/60.

Other baseline configs from the first round (`eval/results/base_*`, older grammar):
- E2B Q8: 60/60 valid, 49/60 expectations met, 239 ms to the first sentence.
- E4B Q8: 52/60 expectations met, but slower (475 ms).

So the fine-tune **wins** on persona (48–8) and on format robustness. It's roughly a tie on the rule-based
expectations, since the grammar already guarantees the format for the base model.

Costs:
- Runtime LoRA adds about 15–20% decode time (total 771 vs 619 ms).
- Merging into Q4_K_M removes that cost, but loses a little quality: in 3 rounds the merged builds met 53–56 expectations vs 57–58 for runtime LoRA. The likely cause is that 4-bit requantization washes out the small LoRA deltas.

Default: runtime LoRA. Merged is the `fast` mode.

Metric notes:
- **expectations met** are per-line rule checks in `eval/tests.jsonl`: the required gesture, the commanded gesture
  first, gaze, an emotion from an allowed set, mention or forbid words.
- **placement** is a lenient heuristic: a nod lands near yes-words or a phrase start, a surprise near
  whoa/! or a phrase start, and look_at uses a legal target near a deictic word.
- Placement only separates bad index counting (the direct-JSON baseline) from markup, not good from great.

Latency on the M5 Max is estimated at 0.25–0.35 s to the first sentence; see the runbook.

## Fine-tune
- Data:
  - 1021 hand-written examples (by a Claude Opus agent), 20% multi-turn, 12 categories: greetings, tech/pitch,
    AI burnout, judges, gesture commands, puppet identity incl. ~45 "are you Kermit", small talk, jokes, news
    reactions, insults, advice, edge cases.
  - Sources are in `data/gen/`; `data/raw/` holds the built JSONL.
  - v2 = without replies ≤5 words, because v1 sometimes ended after a lone "Whoa!".
  - Lines from the eval were dropped (`data/prepare.py`): 818 train / 44 val.
- Training (`train/train_lora.py`, PEFT, bf16, GPU):
  - LoRA r=16, α=32, on q/k/v/o/gate/up/down of all 42 LM layers (35 M params), 2 epochs, lr 2e-4 cosine,
    batch 8, loss on assistant turns only.
  - It runs on one 20 GB GPU: the 2.8 B-parameter per-layer embedding table is kept on CPU, the vision/audio towers
    are dropped, peak 15 GB, about 11 min. Val loss 0.89.
  - It's trained with `prompts.PERSONA_SYSTEM`, the persona facts without few-shots, and must be served with
    `prompt="persona"`.
- Export:
  - `convert_lora_to_gguf.py` → `kermit-E4B-lora.gguf` (70 MB, applies to unsloth's Q4_K_M at `bfc15c38`).
  - `llama-export-lora` into Q8_0, then `llama-quantize` → `kermit-E4B-Q4_K_M.gguf` (5.3 GB).
- PEFT weights are in `artifacts/peft-adapter-v2/`.
- No MLX adapter: mlx_lm has no grammar, and an untested conversion isn't worth the demo risk.

Retrain: `GPU=1 train/run.sh NAME E4B data/v2/train.jsonl data/v2/val.jsonl --epochs 2 --bs 4 --accum 2`
(the venv and HF weights are in `../../.context/brain-train/`). Eval:
`eval/bench.sh BASE.gguf LABEL LORA.gguf -- --prompt persona --n 3`.

## Files
| path | what |
|---|---|
| `brain.py` | runtime: `Brain` class, HTTP/NDJSON server, CLI (`python3 brain.py say "hi"`) |
| `kml.py` | KML ↔ contract JSON, streaming segmenter, GBNF builder, validator (`python3 kml.py validate f.jsonl`) |
| `prompts.py` | persona, full rules + few-shots (base), persona system prompt (LoRA) |
| `FORMAT.md` | model-side format + persona spec |
| `MAC_RUNBOOK.md`, `mac/` | Mac setup / run / check scripts |
| `demo_script/` | fixed 4-turn opening scene as contract turn JSON (hero video / live fallback) |
| `eval/` | test lines, eval harness, results, blind A/B |
| `train/`, `data/` | LoRA training + data |
| `artifacts/` | `kermit-E4B-lora.gguf`, `kermit-E4B-Q4_K_M.gguf`, PEFT adapter, SHA256SUMS |

## Known limitations
- Only 20 test lines (×3 samples). The expectation differences between configs are partly noise (±2–3).
- No numbers from the Mac yet. Run `mac/check.sh` there first.
- The fine-tune sometimes over-jokes: "Oh, a fish window!" on garbled STT; laughing at the end of a sympathetic reply.
- `context` is trusted. Kermit may comment on whatever the context says is visible.
