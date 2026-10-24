# Happy Kermit (Gemma Hardware Hackathon, AGI House, Oct 24 2026)

A Kermit-style hand-rod puppet on a (simulated) xArm6 + hand. Everything runs on-device:
Gemma 4 E4B + Kermit LoRA writes the reply and inline gesture tags in one pass (KML → CONTRACT JSON),
Parakeet hears, Kokoro + frog DSP speaks, the mouth follows the audio envelope, perception tracks people
and the active speaker, Gemma vision picks raised hands, MuJoCo renders the puppet.

Interfaces: [CONTRACT.md](CONTRACT.md). Plan, results and the decision log: [docs/PLAN.md](docs/PLAN.md), [docs/TODO.md](docs/TODO.md).

## Layout
| folder | what | docs |
|---|---|---|
| `brain/` | llama-server + `brain.py` (KML, show phases, `/turn`, `/event`), LoRA training data + artifacts | [README](brain/README.md), [Mac runbook](brain/MAC_RUNBOOK.md) |
| `voice/` | Parakeet STT, Kokoro TTS + frogify, lip sync, timelines | [README](voice/README.md) |
| `turn/` | end-of-turn fusion (turn-1-mini + Smart Turn + LiveKit text), streaming Nemotron partials | [README](turn/README.md), [Mac runbook](turn/MAC_RUNBOOK.md) |
| `perception/` | people.py: YOLO11-pose + ByteTrack + re-ID, YuNet, Light-ASD, mic detector | [README](perception/README.md) |
| `vision/` | Gemma vision set-of-marks: hand picks, look commands (second llama-server :8081) | [README](vision/README.md) |
| `camera/` | Insta360 Link 2 PTZ (verdict: PTZ off, fixed wide) | [README](camera/README.md) |
| `motion/` | MuJoCo puppet rendering (`hero.mp4`, `stage/show.mp4`) | [README](motion/README.md) |
| `room/` | closed-loop room sim (pyroomacoustics, noise, judges); `run/*/score.json` kept | [README](room/README.md) |
| `sweep/` | Gemma model/speed sweep (dev box, CUDA) | [REPORT](sweep/REPORT.md) |
| `run_kermit.py` | the live full-duplex hub: mic + camera + BRAIN + TTS + speaker | docstring |

## Quickstart on a new Mac (M-series, needs internet once)
```bash
brew install uv cmake gh
cd brain && bash mac/setup.sh && cd ..           # llama.cpp (Metal) + Gemma 4 E4B base + MTP head + Kermit LoRA
cd voice && ./setup_mac.sh && .venv/bin/python mac_smoke.py && cd ..   # Parakeet, VAD, Kokoro
cd turn && ./setup_models.sh && cd ..             # end-of-turn models
cd perception && bash setup_mac.sh && cd ..       # YOLO / YuNet / Light-ASD / YOLO-World (+ CLIP)

bash brain/mac/run.sh &                           # BRAIN on :8765 (base + LoRA v3)
bash brain/mac/check.sh                           # 2-min check
voice/.venv/bin/python run_kermit.py --camera 0 --tts-backend mlx --play
#   + Gemma vision: bash vision/mac/run_vision.sh &  then add --vision
```
Fallbacks on the day: `--endpoint vad` → `--ptt`; `--no-aec`; `--bargein off`; `KERMIT_BARGEIN_MERGE_S=0`;
`bash brain/mac/run.sh fast` (merged model); drop `MTP=1`; worst case play `motion/videos/stage/show.mp4`.

## Not in git (fetched or recreated)
- venvs and caches: recreated by the setup scripts above.
- Base models (Gemma GGUF, mmproj, MTP, Parakeet, Nemotron, Kokoro, LiveKit, CLIP): downloaded by the setup scripts at pinned revisions.
- Over GitHub's 100 MB limit, attached to the release **`models-v1`**:
  - `kermit-E4B-Q4_K_M.gguf.part-aa/ab/ac`: merged Kermit Q4 (5.3 GB). `FAST=1 bash brain/mac/setup.sh` downloads and joins it,
    or `gh release download models-v1 -p 'kermit-E4B-Q4_K_M*' && cat kermit-E4B-Q4_K_M.gguf.part-* > brain/models/kermit-E4B-Q4_K_M.gguf`
  - `peft-adapter-v2.safetensors`, `peft-adapter-v3.safetensors`: PEFT LoRA weights (for re-merging/retraining); put back as
    `brain/artifacts/peft-adapter-vN/adapter_model.safetensors`.
- `room/run/*`: only the small result files (`score.json`, `meta.json`, ...); WAVs, videos and ndjson logs stayed on the dev box.
- Dev-box-only scripts (`sweep/*.sh`, `brain/eval/*.sh`, `brain/train/run.sh`, MuJoCo usage lines) still point at
  `/var/lib/happy-agent/...` paths (llama.cpp CUDA build, training venv). Set `KERMIT_MUJOCO_PY` for `room/live.py`.

Safety: sim only. Nothing here sends commands to a real arm or hand.
