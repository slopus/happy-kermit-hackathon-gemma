#!/bin/bash
# One-time: fetch the turn-taking models into kermit/turn/models (~560 MB, needs network once).
#   cd kermit/turn && ./setup_models.sh          (Mac or Linux; then run_kermit.py uses --endpoint eou)
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p models/turn1mini/onnx models/smart-turn models/livekit-ml/onnx
HF=https://huggingface.co
get() { [ -s "$2" ] || { echo "fetch $2"; curl -fL -o "$2" "$1"; }; }
# p99lab turn-1-mini (Apache-2.0), streaming audio end-of-turn
get $HF/p99lab/turn-1-mini/resolve/main/onnx/turn-1-mini.step.onnx models/turn1mini/onnx/turn-1-mini.step.onnx
# Pipecat Smart Turn v3.2 (BSD-2), CPU int8. mel80.npy (Whisper filters) ships with this folder.
get $HF/pipecat-ai/smart-turn-v3/resolve/main/smart-turn-v3.2-cpu.onnx models/smart-turn/smart-turn-v3.2-cpu.onnx
# LiveKit turn-detector v0.4.1-intl text model (LiveKit Model License)
get $HF/livekit/turn-detector/resolve/v0.4.1-intl/onnx/model_q8.onnx models/livekit-ml/onnx/model_q8.onnx
get $HF/livekit/turn-detector/resolve/v0.4.1-intl/tokenizer.json models/livekit-ml/tokenizer.json
# Nemotron-3.5 streaming ASR 0.6b, 160 ms chunks (sherpa-onnx int8)
N=sherpa-onnx-nemotron-3.5-asr-streaming-0.6b-160ms-int8-2026-06-11
if [ ! -s models/nemotron-160ms/encoder.int8.onnx ]; then
  rm -f models/nemotron-160ms
  curl -fL https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/$N.tar.bz2 | tar xj -C models
  mv models/$N models/nemotron-160ms
fi
[ -s models/smart-turn/mel80.npy ] || echo "WARNING: models/smart-turn/mel80.npy missing: copy it from the dev box"
[ -s models/fusion.json ] || echo "WARNING: models/fusion.json missing: copy it from the dev box (bench/fit_live.py)"
echo "turn models ready: $(du -sh models | cut -f1)"
