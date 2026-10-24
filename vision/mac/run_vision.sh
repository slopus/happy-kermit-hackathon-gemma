#!/bin/bash
# GEMMA VISION server on the M5 Max: a second llama-server (Metal) on :8081 with Gemma 4 E4B + mmproj and NO LoRA.
# It mmaps the same base GGUF as BRAIN (kermit/brain/models, from brain/mac/setup.sh), so the extra memory is mostly the
# 1 GB vision projector + KV. Kept separate from BRAIN's :8080 so the Kermit persona LoRA never touches vision prompts.
#   bash kermit/vision/mac/run_vision.sh            # then: python kermit/run_kermit.py --camera 0 --vision ...
set -euo pipefail
cd "$(dirname "$0")/.."
M=../brain/models
mkdir -p models
# same unsloth revision as BRAIN's base model; sha256 ddf46c21d7078e95338cfc22306b19b276a29a5ad089023449dd54d4b6170a51 (990 MB)
[ -f models/mmproj-F16.gguf ] || curl -L --fail -o models/mmproj-F16.gguf \
  "https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF/resolve/bfc15c382204943c3a8fff0c750b94ae2364d7a3/mmproj-F16.gguf"
$M/llama-server -m $M/gemma-4-E4B-it-Q4_K_M.gguf --mmproj models/mmproj-F16.gguf -ngl 99 -fa on -c 8192 -np 2 \
  --reasoning off --reasoning-budget 0 --host 127.0.0.1 --port ${PORT:-8081} > vision-server.log 2>&1 &
SRV=$!
until curl -sf http://127.0.0.1:${PORT:-8081}/health >/dev/null; do sleep 0.5; kill -0 $SRV || { tail vision-server.log; exit 1; }; done
echo "vision llama-server up on :${PORT:-8081} (pid $SRV)"
PY=../voice/.venv/bin/python; [ -x $PY ] || PY=python3
$PY -c "import PIL, numpy" 2>/dev/null || $PY -m pip install -q pillow numpy   # vision.py needs Pillow in the run_kermit venv
$PY smoke.py --url http://127.0.0.1:${PORT:-8081} || true
wait $SRV
