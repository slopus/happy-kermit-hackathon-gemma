#!/bin/bash
# One-time setup on the Mac (Apple Silicon). Needs network once; everything after runs offline.
#   cd kermit/voice && ./setup_mac.sh
set -euo pipefail
cd "$(dirname "$0")"
command -v uv >/dev/null || { echo "install uv first: brew install uv"; exit 1; }

uv venv -p 3.12 .venv
VIRTUAL_ENV=.venv uv pip install -r requirements.txt

mkdir -p models
if [ ! -d models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8 ]; then
  curl -L https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8.tar.bz2 | tar xj -C models
fi
[ -f models/silero_vad.onnx ] || curl -L -o models/silero_vad.onnx https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx

# Pre-fetch TTS weights (Kokoro for PyTorch and for MLX) so the demo runs with HF_HUB_OFFLINE=1
.venv/bin/python - <<'EOF'
from huggingface_hub import snapshot_download
snapshot_download("hexgrad/Kokoro-82M")
snapshot_download("mlx-community/Kokoro-82M-bf16")
EOF
../turn/setup_models.sh  # turn-taking models (end of turn, AEC)
echo "setup done. next: .venv/bin/python mac_smoke.py"
# PERCEPTION: YuNet face detector (230 kB)
mkdir -p ../perception/models
[ -f ../perception/models/face_detection_yunet_2023mar.onnx ] || curl -L -o ../perception/models/face_detection_yunet_2023mar.onnx \
  https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
