#!/bin/bash
# One-time setup on the M5 Max. Needs internet ONCE (build + model download); the demo itself runs offline.
#   cd kermit/brain && bash mac/setup.sh
set -euo pipefail
cd "$(dirname "$0")/.."
BRAIN=$PWD
mkdir -p "$BRAIN/models" "$HOME/kermit-src"

# 1) llama.cpp with Metal, pinned to the commit we tested on CUDA (Gemma 4 + --reasoning flags + GBNF).
xcode-select -p >/dev/null || xcode-select --install
command -v cmake >/dev/null || brew install cmake
if [ ! -d "$HOME/kermit-src/llama.cpp" ]; then
  git clone https://github.com/ggml-org/llama.cpp "$HOME/kermit-src/llama.cpp"
fi
cd "$HOME/kermit-src/llama.cpp"
git fetch --quiet origin && git checkout --quiet 970b51ce7c9ba5fff0726c5e9994f1c2f0e4eedc
cmake -B build -DGGML_METAL=ON -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build -j --target llama-server llama-bench
ln -sf "$PWD/build/bin/llama-server" "$BRAIN/models/llama-server"
ln -sf "$PWD/build/bin/llama-bench" "$BRAIN/models/llama-bench"

# 2) Base model (exact revision the LoRA was trained/merged against). 5.0 GB.
cd "$BRAIN/models"
REV=bfc15c382204943c3a8fff0c750b94ae2364d7a3
[ -f gemma-4-E4B-it-Q4_K_M.gguf ] || curl -L --fail -o gemma-4-E4B-it-Q4_K_M.gguf \
  "https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF/resolve/$REV/gemma-4-E4B-it-Q4_K_M.gguf"

# 2b) Optional MTP drafter for speculative decoding (MTP=1 bash mac/run.sh). 100 MB.
MTP_REV=a69f018d7678a2b9f5e705f8903d6fb29b9b25dc
[ -f mtp-gemma-4-E4B-it-Q8_0.gguf ] || curl -L --fail -o mtp-gemma-4-E4B-it-Q8_0.gguf \
  "https://huggingface.co/ggml-org/gemma-4-E4B-it-GGUF/resolve/$MTP_REV/mtp-gemma-4-E4B-it-Q8_0.gguf"

# 3) Kermit fine-tune. The LoRA (70 MB, applied at load time with --lora) ships in the repo under artifacts/.
[ -f kermit-E4B-lora.gguf ] || cp "$BRAIN/artifacts/kermit-E4B-lora.gguf" .
# 3b) Optional merged Q4 (5.3 GB, ~15% faster decode, `run.sh fast`). Too big for git: it lives in the
#     GitHub release `models-v1` as 3 parts. FAST=1 bash mac/setup.sh  (needs `gh auth login` with repo access)
if [ "${FAST:-0}" = 1 ] && [ ! -f kermit-E4B-Q4_K_M.gguf ]; then
  gh release download models-v1 -R slopus/happy-kermit-hackathon-gemma -p 'kermit-E4B-Q4_K_M.gguf.part-*' -D .
  cat kermit-E4B-Q4_K_M.gguf.part-* > kermit-E4B-Q4_K_M.gguf && rm kermit-E4B-Q4_K_M.gguf.part-*
fi
ls -la "$BRAIN/models"
echo "setup done. next: bash mac/run.sh"
