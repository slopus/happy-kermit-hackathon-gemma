#!/bin/bash
# Start the on-device brain: llama-server (Metal) on :8080 + brain.py (turn JSON / NDJSON) on :8765.
#   bash mac/run.sh          base Q4_K_M + Kermit LoRA v3 (judge Q&A + interjections)          <- default
#   bash mac/run.sh v2       base Q4_K_M + LoRA v2 (previous adapter, fallback)
#   bash mac/run.sh fast     merged v2 Q4_K_M (~50 ms faster to sentence 1)
#   bash mac/run.sh base     prompting-only fallback (base model, rules + few-shots + facts)
# Sessions start in the scripted presentation phase (CONTRACT §6); it switches to Q&A after the hand-off.
# Set PHASE=qa to start in open conversation instead.
# MTP=1 adds the official Gemma 4 MTP drafter (speculative decoding): ~25% faster turns, same quality on CUDA;
# untested on Metal, so compare with mac/check.sh before the show.
set -euo pipefail
cd "$(dirname "$0")/.."
M=models
case "${1:-ft}" in
  base) ARGS=(-m $M/gemma-4-E4B-it-Q4_K_M.gguf); PROMPT=full3 ;;
  fast) ARGS=(-m $M/kermit-E4B-Q4_K_M.gguf); PROMPT=persona3 ;;
  v2)   ARGS=(-m $M/gemma-4-E4B-it-Q4_K_M.gguf --lora $M/kermit-E4B-lora-v2.gguf); PROMPT=persona3 ;;
  *)    ARGS=(-m $M/gemma-4-E4B-it-Q4_K_M.gguf --lora $M/kermit-E4B-lora.gguf); PROMPT=persona3 ;;
esac
[ "${MTP:-0}" = 1 ] && ARGS+=(--spec-type draft-mtp -md $M/mtp-gemma-4-E4B-it-Q8_0.gguf -ngld 99 --spec-draft-n-max 2)
$M/llama-server "${ARGS[@]}" -ngl 99 -fa on -c 8192 -np 1 --reasoning off --reasoning-budget 0 \
  --host 127.0.0.1 --port 8080 > llama-server.log 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null' EXIT
until curl -sf http://127.0.0.1:8080/health >/dev/null; do sleep 0.5; kill -0 $SRV || { tail llama-server.log; exit 1; }; done
echo "llama-server up (${ARGS[*]}), prompt=$PROMPT, phase=${PHASE:-presentation}"
python3 brain.py serve --llm http://127.0.0.1:8080 --prompt $PROMPT --phase ${PHASE:-presentation} --port 8765
