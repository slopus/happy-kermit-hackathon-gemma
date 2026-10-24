#!/bin/bash
# Usage: [SCRIPT=run_ood.py] eval/judge_bench.sh GGUF LABEL PROMPT [LORA_GGUF]   (CUDA llama-server on GPU ${GPU:-1}, port ${PORT:-18080})
set -e
B=/var/lib/happy-agent/happy/Bots/bot_2/.context/gemma-bench
cd "$(dirname "$0")/.."
GGUF=$1 LABEL=$2 PROMPT=$3; LORA=${4:+--lora $4}
PORT=${PORT:-18080}
mkdir -p eval/logs
TMPDIR=$B/tmp CUDA_VISIBLE_DEVICES=${GPU:-1} $B/llama.cpp/build/bin/llama-server -m $GGUF $LORA \
  -ngl 99 -fa on -c 8192 -np 1 --reasoning off --reasoning-budget 0 --port $PORT --host 127.0.0.1 > eval/logs/server_judge_$LABEL.log 2>&1 &
SRV=$!
trap "kill $SRV 2>/dev/null; wait $SRV 2>/dev/null" EXIT
for i in $(seq 180); do curl -sf http://127.0.0.1:$PORT/health >/dev/null && break; sleep 1; done
python3 eval/${SCRIPT:-run_judge.py} --llm http://127.0.0.1:$PORT --label $LABEL --prompt $PROMPT --n ${N:-2}
