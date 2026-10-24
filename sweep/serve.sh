#!/bin/bash
# Usage: GPU=1 PORT=18090 [NP=1] [CTX=8192] sweep/serve.sh LABEL MODEL.gguf [extra llama-server args...]
# Starts a CUDA llama-server in the background, waits for /health, prints its PID. Log: logs/server_LABEL.log
# Two GPUs: GPU=0,1 serve.sh L M.gguf -ts 1,1
B=/var/lib/happy-agent/happy/Bots/bot_2/.context/gemma-bench
cd "$(dirname "$0")"
LABEL=$1 MODEL=$2; shift 2
PORT=${PORT:-18090}
TMPDIR=$B/tmp CUDA_VISIBLE_DEVICES=${GPU:-1} nohup $B/llama.cpp/build/bin/llama-server -m $MODEL -ngl 99 -fa on \
  -c ${CTX:-8192} -np ${NP:-1} --reasoning off --reasoning-budget 0 --port $PORT --host 127.0.0.1 "$@" \
  > logs/server_$LABEL.log 2>&1 &
echo $! > logs/server_$LABEL.pid
for i in $(seq 300); do curl -sf http://127.0.0.1:$PORT/health >/dev/null && { echo "up pid=$(cat logs/server_$LABEL.pid) port=$PORT"; exit 0; }; sleep 1; done
echo "server $LABEL failed"; tail -20 logs/server_$LABEL.log; exit 1
