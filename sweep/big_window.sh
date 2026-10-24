#!/bin/bash
# Two-GPU window: 26B-A4B GPU-resident latency run, then swap the 26B file for the 31B and run it split over both GPUs.
cd "$(dirname "$0")"
M=/var/lib/happy-agent/happy/Bots/bot_2/.context/gemma-bench/models
log() { echo "[$(date +%T)] $*"; }
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader

if [ -f $M/26B/gemma-4-26B_q4_0-it.gguf ] && [ ! -f results/26B_QAT_full_lat.json ]; then
  log "26B latency on GPU1"
  GPU=1 PORT=18091 ./serve.sh 26B_QAT_gpu $M/26B/gemma-4-26B_q4_0-it.gguf --swa-full && \
    python3 run_sweep.py --llm http://127.0.0.1:18091 --label 26B_QAT_full_lat --n 2 > logs/eval_26B_QAT_full_lat.log 2>&1
  tail -1 logs/eval_26B_QAT_full_lat.log
  # split mode with the big model as text writer and the same model annotating (NP=2) is skipped: E4B split covers it
  kill $(cat logs/server_26B_QAT_gpu.pid); sleep 3
fi

log "swap 26B -> 31B"
rm -f $M/26B/gemma-4-26B_q4_0-it.gguf
mkdir -p $M/31B
[ -f $M/31B/gemma-4-31B-it-Q4_K_M.gguf ] || curl -sL -o $M/31B/gemma-4-31B-it-Q4_K_M.gguf \
  https://huggingface.co/unsloth/gemma-4-31B-it-GGUF/resolve/main/gemma-4-31B-it-Q4_K_M.gguf
ls -la $M/31B; df -h / | tail -1

log "31B split GPU1+GPU0"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
GPU=1,0 PORT=18092 ./serve.sh 31B_Q4_split2 $M/31B/gemma-4-31B-it-Q4_K_M.gguf --swa-full -ts 3,2 && \
  python3 run_sweep.py --llm http://127.0.0.1:18092 --label 31B_Q4_full --n 3 > logs/eval_31B_Q4_full.log 2>&1
tail -1 logs/eval_31B_Q4_full.log
kill $(cat logs/server_31B_Q4_split2.pid); sleep 3
log "window done"
