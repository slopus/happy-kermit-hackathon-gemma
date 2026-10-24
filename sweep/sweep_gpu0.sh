#!/bin/bash
# Size sweep, single GPU (GPU0; GPU1 is busy with BRAIN's LoRA training). Mixed (KML) mode, n=3, plus split runs on E4B.
set -u
cd "$(dirname "$0")"
M=/var/lib/happy-agent/happy/Bots/bot_2/.context/gemma-bench/models
A=../brain/artifacts
export GPU=${GPU:-0}

run() {  # run LABEL MODEL "server args" "eval args"
  local L=$1 MODEL=$2 SARGS=$3 EARGS=$4
  echo "=== $L $(date +%T)"
  PORT=18090 NP=${NP:-1} CTX=${CTX:-8192} ./serve.sh $L $MODEL $SARGS || return
  python3 run_sweep.py --llm http://127.0.0.1:18090 --label $L $EARGS > logs/eval_$L.log 2>&1
  tail -1 logs/eval_$L.log
  kill $(cat logs/server_$L.pid); sleep 3
}

for job in "$@"; do
  case $job in
    e2b_q8)     run E2B_Q8_full   $M/E2B/gemma-4-E2B-it-Q8_0.gguf "" "--prompt full --n 3" ;;
    e4b_q4)     run E4B_Q4_full   $M/E4B/gemma-4-E4B-it-Q4_K_M.gguf "" "--prompt full --n 3" ;;
    e4b_lora)   run E4B_Q4_lora   $M/E4B/gemma-4-E4B-it-Q4_K_M.gguf "--lora $A/kermit-E4B-lora.gguf" "--prompt persona --n 3" ;;
    12b_q4)     run 12B_Q4_full   $M/12B/gemma-4-12b-it-Q4_K_M.gguf "--swa-full" "--prompt full --n 3" ;;
    12b_q8)     run 12B_Q8_full   $M/12B/gemma-4-12b-it-Q8_0.gguf "--swa-full" "--prompt full --n 3" ;;
    26b_qat)    run 26B_QAT_full  $M/26B/gemma-4-26B_q4_0-it.gguf "--swa-full" "--prompt full --n 3" ;;
    split_e4b)  NP=2 CTX=16384 run E4B_Q4_split $M/E4B/gemma-4-E4B-it-Q4_K_M.gguf "" "--mode split --prompt full --n 3" ;;
    e4b_lat)    run E4B_Q4_full_lat $M/E4B/gemma-4-E4B-it-Q4_K_M.gguf "" "--prompt full --n 2" ;;
    lora_lat)   run E4B_Q4_lora_lat $M/E4B/gemma-4-E4B-it-Q4_K_M.gguf "--lora $A/kermit-E4B-lora.gguf" "--prompt persona --n 2" ;;
    e2b_lat)    run E2B_Q8_full_lat $M/E2B/gemma-4-E2B-it-Q8_0.gguf "" "--prompt full --n 2" ;;
    split_e2b)  # E4B writes the text, E2B Q8 annotates (second server on the same GPU)
      PORT=18093 ./serve.sh E2B_annot $M/E2B/gemma-4-E2B-it-Q8_0.gguf || continue
      run E4B_text_E2B_annot $M/E4B/gemma-4-E4B-it-Q4_K_M.gguf "" "--mode split --annot http://127.0.0.1:18093 --prompt full --n 3"
      kill $(cat logs/server_E2B_annot.pid) ;;
    *) echo "unknown job $job" ;;
  esac
done
