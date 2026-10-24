#!/bin/bash
# Clean latency pass: every latency-critical config back-to-back on one exclusive GPU (GPU1), n=2 (40 turns) each.
cd "$(dirname "$0")"
M=/var/lib/happy-agent/happy/Bots/bot_2/.context/gemma-bench/models
A=../brain/artifacts
export GPU=${GPU:-1}
E4B=$M/E4B/gemma-4-E4B-it-Q4_K_M.gguf
MTP4="--spec-type draft-mtp -md $M/mtp/mtp-gemma-4-E4B-it-Q8_0.gguf -ngld 99"
MTP26="--spec-type draft-mtp -md $M/mtp/mtp-gemma-4-26B-A4B-it-Q8_0.gguf -ngld 99"

run() {  # run LABEL MODEL "server args" "eval args"
  local L=$1 MODEL=$2 SARGS=$3 EARGS=$4
  echo "=== $L $(date +%T) load=$(cut -d' ' -f1 /proc/loadavg)"
  PORT=18095 NP=${NP:-1} ./serve.sh $L $MODEL $SARGS > /dev/null || { echo "server failed"; tail -3 logs/server_$L.log; return; }
  python3 run_sweep.py --llm http://127.0.0.1:18095 --label $L $EARGS > logs/eval_$L.log 2>&1
  tail -1 logs/eval_$L.log
  grep "draft acceptance" logs/server_$L.log | awk -F'[(/]' '{a+=$2; g+=$3} END {if (g) printf "  mtp acceptance %.2f\n", a/g}'
  kill $(cat logs/server_$L.pid); sleep 3
}

for job in "$@"; do
  case $job in
    e2b)        run C_E2B_Q8          $M/E2B/gemma-4-E2B-it-Q8_0.gguf "" "--prompt full --n 2" ;;
    e4b)        run C_E4B_Q4          $E4B "" "--prompt full --n 2" ;;
    e4b_mtp)    run C_E4B_Q4_mtp      $E4B "$MTP4" "--prompt full --n 2" ;;
    lora)       run C_E4B_lora        $E4B "--lora $A/kermit-E4B-lora.gguf" "--prompt persona --n 2" ;;
    lora_mtp)   run C_E4B_lora_mtp2   $E4B "--lora $A/kermit-E4B-lora.gguf $MTP4 --spec-draft-n-max 2" "--prompt persona --n 2" ;;
    merged_mtp) run C_E4B_merged_mtp2 $A/kermit-E4B-Q4_K_M.gguf "$MTP4 --spec-draft-n-max 2" "--prompt persona --n 2" ;;
    26b)        run C_26B_Q4_0        $M/26B/gemma-4-26B-A4B-it-Q4_0.gguf "--swa-full" "--prompt full --n 2" ;;
    26b_mtp)    run C_26B_Q4_0_mtp    $M/26B/gemma-4-26B-A4B-it-Q4_0.gguf "--swa-full $MTP26" "--prompt full --n 2" ;;
    split_same) NP=2 CTX=16384 run C_split_E4B_E4B $E4B "" "--mode split --prompt full --n 2" ;;
    split_e2b)
      PORT=18096 ./serve.sh C_E2B_annot $M/E2B/gemma-4-E2B-it-Q8_0.gguf > /dev/null || continue
      run C_split_E4B_E2B $E4B "" "--mode split --annot http://127.0.0.1:18096 --prompt full --n 2"
      kill $(cat logs/server_C_E2B_annot.pid) ;;
    *) echo "unknown job $job" ;;
  esac
done
echo "clean pass done $(date +%T)"
