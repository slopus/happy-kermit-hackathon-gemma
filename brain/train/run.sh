#!/bin/bash
# Usage: train/run.sh NAME MODEL(E4B|E2B) TRAIN.jsonl VAL.jsonl [train_lora.py args...]
# Trains on GPU ${GPU:-0}, exports LoRA GGUF to $T/runs/NAME/lora.gguf.
set -e
cd "$(dirname "$0")/.."
T=/var/lib/happy-agent/happy/Bots/bot_2/.context/brain-train
L=/var/lib/happy-agent/happy/Bots/bot_2/.context/gemma-bench/llama.cpp
NAME=$1 M=$2 TR=$3 VA=$4; shift 4
OUT=$T/runs/$NAME
CUDA_VISIBLE_DEVICES=${GPU:-0} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TMPDIR=$T $T/venv/bin/python train/train_lora.py --base $T/hf/gemma-4-$M-it \
  --data $TR --val $VA --out $OUT "$@" 2>&1 | grep -v -i -E "warn|Loading weights"
PYTHONPATH=$L/gguf-py $T/venv/bin/python $L/convert_lora_to_gguf.py $OUT --base $T/hf/gemma-4-$M-it \
  --outfile $OUT/lora.gguf --outtype f16 2>&1 | tail -1
rm -f $OUT/tokenizer.json   # 32 MB copy, not needed
