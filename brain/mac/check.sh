#!/bin/bash
# With mac/run.sh running in another terminal: raw speed, one streamed turn, and the 20-line eval.
#   bash mac/check.sh [ft|v2|fast|base]   (same mode as mac/run.sh; start run.sh with PHASE=qa for this check)
set -euo pipefail
cd "$(dirname "$0")/.."
MODE=${1:-ft}
PROMPT=$([ "$MODE" = base ] && echo full3 || echo persona3)
MODEL=$([ "$MODE" = fast ] && echo models/kermit-E4B-Q4_K_M.gguf || echo models/gemma-4-E4B-it-Q4_K_M.gguf)
echo "== llama-bench ($MODEL): expect tg128 >= 90 t/s on M5 Max (RTX 4000 Ada: 89 t/s)"
models/llama-bench -m "$MODEL" -fa 1 -p 64 -n 128 -r 2
echo "== one streamed turn (timestamps per NDJSON line)"
python3 - <<'EOF'
import json, time, urllib.request
t0 = time.perf_counter()
req = urllib.request.Request("http://127.0.0.1:8765/turn", json.dumps({
    "heard": "whoa kermit is that a robot arm", "context": "one person in front of you. visible: robot arm, laptop",
    "stream": True, "session_id": "check"}).encode(), {"Content-Type": "application/json"})
for line in urllib.request.urlopen(req):
    print(f"{(time.perf_counter() - t0) * 1000:6.0f} ms  {line.decode().strip()}")
EOF
echo "== eval (20 lines x 1). Targets: valid 20/20, ms_header < 150, ms_first_sentence < 400"
python3 eval/run_eval.py --llm http://127.0.0.1:8080 --label mac_$MODE --prompt $PROMPT --n 1 | tail -1
echo "== judge Q&A + interjections (targets: judge 20/20, ms_first_sentence < 450)"
python3 eval/run_judge.py --llm http://127.0.0.1:8080 --label mac_$MODE --prompt $PROMPT --n 1 | tail -1
echo "== off-script robustness, 40 prompts x 2 phases (target >= 76/80)"
python3 eval/run_ood.py --llm http://127.0.0.1:8080 --label mac_$MODE --prompt $PROMPT --n 1 | tail -1
