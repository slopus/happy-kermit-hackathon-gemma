#!/bin/bash
# people.py (specialists) on every clip → pp/<name>.ndjson + events
cd "$(dirname "$0")/../perception"
for f in ../vision/clips/*.mp4; do n=$(basename $f .mp4); [ -s ../vision/pp/$n.ndjson ] && continue
  CUDA_VISIBLE_DEVICES=${GPU:-1} .venv/bin/python people.py --source $f --proc-fps 10 --log ../vision/pp/$n.ndjson --events ../vision/pp/$n.events.ndjson --render ../vision/pp/$n.mp4 2> ../vision/pp/$n.stderr; echo "$n done"; done
