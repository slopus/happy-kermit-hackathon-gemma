#!/bin/bash
# full v3 script × noise level × input config, one after another (wall-clock, ~4 min each)
cd "$(dirname "$0")"
TAG=${1:-s1}
for lvl in quiet normal loud; do for inp in A B; do
  python3 live.py --name ${TAG}_${lvl}_${inp} --noise $lvl --input $inp --camera > logs/${TAG}_${lvl}_${inp}.log 2>&1
done; done
echo SWEEP DONE
