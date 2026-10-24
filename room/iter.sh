#!/bin/bash
# hill-climb iteration: re-run the given (level input) pairs on the current code.  ./iter.sh TAG normal:B quiet:B ...
cd "$(dirname "$0")"
TAG=$1; shift
for p in "$@"; do lvl=${p%%:*}; inp=${p##*:}
  python3 live.py --name ${TAG}_${lvl}_${inp} --noise $lvl --input $inp --camera > logs/${TAG}_${lvl}_${inp}.log 2>&1
done
echo ITER $TAG DONE >> logs/iter.log
