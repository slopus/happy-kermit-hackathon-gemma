#!/bin/bash
cd "$(dirname "$0")"
python3 live.py --name it3_normal_B --noise normal --input B --camera > logs/it3_normal_B.log 2>&1
KERMIT_TURN_TDT=0 python3 live.py --name it3_normal_B_tdt0 --noise normal --input B --camera > logs/it3_normal_B_tdt0.log 2>&1
python3 live.py --name it3_quiet_B --noise quiet --input B --camera > logs/it3_quiet_B.log 2>&1
python3 live.py --name it3_loud_B --noise loud --input B --camera > logs/it3_loud_B.log 2>&1
python3 live.py --name it3_normal_B_pin --noise normal --input B --camera --taskset 0-7 > logs/it3_normal_B_pin.log 2>&1
echo ITER it3 DONE >> logs/iter.log
