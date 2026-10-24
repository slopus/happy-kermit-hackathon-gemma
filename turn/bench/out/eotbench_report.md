# End-of-turn bench: eotbench  (400 turns, 850 hold spans, fusion 5-fold CV by turn)

eot-bench policy sweep (threshold × action delay 0.2–1.0 s × timeout 1–3.5 s). cut@X = share of mid-turn pauses Kermit would cut into at ≤X ms mean dead air; lat@Y = mean dead air after a true end at ≤Y cut-ins.

| method | AUC | cut@300ms | cut@600ms | lat@2% | lat@5% | lat@10% |
|---|---|---|---|---|---|---|
| silence only (ideal VAD) | – | 55.6% | 21.7% | 2500 ms | 2000 ms | 1000 ms |
| turn-1-mini (audio) | 0.970 | 22.7% | 9.6% | 1350 ms | 849 ms | 580 ms |
| Smart Turn v3.2 (audio) | 0.876 | 34.8% | 15.2% | 1601 ms | 1031 ms | 757 ms |
| parakeet EOU P(<EOU>) | 0.725 | – | 21.1% | 2500 ms | 2000 ms | 993 ms |
| parakeet EOU emitted | 0.709 | – | 21.6% | 2500 ms | 2000 ms | 993 ms |
| LiveKit text, Nemotron partials | 0.848 | 41.4% | 13.3% | 1558 ms | 1040 ms | 701 ms |
| LiveKit text, oracle words (0.5 s lag) | 0.910 | 74.3% | 12.9% | 1464 ms | 948 ms | 676 ms |
| Gemma e2b judge, Nemotron partials | 0.748 | 73.5% | 16.2% | 2500 ms | 1184 ms | 820 ms |
| fusion: audio models + prosody + text form | 0.967 | 22.7% | 9.6% | 1610 ms | 843 ms | 591 ms |
| fusion + LiveKit text | 0.970 | 19.1% | 7.5% | 1348 ms | 773 ms | 492 ms |
| fusion audio + Gemma | 0.968 | 21.0% | 8.2% | 1426 ms | 834 ms | 527 ms |
| fusion all | 0.973 | 17.4% | 7.4% | 1124 ms | 712 ms | 445 ms |

## Fixed silence thresholds

| rule | cut-ins (per pause) | turns with ≥1 cut-in | mean dead air |
|---|---|---|---|
| ideal silence ≥ 0.3 s | 55.6% | 38.2% | 300 ms |
| ideal silence ≥ 0.5 s | 28.8% | 23.0% | 500 ms |
| ideal silence ≥ 0.8 s | 13.2% | 12.2% | 800 ms |
| ideal silence ≥ 1.0 s | 9.6% | 9.0% | 1000 ms |
| Silero (p<0.5) silence ≥ 0.3 s | 72.6% | 43.2% | 241 ms |
| Silero (p<0.5) silence ≥ 0.5 s | 36.6% | 28.7% | 442 ms |
| Silero (p<0.5) silence ≥ 0.8 s | 15.5% | 14.5% | 746 ms |
