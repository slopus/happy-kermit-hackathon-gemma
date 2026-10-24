# End-of-turn bench: kermit  (150 turns, 250 hold spans, fusion 5-fold CV by turn)

eot-bench policy sweep (threshold × action delay 0.2–1.0 s × timeout 1–3.5 s). cut@X = share of mid-turn pauses Kermit would cut into at ≤X ms mean dead air; lat@Y = mean dead air after a true end at ≤Y cut-ins.

| method | AUC | cut@300ms | cut@600ms | lat@2% | lat@5% | lat@10% |
|---|---|---|---|---|---|---|
| silence only (ideal VAD) | – | 99.2% | 79.9% | 2000 ms | 2000 ms | 2000 ms |
| turn-1-mini (audio) | 0.913 | 36.8% | 22.6% | 1722 ms | 1502 ms | 1211 ms |
| Smart Turn v3.2 (audio) | 0.561 | 82.0% | 67.8% | 2000 ms | 1969 ms | 1969 ms |
| parakeet EOU P(<EOU>) | 0.523 | – | 78.7% | 2000 ms | 2000 ms | 2000 ms |
| parakeet EOU emitted | 0.536 | – | 82.0% | 2000 ms | 2000 ms | 2000 ms |
| LiveKit text, Nemotron partials | 0.856 | 69.5% | 25.5% | 1593 ms | 1137 ms | 984 ms |
| LiveKit text, oracle words (0.5 s lag) | 0.850 | 96.2% | 79.9% | 1607 ms | 1377 ms | 1201 ms |
| Gemma e2b judge, Nemotron partials | 0.842 | 55.2% | 31.4% | 2000 ms | 2000 ms | 2000 ms |
| fusion: audio models + prosody + text form | 0.988 | 21.3% | 8.8% | 1245 ms | 787 ms | 488 ms |
| fusion + LiveKit text | 0.984 | 18.4% | 3.8% | 1186 ms | 559 ms | 365 ms |
| fusion audio + Gemma | 0.984 | 25.5% | 2.9% | 844 ms | 538 ms | 404 ms |
| fusion all | 0.985 | 20.1% | 4.2% | 900 ms | 574 ms | 375 ms |

## Fixed silence thresholds

| rule | cut-ins (per pause) | turns with ≥1 cut-in | mean dead air |
|---|---|---|---|
| ideal silence ≥ 0.3 s | 99.2% | 94.0% | 300 ms |
| ideal silence ≥ 0.5 s | 85.4% | 88.0% | 500 ms |
| ideal silence ≥ 0.8 s | 63.6% | 68.7% | 800 ms |
| ideal silence ≥ 1.0 s | 48.1% | 58.7% | 1000 ms |
| Silero (p<0.5) silence ≥ 0.3 s | 94.1% | 92.7% | 355 ms |
| Silero (p<0.5) silence ≥ 0.5 s | 80.3% | 84.0% | 555 ms |
| Silero (p<0.5) silence ≥ 0.8 s | 58.6% | 66.0% | 855 ms |
