| config | RTX 1st sentence | RTX total | RTX tok/s | M5 tok/s (est.) | **M5 1st sentence (est.)** | M5 total (est.) |
|---|---|---|---|---|---|---|
| E2B Q8, prompt | 239 ms | 438 ms | 99 | 171 | **157 ms** | 278 ms |
| E4B Q4, prompt | 348 ms | 645 ms | 79 | 137 | **219 ms** | 382 ms |
| E4B Q4 + MTP, prompt | 229 ms | 377 ms | 136 | 238 | **155 ms** | 243 ms |
| **E4B Q4 + LoRA (current)** | 361 ms | 770 ms | 67 | 117 | **227 ms** | 481 ms |
| E4B Q4 + LoRA + MTP (n-max 2) | 306 ms | 589 ms | 90 | 157 | **205 ms** | 378 ms |
| E4B merged-LoRA Q4 + MTP | 245 ms | 484 ms | 113 | 198 | **164 ms** | 305 ms |
| 12B Q4_K_M, prompt | 636 ms | 1478 ms | 33 | 52 | **432 ms** | 948 ms |
| 26B-A4B Q4, prompt | 300 ms | 626 ms | 76 | 67 | **336 ms** | 733 ms |
| 26B-A4B Q4 + MTP, prompt | 259 ms | 535 ms | 98 | 86 | **282 ms** | 574 ms |
| 31B Q4_K_M, prompt | 1293 ms | 3279 ms | 15 | 22 | **969 ms** | 2385 ms |
