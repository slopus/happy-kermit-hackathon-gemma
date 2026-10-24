"""M5 Max (40-core GPU, 614 GB/s) latency ESTIMATES, scaled from the clean RTX 4000 Ada runs (llama.cpp + GBNF).

  tps_M5     = measured server tok/s on the RTX * k,   k = M5 raw llama.cpp decode / RTX raw llama-bench decode (same quant)
               (carries the grammar/sampling overhead, runtime-LoRA cost and MTP gain over unchanged)
  1st sentence_M5 = prefill_M5 + N_fs / tps_M5         total_M5 = prefill_M5 + N_gen / tps_M5
  N_fs       = tokens up to the end of sentence 1 = (t_first_sentence - t_first_token) * tok/s, measured per config
  prefill_M5 = RTX first-token time (M5 Max pp512 3220 t/s vs this RTX 3428 t/s on Llama-2-7B Q4_0: parity); 31B x1.3
"""
import json, os, statistics

HERE = os.path.dirname(os.path.abspath(__file__))
# model -> (M5 raw llama.cpp decode tok/s [estimate, see REPORT.md], RTX raw decode tok/s [llama-bench, measured])
K = {"E2B": (150, 86.9), "E4B": (125, 71.5), "12B": (55, 35.0), "26B": (75, 85.0), "31B": (25, 17.0)}
ROWS = [  # label, name, model
    ("C_E2B_Q8", "E2B Q8, prompt", "E2B"),
    ("C_E4B_Q4", "E4B Q4, prompt", "E4B"),
    ("C_E4B_Q4_mtp", "E4B Q4 + MTP, prompt", "E4B"),
    ("C_E4B_lora", "**E4B Q4 + LoRA (current)**", "E4B"),
    ("C_E4B_lora_mtp2", "E4B Q4 + LoRA + MTP (n-max 2)", "E4B"),
    ("C_E4B_merged_mtp2", "E4B merged-LoRA Q4 + MTP", "E4B"),
    ("12B_Q4_full", "12B Q4_K_M, prompt", "12B"),
    ("C_26B_Q4_0", "26B-A4B Q4, prompt", "26B"),
    ("C_26B_Q4_0_mtp", "26B-A4B Q4 + MTP, prompt", "26B"),
    ("31B_Q4_full", "31B Q4_K_M, prompt", "31B"),
]

print("| config | RTX 1st sentence | RTX total | RTX tok/s | M5 tok/s (est.) | **M5 1st sentence (est.)** | M5 total (est.) |")
print("|---|---|---|---|---|---|---|")
for lab, name, model in ROWS:
    p = os.path.join(HERE, "results", f"{lab}.json")
    if not os.path.exists(p):
        continue
    rows = [r for r in json.load(open(p))["rows"] if r.get("t_first_sentence") and r.get("tps")]
    m = statistics.median
    ft, tps = m(r["t_first_token"] for r in rows), m(r["tps"] for r in rows)
    nfs = m((r["t_first_sentence"] - r["t_first_token"]) * r["tps"] for r in rows)
    ngen = m(r["n_gen"] for r in rows)
    m5raw, rtxraw = K[model]
    tps5 = tps * m5raw / rtxraw
    ft5 = ft * (1.3 if model == "31B" else 1.0)
    print(f"| {name} | {1000 * m(r['t_first_sentence'] for r in rows):.0f} ms | {1000 * m(r['t_total'] for r in rows):.0f} ms | "
          f"{tps:.0f} | {tps5:.0f} | **{1000 * (ft5 + nfs / tps5):.0f} ms** | {1000 * (ft5 + ngen / tps5):.0f} ms |")
