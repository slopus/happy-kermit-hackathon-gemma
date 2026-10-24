"""Gemma as the end-of-turn judge on partial transcripts: one forward pass, P(yes) from the first-token logprobs.

usage: scores_gemma.py DATASET URL TAG [SRC=nemo]    (URL of a llama-server with the Gemma model)
"""
import json, math, os, sys, time
import numpy as np, pandas as pd, requests
from concurrent.futures import ThreadPoolExecutor
from common import OUT, load_rows, load_out

SYSTEM = ("You watch a live voice conversation and decide whether the user has finished their turn and now expects "
          "a reply. The user's words come from streaming speech recognition: they may stop mid-sentence, mid-thought "
          "or after a filler like 'um' or 'so', which means the user is still talking. Answer 'yes' only if the user's "
          "words form a complete turn that the assistant should respond to now; otherwise answer 'no'.")


def prompt(prev_assistant, user_text):
    prev = (prev_assistant or "").strip()
    if len(prev) > 300: prev = "…" + prev[-300:]
    return (f"Assistant said: \"{prev}\"\n" if prev else "") + f"User so far: \"{user_text.strip()}\"\n\nHas the user finished their turn? Answer yes or no."


def p_yes(url, prev, text, sess=requests):
    if not (text or "").strip(): return 0.0, 0.0
    body = {"messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt(prev, text)}],
            "max_tokens": 1, "temperature": 0, "logprobs": True, "top_logprobs": 20}
    t = time.perf_counter()
    r = sess.post(f"{url}/v1/chat/completions", json=body, timeout=60).json()
    dt = time.perf_counter() - t
    lp = {}
    for c in r["choices"][0]["logprobs"]["content"][0]["top_logprobs"]:
        k = c["token"].strip().lower()
        if k in ("yes", "no"): lp[k] = max(lp.get(k, -1e9), c["logprob"])
    y, n = math.exp(lp.get("yes", -30)), math.exp(lp.get("no", -30))
    return y / (y + n), dt


if __name__ == "__main__":
    name, url, tag = sys.argv[1:4]; src = sys.argv[4] if len(sys.argv) > 4 else "nemo"
    rows = load_rows(name)
    prev = {r["id"]: next((m["content"] for m in reversed(r["messages"]) if m.get("role") == "assistant"), "") for r in rows}
    A = load_out(f"{name}_asr")
    keys = sorted({(i, t) for i, t in zip(A.id, A[f"text_{src}"])})
    print(len(keys), "unique prompts", flush=True)
    res = {}; lat = []
    s = requests.Session()
    def job(k):
        return k, p_yes(url, prev[k[0]], k[1], s)
    t0 = time.time()
    with ThreadPoolExecutor(8) as ex:
        for i, (k, (p, dt)) in enumerate(ex.map(job, keys)):
            res[k] = p; lat.append(dt)
            if i % 200 == 0: print(i, f"{time.time() - t0:.0f}s", flush=True)
    A[f"gemma_{tag}"] = [res[(i, t)] for i, t in zip(A.id, A[f"text_{src}"])]
    A[["id", "span_index", "timestamp", f"gemma_{tag}"]].to_parquet(f"{OUT}/{name}_gemma_{tag}.parquet")
    print("latency ms (8 parallel) p50/p90:", np.percentile(lat, 50) * 1000, np.percentile(lat, 90) * 1000)
