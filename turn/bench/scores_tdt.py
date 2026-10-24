"""Text at the pause: Parakeet TDT v2 (offline, what VOICE already runs) on the audio up to each silence start.
Live, endpoint.py runs it once per pause at 0.2 s of silence, so the text features see the last word instead of
a streaming partial that lags by 0.3 s or more. Grid points with silence < 0.2 s keep the Nemotron partial.

usage: SHARD=i/n scores_tdt.py DATASET    then    scores_tdt.py DATASET models
"""
import os, sys, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../voice"))
from common import OUT, SR, load_rows, grid, run_sharded, load_out

_r = {}


def tdt_row(r):
    import stt
    if "r" not in _r: _r["r"] = stt.recognizer("cpu", threads=2)
    out = []
    for k, (s, e) in enumerate(r["spans"]):
        if e - s < 0.1: continue
        a = r["audio"][:int((s + 0.05) * SR)]
        t = time.perf_counter(); txt = stt.transcribe(_r["r"], a).text.strip() if len(a) > 1600 else ""
        out.append(dict(id=r["id"], span_index=k, text_tdt=txt, tdt_ms=(time.perf_counter() - t) * 1000, upto_s=s))
    return out


if __name__ == "__main__":
    name, step = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else "tdt")
    if step == "tdt":
        rows = load_rows(name)
        res, sfx = run_sharded(tdt_row, rows, "tdt")
        pd.DataFrame([x for r in res for x in r]).to_parquet(f"{OUT}/{name}_tdtspan{sfx}.parquet")
    else:
        from scores_text import LiveKit
        T = load_out(f"{name}_tdtspan"); A = load_out(f"{name}_asr")
        rows = {r["id"]: r for r in load_rows(name)}
        M = A.merge(T[["id", "span_index", "text_tdt"]], on=["id", "span_index"], how="left")
        G = grid(list(rows.values()))[["id", "span_index", "timestamp", "silence_dur"]]
        M = M.merge(G, on=["id", "span_index", "timestamp"], how="left")
        M["text_live"] = np.where(M.silence_dur >= 0.2 - 1e-6, M.text_tdt.fillna(M.text_nemo), M.text_nemo)
        lk = LiveKit(); msgs = {i: r["messages"] for i, r in rows.items()}
        M["lk_live"] = [lk(msgs[i], tx) for i, tx in zip(M.id, M.text_live)]
        M[["id", "span_index", "timestamp", "text_live", "lk_live"]].to_parquet(f"{OUT}/{name}_live_text.parquet")
        print(T.tdt_ms.describe())
