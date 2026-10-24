"""Parakeet TDT 0.6b v2 offline (what VOICE runs after the utterance) on the same turns: WER + latency."""
import sys, time, numpy as np, pandas as pd
sys.path.insert(0, "../../voice")
from common import load_rows, OUT
from asr_wer import wer
import stt
rec = stt.recognizer("cpu", threads=4)
for ds in sys.argv[1:]:
    rows = load_rows(ds); out = []
    for r in rows:
        t = time.perf_counter(); h = stt.transcribe(rec, r["audio"]).text; dt = time.perf_counter() - t
        out.append(dict(id=r["id"], ref=" ".join(w for _, _, w in r["words"]), tdt=h, ms=dt * 1000, dur=len(r["audio"]) / 16000))
    D = pd.DataFrame(out); D.to_parquet(f"{OUT}/{ds}_tdt.parquet")
    nw = D.ref.str.split().str.len(); w = np.array([wer(a, b) for a, b in zip(D.ref, D.tdt)])
    print(ds, "TDT WER %.3f" % ((w * nw).sum() / nw.sum()), "ms/turn med", D.ms.median().round(), "rtf", (D.ms / 1000 / D.dur).median().round(3))
