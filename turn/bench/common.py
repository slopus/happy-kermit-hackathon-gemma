"""Shared bits for the end-of-turn bench: dataset rows, the eot-bench scoring grid, metrics.

Grid = eot-bench's: every silence span ≥ 0.1 s, a score point every 0.1 s from the span start; the last span of a
turn is the true end of turn ("eot"), earlier ones are mid-turn pauses ("hold"). Metrics reuse eot-bench's
own policy sweep (ref/metrics.py: threshold × action_delay × timeout), so numbers line up with its leaderboard.
"""
import io, math, os, sys
import numpy as np, pandas as pd, soundfile as sf

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "ref"))
OUT = os.path.join(HERE, "out")
os.makedirs(OUT, exist_ok=True)
SR = 16000


def datasets():
    return {"eotbench": os.path.join(ROOT, "data/eotbench/en.parquet"), "kermit": os.path.join(ROOT, "data/kermit/kermit.parquet")}


def load_rows(name="eotbench"):
    df = pd.read_parquet(datasets()[name])
    rows = []
    for r in df.itertuples():
        a = r.audio
        if isinstance(a, dict) and a.get("bytes") is not None:
            x, sr = sf.read(io.BytesIO(a["bytes"]), dtype="float32")
        else:
            x, sr = np.asarray(a["array"], np.float32), int(a["sampling_rate"])
        if x.ndim > 1: x = x.mean(1)
        assert sr == SR, sr
        rows.append(dict(id=r.id, audio=x, spans=[(float(s["start"]), float(s["end"])) for s in r.silence_spans],
                         words=[(float(w["start"]), float(w["end"]), str(w["word"])) for w in r.words],
                         messages=[dict(m) for m in r.messages] if r.messages is not None else []))
    return rows


def grid(rows, step=0.1):
    out = []
    for r in rows:
        n = len(r["spans"])
        for k, (s, e) in enumerate(r["spans"]):
            if e - s < 0.1 - 1e-6: continue
            m = int(math.floor((e - s) / step + 1e-6))
            ts = [round(s + i * step, 6) for i in range(m + 1)]
            if ts[-1] < e - 1e-6: ts.append(round(e, 6))
            for t in ts:
                out.append(dict(id=r["id"], span_index=k, timestamp=t, silence_dur=round(t - s, 6),
                                label="eot" if k == n - 1 else "hold"))
    return pd.DataFrame(out)


def at(times, values, t, default=0.0):
    """Value of a causal time series (times ascending, value known from time) at t."""
    i = np.searchsorted(times, t + 1e-6, side="right") - 1
    return values[i] if i >= 0 else default


def evaluate(pred, col, score_point=None):
    """eot-bench operating points for score column `col` (any-time read-out unless score_point)."""
    import metrics
    p = pred[["id", "span_index", "timestamp", "silence_dur", "label"]].copy()
    p["p_eot"] = pred[col].astype(float)
    p["language"] = "en"
    trade, summ = metrics.compute_metrics_from_predictions(p, score_point_s=score_point)
    model = trade[trade.policy_type == "model"]
    vad = trade[trade.policy_type != "model"]

    def best_cut(df, lat):
        f = df[df.mean_latency <= lat + 1e-9]
        return float(f.cutoff_rate.min()) if len(f) else math.nan

    def best_lat(df, cut):
        f = df[df.cutoff_rate <= cut + 1e-9]
        return float(f.mean_latency.min()) if len(f) else math.nan

    res = {"auc": summ["auc"], "cut@300": best_cut(model, 0.3), "cut@600": best_cut(model, 0.6),
           "lat@5%": best_lat(model, 0.05), "lat@10%": best_lat(model, 0.10), "lat@2%": best_lat(model, 0.02)}
    vadres = {"cut@300": best_cut(vad, 0.3), "cut@600": best_cut(vad, 0.6), "lat@5%": best_lat(vad, 0.05),
              "lat@10%": best_lat(vad, 0.10), "lat@2%": best_lat(vad, 0.02)}
    return res, vadres, trade


def run_sharded(fn, jobs, tag):
    """Process jobs[shard::nshard] (SHARD=i/n env), no multiprocessing (sandbox has no /dev/shm semaphores)."""
    import time
    i, n = map(int, os.environ.get("SHARD", "0/1").split("/"))
    mine = jobs[i::n]; res = []; t = time.time()
    for k, j in enumerate(mine):
        res.append(fn(j))
        if k % 10 == 0: print(f"[{tag} {i}/{n}] {k}/{len(mine)} {time.time() - t:.0f}s", flush=True)
    return res, (f".{i}" if n > 1 else "")


def load_out(stem):
    """Concatenate out/{stem}.parquet or its shards out/{stem}.N.parquet."""
    import glob
    fs = sorted(glob.glob(f"{OUT}/{stem}.parquet") + glob.glob(f"{OUT}/{stem}.[0-9]*.parquet"))
    return pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
