"""Replay turns through the live endpoint.Endpointer block by block (simulated clock, ASR decoded inline), against
ground truth: did it commit inside a mid-turn pause (cut-in), how long after the true end did it commit, and
what does each 32 ms block cost.

usage: replay.py kermit [N]  |  replay.py wav FILE.wav
"""
import io, json, os, sys, time
import numpy as np, pandas as pd, soundfile as sf

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import endpoint as E


def run(ep, x, log):
    """Feed audio x (16 kHz) to ep with a simulated clock starting at 0. Returns per-block cost (ms)."""
    cost = []
    x = np.concatenate([x, np.zeros(int(2.5 * E.SR), np.float32)])
    for i in range(0, len(x) - E.BLOCK + 1, E.BLOCK):
        t = (i + E.BLOCK) / E.SR
        c = time.perf_counter(); ep.feed(t, x[i:i + E.BLOCK]); cost.append((time.perf_counter() - c) * 1000)
    return cost


def fresh(ep):
    ep.vad.reset(); ep.t1m.reset(); ep.utt = []; ep.t_start = None; ep.pause = None; ep.in_speech = False
    ep.speech_run = ep.quiet_run = 0; ep.speech_s = 0.0; ep.last_commit = None; ep.stalled = False; ep.prefix_text = ""
    if ep.asr: ep.asr.cut()


if __name__ == "__main__":
    events = []
    ep = E.Endpointer(on_turn_end=lambda a, txt, te, ts: None, aec=False, sync=True, log=events.append)
    if sys.argv[1] == "wav":
        x, sr = sf.read(sys.argv[2], dtype="float32")
        if x.ndim > 1: x = x.mean(1)
        if sr != E.SR:
            from scipy.signal import resample_poly
            x = resample_poly(x, E.SR, sr).astype(np.float32)
        cost = run(ep, x, events)
        for e in events: print(json.dumps(e))
        print(f"block cost ms: p50 {np.median(cost):.2f} p99 {np.percentile(cost, 99):.1f} max {max(cost):.1f}")
        sys.exit()
    df = pd.read_parquet(os.path.join(HERE, "../data/kermit/kermit.parquet"))
    n = int(sys.argv[2]) if len(sys.argv) > 2 else len(df)
    rows, costs = [], []
    for r in df.sample(n=min(n, len(df)), random_state=1).itertuples():
        x, _ = sf.read(io.BytesIO(r.audio["bytes"]), dtype="float32")
        fresh(ep); events.clear(); ep.set_context(r.messages[-1]["content"])
        costs += run(ep, x, events)
        end = r.words[-1]["end"]
        holds = [(s["start"], s["end"]) for s in r.silence_spans[:-1] if s["end"] - s["start"] >= 0.2]
        commits = [e for e in events if e["event"] == "turn_end"]
        cands = [e for e in events if e["event"] == "turn_candidate"]
        early = [c for c in commits if c["t"] < end + 0.05]
        final = [c for c in commits if c["t"] >= end + 0.05]
        rows.append(dict(id=r.id, holds=len(holds), cutins=len(early), committed=bool(final),
                         delay=(final[0]["t"] - end) if final else np.nan, by=final[0]["by"] if final else None,
                         cands=len(cands), cancels=sum(e["event"] == "turn_cancel" for e in events),
                         text=final[0]["text"] if final else (early[-1]["text"] if early else "")))
    R = pd.DataFrame(rows); R.to_csv(os.path.join(HERE, "out/replay_kermit.csv"), index=False)
    print(R[["id", "holds", "cutins", "committed", "delay", "by", "cands", "text"]].to_string(max_colwidth=50))
    print(f"\nturns {len(R)}  holds {R.holds.sum()}  cut-ins {R.cutins.sum()} ({R.cutins.sum() / max(1, R.holds.sum()) * 100:.1f}% of pauses, "
          f"{(R.cutins > 0).mean() * 100:.1f}% of turns)  delay after true end: mean {R.delay.mean() * 1000:.0f} ms, "
          f"median {R.delay.median() * 1000:.0f} ms (silence measured from the last word's end)  timeouts {(R.by == 'timeout').mean() * 100:.0f}%")
    print(f"block cost ms: p50 {np.median(costs):.2f} p90 {np.percentile(costs, 90):.1f} p99 {np.percentile(costs, 99):.1f} max {max(costs):.1f}")
