"""Barge-in under speaker leakage: Kermit's TTS plays from the speaker, leaks into the mic through a room, and the
human sometimes starts talking over him. One continuous simulated session per condition (the echo canceller
keeps its adaptation across trials, as it would live).

Each trial: Kermit says a line (2–6 s of frog TTS from voice/samples), echo = delayed + reverberant + slightly
clipped copy at a given echo-to-speech ratio. In half the trials a human (Kokoro voices from data/kermit) starts at a
random moment during Kermit's line. Detector: speech prob ≥ 0.5 for ≥ N consecutive 32 ms blocks while Kermit plays.

Reports: false barge-ins per Kermit line (echo only), detection rate and latency from human onset.
usage: bargein.py
"""
import glob, io, json, os, sys
import numpy as np, pandas as pd, soundfile as sf
from scipy.signal import resample_poly, fftconvolve

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import endpoint as E

SR = 16000
rng = np.random.default_rng(3)


def kermit_lines():
    out = []
    for f in sorted(glob.glob(os.path.join(HERE, "../../voice/samples/**/*.wav"), recursive=True)):
        x, sr = sf.read(f, dtype="float32")
        if x.ndim > 1: x = x.mean(1)
        x = resample_poly(x, SR, sr).astype(np.float32)
        if 1.5 < len(x) / SR < 9: out.append(x / (np.abs(x).max() + 1e-9))
    return out


def human_lines():
    df = pd.read_parquet(os.path.join(HERE, "../data/kermit/kermit.parquet"))
    out = []
    for r in df.itertuples():
        x, _ = sf.read(io.BytesIO(r.audio["bytes"]), dtype="float32")
        s = int(r.words[0]["start"] * SR) - 800
        out.append(x[max(0, s):int(r.words[-1]["end"] * SR) + 1600])
    return out


def rir(rt60=0.3, tail=0.07, n=int(0.3 * SR)):
    """Direct path + exponential diffuse tail (tail 0.07, RT60 0.3 s: direct-to-reverberant ratio about -1 dB)."""
    t = np.arange(n) / SR
    h = rng.standard_normal(n) * np.exp(-6.9 * t / rt60) * tail
    h[0] = 1.0
    return (h / np.sqrt((h ** 2).sum())).astype(np.float32)


def rms(x): return float(np.sqrt(np.mean(x ** 2)) + 1e-12)


def session(cond, K, H, n_trials=60):
    """cond: dict(etr_db, delay_ms, detector, aec). Returns per-trial records."""
    aec = E.EchoCanceller(delay_ms=cond["delay_ms"]) if cond["aec"] else None
    vad = E.Silero()
    h = rir()
    recs = []
    D = int(cond["delay_ms"] * SR / 1000)
    for i in range(n_trials):
        k = K[rng.integers(len(K))] * 0.5
        talk = i >= 4 and (i % 2 == 1)                      # first 4 trials: echo-only warm-up
        gap = np.zeros(int(0.8 * SR), np.float32)
        ref = np.concatenate([k, gap])                       # what the speaker plays (reference)
        echo = fftconvolve(np.concatenate([np.zeros(D, np.float32), ref]), h)[:len(ref)]
        echo = np.tanh(1.5 * echo) / 1.5                      # a little speaker distortion
        mic = echo.copy()
        onset = None
        if talk:
            hu = H[rng.integers(len(H))]
            hu = fftconvolve(hu, rir(0.25))[:len(hu)]
            # human level so that echo-to-human ratio during speech = etr_db
            hu = hu * (rms(echo[:len(k)]) / rms(hu) / 10 ** (cond["etr_db"] / 20))
            onset = int(rng.uniform(0.3, max(0.4, len(k) / SR - 0.8)) * SR)
            end = min(len(mic), onset + len(hu)); mic[onset:end] += hu[:end - onset]
        mic += (rng.standard_normal(len(mic)) * 10 ** (-60 / 20)).astype(np.float32)
        run, fired = 0, None
        for j in range(0, len(mic) - E.BLOCK + 1, E.BLOCK):
            if aec:
                aec.far(ref[j:j + E.BLOCK], SR)
                x = aec.near(mic[j:j + E.BLOCK])
            else:
                x = mic[j:j + E.BLOCK]
            p = vad(x)
            playing = j < len(k)
            run = run + 1 if (p >= cond["thr"] and playing) else 0
            if run >= cond["n"] and fired is None:
                fired = (j + E.BLOCK) / SR
        if i < 4: continue
        recs.append(dict(talk=talk, fired=fired is not None,
                         lat=(fired - onset / SR) if (fired is not None and onset is not None) else np.nan,
                         early=fired is not None and onset is not None and fired < onset / SR))
    return recs


if __name__ == "__main__":
    K, H = kermit_lines(), human_lines()
    print(len(K), "kermit lines,", len(H), "human lines")
    rows = []
    for aec in (False, True):
        for etr in (6, 0, -6):
            for n, thr in ((4, 0.5), (6, 0.5), (6, 0.8)):
                c = dict(aec=aec, etr_db=etr, delay_ms=80, n=n, thr=thr)
                r = pd.DataFrame(session(c, K, H))
                e, t = r[~r.talk], r[r.talk]
                row = dict(aec=aec, echo_vs_human_db=etr, rule=f"p≥{thr} for {n * 32} ms",
                           false_bargein=round(float(e.fired.mean()), 3),
                           detect=round(float((t.fired & ~t.early).mean()), 3),
                           lat_ms_med=round(float(np.nanmedian(t.lat[~t.early]) * 1000)) if (t.fired & ~t.early).any() else None,
                           lat_ms_p90=round(float(np.nanpercentile(t.lat[~t.early], 90) * 1000)) if (t.fired & ~t.early).any() else None)
                rows.append(row); print(row, flush=True)
    df = pd.DataFrame(rows); df.to_csv(os.path.join(HERE, "out/bargein.csv"), index=False)
    print(df.to_string(index=False))
