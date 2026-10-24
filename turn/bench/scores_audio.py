"""Audio-side scores on the eot-bench grid, all causal, all CPU:
  t1m      turn-1-mini (p99lab, streaming, P(eot) every 20 ms)
  smart    Smart Turn v3.2 cpu (Pipecat; last 8 s of audio up to t, Whisper log-mel)
  peou     parakeet_realtime_eou_120m (soniqo ONNX, 160 ms cache-aware streaming): max P(<EOU>) over the
           encoder frames since the silence started, plus whether <EOU> was emitted; its own transcript (peou_text)
  vad      Silero VAD speech prob (latest 32 ms frame) and the trailing-silence length by Silero (vad_sil)
  prosody  cheap features of the last voiced 400 ms: f0 slope, energy slope, last-word duration proxy

usage: scores_audio.py DATASET [N_WORKERS]
"""
import json, os, sys, time
import numpy as np, pandas as pd
from common import ROOT, OUT, SR, load_rows, grid, at, run_sharded

M = os.path.join(ROOT, "models")
_m = {}


def models():
    if _m: return _m
    import onnxruntime as ort
    o = ort.SessionOptions(); o.intra_op_num_threads = 1; o.inter_op_num_threads = 1; o.log_severity_level = 3
    S = lambda p: ort.InferenceSession(p, o, providers=["CPUExecutionProvider"])
    _m["t1m"] = S(f"{M}/turn1mini/onnx/turn-1-mini.step.onnx")
    _m["smart"] = S(f"{M}/smart-turn/smart-turn-v3.2-cpu.onnx")
    _m["enc"], _m["dec"], _m["joint"] = [S(f"{M}/parakeet-eou/parakeet-eou-{n}.onnx") for n in ("encoder", "decoder", "joint")]
    _m["silero"] = S(f"{ROOT}/../voice/models/silero_vad.onnx")
    from transformers import WhisperFeatureExtractor
    _m["wfe"] = WhisperFeatureExtractor(chunk_length=8)
    import librosa
    _m["mel"] = librosa.filters.mel(sr=16000, n_fft=512, n_mels=128, norm="slaney", htk=False)
    _m["cfg"] = json.load(open(f"{M}/parakeet-eou/config.json"))
    _m["vocab"] = json.load(open(f"{M}/parakeet-eou/vocab.json"))
    return _m


def run_t1m(x):
    m = models(); s = m["t1m"]; STEP = 2560
    names = ["audio", "conv1", "conv2", "k", "v", "k_valid", "h6"]
    st = {i.name: np.zeros(i.shape, np.float32) for i in s.get_inputs() if i.name != "audio"}
    outs = ["probs"] + [f"new_state_{n}" for n in names]
    xx = np.pad(x, (0, (-len(x)) % STEP))
    T, P = [], []
    for i in range(0, len(xx), STEP):
        pr, *new = s.run(outs, {"audio": xx[i:i + STEP].reshape(1, STEP), **st})
        st = {f"state_{n}": v for n, v in zip(names, new)}
        for j in range(8):
            T.append((i + (j + 1) * 320) / SR); P.append(float(pr[j]))
    return np.array(T), np.array(P)


def run_smart(x, ts):
    m = models(); N = 8 * SR; feats = []
    for t in ts:
        e = int(np.floor(t * SR + 1e-6)); a = x[max(0, e - N):e]
        a = np.pad(a, (N - len(a), 0))
        f = m["wfe"](a, sampling_rate=SR, return_tensors="np", padding="max_length", max_length=N, truncation=True, do_normalize=True)
        feats.append(f.input_features[0])
    out = []
    for i in range(0, len(feats), 16):
        out.append(m["smart"].run(None, {"input_features": np.stack(feats[i:i + 16]).astype(np.float32)})[0].reshape(-1))
    return np.concatenate(out) if out else np.zeros(0)


def logmel_eou(x):
    import librosa
    m = models()
    a = np.pad(x, (0, SR // 2))
    e = a.copy(); e[1:] -= .97 * a[:-1]
    spec = librosa.stft(e, n_fft=512, hop_length=160, win_length=400, window=np.hanning(400), center=True, pad_mode="constant")
    return np.log(m["mel"] @ (np.abs(spec) ** 2) + 2 ** -24).astype(np.float32)


def run_peou(x):
    """Streaming parakeet EOU. Returns per 80 ms encoder frame: time available, P(EOU) max over the frame's
    decode steps, EOU emitted, text so far (since the last EOU reset NOT applied: we keep the whole text)."""
    m = models(); cfg = m["cfg"]; enc, dec, joint = m["enc"], m["dec"], m["joint"]
    F = logmel_eou(x)
    ins = enc.get_inputs(); names = [i.name for i in ins]
    cache = [np.zeros(i.shape, np.float32 if "float" in i.type else np.int64) for i in ins[2:]]
    h = np.zeros((cfg["decoderLayers"], 1, cfg["decoderHidden"]), np.float32); c = h.copy()
    blank, eou, eob = cfg["blankTokenId"], cfg["eouTokenId"], cfg["eobTokenId"]
    hid, h, c = dec.run(None, {"token": np.array([[blank]], np.int64), "h": h, "c": c})
    W = cfg["streaming"]["melFrames"]; S = cfg["streaming"]["outputFrames"] * cfg["subsamplingFactor"]
    T, P, E, TX = [], [], [], []
    toks = []
    for off in range(0, F.shape[1] - W + 1, S):
        out = enc.run(None, dict(zip(names, [F[None, :, off:off + W], np.array([W], np.int64), *cache])))
        cache = [v.astype(p.dtype) for v, p in zip(out[2:], cache)]
        t_avail = (off + W) * 0.01
        for fr in range(cfg["streaming"]["outputFrames"]):
            pmax, emitted = 0.0, False
            for _ in range(8):
                lg = joint.run(None, {"encoder_output": out[0][:, fr:fr + 1], "decoder_output": hid})[0].reshape(-1)
                pr = np.exp(lg - lg.max()); pr /= pr.sum()
                pmax = max(pmax, float(pr[eou]))
                tok = int(lg.argmax())
                if tok == blank: break
                if tok == eou:
                    emitted = True
                    # NeMo resets the decoder after EOU; mimic: blank-reset state
                    h = np.zeros_like(h); c = np.zeros_like(c)
                    hid, h, c = dec.run(None, {"token": np.array([[blank]], np.int64), "h": h, "c": c})
                    break
                hid, h, c = dec.run(None, {"token": np.array([[tok]], np.int64), "h": h, "c": c})
                if tok != eob: toks.append(tok)
            T.append(t_avail); P.append(pmax); E.append(emitted)
            TX.append(len(toks))
    text_at = lambda n: "".join(m["vocab"][str(t)] for t in toks[:n]).replace("\u2581", " ").strip()
    return np.array(T), np.array(P), np.array(E), TX, text_at


def run_silero(x):
    m = models(); s = m["silero"]
    h = np.zeros((2, 1, 64), np.float32); c = h.copy(); T, P = [], []
    for i in range(0, len(x) - 511, 512):
        p, h, c = s.run(None, {"x": x[None, i:i + 512], "h": h, "c": c})
        T.append((i + 512) / SR); P.append(float(p.reshape(-1)[0]))
    return np.array(T), np.array(P)


def prosody(x, t, vT, vP):
    """Features of the last voiced 0.5 s before t: f0 slope (st/s), energy slope (dB/s), voiced fraction."""
    import librosa
    sp = vT[(vT <= t) & (vP > 0.5)]
    if not len(sp): return 0.0, 0.0, 0.0
    e = int(sp[-1] * SR); s0 = max(0, e - SR // 2)
    seg = x[s0:e]
    if len(seg) < 2048: return 0.0, 0.0, 0.0
    f0, vf, _ = librosa.pyin(seg, fmin=70, fmax=400, sr=SR, frame_length=1024, hop_length=160)
    rms = librosa.feature.rms(y=seg, frame_length=640, hop_length=160)[0]
    db = 20 * np.log10(rms + 1e-6); tt = np.arange(len(db)) * 0.01
    es = float(np.polyfit(tt, db, 1)[0]) if len(db) > 3 else 0.0
    ok = ~np.isnan(f0)
    if ok.sum() > 4:
        st = 12 * np.log2(f0[ok] / np.nanmedian(f0[ok])); ft = (np.arange(len(f0)) * 0.01)[ok]
        fs = float(np.polyfit(ft[-20:], st[-20:], 1)[0]) if ok.sum() > 4 else 0.0
    else:
        fs = 0.0
    return fs, es, float(ok.mean())


def score_row(args):
    r, g = args
    x = r["audio"]; ts = g.timestamp.to_numpy()
    t0 = time.perf_counter()
    T1, P1 = run_t1m(x); c_t1m = time.perf_counter() - t0
    t0 = time.perf_counter(); ps = run_smart(x, ts); c_smart = (time.perf_counter() - t0) / max(1, len(ts))
    t0 = time.perf_counter(); TE, PE, EE, TX, text_at = run_peou(x); c_peou = time.perf_counter() - t0
    vT, vP = run_silero(x)
    out = g.copy()
    out["t1m"] = [at(T1, P1, t) for t in ts]
    out["smart"] = ps
    # peou: max P(EOU) / any emission between span start and t
    starts = (g.timestamp - g.silence_dur).to_numpy()
    pe, ee, txt = [], [], []
    for s, t in zip(starts, ts):
        sel = (TE > s - 0.08) & (TE <= t + 1e-6)
        pe.append(float(PE[sel].max()) if sel.any() else 0.0)
        ee.append(float(EE[sel].any()) if sel.any() else 0.0)
        i = np.searchsorted(TE, t + 1e-6, side="right") - 1
        txt.append(text_at(TX[i]) if i >= 0 else "")
    out["peou"], out["peou_emit"], out["peou_text"] = pe, ee, txt
    out["vad_p"] = [at(vT, vP, t) for t in ts]
    # silero-measured trailing silence at t (speech prob < 0.35)
    vs = []
    for t in ts:
        sp = vT[(vT <= t + 1e-6) & (vP >= 0.5)]
        vs.append(t - sp[-1] if len(sp) else t)
    out["vad_sil"] = vs
    pros = {}
    for s in np.unique(starts):
        pros[s] = prosody(x, s + 0.05, vT, vP)
    out["f0_slope"] = [pros[s][0] for s in starts]
    out["en_slope"] = [pros[s][1] for s in starts]
    out["voiced_frac"] = [pros[s][2] for s in starts]
    dur = len(x) / SR
    cost = dict(id=r["id"], dur=dur, t1m_rtf=c_t1m / dur, smart_ms=c_smart * 1000, peou_rtf=c_peou / dur)
    return out, cost


if __name__ == "__main__":
    name = sys.argv[1]
    rows = load_rows(name); G = grid(rows)
    jobs = [(r, G[G.id == r["id"]]) for r in rows]
    jobs = [j for j in jobs if len(j[1])]
    res, sfx = run_sharded(score_row, jobs, "audio")
    pd.concat([o for o, _ in res]).to_parquet(f"{OUT}/{name}_audio{sfx}.parquet")
    cost = pd.DataFrame([c for _, c in res]); cost.to_parquet(f"{OUT}/{name}_audiocost{sfx}.parquet")
    print(cost.describe().to_string())
