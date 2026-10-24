"""Smoke test + latency check on the target machine. Tries every TTS backend, times STT, and says
which backend to use.   python mac_smoke.py [--backends mlx,torch-mps,torch-cpu]"""
import argparse, json, os, sys, time, traceback
import numpy as np, soundfile as sf

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

ap = argparse.ArgumentParser()
ap.add_argument("--backends", default="mlx,torch-mps,torch-cpu")
ap.add_argument("--voice", default=os.environ.get("KERMIT_VOICE", "am_fenrir"))
a = ap.parse_args()

turns = json.load(open(os.path.join(HERE, "lines", "demo_turns.json")))
firsts = [t["text"].split("!")[0].split("?")[0].split(".")[0] + "!" for t in turns]  # first sentences
out = os.path.join(HERE, "out", "smoke"); os.makedirs(out, exist_ok=True)
summary = {}

from tts import KokoroTTS
for b in a.backends.split(","):
    backend, device = ("mlx", None) if b == "mlx" else ("torch", b.split("-")[1])
    try:
        t = time.perf_counter(); tts = KokoroTTS(a.voice, backend=backend, device=device); load = time.perf_counter() - t
        first, full, ok = [], [], True
        for s in firsts:
            t = time.perf_counter(); tts.synth(s, "excited"); first.append(time.perf_counter() - t)
        for turn in turns:
            t = time.perf_counter(); audio, words = tts.synth(turn["text"], turn["emotion"]); full.append((time.perf_counter() - t) / (len(audio) / 24000))
            ok &= len(words) >= len(turn["text"].split()) - 1
            sf.write(os.path.join(out, f"{b}_{turn['id']}.wav"), audio, 24000)
        summary[b] = {"load_s": round(load, 2), "first_sentence_ms_median": round(1000 * float(np.median(first)), 1),
                      "first_sentence_ms_max": round(1000 * max(first), 1), "rtf_median": round(float(np.median(full)), 4), "word_timings_ok": bool(ok)}
    except Exception as e:
        traceback.print_exc(limit=2); summary[b] = {"error": repr(e)[:200]}
    print(b, summary[b], flush=True)

import stt
for prov in ["cpu"]:
    rec = stt.recognizer(prov); a16 = np.zeros(16000, np.float32); stt.transcribe(rec, a16)
    from scipy.signal import resample_poly
    x, sr = sf.read(os.path.join(HERE, "samples", "open_1.wav"), dtype="float32")
    x16 = resample_poly(x, 16000, sr).astype(np.float32)
    ts = []
    for _ in range(5):
        t = time.perf_counter(); r = stt.transcribe(rec, x16); ts.append(time.perf_counter() - t)
    summary["stt_" + prov] = {"ms_median_for_%.1fs" % (len(x16) / 16000): round(1000 * float(np.median(ts))), "text": r.text}
    print("stt", prov, summary["stt_" + prov])

good = {k: v for k, v in summary.items() if "first_sentence_ms_median" in v and v["word_timings_ok"]}
if good:
    best = min(good, key=lambda k: good[k]["first_sentence_ms_median"])
    be = "mlx" if best == "mlx" else "torch"
    print(f"\nUSE: --tts-backend {be}" + ("" if be == "mlx" else f" --device {best.split('-')[1]}"), f"(first sentence ≈ {good[best]['first_sentence_ms_median']} ms)")
json.dump(summary, open(os.path.join(out, "summary.json"), "w"), indent=1)
print("listen:", out)
