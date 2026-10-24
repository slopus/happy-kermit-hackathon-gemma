"""mlx-audio Kokoro: the exact Mac code path (here on mlx[cuda]). Checks audio, word timings, TTFA."""
import json, os, sys, time
import numpy as np, soundfile as sf
import mlx.core as mx
from mlx_audio.tts.utils import load_model
from mlx_audio.tts.models.kokoro import KokoroPipeline

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "bench", "kokoro", "mlx")
os.makedirs(OUT, exist_ok=True)
voice, speed = (sys.argv[1] if len(sys.argv) > 1 else "am_puck"), 1.15
t = time.perf_counter()
model = load_model("mlx-community/Kokoro-82M-bf16")
pipe = KokoroPipeline(lang_code="a", model=model, repo_id="mlx-community/Kokoro-82M-bf16")
print(f"load {time.perf_counter() - t:.2f}s device={mx.default_device()}")
list(pipe("Warm up, warm up.", voice=voice, speed=speed))
rows = []
for turn in json.load(open(os.path.join(ROOT, "lines", "demo_turns.json"))):
    t0 = time.perf_counter(); ttfa = None; chunks, words, off = [], [], 0.0
    for r in pipe(turn["text"], voice=voice, speed=speed, split_pattern=r"\n+"):
        a = np.array(r.audio, dtype=np.float32).reshape(-1)
        if ttfa is None: ttfa = time.perf_counter() - t0
        for tok in r.tokens or []:
            if tok.start_ts is None or tok.end_ts is None: continue
            if not any(c.isalnum() for c in tok.text):
                if words: words[-1]["w"] += tok.text
                continue
            words.append({"w": tok.text, "t0": round(off + tok.start_ts, 3), "t1": round(off + tok.end_ts, 3)})
        chunks.append(a); off += len(a) / 24000
    total = time.perf_counter() - t0; wav = np.concatenate(chunks)
    sf.write(os.path.join(OUT, f"{turn['id']}.wav"), wav, 24000)
    json.dump(words, open(os.path.join(OUT, f"{turn['id']}.words.json"), "w"))
    rows.append({"id": turn["id"], "ttfa_s": round(ttfa, 4), "total_s": round(total, 4), "audio_s": round(len(wav) / 24000, 3),
                 "n_words": len(words), "n_text_words": len(turn["text"].split())})
    print(rows[-1], flush=True)
json.dump(rows, open(os.path.join(OUT, "results.json"), "w"), indent=1)
