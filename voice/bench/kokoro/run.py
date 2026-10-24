"""Kokoro-82M bench: per-line wavs, word timings, TTFA/RTF on GPU and CPU.

  python bench/kokoro/run.py [--device cuda|cpu] [--voices am_puck,am_echo]
"""
import argparse, json, os, sys, time
import numpy as np, soundfile as sf, torch
from kokoro import KPipeline

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "bench", "kokoro")
SR = 24000

ap = argparse.ArgumentParser()
ap.add_argument("--device", default="cuda")
ap.add_argument("--voices", default="am_puck,am_echo,am_michael,af_heart,am_fenrir")
ap.add_argument("--speed", type=float, default=1.15)
ap.add_argument("--save", action="store_true")
a = ap.parse_args()

turns = json.load(open(os.path.join(ROOT, "lines", "demo_turns.json")))
t = time.perf_counter()
pipe = KPipeline(lang_code="a", device=a.device, repo_id="hexgrad/Kokoro-82M")
load_s = time.perf_counter() - t
results = {"device": a.device, "load_s": load_s, "speed": a.speed, "lines": []}

for voice in a.voices.split(","):
    list(pipe("Warm up, warm up.", voice=voice, speed=a.speed))  # warm-up (also loads voice)
    os.makedirs(os.path.join(OUT, voice), exist_ok=True)
    for turn in turns:
        t0 = time.perf_counter(); ttfa = None; chunks, words, off = [], [], 0.0
        for r in pipe(turn["text"], voice=voice, speed=a.speed, split_pattern=r"\n+"):
            audio = r.audio.cpu().numpy() if torch.is_tensor(r.audio) else np.asarray(r.audio)
            if ttfa is None: ttfa = time.perf_counter() - t0
            for tok in r.tokens or []:
                if tok.start_ts is None or tok.end_ts is None: continue
                if not any(c.isalnum() for c in tok.text):
                    if words: words[-1]["w"] += tok.text  # glue punctuation to the previous word
                    continue
                words.append({"w": tok.text + (tok.whitespace and "" or ""), "t0": round(off + tok.start_ts, 3), "t1": round(off + tok.end_ts, 3)})
            chunks.append(audio); off += len(audio) / SR
        total = time.perf_counter() - t0
        wav = np.concatenate(chunks)
        dur = len(wav) / SR
        if a.device == "cuda" or a.save:
            sf.write(os.path.join(OUT, voice, f"{turn['id']}.wav"), wav, SR)
            json.dump(words, open(os.path.join(OUT, voice, f"{turn['id']}.words.json"), "w"))
        results["lines"].append({"voice": voice, "id": turn["id"], "ttfa_s": round(ttfa, 4), "total_s": round(total, 4),
                                 "audio_s": round(dur, 3), "rtf": round(total / dur, 4), "n_words": len(words),
                                 "n_text_words": len(turn["text"].split())})
        print(voice, turn["id"], f"ttfa={ttfa*1000:.0f}ms total={total*1000:.0f}ms dur={dur:.2f}s words={len(words)}/{len(turn['text'].split())}", flush=True)

json.dump(results, open(os.path.join(OUT, f"results_{a.device}.json"), "w"), indent=1)
