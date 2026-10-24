"""Synthetic demo-like turns for Kermit, in eot-bench's schema: data/kermit/kermit.parquet.

Each line is synthesised in one piece by Kokoro (so the prosody runs on through the pause, as in a real hesitation),
then cut at the word boundaries marked "|" and a 0.35–1.6 s silence is inserted. Natural pauses (commas) stay as
they are. The last span (2 s of trailing silence) is the true end of turn; every other silence ≥ 0.1 s is a hold.
Room noise at about -55 dBFS so the silences are not digital zero.

run in the kokoro venv:  ../../voice/.venvs/kokoro/bin/python make_kermit_set.py
"""
import io, os, random, sys
import pickle
import numpy as np, soundfile as sf
from scipy.signal import resample_poly

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "../../voice"))
OUT = os.path.join(HERE, "../data/kermit"); os.makedirs(OUT, exist_ok=True)

KERMIT_PREV = [
    "Hi-ho! Kermit the Frog here. What would you like to talk about?",
    "Well, that is a really big robot arm. Yaaay!",
    "Oh, I'm just a frog on a hand, but I love a good question.",
    "Hmm, it's not easy being green, you know.",
]
LINES = [  # "|" = the speaker pauses here and then carries on
    "So, Kermit... | how are you feeling about, um... | AI these days?",
    "Kermit, | what do you think about the robot arm?",
    "Okay so | I wanted to ask you | what your favorite song is.",
    "Can you tell me, | uh, | how you got so green?",
    "I was thinking that maybe | we could talk about | the hackathon.",
    "What's your take on | the whole Gemma thing?",
    "Hey Kermit, | do you know | where Miss Piggy is?",
    "So the thing is, | um, | we built you in about | twelve hours.",
    "Do you ever get tired of | being on somebody's hand?",
    "Tell me a joke | about frogs.",
    "And then, | uh, | what happens if the arm | moves too fast?",
    "I think | you're the coolest robot here.",
    "Kermit is our robot puppet, he runs entirely on device. | Anyway, Kermit, what do you think?",
    "This is the demo. | We have a camera, a microphone, and a robot arm. | Kermit, say hi to the judges!",
    "Well, | I guess my question is, | are you real?",
    "If you could have any superpower, | what would it be?",
    "Wait, | wait, | let me think about that.",
    "We trained the model last night, | and honestly, | it was a lot of work. | So, how do you feel?",
    "My name is | Sarah, | and I'm from | San Francisco.",
    "Um, | can you sing something?",
    "Do you like | flies?",
    "So what we did was | we took a Gemma model, | and we fine tuned it | to talk like you.",
    "How old are you, | Kermit?",
    "I'd like to know | what you had for breakfast.",
    "Is it hard to talk | with a robot hand | inside you?",
    "Kermit, look over there. | Who is that?",
    "The judges are | from Google, | so be nice.",
    "Yes.",
    "No, | not really.",
    "That's amazing.",
]
VOICES = ["af_heart", "am_michael", "bf_emma", "am_adam", "af_bella", "bm_george"]


def main():
    import torch
    from tts import KokoroTTS
    rng = random.Random(7)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rows = []
    for vi, voice in enumerate(VOICES):
        tts = KokoroTTS(voice=voice, device=dev, character=False)
        lang_b = voice.startswith("b")
        for li, line in enumerate(LINES):
            if (li + vi) % 2 and len(rows) > 0 and li % 3 == 0: continue  # thin out a little
            parts = [p.strip() for p in line.split("|")]
            text = " ".join(parts)
            a, words = tts.synth(text, "neutral")
            if not len(a): continue
            # word index where each part ends
            cuts, n = [], 0
            for p in parts[:-1]:
                n += len(p.split()); cuts.append(n)
            if len(words) != len(text.split()):
                print("word mismatch", voice, text, len(words), len(text.split())); continue
            pieces, last = [], 0
            for k in cuts:
                t = 0.5 * (words[k - 1]["t1"] + words[k]["t0"])
                i = int(t * 24000); pieces.append(a[last:i]); last = i
            pieces.append(a[last:])
            pauses = [rng.uniform(0.35, 1.6) for _ in cuts]
            sil = lambda d: np.zeros(int(d * 24000), np.float32)
            y = [sil(0.5)]
            word_t = []
            off = 0.5; acc = 0.0
            seg_start = [0.0] + [0.5 * (words[k - 1]["t1"] + words[k]["t0"]) for k in cuts]
            for j, p in enumerate(pieces):
                y.append(p)
                if j < len(pauses): y.append(sil(pauses[j]))
            y.append(sil(2.0))
            y = np.concatenate(y)
            # shift word timings by the inserted silence before them
            shift = lambda t: 0.5 + t + sum(pauses[j] for j in range(len(cuts)) if t > seg_start[j + 1])
            ws = [{"start": round(shift(w["t0"]), 3), "end": round(shift(w["t1"]), 3), "word": w["w"]} for w in words]
            x = resample_poly(y, 2, 3).astype(np.float32)
            x = 0.9 * x / (np.abs(x).max() + 1e-9) * rng.uniform(0.3, 0.9)
            # silence spans from the clean signal (20 ms frames under -40 dB of the loud frames)
            fr = x[: len(x) // 320 * 320].reshape(-1, 320)
            db = 20 * np.log10(np.sqrt((fr ** 2).mean(1)) + 1e-9)
            quiet = db < np.percentile(db[db > -100], 95) - 40
            spans, i = [], 0
            while i < len(quiet):
                if quiet[i]:
                    j = i
                    while j < len(quiet) and quiet[j]: j += 1
                    s, e = i * 0.02, j * 0.02
                    if i > 0 and e - s >= 0.1: spans.append({"start": round(s, 3), "end": round(e, 3)})
                    i = j
                else:
                    i += 1
            spans[-1]["end"] = round(len(x) / 16000, 3)
            nrng = np.random.default_rng(li * 10 + vi)
            noise = np.cumsum(nrng.standard_normal(len(x))).astype(np.float32); noise -= np.convolve(noise, np.ones(400) / 400, "same")
            x = x + noise / (np.abs(noise).max() + 1e-9) * 10 ** (-55 / 20) * 3
            buf = io.BytesIO(); sf.write(buf, x, 16000, format="WAV", subtype="FLOAT")
            rid = f"kermit__{voice}__{li:02d}"
            rows.append(dict(id=rid, audio={"bytes": buf.getvalue(), "path": rid + ".wav"}, language="en",
                             duration=len(x) / 16000, silence_spans=spans, words=ws,
                             messages=[{"role": "assistant", "content": KERMIT_PREV[(li + vi) % len(KERMIT_PREV)]}]))
            print(rid, len(spans), "spans", text[:50], flush=True)
    pickle.dump(rows, open(os.path.join(OUT, "kermit.pkl"), "wb"))  # -> parquet in the bench venv (no pandas here)
    print(len(rows), "turns,", sum(len(r["silence_spans"]) - 1 for r in rows), "holds")


if __name__ == "__main__":
    main()
