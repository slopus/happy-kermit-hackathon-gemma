"""Pre-render the human lines of a scenario with Kokoro (plain voices, no frog DSP) → assets/lines/.

  ../voice/.venvs/kokoro/bin/python humans.py [--script ../brain/demo_script/v3/script.json] [--device cpu]

The humans are scripted actors, so their audio may be rendered ahead of time; WHEN each line plays is
decided live by the room (server.py), never pre-mixed. Output per line: <key>.wav (16 kHz) + <key>.json
(text, words with t0/t1 in seconds from the line start, voice).
"""
import argparse, hashlib, json, os, sys
import numpy as np, soundfile as sf
from scipy.signal import resample_poly

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "voice"))
os.environ.setdefault("HF_HOME", os.path.join(HERE, "..", "voice", ".hf"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")

VOICES = {"kirill": "am_michael", "steve": "am_adam", "judge1": "af_sarah", "judge2": "bm_george", "judge3": "am_eric"}


def line_key(who, text):
    return f"{who}_{hashlib.sha1(text.encode()).hexdigest()[:10]}"


def lines_of(scenario):
    """[(who, text)] for every human line in a scenario (v3 script.json or a room scenario json)."""
    s = json.load(open(scenario))
    steps = s["steps"]
    out = []
    for st in steps:
        out.append((st["speaker"], st["line"]))
        for extra in st.get("extra_lines", []):
            out.append((extra["speaker"], extra["line"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--script", nargs="+", default=[os.path.join(HERE, "..", "brain", "demo_script", "v3", "script.json")])
    ap.add_argument("--out", default=os.path.join(HERE, "assets", "lines"))
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    from tts import KokoroTTS, SR
    engines = {}
    for path in a.script:
        for who, text in lines_of(path):
            key = line_key(who, text)
            if os.path.exists(os.path.join(a.out, key + ".json")):
                continue
            v = VOICES[who]
            if v not in engines:
                engines[v] = KokoroTTS(v, backend="torch", device=a.device, character=False)
            audio, words = engines[v].synth(text, "neutral")
            x16 = resample_poly(audio, 2, 3).astype(np.float32)
            x16 /= np.sqrt(np.mean(x16[np.abs(x16) > 0.01] ** 2)) + 1e-9   # speech-active RMS = 1.0 (level set by the room)
            x16 *= 0.1
            sf.write(os.path.join(a.out, key + ".wav"), x16, 16000)
            json.dump({"who": who, "voice": v, "text": text, "duration": round(len(x16) / 16000, 3), "words": words},
                      open(os.path.join(a.out, key + ".json"), "w"))
            print(key, f"{len(x16) / 16000:.2f}s", text[:60])


if __name__ == "__main__":
    main()
