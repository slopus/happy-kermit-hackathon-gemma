"""Render BRAIN's presentation script (brain/demo_script/presentation_script.json, turns 1–8) as one scene:
Kermit's turns (wav + timeline each), the human's lines in a separate "heard" voice, and a mixed
scene.wav + scene.json with absolute start times, for MOTION's hero video.

Turn 5 is a barge-in: Kermit's "Oh, like..." starts while the human is still mid-sentence (his back
turned), and overlaps him. The script has no words for the human in turns 5–6, so PLACEHOLDER lines
below stand in for them; change HUMAN_OVERRIDE to Kirill's real lines and re-run.

  python make_scene.py [--out samples/scene] [--human-voice am_michael]
"""
import argparse, json, os, re, sys, types
import numpy as np, soundfile as sf

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from tts import KokoroTTS, SR
import kermit_voice
from kermit_voice import Performer, trim
kermit_voice.print = lambda *a, **k: None  # no per-sentence NDJSON here

SCRIPT = os.path.join(HERE, "..", "brain", "demo_script", "presentation_script.json")
HUMAN_OVERRIDE = {  # PLACEHOLDER words for the turned-away beats
    5: "So the whole point is that everything runs right here on the device, and the reason that matters is",
    6: "privacy, latency, and it keeps working even when the conference wifi doesn't.",
}
INTERJECT_AT = 0.72   # turn 5: Kermit cuts in at this fraction of the human's line
REPLY_GAP = 0.7       # human stops → Kermit starts (the measured live latency)
HUMAN_GAP = 0.6       # Kermit stops → human starts


def sentence_case(s):
    s = re.sub(r"\bai\b", "AI", re.sub(r"\bkermit\b", "Kermit", re.sub(r"\bhappy kermit\b", "Happy Kermit", s.strip())))
    return s[0].upper() + s[1:] + ("" if s[-1] in ".?!" else "?" if s.split()[0] in ("so", "is", "and", "how", "what") else ".")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "samples", "scene"))
    ap.add_argument("--human-voice", default="am_michael")
    ap.add_argument("--voice", default=os.environ.get("KERMIT_VOICE", "am_fenrir"))
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    perf = Performer(types.SimpleNamespace(voice=a.voice, tts_backend="torch", device=a.device, no_frog=False,
                                           out=os.path.join(HERE, ".context", "render"), play=False))
    human = KokoroTTS(a.human_voice, device=a.device, character=False)
    script = json.load(open(SCRIPT))

    clips, t = [], 0.5
    for beat in script:
        n = beat["turn"]
        text = HUMAN_OVERRIDE.get(n) or (None if beat["heard"].startswith("(") else sentence_case(beat["heard"]))
        h_start = h_end = None
        if text:
            ha, hw = human.synth(text, "neutral"); ha, hw = trim(ha, hw)
            sf.write(os.path.join(a.out, f"human_{n:02d}.wav"), ha, SR)
            h_start, h_end = t, t + len(ha) / SR
            clips.append({"who": "human", "turn": n, "audio": f"human_{n:02d}.wav", "t": round(h_start, 3),
                          "duration": round(len(ha) / SR, 3), "text": text, "placeholder": n in HUMAN_OVERRIDE,
                          "words": [{"w": w["w"], "t0": round(w["t0"] + h_start, 3), "t1": round(w["t1"] + h_start, 3)} for w in hw]})
        wav, tl = perf.one(turn=beat["turn_json"])
        dst = os.path.join(a.out, f"kermit_{n:02d}")
        os.replace(wav, dst + ".wav"); tl["audio"] = f"kermit_{n:02d}.wav"; tl.pop("latency", None)
        json.dump(tl, open(dst + ".timeline.json", "w"), separators=(",", ":"))
        if beat["trigger"] == "turned_away" and h_start is not None:
            k_start = h_start + INTERJECT_AT * (h_end - h_start)   # barge-in: over the human's speech
        else:
            k_start = (h_end + REPLY_GAP) if h_end is not None else t
        clips.append({"who": "kermit", "turn": n, "beat": beat["beat"], "trigger": beat["trigger"], "audio": tl["audio"],
                      "timeline": f"kermit_{n:02d}.timeline.json", "t": round(k_start, 3), "duration": tl["duration"],
                      "text": tl["text"], "emotion": tl["emotion"], "gaze": tl["gaze"],
                      "events": [dict(e, t=round(e["t"] + k_start, 3)) for e in tl["events"]]})
        t = max(k_start + tl["duration"], h_end or 0) + HUMAN_GAP

    total = max(c["t"] + c["duration"] for c in clips) + 0.8
    mix = np.zeros(int(total * SR), np.float32); kermit_only = mix.copy()
    for c in clips:
        x, _ = sf.read(os.path.join(a.out, c["audio"]), dtype="float32"); i = int(round(c["t"] * SR))
        gain = 0.8 if c["who"] == "human" else 1.0
        mix[i:i + len(x)] += gain * x
        if c["who"] == "kermit": kermit_only[i:i + len(x)] += x
    mix *= 0.95 / max(1e-6, np.abs(mix).max())
    sf.write(os.path.join(a.out, "scene.wav"), mix, SR)
    sf.write(os.path.join(a.out, "scene_kermit_only.wav"), kermit_only, SR)

    # one mouth track for the whole scene (Kermit only), 60 Hz, for convenience
    hz = 60; mouth = np.zeros(int(np.ceil(total * hz)), np.float32)
    for c in clips:
        if c["who"] != "kermit": continue
        m = json.load(open(os.path.join(a.out, c["timeline"])))["mouth"]["values"]; i = int(round(c["t"] * hz))
        mouth[i:i + len(m)] = np.maximum(mouth[i:i + len(m)], m[:len(mouth) - i])
    json.dump({"audio": "scene.wav", "kermit_only_audio": "scene_kermit_only.wav", "sample_rate": SR, "duration": round(total, 3),
               "human_voice": a.human_voice, "kermit_voice": a.voice, "clips": clips,
               "mouth": {"hz": hz, "values": [round(float(v), 3) for v in mouth]},
               "events": sorted([e for c in clips if c["who"] == "kermit" for e in c["events"]], key=lambda e: e["t"])},
              open(os.path.join(a.out, "scene.json"), "w"), indent=None, separators=(",", ":"))
    for c in clips:
        print(f"{c['t']:6.2f}–{c['t'] + c['duration']:6.2f}  {c['who']:6s} t{c['turn']}  {c['text'][:70]}")


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)
