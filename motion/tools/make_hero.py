"""Build the hero-scene timeline: the human's line, then the frog's reply, for the 4 demo turns.

    /var/lib/happy-agent/IsaacLab/.venv/bin/python tools/make_hero.py
    /var/lib/happy-agent/IsaacLab/.venv/bin/python render.py samples/hero.timeline.json -o videos/hero.mp4

While the human talks, the puppet listens: it looks at the speaker, tilts its head, and nods as
the line ends. MOTION-only extension to the CONTRACT timeline: ``captions`` ([t0, t1, who, text])
for lines without word timings, plus ``emotion`` events ({"type": "emotion", "value": ...}).
"""
from __future__ import annotations

import json
import wave
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
KERMIT = HERE.parent
VOICE = KERMIT / "voice" / "samples" / "demo_script"
BRAIN = KERMIT / "brain" / "demo_script" / "opening_turns.json"
OUT = HERE / "samples"
SR, HZ = 24000, 60
GAP_AFTER_HEARD = 0.45   # beat between the question and the answer
GAP_AFTER_TURN = 0.7


def read_wav(path):
    with wave.open(str(path)) as w:
        assert w.getframerate() == SR and w.getnchannels() == 1
        return np.frombuffer(w.readframes(w.getnframes()), np.int16)


def main():
    script = json.loads(BRAIN.read_text())
    audio, words, mouth, events, captions = [np.zeros(int(0.6 * SR), np.int16)], [], [], [], []
    t = 0.6
    first_emotion = None
    for turn in script:
        n = turn["turn"]
        # --- the human speaks; the puppet listens
        heard = read_wav(VOICE / f"heard_{n:02d}.wav")
        hd = len(heard) / SR
        captions.append([round(t, 3), round(t + hd, 3), "Human", turn["heard"].capitalize()])
        events += [{"t": round(t + 0.1, 3), "type": "look_at", "target": "speaker", "intensity": 1.0},
                   {"t": round(t + 0.3, 3), "type": "tilt_left" if n % 2 else "tilt_right", "intensity": 0.45},
                   {"t": round(t + hd - 0.45, 3), "type": "nod", "intensity": 0.35}]
        audio += [heard, np.zeros(int(GAP_AFTER_HEARD * SR), np.int16)]
        t += hd + GAP_AFTER_HEARD
        # --- the frog answers
        tl = json.loads((VOICE / f"turn_{n:02d}.timeline.json").read_text())
        pcm = read_wav(VOICE / tl["audio"])
        first_emotion = first_emotion or tl["emotion"]
        events.append({"t": round(t - 0.3, 3), "type": "emotion", "value": tl["emotion"]})
        if not any(e["type"] == "look_at" and e["t"] < 0.5 for e in tl["events"]):
            events.append({"t": round(t - 0.2, 3), "type": "look_at", "target": tl["gaze"].removeprefix("target:"),
                           "intensity": 0.8})
        events += [dict(e, t=round(e["t"] + t, 3)) for e in tl["events"]]
        words += [dict(w, t0=round(w["t0"] + t, 3), t1=round(w["t1"] + t, 3)) for w in tl["words"]]
        mouth += [0.0] * (int(round(t * HZ)) - len(mouth))
        mouth += tl["mouth"]["values"]
        audio += [pcm, np.zeros(int(GAP_AFTER_TURN * SR), np.int16)]
        t += len(pcm) / SR + GAP_AFTER_TURN
    pcm = np.concatenate(audio)
    OUT.mkdir(exist_ok=True)
    with wave.open(str(OUT / "hero.wav"), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())
    dur = len(pcm) / SR
    mouth += [0.0] * (int(np.ceil(dur * HZ)) - len(mouth))
    out = {"audio": "hero.wav", "sample_rate": SR, "duration": round(dur, 3), "words": words,
           "mouth": {"hz": HZ, "values": mouth}, "events": sorted(events, key=lambda e: e["t"]),
           "emotion": "neutral", "gaze": "speaker", "captions": captions}
    (OUT / "hero.timeline.json").write_text(json.dumps(out, indent=1))
    print(f"hero: {dur:.1f}s, {len(words)} words, {len(events)} events")


if __name__ == "__main__":
    main()
