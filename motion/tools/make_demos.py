"""Build the demo timelines in samples/ from VOICE's sample lines (kermit/voice/samples).

    /var/lib/happy-agent/IsaacLab/.venv/bin/python tools/make_demos.py

Single-gesture demos reuse a VOICE line with replaced events; ``performance`` concatenates four
lines (audio, words, mouth, events) into one ~15 s turn with emotion changes between lines.
"""
from __future__ import annotations

import json
import os
import wave
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
VOICE = HERE.parent / "voice" / "samples"
OUT = HERE / "samples"


def load(name):
    return json.loads((VOICE / f"{name}.timeline.json").read_text())


def word_t(tl, i):
    return tl["words"][i]["t0"]


def write(name, tl):
    OUT.mkdir(exist_ok=True)
    if not str(tl["audio"]).startswith(str(OUT)) and not (OUT / tl["audio"]).exists():
        tl["audio"] = os.path.relpath(VOICE / tl["audio"], OUT)
    (OUT / f"{name}.timeline.json").write_text(json.dumps(tl, indent=1))
    print(f"{name}: {tl['duration']:.2f}s, events {[(round(e['t'], 2), e['type']) for e in tl['events']]}")


def single():
    t = load("open_1")
    t["events"] = []  # pure lip sync: mouth + idle life + speech accents only
    write("talk", t)

    t = load("open_3")  # "You talk, I listen, and your agents get to work. Pretty neat, huh?"
    t["events"] = [{"t": word_t(t, 1), "type": "nod", "intensity": 1.0},
                   {"t": word_t(t, 9), "type": "nod", "intensity": 0.7}]
    write("nod", t)

    t = load("open_2")  # "Whoa! Is that a robot arm? Is that my robot arm?" (surprise, look_at arm, lean_in)
    t["gaze"] = "speaker"  # start facing the speaker so the look over at the arm reads
    write("whoa", t)

    t = load("open_3")
    t["emotion"], t["gaze"] = "happy", "speaker"
    t["events"] = [{"t": word_t(t, 1), "type": "look_at", "target": "left", "intensity": 1.0},
                   {"t": word_t(t, 7), "type": "look_at", "target": "right", "intensity": 1.0},
                   {"t": word_t(t, 12), "type": "look_at", "target": "speaker", "intensity": 1.0}]
    write("turn", t)


def performance():
    parts = [
        ("open_1", [], 0.35),
        ("open_2", [], 0.45),
        ("open_3", [("look_at", 1, {"target": "left"}), ("look_at", 7, {"target": "right"}),
                    ("look_at", 12, {"target": "speaker"})], 0.4),
        ("think_1", [], 0.0),
    ]
    sr = 24000
    audio, words, mouth, events = [], [], [], []
    t_off = 0.0
    hz = 60
    for i, (name, extra, gap) in enumerate(parts):
        tl = load(name)
        with wave.open(str(VOICE / tl["audio"])) as w:
            assert w.getframerate() == sr
            pcm = np.frombuffer(w.readframes(w.getnframes()), np.int16)
        dur = len(pcm) / sr
        n_mouth = int(round((t_off) * hz)) - len(mouth)
        mouth += [0.0] * max(0, n_mouth)
        mouth += tl["mouth"]["values"]
        audio.append(pcm)
        audio.append(np.zeros(int(gap * sr), np.int16))
        words += [dict(w, t0=round(w["t0"] + t_off, 3), t1=round(w["t1"] + t_off, 3)) for w in tl["words"]]
        if i:
            events.append({"t": t_off, "type": "emotion", "value": tl["emotion"]})
            if tl.get("gaze") and not any(e["type"] == "look_at" for e in tl["events"]) and not extra:
                events.append({"t": t_off, "type": "look_at", "target": tl["gaze"].removeprefix("target:"),
                               "intensity": 0.6})
        drop = {"look_at"} if extra else set()
        events += [dict(e, t=round(e["t"] + t_off, 3)) for e in tl["events"] if e["type"] not in drop]
        for typ, wi, kw in extra:
            events.append(dict({"t": round(tl["words"][wi]["t0"] + t_off, 3), "type": typ, "intensity": 1.0}, **kw))
        t_off += dur + gap
    pcm = np.concatenate(audio)
    OUT.mkdir(exist_ok=True)
    with wave.open(str(OUT / "performance.wav"), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    mouth += [0.0] * (int(np.ceil(len(pcm) / sr * hz)) - len(mouth))
    first = load(parts[0][0])
    tl = {"audio": "performance.wav", "sample_rate": sr, "duration": round(len(pcm) / sr, 3), "words": words,
          "mouth": {"hz": hz, "values": mouth}, "events": sorted(events, key=lambda e: e["t"]),
          "emotion": first["emotion"], "gaze": first["gaze"]}
    write("performance", tl)


def walk_trace(path=OUT / "walk_synthetic.ndjson", dur=12.0, hz=20):
    """Synthetic PERCEPTION trace (CONTRACT section 4): a person walks from -40 to +40 deg at ~1.4 m,
    with a 1.6 s tracking dropout in the middle. Stand-in until perception/samples/walk.ndjson exists."""
    cam = np.array([0.05, 0.0, 0.10])
    lines = []
    for i in range(int(dur * hz) + 1):
        t = i / hz
        s = np.clip((t - 1.0) / (dur - 2.0), 0, 1)
        s = s * s * (3 - 2 * s)
        yaw = np.deg2rad(-40 + 80 * s)
        dist = 1.4 + 0.1 * np.sin(2 * np.pi * t / dur)
        face = cam + np.array([dist * np.cos(yaw), dist * np.sin(yaw), 0.42 + 0.015 * np.sin(2 * np.pi * 1.8 * t)])
        present = not (5.2 <= t <= 6.8)
        rel = face - cam
        row = {"t": round(t, 3), "present": present}
        if present:
            row.update(yaw_deg=round(float(np.rad2deg(np.arctan2(rel[1], rel[0]))), 2),
                       pitch_deg=round(float(np.rad2deg(np.arctan2(rel[2], np.hypot(rel[0], rel[1])))), 2),
                       dist_m=round(float(np.linalg.norm(rel)), 3), conf=0.9,
                       target_base=[round(float(x), 3) for x in face])
        lines.append(json.dumps(row))
    path.write_text("\n".join(lines) + "\n")
    print(f"{path.name}: {dur}s, dropout 5.2-6.8 s")


def follow_talk():
    """Two lines back to back, gaze on the speaker, a few nods: talking while following a person."""
    a, b = load("open_1"), load("open_3")
    sr = 24000
    pcm = []
    for tl in (a, b):
        with wave.open(str(VOICE / tl["audio"])) as w:
            pcm.append(np.frombuffer(w.readframes(w.getnframes()), np.int16))
    gap = 1.2
    off = len(pcm[0]) / sr + gap
    audio = np.concatenate([pcm[0], np.zeros(int(gap * sr), np.int16), pcm[1]])
    with wave.open(str(OUT / "follow_talk.wav"), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(audio.tobytes())
    mouth = a["mouth"]["values"] + [0.0] * (int(round(off * 60)) - len(a["mouth"]["values"])) + b["mouth"]["values"]
    keep = {"nod", "head_bob", "tilt_left", "tilt_right"}
    events = [e for e in a["events"] if e["type"] in keep] + \
        [dict(e, t=round(e["t"] + off, 3)) for e in b["events"] if e["type"] in keep]
    words = a["words"] + [dict(w, t0=round(w["t0"] + off, 3), t1=round(w["t1"] + off, 3)) for w in b["words"]]
    tl = {"audio": "follow_talk.wav", "sample_rate": sr, "duration": round(len(audio) / sr, 3), "words": words,
          "mouth": {"hz": 60, "values": mouth}, "events": events, "emotion": "happy", "gaze": "speaker"}
    (OUT / "follow_talk.timeline.json").write_text(json.dumps(tl, indent=1))
    print(f"follow_talk: {tl['duration']:.2f}s")


if __name__ == "__main__":
    single()
    performance()
    walk_trace()
    follow_talk()
