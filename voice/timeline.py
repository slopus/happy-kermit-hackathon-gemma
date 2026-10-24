"""Turn JSON + TTS audio + TTS/ASR word timings → CONTRACT §2 performance timeline."""
import difflib, json, re
import numpy as np
from lipsync import mouth_curve, HZ


def _norm(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())


def align_words(text_words, timed):
    """Give every whitespace word of the turn text a (t0, t1), using timed words that may be split,
    merged or misspelled (TTS tokens, ASR output). Aligns on characters, interpolates gaps."""
    a_chars, a_owner = [], []
    for i, w in enumerate(text_words):
        for c in _norm(w):
            a_chars.append(c); a_owner.append(i)
    b_chars, b_time = [], []
    for w in timed:
        n = _norm(w["w"])
        for k, c in enumerate(n):  # spread each timed word linearly over its characters
            b_chars.append(c)
            b_time.append(w["t0"] + (w["t1"] - w["t0"]) * np.array([k, k + 1]) / len(n))
    t0 = [None] * len(text_words); t1 = [None] * len(text_words)
    sm = difflib.SequenceMatcher(None, a_chars, b_chars, autojunk=False)
    for blk in sm.get_matching_blocks():
        for j in range(blk.size):
            i = a_owner[blk.a + j]; ts = b_time[blk.b + j]
            t0[i] = ts[0] if t0[i] is None else min(t0[i], ts[0])
            t1[i] = ts[1] if t1[i] is None else max(t1[i], ts[1])
    # words with no matched characters: interpolate between neighbours
    end = max((w["t1"] for w in timed), default=0.0)
    for i in range(len(text_words)):
        if t0[i] is None:
            prev = next((t1[k] for k in range(i - 1, -1, -1) if t1[k] is not None), 0.0)
            nxt = next((t0[k] for k in range(i + 1, len(text_words)) if t0[k] is not None), end)
            t0[i], t1[i] = prev, max(prev, nxt)
    return [{"w": w, "t0": round(float(a), 3), "t1": round(float(b), 3)} for w, a, b in zip(text_words, t0, t1)]


def build_timeline(turn, audio, sr, timed_words, audio_name, lead_ms=0.0):
    words = align_words(turn["text"].split(), timed_words)
    events = []
    for g in turn.get("gestures", []):
        i = g["word"]
        if not 0 <= i < len(words): continue
        e = {"t": words[i]["t0"], "type": g["type"], "intensity": g.get("intensity", 0.5)}
        if "target" in g: e["target"] = g["target"]
        e["word"] = i
        events.append(e)
    mouth = mouth_curve(audio, sr, words, turn.get("emotion", "neutral"), HZ, lead_ms)
    return {"audio": audio_name, "sample_rate": sr, "duration": round(len(audio) / sr, 3), "words": words,
            "mouth": {"hz": HZ, "values": [float(v) for v in mouth]}, "events": sorted(events, key=lambda e: e["t"]),
            "emotion": turn.get("emotion", "neutral"), "gaze": turn.get("gaze", "speaker")}


def dump(tl, path):
    """Compact JSON: one key per line, mouth values on one line."""
    with open(path, "w") as f:
        json.dump(tl, f, separators=(",", ":"))
