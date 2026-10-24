"""Build the frame-level test set from clips + people.py logs.

  python testset.py extract            # frames/<clip>_<t>.jpg (+ 3 earlier frames over 1 s), people json, marked png
  python testset.py sheet <clip>       # marked frames of one clip for labelling

Labels live in labels.json (hand-made, see the README): per item the raised-hand ids, descriptions of
some people (as a human would say them), described-look commands with their target id, and the speaker id.
"""
import json, os, sys, subprocess
import numpy as np
from PIL import Image
import gemma

HERE = os.path.dirname(os.path.abspath(__file__))
P = lambda *a: os.path.join(HERE, *a)

# (clip, t) picked from the contact sheets: moments with raised hands / a mic / a clear speaker
PICKS = {
    "biz_hands1": [3.0, 6.0, 9.0], "biz_hands2": [2.0, 8.0], "class_kids": [17.0, 20.0], "class_kids2": [3.0, 6.0],
    "concert_stock": [20.0, 30.0, 38.0], "montessori_hands": [3.0, 9.0], "school_hands": [12.0, 24.0],
    "conf_woman_hand": [1.5, 6.0], "lect_hands": [3.0], "obama_town": [20.0, 40.0, 70.0], "speech_aud": [5.0, 40.0],
    "panel_mic": [5.0], "comedy_crowd_q": [20.0, 45.0], "aud_clap": [20.0], "festival_rear": [10.0], "crowd_hands": [8.0],
}


def load_log(clip):
    return [json.loads(l) for l in open(P("pp", clip + ".ndjson"))]


def at(log, t):
    return min(log, key=lambda o: abs(o["t"] - t))


def frame(clip, t):
    raw = subprocess.run(["ffmpeg", "-v", "quiet", "-ss", f"{t:.3f}", "-i", P("clips", clip + ".mp4"), "-frames:v", "1",
                          "-f", "image2pipe", "-vcodec", "png", "-"], capture_output=True).stdout
    import io
    return Image.open(io.BytesIO(raw)).convert("RGB")


def extract():
    os.makedirs(P("frames"), exist_ok=True); os.makedirs(P("marked"), exist_ok=True)
    items = []
    for clip, ts in PICKS.items():
        if not os.path.exists(P("pp", clip + ".ndjson")): print("no log", clip); continue
        log = load_log(clip)
        for t in ts:
            key = f"{clip}_{t:05.1f}"
            o = at(log, t)
            strip = []
            for k, dt in enumerate((-0.75, -0.5, -0.25, 0.0)):
                f = frame(clip, max(0, t + dt)); f.save(P("frames", f"{key}_{k}.jpg"), quality=90); strip.append(f)
                strip_o = at(log, t + dt)
            ppl = [{k: p[k] for k in ("id", "box", "face_box", "hand_up", "has_mic", "speaking_p", "asd", "yaw_deg", "pitch_deg", "facing") if k in p}
                   for p in o["people"]]
            hist = {round(dt, 2): {p["id"]: p.get("face_box") or p["box"] for p in at(log, t + dt)["people"]} for dt in (-0.75, -0.5, -0.25, 0.0)}
            items.append({"key": key, "clip": clip, "t": t, "size": strip[-1].size, "speaker_id": o.get("speaker_id"),
                          "speech_active": o.get("speech_active"), "people": ppl,
                          "heads": {str(k): {str(i): b for i, b in v.items()} for k, v in hist.items()}})
            gemma.marked(strip[-1], ppl, max_side=1600).save(P("marked", key + ".jpg"), quality=85)
            print(key, len(ppl), "people", "hand_up:", [p["id"] for p in ppl if p.get("hand_up")])
    json.dump(items, open(P("items.json"), "w"), indent=0)


if __name__ == "__main__":
    {"extract": extract}[sys.argv[1]]()
