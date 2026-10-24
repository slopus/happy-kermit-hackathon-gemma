"""(a) Who is speaking: Gemma 4 E4B vs people.py on the Columbia items (items_speaker.json).

  python eval_speaker.py [variant ...]      # variants: frame, strip, faces, faces_wide → results/speaker_<variant>.json
Gemma answers with a track id from an enum (json_schema), so every answer is a valid id.
"""
import json, os, sys, time
import numpy as np
from PIL import Image
import gemma

HERE = os.path.dirname(os.path.abspath(__file__))
DT = ("-0.75", "-0.5", "-0.25", "0.0")

ASK_FRAME = ("This is a video frame of a panel discussion. Each visible person has a numbered tag. Exactly one of them is "
             "speaking right now. Which numbered person is speaking? Look at mouths, gestures and who the others look at.")
ASK_STRIP = ("These are {n} frames from the last second of a panel discussion video, oldest first. Each person has the same "
             "numbered tag in every frame. Exactly one person is speaking. Which numbered person is speaking? Compare the "
             "mouths across frames: the speaker's mouth opens and closes.")
ASK_FACES = ("Each row shows one person's face (number on the left) in 4 frames from the last second, oldest on the left. "
             "Exactly one of these people is talking. Whose mouth is opening and closing across the frames? Answer with the row number.")


def schema(ids):
    return {"type": "object", "properties": {"id": {"type": "integer", "enum": ids}}, "required": ["id"]}


def frames(it):
    return [Image.open(os.path.join(HERE, "frames_col", f"{it['key']}_{k}.jpg")).convert("RGB") for k in range(4)]


def run(variant, items):
    out = []
    for it in items:
        ids = sorted(p["id"] for p in it["people"]); fr = frames(it)
        if variant == "frame":
            imgs, q = [gemma.marked(fr[-1], it["people"])], ASK_FRAME
        elif variant == "strip":
            per = [[{"id": int(i), "box": b} for i, b in it["heads"][d].items()] for d in DT]
            boxes = {p["id"]: p["box"] for p in it["people"]}
            imgs = [gemma.marked(f, [{"id": p["id"], "box": boxes.get(p["id"], p["box"])} for p in pp]) for f, pp in zip(fr, per)]
            q = ASK_STRIP.format(n=len(imgs))
        elif variant in ("faces", "faces_wide"):
            per = [{int(i): b for i, b in it["heads"][d].items() if int(i) in ids} for d in DT]
            imgs = [gemma.crop_grid(fr, per, cell=200 if variant == "faces_wide" else 140, pad=0.6 if variant == "faces_wide" else 0.2)]
            q = ASK_FACES
        pred, ms = gemma.ask_choice(imgs, q + " Answer with just the number.", ids)
        out.append({"key": it["key"], "gt": it["gt_speaker"], "pp": it["pp_speaker"], "gemma": pred, "ms": round(ms), "n": len(ids)})
        print(variant, it["key"], "gt", it["gt_speaker"], "pp", it["pp_speaker"], "gemma", pred, f"{ms:.0f} ms", flush=True)
    return out


def summary(rs):
    n = len(rs); g = sum(r["gemma"] == r["gt"] for r in rs); p = sum(r["pp"] == r["gt"] for r in rs)
    chance = np.mean([1 / r["n"] for r in rs])
    return {"n": n, "gemma_acc": round(100 * g / n, 1), "pp_acc": round(100 * p / n, 1), "chance": round(100 * chance, 1),
            "ms_median": float(np.median([r["ms"] for r in rs])), "ms_p90": float(np.percentile([r["ms"] for r in rs], 90))}


if __name__ == "__main__":
    items = json.load(open(os.path.join(HERE, "items_speaker.json")))
    if os.environ.get("N"): items = items[:int(os.environ["N"])]
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    for v in sys.argv[1:] or ["frame", "strip", "faces"]:
        rs = run(v, items); s = summary(rs)
        json.dump({"summary": s, "items": rs}, open(os.path.join(HERE, "results", f"speaker_{v}.json"), "w"), indent=0)
        print("SUMMARY", v, json.dumps(s), flush=True)
