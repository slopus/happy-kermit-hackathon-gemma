"""Speaker test set from Columbia ASD (5-person panel, front camera) + PERCEPTION's people.py logs on the same windows.

Every STEP s in each window where exactly one GT person speaks and all GT faces map to a people.py track:
store 4 frames over the last 0.75 s, the tracked people (ids, boxes, face boxes) and the GT speaker's track id.
The specialist answer is people.py's speaker_id at that time (full fusion: Light-ASD + VAD + mic + hysteresis).
"""
import collections, glob, json, os, sys
from PIL import Image
import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
PER = os.path.join(HERE, "..", "perception")
sys.path.insert(0, os.path.join(PER, "eval"))
from eval_columbia import ioa, WINDOWS, FPS, LAB  # noqa: E402

STEP = 9.0
DT = (-0.75, -0.5, -0.25, 0.0)


def main():
    gt = collections.defaultdict(list)
    for f in glob.glob(os.path.join(LAB, "*.txt")):
        n = os.path.basename(f)[:-4]
        for l in open(f):
            d = list(map(int, l.split()[:5])); gt[d[0]].append((n, (d[1], d[2], d[1] + d[3], d[2] + d[3]), d[4]))
    os.makedirs(os.path.join(HERE, "frames_col"), exist_ok=True)
    items = []
    for w, ws in WINDOWS.items():
        log = [json.loads(l) for l in open(os.path.join(PER, "eval", "col", f"colw{w}.ndjson"))]
        by_t = {round(o["t"], 3): o for o in log}
        ts = sorted(by_t)
        cap = cv2.VideoCapture(os.path.join(PER, "data", "col", f"colw{w}.mp4"))
        f0 = round(ws * FPS); next_t = 2.0
        for t in ts:
            if t < next_t: continue
            o = by_t[t]; fr = f0 + round(t * FPS)
            if fr not in gt: continue
            who = {}
            for name, box, lab in gt[fr]:
                best, bs = None, 0.3
                for q in o["people"]:
                    s = ioa(q["face_box"], box) if q.get("face_box") else 0
                    if s > bs: best, bs = q, s
                if best: who[name] = best["id"]
            spk = [n for n, _, lab in gt[fr] if lab]
            if len(spk) != 1 or spk[0] not in who or len(who) < len(gt[fr]) or len(o["people"]) < 2: continue
            # the GT speaker must have been speaking for the whole 1 s strip, so a lip-reader has something to see
            if not all(any(n == spk[0] and lab for n, _, lab in gt.get(fr - round(k * FPS), [])) for k in (0.25, 0.5, 0.75, 1.0)): continue
            key = f"colw{w}_{t:06.2f}"; heads = {}
            for k, dt in enumerate(DT):
                cap.set(cv2.CAP_PROP_POS_MSEC, (t + dt) * 1000); ok, img = cap.read()
                if not ok: break
                cv2.imwrite(os.path.join(HERE, "frames_col", f"{key}_{k}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
                tt = min(ts, key=lambda x: abs(x - (t + dt)))
                heads[str(dt)] = {str(p["id"]): p.get("face_box") or p["box"] for p in by_t[tt]["people"]}
            else:
                items.append({"key": key, "clip": f"colw{w}", "t": t, "gt_speaker": who[spk[0]], "gt_name": spk[0],
                              "pp_speaker": o["speaker_id"], "speech_active": o["speech_active"],
                              "people": [{k: p.get(k) for k in ("id", "box", "face_box", "speaking_p", "asd", "has_mic", "yaw_deg")} for p in o["people"]],
                              "heads": heads})
                next_t = t + STEP
    json.dump(items, open(os.path.join(HERE, "items_speaker.json"), "w"), indent=0)
    acc = sum(i["pp_speaker"] == i["gt_speaker"] for i in items) / max(1, len(items))
    print(len(items), "items; people.py speaker_id acc", round(100 * acc, 1), collections.Counter(i["gt_name"] for i in items))


if __name__ == "__main__":
    main()
