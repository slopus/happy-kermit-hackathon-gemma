"""Columbia ASD (Chakravarty & Tuytelaars) scored on people.py logs: the full live pipeline (YOLO-pose + YuNet + causal
Light-ASD + Silero VAD + fusion + speaker hysteresis) on 5 windows of the panel video (12.5 min, 5 speakers).

  python eval/eval_columbia.py            # reads eval/col/colw*.ndjson, writes eval/columbia_results.json
"""
import glob, json, os, collections
import numpy as np
from sklearn.metrics import f1_score

HERE = os.path.dirname(os.path.abspath(__file__))
LAB = os.path.join(HERE, "..", "data", "col", "col_labels", "fusion")
WINDOWS = {1: 900, 2: 1320, 3: 1980, 4: 4800, 5: 5010}
FPS = 29.97


def ioa(gt, b):   # intersection over the GT box (as in the official Columbia script)
    x0, y0, x1, y1 = max(gt[0], b[0]), max(gt[1], b[1]), min(gt[2], b[2]), min(gt[3], b[3])
    return max(0, x1 - x0) * max(0, y1 - y0) / max(1, (gt[2] - gt[0]) * (gt[3] - gt[1]))


def main(logdir=os.path.join(HERE, "col"), out_name="columbia_results.json", quiet=False):
    gt = collections.defaultdict(list)
    for f in glob.glob(os.path.join(LAB, "*.txt")):
        n = os.path.basename(f)[:-4]
        for l in open(f):
            d = list(map(int, l.split()[:5])); gt[d[0]].append((n, (d[1], d[2], d[1] + d[3], d[2] + d[3]), d[4]))
    R = collections.defaultdict(lambda: {"asd": [], "fused": [], "y": [], "asd_y": []})
    spk = {"correct": 0, "total": 0, "none": 0}
    switches = []
    for w, ws in WINDOWS.items():
        p = os.path.join(logdir, f"colw{w}.ndjson")
        if not os.path.exists(p): continue
        f0 = round(ws * FPS)
        gt_spk_prev, change_t, pending = None, None, None
        for line in open(p):
            o = json.loads(line); fr = f0 + round(o["t"] * FPS)
            if fr not in gt: continue
            who = {}                                        # GT name → our person id
            for name, box, lab in gt[fr]:
                best, bs = None, 0.3
                for q in o["people"]:
                    s = ioa(q["face_box"], box) if q.get("face_box") else 0     # our (tighter) face box inside the GT box
                    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
                    if s == 0 and q["box"][0] <= cx <= q["box"][2] and q["box"][1] <= cy <= q["box"][1] + 0.5 * (q["box"][3] - q["box"][1]): s = 0.31
                    if s > bs: best, bs = q, s
                r = R[name]; r["y"].append(lab); r["fused"].append(best["speaking_p"] if best else 0.0)
                if best and best.get("asd") is not None and best.get("face_box"): r["asd"].append(best["asd"]); r["asd_y"].append(lab)
                if best: who[name] = best["id"]
            speakers = [n for n, _, lab in gt[fr] if lab]
            if len(speakers) == 1 and speakers[0] in who:
                spk["total"] += 1; spk["correct"] += int(o["speaker_id"] == who[speakers[0]])
                if o["speaker_id"] is None: spk["none"] += 1
                # switch latency: GT single speaker changes → time until speaker_id follows
                if speakers[0] != gt_spk_prev:
                    if gt_spk_prev is not None: pending = (speakers[0], o["t"])
                    gt_spk_prev = speakers[0]
                if pending and pending[0] == speakers[0] and o["speaker_id"] == who[speakers[0]]:
                    switches.append(round(o["t"] - pending[1], 2)); pending = None
    out = {"per_speaker": {}, "speaker_id_accuracy": None}
    f1s_a, f1s_f = [], []
    for n, r in sorted(R.items()):
        if n == "abbas": continue
        y = np.array(r["y"]); fz = f1_score(y, np.array(r["fused"]) > 0.5, zero_division=0)
        ya = np.array(r["asd_y"]); fa = f1_score(ya, np.array(r["asd"]) > 0, zero_division=0) if len(ya) else None
        out["per_speaker"][n] = {"frames": len(y), "speaking_frac": round(float(y.mean()), 2), "F1_fused": round(100 * fz, 1),
                                 "F1_asd_only": None if fa is None else round(100 * fa, 1), "asd_frames": len(ya),
                                 "acc_fused": round(100 * float(np.mean((np.array(r["fused"]) > 0.5) == y)), 1)}
        f1s_f.append(fz)
        if fa is not None: f1s_a.append(fa)
    out["mean_F1_fused"] = round(100 * float(np.mean(f1s_f)), 1)
    out["mean_F1_asd_only"] = round(100 * float(np.mean(f1s_a)), 1)
    out["speaker_id_accuracy"] = round(100 * spk["correct"] / max(1, spk["total"]), 1)
    out["speaker_id_frames"] = spk["total"]; out["speaker_id_none_frames"] = spk["none"]
    out["switch_latency_s"] = {"n": len(switches), "median": float(np.median(switches)) if switches else None, "all": switches}
    if not quiet: print(json.dumps(out, indent=1))
    if out_name: json.dump(out, open(os.path.join(HERE, out_name), "w"), indent=1)
    return out


if __name__ == "__main__":
    import sys
    main(*(sys.argv[1:3]))
