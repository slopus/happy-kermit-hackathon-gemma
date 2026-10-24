"""Mic (hand-held) detection, person-level, single frame, on hand-labelled real frames (eval/mic/dump.json).
GT: townhall frames where the questioner holds a hand-held mic (#0-2,6-10 man on the right; #16-17 woman in green);
everything else (Columbia table/gooseneck mics, lapel mics, lecture) is negative."""
import json, math, itertools, os
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
D = json.load(open(os.path.join(HERE, "mic", "dump.json")))
POS = {("townhall_cbs.mp4", i): "right" for i in (0, 1, 2, 6, 7, 8, 9, 10, 16, 17)}
D = [r for r in D if r["src"] != "carol.mp4"]          # Carol: too ambiguous to label reliably at this resolution
NOSE, LSHO, RSHO, LWRI, RWRI = 0, 5, 6, 9, 10

def person_has(p, mics, tau, reach, mouth):
    kp = np.array(p["kp"]); b = p["box"]; bw = max(b[2] - b[0], 1)
    sw = abs(kp[LSHO, 0] - kp[RSHO, 0]) if min(kp[LSHO, 2], kp[RSHO, 2]) > 0.3 else 0.35 * bw
    for m in mics:
        if m["conf"] < tau: continue
        x0, y0, x1, y1 = m["box"]; mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        wr = [kp[i, :2] for i in (LWRI, RWRI) if kp[i, 2] > 0.3]
        if not wr: continue
        if min(math.hypot(mx - w[0], my - w[1]) for w in wr) / bw > reach: continue
        if mouth:
            if kp[NOSE, 2] < 0.3: continue
            # mic head (top of box) near the mouth: within ~1 shoulder width of the nose, not far below the shoulders
            if math.hypot((x0 + x1) / 2 - kp[NOSE, 0], y0 - kp[NOSE, 1]) > mouth * sw: continue
        return True
    return False

res = []
for model in ("mic_s", "mic_m"):
    for tau, reach, mouth in itertools.product((0.1, 0.15, 0.2, 0.3, 0.4), (0.3, 0.6), (0, 1.0, 1.5)):
        tp = fp = fn = tn = 0
        for r in D:
            gt_side = POS.get((r["src"], r["i"]))
            for p in r["people"]:
                cx = (p["box"][0] + p["box"][2]) / 2
                gt = gt_side == "right" and cx > r["w"] / 2
                pr = person_has(p, r[model], tau, reach, mouth)
                tp += gt and pr; fp += (not gt) and pr; fn += gt and not pr; tn += (not gt) and not pr
        P = tp / max(1, tp + fp); R = tp / max(1, tp + fn)
        res.append({"model": model, "conf": tau, "reach": reach, "mouth": mouth, "P": round(P, 2), "R": round(R, 2),
                    "F1": round(2 * P * R / max(1e-9, P + R), 2), "tp": tp, "fp": fp, "fn": fn, "tn": tn})
res.sort(key=lambda x: -x["F1"])
for x in res[:12]: print(x)
json.dump({"frames": len(D), "persons": sum(len(r["people"]) for r in D), "best": res[:12], "all": res},
          open(os.path.join(HERE, "mic_results.json"), "w"), indent=1)
