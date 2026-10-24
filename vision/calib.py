"""Gemma box_2d offset on people: ask for every person's box, match to people.py (YOLO) boxes, fit the error.

  python calib.py → results/calib.json
Gemma 4 returns [ymin, xmin, ymax, xmax] on a 0-1000 grid. We compare the native frame vs a frame letterboxed to a
square (the image llama.cpp resizes is then exactly what the model sees, no aspect change).
"""
import json, os, re
import numpy as np
from PIL import Image
import gemma

HERE = os.path.dirname(os.path.abspath(__file__))
KEYS = ["conf_woman_hand_001.5", "conf_woman_hand_006.0", "montessori_hands_003.0", "montessori_hands_009.0", "class_kids2_003.0",
        "class_kids_017.0", "obama_town_020.0", "biz_hands2_002.0", "lect_hands_003.0", "panel_mic_005.0", "school_hands_012.0"]
Q = ('Detect every person. Respond with ONLY a JSON array like [{"box_2d":[ymin,xmin,ymax,xmax]}] with coordinates normalized to 0-1000.')


def iou(a, b):
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    i = max(0, x1 - x0) * max(0, y1 - y0)
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i + 1e-9)


def parse(s):
    try: return [b["box_2d"] for b in json.loads(re.sub(r"^```(json)?|```$", "", s.strip()).strip()) if len(b.get("box_2d", [])) == 4]
    except Exception: return []


def run(mode):
    items = {i["key"]: i for i in json.load(open(os.path.join(HERE, "items.json")))}
    pairs, ms_all = [], []
    for k in KEYS:
        it = items[k]; img = Image.open(os.path.join(HERE, "frames", f"{k}_3.jpg")).convert("RGB"); W, H = img.size
        if mode == "square":
            S = max(W, H); sq = Image.new("RGB", (S, S)); oy = (S - H) // 2; sq.paste(img, (0, oy)); send = sq
            to_px = lambda b: [b[1] * S / 1000, b[0] * S / 1000 - oy, b[3] * S / 1000, b[2] * S / 1000 - oy]
        else:
            send = img; to_px = lambda b: [b[1] * W / 1000, b[0] * H / 1000, b[3] * W / 1000, b[2] * H / 1000]
        txt, ms, _ = gemma.ask([send], Q, max_tokens=400); ms_all.append(ms)
        for b in parse(txt):
            px = to_px(b)
            best = max(it["people"], key=lambda p: iou(p["box"], px), default=None)
            # match on x overlap only, so the y error itself doesn't break the matching
            bx = max(it["people"], key=lambda p: iou([p["box"][0], 0, p["box"][2], 1], [px[0], 0, px[2], 1]), default=None)
            if bx is None or iou([bx["box"][0], 0, bx["box"][2], 1], [px[0], 0, px[2], 1]) < 0.6: continue
            g = bx["box"]
            pairs.append({"key": k, "H": H, "pred": [round(v, 1) for v in px], "yolo": g, "iou": round(iou(g, px), 3)})
    P = np.array([p["pred"] for p in pairs]); G = np.array([p["yolo"] for p in pairs]); Hs = np.array([p["H"] for p in pairs])
    dy0, dy1 = (P[:, 1] - G[:, 1]) / Hs, (P[:, 3] - G[:, 3]) / Hs          # top / bottom edge error, fraction of height
    cyp, cyg = (P[:, 1] + P[:, 3]) / 2 / Hs, (P[:, 1 + 0] * 0 + (G[:, 1] + G[:, 3]) / 2) / Hs
    a, b = np.polyfit(cyp, cyg, 1)                                         # yolo_cy ≈ a·gemma_cy + b
    return {"mode": mode, "n": len(pairs), "median_iou": float(np.median([p["iou"] for p in pairs])),
            "top_err_frac_median": float(np.median(dy0)), "bottom_err_frac_median": float(np.median(dy1)),
            "x_center_err_frac_median": float(np.median(((P[:, 0] + P[:, 2]) - (G[:, 0] + G[:, 2])) / 2 / np.array([items[p['key']]['size'][0] for p in pairs]))),
            "fit_cy": [round(float(a), 3), round(float(b), 3)], "ms_median": float(np.median(ms_all)), "pairs": pairs}


if __name__ == "__main__":
    out = {m: run(m) for m in ("native", "square")}
    for m, r in out.items(): print(m, {k: v for k, v in r.items() if k != "pairs"})
    json.dump(out, open(os.path.join(HERE, "results", "calib.json"), "w"), indent=0)
