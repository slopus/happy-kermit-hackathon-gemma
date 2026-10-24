"""AVA-ActiveSpeaker val: Light-ASD on ground-truth face tracks, scored two ways.

  offline : whole track at once (how papers report it, bidirectional GRU sees the future)
  causal  : what people.py does live: every 0.2 s, a 1.2 s window ending now, mean of the last 8 frame logits,
            held until the next tick (no future frames)
Metric: frame-level AP (SPEAKING_AUDIBLE = positive), sklearn average_precision over all labelled frames.

  python eval/eval_ava.py data/ava/053oq2xB3oU.mkv ... [--minutes 15]
"""
import argparse, collections, csv, json, os, sys, time
import numpy as np, cv2
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, ".."))
import lightasd, people
from sklearn.metrics import average_precision_score
CROP = os.environ.get('CROP', 'box')
CONFIGS = [tuple(float(x) if i == 0 else int(x) for i, x in enumerate(c.split(':'))) for c in os.environ.get('CONFIGS', '1.2:8:0').split(',')]


def run_video(path, asd, minutes):
    vid = os.path.basename(path).split(".")[0]
    rows = collections.defaultdict(list)
    t_end = 900 + 60 * minutes
    for r in csv.reader(open(os.path.join(HERE, "..", "data", "ava", "ava_activespeaker_test_v1.0", f"{vid}-activespeaker.csv"))):
        t = float(r[1])
        if t <= t_end: rows[round(t, 2)].append((r[7], [float(v) for v in r[2:6]], r[6] == "SPEAKING_AUDIBLE"))
    times = np.array(sorted(rows))
    audio = people.load_audio(path)
    import subprocess
    W0, H0 = 640, 360                                          # decode with ffmpeg (accurate seek) at 30 fps, 640×360
    t0 = times[0] - 0.5
    pr = subprocess.Popen(["ffmpeg", "-v", "quiet", "-ss", f"{t0:.3f}", "-i", path, "-t", f"{times[-1] - t0 + 0.2:.3f}",
                           "-vf", f"fps=30,scale={W0}:{H0}", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], stdout=subprocess.PIPE)
    crops = collections.defaultdict(list)        # entity → [(t, crop, label)]
    fi = 0
    while True:
        buf = pr.stdout.read(W0 * H0 * 3)
        if len(buf) < W0 * H0 * 3: break
        img = np.frombuffer(buf, np.uint8).reshape(H0, W0, 3); t = t0 + fi / 30; fi += 1
        j = np.searchsorted(times, t); cand = [k for k in (j - 1, j) if 0 <= k < len(times)]
        k = min(cand, key=lambda k: abs(times[k] - t))
        if abs(times[k] - t) > 0.017: continue
        h, w = img.shape[:2]; gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        for ent, (x1, y1, x2, y2), lab in rows[times[k]]:
            x1, x2, y1, y2 = x1 * w, x2 * w, y1 * h, y2 * h
            crops[ent].append((times[k], lightasd.crop_face(gray, (x1 + x2) / 2, (y1 + y2) / 2, max(x2 - x1, y2 - y1), mode=CROP), lab))
    res = {"offline": [], "label": []}
    for ent, cs in crops.items():
        ts = np.array([c[0] for c in cs]); labs = [c[2] for c in cs]
        grid = np.arange(ts[0], ts[-1] + 1e-6, 0.04)
        idx = np.clip(np.searchsorted(ts, grid), 0, len(ts) - 1)
        faces = np.stack([cs[i][1] for i in idx])
        a = audio[int(ts[0] * 16000): int((grid[-1] + 0.04) * 16000)]
        mf = lightasd.mfcc(a); T = min(len(faces), len(mf) // 4)
        if T < 3: continue
        off = asd(faces[None, :T], mf[:4 * T])[0]
        # causal configs: ticks every 0.2 s, window W s ending at the tick, mean of frame logits [T-a, T-b)
        caus = {}
        for cfg in CONFIGS:
            W, fa, fb = cfg; Tw = int(round(W * 25)); ct, cv = [], []
            for tick in np.arange(ts[0] + 0.2, ts[-1] + 0.2, 0.2):
                g = tick - W + (np.arange(Tw) + 0.5) / 25
                ii = np.clip(np.searchsorted(ts, g), 0, len(ts) - 1)          # before track start: repeat the first crop
                if tick - W < 0: continue
                x = audio[int((tick - W) * 16000): int(tick * 16000)]
                if len(x) < W * 16000 - 160: continue
                m = lightasd.mfcc(x)[:4 * Tw]
                if len(m) < 4 * Tw: m = np.pad(m, ((0, 4 * Tw - len(m)), (0, 0)), mode="edge")
                lg = asd(np.stack([cs[i][1] for i in ii])[None], m)[0]
                ct.append(tick); cv.append(float(np.mean(lg[Tw - fa: Tw - fb])))
            caus[cfg] = (np.array(ct), cv)
        for t, lab in zip(ts, labs):
            gi = min(int(round((t - ts[0]) / 0.04)), T - 1)
            res["offline"].append(float(off[gi]))
            for cfg, (ct, cv) in caus.items():
                ci = np.searchsorted(ct, t + 1e-6) - 1            # latest tick at or before t
                res.setdefault(str(cfg), []).append(cv[ci] if ci >= 0 else cv[0] if len(cv) else 0.0)
            res["label"].append(int(lab))
    return vid, res


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("videos", nargs="+"); ap.add_argument("--minutes", type=float, default=15)
    ap.add_argument("--out", default=os.path.join(HERE, "ava_results.json")); a = ap.parse_args()
    asd = lightasd.LightASD()
    allr = collections.defaultdict(list); per = {}
    for v in a.videos:
        t = time.time(); vid, r = run_video(v, asd, a.minutes)
        keys = [k for k in r if k != "label"]
        per[vid] = {k: round(100 * average_precision_score(r["label"], r[k]), 1) for k in keys}
        per[vid].update(frames=len(r["label"]), pos=int(sum(r["label"])), sec=round(time.time() - t))
        for k in r: allr[k] += r[k]
        print(vid, per[vid], flush=True)
        np.savez(a.out.replace(".json", f"_{vid}.npz"), **{k: np.array(v) for k, v in r.items()})
    y = np.array(allr["label"]); keys = [k for k in allr if k != "label"]
    summ = {k: round(100 * average_precision_score(y, allr[k]), 1) for k in keys}
    summ.update({f"acc@0 {k}": round(100 * float(np.mean((np.array(allr[k]) > 0) == y)), 1) for k in keys})
    out = {"crop": CROP, "configs": "(window_s, a, b): mean of frame logits [T-a, T-b) of a causal window ending now",
           "per_video": per, "all": summ, "frames": len(y), "positives": int(y.sum())}
    print(json.dumps(out, indent=1)); json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
