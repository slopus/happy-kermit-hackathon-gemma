"""Calibrate the oracle's ASD noise: run the oracle on Columbia ground truth (who speaks, per frame) with the same
VAD the real run had, and compare its speaker_id accuracy with the REAL end-to-end pipeline on the same windows.
  python eval/calib_oracle.py [rho ...]"""
import json, glob, os, sys, collections
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, ".."))
import people, oracle, eval_columbia as ec


def run(prof_over):
    gt = collections.defaultdict(dict)
    for f in glob.glob(os.path.join(ec.LAB, "*.txt")):
        n = os.path.basename(f)[:-4]
        for l in open(f):
            d = list(map(int, l.split()[:5])); gt[d[0]][n] = d[4]
    ok = tot = 0
    for w, ws in ec.WINDOWS.items():
        log = [json.loads(l) for l in open(os.path.join(HERE, "col", f"colw{w}.ndjson"))]
        a = people.parse_args(["--oracle", "x"])
        O = oracle.Oracle(a, "/x", None); O.prof.update(prof_over); O._read_truth = lambda: None
        O.prof.update(dropouts_per_min=0.0)
        f0 = round(ws * ec.FPS)
        for o in log:
            fr = f0 + round(o["t"] * ec.FPS)
            if fr not in gt: continue
            ps = [{"id": n, "role": "presenter", "yaw_deg": i * 20.0, "dist_m": 1.5, "facing": "toward", "speaking": bool(s)}
                  for i, (n, s) in enumerate(sorted(gt[fr].items()))]
            O.frames = [(o["t"], ps)]
            O.vad_state = (o["t"], o["speech_active"], -30.0)
            out = O.step(None, o["t"])
            sp = [n for n, s in gt[fr].items() if s]
            if len(sp) == 1:
                tot += 1; ok += int(out["speaker_id"] in O.people and O.people[out["speaker_id"]].name == sp[0])
    return ok / max(1, tot)


if __name__ == "__main__":
    for prof in [dict(asd_logit_speaking=[-0.32, 0.74], asd_logit_silent=[-1.52, 0.76], asd_ar=r) for r in (0.65, 0.9)] + \
                [dict(asd_logit_speaking=[-0.31, 1.37], asd_logit_silent=[-2.29, 0.83], asd_ar=0.65)]:
        print(prof, round(run(prof), 3), flush=True)
