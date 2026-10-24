"""Facing on a real lecture (hand labels at 1 Hz, eval/facing/lect_shm_labels.json) vs people.py's log.
Frame-level confusion (T/S/A), binary 'away' precision/recall, and turned_away event P/R + latency vs label onsets."""
import json, os, collections
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
M = {"toward": "T", "side": "S", "away": "A"}


def main(name="lect_shm"):
    lab = json.load(open(os.path.join(HERE, "facing", f"{name}_labels.json")))
    L = lab["labels"]; step = lab["step_s"]
    log = [json.loads(l) for l in open(os.path.join(HERE, "facing", f"{name}.ndjson"))]
    ts = np.array([o["t"] for o in log])
    conf = collections.Counter(); pred = []
    for i, g in enumerate(L):
        o = log[int(np.argmin(abs(ts - i * step)))]
        if not o["people"]: p = "-"
        else:
            q = max(o["people"], key=lambda q: (q["box"][2] - q["box"][0]) * (q["box"][3] - q["box"][1]))
            p = M[q["facing"]]
        conf[(g, p)] += 1; pred.append(p)
    labs = "TSA"
    mat = {g: {p: conf[(g, p)] for p in labs + "-"} for g in labs}
    y = np.array([g == "A" for g in L]); yp = np.array([p == "A" for p in pred])
    tp, fp, fn = int((y & yp).sum()), int((~y & yp).sum()), int((y & ~yp).sum())
    # 3-class accuracy and "A vs not" for the error profile
    acc = sum(conf[(g, g)] for g in labs) / len(L)
    # events: GT onsets = runs of ≥2 consecutive A seconds (a real turn, not a glance)
    runs, i = [], 0
    while i < len(L):
        if L[i] == "A":
            j = i
            while j < len(L) and L[j] == "A": j += 1
            if j - i >= 2: runs.append((i * step, j * step))
            i = j
        else: i += 1
    ev = [json.loads(l) for l in open(os.path.join(HERE, "facing", f"{name}.events.ndjson"))]
    aw = [e["t"] for e in ev if e["event"] == "turned_away"]
    hit, lat = 0, []
    for s0, s1 in runs:
        m = [t for t in aw if s0 - 1.0 <= t <= s1]
        if m: hit += 1; lat.append(round(m[0] - s0, 2))
    matched = sum(1 for t in aw if any(s0 - 1.0 <= t <= s1 for s0, s1 in runs))
    # the stage semantic ("turned away from Kermit"): runs of ≥ 2 s that are NOT toward (side or away)
    runs2, i = [], 0
    while i < len(L):
        if L[i] != "T":
            j = i
            while j < len(L) and L[j] != "T": j += 1
            if j - i >= 2: runs2.append((i * step, j * step))
            i = j
        else: i += 1
    hit2 = sum(1 for s0, s1 in runs2 if any(s0 - 1.0 <= t <= s1 for t in aw))
    prec2 = sum(1 for t in aw if any(s0 - 1.0 <= t <= s1 for s0, s1 in runs2)) / max(1, len(aw))
    row_norm = {g: {p: round(mat[g][p] / max(1, sum(mat[g].values())), 2) for p in labs} for g in labs}
    out = {"clip": lab["clip"], "seconds": len(L), "confusion_counts": mat, "confusion_rownorm": row_norm,
           "acc_3class": round(acc, 3), "away_frame_P": round(tp / max(1, tp + fp), 2), "away_frame_R": round(tp / max(1, tp + fn), 2),
           "turn_runs_ge2s": len(runs), "turned_away_events": len(aw), "event_recall": round(hit / max(1, len(runs)), 2),
           "event_precision": round(matched / max(1, len(aw)), 2), "event_latency_s": lat,
           "not_toward_runs_ge2s": len(runs2), "event_recall_not_toward": round(hit2 / max(1, len(runs2)), 2),
           "event_precision_not_toward": round(prec2, 2),
           "note": "latency is vs the 1-Hz label onset (±0.5 s resolution); negative = event before the labelled second"}
    print(json.dumps(out, indent=1)); json.dump(out, open(os.path.join(HERE, f"facing_results_{name}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
