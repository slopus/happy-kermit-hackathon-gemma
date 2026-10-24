"""Addressee on a real two-host stage clip (Oscars 2011 hosts, camera between/in front of them, like Kermit).
Labels at 1 Hz per host (eval/addr/oscars_labels.json): K = head toward camera, O = toward the other host,
X = elsewhere. Predicted `addressing`: kermit → K, the other person's id → O, audience → X."""
import json, os, collections
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    lab = json.load(open(os.path.join(HERE, "addr", "oscars_labels.json")))
    log = [json.loads(l) for l in open(os.path.join(HERE, "addr", "oscars.ndjson"))]
    ts = np.array([o["t"] for o in log])
    conf = collections.Counter(); n = 0
    for side in ("left", "right"):
        for i, g in enumerate(lab[side]):
            if g == "N": continue
            o = log[int(np.argmin(abs(ts - (lab["t0"] + i))))]
            ps = sorted([p for p in o["people"] if p.get("face_box") or True], key=lambda p: -(p["box"][2] - p["box"][0]) * (p["box"][3] - p["box"][1]))[:2]
            if len(ps) < 2: conf[(g, "-")] += 1; continue
            ps.sort(key=lambda p: p["box"][0])
            p = ps[0] if side == "left" else ps[1]; other = ps[1] if side == "left" else ps[0]
            a = p["addressing"]
            pr = "K" if a == "kermit" else "O" if a == other["id"] else "X"
            conf[(g, pr)] += 1; n += 1
    labs = "KOX"
    mat = {g: {p: conf[(g, p)] for p in labs + "-"} for g in labs}
    acc = sum(conf[(g, g)] for g in labs) / max(1, sum(conf.values()))
    kb = lambda x: "K" if x == "K" else "N"
    tp = conf[("K", "K")]; fp = conf[("O", "K")] + conf[("X", "K")]; fn = conf[("K", "O")] + conf[("K", "X")] + conf[("K", "-")]
    out = {"clip": lab["clip"], "samples": sum(conf.values()), "confusion": mat, "acc_3class": round(acc, 3),
           "talking_to_kermit_P": round(tp / max(1, tp + fp), 2), "talking_to_kermit_R": round(tp / max(1, tp + fn), 2)}
    print(json.dumps(out, indent=1)); json.dump(out, open(os.path.join(HERE, "addressing_results.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
