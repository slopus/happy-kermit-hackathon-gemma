"""Replay people.py's fusion + speaker hysteresis on logged per-frame signals (ASD logit, face, mic, VAD) with other
weights, then score on Columbia. Used to tune --w-asd/--bias/--w-cont/--switch-* without re-running the models.
  python eval/replay_fusion.py SRC_LOGDIR [--no-mic] [key=value ...]"""
import json, os, sys, itertools, glob, shutil
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, ".."))
import people, eval_columbia


class Replay(people.People):
    def __init__(self, a):
        self.a = a; self.people = {}; self.speaker_id = None; self.challenger = (None, 0.0)
        self.last_speech_t = -1e9; self.events = []
        self.audio = None; self.vad_state = (None, False, -99.0); self.self_speaking = False


def replay(src, dst, a, no_mic=False):
    os.makedirs(dst, exist_ok=True)
    for f in glob.glob(os.path.join(src, "colw*.ndjson")):
        if "events" in f: continue
        R = Replay(a); out = open(os.path.join(dst, os.path.basename(f)), "w")
        for line in open(f):
            o = json.loads(line); t = o["t"]; ids = set()
            for q in o["people"]:
                P = R.people.get(q["id"]) or people.Person(q["id"], t); R.people[q["id"]] = P
                P.t_seen = t; P.box = q["box"]; P.dist = q["dist_m"]; P.confirmed = True; ids.add(q["id"])
                if q.get("asd") is not None and q.get("face"): P.asd_logit, P.asd_t = q["asd"], t
                P.has_mic = False if no_mic else q["has_mic"]
            for pid in [k for k in R.people if k not in ids and t - R.people[k].t_seen > a.lost_s]: del R.people[pid]
            sp = o["speech_active"]
            if sp: R.last_speech_t = t
            R._fuse(t, sp)
            for q in o["people"]: q["speaking_p"] = round(R.people[q["id"]].p, 3)
            o["speaker_id"] = R.speaker_id
            out.write(json.dumps(o) + "\n")


if __name__ == "__main__":
    src = sys.argv[1]; no_mic = "--no-mic" in sys.argv
    grid = {"w_asd": [1.5], "bias": [1.0], "asd_prior": [-1.0], "w_cont": [0.5, 1.0], "w_alone": [2.5],
            "switch_margin": [0.25, 0.5, 1.0], "switch_hold": [0.3], "p_tau": [0.25]}
    res = []
    for vals in itertools.product(*grid.values()):
        kv = dict(zip(grid, vals))
        a = people.parse_args([]); [setattr(a, k, v) for k, v in kv.items()]
        dst = os.path.join(HERE, "..", ".context", "replay"); shutil.rmtree(dst, ignore_errors=True)
        replay(src, dst, a, no_mic)
        o = eval_columbia.main(dst, None, quiet=True)
        res.append({**kv, "F1": o["mean_F1_fused"], "spk_acc": o["speaker_id_accuracy"], "switch_med": o["switch_latency_s"]["median"]})
        print(res[-1], flush=True)
    res.sort(key=lambda r: -(r["F1"] + r["spk_acc"]))
    json.dump(res, open(os.path.join(HERE, "fusion_grid.json"), "w"), indent=1)
    print("BEST", res[:5])
