"""Offline replay of a live-room run through the degraded oracle (same code as live), scored vs truth.
  python eval/replay_room.py ../room/run/quiet_1 [people.py args...]
Speaker accuracy = speaker_id's truth name == the single human speaking (frames with exactly one speaker,
Kermit's loudspeaker off)."""
import json, os, sys, collections
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(HERE, ".."))
import people, oracle


class Shifted(people.FileAudio):
    def __init__(self, x, t0): super().__init__(x); self.t0 = t0
    def get(self, a, b): return super().get(a - self.t0, b - self.t0)


def run(rd, extra=()):
    T = [json.loads(l) for l in open(os.path.join(rd, "truth.ndjson")) if l.strip()]
    meta = json.load(open(os.path.join(rd, "meta.json")))
    frames = [o for o in T if "people" in o]
    kon = sorted((o["t"], o["event"] == "kermit_audio_on") for o in T if o.get("event") in ("kermit_audio_on", "kermit_audio_off"))
    a = people.parse_args(["--oracle", "x"] + list(extra))
    O = oracle.Oracle(a, "/nonexistent", a.error_profile)
    O._read_truth = lambda: None
    x = people.load_audio(os.path.join(rd, "mic.wav")); t0 = meta["t0"]
    O.audio = Shifted(x, t0)
    O.vad_segments = [(s0 + t0, s1 + t0) for s0, s1 in people.silero_segments(x)]
    ki = 0; spk_on = False
    res = collections.Counter(); ev = collections.Counter(); per_who = collections.defaultdict(collections.Counter)
    flick = collections.Counter(); log = []
    for f in frames:
        t = f["t"]
        while ki < len(kon) and kon[ki][0] <= t: spk_on = kon[ki][1]; ki += 1
        O.set_self_speaking(spk_on)
        O.frames = [(t, f["people"])]
        out = O.step(None, t)
        for e in O.events:
            ev[e["event"]] += 1
            if e["event"] == "turned_away":
                nm = O.people[e["id"]].name if e["id"] in O.people else "?"
                tr = next((g for g in f["people"] if g["id"] == nm), None)
                flick["true" if tr and tr["facing"] == "away" else "false"] += 1
                log.append((round(t - meta["t0"], 2), nm, tr and tr["facing"], e.get("while_speaking")))
        sp = [g["id"] for g in f["people"] if g.get("speaking")]
        if len(sp) == 1 and not spk_on:
            name = O.people[out["speaker_id"]].name if out["speaker_id"] in O.people else None
            res["ok" if name == sp[0] else "bad"] += 1
            per_who[sp[0]]["ok" if name == sp[0] else (name or "none")] += 1
    # truth away periods (≥ 1.5 s) for presenters → detection latency of our turned_away
    per, cur = [], {}
    for f in frames:
        for g in f["people"]:
            if g.get("role") != "presenter": continue
            if g["facing"] == "away" and g["id"] not in cur: cur[g["id"]] = f["t"]
            elif g["facing"] != "away" and g["id"] in cur:
                s0 = cur.pop(g["id"])
                if f["t"] - s0 >= 1.5: per.append((g["id"], s0, f["t"]))
    lat = []
    for nm, s0, s1 in per:
        hit = [x[0] for x in log if x[1] == nm and s0 - meta["t0"] - 0.2 <= x[0] <= s1 - meta["t0"]]
        lat.append(round(hit[0] - (s0 - meta["t0"]), 2) if hit else None)
    acc = res["ok"] / max(1, res["ok"] + res["bad"])
    return {"run": os.path.basename(rd), "speaker_acc": round(acc, 3), "n": res["ok"] + res["bad"],
            "per_truth_speaker": {k: dict(v) for k, v in per_who.items()},
            "events": dict(ev), "turned_away_vs_truth_facing": dict(flick), "turned_away_log": log,
            "presenter_away_periods": len(per), "presenter_away_latency_s": lat,
            "presenter_false_turned_away": sum(1 for x in log if x[1] in ("kirill", "steve") and x[2] != "away")}


if __name__ == "__main__":
    print(json.dumps(run(sys.argv[1], sys.argv[2:]), indent=1))
