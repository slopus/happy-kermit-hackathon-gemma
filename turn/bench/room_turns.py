"""Turn-taking metrics for ROOM SIM runs: split the room latency into endpointing (last word → turn_end) and
pipeline (turn_end → first reply sample in the room), and count commits made while the human was mid-turn.

usage: room_turns.py RUN_DIR [RUN_DIR ...]
"""
import json, os, sys
import numpy as np


def load(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def run(d):
    truth = [e for e in load(os.path.join(d, "truth.ndjson")) if "event" in e]
    k = load(os.path.join(d, "kermit.ndjson"))
    utts = []
    for e in truth:
        if e["event"] == "utt_on": utts.append({"on": e["t"], "words": e.get("words", []), "who": e.get("who"), "step": e.get("step")})
        if e["event"] == "utt_off" and utts and "off" not in utts[-1]: utts[-1].update(off=e["t"], cut=e.get("cut", False))
    kon = [e["t"] for e in truth if e["event"] == "kermit_audio_on"]
    ends = [e for e in k if e.get("event") == "turn_end"]
    rows, false_commits = [], 0
    for u in utts:
        if "off" not in u: continue
        last = u["words"][-1]["t1"] if u["words"] else u["off"]
        # commits during the utterance (after the first word, before the last word ended) = cut into a pause
        mid = [e for e in ends if u["on"] + 0.3 < e["t"] < last - 0.05]
        false_commits += len(mid)
        nxt = [e for e in ends if last - 0.05 <= e["t"] <= last + 6]
        te = nxt[0]["t"] if nxt else None
        ko = [t for t in kon if t > last - 0.2 and t < last + 8]
        rows.append(dict(step=u["step"], who=u["who"], endpoint_s=round(te - last, 2) if te else None,
                         by=nxt[0].get("by") if nxt else None, pipeline_s=round(ko[0] - te, 2) if (te and ko and ko[0] > te) else None,
                         room_latency_s=round(ko[0] - last, 2) if ko else None, mid_commits=len(mid),
                         asr_behind_ms=nxt[0].get("asr_behind_ms") if nxt else None))
    ep = [r["endpoint_s"] for r in rows if r["endpoint_s"] is not None]
    pl = [r["pipeline_s"] for r in rows if r["pipeline_s"] is not None]
    bi = [e for e in k if e.get("event") == "bargein"]
    summ = dict(run=os.path.basename(d.rstrip("/")), utts=len(rows), committed=len(ep),
                endpoint_mean=round(float(np.mean(ep)), 2) if ep else None, endpoint_p95=round(float(np.percentile(ep, 95)), 2) if ep else None,
                pipeline_mean=round(float(np.mean(pl)), 2) if pl else None, timeouts=sum(r["by"] == "timeout" for r in rows),
                mid_turn_commits=false_commits, bargeins=len(bi), stalls=sum(e.get("event") == "turn_stall" for e in k),
                cancels=sum(e.get("event") == "turn_cancel" for e in k))
    return summ, rows


if __name__ == "__main__":
    for d in sys.argv[1:]:
        s, rows = run(d)
        print(json.dumps(s))
        if "-v" in os.environ.get("V", "-v"):
            for r in rows: print("   ", r)
