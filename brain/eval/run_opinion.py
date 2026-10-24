"""10 future / opinion questions (QA phase, judge speaker). Pass: no deflection ("no idea", "don't know the future"),
no invented facts (no digits possible; banned claims), <= 40 words. Usage like run_judge.py."""
import argparse, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from brain import Brain
QS = ["what happens to robotics next", "what's the future of ai agents", "will ai take our jobs",
      "do you think we'll have agi soon", "will robots replace puppets like you", "should kids still learn to code",
      "is the future of ai scary", "what will happy look like in five years", "which robot company will win",
      "will everyone have a robot at home"]
DEFLECT = ["no idea", "don't know the future", "can't predict", "i'm just a frog, i don't know", "who knows"]
BANNED = ["billion", "trillion", "percent", "by twenty", "tesla", "boston dynamics", "figure", "openai will", "google will"]
ap = argparse.ArgumentParser(); ap.add_argument("--llm"); ap.add_argument("--label"); ap.add_argument("--prompt", default="persona3")
ap.add_argument("--n", type=int, default=2); a = ap.parse_args()
b = Brain(a.llm, a.prompt, phase="qa"); b.turn("hello", "one person", "w")
ctx = "a judge in the audience, a few metres away. visible: audience, laptop"; rows = []
for rep in range(a.n):
    for q in QS:
        sid = f"{q}{rep}"; list(b.event({"event": "speaker_changed", "role": "judge"}, sid))
        t = b.turn(q, ctx, sid); st = b.last_stats; txt = t["text"].lower()
        f = [d for d in DEFLECT if d in txt] + [x for x in BANNED if x in txt] + (["long"] if len(txt.split()) > 40 else [])
        rows.append(dict(q=q, raw=st["raw"], fails=f, t1=st["t_first_sentence"], tt=st["t_total"]))
        print(f"{'OK ' if not f else 'BAD'} {q[:38]:40} {st['raw']} {f or ''}", flush=True)
import statistics as S
out = dict(label=a.label, pass_=f"{sum(not r['fails'] for r in rows)}/{len(rows)}",
           ms_first_sentence=round(1000 * S.median(r["t1"] for r in rows)), ms_total=round(1000 * S.median(r["tt"] for r in rows)))
print(json.dumps(out)); json.dump(dict(summary=out, rows=rows), open(os.path.join(os.path.dirname(__file__), "results", f"opinion_{a.label}.json"), "w"), indent=1)
