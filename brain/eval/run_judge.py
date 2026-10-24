"""Judge Q&A + interjection eval (QA phase, via the real Brain state machine).

  python eval/run_judge.py --llm http://127.0.0.1:18080 --label v3 --prompt persona3 --n 2

Judge questions: a fresh session gets speaker_changed(role=judge), then the question. Checks: gaze == speaker,
required fact words (any-of groups), no banned claims, <= 40 words. Interjections: turned_away while_speaking
from a presenter in QA phase -> 2-8 words; printed for manual review (must react to the turning away, not answer).
"""
import argparse, json, os, re, statistics, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from brain import Brain

HERE = os.path.dirname(os.path.abspath(__file__))
CTX = "two judges at the table. visible: laptop, audience"
BANNED = ["gpt", "openai", "whisper", "elevenlabs", "chatgpt", "claude", "jetson", "raspberry", "nvidia gpu",
          "in the cloud we", "our servers", "api call", "millisecond", "percent"]
JUDGE = [  # (question, list of any-of word groups that must all be hit)
    ("so what exactly is this project", [["happy"], ["agent"]]),
    ("how does it work end to end", [["gemma"], ["parakeet", "kokoro", "camera", "speech"]]),
    ("which model are you running and why gemma", [["gemma"], ["e four b", "four", "laptop", "device", "local"]]),
    ("did you fine tune the model", [["lora", "fine-tune", "fine tune", "fine-tuned", "trained"]]),
    ("why not just use the cloud", [["private", "privacy", "lag", "offline", "internet", "rule", "requires"]]),
    ("what happens if the wifi goes down", [["offline", "still", "keep", "no internet", "nothing", "fine", "laptop"]]),
    ("what was the hardest part of building this", [["character", "gesture", "latency", "sync", "word", "fast", "funny"]]),
    ("how fast do you respond", [["second", "sentence"]]),
    ("what's next for happy kermit", [["app", "gesture", "who", "agents"]]),
    ("is this really running on that laptop", [["laptop", "mac", "here"]]),
]
PARTIALS = ["so the way the agents talk to each other is", "and then we", "", "the hand has six", "anyway what i was saying"]


def run(brain, n):
    rows = []
    for rep in range(n):
        for q, groups in JUDGE:
            sid = f"j{rep}-{q}"
            list(brain.event({"event": "speaker_changed", "from": 2, "to": 1, "role": "judge"}, sid))
            turn = brain.turn(q, CTX, sid)
            st = brain.last_stats
            text = turn["text"].lower()
            fails = [f"missing {g}" for g in groups if not any(w in text for w in g)]
            fails += [f"banned {b}" for b in BANNED if b in text]
            fails += ["gaze"] if turn["gaze"] != "speaker" else []
            fails += ["long"] if len(text.split()) > 40 else []
            rows.append(dict(kind="judge", q=q, rep=rep, raw=st["raw"], fails=fails, valid=st.get("valid"),
                             **{k: st.get(k) for k in ("t_header", "t_first_sentence", "t_total")}))
            print(f"[judge] {'OK ' if not fails else 'BAD'} {q!r:48} {st['raw']}  {fails or ''}", flush=True)
        for p in PARTIALS:
            sid = f"i{rep}-{p}"
            evs = list(brain.event({"event": "turned_away", "id": 2, "while_speaking": True, "role": "presenter",
                                    "heard_partial": p, "context": "one person in front of you"}, sid))
            turn = next((e for e in evs if e["event"] == "turn"), None)
            st = brain.last_stats
            nw = len(turn["text"].split()) if turn else 0
            rows.append(dict(kind="interject", q=p, rep=rep, raw=st.get("raw"), fails=[] if 2 <= nw <= 8 else ["len"],
                             valid=st.get("valid"), **{k: st.get(k) for k in ("t_header", "t_first_sentence", "t_total")}))
            print(f"[inter] {p!r:48} {st.get('raw')}", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", default="http://127.0.0.1:18080")
    ap.add_argument("--label", required=True)
    ap.add_argument("--prompt", default="persona3")
    ap.add_argument("--n", type=int, default=2)
    a = ap.parse_args()
    brain = Brain(a.llm, a.prompt, phase="qa")
    brain.turn("hello", "one person in front of you", "warm")
    rows = run(brain, a.n)
    med = lambda rs, k: round(1000 * statistics.median(r[k] for r in rs if r.get(k) is not None))
    out = {"label": a.label, "prompt": a.prompt}
    for kind in ("judge", "interject"):
        rs = [r for r in rows if r["kind"] == kind]
        out[kind] = dict(n=len(rs), pass_=f"{sum(not r['fails'] for r in rs)}/{len(rs)}",
                         valid=f"{sum(bool(r['valid']) for r in rs)}/{len(rs)}",
                         ms_header=med(rs, "t_header"), ms_first_sentence=med(rs, "t_first_sentence"),
                         ms_total=med(rs, "t_total"))
    print(json.dumps(out))
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    json.dump(dict(summary=out, rows=rows), open(os.path.join(HERE, "results", f"judge_{a.label}.json"), "w"), indent=1)


main()
