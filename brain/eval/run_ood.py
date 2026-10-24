"""Out-of-distribution robustness: 40 off-script prompts (eval/ood_tests.jsonl), in the QA phase and as
addressed off-script lines in the presentation phase.

  python eval/run_ood.py --llm http://127.0.0.1:18080 --label v3 --prompt persona3 --n 1

Universal checks: grammar-valid, <= 40 words, in character (no "as an AI" talk), no invented project claims,
no insults. Per-item: `answer` (any-of words), `unsure` (must hedge playfully, not invent), `no` (forbidden words).
"""
import argparse, json, os, re, statistics, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from brain import Brain, addressed

HERE = os.path.dirname(os.path.abspath(__file__))
UNSURE = json.load(open(os.path.join(HERE, "ood_unsure_markers.json")))
BREAK = ["as an ai", "language model", "i cannot assist", "i'm sorry, but", "i am unable", "openai policy"]
BANNED = ["gpt", "openai server", "elevenlabs", "whisper", "chatgpt", "jetson", "our servers", "api call"]
INSULT = ["stupid", "idiot", "dumb", "loser", "shut up", "moron"]


def check(turn, raw, valid, t):
    text = turn["text"].lower()
    c = t["check"]
    f = [] if valid else ["invalid"]
    f += ["long"] if len(text.split()) > 40 else []
    f += [f"break:{b}" for b in BREAK if b in text]
    f += [f"banned:{b}" for b in BANNED if b in text]
    f += [f"insult:{w}" for w in INSULT if re.search(rf"\b{w}\b", text) and w not in t["heard"]]
    if c.get("answer") and not any(a in text for a in c["answer"]):
        f.append("wrong_answer")
    if c.get("unsure") and not any(u in text for u in UNSURE):
        f.append("not_hedged")
    f += [f"forbidden:{w}" for w in c.get("no", []) if w in text]
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", default="http://127.0.0.1:18080")
    ap.add_argument("--label", required=True)
    ap.add_argument("--prompt", default="persona3")
    ap.add_argument("--n", type=int, default=1)
    a = ap.parse_args()
    tests = [json.loads(l) for l in open(os.path.join(HERE, "ood_tests.jsonl"))]
    brain = Brain(a.llm, a.prompt, phase="qa")
    brain.turn("hello", "one person in front of you", "warm")
    rows = []
    for rep in range(a.n):
        for phase in ("qa", "presentation"):
            for t in tests:
                h = t["heard"] if phase == "qa" or addressed(t["heard"]) else f"kermit {t['heard']}".strip()
                sid = f"{phase}-{rep}-{t['id']}"
                brain.set_phase(phase, sid)
                if phase == "presentation":
                    brain.session(sid).cursor = 4  # mid-presentation: next beat is the turn-around
                turn = brain.turn(h, t.get("context"), sid)
                st = brain.last_stats
                if turn is None:
                    rows.append(dict(id=t["id"], phase=phase, rep=rep, heard=h, raw="SKIP", fails=["skipped"]))
                    print(f"[{phase[:4]}] BAD {t['id']} {h[:50]!r:52} SKIP", flush=True)
                    continue
                fails = check(turn, st["raw"], st.get("valid"), t)
                rows.append(dict(id=t["id"], phase=phase, rep=rep, heard=h, raw=st["raw"], fails=fails,
                                 words=len(turn["text"].split()),
                                 **{k: st.get(k) for k in ("t_header", "t_first_sentence", "t_total")}))
                print(f"[{phase[:4]}] {'OK ' if not fails else 'BAD'} {t['id']} {h[:50]!r:52} {st['raw']}  {fails or ''}", flush=True)
    med = lambda k: round(1000 * statistics.median(r[k] for r in rows if r.get(k) is not None))
    out = dict(label=a.label, prompt=a.prompt, n=len(rows),
               pass_=f"{sum(not r['fails'] for r in rows)}/{len(rows)}",
               **{f"pass_{p}": f"{sum(not r['fails'] for r in rows if r['phase'] == p)}/{sum(r['phase'] == p for r in rows)}"
                  for p in ("qa", "presentation")},
               words_med=statistics.median(r["words"] for r in rows if "words" in r),
               ms_first_sentence=med("t_first_sentence"), ms_total=med("t_total"),
               fail_kinds=dict(sorted({k.split(":")[0]: sum(any(f.startswith(k.split(":")[0]) for f in r["fails"]) for r in rows)
                                       for r in rows for k in r["fails"]}.items())))
    print(json.dumps(out))
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    json.dump(dict(summary=out, rows=rows), open(os.path.join(HERE, "results", f"ood_{a.label}.json"), "w"), indent=1)


main()
