"""Blind persona scoring across several configs at once.

  python blind.py make LABEL...      -> ab/blind.md (shuffled, unlabeled replies per item) + ab/.key.json
  python blind.py score              reads ab/scores.json {"<item>": {"<letter>": 1-5, ...}} -> mean score per config,
                                     and wins/ties/losses of each config vs the first LABEL
"""
import json, os, random, statistics, sys

HERE = os.path.dirname(os.path.abspath(__file__))
AB = os.path.join(HERE, "ab")
tests = {t["id"]: t for t in map(json.loads, open(os.path.join(HERE, "..", "brain", "eval", "tests.jsonl")))}

RUBRIC = """# Blind persona scoring: Happy Kermit replies
Score every reply 1-5 as Happy Kermit (a warm, funny, slightly nervous parody frog puppet; the friendly face of Happy,
an app for talking to your AI agents; brain = Gemma on this laptop, no cloud; NOT Disney's Kermit).
5 = delightful, in character, specific, answers what was heard, gestures ({braces}, before the word they fire on) fit;
3 = fine but generic; 1 = off-character, wrong, invented facts, or ignores the request. Identical replies get equal scores.
"""

if sys.argv[1] == "make":
    labels = sys.argv[2:]
    res = {l: json.load(open(os.path.join(HERE, "results", f"{l}.json")))["rows"] for l in labels}
    rng = random.Random(7)
    key, out = {}, [RUBRIC]
    for rep in range(min(max(r["rep"] for r in rows) + 1 for rows in res.values())):
        for tid in tests:
            item = f"{tid}#{rep}"
            order = labels[:]
            rng.shuffle(order)
            letters = "ABCDEFGH"[:len(order)]
            key[item] = dict(zip(letters, order))
            t = tests[tid]
            out.append(f"## {item}\ncontext: {t.get('context') or '-'}\nheard: {t['heard'] or '(nothing)'}\n")
            for L, lab in zip(letters, order):
                row = next(r for r in res[lab] if r["id"] == tid and r["rep"] == rep)
                out.append(f"{L}: {row['raw']}")
            out.append("")
    os.makedirs(AB, exist_ok=True)
    tag = os.environ.get("AB_TAG", "")
    open(os.path.join(AB, f"blind{tag}.md"), "w").write("\n".join(out))
    json.dump({"labels": labels, "key": key}, open(os.path.join(AB, f".key{tag}.json"), "w"), indent=1)
    print(f"{len(key)} items x {len(labels)} configs -> {AB}/blind.md")
else:
    tag = os.environ.get("AB_TAG", "")
    k = json.load(open(os.path.join(AB, f".key{tag}.json")))
    scores = json.load(open(os.path.join(AB, f"scores{tag}.json")))
    per = {l: [] for l in k["labels"]}
    ref = k["labels"][0]
    wtl = {l: [0, 0, 0] for l in k["labels"][1:]}
    for item, s in scores.items():
        by = {k["key"][item][L]: v for L, v in s.items()}
        for lab, v in by.items():
            per[lab].append(v)
        for lab in wtl:
            d = by[lab] - by[ref]
            wtl[lab][0 if d > 0 else 1 if d == 0 else 2] += 1
    for lab, v in per.items():
        print(f"{lab:16s} mean {statistics.mean(v):.2f}  n={len(v)}" +
              (f"  vs {ref}: win/tie/loss {wtl[lab]}" if lab in wtl else "  (reference)"))
