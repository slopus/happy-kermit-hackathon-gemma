"""Blind A/B persona-quality comparison between two eval result files.

  python eval/ab_blind.py make  RESULT_A RESULT_B   -> eval/ab/blind.md (for the judge) + eval/ab/key.json
  python eval/ab_blind.py score [KEY.json]           reads eval/ab/verdicts.json {"<pair>": "A"|"B"|"tie"}
  (move key.json out of the judge's reach before judging)
"""
import json, os, random, sys

HERE = os.path.dirname(os.path.abspath(__file__))
AB = os.path.join(HERE, "ab")
tests = {t["id"]: t for t in map(json.loads, open(os.path.join(HERE, "tests.jsonl")))}

if sys.argv[1] == "make":
    os.makedirs(AB, exist_ok=True)
    ra, rb = (json.load(open(p)) for p in sys.argv[2:4])
    rng = random.Random(42)
    key, out = {}, ["# Blind A/B: Happy Kermit replies\n",
                    "For each pair, pick the better reply as Happy Kermit (a warm, funny, slightly nervous parody frog puppet; "
                    "the friendly face of Happy, an app for talking to your AI agents; brain = Gemma on this laptop, no cloud). "
                    "Judge: in character and charming, funny/specific, factually safe (no invented facts), answers what was "
                    "heard, gestures (in {braces}, before the word they fire on) fit the words. Answer A, B or tie.\n"]
    n = 0
    for rep in range(3):
        for x in (r for r in ra["rows"] if r["rep"] == rep):
            y = next(r for r in rb["rows"] if r["id"] == x["id"] and r["rep"] == rep)
            n += 1
            pair = f"p{n:02d}"
            flip = rng.random() < 0.5
            a, b = (y, x) if flip else (x, y)
            key[pair] = {"A": sys.argv[3] if flip else sys.argv[2], "B": sys.argv[2] if flip else sys.argv[3], "id": x["id"]}
            t = tests[x["id"]]
            out.append(f"## {pair}\ncontext: {t.get('context') or '-'}\nheard: {t['heard'] or '(nothing)'}\n\n"
                       f"A: {a['raw']}\n\nB: {b['raw']}\n")
    open(os.path.join(AB, "blind.md"), "w").write("\n".join(out))
    json.dump(key, open(os.path.join(AB, "key.json"), "w"), indent=1)
    print(f"{n} pairs -> {AB}/blind.md")
else:
    key = json.load(open(sys.argv[2] if len(sys.argv) > 2 else os.path.join(AB, "key.json")))
    ver = json.load(open(os.path.join(AB, "verdicts.json")))
    wins = {}
    for pair, v in ver.items():
        who = "tie" if v.strip().lower() == "tie" else os.path.basename(key[pair][v.strip().upper()])
        wins[who] = wins.get(who, 0) + 1
    print(json.dumps(wins))
