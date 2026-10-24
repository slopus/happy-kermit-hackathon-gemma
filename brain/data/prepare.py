"""raw/*.jsonl -> train.jsonl / val.jsonl.  Validates KML, drops examples whose heard line matches an eval test."""
import glob, json, os, random, re, sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import kml

HERE = os.path.dirname(os.path.abspath(__file__))
norm = lambda s: re.sub(r"[^a-z ]", "", s.lower()).strip()
test_heard = {norm(json.loads(l)["heard"]) for l in open(os.path.join(HERE, "..", "eval", "tests.jsonl"))} - {""}

exs, dropped = [], []
for p in sorted(glob.glob(os.path.join(HERE, "raw", "*.jsonl"))):
    for l in open(p):
        if not l.strip():
            continue
        ex = json.loads(l)
        heard = [norm(m["content"].split("heard:", 1)[-1]) for m in ex["messages"] if m["role"] == "user"]
        if test_heard & set(heard):
            dropped.append(heard)
            continue
        for m in ex["messages"]:
            if m["role"] == "assistant":
                kml.parse(m["content"])
        exs.append(ex)
random.Random(0).shuffle(exs)
n_val = max(40, len(exs) // 20)
for name, part in (("val", exs[:n_val]), ("train", exs[n_val:])):
    with open(os.path.join(HERE, f"{name}.jsonl"), "w") as f:
        f.writelines(json.dumps(e) + "\n" for e in part)
print(f"{len(exs)} examples ({len(exs) - n_val} train / {n_val} val); dropped {len(dropped)} overlapping eval: {dropped}")
