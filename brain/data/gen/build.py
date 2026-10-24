"""Build data/raw/D_<nn>.jsonl from the hand-written sources in data/gen/b*.txt.

Source format, one example per line:
    # <category>                         sets the category for the following lines
    <context> ## <heard> ## <reply> [## <heard> ## <reply> ...]
<context> "-" omits the context line; <heard> "_" means nothing was heard.
Output: 100 examples per file, in source order. Fails on duplicate heard lines and on look_at/gaze
targets that are neither standard nor named in the context.
"""
import glob, json, os, re, sys

STD_TARGETS = {"speaker", "audience", "left", "right", "up", "down"}
HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "..", "raw")

examples, seen, errors = [], {}, []
for path in sorted(glob.glob(os.path.join(HERE, "b*.txt"))):
    cat = None
    for ln, line in enumerate(open(path), 1):
        line = line.strip()
        if not line:
            continue
        if line.startswith("#"):
            cat = line[1:].strip()
            continue
        parts = [p.strip() for p in line.split(" ## ")]
        ctx, rest = parts[0], parts[1:]
        if not rest or len(rest) % 2:
            errors.append(f"{path}:{ln}: need context ## heard ## reply pairs")
            continue
        for t in re.findall(r"(?:look_at:|target:)([a-z0-9_]+)", " ".join(rest[1::2])):
            if t not in STD_TARGETS and t.replace("_", " ") not in ctx.lower():
                errors.append(f"{path}:{ln}: look_at target {t!r} not in context")
        msgs = []
        for i in range(0, len(rest), 2):
            heard = "" if rest[i] == "_" else rest[i]
            key = heard.lower().strip()
            if key and key in seen:
                errors.append(f"{path}:{ln}: duplicate heard {heard!r} (first at {seen[key]})")
            if key:
                seen[key] = f"{os.path.basename(path)}:{ln}"
            user = (f"context: {ctx}\n" if ctx != "-" and i == 0 else "") + f"heard: {heard}"
            msgs += [{"role": "user", "content": user}, {"role": "assistant", "content": rest[i + 1]}]
        examples.append({"messages": msgs, "cat": cat})

if errors:
    sys.exit("\n".join(errors))
for f in glob.glob(os.path.join(RAW, "D_*.jsonl")):
    os.remove(f)
for n in range(0, len(examples), 100):
    with open(os.path.join(RAW, f"D_{n // 100 + 1:02d}.jsonl"), "w") as fh:
        for ex in examples[n:n + 100]:
            fh.write(json.dumps(ex, ensure_ascii=False) + "\n")

counts = {}
for ex in examples:
    counts[ex["cat"]] = counts.get(ex["cat"], 0) + 1
multi = sum(len(ex["messages"]) > 2 for ex in examples)
print(f"{len(examples)} examples, {multi} multi-turn ({100 * multi / max(1, len(examples)):.0f}%)")
for k, v in counts.items():
    print(f"  {k}: {v}")
turns = [m["content"] for ex in examples for m in ex["messages"] if m["role"] == "assistant"]
prio = sum(bool(re.search(r"\{(nod|surprise|look_at|look_around)[ :]", t)) for t in turns)
side = sum("left" in ex["messages"][0]["content"].split("\n")[0] or "right" in ex["messages"][0]["content"].split("\n")[0]
           for ex in examples if ex["messages"][0]["content"].startswith("context:"))
print(f"priority-gesture turns {prio}/{len(turns)} ({100 * prio / len(turns):.0f}%), side-position contexts {side}")
