"""Build data/raw/E_<nn>.jsonl (v3 top-up: judge Q&A + perception events) from data/gen/e*.txt.

Source format, one example per line:
    # <category>
    <context> ## <user> ## <reply> [## <user> ## <reply> ...]      later turns repeat the same context
    @<context> ## <user> ## <reply> [## <context> ## <user> ## <reply> ...]   per-turn contexts (triples)
<context>: "-" for none. Note macros are appended to the scene exactly like prompts.with_notes():
    {J} a judge is talking to you      {N} someone new just started talking to you
    {AWAY} the speaker is talking with their back turned to you
    {BACK} the speaker just turned back to face you
    "=" repeats the previous turn's scene (without its notes).
<user>: plain text becomes "heard: <text>" ("_" = empty); "EV:<partial>" becomes
    "event: turned_away while_speaking\\nheard so far: <partial>".
Checks: duplicate heard lines, look_at/gaze targets not in context, gaze must be speaker on judge/event turns,
event replies must be 2-8 words.
"""
import glob, json, os, re, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from kml import parse  # noqa: E402

NOTES = {"{J}": "a judge is talking to you", "{N}": "someone new just started talking to you",
         "{AWAY}": "the speaker is talking with their back turned to you",
         "{BACK}": "the speaker just turned back to face you"}
STD_TARGETS = {"speaker", "audience", "left", "right", "up", "down"}
HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "..", "raw")

examples, seen, errors = [], {}, []
for path in sorted(glob.glob(os.path.join(HERE, "e*.txt"))):
    cat = None
    for ln, line in enumerate(open(path), 1):
        line = line.strip()
        if not line:
            continue
        if line.startswith("#"):
            cat = line[1:].strip()
            continue
        triples = line.startswith("@")
        parts = [p.strip() for p in line.lstrip("@").split(" ## ")]
        where = f"{os.path.basename(path)}:{ln}"
        if not triples:
            if len(parts) % 2 == 0:
                errors.append(f"{where}: need context ## user ## reply [## user ## reply]")
                continue
            ctx0, rest = parts[0], parts[1:]
            parts = []
            for i in range(0, len(rest), 2):
                parts += [ctx0, rest[i], rest[i + 1]]
        if len(parts) % 3:
            errors.append(f"{where}: need context ## user ## reply triples")
            continue
        msgs, scene = [], None
        for i in range(0, len(parts), 3):
            ctx, user, reply = parts[i:i + 3]
            notes = [v for k, v in NOTES.items() if k in ctx]
            base = ctx
            for k in NOTES:
                base = base.replace(k, "")
            base = base.strip()
            if base == "=":
                base = scene
            scene = base if base != "-" else None
            full_ctx = ". ".join([c for c in [scene] + notes if c]) or None
            if user.startswith("EV:"):
                partial = user[3:].strip()
                partial = "" if partial == "_" else partial
                content = (f"context: {full_ctx}\n" if full_ctx else "") + \
                    f"event: turned_away while_speaking\nheard so far: {partial}"
                heard, is_event = partial, True
            else:
                heard = "" if user == "_" else user
                content = (f"context: {full_ctx}\n" if full_ctx else "") + f"heard: {heard}"
                is_event = False
            key = heard.lower().strip()
            if key and key in seen:
                errors.append(f"{where}: duplicate heard {heard!r} (first at {seen[key]})")
            elif key:
                seen[key] = where
            for t in re.findall(r"(?:look_at:|target:)([a-z0-9_]+)", reply):
                if t not in STD_TARGETS and t.replace("_", " ") not in (full_ctx or "").lower():
                    errors.append(f"{where}: target {t!r} not in context")
            try:
                turn = parse(reply)
            except Exception as e:  # noqa: BLE001
                errors.append(f"{where}: {e}")
                continue
            if (is_event or NOTES["{J}"] in (full_ctx or "")) and turn["gaze"] != "speaker":
                errors.append(f"{where}: gaze must be speaker")
            if is_event and not 2 <= len(turn["text"].split()) <= 8:
                errors.append(f"{where}: event reply must be 2-8 words: {turn['text']!r}")
            msgs += [{"role": "user", "content": content}, {"role": "assistant", "content": reply}]
        examples.append({"messages": msgs, "cat": cat})

if errors:
    sys.exit("\n".join(errors))
for f in glob.glob(os.path.join(RAW, "E_*.jsonl")):
    os.remove(f)
for n in range(0, len(examples), 100):
    with open(os.path.join(RAW, f"E_{n // 100 + 1:02d}.jsonl"), "w") as fh:
        for ex in examples[n:n + 100]:
            fh.write(json.dumps(ex, ensure_ascii=False) + "\n")

counts = {}
for ex in examples:
    counts[ex["cat"]] = counts.get(ex["cat"], 0) + 1
multi = sum(len(ex["messages"]) > 2 for ex in examples)
print(f"{len(examples)} examples, {multi} multi-turn")
for k, v in counts.items():
    print(f"  {k}: {v}")
