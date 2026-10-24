"""Score a brain config on eval/tests.jsonl.

  python eval/run_eval.py --llm http://127.0.0.1:18080 --label E4B_Q4_full --mode kml --prompt full --n 3
  --mode json : naive baseline, model writes CONTRACT JSON (word indices) itself under json_schema
"""
import argparse, json, os, re, statistics, sys, time, urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import kml, prompts
from brain import Brain

HERE = os.path.dirname(__file__)
YES = {"yes", "yeah", "yep", "sure", "right", "exactly", "absolutely", "totally", "agree", "agreed", "true", "okay",
       "ok", "definitely", "course", "indeed", "uh-huh", "mhm", "correct", "deal", "i", "you"}
NO = {"no", "nope", "nah", "not", "never", "don't", "can't", "won't", "nothing", "zero", "none", "nuh-uh", "no,", "isn't"}
WOW = {"whoa", "wow", "oh", "what", "huh", "yikes", "gosh", "golly", "really", "wait", "holy", "woah", "eek", "no way"}
DEIXIS = {"this", "that", "these", "those", "here", "there", "you", "everyone", "everybody", "folks", "judges",
          "left", "right", "up", "down", "around", "y'all", "all"}
BASE_TARGETS = {"speaker", "audience", "left", "right", "up", "down"}
SEMANTIC = {"nod": YES, "shake": NO, "surprise": WOW}

JSON_SCHEMA = {"type": "object", "required": ["text", "emotion", "gestures", "gaze"], "properties": {
    "text": {"type": "string"},
    "emotion": {"type": "string", "enum": kml.EMOTIONS},
    "gestures": {"type": "array", "maxItems": 5, "items": {"type": "object", "required": ["word", "type", "intensity"],
                 "properties": {"word": {"type": "integer"}, "type": {"type": "string", "enum": kml.GESTURES},
                                "target": {"type": "string"}, "intensity": {"type": "number"}}}},
    "gaze": {"type": "string"}}}

JSON_RULES = prompts.FORMAT_RULES.split("Reply with exactly ONE line:")[0] + """Reply with ONLY a JSON object:
{"text": "<spoken reply>", "emotion": "<emotion>", "gestures": [{"word": <0-based index of the word in text split on spaces>, "type": "<gesture>", "intensity": <0-1>, "target": "<only for look_at>"}], "gaze": "<gaze>"}
- emotion: """ + ", ".join(kml.EMOTIONS) + """
- gaze: speaker, audience, down, up, or target:<thing from context>
- gesture types: """ + ", ".join(kml.GESTURES) + """
- Use 1 to 4 gestures; the word index is the word the gesture fires on. nod = yes, shake = no, surprise = whoa/wow, head_bob = greetings, look_at/look_around = referring to things, laugh = jokes, sigh = weary, shrug = not sure, flail = big excitement, cover_face = embarrassed, lean_in = questions to the person, tilt = puzzled.
- If asked to do a gesture, it must be the first gesture, on word 0.
- Text: 1 to 3 short sentences, at most 35 words, plain spoken English, numbers spelled out, no emoji."""


def norm(w):
    return re.sub(r"[^\w'-]", "", w.lower())


def context_targets(ctx):
    m = re.search(r"visible:\s*([^.]*)", ctx or "")
    names = {t.strip().replace(" ", "_") for t in m.group(1).split(",")} if m else set()
    words = {norm(w) for w in (ctx or "").split()}  # "a kid in front of you" -> look_at:kid is fine
    return BASE_TARGETS | names | {n.split("_")[-1] for n in names} | words


def placement_ok(turn, g, ctx):
    words = turn["text"].split()
    i = g["word"]
    if not 0 <= i < len(words):
        return False
    window = [norm(w) for w in words[max(0, i - 1):i + 2]]
    boundary = i == 0 or words[i - 1][-1] in ".,!?;:"
    typ = g["type"]
    if typ in SEMANTIC:
        bang = typ == "surprise" and any(w.endswith("!") for w in words[i:i + 2])
        return bool(set(window) & SEMANTIC[typ]) or bang or boundary
    if typ == "look_at":
        tgt = g.get("target", "speaker")
        if tgt not in context_targets(ctx):
            return False
        near = {norm(w) for w in words[max(0, i - 2):i + 3]}
        return boundary or bool(near & (DEIXIS | set(tgt.split("_"))))
    return boundary


def incoherent(turn):
    """emotion/gesture clashes VOICE flagged: flail when not excited, sigh when excited, laugh when sad"""
    ts = [g["type"] for g in turn["gestures"]]
    e = turn["emotion"]
    return (("flail" in ts and e not in ("excited", "surprised", "happy")) or ("sigh" in ts and e == "excited")
            or ("laugh" in ts and e == "sad"))


def check_expect(turn, exp):
    fails = []
    gs = turn["gestures"]
    text = turn["text"].lower()
    if "any_gesture" in exp and not any(g["type"] in exp["any_gesture"] for g in gs):
        fails.append("any_gesture")
    if "first_gesture" in exp:
        ok = gs and gs[0]["type"] == exp["first_gesture"] and gs[0]["word"] <= 2
        if ok and "first_target_in" in exp:
            ok = gs[0].get("target") in exp["first_target_in"]
        if not ok:
            fails.append("first_gesture")
    if "gaze_in" in exp and turn["gaze"] not in exp["gaze_in"] and not any(
            g["type"] == "look_at" and g.get("target") in {"audience", "judges"} for g in gs):
        fails.append("gaze")
    if "emotion_in" in exp and turn["emotion"] not in exp["emotion_in"]:
        fails.append("emotion")
    if "mention" in exp and not any(m in text for m in exp["mention"]):
        fails.append("mention")
    if "mention_all" in exp and not all(m in text for m in exp["mention_all"]):
        fails.append("mention_all")
    if any(f in text for f in exp.get("forbid", [])):
        fails.append("forbid")
    return fails


def json_call(url, heard, ctx, temp):
    sys_ = prompts.PERSONA + "\n\n" + JSON_RULES
    msgs = [{"role": "system", "content": sys_}]
    for c, h, r in prompts.FEWSHOT:
        msgs += [{"role": "user", "content": prompts.user_msg(h, c or None)},
                 {"role": "assistant", "content": json.dumps(kml.parse(r))}]
    msgs.append({"role": "user", "content": prompts.user_msg(heard, ctx)})
    body = {"messages": msgs, "max_tokens": 300, "temperature": temp, "top_p": 0.95, "stream": True,
            "cache_prompt": True, "chat_template_kwargs": {"enable_thinking": False},
            "response_format": {"type": "json_schema", "json_schema": {"schema": JSON_SCHEMA}}}
    t0 = time.perf_counter(); first = None; out = ""; timings = {}
    req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            ev = json.loads(line[6:])
            timings = ev.get("timings", timings)
            for ch in ev.get("choices", []):
                p = (ch.get("delta") or {}).get("content") or ""
                if p and first is None:
                    first = time.perf_counter() - t0
                out += p
    total = time.perf_counter() - t0
    try:
        turn = json.loads(out)
        n = len(turn["text"].split())
        valid = all(0 <= g["word"] < n for g in turn["gestures"]) and n > 0
    except Exception:
        turn, valid = None, False
    # JSON can't be consumed before the text closes and gestures come after it: first useful = total
    return turn, out, valid, dict(t_first_token=first, t_header=total, t_first_sentence=total, t_total=total,
                                  server=timings)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", default="http://127.0.0.1:18080")
    ap.add_argument("--label", required=True)
    ap.add_argument("--mode", choices=["kml", "json"], default="kml")
    ap.add_argument("--prompt", choices=list(prompts.SYSTEMS), default="full")
    ap.add_argument("--no-grammar", action="store_true")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--temp", type=float, default=0.7)
    a = ap.parse_args()
    tests = [json.loads(l) for l in open(os.path.join(HERE, "tests.jsonl")) if l.strip()]
    brain = Brain(a.llm, a.prompt, not a.no_grammar, temperature=a.temp)
    # warm-up (caches the system prompt prefix, as in the live demo)
    list(brain.stream("hello", "one person in front of you", session_id="warm"))
    rows = []
    for rep in range(a.n):
        for t in tests:
            ctx = t.get("context")
            if a.mode == "kml":
                turn = None
                for ev in brain.stream(t["heard"], ctx, session_id=f"{t['id']}-{rep}"):
                    if ev["event"] == "turn":
                        turn = {k: v for k, v in ev.items() if k != "event"}
                st = dict(brain.last_stats); raw = st.pop("raw"); valid = st.pop("valid")
            else:
                turn, raw, valid, st = json_call(a.llm, t["heard"], ctx, a.temp)
            r = dict(id=t["id"], rep=rep, raw=raw, valid=valid, turn=turn,
                     **{k: st.get(k) for k in ("t_first_token", "t_header", "t_first_sentence", "t_total")},
                     n_gen=st.get("server", {}).get("predicted_n"), n_prompt=st.get("server", {}).get("prompt_n"))
            if turn and valid:
                r["lint"] = kml.lint(turn)
                r["fails"] = check_expect(turn, t["expect"])
                r["placement"] = [placement_ok(turn, g, ctx) for g in turn["gestures"]]
                r["incoherent"] = incoherent(turn)
                r["unspeakable"] = bool(re.search(r"[^A-Za-z\u00C0-\u00FF'\u2019.,!?;:\"() -]", turn["text"]))
                r["first_sentence_words"] = len(re.split(r"(?<=[.!?])\s", turn["text"])[0].split())
            rows.append(r)
            print(f"[{a.label}] {t['id']} {'OK ' if valid else 'BAD'} {r.get('fails')} {raw!r}", flush=True)
    ok = [r for r in rows if r["valid"]]
    med = lambda k: statistics.median(r[k] for r in rows if r.get(k) is not None)
    p90 = lambda k: sorted(r[k] for r in rows if r.get(k) is not None)[int(0.9 * len(rows)) - 1]
    gs = [p for r in ok for p in r["placement"]]
    summary = dict(
        label=a.label, mode=a.mode, prompt=a.prompt, n=len(rows),
        valid=f"{len(ok)}/{len(rows)}",
        lint_clean=f"{sum(not r['lint'] for r in ok)}/{len(ok)}",
        expect_pass=f"{sum(not r['fails'] for r in ok)}/{len(ok)}",
        placement_ok=f"{sum(gs)}/{len(gs)}",
        emotion_gesture_clash=f"{sum(r['incoherent'] for r in ok)}/{len(ok)}",
        unspeakable_chars=f"{sum(r['unspeakable'] for r in ok)}/{len(ok)}",
        gestures_per_turn=round(len(gs) / max(len(ok), 1), 2),
        first_sentence_words_med=statistics.median(r["first_sentence_words"] for r in ok) if ok else None,
        ms_first_token_med=round(1000 * med("t_first_token")), ms_header_med=round(1000 * med("t_header")),
        ms_first_sentence_med=round(1000 * med("t_first_sentence")), ms_first_sentence_p90=round(1000 * p90("t_first_sentence")),
        ms_total_med=round(1000 * med("t_total")), gen_tokens_med=med("n_gen"), prompt_tokens_last=rows[-1]["n_prompt"])
    print(json.dumps(summary))
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    json.dump(dict(summary=summary, rows=rows), open(os.path.join(HERE, "results", f"{a.label}.json"), "w"), indent=1)


main()
