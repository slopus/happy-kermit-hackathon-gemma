"""Model-size sweep + mixed-vs-split benchmark on brain/eval/tests.jsonl (same scoring as brain/eval/run_eval.py).

  python run_sweep.py --llm URL --label L --prompt full --n 3                       mixed: one KML pass (production)
  python run_sweep.py --llm URL --label L --mode split --annot URL2 --n 3            split: text pass, then a
        per-sentence gesture-annotation pass (pipelined: sentence k is annotated while the text model writes k+1).
        The annotator's grammar forces the exact words of the sentence, so it can only add tags.

Uses the frozen brain code in snap/ (the live brain/ is being edited). Results -> results/<label>.json
"""
import argparse, json, os, re, statistics, sys, threading, time, urllib.request, queue

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "snap"))
import kml, prompts
from brain import Brain

EVAL = os.path.join(HERE, "..", "brain", "eval")
sys.argv_saved = sys.argv
sys.argv = ["run_eval"]  # run_eval.py parses args in main(); we only want its helpers
_src = open(os.path.join(EVAL, "run_eval.py")).read().replace("\nmain()\n", "\n")
_ns = {"__file__": os.path.join(EVAL, "run_eval.py"), "__name__": "run_eval_helpers"}
exec(compile(_src, "run_eval.py", "exec"), _ns)
sys.argv = sys.argv_saved
check_expect, placement_ok, incoherent = _ns["check_expect"], _ns["placement_ok"], _ns["incoherent"]

TAG_RE = re.compile(r"\{[^}]*\}\s*")

# ---- split mode: pass 1 (text only) -------------------------------------------------------------------------
TEXT_RULES = prompts.FORMAT_RULES.split("- gesture tag:")[0] + \
    "- Write only the spoken words after the header: no gesture tags, no braces.\n" + \
    "- Text: 1 to 3 short sentences, at most 35 words, plain spoken English, numbers spelled out, no emoji, no asterisks, no stage directions.\n" + \
    "- When natural, open with a very short first sentence like \"Whoa!\" or \"Oh, hello!\" so speech can start fast."
TEXT_SYSTEM = prompts.PERSONA + "\n\n" + TEXT_RULES
TEXT_FEWSHOT = [(c, h, TAG_RE.sub("", r)) for c, h, r in prompts.FEWSHOT]


def text_gbnf(min_words=6, max_words=45):
    alt = lambda xs: " | ".join(f'"{x}"' for x in xs)
    return f"""root ::= "[" emotion "|" gaze "] " word (" " word){{{min_words - 1},{max_words - 1}}}
emotion ::= {alt(kml.EMOTIONS)}
gaze ::= {alt(kml.GAZES)} | "target:" [a-z0-9_]{{1,24}}
word ::= [A-Za-z\u00C0-\u00FF'\u2019.,!?;:"()-]{{1,24}}
"""


# ---- split mode: pass 2 (gesture annotation) ----------------------------------------------------------------
ANNOT_SYSTEM = """You animate Happy Kermit, a frog hand-puppet on a robot hand. Kermit has already decided what to say; you add his body language.
You get the context (what the camera sees), what was heard, his emotion, the reply so far (already annotated) and the NEXT sentence. Repeat the next sentence word for word, inserting gesture tags.
- gesture tag: {<type> <intensity>} placed IMMEDIATELY BEFORE the word it fires on; intensity 0.1-1.0. For look_at write {look_at:<target> <intensity>} with target speaker, audience, left, right, up, down, or a thing from context.
- gesture types: """ + ", ".join(kml.GESTURES) + """
- About one tag per sentence, 1 to 4 per reply. nod = yes/agreeing, shake = no, surprise = whoa/wow/shock, head_bob = greetings and cheerful beats, look_at/look_around = referring to things or people, laugh = jokes, sigh = weary or sad, shrug = not sure, flail = big excitement, cover_face = embarrassed, lean_in = secrets or questions to the person, tilt_left/tilt_right = curious or puzzled.
- If the person asked for a gesture, the first sentence must open with it."""


def split_sentences(kml_body):
    """'{a 0.5} Hi! There.' -> ['{a 0.5} Hi!', 'There.'] (same boundary rule as kml.stream_segments)"""
    out, buf = [], kml_body.strip() + " "
    while True:
        m = re.search(r"[.!?]+[\"')]*\s", buf)
        if not m:
            break
        out.append(buf[:m.end()].strip())
        buf = buf[m.end():]
    if buf.strip():
        out.append(buf.strip())
    return out


def annot_user(context, heard, emotion, so_far, sentence):
    return (f"context: {context}\n" if context else "") + f"heard: {heard}\nemotion: {emotion}\n" + \
        f"so far: {so_far or '(start of reply)'}\nnext sentence: {sentence}"


def annot_fewshot():
    msgs = []
    for c, h, r in prompts.FEWSHOT:
        m = kml.HEADER.match(r)
        sents = split_sentences(r[m.end():])
        for i, s in enumerate(sents):
            msgs += [{"role": "user", "content": annot_user(c or None, h, m.group(1), " ".join(sents[:i]), TAG_RE.sub("", s))},
                     {"role": "assistant", "content": s}]
    return msgs


ANNOT_FEWSHOT = annot_fewshot()


def gbnf_lit(s):
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def annot_gbnf(words, targets, first_tag=None):
    """Exactly these words, an optional tag before each (forced first tag for stage commands)."""
    alt = lambda xs: " | ".join(f'"{x}"' for x in xs)
    plain = [g for g in kml.GESTURES if g != "look_at"]
    first = f'"{{{first_tag} " inten "}} "' if first_tag else "(tag \" \")?"
    body = " ".join([first + " " + gbnf_lit(words[0])] + [f'" " (tag " ")? {gbnf_lit(w)}' for w in words[1:]])
    return f"""root ::= {body}
tag ::= "{{" ( ({alt(plain)}) | "look_at:" name ) " " inten "}}"
name ::= {alt(targets)}
inten ::= "0." [1-9] | "1.0"
"""


def _req(url, msgs, grammar, max_tokens, temperature, stream):
    body = {"messages": msgs, "max_tokens": max_tokens, "temperature": temperature, "top_p": 0.95, "stream": stream,
            "cache_prompt": True, "chat_template_kwargs": {"enable_thinking": False}, "grammar": grammar}
    return urllib.request.Request(url.rstrip("/") + "/v1/chat/completions", json.dumps(body).encode(),
                                  {"Content-Type": "application/json"})


def chat_once(url, msgs, grammar, max_tokens, temperature):
    with urllib.request.urlopen(_req(url, msgs, grammar, max_tokens, temperature, False), timeout=120) as r:
        d = json.loads(r.read())
    return d["choices"][0]["message"]["content"], d.get("timings", {})


def chat_stream(url, msgs, grammar, max_tokens, temperature):
    """Yields text deltas, then a final {"timings": ...}."""
    with urllib.request.urlopen(_req(url, msgs, grammar, max_tokens, temperature, True), timeout=120) as r:
        timings = {}
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            ev = json.loads(line[6:])
            timings = ev.get("timings", timings)
            for ch in ev.get("choices", []):
                p = (ch.get("delta") or {}).get("content")
                if p:
                    yield p
        yield {"timings": timings}


def split_turn(a, heard, ctx):
    """Pass 1 streams header + text; each finished sentence goes to the annotator thread at once."""
    msgs = [{"role": "system", "content": TEXT_SYSTEM}]
    for c, h, r in TEXT_FEWSHOT:
        msgs += [{"role": "user", "content": prompts.user_msg(h, c or None)}, {"role": "assistant", "content": r}]
    msgs.append({"role": "user", "content": prompts.user_msg(heard, ctx)})
    targets = kml.targets_from_context(ctx)
    forced = kml.forced_gesture(heard)
    st = {"t_first_token": None, "t_header": None, "t_text_first_sentence": None, "t_first_sentence": None,
          "annot_ms": [], "annot_tps": []}
    q, done = queue.Queue(), []
    emotion = {"v": "neutral"}
    t0 = time.perf_counter()

    def annotator():
        so_far = []
        while True:
            item = q.get()
            if item is None:
                return
            i, sent = item
            words = sent.split()
            ta = time.perf_counter()
            m = [{"role": "system", "content": ANNOT_SYSTEM}] + ANNOT_FEWSHOT + [
                {"role": "user", "content": annot_user(ctx, heard, emotion["v"], " ".join(so_far), sent)}]
            out, tm = chat_once(a.annot, m, annot_gbnf(words, targets, forced if i == 0 else None),
                                16 * len(words) + 40, a.temp)
            out = out.strip()
            so_far.append(out)
            st["annot_ms"].append(round(1000 * (time.perf_counter() - ta)))
            if tm.get("predicted_per_second"):
                st["annot_tps"].append(tm["predicted_per_second"])
            if st["t_first_sentence"] is None:
                st["t_first_sentence"] = time.perf_counter() - t0
            done.append(out)

    th = threading.Thread(target=annotator)
    th.start()
    raw, buf, header, n_sent, timings = "", "", None, 0, {}
    for p in chat_stream(a.llm, msgs, text_gbnf(), a.max_tokens, a.temp):
        if isinstance(p, dict):
            timings = p["timings"]
            continue
        if st["t_first_token"] is None:
            st["t_first_token"] = time.perf_counter() - t0
        raw += p
        if header is None:
            m = kml.HEADER.match(raw)
            if not m:
                continue
            header = m
            emotion["v"] = m.group(1)
            st["t_header"] = time.perf_counter() - t0
            buf = raw[m.end():]
        else:
            buf += p
        while True:
            m = re.search(r"[.!?]+[\"')]*\s", buf)
            if not m:
                break
            sent, buf = buf[:m.end()].strip(), buf[m.end():]
            if sent:
                if st["t_text_first_sentence"] is None:
                    st["t_text_first_sentence"] = time.perf_counter() - t0
                q.put((n_sent, sent)); n_sent += 1
    if buf.strip():
        if st["t_text_first_sentence"] is None:
            st["t_text_first_sentence"] = time.perf_counter() - t0
        q.put((n_sent, buf.strip())); n_sent += 1
    st["t_text_total"] = time.perf_counter() - t0
    q.put(None)
    th.join()
    st["t_total"] = time.perf_counter() - t0
    st["server"] = timings
    line = (raw[:header.end()].strip() + " " if header else "[neutral|speaker] ") + " ".join(done)
    return line, st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", default="http://127.0.0.1:18080")
    ap.add_argument("--annot", help="annotator server for --mode split (default: --llm)")
    ap.add_argument("--label", required=True)
    ap.add_argument("--mode", choices=["kml", "split"], default="kml")
    ap.add_argument("--prompt", choices=list(prompts.SYSTEMS), default="full")
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--max-tokens", type=int, default=120)
    a = ap.parse_args()
    a.annot = a.annot or a.llm
    tests = [json.loads(l) for l in open(os.path.join(EVAL, "tests.jsonl")) if l.strip()]
    brain = Brain(a.llm, a.prompt, True, temperature=a.temp)
    # warm-up (caches the system prompt prefixes, as in the live demo)
    if a.mode == "kml":
        list(brain.stream("hello", "one person in front of you", session_id="warm"))
    else:
        split_turn(a, "hello", "one person in front of you")
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
                raw, st = split_turn(a, t["heard"], ctx)
                try:
                    turn, valid = kml.parse(raw), True
                except kml.KMLError:
                    turn, valid = None, False
            srv = st.get("server", {}) or {}
            r = dict(id=t["id"], rep=rep, raw=raw, valid=valid, turn=turn,
                     **{k: st.get(k) for k in ("t_first_token", "t_header", "t_first_sentence", "t_total",
                                               "t_text_first_sentence", "t_text_total", "annot_ms", "annot_tps")},
                     n_gen=srv.get("predicted_n"), n_prompt=srv.get("prompt_n"), tps=srv.get("predicted_per_second"),
                     prompt_ms=srv.get("prompt_ms"))
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
    vals = lambda k: [r[k] for r in rows if r.get(k) is not None]
    med = lambda k: round(1000 * statistics.median(vals(k))) if vals(k) else None
    p90 = lambda k: round(1000 * sorted(vals(k))[int(0.9 * len(vals(k))) - 1]) if vals(k) else None
    gs = [p for r in ok for p in r["placement"]]
    summary = dict(
        label=a.label, mode=a.mode, prompt=a.prompt, n=len(rows),
        valid=f"{len(ok)}/{len(rows)}",
        lint_clean=f"{sum(not r['lint'] for r in ok)}/{len(ok)}",
        expect_pass=f"{sum(not r['fails'] for r in ok)}/{len(ok)}",
        placement_ok=f"{sum(gs)}/{len(gs)}",
        emotion_gesture_clash=f"{sum(r['incoherent'] for r in ok)}/{len(ok)}",
        gestures_per_turn=round(len(gs) / max(len(ok), 1), 2),
        words_med=statistics.median(len(r["turn"]["text"].split()) for r in ok) if ok else None,
        first_sentence_words_med=statistics.median(r["first_sentence_words"] for r in ok) if ok else None,
        ms_first_token_med=med("t_first_token"), ms_header_med=med("t_header"),
        ms_first_sentence_med=med("t_first_sentence"), ms_first_sentence_p90=p90("t_first_sentence"),
        ms_text_first_sentence_med=med("t_text_first_sentence"),
        ms_total_med=med("t_total"), gen_tokens_med=statistics.median(vals("n_gen")) if vals("n_gen") else None,
        tps_med=round(statistics.median(vals("tps")), 1) if vals("tps") else None,
        annot_ms_med=statistics.median([x for r in rows for x in (r.get("annot_ms") or [])]) if a.mode == "split" else None,
        annot_tps_med=round(statistics.median([x for r in rows for x in (r.get("annot_tps") or [])]), 1)
        if a.mode == "split" and any(r.get("annot_tps") for r in rows) else None)
    print(json.dumps(summary))
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    json.dump(dict(summary=summary, rows=rows), open(os.path.join(HERE, "results", f"{a.label}.json"), "w"), indent=1)
    open(os.path.join(HERE, "results", "matrix.jsonl"), "a").write(json.dumps(summary) + "\n")


main()
