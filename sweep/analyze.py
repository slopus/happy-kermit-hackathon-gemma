"""Summary table over results/*.json, plus a stricter gesture-placement metric.

  python analyze.py LABEL...        (default: all results)

strict placement (only gestures with an obvious anchor word are scored):
  nod   -> anchor word or the next one is a yes-word        shake -> a no-word
  surprise -> whoa/wow/oh... or the word ends with "!"       laugh -> the sentence has a joke cue (ha, haha, funny, joke, kidding, !)
  look_at:<t> -> within ±2 words of a deictic word or the target's name
  head_bob -> a greeting word (hi, hello, hey, bye...) within ±1
"""
import glob, json, os, re, statistics, sys

HERE = os.path.dirname(os.path.abspath(__file__))
norm = lambda w: re.sub(r"[^\w'-]", "", w.lower())
YES = {"yes", "yeah", "yep", "sure", "right", "exactly", "absolutely", "totally", "agree", "agreed", "true", "okay", "ok",
       "definitely", "course", "indeed", "uh-huh", "mhm", "correct", "deal", "yup"}
NO = {"no", "nope", "nah", "not", "never", "don't", "can't", "won't", "nothing", "none", "nuh-uh", "isn't", "i'm", "afraid"}
WOW = {"whoa", "wow", "oh", "what", "huh", "yikes", "gosh", "golly", "really", "wait", "holy", "woah", "eek", "ooh"}
GREET = {"hi", "hello", "hey", "howdy", "bye", "bye-bye", "goodbye", "hiya", "hi-ho", "greetings", "welcome", "see"}
DEIX = {"this", "that", "these", "those", "here", "there", "you", "everyone", "everybody", "folks", "judges", "left",
        "right", "up", "down", "around", "y'all", "all", "over", "audience", "laptop", "crowd", "guys"}


def strict(turn):
    words = turn["text"].split()
    out = []
    for g in turn["gestures"]:
        i, t = g["word"], g["type"]
        win = lambda a, b: {norm(w) for w in words[max(0, i + a):i + b + 1]}
        if t == "nod":
            out.append(bool(win(0, 1) & YES))
        elif t == "shake":
            out.append(bool(win(0, 1) & NO))
        elif t == "surprise":
            out.append(bool(win(0, 1) & WOW) or any(w.endswith("!") for w in words[i:i + 2]))
        elif t == "head_bob":
            out.append(bool(win(-1, 1) & GREET))
        elif t == "look_at":
            out.append(bool(win(-2, 2) & (DEIX | set(g.get("target", "").split("_")))))
    return out


TRIG = [({"whoa", "wow", "woah", "yikes", "gosh", "golly", "ooh"}, {"surprise", "flail"}),
        ({"hi", "hello", "hey", "howdy", "bye", "bye-bye", "goodbye", "hiya", "hi-ho"}, {"head_bob", "look_at", "flail", "nod"}),
        ({"yes", "yeah", "yep", "yup", "absolutely", "definitely", "exactly", "totally", "agreed"}, {"nod"}),
        ({"no", "nope", "nah", "nuh-uh"}, {"shake"})]


def trigger_hits(turn):
    """(hits, opportunities): each trigger word should carry a matching gesture within ±1 word."""
    words = [norm(w) for w in turn["text"].split()]
    at = {}
    for g in turn["gestures"]:
        at.setdefault(g["word"], set()).add(g["type"])
    hits = opp = 0
    for i, w in enumerate(words):
        for trig, ok in TRIG:
            if w in trig:
                opp += 1
                hits += any(at.get(j, set()) & ok for j in (i - 1, i, i + 1))
    return hits, opp


def row(lab):
    d = json.load(open(os.path.join(HERE, "results", f"{lab}.json")))
    s, rows = d["summary"], d["rows"]
    ok = [r for r in rows if r["valid"]]
    st = [x for r in ok for x in strict(r["turn"])]
    s["strict_place"] = f"{sum(st)}/{len(st)}"
    s["strict_pct"] = round(100 * sum(st) / max(len(st), 1))
    th = [trigger_hits(r["turn"]) for r in ok]
    h, o = sum(x for x, _ in th), sum(y for _, y in th)
    s["trigger_hit"] = f"{h}/{o} ({round(100 * h / max(o, 1))}%)"
    return s


labels = sys.argv[1:] or sorted(os.path.basename(p)[:-5] for p in glob.glob(os.path.join(HERE, "results", "*.json")))
cols = ["label", "valid", "expect_pass", "trigger_hit", "gestures_per_turn", "words_med",
        "first_sentence_words_med", "ms_first_token_med", "ms_header_med", "ms_text_first_sentence_med",
        "ms_first_sentence_med", "ms_first_sentence_p90", "ms_total_med", "gen_tokens_med", "tps_med", "annot_ms_med"]
print("| " + " | ".join(cols) + " |")
print("|" + "---|" * len(cols))
for lab in labels:
    s = row(lab)
    print("| " + " | ".join(str(s.get(c, "")) for c in cols) + " |")
