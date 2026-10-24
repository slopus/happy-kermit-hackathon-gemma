"""KML (Kermit markup line) <-> CONTRACT.md turn JSON.  See FORMAT.md.

    python kml.py validate FILE.jsonl...   check every assistant message in chat JSONL files
    python kml.py grammar                  print the GBNF grammar used to constrain decoding
"""
import json, re, sys

EMOTIONS = ["neutral", "happy", "excited", "sad", "surprised", "thinking", "embarrassed", "annoyed"]
GESTURES = ["nod", "shake", "tilt_left", "tilt_right", "head_bob", "look_at", "look_around", "surprise", "laugh",
            "sigh", "shrug", "lean_in", "lean_back", "flail", "cover_face"]
GAZES = ["speaker", "audience", "down", "up"]  # plus target:<name>

HEADER = re.compile(r"^\[(\w+)\|([\w:]+)\]\s*")
TAG = re.compile(r"\{(\w+)(?::([a-z0-9_]+))?\s+([01](?:\.\d+)?)\}")


class KMLError(ValueError):
    pass


def parse(line, strict=True):
    """KML line -> contract turn dict. strict=False repairs what it can (runtime path)."""
    line = line.strip().split("\n")[0].strip()
    m = HEADER.match(line)
    if not m:
        if strict:
            raise KMLError(f"missing [emotion|gaze] header: {line[:60]!r}")
        emotion, gaze, body = "neutral", "speaker", line
    else:
        emotion, gaze, body = m.group(1), m.group(2), line[m.end():]
    if emotion not in EMOTIONS:
        if strict:
            raise KMLError(f"bad emotion {emotion!r}")
        emotion = "neutral"
    if not (gaze in GAZES or re.fullmatch(r"target:[a-z0-9_]+", gaze)):
        if strict:
            raise KMLError(f"bad gaze {gaze!r}")
        gaze = "speaker"

    words, gestures, pending = [], [], []
    pos = 0
    for tm in TAG.finditer(body):
        words += body[pos:tm.start()].split()
        pos = tm.end()
        typ, target, inten = tm.group(1), tm.group(2), float(tm.group(3))
        if typ not in GESTURES:
            if strict:
                raise KMLError(f"bad gesture {typ!r}")
            continue
        if target and typ != "look_at":
            if strict:
                raise KMLError(f"target on non-look_at gesture {typ!r}")
            target = None
        g = {"word": len(words), "type": typ, "intensity": round(min(max(inten, 0.0), 1.0), 2)}
        if typ == "look_at":
            g["target"] = target or "speaker"
        gestures.append(g)
    words += body[pos:].split()
    if any(c in w for w in words for c in "{}[]*"):
        if strict:
            raise KMLError(f"stray markup in text: {body[:80]!r}")
        words = [re.sub(r"[{}\[\]*]", "", w) for w in words]
        words = [w for w in words if w]
    if not words:
        raise KMLError("empty text")
    for g in gestures:  # a trailing tag anchors to the last word
        g["word"] = min(g["word"], len(words) - 1)
    return {"text": " ".join(words), "emotion": emotion, "gestures": gestures, "gaze": gaze}


def lint(turn):
    """Style problems (soft rules from FORMAT.md). Returns a list of strings."""
    probs = []
    n = len(turn["text"].split())
    if n > 40:
        probs.append(f"too long ({n} words)")
    if not 1 <= len(turn["gestures"]) <= 5:
        probs.append(f"{len(turn['gestures'])} gestures")
    if re.search(r"[0-9%$#@&*_~<>]", turn["text"]):
        probs.append("digits/symbols in spoken text")
    if re.search(r"[\U0001F300-\U0001FAFF\u2600-\u27BF]", turn["text"]):
        probs.append("emoji")
    seen = set()
    for g in turn["gestures"]:
        if g["word"] in seen:
            probs.append(f"two gestures on word {g['word']}")
        seen.add(g["word"])
    return probs


def to_kml(turn):
    """Contract turn dict -> KML line (inverse of parse, used for few-shot/data conversion)."""
    words = turn["text"].split()
    by_word = {}
    for g in turn["gestures"]:
        by_word.setdefault(g["word"], []).append(g)
    out = []
    for i, w in enumerate(words):
        for g in by_word.get(i, []):
            t = g["type"] + (f":{g['target']}" if g["type"] == "look_at" else "")
            out.append(f"{{{t} {g['intensity']:.1f}}}")
        out.append(w)
    return f"[{turn['emotion']}|{turn['gaze']}] " + " ".join(out)


BASE_TARGETS = ["speaker", "audience", "left", "right", "up", "down"]


def targets_from_context(context):
    """look_at / gaze targets MOTION can resolve: the fixed directions plus items of a 'visible: a, b' list."""
    m = re.search(r"visible:\s*([^.\n]*)", context or "")
    names = [re.sub(r"[^a-z0-9_]", "", t.strip().lower().replace(" ", "_")) for t in m.group(1).split(",")] if m else []
    return BASE_TARGETS + [n for n in dict.fromkeys(names) if n and n not in BASE_TARGETS]


# explicit stage commands in the heard text -> the gesture the reply must open with
COMMANDS = [
    (r"\bnod(s|ding)?\b", "nod"), (r"\bshake (your|ur|the|that) head|\bshake your head", "shake"),
    (r"\b(look|turn|face|glance)(ing)? (to |over |towards? )?(to )?(your |the |my )?left\b", "look_at:left"),
    (r"\b(look|turn|face|glance)(ing)? (to |over |towards? )?(to )?(your |the |my )?right\b", "look_at:right"),
    (r"\blook(ing)? at me\b|\bface me\b|\bturn (to|towards?) me\b", "look_at:speaker"),
    (r"\blook(ing)? around", "look_around"), (r"\blook(ing)? up\b", "look_at:up"), (r"\blook(ing)? down\b", "look_at:down"),
    (r"\b(look at|face|turn to|turn towards?) (the )?(audience|judges|everyone|everybody|crowd)", "look_at:audience"),
    (r"\bturn around\b", "look_around"),
    (r"(surprised|shocked|whoa) (face|look)|\bact (surprised|shocked)", "surprise"),
    (r"\b(act|look) embarrassed|\bcover your face", "cover_face"), (r"\bshrug", "shrug"), (r"\btilt your head", "tilt_left"),
    (r"\b(do a |do the )?flail|\bwave your arms", "flail"),
]


def forced_gesture(heard):
    """'nod if you agree' -> 'nod'; None when no explicit command."""
    h = (heard or "").lower()
    for pat, g in COMMANDS:
        if re.search(pat, h):
            return g
    return None


def gbnf(targets=None, min_words=6, first_tag=None, gaze=None, max_words=45):
    """GBNF for one KML line. targets=None allows any [a-z0-9_] name; a list restricts look_at/gaze targets.
    min_words stops the model from ending the turn after a lone interjection ("Whoa!").
    first_tag (e.g. "nod", "look_at:left") forces the reply to open with that gesture.
    gaze (e.g. "speaker") forces the header's gaze. max_words caps the length (interjections: 8)."""
    alt = lambda xs: " | ".join(f'"{x}"' for x in xs)
    plain = [g for g in GESTURES if g != "look_at"]
    name = alt(targets) if targets else "[a-z0-9_]{1,24}"
    return f"""root ::= "[" emotion "|" gaze "] " body
emotion ::= {alt(EMOTIONS)}
gaze ::= {f'"{gaze}"' if gaze else alt(GAZES) + ' | "target:" name'}
name ::= {name}
body ::= {f'"{{{first_tag} " inten "}} "' if first_tag else '(tag " ")?'} word (" " (tag " ")? word){{{min_words - 1},{max_words - 1}}}
tag ::= "{{" ( ({alt(plain)}) | "look_at:" name ) " " inten "}}"
inten ::= "0." [1-9] | "1.0"
word ::= [A-Za-z\u00C0-\u00FF'\u2019.,!?;:"()-]{{1,24}}
"""


def stream_segments(pieces):
    """Incremental parser for streaming decode.  Feed text deltas; yields events:
      {"event": "header", "emotion", "gaze"}               as soon as the header closes
      {"event": "sentence", "text", "first_word", "gestures"}  each finished sentence (global word indices)
    The final full turn should still be built with parse() on the concatenated output."""
    buf, header_done, words_emitted = "", False, 0
    for p in pieces:
        buf += p
        if not header_done:
            m = HEADER.match(buf)
            if not m:
                continue
            header_done = True
            yield {"event": "header", "emotion": m.group(1), "gaze": m.group(2)}
            buf = buf[m.end():]
        while True:
            # sentence end = . ! ? followed by space, and not inside a tag
            m = re.search(r"[.!?]+[\"')]*\s", buf)
            if not m or buf[:m.end()].count("{") != buf[:m.end()].count("}"):
                break
            seg, buf = buf[:m.end()], buf[m.end():]
            ev = _segment(seg, words_emitted)
            if ev:
                words_emitted += len(ev["text"].split())
                yield ev
    if header_done and buf.strip():
        ev = _segment(buf, words_emitted)
        if ev:
            yield ev


def _segment(seg, offset):
    try:
        t = parse("[neutral|speaker] " + seg, strict=False)
    except KMLError:
        return None
    for g in t["gestures"]:
        g["word"] += offset
    return {"event": "sentence", "text": t["text"], "first_word": offset, "gestures": t["gestures"]}


def _validate(paths):
    bad = total = 0
    for path in paths:
        for ln, raw in enumerate(open(path), 1):
            if not raw.strip():
                continue
            ex = json.loads(raw)
            for msg in ex["messages"]:
                if msg["role"] != "assistant":
                    continue
                total += 1
                try:
                    probs = lint(parse(msg["content"]))
                except KMLError as e:
                    probs = [str(e)]
                if probs:
                    bad += 1
                    print(f"{path}:{ln}: {'; '.join(probs)} :: {msg['content'][:100]}")
    print(f"{total - bad}/{total} assistant turns OK")
    return bad == 0


if __name__ == "__main__":
    if sys.argv[1:2] == ["validate"]:
        sys.exit(0 if _validate(sys.argv[2:]) else 1)
    elif sys.argv[1:2] == ["grammar"]:
        print(gbnf())
    else:
        print(__doc__)
