"""Happy Kermit BRAIN: heard text (+ context, + perception events) -> CONTRACT §1 turn JSON, via Gemma on a
local OpenAI-compatible server (llama-server, Metal or CUDA; mlx_lm.server also works without the grammar).

Python:   from brain import Brain; b = Brain(prompt="persona3"); for ev in b.stream("hi", ctx, "demo"): ...
HTTP:     python brain.py serve [--port 8765] [--llm http://127.0.0.1:8080] [--prompt persona3|full3|...]
          POST /turn  {"heard", "context", "stream": true, "session_id"}  -> NDJSON header, sentence..., turn
          POST /event {"event": "turned_away"|"turned_back"|"speaker_changed"|"person_arrived"|"person_left",
                       "id", "while_speaking", "role", "heard_partial", "context", "session_id"}
                      -> same NDJSON as /turn when Kermit speaks up, else a single {"event": "skip"}
          POST /event {"event": "hand_pick", "id"|"bearing_deg", "description"}  -> Kermit calls on them (qa phase)
          POST /event {"event": "look_command", "id"|"bearing_deg"|null, "description", "heard"}
                      -> skip; the matching /turn ("look at the guy with the drink") answers looking at person_<id>
          POST /phase {"phase": "presentation"|"qa", "session_id"}   (presentation restarts the script)
          POST /beat  {"id": "<beat id>" | omitted = next, "session_id"}  operator override, plays a scripted beat
          POST /reset                                                 (forget all sessions)
          GET  /state?session_id=…  -> {"phase", "cursor", "next_beat", "speaker_id", "role", "away"}  (read-only)
CLI:      python brain.py say "hey kermit wake up" [--context "..."]

Phases (per session):
  presentation  plays presentation.BEATS (cue-synced, instant, no LLM); off-script lines -> SKIP (Kermit
                stays quiet while the presenter talks) unless addressed to Kermit (name / direct question), which
                Gemma answers without moving the script; after the last beat the session switches to qa
  qa            free answers from the persona + FACTS; a judge speaker forces gaze=speaker
"""
import argparse, json, re, select, socket, sys, threading, time, urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import kml, presentation, prompts

FALLBACK = "[thinking|speaker] {tilt_left 0.5} Hmm, my frog ears missed that. {lean_in 0.5} Say it again?"
SKIP = {"event": "skip"}
INTERJECT_COOLDOWN_S = 20.0
PICK_COOLDOWN_S = 5.0
LIVE_INFO = re.compile(r"how many (people|of us|of you|folks|guys|humans|persons)|what time|time is it|today'?s date|"
                       r"what day|what's the date|weather|price|bitcoin|stock|score|who won|news|exchange rate")
LOOK_WAIT_S, LOOK_KEEP_S = 1.5, 3.0
DESCRIBED_LOOK = re.compile(r"\b(look|looking|turn|face|point) (at|to|towards?) (the|that|this|those|these|him|her|them|"
                           r"my|our|a|an|someone|somebody|everyone)\b")


def person_target(ev):
    """GEMMA VISION event -> look_at target: person_<id>, or a direction from bearing_deg (+ = Kermit's left)."""
    if ev.get("id") is not None:
        return f"person_{int(ev['id'])}"
    b = ev.get("bearing_deg")
    if b is None:
        return None
    return "left" if b > 15 else "right" if b < -15 else "speaker"


ADDRESSED = re.compile(r"\b(kermit|permit|frog)\b|^(hey |so |and |okay |ok |um |uh |well |but )*"
                       r"(what|why|how|who|where|when|which|are you|do you|did you|can you|could you|will you|would you|"
                       r"have you|is it|is that|tell (me|us)|say|sing|show)\b")


def addressed(heard):
    """Is an off-script presentation utterance directed at Kermit (name or a direct question/request)?"""
    return bool(ADDRESSED.search((heard or "").lower().strip()))


class Session:
    def __init__(self, phase):
        self.hist = []                 # (user_content, assistant_kml)
        self.phase, self.cursor = phase, 0
        self.away = False              # current speaker has their back turned
        self.just_back = False         # note for the next turn only
        self.role = None               # role of the current speaker (perception), e.g. "judge"
        self.speaker_name = None       # only if perception sends a name with speaker_changed
        self.speaker_id = None         # PERCEPTION speaker_id (speaker_changed "to")
        self.new_speaker = False       # note for the next turn only
        self.last_interject = -1e9
        self.n_people = 0
        self.look = None               # (target, description, t) from GEMMA VISION look_command
        self.look_ready = threading.Event()
        self.last_pick = {}            # person id -> time of last hand_pick


class Brain:
    def __init__(self, llm="http://127.0.0.1:8080", prompt="full", grammar=True, max_tokens=120,
                 temperature=0.7, history_turns=4, model=None, phase="qa", beats="scripted"):
        self.url = llm.rstrip("/") + "/v1/chat/completions"
        self.system = prompts.SYSTEMS[prompt]
        self.fewshot = prompt.startswith("full")
        self.grammar = grammar
        self.max_tokens, self.temperature, self.history_turns, self.model = max_tokens, temperature, history_turns, model
        self.default_phase, self.beats = phase, beats
        self.sessions = {}
        self.last_stats = {}
        self.lock = threading.RLock()  # one generation at a time (llama-server -np 1, shared prompt cache)

    def session(self, session_id):
        return self.sessions.setdefault(session_id, Session(self.default_phase))

    def set_phase(self, phase, session_id="default"):
        s = self.session(session_id)
        s.phase = phase
        if phase == "presentation":
            s.cursor, s.away, s.just_back = 0, False, False

    # ---- low level -------------------------------------------------------------------------------------------
    def _deltas(self, msgs, grammar):
        body = {"messages": msgs, "max_tokens": self.max_tokens, "temperature": self.temperature, "top_p": 0.95,
                "stream": True, "cache_prompt": True, "chat_template_kwargs": {"enable_thinking": False}}
        if grammar:
            body["grammar"] = grammar
        if self.model:
            body["model"] = self.model
        req = urllib.request.Request(self.url, json.dumps(body).encode(), {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as r:
            for line in r:
                line = line.decode().strip()
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                ev = json.loads(line[6:])
                if "timings" in ev:
                    self.last_stats["server"] = ev["timings"]
                for ch in ev.get("choices", []):
                    piece = (ch.get("delta") or {}).get("content")
                    if piece:
                        yield piece

    def _generate(self, s, user, context, heard="", min_words=6, gaze=None, should_stop=None, max_words=45,
                  first_tag=None, extra_targets=()):
        """LLM turn. Yields header, sentence..., turn; appends to session history.
        should_stop() is polled per token; when it returns True the upstream request is closed (llama-server
        cancels the task) and nothing more is yielded or recorded. Closing this generator does the same."""
        msgs = prompts.messages(heard, context, s.hist[-self.history_turns:], self.system, self.fewshot, user=user)
        grammar = kml.gbnf(kml.targets_from_context(context) + list(extra_targets), min_words=min_words,
                           max_words=max_words, first_tag=first_tag or kml.forced_gesture(heard),
                           gaze=gaze) if self.grammar else None
        t0 = time.perf_counter()
        st = self.last_stats = {"t_first_token": None, "t_header": None, "t_first_sentence": None, "source": "llm"}
        raw = []

        def pieces():
            up = self._deltas(msgs, grammar)
            try:
                for p in up:
                    if should_stop and should_stop():
                        st["cancelled"] = True
                        return
                    if st["t_first_token"] is None:
                        st["t_first_token"] = time.perf_counter() - t0
                    raw.append(p)
                    yield p
            finally:
                up.close()  # closes the HTTP response -> llama-server stops generating

        got_header = False
        try:
            for ev in kml.stream_segments(pieces()):
                if ev["event"] == "header":
                    got_header = True
                    st["t_header"] = time.perf_counter() - t0
                elif st["t_first_sentence"] is None:
                    st["t_first_sentence"] = time.perf_counter() - t0
                yield ev
        except OSError as e:  # LLM server down: still say something
            st["error"] = repr(e)
        if st.get("cancelled"):
            return
        line = "".join(raw).strip()
        st["raw"], st["t_total"] = line, time.perf_counter() - t0
        try:
            turn = kml.parse(line)
            st["valid"] = True
        except kml.KMLError:
            st["valid"] = False
            try:
                turn = kml.parse(line, strict=False)
            except kml.KMLError:
                turn = kml.parse(FALLBACK)
        if not got_header:  # unconstrained backend produced no header: replay from the repaired parse
            yield {"event": "header", "emotion": turn["emotion"], "gaze": turn["gaze"]}
            yield {"event": "sentence", "text": turn["text"], "first_word": 0, "gestures": turn["gestures"]}
        s.hist.append((user, kml.to_kml(turn)))
        yield {"event": "turn", **turn}

    def _play(self, s, line, user, beat_id):
        """Scripted turn: same events as _generate, instantly."""
        t0 = time.perf_counter()
        for ev in kml.stream_segments([line]):
            yield ev
        s.hist.append((user, line))
        self.last_stats = {"source": f"script:{beat_id}", "raw": line, "valid": True, "t_total": time.perf_counter() - t0}
        yield {"event": "turn", **kml.parse(line)}

    # ---- heard utterances ------------------------------------------------------------------------------------
    def stream(self, heard, context=None, session_id="default", should_stop=None):
        """Yields {"event":"header"}, {"event":"sentence"}..., then {"event":"turn", **turn}.
        In the scripted presentation an off-script utterance yields a single SKIP instead."""
        with self.lock:
            s = self.session(session_id)
            notes = [prompts.NOTE_AWAY] if s.away else []
            if s.just_back:
                notes.append(prompts.NOTE_BACK)
            if s.role == "judge":
                notes.append(prompts.NOTE_JUDGE)
            if s.new_speaker:
                notes.append(prompts.NOTE_NEW_SPEAKER)
            if LIVE_INFO.search((heard or "").lower()):
                notes.append(prompts.NOTE_LIVE_INFO)
            if kml.forced_gesture(heard) == "look_at:speaker":  # "look at me": aim at speaker_id, no guessed names
                notes.append(prompts.note_speaker(s.speaker_name))
            s.just_back = s.new_speaker = False
            ctx = prompts.with_notes(context, notes)
            user = prompts.user_msg(heard, ctx)
            if s.phase == "presentation" and self.beats == "scripted":
                beat = self._next_beat(s, heard)
                if beat is None and not (getattr(presentation, "OFF_SCRIPT_LIVE", False) or addressed(heard)):  # presenter monologue: stay quiet
                    self.last_stats = {"source": "script:skip", "heard": heard}
                    yield SKIP
                    return
                if beat is None:  # off-script but asked of Kermit directly: answer, script position unchanged
                    yield from self._generate(s, user, ctx, heard, should_stop=should_stop)
                    return
                yield from self._play(s, presentation.BEATS[beat]["kml"], user, presentation.BEATS[beat]["id"])
                s.cursor = beat + 1
                if any(b["trigger"] == "away" for b in presentation.BEATS[:beat]):
                    s.away = False  # a line from after the turn-around: he's facing us again, turned_back or not
                if s.cursor >= len(presentation.BEATS):
                    s.phase = "qa"
                return
            look = self._await_look(s, heard)
            if look is not None:  # "look at the guy with the drink": GEMMA VISION resolved (or failed) the target
                target, desc = look
                ctx = prompts.with_notes(ctx, [prompts.note_looking_at(desc if target else None)])
                user = prompts.user_msg(heard, ctx)
                target = target or "audience"
                yield from self._generate(s, user, ctx, heard, max_words=20, first_tag=f"look_at:{target}",
                                          gaze=f"target:{target}" if target.startswith("person_") else target
                                          if target in ("audience", "speaker", "up", "down") else None,
                                          extra_targets=[target], should_stop=should_stop)
                return
            yield from self._generate(s, user, ctx, heard, gaze="speaker" if s.role == "judge" else None,
                                      should_stop=should_stop)

    def _next_beat(self, s, heard):
        """Index of the beat a heard utterance should play, or None (off-script)."""
        beats, h = presentation.BEATS, (heard or "").lower()
        later = [i for i in range(s.cursor, len(beats)) if beats[i]["trigger"] != "turned_away"]
        away = next((i for i in later if beats[i]["trigger"] == "away"), None)
        # facing away, or the interjection played and no turned_back came (`away` can get lost): the presenter's
        # next line, whatever it is, gets the back-turned beat, unless it already is the line after it
        if away is not None and (s.away or away == s.cursor):
            nxt = next((i for i in later if i > away and beats[i]["trigger"] == "turn"), None)
            if nxt is None or not beats[nxt]["cue"] or not re.search(beats[nxt]["cue"], h):
                return away
        for i in later:  # the next turn beat whose cue matches (beats without a cue match anything)
            if beats[i]["trigger"] == "turn" and (not beats[i]["cue"] or re.search(beats[i]["cue"], h)):
                return i
        return None

    def play_beat(self, beat_id=None, session_id="default"):
        """Operator override: play beat `beat_id` (or the next one) now, whatever was heard."""
        with self.lock:
            s = self.session(session_id)
            ids = [b["id"] for b in presentation.BEATS]
            i = ids.index(beat_id) if beat_id else min(s.cursor, len(ids) - 1)
            b = presentation.BEATS[i]
            yield from self._play(s, b["kml"], prompts.user_msg(b["heard"], presentation.CTX), b["id"])
            s.phase, s.cursor = ("qa", i + 1) if i + 1 >= len(ids) else ("presentation", i + 1)

    def _await_look(self, s, heard, wait_s=LOOK_WAIT_S):
        """For a described-look command ("look at the guy with the drink"): the (target, description) GEMMA VISION
        sent with look_command (waits up to wait_s for it), (None, None) if it never came; None otherwise."""
        if not DESCRIBED_LOOK.search((heard or "").lower()) or kml.forced_gesture(heard):
            return None
        fresh = lambda: s.look is not None and time.monotonic() - s.look[2] < LOOK_KEEP_S
        if not fresh():
            s.look_ready.clear()
            s.look_ready.wait(wait_s)
        if not fresh():
            return (None, None)
        target, desc, _ = s.look
        s.look = None
        return target, desc

    def turn(self, heard, context=None, session_id="default"):
        """Final turn dict, or None when Kermit stays quiet (SKIP)."""
        result = None
        for ev in self.stream(heard, context, session_id):  # run to the end: the script cursor moves after "turn"
            if ev["event"] == "turn":
                result = {k: v for k, v in ev.items() if k != "event"}
        return result

    # ---- perception events (CONTRACT §5) ---------------------------------------------------------------------
    def event(self, ev, session_id="default", should_stop=None):
        """Update perception state; yields a turn's events when Kermit speaks up, else a single SKIP."""
        if ev.get("event") == "look_command":  # no lock: the /turn it belongs to may be holding it, waiting
            s = self.session(session_id)
            target = person_target(ev)
            s.look = (target, ev.get("description") if target else None, time.monotonic())
            s.look_ready.set()
            yield SKIP
            return
        with self.lock:
            s = self.session(session_id)
            kind = ev.get("event")
            if kind == "phase":
                self.set_phase(ev.get("phase", "qa"), session_id)
            elif kind == "turned_away":
                if s.phase == "qa" and not self._is_speaker_turning(s, ev):
                    yield SKIP  # e.g. a presenter turning to look at the judge who is asking: not our speaker
                    return
                s.away = True
                if ev.get("while_speaking"):
                    yield from self._interject(s, ev, should_stop)
                    return
            elif kind == "turned_back":
                if s.away:
                    s.just_back = True
                s.away = False
                if s.phase == "presentation" and s.cursor < len(presentation.BEATS) \
                        and presentation.BEATS[s.cursor]["trigger"] == "away":
                    s.cursor += 1  # never got to the back-turned line: skip it
            elif kind == "speaker_changed":
                s.role, s.speaker_name = ev.get("role"), ev.get("name")
                if s.phase == "qa":  # in the show only turned_back ends away: a presenter facing away often reads
                    s.away = False   # as a speaker change to whoever is still visible
                s.speaker_id = ev.get("to")
                s.new_speaker = s.phase == "qa"
            elif kind == "hand_pick":
                yield from self._hand_pick(s, ev, should_stop)
                return
            elif kind == "person_arrived":
                s.n_people += 1
            elif kind == "person_left":
                s.n_people = max(0, s.n_people - 1)
            yield SKIP

    @staticmethod
    def _is_speaker_turning(s, ev):
        """QA phase: only the current speaker (per PERCEPTION speaker_id) turning away counts, and never a judge's
        or presenter's turn (presenters naturally turn toward the judge who is asking)."""
        if ev.get("id") is not None and s.speaker_id is not None and ev["id"] != s.speaker_id:
            return False
        return s.role not in ("judge", "presenter") and ev.get("role") not in ("judge", "presenter")

    def _hand_pick(self, s, ev, should_stop=None):
        """GEMMA VISION picked a raised hand: call on that person ("Yes, you with the drink!") and look at them."""
        target, key, now = person_target(ev), ev.get("id", ev.get("bearing_deg")), time.monotonic()
        if s.phase == "presentation" or not target or now - s.last_pick.get(key, -1e9) < PICK_COOLDOWN_S:
            yield SKIP
            return
        s.last_pick[key] = now
        ctx = prompts.with_notes(ev.get("context"), [prompts.note_hand(ev.get("description") or "someone")])
        user = prompts.user_msg("", ctx)
        yield from self._generate(s, user, ctx, min_words=2, max_words=12, first_tag=f"look_at:{target}",
                                  gaze=f"target:{target}" if target.startswith("person_") else "speaker",
                                  extra_targets=[target], should_stop=should_stop)

    def _interject(self, s, ev, should_stop=None):
        now = time.monotonic()
        user = prompts.event_msg("turned_away while_speaking", ev.get("heard_partial", ""), ev.get("context"))
        if s.phase == "presentation" and self.beats == "scripted":
            idx = next((i for i in range(s.cursor, len(presentation.BEATS))
                        if presentation.BEATS[i]["trigger"] == "turned_away"), None)
            if idx is not None:
                s.cursor, s.last_interject = idx + 1, now
                yield from self._play(s, presentation.BEATS[idx]["kml"], user, presentation.BEATS[idx]["id"])
                return
        if s.phase == "presentation" and not getattr(presentation, "INTERJECT", True):
            yield SKIP  # solo talk: the presenter faces the audience on purpose
            return
        if now - s.last_interject < INTERJECT_COOLDOWN_S or s.role == "judge":
            yield SKIP  # don't heckle a judge, and only one cut-in per stretch
            return
        s.last_interject = now
        # a short reaction to the turning away (trained 2-8 words); 12-word grammar cap: a tight cap makes it glue words
        yield from self._generate(s, user, ev.get("context"), min_words=2, max_words=12, gaze="speaker",
                                  should_stop=should_stop)


def serve(brain, port):
    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.0"

        def do_GET(self):
            """GET /state?session_id=kermit -> {"phase", "cursor", "next_beat", "speaker_id", "role", "away"}"""
            from urllib.parse import parse_qs, urlparse
            u = urlparse(self.path)
            if u.path.rstrip("/") != "/state":
                return self._json({"error": "GET /state?session_id=..."}, 404)
            s = brain.session(parse_qs(u.query).get("session_id", ["default"])[0])
            nxt = presentation.BEATS[s.cursor]["id"] if s.phase == "presentation" and s.cursor < len(presentation.BEATS) else None
            return self._json({"phase": s.phase, "cursor": s.cursor, "next_beat": nxt, "speaker_id": s.speaker_id,
                               "role": s.role, "away": s.away})

        def do_POST(self):
            path = self.path.rstrip("/")
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            sid = req.get("session_id", "default")
            if path == "/reset":
                brain.sessions.clear()
                return self._json({"ok": True})
            if path == "/phase":
                brain.set_phase(req.get("phase", "qa"), sid)
                return self._json({"ok": True, "phase": brain.session(sid).phase})
            if path == "/turn":
                if not req.get("stream"):
                    return self._json(brain.turn(req.get("heard", ""), req.get("context"), sid))
                return self._ndjson(brain.stream(req.get("heard", ""), req.get("context"), sid, self._gone))
            if path == "/event":
                return self._ndjson(brain.event(req, sid, self._gone))
            if path == "/beat":
                return self._ndjson(brain.play_beat(req.get("id"), sid))
            return self._json({"error": "POST /turn | /event | /phase | /beat | /reset"}, 404)

        def _gone(self):
            """True once the client closed the connection (Turn-Taking cancels speculative turns this way)."""
            try:
                if not select.select([self.connection], [], [], 0)[0]:
                    return False
                return self.connection.recv(1, socket.MSG_PEEK) == b""
            except OSError:
                return True

        def _ndjson(self, events):
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.end_headers()
            spoke = False
            try:
                for ev in events:
                    spoke |= ev["event"] != "skip"
                    self.wfile.write((json.dumps(ev) + "\n").encode())
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                brain.last_stats["cancelled"] = True
            finally:
                events.close()  # stops generation + upstream request if we bailed out early
            if spoke or brain.last_stats.get("cancelled"):
                print(json.dumps({k: v for k, v in brain.last_stats.items() if k != "server"}), file=sys.stderr)

        def _json(self, obj, code=200):
            data = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *a):
            pass

    # warm-up: compiles GPU kernels and caches the system-prompt prefix so the first real turn is fast
    t0 = time.perf_counter()
    brain.set_phase("qa", "_warmup")  # a presentation-phase "hello" would SKIP and never reach the LLM
    brain.turn("hello", "one person in front of you", session_id="_warmup")
    brain.sessions.pop("_warmup", None)
    print(f"warm-up {time.perf_counter() - t0:.2f}s; brain listening on http://127.0.0.1:{port}/turn "
          f"(phase={brain.default_phase}, beats={brain.beats})", file=sys.stderr)
    ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["serve", "say"])
    ap.add_argument("heard", nargs="?", default="")
    ap.add_argument("--context")
    ap.add_argument("--llm", default="http://127.0.0.1:8080")
    ap.add_argument("--prompt", choices=list(prompts.SYSTEMS), default="full",
                    help="full/full3 = base model (rules + few-shots); persona = Kermit LoRA v2; persona3 = LoRA v3")
    ap.add_argument("--no-grammar", action="store_true", help="for backends without GBNF (mlx_lm.server)")
    ap.add_argument("--model", help="model name field (mlx_lm.server)")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--phase", choices=["qa", "presentation"], default="qa", help="initial phase of new sessions")
    ap.add_argument("--beats", choices=["scripted", "generated"], default="scripted",
                    help="presentation beats: exact scripted lines, or let Gemma improvise them")
    a = ap.parse_args()
    b = Brain(a.llm, a.prompt, not a.no_grammar, model=a.model, phase=a.phase, beats=a.beats)
    if a.cmd == "serve":
        serve(b, a.port)
    else:
        for ev in b.stream(a.heard, a.context):
            print(json.dumps(ev))
        print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in b.last_stats.items()
                          if k != "server"}), file=sys.stderr)
