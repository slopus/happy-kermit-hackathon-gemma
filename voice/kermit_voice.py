"""End to end, one turn: speech (WAV or mic) → Parakeet → BRAIN stream → Kokoro per sentence →
audio.wav + timeline.json (CONTRACT §2), with a latency report.

  python kermit_voice.py --wav q.wav                      # STT + BRAIN (HTTP :8765) + TTS
  python kermit_voice.py --mic [--ptt]                    # loop: listen, answer, repeat
  python kermit_voice.py --turn turn.json                 # skip STT/BRAIN: CONTRACT §1 JSON in
  python kermit_voice.py --text "Hi-ho! Whoa." --emotion excited
Options: --out DIR  --voice am_fenrir  --tts-backend torch|mlx  --device cpu|mps|cuda
         --brain-url URL (default http://127.0.0.1:8765: start BRAIN with `bash ../brain/mac/run.sh`)
         --brain-url inproc --llm URL --brain-prompt persona|full (import brain.py instead)  --play

Sentence streaming: TTS for sentence 1 starts when BRAIN's first `sentence` event arrives. Each
sentence's audio is written as soon as it exists (<out>/turn_NNNN_sK.wav) and announced on stdout as
an NDJSON `segment` event (its own words / mouth / events, times relative to the turn start), so a
player can start on sentence 1. After the last sentence, the concatenated turn_NNNN.wav and
turn_NNNN.timeline.json are written. Latency is measured from speech end (end of WAV, or the VAD
endpoint minus its hangover for the mic).
"""
import argparse, json, os, sys, time
import numpy as np, soundfile as sf

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from timeline import align_words, build_timeline
from lipsync import mouth_curve, HZ
from tts import KokoroTTS, SR

LEAD_SIL, TAIL_SIL, GAP = 0.03, 0.08, 0.12  # trim each sentence to this much silence; pause between sentences


def trim(audio, words, lead=LEAD_SIL, tail=TAIL_SIL, thresh_db=-45):
    """Cut leading/trailing silence (Kokoro pads ~150 ms of it, which is pure latency on sentence 1)."""
    if not len(audio):
        return audio, words
    hop = int(SR * 0.005)
    e = np.array([np.max(np.abs(audio[i:i + hop])) for i in range(0, len(audio), hop)])
    loud = np.nonzero(20 * np.log10(e + 1e-9) > 20 * np.log10(e.max() + 1e-9) + thresh_db)[0]
    if not len(loud):
        return audio, words
    s = max(0, loud[0] * hop - int(lead * SR)); t = min(len(audio), (loud[-1] + 1) * hop + int(tail * SR))
    sh = s / SR
    return audio[s:t], [{"w": w["w"], "t0": max(0.0, round(w["t0"] - sh, 3)), "t1": round(min(w["t1"] - sh, (t - s) / SR), 3)} for w in words]


class Performer:
    def __init__(self, a):
        t = time.perf_counter()
        self.tts = KokoroTTS(a.voice, backend=a.tts_backend, device=a.device, character=not a.no_frog)
        self.load = {"tts_s": round(time.perf_counter() - t, 2)}
        self.a, self.rec, self.brain = a, None, None
        self.on_audio = None  # hub hook: on_audio(k, chunk, seg) gets each sentence's audio (with its leading gap)
        os.makedirs(a.out, exist_ok=True)
        self.n = len([f for f in os.listdir(a.out) if f.endswith(".timeline.json")])
        import threading
        self._lock, self._tts_lock, self._stt_lock = threading.Lock(), threading.Lock(), threading.Lock()  # replies and interjections run on separate threads

    def stt(self):
        if self.rec is None:
            import stt
            t = time.perf_counter(); self.rec = stt.recognizer(self.a.stt_provider)
            stt.transcribe(self.rec, np.zeros(16000, np.float32))  # warm-up: first decode allocates
            self.load["stt_s"] = round(time.perf_counter() - t, 2)
        return self.rec

    def transcribe(self, audio16):
        import stt
        with self._stt_lock:  # one recognizer, shared by the reply path and partial transcripts
            return stt.transcribe(self.stt(), audio16).text.strip()

    def brain_events(self, heard, context):
        if self.a.brain_url != "inproc":
            import urllib.request
            req = urllib.request.Request(self.a.brain_url.rstrip("/") + "/turn", json.dumps(
                {"heard": heard, "context": context, "stream": True, "session_id": "kermit"}).encode(), {"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=60) as r:
                for line in r:
                    if line.strip(): yield json.loads(line)
            return
        if self.brain is None:
            sys.path.insert(0, os.path.join(HERE, "..", "brain"))
            from brain import Brain
            self.brain = Brain(llm=self.a.llm, prompt=self.a.brain_prompt)
        yield from self.brain.stream(heard, context, "kermit")

    @staticmethod
    def events_from_turn(turn):
        """A complete CONTRACT §1 turn → the same header/sentence/turn events BRAIN streams."""
        import re
        yield {"event": "header", "emotion": turn.get("emotion", "neutral"), "gaze": turn.get("gaze", "speaker")}
        words = turn["text"].split(); i = 0
        for sent in re.split(r"(?<=[.!?])\s+", turn["text"].strip()):
            n = len(sent.split())
            yield {"event": "sentence", "text": sent, "first_word": i,
                   "gestures": [g for g in turn.get("gestures", []) if i <= g["word"] < i + n]}
            i += n
        yield {"event": "turn", **turn}

    def perform(self, events, t_speech_end, lat, cancel=None):
        """Consume BRAIN events, synthesise per sentence, emit segments; returns (wav path, timeline).
        `cancel` (threading.Event): checked before each sentence; when set, stop and mark the turn cancelled."""
        with self._lock:
            tag = f"turn_{self.n:04d}"; self.n += 1
        emotion, gaze, pieces, all_words, segs, turn, t_cur = "neutral", "speaker", [], [], [], None, 0.0
        for ev in events:
            if ev["event"] == "header":
                emotion, gaze = ev["emotion"], ev["gaze"]
                lat.setdefault("brain_header_ms", round((time.perf_counter() - t_speech_end) * 1000))
            elif ev["event"] == "sentence":
                if cancel is not None and cancel.is_set():
                    lat["cancelled"] = True
                    if hasattr(events, "close"): events.close()
                    break
                k = len(segs)
                if k == 0: lat["brain_first_sentence_ms"] = round((time.perf_counter() - t_speech_end) * 1000)
                t = time.perf_counter()
                with self._tts_lock:
                    audio, timed = self.tts.synth(ev["text"], emotion)
                audio, timed = trim(audio, timed)
                tts_ms = (time.perf_counter() - t) * 1000
                words = align_words(ev["text"].split(), timed)
                gestures = [dict(g, word=g["word"] - ev["first_word"]) for g in ev.get("gestures", [])]
                seg = build_timeline({"text": ev["text"], "emotion": emotion, "gaze": gaze, "gestures": gestures},
                                     audio, SR, timed, f"{tag}_s{k}.wav")
                if k > 0:  # pause between sentences
                    pieces.append(np.zeros(int(GAP * SR), np.float32)); t_cur += GAP
                sf.write(os.path.join(self.a.out, seg["audio"]), audio, SR)
                for w in seg["words"]: w["t0"] = round(w["t0"] + t_cur, 3); w["t1"] = round(w["t1"] + t_cur, 3)
                for e in seg["events"]: e["t"] = round(e["t"] + t_cur, 3); e["word"] += ev["first_word"]
                seg.update(event="segment", index=k, t_offset=round(t_cur, 3))
                if k == 0:
                    lat["first_audio_ms"] = round((time.perf_counter() - t_speech_end) * 1000)
                    lat["tts_first_sentence_ms"] = round(tts_ms)
                print(json.dumps(dict(turn=tag, kind=lat.get("kind", "reply"), emotion=emotion, gaze=gaze, out_dir=os.path.abspath(self.a.out),
                                      **{x: seg[x] for x in ("event", "index", "audio", "t_offset", "duration", "words", "events", "mouth")})), flush=True)
                chunk = audio if k == 0 else np.concatenate([np.zeros(int(GAP * SR), np.float32), audio])
                if self.on_audio:
                    self.on_audio(k, chunk, seg)
                elif self.a.play:
                    if k == 0: self.player = _play_stream()
                    self.player.send(chunk)
                pieces.append(audio); all_words += seg["words"]; segs.append(seg); t_cur += len(audio) / SR
            elif ev["event"] == "turn":
                turn = {k: v for k, v in ev.items() if k != "event"}
        audio = np.concatenate(pieces) if pieces else np.zeros(1, np.float32)
        lat["turn_ready_ms"] = round((time.perf_counter() - t_speech_end) * 1000)
        # global timeline: words from the per-sentence alignment, mouth recomputed per sentence and concatenated
        mouth = []
        for k, seg in enumerate(segs):
            if k: mouth += [0.0] * (int(round((seg["t_offset"]) * HZ)) - len(mouth))
            mouth += seg["mouth"]["values"]
        mouth = (mouth + [0.0] * int(np.ceil(len(audio) / SR * HZ)))[:int(np.ceil(len(audio) / SR * HZ))]
        tl = {"audio": f"{tag}.wav", "sample_rate": SR, "duration": round(len(audio) / SR, 3), "words": all_words,
              "mouth": {"hz": HZ, "values": mouth}, "events": [e for s in segs for e in s["events"]],
              "emotion": emotion, "gaze": gaze, "text": (turn or {}).get("text", " ".join(w["w"] for w in all_words)),
              "latency": lat}
        wav = os.path.join(self.a.out, f"{tag}.wav")
        sf.write(wav, audio, SR)
        json.dump(tl, open(os.path.join(self.a.out, f"{tag}.timeline.json"), "w"), separators=(",", ":"))
        if turn: json.dump(turn, open(os.path.join(self.a.out, f"{tag}.turn.json"), "w"))
        return wav, tl

    # Canned interjections when BRAIN has no /event endpoint (or is down): the demo still reacts.
    CANNED = {
        "turned_away": {"text": "Oh, like, hello? I'm still over here!", "emotion": "surprised", "gaze": "speaker",
                        "gestures": [{"word": 0, "type": "surprise", "intensity": 0.8}, {"word": 3, "type": "lean_in", "intensity": 0.6}]},
        "turned_back": {"text": "Oh good, you're back!", "emotion": "happy", "gaze": "speaker",
                        "gestures": [{"word": 0, "type": "head_bob", "intensity": 0.7}]},
        "person_arrived": {"text": "Oh, hi there! Welcome!", "emotion": "excited", "gaze": "speaker",
                           "gestures": [{"word": 1, "type": "head_bob", "intensity": 0.8}]},
    }

    def event_turn(self, pev, heard_partial="", context=None):
        """A perception event (CONTRACT §5) → BRAIN POST /event → performed immediately. Returns (wav, tl) or None."""
        t0 = time.perf_counter(); lat = {"kind": "interjection", "trigger": pev.get("event"), "heard_partial": heard_partial}
        def from_brain():
            import urllib.request, urllib.error
            body = dict(pev, heard_partial=heard_partial, context=context, session_id="kermit")
            req = urllib.request.Request(self.a.brain_url.rstrip("/") + "/event", json.dumps(body).encode(), {"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=30) as r:
                if r.status == 204: return
                for line in r:
                    if line.strip(): yield json.loads(line)
        events = None
        if self.a.brain_url != "inproc":
            try:
                gen = from_brain(); first = next(gen, None)
                if first is None or first.get("event") == "skip":
                    return None
                events = _chain(first, gen)
            except Exception as e:  # 404 (no /event yet), connection refused, …
                lat["brain_error"] = repr(e)[:120]
        if events is None:
            canned = self.CANNED.get(pev.get("event"))
            if canned is None or (pev.get("event") == "turned_away" and not pev.get("while_speaking")): return None
            lat["canned"] = True; events = self.events_from_turn(canned)
        wav, tl = self.perform(events, t0, lat)
        print(json.dumps({"event": "done", "turn": os.path.basename(wav)[:-4], "kind": "interjection", "audio": wav,
                          "timeline": wav.replace(".wav", ".timeline.json"), "latency": lat}), flush=True)
        return wav, tl

    def one(self, audio16=None, t_speech_end=None, turn=None, heard=None, cancel=None):
        lat = {}
        t_speech_end = t_speech_end or time.perf_counter()
        if turn is not None:
            events = self.events_from_turn(turn)
        else:
            if heard is None:
                import stt
                t = time.perf_counter(); heard = self.transcribe(audio16)
                lat["stt_ms"] = round((time.perf_counter() - t) * 1000)
            lat["heard"] = heard
            events = self.brain_events(heard, self.a.context)
            first = next(events, None)
            if first is None or first.get("event") == "skip":  # BRAIN chose to stay quiet (off-script line, judge chatter)
                print(json.dumps({"event": "skip", "heard": heard, "latency": lat}), flush=True)
                return None
            events = _chain(first, events)
        wav, tl = self.perform(events, t_speech_end, lat, cancel)
        print(json.dumps({"event": "done", "turn": os.path.basename(wav)[:-4], "kind": lat.get("kind", "reply"), "audio": wav, "timeline": wav.replace(".wav", ".timeline.json"), "latency": lat}), flush=True)
        return wav, tl


def _chain(first, rest):
    yield first
    yield from rest


def _play_stream():
    """Generator-based gapless player (sounddevice). send() audio chunks; plays as they arrive."""
    import sounddevice as sd, queue, threading
    q = queue.Queue()
    def run():
        with sd.OutputStream(samplerate=SR, channels=1, dtype="float32") as out:
            while True:
                c = q.get()
                if c is None: return
                out.write(c.reshape(-1, 1))
    threading.Thread(target=run, daemon=True).start()
    class P:
        def send(self, a): q.put(a.astype(np.float32))
    return P()


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--wav", nargs="+"); src.add_argument("--mic", action="store_true"); src.add_argument("--turn"); src.add_argument("--text")
    src.add_argument("--heard", help="skip STT: pretend the user said this")
    ap.add_argument("--emotion", default="neutral")
    ap.add_argument("--ptt", action="store_true")
    ap.add_argument("--out", default=os.path.join(HERE, "out"))
    ap.add_argument("--voice", default=os.environ.get("KERMIT_VOICE", "am_fenrir"))
    ap.add_argument("--no-frog", action="store_true", help="plain Kokoro voice, no pitch/formant shift + EQ")
    ap.add_argument("--tts-backend", default=os.environ.get("KERMIT_TTS_BACKEND", "torch"), choices=["torch", "mlx"])
    ap.add_argument("--device", default=None)
    ap.add_argument("--stt-provider", default=os.environ.get("KERMIT_STT_PROVIDER", "cpu"))
    ap.add_argument("--brain-url", default=os.environ.get("KERMIT_BRAIN_URL", "http://127.0.0.1:8765"),
                    help="BRAIN server started by brain/mac/run.sh [fast|base] (it owns the model/prompt pairing); "
                         "'inproc' = import brain.py here and talk to llama-server at --llm")
    ap.add_argument("--llm", default=os.environ.get("KERMIT_LLM", "http://127.0.0.1:8080"), help="inproc only")
    ap.add_argument("--brain-prompt", default=os.environ.get("KERMIT_BRAIN_PROMPT", "persona3"),
                    help="inproc only: persona3 with the Kermit LoRA (see brain/README), full with plain base Gemma")
    ap.add_argument("--context", default=None)
    ap.add_argument("--play", action="store_true")
    return ap.parse_args()


def run(p, a):
    if a.turn:
        p.one(turn=json.load(open(a.turn)))
    elif a.text:
        p.one(turn={"text": a.text, "emotion": a.emotion, "gestures": [], "gaze": "speaker"})
    elif a.heard:
        p.one(heard=a.heard)
    elif a.wav:
        from scipy.signal import resample_poly
        p.stt()
        for f in a.wav:
            x, sr = sf.read(f, dtype="float32")
            x = x.mean(1) if x.ndim > 1 else x
            p.one(audio16=resample_poly(x, 16000, sr) if sr != 16000 else x)
    else:
        import stt
        p.stt(); lis = stt.Listener("ptt" if a.ptt else "vad")
        print(f"listening ({'push-to-talk' if a.ptt else 'VAD'}) …", file=sys.stderr)
        for audio16, t_fire in lis.utterances():
            # VAD fires min_silence after the last voiced frame: that hangover is part of the user-felt latency
            hang = 0.0 if a.ptt else lis.vad_min_silence
            p.one(audio16=audio16, t_speech_end=t_fire - hang)


if __name__ == "__main__":
    a = parse_args()
    run(Performer(a), a)
