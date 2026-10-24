"""Live Happy Kermit hub, full duplex: mic + camera + BRAIN + TTS + speaker in one process.

  bash brain/mac/run.sh &                                    # BRAIN on :8765 (default base + Kermit LoRA)
  python run_kermit.py --camera 0 --tts-backend mlx           # VAD turn-taking, Kermit may interject
  python run_kermit.py --camera 0 --ptt --tts-backend mlx     # push-to-talk (Enter / Enter)
  # headless test: a WAV stands in for the mic (real-time paced) + scripted perception events
  python run_kermit.py --mic-wav talk.wav --events-script ev.ndjson --no-speaker --out out/sim

Two paths run side by side:
  reply        mic → Silero VAD (or PTT) → end of utterance → Parakeet → BRAIN /turn → TTS → speaker
  interjection perception event (turned_away while speaking, …) → BRAIN /event with a partial
               transcript of the human's utterance in progress → TTS → speaker, IMMEDIATELY: it does
               not wait for the human to stop (Kirill's barge-in), and it cuts off any queued reply.
The mic is never paused: the human's audio keeps being captured and transcribed while Kermit talks
(echo of Kermit's own voice is dropped by comparing transcripts with what he is saying).

stdout: one NDJSON stream for MOTION
  {"event":"gaze", …§4…}                         ~20 Hz (speaker's bearing with perception v2)
  {"event":"segment","turn":…,"kind":…,"emotion","gaze","mouth":{hz,values},…}  per sentence, when synthesised
      (mouth: relative to the sentence's own audio start; words/events: on the turn clock, i.e. minus t_offset)
  {"event":"play","turn":…,"index":k,"t_wall":…,"t_audio":…}     chunk starts playing / the sentence's own audio starts (perf_counter)
  {"event":"stop","turn":…,"index":k,"at_s":…}                    playback cut off (an interjection preempted it)
  {"event":"done","turn":…,"timeline":…}                          per turn
  {"event":"perception",…}                                        each §5 event, as forwarded to BRAIN
  {"event":"heard","text":…,"echo":bool}                          each finished human utterance
"""
import argparse, json, os, queue, re, sys, threading, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, "voice"), os.path.join(HERE, "perception")]
import kermit_voice
from tts import SR

_out_lock = threading.Lock()


def emit(obj):
    line = json.dumps(obj, separators=(",", ":")) if isinstance(obj, dict) else obj
    with _out_lock:
        sys.stdout.write(line + "\n"); sys.stdout.flush()


kermit_voice.print = lambda *a, **k: emit(a[0])  # segment / done lines go through the same lock


# ---------------------------------------------------------------- speaker
class Speaker(threading.Thread):
    """One output stream for the whole session. Plays queued sentence chunks in 20 ms blocks so an
    interjection can cut in within a block. Announces each sentence start (play) and cut-offs (stop)."""

    BLOCK = int(0.02 * SR)

    def __init__(self, real=True, on_speaking=None):
        super().__init__(daemon=True)
        self.q, self.real, self.on_speaking = queue.Queue(), real, on_speaking or (lambda b: None)
        self.cur, self.gen = None, 0   # [turn, k, audio, pos]; generation counter invalidates queued chunks
        self.spoken = []                # [(t_start, t_end, words)] recent Kermit speech, for echo rejection
        self.speaking = False

    def enqueue(self, turn, k, chunk, words, lead_s=0.0):
        """lead_s: silence at the start of `chunk` before the sentence itself (the inter-sentence gap)."""
        self.q.put((self.gen, turn, k, chunk, words, lead_s))

    def interrupt(self):
        """Drop everything queued or playing (called before an interjection is synthesised)."""
        self.gen += 1
        cur = self.cur
        if cur:
            emit({"event": "stop", "turn": cur[0], "index": cur[1], "at_s": round(cur[3] / SR, 3)})
        self.cur = None

    def run(self):
        out = None
        if self.real:
            import sounddevice as sd
            out = sd.OutputStream(samplerate=SR, channels=1, dtype="float32", blocksize=self.BLOCK); out.start()
        while True:
            try:
                gen, turn, k, chunk, words, lead_s = self.q.get(timeout=0.05)
            except queue.Empty:
                if self.speaking: self.speaking = False; self.on_speaking(False)
                continue
            if gen != self.gen: continue
            cur = self.cur = [turn, k, chunk, 0]
            t0 = time.perf_counter()
            emit({"event": "play", "turn": turn, "index": k, "t_wall": round(t0, 4), "t_audio": round(t0 + lead_s, 4)})
            if not self.speaking: self.speaking = True; self.on_speaking(True)
            self.spoken = (self.spoken + [(t0, t0 + len(chunk) / SR, words)])[-20:]
            while self.cur is cur and cur[3] < len(chunk):
                b = chunk[cur[3]:cur[3] + self.BLOCK]
                if out is not None: out.write(b.reshape(-1, 1))
                else: time.sleep(len(b) / SR)  # headless: keep real-time pacing
                cur[3] += len(b)
            if self.cur is cur: self.cur = None

    def said_recently(self, t_from, t_to):
        return " ".join(" ".join(w) for s, e, w in self.spoken if e >= t_from - 0.3 and s <= t_to + 0.3)


# ---------------------------------------------------------------- mic
class Mic(threading.Thread):
    """16 kHz capture → Silero VAD (or PTT) → utterances. Never pauses. Tracks the utterance in progress so
    an interjection can transcribe what the human has said so far."""

    def __init__(self, a, hub):
        super().__init__(daemon=True)
        import stt
        self.a, self.hub = a, hub
        self.vad = stt.Listener("vad", min_silence=a.vad_silence).vad
        self.live = []            # blocks of the utterance in progress (from speech onset, with pre-roll)
        self.pre = []             # 0.3 s pre-roll ring
        self.in_speech = False
        self.ptt_on = False
        self.t_speech_start = None

    def blocks(self):
        if self.a.mic_wav:
            import soundfile as sf
            from scipy.signal import resample_poly
            x, sr = sf.read(self.a.mic_wav, dtype="float32")
            x = x.mean(1) if x.ndim > 1 else x
            x = resample_poly(x, 16000, sr).astype(np.float32) if sr != 16000 else x
            x = np.concatenate([np.zeros(8000, np.float32), x, np.zeros(16000 * 2, np.float32)])
            t = time.perf_counter()
            for i in range(0, len(x) - 511, 512):
                t += 512 / 16000; time.sleep(max(0.0, t - time.perf_counter()))
                yield x[i:i + 512]
            self.hub.mic_done.set()
            while True: time.sleep(1)
        import sounddevice as sd
        q = queue.Queue()
        with sd.InputStream(channels=1, dtype="float32", samplerate=16000, blocksize=512, callback=lambda d, *_: q.put(d[:, 0].copy())):
            while True: yield q.get()

    def partial(self):
        return np.concatenate(self.live) if self.live else np.zeros(0, np.float32)

    def run(self):
        for b in self.blocks():
            t = time.perf_counter()
            rms_db = float(20 * np.log10(np.sqrt(np.mean(b * b)) + 1e-9))
            self.vad.accept_waveform(b)
            vad_on = self.vad.is_speech_detected()
            self.hub.on_audio(t, b, vad_on, rms_db)
            if self.a.ptt:
                while not self.vad.empty(): self.vad.pop()  # VAD only informs perception in PTT mode
                if self.ptt_on: self.live.append(b)
                self.pre = (self.pre + [b])[-10:]
                continue
            if vad_on and not self.in_speech:
                self.in_speech, self.t_speech_start = True, t
                self.live = list(self.pre)
            if self.in_speech: self.live.append(b)
            self.pre = (self.pre + [b])[-10:]
            while not self.vad.empty():
                seg = np.array(self.vad.front.samples, np.float32); self.vad.pop()
                self.in_speech, self.live = False, []
                self.hub.on_utterance(seg, t - self.a.vad_silence, self.t_speech_start or t)

    def ptt_loop(self):
        """Enter starts, Enter ends. Capture continues in between, even while Kermit interjects."""
        while True:
            input("[Enter = start talking] ")
            self.live, self.t_speech_start, self.ptt_on = list(self.pre), time.perf_counter(), True
            input("[Enter = done] ")
            self.ptt_on = False
            t_end = time.perf_counter(); seg = self.partial(); self.live = []
            self.hub.on_utterance(seg, t_end, self.t_speech_start)


# ---------------------------------------------------------------- perception
class Perception:
    """v2 (people.PeopleThread, CONTRACT §5) when available, else the v1 single-face tracker (§4)."""

    def __init__(self, a, on_event):
        self.v2 = None; self.v1 = None; self.latest_v1 = {"present": False, "n_faces": 0}
        if not a.camera:
            return
        try:
            if a.perception == "v1": raise ImportError
            import people
            pa = people.parse_args(["--source", a.camera, "--hfov", str(a.hfov)] + a.perception_args)
            self.v2 = people.PeopleThread(pa, on_event=on_event); self.v2.start()
        except ImportError:
            import track
            self.v1 = track.parse_args(["--source", a.camera, "--hfov", str(a.hfov)]
                                       + (["--cam-pos", *map(str, a.cam_pos)] if a.cam_pos else [])
                                       + (["--cam-rpy", *map(str, a.cam_rpy)] if a.cam_rpy else []))
            threading.Thread(target=self._run_v1, daemon=True).start()

    def _run_v1(self):
        import track
        tr = track.Tracker(self.v1)
        for img in track.frames(self.v1.source, self.v1.fps):
            self.latest_v1 = tr.step(img)

    @property
    def active(self):
        return bool(self.v1 or self.v2)

    @property
    def gaze(self):
        return self.v2.gaze if self.v2 else self.latest_v1

    def context(self, visible):
        if self.v2:
            return f"{self.v2.context()}. visible: {visible}"
        m = self.latest_v1
        if not self.v1:
            return f"visible: {visible}"
        if not m.get("present"):
            return f"no one is in front of you right now. visible: {visible}"
        side = "to your left" if m["yaw_deg"] > 12 else "to your right" if m["yaw_deg"] < -12 else "right in front of you"
        n = m.get("n_faces", 1)
        who = "one person" if n <= 1 else f"{n} people, the closest one"
        return f"{who} {side}, about {m['dist_m']:.1f} metres away. visible: {visible}"

    def feed(self, t, b, active, rms_db):
        if self.v2:
            self.v2.feed_audio(t, b); self.v2.feed_vad(t, active, rms_db)

    def self_speaking(self, on):
        if self.v2: self.v2.set_self_speaking(on)


# ---------------------------------------------------------------- hub
def _words(s):
    return re.sub(r"[^a-z' ]", " ", s.lower()).split()


class Hub:
    def __init__(self, a):
        self.a = a
        self.mic_done = threading.Event()
        self.perf = kermit_voice.Performer(a)
        self.perf.stt()
        self.percep = Perception(a, self.on_event)
        self.speaker = Speaker(real=not a.no_speaker, on_speaking=self.percep.self_speaking)
        self.perf.on_audio = self._on_tts_audio
        self.replies, self.events = queue.Queue(), queue.Queue()
        self.busy = 0             # turns in flight (for --mic-wav exit)
        self.mic = Mic(a, self)

    # producers --------------------------------------------------------
    def on_audio(self, t, b, active, rms_db):
        self.percep.feed(t, b, active, rms_db)
        if self.a.vad_latest and int(t * 30) != int((t - 512 / 16000) * 30):
            line = json.dumps({"t": round(t, 3), "speech_active": bool(active), "rms_db": round(rms_db, 1)})
            open(self.a.vad_latest + ".tmp", "w").write(line); os.replace(self.a.vad_latest + ".tmp", self.a.vad_latest)

    def on_utterance(self, audio16, t_end, t_start, text=None, cancel=None):
        """A human turn ended. `text`: final transcript if the caller already has one (skips Parakeet).
        `cancel`: threading.Event; setting it stops the reply before its next sentence (speculative starts)."""
        self.replies.put((audio16, t_end, t_start, text, cancel))

    def on_event(self, ev):
        """Perception §5 event (from people.PeopleThread or --events-script). All are forwarded to BRAIN."""
        self.events.put(ev)

    # TTS → speaker ----------------------------------------------------
    def _on_tts_audio(self, k, chunk, seg):
        self.speaker.enqueue(seg["audio"].rsplit("_s", 1)[0], k, chunk, [w["w"] for w in seg["words"]],
                             lead_s=(len(chunk) / SR - seg["duration"]) if k else 0.0)

    # consumers --------------------------------------------------------
    def reply_loop(self):
        while True:
            audio16, t_end, t_start, text, cancel = self.replies.get(); self.busy += 1
            try:
                heard = text if text is not None else self.perf.transcribe(audio16)
                echo = self._is_echo(heard, t_start, t_end)
                emit({"event": "heard", "text": heard, "echo": echo, "dur_s": round(len(audio16) / 16000, 2)})
                if heard and not echo:
                    self.perf.a.context = self.percep.context(self.a.visible)
                    if cancel is None or not cancel.is_set():
                        self.perf.one(heard=heard, t_speech_end=t_end, cancel=cancel)
            except Exception as e:
                emit({"event": "error", "where": "reply", "error": repr(e)[:200]})
            finally:
                self.busy -= 1

    def event_loop(self):
        while True:
            ev = self.events.get(); self.busy += 1
            try:
                emit({"event": "perception", **ev})
                heard_partial = ""
                if ev.get("event") == "turned_away" and ev.get("while_speaking") and len(self.mic.live) > 8:
                    heard_partial = self.perf.transcribe(self.mic.partial())
                if ev.get("event") == "turned_away" and ev.get("while_speaking"):
                    self.speaker.interrupt()   # the interjection goes first; a pending reply would be stale
                self.perf.event_turn(ev, heard_partial, self.percep.context(self.a.visible))
            except Exception as e:
                emit({"event": "error", "where": "event", "error": repr(e)[:200]})
            finally:
                self.busy -= 1

    def gaze_loop(self):
        while True:
            time.sleep(1.0 / self.a.gaze_hz)
            if self.percep.active:
                emit({"event": "gaze", **self.percep.gaze})

    def script_loop(self):
        """--events-script: NDJSON lines {"t": seconds after start, "event": …} replayed on time (tests)."""
        t0 = time.perf_counter()
        for line in open(self.a.events_script):
            if not line.strip(): continue
            ev = json.loads(line); t = ev.pop("t")
            time.sleep(max(0.0, t0 + t - time.perf_counter()))
            ev["t"] = round(time.perf_counter(), 3)
            self.on_event(ev)

    def _is_echo(self, heard, t_start, t_end):
        """Kermit hearing himself: most of the utterance's words are words he was saying at the time."""
        hw = _words(heard)
        if not hw: return False
        said = set(_words(self.speaker.said_recently(t_start, t_end)))
        return sum(w in said for w in hw) / len(hw) >= 0.6

    def idle(self):
        return self.replies.empty() and self.events.empty() and self.busy == 0 and self.speaker.q.empty() and not self.speaker.speaking

    def run(self):
        self.speaker.start()
        for f in (self.reply_loop, self.event_loop, self.gaze_loop):
            threading.Thread(target=f, daemon=True).start()
        if self.a.events_script:
            threading.Thread(target=self.script_loop, daemon=True).start()
        self.mic.start()
        print(f"listening ({'push-to-talk' if self.a.ptt else 'VAD'}); BRAIN at {self.a.brain_url}", file=sys.stderr)
        if self.a.ptt and not self.a.mic_wav:
            self.mic.ptt_loop()
        while True:  # only reached with --mic-wav: exit once the WAV is done and everything has played
            if self.mic_done.wait(0.5) and self.idle():
                time.sleep(self.a.linger)
                if self.idle(): return


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camera", help="camera index / snapshot URL / video; omit for no perception")
    ap.add_argument("--perception", choices=["auto", "v1"], default="auto", help="auto = v2 (perception/people.py) if present")
    ap.add_argument("--hfov", type=float, default=78.0)
    ap.add_argument("--cam-pos", type=float, nargs=3); ap.add_argument("--cam-rpy", type=float, nargs=3)
    ap.add_argument("--gaze-hz", type=float, default=20.0)
    ap.add_argument("--visible", default="audience, laptop", help="appended to BRAIN context as 'visible: …'")
    ap.add_argument("--ptt", action="store_true")
    ap.add_argument("--vad-silence", type=float, default=0.5, help="end-of-utterance silence, s")
    ap.add_argument("--vad-latest", help="write mic VAD state {t, speech_active, rms_db} here at ~30 Hz")
    ap.add_argument("--mic-wav", help="test: feed this WAV into the pipeline in real time as if it were the mic")
    ap.add_argument("--events-script", help="test: NDJSON perception events, t = seconds after start")
    ap.add_argument("--no-speaker", action="store_true", help="headless: no output device (playback simulated in real time)")
    ap.add_argument("--linger", type=float, default=1.5, help="with --mic-wav: exit this long after everything is idle")
    # voice options (as kermit_voice.py)
    ap.add_argument("--out", default=os.path.join(HERE, "voice", "out"))
    ap.add_argument("--voice", default=os.environ.get("KERMIT_VOICE", "am_fenrir"))
    ap.add_argument("--no-frog", action="store_true")
    ap.add_argument("--tts-backend", default=os.environ.get("KERMIT_TTS_BACKEND", "torch"), choices=["torch", "mlx"])
    ap.add_argument("--device", default=None)
    ap.add_argument("--stt-provider", default=os.environ.get("KERMIT_STT_PROVIDER", "cpu"))
    ap.add_argument("--brain-url", default=os.environ.get("KERMIT_BRAIN_URL", "http://127.0.0.1:8765"))
    ap.add_argument("--llm", default=os.environ.get("KERMIT_LLM", "http://127.0.0.1:8080"))
    ap.add_argument("--brain-prompt", default=os.environ.get("KERMIT_BRAIN_PROMPT", "persona3"))
    ap.add_argument("--perception-args", nargs=argparse.REMAINDER, default=[], help="extra args for people.parse_args (must be last)")
    a = ap.parse_args()
    a.play, a.context = False, None
    Hub(a).run()
    sys.stdout.flush()
    os._exit(0)  # camera/audio threads hold native resources; a normal interpreter exit can abort in their destructors


if __name__ == "__main__":
    main()
