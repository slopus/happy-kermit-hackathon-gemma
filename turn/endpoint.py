"""Turn-taking for Happy Kermit: when has the human finished talking, and when are they talking over Kermit?

Silence alone is a bad cue (people pause mid-thought), so every mic block goes through:

  mic 16 kHz ─► [AEC3: Kermit's own voice removed, when playback reference is fed]
             ─► Silero VAD (32 ms)       ─► speech / pause state, trailing silence
             ─► turn-1-mini (160 ms)     ─► P(end of turn) from prosody, streaming, ~1 ms per step
             ─► Nemotron streaming ASR   ─► partial transcript every 160 ms (own thread)
  in a pause, every 100 ms:  Smart Turn v3.2 (audio, once per pause) + LiveKit text model (partial transcript,
             with Kermit's last line as context) + text form ─► fusion (logistic, weights in models/fusion.json)
             ─► candidate (speculative BRAIN start) ─► commit = end of turn  |  speech resumes ─► cancel

Policy (eot-bench style): commit when silence ≥ action_delay and p ≥ threshold, or silence ≥ timeout.
While Kermit is speaking, human speech on the echo-cancelled mic for ≥ bargein_ms is a barge-in.

Everything is CPU/ONNX; models live in kermit/turn/models (see MAC_RUNBOOK.md). Clock: time.perf_counter().
"""
import json, os, queue, re, threading, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.environ.get("KERMIT_TURN_MODELS", os.path.join(HERE, "models"))
VOICE_MODELS = os.environ.get("KERMIT_MODELS", os.path.join(HERE, "..", "voice", "models"))
NEMO_DIR = os.environ.get("KERMIT_NEMO_DIR", os.path.join(MODELS, "nemotron-160ms"))
SR = 16000
BLOCK = 512


def _ort(path, threads=1):
    import onnxruntime as ort
    o = ort.SessionOptions(); o.intra_op_num_threads = threads; o.inter_op_num_threads = 1; o.log_severity_level = 3
    return ort.InferenceSession(path, o, providers=["CPUExecutionProvider"])


# ------------------------------------------------------------------ models
class Silero:
    """Silero VAD v4 (h/c state) on 512-sample blocks → speech probability."""

    def __init__(self, path=os.path.join(VOICE_MODELS, "silero_vad.onnx")):
        self.s = _ort(path); self.reset()

    def reset(self):
        self.h = np.zeros((2, 1, 64), np.float32); self.c = self.h.copy()

    def __call__(self, b):
        p, self.h, self.c = self.s.run(None, {"x": b.reshape(1, -1).astype(np.float32), "h": self.h, "c": self.c})
        return float(p.reshape(-1)[0])


class Turn1Mini:
    """p99lab turn-1-mini streaming graph: 160 ms steps, P(end of turn) per 20 ms frame (Apache-2.0)."""
    STEP = 2560
    NAMES = ["audio", "conv1", "conv2", "k", "v", "k_valid", "h6"]

    def __init__(self, path=os.path.join(MODELS, "turn1mini/onnx/turn-1-mini.step.onnx")):
        self.s = _ort(path); self.outs = ["probs"] + [f"new_state_{n}" for n in self.NAMES]; self.reset()

    def reset(self):
        self.st = {i.name: np.zeros(i.shape, np.float32) for i in self.s.get_inputs() if i.name != "audio"}
        self.buf = np.zeros(0, np.float32); self.p = 0.0

    def push(self, b):
        self.buf = np.concatenate([self.buf, b])
        while len(self.buf) >= self.STEP:
            x, self.buf = self.buf[:self.STEP], self.buf[self.STEP:]
            pr, *new = self.s.run(self.outs, {"audio": x.reshape(1, -1), **self.st})
            self.st = {f"state_{n}": v for n, v in zip(self.NAMES, new)}
            self.p = float(pr[-1])
        return self.p


class SmartTurn:
    """Pipecat Smart Turn v3.2 (BSD-2): last 8 s of audio → P(complete). Whisper log-mel in numpy."""
    N = 8 * SR

    def __init__(self, path=os.path.join(MODELS, "smart-turn/smart-turn-v3.2-cpu.onnx")):
        self.s = _ort(path, threads=2)
        self.mel = np.load(os.path.join(os.path.dirname(path), "mel80.npy"))
        self.win = np.hanning(401)[:-1].astype(np.float32)

    def features(self, a):
        a = a[-self.N:]; a = np.pad(a, (self.N - len(a), 0))
        a = (a - a.mean()) / np.sqrt(a.var() + 1e-7)
        x = np.pad(a, (200, 200), mode="reflect")
        n = 1 + (len(x) - 400) // 160
        fr = np.lib.stride_tricks.as_strided(x, (n, 400), (x.strides[0] * 160, x.strides[0])) * self.win
        mag = np.abs(np.fft.rfft(fr, axis=1)) ** 2
        lm = np.log10(np.maximum(self.mel @ mag[:-1].T, 1e-10))
        lm = np.maximum(lm, lm.max() - 8.0)
        return ((lm + 4.0) / 4.0).astype(np.float32)

    def __call__(self, a):
        return float(self.s.run(None, {"input_features": self.features(a)[None]})[0].reshape(-1)[0])


def norm_text(t):
    import unicodedata
    t = unicodedata.normalize("NFKC", (t or "").lower())
    t = "".join(ch for ch in t if not (unicodedata.category(ch).startswith("P") and ch not in "'-"))
    return re.sub(r"\s+", " ", t).strip()


class TextTurn:
    """LiveKit turn-detector v0.4.1-intl (Qwen2.5-0.5B distilled, int8): P(user turn complete) from the partial
    transcript, with Kermit's last line as context. Cached per prompt."""

    def __init__(self, d=os.path.join(MODELS, "livekit-ml")):
        from tokenizers import Tokenizer
        self.tok = Tokenizer.from_file(os.path.join(d, "tokenizer.json"))
        self.s = _ort(os.path.join(d, "onnx/model_q8.onnx"), threads=4)
        self.cache = {}

    @staticmethod
    def _prompt(user_text, prev_assistant):
        u, a = norm_text(user_text), norm_text(prev_assistant)
        return ((f"<|im_start|>assistant\n{a}<|im_end|>\n" if a else "") + f"<|im_start|>user\n{u}") if u else None

    def cached(self, user_text, prev_assistant=""):
        """Score if already computed (0.0 for empty text), else None."""
        p = self._prompt(user_text, prev_assistant)
        return 0.0 if p is None else self.cache.get(p)

    def __call__(self, user_text, prev_assistant=""):
        p = self._prompt(user_text, prev_assistant)
        if p is None: return 0.0
        if p not in self.cache:
            ids = self.tok.encode(p, add_special_tokens=False).ids[-128:]
            self.cache[p] = float(self.s.run(None, {"input_ids": np.array([ids], np.int64)})[0][0, -1])
            if len(self.cache) > 512: self.cache.pop(next(iter(self.cache)))
        return self.cache[p]


class StreamingASR(threading.Thread):
    """Nemotron-3.5 streaming 0.6b 160 ms (sherpa-onnx, CPU). feed() blocks from the mic thread; text() returns the
    partial of the current utterance; cut() finalises it and starts a new one."""

    def __init__(self, d=NEMO_DIR, threads=4, chunk=0.16, sync=False):
        super().__init__(daemon=True)
        self.sync = sync  # offline replay: decode inline in feed() instead of on the thread
        import sherpa_onnx
        self.rec = sherpa_onnx.OnlineRecognizer.from_transducer(
            encoder=f"{d}/encoder.int8.onnx", decoder=f"{d}/decoder.int8.onnx", joiner=f"{d}/joiner.int8.onnx",
            tokens=f"{d}/tokens.txt", num_threads=threads, provider="cpu", decoding_method="greedy_search")
        self.n = int(chunk * SR); self.q = queue.Queue(); self.lock = threading.Lock()
        self.s = self.rec.create_stream(); self._text = ""; self.lag_ms = 0.0; self._final = ""
        self.t_done = 0.0; self.t_change = 0.0   # capture time of the last decoded block / of the last text change
        self.max_behind, self.skips = float(os.environ.get("KERMIT_ASR_MAX_BEHIND", 1.5)), 0
        self.skip_words = 0
        self.buf, self.nbuf = [], 0
        if not sync: self.start()

    def feed(self, b, t):
        if self.sync: self._step(b, t)
        else: self.q.put((b, t))

    def cut(self):
        """End of the current utterance: returns its text, starts a fresh stream (the queue keeps flowing)."""
        ev = threading.Event()
        if self.sync: self._step("cut", ev); return self._final
        cur = self.text()
        self.q.put(("cut", ev)); ev.wait(0.05)  # don't hold the commit for a backlogged decoder
        return self._final if ev.is_set() else cur

    def restart(self, blocks, t):
        """A new turn opens: words decoded before it (residual echo of Kermit, noise) are no longer part of the
        turn's text. The decoder stream itself keeps running (restarting it costs accuracy and ~0.3 s of latency
        at the start of every turn); text() just skips the words that were there at the turn's onset."""
        self.skip_words = len(self._text.split())

    def text(self):
        w = self._text.split()
        return " ".join(w[self.skip_words:]) if len(w) > self.skip_words else ""

    def run(self):
        while True:
            b, t = self.q.get()
            if not isinstance(b, str) and self.q.qsize() * BLOCK / SR > self.max_behind:
                # CPU-starved: the partial is only an early cue (the final text comes from TDT at the pause), so
                # jump to the newest audio instead of falling further behind real time
                items = [(b, t)]
                while True:
                    try: items.append(self.q.get_nowait())
                    except queue.Empty: break
                keep = {id(it) for it in [it for it in items if not isinstance(it[0], str)][-int(0.5 * SR / BLOCK):]}
                with self.lock:
                    self.s = self.rec.create_stream(); self.buf, self.nbuf = [], 0; self._text = ""; self.skip_words = 0
                self.skips += 1
                for it in items:
                    if isinstance(it[0], str) or id(it) in keep: self._step(*it)
                continue
            self._step(b, t)

    def _step(self, b, t):
        if isinstance(b, str) and b == "restart":
            blocks, t0 = t
            with self.lock:
                self.s = self.rec.create_stream(); self.buf, self.nbuf = list(blocks), sum(len(x) for x in blocks)
                self._text = ""; self.t_change = t0
            return
        if isinstance(b, str):
            with self.lock:
                if self.buf: self.s.accept_waveform(SR, np.concatenate(self.buf)); self.buf, self.nbuf = [], 0
                self.s.accept_waveform(SR, np.zeros(int(0.32 * SR), np.float32))  # flush the right context
                while self.rec.is_ready(self.s): self.rec.decode_stream(self.s)
                self._final = " ".join(self.rec.get_result(self.s).strip().split()[self.skip_words:])
                self.s = self.rec.create_stream(); self._text = ""; self.t_change = self.t_done; self.skip_words = 0
            t.set(); return
        self.buf.append(b); self.nbuf += len(b)
        if self.nbuf >= self.n:
            with self.lock:
                self.s.accept_waveform(SR, np.concatenate(self.buf)); self.buf, self.nbuf = [], 0
                while self.rec.is_ready(self.s): self.rec.decode_stream(self.s)
                txt = self.rec.get_result(self.s).strip()
                if txt != self._text: self.t_change = t
                self._text = txt; self.t_done = t
            if not self.sync: self.lag_ms = (time.perf_counter() - t) * 1000


class EchoCanceller:
    """WebRTC AEC3 (+ noise suppression, high-pass) via livekit's AudioProcessingModule, 10 ms frames at 16 kHz.

    The playback reference is kept on a perf_counter timeline (far(x, sr, t) = samples handed to the device at
    t), and each mic block pulls exactly the reference written during its capture span right before it is
    processed. So render and capture stay aligned however late the mic thread runs (AEC3 itself only has to find
    the acoustic + device delay, ~30-150 ms)."""
    F = 160

    def __init__(self, delay_ms=60, ns=True):
        from livekit import rtc
        self.rtc = rtc
        self.apm = rtc.AudioProcessingModule(echo_cancellation=True, noise_suppression=ns, high_pass_filter=True)
        self.apm.set_stream_delay_ms(delay_ms)
        self.lock = threading.Lock()
        self.ref = np.zeros(SR * 4, np.float32)   # ring: reference sample at absolute index n = round(t * SR)
        self.ref_end = None                        # absolute index after the last written reference sample
        self.cap_end = None                        # absolute index up to which reference was pulled
        self.near_in = np.zeros(0, np.float32); self.near_out = np.zeros(self.F, np.float32)
        self.render = np.zeros(0, np.float32); self.last_ref_db = -120.0

    def _frame(self, x):
        d = np.clip(x * 32767, -32768, 32767).astype(np.int16)
        return self.rtc.AudioFrame(d.tobytes(), SR, 1, len(d))

    def far(self, x, sr=SR, t=None):
        if sr != SR:
            from scipy.signal import resample_poly
            from math import gcd
            g = gcd(sr, SR); x = resample_poly(x, SR // g, sr // g).astype(np.float32)
        t = time.perf_counter() if t is None else t
        with self.lock:
            n0 = int(round(t * SR))
            if self.ref_end is not None and abs(n0 - self.ref_end) < SR // 20: n0 = self.ref_end  # contiguous
            N = len(self.ref)
            idx = (n0 + np.arange(len(x))) % N
            self.ref[idx] = x
            self.ref_end = n0 + len(x)

    def _pull(self, t_end, n):
        """Reference samples for the capture span ending at t_end (zeros where nothing was playing)."""
        e = int(round(t_end * SR)); s = e - n
        if self.cap_end is not None and 0 < s - self.cap_end < SR // 10: s = self.cap_end  # keep it gap-free
        out = np.zeros(e - s, np.float32)
        if self.ref_end is not None:
            N = len(self.ref); lo = max(s, self.ref_end - N + 1)
            hi = min(e, self.ref_end)
            if hi > lo:
                out[lo - s:hi - s] = self.ref[np.arange(lo, hi) % N]
                self.ref[np.arange(lo, hi) % N] = 0.0
        self.cap_end = e
        return out

    def near(self, b, t=None):
        """Mic block captured ending at t → same-length cleaned block (10 ms framing adds ≤10 ms delay)."""
        t = time.perf_counter() if t is None else t
        with self.lock:
            r = self._pull(t, len(b))
            self.last_ref_db = float(20 * np.log10(np.sqrt(np.mean(r * r)) + 1e-9))
            self.render = np.concatenate([self.render, r])
            self.near_in = np.concatenate([self.near_in, b])
            out = [self.near_out]
            while len(self.near_in) >= self.F:
                if len(self.render) >= self.F:     # lockstep: one 10 ms render frame per capture frame
                    self.apm.process_reverse_stream(self._frame(self.render[:self.F])); self.render = self.render[self.F:]
                fr = self._frame(self.near_in[:self.F]); self.near_in = self.near_in[self.F:]
                self.apm.process_stream(fr)
                out.append(np.frombuffer(bytes(fr.data), np.int16).astype(np.float32) / 32768)
            y = np.concatenate(out)
            self.near_out = y[len(b):]
            return y[:len(b)]


FILLERS = {"um", "uh", "so", "and", "but", "or", "like", "the", "a", "to", "of", "about", "with", "because", "if",
           "that", "is", "i", "my", "you", "your", "for", "in", "on", "what", "how", "when", "then", "well", "kermit"}


def text_feats(t):
    t = (t or "").strip()
    w = re.sub(r"[^\w' ]", " ", t.lower()).split()
    return dict(n_words=min(len(w), 30), ends_q=float(t.endswith("?")), ends_p=float(t.endswith((".", "!"))),
                ends_comma=float(t.endswith((",", "...", "…"))), last_filler=float(bool(w) and w[-1] in FILLERS))


class Fusion:
    """Gradient-boosted trees over the live cues, exported to JSON by bench/fit_live.py (trained on eot-bench +
    the Kermit demo set). Also carries the chosen policy (threshold, action_delay, timeout, spec_threshold)."""

    def __init__(self, path=os.path.join(MODELS, "fusion.json")):
        j = json.load(open(path))
        self.feats, self.base, self.policy = j["features"], j["baseline"], j.get("policy", {})
        self.text = j.get("text", "stream")   # which transcript the model was trained on: "tdt" (at the pause) or "stream"
        self.trees = [{k: np.asarray(v) for k, v in t.items()} for t in j["trees"]]

    def __call__(self, f):
        x = [float(f.get(k, np.nan)) for k in self.feats]; raw = self.base
        for t in self.trees:
            i = 0
            while not t["is_leaf"][i]:
                v = x[t["feature_idx"][i]]
                i = t["left"][i] if (t["missing_go_to_left"][i] if v != v else v <= t["num_threshold"][i]) else t["right"][i]
            raw += t["value"][i]
        return float(1 / (1 + np.exp(-raw)))


# ------------------------------------------------------------------ the endpointer
class Endpointer:
    """Feed it every mic block; it calls back:
      on_turn_end(audio16, text, t_end, t_start)   committed end of the human's turn (t_end = last speech)
      on_candidate(audio16, text, t_end, t_start)  likely end, before commit (speculative start); may be followed
                                                   by on_cancel() if the human carries on, or by on_turn_end
      on_cancel()                                  the candidate (or a just-committed turn) was premature
      on_bargein(t)                                the human started talking while Kermit speaks
      on_partial(text)                             transcript so far (for interjections / UI)
    """

    def __init__(self, on_turn_end, on_candidate=None, on_cancel=None, on_bargein=None, on_partial=None,
                 aec=True, asr=True, speculative=True, threshold=None, action_delay=None, timeout=None,
                 spec_threshold=None, min_speech=0.25, vad_on=0.5, vad_off=0.35, bargein_ms=200, log=None, sync=False, tdt=None):
        self.cb = dict(turn_end=on_turn_end, candidate=on_candidate, cancel=on_cancel, bargein=on_bargein, partial=on_partial)
        self.vad, self.t1m, self.smart = Silero(), Turn1Mini(), SmartTurn()
        self.text_model = TextTurn()
        self.tdt = None
        if tdt is None: tdt = os.environ.get("KERMIT_TURN_TDT", "1") == "1"
        if tdt:
            try:
                import sherpa_onnx
                d = os.path.join(VOICE_MODELS, "sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-int8")
                self.tdt = sherpa_onnx.OfflineRecognizer.from_transducer(
                    encoder=f"{d}/encoder.int8.onnx", decoder=f"{d}/decoder.int8.onnx", joiner=f"{d}/joiner.int8.onnx",
                    tokens=f"{d}/tokens.txt", num_threads=4, provider="cpu", decoding_method="greedy_search", model_type="nemo_transducer")
            except Exception as e:
                print(f"endpoint: no TDT at the pause ({e!r:.120})", flush=True)
        self.text_is_final = False
        self.jobs = queue.Queue()
        threading.Thread(target=self._worker, daemon=True).start()
        self.fusion = Fusion()
        pol = self.fusion.policy
        self.threshold = threshold if threshold is not None else pol.get("threshold", 0.5)
        self.action_delay = action_delay if action_delay is not None else pol.get("action_delay", 0.3)
        self.timeout = timeout if timeout is not None else pol.get("timeout", 1.5)
        self.spec_threshold = spec_threshold if spec_threshold is not None else pol.get("spec_threshold", self.threshold * 0.6)
        self.asr = StreamingASR(sync=sync) if asr else None
        self.sync = sync
        self.log = log
        self.aec = None
        if aec:
            try: self.aec = EchoCanceller()
            except Exception as e: self._log({"event": "turn_warn", "aec": f"unavailable: {e!r}"[:160]})
        self.speculative, self.min_speech = speculative, min_speech
        self.vad_on, self.vad_off, self.bargein_blocks = vad_on, vad_off, max(1, round(bargein_ms / 32))
        self.log = log
        self.prev_assistant = ""
        self.kermit_speaking, self.kermit_off_t = False, None
        self.resume_window = 1.0  # speech this soon after a commit = the commit was a cut-in: cancel and merge
        self.last_commit = None   # (t, audio, text) of the last committed turn
        self.recent = []          # pre-roll blocks
        self.utt = []             # blocks of the current human turn (across mid-turn pauses)
        self.in_speech = False; self.speech_run = 0; self.quiet_run = 0
        self.t_start = None; self.t_last_speech = None; self.speech_s = 0.0
        self.pause = None         # dict for the current pause: t0, smart, cand, next_eval
        self.bargein_run = 0; self.bargein_fired = False
        self.hist_p, self.hist_db, self.ref_hist, self.coupling = [], [], [], []
        self.echo_gate_db = 6.0   # barge-in needs the cleaned mic ≥ this many dB above the learned echo residual
        self.stalled, self.prefix_text, self.t_onset, self.opened_over_kermit = False, "", 0.0, False
        self.merge_cand = None   # barge-in this soon after a commit merges the turns (0 = off)
        self.bargein_merge_s = float(os.environ.get("KERMIT_BARGEIN_MERGE_S", 3.0))
        self.max_utt = 20.0       # s: a turn longer than this is forced to a decision (babble, monologue)
        self.lock = threading.Lock()

    # --- inputs from the hub -----------------------------------------------------------------------------------
    def set_context(self, prev_assistant):
        self.prev_assistant = prev_assistant or ""

    def set_kermit_speaking(self, on):
        """Speaker started/stopped. The room's echo tail outlives playback, so 'speaking' lasts 0.6 s longer."""
        if on: self.kermit_speaking, self.kermit_off_t = True, None
        else: self.kermit_off_t = time.perf_counter()

    def _speaking(self, t):
        if self.kermit_speaking and self.kermit_off_t is not None and t > self.kermit_off_t + 0.6:
            self.kermit_speaking = False; self.bargein_run = 0; self.bargein_fired = False
        return self.kermit_speaking

    def far(self, x, sr, t=None):
        """Playback reference: samples handed to the output device at perf_counter t (default now)."""
        if self.aec: self.aec.far(np.asarray(x, np.float32).reshape(-1), sr, t)

    def _log(self, ev):
        if self.log: self.log(ev)

    # --- per mic block ---------------------------------------------------------------------------------------
    def _levels(self, x, p):
        """Adaptive speech threshold: Silero's probability floor and the level floor over the last ~4 s
        (babble lifts both). Returns (speech_on threshold, speech_off threshold, block dB, dB over noise floor)."""
        db = float(20 * np.log10(np.sqrt(np.mean(x * x)) + 1e-9))
        self.hist_p.append(p); self.hist_db.append(db)
        if len(self.hist_p) > 125: self.hist_p.pop(0); self.hist_db.pop(0)
        if len(self.hist_p) >= 30:
            fl_p = float(np.percentile(self.hist_p, 15)); fl_db = float(np.percentile(self.hist_db, 15))
        else:
            fl_p, fl_db = 0.0, -70.0
        on = min(0.92, max(self.vad_on, fl_p + 0.35)); off = min(0.85, max(self.vad_off, fl_p + 0.2))
        return on, off, db, db - fl_db

    def _echo_gate(self, t, x, out_db):
        """Residual-echo gate for barge-in: True if the cleaned mic is clearly louder than the echo the AEC leaves
        at the current playback level. The coupling (residual dB − reference dB) is learned while Kermit talks and
        nobody barges in (90th percentile over the last ~10 s)."""
        if not self.aec or self.echo_gate_db is None: return True
        ref_db = self.aec.last_ref_db
        self.ref_hist.append(ref_db)
        if len(self.ref_hist) > 8: self.ref_hist.pop(0)
        r = max(self.ref_hist)                      # ~250 ms: the room tail outlives the reference
        if r < -60: return True                     # nothing playing recently
        excess = out_db - r
        if not self.bargein_run:                     # learn from blocks that are not (yet) a barge-in
            self.coupling.append(excess)
            if len(self.coupling) > 300: self.coupling.pop(0)
        c = float(np.percentile(self.coupling, 90)) if len(self.coupling) >= 20 else -10.0
        return excess > c + self.echo_gate_db

    def feed(self, t, b):
        """t = perf_counter at the END of this block's capture; b = 512 float32 samples @16 kHz.
        Returns the speech probability (Silero, after AEC)."""
        # The AEC runs on every block (so it stays adapted), but its output is used only while Kermit is audible:
        # outside that, its suppressor and noise suppression only cost the ASR words.
        xa = self.aec.near(b, t) if self.aec else b
        x = xa if self._speaking(t) else b
        p = self.vad(x)
        self.t1m.push(x)
        if self.asr: self.asr.feed(x, t)
        self.recent = (self.recent + [x])[-16:]
        on, off, db, snr = self._levels(x, p)
        speech = p >= (off if self.in_speech else on) and snr > 3.0
        speaking = self._speaking(t)

        # barge-in: sustained speech on the echo-cancelled mic, clearly above the residual echo, while Kermit talks
        if speaking:
            loud = self._echo_gate(t, x, db)
            self.bargein_run = self.bargein_run + 1 if (p >= max(on, 0.6) and loud) else 0
            if self.bargein_run >= self.bargein_blocks and not self.bargein_fired:
                self.bargein_fired = True
                self._log({"event": "bargein_detected", "t": round(t, 3), "p": round(p, 3)})
                if self.cb["bargein"]: self.cb["bargein"](t)
                mc = self.merge_cand
                if mc and t - mc[0] <= self.bargein_merge_s and self.t_start is not None:
                    # Kermit answered a mid-turn pause ("I respect you. | It's just hard…") and the human carried
                    # on over him: the new turn includes the interrupted one, so BRAIN hears the whole thought
                    self.utt = list(mc[1]) + self.utt; self.t_start = mc[3]; self.speech_s += mc[4]
                    self._log({"event": "turn_merge", "t": round(t, 3), "after_commit_s": round(t - mc[0], 2), "prev": mc[2]})
                self.merge_cand = None
            self.last_commit = None   # Kermit is audible: a later resume is a new turn, never a merge
        # The human's audio is never dropped: speech opens / continues a turn whether or not Kermit is talking.
        # A turn that opens while Kermit talks may be residual echo; the hub's transcript echo check filters it,
        # and it gets no speculative start (so it can never cancel the reply that is playing).

        # words stopped arriving: in continuous babble Silero can stay "on" for ever. If the partial transcript
        # has not grown for 0.8 s while we are "in speech", treat it as a pause that started when the words did.
        # (on the ASR's own clock, so a decoder running behind real time is not mistaken for a stall)
        text = self.asr.text() if self.asr else ""
        a = self.asr
        if (self.in_speech and a and self.t_start is not None and text and self.pause is None
                and t - a.t_done < 0.4 and a.t_done - max(a.t_change, self.t_onset) >= 1.0 and t - self.t_start > 1.5):
            self.in_speech = False; self.speech_run = 0; self.stalled = True
            self.pause = {"t0": a.t_change, "next_eval": t, "stall": True, "text0": text}
            self._log({"event": "turn_stall", "t": round(t, 3), "since_s": round(a.t_done - a.t_change, 2)})

        if speech and not getattr(self, "stalled", False):
            self.speech_run += 1; self.quiet_run = 0
            if not self.in_speech and self.speech_run >= 3:   # ~96 ms of speech: onset
                self.in_speech = True; self.t_onset = t
                lc = self.last_commit
                if self.t_start is None and lc and t - lc[0] <= self.resume_window and not speaking:
                    # the human carried on right after a commit, before Kermit made a sound: that commit was a
                    # cut-in. Cancel the reply and continue the same turn (audio + text so far are kept).
                    self._log({"event": "turn_cancel", "t": round(t, 3), "after_commit_s": round(t - lc[0], 3), "merged": True})
                    if self.cb["cancel"]: self.cb["cancel"]()
                    self.t_start, self.utt, self.speech_s = lc[3], list(lc[1]), lc[4]
                    self.prefix_text = lc[2]
                self.last_commit = None
                if self.t_start is None:
                    self.t_start = t - 3 * BLOCK / SR; self.utt = list(self.recent[-6:-1]); self.speech_s = 0.0
                    self.opened_over_kermit = speaking
                    if self.asr: self.asr.restart(self.utt + [x], t)
                if self.pause is not None:                   # speech resumed inside a pause: not the end
                    if self.pause.get("cand"):
                        self._log({"event": "turn_cancel", "t": round(t, 3), "after_s": round(t - self.pause["t0"], 3)})
                        if self.cb["cancel"]: self.cb["cancel"]()
                    self.pause = None
            if self.t_start is not None: self.speech_s += BLOCK / SR
            self.t_last_speech = t
        elif speech and self.stalled:
            if self.pause is None or text != self.pause.get("text0", text):  # new words: really talking again
                self.stalled = False
                if self.pause is not None:
                    if self.pause.get("cand"):
                        self._log({"event": "turn_cancel", "t": round(t, 3), "after_s": round(t - self.pause["t0"], 3)})
                        if self.cb["cancel"]: self.cb["cancel"]()
                    self.pause = None
                self.in_speech = True; self.t_last_speech = t
        else:
            self.speech_run = 0; self.quiet_run += 1
            if self.in_speech and self.quiet_run >= 2:       # ~64 ms quiet: pause starts
                self.in_speech = False
                if self.speech_s >= self.min_speech:
                    self.pause = {"t0": self.t_last_speech, "next_eval": self.t_last_speech + 0.1}
                elif self.pause is None:
                    self.t_start = None; self.utt = []         # a click / cough: drop it
        if self.t_start is not None:
            self.utt.append(x)
            if len(self.utt) * BLOCK / SR > self.max_utt and self.pause is None:   # hard cap: force a decision
                self.pause = {"t0": t, "next_eval": t, "forced": True}
                self._log({"event": "turn_maxlen", "t": round(t, 3)})
        if self.pause is not None and not self.pause.get("committed") and t >= self.pause["next_eval"]:
            self._evaluate(t)
        if self.cb["partial"] and self.asr and self.in_speech and int(t * 4) != int((t - BLOCK / SR) * 4):
            self.cb["partial"](text)
        return p

    def _pause_job(self, pa, audio, prev):
        """Worker thread, once per pause at 0.2 s of silence: Smart Turn on the last 8 s, Parakeet TDT on the whole
        utterance (the text then includes the last word, which the streaming partial is still missing), and the
        LiveKit text model on that text. Results land in the pause dict; the mic thread never waits for them."""
        t0 = time.perf_counter()
        pa["smart"] = self.smart(audio[-SmartTurn.N:]); pa["smart_ms"] = round((time.perf_counter() - t0) * 1000)
        if self.tdt is not None:
            t1 = time.perf_counter()
            s = self.tdt.create_stream(); s.accept_waveform(SR, audio); self.tdt.decode_stream(s)
            txt = s.result.text.strip(); pa["tdt_ms"] = round((time.perf_counter() - t1) * 1000)
            pa["lk_tdt"] = self.text_model(txt, prev) if txt else 0.0
            pa["tdt"] = txt

    def _worker(self):
        while True:
            job = self.jobs.get()
            try: job()
            except Exception as e: self._log({"event": "turn_warn", "worker": repr(e)[:160]})

    def _features(self, t, sil):
        pa = self.pause
        if "job" not in pa and sil >= 0.2 - 1e-3:
            audio = np.concatenate(self.utt) if self.utt else np.zeros(SR, np.float32)
            pa["job"] = True; prev = self.prev_assistant
            if self.sync: self._pause_job(pa, audio, prev)
            else: self.jobs.put(lambda: self._pause_job(pa, audio, prev))
        if "tdt" in pa and self.fusion.text == "tdt":
            text, lk = pa["tdt"], pa["lk_tdt"]
        else:
            text = self.asr.text() if self.asr else ""
            lk = self.text_model.cached(text, self.prev_assistant)
            if lk is None:                              # score the partial off-thread; use the last score meanwhile
                if not pa.get("lk_pending"):
                    pa["lk_pending"] = True
                    def job(pa=pa, text=text, prev=self.prev_assistant):
                        pa["lk_partial"] = self.text_model(text, prev); pa["lk_pending"] = False
                    job() if self.sync else self.jobs.put(job)
                lk = pa.get("lk_partial", 0.0)
        f = dict(silence_dur=sil, t1m=self.t1m.p, smart1=pa.get("smart", 0.5), lk_nemo=lk, **text_feats(text))
        return f, text

    def _evaluate(self, t):
        pa = self.pause; sil = t - pa["t0"]
        pa["next_eval"] = t + 0.1
        f, text = self._features(t, sil)
        p = self.fusion(f)
        pa.setdefault("trace", []).append((round(sil, 2), round(p, 3)))
        audio = np.concatenate(self.utt) if self.utt else np.zeros(0, np.float32)
        if (self.speculative and not pa.get("cand") and sil >= 0.1 and p >= self.spec_threshold
                and not (self.opened_over_kermit and not self.bargein_fired) and not self._speaking(t)):
            pa["cand"] = True
            self._log({"event": "turn_candidate", "t": round(t, 3), "sil": round(sil, 3), "p": round(p, 3), "text": text})
            self.text_is_final = "tdt" in pa
            if self.cb["candidate"]: self.cb["candidate"](audio, text, pa["t0"], self.t_start)
        if (sil >= self.action_delay and p >= self.threshold) or sil >= self.timeout:
            pa["committed"] = True
            streamed = self.asr.cut() if self.asr else text
            self.text_is_final = "tdt" in pa          # TDT on the whole utterance: the hub can skip re-transcribing
            final = pa["tdt"] if self.text_is_final else streamed
            if self.prefix_text and not self.text_is_final:   # TDT already covers the merged audio
                final = (self.prefix_text + " " + (final or text)).strip()
            self.prefix_text = ""
            self._log({"event": "turn_end", "t": round(t, 3), "sil": round(sil, 3), "p": round(p, 3),
                       "by": "model" if p >= self.threshold else "timeout", "text": final,
                       "asr_behind_ms": round((t - self.asr.t_done) * 1000) if self.asr else None,
                       "asr_skips": self.asr.skips if self.asr else None, "text_src": "tdt" if "tdt" in pa else "stream", "feats": {k: round(v, 3) for k, v in f.items()},
                       "ms": {k: pa.get(k) for k in ("smart_ms", "tdt_ms")}})
            self.last_commit = (t, list(self.utt), final or text, self.t_start, self.speech_s)
            self.merge_cand = self.last_commit
            self.cb["turn_end"](audio, final or text, pa["t0"], self.t_start)
            self.t_start = None; self.utt = []; self.speech_s = 0.0; self.pause = None; self.stalled = False
            self.t1m.reset()
