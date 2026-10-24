"""Speech in: Parakeet TDT 0.6b v2 via sherpa-onnx, Silero VAD endpointing, optional push-to-talk.

Same runtime on Linux and Mac: sherpa-onnx int8 on CPU (the Mac's P-cores do a 3 s command in well
under 200 ms; CoreML EP in onnxruntime does not help transducers). On the Linux dev box, provider
"cuda" with the fp16 export is ~20 ms.

Also used as a forced aligner: `words_from_audio()` returns word timings for any TTS output that
lacks its own (pre-rendered lines, other TTS engines).
"""
import os, queue, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.environ.get("KERMIT_MODELS", os.path.join(HERE, "models"))


def recognizer(provider="cpu", threads=4):
    import sherpa_onnx
    q = "fp16" if provider == "cuda" else "int8"
    d = os.path.join(MODELS, f"sherpa-onnx-nemo-parakeet-tdt-0.6b-v2-{q}")
    return sherpa_onnx.OfflineRecognizer.from_transducer(
        encoder=f"{d}/encoder.{q}.onnx", decoder=f"{d}/decoder.{q}.onnx", joiner=f"{d}/joiner.{q}.onnx",
        tokens=f"{d}/tokens.txt", num_threads=threads, provider=provider,
        decoding_method="greedy_search", model_type="nemo_transducer")


def transcribe(rec, audio, sr=16000):
    s = rec.create_stream(); s.accept_waveform(sr, np.asarray(audio, np.float32)); rec.decode_stream(s)
    return s.result


def words_from_audio(rec, audio, sr):
    """ASR word timings: BPE tokens ('▁' or leading space = new word) → [{"w","t0","t1"}]."""
    from scipy.signal import resample_poly
    a16 = resample_poly(audio, 16000, sr) if sr != 16000 else audio
    r = transcribe(rec, a16)
    words, ts = [], list(r.timestamps)
    for i, (tok, t) in enumerate(zip(r.tokens, ts)):
        nxt = ts[i + 1] if i + 1 < len(ts) else len(a16) / 16000
        if tok.startswith((" ", "▁")) or not words:
            words.append({"w": tok.strip(" ▁"), "t0": round(t, 3), "t1": round(min(nxt, t + 0.6), 3)})
        else:
            words[-1]["w"] += tok; words[-1]["t1"] = round(min(nxt, t + 0.6), 3)
    return words


class Listener:
    """Mic → utterances. VAD mode: speak, ~0.5 s silence ends the turn. PTT mode: press Enter to
    start, Enter to stop. Yields (float32 16 kHz audio, t_speech_end perf_counter)."""

    def __init__(self, mode="vad", min_silence=0.5):
        import sherpa_onnx
        self.mode, self.vad_min_silence = mode, min_silence
        cfg = sherpa_onnx.VadModelConfig()
        cfg.silero_vad.model = os.path.join(MODELS, "silero_vad.onnx")
        cfg.silero_vad.min_silence_duration = min_silence
        cfg.silero_vad.min_speech_duration = 0.25
        cfg.silero_vad.threshold = 0.5
        cfg.sample_rate = 16000
        self.vad = sherpa_onnx.VoiceActivityDetector(cfg, buffer_size_in_seconds=60)

    def utterances(self):
        import sounddevice as sd
        q = queue.Queue()
        with sd.InputStream(channels=1, dtype="float32", samplerate=16000, blocksize=512,
                            callback=lambda d, *_: q.put(d[:, 0].copy())):
            if self.mode == "ptt":
                while True:
                    input("[press Enter, speak, press Enter] ")
                    while not q.empty(): q.get()
                    import threading; done = threading.Event()
                    threading.Thread(target=lambda: (input(), done.set()), daemon=True).start()
                    buf = []
                    while not done.is_set(): buf.append(q.get())
                    yield np.concatenate(buf), time.perf_counter()
            else:
                while True:
                    self.vad.accept_waveform(q.get())
                    while not self.vad.empty():
                        seg = np.array(self.vad.front.samples, np.float32); self.vad.pop()
                        yield seg, time.perf_counter()  # VAD fires min_silence after speech ended


if __name__ == "__main__":  # python stt.py file.wav  |  python stt.py --mic [--ptt]
    import soundfile as sf
    rec = recognizer(os.environ.get("KERMIT_STT_PROVIDER", "cpu"))
    if sys.argv[1] == "--mic":
        for a, _ in Listener("ptt" if "--ptt" in sys.argv else "vad").utterances():
            t = time.perf_counter(); print(f"{transcribe(rec, a).text!r}  ({(time.perf_counter() - t) * 1000:.0f} ms)")
    else:
        a, sr = sf.read(sys.argv[1], dtype="float32")
        t = time.perf_counter(); print(words_from_audio(rec, a, sr), f"{(time.perf_counter() - t) * 1000:.0f} ms")
