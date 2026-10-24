"""Kokoro-82M TTS with the frog character applied, sentence by sentence, with word timings.

Character: Kokoro has no emotion or timbre control, so the puppet voice is made in DSP, at no
latency cost:
  * render slower, then resample faster by f = 2^(semitones/12): pitch AND formants go up together
    (smaller "vocal tract" = cartoon creature), while duration returns to the target speed;
  * a nasal/reedy EQ: peak around 1.3 kHz, cut the chest below 180 Hz, a little presence at 3.2 kHz.
Emotion maps to (speed, semitones, energy), e.g. excited = faster and higher.

Backends: "torch" (pip kokoro; cuda / mps / cpu) and "mlx" (mlx-audio, Apple Silicon GPU).
Both expose the same KPipeline token timestamps, so word timings work on either.
"""
import re
from fractions import Fraction
import numpy as np
from scipy.signal import lfilter, resample_poly

SR = 24000

# emotion: (speed, semitones up, output gain). Speed is the final speaking rate.
EMOTION_VOICE = {
    "neutral": (1.08, 3.0, 0.85), "happy": (1.12, 3.5, 0.9), "excited": (1.22, 4.5, 0.95),
    "surprised": (1.15, 5.0, 0.95), "sad": (0.92, 2.0, 0.7), "thinking": (0.98, 2.5, 0.75),
    "embarrassed": (1.05, 3.5, 0.75), "annoyed": (1.1, 2.0, 0.85),
}


def _biquad(kind, f0, gain_db, q, sr=SR):
    """RBJ cookbook peaking / low-shelf filter coefficients."""
    A = 10 ** (gain_db / 40); w = 2 * np.pi * f0 / sr; cw, sw = np.cos(w), np.sin(w); al = sw / (2 * q)
    if kind == "peak":
        b = [1 + al * A, -2 * cw, 1 - al * A]; a = [1 + al / A, -2 * cw, 1 - al / A]
    else:  # lowshelf
        s = 2 * np.sqrt(A) * al
        b = [A * ((A + 1) - (A - 1) * cw + s), 2 * A * ((A - 1) - (A + 1) * cw), A * ((A + 1) - (A - 1) * cw - s)]
        a = [(A + 1) + (A - 1) * cw + s, -2 * ((A - 1) + (A + 1) * cw), (A + 1) + (A - 1) * cw - s]
    return np.array(b) / a[0], np.array(a) / a[0]


# Respellings for words Kokoro's G2P gets wrong (checked with a Parakeet round trip). Same letters
# modulo case where possible, so timeline.align_words still matches them to the turn's text.
RESPELL = {r"\bLoRAs?\b": lambda m: "Lora" + m.group(0)[4:]}  # "LoRA" → "low raw"

EQ = [_biquad("lowshelf", 180, -6, 0.7), _biquad("peak", 1300, 6, 1.4), _biquad("peak", 3200, 3, 1.0)]


def frogify(audio, semitones, gain=0.9, eq=True):
    """Raise pitch+formants by resampling (audio must have been rendered 2^(st/12) slower), then EQ."""
    f = Fraction(2 ** (semitones / 12)).limit_denominator(40)
    y = resample_poly(audio.astype(np.float64), f.denominator, f.numerator)
    if eq:
        for b, a in EQ:
            y = lfilter(b, a, y)
    return (y * (gain / (np.max(np.abs(y)) + 1e-9))).astype(np.float32), float(f)


def _words_from_tokens(tokens, offset, scale):
    words = []
    for tok in tokens or []:
        if tok.start_ts is None or tok.end_ts is None:
            continue
        if not any(c.isalnum() for c in tok.text):  # punctuation token: glue onto the previous word
            if words: words[-1]["w"] += tok.text
            continue
        words.append({"w": tok.text, "t0": round(offset + tok.start_ts / scale, 3), "t1": round(offset + tok.end_ts / scale, 3)})
    return words


class KokoroTTS:
    def __init__(self, voice="am_puck", backend="torch", device=None, character=True):
        self.voice, self.backend, self.character = voice, backend, character
        if backend == "mlx":
            from mlx_audio.tts.utils import load_model
            from mlx_audio.tts.models.kokoro import KokoroPipeline
            repo = "mlx-community/Kokoro-82M-bf16"
            self.pipe = KokoroPipeline(lang_code="a", model=load_model(repo), repo_id=repo)
        else:
            import torch
            from kokoro import KPipeline
            device = device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
            if device == "cpu":  # uncapped torch threads thrash next to onnxruntime/llama: 13 s vs 0.3 s per sentence
                import os
                torch.set_num_threads(int(os.environ.get("KERMIT_TORCH_THREADS", 8)))
            self.pipe = KPipeline(lang_code="a", device=device, repo_id="hexgrad/Kokoro-82M")
        self.synth("Warm up.")  # loads the voice pack and compiles kernels

    def synth(self, text, emotion="neutral"):
        """One sentence (or a few) → (float32 audio @24 kHz, [{"w","t0","t1"}])."""
        for pat, rep in RESPELL.items():
            text = re.sub(pat, rep, text)
        text = re.sub(r"\s*(\.{2,}|…)\s*", ", ", text).strip(" ,")  # Kokoro reads a trailing "like..." as "light"
        text = re.sub(r"[^\w\s.,!?;:'\"()\-]", "", text).strip()  # emoji, markup: not speakable
        if not text:
            return np.zeros(0, np.float32), []
        speed, st, gain = EMOTION_VOICE.get(emotion, EMOTION_VOICE["neutral"])
        if not self.character:
            st = 0.0
        f = 2 ** (st / 12)
        chunks, words, off = [], [], 0.0
        for r in self.pipe(text, voice=self.voice, speed=speed / f, split_pattern=r"\n+"):
            a = np.asarray(r.audio.cpu() if hasattr(r.audio, "cpu") else r.audio, np.float32).reshape(-1)
            if self.character:
                a, f = frogify(a, st, gain)
            words += _words_from_tokens(r.tokens, off, f)
            chunks.append(a); off += len(a) / SR
        return (np.concatenate(chunks) if chunks else np.zeros(0, np.float32)), words
