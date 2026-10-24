"""Mouth openness curve for a puppet that only opens and closes.

Audio envelope, not visemes: a hand puppet has one degree of freedom, and puppeteers flap once per
syllable. Steps: speech-band loudness in dB at 60 Hz → normalised against this utterance's own level →
split into a slow phrase level and a fast syllable part, with the syllable part boosted so the mouth
closes between syllables → fast-open / slower-close smoothing → gates that shut the mouth in pauses
and briefly on m/b/p word onsets (when word timings are known) → per-emotion gain.

Works on one TTS sentence at a time (we synthesise sentence by sentence), so per-utterance
normalisation costs no latency. Pure numpy/scipy.
"""
import numpy as np
from scipy.signal import butter, sosfiltfilt, resample_poly

HZ = 60

# Per-emotion (gain, max openness). Excited/surprised flap wider, sad/embarrassed mumble.
EMOTION_STYLE = {
    "neutral": (1.0, 0.85), "happy": (1.1, 0.9), "excited": (1.2, 1.0), "surprised": (1.2, 1.0),
    "sad": (0.7, 0.6), "thinking": (0.85, 0.7), "embarrassed": (0.75, 0.65), "annoyed": (1.0, 0.85),
}


def _ema(x, a):
    y, s = np.empty_like(x), 0.0
    for i, v in enumerate(x):
        s = a * s + (1 - a) * v; y[i] = s
    return y


def _attack_release(x, att, rel):
    y, s = np.empty_like(x), 0.0
    for i, v in enumerate(x):
        a = att if v > s else rel
        s = a * s + (1 - a) * v; y[i] = s
    return y


def envelope_db(audio, sr, hz=HZ):
    """Speech-band (250–3500 Hz) RMS in dB, one value per 1/hz s frame, 2× overlapped windows."""
    x = sosfiltfilt(butter(4, [250, 3500], btype="band", fs=sr, output="sos"), audio)
    hop = sr / hz; win = int(2 * hop)
    n = int(np.ceil(len(x) / hop))
    x = np.pad(x, (win // 2, win))
    e = np.array([np.mean(x[int(i * hop):int(i * hop) + win] ** 2) for i in range(n)])
    return 10 * np.log10(e + 1e-10)


def mouth_curve(audio, sr, words=None, emotion="neutral", hz=HZ, lead_ms=0.0,
                range_db=28.0, syllable=1.8, phrase=0.3, attack_ms=10, release_ms=45):
    """Openness 0..1 at `hz`, aligned with audio start. `words` = [{"w","t0","t1"}] (seconds) optional."""
    audio = np.asarray(audio, np.float64)
    if sr not in (16000, 22050, 24000, 44100, 48000):
        audio, sr = resample_poly(audio, 24000, sr), 24000
    db = envelope_db(audio, sr, hz)
    voiced = db > db.max() - 40
    ref = np.percentile(db[voiced], 90) if voiced.any() else db.max()
    v = np.clip((db - (ref - range_db)) / range_db, 0, 1.2)       # 0 = silence, ~1 = loud vowel
    slow = _ema(v, np.exp(-1000 / (220 * hz)))                      # phrase level (~220 ms)
    syl = np.clip(v - slow, 0, None)                                # syllable peaks above phrase level
    m = phrase * np.minimum(slow, v) + syllable * syl + 0.3 * np.clip(v - 0.5, 0, None)
    m *= np.clip((v - 0.12) / 0.15, 0, 1)                           # no sound, no mouth
    m = _attack_release(m, np.exp(-1000 / (attack_ms * hz)), np.exp(-1000 / (release_ms * hz)))
    gain, top = EMOTION_STYLE.get(emotion, EMOTION_STYLE["neutral"])
    m = top * np.tanh(m * gain * 0.95 / top)                        # soft clip instead of flat tops
    if words:
        t = np.arange(len(m)) / hz
        talking = np.zeros(len(m), bool)
        for w in words:
            talking |= (t >= w["t0"] - 0.04) & (t <= w["t1"] + 0.03)
        m = np.where(talking, m, m * 0.25)                          # breaths/noise in pauses barely move it
        for w in words:                                             # lips meet on m/b/p onsets
            if w["w"].lstrip("\"'(").lower()[:1] in ("m", "b", "p"):
                c = int(round(w["t0"] * hz))
                m[max(c - 2, 0):max(c + 2, 0)] *= 0.2
    m[m < 0.04] = 0.0
    if lead_ms:                                                     # play the mouth early for a slow actuator
        s = int(round(lead_ms / 1000 * hz))
        m = np.concatenate([m[s:], np.zeros(s)])
    return np.round(m, 3).astype(np.float32)
