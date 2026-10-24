"""Hackathon venue noise: dry source signals for the crowd/HVAC/laptop sources, generated block by block.

Material (small public subsets, assets/noise/):
  * DEMAND PCAFETER (16 kHz, 16 ch, 5 min): crowd babble. Each crowd position reads a different channel at
    a different time offset, so the six positions are decorrelated → a diffuse, moving babble field.
  * ESC-50 (CC BY-NC, 5 s clips): clapping (applause = many clips layered), laughing, footsteps,
    door_wood_creaks (stand-in for chairs), coughing, keyboard_typing, mouse_click.
  * synthesised: HVAC (low-passed pink noise + 60/120/180 Hz hum), laptop fan (band noise + whine).
Levels are set in dBA AT KERMIT'S MIC by the engine, through `calib` (measured through the real RIRs).
"""
import glob, os
import numpy as np, soundfile as sf
from scipy.signal import resample_poly, lfilter, butter, fftconvolve

from acoustics import FS, BLOCK, a_weight_db, PA_TO_DIGITAL, MIC_DBFS_AT_94


def spl_to_dbfs(spl):
    return spl - 94.0 + MIC_DBFS_AT_94

HERE = os.path.dirname(os.path.abspath(__file__))
NOISE = os.path.join(HERE, "assets", "noise")


def _load(path):
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    x = x.mean(1)
    if sr != FS:
        x = resample_poly(x, FS, sr).astype(np.float32)
    return x


def _trim(x, rel_db=-45):
    a = np.abs(x); thr = a.max() * 10 ** (rel_db / 20)
    idx = np.flatnonzero(a > thr)
    if len(idx) == 0: return x[:0]
    y = x[max(0, idx[0] - 160):idx[-1] + 800].copy()
    f = min(160, len(y) // 4)
    if f: y[-f:] *= np.linspace(1, 0, f, dtype=np.float32)
    return y


def _norm_a(x):
    """Scale to A-weighted rms 1.0 (0 dBA re dry unit)."""
    return (x / 10 ** (a_weight_db(x) / 20)).astype(np.float32)


def pink(n, rng):
    w = rng.standard_normal(n)
    b = [0.049922035, -0.095993537, 0.050612699, -0.004408786]; a = [1, -2.494956002, 2.017265875, -0.522189400]
    return lfilter(b, a, w).astype(np.float32)


class Library:
    def __init__(self, seed=0):
        rng = self.rng = np.random.default_rng(seed)
        chans = sorted(glob.glob(os.path.join(NOISE, "demand", "PCAFETER", "ch*.wav")))
        self.babble = [_norm_a(_load(chans[(2 * i) % len(chans)])) for i in range(6)]
        self.clips = {}
        for cat in ["clapping", "laughing", "footsteps", "door_wood_creaks", "coughing", "keyboard_typing", "mouse_click"]:
            xs = [_trim(_load(f)) for f in sorted(glob.glob(os.path.join(NOISE, "esc", cat, "*.wav")))]
            self.clips[cat] = [_norm_a(x) for x in xs if len(x) > FS // 20]
        n = 60 * FS
        b, a = butter(2, 400 / (FS / 2))
        hv = lfilter(b, a, pink(n, rng)); t = np.arange(n) / FS
        hv = hv / np.std(hv) + 0.15 * (np.sin(2 * np.pi * 60 * t) + 0.6 * np.sin(2 * np.pi * 120 * t) + 0.3 * np.sin(2 * np.pi * 180 * t))
        self.hvac = [_norm_a(np.roll(hv, k * n // 3).astype(np.float32)) for k in range(2)]
        b, a = butter(2, [300 / (FS / 2), 3000 / (FS / 2)], "band")
        fan = lfilter(b, a, rng.standard_normal(n)) + 0.3 * np.sin(2 * np.pi * 2140 * t)
        self.fan = _norm_a(fan.astype(np.float32))

    def applause(self, dur, rng, n_clappers=10):
        n = int(dur * FS); y = np.zeros(n + FS * 5, np.float32)
        for _ in range(n_clappers):
            c = self.clips["clapping"][rng.integers(len(self.clips["clapping"]))]
            o = int(rng.uniform(0, 0.4) * FS); seg = c[:n]
            y[o:o + len(seg)] += seg * rng.uniform(0.5, 1.0)
        y = y[:n]
        env = np.ones(n, np.float32); r = int(0.6 * FS)
        if n > 2 * r: env[-r:] = np.linspace(1, 0, r) ** 2
        return _norm_a(y * env)

    def laughter(self, rng, n_people=5):
        n = int(2.5 * FS); y = np.zeros(n, np.float32)
        for _ in range(n_people):
            c = self.clips["laughing"][rng.integers(len(self.clips["laughing"]))][:n]
            o = int(rng.uniform(0, 0.5) * FS); seg = c[:n - o]
            y[o:o + len(seg)] += seg * rng.uniform(0.4, 1.0)
        return _norm_a(y)

    def clip(self, cat, rng):
        c = self.clips[cat]; return c[rng.integers(len(c))]


def calibrate(lib, rirs, names):
    """dBA at the mic per unit (0 dBA) dry level, per noise source, measured through the actual RIRs."""
    out = {}
    for s in names:
        x = (lib.babble[0] if s.startswith("crowd") else lib.hvac[0] if s.startswith("hvac") else lib.fan)[:4 * FS]
        y = fftconvolve(x, rirs[f"{s}|mic"])[:4 * FS] * PA_TO_DIGITAL
        out[s] = a_weight_db(y[FS // 2:]) - a_weight_db(x[FS // 2:])
    return out


class NoiseField:
    """Per-block dry signals for the noise sources, driven by a level schedule.
    level(t_s) → target babble dBA at the mic (venue level). Events add on top."""

    def __init__(self, lib, calib, names, seed=1):
        self.lib, self.calib, self.names = lib, calib, names
        self.rng = np.random.default_rng(seed)
        self.n = 0                           # engine sample index
        self.pos = {s: int(self.rng.integers(0, 60 * FS)) for s in names}
        self.events = []                     # [start_n, source, signal(dry, already gain-scaled), label]
        self.base_dba = 60.0
        self.log = []                        # (n, label, source, dba) for truth

    # --- events (called from the driver / internally) ---
    def add_event(self, label, sig, dba_at_mic, src=None, delay_s=0.0):
        src = src or f"crowd{self.rng.integers(6)}"
        g = 10 ** ((spl_to_dbfs(dba_at_mic) - self.calib[src]) / 20)
        n0 = self.n + int(delay_s * FS)
        self.events.append([n0, src, sig * g, label])
        self.log.append((n0, len(sig), label, src, round(dba_at_mic, 1)))

    def applause(self, rel_db=10.0, dur=3.0, delay_s=0.0):
        """Spread over all crowd positions (each a few clappers)."""
        for i in range(6):
            sig = self.lib.applause(dur + self.rng.uniform(-0.4, 0.4), self.rng, n_clappers=4)
            self.add_event("applause", sig, min(self.base_dba + rel_db, 84) - 7.8, f"crowd{i}", delay_s + self.rng.uniform(0, 0.25))

    def laughter(self, rel_db=6.0, delay_s=0.0):
        for i in self.rng.choice(6, 3, replace=False):
            self.add_event("laughter", self.lib.laughter(self.rng, 3), min(self.base_dba + rel_db, 80) - 4.8, f"crowd{i}", delay_s + self.rng.uniform(0, 0.3))

    def _random_events(self, dt):
        loud = (self.base_dba - 45) / 25                     # 0 quiet … 1 loud
        rates = {"footsteps": 0.03 + 0.07 * loud, "door_wood_creaks": 0.02 + 0.04 * loud,   # per second
                 "coughing": 0.01 + 0.03 * loud, "keyboard_typing": 0.015, "mouse_click": 0.01}
        for cat, r in rates.items():
            if self.rng.random() < r * dt:
                lvl = self.base_dba + self.rng.uniform(-10, 0)
                if cat in ("keyboard_typing", "mouse_click"):
                    self.add_event(cat, self.lib.clip(cat, self.rng), min(lvl, 55), "laptop_fan")
                else:
                    self.add_event(cat, self.lib.clip(cat, self.rng), lvl, None)

    def block(self, base_dba):
        """→ {source: dry block}. base_dba = venue babble level at the mic now."""
        self.base_dba = base_dba
        self._random_events(BLOCK / FS)
        out = {}
        n_crowd = 6
        for s in self.names:
            if s.startswith("crowd"):
                i = int(s[5:]); src, target = self.lib.babble[i], base_dba - 10 * np.log10(n_crowd)
            elif s.startswith("hvac"):
                src, target = self.lib.hvac[int(s[4:])], 40.0 - 3.0
            else:
                src, target = self.lib.fan, 36.0
            p = self.pos[s] % (len(src) - BLOCK)
            out[s] = src[p:p + BLOCK] * 10 ** ((spl_to_dbfs(target) - self.calib[s]) / 20)
            self.pos[s] += BLOCK
        keep = []
        for ev in self.events:
            n0, s, sig, _ = ev
            a, b = self.n - n0, self.n + BLOCK - n0
            if b <= 0: keep.append(ev); continue
            if a >= len(sig): continue
            seg = sig[max(a, 0):min(b, len(sig))]
            o = max(0, -a)
            out[s] = out[s].copy(); out[s][o:o + len(seg)] += seg
            keep.append(ev)
        self.events = keep
        self.n += BLOCK
        return out
