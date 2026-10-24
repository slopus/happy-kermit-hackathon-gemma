"""The venue: one shoebox room, fixed source/listener positions, RIRs, streaming convolution.

Coordinates: everything is specified in Kermit's ARM BASE frame (CONTRACT §4: +x forward toward the
audience, +y = robot's left, +z up, origin at the arm base on the demo table), and converted to room
coordinates for pyroomacoustics. The room is 12 x 8 x 4 m; the stage table sits 1.5 m from the front
wall, centred; the audience fills the rest.

Levels: dry source signals are scaled so that rms 1.0 "dry unit" at the source would be `spl_1m` dB SPL
at 1 m (pyroomacoustics RIRs carry 1/r spreading). The virtual mic maps
94 dB SPL (1 Pa rms) to MIC_DBFS_AT_94 dBFS, i.e. a laptop/USB mic with its AGC settled.
"""
import json, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FS = 16000
BLOCK = 320                      # 20 ms engine block
ROOM_DIM = (12.0, 8.0, 4.0)
RT60 = 0.8
BASE_IN_ROOM = np.array([6.0, 1.5, 0.75])   # arm base: on a 0.75 m table, 1.5 m from the front wall
MIC_DBFS_AT_94 = -6.0
P_REF = 20e-6


def base_to_room(p):
    x, y, z = p
    return np.array([BASE_IN_ROOM[0] - y, BASE_IN_ROOM[1] + x, BASE_IN_ROOM[2] + z])


# --- positions, arm base frame (m) ---
CAM_POS = np.array([0.05, 0.0, 0.10])          # CONTRACT §4 default
MIC_POS = np.array([0.08, 0.0, 0.08])           # at the arm base, next to the camera
AUDIENCE_EAR = np.array([2.6, 0.2, 0.45])       # front row, seated: the "room mixdown" for the video
PEOPLE = {  # mouth positions; standing mouth 1.60 m, seated 1.20 m above the floor (table top at 0.75 m)
    "kirill": {"pos": [0.75, 0.75, 0.85], "role": "presenter"},
    "steve":  {"pos": [0.75, -0.75, 0.85], "role": "presenter"},
    "judge1": {"pos": [3.3, 0.9, 0.45], "role": "judge"},
    "judge2": {"pos": [3.5, 0.0, 0.45], "role": "judge"},
    "judge3": {"pos": [3.3, -0.9, 0.45], "role": "judge"},
}
SPEAKER_POS = np.array([0.0, -0.30, 0.02])      # Kermit's loudspeaker: the laptop, 0.3 m from the mic
LAPTOP_FAN_POS = np.array([0.05, -0.38, 0.0])
HVAC_POS = [np.array([2.5, 3.0, 3.1]), np.array([5.0, -3.5, 3.1])]
CROWD_POS = [np.array(p) for p in [(3.0, 3.5, 0.45), (4.5, 1.5, 0.45), (5.5, -2.5, 0.45),
                                   (2.8, -3.8, 0.45), (6.2, 2.8, 0.45), (4.0, -0.8, 0.45)]]


def head_yaw_to(src_xy, dst_xy):
    d = np.asarray(dst_xy, float)[:2] - np.asarray(src_xy, float)[:2]
    return float(np.degrees(np.arctan2(d[1], d[0])))


def sources():
    """name → dict(pos, directivity yaw or None, group). Presenters get two RIRs: facing Kermit and
    facing the audience; the engine crossfades between them as they turn."""
    S = {}
    for n, p in PEOPLE.items():
        pos = np.array(p["pos"])
        if p["role"] == "presenter":
            S[n + "@kermit"] = dict(pos=pos, yaw=head_yaw_to(pos, MIC_POS), group="human")
            S[n + "@audience"] = dict(pos=pos, yaw=0.0, group="human")
        else:
            S[n] = dict(pos=pos, yaw=head_yaw_to(pos, MIC_POS), group="human")
    S["kermit_spk"] = dict(pos=SPEAKER_POS, yaw=head_yaw_to(SPEAKER_POS, [3.0, 0.0]), group="echo")
    S["laptop_fan"] = dict(pos=LAPTOP_FAN_POS, yaw=None, group="noise")
    for i, p in enumerate(HVAC_POS):
        S[f"hvac{i}"] = dict(pos=p, yaw=None, group="noise")
    for i, p in enumerate(CROWD_POS):
        S[f"crowd{i}"] = dict(pos=p, yaw=None, group="noise")
    return S


def _handheld(name):
    """Config B: a handheld cardioid 6 cm in front of / 3 cm below the mouth, pointing at the mouth."""
    mouth = np.array(PEOPLE[name]["pos"], float)
    d = MIC_POS[:2] - mouth[:2] if PEOPLE[name]["role"] == "presenter" else -mouth[:2]
    d = d / np.linalg.norm(d)
    pos = mouth + np.array([0.06 * d[0], 0.06 * d[1], -0.03])
    v = mouth - pos
    return pos, float(np.degrees(np.arctan2(v[1], v[0]))), float(np.degrees(np.arccos(v[2] / np.linalg.norm(v))))


LISTENERS = {"mic": MIC_POS, "audience": AUDIENCE_EAR}
HANDHELD = {}   # listener name → (pos, yaw_deg base frame, colatitude_deg)
for _n in PEOPLE:
    HANDHELD["hh_" + _n] = _handheld(_n)
    LISTENERS["hh_" + _n] = HANDHELD["hh_" + _n][0]


def compute_rirs(cache=os.path.join(HERE, "assets", "rirs.npz"), rt60=RT60, max_len_s=None):
    """pyroomacoustics ISM (+ air absorption), per source x listener; cached."""
    S = sources()
    key = json.dumps({"rt60": rt60, "dim": ROOM_DIM, "base": BASE_IN_ROOM.tolist(),
                      "S": {k: [list(map(float, v["pos"])), v["yaw"]] for k, v in S.items()},
                      "L": {k: list(map(float, v)) for k, v in LISTENERS.items()}, "v": 4}, sort_keys=True)
    if os.path.exists(cache):
        z = np.load(cache, allow_pickle=False)
        if str(z["key"]) == key:
            return {k: z[k] for k in z.files if k != "key" and k != "meta"}, json.loads(str(z["meta"]))
    import pyroomacoustics as pra
    from pyroomacoustics.directivities import CardioidFamily, DirectionVector
    e_abs, max_order = pra.inverse_sabine(rt60, ROOM_DIM)
    max_order = min(max_order, 24)
    names_L = list(LISTENERS)
    rirs, meta = {}, {"rt60_target": rt60, "max_order": max_order, "e_absorption": e_abs, "rt60_measured": {}, "drr_db": {}}
    for sname, s in S.items():
        room = pra.ShoeBox(ROOM_DIM, fs=FS, materials=pra.Material(e_abs), max_order=max_order,
                           air_absorption=True)
        dirv = None
        if s["yaw"] is not None:   # speech/loudspeaker: broad sub-cardioid (~-6 dB behind)
            dirv = CardioidFamily(orientation=DirectionVector(azimuth=s["yaw"] + 90,  # base yaw → room azimuth
                                                              colatitude=90, degrees=True), p=0.75)
        room.add_source(base_to_room(s["pos"]).tolist(), directivity=dirv)
        mdir = [CardioidFamily(orientation=DirectionVector(azimuth=HANDHELD[n][1] + 90, colatitude=HANDHELD[n][2], degrees=True), p=0.5)
                if n in HANDHELD else None for n in names_L]
        room.add_microphone_array(np.stack([base_to_room(LISTENERS[n]) for n in names_L]).T,
                                  directivity=mdir if any(mdir) else None)
        room.compute_rir()
        for li, ln in enumerate(names_L):
            h = np.asarray(room.rir[li][0], np.float32)
            # trim at -60 dB of the energy decay (Schroeder), cap 1.2 s
            edc = np.cumsum((h ** 2)[::-1])[::-1]; edc /= edc[0]
            n = int(np.searchsorted(-edc, -1e-6)); n = min(max(n, FS // 10), int(1.2 * FS))
            rirs[f"{sname}|{ln}"] = h[:n]
            d = np.linalg.norm(base_to_room(s["pos"]) - base_to_room(LISTENERS[ln]))
            i0 = int(np.argmax(np.abs(h))); w = int(0.0025 * FS)
            meta["drr_db"][f"{sname}|{ln}"] = round(float(10 * np.log10(np.sum(h[max(0, i0 - w):i0 + w] ** 2) /
                                                                  (np.sum(h[i0 + w:] ** 2) + 1e-12))), 1)
            if sname == "crowd0" and ln == "mic":
                try:
                    meta["rt60_measured"][ln] = round(float(pra.experimental.measure_rt60(h, fs=FS)), 3)
                except Exception:
                    pass
            meta.setdefault("dist_m", {})[f"{sname}|{ln}"] = round(float(d), 3)
    np.savez(cache, key=key, meta=json.dumps(meta), **rirs)
    return rirs, meta


class StreamConv:
    """Uniformly partitioned overlap-save convolution of S input streams with S x M RIRs.
    process(X[S, B]) → per-source outputs [S, M, B] (kept per source so groups can be logged)."""

    def __init__(self, H, B=BLOCK):
        # H: list over sources of list over listeners of 1-D RIRs
        self.B, self.N = B, 2 * B
        S, M = len(H), len(H[0])
        L = max(len(h) for row in H for h in row)
        self.P = P = -(-L // B)
        F = B + 1
        self.H = np.zeros((S, M, P, F), np.complex64)
        for s in range(S):
            for m in range(M):
                h = np.zeros(P * B, np.float32); h[:len(H[s][m])] = H[s][m]
                self.H[s, m] = np.fft.rfft(h.reshape(P, B), n=self.N, axis=1)
        self.X = np.zeros((S, P, F), np.complex64)   # frequency-domain delay line (ring)
        self.prev = np.zeros((S, B), np.float32)
        self.head = 0

    def process(self, x, listeners=None):
        """listeners: indices to compute (the input delay line is shared, so switching listeners is seamless)."""
        buf = np.concatenate([self.prev, x], axis=1); self.prev = x.copy()
        self.head = (self.head - 1) % self.P
        self.X[:, self.head] = np.fft.rfft(buf, axis=1)
        idx = (self.head + np.arange(self.P)) % self.P           # delay-ordered partitions
        H = self.H if listeners is None else self.H[:, listeners]
        Y = np.einsum("spf,smpf->smf", self.X[:, idx], H, optimize=True)
        return np.fft.irfft(Y, n=self.N, axis=2)[:, :, self.B:].astype(np.float32)


def a_weight_db(x, fs=FS):
    """A-weighted level (dB re full scale rms) of a signal: frequency-domain A curve."""
    X = np.fft.rfft(x); f = np.fft.rfftfreq(len(x), 1 / fs)
    f2 = f ** 2
    ra = (12194 ** 2 * f2 ** 2) / ((f2 + 20.6 ** 2) * np.sqrt((f2 + 107.7 ** 2) * (f2 + 737.9 ** 2)) * (f2 + 12194 ** 2) + 1e-30)
    A = ra * 10 ** (2.0 / 20)
    p = np.sum(np.abs(X * A) ** 2) / len(x) ** 2 * 2
    return 10 * np.log10(p + 1e-20)


def digital_to_spl(db_fs):
    return db_fs - MIC_DBFS_AT_94 + 94.0


def spl_to_dry_gain(spl_1m):
    """Dry-unit (rms 1) → source scale such that the free-field level at 1 m is spl_1m
    (pyroomacoustics' ISM direct path is 1/r, checked: peak*r ≈ 0.75–0.93 after the fractional-delay sinc)."""
    return P_REF * 10 ** (spl_1m / 20)


PA_TO_DIGITAL = 10 ** (MIC_DBFS_AT_94 / 20)
HH_DBFS_AT_94 = -26.0           # handheld channel at the mixer: speech at 6 cm ≈ -25 dBFS
HH_TO_DIGITAL = 10 ** (HH_DBFS_AT_94 / 20)


if __name__ == "__main__":
    import time
    t = time.time(); R, meta = compute_rirs(); print(f"RIRs {len(R)} in {time.time() - t:.1f}s; RT60 measured {meta['rt60_measured']}")
    for k in sorted(R):
        print(f"{k:28s} len {len(R[k]) / FS:.2f}s  dist {meta['dist_m'][k]:.2f} m  DRR {meta['drr_db'][k]:+.1f} dB")
