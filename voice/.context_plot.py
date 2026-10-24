import json, sys, numpy as np, soundfile as sf, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, ".")
from timeline import build_timeline, dump
turns = {t["id"]: t for t in json.load(open("lines/demo_turns.json"))}
voice = sys.argv[1]
fig, axs = plt.subplots(3, 1, figsize=(14, 9))
for ax, tid in zip(axs, ["open_1", "open_2", "sad_1"]):
    a, sr = sf.read(f"bench/kokoro/{voice}/{tid}.wav", dtype="float32")
    tl = build_timeline(turns[tid], a, sr, json.load(open(f"bench/kokoro/{voice}/{tid}.words.json")), f"{tid}.wav")
    t = np.arange(len(a)) / sr
    ax.plot(t, a / np.abs(a).max() * 0.5 - 0.5, lw=0.3, color="gray")
    m = tl["mouth"]["values"]; ax.plot(np.arange(len(m)) / 60, m, color="green", lw=1.5)
    for w in tl["words"]: ax.axvline(w["t0"], color="b", lw=0.4); ax.text(w["t0"], 1.02, w["w"], fontsize=7, rotation=30)
    for e in tl["events"]: ax.axvline(e["t"], color="r", lw=1.2); ax.text(e["t"], -1.0, e["type"], color="r", fontsize=8)
    ax.set_ylim(-1.05, 1.2); ax.set_title(f"{voice} {tid} ({turns[tid]['emotion']})", fontsize=9)
plt.tight_layout(); plt.savefig(f"bench/mouth_{voice}.png", dpi=90)
