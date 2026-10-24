"""Offline barge-in / self-echo test on a ROOM SIM run: replay its mic.wav through endpoint.Endpointer with the
loudspeaker feed (kermit_dry.wav, advanced by --ref-lead ms, i.e. what Speaker.tap sees) as AEC reference.
Kermit is "speaking" while the reference is non-silent. Ground truth for human speech: mic_human.wav energy.

usage: room_offline.py RUN_DIR [--ref-lead 52] [--bargein-ms 200] [--no-aec] [--no-gate]
"""
import argparse, json, os, sys, time
import numpy as np, soundfile as sf

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import endpoint as E

ap = argparse.ArgumentParser()
ap.add_argument("run"); ap.add_argument("--ref-lead", type=float, default=52); ap.add_argument("--bargein-ms", type=float, default=200)
ap.add_argument("--no-aec", action="store_true"); ap.add_argument("--no-gate", action="store_true")
ap.add_argument("--mic", default="mic.wav"); ap.add_argument("--asr", action="store_true")
a = ap.parse_args()
R = lambda f: sf.read(os.path.join(a.run, f), dtype="float32")[0]
mic, ref, hum = R(a.mic), R("kermit_dry.wav"), R("mic_human.wav")
n = min(len(mic), len(ref), len(hum)); mic, hum = mic[:n], hum[:n]
lead = int(a.ref_lead * 16)
ref = np.concatenate([ref[lead:], np.zeros(lead, np.float32)])[:n]

events = []
ep = E.Endpointer(on_turn_end=lambda *x: None, on_bargein=lambda t: None, aec=not a.no_aec, asr=a.asr, sync=True,
                  log=events.append, speculative=False, bargein_ms=a.bargein_ms)
if a.no_gate: ep.echo_gate_db = None
blk = E.BLOCK
# truth: human speech per block (mic_human level above -50 dBFS-ish relative threshold)
hdb = np.array([20 * np.log10(np.sqrt(np.mean(hum[i:i + blk] ** 2)) + 1e-9) for i in range(0, n - blk + 1, blk)])
hthr = np.percentile(hdb[hdb > -90], 90) - 30
hspeech = hdb > hthr
rdb = np.array([20 * np.log10(np.sqrt(np.mean(ref[i:i + blk] ** 2)) + 1e-9) for i in range(0, n - blk + 1, blk)])
playing = rdb > -55
speaking = False; cost = []
for k, i in enumerate(range(0, n - blk + 1, blk)):
    t = (i + blk) / 16000
    if playing[k] != speaking:
        # Speaker's on_speaking: on when a sentence starts, off when the queue drains (~0.05 s later)
        if playing[k] or not playing[k:k + 12].any():
            speaking = bool(playing[k]); ep.set_kermit_speaking(speaking); ep.kermit_off_t = t if not speaking else None
    ep.far(ref[i:i + blk], 16000, t - blk / 16000)
    c = time.perf_counter(); ep.feed(t, mic[i:i + blk]); cost.append(time.perf_counter() - c)

# score barge-ins
bi = [e["t"] for e in events if e["event"] == "bargein_detected"]
def human_near(t, before=1.0):
    k = int(t * 16000 / blk); return hspeech[max(0, k - int(before * 16000 / blk)):k + 1].any()
true_bi = [t for t in bi if human_near(t)]
false_bi = [t for t in bi if not human_near(t)]
# human onsets while Kermit plays (opportunities)
ons = []
for k in range(1, len(hspeech)):
    if hspeech[k] and not hspeech[max(0, k - 16):k].any() and playing[max(0, k - 3):k + 3].any():
        ons.append(k * blk / 16000)
det = []
for o in ons:
    hit = [t for t in bi if o - 0.05 <= t <= o + 1.5]
    det.append(hit[0] - o if hit else np.nan)
det = np.array(det)
play_s = playing.sum() * blk / 16000
res = dict(run=os.path.basename(a.run.rstrip("/")), aec=not a.no_aec, gate=not a.no_gate, bargein_ms=a.bargein_ms,
           kermit_play_s=round(play_s, 1), bargeins=len(bi), false_bargeins=len(false_bi),
           false_per_min_of_kermit=round(len(false_bi) / play_s * 60, 2) if play_s else None,
           human_onsets_over_kermit=len(ons), detected=int(np.isfinite(det).sum()),
           det_latency_ms_med=round(float(np.nanmedian(det)) * 1000) if np.isfinite(det).any() else None,
           cost_ms_p50=round(np.median(cost) * 1000, 2))
print(json.dumps(res))
print("false at:", [round(t, 1) for t in false_bi][:20])
