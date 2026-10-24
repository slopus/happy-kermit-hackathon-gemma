"""Render BRAIN's v3 show (brain/demo_script/v3/script.json, CONTRACT §6) by driving the LIVE BRAIN server
step by step, exactly as the live hub would: /turn after each human line, /event turned_away mid-line for
the barge-in, speaker_changed(role=judge) before each judge. Scripted beats come back instantly; Q&A
answers are live Gemma (fallback_turn from the script if BRAIN fails or rambles twice).

Output (samples/show/): scene.wav, scene_kermit_only.wav, scene.json (clips with absolute times,
speaker ids, scene-wide mouth + events, brain_log), per-step human_NN.wav / kermit_NN.{wav,timeline.json}.

  brain.py serve --prompt persona3 --phase presentation   # first (base + Kermit LoRA)
  python make_show.py [--brain-url http://127.0.0.1:8765]
  python make_show.py --reroll 17 [--takes 3]    # re-ask one live Q&A step, keep the best take, re-mix in place
Re-rolls never use the script's fallback lines: every take is live Gemma, all takes are logged.
"""
import argparse, json, os, re, sys, time, types, urllib.request
import numpy as np, soundfile as sf

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import kermit_voice
from kermit_voice import Performer, trim
from tts import KokoroTTS, SR
kermit_voice.print = lambda *a, **k: None

SCRIPT = os.path.join(HERE, "..", "brain", "demo_script", "v3", "script.json")
# speaker → (id, role, Kokoro voice, yaw seen from the robot (+ = left), dist m). Same voices as the ROOM SIM.
SPEAKERS = {"kirill": (1, "presenter", "am_michael", 45.0, 1.0), "steve": (2, "presenter", "am_adam", -45.0, 1.0),
            "judge1": (3, "judge", "af_heart", 15.0, 3.5), "judge2": (4, "judge", "bf_emma", 0.0, 3.5),
            "judge3": (5, "judge", "am_eric", -15.0, 3.5)}
MAX_WORDS = 30
REPLY_GAP, HUMAN_GAP, SPEAKER_SWAP_GAP = 0.7, 0.6, 0.4
SESSION = "show_v3"


def post(url, body):
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        data = r.read().decode()
    lines = [json.loads(l) for l in data.splitlines() if l.strip()]
    return lines


def turn_of(events):
    if not events or events[0].get("event") == "skip": return None
    return next(({k: v for k, v in e.items() if k != "event"} for e in events if e.get("event") == "turn"), None)


def write_scene(out, clips, log, voice, total):
    """Mix all clips into scene.wav / scene_kermit_only.wav and write scene.json."""
    mix = np.zeros(int(total * SR), np.float32); kermit_only = mix.copy()
    for c in clips:
        x, _ = sf.read(os.path.join(out, c["audio"]), dtype="float32"); i = int(round(c["t"] * SR))
        mix[i:i + len(x)] += (1.0 if c["who"] == "kermit" else 0.8) * x
        if c["who"] == "kermit": kermit_only[i:i + len(x)] += x
    mix *= 0.95 / max(1e-6, np.abs(mix).max())
    sf.write(os.path.join(out, "scene.wav"), mix, SR); sf.write(os.path.join(out, "scene_kermit_only.wav"), kermit_only, SR)
    hz = 60; mouth = np.zeros(int(np.ceil(total * hz)), np.float32)
    for c in clips:
        if c["who"] != "kermit": continue
        m = json.load(open(os.path.join(out, c["timeline"])))["mouth"]["values"]; i = int(round(c["t"] * hz))
        mouth[i:i + len(m)] = np.maximum(mouth[i:i + len(m)], m[:len(mouth) - i])
    json.dump({"script": "brain/demo_script/v3/script.json", "audio": "scene.wav", "kermit_only_audio": "scene_kermit_only.wav",
               "sample_rate": SR, "duration": round(total, 3), "kermit_voice": voice,
               "people": [{"name": w, "id": i, "role": r, "voice": v, "yaw_deg": y, "dist_m": d} for w, (i, r, v, y, d) in SPEAKERS.items()],
               "clips": clips, "mouth": {"hz": hz, "values": [round(float(v), 3) for v in mouth]},
               "events": sorted([e for c in clips if c["who"] == "kermit" for e in c["events"]], key=lambda e: e["t"]),
               "brain_log": log}, open(os.path.join(out, "scene.json"), "w"), separators=(",", ":"))
    print(f"scene {total:.1f} s → {out}")


DEFLECT = re.compile(r"team would know|not sure|no idea|don't know|ask the team|can't say", re.I)


def score_take(text):
    """Higher is better: answers the question (no deflection), fits the length cap, has some substance."""
    if not text: return -99
    n = len(text.split())
    return (-10 if DEFLECT.search(text) else 0) + (-5 if n > MAX_WORDS else 0) + min(n, 20) / 10


def reroll(a, B, step_no, takes, pick=None):
    """takes > 0: ask live BRAIN that many more times. pick: keep logged take #pick (numbering across all rounds)."""
    script = json.load(open(SCRIPT)); st = next(s for s in script["steps"] if s["step"] == step_no)
    assert st["phase"] == "qa", "only live Q&A steps are re-rolled"
    scene = json.load(open(os.path.join(a.out, "scene.json")))
    sid = SPEAKERS[st["speaker"]][0]; ctx = script["brain"]["context_qa"]
    entry = next(e for e in scene["brain_log"] if e["step"] == step_no)
    logged = entry.setdefault("rerolls", [])
    tries = []
    for k in range(0 if pick else takes):
        sess = f"{SESSION}_reroll{step_no}_{k}_{int(time.time())}"
        post(B + "/phase", {"phase": "qa", "session_id": sess})
        post(B + "/event", {"event": "speaker_changed", "from": None, "to": sid, "role": "judge", "session_id": sess})
        t0 = time.perf_counter(); turn = turn_of(post(B + "/turn", {"heard": st["line"], "context": ctx, "stream": True, "session_id": sess}))
        tries.append({"take": len(logged) + k + 1, "text": turn and turn["text"], "turn": turn, "brain_ms": round((time.perf_counter() - t0) * 1000),
                      "score": score_take(turn and turn["text"]), "deflects": bool(turn and DEFLECT.search(turn["text"]))})
        print(f"take {tries[-1]['take']}: ({tries[-1]['brain_ms']} ms, score {tries[-1]['score']:.1f}) {tries[-1]['text']}", flush=True)
    logged += tries
    best = next(x for x in logged if x["take"] == pick) if pick else max(tries, key=lambda x: x["score"])
    old = next(c for c in scene["clips"] if c["who"] == "kermit" and c["step"] == step_no)
    perf = Performer(types.SimpleNamespace(voice=a.voice, tts_backend="torch", device=a.device, no_frog=False,
                                           out=os.path.join(HERE, ".context", "render"), play=False))
    wav, tl = perf.one(turn=best["turn"])
    dst = os.path.join(a.out, f"kermit_{step_no:02d}")
    os.replace(wav, dst + ".wav"); tl["audio"] = f"kermit_{step_no:02d}.wav"; tl.pop("latency", None)
    json.dump(tl, open(dst + ".timeline.json", "w"), separators=(",", ":"))
    shift = tl["duration"] - old["duration"]; k0 = old["t"]
    old.update(duration=tl["duration"], text=tl["text"], emotion=tl["emotion"], gaze=tl["gaze"], source="live_gemma",
               events=[dict(e, t=round(e["t"] + k0, 3)) for e in tl["events"]])
    for c in scene["clips"]:  # anything after this clip moves by the length difference
        if c["t"] > k0 + 0.01:
            c["t"] = round(c["t"] + shift, 3)
            for key in ("events", "words", "staging"):
                for e in c.get(key, []):
                    for f in ("t", "t0", "t1"):
                        if f in e: e[f] = round(e[f] + shift, 3)
    if "first_take" not in entry: entry["first_take"] = {"text": entry.get("brain_text"), "brain_ms": entry.get("brain_ms")}
    entry.update(brain_text=best["text"], kept_take=best["take"], kept_by="manual pick" if pick else "score")
    write_scene(a.out, scene["clips"], scene["brain_log"], a.voice, scene["duration"] + shift)
    print(f"kept take {best['take']}: {best['text']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--brain-url", default="http://127.0.0.1:8765")
    ap.add_argument("--out", default=os.path.join(HERE, "samples", "show"))
    ap.add_argument("--voice", default=os.environ.get("KERMIT_VOICE", "am_fenrir"))
    ap.add_argument("--device", default=None)
    ap.add_argument("--reroll", type=int, help="re-ask this live Q&A step only")
    ap.add_argument("--takes", type=int, default=3)
    ap.add_argument("--pick", type=int, help="with --reroll: keep this logged take number instead of asking again")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    B = a.brain_url.rstrip("/")
    if a.reroll:
        return reroll(a, B, a.reroll, a.takes, a.pick)
    script = json.load(open(SCRIPT))
    post(B + "/phase", {"phase": "presentation", "session_id": SESSION})

    perf = Performer(types.SimpleNamespace(voice=a.voice, tts_backend="torch", device=a.device, no_frog=False,
                                           out=os.path.join(HERE, ".context", "render"), play=False))
    voices = {}
    clips, log, t, prev_judge = [], [], 0.5, None
    for st in script["steps"]:
        n, who, line, k = st["step"], st["speaker"], st["line"], st.get("kermit")
        sid, role, voice, yaw, dist = SPEAKERS[who]
        if voices.get(voice) is None: voices[voice] = KokoroTTS(voice, device=a.device, character=False)
        ha, hw = voices[voice].synth(line, "neutral"); ha, hw = trim(ha, hw)
        interject = bool(k and "turned_away" in k.get("trigger", ""))
        cut_at = None
        if interject:  # human trails off as Kermit cuts in at the staged word
            at = max([s["at_word"] for s in st.get("staging", []) if "trails off" in s["action"]] or [len(hw) - 1])
            cut_at = hw[min(at, len(hw) - 1)]["t1"] + 0.05
            fade = int(0.08 * SR); i = int(cut_at * SR)
            ha = np.concatenate([ha[:i], ha[i:i + fade] * np.linspace(1, 0, len(ha[i:i + fade]), dtype=np.float32)])
        sf.write(os.path.join(a.out, f"human_{n:02d}.wav"), ha, SR)
        h0, h1 = t, t + len(ha) / SR
        clips.append({"who": who, "speaker_id": sid, "role": role, "voice": voice, "yaw_deg": yaw, "step": n, "phase": st["phase"],
                      "audio": f"human_{n:02d}.wav", "t": round(h0, 3), "duration": round(len(ha) / SR, 3), "text": line,
                      "staging": [dict(s, t=round(h0 + hw[min(s["at_word"], len(hw) - 1)]["t0"], 3)) for s in st.get("staging", [])],
                      "words": [{"w": w["w"], "t0": round(w["t0"] + h0, 3), "t1": round(w["t1"] + h0, 3)} for w in hw if w["t0"] < len(ha) / SR]})

        # --- ask BRAIN, the way the live hub would
        entry = {"step": n, "speaker": who, "line": line}
        ctx = script["brain"]["context_qa" if st["phase"] == "qa" else "context_speech"]
        if role == "judge" and who != prev_judge:
            entry["speaker_changed"] = post(B + "/event", {"event": "speaker_changed", "from": prev_judge, "to": sid, "role": "judge", "session_id": SESSION})
            prev_judge = who
        t_req = time.perf_counter()
        if interject:
            partial = " ".join(w["w"] for w in hw if w["t1"] <= cut_at)
            ev = post(B + "/event", {"event": "turned_away", "id": sid, "while_speaking": True, "role": role,
                                     "heard_partial": partial, "context": ctx, "session_id": SESSION})
        else:
            ev = post(B + "/turn", {"heard": line, "context": ctx, "stream": True, "session_id": SESSION})
        turn = turn_of(ev); entry["brain_ms"] = round((time.perf_counter() - t_req) * 1000)
        entry["brain_text"] = turn and turn["text"]
        if st["phase"] == "qa" and (turn is None or len(turn["text"].split()) > MAX_WORDS):
            ev = post(B + "/turn", {"heard": line, "context": ctx, "stream": True, "session_id": SESSION}); turn = turn_of(ev)
            entry["retry_text"] = turn and turn["text"]
            if turn is None or len(turn["text"].split()) > MAX_WORDS:
                turn = k.get("fallback_turn"); entry["used_fallback"] = True
        expected = k and (k.get("turn") or {}).get("text")
        if expected and turn and turn["text"] != expected: entry["differs_from_script"] = expected
        if k is None and turn is not None: entry["unexpected_reply"] = True
        entry["source"] = "fallback" if entry.get("used_fallback") else ("live_gemma" if st["phase"] == "qa" else "scripted_beat")
        log.append(entry)

        if turn is None:  # Kermit stays quiet
            t = h1 + (SPEAKER_SWAP_GAP if who != "kirill" else HUMAN_GAP)
            print(f"{n:2d} {who:7s} {line[:60]!r}  → (Kermit quiet)", flush=True)
            continue
        wav, tl = perf.one(turn=turn)
        dst = os.path.join(a.out, f"kermit_{n:02d}")
        os.replace(wav, dst + ".wav"); tl["audio"] = f"kermit_{n:02d}.wav"; tl.pop("latency", None)
        json.dump(tl, open(dst + ".timeline.json", "w"), separators=(",", ":"))
        k0 = (h0 + cut_at - 0.15) if interject else h1 + REPLY_GAP
        clips.append({"who": "kermit", "step": n, "phase": st["phase"], "beat": (k or {}).get("beat"), "answering": who,
                      "gaze_target_id": sid, "gaze_yaw_deg": yaw, "source": entry["source"],
                      "audio": tl["audio"], "timeline": f"kermit_{n:02d}.timeline.json", "t": round(k0, 3), "duration": tl["duration"],
                      "text": tl["text"], "emotion": tl["emotion"], "gaze": tl["gaze"],
                      "events": [dict(e, t=round(e["t"] + k0, 3)) for e in tl["events"]]})
        t = max(k0 + tl["duration"], h1) + HUMAN_GAP
        print(f"{n:2d} {who:7s} {line[:60]!r}\n   Kermit [{tl['emotion']}|{tl['gaze']}] ({entry['source']}, {entry['brain_ms']} ms): {tl['text']}", flush=True)

    write_scene(a.out, clips, log, a.voice, t + 0.5)


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)
