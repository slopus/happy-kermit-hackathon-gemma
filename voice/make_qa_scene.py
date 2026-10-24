"""Q&A scene: three judges ask, Kermit answers LIVE from BRAIN (--phase qa), rendered like samples/scene/.

  python make_qa_scene.py [--brain-url http://127.0.0.1:8765] [--out samples/qa]
Needs BRAIN running (`brain.py serve --prompt persona3 --phase qa` on base + Kermit LoRA).
Kermit's words are whatever Gemma says (kept as-is: "honest footage"); an answer over MAX_WORDS is
re-asked once on a fresh session and both attempts are logged in scene.json.
"""
import argparse, json, os, sys, time, types, urllib.request
import numpy as np, soundfile as sf

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import kermit_voice
from kermit_voice import Performer, trim
from tts import KokoroTTS, SR
kermit_voice.print = lambda *a, **k: None

# id, role, voice, where they sit (robot's view: + = left), question
JUDGES = [
    (1, "judge", "af_sarah", 25.0,
     "Okay, honest question. How is this different from a chatbot with a puppet glued on top?"),
    (2, "judge", "bm_george", 0.0,
     "What actually runs where? Is it really all offline, or is something phoning home?"),
    (3, "judge", "am_eric", -25.0,
     "What was the hardest part to get working in one day?"),
]
MAX_WORDS = 28
GAP_Q, GAP_A = 0.7, 0.9   # judge stops → Kermit starts; Kermit stops → next judge
SESSION = "qa_scene"


def post(url, body, stream=False):
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    r = urllib.request.urlopen(req, timeout=60)
    return [json.loads(l) for l in r if l.strip()] if stream else json.load(r)


def side(yaw):
    return "to your left" if yaw > 12 else "to your right" if yaw < -12 else "right in front of you"


def ask(brain, jid, yaw, q, prev):
    ctx = f"3 judges at the table; the one speaking is {side(yaw)}, facing you. visible: judges, audience, laptop, robot arm"
    ev = post(brain + "/event", {"event": "speaker_changed", "from": prev, "to": jid, "role": "judge", "session_id": SESSION}, stream=True)
    t0 = time.perf_counter()
    events = post(brain + "/turn", {"heard": q, "context": ctx, "stream": True, "session_id": SESSION}, stream=True)
    ms = round((time.perf_counter() - t0) * 1000)
    turn = next((e for e in events if e["event"] == "turn"), None)
    return ctx, ev, turn, ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--brain-url", default="http://127.0.0.1:8765")
    ap.add_argument("--out", default=os.path.join(HERE, "samples", "qa"))
    ap.add_argument("--voice", default=os.environ.get("KERMIT_VOICE", "am_fenrir"))
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    post(a.brain_url + "/phase", {"phase": "qa", "session_id": SESSION})

    perf = Performer(types.SimpleNamespace(voice=a.voice, tts_backend="torch", device=a.device, no_frog=False,
                                           out=os.path.join(HERE, ".context", "render"), play=False))
    clips, t, prev, log = [], 0.5, None, []
    for jid, role, voice, yaw, q in JUDGES:
        ha, hw = KokoroTTS(voice, device=a.device, character=False).synth(q, "neutral"); ha, hw = trim(ha, hw)
        sf.write(os.path.join(a.out, f"judge_{jid}.wav"), ha, SR)
        clips.append({"who": "judge", "speaker_id": jid, "role": role, "voice": voice, "yaw_deg": yaw, "audio": f"judge_{jid}.wav",
                      "t": round(t, 3), "duration": round(len(ha) / SR, 3), "text": q,
                      "words": [{"w": w["w"], "t0": round(w["t0"] + t, 3), "t1": round(w["t1"] + t, 3)} for w in hw]})
        t += len(ha) / SR

        ctx, ev, turn, ms = ask(a.brain_url, jid, yaw, q, prev)
        attempts = [{"text": turn and turn["text"], "brain_ms": ms, "words": len(turn["text"].split()) if turn else 0}]
        if turn is None or len(turn["text"].split()) > MAX_WORDS:
            ctx, ev, turn, ms = ask(a.brain_url, jid, yaw, q, prev)
            attempts.append({"text": turn and turn["text"], "brain_ms": ms, "words": len(turn["text"].split()) if turn else 0})
        log.append({"judge": jid, "question": q, "context": ctx, "speaker_changed_reply": ev, "attempts": attempts})
        prev = jid

        wav, tl = perf.one(turn=turn)
        dst = os.path.join(a.out, f"kermit_{jid}")
        os.replace(wav, dst + ".wav"); tl["audio"] = f"kermit_{jid}.wav"; tl.pop("latency", None)
        json.dump(tl, open(dst + ".timeline.json", "w"), separators=(",", ":"))
        k_start = t + GAP_Q
        clips.append({"who": "kermit", "answering": jid, "audio": tl["audio"], "timeline": f"kermit_{jid}.timeline.json",
                      "t": round(k_start, 3), "duration": tl["duration"], "text": tl["text"], "emotion": tl["emotion"],
                      "gaze": tl["gaze"], "gaze_target_id": jid, "gaze_yaw_deg": yaw,
                      "events": [dict(e, t=round(e["t"] + k_start, 3)) for e in tl["events"]]})
        t = k_start + tl["duration"] + GAP_A
        print(f"judge {jid} ({voice}): {q}\n  Kermit [{tl['emotion']}|{tl['gaze']}] ({ms} ms, {len(tl['text'].split())} words): {tl['text']}", flush=True)

    total = t + 0.3
    mix = np.zeros(int(total * SR), np.float32); kermit_only = mix.copy()
    for c in clips:
        x, _ = sf.read(os.path.join(a.out, c["audio"]), dtype="float32"); i = int(round(c["t"] * SR))
        mix[i:i + len(x)] += (0.8 if c["who"] == "judge" else 1.0) * x
        if c["who"] == "kermit": kermit_only[i:i + len(x)] += x
    mix *= 0.95 / max(1e-6, np.abs(mix).max())
    sf.write(os.path.join(a.out, "scene.wav"), mix, SR); sf.write(os.path.join(a.out, "scene_kermit_only.wav"), kermit_only, SR)
    hz = 60; mouth = np.zeros(int(np.ceil(total * hz)), np.float32)
    for c in clips:
        if c["who"] != "kermit": continue
        m = json.load(open(os.path.join(a.out, c["timeline"])))["mouth"]["values"]; i = int(round(c["t"] * hz))
        mouth[i:i + len(m)] = np.maximum(mouth[i:i + len(m)], m[:len(mouth) - i])
    people = [{"id": j, "role": r, "voice": v, "yaw_deg": y, "dist_m": 1.8} for j, r, v, y, _ in JUDGES]
    json.dump({"audio": "scene.wav", "kermit_only_audio": "scene_kermit_only.wav", "sample_rate": SR, "duration": round(total, 3),
               "kermit_voice": a.voice, "people": people, "clips": clips,
               "mouth": {"hz": hz, "values": [round(float(v), 3) for v in mouth]},
               "events": sorted([e for c in clips if c["who"] == "kermit" for e in c["events"]], key=lambda e: e["t"]),
               "brain_log": log},
              open(os.path.join(a.out, "scene.json"), "w"), separators=(",", ":"))


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    os._exit(0)
