"""Fake a LIVE ROOM SIM run directory with the same files ROOM SIM will produce, to test render_log.py.

    /var/lib/happy-agent/IsaacLab/.venv/bin/python tools/mock_room_run.py .context/mock_run
    /var/lib/happy-agent/IsaacLab/.venv/bin/python render_log.py .context/mock_run -o videos/live/mock_run.mp4

Kirill asks, the puppet answers, Steve asks, the puppet answers (VOICE scene clips), driven through
LiveWorld exactly as ROOM SIM would (set_person at 30 Hz, run_kermit-style segment/play events,
speaker_id gaze), but stepped on a synthetic clock instead of wall time.
"""
from __future__ import annotations

import json
import sys
import wave
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))
from live import LiveWorld  # noqa: E402

SCENE = HERE.parent / "voice/samples/scene"
SR = 24000
FPS = 30
T0 = 5000.0  # fake perf_counter at the start of the mix


def wav(path):
    with wave.open(str(path)) as w:
        return np.frombuffer(w.readframes(w.getnframes()), np.int16)


def main(out: Path):
    out.mkdir(parents=True, exist_ok=True)
    people = [{"name": "kirill", "style": "presenter", "mic": True, "pos_base": [0.75, 0.75]},
              {"name": "steve", "style": "judge_b", "mic": True, "pos_base": [0.75, -0.75]},
              {"name": "judge_a", "style": "judge_a", "seated": True},
              {"name": "judge_b", "style": "judge_c", "seated": True},
              {"name": "judge_c", "style": "judge_a", "seated": True}]
    ids = {"kirill": 1, "steve": 2}
    w = LiveWorld(people=people)
    # script: (who, clip, start time in the mix)
    plan, t = [], 0.6
    for who, clip in [("kirill", "human_02"), ("kermit", "kermit_02"), ("steve", "human_03"), ("kermit", "kermit_03")]:
        pcm = wav(SCENE / f"{clip}.wav")
        plan.append((who, clip, t, len(pcm) / SR))
        t += len(pcm) / SR + 0.5
    total = t + 0.8
    mix = np.zeros(int(total * SR), np.int32)
    for who, clip, t0, dur in plan:
        pcm = wav(SCENE / f"{clip}.wav")
        mix[int(t0 * SR):int(t0 * SR) + len(pcm)] += pcm
    with wave.open(str(out / "mix.wav"), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(SR)
        f.writeframes(np.clip(mix, -32768, 32767).astype(np.int16).tobytes())
    (out / "meta.json").write_text(json.dumps({"mix_t0": T0}))
    texts = {c["audio"][:-4]: c["text"] for c in json.loads((SCENE / "scene.json").read_text())["clips"]}

    ws = open(out / "world_state.ndjson", "w")
    kl = open(out / "kermit.ndjson", "w")
    pl = open(out / "people.ndjson", "w")
    ws.write(json.dumps({"config": w.config()}) + "\n")
    emitted = set()
    speaker = None
    for i in range(int(total * FPS)):
        t = i / FPS
        now = T0 + t
        for who, clip, t0, dur in plan:
            active = t0 - 0.2 <= t <= t0 + dur + 0.2
            if who != "kermit" and active:
                speaker = who
            if who == "kermit" and t >= t0 and clip not in emitted:
                emitted.add(clip)
                tl = json.loads((SCENE / f"{clip}.timeline.json").read_text())
                seg = {"event": "segment", "turn": clip, "index": 0, "kind": "reply", "t_offset": 0.0,
                       "duration": tl["duration"], "words": tl["words"], "events": tl["events"], "mouth": tl["mouth"],
                       "emotion": tl["emotion"], "gaze": tl["gaze"], "audio": f"{clip}.wav", "out_dir": str(SCENE)}
                play = {"event": "play", "turn": clip, "index": 0, "t_wall": T0 + t0, "t_audio": T0 + t0}
                for ev in (seg, play):
                    w.kermit_event(ev)
                    kl.write(json.dumps(ev) + "\n")
        # people: presenters face the puppet, turn the head to whoever talks
        for p in people[:2]:
            name = p["name"]
            xy = np.array(p["pos_base"])
            talking = any(who == name and t0 <= t <= t0 + dur for who, _, t0, dur in plan)
            clip = next((c for who, c, t0, dur in plan if who == name and t0 <= t <= t0 + dur), None)
            mouth = 0.0
            if talking:
                c0 = next(t0 for who, c, t0, _ in plan if c == clip)
                k = int((t - c0) * SR)
                pcm = wav(SCENE / f"{clip}.wav")[k:k + SR // 30].astype(float)
                mouth = float(np.clip(np.sqrt(np.mean(pcm ** 2) + 1e-9) / 4000, 0, 1)) if len(pcm) else 0.0
            yaw = np.degrees(np.arctan2(-xy[1], -xy[0])) + (15 if xy[1] < 0 else -15)  # angled toward the audience
            row = dict(name=name, pos_base=list(xy), yaw_deg=yaw, head_yaw_deg=0.0, mouth=mouth, talking=talking)
            w.set_person(**row)
            pl.write(json.dumps(dict(row, t=now, text=texts.get(clip) if talking else None)) + "\n")
        if speaker:
            xy = people[ids[speaker] - 1]["pos_base"]
            g = {"event": "gaze", "speaker_id": ids[speaker],
                 "people": [{"id": ids[speaker], "target_base": [xy[0], xy[1], 0.82]}]}
            w.kermit_event(g)
        w.step(now)
        ws.write(json.dumps(w.state()) + "\n")
    for f in (ws, kl, pl):
        f.close()
    w.close()
    print(f"mock run: {out} ({total:.1f}s)")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / ".context/mock_run")
