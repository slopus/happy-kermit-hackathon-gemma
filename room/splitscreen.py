"""Split-screen "this ran live" debug video from one room run (no re-simulation, logs only).

  .venv/bin/python splitscreen.py run/<name> [-o out.mp4] [--audio audience|mic]

Top: the arm-base sim camera exactly as rendered live (armcam.mp4, re-timed to its logged frame times).
Bottom: Kermit's mic input waveform (what his STT/endpointer got) + a transcript lane:
  ROOM   who is actually speaking and what (ground truth), crowd reactions
  HEARD  what Kermit's STT produced (run_kermit 'heard', at the time it was produced; echo-flagged in grey)
  KERMIT what Kermit is saying, from the moment it is audible in the room; CUT if he stopped mid-sentence
plus a header: run, venue level, input config, and the measured end-of-speech → audio latency per reply.
"""
import argparse, json, os, subprocess
import numpy as np


def ass_time(t):
    t = max(0.0, t); h = int(t // 3600); m = int(t % 3600 // 60); s = t % 60
    return f"{h}:{m:02d}:{s:05.2f}"


def esc(s):
    return (s or "").replace("\\", "/").replace("{", "(").replace("}", ")").replace("\n", " ")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run"); ap.add_argument("-o"); ap.add_argument("--audio", default="audience")
    a = ap.parse_args()
    R = a.run
    meta = json.load(open(os.path.join(R, "meta.json"))); t0 = meta["t0"]
    cam = json.load(open(os.path.join(R, "armcam_times.json")))
    v0 = cam["times"][0]; eff = cam["effective_fps"]
    off = v0 - t0                                   # video starts this long after audio sample 0
    truth = [json.loads(l) for l in open(os.path.join(R, "truth.ndjson"))]
    kerm = [json.loads(l) for l in open(os.path.join(R, "kermit.ndjson"))]
    score = json.load(open(os.path.join(R, "score.json"))) if os.path.exists(os.path.join(R, "score.json")) else {}
    T = lambda t: t - v0                            # → video time
    ev = []   # (start, end, style, text)
    utt = {}
    for e in truth:
        if e.get("event") == "utt_on": utt[e["step"]] = e
        if e.get("event") == "utt_off":
            u = utt[e["step"]]; n = e["n_words"]
            txt = " ".join(w["w"] for w in u["words"][:n]) + (" —" if e["cut"] else "")
            ev.append((u["t"], e["t"] + 0.6, "Room", f"{u['who'].upper()}: {txt}"))
        if e.get("event") == "crowd":
            ev.append((e["t"], e["t"] + 2.5, "Crowd", f"[crowd {e['kind']}]"))
        if e.get("event") == "missed_turn":
            ev.append((e["t"], e["t"] + 2.0, "Bad", "MISSED TURN"))
    for e in kerm:
        if e.get("event") == "heard" and e.get("text"):
            ev.append((e["t_rx"], e["t_rx"] + 3.0, "Echo" if e.get("echo") else "Heard",
                       ("HEARD (own echo, dropped): " if e.get("echo") else "HEARD: ") + e["text"][-110:]))
        if e.get("event") == "bargein":
            ev.append((e["t_rx"], e["t_rx"] + 1.2, "Bad", "BARGE-IN → Kermit stops"))
    # Kermit lines from the room's audio onsets
    ons = [e["t"] for e in truth if e.get("event") == "kermit_audio_on"]
    offs = [e["t"] for e in truth if e.get("event") == "kermit_audio_off"]
    plays = [e for e in kerm if e.get("event") == "play"]
    segs = {}
    for e in kerm:
        if e.get("event") == "segment":
            segs[(e.get("turn"), e.get("index"))] = " ".join(w["w"] for w in e.get("words", []))
    stops = {(e.get("turn"), e.get("index")) for e in kerm if e.get("event") == "stop"}
    for p in plays:
        t = p.get("t_audio", p["t_wall"]) + 0.05
        txt = segs.get((p["turn"], p["index"]), "")
        nxt = [o for o in offs if o > t]
        end = min(nxt[0] if nxt else t + 3, t + 6)
        cut = (p["turn"], p["index"]) in stops
        ev.append((t, end + 0.3, "Kermit", f"KERMIT: {txt}" + ("  [CUT]" if cut else "")))
    lat = {r["step"]: r.get("latency_s") for r in score.get("steps", [])}
    for e in truth:
        if e.get("event") == "utt_off" and lat.get(e["step"]) is not None:
            ev.append((e["t"], e["t"] + lat[e["step"]] + 1.5, "Lat", f"end of speech → Kermit audible: {lat[e['step']]:.2f} s"))
    s = score.get("summary", {})
    head = (f"LIVE ROOM SIM · {os.path.basename(R.rstrip('/'))} · venue {meta['base_dba']:.0f} dBA · input "
            f"{'table mic' if meta.get('input', 'A') == 'A' else 'handheld mics'} · RT60 {meta['rt60']} s · wall-clock, closed loop")
    lanes = {"Room": 0, "Crowd": 0, "Bad": 3, "Heard": 1, "Echo": 1, "Kermit": 2, "Lat": 3}
    W, H = 1280, 1080
    sty = """[Script Info]
ScriptType: v4.00+
PlayResX: 1280
PlayResY: 1080
[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Head,DejaVu Sans,22,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,1,0,0,0,100,100,0,0,1,2,0,7,20,20,12,1
Style: Room,DejaVu Sans,26,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,0,0,0,0,100,100,0,0,1,2,0,7,20,20,0,1
Style: Crowd,DejaVu Sans,26,&H0080FFFF,&H000000FF,&H00000000,&H80000000,0,1,0,0,100,100,0,0,1,2,0,7,20,20,0,1
Style: Heard,DejaVu Sans,26,&H0000FFFF,&H000000FF,&H00000000,&H80000000,0,0,0,0,100,100,0,0,1,2,0,7,20,20,0,1
Style: Echo,DejaVu Sans,26,&H00909090,&H000000FF,&H00000000,&H80000000,0,1,0,0,100,100,0,0,1,2,0,7,20,20,0,1
Style: Kermit,DejaVu Sans,26,&H0050FF50,&H000000FF,&H00000000,&H80000000,1,0,0,0,100,100,0,0,1,2,0,7,20,20,0,1
Style: Bad,DejaVu Sans,26,&H004040FF,&H000000FF,&H00000000,&H80000000,1,0,0,0,100,100,0,0,1,2,0,9,20,20,0,1
Style: Lat,DejaVu Sans,24,&H00FFC080,&H000000FF,&H00000000,&H80000000,0,0,0,0,100,100,0,0,1,2,0,7,20,20,0,1
[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    lines = [f"Dialogue: 0,{ass_time(0)},{ass_time(1e4)},Head,,0,0,0,,{esc(head)}"]
    y0 = 720 + 140
    for st, en, style, text in sorted(ev):
        y = y0 + 48 * lanes[style]
        lines.append(f"Dialogue: 0,{ass_time(T(st))},{ass_time(T(en))},{style},,0,0,0,,{{\\pos(20,{y})}}{esc(text)[:120]}")
    ass = os.path.join(R, "split.ass"); open(ass, "w").write(sty + "\n".join(lines) + "\n")
    out = a.o or os.path.join(R, "split.mp4")
    aud = os.path.join(R, f"{a.audio}.wav"); mic = os.path.join(R, "mic.wav")
    fc = (f"[0:v]setpts=N/({eff})/TB,scale={W}:720,fps=30[cam];"
          f"[2:a]atrim=start={off:.3f},asetpts=PTS-STARTPTS,volume=25,alimiter=limit=0.95,showwaves=s={W}x140:mode=cline:colors=0x50ff50:rate=30[wav];"
          f"color=c=0x101010:s={W}x{H - 720 - 140}:r=30[pad];"
          f"[cam][wav][pad]vstack=inputs=3:shortest=1,ass={ass}[v];"
          f"[1:a]atrim=start={off:.3f},asetpts=PTS-STARTPTS,volume=2.0,aresample=48000[a]")
    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", os.path.join(R, "armcam.mp4"), "-i", aud, "-i", mic, "-filter_complex", fc,
           "-map", "[v]", "-map", "[a]", "-c:v", "libx264", "-preset", "veryfast", "-crf", "22", "-pix_fmt", "yuv420p",
           "-c:a", "aac", "-b:a", "160k", "-shortest", out]
    subprocess.run(cmd, check=True)
    print(out)


if __name__ == "__main__":
    main()
