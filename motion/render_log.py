"""Render a LIVE ROOM SIM run (CONTRACT §6) offline from its logs, in the hero.mp4 style.

    /var/lib/happy-agent/IsaacLab/.venv/bin/python render_log.py room/run/<run>/ -o videos/live/<run>.mp4

Reads from the run directory:
  world_state.ndjson   LiveWorld.state() per frame; the first line is {"config": LiveWorld.config()}
  kermit.ndjson        run_kermit stdout (segment / play / stop ...): subtitles and gesture tags
  people.ndjson        optional: set_person calls; a "text" field on a talking person becomes a caption
  mix.wav              room audio; its first sample is at perf_counter ``mix_t0`` (meta.json), else at the
                       first world_state t
The puppet and people are replayed exactly as they were posed in the run (nothing is re-animated);
only the cameras are chosen offline: the puppet's 3/4 close-up while it talks, a two-shot of the
puppet and whoever else is talking, and a short establishing wide at the start.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from live import LiveWorld  # noqa: E402  (sets MUJOCO_GL)

import mujoco  # noqa: E402

from kermit_motion.animation import CLIP_LEN, Animation, Timeline  # noqa: E402
from kermit_motion.scene import HEAD_REST_POS  # noqa: E402
from render import Hud  # noqa: E402

MIN_SHOT = 2.0


def _ndjson(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def _cameras(cfg: dict) -> dict:
    """name -> (eye, target, fovy) in the hero.mp4 look: warm stage, puppet 3/4, presenters beside it."""
    floor = cfg.get("floor_z", -0.8)
    k = HEAD_REST_POS + np.array([0.07, 0.0, -0.02])
    cams = {"kermit_cu": (k + 1.05 * np.array([np.cos(0.35), np.sin(0.35), 0.14]), k, 34),
            "stage_wide": (np.array([3.6, 1.7, 1.3]), np.array([0.9, 0.0, -0.1]), 46)}
    for p in cfg["people"]:
        if p.get("seated"):
            continue
        xy = np.asarray(p.get("pos_base", (1.0, 0.0)), float)
        P = np.array([xy[0], xy[1], floor + 1.3])
        K = HEAD_REST_POS + np.array([0.0, 0, -0.05])
        line = P - K
        perp = np.array([-line[1], line[0], 0.0])
        perp /= np.linalg.norm(perp)
        if perp[0] < 0:  # stand on the audience side of the puppet-presenter line
            perp = -perp
        eye = K + 0.45 * line + 1.6 * perp + np.array([0, 0, 0.32])
        cams[f"two_{p['name']}"] = (eye, K + 0.45 * line + np.array([0, 0, 0.08]), 50)
    judges = [np.asarray(p.get("pos_base", (3.4, 0)), float) for p in cfg["people"] if p.get("seated")]
    if judges:
        jc = np.mean(judges, axis=0)
        cams["qa_wide"] = (np.array([jc[0] + 1.4, jc[1] + 0.5, 1.4]), HEAD_REST_POS + np.array([0.6, 0, -0.2]), 42)
    return cams


def _kermit_timeline(events: list[dict], t0: float):
    """Words, gesture labels and talking spans on the video clock from run_kermit's stream."""
    segs, words, labels, spans, stops, emotions = {}, [], [], [], {}, []
    for ev in events:
        if ev.get("event") == "stop":
            stops[ev["turn"]] = min(stops.get(ev["turn"], np.inf), ev.get("_t_video", np.inf))
    for ev in events:
        kind = ev.get("event")
        if kind == "segment":
            segs[(ev["turn"], ev["index"])] = ev
        elif kind == "play" and (ev["turn"], ev["index"]) in segs:
            seg = segs[(ev["turn"], ev["index"])]
            start = ev.get("t_audio", ev["t_wall"]) - t0
            turn0 = start - seg.get("t_offset", 0.0)
            dur = seg.get("duration") or len(seg.get("mouth", {}).get("values", [])) / 60
            cut = stops.get(ev["turn"], np.inf)
            end = min(start + dur, cut)
            spans.append((start, end))
            if seg.get("emotion"):
                emotions.append({"t": start - 0.3, "type": "emotion", "value": seg["emotion"]})
            kept = [dict(w, t0=w["t0"] + turn0, t1=min(w["t1"] + turn0, cut)) for w in seg.get("words", [])
                    if w["t0"] + turn0 < cut]
            if kept and len(kept) < len(seg.get("words", [])):  # playback was cut off mid-sentence
                kept[-1]["w"] = kept[-1]["w"].rstrip(".,!?…") + "\u2014"
            words += kept
            labels += [(e["t"] + turn0, e["t"] + turn0 + CLIP_LEN.get(e["type"], 0.9), e["type"])
                       for e in seg.get("events", []) if e["t"] + turn0 < cut]
    return words, labels, spans, emotions


def _clear_angle(tgt, yaw, standing, dist=1.05, clearance=0.42):
    """Camera azimuth around the puppet: ~25 deg off its facing direction toward the audience (+x),
    nudged until no standing person is within ``clearance`` of the line of sight."""
    want = yaw - np.radians(25) * np.sign(yaw) if abs(yaw) > np.radians(12) else np.radians(20)
    for da in np.radians([0, 8, -8, 16, -16, 24, -24, 32, -32, 45, -45]):
        a = want + da
        eye = tgt[:2] + dist * np.array([np.cos(a), np.sin(a)])
        seg = eye - tgt[:2]
        ok = True
        for P in standing:
            u = np.clip((P - tgt[:2]) @ seg / (seg @ seg), 0, 1)
            if np.linalg.norm(tgt[:2] + u * seg - P) < clearance and u > 0.05:
                ok = False
        if ok:
            return a
    return want


_FONT = None


def _heard_tag(img, t, heard, size):
    global _FONT
    cur = [h for h in heard if h[0] <= t <= h[0] + 3.0]
    if not cur:
        return img
    from PIL import Image, ImageDraw, ImageFont
    if _FONT is None:
        _FONT = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", int(17 * size[1] / 720))
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im, "RGBA")
    text = f"live STT heard: \u201c{cur[-1][1]}\u201d"
    s = size[1] / 720
    x, y = 20 * s, 74 * s
    d.rounded_rectangle([x - 8 * s, y - 6 * s, x + d.textlength(text, font=_FONT) + 8 * s, y + 26 * s], 8 * s,
                        fill=(20, 60, 110, 190))
    d.text((x, y), text, font=_FONT, fill=(230, 240, 255, 255))
    return np.asarray(im)


def render_log(run: Path, out: Path, fps: int = 30, size=(1280, 720), hud: bool = True, trim: bool = True):
    t_start = time.time()
    states = _ndjson(run / "world_state.ndjson")
    cfg = next((s["config"] for s in states if "config" in s), None)
    if cfg is None:
        cfg_path = run / "config.json"
        cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else None
    if cfg is None:
        raise SystemExit("no LiveWorld config: log {'config': w.config()} as the first world_state line")
    states = [s for s in states if "qpos" in s]
    ts = np.array([s["t"] for s in states])
    meta = json.loads((run / "meta.json").read_text()) if (run / "meta.json").exists() else {}
    t0 = float(meta.get("mix_t0", meta.get("t0", ts[0])))

    kev = _ndjson(run / "kermit.ndjson")
    for ev in kev:  # stop.at_s is relative to that segment's t_wall
        if ev.get("event") == "stop":
            play = next((p for p in kev if p.get("event") == "play" and p["turn"] == ev["turn"]
                         and p["index"] == ev.get("index")), None)
            if play is not None:
                ev["_t_video"] = play["t_wall"] + ev.get("at_s", 0.0) - t0
    words, labels, k_spans, emotions = _kermit_timeline(kev, t0)

    people_log = []
    for row in _ndjson(run / "people.ndjson"):  # ROOM SIM writes one row per tick with a "people" list
        if "people" in row:
            people_log += [dict(p, t=row["t"]) for p in row["people"]]
        else:
            people_log.append(row)
    captions, talk_spans = [], {}
    cur = {}
    for row in people_log:  # collapse consecutive talking frames into spans (+ caption if text given)
        name, t = row.get("name"), row.get("t", 0) - t0
        talking = bool(row.get("talking"))
        if talking and name not in cur:
            cur[name] = [t, t, row.get("text")]
        elif talking:
            cur[name][1] = t
            cur[name][2] = cur[name][2] or row.get("text")
        elif name in cur:
            a, b, text = cur.pop(name)
            talk_spans.setdefault(name, []).append((a, b))
            if text:
                captions.append([a, b, name.capitalize(), text])
    for name, (a, b, text) in cur.items():
        talk_spans.setdefault(name, []).append((a, b))
        if text:
            captions.append([a, b, name.capitalize(), text])
    # captions from the room-sim driver (line on/off times) + the show script text
    drv = run / "driver_results.json"
    script_path = meta.get("script")
    if not captions and drv.exists() and script_path and Path(script_path).exists():
        script = json.loads(Path(script_path).read_text())
        lines = {st["step"]: (st["speaker"], st["line"]) for st in script.get("steps", [])}
        for r in json.loads(drv.read_text()):
            if r.get("step") in lines and r.get("line_on_t"):
                who_, text = lines[r["step"]]
                captions.append([r["line_on_t"] - t0, r.get("line_off_t", r["line_on_t"] + 3) - t0,
                                 who_.capitalize(), text])
    # what Kermit's live STT heard, shown briefly as proof the loop ran live
    heard = [(ev["t_rx"] - t0, ev["text"]) for ev in kev if ev.get("event") == "heard" and ev.get("text")
             and not ev.get("echo")]

    w = LiveWorld(**{k: v for k, v in cfg.items() if k in ("people", "size", "hfov", "floor_z", "hand", "cam_pos",
                                                         "cam_pitch_deg", "ptz")}, view_size=size)
    m, d = w.k.m, w.k.d
    cams = _cameras(cfg)
    end = max(ts[-1] - t0, max([b for _, b in k_spans], default=0))
    first = min([c[0] for c in captions] + [a for a, _ in k_spans] + [end])
    start = max(0.0, first - 2.5) if trim else 0.0  # skip the room warm-up before the first line
    duration = end - start
    n = int(duration * fps)
    tv = start + np.arange(n) / fps

    standing = [np.asarray(p.get("pos_base", (1, 0)), float) for p in cfg["people"] if not p.get("seated")]

    # puppet head pose per video frame (from the logged joints)
    site = m.site("head").id
    frame_state = [int(np.clip(np.searchsorted(ts, t + t0), 0, len(ts) - 1)) for t in tv]
    head_pos, head_yaw = np.zeros((n, 3)), np.zeros(n)
    for i, j in enumerate(frame_state):
        d.qpos[:] = states[j]["qpos"]
        mujoco.mj_kinematics(m, d)
        head_pos[i] = d.site_xpos[site]
        R = d.site_xmat[site].reshape(3, 3)
        head_yaw[i] = np.arctan2(R[1, 0], R[0, 0])

    # shot list: who is talking at each moment, held >= MIN_SHOT
    def who(t):
        if any(a <= t <= b for a, b in k_spans):
            return "kermit_cu"
        for name, sp in talk_spans.items():
            if any(a <= t <= b for a, b in sp):
                return f"two_{name}" if f"two_{name}" in cams else ("qa_wide" if "qa_wide" in cams else "kermit_cu")
        return None
    shots = [(start, "stage_wide")]
    for t in tv[:: max(1, fps // 10)]:
        c = who(t)
        if c and c != shots[-1][1] and t - shots[-1][0] >= MIN_SHOT:
            shots.append((t, c))
    if len(shots) > 1 and shots[1][0] > start + 2.5 + MIN_SHOT:
        shots.insert(1, (start + 2.5, "kermit_cu"))
    # each close-up sits ~23 deg off the way the puppet faces during that shot, on the audience side,
    # so its face always reads 3/4 (never the back of its head)
    for k, (ta, c) in enumerate(shots):
        if c != "kermit_cu":
            continue
        tb = shots[k + 1][0] if k + 1 < len(shots) else end
        sel = (tv >= ta) & (tv < tb)
        yaw = float(np.mean(head_yaw[sel])) if sel.any() else 0.0
        tgt = np.mean(head_pos[sel], axis=0) + np.array([0.04 * np.cos(yaw), 0.04 * np.sin(yaw), -0.02])
        a = _clear_angle(tgt, yaw, standing)
        name = f"kermit_cu_{k}"
        cams[name] = (tgt + 1.05 * np.array([np.cos(a), np.sin(a), 0.14]), tgt, 34)
        shots[k] = (ta, name)

    tl = Timeline(duration=duration, audio=None, words=words, mouth=np.zeros(1), mouth_hz=60, events=emotions,
                  emotion=cfg.get("emotion", "neutral"), captions=captions)
    overlay = Hud(tl, Animation(t=tv, ch={}, labels=labels), size[0], size[1], f"live room run · {meta.get('base_dba', '?')} dBA · mic {meta.get('input', '?')}",
                  True) if hud else None
    renderer = mujoco.Renderer(m, size[1], size[0])
    audio = run / "audience.wav" if (run / "audience.wav").exists() else run / "mix.wav"
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{size[0]}x{size[1]}",
           "-r", str(fps), "-i", "-"]
    if audio.exists():
        cmd += ["-ss", f"{start:.3f}", "-i", str(audio), "-c:a", "aac", "-b:a", "160k"]
    cmd += ["-c:v", "libx264", "-preset", "medium", "-crf", "19", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            str(out)]
    out.parent.mkdir(parents=True, exist_ok=True)
    ff = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    cam = mujoco.MjvCamera()
    mouth_geoms = {name: w.k.people[name] for name in w.k.people}
    try:
        for i, t in enumerate(tv):
            s = states[frame_state[i]]
            d.qpos[:] = s["qpos"]
            if "mocap_pos" in s and m.nmocap:
                d.mocap_pos[:] = s["mocap_pos"]
                d.mocap_quat[:] = s["mocap_quat"]
            for name, pm in mouth_geoms.items():
                mo = s.get("mouths", {}).get(name, 0.0)
                m.geom_size[pm.mouth_geom] = pm.mouth_size * np.array([1, 1 - 0.15 * mo, 1 + 3.5 * mo])
            mujoco.mj_forward(m, d)
            name = shots[0][1]
            for ta, c in shots:
                if ta <= t:
                    name = c
            eye, tgt, fov = cams[name]
            v = np.asarray(tgt) - np.asarray(eye)
            cam.lookat[:] = tgt
            cam.distance = np.linalg.norm(v)
            cam.azimuth = np.degrees(np.arctan2(v[1], v[0]))
            cam.elevation = np.degrees(np.arcsin(v[2] / np.linalg.norm(v)))
            m.vis.global_.fovy = fov
            renderer.update_scene(d, cam)
            img = renderer.render()
            if overlay:
                img = overlay.draw(img, t)
                img = _heard_tag(img, t, heard, size)
            ff.stdin.write(np.ascontiguousarray(img).tobytes())
    finally:
        ff.stdin.close()
        ff.wait()
        renderer.close()
        w.close()
    print(f"wrote {out}: {n} frames, {duration:.1f}s, shots {[(round(a, 1), b) for a, b in shots]} "
          f"({time.time() - t_start:.0f}s)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", type=Path)
    ap.add_argument("-o", "--out", type=Path, required=True)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--size", default="1280x720")
    ap.add_argument("--no-hud", action="store_true")
    ap.add_argument("--no-trim", action="store_true", help="keep the room warm-up before the first line")
    a = ap.parse_args()
    render_log(a.run, a.out, a.fps, tuple(map(int, a.size.split("x"))), not a.no_hud, not a.no_trim)


if __name__ == "__main__":
    main()
