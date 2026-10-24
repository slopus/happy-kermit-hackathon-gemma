"""GEMMA VISION: Gemma 4 E4B as the semantic layer on top of PERCEPTION's tracks (people.py, CONTRACT §5).

Specialists (people.py, ~20 Hz) keep doing tracking, faces and Light-ASD speaker detection. Gemma runs at ~1 Hz on the
latest frame with people.py's tracks drawn as numbered tags ("set-of-marks"), and answers with a track id, so Gemma's
ids ARE people.py's ids and Gemma never has to output pixels:
  - raised hands: people.py hand_up (precise, misses wrists out of frame / hands at the head) OR Gemma yes/no on the
    person crop (recall 93%); a held hand → hand_pick event with a short description ("the woman in the red top")
  - descriptions: Gemma on the person crop, cached per id
  - described look commands ("Kermit, look at the guy with the drink") → Gemma on frame + crop gallery → id
  - untracked people (wide crowd shots): Gemma box_2d on the native-aspect frame → bearing (no y offset on people,
    see results/calib.json)

Events (agreed with BRAIN, POST :8765/event, session_id "kermit"):
  {"event":"hand_pick","id":3,"description":"the guy with the drink","bearing_deg":12.0,"pitch_deg":4.0,"conf":0.8}
  {"event":"look_command","id":3,"description":"the guy with the drink","heard":"kermit look at …","bearing_deg":12.0}
  bearing_deg: base frame, + = Kermit's left (CONTRACT §4). With a PTZ camera, pass ptz=lambda: (pan_deg, tilt_deg)
  and the current pan/tilt is added to people.py's camera-frame bearing.
Gaze for MOTION: vision.gaze() → §4-style dict aimed at the picked/commanded person for `hold_s`, else None.

  in run_kermit.py:  --vision [--vision-url http://127.0.0.1:8081]
  standalone test:   python vision.py --source clip.mp4 [--say "look at the woman in red"]
"""
import argparse, json, math, os, re, sys, threading, time, urllib.request
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gemma

YN = ("Is this person raising a hand or arm up above their shoulder right now, as if asking a question? "
      "Holding a phone up counts. Answer yes or no.")
DESC = ("Describe this person in at most 8 words so a host can call on them in a crowd, like 'the woman in the red jacket' "
        "or 'the man with glasses and a grey beard'. Use clothing colour, hair, glasses, or something they hold; only mention "
        "an object if it is clearly in their hand. Don't mention pose or where they face. Never mention age, race, body, or "
        "looks. Answer with just the phrase.")
CMD = ("You are Kermit, a puppet on stage. The first image is the whole view with numbered people; the second shows each "
       "numbered person close up. Someone said: \"{cmd}\". Which numbered person do they mean? Left and right are as seen "
       "in the first image. If nobody matches, answer none. Answer with just the number.")
LOCATE = ("Someone told Kermit: \"{cmd}\". Find the one person they mean in this image. Respond with ONLY JSON like "
          "{{\"box_2d\":[ymin,xmin,ymax,xmax]}} normalized to 0-1000, or {{\"box_2d\":null}} if nobody matches.")

HAND_LOCATE = ('Find one person in the audience who is raising a hand. Respond with ONLY JSON like {"box_2d":[ymin,xmin,ymax,xmax]} '
               'normalized to 0-1000 around that person (head and raised arm), or {"box_2d":null} if nobody is.')

# "look at / turn to / face / point at <description>"; bare directions are BRAIN's job (look_at:left/right/up/speaker)
LOOK_RE = re.compile(r"\b(?:(?:look|turn|point|stare|glance)\w*\s+(?:over\s+|back\s+)?(?:at|to|toward|towards)|face)\s+(.+)", re.I)
TAIL = re.compile(r"[.!?;]|,?\s+(?:and|then)\s+(?:say|tell|wave|ask|answer|give|talk|sing|do|make)\b|\s+please\b", re.I)
BARE = re.compile(r"^(?:the\s+|your\s+|my\s+)?(?:left|right|up|down|me|us|here|there|him|her|them|the audience|the crowd|"
                  r"the camera|the ceiling|the floor|the sky)\b[\s.!?]*$", re.I)


def described_target(text):
    """'Kermit, look at the guy with the drink!' → 'the guy with the drink'; None for bare directions / no command."""
    m = LOOK_RE.search(text or "")
    if not m: return None
    tgt = TAIL.split(m.group(1).strip(), maxsplit=1)[0].strip(" ,")
    if not tgt or BARE.match(tgt): return None
    return tgt


def to_pil(img):
    if isinstance(img, Image.Image): return img.convert("RGB")
    return Image.fromarray(np.ascontiguousarray(img[..., ::-1]))          # OpenCV BGR → RGB


def crop(img, box, pad=0.05):
    x0, y0, x1, y1 = box; w, h = x1 - x0, y1 - y0
    return img.crop((max(0, x0 - pad * w), max(0, y0 - pad * h), min(img.width, x1 + pad * w), min(img.height, y1 + pad * h)))


class Vision:
    def __init__(self, url=gemma.URL, brain_url=None, on_event=None, hfov=78.0, ptz=None, hz=1.0, sweep_k=4,
                 hand_hold_s=1.0, pick_every_s=8.0, hold_s=3.0, auto_pick=True, log=None, busy=None, crowd=True):
        self.url, self.brain_url, self.on_event = url, brain_url, on_event
        self.hfov, self.ptz, self.hz, self.sweep_k = hfov, ptz, hz, sweep_k
        self.hand_hold_s, self.pick_every_s, self.hold_s, self.auto_pick = hand_hold_s, pick_every_s, hold_s, auto_pick
        self.log = open(log, "a") if log else None
        self.crowd = crowd                               # point at raised hands Gemma-only when people.py tracks < 3
        self.busy = busy or (lambda: False)              # True while BRAIN is generating: skip the sweep (shared GPU)
        self.frame, self.out, self.t_frame = None, None, 0.0
        self.lock = threading.Lock()
        self.hand_first, self.hand_last = {}, {}        # id → first / last time a hand was seen up (either source)
        self.checked = {}                                # id → last Gemma yes/no time
        self.desc = {}                                   # id → (description, t)
        self.last_pick_t, self.picked = -1e9, {}         # id → last pick time
        self.target = None                               # (id, bearing, pitch, t_until, source)
        self.timing = {"yn_ms": [], "desc_ms": [], "cmd_ms": [], "locate_ms": []}
        self._stop = threading.Event()

    # ------------------------------------------------------------ inputs
    def observe(self, img, out, t=None):
        """Latest camera frame (BGR ndarray or PIL) + the people.py §5 output computed from it."""
        with self.lock:
            self.frame, self.out, self.t_frame = img, out, time.perf_counter() if t is None else t

    def attach(self, people_thread):
        """Tap a running people.PeopleThread: every frame it steps is also handed to observe()."""
        def hook():
            people_thread.ready.wait()
            pp = people_thread.pp; step = pp.step
            def wrapped(img, t=None):
                out = step(img, t)
                if img is not None: self.observe(img, out)
                return out
            pp.step = wrapped
        threading.Thread(target=hook, daemon=True).start()

    def start(self):
        threading.Thread(target=self._loop, daemon=True).start(); return self

    def stop(self): self._stop.set()

    # ------------------------------------------------------------ helpers
    def _snap(self):
        with self.lock:
            if self.frame is None or self.out is None: return None, None
            return to_pil(self.frame), self.out

    def _bearing(self, pid, out):
        p = next((q for q in out.get("people", []) if q["id"] == pid), None)
        if p is None: return None, None
        pan, tilt = self.ptz() if self.ptz else (0.0, 0.0)
        return round(p["yaw_deg"] + pan, 1), round(p.get("pitch_deg", 0.0) + tilt, 1)

    def _px_bearing(self, u, v, w, h):
        """Pixel → bearing with a pinhole from hfov (for people Gemma found but people.py doesn't track)."""
        f = (w / 2) / math.tan(math.radians(self.hfov / 2))
        pan, tilt = self.ptz() if self.ptz else (0.0, 0.0)
        return round(-math.degrees(math.atan2(u - w / 2, f)) + pan, 1), round(-math.degrees(math.atan2(v - h / 2, f)) + tilt, 1)

    def _emit(self, ev, direct=False):
        ev = dict(ev, session_id="kermit")
        if self.log: self.log.write(json.dumps(dict(ev, t=round(time.perf_counter(), 3))) + "\n"); self.log.flush()
        if self.on_event and not direct:
            self.on_event(ev); return
        if self.brain_url:
            try:
                req = urllib.request.Request(self.brain_url.rstrip("/") + "/event", json.dumps(ev).encode(), {"Content-Type": "application/json"})
                t0 = time.perf_counter()
                with urllib.request.urlopen(req, timeout=10) as r: body = r.read().decode()
                if self.log:   # standalone: keep what BRAIN answered (NDJSON) next to the event
                    self.log.write(json.dumps({"brain_reply_to": ev["event"], "ms": round((time.perf_counter() - t0) * 1000),
                                               "lines": [json.loads(l) for l in body.splitlines() if l.strip()][-1:]}) + "\n"); self.log.flush()
            except Exception as e:
                print(f"vision: POST /event failed: {e!r:.120}", file=sys.stderr)
        elif not self.on_event:
            print(json.dumps(ev), flush=True)

    def describe(self, pid, img=None, out=None, max_age=8.0):
        d = self.desc.get(pid)
        if d and time.perf_counter() - d[1] < max_age: return d[0]
        if img is None: img, out = self._snap()
        p = next((q for q in (out or {}).get("people", []) if q["id"] == pid), None)
        if img is None or p is None: return d[0] if d else None
        txt, ms, _ = gemma.ask([crop(img, p["box"])], DESC, max_tokens=24, url=self.url)
        self.timing["desc_ms"].append(ms)
        phrase = re.sub(r"\s+", " ", txt.strip().strip(".\"'")).strip()
        phrase = re.sub(r",?\s*(?:facing|looking|turned)\s+(?:away|forward|the camera|left|right|front)(?:\s+from (?:the )?(?:camera|viewer))?", "", phrase, flags=re.I).strip(" ,")
        phrase = phrase[0].lower() + phrase[1:] if phrase else phrase
        if phrase and not phrase.startswith(("the ", "a ")): phrase = "the " + phrase
        self.desc[pid] = (phrase, time.perf_counter())
        return phrase

    def raised(self, img, out):
        """Update hand state: people.py hand_up OR Gemma yes/no on up to sweep_k least-recently-checked people."""
        now = time.perf_counter(); ppl = out.get("people", [])
        up = {p["id"] for p in ppl if p.get("hand_up")}
        todo = sorted((p for p in ppl if p["id"] not in up), key=lambda p: self.checked.get(p["id"], 0))[:self.sweep_k]
        for p in todo:
            a, ms, _ = gemma.ask([crop(img, p["box"], 0.15)], YN, max_tokens=2, grammar='root ::= "yes" | "no"', url=self.url)
            self.timing["yn_ms"].append(ms); self.checked[p["id"]] = now
            if a.strip() == "yes": up.add(p["id"])
        for pid in up:
            if now - self.hand_last.get(pid, -1e9) > 2.0: self.hand_first[pid] = now
            self.hand_last[pid] = now
        return [pid for pid in up if now - self.hand_first[pid] >= 0 and any(q["id"] == pid for q in ppl)]

    # ------------------------------------------------------------ API
    def _locate(self, img, prompt):
        """Gemma box_2d on the native-aspect frame → (box, yaw, pitch) or None. ~0.6 s. Never letterbox (calib.json)."""
        txt, ms, _ = gemma.ask([img], prompt, max_tokens=40, url=self.url)
        self.timing["locate_ms"].append(ms)
        try:
            b = json.loads(re.sub(r"^```(json)?|```$", "", txt.strip()).strip()).get("box_2d")
        except Exception:
            return None
        if not b or len(b) != 4 or not all(isinstance(x, (int, float)) for x in b): return None
        u, v = (b[1] + b[3]) / 2000 * img.width, (b[0] + 0.25 * (b[2] - b[0])) / 1000 * img.height   # aim at the upper body
        return (b,) + self._px_bearing(u, v, img.width, img.height)

    def phase(self):
        """BRAIN's show phase (GET /state, cached 2 s): auto hand picks only in 'qa'. Unknown/unreachable → 'qa'."""
        now = time.perf_counter()
        if now - getattr(self, "_phase_t", -1e9) > 2.0:
            self._phase_t, self._phase = now, "qa"
            if self.brain_url:
                try:
                    with urllib.request.urlopen(self.brain_url.rstrip("/") + "/state?session_id=kermit", timeout=0.5) as r:
                        self._phase = json.load(r).get("phase") or "qa"
                except Exception:
                    pass
        return self._phase

    def crowd_hand_pick(self):
        """people.py tracks few/no people (wide crowd shot): let Gemma point at a raised hand → bearing-only hand_pick."""
        img, out = self._snap()
        if img is None: return None
        r = self._locate(img, HAND_LOCATE)
        if r is None: return None
        b, yaw, pitch = r
        box = [b[1] * img.width / 1000, b[0] * img.height / 1000, b[3] * img.width / 1000, b[2] * img.height / 1000]
        txt, ms, _ = gemma.ask([crop(img, box, 0.1)], DESC, max_tokens=24, url=self.url); self.timing["desc_ms"].append(ms)
        d = txt.strip().strip(".\"'"); d = ("the " + d[0].lower() + d[1:]) if d and not d.lower().startswith(("the ", "a ")) else d
        ev = {"event": "hand_pick", "id": None, "description": d, "bearing_deg": yaw, "pitch_deg": pitch, "conf": 0.5, "box_2d": b}
        self.last_pick_t = time.perf_counter()
        self.target = (None, yaw, pitch, self.last_pick_t + self.hold_s, "hand_pick")
        self._emit(ev)
        return ev

    def hand_pick(self, force=False):
        """Pick one raised hand (held ≥ hand_hold_s, not picked in the last 20 s if others are up) → hand_pick event."""
        img, out = self._snap()
        if img is None: return None
        now = time.perf_counter()
        # not the person talking right now, and not a presenter on stage (§6: they stand ~1 m from Kermit)
        ok = {q["id"] for q in out.get("people", []) if q["id"] != out.get("speaker_id") and q.get("role") != "presenter"}
        cand = [pid for pid, t1 in self.hand_last.items() if now - t1 < 1.5 and now - self.hand_first.get(pid, now) >= (0 if force else self.hand_hold_s)
                and pid in ok]
        if not cand: return None
        # one pick per raised-hand episode: someone already called on must lower the hand and raise it again
        cand = [c for c in cand if self.picked.get(c, -1e9) < self.hand_first.get(c, now) or force]
        if not cand: return None
        fresh = [c for c in cand if now - self.picked.get(c, -1e9) > 20] or cand
        size = {q["id"]: (q["box"][2] - q["box"][0]) * (q["box"][3] - q["box"][1]) for q in out["people"]}
        pp_up = {q["id"] for q in out["people"] if q.get("hand_up")}
        # people.py's hand_up first (precision 100%), then Gemma-only hands; the biggest box = closest = easiest to address
        pid = max(fresh, key=lambda i: (i in pp_up, size.get(i, 0)))
        d = self.describe(pid, img, out, max_age=3.0)
        yaw, pitch = self._bearing(pid, out)
        ev = {"event": "hand_pick", "id": pid, "description": d, "bearing_deg": yaw, "pitch_deg": pitch,
              "conf": 0.9 if any(q["id"] == pid and q.get("hand_up") for q in out["people"]) else 0.7}
        self.picked[pid] = self.last_pick_t = now
        self.target = (pid, yaw, pitch, now + self.hold_s, "hand_pick")
        self._emit(ev)
        return ev

    def look_command(self, heard, emit=True):
        """Final transcript → look_command event if it names a described target; None if it isn't one."""
        tgt = described_target(heard)
        if tgt is None: return None
        img, out = self._snap()
        ev = {"event": "look_command", "id": None, "description": tgt, "heard": heard, "bearing_deg": None}
        if img is not None:
            ppl = out.get("people", [])
            if ppl:
                mk = gemma.marked(img, ppl); gal = gemma.gallery(img, ppl)
                pid, ms = gemma.ask_choice([mk, gal], CMD.format(cmd=heard), [p["id"] for p in ppl], none=True, url=self.url)
                self.timing["cmd_ms"].append(ms)
                if pid is not None:
                    yaw, pitch = self._bearing(pid, out)
                    ev.update(id=pid, bearing_deg=yaw, pitch_deg=pitch)
            if ev["id"] is None:                       # nobody tracked matches: let Gemma point (crowds, people.py misses)
                r = self._locate(img, LOCATE.format(cmd=heard))
                if r is not None:
                    ev.update(box_2d=r[0], bearing_deg=r[1], pitch_deg=r[2])
        if ev["bearing_deg"] is not None:
            self.target = (ev["id"], ev["bearing_deg"], ev.get("pitch_deg", 0.0), time.perf_counter() + self.hold_s, "look_command")
        if emit: self._emit(ev, direct=True)          # direct: BRAIN holds the /turn ≤1.5 s waiting for this
        return ev

    def on_heard(self, text):
        """Non-blocking hook for the voice loop (final transcript)."""
        if described_target(text) is not None:
            threading.Thread(target=self.look_command, args=(text,), daemon=True).start()

    def gaze(self):
        """§4-style gaze override for MOTION while a pick/command is fresh: follows the track if it moves."""
        if not self.target: return None
        pid, yaw, pitch, until, src = self.target
        if time.perf_counter() > until: return None
        if pid is not None and self.out is not None:
            y2, p2 = self._bearing(pid, self.out)
            if y2 is not None: yaw, pitch = y2, p2
        return {"present": True, "yaw_deg": yaw, "pitch_deg": pitch, "id": pid, "source": "vision:" + src}

    def context(self):
        """Short BRAIN context: who has a hand up (descriptions as cached)."""
        now = time.perf_counter()
        up = [pid for pid, t1 in self.hand_last.items() if now - t1 < 2.5]
        if not up: return ""
        ds = [self.desc[p][0] for p in up if p in self.desc]
        return f"{len(up)} raised hand{'s' if len(up) > 1 else ''}" + (f" ({'; '.join(ds[:3])})" if ds else "")

    # ------------------------------------------------------------ loop
    def _loop(self):
        while not self._stop.is_set():
            t0 = time.perf_counter()
            img, out = self._snap()
            if img is not None and not self.busy():
                try:
                    up = self.raised(img, out)
                    for pid in up[:2]: self.describe(pid, img, out)        # warm the cache for the pick
                    quiet = (self.auto_pick and not out.get("speech_active") and t0 - self.last_pick_t > self.pick_every_s
                             and self.phase() != "presentation")
                    if quiet and up:
                        self.hand_pick()
                    elif quiet and self.crowd and len(out.get("people", [])) < 3:
                        self.crowd_hand_pick()
                except Exception as e:
                    print(f"vision: {e!r:.200}", file=sys.stderr)
            self._stop.wait(max(0.05, 1.0 / self.hz - (time.perf_counter() - t0)))


# ---------------------------------------------------------------- standalone
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, help="video file (offline replay at real time) or camera index")
    ap.add_argument("--url", default=gemma.URL); ap.add_argument("--brain-url")
    ap.add_argument("--hfov", type=float, default=78.0)
    ap.add_argument("--say", action="append", default=[], help="'T:text' — a heard utterance at video time T (s)")
    ap.add_argument("--duration", type=float, default=0); ap.add_argument("--log")
    ap.add_argument("--render", help="write an annotated MP4 (tracks, raised hands people.py/Gemma, picks, commands)")
    ap.add_argument("--perception-args", nargs=argparse.REMAINDER, default=[])
    a = ap.parse_args()
    sys.path.insert(0, os.path.join(HERE, "..", "perception"))
    import people, cv2
    pa = people.parse_args(["--source", a.source, "--hfov", str(a.hfov), "--mic-model", ""] + a.perception_args)
    pp = people.People(pa)
    v = Vision(url=a.url, brain_url=a.brain_url, hfov=a.hfov, log=a.log).start()
    events = []                                                         # (t_wall, text) captions for --render
    if a.render:
        emit0 = v._emit
        def _emit(ev, direct=False):
            emit0(ev, direct)
            txt = (f"hand_pick → #{ev['id']} \"{ev['description']}\"" if ev["event"] == "hand_pick" else
                   f"\"{ev['heard']}\" → #{ev['id']} ({ev['bearing_deg']}°)")
            events.append((time.perf_counter(), txt, ev.get("id")))
        v._emit = _emit
        vw = None
    says = sorted((float(s.split(":", 1)[0]), s.split(":", 1)[1]) for s in a.say)
    cap = cv2.VideoCapture(a.source); fps = cap.get(cv2.CAP_PROP_FPS) or 25; i = 0; t_start = time.perf_counter()
    while True:
        ok, img = cap.read()
        if not ok: break
        t = i / fps; i += 1
        if a.duration and t > a.duration: break
        if i % max(1, round(fps / 10)): continue                       # people.py at 10 Hz
        out = pp.step(img, t); v.observe(img, out)
        while says and says[0][0] <= t:
            ev = v.look_command(says.pop(0)[1]); print("look_command", json.dumps(ev), file=sys.stderr)
        if a.render:
            fr = draw(img, out, v, events)
            if vw is None:
                vw = cv2.VideoWriter(a.render + ".tmp.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 10, (fr.shape[1], fr.shape[0]))
            vw.write(fr)
        time.sleep(max(0, t - (time.perf_counter() - t_start)))         # real time, so the 1 Hz loop sees what it would live
    v.stop()
    if a.render and vw is not None:
        vw.release()
        import subprocess
        subprocess.run(["ffmpeg", "-y", "-v", "quiet", "-i", a.render + ".tmp.mp4", "-c:v", "libx264", "-crf", "24", "-pix_fmt", "yuv420p", a.render])
        os.remove(a.render + ".tmp.mp4")
    print(json.dumps({k: round(float(np.median(x)), 1) for k, x in v.timing.items() if x}), file=sys.stderr)


def draw(img, out, v, events):
    import cv2
    im = img.copy(); now = time.perf_counter(); s = im.shape[1] / 1280
    tgt = v.target[0] if v.target and now < v.target[3] else None
    for p in out.get("people", []):
        x0, y0, x1, y1 = map(int, p["box"]); pid = p["id"]
        g_up = now - v.hand_last.get(pid, -1e9) < 1.5 and not p.get("hand_up")
        col = (0, 255, 255) if pid == tgt else (0, 200, 255) if p.get("hand_up") else (255, 255, 0) if g_up else (200, 200, 200)
        cv2.rectangle(im, (x0, y0), (x1, y1), col, 6 if pid == tgt else 2)
        lab = f"#{pid}" + (" HAND(pp)" if p.get("hand_up") else " HAND(gemma)" if g_up else "")
        d = v.desc.get(pid)
        cv2.putText(im, lab, (x0 + 4, max(20, y0 + 24)), cv2.FONT_HERSHEY_SIMPLEX, 0.8 * s, (0, 0, 0), 5)
        cv2.putText(im, lab, (x0 + 4, max(20, y0 + 24)), cv2.FONT_HERSHEY_SIMPLEX, 0.8 * s, col, 2)
        if d:
            cv2.putText(im, d[0][:48], (x0 + 4, max(44, y0 + 50)), cv2.FONT_HERSHEY_SIMPLEX, 0.6 * s, (0, 0, 0), 4)
            cv2.putText(im, d[0][:48], (x0 + 4, max(44, y0 + 50)), cv2.FONT_HERSHEY_SIMPLEX, 0.6 * s, (255, 255, 255), 1)
    h = im.shape[0]; k = 0
    for te, txt, _ in events[-3:]:
        if now - te < 4:
            y = h - 20 - 36 * k; k += 1
            cv2.putText(im, txt, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9 * s, (0, 0, 0), 6)
            cv2.putText(im, txt, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9 * s, (0, 255, 255), 2)
    return im


if __name__ == "__main__":
    main()
