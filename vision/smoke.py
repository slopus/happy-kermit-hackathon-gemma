"""Mac smoke test for GEMMA VISION (no camera): 3 bundled real frames + their people.py tracks.
  python kermit/vision/smoke.py [--url http://127.0.0.1:8081]
Prints the hand pick, description and look-command resolution per frame, with latency."""
import argparse, json, os, sys, time
from PIL import Image
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import gemma, vision

ap = argparse.ArgumentParser(); ap.add_argument("--url", default="http://127.0.0.1:8081"); a = ap.parse_args()
for s in json.load(open(os.path.join(HERE, "samples", "samples.json"))):
    v = vision.Vision(url=a.url, auto_pick=False, hand_hold_s=0); v._emit = lambda e, direct=False: None
    img = Image.open(os.path.join(HERE, "samples", s["key"] + ".jpg")).convert("RGB")
    out = {"people": s["people"], "speech_active": False}; v.observe(img, out)
    t = time.perf_counter(); up = v.raised(img, out); t_h = time.perf_counter() - t
    ev = v.hand_pick(force=True) if up else None
    t = time.perf_counter(); lc = v.look_command(s["cmd"], emit=False); t_c = time.perf_counter() - t
    print(f"{s['key']}: raised={up} ({t_h*1000:.0f} ms, {len(s['people'])} people)  pick={ev and (ev['id'], ev['description'])}"
          f"\n   '{s['cmd']}' → id {lc['id']} bearing {lc['bearing_deg']}° ({t_c*1000:.0f} ms)")
    print("   timing (ms):", {k: [round(t) for t in x] for k, x in v.timing.items() if x})
