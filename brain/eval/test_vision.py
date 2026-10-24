"""Live test: spatial commands, hand_pick, look_command (VISION before / after / never), qa phase."""
import json, os, sys, threading, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from brain import Brain
b = Brain(sys.argv[1], "persona3", phase="qa"); b.turn("hello", "one person", "w")
CTX = "Kirill on your left. visible: audience, laptop"
def show(tag, evs, t0):
    t = next((e for e in evs if e["event"] == "turn"), None)
    st = b.last_stats
    print(f"{tag:46} {time.perf_counter()-t0:5.2f}s  " + (f"gaze={t['gaze']:16} {st['raw']}" if t else "SKIP"))
for h in ["kermit look to your right", "turn left", "look up", "look at me", "face the audience", "turn around"]:
    for i in range(2):
        t0 = time.perf_counter(); show(f"[dir] {h}", list(b.stream(h, CTX, f"d{h}{i}")), t0)
for d in ["the guy with the drink", "the woman in the red jacket", "the person in the back with a laptop"]:
    for i in range(2):
        t0 = time.perf_counter(); show(f"[hand_pick] {d}", list(b.event({"event": "hand_pick", "id": 3, "bearing_deg": -20, "description": d}, f"h{d}{i}")), t0)
t0 = time.perf_counter(); show("[hand_pick] same id again (cooldown)", list(b.event({"event": "hand_pick", "id": 3, "description": "x"}, "hthe guy with the drink1")), t0)
show("[hand_pick] presentation phase", list(Brain(sys.argv[1], "persona3", phase="presentation").event({"event": "hand_pick", "id": 4, "description": "x"}, "p")), time.perf_counter())
H = "kermit look at the guy with the drink"
# look_command first, then /turn
for i in range(2):
    sid = f"lc-first{i}"; list(b.event({"event": "look_command", "id": 5, "bearing_deg": 25, "description": "the guy with the drink", "heard": H}, sid))
    t0 = time.perf_counter(); show("[look] VISION first", list(b.stream(H, CTX, sid)), t0)
# /turn first, VISION answers 0.8 s later (from another thread)
for i in range(2):
    sid = f"lc-late{i}"
    threading.Timer(0.8, lambda sid=sid: list(b.event({"event": "look_command", "id": 6, "description": "the woman with the coffee", "heard": "look at the woman with the coffee"}, sid))).start()
    t0 = time.perf_counter(); show("[look] VISION 0.8 s late", list(b.stream("look at the woman with the coffee", CTX, sid)), t0)
# unresolved and never
sid = "lc-null"; list(b.event({"event": "look_command", "id": None, "description": "the guy with the hat", "heard": "look at the guy with the hat"}, sid))
t0 = time.perf_counter(); show("[look] VISION id=null", list(b.stream("look at the guy with the hat", CTX, sid)), t0)
t0 = time.perf_counter(); show("[look] VISION never (1.5 s wait)", list(b.stream("look at the lady with the balloon", CTX, "lc-never")), t0)
