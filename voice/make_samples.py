"""Re-render samples/ (demo lines + hero-video script) with the current voice and mouth settings."""
import glob, json, os, sys, types
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kermit_voice import Performer

HERE = os.path.dirname(os.path.abspath(__file__))
a = types.SimpleNamespace(voice=os.environ.get("KERMIT_VOICE", "am_fenrir"), tts_backend=os.environ.get("KERMIT_TTS_BACKEND", "torch"),
                          device=None, no_frog=False, out=os.path.join(HERE, ".context", "render"), play=False)
p = Performer(a)
jobs = [(t, os.path.join(HERE, "samples", t["id"])) for t in json.load(open(os.path.join(HERE, "lines", "demo_turns.json")))]
for f in sorted(glob.glob(os.path.join(HERE, "..", "brain", "demo_script", "turn_0*.json"))):
    jobs.append((json.load(open(f)), os.path.join(HERE, "samples", "demo_script", os.path.basename(f)[:-5])))
for turn, dst in jobs:
    wav, tl = p.one(turn={k: v for k, v in turn.items() if k != "id"})
    os.replace(wav, dst + ".wav")
    tl["audio"] = os.path.basename(dst) + ".wav"
    json.dump(tl, open(dst + ".timeline.json", "w"), separators=(",", ":"))
    json.dump(turn, open(dst + ".turn.json", "w"), indent=1)
for f in glob.glob(os.path.join(a.out, "*")):
    os.remove(f)
