"""Render the v3 stage show (presentation.py: the same lines brain.py plays live) to demo_script/v3/.
   python demo_script/make_script.py
     -> v3/script.json   ordered steps: speaker, human line, staging cues, the Kermit beat it should trigger
                         (scripted turn JSON, or live Gemma in Q&A with a fact-checked fallback)
        v3/turn_NN_<beat>.json   CONTRACT §1 turn JSON per scripted Kermit line
(The 8-beat v2 files at the top of demo_script/ are kept for the existing hero renders.)"""
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import kml, presentation as P

here = os.path.join(os.path.dirname(os.path.abspath(__file__)), "v3")
os.makedirs(here, exist_ok=True)


def turn_of(line, beat):
    t = kml.parse(line)
    probs = [p for p in kml.lint(t) if not (beat == "interject" and p.startswith("two gestures"))]
    assert not probs, (beat, probs)  # "Oh, like..." deliberately stacks look_at + lean_in on "like..."
    return t


TRIGGER = {"turn": "end of the human line (/turn)",
           "turned_away": "PERCEPTION turned_away while_speaking (/event), Kermit barges in",
           "away": "end of the human line while the speaker still faces away (/turn)"}
STAGING = {  # beat id -> staging cues on the human line that leads to it
    "interject": [{"at_word": 3, "who": "kirill", "action": "turns toward the audience and keeps talking to them"},
                  {"at_word": 15, "who": "kirill", "action": "trails off as Kermit cuts in"}],
    "back_turned": [{"at_word": 0, "who": "kirill", "action": "still facing the audience, does not pause"}],
    "respect": [{"at_word": 0, "who": "kirill", "action": "turns back to face Kermit"}],
    "steve_brain": [{"at_word": 0, "who": "steve", "action": "steps in on Kermit's right, facing him"}],
}

steps, n = [], 0
for b in P.BEATS:
    for il in [x for x in P.INTERLUDES if P.BEATS.index(b) and x["after"] == P.BEATS[P.BEATS.index(b) - 1]["id"]]:
        n += 1
        steps.append(dict(step=n, phase=il["phase"], speaker=il["speaker"], line=il["heard"], staging=[],
                          kermit=None, note="presenter to presenter: Kermit stays quiet (BRAIN returns skip)"))
    n += 1
    t = turn_of(b["kml"], b["id"])
    fn = f"turn_{n:02d}_{b['id']}.json"
    json.dump(t, open(os.path.join(here, fn), "w"), indent=1)
    steps.append(dict(step=n, phase=b["phase"], speaker=b["speaker"], line=b["heard"], staging=STAGING.get(b["id"], []),
                      kermit=dict(beat=b["id"], mode="scripted", trigger=TRIGGER[b["trigger"]], kml=b["kml"],
                                  turn=t, file=fn)))
for q in P.QA:
    n += 1
    steps.append(dict(step=n, phase="qa", speaker=q["speaker"], line=q["heard"],
                      staging=[{"at_word": 0, "who": q["speaker"], "action": "asks from the audience, ~3-4 m away"}],
                      kermit=dict(beat=q["id"], mode="live", trigger="speaker_changed(role=judge) then /turn",
                                  fallback_kml=q["fallback"], fallback_turn=turn_of(q["fallback"], q["id"]))))

script = dict(
    version="v3", contract="CONTRACT §6",
    speakers={"kirill": "presenter, ~1 m on Kermit's left, angled ~45°",
              "steve": "presenter, ~1 m on Kermit's right, angled ~45°",
              "judge1": "judge, audience, ~3-4 m", "judge2": "judge, audience, ~3-4 m", "judge3": "judge, audience, ~3-4 m"},
    brain=dict(start="python3 brain.py serve --prompt persona3 --phase presentation",
               presentation="scripted beats, instant; off-script lines not addressed to Kermit -> skip; "
                            "after 'handoff' the session switches to qa",
               qa="live Gemma (persona3 + FACTS); send speaker_changed(role=judge) before each new judge",
               context_speech=P.CTX, context_qa=P.QA_CTX),
    steps=steps)
json.dump(script, open(os.path.join(here, "script.json"), "w"), indent=1)
for s in steps:
    k = s["kermit"]
    print(f"{s['step']:02d} {s['phase']:8} {s['speaker']:7} {s['line'][:60]:62} -> "
          + ("(quiet)" if not k else f"[{k['mode']}] " + (k.get("turn") or k["fallback_turn"])["text"][:70]))
