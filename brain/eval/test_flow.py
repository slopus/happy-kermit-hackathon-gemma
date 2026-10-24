"""Offline test of presentation beats + perception events (LLM stubbed)."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from brain import Brain

b = Brain(prompt="persona3", phase="presentation")
b._deltas = lambda msgs, grammar: (p for p in ["[happy|speaker] {nod 0.5} Generated reply ", "for the test, okay."])
said = lambda evs: next((e.get("text") for e in evs if e["event"] == "turn"), "SKIP" if any(e["event"] == "skip" for e in evs) else None)
S = "kirill"
steps = [
    ("turn", "hey kermit wake up there's a lot of people here"),
    ("turn", "uh so like how are you feeling about ai these days"),
    ("turn", "is that why you're here"),
    ("turn", "and wait are you running in the cloud"),
    ("turn", "so let me tell you how we built this"),                     # off-script -> SKIP
    ("event", {"event": "turned_away", "id": 2, "while_speaking": True, "heard_partial": "so the hand has"}),
    ("event", {"event": "turned_away", "id": 2, "while_speaking": True}),  # duplicate -> SKIP (cooldown)
    ("turn", "and the perception tracks everyone in the room"),         # still away -> back_turned beat
    ("event", {"event": "turned_back", "id": 2}),
    ("turn", "well kermit i respect you it's just hard not to get distracted by this cheerful audience"),
    ("turn", "and that's happy kermit any questions"),                   # handoff -> phase qa
    ("event", {"event": "speaker_changed", "from": 2, "to": 1, "role": "judge"}),
    ("turn", "what model are you running"),                             # qa: generated, judge notes, gaze forced
]
for kind, x in steps:
    evs = list(b.stream(x, "one person in front of you", S) if kind == "turn" else b.event(x, S))
    s = b.session(S)
    print(f"{kind:5} {str(x)[:60]:62} -> {said(evs)!r:70} [phase={s.phase} cursor={s.cursor} away={s.away} role={s.role}]")
print("last user msg:", repr(b.session(S).hist[-1][0]))
# operator override + qa interjection rules
b.set_phase("presentation", "op")
print("beat override:", said(list(b.play_beat("handoff", "op"))), b.session("op").phase)
b2 = Brain(prompt="persona3", phase="qa"); b2._deltas = b._deltas
print("qa interject (presenter):", said(list(b2.event({"event": "turned_away", "while_speaking": True}, "x"))))
b2.event  # judge -> no heckling
list(b2.event({"event": "speaker_changed", "role": "judge"}, "y"))
print("qa interject (judge):", said(list(b2.event({"event": "turned_away", "while_speaking": True}, "y"))))
