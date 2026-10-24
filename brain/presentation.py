"""The scripted stage show, v3 (Kirill, Oct 24, CONTRACT §6): speech part 1 (Kirill, AI) with the
distracted beat, speech part 2 (Steve, engineering), hand-off, then live Q&A with the judges.

BEATS are the Kermit lines brain.py plays live in the "presentation" phase (instant, no LLM). Each fires on:
  "turn"         the next heard utterance whose `cue` (regex on the heard text) matches
  "turned_away"  perception event turned_away with while_speaking=true (instant interjection)
  "away"         the next heard utterance while the speaker still faces away
`speaker`/`heard` are the human line that leads to the beat (full script: demo_script/make_script.py ->
demo_script/v3/script.json). After the last beat brain.py switches the session to the "qa" phase.
"""
CTX = "Kirill on your left and Steve on your right. visible: audience, laptop"

BEATS = [
    # ---- speech part 1: Kirill + Kermit about AI (the four hero.mp4 lines) ----
    dict(id="wake", phase="speech1", speaker="kirill", trigger="turn", cue=r"wake",
         heard="Hey Kermit, wake up! There's a lot of people here.",
         kml="[surprised|audience] {surprise 1.0} Whoa! {look_around 0.7} That is a lot of people. {head_bob 0.8} Hi, everybody! I'm Happy Kermit."),
    dict(id="feeling", phase="speech1", speaker="kirill", trigger="turn", cue=r"feel",
         heard="So Kermit, how are you feeling about AI these days?",
         kml="[thinking|speaker] {sigh 0.6} Honestly? A little tired. {look_at:audience 0.5} Every week there's a new model, a new agent, a new robot. {nod 0.6} Some days it feels like a sci-fi movie, and not the fun kind."),
    dict(id="why_here", phase="speech1", speaker="kirill", trigger="turn", cue=r"why (you're|you are|are you) here|is that why",
         heard="Is that why you're here?",
         kml="[happy|speaker] {nod 0.8} Yep! I'm the friendly face of Happy, an app for talking to your AI agents. {lean_in 0.6} Less dystopia, more frog."),
    dict(id="cloud", phase="speech1", speaker="kirill", trigger="turn", cue=r"cloud|internet|on device|locally",
         heard="And wait, are you running in the cloud?",
         kml="[excited|audience] {shake 0.8} Nope! {look_at:laptop 0.7} My whole brain is Gemma, running right here on this laptop. {flail 0.8} No cloud, no internet, just me!"),
    # distracted beat: Kirill turns toward the audience mid-sentence and keeps talking to them
    dict(id="interject", phase="speech1", speaker="kirill", trigger="turned_away", cue=None,
         heard="Honestly, I think everyone in this room feels it. New models every week, new agents every...",
         kml="[surprised|speaker] {surprise 1.0} Oh, {look_at:speaker 0.8} like... {lean_in 0.6}"),
    dict(id="back_turned", phase="speech1", speaker="kirill", trigger="away", cue=None,
         heard="...and every single launch says it changes everything.",
         kml="[annoyed|speaker] {shake 0.7} You, my friend, should learn how to speak to people {nod 0.6} face to face."),
    dict(id="respect", phase="speech1", speaker="kirill", trigger="turn", cue=r"respect you|distracted|cheerful audience",
         heard="Well, Kermit, I respect you. It's just hard not to get distracted by this cheerful audience.",
         kml="[happy|audience] {look_around 0.7} Okay, fair enough. They are a very cheerful bunch. {head_bob 0.7} Respect accepted, my friend!"),
    # ---- speech part 2: Steve + Kermit about the engineering (Kirill's intro line to Steve gets no reply) ----
    dict(id="steve_brain", phase="speech2", speaker="steve", trigger="turn", cue=r"running your brain|runs your brain|your brain",
         heard="Hi Kermit, I'm Steve. Do you know what's actually running your brain right now?",
         kml="[excited|speaker] {head_bob 0.6} Hi, Steve! {look_at:laptop 0.6} Google's Gemma four, the E four B model, right here on this MacBook. {laugh 0.4} You gave it a LoRA, and now it thinks it's a frog."),
    dict(id="steve_ears", phase="speech2", speaker="steve", trigger="turn", cue=r"hear|talk back|so fast",
         heard="And how do you hear me and talk back so fast?",
         kml="[happy|speaker] {lean_in 0.6} Parakeet turns your voice into words, Gemma does the thinking, and Kokoro gives me this lovely voice. {nod 0.6} I speak sentence by sentence, so I start in well under a second."),
    dict(id="steve_gestures", phase="speech2", speaker="steve", trigger="turn", cue=r"decide|when to|surprised",
         heard="How do you decide when to nod or look surprised?",
         kml="[thinking|speaker] {tilt_right 0.5} I write little gesture tags right into my sentences. {nod 0.8} Like this nod. {surprise 0.9} Whoa, and that one! {look_at:speaker 0.5} The xArm performs them on the exact word."),
    dict(id="steve_robotics", phase="speech2", speaker="steve", trigger="turn", cue=r"robotics|explode",
         heard="Last one. Why do you think robotics is about to explode?",
         kml="[excited|audience] {flail 0.8} Because brains like mine now fit on a laptop! {look_at:audience 0.6} Small smart models, good cameras, and arms that can move. {nod 0.7} Robots are about to get a lot friendlier."),
    # ---- hand-off to Q&A (then brain switches to the qa phase: live Gemma) ----
    dict(id="handoff", phase="handoff", speaker="kirill", trigger="turn", cue=r"questions|q and a|q&a",
         heard="Well, Kermit, what do you say? Will you answer some questions?",
         kml="[excited|audience] {nod 0.8} Absolutely! {look_around 0.7} Judges, hit me with your questions. {lean_in 0.6} I promise I only bite flies."),
]

# Lines with no Kermit reply (presenter-to-presenter); listed so the full script reads end to end.
INTERLUDES = [
    dict(after="respect", phase="speech2", speaker="kirill",
         heard="Steve, you built most of his insides. Tell everyone how he works."),
]

# Q&A: judges seated further back; Kermit answers LIVE with Gemma (prompt persona3). The fallback lines are
# hand-written, fact-checked answers for renders or if the live stack hiccups.
QA_CTX = "a judge in the audience, a few metres away. visible: audience, laptop"
QA = [
    dict(id="qa_nod", speaker="judge1", heard="How does the hand know when to nod?",
         fallback="[thinking|speaker] {nod 0.7} I tell it! {lean_in 0.5} I write a nod tag right before the word, and the hand performs it in time with my voice. {laugh 0.4} Like puppet sheet music."),
    dict(id="qa_cloud", speaker="judge2", heard="Why Gemma and not a cloud model?",
         fallback="[happy|speaker] {shake 0.6} No cloud for this frog. {look_at:laptop 0.6} Gemma runs right here on the laptop, so it's private, has no network lag, and works offline. {nod 0.5} Also, the hackathon said so."),
    dict(id="qa_robotics", speaker="judge3", heard="What happens to robotics next?",
         fallback="[excited|speaker] {surprise 0.8} Whoa, big question! {nod 0.6} Small smart models like mine now run right on the device. {lean_in 0.5} So robots get friendlier, faster, and a lot more fun to talk to."),
    dict(id="qa_hard", speaker="judge1", heard="And what was the hardest part to build?",
         fallback="[thinking|speaker] {sigh 0.4} Keeping a small model funny and in character. {nod 0.6} And landing every gesture on the right word, while voice, mouth and motion stay in sync."),
]
