"""The scripted stage show, v4 (Kirill solo, Oct 24): a 7-slide talk on what agents can build in a day and what
still needs us. Kermit comments, agrees and adds references, then the session goes to live Q&A with the judges.

BEATS are the Kermit lines brain.py plays live in the "presentation" phase (instant, no LLM). Each fires on the
next heard utterance whose `cue` (regex on the heard text) matches. With OFF_SCRIPT_LIVE, anything else Kirill
says gets a live Gemma reply and the script position stays put. `heard` is the human line that leads to the beat
(full script: demo_script/make_script.py -> demo_script/v3/script.json). After the last beat brain.py switches the
session to the "qa" phase.
"""
CTX = "Kirill on your left, presenting. visible: audience, laptop"
OFF_SCRIPT_LIVE = True  # unexpected lines get a live reply instead of silence
INTERJECT = False  # Kirill faces the audience a lot in a solo talk: no turned-away heckles

BEATS = [
    dict(id="intro", phase="talk", speaker="kirill", trigger="turn",
         cue=r"i'm kirill|i am kirill|my name|hi everyone|hello everyone|this is kermit",
         heard="Hi everyone, I'm Kirill, and this is Kermit.",
         kml="[happy|audience] {head_bob 0.8} Hi, everybody! {sigh 0.5} I've been hanging off this arm all day. {nod 0.6} I could really use a reset."),
    dict(id="agents", phase="talk", speaker="kirill", trigger="turn",
         cue=r"agents|looked at the code|pipeline|fine.?tun|hour of my",
         heard="The latest agents finally work really well. This hackathon I barely looked at the code. They wrote the pipeline, fine-tuned the models, ran the simulations. About an hour of my time.",
         kml="[happy|audience] {nod 0.7} It's true. {look_at:audience 0.6} Half of X this month is people saying, I didn't write a single line of code. {laugh 0.4} He's one of them."),
    dict(id="how", phase="talk", speaker="kirill", trigger="turn",
         cue=r"walk (us|me) through|how you work|how do you work|how it works",
         heard="Kermit, walk us through how you work.",
         kml="[excited|speaker] {lean_in 0.6} Parakeet hears you, Gemma thinks, that's my brain, {look_at:laptop 0.6} Kokoro talks, and the arm moves on the exact word. {flail 0.7} All on this laptop."),
    dict(id="real_world", phase="talk", speaker="kirill", trigger="turn",
         cue=r"real world|stuck finger|wouldn.t turn on|electronics|in reality",
         heard="Then came the real world. In simulation, everything looked great. In reality, one hand had a stuck finger and the other wouldn't turn on. We swapped the electronics to get it working.",
         kml="[thinking|speaker] {sigh 0.6} Sim to real. {tilt_right 0.5} In the simulation, I had perfect posture."),
    dict(id="impressed", phase="talk", speaker="kirill", trigger="turn",
         cue=r"underwhelm|impressed|fans|simulator",
         heard="Talking to him, I'm underwhelmed and impressed at the same time. My fans don't even turn on. The iOS simulator works this Mac harder than his brain.",
         kml="[happy|audience] {head_bob 0.6} Five gigs of brain, zero fans. {laugh 0.4} I don't think very hard, but I think very fast."),
    dict(id="value", phase="talk", speaker="kirill", trigger="turn",
         cue=r"added value|dumb pipe|chores|laundry|my value",
         heard="So where's the added value? Local models are great dumb pipes. But chores? Nobody wants half-folded laundry. And if agents write all the code, where's my value as an engineer?",
         kml="[thinking|speaker] {nod 0.6} François Chollet said it last week. {look_at:audience 0.5} Agents write almost all the code, and engineers have never worked harder. {nod 0.5} The bottleneck just moves."),
    dict(id="close", phase="handoff", speaker="kirill", trigger="turn",
         cue=r"no offen[cs]e|your position|kermit's position|in your shoes",
         heard="So, no offence, Kermit... but you don't want to be in Kermit's position.",
         kml="[happy|speaker] {shake 0.6} None taken. {look_at:audience 0.6} Someone else's hand does all the work, and I take the credit. {surprise 0.9} Wait. {look_around 0.7} That's you guys with your agents."),
]

INTERLUDES = []  # no presenter-to-presenter lines

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
