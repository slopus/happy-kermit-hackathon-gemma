"""The scripted stage presentation (Kirill, Oct 24): opening scene, turn-around beat, hand-off to Q&A.

Each beat fires on a trigger:
  "turn"         the next heard utterance (optionally re-synced by `cue`, a regex on the heard text)
  "turned_away"  perception event turned_away with while_speaking=true (instant interjection)
  "away"         the next heard utterance while the speaker still has their back turned
`heard` is the expected human line, used for renders and docs, not for matching. After the last beat
brain.py switches the session to the "qa" phase.
"""
CTX = "one person in front of you. visible: audience, laptop"

BEATS = [
    dict(id="wake", trigger="turn", cue=r"wake", heard="hey kermit wake up there's a lot of people here",
         kml="[surprised|audience] {surprise 1.0} Whoa! {look_around 0.7} That is a lot of people. {head_bob 0.8} Hi, everybody! I'm Happy Kermit."),
    dict(id="feeling", trigger="turn", cue=r"feel", heard="so kermit how are you feeling about ai these days",
         kml="[thinking|speaker] {sigh 0.6} Honestly? A little tired. {look_at:audience 0.5} Every week there's a new model, a new agent, a new robot. {nod 0.6} Some days it feels like a sci-fi movie, and not the fun kind."),
    dict(id="why_here", trigger="turn", cue=r"why (you're|you are|are you) here|is that why", heard="is that why you're here",
         kml="[happy|speaker] {nod 0.8} Yep! I'm the friendly face of Happy, an app for talking to your AI agents. {lean_in 0.6} Less dystopia, more frog."),
    dict(id="cloud", trigger="turn", cue=r"cloud|internet|on device|locally", heard="and wait are you running in the cloud",
         kml="[excited|audience] {shake 0.8} Nope! {look_at:laptop 0.7} My whole brain is Gemma, running right here on this laptop. {flail 0.8} No cloud, no internet, just me!"),
    # turn-around beat: Kirill turns his back while talking, keeps talking, then turns back
    dict(id="interject", trigger="turned_away", cue=None, heard="(Kirill turns his back mid-sentence)",
         kml="[surprised|speaker] {surprise 1.0} Oh, {look_at:speaker 0.8} like... {lean_in 0.6}"),
    dict(id="back_turned", trigger="away", cue=None, heard="(Kirill keeps talking with his back turned)",
         kml="[annoyed|speaker] {shake 0.7} You, my friend, should learn how to speak to people {nod 0.6} face to face."),
    dict(id="respect", trigger="turn", cue=r"respect you|distracted|cheerful audience",
         heard="well kermit i respect you it's just hard not to get distracted by this cheerful audience",
         kml="[happy|audience] {look_around 0.7} Okay, fair enough. They are a very cheerful bunch. {head_bob 0.7} Respect accepted, my friend!"),
    # hand-off: the presenter wraps up; Kermit opens the floor (then brain switches to the qa phase)
    dict(id="handoff", trigger="turn", cue=r"question|q and a|q&a|that's (it|our|all)|thank you|judges",
         heard="and that's happy kermit any questions",
         kml="[excited|audience] {look_around 0.7} Okay, judges, {lean_in 0.6} hit me with your questions! {nod 0.6} I promise I only bite flies."),
]
