# Happy Kermit: interfaces between workstreams (v0, the coordinator owns this file)

Pipeline per turn (all on-device; target is Kirill's MacBook M5 Max 128 GB, dev on this Linux node):

```
mic → STT (Parakeet) → BRAIN (Gemma E4B, in character) → turn JSON
turn JSON → VOICE (local TTS) → audio + word timings + mouth curve
mouth curve + gesture events + gaze → MOTION (xArm6 + dexterous hand puppet)
camera → face tracker → gaze target (MOTION, real time)
```

## 1. BRAIN → VOICE / MOTION: turn JSON
```json
{
  "text": "Hi-ho! Well, that is a really big robot arm.",
  "emotion": "excited",
  "gestures": [
    {"word": 0, "type": "head_bob", "intensity": 0.8},
    {"word": 5, "type": "look_at", "target": "arm", "intensity": 0.5},
    {"word": 7, "type": "surprise", "intensity": 1.0}
  ],
  "gaze": "speaker"
}
```
- `word` is a 0-based index of a word in `text`, split on whitespace. VOICE turns it into seconds using the TTS word timings.
- `emotion` sets the baseline posture and mouth energy. Values: neutral, happy, excited, sad, surprised, thinking, embarrassed, annoyed.
- `intensity` runs from 0 to 1.
- `gaze` values: speaker (track a face), audience, down, up, `target:<name>`.

### Gesture vocabulary (v0, MOTION implements each one)
nod, shake, tilt_left, tilt_right, head_bob, look_at, look_around, surprise (mouth wide + head back), laugh (fast mouth + bob), sigh (slow droop), shrug, lean_in, lean_back, flail (arms/fingers, Kermit-style excitement), cover_face.

## 2. VOICE → MOTION: performance timeline
```json
{
  "audio": "turn_0007.wav",
  "sample_rate": 24000,
  "words": [{"w": "Hi-ho!", "t0": 0.00, "t1": 0.42}],
  "mouth": {"hz": 60, "values": [0.0, 0.12, 0.55]},
  "events": [{"t": 0.00, "type": "head_bob", "intensity": 0.8}],
  "emotion": "excited",
  "gaze": "speaker"
}
```
- `mouth` holds openness from 0 to 1, sampled at `hz` and aligned with the audio start. It comes from the lip-sync model or the audio envelope.
- v0.1 additions (VOICE): a top-level `duration` (seconds), plus `word` (source word index) and `target` (for look_at) on each event. The mouth curve is one flap per syllable, closes in pauses, dips on m/b/p, and is scaled by emotion. There is no actuator lead by default; use `lead_ms` to regenerate with one.
- Samples: kermit/voice/samples/<id>.{wav,timeline.json,turn.json}

## 1b. BRAIN → VOICE streaming
NDJSON: a header event (emotion, gaze), then one event per sentence (text plus gestures with global word indices), then the final turn event. Details are in kermit/brain/FORMAT.md.
Endpoint: `POST :8765/turn` (BRAIN server), response streamed as NDJSON. Agreed by BRAIN and VOICE.
Runtime: llama.cpp + GBNF grammar (without the grammar, E4B invents tags).
- MOTION starts audio playback and motion on the same clock. Target A/V offset is under 80 ms.

## 4. PERCEPTION → MOTION: person tracking (Kirill, Oct 24)
The camera is static and mounted at the base of the arm, looking forward along the arm's +x (the front of the base). Its pose relative to the arm base frame is a fixed transform in config, with defaults: position (0.05, 0, 0.10) m, no rotation (optical axis = base +x).

PERCEPTION streams ~15–30 Hz NDJSON (or a shared latest-value):
```json
{"t": 12.345, "present": true, "yaw_deg": -18.2, "pitch_deg": 6.5, "dist_m": 1.4, "conf": 0.92,
 "target_base": [1.30, -0.43, 0.25]}
```
- `yaw_deg` / `pitch_deg`: bearing of the main person's face in the arm base frame. +yaw = toward base +y (left as seen by the robot).
- `dist_m`: estimated from face size (assume a 0.16 m face width). `target_base`: face point in base frame, metres.
- Main person = largest/closest face, with hysteresis (switch only if the new face is >1.3× larger for >0.5 s).
- No face for >1 s: `present: false`. MOTION eases back to the neutral/audience direction over ~1 s.

MOTION uses it as a **bias**, not a lock:
- The whole arm pose (base yaw joint1 plus head aim) leans toward the person, smoothed with a ~0.3–0.5 s time constant and clamped to ±60° yaw, ±20° pitch.
- Blend: the gaze bias sets the "home" direction; gestures (nod, whoa, turn, etc.) play on top of it, relative to that direction.
- `gaze: "speaker"` in a turn means full weight on the person; `audience` means a partial bias (~0.4); `target:x` overrides temporarily.
- Puppetry rule: Kermit faces the person, with a slight 3/4 angle toward the audience camera if one is defined.
- As implemented: joint1 takes 60% of the yaw and the head aim takes 40%. Critically damped, τ = 0.4 s. Weights: speaker 1.0 / audience 0.4 / look_at 0 (override).
- v0.2 additions: an `n_faces` field. Camera FOV limits coverage (78° HFOV = ±39°; a C922 is ~70°, so ±35°), and the ±60° clamp only matters with a wider lens. Set `--hfov` to the real camera.
- Live transport: PERCEPTION writes a latest-value JSON file (`--latest path`). MOTION polls it and resamples on its own clock.

## 5. PERCEPTION v2: multi-person, active speaker, attention (Kirill, Oct 24)
§4 tracks one "main person". v2 tracks everyone and decides **who is speaking** and **whether they are facing Kermit**.

Per frame, PERCEPTION writes a latest-value JSON (and an NDJSON log):
```json
{"t": 12.3, "speaker_id": 2, "speech_active": true,
 "people": [
   {"id": 1, "yaw_deg": -25.0, "pitch_deg": 4.0, "dist_m": 1.6, "facing": "toward",
    "speaking_p": 0.08, "has_mic": false, "role": "judge", "conf": 0.9},
   {"id": 2, "yaw_deg": 12.0, "pitch_deg": 6.0, "dist_m": 1.2, "facing": "away",
    "speaking_p": 0.91, "has_mic": true, "role": "presenter", "conf": 0.8}]}
```
- `id`: a stable track ID that survives the person turning around (body/re-ID tracking, not just faces).
- `facing`: toward | side | away. "away" means the body is present but the face is not visible or the head is turned more than 100°.
- `speaking_p`: fused active-speaker probability from:
  - lip/face activity (an ASD model, when the face is visible)
  - holding a microphone (open-vocabulary detector)
  - VAD from the mic, combined with temporal continuity
  - optional: direction of arrival from a mic array
- `speaker_id`: argmax with hysteresis. MOTION's gaze target is `speaker_id`'s bearing (it replaces §4's main person).

Discrete events go to BRAIN (NDJSON, deduplicated):
- `{"event":"turned_away","id":2,"while_speaking":true}`
- `{"event":"turned_back","id":2}`
- `{"event":"speaker_changed","from":2,"to":1,"role":"judge"}`
- `{"event":"person_arrived"|"person_left","id":…}`

BRAIN can react to `turned_away while_speaking` proactively, without waiting for a question.

## 6. Stage v3 and the live room simulation (Kirill, Oct 24 ~03:00 PT)
**Reference look: `kermit-demo-latest.mp4` = motion/videos/hero.mp4** (byte-identical). All new renders use its framing and lighting.

### Run of show
1. **Speech, part 1: Kirill + Kermit about AI.** Burnt out by the pace of AI, dystopian vibes, Happy as a friendly face. Kirill gets distracted by the cheerful audience and turns toward them; Kermit calls him out ("You, my friend, should learn how to speak to people face to face."); Kirill: "Well, Kermit, I respect you. It's just hard not to get distracted by this cheerful audience."
2. **Speech, part 2: Steve + Kermit about the engineering.** How it's built (Gemma on-device, fine-tune, voice, hand), and why robotics is about to explode.
3. **Hand-off.** Kirill: "Well, Kermit, what do you say? Will you answer some questions?" Kermit agrees.
4. **Q&A.** Judges at the very end, seated further back (~3–4 m), in the audience.

### Layout
- Kermit is centre stage. **Kirill and Steve stand close (~0.8–1.2 m), one on each side**, angled about 45° so they can see the audience peripherally while talking to Kermit.
- Audience and judges are further away. Perception tracks the **near people** (presenters) for speaker and addressee, and the judges only in the Q&A phase.
- Facing is a soft cue for who is being addressed, not a hard trigger. The "distracted" beat is scripted.

### One shared acoustic room (the real-time test)
Everything is a sound source in ONE simulated room (reverb, distances), and Kermit has ONE mic in that room:
- sources: Kirill, Steve, judges, **Kermit's own loudspeaker**, crowd babble, claps, footsteps, chairs, laptop fans; the venue noise level varies over time
- Kermit's mic hears the mix, including his own voice, so **echo cancellation / self-voice suppression is required**
- the harness runs **in wall-clock real time**: the humans' lines play when the script says so (or when Kermit finishes), Kermit's pipeline runs live (streaming STT → end-of-turn → BRAIN → TTS → speaker into the room), and nothing is pre-mixed
- the sim camera at the arm base renders the scene live (MuJoCo), and its frames feed PERCEPTION
- **Camera pose (MOTION finding):** at §4's base pose (z = 0.10, +x), the presenters at ±45°/1 m are cropped at the waist and the puppet's body occludes the lens. Default for the sim is now a raised side post at about (0.10, 0.25, 0.55) in the base frame, looking forward and slightly down, unless Kirill picks otherwise. A wider lens (≥100° HFOV) helps keep both presenters in frame. Samples: motion/videos/live/armcam_{A,B,C}.png
- everything is logged with timestamps, and the video is rendered afterwards from the log; the render is offline, the run is real

## Priority gestures (Kirill, Oct 23)
These must look good first:
1. **talking**: mouth opens and closes, following the audio.
2. **nod**
3. **whoa / surprise**: mouth wide, head jerks back, maybe a shake.
4. **head turn**: small look left and right.

The rest of the vocabulary comes after these.

## 3. Safety / scope
- Git: kermit/ is the private repo slopus/happy-kermit-hackathon-gemma (Kirill, Oct 24). Only the coordinator commits and pushes; subtasks do not.
- MOTION is **simulation only**. The deliverable is a rendered video of the hand puppet performing, with the audio muxed in.
- No subtask sends commands to the real arm or hand.
