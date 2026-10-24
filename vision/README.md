# Happy Kermit: GEMMA VISION (Gemma 4 E4B as the semantic layer on top of PERCEPTION)

**Answer to "can Gemma do the vision part?"**
- **Who is speaking:** no, keep the specialists (Light-ASD + VAD in people.py).
- **Raised hands, describing people, described look commands:** yes. Gemma does these well and fast, as long as it
  works on people.py's tracks (set-of-marks) rather than outputting pixel boxes.
- **Fine-tune:** not needed (see below).

## Results (Gemma 4 E4B Q4_K_M + mmproj F16, llama.cpp 970b51c, one RTX 4000 Ada; real stage-view footage)
| task | specialist (people.py) | Gemma best variant | Gemma ms |
|---|---|---|---|
| (a) who is speaking (Columbia panel, 81 items, 1 GT speaker) | final fusion speaker_id **90.1%** (raw ASD argmax 88.9%) | 4-frame marked strip 65.4% (1 frame 46.9%, face-crop grid 40.7%; chance 37.6%) | 383 |
| (b) pick a raised hand (23 frames) | hand_up 87% (P 100 / R 64) | yes/no per person crop 87% (P 89 / R 93); crop gallery 78%; marked frame 61% | 140 per person |
| (b) **hybrid**: hand_up first, then Gemma yes/no | | **95.7%** (set P 89 / R 96) | |
| (b) describe a person, ≤8 words (64 people) | torso-colour heuristic, ~45% usable | crop → "the man in brown jacket holding a microphone", ~90% usable (judged by eye) | 205 |
| (c) described look command (41 commands) | colour and left/right word rules 14.6% | marked frame + crop gallery **82.9%** (marked frame alone 61%) | 281 |
| (c) crowd with no tracks (Gemma box_2d on the full frame) | — | 4 of 6 boxes plausible (results/ex_locate_crowd.jpg) | 600 |

Specialists by task:
- (a) speaker = people.py; Gemma is not a lip-reader.
- (b) and (c) = Gemma over people.py's tracks.

Decisions:
- **Prompt format:** Gemma answers with a track id, constrained by a GBNF grammar (`gemma.ask_choice`). Every answer is
  a valid id or `none`. Gemma never volunteers "none" on frames with no raised hand (0/7), so people.py tracks plus a
  per-person yes/no gate the picks.
- **Box offset:** fitted against YOLO person boxes on native-aspect frames (results/calib.json, 49 pairs):
  - x-centre error 0.1% W, bottom edge 0.1% H, top +3.8% H, median IoU 0.64. **No meaningful offset for people.**
  - Letterboxing to a square adds a +22% H error (the model ignores the black bars). So: send native aspect, never pad.
  - Box listing takes ~3 s per frame, so it's used only as the crowd fallback.
- **Fine-tune:** none.
  - (a): Light-ASD does lip-sync better than any prompt and runs at 5 Hz.
  - (b) and (c): already at or above the specialists. The labelled set (23 frames, ~140 person crops) is also too
    small to train on without testing on the training data.
  - The Kermit persona LoRA stays BRAIN-only: vision runs on a **separate llama-server without LoRA**.
- **Speaker tie-break (dropped):** using Gemma only when ASD's margin is small scored 85.2%, below ASD alone (88.9%).

## How it runs (hybrid)
```
camera → people.py (YOLO-pose + ByteTrack + YuNet + Light-ASD, ~20 Hz) → tracks {id, box, yaw, hand_up, speaking_p…}
                         └─ vision.Vision.observe(frame, out) (tap on PeopleThread.step)
vision loop, ~1 Hz, paused while BRAIN is generating:
   Gemma yes/no on ≤4 least-recently-checked crops  ∪ people.py hand_up  → raised-hand set
   descriptions cached per id
   hand held ≥1 s, nobody talking, ≥8 s since the last pick, BRAIN phase != presentation (GET /state) → POST /event hand_pick
     (people.py hand_up first, then the biggest box; never the current speaker or a presenter;
      one pick per raised-hand episode)
   fewer than 3 tracks (wide crowd shot) → Gemma box_2d "a raised hand" → bearing-only hand_pick
heard (final transcript) "… look at / turn to / face <description>" → Gemma(marked frame + crop gallery) → id
   → POST /event look_command (inside BRAIN's 1.5 s hold window; measured ~0.15–0.45 s);
     nobody tracked matches → Gemma box_2d → bearing only
gaze: vision.gaze() overrides PERCEPTION's §5 gaze for 3 s after a pick or command (follows the track's bearing)
```

Events, agreed with BRAIN (implemented there, `person_<id>` targets). Shape:
```
{"event":"hand_pick","id":7,"description":"the person in red jacket holding dark object","bearing_deg":-6.6,"pitch_deg":-14.9,"conf":0.9,"session_id":"kermit"}
{"event":"look_command","id":3,"description":"the guy in the orange hoodie","heard":"Kermit, look at the guy in the orange hoodie.","bearing_deg":-12.9,"pitch_deg":-17.3,"session_id":"kermit"}
```
- `id`: people.py's track id, or null when the target came from the crowd fallback (then `box_2d` is included).
- `bearing_deg`: base frame, + = Kermit's left (§4). With a PTZ camera, `Vision(ptz=lambda: (pan, tilt))` adds the
  current pan/tilt.
- Simple directions ("look left / right / up / at me") are not sent: BRAIN's grammar handles them.

## Verified end to end (this box)
`run_kermit.py --camera clips/biz_hands2.mp4 --vision --mic-wav "Kermit, look at the guy in the orange hoodie."`:
- **hand_pick:** id 7 "the person in red jacket holding dark object" → BRAIN interjection, gaze target `person_7`.
- **look command:** heard → `look_command` id 3 → reply "Oh! He's got a nice hoodie on. What about him catches your
  eye?" with `look_at:person_3` and gaze `target:person_3`.
- **MOTION:** gaze lines `{"source":"vision:hand_pick","id":7,…}`.

Annotated runs: demo/*.mp4 (boxes: grey = tracked, orange = hand_up from people.py, cyan = hand from Gemma only,
thick yellow = Kermit's current target). Example frames: results/ex_*.jpg.

## Mac (M5 Max)
```bash
bash kermit/brain/mac/setup.sh                 # once: llama.cpp Metal + base E4B (shared with vision)
bash kermit/brain/mac/run.sh &                 # BRAIN: :8080 model (+ Kermit LoRA), :8765 brain
bash kermit/vision/mac/run_vision.sh &         # VISION: :8081, same base GGUF + mmproj (990 MB, fetched once), no LoRA, then smoke.py
python kermit/run_kermit.py --camera 0 --hfov 70 --ptt --tts-backend mlx --play --vision \
       --perception-args --mic-model ''        # '' if the venue has table/stand mics (PERCEPTION's advice)
```
- `smoke.py` (no camera, 3 bundled frames) prints the hand pick, description and look-command id plus latency per call.
- Here: yes/no ~90–150 ms per person, description ~140–250 ms, command ~100–330 ms. Expect the same or better on the
  M5 Max (not measured).
- Memory: the second server mmaps the same 5 GB GGUF, plus ~1 GB mmproj and KV.
- If memory is tight, drop the separate server and point `--vision-url` at :8080 with `--mmproj` added there. **But**
  the LoRA would then apply to vision prompts too (untested), so the separate server is the recommendation.

## Files
- `vision.py`: the module (Vision class: observe/attach/start, hand_pick, look_command, on_heard, gaze, context;
  standalone `--source clip --say "T:text" --render out.mp4`)
- `gemma.py`: llama-server client, GBNF id choice, set-of-marks drawing, crop gallery
- `testset.py`, `speaker_set.py`, `labels.json`, `items*.json`: test set (clips/, frames/, frames_col/, pp/ =
  people.py logs per clip)
- `eval_speaker.py`, `eval_people.py`, `eval_variants.py`, `calib.py`, `locate_test.py` → results/*.json
- Sources:
  - YouTube stock/CC snippets ≤90 s (yt-dlp sections): concert, classroom, conference hand-raise, PBS town hall
    with mic, panel mic, comedy, auditorium
  - Columbia ASD (PERCEPTION's copy)

## Limits
- Hand yes/no false positives: a teacher pointing, kids with hands at their heads. people.py's presenter role filter
  only knows distance, so a teacher at 2–3 m isn't excluded.
- Descriptions sometimes get gender or colour wrong in dark scenes (concert), or call a blonde person brown-haired.
  The prompt bans age, race, body and looks.
- On wide crowd shots people.py tracks 0–3 people at imgsz 640. The crowd fallback (Gemma box_2d) only gives a
  bearing, and it was right on about 4 of 6 frames.
- All numbers come from a small hand-labelled set (23 frames / 41 commands / 81 speaker items), so they are
  ±10 points.
