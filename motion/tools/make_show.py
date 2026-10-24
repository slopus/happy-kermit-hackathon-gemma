"""Stage script for the v3 show (CONTRACT §6) from VOICE's samples/show/scene.json.

    /var/lib/happy-agent/IsaacLab/.venv/bin/python tools/make_show.py
    /var/lib/happy-agent/IsaacLab/.venv/bin/python render_stage.py samples/stage/show.stage.json -o videos/stage/show.mp4

Layout from scene.json ``people`` (bearing / distance from the robot): Kirill +45 deg and Steve -45 deg at
~1 m, standing angled ~35 deg toward the audience; judges seated at 3.5 m with a few audience extras
behind. VOICE ``staging`` cues become actions (turn toward the audience, turn back, step in); the edit
follows the dialogue: two-shot on each presenter question, the puppet's 3/4 close-up (aimed at whoever
it answers) on its replies, a three-shot for the hand-off, and the judges' wide in the Q&A.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
SHOW = HERE.parent / "voice/samples/show/scene.json"
OUT = HERE / "samples/stage/show.stage.json"
STYLES = {"kirill": "presenter", "steve": "judge_b", "judge1": "judge_a", "judge2": "judge_c", "judge3": "judge_a"}
TOWARD_AUDIENCE = 35  # deg the presenters stand turned from facing the puppet toward the audience


def main():
    vs = json.loads(SHOW.read_text())
    people, pos = [], {}
    for p in vs["people"]:
        y = np.radians(p["yaw_deg"])
        xy = np.array([0.05 + p["dist_m"] * np.cos(y), p["dist_m"] * np.sin(y)])
        pos[p["name"]] = xy
        entry = {"id": p["id"], "name": p["name"], "style": STYLES.get(p["name"], "judge_a"), "role": p["role"],
                 "pos": [round(float(v), 3) for v in xy]}
        if p["role"] == "presenter":
            face_puppet = np.degrees(np.arctan2(-xy[1], 0.45 - xy[0]))
            entry["yaw"] = float(np.radians(face_puppet + np.sign(xy[1]) * TOWARD_AUDIENCE))
            entry["mic"] = True
        else:
            entry["seated"] = True
        people.append(entry)
    # audience extras behind the judges (silent)
    for k, (yd, dist, style) in enumerate([(30, 4.3, "judge_b"), (-30, 4.3, "judge_c"), (8, 4.6, "judge_b"),
                                           (-10, 4.7, "presenter")]):
        y = np.radians(yd)
        people.append({"id": 10 + k, "name": f"aud{k}", "style": style, "role": "audience", "seated": True,
                       "pos": [round(0.05 + dist * np.cos(y), 3), round(dist * np.sin(y), 3)]})

    actions = []
    steve_in = None
    for c in vs["clips"]:
        for s in c.get("staging", []):
            act, who, t = s["action"], s["who"], s["t"]
            if "toward the audience" in act:
                actions.append({"t": t - 0.2, "who": who, "do": "turn_to", "deg": float(np.degrees(np.arctan2(
                    -0.3 * np.sign(pos[who][1]), 1.0))), "dur": 1.0})
            elif "turns back" in act:
                presenter = next(p for p in people if p["name"] == who)
                actions.append({"t": t, "who": who, "do": "turn_to", "deg": float(np.degrees(presenter["yaw"])),
                                "dur": 0.9})
            elif "steps in" in act:
                steve_in = t
    # Steve waits at the side of the stage and steps in when called (step 8 -> 9)
    steve = next(p for p in people if p["name"] == "steve")
    if steve_in is not None:
        steve["start"] = [steve["pos"][0] + 0.9, steve["pos"][1] - 0.9]
        actions.append({"t": steve_in - 2.2, "who": "steve", "do": "move_to", "pos": steve["pos"], "dur": 2.0,
                        "deg": float(np.degrees(steve["yaw"]))})
    # after the hand-off the presenters step back and open up to the room, clearing the stage for the Q&A
    handoff_end = max(c["t"] + c["duration"] for c in vs["clips"] if c.get("phase") == "handoff")
    for p in people:
        if p["role"] == "presenter":
            xy = np.asarray(p["pos"])
            ang = np.arctan2(xy[1], xy[0] - 0.05) * 1.3
            back = np.array([0.05 + 1.5 * np.cos(ang), 1.5 * np.sin(ang)])
            actions.append({"t": handoff_end + 0.2, "who": p["name"], "do": "move_to", "pos": back.round(3).tolist(),
                            "dur": 1.8, "deg": float(np.degrees(np.arctan2(-back[1], 2.5 - back[0])))})
    for c in vs["clips"]:
        if c["who"].startswith("judge"):
            actions.append({"t": c["t"] - 0.7, "who": c["who"], "do": "raise_hand", "hold": 0.9})

    # edit: (time, camera); a cut lands ~0.2 s before the line it introduces
    shots = [[0.0, "stage_wide"]]
    last_presenter = "kirill"
    for c in vs["clips"]:
        who, t = c["who"], max(c["t"] - 0.2, 0.0)
        if who in ("kirill", "steve"):
            last_presenter = who
            cam = f"two_{who}"
        elif who.startswith("judge"):
            cam = "qa_wide"
        else:
            if c.get("phase") == "handoff":
                cam = "three_shot"
            else:
                cam = f"kermit_cu@{c.get('gaze_yaw_deg', 0.0):g}"
        if c["step"] == 8:
            cam = "three_shot"  # Kirill hands over to Steve; Steve walks in
        if cam != shots[-1][1]:
            shots.append([round(t, 2), cam])
        # long puppet answers: cut back to the two-shot for the second half so the listener's reaction shows
        if who == "kermit" and c["duration"] > 6.0 and c.get("phase") in ("speech1", "speech2"):
            shots.append([round(c["t"] + 0.55 * c["duration"], 2), f"two_{last_presenter}"])
    shots.append([round(vs["clips"][-1]["t"] + vs["clips"][-1]["duration"] + 0.4, 2), "stage_wide"])

    S = {"voice_scene": "../../../voice/samples/show/scene.json", "floor_z": -0.8,
         "labels": {"kirill": "Kirill", "steve": "Steve", "judge1": "Judge 1", "judge2": "Judge 2", "judge3": "Judge 3"},
         "people": people, "actions": actions, "shots": shots}
    OUT.write_text(json.dumps(S, indent=1))
    print(f"{OUT.name}: {len(people)} people, {len(actions)} actions, {len(shots)} shots")


if __name__ == "__main__":
    main()
