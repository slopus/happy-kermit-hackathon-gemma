"""(b) raised-hand pick + description and (c) described-look commands: Gemma 4 E4B vs a specialist baseline.

  python eval_people.py            → results/hands.json, results/cmds.json, results/describe.json
Specialist baseline:
  (b) pick = people.py hand_up (largest box first); description = torso colour name ("the person in the red top")
  (c) colour words + left/right/front words in the command matched against tracks (no semantics beyond that)
"""
import json, os, re, sys, time
import numpy as np
from PIL import Image
import gemma

HERE = os.path.dirname(os.path.abspath(__file__))

HAND_Q = ("You are Kermit, a puppet on stage, looking out at the audience. Each tracked person has a numbered tag. "
          "Pick ONE person who is raising a hand right now (a hand or arm held up, above the shoulder, as if asking to speak). "
          "Ignore the teacher or presenter at the front. If nobody in a tag is raising a hand, answer none. Answer with just the number.")
DESC_Q = ("Describe the person in the {where} in at most 8 words, as a host would call on them in a crowd, e.g. "
          "'the woman in the red jacket' or 'the guy with the drink'. Mention clothing colour or an object they hold. "
          "Never mention age, race, body, or attractiveness. Answer with just the phrase.")
CMD_Q = ("You are Kermit, a puppet on stage. Each tracked person has a numbered tag. Someone said: \"{cmd}\". "
         "Which numbered person do they mean? Left and right are as seen in this image. If nobody matches, answer none. Answer with just the number.")

COLORS = {"red": [(0, 10), (170, 180)], "orange": [(10, 22)], "yellow": [(22, 35)], "green": [(35, 85)], "teal": [(85, 100)],
          "blue": [(100, 130)], "purple": [(130, 160)], "pink": [(160, 170)]}


def torso_color(img, box):
    import cv2
    x0, y0, x1, y1 = map(int, box); h, w = y1 - y0, x1 - x0
    a = np.asarray(img)[int(y0 + .3 * h):int(y0 + .6 * h), int(x0 + .25 * w):int(x1 - .25 * w)]
    if a.size == 0: return "dark"
    hsv = cv2.cvtColor(a[..., ::-1].copy(), cv2.COLOR_BGR2HSV).reshape(-1, 3)
    s, v = np.median(hsv[:, 1]), np.median(hsv[:, 2])
    if v < 60: return "black"
    if s < 50: return "white" if v > 170 else "grey"
    hue = np.median(hsv[hsv[:, 1] > 50, 0]) if (hsv[:, 1] > 50).any() else 0
    return next((n for n, rs in COLORS.items() if any(a <= hue < b for a, b in rs)), "colourful")


def spec_desc(img, p): return f"the person in the {torso_color(img, p['box'])} top"


def spec_cmd(img, people, cmd):
    c = cmd.lower(); W = img.width
    cand = people
    col = [n for n in list(COLORS) + ["black", "white", "grey", "gray"] if re.search(rf"\b{n}\b", c)]
    if col:
        m = [p for p in people if torso_color(img, p["box"]) in [x.replace("gray", "grey") for x in col]]
        cand = m or cand
    cx = lambda p: (p["box"][0] + p["box"][2]) / 2
    if "left" in c: return min(cand, key=cx)["id"]
    if "right" in c: return max(cand, key=cx)["id"]
    if col and len(cand) < len(people): return max(cand, key=lambda p: (p["box"][2] - p["box"][0]) * (p["box"][3] - p["box"][1]))["id"]
    return None


def crop_person(img, box, pad=0.15):
    x0, y0, x1, y1 = box; w, h = x1 - x0, y1 - y0
    return img.crop((max(0, x0 - pad * w), max(0, y0 - pad * h), min(img.width, x1 + pad * w), min(img.height, y1 + pad * h)))


def main():
    items = {i["key"]: i for i in json.load(open(os.path.join(HERE, "items.json")))}
    labels = {k: v for k, v in json.load(open(os.path.join(HERE, "labels.json"))).items() if not k.startswith("_")}
    H, C, D = [], [], []
    for key, lab in labels.items():
        it = items[key]; ppl = it["people"]; ids = [p["id"] for p in ppl]
        img = Image.open(os.path.join(HERE, "frames", f"{key}_3.jpg")).convert("RGB")
        mk = gemma.marked(img, ppl)
        good = set(lab["hands"]) | set(lab.get("ok", [])); neg = set(lab.get("neg", []))
        # (b) pick
        g, ms = gemma.ask_choice([mk], HAND_Q, ids, none=True)
        hu = sorted([p for p in ppl if p.get("hand_up")], key=lambda p: -(p["box"][2] - p["box"][0]) * (p["box"][3] - p["box"][1]))
        s = hu[0]["id"] if hu else None
        ok = lambda x: (x in good and x not in neg) if lab["hands"] else (x is None or x in good)
        H.append({"key": key, "gt": lab["hands"], "gemma": g, "spec": s, "gemma_ok": ok(g), "spec_ok": ok(s), "ms": round(ms)})
        print("hand", key, lab["hands"], "gemma", g, "spec", s, f"{ms:.0f}ms", flush=True)
        # describe: whole frame with the person tagged vs. a crop of just that person
        for pid, ref in lab["desc"].items():
            p = next((q for q in ppl if q["id"] == int(pid)), None)
            if p is None: continue
            t0 = time.perf_counter()
            d1, ms1, _ = gemma.ask([gemma.marked(img, [p])], DESC_Q.format(where=f"box tagged {pid}"), max_tokens=24)
            d2, ms2, _ = gemma.ask([crop_person(img, p["box"])], DESC_Q.format(where="picture"), max_tokens=24)
            D.append({"key": key, "id": int(pid), "ref": ref, "gemma_tag": d1.strip(), "gemma_crop": d2.strip(), "spec": spec_desc(img, p),
                      "ms_tag": round(ms1), "ms_crop": round(ms2)})
            print("desc", key, pid, "|", ref, "|", d1.strip(), "|", d2.strip(), "|", spec_desc(img, p), flush=True)
        # (c) commands
        for cmd, tgt in lab["cmds"]:
            g, ms = gemma.ask_choice([mk], CMD_Q.format(cmd=cmd), ids, none=True)
            s = spec_cmd(img, ppl, cmd)
            yaw = next((p["yaw_deg"] for p in ppl if p["id"] == g), None)
            C.append({"key": key, "cmd": cmd, "gt": tgt, "gemma": g, "spec": s, "gemma_ok": g == tgt, "spec_ok": s == tgt, "ms": round(ms), "bearing_deg": yaw})
            print("cmd", key, cmd, "gt", tgt, "gemma", g, "spec", s, f"{ms:.0f}ms", flush=True)
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    summ = lambda R: {"n": len(R), "gemma_acc": round(100 * np.mean([r["gemma_ok"] for r in R]), 1), "spec_acc": round(100 * np.mean([r["spec_ok"] for r in R]), 1),
                      "ms_median": float(np.median([r["ms"] for r in R]))}
    json.dump({"summary": summ(H), "items": H}, open(os.path.join(HERE, "results", "hands.json"), "w"), indent=0)
    json.dump({"summary": summ(C), "items": C}, open(os.path.join(HERE, "results", "cmds.json"), "w"), indent=0)
    json.dump({"items": D, "ms_tag_median": float(np.median([d["ms_tag"] for d in D])), "ms_crop_median": float(np.median([d["ms_crop"] for d in D]))},
              open(os.path.join(HERE, "results", "describe.json"), "w"), indent=0)
    print("SUMMARY hands", summ(H)); print("SUMMARY cmds", summ(C))


if __name__ == "__main__":
    main()
