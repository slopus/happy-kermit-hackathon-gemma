"""Prompt/input variants for (b) and (c) on the labelled frames (see eval_people.py for the baseline).

  python eval_variants.py TAG      → results/variants_<TAG>.json   (TAG names the server config, e.g. tok_default / tok_560)
"""
import json, os, sys, time
import numpy as np
from PIL import Image
import gemma
from eval_people import HAND_Q, CMD_Q, crop_person

HERE = os.path.dirname(os.path.abspath(__file__))
G_HAND = ("Each tile shows one person from the audience, numbered. Pick ONE person who is raising a hand right now "
          "(hand or arm held up above the shoulder, as if asking to speak). If none of them is, answer none. Answer with just the number.")
G_CMD = ("You are Kermit, a puppet on stage. The first image is the whole view with numbered people; the second shows each "
         "numbered person close up. Someone said: \"{cmd}\". Which numbered person do they mean? Left and right are as seen "
         "in the first image. If nobody matches, answer none. Answer with just the number.")
YN = ("Is this person raising a hand or arm up above their shoulder right now, as if asking a question? "
      "Holding a phone up counts. Answer yes or no.")
DESC = ("Describe this person in at most 8 words so a host can call on them in a crowd, like 'the woman in the red jacket' "
        "or 'the man with glasses and a grey beard'. Use clothing colour, hair, glasses, or something they hold; only mention "
        "an object if it is clearly in their hand. Never mention age, race, body, or looks. Answer with just the phrase.")


def main(tag):
    items = {i["key"]: i for i in json.load(open(os.path.join(HERE, "items.json")))}
    labels = {k: v for k, v in json.load(open(os.path.join(HERE, "labels.json"))).items() if not k.startswith("_")}
    R = {"hand_marked": [], "hand_gallery": [], "hand_yesno": [], "cmd_marked": [], "cmd_both": [], "desc": []}
    for key, lab in labels.items():
        it = items[key]; ppl = it["people"]; ids = [p["id"] for p in ppl]
        img = Image.open(os.path.join(HERE, "frames", f"{key}_3.jpg")).convert("RGB")
        mk = gemma.marked(img, ppl); gal = gemma.gallery(img, ppl)
        good = set(lab["hands"]) | set(lab.get("ok", [])); neg = set(lab.get("neg", []))
        ok = lambda x: (x in good and x not in neg) if lab["hands"] else (x is None or x in good)
        for name, imgs, q in (("hand_marked", [mk], HAND_Q), ("hand_gallery", [gal], G_HAND)):
            g, ms = gemma.ask_choice(imgs, q, ids, none=True)
            R[name].append({"key": key, "pred": g, "ok": ok(g), "ms": round(ms)})
        # per-person yes/no: every tracked person, one call each → set of raised hands
        ys, t0 = [], time.perf_counter()
        for p in ppl:
            a, _, _ = gemma.ask([crop_person(img, p["box"])], YN, max_tokens=2, grammar='root ::= "yes" | "no"')
            if a.strip() == "yes": ys.append(p["id"])
        ms = (time.perf_counter() - t0) * 1000
        pick = max((p for p in ppl if p["id"] in ys), key=lambda p: (p["box"][2] - p["box"][0]) * (p["box"][3] - p["box"][1]), default=None)
        g = pick["id"] if pick else None
        tp = len(set(ys) & good); fp = len(set(ys) - good); fn = len(set(lab["hands"]) - set(ys))
        R["hand_yesno"].append({"key": key, "pred": g, "yes": ys, "ok": ok(g), "ms": round(ms), "tp": tp, "fp": fp, "fn": fn})
        for cmd, tgt in lab["cmds"]:
            for name, imgs, q in (("cmd_marked", [mk], CMD_Q.format(cmd=cmd)), ("cmd_both", [mk, gal], G_CMD.format(cmd=cmd))):
                g, ms = gemma.ask_choice(imgs, q, ids, none=True)
                R[name].append({"key": key, "cmd": cmd, "gt": tgt, "pred": g, "ok": g == tgt, "ms": round(ms)})
        for pid, ref in lab["desc"].items():
            p = next((q for q in ppl if q["id"] == int(pid)), None)
            if p is None: continue
            d, ms, _ = gemma.ask([crop_person(img, p["box"], pad=0.05)], DESC, max_tokens=24)
            R["desc"].append({"key": key, "id": int(pid), "ref": ref, "gemma": d.strip(), "ms": round(ms)})
        print(key, {k: v[-1].get("ok") for k, v in R.items() if v and v[-1]["key"] == key}, flush=True)
    S = {k: {"n": len(v), "acc": round(100 * np.mean([r["ok"] for r in v]), 1), "ms_median": float(np.median([r["ms"] for r in v]))}
         for k, v in R.items() if k != "desc"}
    yn = R["hand_yesno"]; tp, fp, fn = (sum(r[k] for r in yn) for k in ("tp", "fp", "fn"))
    S["hand_yesno"].update({"precision": round(100 * tp / max(1, tp + fp), 1), "recall": round(100 * tp / max(1, tp + fn), 1)})
    S["desc"] = {"n": len(R["desc"]), "ms_median": float(np.median([r["ms"] for r in R["desc"]]))}
    json.dump({"summary": S, **R}, open(os.path.join(HERE, "results", f"variants_{tag}.json"), "w"), indent=0)
    print("SUMMARY", tag, json.dumps(S))


if __name__ == "__main__":
    main(sys.argv[1])
