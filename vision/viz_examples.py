"""Annotated example frames for the report: the tracked people (thin), Gemma's pick (thick) and its line."""
import json, os
from PIL import Image, ImageDraw
import gemma

HERE = os.path.dirname(os.path.abspath(__file__))
V = json.load(open(os.path.join(HERE, "results", "variants_tok_default.json")))
items = {i["key"]: i for i in json.load(open(os.path.join(HERE, "items.json")))}


def show(key, pick, lines, out):
    it = items[key]; img = Image.open(os.path.join(HERE, "frames", f"{key}_3.jpg")).convert("RGB")
    im = gemma.marked(img, it["people"], max_side=1280, thick=2); s = im.width / img.width; d = ImageDraw.Draw(im)
    p = next(p for p in it["people"] if p["id"] == pick)
    d.rectangle([v * s for v in p["box"]], outline=(255, 255, 0), width=8)
    F = gemma.font(26); y = im.height - 40 * len(lines) - 10
    d.rectangle([0, y - 8, im.width, im.height], fill=(0, 0, 0))
    for k, l in enumerate(lines): d.text((14, y + 40 * k), l, fill=(255, 255, 0) if k == 0 else (255, 255, 255), font=F)
    im.save(os.path.join(HERE, "results", out), quality=88)


def get(kind, key, **kw):
    return next(r for r in V[kind] if r["key"] == key and all(r.get(k) == v for k, v in kw.items()))


yn = get("hand_yesno", "conf_woman_hand_001.5"); ds = get("desc", "conf_woman_hand_001.5", id=2)
show("conf_woman_hand_001.5", 2, [f"hand_pick → id 2: \"{ds['gemma']}\"",
     f"Gemma yes/no per person: raised = {yn['yes']}  ({yn['ms']} ms for {len(items['conf_woman_hand_001.5']['people'])} people)",
     "people.py hand_up: none (wrist above the frame top)"], "ex_hand_pick.jpg")
c = get("cmd_both", "class_kids2_003.0", cmd="look at the girl with the striped backpack")
show("class_kids2_003.0", c["pred"], [f"\"Kermit, {c['cmd']}\" → id {c['pred']} ({c['ms']} ms)",
     f"bearing for MOTION: yaw {next(p['yaw_deg'] for p in items['class_kids2_003.0']['people'] if p['id'] == c['pred'])}° (people.py track)"], "ex_look_command.jpg")
ds = get("desc", "obama_town_020.0", id=6)
show("obama_town_020.0", 6, [f"describe id 6 → \"{ds['gemma']}\" ({ds['ms']} ms)", "people.py has_mic (YOLO-World) for the same frame: " +
     str(next(p['has_mic'] for p in items['obama_town_020.0']['people'] if p['id'] == 6))], "ex_describe_mic.jpg")
print("ok")
