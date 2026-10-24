"""Gemma pointing on frames people.py can't track (wide crowd shots): box_2d for a described person / a raised hand."""
import json, os, re, sys
from PIL import Image, ImageDraw
import gemma, vision
HERE = os.path.dirname(os.path.abspath(__file__))
HAND = ('Find one person in the audience who is raising a hand. Respond with ONLY JSON like {"box_2d":[ymin,xmin,ymax,xmax]} '
        'normalized to 0-1000 around that person (head and raised arm), or {"box_2d":null} if nobody is.')
Q = [("speech_aud_005.0", "look at the woman in the red top in the front row"), ("speech_aud_040.0", HAND),
     ("biz_hands1_003.0", HAND), ("concert_stock_030.0", HAND), ("obama_town_040.0", "look at the woman in the red dress"),
     ("aud_clap_020.0", HAND)]
outs = []
for key, q in Q:
    img = Image.open(os.path.join(HERE, "frames", f"{key}_3.jpg")).convert("RGB")
    prompt = q if q is HAND else vision.LOCATE.format(cmd=q)
    txt, ms, _ = gemma.ask([img], prompt, max_tokens=40)
    try: b = json.loads(re.sub(r"^```(json)?|```$", "", txt.strip()).strip()).get("box_2d")
    except Exception: b = None
    d = ImageDraw.Draw(img); W, H = img.size
    if b: d.rectangle([b[1] * W / 1000, b[0] * H / 1000, b[3] * W / 1000, b[2] * H / 1000], outline=(255, 255, 0), width=5)
    d.rectangle([0, 0, W, 34], fill=(0, 0, 0)); d.text((8, 4), f"{'raised hand?' if q is HAND else q} → {b} ({ms:.0f} ms)", fill=(255, 255, 0), font=gemma.font(22))
    outs.append(img.resize((640, round(640 * H / W)))); print(key, b, round(ms))
G = Image.new("RGB", (1280, 3 * max(o.height for o in outs)))
for k, o in enumerate(outs): G.paste(o, ((k % 2) * 640, (k // 2) * max(x.height for x in outs)))
G.save(os.path.join(HERE, "results", "ex_locate_crowd.jpg"), quality=85)
