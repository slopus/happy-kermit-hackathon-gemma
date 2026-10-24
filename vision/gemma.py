"""Thin Gemma 4 (llama-server + mmproj) client for the vision layer, plus set-of-marks rendering.

Gemma never has to output pixel boxes here: people.py's tracks are drawn on the frame as numbered tags and
Gemma answers with a tag number. That sidesteps the llama.cpp box offset (see calib.py) and keeps ids = track ids.
"""
import base64, io, json, os, time, urllib.request
from PIL import Image, ImageDraw, ImageFont

URL = os.environ.get("KERMIT_VISION_URL", "http://127.0.0.1:8079")
_FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
if not os.path.exists(_FONT):                      # macOS
    _FONT = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
PALETTE = [(255, 59, 48), (52, 199, 89), (0, 122, 255), (255, 204, 0), (175, 82, 222), (255, 149, 0),
           (90, 200, 250), (255, 45, 85), (162, 132, 94), (0, 199, 190)]


def font(px):
    try: return ImageFont.truetype(_FONT, px)
    except OSError: return ImageFont.load_default()


def jpeg_b64(im, q=88):
    buf = io.BytesIO(); im.convert("RGB").save(buf, "JPEG", quality=q); return base64.b64encode(buf.getvalue()).decode()


def ask(images, prompt, max_tokens=80, schema=None, system=None, temperature=0.0, url=URL, timeout=60, grammar=None):
    """images: list of PIL images (sent in order, before the text). → (text, ms, usage)"""
    content = [{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + jpeg_b64(im)}} for im in images]
    content.append({"type": "text", "text": prompt})
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": content}]
    body = {"messages": msgs, "max_tokens": max_tokens, "temperature": temperature, "cache_prompt": True,
            "chat_template_kwargs": {"enable_thinking": False}}
    if grammar is not None:
        body["grammar"] = grammar
    if schema is not None:
        body["response_format"] = {"type": "json_schema", "json_schema": {"name": "a", "schema": schema}}
    t = time.perf_counter()
    req = urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    r = json.load(urllib.request.urlopen(req, timeout=timeout))
    ms = (time.perf_counter() - t) * 1000
    return r["choices"][0]["message"]["content"], ms, r.get("usage", {})


def ask_json(images, prompt, schema, **kw):
    txt, ms, u = ask(images, prompt, schema=schema, **kw)
    try: return json.loads(txt), ms
    except json.JSONDecodeError: return None, ms


def ask_choice(images, prompt, ids, none=False, **kw):
    """Constrain the answer to one of the track ids (GBNF) → (id or None, ms)."""
    alts = [f'"{i}"' for i in ids] + (['"none"'] if none else [])
    txt, ms, _ = ask(images, prompt, max_tokens=4, grammar="root ::= " + " | ".join(alts), **kw)
    txt = txt.strip()
    return (int(txt) if txt.isdigit() else None), ms


def marked(img, people, max_side=1024, thick=None, label_pos="top"):
    """Draw each person's box + a big numbered tag (the track id). img: PIL RGB; people: [{"id", "box":[x0,y0,x1,y1]}]."""
    im = img.convert("RGB").copy()
    s = min(1.0, max_side / max(im.size))
    if s < 1: im = im.resize((round(im.width * s), round(im.height * s)), Image.BICUBIC)
    d = ImageDraw.Draw(im)
    th = thick or max(2, round(im.width / 400))
    fsz = max(14, round(im.width / 40)); F = font(fsz)
    for p in people:
        x0, y0, x1, y1 = [v * s for v in p["box"]]
        col = PALETTE[p["id"] % len(PALETTE)]
        d.rectangle([x0, y0, x1, y1], outline=col, width=th)
        tag = str(p["id"]); tw = d.textlength(tag, font=F)
        tx, ty = x0, (y0 - fsz - 6 if label_pos == "top" and y0 > fsz + 6 else y0)
        d.rectangle([tx, ty, tx + tw + 8, ty + fsz + 6], fill=col)
        d.text((tx + 4, ty + 1), tag, fill=(0, 0, 0) if sum(col) > 450 else (255, 255, 255), font=F)
    return im


def crop_grid(frames, people, cols=None, cell=160, key="face_box", pad=0.35):
    """One row per person: that person's head crop in each frame (left→right = oldest→newest), id tag on the left.
    frames: list of PIL frames; people: list of per-frame dicts {id: box} or a single {id: box} for all frames."""
    ids = sorted({i for fr in people for i in fr}) if isinstance(people, list) else sorted(people)
    per = people if isinstance(people, list) else [people] * len(frames)
    n = len(frames); tagw = 56
    out = Image.new("RGB", (tagw + n * cell, len(ids) * cell), (30, 30, 30)); d = ImageDraw.Draw(out); F = font(30)
    for r, pid in enumerate(ids):
        col = PALETTE[pid % len(PALETTE)]
        d.rectangle([0, r * cell, tagw - 4, (r + 1) * cell - 4], fill=col)
        d.text((8, r * cell + cell // 2 - 18), str(pid), fill=(0, 0, 0), font=F)
        for c, (fr, boxes) in enumerate(zip(frames, per)):
            b = boxes.get(pid)
            if b is None: continue
            x0, y0, x1, y1 = b; w, h = x1 - x0, y1 - y0; m = max(w, h) * (1 + pad)
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            cr = fr.crop((int(cx - m / 2), int(cy - m / 2), int(cx + m / 2), int(cy + m / 2))).resize((cell - 4, cell - 4), Image.BICUBIC)
            out.paste(cr, (tagw + c * cell, r * cell))
    return out


def gallery(img, people, h=256, cols=4, pad=0.12):
    """Each tracked person cropped (box + pad), scaled to height h, id tag in the corner, tiled in a grid."""
    tiles = []
    for p in people:
        x0, y0, x1, y1 = p["box"]; w, hh = x1 - x0, y1 - y0
        c = img.crop((max(0, x0 - pad * w), max(0, y0 - pad * hh), min(img.width, x1 + pad * w), min(img.height, y1 + pad * hh)))
        c = c.resize((max(1, round(c.width * h / c.height)), h), Image.BICUBIC)
        if c.width > 1.4 * h: c = c.resize((round(1.4 * h), h), Image.BICUBIC)
        tiles.append((p["id"], c))
    if not tiles: return None
    tw = max(t.width for _, t in tiles); rows = (len(tiles) + cols - 1) // cols
    out = Image.new("RGB", (min(cols, len(tiles)) * (tw + 6), rows * (h + 6)), (20, 20, 20)); d = ImageDraw.Draw(out); F = font(34)
    for k, (pid, t) in enumerate(tiles):
        x, y = (k % cols) * (tw + 6), (k // cols) * (h + 6)
        out.paste(t, (x + (tw - t.width) // 2, y)); col = PALETTE[pid % len(PALETTE)]
        tag = str(pid); d.rectangle([x, y, x + d.textlength(tag, font=F) + 12, y + 42], fill=col)
        d.text((x + 6, y + 2), tag, fill=(0, 0, 0) if sum(col) > 450 else (255, 255, 255), font=F)
    return out
