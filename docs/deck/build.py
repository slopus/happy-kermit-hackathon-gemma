"""Build the Kermit deck: one HTML page per slide, rendered to ../slides/*.png by render.cjs.

Look follows happy.engineering: paper #f5f0e7, ink #183f38, Space Grotesk titles, DM Sans body,
DM Mono eyebrows, and hand-drawn Shantell Sans diagrams like /img/tailcat.svg.
"""
import math
import random
from pathlib import Path

OUT = Path(__file__).with_name("out")
INK, MUTED, CREAM, GREEN, PAPER, SAND, KERMIT = "#183f38", "#6b7f78", "#f6eadf", "#e3efe9", "#f5f0e7", "#efe9dc", "#cfe3c0"
HAND = "'Shantell Sans', 'Comic Sans MS', 'Chalkboard SE', cursive"
rng = random.Random(7)


# ---------------------------------------------------------------- hand-drawn primitives
def j(amount):
    return rng.uniform(-amount, amount)


def stroke(points, color=INK, width=1.6, dash=None, wobble=0.6, step=34):
    """Polyline with a little perpendicular wobble, like the docs diagrams."""
    out = []
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        length = math.hypot(x1 - x0, y1 - y0) or 1
        nx, ny = -(y1 - y0) / length, (x1 - x0) / length
        n = max(1, round(length / step))
        for i in range(n + (1 if (x1, y1) == points[-1] else 0)):
            t = i / n
            k = 0 if i in (0, n) else j(wobble)
            out.append((x0 + (x1 - x0) * t + nx * k, y0 + (y1 - y0) * t + ny * k))
    d = "M" + " L".join(f"{x:.1f} {y:.1f}" for x, y in out)
    extra = f' stroke-dasharray="{dash}"' if dash else ""
    return (f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{width}" '
            f'stroke-linecap="round" stroke-linejoin="round"{extra}/>')


def box(x, y, w, h, fill=CREAM, width=1.6, color=INK):
    fill_path = (f'<path d="M{x + j(1):.0f} {y + j(1):.0f} L{x + w + j(1):.0f} {y + j(1):.0f} '
                 f'L{x + w + j(1):.0f} {y + h + j(1):.0f} L{x + j(1):.0f} {y + h + j(1):.0f} Z" fill="{fill}"/>')
    o = lambda: rng.uniform(.6, 2.2)
    sides = [
        [(x - o(), y + j(1)), (x + w + o(), y + j(1))],
        [(x + w + j(1), y - o()), (x + w + j(1), y + h + o())],
        [(x + w + o(), y + h + j(1)), (x - o(), y + h + j(1))],
        [(x + j(1), y + h + o()), (x + j(1), y - o())],
    ]
    return fill_path + "".join(stroke(s, color, width) for s in sides)


def arrow(points, color=MUTED, width=1.5, dash=None):
    (x0, y0), (x1, y1) = points[-2], points[-1]
    a = math.atan2(y1 - y0, x1 - x0)
    head = []
    for side in (1, -1):
        bx = x1 - 11 * math.cos(a) + side * 5.5 * math.sin(a)
        by = y1 - 11 * math.sin(a) - side * 5.5 * math.cos(a)
        head.append(stroke([(bx, by), ((bx + x1) / 2 + j(.5), (by + y1) / 2 + j(.5)), (x1, y1)], color, width))
    return stroke(points, color, width, dash) + "".join(head)


def circle(cx, cy, r, fill="none", color=INK, width=1.8, wobble=.025):
    pts = []
    start = rng.uniform(0, math.tau)
    for i in range(27):
        t = start + i / 24 * math.tau
        rr = r * (1 + j(wobble))
        pts.append((cx + rr * math.cos(t), cy + rr * math.sin(t)))
    d = "M" + " L".join(f"{x:.1f} {y:.1f}" for x, y in pts)
    fill_el = f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{fill}"/>' if fill != "none" else ""
    return fill_el + (f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{width}" '
                      f'stroke-linecap="round" stroke-linejoin="round"/>')


def text(x, y, value, size=18, weight=400, color=INK, anchor="start"):
    return (f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}" fill="{color}" '
            f'text-anchor="{anchor}">{value}</text>')


def svg(w, h, body, label, scale=1):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w * scale:.0f}" height="{h * scale:.0f}" '
            f'font-family="{HAND}" role="img" aria-label="{label}">{body}</svg>')


# ---------------------------------------------------------------- doodles
def frog(scale=1.0):
    """Kermit in the same ink: head, two big eyes, a smile, the pointy collar."""
    p = []
    collar = []
    for i in range(10):  # the pointy collar fans out under the chin
        a0, a1 = math.radians(28 + i * 12.4), math.radians(28 + (i + .5) * 12.4)
        collar += [(200 + 120 * math.cos(a0), 215 + 70 * math.sin(a0)),
                   (200 + 190 * math.cos(a1) + j(3), 215 + 135 * math.sin(a1) + j(3))]
    collar.append((200 + 120 * math.cos(math.radians(152)), 215 + 70 * math.sin(math.radians(152))))
    p.append('<path d="M' + ' L'.join(f'{x:.1f} {y:.1f}' for x, y in collar) + f' Z" fill="{KERMIT}"/>')
    p.append(stroke(collar, INK, 2.0, wobble=.3, step=99))
    head = [(200 + 160 * math.cos(t) * (1 + j(.012)), 200 + 92 * math.sin(t) * (1 + j(.02)))
            for t in (i / 30 * math.tau for i in range(32))]
    p.append(f'<ellipse cx="200" cy="200" rx="160" ry="92" fill="{KERMIT}"/>')
    p.append(stroke(head, INK, 2.2, wobble=.4, step=60))
    for cx in (140, 262):
        p.append(circle(cx, 110, 50, "#fffdf8", INK, 2.2))
        p.append(f'<circle cx="{cx + 6}" cy="126" r="9" fill="{INK}"/>')
    smile = [(84 + 232 * t, 222 + 70 * math.sin(math.pi * t) + j(1)) for t in (i / 14 for i in range(15))]
    p.append(stroke(smile, INK, 2.4, wobble=.3, step=99))
    return f'<g transform="scale({scale})">' + "".join(p) + "</g>"


# ---------------------------------------------------------------- the system diagram
def system_diagram():
    p = []
    title = lambda x, y, s: text(x, y, s, 24, 700, INK, "middle")
    sub = lambda x, y, s: text(x, y, s, 17, 400, MUTED, "middle")

    # This laptop: like the docs' "Team server" frame, with a dark title band.
    p.append(box(200, 0, 980, 590, SAND, 1.6))
    p.append(f'<path d="M201 1 L1179 1 L1179 56 L201 56 Z" fill="{INK}"/>')
    p.append(text(226, 37, "This MacBook: everything on-device, no cloud", 22, 700, "#fffdf8"))

    # Inputs.
    p.append(box(0, 130, 150, 70)); p.append(title(75, 173, "Mic"))
    p.append(box(0, 330, 150, 100)); p.append(title(75, 375, "Camera")); p.append(sub(75, 402, "Insta360"))

    # Voice loop.
    p.append(box(230, 110, 210, 110)); p.append(title(335, 156, "Parakeet")); p.append(sub(335, 184, "speech → text"))
    p.append(box(520, 100, 300, 130, GREEN, 1.9))
    p.append(title(670, 146, "Gemma 4 E4B")); p.append(sub(670, 176, "Kermit LoRA"))
    p.append(sub(670, 202, "writes reply + gesture tags"))
    p.append(box(900, 110, 200, 110)); p.append(title(1000, 156, "Kokoro")); p.append(sub(1000, 184, "voice + word times"))

    # Seeing.
    p.append(box(230, 330, 340, 100)); p.append(title(400, 372, "Perception")); p.append(sub(400, 400, "people tracks · who's speaking"))
    p.append(box(650, 330, 250, 100)); p.append(title(775, 372, "Following")); p.append(sub(775, 400, "face the speaker"))

    # Watching it all.
    p.append(box(230, 482, 870, 76, PAPER, 1.3, MUTED))
    p.append(text(665, 528, '<tspan font-weight="700" fill="#183f38">Dashboard + controller</tspan>'
                  '<tspan fill="#6b7f78">   ON / OFF · live status · these slides</tspan>', 20, 400, INK, "middle"))

    # Outputs.
    p.append(box(1260, 30, 240, 76)); p.append(title(1380, 66, "Speaker")); p.append(sub(1380, 92, "Kermit's voice"))
    p.append(box(1260, 128, 240, 76)); p.append(title(1380, 164, "Inspire hand")); p.append(sub(1380, 190, "mouth, on the audio"))
    p.append(box(1260, 240, 240, 190)); p.append(title(1380, 322, "xArm")); p.append(sub(1380, 350, "head + gestures,"))
    p.append(sub(1380, 374, "on the exact word"))

    # Arrows.
    p.append(arrow([(152, 165), (226, 165)]))
    p.append(arrow([(442, 165), (516, 165)]))
    p.append(arrow([(822, 165), (896, 165)]))
    p.append(arrow([(1102, 142), (1140, 142), (1140, 68), (1256, 68)]))
    p.append(arrow([(1102, 166), (1256, 166)]))
    p.append(arrow([(1102, 190), (1140, 190), (1140, 280), (1256, 280)]))
    p.append(arrow([(152, 380), (226, 380)]))
    p.append(arrow([(572, 380), (646, 380)]))
    p.append(arrow([(902, 380), (1256, 380)], dash="6 6"))
    p.append(arrow([(545, 326), (545, 236)]))
    p.append(text(560, 278, "turned away · new speaker", 16, 400, MUTED))
    return svg(1500, 594, "".join(p), "Kermit system: mic to Parakeet to Gemma 4 E4B to Kokoro, out to speaker, Inspire hand mouth and xArm gestures; camera to perception to following; dashboard and controller watch it all", scale=1408 / 1500)


# ---------------------------------------------------------------- slides
CSS = """
@font-face { font-family: 'Shantell Sans'; src: url(../fonts/ShantellSans.ttf); }
@font-face { font-family: 'DM Sans'; src: url(../fonts/DMSans.ttf); }
@font-face { font-family: 'DM Mono'; src: url(../fonts/DMMono-Medium.ttf); font-weight: 500; }
@font-face { font-family: 'Space Grotesk'; src: url(../fonts/SpaceGrotesk.ttf); }
* { box-sizing: border-box; margin: 0; }
html, body { width: 1600px; height: 900px; overflow: hidden; background: #f5f0e7; color: #183f38;
  font-family: 'DM Sans', sans-serif; -webkit-font-smoothing: antialiased; }
.slide { position: relative; width: 1600px; height: 900px; padding: 78px 96px; }
.eyebrow { font-family: 'DM Mono', monospace; font-weight: 500; font-size: 20px; letter-spacing: .1em;
  text-transform: uppercase; color: #49665d; display: flex; align-items: center; gap: 14px; }
.eyebrow::before { content: ''; width: 34px; height: 2px; background: #183f38; }
h1 { font-family: 'Space Grotesk', sans-serif; font-weight: 600; letter-spacing: -.03em; line-height: 1.02; }
.foot { position: absolute; left: 96px; right: 96px; bottom: 56px; display: flex; justify-content: space-between;
  font-family: 'DM Mono', monospace; font-weight: 500; font-size: 18px; letter-spacing: .06em; color: #6b7f78; }
.hl { background: #f3d98a; border-radius: .2em; padding: 0 .16em; }
"""


def page(body, extra=""):
    return f'<!doctype html><meta charset="utf-8"><style>{CSS}{extra}</style><div class="slide">{body}</div>'


SLIDES = {}

SLIDES["01-title"] = page(f"""
<div class="eyebrow">Gemma Hardware Hackathon · AGI House</div>
<h1 style="font-size:150px;margin:90px 0 34px">Happy<br>Kermit</h1>
<p style="font-size:40px;line-height:1.3;max-width:800px">What agents can build in a day,<br>
<span style="color:#6b7f78">and what still needs us.</span></p>
<div style="position:absolute;right:120px;top:210px">{svg(400, 360, frog(), "Kermit doodle")}</div>
<div class="foot"><span>Kirill + Steve · happy.engineering</span><span>Oct 24, 2026</span></div>
""")

BULLETS = """
.list { margin-top: 56px; display: grid; gap: 30px; max-width: 1300px; }
.row { display: grid; grid-template-columns: 220px 1fr; gap: 36px; border-top: 2px solid #183f38; padding-top: 20px; }
.k { font-family: 'Space Grotesk', sans-serif; font-weight: 600; font-size: 44px; letter-spacing: -.02em; }
.v { font-size: 30px; line-height: 1.38; color: #49665d; }
.quote { position: absolute; left: 96px; right: 96px; bottom: 70px; font-size: 26px; color: #6b7f78; }
"""


def rows(items):
    return '<div class="list">' + "".join(f'<div class="row"><div class="k">{k}</div><div class="v">{v}</div></div>'
                                          for k, v in items) + "</div>"


SLIDES["02-agents-built-it"] = page("""
<div class="eyebrow">01 · The agents built it</div>
<h1 style="font-size:84px;margin:36px 0 0">I barely looked at the code.</h1>
""" + rows([
    ("~1 hr", "of my time. Agents wrote the pipeline, fine-tuned the models, and ran the room simulations."),
    ("Models", "Parakeet hears, Gemma 4 E4B + a Kermit LoRA thinks, Kokoro talks, perception watches."),
    ("Me", "Pointed, reviewed, said no. Mostly said no."),
]) + """
<div class="quote">“I didn't write a single line of code.” Half of X, this month.</div>
""", BULLETS)

SLIDES["03-how-it-works"] = page(f"""
<div class="eyebrow">02 · How it works</div>
<h1 style="font-size:60px;margin:22px 0 0">Hear, think, speak, move. All on this laptop.</h1>
<div style="position:absolute;left:96px;top:236px">{system_diagram()}</div>
""", ".slide { padding-top: 64px; }")

SLIDES["04-real-world"] = page("""
<div class="eyebrow">03 · Then came the real world</div>
<h1 style="font-size:84px;margin:36px 0 0">The sim was perfect. <span class="hl">Reality wasn't.</span></h1>
""" + rows([
    ("Sim", "Camera tracking looked great. Hand motion looked great."),
    ("Hand 1", "A stuck finger, so the puppet couldn't be worn."),
    ("Hand 2", "Wouldn't turn on. We swapped the electronics and got it working."),
]) + """
<div class="quote">Agents can write the code. They can't unstick a finger.</div>
""", BULLETS)

FACTS = [
    ("~0.4 s", "to Kermit's first sentence. Measured on this Mac."),
    ("5.3 GB", "the whole brain: Gemma 4 E4B + Kermit LoRA, 4-bit."),
    ("0", "cloud calls. Works offline."),
    ("0 fans", "The iOS simulator works this Mac harder."),
]
SLIDES["05-impressed"] = page("""
<div class="eyebrow">04 · Talking to it</div>
<h1 style="font-size:84px;margin:36px 0 0">Underwhelmed and impressed,<br>at the same time.</h1>
<div class="facts">""" + "".join(f"""
  <div class="fact"><div class="big">{b}</div><p>{d}</p></div>""" for b, d in FACTS) + """
</div>""", """
.facts { position: absolute; left: 96px; right: 96px; bottom: 96px; display: grid; grid-template-columns: repeat(4, 1fr); gap: 40px; }
.fact { border-top: 2px solid #183f38; padding-top: 22px; }
.big { font-family: 'Space Grotesk', sans-serif; font-weight: 600; font-size: 72px; letter-spacing: -.03em; line-height: 1; white-space: nowrap; }
.fact p { margin-top: 18px; font-size: 26px; line-height: 1.42; color: #49665d; }
""")

SLIDES["06-added-value"] = page("""
<div class="eyebrow">05 · So where is the added value?</div>
<h1 style="font-size:84px;margin:36px 0 0">Local models are great <span class="hl">dumb pipes.</span></h1>
""" + rows([
    ("Local", "Great dumb pipes: hear, route, classify, talk. Private, fast, quiet. Not your chores: nobody wants half-folded laundry."),
    ("Frontier", "Writes almost all the code now. It vibe-coded this whole stack in about an hour of my time."),
    ("Us", "Deciding what to build, fixing the real world, and checking that it's actually done."),
]) + """
<div class="quote">“Software engineers have never worked harder than they do now… agents write ~100% of the code.” François Chollet, Oct 17</div>
""", BULLETS)

SLIDES["07-close"] = page(f"""
<div class="eyebrow">Thank you</div>
<h1 style="font-size:112px;margin:100px 0 34px;max-width:1000px">No offence, Kermit…</h1>
<p style="font-size:44px;line-height:1.3;max-width:900px">but you don't want to be in <span class="hl">Kermit's position.</span></p>
<p style="font-size:32px;color:#6b7f78;margin-top:40px">Questions? Ask Kermit.</p>
<div style="position:absolute;right:120px;top:250px">{svg(400, 360, frog(.92), "Kermit doodle")}</div>
<div class="foot"><span>happy.engineering</span><span>Built with Gemma · on-device</span></div>
""")


def main():
    OUT.mkdir(exist_ok=True)
    for old in OUT.glob("*.html"):
        old.unlink()
    for name, html in SLIDES.items():
        (OUT / f"{name}.html").write_text(html)
    (OUT / "system-diagram.svg").write_text(system_diagram())
    print("\n".join(sorted(SLIDES)))


if __name__ == "__main__":
    main()
