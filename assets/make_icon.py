"""Draw the GoblinScript app icon -> assets/goblin.ico (multi-size Windows ICO).

The ASCII mascot (`theme.rs`), drawn in pixels -- the same goblin, stood there:

      /\\,/\\        narrow ears, close together and upright
     \\(o.o)/       round eyes, small nose, arms up and out
      |___|        body
       / \\         legs

Same figure as the review page's logo and favicon (`review.html`), same
palette. Reproducible -- re-run to regenerate the .ico.

    venv\\Scripts\\python.exe goblinscript/assets/make_icon.py
"""
from pathlib import Path
from PIL import Image, ImageDraw

S = 256  # master canvas; the .ico is downscaled from here
FACE   = (139, 195, 74, 255)   # #8bc34a
EAR    = (124, 179, 66, 255)   # #7cb342
NOSE   = (104, 159, 56, 255)   # #689f38
DARK   = (27, 58, 18, 255)     # #1b3a12
CREAM  = (241, 248, 233, 255)  # #f1f8e9


def draw(scale):
    """The figure at 256 px, then resampled.

    Drawn back to front: ears, then the limbs and body, then the head OVER
    them, then the face. The head has to land last of the big shapes so the
    arms and torso tuck behind it the way they do in the ASCII, where the face
    row sits on top of everything.

    The head is deliberately most of the figure. At 16 px -- which is where a
    taskbar and a title bar actually show this -- the body and legs resolve to a
    couple of green pixels, and the head is the whole of what anyone reads.
    """
    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    # ears: two narrow uprights sitting close together, the ASCII's `/\,/\`
    # rather than the wide swept pair this icon used to carry
    d.polygon([(88, 80), (104, 12), (124, 72)], fill=EAR)
    d.polygon([(168, 80), (152, 12), (132, 72)], fill=EAR)
    # arms, up and out: the `\` and `/` either side of the face
    d.line([(80, 184), (36, 144)], fill=EAR, width=16, joint="curve")
    d.line([(176, 184), (220, 144)], fill=EAR, width=16, joint="curve")
    # body, and legs under it
    d.line([(112, 220), (104, 248)], fill=EAR, width=16)
    d.line([(144, 220), (152, 248)], fill=EAR, width=16)
    d.rounded_rectangle([96, 160, 160, 220], radius=16, fill=FACE)
    # head, over the lot
    d.ellipse([60, 52, 196, 172], fill=FACE)
    # eyes: cream sclera + dark pupils
    d.ellipse([81, 89, 119, 127], fill=CREAM)
    d.ellipse([137, 89, 175, 127], fill=CREAM)
    d.ellipse([92, 100, 112, 120], fill=DARK)
    d.ellipse([144, 100, 164, 120], fill=DARK)
    # nose
    d.ellipse([120, 124, 136, 140], fill=NOSE)
    # a plain smile -- the fangs went with the scowl
    d.arc([100, 126, 156, 166], start=20, end=160, fill=DARK, width=8)
    return img.resize((scale, scale), Image.LANCZOS)


def main():
    master = draw(S)
    out = Path(__file__).with_name("goblin.ico")
    sizes = [16, 24, 32, 48, 64, 128, 256]
    master.save(out, format="ICO", sizes=[(s, s) for s in sizes])
    print(f"wrote {out}  ({', '.join(str(s) for s in sizes)} px)")


if __name__ == "__main__":
    main()
