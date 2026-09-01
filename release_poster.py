"""The release comparison poster: three shipped versions side by side, for USERS.

Every version drafted the same 11 clips on the same machine, each at its OWN
shipped defaults -- the newer auto-crop and exposure stages do not exist in
v0.1.0, and disabling them would measure a build nobody ships. Every draft is
scored against the clip's human funscript.

Two sections, because three versions need a structure a verdict tag cannot
carry: how close the script is to the person's, and how much movement it puts
there that the person did not. ONE mechanism sets them together: a version
that creates more
movement follows the action better AND makes more movement the person never
made. The three stand against the person's own count -- Version 0.1.0 far
short of it, the two newer versions past it -- which is a reading no single
"better" or "worse" label can hold, and the whole reason the sections exist.

Each row is a quantity a reader already has a feel for -- a percentage of
correct timing, a count per minute, a number of minutes -- so nothing on the
page needs the project to explain it. Rows that need explaining stay off the
page; the one score that survives carries its own scale in the note under it.

Colour is VERSION IDENTITY here and nothing else: one hue per version, in a
fixed order, on every row. A bar cannot be recoloured to mean "this one is
worse" -- with three of them the reader would lose track of which version they
are looking at, which is the one thing the page cannot afford. The reading
lives in the white human-reference line, in the lead tag, and in the note.

Rows measured against the person's own count carry a white line at that count:
a bar short of the line is read as short, a bar past it as too many -- without
the mark, a doubling toward the target would sell as a straight win and an
overshoot would too. Those rows carry NO lead tag, because "nearest the line"
would award an overshoot, which is exactly the confusion the line prevents.

The copy follows ASD-STE100 Simplified Technical English, and it names no
internal artifact: no clip IDs, no host hardware, no metric names. The goblin
is `src/mascot.rs`'s own, poses and faces from its own list, so the poster is
drawn by the same hand as the app.

Numbers come from probes/draft_vs_script_pos.py, probes/draft_vs_script_ms.py
and artifact_speed.py. They are written down here
rather than recomputed, so the poster renders in seconds and the values on it
are exactly the ones quoted in the run.

The first four rows are read where the person WROTE a script. A video the
person left unscripted for three minutes has no answer there to compare a
draft against, so those spans count neither as a miss nor as an invention, and
one convention covers all three versions: the page compares versions and never
conventions. The last two rows read written speed alone and take the person's
line from the script's own actions, so they never needed the distinction.

    python goblinscript/release_poster.py --out infer_out/release_v040
"""
import argparse
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                      # noqa: E402
from matplotlib.patches import FancyBboxPatch        # noqa: E402

# --- the phosphor palette, straight off src/theme.rs (xterm-256 -> hex) ------
BG = "#000000"
INK = "#87d787"        # text 114
BRIGHT = "#87ff00"     # logo 118
DIM = "#5f875f"        # muted 65
WARN = "#ffd700"       # warn 220
WHITE = "#ffffff"      # accent 231

# One colour per version, oldest first, and this order never changes. The
# newest release takes the logo green because the page is published for it;
# the middle version takes the amber step of theme.rs's own ramp, which sits
# between the two greens in the ramp's order exactly as that version sits
# between them in time. The two greens differ enough in lightness to separate
# on black, and every bar is named in the gutter anyway, so identity never
# rests on colour.
VERSIONS = [("Version 0.1.0", DIM),
            ("Version 0.3.0", WARN),
            ("Version 0.4.0", BRIGHT)]
NEW_VER = "Version 0.4.0"

MONO = ["DejaVu Sans Mono", "Consolas", "monospace"]

# --- the goblin, from src/mascot.rs -----------------------------------------
# His ears are up in every pose; the face carries the mood. Only faces from
# mascot.rs's own list appear here.
REST = [r" /\,/\ ", r"\(o.o)/", r" |___| ", r"  / \  "]
WAVE = [r" /\,/\ ", r"\(^.^)/", r" |___| ", r"  / \  "]
SQUINT = [r" /\,/\ ", r"\(>.<)/", r" |___| ", r"  / \  "]


def march(pose, n, gap=3):
    """A file of goblins, side by side -- the parade mascot.rs marches across
    the header about once every five minutes."""
    return "\n".join((" " * gap).join([row] * n) for row in pose)


# --- the measurements -------------------------------------------------------
# (label, person, [v0.1.0, v0.3.0, v0.4.0], fmt, lead, note)
# `person` is the human script's own value, drawn as a line, or None.
# `lead` names the version that is best on the row, or None where the human
# line already carries the reading. A value of None inside the triple is a
# measurement that was not made.
SECTIONS = [
    ("HOW CLOSE TO THE PERSON'S SCRIPT", BRIGHT, WAVE,
     "A longer bar is better on all three rows.", [
        ("How well the script follows the action",
         None, [0.793, 0.823, 0.833], "{:.3f}", 2,
         "1.000 is a perfect match with the person's script. Version 0.4.0 "
         "is the nearest of the three."),
        ("How well the script follows the action, on difficult videos",
         None, [0.716, 0.787, 0.793], "{:.3f}", 2,
         "The two most difficult videos of the eleven. Every version is "
         "further behind here, and Version 0.1.0 the furthest. Version "
         "0.4.0 is ahead here too."),
        ("Changes of direction at the correct time",
         None, [83.8, 88.1, 87.9], "{:.1f}%", 1,
         "Correct to 1/15 second. The two newer versions are close here."),
    ]),
    ("MOVEMENT THAT IS NOT IN THE PERSON'S SCRIPT", WARN, SQUINT,
     "Longer is worse. A white line marks the person's own count.", [
        ("Changes of direction that the person did not make",
         None, [28.3, 34.3, 34.9], "{:.1f}%", 0,
         "A version that creates more movements finds more of the person's "
         "movements, and also makes more that the person did not."),
        ("Fast movements in each minute",
         22.5, [4.5, 33.8, 34.3], "{:.1f}", None,
         "Version 0.1.0 creates far fewer than the person. The two newer "
         "versions create more than the person, and they are close to each "
         "other."),
        ("Sudden fast movements where the video is slow, in each minute",
         0.5, [7.3, 14.3, 17.2], "{:.1f}", None,
         "The person creates almost none. Every version creates too many, "
         "and Version 0.4.0 the most: it creates the most movement overall, "
         "and these two rows move together."),
    ]),
]

GUTTER, TRACK = 0.235, 0.50
# The note sits at the bar track's own left edge, so the page width left to it
# is fixed and a note WRAPS rather than running off the paper. Monospace at
# this size fits this many characters in that span; the row grows downward by
# the lines it took, so a longer note costs height and never a lost sentence.
NOTE_COLS, NOTE_LEAD = 72, 0.26

# What the newest version changed. It adds no command and no button: the
# goblins learned the same job better, so the whole of it is on the bars
# above and the page says so in one line rather than dressing tuning up as
# a feature list.
WHATS_NEW = (
    "The goblins learned from more videos, and they follow the action more "
    "closely than before.\n"
    "Movement at a shot cut is smooth now. Before, a cut could ask for a "
    "large move in one frame.\n"
    "A new seed option makes a small variation of the same script."
)


def bar_row(ax, y, label, person, values, fmt, lead, note):
    """One comparison, three bars. The person's own count, where there is one,
    is a line the bars are read against: the reader then sees an overshoot as
    an overshoot. Returns the height the row took, in page units."""
    have = [v for v in values if v is not None]
    span = max(have + ([] if person is None else [person]))
    span = span * 1.06 or 1.0

    ax.text(0, y + 0.36, label, color=INK, fontsize=11, family=MONO,
            va="bottom", ha="left")
    for i, (v, (tag, col)) in enumerate(zip(values, VERSIONS)):
        by = y - 0.14 - i * 0.30
        ax.text(GUTTER - 0.008, by, tag, color=col, fontsize=8.8, family=MONO,
                va="center", ha="right")
        if v is None:
            ax.text(GUTTER + 0.014, by, "not measured", color=DIM,
                    fontsize=9, family=MONO, va="center", ha="left",
                    style="italic")
            continue
        w = max(v / span * TRACK, 0.004)
        ax.add_patch(FancyBboxPatch(
            (GUTTER, by - 0.100), w, 0.200,
            boxstyle="round,pad=0,rounding_size=0.012",
            facecolor=col, edgecolor="none", mutation_aspect=0.4))
        ax.text(GUTTER + w + 0.014, by, fmt.format(v), color=col, fontsize=10,
                family=MONO, va="center", ha="left", weight="bold")
    # The person's count is a TARGET, so it is a line the bars are read
    # against -- not a fourth bar competing with them. A bar that overshoots
    # the mark then crosses the line, which is the whole point of the row.
    # White, because the mark belongs to no version and every other colour
    # here is spoken for; at the weight of a line it states itself without
    # taking the row, which a white BAR did.
    if person is not None:
        px = GUTTER + person / span * TRACK
        ax.plot([px, px], [y + 0.04, y - 0.90], color=WHITE, lw=1.5, zorder=5)
        # a mark near the left edge takes its label to the right, so the
        # label never runs into the row tags in the gutter
        near_edge = px < GUTTER + 0.09
        ax.text(px if not near_edge else px + 0.012, y + 0.10,
                "the person: " + fmt.format(person),
                color=WHITE, fontsize=8.8, family=MONO, va="bottom",
                ha="center" if not near_edge else "left")
    # The lead tag names a version, so it is drawn in that version's own
    # colour -- the same hue the reader has been tracking down the page. It
    # sits on the ROW TITLE's line: the question is then on the left and its
    # answer on the right, and the tag can never run into the value label of
    # a bar that fills the track.
    if lead is not None:
        tag, col = VERSIONS[lead]
        ax.text(1.0, y + 0.36, tag + " leads", color=col, fontsize=9.5,
                family=MONO, weight="bold", va="bottom", ha="right")
    ny = y - 0.14 - len(values) * 0.30 - 0.14
    lines = textwrap.wrap(note, NOTE_COLS)
    for line in lines:
        ax.text(GUTTER, ny, line, color=DIM, fontsize=8.5, family=MONO,
                va="center", ha="left", style="italic")
        ny -= NOTE_LEAD
    return (2.35 if person is None else 2.60) + NOTE_LEAD * (len(lines) - 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="infer_out/release_v040")
    ap.add_argument("--name", default="goblinscript_versions.png")
    args = ap.parse_args()

    # The page is laid out downward from TOP in its own units and the canvas is
    # cut to fit afterwards, at a fixed UNIT of inches per unit. A row added or
    # removed then changes the poster's HEIGHT and never its density -- with a
    # figure sized up front, the same edit leaves a slab of dead black at the
    # bottom or crushes the footer off the page.
    TOP, UNIT = 40.6, 0.40
    fig = plt.figure(figsize=(8.8, 16.4), dpi=150, facecolor=BG)
    ax = fig.add_axes([0.055, 0.02, 0.90, 0.96])
    ax.set_facecolor(BG)
    ax.set_xlim(0, 1)
    ax.axis("off")

    y = TOP
    ax.text(0, y, "GOBLINSCRIPT", color=BRIGHT, fontsize=27, family=MONO,
            weight="bold", va="top", ha="left")
    ax.text(0, y - 1.05, "Version 0.4.0, Version 0.3.0 and Version 0.1.0",
            color=WHITE, fontsize=13, family=MONO, va="top", ha="left")
    ax.text(0, y - 2.00,
            "The goblins created scripts for 11 videos that they did not see\n"
            "before. Then we compared the scripts with scripts that a person\n"
            "created for the same videos.",
            color=INK, fontsize=10, family=MONO, va="top", ha="left",
            linespacing=1.5)
    ax.text(1.0, y - 0.05, "\n".join(WAVE), color=BRIGHT, fontsize=15,
            family=MONO, va="top", ha="right", linespacing=1.05)
    ax.plot([0, 1], [y - 3.85, y - 3.85], color=DIM, lw=1.1)

    y = 36.15
    for title, colour, pose, sub, rows in SECTIONS:
        ax.text(0, y, title, color=colour, fontsize=13, family=MONO,
                weight="bold", va="center", ha="left")
        ax.text(0, y - 0.40, sub, color=DIM, fontsize=9, family=MONO,
                va="center", ha="left")
        ax.text(1.0, y + 0.42, "\n".join(pose), color=colour, fontsize=8.5,
                family=MONO, va="top", ha="right", linespacing=1.05)
        y -= 1.75
        for row in rows:
            y -= bar_row(ax, y, *row)
        y -= 0.35
        if title != SECTIONS[-1][0]:
            ax.plot([0, 1], [y + 0.30, y + 0.30], color=DIM, lw=1.1)
            y -= 0.55

    y -= 0.10
    ax.plot([0, 1], [y + 0.30, y + 0.30], color=DIM, lw=1.1)
    y -= 0.50
    ax.text(0, y, "NEW IN " + NEW_VER.upper(), color=BRIGHT, fontsize=13,
            family=MONO, weight="bold", va="center", ha="left")
    ax.text(1.0, y + 0.42, "\n".join(REST), color=BRIGHT, fontsize=8.5,
            family=MONO, va="top", ha="right", linespacing=1.05)
    y -= 0.95
    ax.text(0.012, y, WHATS_NEW, color=INK, fontsize=9.6, family=MONO,
            va="top", ha="left", linespacing=1.6)
    y -= 1.35

    y -= 0.30
    ax.plot([0, 1], [y, y], color=DIM, lw=1.1)
    y -= 0.40
    ax.text(0.5, y, march(REST, 7), color=DIM, fontsize=7.6, family=MONO,
            va="top", ha="center", linespacing=1.05)
    y -= 1.95
    ax.text(0, y, "11 videos. 2 hours and 19 minutes of video. "
                  "Each version has its usual settings.",
            color=DIM, fontsize=8.4, family=MONO, va="bottom", ha="left")

    y -= 0.30
    ax.set_ylim(y, TOP + 0.4)
    fig.set_size_inches(8.8, (TOP + 0.4 - y) * UNIT)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    p = out / args.name
    fig.savefig(p, facecolor=BG, dpi=150)
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
