# GoblinScript

Drop in a video, get a funscript. **Made by real goblins.**

A goblin watches the video and scrawls a matching script: strokes where the
action is, timed to the motion on screen. The result lands next to the video
as `video.funscript`, ready for any player that reads funscripts. Treat it as
a solid first draft -- the goblin aims to nail the rhythm and the timing, and
you can fine-tune anything else in your favorite editor.

## Using it

**Easiest way:** double-click `goblinscript.exe`. A picker opens right in the
terminal -- browse to your videos with the arrow keys, select them with
Space or Enter, toggle overwrite with the highlighted key, and
press S to set the goblins to work. **R** re-reads the folder (handy when a
download has just landed) and **E** opens it in Explorer, with whatever the
cursor is on already selected.

**Space on a folder takes the whole folder** -- every video sitting in it, so a
fresh drop of clips goes into the batch without stepping inside. Just that one
folder: videos in the folders below it are left alone, and you pick those by
going in. The box on the folder's row says how many of its videos are in.

These keys change the mood rather than the work, and they keep working **while
a draft is running** -- a video takes a while, and the volume should not have to
wait for it:

* **T** cycles the colour scheme -- goblin green, an amber monitor, CGA
  cyan/magenta, or plain white phosphor. Every screen follows, review page
  included.
* **M** cycles the sound: work music, blips only, or silence. The goblins
  put a record on when they start; it plays for as long as the app is open,
  picking a track at random and then working through the rest.
* **V** sets how loud that music is -- quiet, normal or loud.
* **N** skips to the next track.

All of it is remembered for next time, so turning the music down (or off) is a
one-time decision. From a command line the same settings are `--theme`,
`--music` / `--mute` and `--volume`.

**When something goes wrong, the goblins say so and keep saying so.** A video
that fails does not take the rest of the batch with it -- the goblins move on
to the next one, and when the picker comes back it opens on a report: which
video, which step it got to, and why it stopped, in as many lines as it takes.
Press any key to put the listing back, **X** to read it again, and **E** there
to open the log folder.

That log is the copy that outlives the session: every failure is appended to
`goblinscript.log` next to the exe, in English, with a date on each line and
the version and machine at the top of each run. It is the right thing to send
when you ask what happened. (It is capped -- once it passes half a megabyte the
old one becomes `goblinscript.log.old` and a fresh one starts.)

**Ctrl-C stops.** It ends the draft and closes the app -- no work carries on in
the background afterwards. Whatever the goblins had finished is kept, so running
that video again picks up where they left off instead of starting over. If they
are in the middle of a long step they will take a few seconds to down tools;
press it a second time to leave immediately.

Or from a command line:

```
goblinscript video.mp4                  the script appears next to the video
goblinscript a.mp4 b.mp4 c.mp4          several videos, one after another
goblinscript D:\clips                   every video in that folder
goblinscript video.mp4 --out D:\scripts put the scripts somewhere else
goblinscript video.mp4 --theme amber    pick a colour scheme up front
goblinscript video.mp4 --music          work music on; --mute for silence
goblinscript video.mp4 --quiet          skip the startup report
```

Every run opens with a startup report -- the goblins counting their memory,
finding ffmpeg, and naming the hardware they'll be thinking with. It's worth a
glance if a draft ever misbehaves: it says which graphics backend they landed
on and whether any interrupted work is waiting to resume.

**Your existing scripts are safe.** If a video already has a `.funscript`
next to it, the goblins skip that video and say so -- they never overwrite a
script unless you explicitly pass `--force`.

Regular (flat) videos need no flags at all.

## Speaking your language

The goblins speak whatever your machine speaks, as long as a phrasebook for it
is sitting in the `languages/` folder next to the exe. **English and Chinese
ship with them.** **G** in the picker moves between the ones you have and the
choice is remembered; from a command line it is `--lang zh-CN`, and that
includes `--help`.

**Yours not there? Add it -- no build, no toolchain.** Copy
`languages/en-US.json`, name it after your language (`languages/de-DE.json`,
say), and translate the right-hand side of every line. Leave the left-hand
names alone, and keep anything in `{braces}` exactly as it is -- those are
where the numbers and filenames get dropped in. Save it in whatever encoding
your editor offers, start the app, and your language is on the G key. A line you
have not translated yet simply comes out in English, so a half-finished file is
perfectly usable -- and a file that cannot be read at all says so by name when
the app starts, rather than quietly not being there.

## VR videos

A goblin cannot read a VR video the way it comes: the picture is a warped
360-degree bubble, and a goblin has only ever watched ordinary flat footage.
So it asks you where to look, once.

Hand it a VR video and it says so, then opens a page in your browser showing
one flat view cut out of the bubble. **Drag the picture to aim it** -- the same
way you would turn your head in a headset -- until the action sits in frame.
That is the whole job. A few things help you get it right:

* The **filmstrip** underneath is that same aim, ten moments spread across the
  video. Every cell should still hold the action -- that is how you know one
  aim covers the whole thing rather than just the bit you were looking at.
* **B** plays about two seconds of real motion at the cursor, looping, and you
  can keep aiming while it runs. A still frame cannot tell you whether the
  stroke wanders out of view; this can.
* The **strip below the picture** is the whole video from above, with a box
  showing where you are pointing -- handy for finding the action in the first
  place.
* Long video? **I** and **O** mark a start and an end, and only that stretch is
  drafted. The funscript you get is still timed to the full-length video, so it
  drops straight into your player.

Press **Done** and the goblins get to work. Several VR videos in one batch are
all aimed in the same sitting -- arrow keys switch between them -- so the
drafting itself runs unattended afterwards.

Your aim is remembered, so running the same video again never asks twice.
If a video is flagged as VR by mistake, press **K** ("not VR") and it is
drafted as ordinary footage. `--no-vr` skips the whole step, and `--vr` forces
the page open on a video the goblins thought was flat.

## What you need

* **Windows 10 (64-bit) or Windows 11.** GoblinScript is Windows-only.
* A **DirectX 12 graphics card** -- NVIDIA, AMD, and Intel all work. Plan on
  **8 GB of video memory (VRAM)**; that is where the goblins do their
  thinking. A card is genuinely required: the goblins think in a form only a
  GPU can hold, so there is no CPU-only mode in this build. (`--cpu` exists
  for the developers -- it needs a bundle built with `--cpu-attn` -- and will
  tell you so if you try it.)
* **8 GB of system memory**, or more. The startup report flags anything less;
  long videos are where it bites.
* [ffmpeg](https://ffmpeg.org/) on your PATH -- any recent build works.
  Easiest install: open a terminal and run

  ```
  winget install --id Gyan.FFmpeg
  ```

  then open a NEW terminal (the PATH change doesn't reach already-open ones).
* Keep the `.dll` files from the zip in the same folder as `goblinscript.exe`.

**How long does it take?** Very roughly half the video's running time on a
fast GPU -- a one-hour video takes 25-30 minutes. Running the same video
again is much faster (see below), and several videos queue up fine overnight.

## The cache folder

While the goblins are working, they keep their scratch files in a `cache`
folder next to the exe (a converted working copy of the video and some
intermediate data). The folder cleans itself up: once a video's script is
written, its working files are removed.

The one exception is a draft that didn't finish -- a crash, or you closed the
window. Its working files stay, so running the same video again picks up
where the goblins left off instead of starting over.

* `--cache D:\somewhere` puts the working files on another drive.
* `--keep-cache` keeps them even after success; re-drafting that video later
  (say, with a newer version) then skips most of the work.
* Deleting the cache folder by hand is always safe; the next run just redoes
  the work.

The failure log (`goblinscript.log`) sits beside that folder and is just as
safe to delete -- it is written for you to read, never read back by the app.

## Good to know

* Everything runs on your machine. No uploads, no internet connection needed.
* The video file itself is never modified.
* `goblinscript --help` lists every option.
* Funscripts drive adult toys, and adult video is what the goblins were taught
  on. Nothing here is for anyone under 18.

## The source

GoblinScript is MIT licensed and the source is all here -- the picker, the
video handling, the decode that turns what the goblins see into strokes, and
the tests that hold it together.

**The goblins themselves are not.** What the app knows about watching a video
is a ~250 MB bundle of neural-network graphs, trained separately and baked
into the released `goblinscript.exe`. It is a build product, not source, and
it is not in this repository. So:

```
cargo build --release     # builds, and needs a bundle to draft anything
cargo test                # green on a bare checkout
```

A build from source has no goblins in it until you point it at a bundle with
`--bundle <DIR>`. The one inside a release zip is the one the release drafts
with.

The soundtrack is not in here either -- `music/` is empty in a checkout and
the app runs silent, which is a setting it already has. Any General MIDI
`.mid` files dropped in that folder become the playlist.

`DEVELOPMENT.md` is the map: what each stage does, why it is shaped that way,
and what is measured to keep it honest.
