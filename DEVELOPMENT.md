# GoblinScript internals

The user-facing side lives in `README.md`; everything below is for working on
GoblinScript itself. (To users, the drafts are made by real goblins; in here,
the goblin is a frozen V-JEPA 2.1 encoder and a trained head.)

A frozen **V-JEPA 2.1** encoder and the trained head, wrapped as a single Rust
CLI. The model is baked into the binary; the only external dependency is
**ffmpeg** (and **ffprobe**) on PATH -- any build carrying the `v360` filter
(≥ 4.2; the `winget install --id Gyan.FFmpeg` release is what's tested).
**Windows 10 (64-bit, 1903+ -- the DirectML floor) or 11**, and 8 GB of system
RAM (`bios.rs` warns below that; distinct from the ~4.5 GB VRAM working set).

Runs on any DirectX 12 GPU (NVIDIA / AMD / Intel) via DirectML. On an RTX 4090
the encoder forward is **63 ms** (`--bench`), and the encoding runs at
**~7.4x realtime** -- 4 forwards per 64-frame window on the 30 Hz grid, so the
forward alone sets an 8.5x ceiling. The stage that encoding is now half of also
carries the head, and the head has no GPU to run on: a 1024-row chunk is
**~460 ms** of CPU (the forward covers chunk + 2x ctx = 3072 rows; the envelope
decode's 3072 stepped calls are 38 ms of that). It runs on its own thread, so
that cost lands beside the encoder rather than in front of it -- see the head's
own section. Quote the stage, not the forward: the same
63 ms read as 17.7x on the 15 Hz grid, where a window was 32 frames and two
forwards, and that number outlived the grid it was measured on. The bundle is
fp16, the encoder's production dtype. A `--features cuda` build still exists and
still works, but it is no longer a fast path worth chasing: CUDA measures ~56 ms
against DirectML's 60, so the vendor-neutral backend is now at parity and the
default build is the one to ship.

That parity is recent and it came from the shape the bundle's encoder carries
attention in. Two things: attention is ONE fused `MultiHeadAttention` node
rather than a MatMul -> Softmax -> MatMul, and that node is fed a **packed**
`(B,S,N,3,H)` q/k/v tensor rather than three separate ones.
The second half is where DirectML's win is: the separate layout reaches a
generic path that still materializes the score matrix (144 ms), the packed one
reaches DirectML's native fused attention operator (59 ms). Attention went from
181 ms of a 241 ms forward to ~39 ms of a 60 ms one.

**There is no CPU fallback in a shipped bundle.** ONNX Runtime has no CPU kernel
for packed attention, so the encoder graph cannot run there at all, and `--cpu`
refuses up front (`manifest.attn`) rather than dying inside the encode stage.
That costs a real debugging tool -- the CPU provider is fp32 with no vendor EP in
the path, which is the reference a suspect draft gets bisected against when a
GPU backend MISCOMPUTES rather than fails (the fp16 overflow `encode.rs` guards
against). The replacement is a different bundle: a separate-qkv export runs
everywhere at 2.4x the DirectML cost, and its CPU forward is 4.5 s -- 74x the
shipped GPU path (peak working set 1.7 GB, down from 10 GB: ORT's CPU attention
kernel does not materialize either). `--bundle <DIR>` loads one without a
rebuild, so the reference is a bundle swap.

**VRAM did not move**: 4.5 GB on DirectML, measured before and after. The 8 GB
the README asks for is still the number.

## Pipeline

Four stages. Only the middle two involve a neural net; everything else is
plain Rust.

Two front ends, one pipeline: flags on the command line, or -- with no
arguments, which is what double-clicking the exe produces -- the `tui.rs`
picker (ratatui). The picker is deliberately only a LAUNCHER: it collects
videos + the force toggle, closes its alternate screen, and the run
itself uses the ordinary styled console stream (indicatif bars + console
colors), so the per-video history stays scrollable and identical either way.
A full-screen TUI for the run itself was rejected for exactly that reason:
alternate-screen apps lose the scrollback record of an overnight batch and
break piping output to a file. In interactive mode the window pauses on exit
so a double-clicked run (or its error) can actually be read.

**The listing is one `read_dir` and nothing else.** A folder of thousands is a
folder the picker still has to feel like a text field in, so everything that
would cost per-video I/O is either folded into that one pass or moved off the
draw thread: whether a video has a script beside it comes from the NAME SET the
same pass collected (not a `stat` apiece), what WROTE that script is a file read
and lands from a background pool, and the ffprobe durations are a want-list the
draw rewrites to the visible rows plus a lookahead -- a probe is a process
launch, and a folder's worth of them for rows nobody scrolls to is minutes of
disk the app is competing with. The filter is then a pure predicate over the
model already in memory, and the draw assembles only the rows it will paint,
which is why `App` owns its own scroll offset. At 4000 videos that is 12 ms to
list and ~0.6 ms per filter keystroke, against 124 ms per keystroke when
`refresh()` re-walked the folder and re-opened every script. The drive list
reads `GetLogicalDrives()` rather than `Path::exists()` per letter, which
TOUCHES each drive -- an empty optical bay spins up and a dead network mapping
waits out its timeout, on the one screen the picker opens to.

A picker-mode batch keeps the picker's four-row header pinned over that stream
(`chrome.rs`), which is a **scroll region** (`DECSTBM`) and not an alternate
screen: the log below prints and scrolls inside its own margins, the header
rows sit above them, and everything the console stream already did it still
does. A named-video run keeps the plain console -- that terminal belongs to
whoever typed into it -- and so does any redirected run, where there is no top
of the screen to pin anything to.

| Stage | What happens | Net |
|---|---|---|
| transcode | 480p / 30 fps / crf 23 | -- |
| boundaries | TransNetV2 per-frame transition probability -> shot cuts | `transnet.onnx` |
| encode | 384x384 2-frame tubelets -> PCA latents -> int8, then the head on every 1024 rows: mask net -> attention pool -> dilated TCN -> the five heads (pos, env, ext, plat, rev) | `vjepa_pca.onnx`, `mask.onnx`, `head.onnx`, `env_step.onnx` |
| styling | reversal times, level, amplitude, band snapping -> actions | -- |

**There is no model stage.** The head used to be one, reading the finished
cache back; it now runs INSIDE encode, on rows the encoder wrote seconds
earlier. `heads::Heads` takes rows through `push` and forwards each whole 1024-
row chunk on the spot -- the boundaries are still counted from row 0 and the
short tail's lookback is still decided at the end, so the chunks a streamed run
forwards are the chunks a whole-cache run would have and the tracks are
IDENTICAL (a pre-change build and this one write the same funscript byte for
byte from the same cache; that is the check to re-run on anything touching
`Heads`). What it buys is that the tracks exist while the long stage is still
running -- which is what puts the strokes on the viewport instead of a bar and
an ETA. A run that already HAS its latents streams them back off disk through
the same `Heads` (`heads::stream_cache`), so there is one path, not two.

**The head forwards on its own thread**, and that is what makes folding it in
free. Inline it was ~460 ms of CPU every 1024 rows in which nothing fed the
encoder, and a live draft showed it: the graphics card at 1-3% for the whole
hitch, one per chunk, ~16% of the stage's wall clock at the rate that draft was
running. `Heads` is now a handle over a worker that owns the two sessions;
`push` copies rows into a queue bounded in ROWS (so a cache replay pushing whole
chunks does not queue more memory than the encoder pushing windows) and returns.
One worker, FIFO, `Finish` last -- so the chunking, the ctx padding and the
envelope's history are what a single thread produced, and the byte-identity
check above is the one that proves it. Measured on a 3.5-minute clip (6 chunks):
the encode stage 37 s -> 34 s, and the GPU trace goes from ~6 idle gaps of
0.55-1.1 s inside the stage to a single unbroken 36 s of work. The one thing the
queue does change is WHEN a failure surfaces: a chunk that fails is reported at
the next `push` or at `finish`.

In a batch these do not run strictly in sequence. `normalize` is libx264 on the
CPU and the two stages after it are the GPU's, so `prefetch.rs` starts the
NEXT video's transcode the moment this one's finishes -- it runs in the
current video's GPU-bound shadow and is usually done before that video is.
One video ahead, never more, and only for videos that will actually be drafted
(normalizing one that is about to be skipped for having a script already is the
single way a head start can cost more than it saves). Measured on a 3-video
batch: 47 s -> 34 s, with byte-identical drafts. `--no-prefetch` is the
sequential baseline that number comes from.

Inside a video the head is the overlap that pays (above). The frame read is not:
`encode.rs` reads frames inline, between one `sess.run` and the next, and moving
that read onto its own thread ahead of the encoder is **measured a wash** --
see Open.

**The transcode is not an optimization.** Every clip the model has ever seen
reached it through that same re-encode. A pristine 4K source fed straight to
the encoder is off-distribution in a way nobody has measured, so GoblinScript
normalizes first and reads only from the normalized copy.

**High-fps sources draft on the normal path**: the fps filter lands every
source on the 30 fps deploy grid (a 60 fps source contributes every other
frame). Real-speed motion at that grid is the only distribution the model
has ever seen, and the only one it scripts well.

### VR (`vr.rs`, `vr.html`) -- an optional stage 0

An equirectangular eye is off-distribution for the frozen encoder, so a VR
source has to become flat before it becomes latents. That needs one human
decision -- WHERE to point -- and `vr.rs` serves a loopback browser page to
collect it, on the `review.rs` pattern.

Three things carry the design:

* **The aim folds into the transcode.** `Config::filter_prefix` emits
  `crop=<eye>,v360=<aim>` and `ffmpeg::transcode` puts it in FRONT of the
  existing `scale,fps` chain. One pass, no intermediate render: no flat copy
  is ever materialized, because nothing here needs one to exist afterwards.
  A static aim means v360 builds its LUT once, so the pass costs
  about what a flat transcode costs. A range trim becomes an input seek.
* **The preview IS the render.** `src/projector.js` -- baked into the exe with
  `include_str!` -- reproduces v360's mapping per pixel, so the
  page projects server-supplied eye frames itself and re-aiming is a redraw,
  not a round trip. **V** flips to a frame ffmpeg's own v360 produced: the one
  check that catches the shader and the renderer disagreeing. There is exactly
  one copy of the mapping, so the preview and the pass cannot drift apart.
* **The whole batch is aimed before any of it drafts.** `prefetch.rs` starts
  the next video's transcode a whole draft early and cannot do that for a video
  whose aim is an open question -- so `vr_prep` asks every question up front
  and the batch then runs unattended.

The aim lands in `cache/vr/<source-fingerprint>.json` (source identity, so it
survives re-aiming), and it is folded
into the DRAFT cache key (`Cache::open`) -- re-aiming changes every pixel the
encoder sees, so it must not reuse the old aim's transcode.

**A trimmed range shifts the clock.** The draft runs on the clip's clock;
`restyle` adds `t0_ms` back at write time so the file on disk is always for the
full-length video, and the review page rebases the fetched script by the same
amount because it streams the trimmed copy.

**The viewport aspect is the anamorphic squash.** `encode.rs` opens the decoder
at `scale=enc_res:enc_res` -- a flat squeeze into a square -- so the aspect
chosen on the page IS the stretch the model sees. 16:9 is the shape the model
was trained at and the one every VR clip reached it in, so
`Config::aspect_warning` flags anything else; the preview looks perfectly aimed
either way, which is what makes it worth a warning.

Detection (`vr::detect`) is recall-biased and deliberately not the last word:
stereo/spherical side data decides when present, frame shape otherwise (a 2:1
frame at VR resolution is SBS, a square one is top-and-bottom). A false
positive costs one glance at a page with a "not VR" button (**K**); a miss
costs a full-length draft of nonsense. `--vr` forces the page, `--no-vr` skips
the stage, `--vr-only` runs it without loading the model (aiming needs no GPU).

### The crop (`autocrop.rs`) and its page (`cropedit.rs`, `cropedit.html`)

A sparse probe runs the bundle's own mask net over short native-rate windows,
mixes the attention heads by their concentration, and reduces what is left to
one rect per shot. The rect rides the ENCODE decode chain (`SegmentedDecoder`),
never a re-encode: a re-encode was measured to land each render path on a
different clock, and the synthesized frame grid cannot drift.

**A rect is fractions of the frame, not attention cells.** The map is sampled
on the encoder's 24x24 grid and read there, but `SUBCELL`-fold bilinear
refinement between cell centres puts each row's box at 1/96 of the frame, the
edge votes are continuous percentiles, and the size, the placement search's
candidates and the picture box (letterbox bars, read per pixel LINE) all follow.
A cell is 4.2% of the frame; that used to be the step of every edge and the rung
of a six-value zoom ladder. `autocrop.py` in the training tree is the twin, and
`grid_check.py` binds them BOTH ways -- the constants by name, and a Gaussian
fixture through the whole refinement (`CROP_FIXTURE` in this file's tests).

**The page is a transport first.** A rect is judged against the action moving
under it, so the bar is a real scrub -- press and drag anywhere along it, and
the picture, the playhead and the overlay follow the pointer; the shot bands
are paint under it (`pointer-events: none`), not click targets. Arrows step a
frame (Shift a second, Alt a shot), **L** loops the shot being judged, and
**G** is the goblins' own view: the rect blown up to fill the stage, which is
what the encoder is actually handed. `probes/crop_page_check.py` in the
training tree drives all of that in a real Chromium and reports where the drag
put the video -- the crate's own test drives the SERVER and cannot press
anything.

**The page opens by default** (`--no-crop-edit`, or **K** in the picker, to
skip it; `--crop-edit` demands it where it would be skipped). It is served on
the `review.rs` pattern -- loopback, embedded, streaming the normalized copy the
crop is applied to -- and it sits between the probe and the encode, where a
correction is free: nothing downstream has run yet. The person drags a rect;
what they draw replaces the vote for that shot, or for every shot.

A hand-drawn plan is written back to `autocrop.json` marked `manual`, and
`autocrop::read_cached` then keeps it through a new checkpoint, a retuned recipe
and a different exposure -- an aim made against the picture answers to none of
them. That is also why the page is skipped when the plan came off the cache: the
answer it would ask for is already on disk. The rects ride the latent cache key
(`Plan::key`), so re-drawing one re-encodes rather than reusing the old pixels,
and the review page draws the plan it was drafted with either way.

## Presentation

Four surfaces (intro demo, picker, processing console, review page) draw from
ONE palette in `theme.rs`. Terminal colours are xterm-256 **indices**, not
truecolour: `console` (the processing console) has no truecolour, and an index
renders identically through `console` and `ratatui`, so the picker and the
console cannot drift. `Palette` has four members -- `phosphor` (house green),
`amber`, `cga`, `mono` -- selected by `--theme`, cycled live with **T** in the
picker, and remembered in `settings.json`. The active one is an `AtomicU8`, so a
keypress changes every surface on its next frame with nothing threaded through
call stacks that exist for other reasons.

The intro reads its colours from the same table. Its sweep is expressed as
FRACTIONS of the palette's hue band rather than absolute degrees, so a palette
narrows the rainbow instead of losing it: phosphor sweeps all 360°, amber
shimmers across 45°, and mono (zero-width band, zero saturation) resolves to a
greyscale ramp. The mascot dances the chime out under the settled wordmark,
fading in once the letters have landed -- two entrances at once is one busy
screen -- and off the animation's own clock, so every replay is the same
performance. He is drawn only where he fits WHOLE; the intro is the one screen
with no obligation to say anything.

The review page ships the phosphor scheme as its stylesheet
fallback and overrides every custom property from `/state.theme`; its canvases
can't read CSS variables, so `C` in the page mirrors the same values as strings.

`bios.rs` is the startup report, dressed as a power-on self-test. Every line is
a **real** check -- ffmpeg's version, host memory and core count, resumable
drafts in the cache, the bundle's run name, the provider chain this build will
try. The framing is invention; the diagnostics are not, because this screen is
also the first thing a user pastes into a bug report.

Startup runs in **boot order -- POST, then the demo, then the picker**: the
machine wakes up before it shows off. In picker mode the report reveals line by
line (and counts the memory up) over ~5 s while the ONNX bundle loads on a
background thread; the host lines need nothing from the bundle, and the device
lines join the loader first, so a slow load reads as the report enumerating
devices rather than as a stall. The intro then plays with everything in hand.
Pacing is the file's six `*_MS`/`*_STEPS` constants and nothing else, with a
test pinning the total to 4-7 s. A named-video run prints the report instantly
(that user is driving a tool, not booting a machine) and `--quiet` skips it.
Column widths are shared with the processing log's stage lines, so a finished run
reads as one printout.

The goblin in that banner **moves through the whole report**: each pause is
spent repainting his four rows in place (cursor up, art columns only, cursor
back) rather than sleeping through it, so the stretch where the machine is
loading a bundle does not read as a machine that has stopped. Only the art
columns are written, so a repaint cannot disturb a character of the diagnostics.
He holds still in the two cases where the arithmetic that finds those rows stops
being sound: a report that has scrolled the banner off the top, and a console
narrower than the report's own columns, where a wrapped line is a row nothing
counted.

`mascot.rs` is the goblin himself -- every pose he can strike and the beat he
strikes them on -- shared by all four surfaces, because the picker's header, the
processing header, the startup banner and the intro all draw the same fellow
doing the same thing at the same moment. What he is doing is a pure function of
the beat clock and the audio state, so surfaces agree without any of them owning
animation state, and a screen that opens mid-dance opens mid-dance. He DANCES
while music plays (a move per phrase, a set off now and then, so he is neither
furniture nor a metronome) and idles when it does not (blink, crane, stretch,
doze). The boot table is the exception with its own short cycle: the report is
over in five seconds, and the picker's long idle cycle would let a whole boot
pass with nothing happening. Tests hold the art itself -- every pose fits the
block, stays ASCII, wears a face off the readable list, and keeps its ears up.

**The parade** is the rare one: about once every five minutes, a file of eight
small goblins marches across the header, out from behind the band and away
behind the wordmark, and the mascot stops to watch them go past. They are the
same fellow again with his torso row dropped -- the trade the band already makes
-- so their heads sit a row below his and their feet stand in the same rule,
which they interrupt rather than blank out (a space in a pose paints nothing).
One of them is a straggler: a beat out of step with the file, and now and then
over his own feet. It runs only where the corridor between the tagline and the
band is at least four blocks wide, which rules out an 80-column terminal
entirely, for the reason the drummer is dropped at the same sort of width -- no
band beats a broken one. A goblin is drawn WHOLE or not at all, so the file
arrives block by block at the corridor's edge, and the entering and leaving
marchers are drawn in the secondary ink to turn that pop into an arrival.
Rarity is the design and not a setting: this is the one thing the header does
that is worth looking up at, and it stops being that the moment it is expected.

`sound.rs` holds two independent halves. The **effects** are synthesized --
boot chime, finished-batch jingle, stage blip, failure buzz, skip grumble, UI
click -- so they need no assets and no decoder. The **music** is the `music/`
playlist: every `.mid` in that folder, enumerated by `build.rs` into an
`include_bytes!` table (adding a track is dropping a file in the folder), parsed
by `midly`, and played through the OS General MIDI synthesizer via `winmm`'s
`midiOutShortMsg`. The playlist starts on a random track and runs in order,
wrapping forever.

That path is the second design. The first rendered every score through this
file's own square-wave and LFSR-noise voices, which needed no synth and worked
everywhere -- and turned a full arrangement into an 8-bit cover. It was rejected
on hearing it. The consequence is that **music is Windows-only**; the effects
are not, and the port would be a soundfont synth behind the same `set_music`
door, with nothing outside that section changing.

The player owns its own thread. `clock` there counts time SPENT PLAYING rather
than wall time, which is what makes M pause and resume land the track exactly
where it stopped; pausing also calls `midiOutReset`, because held notes on a
hardware synth ring forever otherwise.

**Volume rides MIDI channel volume (CC 7)**, not note velocity and not
`midiOutSetVolume`. Velocity is fixed when a note starts, so a velocity-scaled
change would not be heard until the next note; `midiOutSetVolume` reaches into a
device shared with anything else using it.

`Mixer` multiplies THREE levels into each wire value: the balance the score
asked for (its own CC 7), the track's levelling gain, and the user's step. They
combine rather than override -- replacing the score's CC 7 with a master number
would flatten an arrangement's internal mix.

**They combine as AMPLITUDES, converted to a byte once at the end.** MIDI
channel volume is quadratic -- the spec defines it as `40 * log10(cc7/127)` dB,
so `amplitude = (cc7/127)^2`. Multiplying the byte therefore squares each factor
and stacks the squares: levelling a track by 0.48 and applying a 0.55 step took
a channel to **-27 dB** when about -17 dB was intended, and the music was
inaudible under a working app. `Volume::amp` is an amplitude ratio for this
reason, and `Mixer::cc7` does the single `sqrt` back to the wire. The result is
also CLAMPED, not masked: `& 0x7f` on an over-range value wraps, which would
send a channel that should be at full volume to near silence.
`no_track_is_mixed_into_inaudibility` pins every track's normal-step level into
an audible band, in dB, so this cannot come back.

**Tracks are levelled to each other.** MIDI carries no loudness, so `loudness()`
measures the score: the RMS of the energy sounding at each moment, each note
contributing its velocity scaled by its channel volume, with each note's counted
length capped (a synth's note decays whether or not the score released the key,
so a pedalled chord must not read as minutes of full-level sound). Measured raw,
the current set spans 86-230 -- about 8.5 dB, very audible when the playlist
advances. `track_gain()` divides each into `TARGET_LOUDNESS`, which is a FIXED
reference rather than the playlist's own average, so dropping in a new `.mid`
cannot re-level everything else. The clamp band is asymmetric because attenuation
is nearly free while a boost has to stay under CC 7's ceiling. The `#[ignore]`d
`report_loudness` test prints the table and is the tool for retuning the target
after adding music.

Three controls, all one key each in the picker: **M** cycles audio (music ->
blips -> silent -- "track but no blips" is a combination nobody has asked for
and a second key to explain), **V** cycles volume (quiet/normal/loud, the same
preset idiom as `--stillness`), **N** skips the track. V and N are hidden from
the status strip and ignored as keys while the music is off, because a key that
would do nothing is worse than no key at all. Music is ON by default, and starts
after the boot sequence -- the intro has its own chime, and two pieces of music
at once is neither of them. `fire()` declines when muted OR when stdout is not a
terminal, in one place, so an overnight batch cannot blip at an empty room.

### Failures (`errlog.rs`)

**A failure has to outlive the screen it was printed on.** The picker reopens
into the alternate screen the moment a batch ends, so a red line printed on the
way out is visible for one frame and then painted over -- which reads as the
app having thrown the work away for no reason. So a failure is delivered three
times over, and the console is the least of them:

* **The console** keeps its red line, for the run that was started from a
  terminal and still has one.
* **The picker's report** (`draw_errors` in `tui.rs`) opens BY ITSELF when the
  batch that just ran had failures, covering the listing: each video named, the
  stage it reached, the error and every cause under it. Any key that is not
  about reading it closes it -- **Q** included, which here means "done reading"
  and never "quit the session" -- **X** brings it back for as long as it is the
  latest news, and **E** opens the log's folder.
* **`goblinscript.log`**, beside the exe with `settings.json` and the cache.
  Appended to, one run header (version, OS, clock) plus one entry per failure,
  rotated at 512 KB into a single `.old` companion.

All three read the same `errlog::Failure` value -- what it was working on, the
stage from the live marker, and `anyhow`'s chain outermost-first -- so they
cannot say different things about one failure. The log's copy is **English
whatever the interface is speaking**: it is written to be pasted into a bug
report, and a stack of Chinese context lines is not a thing this repo can read
back. That is what `lang::en` exists for, and it is why `Live::setup`/`stage`/
`done` take a catalog KEY rather than a label -- the screen draws `lang::t(key)`
and the log remembers `lang::en(key)`, from one call, so the two can never name
different stages.

Writing is best-effort throughout, like `settings`: `record` returns the path it
wrote to and `None` when it could not (a read-only install folder), and the
report says which of those happened. A log that cannot be written never fails a
draft that otherwise worked.

**In picker mode a failure ends the BATCH, never the session.** The prep step
(link resolution + VR aims) runs as one fallible block for that reason: an
unreachable link is this batch's failure, reported like any other, and the
picker comes back. A named-video command line keeps the old contract -- that run
asked for those videos, so a failure there is the process's failure, and its
exit message names the log.

### Language (`lang.rs`, `languages/`)

**The words are not in the code.** `languages/<tag>.json` is one flat map of
key -> string; `t!("picker.act.start")` reads it, `t!("key", n = 3)` fills `{n}`
placeholders. A translator opens the file, replaces the right-hand sides, and
saves it under their own tag -- no build, no toolchain, and nothing to
understand about the structure. That is the whole design goal, and it is why the
catalogs ride BESIDE the exe (`cargo xtask dist` packs the folder) rather than
inside it.

English is also `include_str!`d, as the per-key fallback. A key a translation
has not reached, a file half-finished, a typo in a key: all of them draw the
English rather than a blank or a raw key, so an incomplete translation is
partly translated instead of broken. The embedded copy stays reachable
underneath a disk `en-US.json` that replaces it, which is what keeps that
fallback true for a user's own correction of our wording -- and what `lang::en`
reads, since the failure log's English cannot be a file's to move. Catalogs are
sorted by tag with **English pinned at index 0**: that index is where `ACTIVE`
starts and where every lookup ends, so a tag that sorts ahead of `en-US`
(`de-DE`, `cs-CZ`) must not be able to take the seat the app opens in.

**A catalog loads whatever it was saved as.** The invitation is "copy the file
and edit it", so the loader meets the editors a translator actually has: a
byte-order mark (UTF-8's, or UTF-16's in either byte order -- PowerShell's `>`
writes the latter) is honoured rather than fed to `serde_json`, which fails on
it at the first character, and bytes that are still not Unicode are read lossily
so a codepage save comes out with visible replacement characters instead of no
language at all. A file that is there and does not load is NAMED at startup with
the reason -- `lang::unreadable` -- because "your language is not installed" is
the one answer that would send its author looking in the wrong place.

Selection is the palette's idiom exactly: an `AtomicUsize` index every surface
reads, `--lang` over the remembered choice over the system locale
(`GetUserDefaultLocaleName`), cycled live with **G** in the picker, remembered
in `settings.json`. G and the strip's language chip appear only where a second
catalog is installed. The language is settled BEFORE clap parses, off the raw
argv, because `--help` is written in it -- and clap's own furniture (`Usage:`,
`Options:`) is not reachable from `mut_arg` and stays English.

The two browser pages fetch the catalog (`/lang`, `/api/lang`) and translate
themselves through `data-t` attributes; their markup keeps the English as what
shows before the fetch lands. An argument's English help is its DOC COMMENT --
`cli.*` keys only ever override it -- which is why that one family is exempt
from the catalogs-carry-the-same-keys test.

**Width is display width, not `chars().count()`.** A CJK glyph is two terminal
cells, so the picker's row padding, footer packing and name elision all measure
through `console::measure_text_width`; measured as characters, a Chinese footer
packs past the right edge and loses its tail, where start and quit are.
`a_translated_picker_still_holds_its_columns` renders the picker in Chinese and
checks both. The processing console shares that rule: `done_line` and
`stage_body` lay out the dot leader, the detail field and the bar's leftover
width in cells, so a translated stage name keeps the `[ OK ]` chip in the
column the startup report puts it in --
`a_translated_stage_line_puts_its_chip_in_the_same_column` and
`a_translated_live_line_stays_inside_the_terminal` are the two tests.

The same rule reaches everything else drawn per cell: the viewport's panel
captions (`viz::caption` walks columns, and a wide glyph with one cell left is
dropped rather than half-drawn), the header's tagline (which is what sizes the
rule beside it), and the intro marquee, whose scroll is a column index -- so
`marquee` lays the banner out as one entry per CELL, `None` for the second half
of a wide glyph.

A style knob's `label()` is the WIRE value: `from_label` round-trips it and
`flags_line` writes it onto a command line. Both screens therefore translate it
at the last moment for display only, through the same `page.preset.*` keys.
`Palette`, `Audio` and `Volume` labels are display-only outright -- settings
persist those as serde enums -- so those translate in place.

`the_pages_only_ask_for_keys_the_catalog_has` is the guard that a page cannot
name a key nobody wrote: it scans `data-t`, `data-t-title`,
`data-t-placeholder`, literal `T(...)` in either quote, and each `?` button's
`data-h` as the title/body pair it resolves to.

The random start track uses `RandomState` (OS-seeded). The obvious clock-based
version is BROKEN on Windows and silently so: `SystemTime` has 100 ns
granularity, so `subsec_nanos()` is always a multiple of 100, and 100 is
divisible by 4 -- with four tracks `nanos % len` was always zero and the same
song played every launch. `the_launch_pick_reaches_every_track` is the test that
would have caught it; the in-range test it sat next to passed the whole time.

**The draft has a keyboard.** `Live` (in `main.rs`) puts the console in raw mode
for the length of a draft and its render thread drains key events on the same
120 ms clock it animates on -- so M/V/N/T stay live through an hours-long encode
instead of queueing up behind it. Raw mode is what makes that possible and it
takes Ctrl-C's signal with it, so the key IS the interrupt now: `cancel.rs`
counts the asks, every long loop `cancel::check()?`s between chunks, and the
`Cancelled` error unwinds the batch into a "stopped" line and exit 130. A second
Ctrl-C exits on the spot, because the first cannot land until the forward in
flight returns (~25 s on the CPU path). Stopping abruptly is safe by
construction, not by care: latents land in a `.part` file, the funscript is
written once at the end, and the cache is built to be resumed -- the same
property that makes a power cut survivable. Only presentation keys are live
during a draft; nothing that would change what is being computed, and quitting
is Ctrl-C rather than `q` so a stray keystroke cannot cost an hour of encoding.

**The draft has a viewport** (`viz.rs`): two panels above the status line,
drawn by the same render thread on the same clock for as long as the goblins are
working. *Goblin vision* is the encoder's own 24x24 latent grid as a single heat
field. Two grid rows go into each terminal row (upper-half block, foreground
over background), so 24 columns over 12 rows draws the grid SQUARE on a cell
twice as tall as it is wide; the grid is SAMPLED onto those rows rather than
indexed, so a bundle with a different grid fills the panel instead of showing
its top half.

**Beside it, *the film***: the same instant of video in colour, box-averaged
onto the SAME 24x24 grid, one cell to one cell. That 1:1 correspondence is the
entire point of putting them side by side and the reason the film is not drawn
at a nicer resolution -- a bright cell on the left names a cell on the right.
The frames are the encoder's own input, so what is on screen is the
anamorphically squashed square the model sees, not the 16:9 the player would
show. It is the one panel the palette does not own (a frame of the video is the
video's colours or it is not the film), quantized to the xterm-256 cube like
everything else -- with near-grey taken to the 24-step grey ramp rather than the
cube's four greys, because a dim interior is most of what a frame of film is.
`encode.rs` publishes it from `win[group/2]`, the same window row the heat field
comes from, so the two panels are the same instant.

The two are drawn against the log's height, and the film goes before the field
does: a terminal too narrow for the pair (`GAP` between them) keeps the field
alone, one without the rows (`MIN_LOG_H`) draws nothing at all and the run looks
as it did before there was a viewport. The panel that says what the goblins are
looking at is the one worth the rows.

**A third panel is not there, and it was tried twice.** The tracks exist during
the encode stage now, so the funscript being written is drawable -- first as a
scrolling min/max strip, then as a rule-90 automaton seeded off the decoded
reversals. Both worked and neither was worth its rows: the strip is a plot of
position with the axes filed off, and the automaton is beautiful but says almost
nothing a viewer can act on. What the goblins are LOOKING AT is the panel that
earns its space; what they are writing is what the review page is for. The
history is in git if a third panel is ever wanted back.

**The field is the mask net's attention** -- the ROI itself, the weighting the
trunk is about to pool each frame with. `mask.onnx` is the mask stack alone, one
latent row in and its eight sigmoid maps out, and `encode.rs` runs it on one row
per window. It is the REAL gate rather than a likeness: the mask net is a 2-D
conv stack over a single grid with no cut flag, no temporal context and no
dropout (`frontend` is deterministic by construction), so one row through this
graph computes exactly what the draft's own attention pooling used on that row.

**The heads are combined the way the pooling combines them**: each head
normalized over the grid to sum to one, then averaged. That is not a taste
choice between mean and max. Attention pooling divides each head's gate by its
own sum before it weights anything (`att = gate / gate.sum()`), so the
normalized map IS the weight that head carries and the raw sigmoid is not -- a
head sitting at 0.9 across the whole grid is loud raw and says nothing about
where the model looks. Normalized, it contributes a flat nothing and the peaked
heads show through; a test pins the contrast against what averaging the raw
sigmoids would have given. The result is one distribution over the grid,
whatever head count the checkpoint carries.

It is a separate graph for a rate reason, not an arithmetic one. `head.onnx`
already computes this gate and discards it unread, and it now runs in this same
stage -- but it runs once per 1024-row CHUNK, which is one picture per ~34 s of
video against one per window here, and it would be ~19 MB of gate to carry back
for it against 18 KB for the one row this graph is asked for.

**The cost is bounded on purpose.** One row per window is the whole control: a
window is a fraction of a rendered frame at this speed, so a single row keeps
the picture moving, and it caps the decoration at one small CPU forward per
~120 ms of wall clock. The stack is ~40 M multiply-adds (128->48 3x3, 48->32
3x3, 32->8 1x1 over 576 cells), measured at **0.43 ms on one thread** -- 0.4% of
a window -- and the session is pinned to exactly that one intra-op thread,
because everything around it is already spending the machine: the encoder owns
the GPU, and in a batch `prefetch` owns the cores with libx264. The session is
not even built on a piped run, where nothing would be drawn with it.

**Latent band energy is the fallback**, for a bundle exported before the mask
graph: the per-cell L2 norm across the PCA dimensions of the row the encoder
just wrote. It is free -- already in registers on its way to the cache -- and it
shows where the picture is busy rather than where the model is looking. A
malformed gate is refused rather than drawn, and a mask forward that fails takes
the graph out of the run instead of being retried every window. The caption
names whichever source is on screen.

It is decoration and nothing else: the draft is byte-identical with the panel on
or off. Malformed input is dropped rather than raised, for the same reason: a
picture must never take a draft down.

**Colour is a thermal ramp, not a brightness ramp.** Each palette's `ramp` is
twelve xterm-256 steps running black to white THROUGH that palette's own hues --
phosphor climbs green to yellow to white, amber walks a filament's red to orange
to yellow, cga runs blue to magenta to pink to cyan. The eye separates green from
yellow from white far better than it separates two greens, so the same data
carries more of itself; every ramp still runs dark to light (a test holds it) so
the field survives having the colour taken away. T recolours it live.

**Two things stop it flickering, and both are load-bearing.** The field is
smoothed in TIME and normalized on its OWN range: normalizing against its own
min/max is what makes the contrast worth watching -- an attention map living
between 0.4 and 0.6 of its peak is one flat grey on an absolute scale -- but
done raw it rescales the whole picture every window and a still scene strobes,
so the field runs through an EMA and its normalization window chases its
extremes more slowly still (`FIELD_EMA`, `RANGE_EMA`). A source change
(attention to bands) reseeds both outright rather than easing, so the panel
never blends two quantities that were never measured together. And the block is
repainted IN PLACE -- `home_live` moves the cursor and clears nothing, each line
erasing only its own tail as it is written.
Blanking thirteen rows before refilling them lets the terminal present the blank
frame, which is exactly what flicker is. Only two cases still wipe: a resize,
which has reflowed the rows underneath, and a block that got shorter, which
would otherwise leave its old bottom rows standing.

The panel enlarges the live area from one line to fourteen, which
is why `last_vis` is a VECTOR of visible widths -- one per drawn line. The cursor
climbs the sum of the rows those widths occupy at the CURRENT terminal width, so
a shrink that reflowed three of them is still accounted for exactly, and the log
above never moves. A terminal without the rows to spare (`MIN_LOG_H`, counted
against the log's own height -- the pinned header takes four) or the columns for
even the small layout simply does not get it, and the display is the single line
it always was.

**The cursor is hidden for as long as the header is pinned.** `chrome::frame`
paints by addressing the header rows and handing the cursor back
(`\x1b7` ... `\x1b8`), which is correct and still leaves it standing up there
for the length of the burst -- and a terminal draws its cursor from wherever it
has read to, so several times a second it appeared in the bottom-left of the
header and went again. Nothing on that screen is waiting to be typed into
(every key the draft takes is a hotkey), so `Header::begin` takes the cursor
away and every path that stops pinning gives it back: `Drop`, and `release` for
the second Ctrl-C that leaves by `exit` and runs no destructor. A cursor left
hidden outlives the process and belongs to the user's next shell, which is why
that last one is a test.

Two restraints the flair does not get to cross. The processing console must
survive `> log.txt`: it is deliberately not an alternate-screen TUI (see above),
its status chips are plain ASCII, and `console` drops the colour itself when
stdout is redirected. And the review page's CRT scanlines are pinned to the
chrome only -- `#stage`, `#timeline` and `#strip` sit on a higher layer -- because
that page's job is judging stroke timing against real footage, and a decorative
overlay on the instrument would be a lie about what the user is looking at.

## Styling parameters & the review loop

Every user-facing knob acts DOWNSTREAM of the head's tracks (`style::Params`):
dwell-call confidence (`--dwells`, three levels on the peak filter), the
stillness gate (`--stillness`), and pure output shaping (`--intensity`,
`--range`, `--max-speed` -- funscript-domain transforms on the finished
actions). At the defaults the surface is an identity on everything a player
could have performed anyway, and a bare run reproduces the manifest's draft
bit for bit (`default_params_are_an_identity`).

`--max-speed` is the one exception, and deliberately so: it defaults to
`MAX_POS_RATE`, the cap `jepa_infer` applies to every list IT writes, so a
transition asking for depth no device delivers is pulled in (timing never
moves). Shipping it opt-in meant every draft carried transitions the Python
decode would have clamped -- 48 per clip over the cap, peaking at 1305 pos/s,
across the out-of-sample set. `--max-speed 0` is the way back to the model's
uncapped reach; `grid_check`'s mirror set now holds the constant equal on both
sides. The manifest stays the single source of "normal"; presets override it,
they never redefine it.

`--review` (implicit when launched from the picker) keeps the process open
after drafting. The tracks stay in memory, so a parameter change re-composes
and rewrites every drafted script in milliseconds -- the encoder never runs
again. On exit the console prints the equivalent flags, so a batch re-run
reproduces the accepted setting.

The review surface is a browser page served by the exe itself (`review.rs` +
the embedded `review.html`; loopback only, ephemeral port, nothing written to
disk, no external requests): the video plays with a SIMULATOR overlaid -- the
scripted position riding the frame's right edge -- above a scrubbable script
strip and the style controls; every change POSTs back, the scripts on disk
rewrite, and the page redraws, so the user's own player stays a valid second
screen. The page also surfaces the model's per-row confidence head (`conf` in
`head.onnx`, trained so its window mean tracks the phase correlation the model
expects against a human script): a live readout at the playhead plus a cool
slate band along the strip. It rides `/conf/N` and is served ONCE per clip, not
folded into `/state` -- confidence is a property of the frozen trunk, so a
re-style never changes it (a bundle exported before the head existed serves
`conf: null` and the page hides the readout).

The written funscript carries nothing but the actions and an author stamp:
`metadata` is the author alone, and it serializes ahead of the long `actions`
array so a text editor shows it at the top of the file.
Which file `/video/N` streams is probe-and-fallback
(`ffmpeg::browser_playable`): the ORIGINAL when its container/codecs pass the
every-browser whitelist (full quality -- most flat MP4 sources), otherwise the
cache's normalized copy, H.264+AAC MP4, which plays everywhere; the page retries on
the copy client-side if playback still errors. **Every request is answered at once, and the re-style happens elsewhere.**
tiny_http accepts on a background thread but ANSWERS from the request loop, so a
re-style running there stalls the video stream, the timeline and the Done button
along with it. A long video makes that unmissable: the emission-prior fit alone
is ~64 runs of the event decode over the whole clip (`bias_fit_s` is 8000 s, so
at 30 rows/s the window IS the clip), which is 220 ms at two hours and rises
from there. So `/params` records the change, hands it to a styler thread and
returns; the clip is marked `styling` in `/state`, and the page waits on that
flag before re-reading the script rather than by holding a socket open. File
requests still go to short-lived threads and answer ranges in capped 206 chunks,
so a browser buffering a multi-GB original was never the problem here.

At most ONE re-style is outstanding per clip: a knob turned while an older change
is still running REPLACES it, because the older answer is a script nobody will
read and paying for it would put every later change that many seconds further
behind. The queue is drained on the way out rather than abandoned, so someone who
turns a knob and hits Done a second later still finds that knob's script on disk.
`the_queue_supersedes_in_flight_work_and_drains_on_the_way_out` holds both.

The decode itself got cheaper by the same measurement: the viterbi's three
per-row `Vec`s are hoisted out of its loop (a clip-length window made that
millions of allocations per decode, and the fit runs the decode dozens of times),
and the bias fit refills one prior array instead of rebuilding it per probe.
Bit-identical output, 716 ms -> 220 ms for a two-hour fit -- see the
`#[ignore]`d `fit_cost_at_two_hours`.

If the server cannot start, the terminal review screen (`tui.rs`) takes over --
same knobs, same restyle contract.

## Cache

`cache/<fingerprint>/` next to the executable holds the transcode, the cuts and
the encoder's latents (~1 MB per second of video) WHILE a draft runs. The
fingerprint is a blake3 of the source's size plus its head and tail megabyte
-- deliberately not the filename. The cache is ephemeral:
a video's directory is removed the moment its funscript is written (with
`--review`, the moment the review closes -- the review page may be streaming
the normalized copy out of it). Its actual
job is resume-after-interrupt -- a crashed or killed draft keeps its files, and
the re-run picks up at the first missing artifact instead of paying for hours
again. `--keep-cache` keeps them after success too (a re-draft then costs
seconds); `--cache <DIR>` moves the root.

A fully in-memory pipeline was considered and rejected: the transcode must
exist as an encoded stream (the model's input distribution includes the x264
round trip) and is read by two passes, and latents for a 6 h video are ~22 GB
-- streaming them through the head would fit, but a crash at hour two would
then restart from zero.

The latents record which bundle produced them: a new champion or a refit PCA
basis invalidates them rather than feeding a new head someone else's features.

## Building

`cargo build` is the whole of it, and `cargo test` runs green on a bare
checkout -- everything below the CLI is plain Rust with no model in it.

What a checkout does not carry is the **bundle**: the ONNX graphs and the
manifest of deploy constants that make up the goblin itself. It is a build
product of the training tree -- exported from a champion checkpoint, weighing
~250 MB -- so it is neither committed nor part of the source distribution. A
released `goblinscript.exe` has one baked in.

```bash
cargo build --release                        # no model: needs --bundle <DIR>
cargo build --release --features embed       # bakes bundle/ into the exe
```

A release binary will hand its own bundle over: `goblinscript --dump-bundle
<DIR>` writes the baked-in graphs back out, byte for byte and named as the
manifest names them, so the directory it leaves is one `--bundle <DIR>` reads.
That is the supported way to get a model without the training tree, and it
needs no GPU and no ffmpeg -- it is bytes out of the exe, dispatched before
any of that is looked for. `--features embed` reads `bundle/` at compile time
and will not build without it.

**Two platforms, one floor each.** The session chain in `bundle.rs` is
target-gated. Windows registers DirectML, which every DirectX 12 vendor
serves, so the Windows zip runs on any card with nothing installed beyond the
driver; the `cuda` feature puts a CUDA attempt in front of it, falling back
at runtime. Linux registers CUDA, and there is nothing under it. That is a
narrower floor on purpose: the encoder's packed attention has no CPU kernel,
so a platform with no GPU provider has no floor at all, and the
vendor-neutral option -- ONNX Runtime's WebGPU provider over Dawn -- costs
2.9x (152 ms a forward against CUDA's 52 on the same 4090) because it reaches
the tensor cores from nothing but `MatMulNBits`, while asking for a second
model export and a second set of graph rewrites to feed it. So off Windows
CUDA is a dependency rather than a feature
(`[target.'cfg(not(windows))'.dependencies]`), a Linux build cannot be
configured without it, and the plain `cargo build` CI runs is the build the
release ships. The Linux binary carries `$ORIGIN` in its rpath, so the ORT
CUDA provider libraries beside it are found the way `DirectML.dll` is found
beside the exe -- and so are NVIDIA's own CUDA and cuDNN libraries when they
were dropped in there instead of installed. `cargo xtask dist` packs
whichever platform it runs on.

**One bundle, both platforms.** DirectML and CUDA each have a kernel for the
packed-QKV `MultiHeadAttention` and for Conv3d, so the same `bundle.zip`
serves Windows and Linux and there is nothing to export twice. On a 4090 the
encoder forward is 59 ms on DirectML and 52 on CUDA, and the shot-cut
detector's 100-frame window is 4 ms on CUDA against 60 on the CPU. `--bench`
runs the detector once as well as the encoder, because a GPU provider builds
a session it cannot execute and only says so at the first node.

**Why not a vendor-neutral Linux floor.** ONNX Runtime's WebGPU provider over
Dawn runs this model correctly on any Vulkan driver, and it was the Linux
floor for one release. It costs 2.9x -- 152 ms a forward against CUDA's 52 --
and the reason is not the graph: attention is 92 ms of it and the MLP 35,
both at ~32 TFLOP/s, which is ~40% of the card's *vector-ALU* fp16 peak,
because that provider reaches the tensor cores from nothing but
`MatMulNBits` while DirectML's metacommands and cuBLAS use them for
everything. The attention was already on its FlashAttention path (written
unfused it is 363 ms, and its peak allocation never holds a score matrix),
and no shape of the graph closed the gap. It also has no kernel for packed
attention or Conv3d and no 8-bit type, so it needed a second model export
with four rewrites of its own to feed it. Two artifacts and a second export
path, for a third of the speed, is what was dropped.

**The release zips come from GitHub Actions.** A `v*` tag runs
`.github/workflows/release.yml`: one runner per platform fetches the inputs
a checkout lacks from the permanent `bundle` release of this repository --
the model, `bundle.zip`, and `music.zip`, the soundtrack -- checks the model
against `bundle.sha256`, packs with
`cargo xtask dist`, and attaches the zip to the tag's release, creating a
draft when none exists. The asset names never change, so the workflow never
does; a new champion is one re-uploaded bundle zip and one new line in
`bundle.sha256`, in the commit that bumps the version. The pin is what keeps
an old tag honest: rebuilt a year later, it fails instead of embedding
whatever model is current.

The vision graphs a shipped bundle carries are **fp16**, with every Softmax
fused into `com.microsoft.MultiHeadAttention` -- a partly-fused encoder is not
a thing to ship, and `cargo xtask dist` refuses a non-fp16 bundle outright.
An fp32 encoder is twice the size and ~6.4x slower on DirectML; it exists as a
range-proof reference to attribute numerics against, never as a release.

**A build has exactly one GPU provider.** `cargo build --release --features
embed` is DirectML and `--features embed,cuda` is CUDA, and neither falls back
to the other, because the two do not live in one ONNX Runtime. `error_on_failure`
is what makes a machine without a usable card FAIL the attempt rather than
silently land on the CPU provider registered beside it, which has no kernel for
the packed attention and would die at its first node with a message about the
kernel rather than about the card. The draft is EP-independent at the product
level: the CUDA build's funscript scores the same against the reference draft
as DirectML's does.

**Which ONNX Runtime is a per-package choice, and it is about Blackwell.** The
DirectML build links the runtime the `ort` crate downloads, which is the only
one of the two that has a DirectML EP. Its CUDA provider is compiled for
sm_75, sm_80 and sm_90 with NO PTX, so it has no kernel for a Blackwell card
and nothing to JIT one from -- an RTX 50-series machine would have no GPU path
at all on Linux. So the CUDA builds link Microsoft's own
`onnxruntime-*-gpu_cuda12` package instead, which covers sm_61 to sm_120 on
Windows and sm_60 to sm_120 on Linux, and they ship it beside the binary.
`ORT_LIB_LOCATION` points at its `lib` folder; it is the variable cargo reads
and the one `xtask` packs from, so a zip cannot carry a runtime its binary was
not built against. On Linux that package holds the `.so` alone, and
`ort-sys` links a lib folder statically unless `ORT_PREFER_DYNAMIC_LINK=1`
says otherwise -- without it the build looks for `libonnxruntime.a`, finds
none, and stops with "could not link". Windows needs no such thing: its
`onnxruntime.lib` is the import library `ort-sys` asks for by name.

Microsoft's runtime needs CUDA **12.9** beside it, not 12.6:
the older libraries load and then fail on a missing symbol, saying
`undefined symbol: cudaLibraryGetKernel` on Linux and nothing useful at all on
Windows (`Error 127`).

**The ONNX Runtime version is PAIRED to the `ort` crate, and the pairing is not
advisory.** Each `ort-sys` release names the runtime it carries bindings for --
rc.12 is 1.24, rc.13 is 1.28 -- and the crate's `api-NN` feature has to match:
`api-28` for 1.28. Running rc.12's bindings against 1.29 builds, links, draws a
numerically correct draft, and then ABORTS on Linux at teardown with `corrupted
double-linked list` (exit 134). It is silent on Windows, which is the trap: the
same mismatch is there and only one platform says so. The isolation that found
it is worth keeping: ORT 1.24.4 linked dynamically exits cleanly, so it is the
version gap and not the dynamic linking. When bumping either side, bump both,
and check the CUDA provider's architectures with `cuobjdump --list-elf` before
believing a version reaches the cards it claims to -- it carries NO PTX, so
what is not compiled in cannot be JIT'd.

`cargo xtask dist` packs `dist/goblinscript-<ver>-windows-dml.zip` (exe +
DirectML.dll + `THIRD-PARTY-NOTICES.txt` + the operator's `README.txt` -- runs
on any DX12 GPU). The version is SemVer
with a shipped-release rule: a bare `X.Y.Z` names a build the user has called
shipped, and every build ahead of that call carries a `-rc.N` suffix in
`Cargo.toml` -- so a shipped-looking dist name on disk is proof of a ship,
and a candidate can never overwrite a release. On Linux it packs
`dist/goblinscript-<ver>-linux-cuda.zip`, and on Windows
`cargo xtask dist --cuda` adds `...-windows-cuda.zip`. Whatever `bundle/`
holds is what ships -- the xtask prints its checkpoint so a stale bundle is
caught before it goes out.

**A CUDA zip carries NVIDIA's runtime.** ONNX Runtime loads its provider
library in a way that does NOT resolve that library's own dependencies from
PATH: `cudnn64_9.dll` on PATH is reported missing, and only a copy beside the
binary is found. So an installed CUDA on the user's machine buys nothing, and
the ten libraries `cargo xtask cuda-libs` names ride in the zip instead. They
come from NVIDIA's redistributable archives into `cuda-libs/`
(or `CUDA_LIBS_DIR`); the release workflow fetches them, a local pack does
not, and a missing one stops the pack by name. `CUDA_RUNTIME_WINDOWS` in
`xtask` is the list, and it is the MINIMUM measured by removing libraries
until the graphs broke -- `cudnn_engines_precompiled` is what the shot-cut
detector's Conv3d builds its engine from, while `cudnn_adv`, `cufft`, `curand`
and `nvrtc` are another 638 MB that changed no timing. NVIDIA's redistribution
terms go into that zip's notices file.

That runtime is ~1.3 GB, which is why Windows gets two zips: an NVIDIA owner
takes the CUDA one and reads 60 ms against DirectML's 72 on a 4090, one sixth
off every draft, and everyone else takes the dml zip and carries none of the
weight. On Linux there is no choice to make -- CUDA is the only GPU path, so
the one Linux zip carries it and the user needs an NVIDIA card and its driver
and nothing more.

### Licensing and third-party notices

GoblinScript is MIT (`LICENSE`). The zips carry that file and
`THIRD-PARTY-NOTICES.txt` beside the exe, and `xtask` refuses to pack without
either: MIT asks that its notice travel with every copy, and the notices file
names `LICENSE` as where the app's own terms live.

`THIRD-PARTY-NOTICES.txt` is attribution for every third-party crate and model
weight GoblinScript ships (V-JEPA 2 / TransNetV2 weights, ONNX Runtime,
DirectML, and the Rust crate tree). The whole stack is permissive (MIT /
Apache-2.0 / BSD / Unicode), and permissive means one obligation: reproduce
those licenses.

It is generated, not hand-edited -- the crate section comes from `cargo-about`
(`about.toml` config + `about.hbs` template, which also holds the hand-written
header for the non-crate components). Regenerate it whenever the dependency tree
changes:

```bash
cargo install cargo-about --features cli   # once
cd goblinscript
cargo about generate --manifest-path Cargo.toml about.hbs -o THIRD-PARTY-NOTICES.txt
```

`about.toml`'s `accepted` list is the permissive set only, so a copyleft
dependency (GPL/LGPL/AGPL) entering the tree makes `cargo about` fail rather
than silently shipping a license violation -- treat that failure as "this
dependency cannot ship inside a prebuilt MIT binary," not as a config to widen.
`ffmpeg` is deliberately absent from the notices: it is invoked as an external
program the user installs, never linked or bundled, so its GPL/LGPL does not
reach the binary. Never ship an `ffmpeg` build inside the zip -- that would pull
its license in.

Without `--features embed` the binary has no model in it and needs `--bundle
<DIR>` -- which is how you iterate on a bundle without a ~250 MB rebuild each
time.

The bundle is the only seam between the research tree and the product. It is
traced from the checkpoint's own arch, every graph in it is checked against
PyTorch before it is written, and the manifest beside them carries every deploy
constant (v_std, the level-decode temperature, the stillness gate, the boundary
thresholds) so that none of them are re-guessed in Rust.

## Parity

The graphs are not trusted because they exported: a bundle is checked against
a reference forward before it ships. Two things here exist for that check and
for nothing else.

**`--from-latents`** (hidden) feeds the head stage rows read off disk instead
of an encode. It is the only way `heads.rs`'s own chunk boundaries, ctx
padding, short-tail lookback and stepped envelope buffer are the things under
test rather than a re-implementation of them, and it reaches them with the
encoder taken out of the picture. Read the result by comparing max|d| at the
chunk seams against max|d| in the interior: the two moving together is
execution-provider numerics, a seam far above the interior is the tiling.

**Determinism**, so that a comparison at the funscript level means something.
A re-draft from a kept cache is byte-identical, and a draft is EP-independent
at the product level -- so a moved number is a real change and never noise.
That level is worth checking on its own: a port that reproduces every track
and ships different actions is still a broken port, and only the actions show
it. Run one with `--no-transcode --no-autocrop --no-exposure` on a clip that
is already at the deploy spec, or the comparison measures the stages rather
than the port.

## A new champion

A new champion arrives as a new bundle: drop it in and rebuild. The manifest
carries the arch, so a checkpoint that only retunes the existing heads needs no
Rust change. A checkpoint that adds a NEW head -- or new DECODE logic -- needs
the styler taught what to do with it: `style.rs` implements exactly the
composed styling the current champion deploys and nothing speculative —
band-rail snapping, the stillness hold, and the trapezoid-dwell
level lock with its peak call filter (`dwell_kind` + `level_lock`,
ported from jepa_infer with its numpy edge conventions). The lock's
three thresholds are per-CHECKPOINT calibration read from the manifest,
not constants: they threshold the dwell head's published posterior, so
a checkpoint whose head folds its class prior out needs its own triple
and a stale literal here would silently ship the wrong operating point
(`jepa_infer.PLAT_*` is the one definition both decoders read). The split runs
right down that line: a head is a graph and rides the export untouched, while
anything that reads its output is decode logic and lands here, as a port plus
whatever manifest constant it thresholds against.

## Open

* AMD is unvalidated locally (no AMD box); DirectML semantics are
  vendor-neutral and `--bench` self-checks finiteness and latent std in the
  field.
* **Four attention levers are measured dead -- do not re-run them.** All at the
  real shape (1x12x9216x64 fp16, a 4090), all against the full forward:
  * *Tiling the score matrix, unfused*: flat 15.0-15.3 ms per block from 2 GB
    tiles down to 16 MB, bit-identical output. Tiling does not reduce total
    traffic, because separate ONNX nodes cannot hold a tile across
    MatMul -> Softmax -> MatMul.
  * *Tiling the FUSED node*: monotonically WORSE -- 0.98x at chunk 2304 falling
    to 0.70x at 144. A smaller score buffer buys nothing and the dispatches cost.
  * *A newer DirectML*: 1.15.4 is the newest that exists (nuget
    `Microsoft.AI.DirectML`, 28 versions). There is no version to upgrade to.
  * *The WebGPU EP*, the other vendor-neutral backend: runs the graph correctly
    at 383 ms against DirectML's 144 -- 2.5x slower.

  What DID work was the input layout, which is a property of the graph and not
  of the backend: see `fuse_attention`. Two supporting measurements from the same
  session -- DirectML sustains 165-199 TFLOP/s on plain fp16 GEMM (faster than
  cuBLAS on the same card), and disabling metacommands costs 2.1x. The EP was
  never the bottleneck and the vendor kernels were always live; the graph was
  simply asking for them in the wrong shape.
* The next thing in the encoder, and now the largest single item: everything
  that is NOT attention costs **21 ms of the 60 ms** forward (measured by
  ablating the 12 MHA nodes to `Identity`). The graph carries 2337 nodes, of
  which 327 Cast / 230 Reshape / 253 Unsqueeze / 108 Concat / 72 Neg are RoPE's
  rotate-half and the explicit dtype casts `_patch_rope_for_onnx` writes down.
  The 72 `Neg` are the count that matters: V-JEPA 2.1's RoPE is 3D, applied to
  three slices of the head dim per q and per k, so 12 blocks x 2 x 3 axes = **72
  sites**, each ~11 elementwise passes over a 14 MB tensor. Not all 21 ms is
  reachable -- the MLP and qkv GEMMs are ~9 ms of it at DirectML's measured
  fp16 rate -- so the ceiling here is roughly **6-9 ms of the 60**.

  **It is worth 1.07x and it costs numerics -- measured, and the ceiling is now
  known.** ORT profiling cannot see inside the encoder (DirectML collapses the
  whole graph into one `DmlFusedNode`, and `session.disable_dml_graph_fusion`
  does not change that), so the site was benched standalone at the real shape:
  `(1, 12, 9216, 20)` fp16, 24 chained, on a 4090. (`--bench` read 65 ms that
  session against the 60 ms recorded above -- the machine was a few percent
  down. The shares are what the table is for, not the absolute clock.)

  | variant | ms/site | x72 | provider |
  |---|---|---|---|
  | the chain as exported (fp32 math) | 0.127 | **9.1 ms** | Dml |
  | the same chain kept in fp16 | 0.103 | 7.4 ms | Dml |
  | `com.microsoft.RotaryEmbedding`, fp16 | 0.071 | **5.1 ms** | Dml |
  | `ai.onnx` v23 `RotaryEmbedding`, fp16 | 0.429 | 30.9 ms | **CPU** |
  | `Cast -> com.microsoft fp32 -> Cast` | 3.666 | 264 ms | **CPU** |

  So RoPE is 9.1 ms of a 65 ms forward and the best fusion leaves 5.1 -- a 4 ms
  saving, **1.07x**, for a graph rewrite plus a parity re-run. Four things fell
  out of it, all of which kill the version that would have been safe:
  * **There is no bit-exact route.** DirectML's kernel is fp16-only; the fp32
    node falls to the CPU and runs 29x SLOWER than the chain it replaces. Any
    win here is a numerics change (max|d| 2.8e-02 over 24 chained sites), so it
    is the parity gate's business, not a free lunch.
  * **Only the contrib domain reaches the GPU.** The standard opset-23 op is
    CPU-only on this ORT/DirectML, which makes `torch.onnx.ops.rotary_embedding`
    (it emits the standard op) the wrong emitter -- it wants
    `torch.onnx.ops.symbolic` against `com.microsoft`.
  * **Fusing is only half the win.** 1.7 ms of the 4.0 comes from dropping
    `_patch_rope_for_onnx`'s three fp32 casts per site, which is a three-line
    change and no new op at all. It carries the same numerics cost.
  * The cache must cover the SEQUENCE and be an initializer (a per-position
    cache is refused: "Updating cos_cache and sin_cache in RotaryEmbedding is
    not currently supported"). That part is free -- the token grid is fixed, so
    one row per token with `position_ids = arange` is ~180 KB per axis, shared
    by q and k across all twelve blocks, and it bakes out `interpolate_rope`'s
    non-integer spatial positions.

  1.07x is the honest number to weigh the rewrite against, and it replaces this
  section's old claim that RoPE was where the encoder's remaining time lived.
  The forward's 65 ms is roughly 39 attention / 9 RoPE / 9 GEMM / 8 the rest:
  **attention is still 60% of it, and it is already fused.**
* Untried: **one forward for both alignments**. `encode.rs` runs the graph twice
  per window at batch 1 -- once per alignment -- and a batch of 2 would pay the
  node launches once. The measurements above argue it is worth little: DirectML
  already fuses the graph into a single node, so there are not 2300 launches to
  amortize, and at 0.127 ms for 24 chained elementwise ops the graph is reading
  memory rather than waiting on dispatch. The thing to watch if it is tried
  anyway is not the clock but the **working set**: 4.5 GB of VRAM is what makes
  the 8 GB floor in `README.md` honest, and a batch of 2 that takes it past 6 GB
  buys speed on a 4090 by dropping the cards this is supposed to run on. Treat a
  VRAM regression as disqualifying rather than as a trade.
* Deeper overlap than one video ahead. `prefetch.rs` hides the CPU-side
  transcode behind the previous video's GPU work, which took a 3-video batch
  from 47 s to 34 s (1.4x, drafts byte-identical). What it does NOT hide is the
  first video's transcode, or the remainder when the transcode is longer than
  the draft it hides behind -- on the synthetic clips above, 9 s of normalize
  against ~7 s of GPU leaves 2 s exposed every time. What is left is running the
  transcode at a lower preset, which is not available: the spec is corpus
  distribution, not a tuning knob.

  **Overlapping WITHIN a video is measured dead -- do not re-run it.** A frame
  reader on its own thread, one window ahead of the encoder, drafted a 3-minute
  clip at **13.3x realtime against the inline reader's 13.2x** (release build,
  same clip, byte-identical funscripts). The ~5-10% this section used to claim
  was there was never there: ffmpeg already decodes in its own process, so the
  only serial part was the pipe read, and the OS buffer covers that while the
  GPU works. A 4090 is also the most FAVOURABLE case for the idea -- the faster
  the GPU relative to the CPU, the more a read costs it -- so a wash here is a
  wash everywhere.

  That still holds at 30 Hz, where the GPU cost per frame HALVED and the
  decoder's share of the clock therefore doubled: the exact decode
  `Decoder::open` asks for (`fps=30,scale=384:384`, rgb24) standalone runs at
  **5028 fps against the 255 fps the encoder consumes**, a 20x margin. The
  decoder cannot be the limiter on any grid this encoder will run on.
* **The CPU work around the forward is ~38 ms of a 290 ms window, and it is
  memcpy, not allocation.** The encoding runs at 7.4x against the forward's 8.5x
  ceiling, and that gap is where the last in-process headroom is. What was
  reclaimed of it: the input staging buffer, the row slab and the frames are
  now allocated once per clip rather than per window (`encode.rs` -- a fresh
  frame `Vec` cost an allocation AND a 442 KB zero-fill that `read_exact`
  immediately overwrote, 64 times a window), and the viewport's mask forward
  moved to its own thread. Measured **19.14 s -> 18.80 s** on a 120 s clip,
  4 alternating reps, funscripts byte-identical: **1.8%**, and every rep
  favoured it.

  The hypothesis that this was worth ~13% was WRONG, and the accounting says
  why: a window moves **83 MB** through the CPU -- 29 MB read off the decode
  pipe and 54 MB gathered into the staging buffer -- and that gather is
  irreducible. Alignment `j` needs frames `(a, a+k)` for `a = j, j+2k, ...`,
  which are strided in the decode order and contiguous in the tensor, and ONNX
  takes no strided input. Filling per-alignment buffers as frames ARRIVE costs
  exactly the same: each frame is the head of one tubelet and the tail of
  another, so it lands in exactly 2 alignment buffers either way. The only
  reachable piece left is reading each frame straight into one of its two
  slots (1 read + 1 copy instead of 1 read + 2), worth maybe another 2% for a
  tail path that would have to pad into alignment buffers rather than frames.
