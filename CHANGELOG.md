# Changelog

## 0.4.0-rc.2

**A candidate, not a release.** These goblins are on disk and under test;
the shipped build is still 0.3.4.

**New goblins.** They learned on more videos than the ones before them, they
look further along a video before they decide, and they were taught to count
their own movements against the person's. This is the whole of it: there is
no new command and no new button.

**Less movement that is not in the video.** Where a video is slow, the goblins
used to put in sudden fast movements the person never wrote. There are about
15% fewer of them now. Over a whole video they write 26 fast movements each
minute where the goblins before them wrote 31, and a person writes 22.

**They follow the action a little better**, and they invent slightly fewer
changes of direction. They do find a few less of the person's changes of
direction than the goblins before them did -- the poster on the release page
puts every version next to the other, and next to a person, so you can see
the trade.

**Everything else is the same.** Your settings, the review page, the languages
and the folder picker do not change.

## 0.3.4

**GoblinScript is open source.** The app is at
<https://github.com/ofs69/GoblinScript> under the MIT license -- the picker,
the video handling, the review page, and the styling that turns what the
goblins see into strokes. The goblins themselves are not source code: what
they know is a trained model, and it is still baked into the exe you download.

**Take the goblins with you.** `goblinscript --dump-bundle <FOLDER>` writes
that model out, and a GoblinScript built from source picks it up with
`--bundle <FOLDER>` -- from there it drafts exactly as this one does. If you
are not building anything, you never need it.

None of this changes what the goblins write.

## 0.3.3

**Take a whole folder at once.** Space on a folder in the picker puts every
video in it into the batch, so a fresh drop of clips goes in from the row above
them instead of a trip inside -- and the box on that row says how many of its
videos are in. From a command line, `goblinscript D:\clips` does the same. Just
that one folder either way: the folders below it are somebody else's batch, and
you pick those by going in.

**Videos with a timecode track draft again.** A file whose video is tied to a
timecode track has that video reported twice by FFmpeg 9, and the goblins read
the second copy's width where the first one's height belonged -- "no height",
right before auto-crop. They take the picture's own size now, however many
times it is listed, and the copy they normalize carries no timecode paperwork
of its own.

**A failure outlives the screen it was printed on.** Whatever goes wrong is
written to `goblinscript.log`, beside the exe with the settings and the cache:
which video, how far it got, and the whole chain of reasons. Always in English,
whichever language the goblins are speaking, so it can be pasted straight into
a report.

## 0.3.2

**The review knobs work in every language.** In Chinese, turning a knob was
refused and the gap filler's presets did nothing. Both work.

## 0.3.1

**The goblins speak your language.** English and Chinese ship with them, and
anyone can add a third: copy `languages/en-US.json`, translate the right-hand
side of every line, save it under your own tag. No build needed, and an
untranslated line comes out in English. Your machine's language is picked
automatically, **G** cycles, `--lang zh-CN` sets it -- `--help` included.

**Big folders open instantly.** Listing a few thousand videos is about ten
times quicker, and filtering no longer re-reads the folder, so typing a search
is instant.

**The review page stays live on long videos.** Turning a knob no longer freezes
the page -- it keeps playing while the script is rewritten, and Done never
loses a change still in flight. The rewrite itself is about three times quicker
on a two-hour video.

None of this changes what the goblins write.
