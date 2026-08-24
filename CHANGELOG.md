# Changelog

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
