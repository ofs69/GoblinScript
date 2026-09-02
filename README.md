# GoblinScript

Drop in a video, get a funscript. **Made by real goblins.**

A goblin watches the video and scrawls a matching script: strokes where the
action is, timed to the motion on screen. The result lands next to the video
as `video.funscript`, ready for any player that reads funscripts. Treat it as
a solid first draft -- the goblin aims to nail the rhythm and the timing, and
you can fine-tune anything else in your favorite editor.

> **This repository is the inference client only.**
>
> It is the app that *runs* a trained model -- decode, perception, and the
> styling that turns what the goblins see into strokes. It is **not** the
> training pipeline, and it does **not** contain the model itself. That is a
> separate ~250 MB bundle, baked into released binaries; a build from source
> needs one passed with `--bundle`. The training tree is not public yet.

## Using it

Double-click `goblinscript.exe` and a picker opens in the terminal: browse with
the arrow keys, select with Space, press **S** to set the goblins to work.
Or from a command line:

```
goblinscript video.mp4                  the script appears next to the video
goblinscript D:\clips                   every video in that folder
goblinscript video.mp4 --out D:\scripts put the scripts somewhere else
```

Before the goblins start, a page opens in your browser with the part of the
picture they are about to watch drawn on each shot. Drag that rect onto the
action if they picked the wrong part, or click through if they got it right;
what you draw is kept for that video. Turn the page off with `--no-crop-edit`,
or with **K** in the picker, and a batch runs with nobody at the keyboard.

`goblinscript --help` lists every option. A VR video opens a page in your
browser to aim it once; flat video needs no flags at all. English and Chinese
ship with it, and any `languages/*.json` you translate shows up on the **G**
key. Existing scripts are never overwritten without `--force`, and the video
file itself is never touched.

## What you need

* **Windows 10 (64-bit) or 11**, or **Linux** on x86-64 (Ubuntu 22.04 or
  newer, or the like).
* A graphics card with **8 GB of VRAM**: **DirectX 12** on Windows (NVIDIA,
  AMD, Intel), **Vulkan** on Linux (Mesa serves AMD and Intel, NVIDIA's own
  driver serves NVIDIA -- nothing from CUDA to install). There is no CPU-only
  mode: the goblins think in a form only a GPU can hold.
* **8 GB of system memory**, or more.
* [ffmpeg](https://ffmpeg.org/) on your PATH -- `winget install --id Gyan.FFmpeg`
  and then a new terminal on Windows, `sudo apt install ffmpeg` on Linux.
* Keep the `.dll` or `.so` from the zip beside the goblinscript binary.

Roughly half the video's running time on a fast GPU. Everything runs on your
machine: no uploads, no internet needed.

Funscripts drive adult toys, and adult video is what the goblins were taught
on. Nothing here is for anyone under 18.

## Building it

```
cargo build --release     # builds; no goblins in it yet
cargo test                # green on a bare checkout
```

The model is not in this repository, but a **released** `goblinscript.exe`
hands its own over. Ask it for one, and point your build at what it writes:

```
goblinscript --dump-bundle bundle          # from a release exe, into ./bundle
cargo run --release -- --bundle bundle VIDEO
```

Those are the same bytes the release drafts with, so your build now drafts
identically. To bake them in instead -- your own standalone exe, or a whole
release zip beside `LICENSE` and the third-party notices:

```
cargo build --release --features embed     # reads ./bundle at compile time
cargo xtask dist                           # -> dist/goblinscript-VER-dml.zip on Windows,
                                           #    dist/goblinscript-VER-linux-x64.zip on Linux
```

A Linux build wants `libasound2-dev` and `pkg-config` from the distribution;
everything else, the ONNX Runtime and its WebGPU provider included, is fetched
by cargo.

`music/` is empty in a checkout and the app runs silent; any General MIDI
`.mid` files dropped in there become the playlist.

The release zips are built by GitHub Actions from a version tag. The inputs a
checkout lacks sit on the permanent `bundle` release of this repository:
`bundle.zip` for Windows, `bundle-linux.zip` for Linux (the same model,
exported in the forms the WebGPU provider has kernels for), and `music.zip`.
`bundle.sha256` pins the model a version embeds, and the workflow refuses any
other.

`DEVELOPMENT.md` is the map: what each stage does, why it is shaped that way,
and what is measured to keep it honest.

## License

MIT -- see `LICENSE`. Third-party components (ONNX Runtime, DirectML, the model
weights, the Rust crate tree) are attributed in `THIRD-PARTY-NOTICES.txt`.
