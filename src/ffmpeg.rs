//! Everything that touches the video file: probe, transcode, decode.
//!
//! Two rules the rest of the program depends on:
//!
//! * **The transcode is not an optimization.** Every clip the model has ever
//!   seen -- the training corpus and every draft to date -- was first
//!   re-encoded to 480p/30fps/crf-23. A pristine 4K source fed straight to the
//!   encoder is off-distribution in a way nobody has measured, so GoblinScript
//!   normalizes first and reads only from the normalized copy.
//! * **Time is absolute.** Frame `i` of the decode grid sits at
//!   `i / grid_fps` s of the video, always (and latent row `j` at
//!   `Manifest::row_ms(j)`, its tubelet pair's midpoint). Nothing downstream
//!   re-bases either clock.

use anyhow::{bail, Context, Result};

/// How far before a segment's first frame the input seek aims. Small enough
/// to stay far inside the frame it is reaching for, large enough to survive
/// the millisecond formatting `-ss` takes.
const SEEK_GUARD_S: f64 = 0.001;

/// Which frames this decode chain hands the encoder, as a number the latent
/// cache can compare. Everything else in the cache key describes the INPUT
/// (source, crop plan, normalize, exposure gamma); this describes the
/// reader. Bump it whenever a change here would give one of those inputs
/// different pixels, so caches written by the older reader re-encode instead
/// of feeding a fixed decode someone else's frames.
pub const DECODE_REV: u32 = 1;
use std::io::{BufRead, BufReader, Read};
use std::path::Path;
use std::process::{Child, Command, Stdio};

use crate::bundle::Transcode;

pub fn have_tools() -> Result<()> {
    for tool in ["ffmpeg", "ffprobe"] {
        Command::new(tool)
            .arg("-version")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .with_context(|| {
                format!(
                    "{tool} is not on PATH -- the goblins need ffmpeg. Install it with:\n  \
                     winget install --id Gyan.FFmpeg\nthen open a new terminal and try again"
                )
            })?;
    }
    Ok(())
}

/// ffmpeg's version as a short string ("7.1"), for the startup report. Purely
/// informational: `None` on anything unexpected, and no caller treats that as
/// an error -- `have_tools` is what decides whether ffmpeg is usable.
pub fn version() -> Option<String> {
    let out = Command::new("ffmpeg").arg("-version").output().ok()?;
    let first = String::from_utf8_lossy(&out.stdout);
    // "ffmpeg version 7.1-full_build-www.gyan.dev Copyright (c) ..." on a
    // release build, "ffmpeg version 2025-06-11-git-abc123 ..." on a git one
    let v = first.lines().next()?.split_whitespace().nth(2)?;
    // A dated git build carries its whole date, which is the only version
    // information it has; a release build drops the distributor's suffix
    // ("7.1-full_build" -> "7.1", "n6.0" -> "n6.0").
    let is_dated = v.len() >= 10
        && v.as_bytes()[..10]
            .iter()
            .enumerate()
            .all(|(i, c)| if i == 4 || i == 7 { *c == b'-' } else { c.is_ascii_digit() });
    Some(if is_dated {
        v[..10].to_string()
    } else {
        v.split('-').next().unwrap_or(v).to_string()
    })
}

/// Coded width/height of the first video stream, via ffprobe. The normalized
/// clip this runs on is square-pixel by construction (the transcode's `scale`
/// writes no SAR), so these dims ARE the displayed aspect.
pub fn dims(path: &Path) -> Result<(usize, usize)> {
    let out = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "json",
        ])
        .arg(path)
        .output()
        .context("ffprobe failed to run")?;
    if !out.status.success() {
        bail!("ffprobe could not read the video's dimensions");
    }
    first_dims(&String::from_utf8_lossy(&out.stdout))
        .context("ffprobe reported no video dimensions")
}

/// The first stream in an ffprobe report that actually has a picture size.
///
/// One video stream does not mean one entry. An MP4 that carries a timecode
/// track ties it to the video with a track reference, and an ffprobe new
/// enough to read those as a stream group reports the same video TWICE -- so
/// the answer is the first entry with dimensions, not the only one. Streams
/// that carry no picture (a timecode or data member listed alongside) have no
/// width at all and are skipped for the same reason.
fn first_dims(report: &str) -> Option<(usize, usize)> {
    let v: serde_json::Value = serde_json::from_str(report).ok()?;
    v["streams"].as_array()?.iter().find_map(|s| {
        let w = s["width"].as_u64()? as usize;
        let h = s["height"].as_u64()? as usize;
        (w > 0 && h > 0).then_some((w, h))
    })
}

/// Duration in milliseconds, via ffprobe.
pub fn duration_ms(path: &Path) -> Result<f64> {
    let out = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
        ])
        .arg(path)
        .output()
        .context("ffprobe failed to run")?;
    if !out.status.success() {
        bail!("ffprobe could not read the video");
    }
    let s = String::from_utf8_lossy(&out.stdout);
    let secs: f64 = s
        .trim()
        .parse()
        .with_context(|| format!("ffprobe returned no duration ({:?})", s.trim()))?;
    Ok(secs * 1000.0)
}

/// Can a browser's `<video>` element play this file as-is? Decides whether the
/// review page streams the ORIGINAL (full quality) or the cache's normalized
/// copy (H.264+AAC MP4 -- plays everywhere, but 480p). Anything that stops the
/// probe -- ffprobe failing, unparseable output -- answers `false`: the
/// fallback always exists, an error page never has to.
pub fn browser_playable(path: &Path) -> bool {
    let out = match Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-show_entries",
            "format=format_name",
            "-show_entries",
            "stream=codec_type,codec_name",
            "-of",
            "json",
        ])
        .arg(path)
        .output()
    {
        Ok(o) if o.status.success() => o,
        _ => return false,
    };
    let v: serde_json::Value = match serde_json::from_slice(&out.stdout) {
        Ok(v) => v,
        Err(_) => return false,
    };
    let container = v["format"]["format_name"].as_str().unwrap_or("");
    let codec_of = |kind: &str| -> Option<String> {
        v["streams"].as_array()?.iter().find_map(|s| {
            (s["codec_type"].as_str() == Some(kind))
                .then(|| s["codec_name"].as_str().unwrap_or("").to_string())
        })
    };
    let vcodec = match codec_of("video") {
        Some(c) => c,
        None => return false,
    };
    playable(container, &vcodec, codec_of("audio").as_deref())
}

/// The conservative every-browser whitelist. Notably OUT: HEVC (hardware- and
/// browser-dependent) and H.264 inside matroska (Chrome demuxes it, Firefox
/// refuses). A miss only costs quality -- the review falls back to the
/// normalized copy.
fn playable(container: &str, vcodec: &str, acodec: Option<&str>) -> bool {
    // ffprobe reports the demuxer's alias list: "mov,mp4,m4a,3gp,3g2,mj2" for
    // every MP4-family file, and "matroska,webm" for BOTH .mkv and .webm --
    // the container can't tell them apart, the codec subset can: a matroska
    // file restricted to the WebM codecs IS a WebM to a browser.
    let c: Vec<&str> = container.split(',').collect();
    if c.contains(&"mp4") {
        matches!(vcodec, "h264" | "vp9" | "av1")
            && match acodec {
                None => true,
                Some(a) => matches!(a, "aac" | "mp3" | "opus" | "flac"),
            }
    } else if c.contains(&"webm") {
        matches!(vcodec, "vp8" | "vp9" | "av1")
            && match acodec {
                None => true,
                Some(a) => matches!(a, "opus" | "vorbis"),
            }
    } else {
        false
    }
}

/// Normalize a source video into the encode spec the model was trained on.
///
/// With a `vr` config the same pass ALSO flattens the source: `crop` the eye,
/// `v360` it to a flat viewport at the saved aim, and the scale/fps chain below
/// takes it from there. One pass, no intermediate render -- and because the aim
/// is static, v360 builds its projection LUT once instead of per frame, so this
/// costs about what a flat transcode of the same footage costs. A trimmed range
/// becomes an input seek, which is frame-accurate under re-encode.
///
/// Calls `on_progress` with a 0..1 fraction as ffmpeg reports it; `total_ms`
/// is the output timeline's length -- for a trimmed VR source that is the
/// RANGE, not the whole video.
pub fn transcode(
    src: &Path,
    dst: &Path,
    spec: &Transcode,
    total_ms: f64,
    vr: Option<&crate::vr::Config>,
    mut on_progress: impl FnMut(f64),
) -> Result<()> {
    let part = dst.with_extension("part.mp4");
    // Hardware decode is attempted only for VR, where the source is typically
    // 6-8K HEVC and software decode is brutal -- and it is only ever an
    // ATTEMPT. Some driver/codec pairs accept `-hwaccel auto` and then fail
    // mid-stream, so a failure retries in software before it becomes an error
    // (the same fallback `vr_project.py`'s renderer runs for the same reason).
    // The flat path never asked for it and does not start.
    let try_hw = vr.is_some();
    match run_transcode(src, &part, spec, total_ms, vr, try_hw, &mut on_progress) {
        Ok(()) => {}
        Err(e) if try_hw && !crate::cancel::is_cancel(&e) => {
            run_transcode(src, &part, spec, total_ms, vr, false, &mut on_progress)
                .context("the source would not decode, with or without the graphics card")?;
        }
        Err(e) => return Err(e),
    }
    // rename last: a half-written file must never look like a finished one to
    // the next run's cache check
    std::fs::rename(&part, dst).context("could not finalize the transcode")?;
    on_progress(1.0);
    Ok(())
}

/// One ffmpeg attempt, writing `part`. Split out so a hardware-decode failure
/// can be retried in software without duplicating the command.
fn run_transcode(
    src: &Path,
    part: &Path,
    spec: &Transcode,
    total_ms: f64,
    vr: Option<&crate::vr::Config>,
    hw: bool,
    on_progress: &mut impl FnMut(f64),
) -> Result<()> {
    let mut vf = String::new();
    if let Some(v) = vr {
        vf.push_str(&v.filter_prefix(spec.height));
        vf.push(',');
    }
    vf.push_str(&format!(
        "scale=-2:{}:flags=bicubic,fps={}",
        spec.height, spec.fps
    ));
    let gop = (spec.fps * 2.0).round().max(1.0) as i64;

    let mut cmd = Command::new("ffmpeg");
    cmd.args(["-y", "-nostdin", "-loglevel", "error"]);
    // Input options, before -i: a VR range trim seeks the SOURCE (frame-exact
    // under re-encode), and hardware decode has to be asked for before the
    // input it applies to. Both fall away entirely on the flat path.
    if let Some(v) = vr {
        let dur = duration_ms(src).unwrap_or(0.0);
        let (t0, t1) = v.range(dur);
        if t0 > 0.0 || t1 < dur {
            cmd.args(["-ss", &format!("{:.3}", t0 / 1000.0)]);
            cmd.args(["-t", &format!("{:.3}", (t1 - t0) / 1000.0)]);
        }
        if hw {
            cmd.args(["-hwaccel", "auto"]);
        }
    }
    cmd.arg("-i")
        .arg(src)
        .args(["-map", "0:v:0", "-map", "0:a:0?"]);
    cmd.args(["-vf", &vf])
        .args(["-c:v", "libx264", "-crf", &spec.crf.to_string()])
        .args(["-preset", &spec.preset])
        .args(["-pix_fmt", "yuv420p"])
        .args(["-g", &gop.to_string(), "-keyint_min", &gop.to_string()])
        .args(["-x264-params", "scenecut=0:open-gop=0"]);
    cmd.args(["-c:a", "aac", "-b:a", "128k"]);
    // No timecode track in the normalize. A source that carries one hands its
    // timecode to the muxer as metadata, which writes a tmcd track next to the
    // video the explicit maps above already chose -- paperwork the normalize
    // has no use for, on the one file every later stage reads.
    cmd.args(["-write_tmcd", "0"]);
    let mut child = cmd
        .args(["-progress", "pipe:1", "-nostats"])
        .arg(part)
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .context("could not start ffmpeg")?;

    if let Some(stdout) = child.stdout.take() {
        for line in BufReader::new(stdout).lines().map_while(Result::ok) {
            // ffmpeg reports every few hundred ms, which makes this the transcode's
            // stop point. The half-written file is a `.part` nobody reads.
            if crate::cancel::requested() {
                let _ = child.kill();
                let _ = child.wait();
                let _ = std::fs::remove_file(part);
                crate::cancel::check()?;
            }
            if let Some(us) = line.strip_prefix("out_time_us=") {
                if let Ok(us) = us.trim().parse::<f64>() {
                    if total_ms > 0.0 {
                        on_progress((us / 1000.0 / total_ms).clamp(0.0, 1.0));
                    }
                }
            }
        }
    }
    let st = child.wait()?;
    if !st.success() {
        let _ = std::fs::remove_file(part);
        bail!("ffmpeg failed to transcode the source (exit {st})");
    }
    Ok(())
}

/// Raw rgb24 frames off an ffmpeg pipe, on the deploy frame grid.
pub struct Decoder {
    child: Child,
    stdout: BufReader<std::process::ChildStdout>,
    frame_bytes: usize,
    index: usize,
}

impl Decoder {
    pub fn open(
        path: &Path,
        width: usize,
        height: usize,
        fps: f64,
        max_s: Option<f64>,
    ) -> Result<Self> {
        Self::open_at(path, width, height, fps, max_s, 0, None)
    }

    /// `Decoder::open` from an absolute FRAME index, with an optional filter
    /// stage between the fps gate and the squash scale -- the auto-crop's
    /// `crop=W:H:X:Y` (source pixels), or the uncropped path's softening
    /// `scale` down to the bundle's spec height when the one normalized file
    /// is taller than it.
    ///
    /// The seek target is `start_frame`'s own time less `SEEK_GUARD_S`, and
    /// both halves of that matter. An input seek rebases timestamps onto the
    /// target, so the `fps` gate downstream resamples from THAT origin: only
    /// a whole number of frames between the seeked origin and an unseeked
    /// pipe's keeps the two on one grid, which is what lets a segment plan
    /// restart mid-clip without moving the clock. The guard is what stops the
    /// seek from overshooting `start_frame` on millisecond formatting, and it
    /// is a millisecond rather than the half frame it used to be because half
    /// a frame IS the gate's rounding boundary -- exact on a source whose
    /// video starts at zero, one frame off on one that declares a start time,
    /// an edit list, or a video track muxed behind its audio.
    pub fn open_at(
        path: &Path,
        width: usize,
        height: usize,
        fps: f64,
        max_s: Option<f64>,
        start_frame: usize,
        filter: Option<&str>,
    ) -> Result<Self> {
        let mut cmd = Command::new("ffmpeg");
        cmd.args(["-v", "error", "-nostdin"]);
        if start_frame > 0 {
            let ss = start_frame as f64 / fps - SEEK_GUARD_S;
            cmd.args(["-ss", &format!("{ss:.3}")]);
        }
        cmd.arg("-i").arg(path);
        if let Some(t) = max_s {
            cmd.args(["-t", &format!("{t:.3}")]);
        }
        let vf = match filter {
            Some(f) => format!("fps={fps},{f},scale={width}:{height}"),
            None => format!("fps={fps},scale={width}:{height}"),
        };
        let mut child = cmd
            .args(["-vf", &vf])
            .args(["-pix_fmt", "rgb24", "-f", "rawvideo", "-"])
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .context("could not start ffmpeg to decode")?;
        let stdout = child.stdout.take().expect("piped");
        let frame_bytes = width * height * 3;
        Ok(Self {
            child,
            stdout: BufReader::with_capacity(frame_bytes * 4, stdout),
            frame_bytes,
            index: 0,
        })
    }

    /// The next frame, or `None` at end of stream. A short read is the end of
    /// the stream, never a partial frame.
    ///
    /// The pipe closing is how a decode ENDS and also how it DIES: a source
    /// that goes bad halfway through leaves ffmpeg exiting non-zero with frames
    /// still owed, and from this side that reads exactly like the last frame of
    /// a shorter clip. So the child is asked which of the two it was. It
    /// matters far past this call: a half-decoded clip that returns `false`
    /// here becomes a latent cache of half a video, internally consistent and
    /// indistinguishable from a complete one, and every later run of that clip
    /// reads it as the whole film.
    pub fn next_frame(&mut self, buf: &mut Vec<u8>) -> Result<bool> {
        buf.resize(self.frame_bytes, 0);
        match self.stdout.read_exact(buf) {
            Ok(()) => {
                self.index += 1;
                Ok(true)
            }
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                let st = self.child.wait().context("waiting on the decode")?;
                if !st.success() {
                    bail!(
                        "the decode failed {} frames in (ffmpeg exit {st}) -- the video \
                         goes bad partway through, and half of it is not a draft",
                        self.index
                    );
                }
                Ok(false)
            }
            Err(e) => Err(e).context("decode pipe broke"),
        }
    }
}

impl Drop for Decoder {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

/// Frames from a plan of `(start_frame, filter)` segments -- one ffmpeg pipe
/// per segment, switched at exact frame indices. This is how a per-shot
/// auto-crop reaches the encoder WITHOUT touching the clock: the crop lives in
/// each pipe's `-vf` chain (no re-encode, no remux -- a re-encode was measured
/// to land every render path on a different clock), every pipe opens on the
/// absolute frame grid via `open_at`, and timestamps are never read back, so
/// frame `i` sits at `i / fps` by construction. A pipe that under-delivers
/// before its segment boundary is a hard error, never a silent shift: every
/// later frame would be mis-indexed, which is exactly the drift this design
/// exists to make impossible.
pub struct SegmentedDecoder {
    path: std::path::PathBuf,
    width: usize,
    height: usize,
    fps: f64,
    max_s: Option<f64>,
    /// (first frame, filter) per segment, sorted, first at frame 0.
    segs: Vec<(usize, Option<String>)>,
    cur: Option<Decoder>,
    seg: usize,
    index: usize, // absolute frame index of the NEXT frame
}

impl SegmentedDecoder {
    pub fn open(
        path: &Path,
        width: usize,
        height: usize,
        fps: f64,
        max_s: Option<f64>,
        segs: Vec<(usize, Option<String>)>,
    ) -> Result<Self> {
        anyhow::ensure!(
            segs.first().map(|s| s.0) == Some(0),
            "a segment plan must start at frame 0"
        );
        anyhow::ensure!(
            segs.windows(2).all(|w| w[0].0 < w[1].0),
            "segment plan is not strictly ordered"
        );
        Ok(Self {
            path: path.to_path_buf(),
            width,
            height,
            fps,
            max_s,
            segs,
            cur: None,
            seg: 0,
            index: 0,
        })
    }

    fn seg_end(&self) -> usize {
        self.segs.get(self.seg + 1).map(|s| s.0).unwrap_or(usize::MAX)
    }

    pub fn next_frame(&mut self, buf: &mut Vec<u8>) -> Result<bool> {
        if let Some(t) = self.max_s {
            // the same cutoff a single pipe's `-t` produces on the fps grid
            if self.index as f64 >= t * self.fps {
                return Ok(false);
            }
        }
        if self.index >= self.seg_end() {
            self.seg += 1;
            self.cur = None;
        }
        if self.cur.is_none() {
            let (start, filter) = &self.segs[self.seg];
            // each pipe keeps the run's own -t semantics, measured from 0
            let rem = self.max_s.map(|t| t - *start as f64 / self.fps);
            self.cur = Some(Decoder::open_at(
                &self.path,
                self.width,
                self.height,
                self.fps,
                rem,
                *start,
                filter.as_deref(),
            )?);
        }
        if self.cur.as_mut().expect("opened above").next_frame(buf)? {
            self.index += 1;
            return Ok(true);
        }
        // the pipe ended: at the last segment that IS the end of the clip, but
        // before a boundary it means every later frame would be mis-indexed
        anyhow::ensure!(
            self.seg + 1 >= self.segs.len(),
            "decode under-delivered: the pipe for segment {} ended at frame {} \
             before the next rect boundary at frame {} -- refusing to continue \
             on a shifted clock",
            self.seg,
            self.index,
            self.seg_end(),
        );
        Ok(false)
    }
}

#[cfg(test)]
mod tests {
    use super::{first_dims, playable, Decoder, SegmentedDecoder};
    use std::path::{Path, PathBuf};

    const MP4: &str = "mov,mp4,m4a,3gp,3g2,mj2";
    const MKV: &str = "matroska,webm";

    #[test]
    fn dims_read_the_one_video_stream() {
        let r = r#"{"streams":[{"width":1920,"height":1080}]}"#;
        assert_eq!(first_dims(r), Some((1920, 1080)));
    }

    /// THE timecode case: an MP4 whose video is tied to a tmcd track by a
    /// track reference is reported twice by ffprobe 9, and the answer is
    /// still the video's size rather than an error.
    #[test]
    fn dims_survive_a_video_reported_twice() {
        let r = r#"{"streams":[{"width":854,"height":480},{"width":854,"height":480}]}"#;
        assert_eq!(first_dims(r), Some((854, 480)));
    }

    /// A member with no picture is skipped rather than answered with.
    #[test]
    fn dims_skip_a_stream_without_a_picture() {
        let r = r#"{"streams":[{"codec_type":"data"},{"width":1920,"height":1080}]}"#;
        assert_eq!(first_dims(r), Some((1920, 1080)));
    }

    #[test]
    fn dims_of_a_report_with_no_picture_at_all_are_none() {
        assert_eq!(first_dims(r#"{"streams":[]}"#), None);
        assert_eq!(first_dims("not json"), None);
    }

    /// A short synthetic clip in the normalize encode spec (H.264, fixed
    /// GOP), built once under target/. Motion in every frame, so an
    /// off-by-one seam cannot pass as identical bytes.
    fn seam_clip() -> PathBuf {
        let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target/seam_test");
        std::fs::create_dir_all(&dir).unwrap();
        let clip = dir.join("testsrc.mp4");
        if !clip.exists() {
            let st = std::process::Command::new("ffmpeg")
                .args(["-v", "error", "-y", "-f", "lavfi", "-i"])
                .arg("testsrc2=size=854x480:rate=30:duration=12")
                .args(["-c:v", "libx264", "-crf", "23", "-pix_fmt", "yuv420p"])
                .args(["-g", "60", "-keyint_min", "60"])
                .args(["-x264-params", "scenecut=0:open-gop=0"])
                .arg(&clip)
                .status()
                .expect("ffmpeg runs");
            assert!(st.success(), "test clip encode failed");
        }
        clip
    }

    fn all_frames(dec: &mut impl FnMut(&mut Vec<u8>) -> anyhow::Result<bool>) -> Vec<Vec<u8>> {
        let mut frames = Vec::new();
        let mut buf = Vec::new();
        while dec(&mut buf).unwrap() {
            frames.push(buf.clone());
        }
        frames
    }

    /// THE drift check: a multi-segment identity plan must reproduce the
    /// single-pass decode byte for byte -- every pipe restart lands on its
    /// exact frame, including across a single-frame segment. This is what
    /// makes a per-shot crop unable to move the clock.
    #[test]
    #[ignore = "decodes with the real ffmpeg; run explicitly: cargo test -- --ignored seam"]
    fn seam_segmented_decode_is_byte_identical_to_single_pass() {
        let clip = seam_clip();
        let mut single = Decoder::open(&clip, 384, 384, 30.0, None).unwrap();
        let a = all_frames(&mut |b| single.next_frame(b));

        let segs = vec![(0, None), (100, None), (101, None), (250, None)];
        let mut seg = SegmentedDecoder::open(&clip, 384, 384, 30.0, None, segs).unwrap();
        let b = all_frames(&mut |b| seg.next_frame(b));

        assert_eq!(a.len(), b.len(), "segment plan changed the frame count");
        assert!(a.len() > 300, "test clip too short to stress the seams");
        for (i, (fa, fb)) in a.iter().zip(&b).enumerate() {
            assert_eq!(fa, fb, "frame {i} differs across a segment seam");
        }
    }

    /// A cropped segment keeps the count (the clock) while changing the
    /// pixels -- and only inside its own segment.
    #[test]
    #[ignore = "decodes with the real ffmpeg; run explicitly: cargo test -- --ignored seam"]
    fn seam_cropped_segment_changes_pixels_not_the_clock() {
        let clip = seam_clip();
        let mut single = Decoder::open(&clip, 384, 384, 30.0, None).unwrap();
        let a = all_frames(&mut |b| single.next_frame(b));

        let segs = vec![
            (0, None),
            (120, Some("crop=534:300:160:90".to_string())),
            (240, None),
        ];
        let mut seg = SegmentedDecoder::open(&clip, 384, 384, 30.0, None, segs).unwrap();
        let b = all_frames(&mut |b| seg.next_frame(b));

        assert_eq!(a.len(), b.len(), "a crop must never change the frame count");
        assert_eq!(a[0], b[0]);
        assert_eq!(a[119], b[119], "last frame before the crop segment");
        assert_ne!(a[120], b[120], "the cropped segment shows different pixels");
        assert_eq!(a[240], b[240], "first frame after the crop segment");
        assert_eq!(a[a.len() - 1], b[b.len() - 1]);
    }

    /// The seam invariant on containers that carry their OWN timeline. The
    /// clip above starts at zero with no edit list, which is the easy case;
    /// a real source can declare a start time, mux its video behind its
    /// audio, or hold a copy-seek's edit list, and the normalized copy
    /// adopts that timeline. Each pipe restart seeks into it, so this is
    /// where a segment plan would pick up a constant offset the single pass
    /// never sees -- and a per-shot crop would then move the clock after all.
    #[test]
    #[ignore = "decodes with the real ffmpeg; run explicitly: cargo test -- --ignored seam"]
    fn seam_holds_on_containers_that_declare_their_own_timeline() {
        let dir = clock_dir();
        let b = clock_base().to_str().unwrap().to_string();
        let startts = dir.join("seam_startts.mp4");
        ffmpeg_ok(&["-i", &b, "-c", "copy", "-output_ts_offset", "0.066",
                    startts.to_str().unwrap()]);
        let audio = dir.join("seam_audiolate.mp4");
        ffmpeg_ok(&["-itsoffset", "0.1", "-i", &b,
                    "-f", "lavfi", "-i", "sine=f=440:r=48000:d=8",
                    "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac",
                    audio.to_str().unwrap()]);
        let elst = dir.join("seam_elst.mp4");
        ffmpeg_ok(&["-ss", "0.066", "-i", &b, "-c", "copy", elst.to_str().unwrap()]);

        let mut bad = Vec::new();
        for (name, src) in [
            ("start_time", startts),
            ("audio_ahead", audio),
            ("edit_list", elst),
        ] {
            let mut single = Decoder::open(&src, 64, 64, 30.0, None).unwrap();
            let a = all_frames(&mut |x| single.next_frame(x));
            let segs = vec![(0, None), (37, None), (38, None), (120, None)];
            let mut seg = SegmentedDecoder::open(&src, 64, 64, 30.0, None, segs).unwrap();
            let c = all_frames(&mut |x| seg.next_frame(x));
            assert!(a.len() > 150, "{name}: decoded nothing to compare");
            if a.len() != c.len() {
                bad.push(format!("{name}: segment plan changed the frame count \
                                  ({} -> {})", a.len(), c.len()));
                continue;
            }
            if let Some(i) = (0..a.len()).find(|&i| a[i] != c[i]) {
                bad.push(format!("{name}: first differing frame is {i} \
                                  (segment starts are 0/37/38/120)"));
            }
        }
        assert!(bad.is_empty(), "a segment seam moved the clock:\n  {}", bad.join("\n  "));
    }

    // ---- the normalize clock -------------------------------------------------
    //
    // The funscript plays against the SOURCE, but every row it was written from
    // was decoded from the NORMALIZED copy, so the two have to hold the same
    // picture at the same time. Progressive drift is excluded by construction
    // (the `fps` filter maps by presentation time), but a CONSTANT offset can
    // ride in on container timing -- a video start_time, an mp4 edit list, an
    // audio track muxed ahead of the video. Training would absorb such an
    // offset in the per-clip lag fit; deploy has no lag stage and would ship it.
    // So these engineer the offsets deliberately and measure what comes out.

    fn clock_dir() -> PathBuf {
        let d = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("target/clock_test");
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn ffmpeg_ok(args: &[&str]) {
        let st = std::process::Command::new("ffmpeg")
            .args(["-v", "error", "-y"])
            .args(args)
            .status()
            .expect("ffmpeg runs");
        assert!(st.success(), "ffmpeg failed: {args:?}");
    }

    /// The base source: every frame a FLAT grey whose value is two
    /// incommensurate sines of the frame index. That makes the mean-luma
    /// series unique across the clip (the sines only realign past its end), so
    /// correlating two decodes of it has exactly one peak -- and it survives
    /// H.264 and a rescale, which a fine pattern would not.
    fn clock_base() -> PathBuf {
        let clip = clock_dir().join("base.mp4");
        if !clip.exists() {
            ffmpeg_ok(&[
                "-f", "lavfi", "-i", "color=c=black:s=320x180:r=30:d=8",
                "-vf", "geq=lum=128+60*sin(N/7)+50*sin(N/13):cb=128:cr=128",
                "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p",
                "-g", "30", "-keyint_min", "30", "-x264-params", "scenecut=0:open-gop=0",
                clip.to_str().unwrap(),
            ]);
        }
        clip
    }

    /// Per-frame mean luma on the DEPLOY grid -- the same decode path the
    /// encoder is fed from, so what this measures is what the model sees.
    fn luma_series(path: &Path) -> Vec<f64> {
        let mut dec = Decoder::open(path, 32, 32, 30.0, None).unwrap();
        let (mut out, mut buf) = (Vec::new(), Vec::new());
        while dec.next_frame(&mut buf).unwrap() {
            out.push(buf.iter().map(|&b| b as f64).sum::<f64>() / buf.len() as f64);
        }
        out
    }

    /// The whole-frame shift that best aligns `b` onto `a` (positive = `b`'s
    /// content sits LATER), with the correlation it reached.
    fn best_shift(a: &[f64], b: &[f64], max: i64) -> (i64, f64) {
        let mut best = (0i64, f64::NEG_INFINITY);
        for s in -max..=max {
            let (mut xs, mut ys) = (Vec::new(), Vec::new());
            for i in 0..a.len() as i64 {
                let j = i - s; // b[j] should hold what a[i] holds
                if j >= 0 && (j as usize) < b.len() {
                    xs.push(a[i as usize]);
                    ys.push(b[j as usize]);
                }
            }
            if xs.len() < 60 {
                continue;
            }
            let n = xs.len() as f64;
            let (mx, my) = (xs.iter().sum::<f64>() / n, ys.iter().sum::<f64>() / n);
            let mut num = 0.0;
            let (mut dx, mut dy) = (0.0, 0.0);
            for (x, y) in xs.iter().zip(&ys) {
                num += (x - mx) * (y - my);
                dx += (x - mx).powi(2);
                dy += (y - my).powi(2);
            }
            let r = if dx > 0.0 && dy > 0.0 { num / (dx * dy).sqrt() } else { 0.0 };
            if r > best.1 {
                best = (s, r);
            }
        }
        best
    }

    /// THE deploy clock check: normalizing a source must not move its content
    /// by even one frame, whatever the container did with its timestamps. Each
    /// case is a source built to carry the offset named.
    #[test]
    #[ignore = "transcodes with the real ffmpeg; run explicitly: cargo test -- --ignored clock"]
    fn clock_normalize_holds_the_source_timeline() {
        let base = clock_base();
        let dir = clock_dir();
        let b = base.to_str().unwrap().to_string();

        // a container start_time: the whole timeline pushed out by 66 ms, which
        // in mp4 lands as an edit list -- the shape 004866's normalized file has
        let startts = dir.join("startts.mp4");
        ffmpeg_ok(&["-i", &b, "-c", "copy", "-output_ts_offset", "0.066",
                    startts.to_str().unwrap()]);
        // video muxed 100 ms BEHIND its audio: the container starts at the
        // audio, so the video stream's own first frame is not at zero
        let audio = dir.join("audiolate.mp4");
        ffmpeg_ok(&["-itsoffset", "0.1", "-i", &b,
                    "-f", "lavfi", "-i", "sine=f=440:r=48000:d=8",
                    "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac",
                    audio.to_str().unwrap()]);
        // an edit list from a copy-seek: the mp4 says "start playing 66 ms in"
        let elst = dir.join("elst.mp4");
        ffmpeg_ok(&["-ss", "0.066", "-i", &b, "-c", "copy", elst.to_str().unwrap()]);
        // genuine VFR: one frame in ten dropped with its neighbours' timestamps
        // kept, so the timeline has holes the fps filter has to fill in place
        let vfr = dir.join("vfr.mp4");
        ffmpeg_ok(&["-i", &b, "-vf", "select=not(eq(mod(n\\,10)\\,3))",
                    "-fps_mode", "passthrough",
                    "-c:v", "libx264", "-crf", "16", "-pix_fmt", "yuv420p",
                    vfr.to_str().unwrap()]);

        let spec = crate::bundle::Transcode {
            height: 480,
            fps: 30.0,
            crf: 23,
            preset: "veryfast".to_string(),
        };
        let mut bad = Vec::new();
        for (name, src) in [
            ("plain", base.clone()),
            ("start_time", startts),
            ("audio_ahead", audio),
            ("edit_list", elst),
            ("vfr", vfr),
        ] {
            let norm = dir.join(format!("{name}_norm.mp4"));
            let _ = std::fs::remove_file(&norm);
            super::transcode(&src, &norm, &spec, 8000.0, None, |_| {}).unwrap();
            let (a, n) = (luma_series(&src), luma_series(&norm));
            assert!(a.len() > 120 && n.len() > 120, "{name}: decoded nothing to compare");
            let (shift, r) = best_shift(&a, &n, 10);
            println!("{name}: shift {shift} frames, corr {r:.3} ({} vs {} frames)",
                     a.len(), n.len());
            assert!(r > 0.9, "{name}: no alignment peak at all (corr {r:.3})");
            // Both sides read the container's OWN timeline, so a source that
            // declares a head (video muxed late) or a trim (an edit list) comes
            // out with that many frames on both sides. Equal counts are what
            // says the normalize pass adopted the source's timeline rather than
            // inventing one -- a shift of 0 between two differently-long
            // decodes would only mean they agree where they overlap.
            assert_eq!(
                a.len(), n.len(),
                "{name}: normalize changed the frame count ({} -> {})", a.len(), n.len()
            );
            if shift != 0 {
                bad.push(format!("{name}: normalize moved the content {shift} frames \
                                  ({:.0} ms, corr {r:.3})", shift as f64 / 30.0 * 1000.0));
            }
        }
        assert!(bad.is_empty(), "the normalize pass is not clock-neutral:\n  {}", bad.join("\n  "));
    }

    #[test]
    fn common_mp4_plays() {
        assert!(playable(MP4, "h264", Some("aac")));
        assert!(playable(MP4, "h264", None)); // no audio track is fine
        assert!(playable(MP4, "av1", Some("opus")));
    }

    #[test]
    fn webm_subset_plays_mkv_h264_does_not() {
        assert!(playable(MKV, "vp9", Some("opus")));
        // same bytes Chrome would demux -- but Firefox refuses h264/aac in
        // matroska, and the whitelist is every-browser or fallback
        assert!(!playable(MKV, "h264", Some("aac")));
        assert!(!playable(MKV, "vp9", Some("aac")));
    }

    #[test]
    fn browser_hostile_codecs_fall_back() {
        assert!(!playable(MP4, "hevc", Some("aac"))); // hardware-dependent
        assert!(!playable(MP4, "h264", Some("ac3"))); // silent video otherwise
        assert!(!playable("avi", "h264", Some("aac")));
        assert!(!playable("asf", "wmv3", Some("wmav2")));
    }
}
