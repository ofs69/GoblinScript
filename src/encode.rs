//! The expensive pass: frames -> PCA latents -> the int8 cache.
//!
//! This is the one stage worth caching, so it streams straight to disk and
//! never holds the clip in RAM. It is the stage that dominates a draft: at
//! ~7.7x realtime it costs ~7.8 s per minute of video, against 1-2 s for the
//! libx264 normalize pass ahead of it.
//!
//! The GPU forward is 4/5 of that and every lever on it is measured.
//! What is left is the CPU work wrapped around a blocking
//! `Session::run`, so everything here that touches a frame or a row is written
//! to move or reuse memory rather than allocate it: the frame pool, the input
//! staging buffer and the row slab are each allocated once for the whole clip,
//! the viewport's forward runs on its own thread, and the ffmpeg decode runs
//! on another, one window ahead of the GPU.
//!
//! The grid is the contract, and it is subtle. EVERY ROW IS OWNED BY A FIRST-
//! FRAME INDEX `a`, and its tubelet is the frame pair `(a, a+k)`. Alignment `j`
//! covers `a = j, j+2k, j+4k, ...`; each is one encoder forward over a
//! contiguous decimation of the video, and interleaving all `2k` of them
//! restores one row per frame. Get this wrong and every downstream time is off
//! by a frame, silently.
//!
//! Extraction is CUT-BLIND on purpose: windows sit on a fixed frame grid and
//! never consult the boundaries. Cuts reach the model only through the cut-flag
//! channel, which is what it was trained on.

use anyhow::{Context, Result};
use indicatif::ProgressBar;
use ort::session::Session;
use std::collections::VecDeque;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use crate::bundle::{ort_err, Manifest};
use crate::heads::Heads;
use crate::viz::Viewport;

/// The half-written cache, which is the cache's ABSENCE until the rename.
///
/// A draft that stops in this stage -- Ctrl-C, a broken pipe, a window that
/// failed -- leaves the cache directory behind on purpose, so the next run
/// resumes into it. What it must not leave behind is the partial file, which is
/// the full size of the finished one (~1 MB per second of video) and which
/// nothing would clear until the next SUCCESSFUL draft of the same clip. After
/// the rename the path is gone and this is a no-op.
struct PartFile(std::path::PathBuf);

impl Drop for PartFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

pub struct Latents {
    pub path: std::path::PathBuf,
    pub rows: usize,
    pub row_bytes: usize,
}

/// One window of `group + k` frames -> `group` latent rows, in video order,
/// written into `slab` at `row_bytes` apiece.
///
/// The window size is FIXED: V-JEPA 2.1's rotary embedding guards on its token
/// count (`576 * (T/2) == 9216`), so the graph is specialized to `clip_len`
/// frames and a short window is not a thing it can be asked for.
///
/// `x` is the caller's staging buffer. It is 13.5 MB at the shipped config and
/// it is refilled once per alignment, so owning it here would mean allocating
/// -- and first-touching, which on Windows is a page fault per 4 KB -- that
/// much four times per window for the whole clip.
pub(crate) fn encode_window(
    sess: &mut Session,
    man: &Manifest,
    win: &[&[u8]],
    scale: f32,
    row_bytes: usize,
    x: &mut Vec<u8>,
    slab: &mut [i8],
) -> Result<()> {
    let (res, k) = (man.enc_res, man.tubelet_stride);
    let group = (man.clip_len / 2) * 2 * k;

    for j in 0..man.alignments {
        let aa: Vec<usize> = (j..group).step_by(2 * k).take(man.clip_len / 2).collect();
        x.clear();
        for &a in &aa {
            x.extend_from_slice(win[a]);
            x.extend_from_slice(win[a + k]);
        }
        let shape = vec![1i64, (aa.len() * 2) as i64, res as i64, res as i64, 3];
        let t = ort::value::TensorRef::from_array_view((shape, &x[..])).map_err(ort_err)?;
        let o = sess.run(ort::inputs!["frames" => t]).map_err(ort_err)?;
        let (_s, lat) = o["lat"].try_extract_tensor::<f32>().map_err(ort_err)?;
        for (i, &a) in aa.iter().enumerate() {
            let src = &lat[i * row_bytes..(i + 1) * row_bytes];
            let dst = &mut slab[a * row_bytes..(a + 1) * row_bytes];
            // the int8 step (8/127) is what the model trained against: quantize
            // here, not "for storage" -- it is part of the input
            for (d, &v) in dst.iter_mut().zip(src) {
                *d = (v / scale).round().clamp(-127.0, 127.0) as i8;
            }
        }
    }
    Ok(())
}

/// The first window must contain actual signal.
///
/// A GPU backend can load this graph, run it, and return NaN for every value --
/// DirectML did exactly that on an fp16 attention overflow, and NaN quantizes to
/// int8 zero, so the cache filled with zeros, the head saw nothing, and the draft
/// came out flat. Nothing anywhere raised. Whitened latents are ~unit variance by
/// construction, so a first window with no variance is not a quiet video, it is a
/// broken encoder -- and it costs one check to say so before spending eight
/// minutes producing garbage.
fn assert_signal(slab: &[i8]) -> Result<()> {
    let n = slab.len();
    let nz = slab.iter().filter(|&&v| v != 0).count();
    if n > 0 && nz * 100 < n {
        anyhow::bail!(
            "the encoder returned an empty first window ({:.1}% non-zero of \
             {n} values). The graph ran but produced no signal -- this is the \
             GPU backend miscomputing it, not the video. Try --bench to \
             confirm, and file it: a draft from this would be flat.",
            100.0 * nz as f64 / n as f64,
        );
    }
    Ok(())
}

fn write_rows(out: &mut BufWriter<File>, rows: &[i8]) -> Result<()> {
    let bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(rows.as_ptr() as *const u8, rows.len()) };
    out.write_all(bytes)?;
    Ok(())
}

/// Encode `video` into an int8 latent cache at `dst`, feeding every row to the
/// head as it is made.
///
/// The head is `heads`, and it runs INSIDE this stage: a window's rows go to
/// disk and to the model in the same breath, so the tracks are finished when
/// the encoding is and the strokes are on screen while it runs. It costs the
/// head's own forwards, which are minutes of nothing against the encoder's
/// hours, and it buys the whole of the stage's watchability.
///
/// `mask` is the viewport's graph, and it is `None` whenever there is nobody to
/// draw for (a piped run) or the bundle predates it. It feeds `viz` and nothing
/// else -- the cache this writes is byte-identical either way.
///
/// `plan` is the auto-crop segment plan (`autocrop::Plan::segments`): the
/// decode runs through `SegmentedDecoder`, one pipe per rect, switched on the
/// absolute frame grid. `None` is the identity plan -- one uncropped pipe,
/// the decode this stage always ran.
#[allow(clippy::too_many_arguments)]
pub fn encode(
    video: &Path,
    sess: &mut Session,
    mask: Option<Session>,
    man: &Manifest,
    dst: &Path,
    n_frames_hint: u64,
    max_s: Option<f64>,
    plan: Option<Vec<(usize, Option<String>)>>,
    pb: &ProgressBar,
    heads: &mut Heads,
) -> Result<Latents> {
    let (res, k) = (man.enc_res, man.tubelet_stride);
    let group = (man.clip_len / 2) * 2 * k;
    let row_bytes = man.dim * man.grid * man.grid;
    let scale = man.int8_scale;
    // The alignments have to TILE the window: alignment `j` owns first-frame
    // indices `j, j+2k, ...`, so it takes exactly `2k` of them to give every
    // frame in `0..group` a row, which is the "one row per frame" the whole
    // grid is built on. Anything else leaves rows unwritten (or writes them
    // twice), and the manifest is where that would come from.
    anyhow::ensure!(
        man.alignments == 2 * k,
        "bundle grid is inconsistent: {} alignments at tubelet stride {k} cannot \
         tile a window (expected {})",
        man.alignments,
        2 * k
    );

    let part = PartFile(dst.with_extension("part"));
    let mut out = BufWriter::with_capacity(1 << 22, File::create(&part.0)?);
    let segs = plan.unwrap_or_else(|| vec![(0, None)]);
    let dec =
        crate::ffmpeg::SegmentedDecoder::open(video, res, res, man.grid_fps, max_s, segs)?;
    let view = Viewport::spawn(mask, man.dim, man.grid);

    let frame_bytes = res * res * 3;
    let mut x: Vec<u8> = Vec::with_capacity((man.clip_len / 2) * 2 * frame_bytes);
    let mut slab: Vec<i8> = vec![0; group * row_bytes];

    let mut buf: Vec<Vec<u8>> = Vec::new(); // frames not yet turned into rows
    let mut recent: VecDeque<Vec<u8>> = VecDeque::new(); // the frames before those
    let mut rows_written = 0usize;
    let mut n_frames = 0usize;

    // The decode runs on its own thread, up to one window ahead of the GPU:
    // the pipe read that used to sit between two `Session::run`s fills the
    // next window while the current one encodes. The frames travel in decode
    // order into the same `buf`, so the tensors are identical in identical
    // order and the cache is byte-for-byte the serial loop's. The channel is
    // bounded at one window of frames (~57 MB at the shipped config), and
    // frames are recycled, never freed -- a fresh `Vec` costs an allocation
    // AND a 442 KB zero-fill per frame -- by sending the ones that fall out
    // of `recent` back over the pool channel, where `next_frame`'s resize on
    // a right-sized buffer is a no-op. Both channels live inside the scope
    // closure so an early error return drops the receiver, which unblocks
    // and ends the decoder thread before the scope joins it.
    std::thread::scope(|sc| -> Result<()> {
        let (tx_f, rx_f) = std::sync::mpsc::sync_channel::<Result<Vec<u8>>>(group + k);
        let (tx_pool, rx_pool) = std::sync::mpsc::channel::<Vec<u8>>();
        sc.spawn(move || {
            let mut dec = dec;
            loop {
                let mut f = rx_pool.try_recv().unwrap_or_default();
                match dec.next_frame(&mut f) {
                    Ok(true) => {
                        if tx_f.send(Ok(f)).is_err() {
                            break; // the encode side is gone; so is the need
                        }
                    }
                    Ok(false) => break,
                    Err(e) => {
                        let _ = tx_f.send(Err(e));
                        break;
                    }
                }
            }
        });
        for got in rx_f {
            n_frames += 1;
            buf.push(got?);

            if buf.len() >= group + k {
                // one window is the grain the user can stop at: the `.part`
                // file is never the cache until the rename, so unwinding here
                // loses only the seconds since the last window
                crate::cancel::check()?;
                let win: Vec<&[u8]> = buf[..group + k].iter().map(|f| f.as_slice()).collect();
                encode_window(sess, man, &win, scale, row_bytes, &mut x, &mut slab)?;
                if rows_written == 0 {
                    assert_signal(&slab)?;
                }
                // the window's middle row and the frame it was computed from,
                // so the two panels are the same instant of film
                view.publish(&slab[(group / 2) * row_bytes..(group / 2 + 1) * row_bytes]);
                crate::viz::publish_frame(win[group / 2], res, res, man.grid);
                write_rows(&mut out, &slab)?;
                heads.push(&slab)?;
                rows_written += group;
                // the frames this window consumed ARE the history the tail
                // needs, so they move into `recent` rather than being cloned
                // into it as they arrive; what falls off the far end goes
                // back to the decoder's pool
                recent.extend(buf.drain(..group));
                while recent.len() > group + k {
                    if let Some(f) = recent.pop_front() {
                        let _ = tx_pool.send(f);
                    }
                }
                pb.set_position(rows_written as u64);
            }
        }
        Ok(())
    })?;

    // The tail. Every frame owns a row, so the frames after the last full
    // window still need theirs -- but the graph only takes a full window.
    // Python encodes them in a correspondingly SHORT window; the fixed-shape
    // graph cannot, so instead the window slides BACK over the clip's last
    // frames and only the still-missing rows are kept.
    //
    // Every one of those rows keeps its own tubelet `(a, a+k)`, which is what
    // fixes its meaning; what differs from Python is the window CONTEXT around
    // it -- `group` real frames here instead of a short window. That touches at
    // most the final ~2 s of a clip, and it uses more real footage than the
    // Python tail does, not less.
    //
    // A window covers `group` rows and up to `group + k - 1` can be owed (the
    // loop leaves `k` frames behind and takes another window at `group + k`),
    // so this is a LOOP: one pass for all but the last `group`, one for the
    // rest. At `k = 1` the second pass never runs.
    if n_frames > 0 {
        // the clip's last frames, oldest first, and the video index of tail[0]
        let tail: Vec<Vec<u8>> = recent.into_iter().chain(buf).collect();
        let base = n_frames - tail.len();
        while rows_written < n_frames {
            // slide back only as far as the still-missing rows need: a window
            // whose first frame is `start` writes rows `start..start+group`
            let start = rows_written.min(n_frames.saturating_sub(group));
            let win: Vec<&[u8]> = (0..group + k)
                .map(|i| {
                    // past the end of the clip the last frame repeats, so the
                    // final rows' tubelets are STATIC pairs (exact pose, zero
                    // observed motion) -- the convention extraction closes a
                    // clip with. The clamp can only bite beyond `n_frames`.
                    tail[(start + i - base).min(tail.len() - 1)].as_slice()
                })
                .collect();
            encode_window(sess, man, &win, scale, row_bytes, &mut x, &mut slab)?;
            if rows_written == 0 {
                assert_signal(&slab)?;
            }
            let keep = rows_written - start;
            let take = (start + group).min(n_frames) - rows_written;
            let rows = &slab[keep * row_bytes..(keep + take) * row_bytes];
            write_rows(&mut out, rows)?;
            heads.push(rows)?;
            rows_written += take;
            pb.set_position(rows_written as u64);
        }
    }

    out.flush()?;
    drop(out);
    // A decode that opens the clip and delivers nothing looks, on disk, like a
    // cache of a video with no frames in it -- zero bytes, zero rows, internally
    // consistent, and valid forever. It is a broken source or a broken decode,
    // and it says so here instead of becoming the answer to every later run.
    anyhow::ensure!(
        rows_written > 0,
        "the decode delivered no frames at all, so there is nothing to encode -- \
         the source (or the normalized copy of it) cannot be read"
    );
    std::fs::rename(&part.0, dst).context("could not finalize the latent cache")?;
    pb.set_position(n_frames_hint.max(rows_written as u64));

    Ok(Latents {
        path: dst.to_path_buf(),
        rows: rows_written,
        row_bytes,
    })
}

#[cfg(test)]
mod tests {
    /// The tail bookkeeping, without a GPU: how many rows each pass writes and
    /// which frames it reads. `group + k - 1` rows can be owed at end of
    /// stream, which is one more than a window produces -- the case that used
    /// to underflow `group - missing`.
    fn tail_passes(n: usize, group: usize, k: usize) -> Vec<(usize, usize, usize)> {
        let mut rows_written = 0;
        let mut buf = 0;
        for _ in 0..n {
            buf += 1;
            if buf >= group + k {
                rows_written += group;
                buf -= group;
            }
        }
        let mut passes = Vec::new();
        while rows_written < n {
            let start = rows_written.min(n.saturating_sub(group));
            let take = (start + group).min(n) - rows_written;
            // (first frame of the window, first row kept, rows kept)
            passes.push((start, rows_written - start, take));
            rows_written += take;
            assert!(passes.len() <= 4, "tail did not converge for n={n}");
        }
        assert_eq!(rows_written, n, "n={n}: every frame owns a row");
        passes
    }

    #[test]
    fn every_clip_length_gets_exactly_one_row_per_frame() {
        for k in [1usize, 2, 4] {
            let group = 16 * k;
            for n in 1..=(6 * group) {
                for (start, keep, take) in tail_passes(n, group, k) {
                    assert!(keep + take <= group, "n={n} k={k}: kept past the window");
                    assert!(start + take <= n, "n={n} k={k}: kept a padding row");
                    assert!(take > 0, "n={n} k={k}: a pass that writes nothing");
                }
            }
        }
    }

    // The shipped grid: 64-frame windows at stride 2. A clip of 64W+65 frames
    // owes 65 rows against a 64-row window, which is the two-pass case: the
    // first pass writes what it can from where the rows are owed, the second
    // slides to the clip's end for the one row left over.
    #[test]
    fn one_row_over_a_window_takes_two_passes() {
        assert_eq!(tail_passes(65, 64, 2), vec![(0, 0, 64), (1, 63, 1)]);
        assert_eq!(tail_passes(129, 64, 2), vec![(64, 0, 64), (65, 63, 1)]);
    }

    // Anything that fits in one window is the ORIGINAL single-pass behavior,
    // unchanged: the window is the clip's last `group` frames and the rows kept
    // are its last `missing`.
    #[test]
    fn a_tail_within_one_window_is_the_single_slide_back() {
        for (n, missing) in [(100usize, 36usize), (66, 2), (127, 63), (128, 64)] {
            assert_eq!(
                tail_passes(n, 64, 2),
                vec![(n - 64, 64 - missing, missing)],
                "n={n}"
            );
        }
    }

    // A clip shorter than one window is all tail: its rows are the FIRST `n` of
    // the window, since the padding frames sit after the real ones.
    #[test]
    fn a_clip_shorter_than_a_window_keeps_its_real_rows() {
        assert_eq!(tail_passes(10, 64, 2), vec![(0, 0, 10)]);
        assert_eq!(tail_passes(1, 64, 2), vec![(0, 0, 1)]);
    }
}
