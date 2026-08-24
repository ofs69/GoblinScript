//! Shot boundaries: TransNetV2 per-frame transition probability.
//!
//! TransNetV2 (a small dilated-3D-conv net) reads 27x48 frames and emits a
//! per-frame transition probability. The model reads cuts through a single
//! flag channel and the styler treats every shot as its own span, so cuts have
//! to be found before anything else runs. A run of `prob > thr` is one cut,
//! placed at the run's peak frame.
//!
//! Port of `boundaries.py`'s transnet detector -- same input, same windowing
//! (pad 25, window 100, step 50, keep the middle 50 of each window's output),
//! same threshold, same clock. Cuts need not be byte-identical to the Python
//! detector: the trunk is measured insensitive to the boundary source, so
//! near-parity is the target, not a bitwise match.

use anyhow::Result;
use indicatif::ProgressBar;
use ort::session::Session;
use std::path::Path;

use crate::bundle::{ort_err, Manifest, Transnet};
use crate::ffmpeg::Decoder;

/// Runs of `prob > thr` -> each run's peak frame -> cut times (ms), merged to
/// `min_gap_s` spacing. Frame `i`'s time is `i / fps` on the absolute clock.
fn detect_cuts(preds: &[f32], fps: f64, t: &Transnet) -> Vec<f64> {
    let min_gap = t.min_gap_s * 1000.0;
    let mut cuts: Vec<f64> = Vec::new();
    let mut i = 0usize;
    while i < preds.len() {
        if (preds[i] as f64) <= t.thr {
            i += 1;
            continue;
        }
        let mut j = i;
        let mut peak = i;
        while j < preds.len() && (preds[j] as f64) > t.thr {
            if preds[j] > preds[peak] {
                peak = j;
            }
            j += 1;
        }
        let time = peak as f64 / fps * 1000.0;
        match cuts.last() {
            Some(&last) if time - last < min_gap => {}
            _ => cuts.push(time),
        }
        i = j;
    }
    cuts
}

/// Decode the normalized video and find its cuts.
pub fn find_cuts(
    video: &Path,
    sess: &mut Session,
    man: &Manifest,
    n_frames_hint: u64,
    max_s: Option<f64>,
    pb: &ProgressBar,
) -> Result<Vec<f64>> {
    let t = &man.transnet;
    let (w, h) = (t.input_w, t.input_h);
    let fb = w * h * 3; // bytes per frame
    let mut dec = Decoder::open(video, w, h, man.grid_fps, max_s)?;

    // decode every frame at 27x48 into one contiguous buffer (a compilation is
    // hours long, but 27x48x3 per frame is ~4 KB -- ~1.3 GB for a 6 h clip)
    let mut all: Vec<u8> = Vec::new();
    let mut frame = Vec::new();
    let mut n = 0usize;
    while dec.next_frame(&mut frame)? {
        all.extend_from_slice(&frame);
        n += 1;
        if n.is_multiple_of(4096) {
            crate::cancel::check()?;
            pb.set_position(n as u64);
        }
    }
    // No frames is not a video with no cuts in it: the decoder handed back
    // nothing, and an empty list cached under this clip's name would answer
    // "one shot, no boundaries" for every run after it.
    anyhow::ensure!(n > 0, "the decode delivered no frames to look for cuts in");

    // TransNetV2 predict_frames windowing: 25 copies of the first frame lead,
    // enough copies of the last frame trail to make the length a multiple of
    // the step past the 25-frame tail; slide a 100-frame window by 50 and keep
    // rows [25..75] of each window's output.
    let (window, step, lead) = (t.window, t.step, 25usize);
    let tail = 25 + step - if n.is_multiple_of(step) { step } else { n % step };
    let padded = lead + n + tail;
    let frame_at = |pi: usize| -> &[u8] {
        let fi = if pi < lead {
            0
        } else if pi >= lead + n {
            n - 1
        } else {
            pi - lead
        };
        &all[fi * fb..(fi + 1) * fb]
    };

    let mut preds: Vec<f32> = Vec::with_capacity(n);
    let mut buf: Vec<u8> = vec![0u8; window * fb];
    let shape = vec![1i64, window as i64, h as i64, w as i64, 3];
    let mut ptr = 0usize;
    while ptr + window <= padded {
        crate::cancel::check()?;
        for k in 0..window {
            buf[k * fb..(k + 1) * fb].copy_from_slice(frame_at(ptr + k));
        }
        let x = ort::value::TensorRef::from_array_view((shape.clone(), &buf[..]))
            .map_err(ort_err)?;
        let out = sess.run(ort::inputs!["frames" => x]).map_err(ort_err)?;
        let (_s, prob) = out["prob"].try_extract_tensor::<f32>().map_err(ort_err)?;
        // keep the non-overlapping middle 50 rows (25..75)
        for &p in &prob[lead..lead + step] {
            if preds.len() < n {
                preds.push(p);
            }
        }
        ptr += step;
        pb.set_position((ptr.min(n)) as u64);
    }
    preds.truncate(n);
    pb.set_position(n_frames_hint);

    Ok(detect_cuts(&preds, man.grid_fps, t))
}

/// Cut times (ms) -> the latent rows they land on, ascending and deduplicated.
///
/// `row_ms` is the clip's row clock (`Manifest::row_ms`), passed in rather
/// than restated: a cut belongs to the first row whose time is at or after it,
/// which is the `searchsorted` convention `JepaClip` uses, and the two only
/// agree while both read the same clock.
///
/// The row count is deliberately NOT an argument. The head is forwarded chunk
/// by chunk while the encoder is still producing rows, so its cut-flag channel
/// has to be answerable for a row before the clip's length is known; a cut past
/// the end simply never gets asked about.
pub fn cut_rows(cuts_ms: &[f64], row_ms: impl Fn(usize) -> f64) -> Vec<usize> {
    let mut rows: Vec<usize> = Vec::new();
    for &c in cuts_ms {
        // first row whose time is >= the cut
        let (mut lo, mut hi) = (0usize, 1usize);
        while row_ms(hi) < c {
            hi *= 2;
        }
        while lo < hi {
            let mid = (lo + hi) / 2;
            if row_ms(mid) < c {
                lo = mid + 1;
            } else {
                hi = mid;
            }
        }
        if lo > 0 && rows.last() != Some(&lo) {
            rows.push(lo);
        }
    }
    rows
}

/// The model's cut-flag channel for rows `fs..e` of the timeline.
pub fn cut_flags(cut_rows: &[usize], fs: usize, e: usize) -> Vec<f32> {
    let mut cut = vec![0.0f32; e - fs];
    let first = cut_rows.partition_point(|&r| r < fs.max(1));
    for &r in &cut_rows[first..] {
        if r >= e {
            break;
        }
        cut[r - fs] = 1.0;
    }
    cut
}

/// Row edges of every shot on a clip of `n_rows`: `[0, ..cuts.., n_rows]`.
pub fn shot_edges(cuts_ms: &[f64], n_rows: usize, row_ms: impl Fn(usize) -> f64) -> Vec<usize> {
    let mut edges = vec![0usize];
    for r in cut_rows(cuts_ms, row_ms) {
        if r < n_rows && *edges.last().unwrap() != r {
            edges.push(r);
        }
    }
    edges.push(n_rows);
    edges
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The shipped grid's row clock: 30 fps frames, tubelet stride 2, so row
    /// `i` is the pair `(i, i+2)` and sits at frame `i+1`. Spelled out here
    /// rather than borrowed from a manifest so a change to either one has to
    /// be made deliberately in both places.
    fn shipped_clock() -> impl Fn(usize) -> f64 + Copy {
        |i: usize| (i as f64 + 1.0) / 30.0 * 1000.0
    }

    /// The flag channel is built per CHUNK now, from rows the encoder has not
    /// finished producing -- so it has to agree with the whole-clip edges it
    /// used to be cut from, on every window of the timeline.
    #[test]
    fn chunked_flags_match_the_whole_clip_edges() {
        let row = shipped_clock();
        let cuts = vec![0.0, 33.4, 1000.0, 1001.0, 4321.0, 9_000.0];
        let n = 300usize;
        let rows = cut_rows(&cuts, row);
        let edges = shot_edges(&cuts, n, row);
        // the whole-clip flag vector, the way it was built before
        let mut want = vec![0.0f32; n];
        for &e in &edges[1..edges.len() - 1] {
            want[e] = 1.0;
        }
        assert!(want.iter().any(|&f| f > 0.0), "the fixture has cuts in range");
        for &chunk in &[1usize, 7, 64, 300] {
            let mut got = Vec::with_capacity(n);
            let mut s = 0;
            while s < n {
                let e = (s + chunk).min(n);
                got.extend(cut_flags(&rows, s, e));
                s = e;
            }
            assert_eq!(got, want, "chunk {chunk}");
        }
    }

    /// A cut lands on the first row whose midpoint is at or after it, cuts
    /// sharing a row collapse, and row 0 is never an edge.
    #[test]
    fn a_cut_lands_on_its_first_row() {
        let row = shipped_clock();
        // row i sits at its tubelet pair's midpoint, (i + 1) / 30 s
        assert!(cut_rows(&[0.0, 20.0], row).is_empty()); // both inside row 0
        assert_eq!(cut_rows(&[60.0], row), vec![1]); // 33.3 < 60 <= 66.7
        assert_eq!(cut_rows(&[66.8, 90.0], row), vec![2]); // one row, one edge
        assert_eq!(shot_edges(&[60.0], 10, row), vec![0, 1, 10]);
        assert_eq!(shot_edges(&[60.0], 1, row), vec![0, 1], "past the end, dropped");
    }
}
