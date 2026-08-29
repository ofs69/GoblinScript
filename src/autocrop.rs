//! Auto-crop: the mask net's attention -> per-shot conservative crop rects.
//!
//! The deploy twin of `autocrop.py` (constants and recipe are 1:1 -- change
//! them together). A sparse PROBE pass decodes short native-rate windows of
//! the normalized clip, runs them through the encoder and `mask.onnx` (both
//! already in the bundle), and reduces the attention to one static rect per
//! shot: heads mixed weighted by their concentration (the diffuse heads'
//! whole-grid floor otherwise vetoes every crop), the row's background floor
//! subtracted (a content-independent border/corner prior that otherwise pins
//! the box to the frame in 63% of rows), per-row top-mass bboxes,
//! shot edges voted by the sharper half of the rows and clamped into the
//! PICTURE box (the frame minus dead letterbox/pillarbox bars, read off the
//! probe's own frames -- dead pixels carry nothing, so no rect covers them),
//! a one-cell margin, a zoom cap, identity snap near full frame. Rects
//! change only at detected cuts, so cropping introduces no motion or
//! transitions the source lacks.
//!
//! **A rect is CONTINUOUS -- fractions of the frame, not attention cells.**
//! The attention is sampled on the encoder's grid, so that is where the map
//! is read; the RECT the map decides is not confined to it. The map is
//! refined by `SUBCELL` between cell centres before the box is taken, the
//! votes are percentiles of continuous edges, and the size and placement that
//! come out of them are frame fractions the decode turns into pixels. On the
//! deploy grid one cell is 4.2% of the frame, which used to be the step of
//! every edge, every zoom (a ladder of six between the cap and the identity
//! snap) and every candidate the placement search could reach; the picture
//! box was quantized the same way, so a letterbox bar was trimmed to the
//! nearest 4.2% of the height. Sub-cell edges are real information: the
//! attention field is smooth and its samples say where between two cells the
//! mass sits, and a shot's rect averages hundreds of those samples.
//!
//! The crop is applied in the ENCODE DECODE chain (`SegmentedDecoder`), never
//! as a transcode: a re-encode was measured to land each render path on a
//! different clock (container start-time offsets) and charge kappa for it.
//! The clock here stays the synthesized frame grid, so drift is impossible
//! by construction; zoom itself is measured FREE to x2 on the frozen trunk.

use anyhow::{Context, Result};
use indicatif::ProgressBar;
use ort::session::Session;
use std::path::Path;

use crate::bundle::{ort_err, Manifest};
use crate::ffmpeg::Decoder;

// `autocrop.py` twins -- keep in lockstep.
const FLOOR_Q: f64 = 75.0; // background floor subtracted from the mixed map
const TOP_MASS_Q: f32 = 0.6; // per-row bbox: mixed-attention mass it holds
const CONC_CELLS: usize = 10; // sharpness = mass in this many top cells
const EDGE_Q: f64 = 10.0; // shot edge percentile over the sharper rows
const MARGIN_CELLS: f64 = 1.0; // extra cells each side of the shot bbox --
                     // an attention CELL, converted to a frame fraction by
                     // the grid it was read on, so the margin is the same
                     // piece of picture it always was
const SUBCELL: usize = 4; // sub-cell refinement of the attention map before
                     // the per-row box is taken: the map is interpolated onto
                     // this many samples per cell per axis, so an edge lands
                     // on 1/(4*grid) of the frame instead of 1/grid. The
                     // field is smooth and sampled at cell centres, so where
                     // a bright cell sits next to a dim one the crossing is
                     // between them, and this is what reads it
pub(crate) const MIN_SIDE_FRAC: f64 = 0.6667; // rect side floor (zoom cap x1.5)
const IDENT_FRAC: f64 = 0.88; // side >= this fraction of the grid -> no crop
const MIN_SHOT_ROWS: usize = 16; // sampled rows a shot needs to vote alone
pub(crate) const PIC_DEAD_LUMA: usize = 26; // a cell whose brightest pixel never exceeds
                     // this across the sampled frames is DEAD -- a letterbox/
                     // pillarbox bar, not picture. Bars decode to a few counts
                     // of codec noise; content that never clears this in any
                     // sample is indistinguishable from a bar, and trimming it
                     // is the right call either way.
const PROBE_EVERY_S: f64 = 15.0; // one window starts every this many seconds
const SEARCH_WINDOWS: usize = 2; // windows per shot the placement search reads
                     // -- each candidate costs one encode of each, so this is
                     // what keeps the search a few percent of a run
const SEARCH_MARGIN: f64 = 0.03; // a candidate must beat the incumbent's conf
                     // by this much to move the rect. The A/B's damaging moves
                     // rode +0.008/+0.023 while every winning move rode
                     // +0.033..+0.153 -- the read's
                     // errors are SMALL-margin, so the gate blocks them while
                     // passing every measured win
const SEARCH_STEP_CELLS: f64 = 2.0; // how far a candidate placement sits from
                     // the attention's own, in attention cells. The margins
                     // below were tuned with moves this size, so the step
                     // stays the distance it was measured at even though the
                     // rect it moves is no longer quantized to it
const SEARCH_KEEP_CONF: f64 = 0.88; // incumbent conf at which a shot is kept
                     // without searching the other candidates: no measured
                     // WINNING move ever came from an incumbent above 0.848
                     // (every move from 0.91+ rode +0.0002..+0.011, one of
                     // them the damaging one), so 0.88 sits in the gap and
                     // spends the 6 candidate encodes only where wins have
                     // ever lived -- on high-conf cutty clips this is where
                     // the probe minutes go

/// One crop rect as FRACTIONS of the frame -- `(x, y, w, h)`, origin top
/// left; identity is `(0.0, 0.0, 1.0, 1.0)`. Fractions and not pixels because
/// the same rect describes every copy of the picture: the normalized file the
/// decode crops, the original the review page streams, and the squashed
/// square the encoder reads (grid cells rescale the frame linearly on both
/// axes, so a fraction is a fraction in all three).
pub type Rect = (f64, f64, f64, f64);

/// The identity rect: the whole frame, which is what "no crop" is.
pub const IDENTITY: Rect = (0.0, 0.0, 1.0, 1.0);

/// A rect is the whole frame when both sides reach it -- within a rounding
/// hair, since these are computed fractions rather than counted cells.
fn is_whole(r: Rect) -> bool {
    r.2 >= 1.0 - 1e-6 && r.3 >= 1.0 - 1e-6
}

pub struct Plan {
    /// (first frame, rect) per segment, consecutive equal rects merged.
    pub segs: Vec<(usize, Rect)>,
    /// The rects the PROBE decided, before any hand correction: what the crop
    /// page's "auto" restores, and what its readout compares a drawn rect to.
    pub auto: Vec<(usize, Rect)>,
    /// A human drew the rects in `segs`. A hand-aimed plan is the aim, so it
    /// is never re-probed and never expires with a retuned recipe.
    pub manual: bool,
    pub grid: usize,
    /// Fraction of sampled rows with > 20% of their attention outside their
    /// rect -- the escape instrument, printed with the stage.
    pub escape_share: f64,
    /// Shots whose rect the confidence search MOVED off the attention's own
    /// placement, for the stage line.
    pub placed: usize,
}

impl Plan {
    pub fn is_identity(&self) -> bool {
        self.segs.iter().all(|&(_, r)| is_whole(r))
    }

    /// Median zoom over segments (1 / rect height), for the stage line.
    pub fn median_zoom(&self) -> f64 {
        let mut z: Vec<f64> = self.segs.iter().map(|(_, r)| 1.0 / r.3).collect();
        z.sort_by(|a, b| a.partial_cmp(b).unwrap());
        z[z.len() / 2]
    }

    /// The decode segments: rect -> "W:H:X:Y" in source pixels of the `w`x`h`
    /// normalized clip (even-rounded for the codec; identity -> no filter).
    pub fn segments(&self, w: usize, h: usize) -> Vec<(usize, Option<String>)> {
        self.segs
            .iter()
            .map(|&(start, rect)| (start, crop_arg(rect, w, h)))
            .collect()
    }

    /// What the review page draws. `fps` is the FRAME grid the segment starts
    /// are counted on (`Manifest::grid_fps`), so the times come out on the
    /// VIDEO clock the page already plays against.
    pub fn view(&self, fps: f64) -> View {
        View {
            segs: self
                .segs
                .iter()
                .map(|&(f, (x, y, w, h))| ViewSeg { t_ms: f as f64 / fps * 1000.0, x, y, w, h })
                .collect(),
            zoom: self.median_zoom(),
            escape_share: self.escape_share,
        }
    }

    /// Can a segmented decode actually run this plan?
    ///
    /// The two rules `SegmentedDecoder::open` enforces: the first segment
    /// starts the clip, and every later one starts strictly after the one
    /// before. A plan built by `probe` cannot break them -- `shot_frames` sees
    /// to that -- but a plan read back off disk was built by whatever wrote
    /// it, and one written before the frame-0 rule existed carries two
    /// segments on frame 0. Every other field of that cache entry still
    /// matches, the checkpoint included, so nothing else would refuse it and
    /// the run would stop at ENCODE exactly as it did the first time.
    /// Checking here is what lets the cache heal: the plan is turned down, the
    /// probe runs again, and the corrected plan takes its place on disk.
    pub fn is_runnable(&self) -> bool {
        self.segs.first().map(|s| s.0) == Some(0)
            && self.segs.windows(2).all(|w| w[0].0 < w[1].0)
    }

    /// A stable identity for the plan, part of the latent cache key: cached
    /// latents from a different plan (or none) must not be reused.
    pub fn key(&self) -> String {
        let s: Vec<String> = self
            .segs
            .iter()
            .map(|(f, (x, y, w, h))| format!("{f}:{x:.5},{y:.5},{w:.5},{h:.5}"))
            .collect();
        format!("crop-v2[{}]", s.join(";"))
    }
}

/// One segment as the review page reads it: when it starts on the video
/// clock, and its rect in FRAME FRACTIONS. Grid cells are a linear rescale of
/// the frame on both axes (the encoder decodes the picture squashed to a
/// square), so the fractions hold for whichever copy of the picture the page
/// happens to stream -- the original, or the normalized 480p one.
#[derive(Clone, serde::Serialize)]
pub struct ViewSeg {
    pub t_ms: f64,
    pub x: f64,
    pub y: f64,
    pub w: f64,
    pub h: f64,
}

/// The plan as a readout: the segments plus the two numbers the stage line
/// prints, so the page can say what the crop did as well as show it.
#[derive(Clone, serde::Serialize)]
pub struct View {
    pub segs: Vec<ViewSeg>,
    pub zoom: f64,
    pub escape_share: f64,
}

/// One sampled row: which shot it fell in, its tight bbox, its sharpness.
struct RowVote {
    shot: usize,
    /// `x0, x1, y0, y1` as FRACTIONS of the frame, read between cell centres.
    bbox: (f64, f64, f64, f64),
    conc: f32,
    map: Vec<f32>, // mixed attention, grid*grid, sums to 1 (escape instrument)
}

/// Cut times (ms) -> the frames a shot may START on: ascending, deduplicated,
/// and never frame 0.
///
/// Those three rules are what make the emitted plan a strictly ordered segment
/// list, which is what `SegmentedDecoder` requires of one. Frame 0 is already
/// every clip's first shot start, so a cut landing on it describes a shot of
/// zero frames -- one no sampled row can fall in, which therefore takes the
/// clip-global rect and then collides with the shot that really does begin
/// there. Two cuts inside one frame's width are that same collision arriving
/// the other way, and a `--cuts` file is under no obligation to arrive sorted.
///
/// `boundaries::cut_rows` holds these rules on the ROW grid, for the model's
/// own cut-flag channel. A shot plan needs them on the FRAME grid.
fn shot_frames(cuts_ms: &[f64], fps: f64) -> Vec<usize> {
    let mut f: Vec<usize> = cuts_ms
        .iter()
        .filter(|c| c.is_finite())
        .map(|&c| (c / 1000.0 * fps).round() as usize)
        .filter(|&f| f > 0)
        .collect();
    f.sort_unstable();
    f.dedup();
    f
}

/// The probe: sparse native-rate windows -> attention votes -> the plan.
///
/// `cuts_ms` are cut times in MILLISECONDS (as `boundaries::find_cuts`
/// returns them). Windows never need to respect them -- the mask net was
/// trained on cut-blind windows too.
#[allow(clippy::too_many_arguments)]
pub fn probe(
    video: &Path,
    enc: &mut Session,
    mask: &mut Session,
    head: Option<&mut Session>,
    man: &Manifest,
    cuts_ms: &[f64],
    n_frames: usize,
    photo: Option<&str>,
    pb: &ProgressBar,
) -> Result<Plan> {
    let (res, k) = (man.enc_res, man.tubelet_stride);
    let group = (man.clip_len / 2) * 2 * k;
    let grid = man.grid;
    let row_bytes = man.dim * grid * grid;

    // window starts: one every PROBE_EVERY_S, always at least one
    let step = ((PROBE_EVERY_S * man.grid_fps).round() as usize).max(group + k);
    let mut starts: Vec<usize> = (0..n_frames.saturating_sub(group + k)).step_by(step).collect();
    if starts.is_empty() {
        starts.push(0);
    }
    pb.set_length(starts.len() as u64);

    let cut_frames = shot_frames(cuts_ms, man.grid_fps);
    let shot_of = |frame: usize| cut_frames.partition_point(|&c| c <= frame);

    let mut x: Vec<u8> = Vec::with_capacity((man.clip_len / 2) * 2 * res * res * 3);
    let mut slab: Vec<i8> = vec![0; group * row_bytes];
    let mut votes: Vec<RowVote> = Vec::new();
    // the picture box is read per PIXEL line, not per cell: a bar is trimmed
    // where it actually ends
    let mut row_max = vec![0u8; res];
    let mut col_max = vec![0u8; res];

    // uncropped reads of a taller-than-spec normalize soften to the spec
    // height first -- the same pixels the uncropped ENCODE decode will read
    // -- and every read applies the clip's exposure correction (`photo`)
    let (vw, vh) = crate::ffmpeg::dims(video)?;
    let soften = soften_arg(vh as u32, man.transcode.height);
    let plain = crate::exposure::join_filters(photo, soften.as_deref());

    // The probe decodes run one window ahead of the GPU on their own thread:
    // each window is a fresh ffmpeg seek plus ~2 s of frames, and it used to
    // sit between two `encode_window` calls. The channel is bounded at one
    // window (~57 MB); windows arrive in probe order with identical frames,
    // so the votes -- and the rects they decide -- are unchanged. A short
    // window means the clip ended there: the worker stops and the main loop
    // breaks, and the earlier windows stand.
    std::thread::scope(|sc| -> Result<()> {
        let (tx, rx) = std::sync::mpsc::sync_channel::<Result<Option<Vec<Vec<u8>>>>>(1);
        let starts_ref = &starts;
        let plain = plain.as_deref();
        sc.spawn(move || {
            for &start in starts_ref {
                let one = (|| -> Result<Option<Vec<Vec<u8>>>> {
                    let mut dec =
                        Decoder::open_at(video, res, res, man.grid_fps, None, start, plain)?;
                    let mut frames: Vec<Vec<u8>> = Vec::with_capacity(group + k);
                    let mut f = Vec::new();
                    while frames.len() < group + k && dec.next_frame(&mut f)? {
                        frames.push(std::mem::take(&mut f));
                    }
                    Ok((frames.len() == group + k).then_some(frames))
                })();
                let stop = matches!(one, Ok(None) | Err(_));
                if tx.send(one).is_err() || stop {
                    break; // the probe side is done, or this clip is
                }
            }
        });
        for (i, got) in rx.into_iter().enumerate() {
            crate::cancel::check()?;
            let Some(frames) = got? else { break };
            let start = starts[i];
            accum_edges(&frames[0], res, &mut row_max, &mut col_max);
            let win: Vec<&[u8]> = frames.iter().map(|v| v.as_slice()).collect();
            crate::encode::encode_window(
                enc, man, &win, man.int8_scale, row_bytes, &mut x, &mut slab,
            )?;
            for a in 0..group {
                let row = &slab[a * row_bytes..(a + 1) * row_bytes];
                if let Some(v) = row_vote(mask, man, row, shot_of(start + a))? {
                    votes.push(v);
                }
            }
            pb.set_position(i as u64 + 1);
        }
        Ok(())
    })?;
    anyhow::ensure!(!votes.is_empty(), "the auto-crop probe sampled no rows");

    // per-shot rects; under-sampled shots take the clip-global rect
    let pic = picture_box(&row_max, &col_max, res);
    let n_shots = cut_frames.len() + 1;
    let all: Vec<&RowVote> = votes.iter().collect();
    let glob = shot_rect(&all, grid, pic);
    let shot_start = |s: usize| if s == 0 { 0 } else { cut_frames[s - 1] };

    let mut rects: Vec<Rect> = Vec::with_capacity(n_shots);
    for s in 0..n_shots {
        let sv: Vec<&RowVote> = votes.iter().filter(|v| v.shot == s).collect();
        if sv.len() < MIN_SHOT_ROWS {
            rects.push(glob);
        } else {
            rects.push(shot_rect(&sv, grid, pic));
        }
    }

    // The attention decides how big the rect is and roughly where; the MODEL
    // decides which placement it actually reads best. Every candidate shows it
    // the same amount of picture, so the only variable is position -- and a
    // like-for-like comparison is the only one conf can make, since its value
    // carries a systematic offset between framings of different zoom.
    let mut moved = 0usize;
    if let Some(head) = head {
        for s in 0..n_shots {
            let rect = rects[s];
            if is_whole(rect) {
                continue; // nothing placed, nothing to place better
            }
            let mine: Vec<usize> = starts
                .iter()
                .copied()
                .filter(|&st| shot_of(st) == s)
                .take(SEARCH_WINDOWS)
                .collect();
            if mine.is_empty() {
                continue;
            }
            // score one candidate over this shot's sampled windows
            let mut score_cand = |cand: Rect| -> Result<Option<f64>> {
                let crop =
                    crate::exposure::join_filters(photo, crop_arg(cand, vw, vh).as_deref());
                let (mut sum, mut n) = (0.0, 0usize);
                for &st in &mine {
                    crate::cancel::check()?;
                    let mut dec = Decoder::open_at(
                        video, res, res, man.grid_fps, None, st, crop.as_deref(),
                    )?;
                    let mut frames: Vec<Vec<u8>> = Vec::with_capacity(group + k);
                    let mut f = Vec::new();
                    while frames.len() < group + k && dec.next_frame(&mut f)? {
                        frames.push(std::mem::take(&mut f));
                    }
                    if frames.len() < group + k {
                        continue;
                    }
                    let win: Vec<&[u8]> = frames.iter().map(|v| v.as_slice()).collect();
                    crate::encode::encode_window(
                        enc, man, &win, man.int8_scale, row_bytes, &mut x, &mut slab,
                    )?;
                    if let Some(c) = block_conf(head, man, &slab, group)? {
                        sum += c;
                        n += 1;
                    }
                }
                Ok((n > 0).then(|| sum / n as f64))
            };
            // the incumbent first: a shot the model already reads confidently
            // never produced a move that cleared the margin, so it is kept
            // without spending the other candidates' encodes
            let incumbent = match score_cand(rect)? {
                Some(c) if c < SEARCH_KEEP_CONF => c,
                _ => continue,
            };
            let mut best = (rect, incumbent);
            for cand in placements(rect, grid, pic).into_iter().skip(1) {
                if let Some(score) = score_cand(cand)? {
                    if score > best.1 {
                        best = (cand, score);
                    }
                }
            }
            // the margin gate: the read's measured errors are small-margin
            if best.0 != rect && best.1 - incumbent >= SEARCH_MARGIN {
                moved += 1;
                rects[s] = best.0;
            }
        }
    }

    // escape instrument over every sampled row, against its shot's rect
    let escaped = votes
        .iter()
        .filter(|v| 1.0 - inside_mass(&v.map, grid, rects[v.shot]) > 0.2)
        .count();

    // shot starts on the frame grid, consecutive equal rects merged
    let mut segs: Vec<(usize, Rect)> = Vec::new();
    for (s, &r) in rects.iter().enumerate() {
        let start = shot_start(s);
        match segs.last() {
            Some(&(_, last)) if last == r => {}
            _ => segs.push((start, r)),
        }
    }

    Ok(Plan {
        auto: segs.clone(),
        segs,
        manual: false,
        grid,
        escape_share: escaped as f64 / votes.len() as f64,
        placed: moved,
    })
}

/// One rect -> the decode-chain `crop` filter in pixels of a `w`x`h` picture,
/// even-rounded for the codec. `None` for the identity rect, which is no
/// filter at all. The one place fractions become pixels, so the plan the
/// probe searches and the plan the decode applies cannot drift apart.
fn crop_arg(rect: Rect, w: usize, h: usize) -> Option<String> {
    let (x0, y0, rw, rh) = rect;
    if is_whole(rect) {
        return None;
    }
    let cw = ((rw * w as f64 / 2.0).round() as usize * 2).clamp(2, w & !1);
    let ch = ((rh * h as f64 / 2.0).round() as usize * 2).clamp(2, h & !1);
    let cx = ((x0 * w as f64).round().max(0.0) as usize).min(w - cw);
    let cy = ((y0 * h as f64).round().max(0.0) as usize).min(h - ch);
    Some(format!("crop={cw}:{ch}:{cx}:{cy}"))
}

/// The softening stage an UNCROPPED decode of the one normalized file needs
/// when that file is taller than the bundle's spec: down to the spec height
/// first, so the encoder sees the scale ratio it was trained on. `None` when
/// the file already is the spec height -- the decode is then byte-identical
/// to the era of a spec-height normalize. A CROPPED decode never softens:
/// the extra lines are exactly what the crop is there to keep.
pub fn soften_arg(norm_h: u32, spec_h: u32) -> Option<String> {
    (norm_h > spec_h).then(|| format!("scale=-2:{spec_h}:flags=bicubic"))
}

/// Mean confidence the head reports over one block of latent rows.
///
/// The conf head is the only readout that says whether a framing SUITS the
/// model without a reference script -- and it ranks crop damage where the
/// escape instrument does not. Read on a block
/// this short the value carries a constant negative bias against a full-length
/// draft (-0.072 measured), but the ORDER survives (rank rho +0.969 over 32
/// spans), and an order is all a search between candidates needs.
fn block_conf(head: &mut Session, man: &Manifest, rows: &[i8], n: usize) -> Result<Option<f64>> {
    if !head.outputs().iter().any(|o| o.name() == "conf") {
        return Ok(None); // a bundle from before the conf head
    }
    let xs = vec![1i64, n as i64, man.dim as i64, man.grid as i64, man.grid as i64];
    let xt = ort::value::TensorRef::from_array_view((xs, rows)).map_err(ort_err)?;
    let cut = vec![0f32; n];
    let ct = ort::value::TensorRef::from_array_view((vec![1i64, n as i64], &cut[..]))
        .map_err(ort_err)?;
    let out = head.run(ort::inputs!["x_i8" => xt, "cut" => ct]).map_err(ort_err)?;
    let (_s, v) = out["conf"].try_extract_tensor::<f32>().map_err(ort_err)?;
    let (mut sum, mut cnt) = (0.0, 0usize);
    for &c in v.iter() {
        if c.is_finite() {
            sum += c as f64;
            cnt += 1;
        }
    }
    Ok((cnt > 0).then(|| sum / cnt as f64))
}

/// Per-LINE running maximum of the frame's brightest channel byte, down each
/// axis -- the evidence the picture box is read from. Bars stay at codec
/// noise across every sample; any content line clears them the first time it
/// is lit. Per line rather than per cell, so a bar is measured where it ends
/// instead of at the nearest 4.2% of the frame.
fn accum_edges(frame: &[u8], res: usize, row_max: &mut [u8], col_max: &mut [u8]) {
    for py in 0..res {
        let row = &frame[py * res * 3..(py + 1) * res * 3];
        for px in 0..res {
            let p = &row[px * 3..px * 3 + 3];
            let v = p[0].max(p[1]).max(p[2]);
            if v > row_max[py] {
                row_max[py] = v;
            }
            if v > col_max[px] {
                col_max[px] = v;
            }
        }
    }
}

/// The PICTURE box `(x0, x1, y0, y1)` as fractions of the frame: the frame
/// minus its dead edge lines (letterbox and pillarbox bars). Dead pixels
/// carry nothing, so no rect has business covering them -- but a "picture"
/// too small for the zoom cap's smallest rect is not one the cap can serve,
/// and the whole frame stands in for it.
///
/// `res` is the square the probe decoded, so a line here is a line of the
/// SQUASHED picture; both axes divide by the same `res` and come out as
/// fractions of the real frame either way.
fn picture_box(row_max: &[u8], col_max: &[u8], res: usize) -> (f64, f64, f64, f64) {
    let live = |v: &[u8], i: usize| (v[i] as usize) > PIC_DEAD_LUMA;
    let span = |v: &[u8]| -> (usize, usize) {
        let (mut a, mut b) = (0usize, res);
        while a < b && !live(v, a) {
            a += 1;
        }
        while b > a && !live(v, b - 1) {
            b -= 1;
        }
        (a, b)
    };
    let (y0, y1) = span(row_max);
    let (x0, x1) = span(col_max);
    let r = res as f64;
    let (x0, x1, y0, y1) = (x0 as f64 / r, x1 as f64 / r, y0 as f64 / r, y1 as f64 / r);
    if x1 - x0 < MIN_SIDE_FRAC || y1 - y0 < MIN_SIDE_FRAC {
        return (0.0, 1.0, 0.0, 1.0);
    }
    (x0, x1, y0, y1)
}

/// Clamp a rect start so `[start, start+side)` sits inside the picture span
/// `[p0, p1)` -- centered overhang when the side outgrows the span -- and
/// always inside the frame.
fn clamp_into(start: f64, side: f64, p0: f64, p1: f64) -> f64 {
    let hi = p1 - side;
    let v = if hi < p0 { (p0 + p1 - side) / 2.0 } else { start.clamp(p0, hi) };
    v.clamp(0.0, (1.0 - side).max(0.0))
}

/// The attention mass of one row that falls INSIDE a rect. Cells are boxes
/// and the rect is continuous, so an edge cell counts by the area of it the
/// rect covers -- the same reading the crop itself makes of the picture.
fn inside_mass(map: &[f32], grid: usize, rect: Rect) -> f64 {
    let (x, y, w, h) = rect;
    let g = grid as f64;
    let overlap = |i: usize, lo: f64, hi: f64| -> f64 {
        let (a, b) = (i as f64 / g, (i + 1) as f64 / g);
        (b.min(hi) - a.max(lo)).max(0.0) * g
    };
    let mut sum = 0.0;
    for cy in 0..grid {
        let fy = overlap(cy, y, y + h);
        if fy <= 0.0 {
            continue;
        }
        for cx in 0..grid {
            let fx = overlap(cx, x, x + w);
            if fx > 0.0 {
                sum += map[cy * grid + cx] as f64 * fx * fy;
            }
        }
    }
    sum
}

/// Candidate placements for a rect of a fixed size: where the attention put
/// it, and the same box nudged around it -- every candidate inside the
/// picture box, since a spot on the bars is not a placement worth an encode.
///
/// SIZE is deliberately not searched. Confidence carries a systematic offset
/// between framings of different zoom -- it fell on every clip measured,
/// including one the crop helped -- so ranking sizes against each other would
/// just re-elect the loosest. Position has no such confound: every candidate
/// shows the model the same amount of picture, so the comparison is like for
/// like and the only question left is WHERE.
fn placements(rect: Rect, grid: usize, pic: (f64, f64, f64, f64)) -> Vec<Rect> {
    let (x, y, w, h) = rect;
    let step = SEARCH_STEP_CELLS / grid as f64;
    let mut out = vec![rect];
    for (dx, dy) in [(-1.0, 0.0), (1.0, 0.0), (0.0, -1.0), (0.0, 1.0), (-1.0, -1.0), (1.0, 1.0)] {
        let nx = clamp_into(x + dx * step, w, pic.0, pic.1);
        let ny = clamp_into(y + dy * step, h, pic.2, pic.3);
        let cand = (nx, ny, w, h);
        if !out.iter().any(|&o: &Rect| same_rect(o, cand)) {
            out.push(cand);
        }
    }
    out
}

/// Two rects the decode cannot tell apart (a hair under a tenth of a pixel of
/// a 4K line). A candidate this close to one already queued is not worth the
/// encodes it would cost to rank.
fn same_rect(a: Rect, b: Rect) -> bool {
    (a.0 - b.0).abs() < 1e-5
        && (a.1 - b.1).abs() < 1e-5
        && (a.2 - b.2).abs() < 1e-5
        && (a.3 - b.3).abs() < 1e-5
}

/// One latent row through the mask net -> its attention vote.
fn row_vote(mask: &mut Session, man: &Manifest, row: &[i8], shot: usize) -> Result<Option<RowVote>> {
    let grid = man.grid;
    let shape = vec![1i64, man.dim as i64, grid as i64, grid as i64];
    let t = ort::value::TensorRef::from_array_view((shape, row)).map_err(ort_err)?;
    let out = mask.run(ort::inputs!["x_i8" => t]).map_err(ort_err)?;
    let (s, gate) = out["gate"].try_extract_tensor::<f32>().map_err(ort_err)?;
    if s.len() != 4 {
        return Ok(None);
    }
    let heads = s[1] as usize;
    let cells = grid * grid;

    // per-head normalize + concentration weights, mixed to one map (sum 1)
    let mut map = vec![0f32; cells];
    let mut top = vec![0f32; cells];
    for h in 0..heads {
        let g = &gate[h * cells..(h + 1) * cells];
        let sum: f32 = g.iter().sum::<f32>() + 1e-9;
        top.copy_from_slice(g);
        top.sort_by(|a, b| b.partial_cmp(a).unwrap());
        let conc_h: f32 = top[..CONC_CELLS.min(cells)].iter().sum::<f32>() / sum;
        for (m, &v) in map.iter_mut().zip(g) {
            *m += conc_h * v / sum;
        }
    }
    let total: f32 = map.iter().sum::<f32>() + 1e-9;
    for m in map.iter_mut() {
        *m /= total;
    }

    // Subtract the row's background floor. The mixed map is an ROI sitting on
    // a near-uniform background that is measured CONTENT-INDEPENDENT (the
    // border ring holds 18% of the mass and the corner blocks 5.5%, and
    // neither falls in rows whose ROI is centred and sharp). The bbox below
    // is the bounding box of a cell SET, so one such speck pins an edge to
    // the frame -- it does in 63% of rows. Removing the floor drops specks by
    // their MASS rather than their position, so a genuinely bright cell at
    // the frame edge survives. Percentile-subtract-then-renormalize is scale
    // invariant, so applying it here matches `autocrop.py` doing it to the
    // un-normalized mix.
    top.copy_from_slice(&map);
    top.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let floor = percentile(&top.iter().map(|&v| v as f64).collect::<Vec<_>>(), FLOOR_Q) as f32;
    let mut kept = 0f32;
    for m in map.iter_mut() {
        *m = (*m - floor).max(0.0);
        kept += *m;
    }
    for m in map.iter_mut() {
        *m /= kept + 1e-9;
    }

    // the row's sharpness, read on the cells the attention was sampled on --
    // its weight in the head mix and its vote weight in the shot
    top.copy_from_slice(&map);
    top.sort_by(|a, b| b.partial_cmp(a).unwrap());
    let conc: f32 = top[..CONC_CELLS.min(cells)].iter().sum();

    Ok(Some(RowVote { shot, bbox: top_mass_box(&map, grid), conc, map }))
}

/// The row's box as FRACTIONS of the frame: the smallest descending-value set
/// of the REFINED map holding `TOP_MASS_Q` of the mass.
///
/// The refinement is what takes the box off the cell lattice. The set itself
/// is the same definition it always was -- values in descending order until
/// the mass is held -- but it is taken on a map interpolated `SUBCELL` times
/// finer, so the edge lands where the field crosses the threshold rather than
/// on whichever cell centre was nearest.
fn top_mass_box(map: &[f32], grid: usize) -> (f64, f64, f64, f64) {
    let sub = grid * SUBCELL;
    let fine = refine(map, grid);
    let mut srt = fine.clone();
    srt.sort_by(|a, b| b.partial_cmp(a).unwrap());
    if srt[0] <= 0.0 {
        return (0.0, 1.0, 0.0, 1.0); // no mass anywhere: no opinion
    }
    let mut mass = 0f32;
    let mut thr = srt[srt.len() - 1];
    for &v in &srt {
        mass += v;
        if mass >= TOP_MASS_Q {
            thr = v;
            break;
        }
    }
    let (mut x0, mut x1, mut y0, mut y1) = (sub, 0usize, sub, 0usize);
    for (i, &v) in fine.iter().enumerate() {
        if v >= thr {
            let (cx, cy) = (i % sub, i / sub);
            x0 = x0.min(cx);
            x1 = x1.max(cx + 1);
            y0 = y0.min(cy);
            y1 = y1.max(cy + 1);
        }
    }
    let s = sub as f64;
    (x0 as f64 / s, x1 as f64 / s, y0 as f64 / s, y1 as f64 / s)
}

/// Bilinear refinement of the attention map: `SUBCELL` samples per cell per
/// axis, taken between CELL CENTRES and held flat outside the outermost ones,
/// renormalized to sum 1. `autocrop.py`'s twin -- one interpolation, or the
/// two languages read different edges off the same attention.
fn refine(map: &[f32], grid: usize) -> Vec<f32> {
    let sub = grid * SUBCELL;
    // for each fine index: the two cell centres it sits between, and how far
    let axis: Vec<(usize, usize, f32)> = (0..sub)
        .map(|j| {
            let f = ((j as f64 + 0.5) / SUBCELL as f64 - 0.5).clamp(0.0, (grid - 1) as f64);
            let i0 = f.floor() as usize;
            let i1 = (i0 + 1).min(grid - 1);
            (i0, i1, (f - i0 as f64) as f32)
        })
        .collect();
    let mut out = vec![0f32; sub * sub];
    let mut total = 0f32;
    for (jy, &(y0, y1, ty)) in axis.iter().enumerate() {
        for (jx, &(x0, x1, tx)) in axis.iter().enumerate() {
            let a = map[y0 * grid + x0] * (1.0 - tx) + map[y0 * grid + x1] * tx;
            let b = map[y1 * grid + x0] * (1.0 - tx) + map[y1 * grid + x1] * tx;
            let v = a * (1.0 - ty) + b * ty;
            out[jy * sub + jx] = v;
            total += v;
        }
    }
    if total > 0.0 {
        for v in out.iter_mut() {
            *v /= total;
        }
    }
    out
}

/// Rows of one shot (or the whole clip) -> its rect: edges voted at EDGE_Q
/// percentiles by the sharper half of rows, clamped into the picture box
/// (attention on dead bars is floor noise, and clamping is also what keeps a
/// diffuse vote from reaching the identity snap through the bars), a margin,
/// the zoom cap, and an identity snap when what is left is most of the frame
/// anyway.
fn shot_rect(votes: &[&RowVote], grid: usize, pic: (f64, f64, f64, f64)) -> Rect {
    let (x0, x1, y0, y1) = vote_edges(votes, grid);
    let (x0, x1) = (x0.max(pic.0), x1.min(pic.1));
    let (y0, y1) = (y0.max(pic.2), y1.min(pic.3));
    // one side for both axes: a fraction of the width equals the same
    // fraction of the height in grid cells, which is what keeps the crop the
    // shape of the source and the squash the encoder was trained on
    let side = (x1 - x0).max(y1 - y0).max(MIN_SIDE_FRAC).min(1.0);
    if side >= IDENT_FRAC {
        return IDENTITY;
    }
    let cx = (x0 + x1) / 2.0;
    let cy = (y0 + y1) / 2.0;
    let rx = clamp_into(cx - side / 2.0, side, pic.0, pic.1);
    let ry = clamp_into(cy - side / 2.0, side, pic.2, pic.3);
    (rx, ry, side, side)
}

/// The voted edges of a set of rows: the sharper half decides, at the EDGE_Q
/// percentiles, plus the safety margin. The one definition of "where the
/// attention is", read once and used by every rect the plan emits.
fn vote_edges(votes: &[&RowVote], grid: usize) -> (f64, f64, f64, f64) {
    let mut conc: Vec<f32> = votes.iter().map(|v| v.conc).collect();
    conc.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let med = conc[conc.len() / 2];
    let sharp: Vec<&&RowVote> = votes.iter().filter(|v| v.conc >= med).collect();

    let col = |f: fn(&RowVote) -> f64| -> Vec<f64> {
        let mut v: Vec<f64> = sharp.iter().map(|r| f(r)).collect();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v
    };
    let m = MARGIN_CELLS / grid as f64;
    (
        percentile(&col(|r| r.bbox.0), EDGE_Q) - m,
        percentile(&col(|r| r.bbox.1), 100.0 - EDGE_Q) + m,
        percentile(&col(|r| r.bbox.2), EDGE_Q) - m,
        percentile(&col(|r| r.bbox.3), 100.0 - EDGE_Q) + m,
    )
}

/// numpy-compatible linear-interpolation percentile over a SORTED slice.
fn percentile(sorted: &[f64], q: f64) -> f64 {
    let n = sorted.len();
    if n == 1 {
        return sorted[0];
    }
    let pos = q / 100.0 * (n - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo as f64)
}

/// The height of the ONE normalized file, derived from the ENCODER's input:
/// the least height at which a cap-tight crop still reads only real source
/// lines -- `enc_res / MIN_SIDE_FRAC` = 576 at the shipped bundle.
///
/// A crop keeps `MIN_SIDE_FRAC` of the file's height, so at this height the
/// tightest legal crop delivers exactly `enc_res` lines: nothing invented,
/// at scale ratio 1.0 against the trained 1.25x. Deriving from the SPEC
/// height instead (480/0.667 = 720) would land the cap crop at the trained
/// ratio, but that purity is unmeasured while its cost is not (a ~50% taller
/// transcode, and the VR reprojection renders at this height) -- and every
/// measured crop result was collected with crops UPSCALED from 480p, so the
/// evidence floor is far below either. 576 is the min/max (user call,
/// 2026-07-30, explicitly without an A/B). The uncropped read of the file
/// still softens to the spec height first (`soften_arg`), and the caller
/// caps by the source's own lines (a 480p source normalizes at 480,
/// byte-identical to a spec-height normalize).
///
/// Deploy only, and deliberately: the corpus keeps its 480p normalize (the
/// caches and the trunk are built on it), so this closes the gap for the
/// footage the crop exists for without touching what the model learned from.
pub fn crop_norm_height(enc_res: u32) -> u32 {
    let h = (enc_res as f64 / MIN_SIDE_FRAC).round() as u32;
    h + (h % 2) // even, for the codec
}

/// The probe needs the bundle's mask graph; refuse up front, not mid-run.
pub fn require_mask(mask: &Option<Session>) -> Result<()> {
    anyhow::ensure!(
        mask.is_some(),
        "--autocrop needs a bundle with mask.onnx (this one predates it) -- \
         re-export the bundle"
    );
    Ok(())
}

/// The recipe half of a plan's identity: every constant that shapes the rects.
/// The probe is a pure function of (normalized clip, bundle, recipe), and the
/// bundle stamp alone cannot see a retuned constant -- a plan cached before a
/// retune would be silently reused under the same checkpoint. A missing or
/// differing stamp re-probes.
fn recipe_id() -> String {
    format!(
        "f{FLOOR_Q}-m{TOP_MASS_Q}-c{CONC_CELLS}-e{EDGE_Q}-g{MARGIN_CELLS}-\
         s{MIN_SIDE_FRAC}-i{IDENT_FRAC}-r{MIN_SHOT_ROWS}-p{PROBE_EVERY_S}-\
         w{SEARCH_WINDOWS}-d{PIC_DEAD_LUMA}-sm{SEARCH_MARGIN}-\
         sk{SEARCH_KEEP_CONF}-u{SUBCELL}-ss{SEARCH_STEP_CELLS}"
    )
}

/// The plan on disk (`autocrop.json` in the video's cache dir): the probe is
/// a pure function of (normalized clip, bundle, recipe), so a re-run reuses
/// it -- but a new checkpoint is a new mask net and a retuned constant is a
/// new recipe, so both identities are keys.
#[derive(serde::Serialize, serde::Deserialize)]
struct PlanFile {
    checkpoint: String,
    epoch: i64,
    basis_id: String,
    grid: usize,
    recipe: String,
    /// The exposure gamma the probe saw the pixels under -- a different
    /// correction is a different picture, so it is part of the identity.
    gamma: f64,
    escape_share: f64,
    segs: Vec<(usize, Rect)>,
    /// The probe's own rects, kept beside a hand correction so "auto" in the
    /// crop page has something to restore.
    #[serde(default)]
    auto: Vec<(usize, Rect)>,
    /// A human drew `segs`.
    #[serde(default)]
    manual: bool,
}

/// The cached plan, when it is one this run may use.
///
/// A PROBED plan is a pure function of (clip, bundle, recipe, exposure), so
/// every one of those is a key and any mismatch re-probes. A HAND-AIMED one
/// answers to none of them: the rect was drawn against the picture, and a new
/// checkpoint or a retuned constant is no reason to throw away the one thing
/// in the pipeline a person decided themselves. Only the grid has to match,
/// because that is what the rects were read on.
pub fn read_cached(dir: &Path, man: &Manifest, gamma: f64) -> Option<Plan> {
    let p: PlanFile =
        serde_json::from_slice(&std::fs::read(dir.join("autocrop.json")).ok()?).ok()?;
    let fresh = p.checkpoint == man.checkpoint
        && p.epoch == man.epoch
        && p.basis_id == man.basis_id
        && p.recipe == recipe_id()
        && p.gamma == gamma;
    (p.grid == man.grid && (fresh || p.manual))
        .then_some(Plan {
            placed: 0,
            auto: if p.auto.is_empty() { p.segs.clone() } else { p.auto },
            segs: p.segs,
            manual: p.manual,
            grid: p.grid,
            escape_share: p.escape_share,
        })
        .filter(Plan::is_runnable)
}

pub fn write_cached(dir: &Path, man: &Manifest, plan: &Plan, gamma: f64) -> Result<()> {
    let f = PlanFile {
        checkpoint: man.checkpoint.clone(),
        epoch: man.epoch,
        basis_id: man.basis_id.clone(),
        grid: plan.grid,
        recipe: recipe_id(),
        gamma,
        escape_share: plan.escape_share,
        segs: plan.segs.clone(),
        auto: plan.auto.clone(),
        manual: plan.manual,
    };
    std::fs::write(dir.join("autocrop.json"), serde_json::to_vec_pretty(&f)?)
        .context("could not write the auto-crop plan")?;
    Ok(())
}

/// The stage's done line: what the crop decided, and its instrument.
pub fn stage_line(plan: &Plan, reused: bool) -> String {
    let tag = match (plan.manual, reused) {
        (true, _) => crate::t!("console.crop.hand"),
        (false, true) => crate::t!("console.crop.reused"),
        _ => "",
    };
    if plan.is_identity() {
        crate::t!("console.crop.identity", tag = tag)
    } else {
        let placed = if plan.placed > 0 {
            crate::t!("console.crop.replaced", n = plan.placed)
        } else {
            String::new()
        };
        crate::t!(
            "console.crop.rects",
            n = plan.segs.len(),
            placed = placed,
            zoom = format!("{:.2}", plan.median_zoom()),
            esc = format!("{:.1}", plan.escape_share * 100.0),
            tag = tag,
        )
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn percentile_matches_numpy() {
        let v = [1.0, 2.0, 3.0, 4.0];
        assert_eq!(percentile(&v, 0.0), 1.0);
        assert_eq!(percentile(&v, 100.0), 4.0);
        assert_eq!(percentile(&v, 50.0), 2.5); // np.percentile linear
        assert!((percentile(&v, 10.0) - 1.3).abs() < 1e-12);
        assert_eq!(percentile(&[7.0], 90.0), 7.0);
    }

    /// A cut on frame 0 is the one every clip already has. Kept, it hands the
    /// plan two segments starting on frame 0 and the decoder refuses the run.
    #[test]
    fn a_cut_on_the_first_frame_is_not_a_shot_start() {
        // the reported clip: cuts at 0 and 2966.67 ms on the 30 fps grid
        assert_eq!(shot_frames(&[0.0, 2966.67, 19933.33], 30.0), vec![89, 598]);
        // and the same list with no frame-0 cut is untouched
        assert_eq!(shot_frames(&[2966.67, 19933.33], 30.0), vec![89, 598]);
    }

    /// Two cuts inside one frame's width are one shot start, not two.
    #[test]
    fn cuts_landing_on_one_frame_collapse() {
        // 30.0, 30.15 and 30.3 frames all round onto frame 30
        assert_eq!(shot_frames(&[1000.0, 1005.0, 1010.0], 30.0), vec![30]);
        assert_eq!(shot_frames(&[1000.0, 1100.0], 30.0), vec![30, 33]);
    }

    /// A cuts file is under no obligation to arrive sorted, and
    /// `partition_point` is only answerable on an ascending list.
    #[test]
    fn an_unsorted_cut_list_still_yields_an_ordered_plan() {
        let f = shot_frames(&[19933.33, 0.0, 2966.67, 2966.67], 30.0);
        assert_eq!(f, vec![89, 598]);
        assert!(f.windows(2).all(|w| w[0] < w[1]));
    }

    /// Nothing in a cuts file is trusted to be a number a frame index can be
    /// made of -- an infinity would otherwise saturate to usize::MAX.
    #[test]
    fn unusable_cut_times_are_dropped() {
        let f = shot_frames(&[f64::NAN, f64::INFINITY, -500.0, 1000.0], 30.0);
        assert_eq!(f, vec![30]);
    }

    fn plan_of(starts: &[usize]) -> Plan {
        let segs: Vec<(usize, Rect)> =
            starts.iter().map(|&f| (f, (0.0, 0.0, 0.6667, 0.6667))).collect();
        Plan { auto: segs.clone(), segs, manual: false, grid: 24, escape_share: 0.0, placed: 0 }
    }

    /// A cache written before the frame-0 rule existed must not be handed
    /// back: every other field of it still matches, so this is the only thing
    /// standing between a stale plan and the same failure a second time.
    #[test]
    fn a_plan_that_starts_twice_on_frame_zero_is_not_runnable() {
        assert!(!plan_of(&[0, 0, 89, 598]).is_runnable());
        assert!(plan_of(&[0, 89, 598, 715, 1235]).is_runnable());
    }

    /// A plan that does not open the clip cannot be run either -- the decoder
    /// has nothing to deliver the frames before its first segment.
    #[test]
    fn a_plan_must_open_on_the_first_frame() {
        assert!(!plan_of(&[89, 598]).is_runnable());
        assert!(!plan_of(&[]).is_runnable());
        assert!(plan_of(&[0]).is_runnable());
    }

    fn vote(bbox: (f64, f64, f64, f64), conc: f32) -> RowVote {
        RowVote { shot: 0, bbox, conc, map: Vec::new() }
    }

    /// A row whose box is stated in whole cells, which is how these fixtures
    /// read: the recipe no longer rounds to cells, but a cell is still the
    /// unit the attention arrives in.
    fn cells(x0: f64, x1: f64, y0: f64, y1: f64, conc: f32) -> RowVote {
        vote((x0 / 24.0, x1 / 24.0, y0 / 24.0, y1 / 24.0), conc)
    }

    fn refs(v: &[RowVote]) -> Vec<&RowVote> {
        v.iter().collect()
    }

    const FULL: (f64, f64, f64, f64) = (0.0, 1.0, 0.0, 1.0);

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9
    }

    #[test]
    fn tight_agreeing_rows_crop_with_margin_and_floor() {
        // every sharp row says cells 8..14 (6 wide): the margin adds one cell
        // each side (8 of 24 = 0.3333), and the zoom cap lifts it to 0.6667
        let votes: Vec<RowVote> = (0..40).map(|_| cells(8.0, 14.0, 8.0, 14.0, 1.0)).collect();
        let (x0, y0, w, h) = shot_rect(&refs(&votes), 24, FULL);
        assert!(close(w, MIN_SIDE_FRAC) && close(h, MIN_SIDE_FRAC));
        // centred on the vote (cells 7..15, centre 11 of 24), in frame
        assert!(x0 <= 7.0 / 24.0 && x0 + w >= 15.0 / 24.0, "rect misses the vote");
        assert!(close(x0, 11.0 / 24.0 - MIN_SIDE_FRAC / 2.0));
        assert!(close(y0, x0));
    }

    /// The rect is no longer a whole number of cells: a vote a third of a cell
    /// wider moves the rect by a third of a cell, where it used to move by
    /// nothing at all until the vote crossed a whole one.
    #[test]
    fn a_sub_cell_vote_moves_the_rect_by_a_sub_cell_amount() {
        let tight: Vec<RowVote> = (0..40).map(|_| cells(8.0, 14.0, 8.0, 14.0, 1.0)).collect();
        let a = shot_rect(&refs(&tight), 24, FULL);
        let shifted: Vec<RowVote> = (0..40)
            .map(|_| vote((8.3 / 24.0, 14.3 / 24.0, 8.0 / 24.0, 14.0 / 24.0), 1.0))
            .collect();
        let b = shot_rect(&refs(&shifted), 24, FULL);
        assert!(close(b.0 - a.0, 0.3 / 24.0), "moved {} of a cell", (b.0 - a.0) * 24.0);
        assert!(close(b.1, a.1), "the untouched axis must not move");
    }

    /// The zoom is continuous too: between the cap and the identity snap the
    /// old recipe could only reach six sizes (24/16 .. 24/21).
    #[test]
    fn the_zoom_is_not_a_ladder_of_cells() {
        let a: Vec<RowVote> = (0..40).map(|_| cells(4.0, 20.0, 4.0, 20.0, 1.0)).collect();
        let b: Vec<RowVote> = (0..40)
            .map(|_| vote((4.0 / 24.0, 20.4 / 24.0, 4.0 / 24.0, 20.0 / 24.0), 1.0))
            .collect();
        let (za, zb) = (shot_rect(&refs(&a), 24, FULL).2, shot_rect(&refs(&b), 24, FULL).2);
        assert!(close(za, 18.0 / 24.0), "cap-free vote of 18 cells: {za}");
        assert!(close(zb - za, 0.4 / 24.0), "sizes {za} and {zb} are a cell apart");
    }

    /// The floor subtraction is what keeps a corner speck from pinning the
    /// box: at the shipped FLOOR_Q a uniform background is erased, while a
    /// cell genuinely brighter than the background survives it.
    #[test]
    fn the_background_floor_drops_specks_not_bright_cells() {
        let n = 100;
        let mut v: Vec<f64> = vec![0.001; n]; // the uniform background
        v[0] = 0.5; // the ROI peak
        v[99] = 0.02; // a dim speck: brighter than the floor, far from the peak
        let mut s = v.clone();
        s.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let floor = percentile(&s, FLOOR_Q);
        assert!(floor > 0.0 && floor < 0.02, "floor {floor} is not the background");
        let kept: Vec<f64> = v.iter().map(|x| (x - floor).max(0.0)).collect();
        assert_eq!(kept.iter().filter(|&&x| x > 0.0).count(), 2, "background survived");
        assert!(kept[0] > kept[99], "the peak must outlive the speck");
        // and the speck keeps only what it had ABOVE the background
        assert!((kept[99] - (0.02 - floor)).abs() < 1e-12);
    }

    #[test]
    fn wide_votes_snap_to_identity() {
        let votes: Vec<RowVote> = (0..40).map(|_| cells(1.0, 23.0, 0.0, 24.0, 1.0)).collect();
        assert_eq!(shot_rect(&refs(&votes), 24, FULL), IDENTITY);
    }

    #[test]
    fn diffuse_rows_do_not_vote() {
        // half the rows are sharp and tight, half diffuse and full-frame;
        // only the sharp half votes, so the rect stays tight
        let mut votes: Vec<RowVote> = (0..20).map(|_| cells(9.0, 14.0, 9.0, 14.0, 0.9)).collect();
        votes.extend((0..20).map(|_| cells(0.0, 24.0, 0.0, 24.0, 0.1)));
        let (_, _, w, h) = shot_rect(&refs(&votes), 24, FULL);
        assert!(close(w, MIN_SIDE_FRAC) && close(h, MIN_SIDE_FRAC));
    }

    /// A vote near the picture's edge used to centre the cap-lifted rect
    /// onto the bars; the clamp keeps every rect cell on real pixels.
    #[test]
    fn the_rect_stays_inside_the_picture_box() {
        let pic = (0.125, 0.875, 0.125, 0.875); // content on 3/4 of the frame
        // votes hug the content's top-left corner: the capped rect centred on
        // them would start above the picture and used to clamp onto the bars
        let votes: Vec<RowVote> = (0..40).map(|_| cells(5.0, 11.0, 5.0, 11.0, 1.0)).collect();
        let r = shot_rect(&refs(&votes), 24, pic);
        assert!(close(r.0, pic.0) && close(r.1, pic.2), "rect {r:?} is off the picture");
        // and the same clamp bounds the placement search's candidates
        for (x, y, w, h) in placements(r, 24, pic) {
            assert!(x >= pic.0 - 1e-9 && x + w <= pic.1 + 1e-9);
            assert!(y >= pic.2 - 1e-9 && y + h <= pic.3 + 1e-9);
        }
        // a picture NARROWER than the zoom cap's rect has no placement that
        // fits: the rect centres on it and overhangs both bars equally
        let tight = (1.0 / 6.0, 5.0 / 6.0, 1.0 / 6.0, 5.0 / 6.0);
        let r = shot_rect(&refs(&votes), 24, tight);
        assert!(close(r.0, (tight.0 + tight.1 - r.2) / 2.0));
        assert!(close(r.1, (tight.2 + tight.3 - r.3) / 2.0));
    }

    /// A diffuse vote spans the frame THROUGH the bars, which used to read
    /// as "most of the frame" and snap to identity -- on a barred frame the
    /// honest wide answer is the picture box, not the whole frame.
    #[test]
    fn a_wide_vote_crops_to_the_picture_not_identity() {
        let pic = (1.0 / 6.0, 5.0 / 6.0, 1.0 / 6.0, 5.0 / 6.0);
        let votes: Vec<RowVote> = (0..40).map(|_| cells(0.0, 24.0, 0.0, 24.0, 1.0)).collect();
        let r = shot_rect(&refs(&votes), 24, pic);
        assert!(!is_whole(r), "a barred frame is not its own picture");
        assert!(close(r.2, MIN_SIDE_FRAC));
    }

    #[test]
    fn the_picture_box_trims_dead_lines_only() {
        let res = 384;
        let lit = (PIC_DEAD_LUMA + 1) as u8;
        // a letterbox: 40 dead lines top and bottom, nothing at the sides
        let mut rows = vec![lit; res];
        for v in rows.iter_mut().take(40) {
            *v = 0;
        }
        for v in rows.iter_mut().skip(res - 40) {
            *v = 0;
        }
        let cols = vec![lit; res];
        let p = picture_box(&rows, &cols, res);
        assert_eq!(p, (0.0, 1.0, 40.0 / 384.0, 344.0 / 384.0));
        // one line finer than a cell of the deploy grid, which is 16 lines
        let mut rows1 = rows.clone();
        rows1[40] = 0;
        assert_eq!(picture_box(&rows1, &cols, res).2, 41.0 / 384.0);
        // a dead line INSIDE the picture is content, never trimmed
        let mut inner = rows.clone();
        inner[200] = 0;
        assert_eq!(picture_box(&inner, &cols, res), p);
        // a picture the cap's smallest rect cannot fit inside is not one the
        // crop can serve: the whole frame stands in
        let mut tiny = vec![0u8; res];
        for v in tiny.iter_mut().take(res / 2).skip(res / 4) {
            *v = lit;
        }
        assert_eq!(picture_box(&tiny, &cols, res), (0.0, 1.0, 0.0, 1.0));
        // and an all-dead accumulation (the probe saw only black) does too
        assert_eq!(picture_box(&vec![0u8; res], &vec![0u8; res], res), (0.0, 1.0, 0.0, 1.0));
    }

    #[test]
    fn plan_segments_merge_and_map_to_even_pixels() {
        let segs = vec![(0, (4.0 / 24.0, 4.0 / 24.0, 14.0 / 24.0, 14.0 / 24.0)), (900, IDENTITY)];
        let plan =
            Plan { auto: segs.clone(), segs, manual: false, grid: 24, placed: 0, escape_share: 0.0 };
        let segs = plan.segments(854, 480);
        assert_eq!(segs.len(), 2);
        let c = segs[0].1.as_ref().expect("cropped");
        // 14/24 of 854 = 498.2 -> 498; 14/24 of 480 = 280
        assert_eq!(c, "crop=498:280:142:80");
        assert!(segs[1].1.is_none(), "identity segment carries no filter");
        assert!(!plan.is_identity());
        assert!(plan.key().contains("900:0.00000,0.00000,1.00000,1.00000"));
    }

    /// A cached plan carries the recipe that produced it: a plan file from
    /// before the stamp (no `recipe` field) must fail to parse, so
    /// `read_cached` re-probes instead of reusing rects from a retuned
    /// recipe under an unchanged bundle.
    #[test]
    fn pre_stamp_plan_files_do_not_parse() {
        let old = r#"{"checkpoint":"c","epoch":1,"basis_id":"b","grid":24,
                      "escape_share":0.1,"segs":[[0,[4,4,16,16]]]}"#;
        assert!(serde_json::from_str::<PlanFile>(old).is_err());

        let stamped = PlanFile {
            checkpoint: "c".into(),
            epoch: 1,
            basis_id: "b".into(),
            grid: 24,
            recipe: recipe_id(),
            gamma: 1.0,
            escape_share: 0.1,
            segs: vec![(0, (0.1, 0.1, 0.7, 0.7))],
            auto: vec![(0, (0.1, 0.1, 0.7, 0.7))],
            manual: false,
        };
        let back: PlanFile =
            serde_json::from_slice(&serde_json::to_vec(&stamped).unwrap()).unwrap();
        assert_eq!(back.recipe, recipe_id());
        // the stamp reads every recipe constant; spot-check the sub-cell
        // refinement and the zoom cap
        assert!(back.recipe.contains(&format!("u{SUBCELL}")));
        assert!(back.recipe.contains(&format!("s{MIN_SIDE_FRAC}")));
    }

    /// A hand-drawn plan answers to the person who drew it and not to the
    /// recipe: a retuned constant re-probes an automatic plan and leaves a
    /// manual one exactly where it is.
    #[test]
    fn a_hand_drawn_plan_outlives_the_recipe_that_never_made_it() {
        // the fixture manifest, for the same reason `bundle.rs` uses it: a
        // bundle is a build product a source checkout does not have
        let man: Manifest = serde_json::from_slice(
            &std::fs::read(
                std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                    .join("tests/fixtures/manifest.json"),
            )
            .unwrap(),
        )
        .unwrap();
        let dir = std::env::temp_dir().join(format!("gs-croptest-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let segs = vec![(0usize, (0.2, 0.1, 0.5, 0.5))];
        let plan =
            Plan { auto: segs.clone(), segs, manual: true, grid: man.grid, escape_share: 0.0, placed: 0 };
        write_cached(&dir, &man, &plan, 1.0).unwrap();
        // the same run reads it back
        let back = read_cached(&dir, &man, 1.0).expect("a manual plan is its own key");
        assert!(back.manual && back.segs[0].1 .0 == 0.2);
        // and so does a run whose checkpoint, epoch and exposure all moved
        let mut other = man.clone();
        other.checkpoint = "another".into();
        other.epoch += 1;
        assert!(read_cached(&dir, &other, 1.4).is_some());
        // an automatic plan under the same move does not survive
        let auto = Plan { manual: false, ..plan };
        write_cached(&dir, &man, &auto, 1.0).unwrap();
        assert!(read_cached(&dir, &other, 1.0).is_none());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn view_is_fractions_on_the_video_clock() {
        let segs = vec![(0, (4.0 / 24.0, 4.0 / 24.0, 0.5, 0.5)), (900, IDENTITY)];
        let plan =
            Plan { auto: segs.clone(), segs, manual: false, grid: 24, placed: 0, escape_share: 0.05 };
        let v = plan.view(30.0);
        assert_eq!(v.segs.len(), 2);
        assert_eq!(v.segs[0].t_ms, 0.0);
        assert!((v.segs[0].x - 1.0 / 6.0).abs() < 1e-12);
        assert!((v.segs[0].w - 0.5).abs() < 1e-12);
        // frame 900 at 30 fps is 30 s in, and the identity rect is the frame
        assert!((v.segs[1].t_ms - 30_000.0).abs() < 1e-9);
        assert_eq!((v.segs[1].x, v.segs[1].w), (0.0, 1.0));
        assert!((v.zoom - 2.0).abs() < 1e-12); // median_zoom's upper of x1, x2
        assert_eq!(v.escape_share, 0.05);
    }

    /// The escape instrument reads a continuous rect against square cells:
    /// an edge cell counts by the area the rect covers of it.
    #[test]
    fn escaped_mass_counts_edge_cells_by_area() {
        let grid = 4;
        let mut map = vec![0f32; grid * grid];
        map[5] = 0.5; // cell (1,1), fully inside the rect below
        map[6] = 0.5; // cell (2,1), half covered by it
        let r = (0.25, 0.25, 0.375, 0.5); // x 0.25..0.625, y 0.25..0.75
        assert!((inside_mass(&map, grid, r) - 0.75).abs() < 1e-9);
        assert!((inside_mass(&map, grid, IDENTITY) - 1.0).abs() < 1e-9);
    }

    /// One Gaussian blob whose centre sits BETWEEN cell centres, read through
    /// the whole sub-cell path. `grid_check.py` pins the same numbers on the
    /// Python side and reads the CROP_FIXTURE lines below to compare them, so
    /// the two languages cannot refine one attention map differently.
    fn fixture_map() -> Vec<f32> {
        let g = 24usize;
        let mut m = vec![0f32; g * g];
        let mut total = 0f32;
        for y in 0..g {
            for x in 0..g {
                let d = (x as f64 - 11.3).powi(2) + (y as f64 - 9.7).powi(2);
                let v = (-d / (2.0 * 2.5f64.powi(2))).exp() as f32;
                m[y * g + x] = v;
                total += v;
            }
        }
        for v in m.iter_mut() {
            *v /= total;
        }
        m
    }

    #[test]
    fn the_sub_cell_box_is_pinned_across_languages() {
        let m = fixture_map();
        let b = top_mass_box(&m, 24);
        // CROP_FIXTURE box 0.34375000 0.63541667 0.28125000 0.57291667
        let want = [0.34375000, 0.63541667, 0.28125000, 0.57291667];
        for (got, want) in [b.0, b.1, b.2, b.3].iter().zip(want) {
            assert!((got - want).abs() < 1e-6, "box {got} != {want}");
        }
        // an edge landing between cell centres is the whole point: 0.34375 is
        // 8.25 cells, which the cell lattice could not have said
        assert!((b.0 * 24.0 - 8.25).abs() < 1e-6);

        let rows: Vec<RowVote> = (0..40)
            .map(|_| RowVote { shot: 0, bbox: b, conc: 0.224346, map: m.clone() })
            .collect();
        let r = shot_rect(&refs(&rows), 24, FULL);
        // CROP_FIXTURE rect 0.15623333 0.09373333 0.66670000 0.66670000
        let want = [0.15623333, 0.09373333, 0.66670000, 0.66670000];
        for (got, want) in [r.0, r.1, r.2, r.3].iter().zip(want) {
            assert!((got - want).abs() < 1e-6, "rect {got} != {want}");
        }
    }
}
