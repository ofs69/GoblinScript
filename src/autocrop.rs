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
const MARGIN_CELLS: i64 = 1; // extra cells each side of the shot bbox
const MIN_SIDE_FRAC: f64 = 0.6667; // rect side floor (zoom cap x1.5)
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
const SEARCH_KEEP_CONF: f64 = 0.88; // incumbent conf at which a shot is kept
                     // without searching the other candidates: no measured
                     // WINNING move ever came from an incumbent above 0.848
                     // (every move from 0.91+ rode +0.0002..+0.011, one of
                     // them the damaging one), so 0.88 sits in the gap and
                     // spends the 6 candidate encodes only where wins have
                     // ever lived -- on high-conf cutty clips this is where
                     // the probe minutes go

/// One crop rect in grid cells; identity is `(0, 0, grid, grid)`.
pub type Rect = (usize, usize, usize, usize);

pub struct Plan {
    /// (first frame, rect) per segment, consecutive equal rects merged.
    pub segs: Vec<(usize, Rect)>,
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
        self.segs.iter().all(|(_, r)| r.2 == self.grid && r.3 == self.grid)
    }

    /// Median zoom over segments (grid / rect side), for the stage line.
    pub fn median_zoom(&self) -> f64 {
        let mut z: Vec<f64> =
            self.segs.iter().map(|(_, r)| self.grid as f64 / r.3 as f64).collect();
        z.sort_by(|a, b| a.partial_cmp(b).unwrap());
        z[z.len() / 2]
    }

    /// The decode segments: rect -> "W:H:X:Y" in source pixels of the `w`x`h`
    /// normalized clip (even-rounded for the codec; identity -> no filter).
    pub fn segments(&self, w: usize, h: usize) -> Vec<(usize, Option<String>)> {
        self.segs
            .iter()
            .map(|&(start, rect)| (start, crop_arg(rect, self.grid, w, h)))
            .collect()
    }

    /// What the review page draws. `fps` is the FRAME grid the segment starts
    /// are counted on (`Manifest::grid_fps`), so the times come out on the
    /// VIDEO clock the page already plays against.
    pub fn view(&self, fps: f64) -> View {
        let g = self.grid as f64;
        View {
            segs: self
                .segs
                .iter()
                .map(|&(f, (x, y, w, h))| ViewSeg {
                    t_ms: f as f64 / fps * 1000.0,
                    x: x as f64 / g,
                    y: y as f64 / g,
                    w: w as f64 / g,
                    h: h as f64 / g,
                })
                .collect(),
            zoom: self.median_zoom(),
            escape_share: self.escape_share,
        }
    }

    /// A stable identity for the plan, part of the latent cache key: cached
    /// latents from a different plan (or none) must not be reused.
    pub fn key(&self) -> String {
        let s: Vec<String> = self
            .segs
            .iter()
            .map(|(f, (x, y, w, h))| format!("{f}:{x},{y},{w},{h}"))
            .collect();
        format!("crop-v1[{}]", s.join(";"))
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
    bbox: (usize, usize, usize, usize), // x0, x1, y0, y1 (exclusive)
    conc: f32,
    map: Vec<f32>, // mixed attention, grid*grid, sums to 1 (escape instrument)
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

    let cut_frames: Vec<usize> = cuts_ms
        .iter()
        .map(|&c| (c / 1000.0 * man.grid_fps).round() as usize)
        .collect();
    let shot_of = |frame: usize| cut_frames.partition_point(|&c| c <= frame);

    let mut x: Vec<u8> = Vec::with_capacity((man.clip_len / 2) * 2 * res * res * 3);
    let mut slab: Vec<i8> = vec![0; group * row_bytes];
    let mut votes: Vec<RowVote> = Vec::new();
    let mut cell_max = vec![0u8; grid * grid];

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
            accum_cell_max(&frames[0], res, grid, &mut cell_max);
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
    let pic = picture_box(&cell_max, grid);
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
            if rect.2 == grid {
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
                    crate::exposure::join_filters(photo, crop_arg(cand, grid, vw, vh).as_deref());
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
        .filter(|v| {
            let (x0, y0, w, h) = rects[v.shot];
            let inside: f32 = (y0..y0 + h)
                .flat_map(|y| (x0..x0 + w).map(move |x| v.map[y * grid + x]))
                .sum();
            1.0 - inside > 0.2
        })
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
        segs,
        grid,
        escape_share: escaped as f64 / votes.len() as f64,
        placed: moved,
    })
}

/// One rect in grid cells -> the decode-chain `crop` filter in pixels of a
/// `w`x`h` picture, even-rounded for the codec. `None` for the identity rect,
/// which is no filter at all. The one place cells become pixels, so the plan
/// the probe searches and the plan the decode applies cannot drift apart.
fn crop_arg(rect: Rect, grid: usize, w: usize, h: usize) -> Option<String> {
    let (x0, y0, rw, rh) = rect;
    if rw == grid && rh == grid {
        return None;
    }
    let cw = ((rw * w) as f64 / grid as f64 / 2.0).round() as usize * 2;
    let ch = ((rh * h) as f64 / grid as f64 / 2.0).round() as usize * 2;
    let cx = (((x0 * w) as f64 / grid as f64).round() as usize).min(w - cw);
    let cy = (((y0 * h) as f64 / grid as f64).round() as usize).min(h - ch);
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

/// Per-cell running maximum of the frame's brightest channel byte -- the
/// evidence the picture box is read from. Bars stay at codec noise across
/// every sample; any content cell clears them the first time it is lit.
fn accum_cell_max(frame: &[u8], res: usize, grid: usize, cell_max: &mut [u8]) {
    let block = res / grid;
    for py in 0..res {
        let cy = (py / block).min(grid - 1);
        let row = &frame[py * res * 3..(py + 1) * res * 3];
        for px in 0..res {
            let p = &row[px * 3..px * 3 + 3];
            let v = p[0].max(p[1]).max(p[2]);
            let c = &mut cell_max[cy * grid + (px / block).min(grid - 1)];
            if v > *c {
                *c = v;
            }
        }
    }
}

/// The PICTURE box `(x0, x1, y0, y1)` in cells, exclusive: the frame minus
/// its dead edge rows/columns (letterbox and pillarbox bars). Dead pixels
/// carry nothing, so no rect has business covering them -- but a "picture"
/// too small for the zoom cap's smallest rect is not one the cap can serve,
/// and the whole frame stands in for it.
fn picture_box(cell_max: &[u8], grid: usize) -> (i64, i64, i64, i64) {
    let dead_row = |y: usize| (0..grid).all(|x| (cell_max[y * grid + x] as usize) <= PIC_DEAD_LUMA);
    let dead_col = |x: usize| (0..grid).all(|y| (cell_max[y * grid + x] as usize) <= PIC_DEAD_LUMA);
    let (mut x0, mut x1, mut y0, mut y1) = (0usize, grid, 0usize, grid);
    while y0 < y1 && dead_row(y0) {
        y0 += 1;
    }
    while y1 > y0 && dead_row(y1 - 1) {
        y1 -= 1;
    }
    while x0 < x1 && dead_col(x0) {
        x0 += 1;
    }
    while x1 > x0 && dead_col(x1 - 1) {
        x1 -= 1;
    }
    let min_side = (grid as f64 * MIN_SIDE_FRAC).round() as usize;
    if x1 - x0 < min_side || y1 - y0 < min_side {
        return (0, grid as i64, 0, grid as i64);
    }
    (x0 as i64, x1 as i64, y0 as i64, y1 as i64)
}

/// Clamp a rect start so `[start, start+side)` sits inside the picture span
/// `[p0, p1)` -- centered overhang when the side outgrows the span -- and
/// always inside the grid.
fn clamp_into(start: i64, side: i64, p0: i64, p1: i64, grid: i64) -> i64 {
    let hi = p1 - side;
    let v = if hi < p0 { (p0 + p1 - side) / 2 } else { start.clamp(p0, hi) };
    v.clamp(0, grid - side)
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
fn placements(rect: Rect, grid: usize, pic: (i64, i64, i64, i64)) -> Vec<Rect> {
    let (x, y, w, h) = rect;
    let mut out = vec![rect];
    for (dx, dy) in [(-2i64, 0i64), (2, 0), (0, -2), (0, 2), (-2, -2), (2, 2)] {
        let nx = clamp_into(x as i64 + dx, w as i64, pic.0, pic.1, grid as i64) as usize;
        let ny = clamp_into(y as i64 + dy, h as i64, pic.2, pic.3, grid as i64) as usize;
        if !out.contains(&(nx, ny, w, h)) {
            out.push((nx, ny, w, h));
        }
    }
    out
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

    // tight bbox: smallest descending-value cell set holding TOP_MASS_Q
    let mut order: Vec<usize> = (0..cells).collect();
    order.sort_by(|&a, &b| map[b].partial_cmp(&map[a]).unwrap());
    let (mut x0, mut x1, mut y0, mut y1) = (grid, 0usize, grid, 0usize);
    let mut mass = 0f32;
    let mut conc = 0f32;
    for (i, &c) in order.iter().enumerate() {
        if i < CONC_CELLS {
            conc += map[c];
        }
        if mass >= TOP_MASS_Q {
            continue;
        }
        mass += map[c];
        let (cx, cy) = (c % grid, c / grid);
        x0 = x0.min(cx);
        x1 = x1.max(cx + 1);
        y0 = y0.min(cy);
        y1 = y1.max(cy + 1);
    }
    Ok(Some(RowVote { shot, bbox: (x0, x1, y0, y1), conc, map }))
}

/// Rows of one shot (or the whole clip) -> its rect: edges voted at EDGE_Q
/// percentiles by the sharper half of rows, clamped into the picture box
/// (attention on dead bars is floor noise, and clamping is also what keeps a
/// diffuse vote from reaching the identity snap through the bars), a margin,
/// the zoom cap, and an identity snap when what is left is most of the frame
/// anyway.
fn shot_rect(votes: &[&RowVote], grid: usize, pic: (i64, i64, i64, i64)) -> Rect {
    let (x0, x1, y0, y1) = vote_edges(votes);
    let (x0, x1) = (x0.max(pic.0), x1.min(pic.1));
    let (y0, y1) = (y0.max(pic.2), y1.min(pic.3));
    let g = grid as i64;
    let side = (x1 - x0)
        .max(y1 - y0)
        .max((grid as f64 * MIN_SIDE_FRAC).round() as i64);
    if side as f64 >= grid as f64 * IDENT_FRAC {
        return (0, 0, grid, grid);
    }
    let cx = (x0 + x1) as f64 / 2.0;
    let cy = (y0 + y1) as f64 / 2.0;
    let rx = clamp_into((cx - side as f64 / 2.0).round() as i64, side, pic.0, pic.1, g);
    let ry = clamp_into((cy - side as f64 / 2.0).round() as i64, side, pic.2, pic.3, g);
    (rx as usize, ry as usize, side as usize, side as usize)
}

/// The voted edges of a set of rows: the sharper half decides, at the EDGE_Q
/// percentiles, plus the safety margin. The one definition of "where the
/// attention is", read once and used by every rect the plan emits.
fn vote_edges(votes: &[&RowVote]) -> (i64, i64, i64, i64) {
    let mut conc: Vec<f32> = votes.iter().map(|v| v.conc).collect();
    conc.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let med = conc[conc.len() / 2];
    let sharp: Vec<&&RowVote> = votes.iter().filter(|v| v.conc >= med).collect();

    let col = |f: fn(&RowVote) -> usize| -> Vec<f64> {
        let mut v: Vec<f64> = sharp.iter().map(|r| f(r) as f64).collect();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v
    };
    (
        percentile(&col(|r| r.bbox.0), EDGE_Q) as i64 - MARGIN_CELLS,
        percentile(&col(|r| r.bbox.1), 100.0 - EDGE_Q) as i64 + MARGIN_CELLS,
        percentile(&col(|r| r.bbox.2), EDGE_Q) as i64 - MARGIN_CELLS,
        percentile(&col(|r| r.bbox.3), 100.0 - EDGE_Q) as i64 + MARGIN_CELLS,
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
         sk{SEARCH_KEEP_CONF}"
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
}

pub fn read_cached(dir: &Path, man: &Manifest, gamma: f64) -> Option<Plan> {
    let p: PlanFile =
        serde_json::from_slice(&std::fs::read(dir.join("autocrop.json")).ok()?).ok()?;
    (p.checkpoint == man.checkpoint
        && p.epoch == man.epoch
        && p.basis_id == man.basis_id
        && p.grid == man.grid
        && p.recipe == recipe_id()
        && p.gamma == gamma)
        .then_some(Plan {
            placed: 0,
            segs: p.segs,
            grid: p.grid,
            escape_share: p.escape_share,
        })
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
    };
    std::fs::write(dir.join("autocrop.json"), serde_json::to_vec_pretty(&f)?)
        .context("could not write the auto-crop plan")?;
    Ok(())
}

/// The stage's done line: what the crop decided, and its instrument.
pub fn stage_line(plan: &Plan, reused: bool) -> String {
    let tag = if reused { crate::t!("console.crop.reused") } else { "" };
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

    fn vote(bbox: (usize, usize, usize, usize), conc: f32) -> RowVote {
        RowVote { shot: 0, bbox, conc, map: Vec::new() }
    }

    fn refs(v: &[RowVote]) -> Vec<&RowVote> {
        v.iter().collect()
    }

    const FULL: (i64, i64, i64, i64) = (0, 24, 0, 24);

    #[test]
    fn tight_agreeing_rows_crop_with_margin_and_floor() {
        // every sharp row says cells 8..14 (6 wide): margin adds one each
        // side (8), the zoom cap lifts it to round(24 * 0.6667) = 16
        let votes: Vec<RowVote> = (0..40).map(|_| vote((8, 14, 8, 14), 1.0)).collect();
        let (x0, y0, w, h) = shot_rect(&refs(&votes), 24, FULL);
        assert_eq!((w, h), (16, 16));
        // centered on the vote (cells 7..15 center 11), clamped in-grid
        assert!(x0 <= 7 && x0 + w >= 15, "rect {x0}..{} misses the vote", x0 + w);
        assert_eq!((x0, y0), (3, 3));
    }

    /// The floor subtraction is what keeps a corner speck from pinning the
    /// box: at the shipped FLOOR_Q a uniform background is erased, while a
    /// cell genuinely brighter than the background survives it.
    #[test]
    fn the_background_floor_drops_specks_not_bright_cells() {
        let cells = 100;
        let mut v: Vec<f64> = vec![0.001; cells]; // the uniform background
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
        let votes: Vec<RowVote> = (0..40).map(|_| vote((1, 23, 0, 24), 1.0)).collect();
        assert_eq!(shot_rect(&refs(&votes), 24, FULL), (0, 0, 24, 24));
    }

    #[test]
    fn diffuse_rows_do_not_vote() {
        // half the rows are sharp and tight, half diffuse and full-frame;
        // only the sharp half votes, so the rect stays tight
        let mut votes: Vec<RowVote> = (0..20).map(|_| vote((9, 14, 9, 14), 0.9)).collect();
        votes.extend((0..20).map(|_| vote((0, 24, 0, 24), 0.1)));
        let (_, _, w, h) = shot_rect(&refs(&votes), 24, FULL);
        assert_eq!((w, h), (16, 16)); // the tight vote, lifted by the zoom cap
    }

    /// A vote near the picture's edge used to centre the cap-lifted rect
    /// onto the bars; the clamp keeps every rect cell on real pixels.
    #[test]
    fn the_rect_stays_inside_the_picture_box() {
        let pic = (4i64, 20i64, 4i64, 20i64); // content at 2/3, centred
        // votes hug the content's top-left corner: the 16-cell rect centred
        // on them would start at negative cells and clamp to the frame,
        // covering 4 columns and rows of bar
        let votes: Vec<RowVote> = (0..40).map(|_| vote((5, 11, 5, 11), 1.0)).collect();
        assert_eq!(shot_rect(&refs(&votes), 24, pic), (4, 4, 16, 16));
        // and the same clamp bounds the placement search's candidates
        for (x, y, w, h) in placements((4, 4, 16, 16), 24, pic) {
            assert!(x as i64 >= pic.0 && (x + w) as i64 <= pic.1);
            assert!(y as i64 >= pic.2 && (y + h) as i64 <= pic.3);
        }
    }

    /// A diffuse vote spans the frame THROUGH the bars, which used to read
    /// as "most of the frame" and snap to identity -- on a barred frame the
    /// honest wide answer is the picture box, not the whole frame.
    #[test]
    fn a_wide_vote_crops_to_the_picture_not_identity() {
        let pic = (4i64, 20i64, 4i64, 20i64);
        let votes: Vec<RowVote> = (0..40).map(|_| vote((0, 24, 0, 24), 1.0)).collect();
        assert_eq!(shot_rect(&refs(&votes), 24, pic), (4, 4, 16, 16));
    }

    #[test]
    fn picture_box_trims_dead_edges_only() {
        let grid = 24;
        let lit = (PIC_DEAD_LUMA + 1) as u8;
        // the padded layout: content cells 4..20 on both axes
        let mut cells = vec![0u8; grid * grid];
        for y in 4..20 {
            for x in 4..20 {
                cells[y * grid + x] = lit;
            }
        }
        assert_eq!(picture_box(&cells, grid), (4, 20, 4, 20));
        // a dark region INSIDE the picture is content, never trimmed
        for x in 4..20 {
            cells[10 * grid + x] = 0;
        }
        assert_eq!(picture_box(&cells, grid), (4, 20, 4, 20));
        // a barless frame is its own picture
        let full = vec![lit; grid * grid];
        assert_eq!(picture_box(&full, grid), (0, 24, 0, 24));
        // a picture the cap's smallest rect cannot fit inside is not one the
        // crop can serve: the whole frame stands in
        let mut tiny = vec![0u8; grid * grid];
        for y in 8..20 {
            for x in 8..20 {
                tiny[y * grid + x] = lit;
            }
        }
        assert_eq!(picture_box(&tiny, grid), (0, 24, 0, 24));
        // and an all-dead accumulation (probe saw only black) does too
        assert_eq!(picture_box(&vec![0u8; grid * grid], grid), (0, 24, 0, 24));
    }

    #[test]
    fn plan_segments_merge_and_map_to_even_pixels() {
        let plan = Plan {
            segs: vec![(0, (4, 4, 14, 14)), (900, (0, 0, 24, 24))],
            grid: 24,
            placed: 0,
            escape_share: 0.0,
        };
        let segs = plan.segments(854, 480);
        assert_eq!(segs.len(), 2);
        let c = segs[0].1.as_ref().expect("cropped");
        // 14/24 of 854 = 498.2 -> 498; 14/24 of 480 = 280
        assert_eq!(c, "crop=498:280:142:80");
        assert!(segs[1].1.is_none(), "identity segment carries no filter");
        assert!(!plan.is_identity());
        assert!(plan.key().contains("900:0,0,24,24"));
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
            segs: vec![(0, (4, 4, 16, 16))],
        };
        let back: PlanFile =
            serde_json::from_slice(&serde_json::to_vec(&stamped).unwrap()).unwrap();
        assert_eq!(back.recipe, recipe_id());
        // the stamp reads every recipe constant; spot-check the two that
        // were retuned the day the gap was found
        assert!(back.recipe.contains(&format!("f{FLOOR_Q}")));
        assert!(back.recipe.contains(&format!("s{MIN_SIDE_FRAC}")));
    }

    #[test]
    fn view_is_fractions_on_the_video_clock() {
        let plan = Plan {
            segs: vec![(0, (4, 4, 12, 12)), (900, (0, 0, 24, 24))],
            grid: 24,
            placed: 0,
            escape_share: 0.05,
        };
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
}
