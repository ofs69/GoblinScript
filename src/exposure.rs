//! Exposure: one corrective gamma per clip, estimated from the picture.
//!
//! The corpus the model learned from occupies a luma BAND (median-of-frame-
//! medians 0.28..0.65 normalized, p10..p90 over 60 strided clips), and a
//! clip inside it is the model's
//! ordinary weather: naturally dark scenes live there and are left alone. A
//! clip OUTSIDE the band -- a crushed master, a blown transfer -- is the
//! measured failure case, and one gamma brings it TO the band's edge:
//! minimal intervention, since the band edge is a legitimate corpus
//! operating point, and mapping to the band CENTER instead was measured to
//! damage a real near-band corpus clip (-0.09 corr from gamma 2.0 on a
//! 0.25-median clip). Deep degradations clamp to the measured-safe x2.0
//! envelope (the oracle restore the benchmark validated).
//!
//! The correction is a filter stage in the decode chains that feed the MODEL
//! (the auto-crop probe and the encode), exactly like the crop and the
//! soften: no re-encode, no clock, no second file. An in-band clip gets
//! gamma 1.0, which is no filter at all -- byte-identical to a run without
//! the stage.

use anyhow::Result;
use std::path::Path;

use crate::ffmpeg::Decoder;

const SAMPLES: usize = 9; // frames sampled, spread across the clip
const LUMA_LO: f64 = 0.28; // the corpus band (p10, probes/luma_ref.py)
const LUMA_HI: f64 = 0.65; // (p90)
const GAMMA_DEADZONE: f64 = 0.15; // |gamma - 1| below this is left alone: a
                     // clip NEAR the band is corpus weather, and gamma 2.0
                     // on a real 0.25-median corpus clip measured -0.09 corr
                     // (the to-center rule this replaced)
const GAMMA_MIN: f64 = 0.5; // the measured-safe envelope: the oracle x2.0
const GAMMA_MAX: f64 = 2.0; // restore is the strongest correction validated

/// The verdict: the clip's sampled median luma, and the one gamma the decode
/// chains apply (1.0 = in band, nothing applied).
pub struct Exposure {
    pub gamma: f64,
    pub median: f64,
}

impl Exposure {
    /// The `eq` stage for a decode filter chain, `None` when in band.
    pub fn filter(&self) -> Option<String> {
        ((self.gamma - 1.0).abs() > 1e-3).then(|| format!("eq=gamma={:.3}", self.gamma))
    }

    pub fn stage_line(&self) -> String {
        let luma = format!("{:.2}", self.median);
        if self.filter().is_none() {
            crate::t!("console.expo.inband", luma = luma)
        } else {
            crate::t!(
                "console.expo.gamma",
                luma = luma,
                gamma = format!("{:.2}", self.gamma)
            )
        }
    }
}

/// Median rec601 luma of one rgb24 frame, ignoring dead bar rows/columns
/// (the same dead-luma rule the picture box uses -- a letterboxed dark clip
/// must not read darker for its bars).
fn frame_median(frame: &[u8], res: usize) -> Option<f64> {
    let mut luma = vec![0f32; res * res];
    for (i, px) in frame.chunks_exact(3).enumerate() {
        luma[i] = 0.299 * px[0] as f32 + 0.587 * px[1] as f32 + 0.114 * px[2] as f32;
    }
    let dead = crate::autocrop::PIC_DEAD_LUMA as f32;
    let live_row: Vec<bool> = (0..res)
        .map(|y| luma[y * res..(y + 1) * res].iter().any(|&v| v > dead))
        .collect();
    let live_col: Vec<bool> = (0..res)
        .map(|x| (0..res).any(|y| luma[y * res + x] > dead))
        .collect();
    let mut vals: Vec<f32> = Vec::with_capacity(res * res);
    for y in (0..res).filter(|&y| live_row[y]) {
        for x in (0..res).filter(|&x| live_col[x]) {
            vals.push(luma[y * res + x]);
        }
    }
    if vals.is_empty() {
        return None; // an all-dead frame (a fade) has no opinion
    }
    vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
    Some(vals[vals.len() / 2] as f64 / 255.0)
}

/// Sample the clip -> its exposure verdict. Reads the same file every other
/// stage decodes, one frame per sample point.
pub fn probe(video: &Path, res: usize, fps: f64, n_frames: usize) -> Result<Exposure> {
    let last = n_frames.saturating_sub(1);
    let mut meds: Vec<f64> = Vec::with_capacity(SAMPLES);
    for i in 0..SAMPLES {
        crate::cancel::check()?;
        // 2%..98% of the clip, matching the python reference measurement
        let frame = ((0.02 + 0.96 * i as f64 / (SAMPLES - 1) as f64) * last as f64) as usize;
        let mut dec = Decoder::open_at(video, res, res, fps, None, frame, None)?;
        let mut f = Vec::new();
        if dec.next_frame(&mut f)? {
            if let Some(m) = frame_median(&f, res) {
                meds.push(m);
            }
        }
    }
    if meds.is_empty() {
        return Ok(Exposure { gamma: 1.0, median: 0.5 }); // no picture, no opinion
    }
    meds.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let m = meds[meds.len() / 2].clamp(0.01, 0.99);
    // The correction brings an outlier TO the corpus band, never further:
    // an in-band target equals the median (gamma exactly 1), a near-band
    // clip's small gamma dies in the dead-zone, and a far-out clip clamps
    // to the measured-safe envelope. Mapping to the band CENTER instead was
    // measured to damage a real 0.25-median corpus clip.
    let target = m.clamp(LUMA_LO, LUMA_HI);
    let mut gamma = (m.ln() / target.ln()).clamp(GAMMA_MIN, GAMMA_MAX);
    if (gamma - 1.0).abs() < GAMMA_DEADZONE {
        gamma = 1.0;
    }
    Ok(Exposure { gamma, median: m })
}

/// Two optional filter stages -> one chain (the exposure `eq` composed with
/// a crop or soften), comma-joined in the order given.
pub fn join_filters(a: Option<&str>, b: Option<&str>) -> Option<String> {
    match (a, b) {
        (Some(a), Some(b)) => Some(format!("{a},{b}")),
        (Some(a), None) => Some(a.to_string()),
        (None, Some(b)) => Some(b.to_string()),
        (None, None) => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn in_band_is_no_filter() {
        let e = Exposure { gamma: 1.0, median: 0.5 };
        assert!(e.filter().is_none());
    }

    fn rule(m: f64) -> f64 {
        let target = m.clamp(LUMA_LO, LUMA_HI);
        let g: f64 = (m.ln() / target.ln()).clamp(GAMMA_MIN, GAMMA_MAX);
        if (g - 1.0).abs() < GAMMA_DEADZONE { 1.0 } else { g }
    }

    #[test]
    fn the_band_edge_rule_spares_the_corpus_and_clamps_the_crushed() {
        // in band: gamma exactly 1 by construction
        assert_eq!(rule(0.40), 1.0);
        // the real 0.25-median corpus clip that gamma 2.0 damaged: its
        // to-edge gamma (x1.09) dies in the dead-zone -- untouched
        assert_eq!(rule(0.25), 1.0);
        // the benchmark's deep darken (0.02): clamped to the oracle x2.0
        assert!((rule(0.02) - 2.0).abs() < 1e-9);
        // a moderate darken fires a moderate, sub-clamp gamma
        let g = rule(0.14);
        assert!(g > 1.5 && g < 1.6, "moderate correction, got {g}");
    }

    #[test]
    fn bars_do_not_darken_the_reading() {
        let res = 8;
        let mut frame = vec![0u8; res * res * 3]; // black bars everywhere
        // picture rows 2..6, cols 2..6 at mid grey
        for y in 2..6 {
            for x in 2..6 {
                for c in 0..3 {
                    frame[(y * res + x) * 3 + c] = 128;
                }
            }
        }
        let m = frame_median(&frame, res).unwrap();
        assert!((m - 128.0 / 255.0).abs() < 1e-6, "bars leaked into the median: {m}");
        // an all-dead frame abstains instead of reading as black
        assert!(frame_median(&vec![0u8; res * res * 3], res).is_none());
    }

    #[test]
    fn filters_join_in_order() {
        assert_eq!(
            join_filters(Some("eq=gamma=1.900"), Some("crop=1:2:3:4")).unwrap(),
            "eq=gamma=1.900,crop=1:2:3:4"
        );
        assert_eq!(join_filters(None, Some("x")).unwrap(), "x");
        assert_eq!(join_filters(Some("x"), None).unwrap(), "x");
        assert!(join_filters(None, None).is_none());
    }
}
