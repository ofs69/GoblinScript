//! Composed styling: the head's tracks -> a funscript.
//!
//! Each head contributes the axis it is best at. Reversal TIMES come from the
//! marginal velocity's zero crossings (never relaxed, never smoothed away),
//! stroke LEVEL from the position head, stroke AMPLITUDE from the generated
//! envelope, and stroke ENDPOINTS snap to the predicted band edges (scripters
//! tap the extremes at reversals; level +- envelope/2 stops short of them).
//!
//! This is a port of `jepa_infer.style_positions_composed` and it is a port on
//! purpose -- including numpy's edge conventions. `np.convolve(..., "same")`
//! pads with ZEROS, so the first and last few rows of a smoothed track are
//! genuinely pulled toward zero; that artifact is in every draft the model has
//! ever been judged on. Reproducing the intent instead of the arithmetic would
//! quietly make this a different styler.

/// `np.convolve(x, np.ones(k)/k, mode="same")` -- box smoother, ZERO-padded at
/// the edges (not edge-padded: the difference shows up in the first and last
/// k/2 rows, and the Python it mirrors has it too).
fn box_same(x: &[f64], k: usize) -> Vec<f64> {
    let n = x.len();
    let half = k / 2;
    let mut out = vec![0.0; n];
    for (i, o) in out.iter_mut().enumerate() {
        let mut acc = 0.0;
        for j in 0..k {
            // full convolution index, then take the centered window
            let idx = i as isize + j as isize - half as isize;
            if idx >= 0 && (idx as usize) < n {
                acc += x[idx as usize];
            }
        }
        *o = acc / k as f64;
    }
    out
}

/// Styling durations in SECONDS -- mirrors of common.py's set, value for
/// value. `rows_at` restates them on the bundle's row grid, which is the only
/// place a row count is formed. Keep these in step with common.py: the two
/// decoders are only the same decoder while these agree.
const DECODE_SMOOTH_S: f64 = 1.0; // level + band-rail low-pass
const LOCK_SMOOTH_S: f64 = 0.466667; // level-lock local mean
const LOCK_EDGE_S: f64 = 0.2; // lock ramp seconds per side
const DWELL_GAP_S: f64 = 0.266667; // dwell-call hole-fill
const DWELL_MIN_CALL_S: f64 = 0.266667; // shortest surviving call

/// `common.rows_at`: a duration as a row count on a given grid (>= 1 row);
/// `odd` for symmetric kernels.
fn rows_at(seconds: f64, row_hz: f64, odd: bool) -> usize {
    let n = (seconds * row_hz).round().max(1.0) as usize;
    if odd { n | 1 } else { n }
}

fn sign(x: f64) -> f64 {
    if x > 0.0 {
        1.0
    } else if x < 0.0 {
        -1.0
    } else {
        0.0
    }
}

/// Sign flips of the 3-frame-smoothed track: where a stroke reverses.
fn crossings(s: &[f64]) -> Vec<usize> {
    let mut sg: Vec<f64> = s.iter().map(|&v| sign(v)).collect();
    for v in sg.iter_mut() {
        if *v == 0.0 {
            *v = 1.0; // numpy: sg[sg == 0] = 1
        }
    }
    (1..s.len()).filter(|&i| sg[i] * sg[i - 1] < 0.0).collect()
}

/// Most-likely ALTERNATING reversal sequence from the rev head's per-row
/// probabilities -> `(rows, kinds)`, kind +1 peak / -1 valley. Port of
/// `common._alternating_gapped`.
///
/// A funscript alternates by construction, so that structure IS the decode:
/// a 3-state Viterbi over the head's own classes with same-type transitions
/// forbidden -- not a probability threshold. `g` is the refractory in rows;
/// without it a fitted prior's surplus lands as adjacent peak/valley pairs no
/// author writes (33 ms half-strokes at 30 rows/s), chopping sustained fast
/// sections into stubs. State = (kind of last event, rows since it, saturated
/// at `g`), so the count is `1 + 2 * (g + 1)`. Exact, one pass.
///
/// The prior is PER ROW, which is what a band-fitted bias needs -- the Python
/// broadcasts `bias[band_of]` into the same argument, so one row's prior
/// depends on which band it is in. A constant prior is the same array filled.
fn alternating_events_rows(
    p_peak: &[f64],
    p_valley: &[f64],
    bias: &[f64],
    g: usize,
) -> (Vec<usize>, Vec<i8>) {
    let t_len = p_peak.len();
    if t_len == 0 {
        return (Vec::new(), Vec::new());
    }
    let n_a = g + 1; // ages 0..g-1, then FREE at index g
    let ns = 1 + 2 * n_a;
    let free = g;
    let neg = f64::NEG_INFINITY;
    let base = |k: usize| 1 + k * n_a;

    // NaN reads as 0 (numpy's nan_to_num) before the clamp, matching the
    // Python, which nan_to_num's the tracks at the call site.
    let clamp = |v: f64| {
        let x = if v.is_finite() { v } else { 0.0 };
        x.clamp(1e-9, 1.0)
    };
    let mut score = vec![neg; ns];
    score[0] = 0.0;
    let mut back = vec![0u16; t_len * ns];
    let mut emit = vec![0i8; t_len * ns];

    // The row's three working arrays live OUTSIDE the loop: `next` is swapped
    // with `score` at the end of each row, and the back-pointer and emission
    // arrays are written straight into their row of the tables rather than
    // built beside them and copied in. All three used to be allocated per row,
    // which on a clip-length fit window is millions of allocations per decode --
    // and the emission-prior fit runs this decode dozens of times over.
    let mut next = vec![neg; ns];
    for i in 0..t_len {
        let pk = clamp(p_peak[i]);
        let vl = clamp(p_valley[i]);
        let nn = clamp(1.0 - pk - vl);
        let bi = bias[i];
        let (lpk, lvl_, lnn) = (pk.ln() + bi, vl.ln() + bi, nn.ln());
        next.fill(neg);
        let (nb, ne) = (&mut back[i * ns..(i + 1) * ns], &mut emit[i * ns..(i + 1) * ns]);
        // emit nothing: every state ages one row (FREE absorbs)
        next[0] = score[0] + lnn;
        nb[0] = 0;
        for k in 0..2 {
            let b0 = base(k);
            for a in 1..n_a {
                next[b0 + a] = score[b0 + a - 1] + lnn;
                nb[b0 + a] = (b0 + a - 1) as u16;
            }
            let stay = score[b0 + free] + lnn;
            if stay > next[b0 + free] {
                next[b0 + free] = stay;
                nb[b0 + free] = (b0 + free) as u16;
            }
        }
        // emit: legal from "none yet" and from the OPPOSITE kind at FREE age
        for (k, lp) in [(0usize, lpk), (1usize, lvl_)] {
            let src_free = base(1 - k) + free;
            let src = if score[0] >= score[src_free] { 0 } else { src_free };
            let c = score[0].max(score[src_free]) + lp;
            if c > next[base(k)] {
                next[base(k)] = c;
                nb[base(k)] = src as u16;
                ne[base(k)] = if k == 0 { 1 } else { 2 };
            }
        }
        std::mem::swap(&mut score, &mut next);
    }

    let mut s = 0usize;
    let mut best = neg;
    for (i, &v) in score.iter().enumerate() {
        if v > best {
            best = v;
            s = i;
        }
    }
    let mut rows = Vec::new();
    let mut kinds = Vec::new();
    for i in (0..t_len).rev() {
        let e = emit[i * ns + s];
        if e != 0 {
            rows.push(i);
            kinds.push(if e == 1 { 1i8 } else { -1i8 });
        }
        s = back[i * ns + s] as usize;
    }
    rows.reverse();
    kinds.reverse();
    (rows, kinds)
}

/// Emission prior (log-odds) whose MAP decode emits about `target` events on
/// the rows given. Port of `common.fit_emission_bias`: monotone in bias, so a
/// bisection is exact to the resolution of the event count.
///
/// The prior is NOT a tuned knob and NOT a constant. The MAP sequence's event
/// RATE is set by the head's class prior, which no training weighting
/// calibrates for emission, so it is fitted per clip. Deploy has no script to
/// fit against, so the target is the head's OWN probability mass over the fit
/// window -- the count it already believes it is seeing.
fn fit_emission_bias(p_peak: &[f64], p_valley: &[f64], target: usize, g: usize) -> f64 {
    let (mut lo, mut hi) = (-6.0f64, 6.0f64);
    // One prior array, refilled per probe. A clip-length fit window makes this
    // a megabyte, and the bisection asks two dozen questions of it.
    let mut bias = vec![0.0f64; p_peak.len()];
    for _ in 0..24 {
        let mid = 0.5 * (lo + hi);
        bias.fill(mid);
        let n = alternating_events_rows(p_peak, p_valley, &bias, g).0.len();
        if n < target {
            lo = mid;
        } else {
            hi = mid;
        }
        if hi - lo < 1e-3 {
            break;
        }
    }
    0.5 * (lo + hi)
}

/// One emission prior PER BAND, fitted so the MAP decode emits about
/// `targets[k]` events on band `k`'s rows. Port of
/// `common.fit_emission_bias_bands`.
///
/// A single global prior provably misplaces a global surplus: the refractory
/// pins the fast band's emitted rate, so whatever the head's sum is over
/// overflows into the rows where insertions are most jarring. An unscripted
/// GAP is the sharpest case of that -- the head finds real reversals there and
/// the script asserts nothing -- so fitting still and moving rows separately
/// stops gap mass inflating the prior that governs moving rows.
///
/// Coordinate bisection, `rounds` passes: each coordinate is monotone (more
/// bias in a band -> weakly more events in it), and the coupling through the
/// alternation constraint is what the second pass absorbs.
fn fit_emission_bias_bands(
    p_peak: &[f64],
    p_valley: &[f64],
    band_of: &[usize],
    targets: &[usize],
    g: usize,
) -> Vec<f64> {
    let mut b = vec![0.0f64; targets.len()];
    // The per-row prior, refilled in place rather than rebuilt. This is the
    // hottest allocation in the whole re-style: two rounds x one band each x
    // sixteen bisection steps is up to 64 probes, and on a long video each one
    // was a fresh clip-length array behind a decode that is already clip-length.
    let mut rows = vec![0.0f64; band_of.len()];
    for _ in 0..2 {
        for k in 0..targets.len() {
            let (mut lo_k, mut hi_k) = (-6.0f64, 6.0f64);
            for _ in 0..16 {
                let mid = 0.5 * (lo_k + hi_k);
                b[k] = mid;
                for (r, &j) in rows.iter_mut().zip(band_of) {
                    *r = b[j];
                }
                let (ev, _) = alternating_events_rows(p_peak, p_valley, &rows, g);
                let n = ev.iter().filter(|&&r| band_of[r] == k).count();
                if n == targets[k] {
                    lo_k = mid;
                    hi_k = mid;
                    break;
                }
                if n < targets[k] {
                    lo_k = mid;
                } else {
                    hi_k = mid;
                }
                if hi_k - lo_k < 1e-3 {
                    break;
                }
            }
            b[k] = 0.5 * (lo_k + hi_k);
        }
    }
    b
}

pub struct Tracks {
    pub vmarg: Vec<f64>, // marginal velocity, pos-units/s (already * v_std)
    pub level: Vec<f64>, // position head, 0..100
    pub env: Vec<f64>,   // generated envelope, pos-units/s (already * v_std)
    pub band: Option<(Vec<f64>, Vec<f64>)>, // ext head: (floor, ceiling)
    pub plat: Option<(Vec<f64>, Vec<f64>)>, // dwell head: P(top), P(bottom)
    pub rev: Option<(Vec<f64>, Vec<f64>)>,  // rev head: P(peak), P(valley)
    // per-row confidence [0,1] (NaN = unforwarded row); empty when the bundle
    // has no conf head. Read only by the review page, never by styling -- it is
    // a property of the frozen trunk, so a re-style never changes it.
    pub conf: Vec<f64>,
}

pub struct StyleCfg {
    pub fps: f64,
    pub still_eps: f64,
    pub ext_snap: f64,
    pub plat_thr: f64,
    pub plat_lo: f64,
    pub plat_peak: f64,
    pub plat_veto: f64,
    /// Per-row rail target inside a dwell lock (follows slow plateau
    /// drift) instead of one constant level.
    pub plat_rail_track: bool,
    /// Peak-confidence lock scaling: correction x clamp((peak - .0) /
    /// (.1 - .0), 0, 1). `.1 <= .0` = full strength.
    pub plat_soft: (f64, f64),
    /// Cap on the lock's per-row mean correction, position units;
    /// 0 = uncapped.
    pub plat_shift_cap: f64,
    /// Crossing snap radius in ROWS (the manifest's `rev_snap_s` on this
    /// grid); 0 = off.
    pub rev_snap: usize,
    /// Reversal SEGMENTATION from the alternating event decode over the rev
    /// head instead of the marginal's zero crossings (jepa_infer
    /// `--rev-source viterbi`, the standing 30 rows/s operating point). The
    /// CARRIER is untouched either way: level, rails, envelope amplitude,
    /// stillness gate and the dwell lock place positions identically. Off
    /// without a rev head; a shot that decodes no events falls back to
    /// crossings.
    pub rev_viterbi: bool,
    /// Refractory in ROWS for the event decode (the manifest's `rev_gap_s`
    /// on this grid). This struct is the RUNTIME config, so it carries rows;
    /// the manifest carries seconds and main.rs converts once.
    pub rev_gap_rows: usize,
    /// Fit the event decode's emission prior separately on STILL and MOVING
    /// rows (jepa_infer `--rev-gap-prior`), membership from the stillness
    /// gate below. The global fit takes its count target from the head's sum
    /// over the whole clip, unscripted gaps included, and force-emits the
    /// difference into slow sections as speed spikes.
    pub rev_gap_prior: bool,
    /// Box width in SECONDS for the |vmarg| the gap prior reads its still /
    /// moving membership off (jepa_infer's `SPEED_REF_S`).
    pub speed_ref_s: f64,
    /// Pre-derivative crossing smoother, SECONDS. 0.1 is the operating
    /// point; 0.2 collapses fast capture.
    pub rev_smooth_s: f64,
    /// Rows of the emission-prior fit window (`bias_fit_s` on this grid).
    pub bias_fit_rows: usize,
    /// Sub-frame reversal times via the event-head parabola (jepa_infer
    /// --subframe rev, the champion decode). Off without a rev head.
    pub subframe_rev: bool,
    /// Background filler rhythm (jepa_infer --filler-*): detected gaps
    /// whose STILL time totals `filler_gap_s` -- row-level
    /// smoothed-|vmarg| stillness, garbage islands bridged, kept islands
    /// pausing the rhythm without restarting the count -- are REPLACED
    /// with a triangle wave at `filler_rate` strokes/min, +-`filler_amp`
    /// units around the smoothed level. An island survives only on
    /// sustained evidence: `filler_min_real_s` of smoothed |v| at or
    /// above `filler_real_v` in total. 0.0 gap = off, the default
    /// and the identity. AUTHORSHIP, deliberately uncorrelated with
    /// on-screen motion -- a user Params knob, never a
    /// manifest constant.
    pub filler_gap_s: f64,
    pub filler_min_real_s: f64,
    pub filler_real_v: f64,
    /// The ONE dial (0..1) for how much the model's output shapes a
    /// filled gap, scaling both influence axes together: the island
    /// evidence bar rises from `filler_min_real_s` (1.0) to the
    /// max-bridge length (0.0 -- no sub-bridge island survives, the
    /// rhythm runs pure), and the rhythm's base blends from the
    /// smoothed level (1.0) to a constant per-gap anchor from the
    /// gap's entry/exit levels (0.0). The >= `filler_max_bridge_s`
    /// length pass is untouched at every setting.
    pub filler_model_w: f64,
    pub filler_rate: f64,
    pub filler_amp: f64,
    pub filler_ramp_s: f64,
    /// Islands at least this long (seconds) always survive, whatever
    /// their amplitude: bridging only ever eats SHORT weak blips, so a
    /// long genuinely-slow passage (above the still floor, below the
    /// confidence bar) can never be overwritten. Also the gap-accounting
    /// reset: a kept island shorter than this interrupts ONE gap, an
    /// island at least this long ends it.
    pub filler_max_bridge_s: f64,
    /// Fractional amplitude modulation (0 = metronome) with period
    /// `filler_sway_s` seconds, phase seeded per gap -- deterministic.
    pub filler_sway: f64,
    pub filler_sway_s: f64,
    /// Rhythm shape: continuous triangle, or bursts of `filler_burst`
    /// strokes separated by `filler_rest_s` seconds of hold.
    pub filler_pattern: FillerPattern,
    pub filler_burst: usize,
    pub filler_rest_s: f64,
    /// Depth-uniformity numbers (`None` = off, the identity): the resolved
    /// `Params::depth_params()`, applied after the strokes are composed.
    pub depth: Option<DepthParams>,
}

/// Filler rhythm shape. `Steady` = one continuous triangle; `Burst` =
/// `filler_burst` strokes, then `filler_rest_s` seconds parked at the
/// base, repeating -- closer to how authors write dead time.
#[derive(Clone, Copy, PartialEq, Eq, Debug, clap::ValueEnum, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum FillerPattern {
    Steady,
    Burst,
}

impl FillerPattern {
    pub fn label(self) -> &'static str {
        match self {
            FillerPattern::Steady => "steady",
            FillerPattern::Burst => "burst",
        }
    }
    pub fn from_label(s: &str) -> Option<Self> {
        match s {
            "steady" => Some(FillerPattern::Steady),
            "burst" => Some(FillerPattern::Burst),
            _ => None,
        }
    }
}

/// Filler preset: a bundle of the primary numbers (gap, rate, amp,
/// pattern), the everyday way to turn the feature on. Every raw
/// `--filler-*` flag still overrides its number, the expert idiom the
/// other presets follow. The resolved Params carry only numbers, so the
/// flags line reproduces exactly what ran preset or not.
#[derive(Clone, Copy, PartialEq, Eq, Debug, clap::ValueEnum, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum FillerPreset {
    Off,
    Subtle,
    Steady,
    Bursts,
}

impl FillerPreset {
    /// Write this preset's bundle into `p` (Off writes nothing: the gap
    /// stays 0 unless an expert flag sets it).
    pub fn apply(self, p: &mut Params) {
        let (gap, rate, amp, pat) = match self {
            FillerPreset::Off => return,
            FillerPreset::Subtle => (20.0, 30.0, 10.0, FillerPattern::Steady),
            FillerPreset::Steady => (10.0, 40.0, 15.0, FillerPattern::Steady),
            FillerPreset::Bursts => (10.0, 50.0, 18.0, FillerPattern::Burst),
        };
        p.filler_gap_s = gap;
        p.filler_rate = rate;
        p.filler_amp = amp;
        p.filler_pattern = pat;
    }
    pub fn base(self) -> Option<(f64, f64, f64, FillerPattern)> {
        match self {
            FillerPreset::Off => None,
            FillerPreset::Subtle => Some((20.0, 30.0, 10.0, FillerPattern::Steady)),
            FillerPreset::Steady => Some((10.0, 40.0, 15.0, FillerPattern::Steady)),
            FillerPreset::Bursts => Some((10.0, 50.0, 18.0, FillerPattern::Burst)),
        }
    }
}

/// How confident the dwell head must be before a hold is parked at its level.
///
/// The knob is the START of the lock's confidence ramp (`plat_soft.0`): a
/// consolidated call locks at full strength once its peak probability reaches
/// the ramp's top, at nothing below the start, and proportionally between. A
/// preset moves that start. `Normal` is the manifest's own, `Cautious` parks
/// only high-confidence holds -- the better trade on material unlike the
/// training corpus, where mid-confidence calls over-commit at low-motion
/// regions -- and `Eager` parks more of what the head offers.
///
/// It is deliberately NOT the peak filter (`plat_peak`), which drops a call
/// whole. Both read the same quantity, so a filter below the ramp's start
/// admits calls the ramp then multiplies by zero: the two controls overlap,
/// and the ramp is the one that still discriminates once they do. Presets on
/// the filter silently stop meaning anything the moment the manifest's ramp
/// starts above them, which is what `presets_ordered` now refuses to ship.
#[derive(Clone, Copy, PartialEq, Eq, Debug, clap::ValueEnum, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Dwells {
    Cautious,
    Normal,
    Eager,
}

impl Dwells {
    /// Where this preset starts the lock's confidence ramp. The manifest sets
    /// the ramp's TOP, which no preset moves -- a call the head is certain
    /// about locks fully whichever preset is on, and the presets differ only
    /// in how far down the confidence scale they keep locking.
    pub fn ramp_start(self, manifest: (f64, f64)) -> f64 {
        match self {
            Dwells::Cautious => 0.85,
            Dwells::Normal => manifest.0,
            Dwells::Eager => 0.3,
        }
    }

    /// The full ramp this preset resolves to against a manifest.
    pub fn plat_soft(self, manifest: (f64, f64)) -> (f64, f64) {
        (self.ramp_start(manifest), manifest.1)
    }
    pub fn cycle(self) -> Self {
        match self {
            Dwells::Cautious => Dwells::Normal,
            Dwells::Normal => Dwells::Eager,
            Dwells::Eager => Dwells::Cautious,
        }
    }
    pub fn label(self) -> &'static str {
        match self {
            Dwells::Cautious => "cautious",
            Dwells::Normal => "normal",
            Dwells::Eager => "eager",
        }
    }
    pub fn from_label(s: &str) -> Option<Self> {
        [Dwells::Cautious, Dwells::Normal, Dwells::Eager]
            .into_iter()
            .find(|d| d.label() == s)
    }
}

/// How aggressively near-still passages quiet down (the stillness gate's
/// velocity floor, pos-units/s). Below the gate, amplitude passes from the
/// generative envelope to the marginal's own predicted travel (the SOFT
/// gate): true holds collapse to sub-texture ripple while authored
/// micro-strokes keep their video-predicted excursion. `Low` trusts more
/// micro-motion, `High` quiets phantom-prone passages harder; the ceiling
/// stays under the measured point where the correlation tail starts
/// collapsing, which the manifest's own floor sets the distance to.
#[derive(Clone, Copy, PartialEq, Eq, Debug, clap::ValueEnum, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Stillness {
    Low,
    Normal,
    High,
}

impl Stillness {
    pub fn still_eps(self, manifest: f64) -> f64 {
        match self {
            Stillness::Low => 8.0,
            Stillness::Normal => manifest,
            Stillness::High => 26.0,
        }
    }
    pub fn cycle(self) -> Self {
        match self {
            Stillness::Low => Stillness::Normal,
            Stillness::Normal => Stillness::High,
            Stillness::High => Stillness::Low,
        }
    }
    pub fn label(self) -> &'static str {
        match self {
            Stillness::Low => "low",
            Stillness::Normal => "normal",
            Stillness::High => "high",
        }
    }
    pub fn from_label(s: &str) -> Option<Self> {
        [Stillness::Low, Stillness::Normal, Stillness::High]
            .into_iter()
            .find(|d| d.label() == s)
    }
}

/// Every preset ladder, resolved against the manifest it will actually run
/// with, must stay STRICTLY ORDERED.
///
/// This is checked at load rather than asserted in a comment because it cannot
/// be checked any earlier: a preset is a compile-time constant, the value it
/// stands beside arrives at runtime from the bundle, and nothing brings the two
/// together until here. A tuning pass that moves a manifest number is not
/// touching this file, and the rung it invalidates is in this file.
///
/// The failure mode is silence, which is what makes it worth an error. A preset
/// whose constant crosses the manifest's value does not break anything a test
/// would notice: the menu entry is still there, still selectable, and simply
/// decodes identically to its neighbour -- or, worse, ranks the wrong way round
/// and quietly does the opposite of its own label.
pub fn presets_ordered(still_eps: f64, plat_soft: (f64, f64)) -> Result<(), String> {
    let dwells: Vec<(&str, f64)> = [Dwells::Eager, Dwells::Normal, Dwells::Cautious]
        .iter()
        .map(|d| (d.label(), d.ramp_start(plat_soft)))
        .collect();
    let stills: Vec<(&str, f64)> = [Stillness::Low, Stillness::Normal, Stillness::High]
        .iter()
        .map(|s| (s.label(), s.still_eps(still_eps)))
        .collect();
    for (flag, rungs) in [("--dwells", &dwells), ("--stillness", &stills)] {
        for (lo, hi) in rungs.iter().zip(rungs.iter().skip(1)) {
            if hi.1 <= lo.1 {
                return Err(format!(
                    "{flag}: {} resolves to {:.3} and {} to {:.3}, so the two decode \
                     the same draft -- a preset constant in style.rs has crossed the \
                     manifest's own value and that rung of the ladder no longer means \
                     anything",
                    lo.0, lo.1, hi.0, hi.1
                ));
            }
        }
    }
    Ok(())
}

/// The user-tunable styling parameters. Every one acts DOWNSTREAM of the
/// head's tracks, so a re-style is compose + write -- milliseconds -- and the
/// encoder never runs again. At the defaults the whole set is an identity:
/// the draft is bit-for-bit the manifest's.

/// Depth uniformity: pull each stroke's reversal endpoints toward the running
/// mean of same-polarity endpoints, so tops reach a consistent height and
/// bottoms a consistent depth. The stroke SHAPE survives -- each monotonic
/// segment is affine-remapped onto its relocated endpoints, so the model's
/// velocity profile still draws it; only where it starts and ends moves. That
/// is the trade: the velocity-derived stroke amplitude gives way to level
/// consistency. `Off` is the model's own reach; the presets dial in more
/// uniformity over a wider window. HOW HARD and HOW LOCAL are the
/// `DepthParams` numbers; WHAT it targets is the per-polarity local mean.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, clap::ValueEnum, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DepthUniformity {
    #[default]
    Off,
    Subtle,
    Even,
    Locked,
}

/// Resolved depth-uniformity numbers (the preset a `DepthUniformity` carries,
/// refined by the expert overrides). `dose` is the blend toward the running
/// mean (0 = the model's endpoint, 1 = the local mean); `window_s` is how many
/// seconds of same-polarity endpoints that mean spans (short = neighbouring
/// strokes match, long = the whole passage converges on one band).
#[derive(Clone, Copy)]
pub struct DepthParams {
    pub dose: f64,
    pub window_s: f64,
}

impl DepthUniformity {
    pub fn base(self) -> Option<DepthParams> {
        match self {
            DepthUniformity::Off => None,
            DepthUniformity::Subtle => Some(DepthParams { dose: 0.35, window_s: 6.0 }),
            DepthUniformity::Even => Some(DepthParams { dose: 0.60, window_s: 10.0 }),
            DepthUniformity::Locked => Some(DepthParams { dose: 0.85, window_s: 20.0 }),
        }
    }
    pub fn cycle(self) -> Self {
        match self {
            DepthUniformity::Off => DepthUniformity::Subtle,
            DepthUniformity::Subtle => DepthUniformity::Even,
            DepthUniformity::Even => DepthUniformity::Locked,
            DepthUniformity::Locked => DepthUniformity::Off,
        }
    }
    pub fn label(self) -> &'static str {
        match self {
            DepthUniformity::Off => "off",
            DepthUniformity::Subtle => "subtle",
            DepthUniformity::Even => "even",
            DepthUniformity::Locked => "locked",
        }
    }
    pub fn from_label(s: &str) -> Option<Self> {
        [DepthUniformity::Off, DepthUniformity::Subtle, DepthUniformity::Even, DepthUniformity::Locked]
            .into_iter()
            .find(|d| d.label() == s)
    }
}

/// Position synthesis for the artifact (jepa_infer `--style`). COMPOSED is
/// the production decode: level + envelope + rails + locks + event
/// segmentation. LEVEL strokes around the position head's level with the
/// marginal's own travel (`compose_level`) -- a far cleaner speed tail
/// (fewer over-speed strokes and gross excursions), at the cost of band
/// reach and steady amplitude. The two decoders score both; the choice is
/// the user's.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, clap::ValueEnum, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Style {
    #[default]
    Composed,
    Level,
}

impl Style {
    pub fn label(self) -> &'static str {
        match self {
            Style::Composed => "composed",
            Style::Level => "level",
        }
    }
    pub fn from_label(s: &str) -> Option<Self> {
        [Style::Composed, Style::Level]
            .into_iter()
            .find(|t| t.label() == s)
    }
}

#[derive(Clone, Copy, PartialEq, Debug, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct Params {
    /// Position synthesis (composed = production; level = carrier-only).
    pub style: Style,
    pub dwells: Dwells,
    pub stillness: Stillness,
    /// Stroke amplitude scale about mid-range; 1.0 = the model's own.
    pub intensity: f64,
    /// The 0..100 track is mapped linearly onto this span (device limits).
    pub range: (f64, f64),
    /// Cap on position speed between actions, pos-units/s; 0 = off. Defaults
    /// to `MAX_POS_RATE`, the device cap the Python decode always applies.
    pub max_speed: f64,
    /// EXPERT override for where the dwell lock's confidence ramp STARTS
    /// (0..1): `Some` uses this raw value in place of whatever `dwells` would
    /// resolve to. This is the numeric knob behind the review page's expert
    /// mode, and it moves the same axis the preset does -- an override that
    /// stood on a different field would leave the dropdown beside it
    /// selecting something else.
    pub dwell_ramp: Option<f64>,
    /// EXPERT override for the stillness velocity floor (pos-units/s): `Some`
    /// uses this raw value in place of whatever `stillness` resolves to.
    pub still_eps: Option<f64>,
    /// Depth-uniformity preset (regularize stroke endpoints toward the
    /// per-polarity running median).
    pub depth: DepthUniformity,
    /// EXPERT overrides on the resolved depth preset's numbers (blend 0..1 /
    /// window seconds); ignored while `depth` is Off.
    pub depth_dose: Option<f64>,
    pub depth_window: Option<f64>,
    /// Background filler rhythm: detected motion-free gaps whose still
    /// time totals at least this (seconds) are REPLACED with the rhythm.
    /// 0 = off (the default, and the byte identity).
    pub filler_gap_s: f64,
    /// A motion island inside a gap survives only if its smoothed
    /// |vmarg| holds `filler_real_v` for this much time in total
    /// (seconds); islands with less sustained evidence are garbage and
    /// are replaced.
    pub filler_min_real_s: f64,
    /// pos-units/s: the confidence bar a gap island's smoothed |vmarg|
    /// must SUSTAIN (`filler_min_real_s` in total) to count as real.
    pub filler_real_v: f64,
    /// 0..1, the model-influence slider: how much the model's output
    /// shapes a filled gap. 1 = islands survive at the evidence bar and
    /// the rhythm rides the smoothed level; 0 = pure rhythm -- no
    /// sub-bridge island survives and the base is one constant per-gap
    /// anchor (entry/exit levels). >= max-bridge islands always survive.
    pub filler_model_w: f64,
    /// Filler rate, strokes (direction legs) per minute.
    pub filler_rate: f64,
    /// Filler amplitude, position units (half peak-to-peak around the
    /// smoothed level base).
    pub filler_amp: f64,
    /// Seam cross-fade seconds at each end of a gap; 0 = one stroke leg.
    pub filler_ramp_s: f64,
    /// Islands at least this long always survive, whatever their
    /// amplitude (a long slow passage is never overwritten). Also the
    /// gap-accounting reset: a kept island shorter than this pauses the
    /// rhythm inside ONE gap, one at least this long ends the gap.
    pub filler_max_bridge_s: f64,
    /// Fractional amplitude sway (0 = metronome) over `filler_sway_s`
    /// seconds; deterministic, phase seeded per gap.
    pub filler_sway: f64,
    pub filler_sway_s: f64,
    /// Rhythm shape (steady triangle / bursts with rests).
    pub filler_pattern: FillerPattern,
    pub filler_burst: usize,
    pub filler_rest_s: f64,
}

impl Params {
    /// The depth-uniformity numbers this parameter set resolves to (None =
    /// off): the expert overrides refine the chosen preset.
    pub fn depth_params(&self) -> Option<DepthParams> {
        let mut d = self.depth.base()?;
        if let Some(v) = self.depth_dose {
            d.dose = v;
        }
        if let Some(v) = self.depth_window {
            d.window_s = v;
        }
        Some(d)
    }
}

impl Default for Params {
    fn default() -> Self {
        Self {
            style: Style::Composed,
            dwells: Dwells::Normal,
            stillness: Stillness::Normal,
            intensity: 1.0,
            range: (0.0, 100.0),
            max_speed: MAX_POS_RATE,
            dwell_ramp: None,
            still_eps: None,
            depth: DepthUniformity::Off,
            depth_dose: None,
            depth_window: None,
            filler_gap_s: 0.0,
            filler_min_real_s: 1.0,
            filler_real_v: 45.0,
            filler_model_w: 1.0,
            filler_rate: 40.0,
            filler_amp: 15.0,
            filler_ramp_s: 0.0,
            filler_max_bridge_s: 5.0,
            filler_sway: 0.15,
            filler_sway_s: 16.0,
            filler_pattern: FillerPattern::Steady,
            filler_burst: 4,
            filler_rest_s: 2.0,
        }
    }
}

/// The CLI flags that reproduce `p` -- what the console echoes after a review
/// and what the review page displays. One formatter, so they cannot drift.
pub fn flags_line(p: &Params) -> String {
    // An expert override replaces its preset flag -- `--dwell-ramp 0.70`
    // instead of `--dwells normal`, so the line reproduces exactly what ran.
    let dwell = match p.dwell_ramp {
        Some(pk) => format!("--dwell-ramp {pk:.2}"),
        None => format!("--dwells {}", p.dwells.label()),
    };
    let still = match p.still_eps {
        Some(se) => format!("--still-eps {se:.0}"),
        None => format!("--stillness {}", p.stillness.label()),
    };
    // Depth: the preset flag, then any expert numbers that refine it -- an
    // override echoes so the line reproduces exactly what ran.
    let depth = if p.depth == DepthUniformity::Off {
        String::new()
    } else {
        let mut s = format!(" --depth-uniformity {}", p.depth.label());
        if let Some(v) = p.depth_dose {
            s.push_str(&format!(" --depth-dose {v:.2}"));
        }
        if let Some(v) = p.depth_window {
            s.push_str(&format!(" --depth-window {v:.1}"));
        }
        s
    };
    // Filler: off does not echo; on echoes the primary numbers always and
    // the detector/seam knobs when they differ from their defaults, so the
    // line reproduces exactly what ran.
    let filler = if p.filler_gap_s > 0.0 {
        let mut s = format!(
            " --filler-gap {:.0} --filler-rate {:.0} --filler-amp {:.0}",
            p.filler_gap_s, p.filler_rate, p.filler_amp
        );
        if (p.filler_min_real_s - 1.0).abs() > 1e-9 {
            s.push_str(&format!(" --filler-min-real {:.1}", p.filler_min_real_s));
        }
        if (p.filler_real_v - 45.0).abs() > 1e-9 {
            s.push_str(&format!(" --filler-real-v {:.0}", p.filler_real_v));
        }
        if (p.filler_model_w - 1.0).abs() > 1e-9 {
            s.push_str(&format!(" --filler-model-w {:.2}", p.filler_model_w));
        }
        if p.filler_ramp_s > 0.0 {
            s.push_str(&format!(" --filler-ramp {:.1}", p.filler_ramp_s));
        }
        if (p.filler_max_bridge_s - 5.0).abs() > 1e-9 {
            s.push_str(&format!(" --filler-max-bridge {:.1}", p.filler_max_bridge_s));
        }
        if (p.filler_sway - 0.15).abs() > 1e-9 {
            s.push_str(&format!(" --filler-sway {:.2}", p.filler_sway));
        }
        if (p.filler_sway_s - 16.0).abs() > 1e-9 {
            s.push_str(&format!(" --filler-sway-s {:.0}", p.filler_sway_s));
        }
        if p.filler_pattern != FillerPattern::Steady {
            s.push_str(&format!(" --filler-pattern {}", p.filler_pattern.label()));
            if p.filler_burst != 4 {
                s.push_str(&format!(" --filler-burst {}", p.filler_burst));
            }
            if (p.filler_rest_s - 2.0).abs() > 1e-9 {
                s.push_str(&format!(" --filler-rest {:.1}", p.filler_rest_s));
            }
        }
        s
    } else {
        String::new()
    };
    // `--style composed` is the default and does not echo.
    let sty = if p.style == Style::Composed {
        String::new()
    } else {
        format!("--style {} ", p.style.label())
    };
    format!(
        "{sty}{dwell} {still} --intensity {:.2}{}{}{depth}{filler}",
        p.intensity,
        if p.range != (0.0, 100.0) {
            format!(" --range {:.0}-{:.0}", p.range.0, p.range.1)
        } else {
            String::new()
        },
        if p.max_speed > 0.0 {
            format!(" --max-speed {:.0}", p.max_speed)
        } else {
            String::new()
        },
    )
}

/// Output-domain shaping on the finished actions: intensity scales positions
/// about mid-range, `range` maps 0..100 onto the device's span, `max_speed`
/// then caps how fast consecutive actions may move (so the cap reads final
/// device units). The model's draft upstream is untouched.
pub fn shape_actions(actions: &mut [Action], p: &Params) {
    let (lo, hi) = p.range;
    for a in actions.iter_mut() {
        let x = (50.0 + (a.pos as f64 - 50.0) * p.intensity).clamp(0.0, 100.0);
        a.pos = (lo + x / 100.0 * (hi - lo)).round() as i64;
    }
    if p.max_speed > 0.0 {
        // Port of `common.clamp_speed`, and it has to be that port exactly:
        // causal on the ALREADY-CLAMPED predecessor, and the step TRUNCATED
        // rather than rounded -- across a 1-2 ms gap half a unit is hundreds
        // of pos/s, so rounding can hand back a transition above the cap.
        for i in 1..actions.len() {
            let dt = (actions[i].at - actions[i - 1].at) as f64 / 1000.0;
            let lim = p.max_speed * dt;
            let prev = actions[i - 1].pos;
            let d = actions[i].pos - prev;
            if (d as f64).abs() > lim {
                let step = lim as i64;
                actions[i].pos = (prev + if d > 0 { step } else { -step })
                    .clamp(0, 100);
            }
        }
    }
}

/// Row spans where `m` is true, as `[a, b)`.
fn runs(m: &[bool]) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < m.len() {
        if m[i] {
            let a = i;
            while i < m.len() && m[i] {
                i += 1;
            }
            out.push((a, i));
        } else {
            i += 1;
        }
    }
    out
}

/// `common.local_mean` -- centered moving mean, EDGE-padded (not zero-padded
/// like `box_same`: oscillation averages out of it, a real traverse does not).
fn local_mean(x: &[f64], k: usize) -> Vec<f64> {
    let k = k | 1;
    let half = k / 2;
    let n = x.len();
    (0..n)
        .map(|i| {
            let mut acc = 0.0;
            for j in 0..k {
                let idx = i as isize + j as isize - half as isize;
                acc += x[idx.clamp(0, n as isize - 1) as usize];
            }
            acc / k as f64
        })
        .collect()
}

/// Dwell-head tracks -> a per-row dwell kind in {0, +1 top, -1 bottom}.
///
/// The head's raw calls are FRAGMENTARY: it fires somewhere inside most script
/// dwells but rarely covers a whole one. That is the worst case for the level
/// lock, which pays for partial coverage -- the step between a locked and a free
/// row ADDS excursion -- so a call is consolidated into a whole dwell before the
/// decode sees it. A run is SEEDED where the probability clears `thr` and grows
/// while it stays above `thr_lo` (hysteresis), so a plateau the head is sure
/// about in the middle and unsure about at the corners comes out whole.
fn dwell_kind(
    plat: &(Vec<f64>, Vec<f64>),
    thr: f64,
    thr_lo: f64,
    peak: f64,
    fps: f64,
) -> Vec<i8> {
    let gap = rows_at(DWELL_GAP_S, fps, false); // short holes closed
    // a dwell briefer than ~0.25 s is just a reversal
    let min_rows = rows_at(DWELL_MIN_CALL_S, fps, false);
    let n = plat.0.len();
    let mut masks: Vec<Vec<bool>> = Vec::new();
    for p in [&plat.0, &plat.1] {
        let mut m = vec![false; n];
        for (a, b) in runs(&p.iter().map(|&v| v >= thr_lo).collect::<Vec<_>>()) {
            if p[a..b].iter().any(|&v| v >= thr) {
                m[a..b].fill(true); // seed, then grow
            }
        }
        let free: Vec<bool> = m.iter().map(|&v| !v).collect();
        for (a, b) in runs(&free) {
            if a > 0 && b < n && b - a <= gap {
                m[a..b].fill(true);
            }
        }
        for (a, b) in runs(&m.clone()) {
            if b - a < min_rows {
                m[a..b].fill(false);
            }
        }
        masks.push(m);
    }
    let (top, bot) = (&masks[0], &masks[1]);
    let mut kind: Vec<i8> = (0..n)
        .map(|i| {
            let t = top[i] && (!bot[i] || plat.0[i] >= plat.1[i]);
            let b = bot[i] && (!top[i] || plat.1[i] > plat.0[i]);
            if t {
                1
            } else if b {
                -1
            } else {
                0
            }
        })
        .collect();
    // Peak filter (> thr to bite): a consolidated call whose probability never
    // reaches `peak` is dropped whole -- false calls are low-peak (median
    // 0.60-0.69 vs 0.76-0.83 inside a real dwell), so this sheds misplaced
    // rail pins while the dwell response stays put. 0 = off.
    if peak > 0.0 {
        for (k, p) in [(1i8, &plat.0), (-1i8, &plat.1)] {
            let is_k: Vec<bool> = kind.iter().map(|&v| v == k).collect();
            for (a, b) in runs(&is_k) {
                if p[a..b].iter().fold(f64::MIN, |m, &v| m.max(v)) < peak {
                    kind[a..b].fill(0);
                }
            }
        }
    }
    kind
}

/// Pin the styled track's LOCAL MEAN to the extremum-side rail inside each
/// predicted dwell, leaving its residual ripple untouched.
///
/// A dwell is a level regime, not a stillness one: 95% of scripted plateaus
/// oscillate at the extreme, and the composed draft already generates about the
/// right amount of that ripple. What it gets wrong is RESIDENCY -- its local mean
/// sweeps through the plateau, deleting the dwell's duration as a triangle apex.
/// So the fix is subtraction, not replacement: decompose into local mean +
/// residual, replace the mean with the rail level, add the residual back. The
/// traverse lives in the mean and dies; the ripple lives in the residual and
/// survives. (Overwriting the span with a constant would take the ripple with
/// it -- and a range-based dwell metric would happily score that as a hold.)
///
/// The correction runs at full strength across the whole dwell and ramps in over
/// `EDGE` free rows on either SIDE of it. The ramp has to live in the flanking
/// strokes, not in the dwell: a plateau's corner rows are part of the plateau --
/// the script's own mean is already parked there -- so tapering inside would
/// leave exactly those rows sweeping.
///
/// `rail_track` locks to the smoothed PER-ROW rail inside the dwell instead of
/// one constant level (a long or drifting plateau follows its rail); `soft`
/// scales the whole correction by the call's head-peak confidence, ramping
/// 0 -> 1 over `[soft.0, soft.1]` -- false calls are separably low-peak, so a
/// weak call gets a weak lock. `soft.1 <= soft.0` (the pre-variant manifests'
/// [0, 0]) means full strength; `plat` is the head tuple the peak comes from.
/// `shift_cap` clamps each row's raw correction to +-cap position units
/// BEFORE the soft/ramp scaling (matching jepa_infer): a confident call
/// shifting too far is half the gross-error rate. 0 = uncapped.
#[allow(clippy::too_many_arguments)]
fn level_lock(
    p: &[f64],
    kind: &[i8],
    rail_lo: &[f64],
    rail_hi: &[f64],
    plat: Option<&(Vec<f64>, Vec<f64>)>,
    soft: (f64, f64),
    rail_track: bool,
    shift_cap: f64,
    fps: f64,
) -> Vec<f64> {
    let smooth = rows_at(LOCK_SMOOTH_S, fps, true);
    let edge = rows_at(LOCK_EDGE_S, fps, false);
    let n = p.len();
    let m = local_mean(p, smooth);
    let mut out = p.to_vec();
    let free: Vec<bool> = kind.iter().map(|&k| k == 0).collect();
    let mut claimed = vec![false; n]; // a ramp row is borrowed once
    for (a, b) in runs(&kind.iter().map(|&k| k != 0).collect::<Vec<_>>()) {
        let rail = if kind[a] > 0 { rail_hi } else { rail_lo };
        let tgt_in: Vec<f64> = if rail_track {
            local_mean(&rail[a..b], smooth)
                .iter()
                .map(|t| t.clamp(0.0, 100.0))
                .collect()
        } else {
            let c = (rail[a..b].iter().sum::<f64>() / (b - a) as f64).clamp(0.0, 100.0);
            vec![c; b - a]
        };
        let strength = match (plat, soft.1 > soft.0) {
            (Some((pt, pb)), true) => {
                let track = if kind[a] > 0 { pt } else { pb };
                let pk = track[a..b]
                    .iter()
                    .cloned()
                    .filter(|v| v.is_finite())
                    .fold(f64::NEG_INFINITY, f64::max);
                ((pk - soft.0) / (soft.1 - soft.0)).clamp(0.0, 1.0)
            }
            _ => 1.0,
        };
        let mut lo = a;
        while lo > 0 && a - lo < edge && free[lo - 1] && !claimed[lo - 1] {
            lo -= 1;
        }
        let mut hi = b;
        while hi < n && hi - b < edge && free[hi] && !claimed[hi] {
            hi += 1;
        }
        claimed[lo..a].fill(true);
        claimed[b..hi].fill(true);
        for i in lo..hi {
            // w ramps 0 -> 1 across the borrowed rows before the dwell, holds at
            // 1 inside it, and ramps back to 0 after: no seam at either end
            let w = if i < a {
                (i - lo) as f64 / (a - lo) as f64
            } else if i >= b {
                1.0 - (i - b + 1) as f64 / (hi - b) as f64
            } else {
                1.0
            };
            // ramp rows carry the nearest corner's target level
            let tgt = if i < a {
                tgt_in[0]
            } else if i >= b {
                tgt_in[b - a - 1]
            } else {
                tgt_in[i - a]
            };
            let corr = if shift_cap > 0.0 {
                (tgt - m[i]).clamp(-shift_cap, shift_cap)
            } else {
                tgt - m[i]
            };
            out[i] = (p[i] + strength * w * corr).clamp(0.0, 100.0);
        }
    }
    out
}

/// Drop consolidated dwell calls that have no adjoining stroke.
///
/// The label side (`dwell_spans`) DEFINES a dwell by its adjoining stroke,
/// but the head reads rows, not that guard, so it can lock stroke-free
/// regions -- on unseen material it over-commits at low-motion
/// intros/outros. A call is vetoed when the predicted |vmarg| of BOTH
/// ~2 s flanks stays under `thr` (pos-units/s); a clip edge counts as a
/// still flank (that IS the intro/outro case). In-corpus dwells adjoin
/// strokes (moving |v| ~107 pos/s vs ~13 during holds), so real calls
/// clear any threshold between those by a wide margin.
fn stroke_veto(kind: &[i8], vmarg: &[f64], thr: f64, flank: usize) -> Vec<i8> {
    let n = kind.len();
    let mut out = kind.to_vec();
    let mean_abs = |lo: usize, hi: usize| -> f64 {
        if hi <= lo {
            return 0.0;
        }
        vmarg[lo..hi]
            .iter()
            .map(|v| if v.is_finite() { v.abs() } else { 0.0 })
            .sum::<f64>()
            / (hi - lo) as f64
    };
    let mut i = 0;
    while i < n {
        if out[i] == 0 {
            i += 1;
            continue;
        }
        let a = i;
        let mut b = i;
        while b < n && out[b] != 0 {
            b += 1;
        }
        let lo_m = mean_abs(a.saturating_sub(flank), a);
        let hi_m = mean_abs(b, (b + flank).min(n));
        if lo_m.max(hi_m) < thr {
            out[a..b].fill(0);
        }
        i = b;
    }
    out
}

/// Regularize stroke depth: pull each reversal endpoint toward the running median
/// of same-polarity endpoints, keeping the stroke's shape.
///
/// A stroke's reversal is a local extremum of the composed track. This finds
/// them with the SAME smoothed-derivative sign-flip detector `extrema_actions`
/// writes with (so the rows it moves are exactly the ones that become actions),
/// splits them into tops and bottoms, and blends each toward its polarity's
/// mean over a `window_s` slice of neighbours. The shot's own ends are fixed
/// anchors; each monotonic run between anchors is then affine-remapped from its
/// old endpoint pair onto the new one -- the within-run progress profile (the
/// model's integrated-velocity shape) is preserved exactly, only the reach
/// changes. `dose <= 0` returns the input untouched (the identity).
fn depth_uniformize(
    p: &[f64],
    shot_edges: &[usize],
    times_ms: &[f64],
    prm: &DepthParams,
) -> Vec<f64> {
    let dose = prm.dose.clamp(0.0, 1.0);
    if dose <= 0.0 {
        return p.to_vec();
    }
    let win_ms = prm.window_s.max(0.0) * 1000.0;
    let mut out = p.to_vec();
    for w in shot_edges.windows(2) {
        let (lo, hi) = (w[0], w[1]);
        if hi - lo < 3 {
            continue;
        }
        let seg = &p[lo..hi];
        let n = seg.len();
        // interior extrema of the smoothed track: (local row, +1 top / -1 bottom)
        let s = box_same(seg, 3);
        let mut d: Vec<f64> = s.windows(2).map(|w| sign(w[1] - w[0])).collect();
        for v in d.iter_mut() {
            if *v == 0.0 {
                *v = 1.0;
            }
        }
        let ext: Vec<(usize, i8)> = (1..d.len())
            .filter(|&i| d[i] * d[i - 1] < 0.0)
            .map(|i| (i, if d[i - 1] > 0.0 { 1 } else { -1 }))
            .collect();
        if ext.is_empty() {
            continue;
        }
        // per-polarity sliding-window MEDIAN of endpoint values -> blend
        // target. The median is the level that lines the window up with
        // the least total position change: neighbours already at a common
        // depth stay put and the outlier comes to them, where a mean is
        // dragged by the outlier and moves every point (user call,
        // 2026-07-30).
        let mut target = vec![f64::NAN; n];
        for pol in [1i8, -1i8] {
            let pts: Vec<usize> =
                ext.iter().filter(|&&(_, q)| q == pol).map(|&(i, _)| i).collect();
            let (mut a, mut b) = (0usize, 0usize);
            for &i in &pts {
                let t = times_ms[lo + i];
                while times_ms[lo + pts[a]] < t - win_ms / 2.0 {
                    a += 1;
                }
                while b < pts.len() && times_ms[lo + pts[b]] <= t + win_ms / 2.0 {
                    b += 1;
                }
                let mut w: Vec<f64> = pts[a..b].iter().map(|&j| seg[j]).collect();
                w.sort_by(|x, y| x.partial_cmp(y).expect("finite positions"));
                let m = w.len();
                let med = if m % 2 == 1 {
                    w[m / 2]
                } else {
                    0.5 * (w[m / 2 - 1] + w[m / 2])
                };
                target[i] = seg[i] + dose * (med - seg[i]);
            }
        }
        // anchors (row, old, new): the shot ends stay put, interior extrema move
        let mut anchors: Vec<(usize, f64, f64)> = Vec::with_capacity(ext.len() + 2);
        anchors.push((0, seg[0], seg[0]));
        for &(i, _) in &ext {
            anchors.push((i, seg[i], target[i]));
        }
        anchors.push((n - 1, seg[n - 1], seg[n - 1]));
        // affine-remap each monotonic run onto its relocated endpoints
        for aw in anchors.windows(2) {
            let (i0, o0, t0) = aw[0];
            let (i1, o1, t1) = aw[1];
            let denom = o1 - o0;
            for i in i0..=i1 {
                let frac = if denom.abs() < 1e-9 {
                    // flat old run: no shape to preserve, ramp on row index
                    (i - i0) as f64 / (i1 - i0).max(1) as f64
                } else {
                    (seg[i] - o0) / denom
                };
                out[lo + i] = (t0 + (t1 - t0) * frac).clamp(0.0, 100.0);
            }
        }
    }
    out
}

/// Composed styling over shot spans -> (0..100 position track, sub-frame
/// reversal times: apex row -> ms, filled under `cfg.subframe_rev` with a
/// rev head -- port of jepa_infer's `--subframe rev` parabola).
pub fn compose(
    t: &Tracks,
    shot_edges: &[usize],
    cfg: &StyleCfg,
    times_ms: &[f64],
) -> (Vec<f64>, std::collections::HashMap<usize, f64>, Vec<usize>, Vec<(usize, usize)>) {
    let dt = 1.0 / cfg.fps;
    let n_all = t.vmarg.len();
    let mut p = vec![50.0; n_all];
    let mut sub = std::collections::HashMap::new();
    // apex rows the event decode CALLED: handed to extrema_actions so the
    // written artifact carries every called vertex (RDP still prunes
    // sub-unit prominence). Empty in cross mode.
    let mut force: Vec<usize> = Vec::new();

    let kd = rows_at(DECODE_SMOOTH_S, cfg.fps, true);
    let kr = rows_at(cfg.rev_smooth_s, cfg.fps, true);

    // The emission prior is fitted ONCE for the clip, not per shot: the
    // per-shot event count is far too small to fit against, and jepa_infer
    // fits it on a clip-level slice. Deploy has no script, so the target is
    // the head's own probability mass over the fit window. Fit THROUGH the
    // decode that will run -- the refractory changes the emitted rate at a
    // given prior.
    let viterbi = cfg.rev_viterbi && t.rev.is_some();
    // Per-row, because a band-fitted prior differs row to row; a global fit
    // fills it with one value and the decode cannot tell the two apart.
    let rev_bias: Vec<f64> = if viterbi {
        let (rt, rb) = t.rev.as_ref().unwrap();
        let b0 = cfg.bias_fit_rows.min(rt.len());
        let nz = |v: &f64| if v.is_finite() { *v } else { 0.0 };
        if cfg.rev_gap_prior {
            // membership is the styling's own stillness gate, so no script is
            // involved and the fit deploys
            let kb = rows_at(cfg.speed_ref_s, cfg.fps, true);
            let sm = box_same(
                &t.vmarg.iter().map(|v| v.abs()).collect::<Vec<f64>>(),
                kb,
            );
            let band: Vec<usize> = sm
                .iter()
                .map(|&s| if s >= cfg.still_eps { 1 } else { 0 })
                .collect();
            let mut targets = [0.0f64; 2];
            for i in 0..b0 {
                targets[band[i]] += nz(&rt[i]) + nz(&rb[i]);
            }
            let tg = [targets[0].round() as usize, targets[1].round() as usize];
            let bb = fit_emission_bias_bands(
                &rt[..b0],
                &rb[..b0],
                &band[..b0],
                &tg,
                cfg.rev_gap_rows,
            );
            band.iter().map(|&k| bb[k]).collect()
        } else {
            let want: f64 = rt[..b0].iter().map(nz).sum::<f64>()
                + rb[..b0].iter().map(nz).sum::<f64>();
            let b = fit_emission_bias(
                &rt[..b0],
                &rb[..b0],
                want.round() as usize,
                cfg.rev_gap_rows,
            );
            vec![b; rt.len()]
        }
    } else {
        vec![0.0; t.vmarg.len()]
    };
    let lv = box_same(&t.level, kd);
    let ev = &t.env;
    let band = t.band.as_ref().map(|(lo, hi)| (box_same(lo, kd), box_same(hi, kd)));
    let dk = t
        .plat
        .as_ref()
        .map(|p| dwell_kind(p, cfg.plat_thr, cfg.plat_lo, cfg.plat_peak, cfg.fps))
        .map(|k| {
            if cfg.plat_veto > 0.0 {
                stroke_veto(&k, &t.vmarg, cfg.plat_veto, (2.0 * cfg.fps).round() as usize)
            } else {
                k
            }
        });

    for w in shot_edges.windows(2) {
        let (lo, hi) = (w[0], w[1]);
        let n = hi - lo;
        if n < 2 {
            continue;
        }
        let v = &t.vmarg[lo..hi];
        let lvl = &lv[lo..hi];
        let s = box_same(v, kr);

        let mut cross = crossings(&s);
        // The event decode replaces the SEGMENTATION itself (not just the
        // boundary position): apex on the segment's LAST row, so event row f
        // becomes boundary f + 1, the same convention rev_snap uses. Segment
        // i ends at its event, so an up-stroke's direction IS the event's
        // kind (+1 peak); the tail segment continues the alternation. A shot
        // that decodes nothing keeps the crossings.
        let mut seg_dirs: Option<Vec<f64>> = None;
        if viterbi {
            let (rt_g, rb_g) = t.rev.as_ref().unwrap();
            let (vrows, vkinds) = alternating_events_rows(
                &rt_g[lo..hi],
                &rb_g[lo..hi],
                &rev_bias[lo..hi],
                cfg.rev_gap_rows,
            );
            let keep: Vec<usize> = vrows
                .iter()
                .copied()
                .enumerate()
                .filter(|&(_, r)| r + 1 <= n - 1)
                .map(|(i, _)| i)
                .collect();
            if !keep.is_empty() {
                cross = keep.iter().map(|&i| vrows[i] + 1).collect();
                let mut d: Vec<f64> =
                    keep.iter().map(|&i| vkinds[i] as f64).collect();
                d.push(-(vkinds[*keep.last().unwrap()] as f64));
                seg_dirs = Some(d);
                force.extend(cross.iter().map(|&c| lo + c - 1));
            }
        }
        // Re-localize each crossing to the reversal-event head's local
        // argmax (port of jepa_infer's --rev-snap): the composed stroke's
        // apex lands on its segment's LAST row (c - 1), so a head max at
        // frame f puts the crossing at f + 1. Direction from the ORIGINAL
        // segmentation (a rising segment ends at a PEAK); snapped
        // boundaries stay strictly ordered -- collisions drop, merging the
        // stroke. NaN probability rows read as 0, numpy's nan_to_num.
        // rev_snap is SUBSUMED by the event segmentation -- it re-localizes
        // crossings, and in viterbi mode there are none left to move.
        if seg_dirs.is_none() && cfg.rev_snap > 0 && !cross.is_empty() {
            if let Some((rt_g, rb_g)) = &t.rev {
                let (rt, rb) = (&rt_g[lo..hi], &rb_g[lo..hi]);
                let r = cfg.rev_snap;
                let mut snapped = Vec::with_capacity(cross.len());
                let mut prev = 0usize;
                let mut a = 0usize;
                for &c in &cross {
                    let d = if c > a {
                        sign(s[a..c].iter().sum::<f64>() / (c - a) as f64)
                    } else {
                        0.0
                    };
                    let track = if d > 0.0 { rt } else { rb };
                    let w0 = prev.max((c - 1).saturating_sub(r));
                    let w1 = (c - 1 + r).min(n - 2);
                    let c2 = if w1 < w0 {
                        c
                    } else {
                        // first max wins ties, matching np.argmax
                        let mut best = w0;
                        let mut bv = f64::NEG_INFINITY;
                        for (o, v) in track[w0..=w1].iter().enumerate() {
                            let x = if v.is_finite() { *v } else { 0.0 };
                            if x > bv {
                                bv = x;
                                best = w0 + o;
                            }
                        }
                        best + 1
                    };
                    if c2 > prev && c2 < n {
                        snapped.push(c2);
                        prev = c2;
                    }
                    a = c;
                }
                cross = snapped;
            }
        }
        let mut bounds = vec![0usize];
        bounds.extend(cross);
        bounds.push(n);

        let mut cur = lvl[0];
        for (si, bw) in bounds.windows(2).enumerate() {
            let (a, b) = (bw[0], bw[1]);
            if b <= a {
                continue;
            }
            // a CALLED reversal reverses: the direction is the event's kind,
            // not the sign the smoothed marginal happens to average to
            let d = match &seg_dirs {
                Some(ds) => ds.get(si).copied().unwrap_or(0.0),
                None => sign(s[a..b].iter().sum::<f64>() / (b - a) as f64),
            };
            let abs_v: Vec<f64> = v[a..b].iter().map(|x| x.abs()).collect();
            let tr_raw: f64 = abs_v.iter().sum::<f64>() * dt;
            let mean_abs_v = tr_raw / dt / (b - a) as f64;
            let mean_env: f64 =
                ev[lo + a..lo + b].iter().sum::<f64>() / (b - a) as f64;
            let tr_env = mean_env * (b - a) as f64 * dt;

            if tr_raw <= 1e-9 || d == 0.0 || tr_env <= 1e-9 {
                p[lo + a..lo + b].copy_from_slice(&lvl[a..b]);
                cur = lvl[b - 1];
                continue;
            }

            // Below the stillness gate the AR envelope's amplitude would
            // mint a phantom (during target holds the marginal's magnitude
            // collapses ~6x while its SIGN keeps wiggling and the envelope
            // never quiets), so amplitude passes to the MARGINAL's own
            // predicted travel: true holds collapse to sub-texture ripple
            // (~5 pos units), authored micro-strokes keep their
            // video-predicted excursion. Quiet segments skip the band snap
            // -- ext_snap would stretch a ripple to the band edge.
            let quiet = mean_abs_v < cfg.still_eps;
            let travel = if quiet { tr_raw.min(tr_env) } else { tr_env };
            let mut end = (lvl[b - 1] + d * travel.min(100.0) / 2.0).clamp(0.0, 100.0);
            if !quiet {
                if let Some((blo, bhi)) = &band {
                    if d < 0.0 && end < blo[lo + b - 1] + cfg.ext_snap {
                        end = blo[lo + b - 1].clamp(0.0, 100.0);
                    } else if d > 0.0 && end > bhi[lo + b - 1] - cfg.ext_snap {
                        end = bhi[lo + b - 1].clamp(0.0, 100.0);
                    }
                }
            }
            // move along the integrated |velocity| profile, so the stroke's
            // shape is the model's, not a ramp
            let mut acc = 0.0;
            for (i, av) in abs_v.iter().enumerate() {
                acc += av;
                let frac = (acc * dt / tr_raw).clamp(0.0, 1.0);
                p[lo + a + i] = cur + (end - cur) * frac;
            }
            cur = end;
        }
        // Sub-frame reversal times (port of jepa_infer --subframe rev):
        // a 3-point parabola on the event head's probability around the
        // stroke's apex row (b - 1, the snapped argmax); the vertex
        // offset, clamped to +-0.5 rows, lands `at` between rows.
        if cfg.subframe_rev {
            if let Some((rt_g, rb_g)) = &t.rev {
                for i in 0..bounds.len().saturating_sub(2) {
                    let (a, b) = (bounds[i], bounds[i + 1]);
                    if b <= a {
                        continue;
                    }
                    // same direction source as the carrier loop: a called
                    // reversal's apex kind IS the event's kind, and it picks
                    // which head track the parabola is fitted on
                    let d = match &seg_dirs {
                        Some(ds) => ds.get(i).copied().unwrap_or(0.0),
                        None => sign(s[a..b].iter().sum::<f64>() / (b - a) as f64),
                    };
                    if d == 0.0 {
                        continue;
                    }
                    let gl = lo + b - 1;
                    let y = if d > 0.0 { rt_g } else { rb_g };
                    if gl < 1 || gl + 1 >= y.len() {
                        continue;
                    }
                    let fin = |x: f64| if x.is_finite() { x } else { 0.0 };
                    let (y0, y1, y2) = (fin(y[gl - 1]), fin(y[gl]), fin(y[gl + 1]));
                    let den = y0 - 2.0 * y1 + y2;
                    let dlt = if den.abs() < 1e-9 {
                        0.0
                    } else {
                        (0.5 * (y0 - y2) / den).clamp(-0.5, 0.5)
                    };
                    let dt_ms = if gl + 1 < times_ms.len() {
                        times_ms[gl + 1] - times_ms[gl]
                    } else {
                        1000.0 * dt
                    };
                    sub.insert(gl, times_ms[gl] + dlt * dt_ms);
                }
            }
        }
    }

    // Regularize stroke depth toward the per-polarity running median (identity
    // when off). Runs on the built strokes but BEFORE the plateau park:
    // uniformity moves stroke reach, the level lock then pins the dwells --
    // reordering would let uniformity drag a just-parked plateau off its rail.
    let p = if let Some(dp) = &cfg.depth {
        depth_uniformize(&p, shot_edges, times_ms, dp)
    } else {
        p
    };

    // The strokes are built; now park the ones called a plateau, keeping the
    // ripple they already carry. The rail is what carries the level here -- it
    // predicts plateau levels within a few units where the level head's ~2 s
    // window is too coarse to speak for a 0.5 s plateau.
    let p = match (&dk, &band) {
        (Some(k), Some((blo, bhi))) => level_lock(
            &p, k, blo, bhi, t.plat.as_ref(), cfg.plat_soft, cfg.plat_rail_track,
            cfg.plat_shift_cap, cfg.fps,
        ),
        (Some(k), None) => level_lock(
            &p, k, &lv, &lv, t.plat.as_ref(), cfg.plat_soft, cfg.plat_rail_track,
            cfg.plat_shift_cap, cfg.fps,
        ),
        (None, _) => p,
    };
    let (p, filled) = if cfg.filler_gap_s > 0.0 {
        fill_still_spans(&p, &t.vmarg, &t.level, cfg)
    } else {
        (p, Vec::new())
    };
    (p, sub, force, filled)
}

/// Background filler rhythm (port of jepa_infer `fill_still_spans`):
/// REPLACE the decode's output with the rhythm inside detected gaps.
///
/// AUTHORSHIP, not transcription: deliberately uncorrelated
/// with on-screen motion, OFF by default. Detection is ROW-level -- a
/// garbage passage defeats a segment gate by construction (its phantom
/// wiggles chop one long gap into short still runs): smoothed |vmarg|
/// under `still_eps` marks still rows, and a motion ISLAND between still
/// rows survives -- kept as decoded, interrupting the rhythm -- only on
/// SUSTAINED evidence: its rows at or above `filler_real_v` (smoothed;
/// holds read ~11 pos/s, real motion ~70-107) must total
/// `filler_min_real_s`. A smoothed peak is one row of evidence, so a
/// hallucination burst that spikes once inside a dead passage is
/// bridged, while a real stroke run holds the bar for its duration.
/// Only a SHORT island can be bridged: one at least
/// `filler_max_bridge_s` long always survives whatever its amplitude,
/// so a long genuinely-slow passage is never overwritten.
///
/// Gap accounting is JOINT across kept interruptions: still runs
/// separated by surviving islands shorter than `filler_max_bridge_s`
/// form ONE gap, which fills when its STILL rows total the gap
/// threshold -- a kept island costs its own duration and pauses the
/// rhythm, it never vetoes the gap or restarts the count. An island at
/// least `filler_max_bridge_s` long is sustained real motion and does
/// restart it. One rhythm clock runs across the whole gap (triangle/
/// burst position and sway measured from the gap's first row), so the
/// rhythm resumes after an interruption where the beat would be. A gap
/// totalling under the threshold is left exactly as decoded.
///
/// Inside a gap the output is REPLACED, not decorated: base = the
/// smoothed LEVEL track (the model's pose estimate; the garbage carrier
/// is discarded) plus a bipolar triangle, cross-faded with the carrier
/// over `filler_ramp_s` seconds at each seam of every filled run (0 =
/// one stroke leg), so the track EQUALS the kept output at the
/// boundary rows.
///
/// `filler_model_w` (0..1) is the ONE dial for how much the model's
/// output shapes a filled gap, scaling both influence axes together:
/// the island evidence requirement interpolates from
/// `filler_min_real_s` (1) up to the max-bridge length (0 -- a
/// sub-bridge island can never hold the bar that long, so no island
/// interrupts and the rhythm runs pure), and the base blends from the
/// smoothed level (1) toward a constant per-gap anchor, the mean of
/// the gap's entry and exit levels (0). The >= `filler_max_bridge_s`
/// length pass is untouched at every setting.
fn fill_still_spans(
    p: &[f64],
    vmarg: &[f64],
    level: &[f64],
    cfg: &StyleCfg,
) -> (Vec<f64>, Vec<(usize, usize)>) {
    let fps = cfg.fps;
    let dt = 1.0 / fps;
    let n = p.len();
    let k = rows_at(DECODE_SMOOTH_S, fps, true);
    let av: Vec<f64> = vmarg.iter().map(|v| if v.is_finite() { v.abs() } else { 0.0 }).collect();
    let sm = box_same(&av, k);
    let lv: Vec<f64> = level.iter().map(|&v| if v.is_finite() { v } else { 50.0 }).collect();
    let base = box_same(&lv, k);
    let mut still: Vec<bool> = sm.iter().map(|&s| s < cfg.still_eps).collect();
    let gap_rows = rows_at(cfg.filler_gap_s, fps, false);
    let min_real = rows_at(cfg.filler_min_real_s, fps, false);
    let max_bridge = rows_at(cfg.filler_max_bridge_s, fps, false);
    // bridge: an island failing the duration-AND-confidence test is garbage
    // between rests and joins its gap -- but only a SHORT one. An island at
    // least max_bridge long always survives, whatever its amplitude, so a
    // long genuinely-slow passage can never be overwritten.
    let mw = cfg.filler_model_w.clamp(0.0, 1.0);
    // the model_w dial raises the island evidence bar toward the
    // max-bridge length, which a sub-bridge island cannot reach
    let req = min_real as f64
        + (1.0 - mw) * (max_bridge as f64 - min_real as f64).max(0.0);
    let mut i = 0;
    while i < n {
        if still[i] {
            i += 1;
            continue;
        }
        let mut j = i;
        while j < n && !still[j] {
            j += 1;
        }
        // confidently real = SUSTAINED evidence: rows at or above the
        // bar totalling min_real, not a single smoothed peak
        let hot = sm[i..j].iter().filter(|&&s| s >= cfg.filler_real_v).count();
        if 0 < i && j < n && j - i < max_bridge && (hot as f64) < req {
            still[i..j].fill(true);
        }
        i = j;
    }
    let mut runs: Vec<(usize, usize)> = Vec::new();
    let mut i = 0;
    while i < n {
        if !still[i] {
            i += 1;
            continue;
        }
        let mut j = i;
        while j < n && still[j] {
            j += 1;
        }
        runs.push((i, j));
        i = j;
    }
    let mut out = p.to_vec();
    let mut spans = Vec::new();
    let half_s = 60.0 / cfg.filler_rate.max(1e-9); // seconds per stroke leg
    let rs = if cfg.filler_ramp_s > 0.0 { cfg.filler_ramp_s } else { half_s };
    // burst cycle length in stroke legs (rest converted to legs)
    let cyc = cfg.filler_burst as f64 + cfg.filler_rest_s / half_s;
    // bipolar triangle: zero at xx=0, first peak at xx=0.5
    let tri = |xx: f64| {
        let u = (xx - 0.5).rem_euclid(2.0);
        if u <= 1.0 { 1.0 - 2.0 * u } else { 2.0 * u - 3.0 }
    };
    let mut g = 0;
    while g < runs.len() {
        // one GAP = consecutive still runs whose kept interruptions are
        // each shorter than max_bridge; a longer island is sustained
        // real motion and starts the accounting over
        let mut ge = g + 1;
        while ge < runs.len() && runs[ge].0 - runs[ge - 1].1 < max_bridge {
            ge += 1;
        }
        let total: usize = runs[g..ge].iter().map(|&(a, b)| b - a).sum();
        if total >= gap_rows {
            let g0 = runs[g].0;
            // the model_w dial slides the rhythm's base from the
            // smoothed level toward one constant anchor per gap (mean
            // of the gap's entry and exit levels)
            let anchor = 0.5 * (base[g0] + base[runs[ge - 1].1 - 1]);
            // deterministic per-gap phase: same clip, same knobs -> the
            // same draft, run after run (no RNG at decode)
            let phase = (g0 % 997) as f64 / 997.0 * std::f64::consts::TAU;
            for &(a, b) in &runs[g..ge] {
                let t_end = (b - a - 1) as f64 * dt;
                for q in 0..(b - a) {
                    // ONE rhythm clock for the whole gap: a kept island
                    // gates the rhythm off, and it resumes where the
                    // beat would be, not from zero
                    let tsec = (a + q - g0) as f64 * dt;
                    let x = tsec / half_s; // stroke legs elapsed
                    let w = match cfg.filler_pattern {
                        FillerPattern::Steady => tri(x),
                        FillerPattern::Burst => {
                            let c = x.rem_euclid(cyc.max(1e-9));
                            if c >= cfg.filler_burst as f64 {
                                0.0 // the rest between bursts: parked at base
                            } else {
                                tri(c)
                            }
                        }
                    };
                    let amp_t = cfg.filler_amp
                        * (1.0
                            + cfg.filler_sway
                                * (std::f64::consts::TAU * tsec / cfg.filler_sway_s.max(1e-9)
                                    + phase)
                                    .sin());
                    // the seam cross-fade stays RUN-local: every filled
                    // run hands over to the kept output at its own edges
                    let tr = q as f64 * dt;
                    let r = (tr.min(t_end - tr) / rs.max(1e-9)).clamp(0.0, 1.0);
                    let bg = (1.0 - mw) * anchor + mw * base[a + q];
                    let tgt = (bg + amp_t * w).clamp(0.0, 100.0);
                    out[a + q] = ((1.0 - r) * p[a + q] + r * tgt).clamp(0.0, 100.0);
                }
                spans.push((a, b));
            }
        }
        g = ge;
    }
    (out, spans)
}

/// Port of jepa_infer's `style_positions_level` (`--style level`): strokes
/// oscillate around the position head's local level -- each stroke ends at
/// `level +- travel/2`, holds hold the level. Drift-free: the level track
/// re-anchors every stroke, so amplitude errors never accumulate. The raw
/// predicted travel is mean-collapsed (~0.7x target -- timid strokes);
/// composed styling replaces it with the generated envelope, buying reach
/// and steady amplitude while paying in speed tail and gross excursions.
/// No rails, no locks, no event segmentation, no sub-frame times: the
/// carrier is the whole decode.
pub fn compose_level_full(
    t: &Tracks,
    shot_edges: &[usize],
    cfg: &StyleCfg,
    times_ms: &[f64],
) -> (Vec<f64>, Vec<(usize, usize)>) {
    let p = compose_level(t, shot_edges, cfg);
    // the carrier-agnostic authorship stages apply to THIS style too --
    // depth uniformity regularizes whatever strokes exist, and the filler
    // replaces detected gaps in whatever carrier is there. Same order as
    // compose (uniformity first, filler last); level has no lock between.
    let p = if let Some(dp) = &cfg.depth {
        depth_uniformize(&p, shot_edges, times_ms, dp)
    } else {
        p
    };
    if cfg.filler_gap_s > 0.0 {
        fill_still_spans(&p, &t.vmarg, &t.level, cfg)
    } else {
        (p, Vec::new())
    }
}

pub fn compose_level(t: &Tracks, shot_edges: &[usize], cfg: &StyleCfg) -> Vec<f64> {
    let dt = 1.0 / cfg.fps;
    let mut p = vec![50.0; t.vmarg.len()];
    let kd = rows_at(DECODE_SMOOTH_S, cfg.fps, true);
    let kr = rows_at(cfg.rev_smooth_s, cfg.fps, true);
    let level: Vec<f64> = t
        .level
        .iter()
        .map(|&v| if v.is_finite() { v } else { 50.0 })
        .collect();
    let lv = box_same(&level, kd);
    for w in shot_edges.windows(2) {
        let (lo, hi) = (w[0], w[1]);
        let n = hi - lo;
        if n < 2 {
            continue;
        }
        let v = &t.vmarg[lo..hi];
        let lvl = &lv[lo..hi];
        let s = box_same(v, kr);
        let cross = crossings(&s);
        let mut bounds = Vec::with_capacity(cross.len() + 2);
        bounds.push(0);
        bounds.extend(cross);
        bounds.push(n);
        let mut cur = lvl[0];
        for bw in bounds.windows(2) {
            let (a, b) = (bw[0], bw[1]);
            let tr: f64 = v[a..b].iter().map(|x| x.abs()).sum::<f64>() * dt;
            let d = if b > a {
                sign(s[a..b].iter().sum::<f64>() / (b - a) as f64)
            } else {
                0.0
            };
            if tr <= 1e-9 || d == 0.0 {
                p[lo + a..lo + b].copy_from_slice(&lvl[a..b]);
                if b > a {
                    cur = lvl[b - 1];
                }
                continue;
            }
            let end = (lvl[b - 1] + d * tr.min(100.0) / 2.0).clamp(0.0, 100.0);
            let mut acc = 0.0;
            for (i, x) in v[a..b].iter().enumerate() {
                acc += x.abs();
                p[lo + a + i] = cur + (end - cur) * (acc * dt / tr).clamp(0.0, 1.0);
            }
            cur = end;
        }
    }
    p
}

#[derive(serde::Serialize, Clone, Copy)]
pub struct Action {
    pub at: i64,
    pub pos: i64,
}

/// Funscript actions: every shot's endpoints plus its position extrema.
/// `sub` (apex row -> ms): sub-frame reversal times from `compose`; an
/// extremum row within +-1 of a refined row emits at the refined time
/// (the smoothed-extremum row can sit one off the composed apex).
/// `force` (global rows): decode-called apex rows from the Viterbi graft,
/// emitted as vertices even where the smoothed track shows no flip -- the
/// composed track reverses there by construction, and the smoother would
/// otherwise erase exactly the fast strokes the call recovered.
pub fn extrema_actions(
    p: &[f64],
    times_ms: &[f64],
    shot_edges: &[usize],
    sub: Option<&std::collections::HashMap<usize, f64>>,
    row_hz: f64,
    rev_smooth_s: f64,
    force: &[usize],
) -> Vec<Action> {
    let kr = rows_at(rev_smooth_s, row_hz, true);
    let mut actions: Vec<Action> = Vec::new();
    for w in shot_edges.windows(2) {
        let (lo, hi) = (w[0], w[1]);
        if hi - lo < 2 {
            continue;
        }
        let seg = &p[lo..hi];
        let s = box_same(seg, kr);
        let mut d: Vec<f64> = s.windows(2).map(|w| sign(w[1] - w[0])).collect();
        for v in d.iter_mut() {
            if *v == 0.0 {
                *v = 1.0;
            }
        }
        // np.where(d[1:] * d[:-1] < 0)[0] + 1 -- the extremum sits at the row
        // where the smoothed derivative changed sign, NOT the row after it
        let mut idx: Vec<usize> = vec![0];
        idx.extend((1..d.len()).filter(|&i| d[i] * d[i - 1] < 0.0));
        idx.push(seg.len() - 1);
        idx.extend(force.iter().filter(|&&r| r >= lo && r < hi).map(|&r| r - lo));
        // forced rows arrive out of order, so this sorts before dedup
        // (np.unique); without them idx was ascending by construction
        idx.sort_unstable();
        idx.dedup();
        for i in idx {
            let gl = lo + i;
            let mut t = times_ms[gl];
            if let Some(sub) = sub {
                // lookup order matches jepa_infer: exact, then -1, then +1
                for r in [Some(gl), gl.checked_sub(1), Some(gl + 1)]
                    .into_iter()
                    .flatten()
                {
                    if let Some(&ms) = sub.get(&r) {
                        t = ms;
                        break;
                    }
                }
            }
            actions.push(Action {
                at: t.round() as i64,
                pos: seg[i].clamp(0.0, 100.0).round() as i64,
            });
        }
    }
    actions.sort_by_key(|a| a.at);
    // a cut can put two actions on one millisecond; the first one wins
    actions.dedup_by_key(|a| a.at);
    rdp(&actions, RDP_EPS)
}

/// Douglas-Peucker epsilon for the written action list, pos units --
/// the same value jepa_infer's `--rdp-eps` default applies (port of
/// `common.rdp_actions`). Artifact-only output hygiene: a dense decode
/// emits near-collinear runs on dwells (sub-eps ripple wiggle,
/// int-rounding plateaus) that carry no device motion; a real reversal
/// deviates by its amplitude and survives any eps below it.
const RDP_EPS: f64 = 1.0;

/// The device's peak position speed, pos-units/s -- `common.MAX_POS_RATE`.
/// A funscript cannot be played faster than this, so a transition asking for
/// more depth than the time allows is not a stroke any device performs. The
/// Python decode clamps every list it writes; this is the same cap, applied
/// at the same place (last, on the emitted actions), which is what makes the
/// two one decode. `--max-speed 0` is the deliberate opt-out.
pub const MAX_POS_RATE: f64 = 600.0;

/// Vertical-deviation Douglas-Peucker over a sorted action list.
fn rdp(actions: &[Action], eps: f64) -> Vec<Action> {
    if actions.len() < 3 {
        return actions.to_vec();
    }
    let n = actions.len();
    let mut keep = vec![false; n];
    keep[0] = true;
    keep[n - 1] = true;
    let mut stack = vec![(0usize, n - 1)];
    while let Some((a, b)) = stack.pop() {
        if b - a < 2 {
            continue;
        }
        let (ta, pa) = (actions[a].at as f64, actions[a].pos as f64);
        let (tb, pb) = (actions[b].at as f64, actions[b].pos as f64);
        let dt = (tb - ta).max(1e-9);
        let mut best = a + 1;
        let mut bd = -1.0f64;
        for (o, act) in actions[a + 1..b].iter().enumerate() {
            let line = pa + (pb - pa) * (act.at as f64 - ta) / dt;
            let d = (act.pos as f64 - line).abs();
            if d > bd {
                bd = d;
                best = a + 1 + o;
            }
        }
        if bd > eps {
            keep[best] = true;
            stack.push((a, best));
            stack.push((best, b));
        }
    }
    actions
        .iter()
        .zip(keep)
        .filter(|(_, k)| *k)
        .map(|(a, _)| Action { at: a.at, pos: a.pos })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;

    /// The decode under a CONSTANT prior. Deploy always fits a per-row one (a
    /// band-fitted bias differs row to row), so this is the tests' own way of
    /// asking the scalar question the fixtures below are written against.
    fn alternating_events_gapped(
        p_peak: &[f64],
        p_valley: &[f64],
        bias: f64,
        g: usize,
    ) -> (Vec<usize>, Vec<i8>) {
        alternating_events_rows(p_peak, p_valley, &vec![bias; p_peak.len()], g)
    }

    /// What the emission-prior fit costs at the length that hurts, printed
    /// rather than asserted: 2 hours on the 30 Hz grid, which is the whole
    /// window (`bias_fit_s` is 8000 s) and therefore 216,000 rows behind every
    /// one of the bisection's ~64 probes. This is the dominant term in a review
    /// re-style, and the reason a knob on a long video used to stall the page.
    /// `cargo test --release -- --ignored fit_cost --nocapture`.
    #[test]
    #[ignore]
    fn fit_cost_at_two_hours() {
        let n = 216_000; // 2 h at 30 rows/s
        // a plausible track: a reversal every ~12 rows, alternating, plus floor
        let pk: Vec<f64> = (0..n).map(|i| if i % 24 == 3 { 0.55 } else { 0.02 }).collect();
        let vl: Vec<f64> = (0..n).map(|i| if i % 24 == 15 { 0.50 } else { 0.02 }).collect();
        // two bands, the way the gap prior splits still rows from moving ones
        let band: Vec<usize> = (0..n).map(|i| usize::from((i / 900) % 3 == 0)).collect();
        let targets = [n / 40, n / 30];

        let t = Instant::now();
        let b = fit_emission_bias_bands(&pk, &vl, &band, &targets, 2);
        let fit = t.elapsed();
        let t = Instant::now();
        let rows: Vec<f64> = band.iter().map(|&j| b[j]).collect();
        let (ev, _) = alternating_events_rows(&pk, &vl, &rows, 2);
        let one = t.elapsed();
        println!(
            "  {n} rows: band fit {fit:?}, one decode {one:?}, {} events, bias {b:?}",
            ev.len()
        );
    }

    // A preset ladder is three compile-time constants standing beside a number
    // that arrives at runtime from the bundle. Nothing compares them until the
    // manifest loads, so a decode tuning pass -- which never opens this file --
    // can flatten a rung and leave a menu entry that decodes exactly like its
    // neighbour. It has happened twice: `stillness high` against a stillness
    // floor that rose to meet it, and `dwells eager` against a lock ramp that
    // started above it. Both were silent.
    #[test]
    fn preset_ladders_stay_ordered_against_the_shipped_manifest() {
        // the values the shipped bundle carries
        assert!(presets_ordered(22.0, (0.5, 1.0)).is_ok());
    }

    #[test]
    fn a_flattened_preset_rung_is_refused() {
        // a stillness floor raised onto `high`'s constant: the preset survives
        // in the menu and stops meaning anything
        let e = presets_ordered(26.0, (0.5, 1.0)).unwrap_err();
        assert!(e.contains("--stillness"), "{e}");
        // a lock ramp starting above `eager`'s: same failure, other ladder
        let e = presets_ordered(22.0, (0.3, 1.0)).unwrap_err();
        assert!(e.contains("--dwells"), "{e}");
        // and the inversion, which is worse than a tie -- the rung does the
        // opposite of its own label
        let e = presets_ordered(22.0, (0.9, 1.0)).unwrap_err();
        assert!(e.contains("--dwells"), "{e}");
    }

    #[test]
    fn every_dwell_preset_resolves_somewhere_the_ramp_can_still_discriminate() {
        // the manifest sets the ramp's TOP; a preset that started at or above
        // it would lock nothing at all, whatever the head reported
        let man = (0.5, 1.0);
        for d in [Dwells::Cautious, Dwells::Normal, Dwells::Eager] {
            let (p0, p1) = d.plat_soft(man);
            assert!(p0 < p1, "{} resolves to a dead ramp {p0}..{p1}", d.label());
            assert!((0.0..=1.0).contains(&p0), "{} is off-scale", d.label());
        }
    }

    #[test]
    fn box_same_is_zero_padded_like_numpy() {
        // np.convolve([1,1,1], ones(3)/3, "same") -> [0.667, 1.0, 0.667]
        let got = box_same(&[1.0, 1.0, 1.0], 3);
        assert!((got[0] - 2.0 / 3.0).abs() < 1e-12);
        assert!((got[1] - 1.0).abs() < 1e-12);
        assert!((got[2] - 2.0 / 3.0).abs() < 1e-12);
    }

    #[test]
    fn events_alternate_and_respect_the_refractory() {
        // two clear peaks with a valley between, plus a decoy peak one row
        // after the first: the refractory must reject the adjacent pair and
        // the alternation must forbid peak-after-peak outright.
        let n = 24;
        let mut pk = vec![0.01; n];
        let mut vl = vec![0.01; n];
        pk[4] = 0.9;
        pk[5] = 0.8; // adjacent decoy
        vl[12] = 0.9;
        pk[20] = 0.9;
        let (rows, kinds) = alternating_events_gapped(&pk, &vl, 0.0, 3);
        assert!(!rows.is_empty(), "decode emitted nothing");
        for w in kinds.windows(2) {
            assert!(w[0] != w[1], "same-type transition survived: {kinds:?}");
        }
        for w in rows.windows(2) {
            assert!(w[1] - w[0] >= 3, "refractory violated: {rows:?}");
        }
    }

    /// The gap prior hands the decode a bias that differs ROW TO ROW, which
    /// is a different code path from the scalar one every earlier decode
    /// used. Fixtures are `common.alternating_events`'s own output on the
    /// same closed-form input, so this fails if the two languages ever stop
    /// agreeing about what a per-row prior means.
    #[test]
    fn a_per_row_prior_decodes_like_the_python() {
        let n = 40;
        let pk: Vec<f64> = (0..n).map(|i| if i % 6 == 1 { 0.40 } else { 0.05 }).collect();
        let vl: Vec<f64> = (0..n).map(|i| if i % 6 == 4 { 0.35 } else { 0.05 }).collect();
        let bias: Vec<f64> = (0..n).map(|i| if i < 20 { -1.0 } else { 2.0 }).collect();
        let (rows, kinds) = alternating_events_rows(&pk, &vl, &bias, 2);
        assert_eq!(rows, vec![22, 25, 28, 31, 34, 37]);
        assert_eq!(kinds, vec![-1, 1, -1, 1, -1, 1]);
        // and a CONSTANT per-row prior is the scalar path exactly -- the
        // refactor that introduced the vector must not have moved it
        let (rc, _) = alternating_events_gapped(&pk, &vl, 0.5, 2);
        assert_eq!(rc, vec![1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34, 37]);
        let (rv, _) = alternating_events_rows(&pk, &vl, &vec![0.5; n], 2);
        assert_eq!(rc, rv);
    }

    /// The refractory's exact reach, pinned in BOTH languages: `g` is `g`
    /// event-free rows after an event, so a strong pair `g` rows apart
    /// loses one event and a pair `g + 1` apart keeps both. The two
    /// fixtures above do not bind on the refractory -- their events are
    /// already 3 and 4 rows apart -- so without this one a reach that moved
    /// on one side alone would pass every cross-language test. Values are
    /// `common.alternating_events`'s own, and `grid_check.py` pins the same
    /// pair on the Python side.
    #[test]
    fn the_refractory_reach_is_one_row_past_the_gap() {
        for g in [2usize, 3] {
            let mut pk = vec![1e-3f64; 8];
            pk[0] = 0.98;
            let mut vl = vec![1e-3f64; 8];
            vl[g] = 0.98;
            let bias = vec![0.0f64; 8];
            let (at_g, _) = alternating_events_rows(&pk, &vl, &bias, g);
            assert_eq!(at_g.len(), 1, "g={g}: a pair {g} apart must lose one");
            vl[g] = 1e-3;
            vl[g + 1] = 0.98;
            let (past, _) = alternating_events_rows(&pk, &vl, &bias, g);
            assert_eq!(past.len(), 2, "g={g}: a pair {} apart must keep both", g + 1);
        }
    }

    /// The band fit is a coordinate bisection whose coordinates COUPLE
    /// through the alternation constraint, so it is not guaranteed to seat
    /// every band on its target -- on real clips it lands within two events,
    /// on an adversarial input it can stall. Either way the two languages
    /// must agree, which is what parity means, so the fixture is
    /// `common.fit_emission_bias_bands`'s own answer on this input rather
    /// than the targets themselves.
    #[test]
    fn the_band_fit_agrees_with_the_python() {
        let n = 400usize;
        let mut pk: Vec<f64> = (0..n).map(|i| if i % 8 == 1 { 0.42 } else { 0.06 }).collect();
        let mut vl: Vec<f64> = (0..n).map(|i| if i % 8 == 5 { 0.38 } else { 0.06 }).collect();
        for i in 150..250 {
            pk[i] *= 0.4;
            vl[i] *= 0.4;
        }
        let band: Vec<usize> = (0..n).map(|i| usize::from(!(150..250).contains(&i))).collect();
        let bb = fit_emission_bias_bands(&pk, &vl, &band, &[8, 61], 2);
        assert!((bb[0] - 1.6307373046875).abs() < 1e-9, "band0 {}", bb[0]);
        assert!((bb[1] - 0.3006591796875).abs() < 1e-9, "band1 {}", bb[1]);
    }

    #[test]
    fn the_emission_prior_is_monotone_in_event_count() {
        // what makes the bisection in fit_emission_bias exact
        let n = 60;
        let pk: Vec<f64> = (0..n).map(|i| if i % 10 == 3 { 0.4 } else { 0.02 }).collect();
        let vl: Vec<f64> = (0..n).map(|i| if i % 10 == 8 { 0.4 } else { 0.02 }).collect();
        let low = alternating_events_gapped(&pk, &vl, -3.0, 2).0.len();
        let mid = alternating_events_gapped(&pk, &vl, 0.0, 2).0.len();
        let high = alternating_events_gapped(&pk, &vl, 3.0, 2).0.len();
        assert!(low <= mid && mid <= high, "{low} {mid} {high}");
        // and the fit lands on a prior that reaches the asked-for count
        let b = fit_emission_bias(&pk, &vl, mid.max(1), 2);
        let got = alternating_events_gapped(&pk, &vl, b, 2).0.len();
        assert!(got >= mid.max(1), "fit under-emitted: {got} < {mid}");
    }

    #[test]
    fn crossings_are_sign_flips() {
        assert_eq!(crossings(&[1.0, 1.0, -1.0, -1.0, 1.0]), vec![2, 4]);
    }

    #[test]
    fn default_params_are_an_identity() {
        // the acceptance bar for the whole knob surface: a bare run's draft
        // is bit-for-bit what the manifest alone produces
        // 75 units over 200 ms is 375 pos/s -- inside the device cap, so the
        // defaults have nothing to do to it
        let mut a = vec![Action { at: 0, pos: 13 }, Action { at: 200, pos: 88 }];
        shape_actions(&mut a, &Params::default());
        assert_eq!((a[0].pos, a[1].pos), (13, 88));
    }

    #[test]
    fn max_speed_caps_the_step() {
        let mut a = vec![Action { at: 0, pos: 0 }, Action { at: 1000, pos: 100 }];
        shape_actions(&mut a, &Params { max_speed: 40.0, ..Params::default() });
        assert_eq!(a[1].pos, 40); // 40 units/s over one second
    }

    #[test]
    fn the_device_cap_is_on_by_default() {
        // 75 units over 67 ms asks for ~1119 pos/s. The Python decode clamps
        // this on every list it writes; a draft that ships it is a decode the
        // harness never scored.
        let mut a = vec![Action { at: 0, pos: 13 }, Action { at: 67, pos: 88 }];
        shape_actions(&mut a, &Params::default());
        assert_eq!(a[1].pos, 53); // 13 + trunc(600 * 0.067)
        // ...and 0 is still the way out of it
        let mut b = vec![Action { at: 0, pos: 13 }, Action { at: 67, pos: 88 }];
        shape_actions(&mut b, &Params { max_speed: 0.0, ..Params::default() });
        assert_eq!(b[1].pos, 88);
    }

    #[test]
    fn the_cap_truncates_rather_than_rounds() {
        // 1 ms at 600 pos/s allows 0.6 units: rounding would write 1 and hand
        // back 1000 pos/s, which is the bug this clamp exists to prevent.
        let mut a = vec![Action { at: 0, pos: 50 }, Action { at: 1, pos: 90 }];
        shape_actions(&mut a, &Params::default());
        assert_eq!(a[1].pos, 50);
    }

    #[test]
    fn shift_cap_bounds_the_lock_correction() {
        // a flat mid track called a top dwell against a far rail: uncapped,
        // the lock walks the mean the whole way; capped, no row moves more
        // than the cap (0 = uncapped stays the pre-cap behavior)
        let n = 60;
        let p = vec![50.0; n];
        let kind: Vec<i8> = (0..n).map(|i| if (20..40).contains(&i) { 1 } else { 0 }).collect();
        let rail = vec![90.0; n];
        let free = level_lock(&p, &kind, &rail, &rail, None, (0.0, 0.0), false, 0.0, 30.0);
        let capped = level_lock(&p, &kind, &rail, &rail, None, (0.0, 0.0), false, 25.0, 30.0);
        assert!((free[30] - 90.0).abs() < 1e-9); // full walk to the rail
        assert!((capped[30] - 75.0).abs() < 1e-9); // 50 + 25, no further
        assert!(capped.iter().zip(&p).all(|(c, o)| (c - o).abs() <= 25.0 + 1e-9));
    }


    fn filler_cfg() -> StyleCfg {
        StyleCfg {
            fps: 30.0,
            still_eps: 15.0,
            ext_snap: 20.0,
            plat_thr: 0.5,
            plat_lo: 0.3,
            plat_peak: 0.65,
            plat_veto: 15.0,
            plat_rail_track: true,
            plat_soft: (0.0, 0.0),
            plat_shift_cap: 25.0,
            rev_snap: 0,
            rev_viterbi: false,
            rev_gap_rows: 2,
            rev_gap_prior: false,
            speed_ref_s: 0.6,
            rev_smooth_s: 0.1,
            bias_fit_rows: 0,
            subframe_rev: false,
            filler_gap_s: 10.0,
            filler_min_real_s: 1.0,
            filler_real_v: 45.0,
            filler_model_w: 1.0,
            filler_rate: 60.0,
            filler_amp: 20.0,
            filler_ramp_s: 0.0,
            filler_max_bridge_s: 5.0,
            filler_sway: 0.0, // metronome: the extremes assert exactly
            filler_sway_s: 16.0,
            filler_pattern: FillerPattern::Steady,
            filler_burst: 4,
            filler_rest_s: 2.0,
            depth: None,
        }
    }

    #[test]
    fn filler_replaces_gaps_and_bridges_garbage_islands() {
        // 30 rows/s, 40 s of quiet vmarg carrying three motion islands: a
        // weak 0.5 s wiggle (garbage -- bridged, its output replaced), a
        // strong 2 s stroke burst (real -- splits the gap, rows untouched)
        // and a LONG weak passage (6.7 s at 25 pos/s -- longer than
        // max_bridge, so it survives whatever its amplitude: a slow
        // passage is never overwritten). Seam rows equal the carrier.
        let n = 1500;
        let mut v = vec![0.0; n];
        for i in 300..315 {
            v[i] = 60.0; // weak island: smoothed peak ~29 < 45
        }
        for i in 600..660 {
            v[i] = 120.0; // real island: smoothed peak >= 45, 2 s >= 1 s
        }
        for i in 800..1000 {
            v[i] = 25.0; // long weak passage: > max_bridge, survives
        }
        let level = vec![50.0; n];
        // a garbage carrier that wiggles off the level everywhere
        let p: Vec<f64> = (0..n).map(|i| 50.0 + ((i % 7) as f64) - 3.0).collect();
        let (out, spans) = fill_still_spans(&p, &v, &level, &filler_cfg());
        assert!((out[0] - p[0]).abs() < 1e-9); // clip-start seam = carrier
        // deep inside the first gap the wiggle is GONE: the track is the
        // level base +- the wave, whose extremes hit 50 +- 20
        let lo = out[60..240].iter().cloned().fold(f64::INFINITY, f64::min);
        let hi = out[60..240].iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        assert!((lo - 30.0).abs() < 1.5 && (hi - 70.0).abs() < 1.5);
        // the weak island's rows were bridged: replaced, not preserved
        assert!((out[307] - p[307]).abs() > 0.5);
        // the strong island's rows are untouched -- tracking is kept
        assert_eq!(&out[615..645], &p[615..645]);
        // the long weak passage's rows are untouched too (bridge cap)
        assert_eq!(&out[850..950], &p[850..950]);
        // JOINT accounting: the kept island paused the gap instead of
        // vetoing it, so the ~4.4 s still run behind it (never long
        // enough alone) fills as part of the same gap
        let lo2 = out[705..770].iter().cloned().fold(f64::INFINITY, f64::min);
        let hi2 = out[705..770].iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        assert!((lo2 - 30.0).abs() < 1.5 && (hi2 - 70.0).abs() < 1.5);
        // ...while the >= max_bridge slow passage ENDS the gap, leaving
        // three filled runs: before the island, behind it, and after
        // the slow passage
        assert_eq!(spans.len(), 3);
        assert!(!spans.is_empty() && spans[0].0 == 0);
        assert!(out.iter().all(|&x| (0.0..=100.0).contains(&x)));
    }

    #[test]
    fn filler_bridges_islands_without_sustained_evidence() {
        // a 3 s island of slow wobble (30 pos/s) carrying one strong
        // 0.5 s spike: its smoothed peak clears the confidence bar, but
        // the rows at/above the bar total ~1.4 s < the 2 s sustained
        // requirement -- a hallucination burst, bridged and replaced
        // (a single peak is one row of evidence, not a certificate)
        let n = 1200;
        let mut v = vec![0.0; n];
        for i in 600..690 {
            v[i] = 30.0;
        }
        for i in 630..645 {
            v[i] = 200.0; // smoothed peak ~112 >= 45, hot run ~41 rows
        }
        let level = vec![50.0; n];
        let p = vec![40.0; n];
        let mut cfg = filler_cfg();
        cfg.filler_min_real_s = 2.0;
        let (out, spans) = fill_still_spans(&p, &v, &level, &cfg);
        // the whole clip is ONE filled gap: the island is gone
        assert_eq!(spans, vec![(0, n)]);
        // deep inside, the rhythm asserts exactly (peak row: base + amp)
        assert!((out[615] - 70.0).abs() < 1e-9);
    }

    #[test]
    fn filler_joint_gap_accounting_resumes_on_the_beat() {
        // two ~6.3 s still runs around a kept 2 s real island: neither
        // run reaches the 10 s gap alone, their STILL time does -- both
        // fill, the island's rows stay as decoded, and the rhythm
        // resumes behind it ON the group clock (the beat that started
        // at row 0), not from zero
        let n = 460;
        let mut v = vec![0.0; n];
        for i in 200..260 {
            v[i] = 100.0; // sustained >= 45 for ~2 s: kept as real
        }
        let level = vec![50.0; n];
        let p = vec![40.0; n];
        let (out, spans) = fill_still_spans(&p, &v, &level, &filler_cfg());
        assert_eq!(spans.len(), 2);
        assert_eq!(&out[210..250], &p[210..250]); // island untouched
        assert!((out[0] - 40.0).abs() < 1e-9); // seam = carrier
        // peak rows of the ONE clock (60/min from row 0): row 75 in the
        // first run, row 315 in the second -- a run-local clock would
        // put row 315 near a VALLEY (~31), not the peak
        assert!((out[75] - 70.0).abs() < 1e-9);
        assert!((out[315] - 70.0).abs() < 1e-9);
    }

    #[test]
    fn filler_model_w_dial_scales_both_influence_axes() {
        // one 2 s strong island (kept at w=1) inside 30 s of quiet, on a
        // level that drifts 40 -> 60. The dial: at 1.0 the island
        // interrupts and the rhythm rides the drifting level; at 0.5 the
        // evidence bar has risen to 3 s (min_real 1 s + half the gap to
        // max_bridge 5 s) so the 2 s island is garbage; at 0.0 the
        // rhythm is pure -- nothing interrupts, and the base is ONE
        // constant anchor per gap, so rows one full period apart write
        // the SAME value while the w=1 fill drifts between them.
        let n = 900;
        let mut v = vec![0.0; n];
        for i in 400..460 {
            v[i] = 120.0; // sustained >= 45 for ~2 s
        }
        let level: Vec<f64> =
            (0..n).map(|i| 40.0 + 20.0 * i as f64 / (n - 1) as f64).collect();
        let p = vec![40.0; n];
        let cfg = filler_cfg();
        let (w1, s1) = fill_still_spans(&p, &v, &level, &cfg);
        assert_eq!(s1.len(), 2); // island kept: two filled runs, one gap
        assert_eq!(&w1[420..440], &p[420..440]);
        let mut cfg_h = filler_cfg();
        cfg_h.filler_model_w = 0.5;
        let (wh, sh) = fill_still_spans(&p, &v, &level, &cfg_h);
        assert_eq!(sh, vec![(0, n)]); // bar risen past the island
        assert!((wh[430] - p[430]).abs() > 0.5);
        let mut cfg_0 = filler_cfg();
        cfg_0.filler_model_w = 0.0;
        let (w0, s0) = fill_still_spans(&p, &v, &level, &cfg_0);
        assert_eq!(s0, vec![(0, n)]);
        // rate 60/min at 30 fps: one full period = 60 rows. Anchored
        // base -> equal values a period apart; level base -> drift
        assert!((w0[300] - w0[600]).abs() < 1e-9);
        assert!((w1[300] - w1[600]).abs() > 3.0);
    }

    #[test]
    fn filler_long_real_island_ends_the_gap() {
        // the same layout but the island is 6 s -- past max_bridge, so
        // it is sustained real motion: it RESETS gap accounting, each
        // flanking still run stands alone under the threshold, and
        // nothing fills
        let n = 580;
        let mut v = vec![0.0; n];
        for i in 200..380 {
            v[i] = 100.0;
        }
        let level = vec![50.0; n];
        let p = vec![40.0; n];
        let (out, spans) = fill_still_spans(&p, &v, &level, &filler_cfg());
        assert!(spans.is_empty());
        assert_eq!(out, p);
    }

    #[test]
    fn filler_burst_pattern_rests_at_the_base() {
        // pattern burst, 2 strokes then 1 s rest at 60/min: the cycle is 3
        // legs (90 rows). Inside the rest window the track sits ON the
        // base; at a burst peak it reaches base + amp.
        let n = 600;
        let v = vec![0.0; n];
        let level = vec![50.0; n];
        let p = vec![50.0; n];
        let mut cfg = filler_cfg();
        cfg.filler_pattern = FillerPattern::Burst;
        cfg.filler_burst = 2;
        cfg.filler_rest_s = 1.0;
        let (out, _) = fill_still_spans(&p, &v, &level, &cfg);
        assert!((out[75] - 50.0).abs() < 1e-9); // rest: parked at base
        // first full-strength burst peak (past the seam ramp): x = 3.5
        // legs -> cycle pos 0.5 -> +amp
        assert!((out[105] - 70.0).abs() < 1e-9);
    }

    #[test]
    fn depth_uniformity_off_is_identity() {
        // dose 0 must return the track byte-for-byte -- the acceptance bar the
        // whole knob shares (a bare run's draft is the manifest's)
        let p = vec![50.0, 80.0, 50.0, 30.0, 50.0, 90.0, 50.0];
        let t: Vec<f64> = (0..p.len()).map(|i| i as f64 * 66.67).collect();
        let out = depth_uniformize(&p, &[0, p.len()], &t, &DepthParams { dose: 0.0, window_s: 10.0 });
        assert_eq!(out, p);
    }

    #[test]
    fn depth_uniformity_pulls_peaks_together() {
        // two tops at different heights (80, 100); a full-dose global window
        // pulls both to their common level (the median of two IS their
        // mean, 90) while the lone valley stays put
        let mut p = vec![50.0];
        for (a, b) in [(50.0, 80.0), (80.0, 50.0), (50.0, 100.0), (100.0, 50.0)] {
            for j in 1..=4 {
                p.push(a + (b - a) * j as f64 / 4.0);
            }
        } // rows: peak 80 @4, valley 50 @8, peak 100 @12
        let t: Vec<f64> = (0..p.len()).map(|i| i as f64 * 66.67).collect();
        let n = p.len();
        let out = depth_uniformize(&p, &[0, n], &t, &DepthParams { dose: 1.0, window_s: 1000.0 });
        assert!((out[4] - 90.0).abs() < 1e-6, "top1: {out:?}");
        assert!((out[12] - 90.0).abs() < 1e-6, "top2: {out:?}");
        assert!((out[8] - 50.0).abs() < 1e-6, "valley: {out:?}");
        // shot ends are fixed anchors
        assert_eq!(out[0], 50.0);
        assert_eq!(out[n - 1], 50.0);
    }

    #[test]
    fn depth_uniformity_moves_the_outlier_not_the_matched_pair() {
        // three tops 80/80/100: the MEDIAN target lines the window up with
        // the least total movement -- the matched pair stays exactly put
        // and the outlier comes to them (a mean would drag all three)
        let mut p = vec![50.0];
        for (a, b) in [
            (50.0, 80.0),
            (80.0, 50.0),
            (50.0, 80.0),
            (80.0, 50.0),
            (50.0, 100.0),
            (100.0, 50.0),
        ] {
            for j in 1..=4 {
                p.push(a + (b - a) * j as f64 / 4.0);
            }
        } // peaks at rows 4, 12, 20
        let t: Vec<f64> = (0..p.len()).map(|i| i as f64 * 66.67).collect();
        let n = p.len();
        let out = depth_uniformize(&p, &[0, n], &t, &DepthParams { dose: 1.0, window_s: 1000.0 });
        assert!((out[4] - 80.0).abs() < 1e-6, "matched top moved: {out:?}");
        assert!((out[12] - 80.0).abs() < 1e-6, "matched top moved: {out:?}");
        assert!((out[20] - 80.0).abs() < 1e-6, "outlier not aligned: {out:?}");
    }

    #[test]
    fn extrema_land_on_the_reversal_row() {
        // a single clean peak: up, peak, down. The action belongs AT the peak.
        // (An off-by-one here shifts every reversal a frame late, which is the
        // one thing the product is judged on.)
        let p = [0.0, 50.0, 100.0, 50.0, 0.0];
        let t: Vec<f64> = (0..5).map(|i| i as f64 * 66.67).collect();
        let acts = extrema_actions(&p, &t, &[0, 5], None, 15.0, 3.0 / 15.0, &[]);
        let peak = acts.iter().max_by_key(|a| a.pos).unwrap();
        assert_eq!(peak.at, 133); // row 2, not row 3
    }
}
