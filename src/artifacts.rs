//! Artifact rarity events of the WRITTEN action list — the port of
//! jepa_infer's `artifact_events`, reading the SAME embedded
//! `artifact_prior.json` (the one prior definition, include_str!'d like
//! projector.js, so the two sides cannot drift). An artifact is a
//! non-sensical action SEQUENCE — one that almost never occurs in the
//! training data — so the instrument is windowed bigram NLL over
//! quantized stroke tokens, fit on authored corpus scripts and grounded
//! against the user's eye. Reference-free
//! and action-domain: it scores the artifact layer on any clip.
//!
//! An INSTRUMENT, never a gate: the spans are stamped into the written
//! funscript's `metadata.artifacts` (the review page draws them as a
//! band) and the rate prints next to the authors' own anchor rate.
//! Nothing is filtered or replaced.

use std::sync::OnceLock;

use crate::style::Action;

#[derive(serde::Deserialize)]
pub struct Prior {
    pub dur_edges_s: Vec<f64>,
    pub amp_edges: Vec<f64>,
    pub win: usize,
    pub thr: f64,
    pub anchor_per_min: f64,
    pub logp: Vec<Vec<f64>>,
}

pub fn prior() -> &'static Prior {
    static P: OnceLock<Prior> = OnceLock::new();
    P.get_or_init(|| {
        serde_json::from_str(include_str!("../../artifact_prior.json"))
            .expect("artifact_prior.json embedded at build time")
    })
}

/// `np.digitize(x, edges)`: the count of edges <= x.
fn digitize(x: f64, edges: &[f64]) -> usize {
    edges.iter().filter(|e| **e <= x).count()
}

/// `np.sign`: zero for zero (f64::signum(0.0) is 1.0, which would
/// segment a stroke starting on a flat step differently than python).
fn sign(x: f64) -> f64 {
    if x > 0.0 {
        1.0
    } else if x < 0.0 {
        -1.0
    } else {
        0.0
    }
}

/// Hot rarity spans (ms, on the actions' own clock) and their rate per
/// minute of scripted time. Mirrors jepa_infer row for row: strokes =
/// reversal-to-reversal segments (zero-delta steps extend the current
/// segment), token = duration bin x 6 + amplitude bin, score =
/// `win`-stroke mean bigram NLL, a span runs from its first hot
/// window's stroke to `win` strokes past its last.
pub fn artifact_events(actions: &[Action]) -> (Vec<[i64; 2]>, f64) {
    let pr = prior();
    let a: Vec<(f64, f64)> = actions
        .iter()
        .map(|x| (x.at as f64 / 1000.0, x.pos as f64))
        .collect();
    let mut strokes: Vec<(f64, f64, f64)> = Vec::new(); // t0_s, dur_s, amp
    let mut i = 0;
    while i + 1 < a.len() {
        let mut j = i + 1;
        let d0 = sign(a[j].1 - a[i].1);
        while j + 1 < a.len()
            && (sign(a[j + 1].1 - a[j].1) == d0 || a[j + 1].1 == a[j].1)
        {
            j += 1;
        }
        strokes.push((a[i].0, a[j].0 - a[i].0, (a[j].1 - a[i].1).abs()));
        i = j;
    }
    if strokes.len() < pr.win + 1 {
        return (Vec::new(), 0.0);
    }
    let tok: Vec<usize> = strokes
        .iter()
        .map(|s| digitize(s.1, &pr.dur_edges_s) * 6 + digitize(s.2, &pr.amp_edges))
        .collect();
    let nll: Vec<f64> = tok
        .windows(2)
        .map(|w| -pr.logp[w[0]][w[1]])
        .collect();
    let w: Vec<f64> = nll
        .windows(pr.win)
        .map(|x| x.iter().sum::<f64>() / pr.win as f64)
        .collect();
    let mut spans = Vec::new();
    let mut i = 0;
    while i < w.len() {
        if w[i] < pr.thr {
            i += 1;
            continue;
        }
        let mut j = i;
        while j < w.len() && w[j] >= pr.thr {
            j += 1;
        }
        let e = (j + pr.win).min(strokes.len() - 1);
        spans.push([
            (strokes[i].0 * 1000.0).round() as i64,
            (strokes[e].0 * 1000.0).round() as i64,
        ]);
        i = j;
    }
    let last = strokes.last().unwrap();
    let dur_min = (last.0 + last.1 - strokes[0].0) / 60.0;
    (spans.clone(), spans.len() as f64 / dur_min.max(1e-9))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn acts(pairs: &[(i64, i64)]) -> Vec<Action> {
        pairs.iter().map(|&(at, pos)| Action { at, pos }).collect()
    }

    #[test]
    fn parity_with_jepa_infer_on_the_synthetic_nonsense_cycle() {
        // the EXACT sequence evaluated against python's artifact_events
        // with the same embedded artifact_prior.json: 30 common strokes
        // (500 ms, amp 50), five 4-stroke nonsense cycles alternating
        // the corpus's rarest realizable token pair (100 ms/amp 80 vs
        // 200 ms/amp 20, net drift zero), 30 common again. Python:
        // spans [[12500, 20500]] ms, rate 20/11 per min. A drift here
        // means the two sides no longer read the same instrument.
        let mut p = vec![(0i64, 10i64)];
        let (mut t, mut pos) = (0i64, 10i64);
        for _ in 0..30 {
            t += 500;
            pos = if pos == 10 { 60 } else { 10 };
            p.push((t, pos));
        }
        for _ in 0..5 {
            t += 100;
            p.push((t, 90));
            t += 200;
            p.push((t, 70));
            t += 200;
            p.push((t, 90));
            t += 100;
            p.push((t, 10));
        }
        pos = 10;
        for _ in 0..30 {
            t += 500;
            pos = if pos == 10 { 60 } else { 10 };
            p.push((t, pos));
        }
        let (spans, rate) = artifact_events(&acts(&p));
        assert_eq!(spans, vec![[12500, 20500]]);
        assert!((rate - 20.0 / 11.0).abs() < 1e-12, "rate {rate}");
    }

    #[test]
    fn authored_rhythm_reads_quiet() {
        // plain alternation at a common stroke shape: no hot span, so
        // the metadata field stays absent and drafts stay byte-stable
        let mut p = vec![(0i64, 10i64)];
        let (mut t, mut pos) = (0i64, 10i64);
        for _ in 0..60 {
            t += 500;
            pos = if pos == 10 { 60 } else { 10 };
            p.push((t, pos));
        }
        let (spans, rate) = artifact_events(&acts(&p));
        assert!(spans.is_empty());
        assert_eq!(rate, 0.0);
    }
}
