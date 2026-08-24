//! The viewport: what the goblins are looking at, painted over the log.
//!
//! Two panels above the live status line, for as long as a draft is running.
//!
//! **Goblin vision**, the encoder's own 24x24 latent grid painted as a single
//! heat field, and beside it **the film**: the same frame in colour, box-
//! averaged to the SAME 24x24 grid. One cell of one is one cell of the other --
//! that is the whole point of drawing them side by side, and it is why the film
//! is not drawn at a nicer resolution. The frames are the encoder's own input
//! (a square anamorphic squash of the 16:9 picture, which is what the model
//! sees), so a bright cell in the left panel names a cell in the right one.
//!
//! **The field is the mask net's attention** -- the ROI itself, the weighting
//! the trunk is about to pool the frame with. It comes from `mask.onnx`, a
//! graph of the mask stack alone that `encode.rs` runs on one latent row per
//! window. It is the REAL gate and not a likeness of it: the mask net is a 2-D
//! conv stack over a single grid with no cut flag, no temporal context and no
//! dropout, so one row through it computes exactly what the draft's own
//! attention pooling used on that row. The head's copy is discarded unread, and
//! asking `head.onnx` for it instead would light the panel only for the seconds
//! that stage lasts, after the minutes of encoding it exists to fill.
//!
//! **The eight heads are combined the way the pooling combines them.** Each
//! head is normalized over the grid to sum to one and the normalized maps are
//! averaged. That is not a stylistic choice between mean and max: attention
//! pooling divides each head's gate by its own sum before it weights anything
//! (`att = gate / gate.sum()`), so the normalized map IS the weight that head
//! carries, and the raw sigmoid is not. A head sitting at 0.9 across the whole
//! grid is loud in the raw gate and says nothing about where the model looks;
//! normalized, it contributes a flat nothing and the peaked heads show through.
//! The combined field is one probability distribution over the grid, which is
//! exactly what "where is it looking" means here.
//!
//! **Latent band energy is the fallback**, for a bundle exported before the
//! mask graph: the per-cell L2 norm across the PCA dimensions of the row the
//! encoder just wrote. It is free -- those numbers are already in registers on
//! their way to the cache -- and it shows where the picture is busy rather than
//! where the model is looking. The caption says which of the two is on screen.
//!
//! It feeds nothing. The viewport is DECORATION, the draft is byte-identical
//! with it on or off, and malformed input is dropped rather than raised -- a
//! panel that could fail in a way the funscript noticed would be a bad trade
//! for a picture.
//!
//! Two things keep it from flickering, and both are load-bearing.
//!
//! **The field is smoothed in time, and normalized on its own range.**
//! Normalizing against the field's own min/max is what makes the contrast worth
//! watching -- an attention map that lives between 0.4 and 0.6 of its peak is
//! one flat grey on an absolute scale -- but done raw it rescales the whole
//! picture every window, so a still scene strobes. The field runs through an
//! EMA and its normalization window chases its extremes more slowly still. A
//! change of SOURCE reseeds both outright rather than easing, since attention
//! and band energy are different quantities and a blend of them was never
//! measured.
//!
//! **The panels are repainted in place**, never cleared first -- see
//! `render_loop` in `main.rs`. Clearing a 13-row block and refilling it eight
//! times a second is visible as a flash; overwriting each row and erasing only
//! its tail is not.
//!
//! Colour is the active palette's `ramp` (`theme.rs`): twelve xterm-256 steps
//! that run through the palette's own hues from black to white, so the field
//! reads as a thermal map rather than one hue's brightness, and T recolours it
//! live. The film is the one panel the palette does not own -- a frame of the
//! video is the video's colours or it is not the film -- and it is quantized to
//! the xterm-256 cube like everything else. No truecolour anywhere: an index
//! renders the same in every stack the app draws through.

use std::sync::mpsc::{sync_channel, SyncSender, TrySendError};
use std::sync::Mutex;
use std::thread::JoinHandle;

use ort::session::Session;

use crate::theme::theme;

/// Panel height in terminal rows. The panel packs two grid rows into one
/// terminal row (upper-half block, foreground over background), so a 24x24 grid
/// lands in 12 -- and since a terminal cell is about twice as tall as it is
/// wide, 24 columns over 12 rows is what draws the grid SQUARE.
const PANEL_H: usize = 12;

/// Left margin, matching the live line's own indent.
const INDENT: usize = 2;

/// Columns between goblin vision and the film.
const GAP: usize = 2;

/// Rows the viewport occupies: the panels plus their caption.
const VIEW_H: usize = PANEL_H + 1;

/// The log needs this many rows before a viewport is worth taking some of them.
/// Below it the panels are simply not drawn and the run looks as it always did.
const MIN_LOG_H: usize = VIEW_H + 8;

/// How much of each new row's field is taken (the rest is what is already on
/// screen). At one window every ~110 ms of wall clock this settles in about a
/// third of a second -- fast enough to follow a cut, slow enough not to strobe.
const FIELD_EMA: f32 = 0.35;

/// How fast the normalization window chases the field's own extremes. Slower
/// than the field itself on purpose: the RANGE jumping is what rescales the
/// whole picture at once, which reads as a flash even when the field is calm.
const RANGE_EMA: f32 = 0.12;

/// Which of the two sources filled the field. The caption says so, and a change
/// of source reseeds rather than easing one quantity into the other.
#[derive(Default, Clone, Copy, PartialEq)]
enum Source {
    #[default]
    Bands,
    Attention,
}

impl Source {
    fn label(self) -> &'static str {
        match self {
            Source::Attention => crate::t!("viewport.src.attention"),
            Source::Bands => crate::t!("viewport.src.bands"),
        }
    }
}

/// What the render thread paints. Written by the encode stage, read by the
/// render thread.
#[derive(Default)]
struct View {
    /// Smoothed per-cell field, row-major `g` x `g`, in its raw units (pooled
    /// attention weight, or band energy) -- the EMA runs here rather than on
    /// the normalized field so that a moving range cannot feed back into it.
    field: Vec<f32>,
    /// The normalization window, chasing the field's extremes.
    lo: f32,
    hi: f32,
    g: usize,
    src: Source,
    /// False until the first row, which seeds the field and the range outright
    /// instead of easing into them from zero.
    seeded: bool,
    /// The film: `fg` x `fg` RGB cells, empty until a frame arrives (and it
    /// never does on a run reading its latents back from the cache -- there is
    /// no decode there to take a frame from).
    frame: Vec<[u8; 3]>,
    fg: usize,
}

static VIEW: Mutex<Option<View>> = Mutex::new(None);

fn with<R>(f: impl FnOnce(&mut View) -> R) -> R {
    let mut g = VIEW.lock().unwrap();
    f(g.get_or_insert_with(View::default))
}

/// Drop what is on screen -- called between videos, so a batch never shows the
/// last film under the current one's progress.
pub fn clear() {
    *VIEW.lock().unwrap() = None;
}

/// Is the viewport being drawn at all? A short terminal keeps its log.
fn fits(log_h: usize) -> bool {
    log_h >= MIN_LOG_H
}

/// The mask net's gate for one row -> the field.
///
/// `gate` is `mask.onnx`'s output for a single latent row, laid out
/// `(heads, g, g)`: the per-cell sigmoid, before the pooling normalizes it.
/// Each head is normalized here exactly as the pooling does and the maps are
/// averaged, so the field is one distribution over the grid however many heads
/// the checkpoint carries.
///
/// Returns whether the gate was drawn -- `false` is a malformed buffer, which
/// the caller reads as "this graph has no picture for me" and falls back to
/// band energy.
#[must_use]
pub fn publish_gate(gate: &[f32], heads: usize, g: usize) -> bool {
    let cells = g * g;
    if cells == 0 || heads == 0 || gate.len() < heads * cells {
        return false;
    }
    let mut f = vec![0.0f32; cells];
    for h in 0..heads {
        let m = &gate[h * cells..(h + 1) * cells];
        // a head that gates nothing anywhere carries no weight and no picture;
        // dividing by its sum would be an infinity where it means a zero
        let sum: f32 = m.iter().sum();
        if !sum.is_finite() || sum <= 1e-6 {
            continue;
        }
        for (x, &v) in f.iter_mut().zip(m.iter()) {
            *x += v / sum;
        }
    }
    for v in f.iter_mut() {
        *v /= heads as f32;
    }
    publish(f, g, Source::Attention);
    true
}

/// One latent row -> the field, the fallback source.
///
/// `row` is the int8 row the encoder just wrote, laid out `(dim, g, g)`, so the
/// energy is one pass over the planes.
pub fn publish_heat(row: &[i8], dim: usize, g: usize) {
    let cells = g * g;
    if cells == 0 || dim == 0 || row.len() < dim * cells {
        return;
    }
    let mut e = vec![0.0f32; cells];
    for d in 0..dim {
        let plane = &row[d * cells..(d + 1) * cells];
        for (c, &v) in plane.iter().enumerate() {
            e[c] += (v as f32) * (v as f32);
        }
    }
    for v in e.iter_mut() {
        *v = v.sqrt();
    }
    publish(e, g, Source::Bands);
}

/// Ease a freshly computed field into what is on screen.
///
/// The smoothing is the same whichever source produced it: the flicker it
/// exists to stop is a property of repainting a small field eight times a
/// second, not of what the numbers mean.
fn publish(e: Vec<f32>, g: usize, src: Source) {
    with(|s| {
        if !s.seeded || s.g != g || s.src != src || s.field.len() != e.len() {
            s.lo = e.iter().copied().fold(f32::MAX, f32::min);
            s.hi = e.iter().copied().fold(f32::MIN, f32::max);
            s.field = e;
            s.g = g;
            s.src = src;
            s.seeded = true;
            return;
        }
        for (x, v) in s.field.iter_mut().zip(e.iter()) {
            *x += (v - *x) * FIELD_EMA;
        }
        let lo = s.field.iter().copied().fold(f32::MAX, f32::min);
        let hi = s.field.iter().copied().fold(f32::MIN, f32::max);
        s.lo += (lo - s.lo) * RANGE_EMA;
        s.hi += (hi - s.hi) * RANGE_EMA;
    });
}

/// The heat field's own worker thread: the mask net's attention maps when the
/// bundle carries the graph, latent band energy when it does not.
///
/// ONE row per window is the whole cost control, and the thread is the other
/// half of it. The forward is ~40 M multiply-adds -- small next to a window,
/// but on the encode thread it would sit between two `Session::run` calls,
/// where small still means 100% of it is on the critical path. Off-thread it is
/// bounded by whatever the encode thread does NOT need: the mailbox holds
/// nothing, so a row offered while the previous one is still being drawn is
/// dropped rather than queued. Decoration never waits for the goblins and the
/// goblins never wait for it.
///
/// A mask forward that fails takes the graph out of the run rather than being
/// retried every window: the fallback is band energy and silence, never an
/// error. Nothing here can fail a draft.
pub struct Viewport {
    tx: Option<SyncSender<Vec<i8>>>,
    handle: Option<JoinHandle<()>>,
}

impl Viewport {
    pub fn spawn(mut mask: Option<Session>, dim: usize, grid: usize) -> Self {
        // capacity 0: a send lands only while the worker is parked in `recv`,
        // which is exactly "draw it if you are free, otherwise skip this window"
        let (tx, rx) = sync_channel::<Vec<i8>>(0);
        let handle = std::thread::spawn(move || {
            while let Ok(row) = rx.recv() {
                if let Some(sess) = mask.as_mut() {
                    if gate_of(sess, dim, grid, &row).is_some() {
                        continue;
                    }
                    mask = None;
                }
                publish_heat(&row, dim, grid);
            }
        });
        Self { tx: Some(tx), handle: Some(handle) }
    }

    pub fn publish(&self, row: &[i8]) {
        if let Some(tx) = &self.tx {
            match tx.try_send(row.to_vec()) {
                Ok(()) | Err(TrySendError::Full(_)) => {}
                // the worker is gone (it cannot return, but a panic would);
                // the draft is not its business either way
                Err(TrySendError::Disconnected(_)) => {}
            }
        }
    }
}

impl Drop for Viewport {
    /// The viewport stops when the stage does. Dropping the sender ends the
    /// worker's `recv`, and the join is what keeps a late frame from being
    /// drawn over the next stage's display.
    fn drop(&mut self) {
        self.tx.take();
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

/// The mask net on one row -> the field, or `None` if this graph cannot give
/// it a picture.
fn gate_of(sess: &mut Session, dim: usize, grid: usize, row: &[i8]) -> Option<()> {
    let shape = vec![1i64, dim as i64, grid as i64, grid as i64];
    let t = ort::value::TensorRef::from_array_view((shape, row)).ok()?;
    let out = sess.run(ort::inputs!["x_i8" => t]).ok()?;
    let (s, gate) = out["gate"].try_extract_tensor::<f32>().ok()?;
    // (1, heads, g, g) -- the head count is the graph's, never assumed
    let heads = (s.len() == 4).then(|| s[1] as usize)?;
    publish_gate(gate, heads, grid).then_some(())
}

/// One decoded frame -> the film panel, box-averaged onto the `g` x `g` grid.
///
/// `rgb` is the encoder's own input frame, `w` x `h` rgb24 -- so the averaging
/// is over exactly the pixels one latent cell was computed from, which is what
/// makes the two panels line up cell for cell. A frame that is not the size it
/// says it is is dropped: the viewport never takes a draft down.
pub fn publish_frame(rgb: &[u8], w: usize, h: usize, g: usize) {
    if g == 0 || w < g || h < g || rgb.len() < w * h * 3 {
        return;
    }
    let mut cells = vec![[0u8; 3]; g * g];
    for r in 0..g {
        let (y0, y1) = (r * h / g, ((r + 1) * h / g).max(r * h / g + 1));
        for c in 0..g {
            let (x0, x1) = (c * w / g, ((c + 1) * w / g).max(c * w / g + 1));
            let mut sum = [0u32; 3];
            for y in y0..y1 {
                let row = &rgb[(y * w) * 3..];
                for x in x0..x1 {
                    for (s, v) in sum.iter_mut().zip(&row[x * 3..x * 3 + 3]) {
                        *s += *v as u32;
                    }
                }
            }
            let n = ((y1 - y0) * (x1 - x0)) as u32;
            for (o, s) in cells[r * g + c].iter_mut().zip(sum) {
                *o = (s / n) as u8;
            }
        }
    }
    with(|s| {
        s.frame = cells;
        s.fg = g;
    });
}


/// An RGB triple onto the xterm-256 palette.
///
/// Near-grey goes to the 24-step grey ramp rather than the 6x6x6 cube, which
/// has four greys in it: a dim interior or a black-and-white scene is most of
/// what a frame of film is, and the cube renders it as four flat plateaus.
fn xterm256(px: [u8; 3]) -> u8 {
    let (mx, mn) = (px.iter().copied().max().unwrap(), px.iter().copied().min().unwrap());
    if mx - mn < 12 {
        let l = (px.iter().map(|&v| v as u16).sum::<u16>() / 3) as u8;
        return match l {
            0..=7 => 16,
            248..=255 => 231,
            // the ramp runs 8, 18, ... 238 in tens
            _ => 232 + ((l as u16 - 8) / 10).min(23) as u8,
        };
    }
    // the cube's levels are 0, 95, 135, 175, 215, 255
    const LEVELS: [u8; 6] = [0, 95, 135, 175, 215, 255];
    let q = |v: u8| -> u16 {
        LEVELS
            .iter()
            .enumerate()
            .min_by_key(|(_, &l)| (v as i16 - l as i16).abs())
            .map(|(i, _)| i as u16)
            .unwrap()
    };
    (16 + 36 * q(px[0]) + 6 * q(px[1]) + q(px[2])) as u8
}

/// An xterm-256 foreground/background pair, emitted only where it CHANGES.
///
/// A panel row is two dozen styled cells and a few hundred bytes of escape
/// codes; most neighbours share a colour, and run-length painting cuts that to
/// a fraction. The terminal is being repainted eight times a second with a
/// header animating over it -- the bytes are worth not sending.
#[derive(Default)]
struct Pen {
    fg: Option<u8>,
    bg: Option<u8>,
    out: String,
}

impl Pen {
    fn put(&mut self, fg: u8, bg: Option<u8>, ch: char) {
        if self.fg != Some(fg) {
            self.out.push_str(&format!("\x1b[38;5;{fg}m"));
            self.fg = Some(fg);
        }
        if self.bg != bg {
            match bg {
                Some(b) => self.out.push_str(&format!("\x1b[48;5;{b}m")),
                None => self.out.push_str("\x1b[49m"),
            }
            self.bg = bg;
        }
        self.out.push(ch);
    }

    /// Append an already-closed segment (another `Pen`'s output). Its trailing
    /// reset has cleared the terminal's colours, so the running state has to be
    /// forgotten or the next `put` would skip an escape it still needs.
    fn raw(&mut self, s: &str) {
        self.out.push_str(s);
        self.fg = None;
        self.bg = None;
    }

    /// Close the row. Colour never leaks past a line the app drew.
    fn finish(mut self) -> String {
        self.out.push_str("\x1b[0m");
        self.out
    }
}

/// Map 0..1 onto the palette's thermal ramp.
fn shade(v: f32) -> u8 {
    let r = theme().ramp;
    let i = (v.clamp(0.0, 1.0) * (r.len() - 1) as f32).round() as usize;
    r[i.min(r.len() - 1)]
}

/// One panel's caption: its name, with a tag right-aligned under the field
/// where the width allows it and dropped where it does not. No indent -- a
/// caption row can carry two of these side by side.
///
/// Laid out in CELLS. The caption sits directly under a panel exactly `width`
/// cells wide, and the names are translated: a CJK glyph fills two cells and a
/// byte-per-cell walk would hand the terminal half a character.
fn caption(width: usize, name: &str, tag: &str) -> String {
    let t = theme();
    let (nw, tw) = (cols(name), cols(tag));
    let room = !tag.is_empty() && width >= nw + 2 + tw;
    let tag_at = width.saturating_sub(tw);
    let mut cap = Pen::default();
    let (mut name_ch, mut tag_ch) = (name.chars(), tag.chars());
    let mut col = 0;
    while col < width {
        let ch = if col < nw {
            name_ch.next().unwrap_or(' ')
        } else if room && col >= tag_at {
            tag_ch.next().unwrap_or(' ')
        } else {
            ' '
        };
        let w = cols(&ch.to_string()).max(1);
        // a wide glyph with one cell left would spill into the panel beside it
        if col + w > width {
            cap.put(t.bar_dim, None, ' ');
            col += 1;
            continue;
        }
        cap.put(if ch == ' ' { t.bar_dim } else { t.muted }, None, ch);
        col += w;
    }
    cap.finish()
}

/// The CELLS a string occupies, which is what the terminal advances by.
fn cols(s: &str) -> usize {
    console::measure_text_width(s)
}

/// A run of blank cells in no colour, between two panels on one row.
fn gap(pen: &mut Pen) {
    for _ in 0..GAP {
        pen.put(theme().bar_dim, None, ' ');
    }
}

/// The viewport's lines, top to bottom, each one complete and self-contained.
///
/// Empty when nothing has been published, when the terminal is too short, or
/// when it is too narrow -- in every one of those cases the caller draws its
/// single live line exactly as it did before there was a viewport at all. The
/// film is dropped before the heat field is: the panel that says what the
/// goblins are looking at is the one worth the rows.
pub fn render(term_w: usize, log_h: usize) -> Vec<String> {
    if !fits(log_h) {
        return Vec::new();
    }
    let guard = VIEW.lock().unwrap();
    let Some(v) = guard.as_ref() else { return Vec::new() };
    if !v.seeded || v.g < 2 || v.field.len() != v.g * v.g {
        return Vec::new();
    }
    let room = term_w.saturating_sub(INDENT + 1);
    if room < v.g {
        return Vec::new();
    }
    // the film sits beside the field, at the same size, or not at all
    let film = v.fg >= 2 && v.frame.len() == v.fg * v.fg && room >= v.g + GAP + v.fg;

    let span = (v.hi - v.lo).max(1e-6);
    let at = |r: usize, c: usize| ((v.field[r * v.g + c] - v.lo) / span).clamp(0.0, 1.0);
    // The grid is sampled onto `2 * PANEL_H` rows rather than indexed directly,
    // so a bundle whose grid is not exactly 2 * PANEL_H tall fills the panel
    // instead of showing its top half.
    let src = |g: usize, k: usize| (k * g / (2 * PANEL_H)).min(g - 1);

    let mut lines = Vec::with_capacity(VIEW_H);
    for r in 0..PANEL_H {
        let mut pen = Pen::default();
        pen.out.push_str(&" ".repeat(INDENT));
        for c in 0..v.g {
            // the upper-half block paints the top grid row over the bottom one
            let (t, b) = (src(v.g, r * 2), src(v.g, r * 2 + 1));
            pen.put(shade(at(t, c)), Some(shade(at(b, c))), '\u{2580}');
        }
        if film {
            gap(&mut pen);
            let (t, b) = (src(v.fg, r * 2), src(v.fg, r * 2 + 1));
            for c in 0..v.fg {
                let fg = xterm256(v.frame[t * v.fg + c]);
                let bg = xterm256(v.frame[b * v.fg + c]);
                pen.put(fg, Some(bg), '\u{2580}');
            }
        }
        lines.push(pen.finish());
    }
    let mut cap = Pen::default();
    cap.out.push_str(&" ".repeat(INDENT));
    cap.raw(&caption(v.g, crate::t!("viewport.vision"), v.src.label()));
    if film {
        gap(&mut cap);
        cap.raw(&caption(v.fg, crate::t!("viewport.film"), "1:1"));
    }
    lines.push(cap.finish());
    lines
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The viewport is ONE screen and so it is one global, which means these
    /// tests share it: cargo runs them on parallel threads, and without this
    /// they publish over each other's fixtures. Poisoning is ignored on
    /// purpose -- a panicking test has already reported the real failure, and
    /// every test here starts by publishing its own state anyway.
    static ONE_SCREEN: Mutex<()> = Mutex::new(());

    fn alone() -> std::sync::MutexGuard<'static, ()> {
        let g = ONE_SCREEN.lock().unwrap_or_else(|e| e.into_inner());
        clear();
        g
    }

    /// A row whose energy sits in one cell.
    fn hot_row(dim: usize, g: usize, at: usize) -> Vec<i8> {
        let mut row = vec![0i8; dim * g * g];
        for d in 0..dim {
            row[d * g * g + at] = 100;
        }
        row
    }

    /// A gate where head `h` attends cell `at` alone, every other head flat.
    fn gate(g: usize, heads: usize, h: usize, at: usize) -> Vec<f32> {
        let mut m = vec![0.5f32; heads * g * g];
        let plane = &mut m[h * g * g..(h + 1) * g * g];
        plane.fill(0.0);
        plane[at] = 1.0;
        m
    }

    /// The point of normalizing before averaging: a head that gates everything
    /// equally has no opinion about WHERE, however loud its raw sigmoid is, and
    /// must not dilute the one head that does. The peaked head's cell has to
    /// come out above the flat background.
    #[test]
    fn a_flat_head_does_not_wash_out_a_peaked_one() {
        let _screen = alone();
        let m = gate(4, 8, 3, 5);
        assert!(publish_gate(&m, 8, 4));
        let f = with(|s| s.field.clone());
        assert_eq!(f.len(), 16);
        assert_eq!(f.iter().cloned().fold(f32::MIN, f32::max), f[5], "the peak is the peak");

        // seven flat heads each spread 1/16, the eighth puts all of its weight
        // on one cell, so the field is still one distribution over the grid
        assert!((f.iter().sum::<f32>() - 1.0).abs() < 1e-5, "sums to one");

        // and the contrast is the reason for normalizing first: averaging the
        // RAW sigmoids instead lets seven opinionless heads at 0.5 drown the
        // one head that has an opinion
        let raw = |c: usize| (0..8).map(|h| m[h * 16 + c]).sum::<f32>() / 8.0;
        assert!(
            f[5] / f[0] > 3.0 && raw(5) / raw(0) < 1.5,
            "normalized {:.2}x vs raw {:.2}x",
            f[5] / f[0],
            raw(5) / raw(0)
        );
    }

    /// Any head count draws: the field is an average over whatever the
    /// checkpoint carries, not a fixed number of tiles.
    #[test]
    fn any_head_count_draws() {
        for heads in [1usize, 4, 8, 12] {
            let _screen = alone();
            assert!(publish_gate(&gate(4, heads, 0, 7), heads, 4), "{heads} heads");
            let f = with(|s| s.field.clone());
            assert!((f.iter().sum::<f32>() - 1.0).abs() < 1e-5, "{heads} heads sum to one");
        }
    }

    /// A head that gates nothing anywhere is a zero, not an infinity -- the
    /// division by its own sum is where a dead head would poison the panel.
    #[test]
    fn a_dead_head_contributes_nothing() {
        let _screen = alone();
        let mut m = vec![0.0f32; 2 * 16];
        m[16 + 5] = 1.0; // head 0 all zero, head 1 peaked
        assert!(publish_gate(&m, 2, 4));
        let f = with(|s| s.field.clone());
        assert!(f.iter().all(|v| v.is_finite()), "no infinities from the dead head");
        assert_eq!(f[5], 0.5, "the live head's weight, halved across two heads");
    }

    /// A malformed buffer must be refused rather than drawn or panicked on:
    /// the caller reads the `false` and falls back to band energy.
    #[test]
    fn a_short_gate_is_refused() {
        let _screen = alone();
        assert!(!publish_gate(&[1.0, 2.0, 3.0], 8, 24));
        assert!(!publish_gate(&[], 0, 24), "no heads at all");
        assert!(with(|s| !s.seeded && s.field.is_empty()));
    }

    /// A malformed row must be dropped, not panic the encode stage: the
    /// viewport is decoration, and decoration never takes a draft down.
    #[test]
    fn a_short_row_is_ignored() {
        let _screen = alone();
        publish_heat(&[1, 2, 3], 128, 24);
        assert!(with(|s| !s.seeded && s.field.is_empty()));
    }

    /// The first row seeds outright -- easing into it from zero would open
    /// every draft on a blank panel.
    #[test]
    fn the_first_row_seeds_the_field_outright() {
        let _screen = alone();
        publish_heat(&hot_row(64, 4, 5), 64, 4);
        let (f, lo, hi) = with(|s| (s.field.clone(), s.lo, s.hi));
        assert_eq!(lo, 0.0);
        assert!(hi > 0.0 && f[5] == hi);
    }

    /// The whole point of the smoothing: a field that changes must MOVE toward
    /// the new one, not snap to it, or a still scene strobes as each window
    /// rescales the picture.
    #[test]
    fn later_rows_ease_the_field_and_the_range() {
        let _screen = alone();
        publish_heat(&hot_row(64, 4, 5), 64, 4);
        let first = with(|s| s.field[5]);
        publish_heat(&hot_row(64, 4, 9), 64, 4);
        let (f5, f9, hi) = with(|s| (s.field[5], s.field[9], s.hi));
        assert!(f5 < first && f5 > 0.0, "cell 5 eased, not snapped: {f5} vs {first}");
        assert!(f9 > 0.0 && f9 < first, "cell 9 rose part way: {f9}");
        assert!(hi > f5.max(f9), "the range still remembers the old peak");
    }

    /// Attention and band energy are different quantities on different scales,
    /// so a run that switches sources reseeds instead of easing one into the
    /// other -- otherwise the seconds after the switch draw a blend of two
    /// things that were never measured together.
    #[test]
    fn changing_source_reseeds() {
        let _screen = alone();
        publish_heat(&hot_row(64, 4, 5), 64, 4);
        assert!(publish_gate(&gate(4, 1, 0, 5), 1, 4));
        let (f, hi) = with(|s| (s.field[5], s.hi));
        assert_eq!(f, 1.0, "the gate seeded outright, not eased from band energy");
        assert_eq!(hi, 1.0);
    }

    /// Every drawn line is exactly the panel's width in VISIBLE cells -- the
    /// escape codes must not count, or the caller's repaint walks the wrong
    /// number of rows and leaves the screen in pieces.
    #[test]
    fn lines_measure_their_visible_width() {
        let _screen = alone();
        publish_heat(&vec![7i8; 128 * 24 * 24], 128, 24);
        let lines = render(200, 60);
        assert_eq!(lines.len(), VIEW_H);
        for (i, l) in lines.iter().enumerate() {
            assert_eq!(console::measure_text_width(l), INDENT + 24, "line {i}");
        }
    }

    /// The caption names the source, and drops the tag rather than overflowing
    /// the panel on a grid too narrow to carry it.
    #[test]
    fn the_caption_names_the_source_where_it_fits() {
        let name = "goblin vision";
        let wide = console::strip_ansi_codes(&caption(24, name, "attention")).to_string();
        assert!(wide.trim_end().ends_with("attention"), "{wide:?}");
        assert!(wide.starts_with(name), "{wide:?}");
        assert_eq!(wide.chars().count(), 24);
        let narrow = console::strip_ansi_codes(&caption(16, name, "bands")).to_string();
        assert!(!narrow.contains("bands"), "no room for the tag: {narrow:?}");
        assert_eq!(narrow.chars().count(), 16);
    }

    /// The caption sits under a panel exactly as wide as the field, and two of
    /// them share a row. A translated name is fewer CHARACTERS than cells, so
    /// the row is measured in cells or the second panel starts in the wrong
    /// column -- and a wide glyph that would hang off the end is dropped
    /// rather than half-drawn.
    #[test]
    fn a_translated_caption_fills_its_panel_exactly() {
        let _lang = crate::lang::speaking("zh-CN");
        for width in 8..40usize {
            let row = console::strip_ansi_codes(&caption(
                width,
                crate::t!("viewport.vision"),
                crate::t!("viewport.src.attention"),
            ))
            .to_string();
            assert_eq!(cols(&row), width, "{width}: {row:?}");
        }
        // and the tag still lands on the right edge where there is room for it
        let wide = console::strip_ansi_codes(&caption(
            32,
            crate::t!("viewport.vision"),
            crate::t!("viewport.src.bands"),
        ))
        .to_string();
        assert!(wide.trim_end().ends_with(crate::t!("viewport.src.bands")), "{wide:?}");
    }

    /// A 24x24 frame of film sits beside a 24x24 field, one cell to one cell --
    /// which is the only reason the two panels are worth putting side by side,
    /// and it means the block is exactly twice a panel plus the gap.
    #[test]
    fn the_film_draws_beside_the_field_at_the_same_size() {
        let _lang = crate::lang::speaking("en-US");
        let _screen = alone();
        publish_heat(&vec![7i8; 128 * 24 * 24], 128, 24);
        publish_frame(&vec![90u8; 384 * 384 * 3], 384, 384, 24);
        let both = INDENT + 24 + GAP + 24;
        let lines = render(both + 1, 60);
        assert_eq!(lines.len(), VIEW_H);
        for (i, l) in lines.iter().enumerate() {
            assert_eq!(console::measure_text_width(l), both, "line {i}");
        }
        let film = crate::t!("viewport.film");
        assert!(
            console::strip_ansi_codes(lines.last().unwrap()).contains(film),
            "the second panel is captioned too"
        );
        // one column short of the pair: the field keeps the row alone
        let lines = render(both, 60);
        assert_eq!(console::measure_text_width(&lines[0]), INDENT + 24);
        assert!(!console::strip_ansi_codes(lines.last().unwrap()).contains(film));
    }

    /// A frame is averaged onto the grid, so a picture split down the middle
    /// comes out split down the middle -- and one that is the wrong size is
    /// dropped rather than drawn from whatever bytes were there.
    #[test]
    fn a_frame_is_averaged_onto_the_grid() {
        let _screen = alone();
        let (w, g) = (48usize, 4usize);
        let mut rgb = vec![0u8; w * w * 3];
        for y in 0..w {
            for x in 0..w {
                // left half red, right half blue
                let px = if x < w / 2 { [200u8, 0, 0] } else { [0, 0, 200] };
                rgb[(y * w + x) * 3..(y * w + x) * 3 + 3].copy_from_slice(&px);
            }
        }
        publish_frame(&rgb, w, w, g);
        let f = with(|s| s.frame.clone());
        assert_eq!(f.len(), g * g);
        assert_eq!(f[0], [200, 0, 0], "left cell is the red half");
        assert_eq!(f[g - 1], [0, 0, 200], "right cell is the blue half");
        publish_frame(&rgb[..10], w, w, g);
        assert_eq!(with(|s| s.frame.len()), g * g, "a short frame changed nothing");
    }

    /// Grey is the grey ramp, not the colour cube's four greys -- a dim
    /// interior is most of what a frame of film is.
    #[test]
    fn near_grey_takes_the_grey_ramp() {
        assert_eq!(xterm256([0, 0, 0]), 16);
        assert_eq!(xterm256([255, 255, 255]), 231);
        for l in [40u8, 128, 200] {
            let c = xterm256([l, l, l]);
            assert!((232..=255).contains(&c), "{l} -> {c}");
        }
        // and a coloured pixel is the 6x6x6 cube, brightest red at its corner
        assert_eq!(xterm256([255, 0, 0]), 196);
        assert_eq!(xterm256([0, 0, 255]), 21);
    }

    /// A terminal without the rows (or the columns) keeps its log intact.
    #[test]
    fn a_small_terminal_draws_nothing() {
        let _screen = alone();
        publish_heat(&vec![7i8; 128 * 24 * 24], 128, 24);
        assert!(fits(MIN_LOG_H) && !fits(MIN_LOG_H - 1));
        assert!(render(200, MIN_LOG_H - 1).is_empty(), "too short");
        assert!(render(24 + INDENT, 60).is_empty(), "too narrow by one");
        assert!(!render(24 + INDENT + 1, 60).is_empty(), "and wide enough by one");
    }

    /// Nothing published yet is not a blank panel, it is no panel: the first
    /// stages of a draft run before the encoder has produced a row.
    #[test]
    fn nothing_published_draws_nothing() {
        let _screen = alone();
        assert!(render(200, 60).is_empty());
    }

    /// Every palette's ramp has to run dark to light, or the field reads as
    /// noise: `shade` maps 0..1 onto it and 0 must be the dark end.
    #[test]
    fn every_ramp_runs_dark_to_light() {
        let _screen = alone();
        for p in crate::theme::Palette::ALL {
            crate::theme::set(p);
            let r = theme().ramp;
            assert_eq!(shade(0.0), r[0], "{}", p.label());
            assert_eq!(shade(1.0), r[r.len() - 1], "{}", p.label());
            assert_eq!(shade(-5.0), r[0], "{} clamps below", p.label());
            assert_eq!(shade(5.0), r[r.len() - 1], "{} clamps above", p.label());
        }
        crate::theme::set(crate::theme::Palette::Phosphor);
    }
}


