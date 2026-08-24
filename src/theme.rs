//! The one palette, shared by every surface the goblins draw on: the intro
//! demo, the picker, the processing console, and the browser review page.
//!
//! Terminal colours are xterm-256 INDICES rather than truecolour, for two
//! reasons: `console` (the processing console) has no truecolour, and an index
//! renders identically in both stacks, so the picker and the console cannot
//! drift apart. The intro and the web page need real channels, so each palette
//! also carries an RGB brand colour and the hue band its plasma sweeps.
//!
//! The active palette is a process-wide atomic: the picker cycles it live (T)
//! and every surface re-reads it on the next frame, with no plumbing through
//! call stacks that exist for other reasons.

use std::sync::atomic::{AtomicU8, Ordering};

#[derive(Clone, Copy, PartialEq, Eq, Debug, clap::ValueEnum, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Palette {
    /// Green phosphor -- the goblin house style.
    Phosphor,
    /// Amber P3 monitor.
    Amber,
    /// Black / cyan / magenta, the way a CGA card did it.
    Cga,
    /// White phosphor, no hue at all.
    Mono,
}

impl Palette {
    pub const ALL: [Palette; 4] = [Palette::Phosphor, Palette::Amber, Palette::Cga, Palette::Mono];

    pub fn label(self) -> &'static str {
        match self {
            Palette::Phosphor => crate::t!("theme.phosphor"),
            Palette::Amber => crate::t!("theme.amber"),
            Palette::Cga => crate::t!("theme.cga"),
            Palette::Mono => crate::t!("theme.mono"),
        }
    }

    fn idx(self) -> u8 {
        match self {
            Palette::Phosphor => 0,
            Palette::Amber => 1,
            Palette::Cga => 2,
            Palette::Mono => 3,
        }
    }

    fn from_idx(i: u8) -> Palette {
        Palette::ALL[(i as usize) % Palette::ALL.len()]
    }

    pub fn next(self) -> Palette {
        Palette::from_idx(self.idx() + 1)
    }
}

/// The active palette. `AtomicU8` (not a lock) because every draw reads it and
/// only a keypress writes it -- a torn read is impossible for a single byte.
static ACTIVE: AtomicU8 = AtomicU8::new(0);

pub fn active() -> Palette {
    Palette::from_idx(ACTIVE.load(Ordering::Relaxed))
}

pub fn set(p: Palette) {
    ACTIVE.store(p.idx(), Ordering::Relaxed);
}

/// Advance to the next palette and return it (the picker's T key).
pub fn cycle() -> Palette {
    let p = active().next();
    set(p);
    p
}

/// Every colour a surface can ask for. Terminal fields are xterm-256 indices;
/// `brand` and the hue band drive the intro's own rendering, which paints in
/// real channels.
#[derive(Clone, Copy)]
pub struct Theme {
    /// The goblin mascot and wordmark.
    pub logo: u8,
    /// Selected / emphasised values -- the brightest thing on screen.
    pub accent: u8,
    /// Ordinary body text.
    pub text: u8,
    /// Secondary text: paths, counts, parentheticals.
    pub muted: u8,
    /// The key-hint footer.
    pub help: u8,
    /// The inverse-video background behind hotkey chips and the list cursor.
    pub chrome_bg: u8,
    pub ok: u8,
    pub warn: u8,
    pub bad: u8,
    /// The filled and unfilled halves of a progress bar.
    pub bar: u8,
    pub bar_dim: u8,
    /// Twelve steps from black to white, THROUGH the palette's own hues -- the
    /// thermal ramp the viewport paints its heat field with (`viz.rs`). A
    /// picture needs more than the handful of roles above, and it needs them
    /// ORDERED: the palette decides what colours the goblins see in, and a ramp
    /// is the only way a field can answer to that.
    ///
    /// It runs through hues rather than up one hue's brightness because a
    /// single-hue field reads flat -- the eye separates green from yellow from
    /// white far better than it separates two greens, so the same data carries
    /// more of itself. Every ramp still runs DARK TO LIGHT (a test holds it),
    /// so the field is readable with the colour taken away.
    pub ramp: [u8; 12],
    /// The colour the intro's wordmark settles on, in linear 0..1 channels.
    pub brand: (f32, f32, f32),
    /// The hue band (degrees) the intro's plasma and sweep are confined to, and
    /// how saturated they are. A zero span with zero saturation is greyscale.
    pub hue_base: f32,
    pub hue_span: f32,
    pub sat: f32,
}

/// The palette table. One row per palette, no computation -- a palette is a set
/// of decisions, and deriving one from another only makes both harder to tune.
pub fn theme() -> Theme {
    match active() {
        Palette::Phosphor => Theme {
            logo: 118,
            accent: 231,
            text: 114,
            muted: 65,
            help: 252,
            chrome_bg: 28,
            ok: 82,
            warn: 220,
            bad: 196,
            bar: 82,
            bar_dim: 238,
            // black -> green -> yellow-green -> yellow -> white: a phosphor
            // scope run hot
            ramp: [16, 22, 28, 34, 40, 46, 82, 118, 154, 190, 226, 231],
            brand: (0.59, 0.87, 0.35),
            hue_base: 0.0,
            hue_span: 360.0,
            sat: 0.85,
        },
        Palette::Amber => Theme {
            logo: 214,
            accent: 231,
            text: 214,
            muted: 130,
            help: 223,
            chrome_bg: 94,
            ok: 214,
            warn: 227,
            bad: 203,
            bar: 214,
            bar_dim: 236,
            // black -> deep red -> orange -> amber -> yellow -> white: the
            // ramp a hot filament actually walks
            ramp: [16, 52, 88, 124, 160, 196, 202, 208, 214, 220, 226, 231],
            brand: (1.0, 0.69, 0.0),
            hue_base: 20.0,
            hue_span: 45.0,
            sat: 0.9,
        },
        Palette::Cga => Theme {
            logo: 51,
            accent: 231,
            text: 51,
            muted: 244,
            help: 231,
            chrome_bg: 54,
            ok: 51,
            warn: 213,
            bad: 201,
            bar: 51,
            bar_dim: 236,
            // black -> blue -> purple -> magenta -> pink -> pale cyan -> white,
            // which is the card's two signature hues with the road between them
            ramp: [16, 17, 19, 21, 57, 93, 129, 165, 201, 207, 123, 231],
            brand: (0.33, 1.0, 1.0),
            hue_base: 180.0,
            hue_span: 120.0,
            sat: 0.95,
        },
        Palette::Mono => Theme {
            logo: 252,
            accent: 231,
            text: 250,
            muted: 244,
            help: 252,
            chrome_bg: 240,
            ok: 252,
            warn: 231,
            bad: 210,
            bar: 252,
            bar_dim: 236,
            // no hue at all, by the palette's definition: the grey ramp is what
            // the other three reduce to with the colour taken away
            ramp: [16, 233, 235, 237, 239, 241, 243, 245, 248, 250, 252, 231],
            brand: (0.88, 0.88, 0.90),
            hue_base: 0.0,
            hue_span: 0.0,
            sat: 0.0,
        },
    }
}

/// The goblin mascot, four lines, with NO indent baked in -- every surface
/// applies its own and pads to `MASCOT_W`.
///
/// It lives here, once, because it was drawn twice: the picker carried per-line
/// indents inside its strings while the startup report applied one uniform
/// indent to unindented copies, which put the ears a column left of the face and
/// the mouth two columns right of it. Eyeballed art drifts the first time
/// somebody edits one copy. `mascot_is_centred` is the test that holds it.
/// The goblin, full length: the same fellow who drums and DJs in the corner of
/// the header (`tui.rs`), stood still. Same ears, same face, same arms -- where
/// the band members have a kit or a deck on their bottom row, this one has legs.
///
/// Drawing it as the band's own goblin is the point. Two surfaces showing two
/// different creatures is how a mascot stops being a mascot. This is the pose
/// he comes back to: the picker header animates him from a table of poses built
/// on these same four rows (`tui.rs`), so every blink, stretch and dance step is
/// this goblin moving rather than a second drawing of one.
///
/// `mascot_is_centred` requires every row to sit on ONE axis. With the axis at
/// 6 that makes every row an odd number of visible columns (a row of width `L`
/// starts at `(7 - L) / 2`); an even-width row cannot land on it.
pub const MASCOT: [&str; 4] = [
    " /\\,/\\",
    "\\(o.o)/",
    " |___|",
    "  / \\",
];

/// The width every mascot line pads to, so whatever a surface prints beside the
/// art starts in ONE column rather than stepping in and out with the face.
pub const MASCOT_W: usize = 8;

/// One row of mascot-sized art, indented by `indent` and padded out to the art
/// column. Takes the row rather than an index because the header draws poses
/// (`tui.rs`) and not only the resting drawing -- and every one of them has to
/// land in the same block, or the wordmark beside it steps in and out.
pub fn art_line(row: &str, indent: usize) -> String {
    format!("{:indent$}{:<MASCOT_W$}", "", row, indent = indent)
}

/// One line of the resting mascot, indented and padded.
pub fn mascot_line(i: usize, indent: usize) -> String {
    art_line(MASCOT[i], indent)
}

/// An xterm-256 index as a `console` colour (the processing console).
pub fn con(idx: u8) -> console::Color {
    console::Color::Color256(idx)
}

/// An xterm-256 index as a `ratatui` colour (the intro, picker and review
/// screens). Same index, same cell -- the two stacks cannot drift.
pub fn rat(idx: u8) -> ratatui::style::Color {
    ratatui::style::Color::Indexed(idx)
}

/// The review page's custom properties for the active palette. The page ships
/// the phosphor values in its stylesheet and overrides them from here on load,
/// so a page served by an older build (or with the fetch failed) still renders.
pub fn css_vars() -> serde_json::Value {
    let (bg, panel, panel2, edge, hi, hi_bright, hi_dim, ink, ink_soft, fang, danger, danger_soft, cool) =
        match active() {
            Palette::Phosphor => (
                "#0e130c", "#151d14", "#1b241a", "#4c6144", "#8bc34a", "#b6e07a", "#bcda90",
                "#f4f9f0", "#d3e2c7", "#f1f8e9", "#ef5350", "#ef9a9a", "150,180,200",
            ),
            Palette::Amber => (
                "#140d02", "#1d1405", "#241a08", "#6b4a12", "#ffb000", "#ffd166", "#e0a33c",
                "#fff4e0", "#e8d5b0", "#fff8ec", "#ff5f4f", "#ffa08f", "140,152,172",
            ),
            Palette::Cga => (
                "#000000", "#0a0a16", "#101022", "#3f3f7f", "#55ffff", "#aaffff", "#00aaaa",
                "#ffffff", "#aaaaaa", "#ffffff", "#ff55ff", "#ffaaff", "170,170,170",
            ),
            Palette::Mono => (
                "#08080a", "#101014", "#16161c", "#4a4a52", "#d8d8dc", "#ffffff", "#a8a8b0",
                "#f2f2f4", "#c0c0c6", "#ffffff", "#d08b84", "#e3b3ae", "138,138,146",
            ),
        };
    serde_json::json!({
        "name": active().label(),
        "bg": bg, "panel": panel, "panel-2": panel2, "edge": edge,
        "hi": hi, "hi-bright": hi_bright, "hi-dim": hi_dim,
        "ink": ink, "ink-soft": ink_soft, "fang": fang,
        "danger": danger, "danger-soft": danger_soft,
        // the cool slate (the confidence band) as raw channels: the canvas
        // builds rgba() strings from it at a dozen different alphas
        "cool": cool,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    // The mascot is drawn on three surfaces and read at a glance on all of
    // them, so its lines must share ONE vertical axis: ears over the face,
    // mouth centred under it. (`first + last` rather than a midpoint, to keep
    // the comparison in integers.) This is the check that caught the ears and
    // the mouth drifting apart when the art existed in two places.
    #[test]
    fn mascot_is_centred() {
        let axis: Vec<usize> = MASCOT
            .iter()
            .map(|l| {
                let first = l.len() - l.trim_start().len();
                let last = l.trim_end().len() - 1;
                first + last
            })
            .collect();
        assert!(
            axis.windows(2).all(|w| w[0] == w[1]),
            "every mascot line shares one axis, got {axis:?}"
        );
        // and none may overrun the column the art is padded into
        for l in MASCOT {
            assert!(l.len() <= MASCOT_W, "{l:?} fits in {MASCOT_W} columns");
        }
    }

    // Whatever a surface prints beside the art has to start in one column, or
    // the header text steps in and out as the face widens and narrows.
    #[test]
    fn mascot_lines_pad_to_one_column() {
        let widths: Vec<usize> = (0..MASCOT.len()).map(|i| mascot_line(i, 2).len()).collect();
        assert!(widths.iter().all(|w| *w == 2 + MASCOT_W), "got {widths:?}");
    }

    // Cycling must visit every palette and come home -- the picker's T key is
    // the only way most users will ever meet the non-default ones.
    #[test]
    fn cycle_is_a_full_ring() {
        let start = Palette::Phosphor;
        let mut p = start;
        let mut seen = Vec::new();
        for _ in 0..Palette::ALL.len() {
            seen.push(p);
            p = p.next();
        }
        assert_eq!(p, start, "cycling {} times returns to the start", Palette::ALL.len());
        for q in Palette::ALL {
            assert!(seen.contains(&q), "{} is reachable by cycling", q.label());
        }
    }

    // Every palette must define every CSS variable the page overrides, or a
    // theme switch would leave the page half-styled.
    #[test]
    fn every_palette_defines_every_css_var() {
        let keys: Vec<String> = {
            set(Palette::Phosphor);
            css_vars().as_object().unwrap().keys().cloned().collect()
        };
        for p in Palette::ALL {
            set(p);
            let v = css_vars();
            let o = v.as_object().unwrap();
            for k in &keys {
                assert!(o.contains_key(k), "{} defines --{k}", p.label());
                assert!(!o[k].as_str().unwrap().is_empty(), "{} --{k} is non-empty", p.label());
            }
        }
        set(Palette::Phosphor);
    }
}
