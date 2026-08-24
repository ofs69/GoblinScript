//! The startup report, dressed as a BIOS power-on self-test.
//!
//! Every line is a REAL check -- ffmpeg's version, the host's memory and core
//! count, the resumable drafts sitting in the cache, the bundle's checkpoint and
//! the provider chain this build will try. The only invention is the framing (and
//! the goblin headcount, which is the thread count wearing a hat). That is the
//! rule the whole screen lives by: a fake BIOS is fine, a fake diagnostic is not,
//! because this is also the first thing a user pastes into a bug report.
//!
//! Timing follows the terminal. Interactive, it reveals line by line, counts
//! the memory up the way the real thing did, and keeps the goblin in the banner
//! moving the whole way down -- the pauses between devices are exactly the
//! stretch where a still drawing starts to look like a hung machine. Piped or
//! redirected, every line lands at once with no escapes in it, so a log file
//! stays a log file.

use crate::theme::{con, theme};
use console::{measure_text_width, style};
use std::cell::Cell;
use std::io::{IsTerminal, Write};
use std::time::{Duration, Instant};

/// The right-hand verdict on a POST line.
#[derive(Clone, Copy)]
pub enum Status {
    Ok,
    Ready,
    /// The check passed but the number is below what the goblins want. Never a
    /// refusal -- the app runs anyway, and this is the line the user quotes when
    /// they ask why it is slow.
    Warn,
}

impl Status {
    fn text(self) -> &'static str {
        match self {
            Status::Ok => "[ OK ]",
            Status::Ready => "[READY]",
            Status::Warn => "[WARN]",
        }
    }

    fn color(self, t: &crate::theme::Theme) -> u8 {
        match self {
            Status::Ok | Status::Ready => t.ok,
            Status::Warn => t.warn,
        }
    }
}

// Pacing. A power-on self-test that flashes past is just a wall of text with
// extra steps -- the point is watching the machine wake up, one device at a
// time. These are the whole tuning surface; the report runs ~5 s at them, which
// is also long enough to cover the model bundle loading on the other thread.
/// Between the banner's lines, and again before the first device line.
const BANNER_MS: u64 = 260;
/// After each completed POST line.
const LINE_MS: u64 = 220;
/// After an indented aside.
const NOTE_MS: u64 = 150;
/// One frame of the memory count-up, and how many frames it takes.
const MEM_FRAME_MS: u64 = 46;
const MEM_STEPS: u64 = 30;
/// The beat on the closing rule, before the screen hands over.
const END_MS: u64 = 420;

/// Where the label's dot leader ends. Wide enough for the longest label, so the
/// values line up in a column instead of stair-stepping.
const LEADER: usize = 34;
/// How wide the value field is before the status chip.
const VALUE: usize = 26;

/// Keep a value inside its column so the status chips stay in one line down the
/// right edge. An over-long value is cut with a trailing `~` rather than being
/// allowed to shove the chip out of alignment.
fn fit(v: &str) -> String {
    if v.chars().count() <= VALUE {
        return v.to_string();
    }
    let mut s: String = v.chars().take(VALUE - 1).collect();
    s.push('~');
    s
}

/// Rows the mascot occupies in the banner.
const MASCOT_H: usize = crate::theme::MASCOT.len();

/// The width the report is laid out for: the indent, the leader, the value
/// column and the status chip. A console narrower than this wraps its lines,
/// and a wrapped line is a row the banner's repaint did not count -- so on one
/// the goblin holds still rather than being redrawn over a device line.
const REPORT_W: usize = 2 + LEADER + 1 + VALUE + 1 + 6;

pub struct Post {
    /// Whether to pace the reveal. Off when piped, and off for a run that named
    /// videos on the command line -- that user is driving a tool, not booting a
    /// machine, and should not wait on an animation.
    animate: bool,
    /// Terminal rows emitted since the banner's LAST art row -- which is how
    /// far back up the screen the goblin is, and the one number the repaint
    /// needs. Counted rather than measured because there is nothing to ask: a
    /// console reports where the cursor is only by being read back, and this
    /// report is writing.
    rows: Cell<usize>,
    /// Is the banner still whole and on the screen? It stops being both when a
    /// long report scrolls it off the top, and the goblin is left where he was
    /// rather than being redrawn over whatever now occupies those rows.
    live: Cell<bool>,
    /// The beat last drawn, so a pause that ends inside one beat and a pause
    /// that begins inside the same one do not both redraw it.
    painted: Cell<usize>,
}

impl Post {
    /// Open the report and print the banner.
    pub fn begin(animate: bool) -> Post {
        let p = Post {
            animate: animate && std::io::stdout().is_terminal(),
            rows: Cell::new(0),
            live: Cell::new(false),
            painted: Cell::new(usize::MAX),
        };
        p.banner();
        p
    }

    /// Print a permanent line and account for the rows it takes -- including
    /// the ones it wrapped into on a console too narrow for it.
    fn emit(&self, line: &str) {
        println!("{line}");
        let width = console::Term::stdout().size().1.max(1) as usize;
        let wrapped = measure_text_width(line) / width;
        self.rows.set(self.rows.get() + 1 + wrapped);
    }

    /// Wait `ms`, and spend the wait animating the banner rather than sleeping
    /// through it. The pose comes off the shared beat clock, so the goblin the
    /// picker opens with is mid-way through whatever he started here.
    fn hold(&self, ms: u64) {
        if !self.animate {
            return;
        }
        std::io::stdout().flush().ok();
        if !self.live.get() {
            std::thread::sleep(Duration::from_millis(ms));
            return;
        }
        // A frame at half the beat, so no pose is ever skipped and no pause is
        // overshot by more than that.
        let step = Duration::from_millis((crate::mascot::BEAT_MS as u64 / 2).max(1));
        let deadline = Instant::now() + Duration::from_millis(ms);
        loop {
            let tick = crate::mascot::beat_tick();
            if tick != self.painted.get() {
                self.repaint(tick);
                self.painted.set(tick);
            }
            let left = deadline.saturating_duration_since(Instant::now());
            if left.is_zero() {
                return;
            }
            std::thread::sleep(step.min(left));
        }
    }

    /// Redraw the banner's four art rows in place, up where they were printed.
    ///
    /// Only the art columns are written: the copyright block beside them is not
    /// cleared and not touched, so a repaint cannot disturb a single character
    /// of the report that is the actual point of this screen.
    fn repaint(&self, tick: usize) {
        let rows = self.rows.get();
        // The cursor sits one row below the last line printed, so the top of
        // the block is this far up -- and if that is off the top of the screen
        // the block has scrolled away and there is nothing to repaint.
        let up = rows + MASCOT_H;
        let height = console::Term::stdout().size().0 as usize;
        if up >= height {
            self.live.set(false);
            return;
        }
        let mut out = std::io::stdout().lock();
        Self::art_frame(&mut out, rows, &crate::mascot::boot_pose(tick));
        let _ = out.flush();
    }

    /// One repaint of the art block, written into `out`: up over the `rows`
    /// emitted since the block's last line and the block's own height, a pose
    /// row per line, and back down to where it started.
    ///
    /// It is written out here, away from stdout, because the arithmetic is the
    /// whole risk: a frame that comes back one row shy walks the goblin down
    /// the report, over the diagnostics, one beat at a time.
    fn art_frame(out: &mut impl Write, rows: usize, pose: &[&str; MASCOT_H]) {
        let t = theme();
        let _ = write!(out, "\x1b[{}A", rows + MASCOT_H);
        for (i, row) in pose.iter().enumerate() {
            let _ = write!(
                out,
                "\r{}",
                style(crate::theme::art_line(row, 2)).fg(con(t.logo)).bold()
            );
            if i + 1 < pose.len() {
                let _ = write!(out, "\x1b[1B");
            }
        }
        let _ = write!(out, "\x1b[{}B\r", rows + 1);
    }

    /// The mascot and the copyright block, revealed a line at a time so the
    /// goblin assembles itself rather than appearing whole. He starts moving
    /// once he is all there -- an arm waving off a body that has not been drawn
    /// yet is not an entrance, it is a glitch.
    fn banner(&self) {
        let t = theme();
        // The art is padded to one column by `mascot_line`, so the text beside
        // it starts in the same place on every row instead of stepping in and
        // out as the face widens and narrows.
        let art = |i: usize| style(crate::theme::mascot_line(i, 2)).fg(con(t.logo)).bold();
        let head = |s: &str| style(s.to_string()).fg(con(t.accent)).bold();
        let sub = |s: &str| style(s.to_string()).fg(con(t.muted));
        println!();
        self.hold(BANNER_MS);
        println!(
            "{}    {}",
            art(0),
            head(&format!(
                "GOBLIN INDUSTRIES (R) SCRIPT BIOS v{}",
                env!("CARGO_PKG_VERSION")
            ))
        );
        self.hold(BANNER_MS);
        println!("{}    {}", art(1), sub("Copyright (C) 1987-2026, Goblin Industries Ltd."));
        self.hold(BANNER_MS);
        println!("{}    {}", art(2), sub("All goblins reserved."));
        self.hold(BANNER_MS);
        println!("{}", art(3));
        // Everything from here is counted against the block's bottom row, which
        // is what `repaint` climbs back over.
        self.rows.set(0);
        let width = console::Term::stdout().size().1 as usize;
        self.live.set(self.animate && width >= REPORT_W);
        self.emit(&format!("  {}", style("-".repeat(66)).fg(con(t.muted))));
        self.hold(BANNER_MS * 2);
    }

    /// One POST line: `label .......... value        [ OK ]`.
    pub fn line(&self, label: &str, value: &str, status: Status) {
        let t = theme();
        let dots = LEADER.saturating_sub(label.chars().count());
        self.emit(&format!(
            "  {} {} {} {}",
            style(label.to_string()).fg(con(t.text)),
            style(".".repeat(dots)).fg(con(t.muted)),
            style(format!("{:<VALUE$}", fit(value))).fg(con(t.accent)),
            style(status.text().to_string()).fg(con(status.color(&t))).bold(),
        ));
        self.hold(LINE_MS);
    }

    /// An unindented aside under a line -- used for the things a user needs to
    /// act on (a missing tool, resumable work) rather than merely observe.
    pub fn note(&self, text: &str) {
        let t = theme();
        self.emit(&format!("       {}", style(text.to_string()).fg(con(t.muted))));
        self.hold(NOTE_MS);
    }

    /// The memory line, counted up in place the way a real POST did. Piped, it
    /// prints once with the final figure and no carriage returns.
    pub fn memory(&self, total_kb: u64, status: Status) {
        let t = theme();
        let label = "Memory test";
        let dots = LEADER.saturating_sub(label.chars().count());
        let head = format!(
            "  {} {} ",
            style(label.to_string()).fg(con(t.text)),
            style(".".repeat(dots)).fg(con(t.muted)),
        );
        if self.animate {
            // counted up in steps, then the true figure -- the last frame is
            // always the real number, never a rounded step. The count-up is the
            // longest single beat of the report, so the goblin keeps moving
            // through it: `hold` leaves the cursor back where this found it,
            // and the next frame opens with a carriage return regardless.
            let steps = MEM_STEPS;
            for i in 1..=steps {
                let shown = total_kb * i / steps;
                let mut out = std::io::stdout().lock();
                let _ = write!(
                    out,
                    "\r{head}{}",
                    style(format!("{:<VALUE$}", format!("{shown} KB")))
                        .fg(con(t.accent))
                );
                let _ = out.flush();
                drop(out);
                self.hold(MEM_FRAME_MS);
            }
            let _ = write!(std::io::stdout(), "\r");
        }
        self.emit(&format!(
            "{head}{} {}",
            style(format!("{:<VALUE$}", format!("{total_kb} KB"))).fg(con(t.accent)),
            style(status.text().to_string()).fg(con(status.color(&t))).bold(),
        ));
        self.hold(LINE_MS);
    }

    /// Close the report.
    pub fn end(&self) {
        let t = theme();
        self.emit(&format!("  {}", style("-".repeat(66)).fg(con(t.muted))));
        self.hold(END_MS);
    }
}

/// The host half of the report: everything knowable without the model bundle,
/// so it can print while the bundle is still loading on another thread.
pub fn system(p: &Post, cache_root: &std::path::Path) {
    match crate::ffmpeg::version() {
        Some(v) => p.line("Media subsystem (ffmpeg)", &v, Status::Ok),
        // have_tools() has already run and passed, so a version we cannot parse
        // is a distributor quirk, not a missing tool
        None => p.line("Media subsystem (ffmpeg)", "present", Status::Ok),
    }

    // Links are an OPTIONAL trick, and the tool is the user's own. Present, the
    // picker offers a paste key; absent, the feature is not offered at all -- so
    // this line is the only place that says why the key is missing.
    if crate::dl::available() {
        // a version we cannot read is a packaging quirk, not a missing tool
        p.line("Link fetcher (yt-dlp)", crate::dl::version().unwrap_or("on PATH"), Status::Ok);
    } else {
        p.line("Link fetcher (yt-dlp)", "not installed", Status::Warn);
        p.note("no yt-dlp on PATH -- pasting links is off; local files are unaffected");
    }

    let crew = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    p.line("Goblin crew", &format!("{crew} on duty"), Status::Ok);

    // 8 GB is the floor the goblins are documented to want. Below it the app
    // still runs -- this is the line that explains why it struggled, not a gate.
    let mut sys = sysinfo::System::new();
    sys.refresh_memory();
    let total = sys.total_memory();
    let status = if total < 8 * 1024 * 1024 * 1024 { Status::Warn } else { Status::Ok };
    p.memory(total / 1024, status);
    if matches!(status, Status::Warn) {
        p.note("under 8 GB of system memory -- long videos may struggle");
    }

    // Resumable work is the one line here a user may need to act on: a cache
    // directory left behind means a draft was interrupted and will pick up
    // where it stopped rather than paying for the encoder again.
    let resumable = std::fs::read_dir(cache_root)
        .map(|rd| rd.flatten().filter(|e| e.path().is_dir()).count())
        .unwrap_or(0);
    if resumable > 0 {
        p.line("Scratch store", &format!("{resumable} draft(s) held"), Status::Ok);
        p.note("interrupted drafts found -- re-running those videos resumes them");
    } else {
        p.line("Scratch store", "clear", Status::Ok);
    }
}

/// The device half: what the model is and which backends this build will try,
/// in the order it will try them. Printed once the bundle has loaded.
pub fn devices(p: &Post, man: &crate::bundle::Manifest) {
    p.line(
        "Neural core",
        &format!("{} @ {}px", man.encoder, man.enc_res),
        Status::Ok,
    );
    p.line(
        "Perception grid",
        &format!("{}x{} @ {} Hz", man.grid, man.grid, man.grid_fps),
        Status::Ok,
    );
    // the RUN name, not the checkpoint path: "runs/v21b_c253_rev_s888/
    // jepa_best.pt" is three times the column and only one part of it varies
    let run = std::path::Path::new(&man.checkpoint)
        .parent()
        .and_then(|d| d.file_name())
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| man.checkpoint.clone());
    p.line("Weight bank", &run, Status::Ok);

    // The chain the session builder will actually walk (bundle::session), not a
    // claim about this machine -- which backend wins is only known once a
    // session is built, and that prints itself when it happens. `--cpu` takes
    // the GPU out of that chain, and gets the WARN chip: the run still works,
    // it is just the ~74x slower one, and this is the line that explains it.
    // A bundle whose attention cannot run on the CPU at all never reaches here
    // -- `main` refuses that combination before the report starts.
    if crate::bundle::force_cpu() {
        p.line("Accelerator chain", "CPU (forced by --cpu)", Status::Warn);
        p.note("the GPU is skipped -- expect hours where a graphics card takes minutes");
    } else {
        let chain = if cfg!(feature = "cuda") {
            "CUDA -> DirectML -> CPU"
        } else if cfg!(windows) {
            "DirectML -> CPU"
        } else {
            "CPU"
        };
        p.line("Accelerator chain", chain, Status::Ok);
    }
}

/// The sign-off.
pub fn ready(p: &Post) {
    p.line("Goblin runtime", "loaded", Status::Ready);
    p.end();
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Net vertical cursor movement in an escape sequence: up is negative, down
    /// positive, everything else ignored.
    fn drift(s: &str) -> i32 {
        let mut net = 0;
        let mut rest = s;
        while let Some(i) = rest.find("\x1b[") {
            let tail = &rest[i + 2..];
            let end = tail.find(|c: char| !c.is_ascii_digit()).unwrap_or(tail.len());
            let n: i32 = tail[..end].parse().unwrap_or(1);
            match tail[end..].chars().next() {
                Some('A') => net -= n,
                Some('B') => net += n,
                _ => {}
            }
            rest = &tail[end..];
        }
        net
    }

    // The report keeps printing under the banner while the banner is repainted
    // in place, so a frame has to come back to the exact row it started on --
    // at every distance the block ever sits at, and for every pose, including
    // the ones that lean or leave a row blank.
    #[test]
    fn a_repaint_comes_back_to_where_it_started() {
        for rows in 0..40usize {
            for tick in 0..64 {
                let pose = crate::mascot::boot_pose(tick);
                let mut buf: Vec<u8> = Vec::new();
                Post::art_frame(&mut buf, rows, &pose);
                let s = String::from_utf8(buf).unwrap();
                assert_eq!(drift(&s), 0, "a repaint drifted, {rows} rows below the block");
                // and it climbed to the block's TOP row, not into the report
                assert!(
                    s.starts_with(&format!("\x1b[{}A", rows + MASCOT_H)),
                    "the climb misses the top of the block: {s:?}"
                );
            }
        }
    }

    // A value that outgrows its column must be cut, not allowed to shove the
    // status chip out of the right-hand line -- the alignment IS the read.
    #[test]
    fn long_values_are_cut_to_the_column() {
        assert_eq!(fit("v21b_c253_rev_s888"), "v21b_c253_rev_s888");
        let long = "runs/some_extremely_long_run_name/jepa_best.pt";
        let cut = fit(long);
        assert_eq!(cut.chars().count(), VALUE);
        assert!(cut.ends_with('~'), "a cut value is marked as cut: {cut}");
    }

    // The whole report is paced by the constants above, and the pacing is the
    // feature. Pin the total: it has to be long enough to watch and to cover
    // the bundle load, and short enough that launching the app twice in a row
    // is not a chore.
    #[test]
    fn the_report_runs_about_five_seconds() {
        // banner: a lead-in, three inter-line beats, then a double beat
        let banner = BANNER_MS * 4 + BANNER_MS * 2;
        // system: ffmpeg, links, crew, scratch (4 lines) + the memory count-up
        let system = LINE_MS * 4 + MEM_STEPS * MEM_FRAME_MS + LINE_MS;
        // devices: core, grid, weights, chain
        let devices = LINE_MS * 4;
        let sign_off = LINE_MS + END_MS;
        let total = banner + system + devices + sign_off;
        assert!(
            (4_000..=7_000).contains(&total),
            "the POST should run 4-7 s, this one runs {total} ms"
        );
    }
}
