//! What went wrong, kept: one plain-text file beside the exe, and the same
//! words the picker shows back.
//!
//! A failure has to survive the screen it was printed on. The picker reopens
//! into the alternate screen the moment a batch ends, which takes the console
//! away with it -- so a red line is the right thing to print and the wrong
//! thing to rely on. Every failure is therefore recorded here first, and the
//! picker draws its report from the very same `Failure` values it appends.
//!
//! **The log is English, always.** The interface speaks whatever the user
//! chose; the log is what gets pasted into a bug report, so it is written in
//! the one language this repo can read back -- stage names come from
//! `lang::en`, and nothing translated is ever written to it.
//!
//! Best-effort throughout, like `settings`: a log that cannot be written costs
//! the run nothing, and `record` says so by returning `None` rather than by
//! failing a draft that otherwise worked.

use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;

/// Rotate at half a megabyte, keeping one older file. Two bounded files rather
/// than one unbounded one: a user who drafts nightly for a year should not
/// discover the log by running out of disk, and the failure they are chasing
/// is always in the newer of the two.
const MAX_BYTES: u64 = 512 * 1024;

/// The stage a draft is in, in English, so a failure can say where it got to.
/// Written by `Live` as each stage opens and cleared as it closes (`main`),
/// which is the one place that knows -- a stage is a screen line, and this is
/// the same line spelled for the log.
static STAGE: Mutex<Option<String>> = Mutex::new(None);

/// Has the run's header been written? The log is only ever opened when
/// something has gone wrong, so the header is written with the first failure
/// rather than at startup: a run that succeeds leaves the file untouched.
static HEADED: AtomicBool = AtomicBool::new(false);

/// `goblinscript.log`, beside the exe -- where `settings.json` and the cache
/// live, so everything the app leaves behind is in one folder the user can
/// find without being told a second path.
pub fn path() -> PathBuf {
    std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|d| d.join("goblinscript.log")))
        .unwrap_or_else(|| PathBuf::from("goblinscript.log"))
}

/// The stage now running, in English. `None` clears it -- between stages there
/// is no answer to "where did it get to", and a stale one is worse than none.
pub fn stage(name: Option<&str>) {
    if let Ok(mut s) = STAGE.lock() {
        *s = name.map(str::to_string);
    }
}

/// One failure, in English: what it was working on, how far it got, and the
/// error's whole chain of causes.
///
/// Held by value rather than formatted on the spot because two surfaces read
/// it -- the log file and the picker's report -- and they must not be able to
/// say different things about the same failure.
#[derive(Clone, Debug)]
pub struct Failure {
    /// The video's file name, or the step that failed when no one video owns
    /// it (fetching a link, reading a batch's VR aims).
    pub what: String,
    /// The stage in flight, when the failure came from inside a draft.
    pub stage: Option<String>,
    /// The error, then each cause under it, outermost first. Never empty.
    pub causes: Vec<String>,
}

impl Failure {
    /// Read one off an `anyhow` error, taking the stage from whatever was
    /// running when it happened.
    pub fn of(what: &str, e: &anyhow::Error) -> Failure {
        Failure {
            what: what.to_string(),
            // TAKEN, not read: a stage that ended by failing never cleared
            // itself, and the one thing worse than not knowing where a failure
            // got to is telling the next one it got to the same place.
            stage: STAGE.lock().ok().and_then(|mut s| s.take()),
            // `anyhow`'s chain is the error and every context above it,
            // outermost first -- which is also the order a reader wants: what
            // failed, then why, then why that.
            causes: e.chain().map(|c| c.to_string()).collect(),
        }
    }

    /// The log entry, English and self-contained -- readable a month later by
    /// someone who was not there.
    pub fn entry(&self, stamp: &str) -> String {
        let mut out = format!("[{stamp}] failed: {}\n", self.what);
        if let Some(s) = &self.stage {
            out.push_str(&format!("  during: {s}\n"));
        }
        let mut causes = self.causes.iter();
        out.push_str(&format!("  error: {}\n", causes.next().map_or("(no message)", |c| c)));
        for c in causes {
            out.push_str(&format!("  because: {c}\n"));
        }
        out
    }
}

/// Append a failure to the log; the path it landed in comes back so the user
/// can be told where to look, and `None` says the log could not be written --
/// a read-only install folder, most likely, which is worth knowing but never
/// worth failing a draft over.
pub fn record(f: &Failure) -> Option<PathBuf> {
    let p = path();
    let mut text = String::new();
    if !HEADED.swap(true, Ordering::Relaxed) {
        rotate(&p);
        text.push_str(&header());
    }
    text.push_str(&f.entry(&stamp(now())));
    let mut file = std::fs::OpenOptions::new().create(true).append(true).open(&p).ok()?;
    file.write_all(text.as_bytes()).ok()?;
    Some(p)
}

/// A line for something that failed outside any draft -- the run itself. Same
/// file, same shape, so a fatal error and a failed video read alike.
pub fn record_fatal(e: &anyhow::Error) -> Option<PathBuf> {
    record(&Failure::of("goblinscript", e))
}

/// What every entry in this run sits under: which build, on which machine's
/// operating system, started when. Three questions a bug report always has to
/// answer, asked once instead of per line.
fn header() -> String {
    format!(
        "\n==== goblinscript {} on {} -- run started {} ====\n",
        env!("CARGO_PKG_VERSION"),
        std::env::consts::OS,
        stamp(now())
    )
}

/// Move a log that has grown past its cap aside, once per run and before the
/// first entry -- so this run's failures are always at the end of the file the
/// user opens, and the previous set is still one file away.
fn rotate(p: &std::path::Path) {
    let big = std::fs::metadata(p).map(|m| m.len() > MAX_BYTES).unwrap_or(false);
    if big {
        let _ = std::fs::rename(p, p.with_extension("log.old"));
    }
}

fn now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// `2026-08-19 12:34:56 UTC` from a Unix timestamp.
///
/// UTC rather than local time, and said so on every line: the log crosses
/// machines (it is written to be sent), and a bare local time from an unknown
/// zone is a number nobody can line up with anything else. The run header
/// gives the reader their own anchor.
fn stamp(secs: u64) -> String {
    let (days, rem) = (secs / 86_400, secs % 86_400);
    let (y, m, d) = civil(days as i64);
    let (hh, mm, ss) = (rem / 3600, (rem % 3600) / 60, rem % 60);
    format!("{y:04}-{m:02}-{d:02} {hh:02}:{mm:02}:{ss:02} UTC")
}

/// Days since the epoch -> (year, month, day), by the shift-the-year-to-March
/// method: with March as month 0 the leap day lands at the END of the year, so
/// the whole calendar becomes arithmetic with no table and no special cases.
fn civil(z: i64) -> (i64, u64, u64) {
    let z = z + 719_468; // re-based on 0000-03-01
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u64; // day of era, [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365; // [0, 399]
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // day of that year, from March 1
    let mp = (5 * doy + 2) / 153; // month, March = 0
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    (yoe as i64 + era * 400 + i64::from(m <= 2), m, d)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The stage marker is process-wide -- one app, one draft at a time -- and
    /// reading it CONSUMES it, so any test that builds a `Failure` takes its
    /// turn. Tests run in parallel, and a stage stolen by a neighbour is the
    /// intermittent kind of failure.
    fn one_draft_at_a_time() -> std::sync::MutexGuard<'static, ()> {
        static LOCK: Mutex<()> = Mutex::new(());
        let g = LOCK.lock().unwrap_or_else(|e| e.into_inner());
        stage(None);
        g
    }

    /// The whole point of the file: a failure that scrolled off the screen is
    /// still readable, with everything needed to act on it -- which video,
    /// which stage, and the causes all the way down.
    #[test]
    fn an_entry_names_the_video_the_stage_and_every_cause() {
        let _one = one_draft_at_a_time();
        let e = anyhow::anyhow!("program not found (os error 2)")
            .context("could not start ffmpeg")
            .context("normalizing the source");
        let mut f = Failure::of("clip.mp4", &e);
        f.stage = Some("normalize".into());
        let entry = f.entry("2026-08-19 12:34:56 UTC");
        assert!(entry.contains("clip.mp4"), "{entry}");
        assert!(entry.contains("during: normalize"), "{entry}");
        assert!(entry.contains("error: normalizing the source"), "{entry}");
        assert!(entry.contains("because: could not start ffmpeg"), "{entry}");
        assert!(entry.contains("because: program not found (os error 2)"), "{entry}");
    }

    /// A stage that ends by failing never clears itself, so the failure that
    /// reads it has to -- or the NEXT failure, from anywhere in the app, would
    /// claim to have happened in a stage that stopped running minutes ago.
    #[test]
    fn a_failure_takes_the_stage_with_it() {
        let _one = one_draft_at_a_time();
        stage(Some("encode"));
        let first = Failure::of("clip.mp4", &anyhow::anyhow!("fell over"));
        assert_eq!(first.stage.as_deref(), Some("encode"));
        let second = Failure::of("preparing the batch", &anyhow::anyhow!("no such link"));
        assert_eq!(second.stage, None, "the second failure inherited a dead stage");
    }

    /// An error with no context above it is still an entry: `causes` is never
    /// empty, and the entry it writes must not read as though the line went
    /// missing.
    #[test]
    fn a_bare_error_is_still_a_whole_entry() {
        let _one = one_draft_at_a_time();
        let f = Failure::of("clip.mp4", &anyhow::anyhow!("the goblins fell over"));
        let entry = f.entry("2026-08-19 12:34:56 UTC");
        assert!(entry.contains("error: the goblins fell over"), "{entry}");
        assert!(!entry.contains("during:"), "no stage was running: {entry}");
    }

    /// Two entries from the same run must be tellable apart by when they
    /// happened, so the stamp is checked against dates whose answers are known
    /// -- including a leap day, which is the one the arithmetic can get wrong.
    #[test]
    fn the_stamp_reads_as_a_date_anyone_can_check() {
        assert_eq!(stamp(0), "1970-01-01 00:00:00 UTC");
        assert_eq!(stamp(1_000_000_000), "2001-09-09 01:46:40 UTC");
        assert_eq!(stamp(1_709_164_800), "2024-02-29 00:00:00 UTC");
        assert_eq!(stamp(1_755_561_600), "2025-08-19 00:00:00 UTC");
    }

    /// The log grows by a few lines per failure and is never read by the app,
    /// so nothing bounds it but this. The older file survives the rotation:
    /// the failure being chased is often the one BEFORE the flood that filled
    /// the file.
    #[test]
    fn an_oversized_log_is_moved_aside_rather_than_dropped() {
        let dir = std::env::temp_dir().join("goblinscript-errlog-test");
        let _ = std::fs::create_dir_all(&dir);
        let p = dir.join("goblinscript.log");
        let old = p.with_extension("log.old");
        let _ = std::fs::remove_file(&old);
        std::fs::write(&p, vec![b'x'; MAX_BYTES as usize + 1]).unwrap();
        rotate(&p);
        assert!(!p.exists(), "the oversized log stayed where it was");
        assert_eq!(std::fs::metadata(&old).unwrap().len(), MAX_BYTES + 1);
        // and a log still inside its cap is left alone
        std::fs::write(&p, "one line\n").unwrap();
        rotate(&p);
        assert_eq!(std::fs::read_to_string(&p).unwrap(), "one line\n");
        let _ = std::fs::remove_file(&p);
        let _ = std::fs::remove_file(&old);
    }
}
