//! Links in, local videos out: `yt-dlp` fetches a pasted URL and the rest of
//! the pipeline never learns it was not a file all along.
//!
//! yt-dlp is the USER'S OWN install, found on PATH. We do not ship it, fetch it,
//! or update it -- it moves faster than our release cycle and carries its own
//! extractor politics, and a tool the user installed is a tool the user can fix.
//!
//! A download lands in the directory the user is working in: the folder the
//! picker is browsing, or the shell's working directory from a command line. The
//! script then lands beside it by the ordinary rule, which is the whole reason
//! this is not a cache: a funscript is useless without the video it was timed
//! against, so a download is a document the user keeps, not scratch we clean up.

use anyhow::{bail, Context, Result};
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

/// The filename yt-dlp will write. PINNED rather than inherited: the target path
/// is predicted before the download starts (so a link already fetched is not
/// fetched twice), and a default that shifted under us would break that
/// prediction silently. The `[id]` keeps two videos of the same name apart.
const OUT_TEMPLATE: &str = "%(title)s [%(id)s].%(ext)s";

/// Best video + best audio, remuxed (not re-encoded) into mp4, falling back to
/// the best single stream. One container for every source keeps the prediction
/// above exact, and mp4 is also what the review page can stream straight from
/// the original instead of the normalized working copy.
const FORMAT: &[&str] = &["-f", "bv*+ba/b", "--merge-output-format", "mp4"];

/// Every yt-dlp launch starts here, because every one of them has to agree with
/// us about what its output means.
///
/// yt-dlp is Python, and Python encodes a piped stdout in the machine's own
/// codepage -- cp1252, cp932, cp936, whatever the user's Windows was installed
/// as -- with unmappable characters DROPPED rather than flagged. The path it
/// prints for a video titled outside that codepage would then come back as a
/// name no file has, and the prediction the whole feature rests on (fetch once,
/// find it again next run) would miss on exactly the titles a person is most
/// likely to paste. `PYTHONIOENCODING` is what sets that encoding, and it is
/// read by a frozen yt-dlp.exe as much as by a script; a Windows console
/// handle is written through the wide API either way, so the bar the user
/// watches during a blocking fetch is unaffected.
fn yt_dlp() -> Command {
    let mut c = Command::new("yt-dlp");
    c.env("PYTHONIOENCODING", "utf-8");
    c
}

/// Is this argument a link rather than a path? Deliberately narrow: only the two
/// schemes a browser puts on the clipboard, so a local file whose name happens to
/// contain a colon is never mistaken for one.
pub fn is_url(s: &str) -> bool {
    let s = s.trim();
    s.starts_with("http://") || s.starts_with("https://")
}

/// How to get the tool, said the same way wherever we have to say it.
pub const INSTALL_HINT: &str = "yt-dlp is not on PATH -- the goblins fetch links with it, but do \
                                not install it for you. Get it with:\n  winget install --id \
                                yt-dlp.yt-dlp\nthen open a new terminal and try again";

/// The probe, run ONCE and remembered: `None` when yt-dlp is not on PATH, else
/// the version it named -- empty if it ran but printed nothing readable. Asked
/// while drawing (the picker offers the paste key only when the tool is there),
/// and a process launch per frame is not a thing to do while drawing.
fn probe() -> Option<&'static str> {
    static FOUND: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
    FOUND
        .get_or_init(|| {
            let out = yt_dlp().arg("--version").output().ok()?;
            if !out.status.success() {
                return None;
            }
            // yt-dlp prints its version and nothing else: "2025.06.09" from a
            // release, "2025.06.09.232809" from a nightly.
            Some(String::from_utf8_lossy(&out.stdout).trim().to_string())
        })
        .as_deref()
}

/// Is yt-dlp on PATH? Without it the whole feature is simply not offered -- the
/// picker shows no paste key and the startup report says links are off.
pub fn available() -> bool {
    probe().is_some()
}

/// yt-dlp's version, for the startup report. Purely informational: `None` on
/// anything unexpected, and no caller reads that as an answer about whether
/// links work -- that one is `available`'s to give.
pub fn version() -> Option<&'static str> {
    probe().filter(|v| !v.is_empty())
}

/// yt-dlp on PATH, or an error that says how to get it. For the one path where
/// the user named a link outright and silence would be wrong.
pub fn have_tool() -> Result<()> {
    if !available() {
        bail!("{INSTALL_HINT}");
    }
    Ok(())
}

/// Where this link WOULD land, without fetching a byte. `--simulate` resolves
/// the title and id the same way the real run will, so an existing file here is
/// the same video and the caller can skip the download entirely.
///
/// `extra` is the user's own yt-dlp arguments, and they are passed HERE as well
/// as to the download itself -- a format or template choice has to reach both
/// calls or the prediction stops describing what the download will write.
pub fn target(url: &str, dir: &Path, extra: &[String]) -> Result<PathBuf> {
    let out = yt_dlp()
        .args(["--simulate", "--no-playlist", "--print", "filename", "-o", OUT_TEMPLATE])
        .args(FORMAT)
        .arg("-P")
        .arg(dir)
        .args(extra)
        .arg(url)
        .output()
        .context("yt-dlp failed to run")?;
    if !out.status.success() {
        bail!(
            "yt-dlp could not read that link: {}",
            String::from_utf8_lossy(&out.stderr).lines().last().unwrap_or("no reason given")
        );
    }
    let name = String::from_utf8_lossy(&out.stdout);
    let name = name.trim();
    if name.is_empty() {
        bail!("yt-dlp named no file for that link");
    }
    // yt-dlp prints the path already joined to -P; take only the file name so
    // the caller's directory is the one that decides where this goes
    Ok(dir.join(Path::new(name).file_name().unwrap_or_else(|| name.as_ref())))
}

/// How far along a download is, as the progress line reports it.
#[derive(Clone, Default)]
pub struct Progress {
    pub pct: f64,
    /// Total size and ETA as yt-dlp formats them -- passed through as strings
    /// because they are for a human to read, not for us to compute with.
    pub size: String,
    pub eta: String,
}

/// What a fetch has to say for itself, as it happens.
enum Msg {
    /// Where it is going, once yt-dlp has resolved the title (a round trip).
    Dest(PathBuf),
    Progress(Progress),
    Done(Result<PathBuf>),
}

/// A running download, as seen from a surface with its own event loop. Every
/// blocking part -- resolving the name, spawning, reading progress -- happens on
/// the worker thread, because the picker asks for this from inside a 120 ms draw
/// loop and a UI that freezes for a network round trip is a broken UI.
pub struct Job {
    rx: std::sync::mpsc::Receiver<Msg>,
    stop: std::sync::Arc<std::sync::atomic::AtomicBool>,
    dest: Option<PathBuf>,
}

/// Progress lines are asked for in this shape -- one per line (`--newline`), so
/// a reader can take them a line at a time instead of parsing carriage returns.
const PROGRESS_TEMPLATE: &str = "GS %(progress._percent_str)s|%(progress._total_bytes_str)s|%(progress._eta_str)s";

/// Begin fetching `url` into `dir`, without blocking the caller. A link already
/// on disk finishes on the first pump rather than through a second code path.
pub fn start(url: &str, dir: &Path, extra: &[String]) -> Result<Job> {
    std::fs::create_dir_all(dir)
        .with_context(|| format!("cannot write downloads to {}", dir.display()))?;
    let (tx, rx) = std::sync::mpsc::channel();
    let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let (url_s, dir_s, stop_w) = (url.to_string(), dir.to_path_buf(), std::sync::Arc::clone(&stop));
    let extra_w = extra.to_vec();
    std::thread::spawn(move || {
        let done = work(&url_s, &dir_s, &extra_w, &tx, &stop_w);
        let _ = tx.send(Msg::Done(done));
    });
    Ok(Job { rx, stop, dest: None })
}

/// The worker: resolve the name, then run the download and relay its progress.
/// Split out so every early return lands in one `Msg::Done`.
fn work(
    url: &str,
    dir: &Path,
    extra: &[String],
    tx: &std::sync::mpsc::Sender<Msg>,
    stop: &std::sync::atomic::AtomicBool,
) -> Result<PathBuf> {
    use std::sync::atomic::Ordering;
    let started = std::time::SystemTime::now();
    let dest = target(url, dir, extra)?;
    let _ = tx.send(Msg::Dest(dest.clone()));
    // Fetched before and still here: yt-dlp would say so itself, but answering
    // from disk costs nothing and keeps a re-paste instant.
    if dest.exists() {
        return Ok(dest);
    }
    let mut child = yt_dlp()
        .args(["--no-playlist", "--newline", "--progress", "--progress-template", PROGRESS_TEMPLATE])
        .args(["-o", OUT_TEMPLATE])
        .args(FORMAT)
        .arg("-P")
        .arg(dir)
        // last, so a user argument beats ours wherever yt-dlp takes the last word
        .args(extra)
        .arg(url)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .context("yt-dlp failed to start")?;
    if let Some(out) = child.stdout.take() {
        let mut r = BufReader::new(out);
        let mut line = String::new();
        // A cancel is noticed between lines. With --newline yt-dlp reports every
        // chunk, so that is a fraction of a second in practice.
        while matches!(r.read_line(&mut line), Ok(n) if n > 0) {
            if stop.load(Ordering::Relaxed) {
                let _ = child.kill();
                let _ = child.wait();
                bail!("cancelled");
            }
            if let Some(p) = parse_progress(&line) {
                let _ = tx.send(Msg::Progress(p));
            }
            line.clear();
        }
    }
    let status = child.wait().context("yt-dlp vanished mid-download")?;
    if !status.success() {
        let mut why = String::new();
        if let Some(err) = child.stderr.as_mut() {
            let _ = std::io::Read::read_to_string(err, &mut why);
        }
        match why.lines().rev().find(|l| !l.trim().is_empty()) {
            Some(last) => bail!("yt-dlp failed: {last}"),
            None => bail!("yt-dlp failed"),
        }
    }
    // A merge can land a different container than predicted; trust the disk.
    Ok(settled(&dest, started))
}

impl Job {
    /// Where the video is going, once yt-dlp has said. `None` until then.
    pub fn dest(&self) -> Option<&Path> {
        self.dest.as_deref()
    }

    /// Take everything the worker has said since the last call. `None` while it
    /// is still going; `Some` once it is finished, one way or the other.
    ///
    /// Never blocks: the caller is a draw loop.
    pub fn pump(&mut self, on: &mut impl FnMut(Progress)) -> Option<Result<PathBuf>> {
        loop {
            match self.rx.try_recv() {
                Ok(Msg::Dest(p)) => self.dest = Some(p),
                Ok(Msg::Progress(p)) => on(p),
                Ok(Msg::Done(r)) => return Some(r),
                // the worker always sends Done before dropping the sender, so a
                // dead channel without one means the thread itself died
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    return Some(Err(anyhow::anyhow!("the fetch died without saying why")))
                }
                Err(std::sync::mpsc::TryRecvError::Empty) => return None,
            }
        }
    }

    /// Ask the download to stop. The part-file stays where it is, and pasting the
    /// same link again resumes from it.
    pub fn cancel(&mut self) {
        self.stop.store(true, std::sync::atomic::Ordering::Relaxed);
    }
}

/// Fetch a link and wait for it, letting yt-dlp draw its own progress on the
/// terminal it was started from. For the command line, where the user is looking
/// at a console and yt-dlp's bar is the one they already know.
pub fn fetch_blocking(url: &str, dir: &Path, extra: &[String]) -> Result<PathBuf> {
    std::fs::create_dir_all(dir)
        .with_context(|| format!("cannot write downloads to {}", dir.display()))?;
    let started = std::time::SystemTime::now();
    let dest = target(url, dir, extra)?;
    if dest.exists() {
        return Ok(dest);
    }
    let status = yt_dlp()
        .args(["--no-playlist", "-o", OUT_TEMPLATE])
        .args(FORMAT)
        .arg("-P")
        .arg(dir)
        .args(extra)
        .arg(url)
        .status()
        .context("yt-dlp failed to start")?;
    if !status.success() {
        bail!("yt-dlp could not fetch that link");
    }
    Ok(settled(&dest, started))
}

/// The predicted path if it exists, else the same stem under whatever extension
/// the merge actually produced (`.mkv` when a codec cannot live in mp4), else
/// the video this download is the only plausible author of.
///
/// That last step is the net under the prediction rather than a second way of
/// making it. The name comes back from yt-dlp as text, and text can arrive
/// spelled differently than it landed on disk -- a title normalized one way in
/// the print and the other in the filename, a character the tool's own output
/// encoding could not carry. A download that yt-dlp says succeeded HAS left a
/// file, so a newly written video in the directory it was told to write to is
/// that file, and answering with it beats handing the caller a path nothing is
/// at.
fn settled(dest: &Path, since: std::time::SystemTime) -> PathBuf {
    if dest.exists() {
        return dest.to_path_buf();
    }
    let (Some(dir), Some(stem)) = (dest.parent(), dest.file_stem()) else {
        return dest.to_path_buf();
    };
    let Ok(rd) = std::fs::read_dir(dir) else {
        return dest.to_path_buf();
    };
    // Filesystem timestamps are coarser than a call to the clock, so the window
    // opens slightly before the download did rather than exactly at it.
    let window = since - std::time::Duration::from_secs(2);
    let mut newest: Option<(std::time::SystemTime, PathBuf)> = None;
    for p in rd.flatten().map(|e| e.path()) {
        if p.extension() == Some("part".as_ref()) {
            continue;
        }
        if p.file_stem() == Some(stem) {
            return p;
        }
        if !crate::is_video(&p) {
            continue;
        }
        let Ok(t) = p.metadata().and_then(|m| m.modified()) else { continue };
        if t >= window && newest.as_ref().is_none_or(|(best, _)| t > *best) {
            newest = Some((t, p));
        }
    }
    newest.map(|(_, p)| p).unwrap_or_else(|| dest.to_path_buf())
}

/// `GS  42.3%|  1.20GiB|00:31` -> a `Progress`. Anything else is yt-dlp talking
/// about something other than progress, and is not our business.
fn parse_progress(line: &str) -> Option<Progress> {
    let rest = line.trim().strip_prefix("GS ")?;
    let mut f = rest.split('|');
    let pct = f.next()?.trim().trim_end_matches('%').parse::<f64>().ok()?;
    Some(Progress {
        pct,
        size: f.next().unwrap_or("").trim().to_string(),
        eta: f.next().unwrap_or("").trim().to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_http_links_are_links() {
        assert!(is_url("https://example.com/watch?v=abc"));
        assert!(is_url("  http://example.com/x  "));
        assert!(!is_url(r"C:\videos\clip.mp4"));
        assert!(!is_url(r"D:\odd name; http.mp4"));
        assert!(!is_url("clip.mp4"));
        assert!(!is_url("ftp://example.com/x.mp4"));
    }

    #[test]
    fn progress_lines_parse_and_others_are_ignored() {
        let p = parse_progress("GS  42.3%|  1.20GiB|00:31\n").expect("a progress line");
        assert!((p.pct - 42.3).abs() < 1e-9);
        assert_eq!(p.size, "1.20GiB");
        assert_eq!(p.eta, "00:31");
        // yt-dlp's own chatter, and a half-written line, are not progress
        assert!(parse_progress("[youtube] Extracting URL: https://...").is_none());
        assert!(parse_progress("GS NA|NA|NA").is_none());
        assert!(parse_progress("").is_none());
    }

    #[test]
    fn a_finished_download_is_found_under_any_extension() {
        let dir = std::env::temp_dir().join(format!("gs_dl_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let started = std::time::SystemTime::now();
        let predicted = dir.join("Clip [abc123].mp4");
        // the merge produced .mkv instead, and left a part-file behind
        std::fs::write(dir.join("Clip [abc123].mkv"), b"x").unwrap();
        std::fs::write(dir.join("Clip [abc123].mp4.part"), b"x").unwrap();
        assert_eq!(settled(&predicted, started), dir.join("Clip [abc123].mkv"));
        // and when the prediction was right, it wins outright
        std::fs::write(&predicted, b"x").unwrap();
        assert_eq!(settled(&predicted, started), predicted);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The name yt-dlp printed and the name it wrote can differ by more than an
    /// extension -- a title spelled one way in the print and another on disk is
    /// a mismatch no stem search closes. The download still happened, so the
    /// video written while it ran is the answer, and a file that was already
    /// sitting in the folder is not.
    #[test]
    fn a_download_that_landed_under_another_name_is_still_found() {
        let dir = std::env::temp_dir().join(format!("gs_dl_name_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let old = dir.join("something else.mp4");
        std::fs::write(&old, b"x").unwrap();
        // the clock the fetch started on, a beat after the file that predates it
        let started = std::time::SystemTime::now() + std::time::Duration::from_secs(60);
        let landed = dir.join("Cafe [abc123].mp4");
        std::fs::write(&landed, b"x").unwrap();
        filetime_now(&landed, started);
        let predicted = dir.join("Caf\u{fffd} [abc123].mp4");
        assert_eq!(settled(&predicted, started), landed);
        // nothing new in the folder: the prediction comes back unchanged rather
        // than an unrelated file the user already had
        let _ = std::fs::remove_file(&landed);
        assert_eq!(settled(&predicted, started), predicted);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Give a file a modification time, so the window `settled` reads can be
    /// tested without waiting on a clock.
    fn filetime_now(p: &Path, t: std::time::SystemTime) {
        let f = std::fs::OpenOptions::new().write(true).open(p).unwrap();
        f.set_modified(t).unwrap();
    }
}
