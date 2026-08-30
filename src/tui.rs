//! The interactive picker: run goblinscript with no arguments (double-clicking
//! the exe does exactly that) and choose videos + options in a small terminal
//! UI instead of typing flags. It is only a LAUNCHER -- once the user hits
//! start it hands a normal `Pick` back to main, the alternate screen closes,
//! and the run itself uses the ordinary console output, which keeps the
//! per-video history scrollable and identical between both ways of running.

use crate::mascot::{self, BAND_MIN_W, BAND_W, BLOCK_W, GOBLIN_H, SOLO_MIN_W, WAVE_MIN_W, WAVE_W};
use crate::style;
use crate::t;
use crate::theme::{rat, theme, Theme};
use crate::{is_video, script_kind, videos_in, ScriptKind};
use anyhow::Result;
use ratatui::crossterm::event::{self, Event, KeyCode, KeyEvent, KeyEventKind};
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Color, Modifier, Style, Stylize};
use ratatui::text::{Line, Span};
use ratatui::widgets::{List, ListItem, ListState, Paragraph};
use ratatui::Frame;
use std::collections::{BTreeSet, HashMap};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// What the user chose; mirrors the CLI flags it replaces.
pub struct Pick {
    pub videos: Vec<PathBuf>,
    pub force: bool,
    /// Auto-crop the batch (C in the picker) -- `--autocrop`'s toggle twin.
    pub autocrop: bool,
    /// Show the crop page before each video (K in the picker) --
    /// `--crop-edit`'s toggle twin. Off is what a batch left running wants.
    pub crop_edit: bool,
    /// Where the browser was when the user hit start, so the next pick reopens
    /// there instead of jumping back to the launch directory.
    pub dir: Option<PathBuf>,
    /// Videos the user marked as VR by hand (B). The detector is recall-biased
    /// but not perfect, and a missed VR source costs a full-length draft of
    /// nonsense -- so the listing is where that can be corrected, per video.
    pub vr: BTreeSet<PathBuf>,
}

/// What the batch that just ran left behind, for the picker to hand back.
///
/// A batch ends by returning here, and the picker's first act is to take the
/// screen -- so whatever the console said about it is wiped before anyone can
/// read it. This is that news carried ACROSS the handover instead: the
/// one-line outcome, every failure in full, and the file they were written to.
#[derive(Default)]
pub struct Report {
    /// Drafted/skipped/failed, for the status line.
    pub status: Option<String>,
    /// Everything that went wrong, in the same words the log has. Empty is the
    /// ordinary case and costs nothing.
    pub failures: Vec<crate::errlog::Failure>,
    /// Where those were appended, so the screen can say where to look again
    /// tomorrow. `None` means the log could not be written.
    pub log: Option<PathBuf>,
}

/// What one probe told us about a listed video: how long it runs, and whether
/// the VR detector reads it as VR.
#[derive(Clone, Copy)]
struct Probed {
    /// `None` = ffprobe could not read it, which is not the same as pending.
    dur_ms: Option<f64>,
    vr: bool,
}

/// Every probe this session, keyed by path. Workers insert, the draw reads; a
/// path that is absent is still in flight and draws as `--:--`.
type Probes = Arc<Mutex<HashMap<PathBuf, Probed>>>;

/// What wrote the script beside each video, once anyone has looked. Filled by
/// the classifier pool, read by the draw and by the folder counts; a path that
/// is absent has a script nobody has opened yet, which is not the same as no
/// script (that is `Video::scripted`, and it is known immediately).
type Kinds = Arc<Mutex<HashMap<PathBuf, ScriptKind>>>;

/// The rows the probe pool should look at next -- rewritten by the draw to
/// whatever is on screen, so the work follows the cursor.
type Want = Arc<Mutex<Vec<PathBuf>>>;

/// Probes run on this many threads. ffprobe is a process LAUNCH per video, which
/// is why the pool is fed the visible rows and not the directory: a folder of
/// thousands would otherwise spend minutes spawning processes for rows nobody
/// will scroll to, in competition with the app for the same disk.
const PROBE_WORKERS: usize = 4;

/// Classifying scripts is a file read rather than a process, so it can afford to
/// cover the WHOLE folder (the counts on the location line are over the whole
/// folder) -- but it is still I/O, and it still belongs off the draw thread.
const KIND_WORKERS: usize = 2;

/// Rows either side of the screen the probe pool reads ahead by, so a slow
/// scroll finds its durations already home instead of watching them arrive.
const PROBE_LOOKAHEAD: usize = 40;

// The palette lives in `theme` -- shared with the processing console and the
// review page, and cycled live with T. Whichever one is active, the contract is
// the same: bright foregrounds, no dim greys, because legibility is not an
// optional mode.

/// The drive letters that exist, from the bitmask the OS already holds. NOT a
/// `Path::exists()` per letter: that TOUCHES each drive, so an empty optical
/// bay spins up and a disconnected network mapping waits out its timeout --
/// seconds of freeze, on the one screen the picker opens to.
#[cfg(windows)]
fn drives() -> Vec<PathBuf> {
    let mask = unsafe { windows::Win32::Storage::FileSystem::GetLogicalDrives() };
    (0..26u32)
        .filter(|i| mask & (1 << i) != 0)
        .map(|i| PathBuf::from(format!("{}:\\", (b'A' + i as u8) as char)))
        .collect()
}

/// Everywhere else there is one root, and it is always there.
#[cfg(not(windows))]
fn drives() -> Vec<PathBuf> {
    vec![PathBuf::from("/")]
}

enum Entry {
    Up,
    Dir(PathBuf),
    /// A video. What WROTE the script beside it is not carried here: that is a
    /// file read, so it lands in `App::kinds` from the classifier pool and the
    /// row picks it up on whatever frame it arrives. Whether there is a script
    /// at all is known the moment the folder is listed -- see `Video::scripted`.
    File(PathBuf),
}

/// One row of the folder being browsed, as the single `read_dir` pass left it.
/// Nothing here costs a syscall of its own: the name set that pass collected
/// answers "is there a script beside it" for every video at once.
struct Row {
    path: PathBuf,
    /// The lowercased file name the filter matches against, folded ONCE when
    /// the folder is listed. The filter runs on every keystroke over every row,
    /// so folding there instead would be an allocation per row per keypress.
    key: String,
    /// A `.funscript` sits beside this video. Meaningless on a directory row.
    scripted: bool,
}

struct App {
    /// `None` = the drive list (above every root).
    cur: Option<PathBuf>,
    /// The whole folder as read from disk, subdirectories then videos. Rebuilt
    /// only when the FOLDER changes (or R): the filter is a predicate over
    /// names that has no business touching a disk.
    folders: Vec<Row>,
    videos: Vec<Row>,
    /// What is on screen: the model above, narrowed by `filter`. Rebuilding it
    /// costs a substring compare per row and no I/O, which is what makes typing
    /// a filter in a folder of thousands feel like typing.
    entries: Vec<Entry>,
    /// The first row drawn. Owned here rather than left to `ListState` because
    /// the draw builds items for the VISIBLE window only -- a folder of
    /// thousands must not cost thousands of assembled rows every frame.
    offset: usize,
    list: ListState,
    selected: BTreeSet<PathBuf>,
    force: bool,
    /// Auto-crop for the next batch (C). Seeded from the remembered setting;
    /// the choice sticks when a batch starts with it.
    autocrop: bool,
    /// The crop page before each video of the next batch (K), on the same
    /// terms. Off is what a batch left running unattended wants.
    crop_edit: bool,
    /// Feedback from the batch that just ran (drafted/skipped/failed) -- shown
    /// so an instant, all-skipped start is not silent.
    status: Option<String>,
    /// What that batch could not do, and the file it was appended to. Kept
    /// whole rather than squeezed into the status line: an error is several
    /// lines of English and that line has room for none of them.
    failures: Vec<crate::errlog::Failure>,
    log: Option<PathBuf>,
    /// The failure report is up, covering the listing. It opens BY ITSELF when
    /// a batch failed -- a report nobody asked for is exactly what a failure
    /// needs -- and X brings it back for as long as it is the latest news.
    errors: bool,
    /// The first report line drawn: a batch of ten failures is longer than any
    /// screen, and the last one is as worth reading as the first.
    err_off: usize,
    /// A live substring filter over the current listing (`/` starts it). While
    /// `filtering`, typed characters extend it and the listing narrows.
    filter: String,
    filtering: bool,
    /// Videos in the current directory, and how many already have a script,
    /// split by what wrote it -- shown so a folder that is mostly done, or one
    /// full of hand work, reads at a glance. The first two are exact the moment
    /// the folder is listed; the split fills in as the classifier pool reads the
    /// scripts, because that is a file read per video and the listing does not
    /// wait for one.
    n_videos: usize,
    n_scripted: usize,
    n_ai: Arc<AtomicUsize>,
    n_hand: Arc<AtomicUsize>,
    /// Videos the user marked as VR (B) because the detector read them as flat.
    /// Per video and per session: what persists is the answer the prep page
    /// saves to the sidecar afterwards.
    vr_marks: BTreeSet<PathBuf>,
    /// A link being typed (L), the fetch it turned into, and the one line either
    /// has to say for itself. Only ever set when yt-dlp is on PATH -- without it
    /// the key is not offered and none of this can start.
    url: Option<String>,
    job: Option<crate::dl::Job>,
    dl_line: Option<String>,
    /// Where a fetched video goes (`--dl-dir`), or the browsed folder when unset,
    /// and the user's own yt-dlp arguments from after a `--`.
    dl_dir: Option<PathBuf>,
    dl_args: Vec<String>,
    probes: Probes,
    /// The rows the probe pool is working on: the visible window plus a
    /// lookahead, rewritten by the draw. The pool itself is spawned once and
    /// lives as long as the picker, so walking through folders costs no threads.
    want: Want,
    kinds: Kinds,
    /// Bumped on every listing. Both pools stop as soon as they see a newer one,
    /// so leaving a folder abandons its queue instead of draining it.
    probe_gen: Arc<AtomicU64>,
    /// Set when the picker returns, so the pools' threads end with it.
    stop: Arc<AtomicBool>,
    /// When this picker opened -- the clock the blinking filter cursor runs on.
    started: Instant,
}

impl App {
    fn new(
        force: bool,
        autocrop: bool,
        crop_edit: bool,
        start: Option<PathBuf>,
        report: Report,
        dl_dir: Option<PathBuf>,
        dl_args: Vec<String>,
    ) -> Self {
        let mut a = Self {
            cur: start,
            folders: Vec::new(),
            videos: Vec::new(),
            entries: Vec::new(),
            offset: 0,
            list: ListState::default(),
            selected: BTreeSet::new(),
            force,
            autocrop,
            crop_edit,
            status: report.status,
            errors: !report.failures.is_empty(),
            err_off: 0,
            failures: report.failures,
            log: report.log,
            filter: String::new(),
            filtering: false,
            n_videos: 0,
            n_scripted: 0,
            n_ai: Arc::new(AtomicUsize::new(0)),
            n_hand: Arc::new(AtomicUsize::new(0)),
            vr_marks: BTreeSet::new(),
            url: None,
            job: None,
            dl_line: None,
            dl_dir,
            dl_args,
            probes: Probes::default(),
            want: Want::default(),
            kinds: Kinds::default(),
            probe_gen: Arc::new(AtomicU64::new(0)),
            stop: Arc::new(AtomicBool::new(false)),
            started: Instant::now(),
        };
        a.spawn_probe_pool();
        a.refresh();
        a
    }

    /// Re-read the folder from disk, then rebuild the view. The ONLY entry
    /// point that touches a disk -- everything the filter does goes through
    /// `refilter` instead.
    ///
    /// One `read_dir` pass and nothing else. The names it walks past are
    /// already the answer to "does this video have a script beside it", so that
    /// question costs a set lookup rather than a `stat` per video; and what
    /// WROTE each script is a file read, so it is left to the classifier pool
    /// and arrives on a later frame.
    fn refresh(&mut self) {
        self.folders.clear();
        self.videos.clear();
        self.n_videos = 0;
        self.n_scripted = 0;
        let key = |p: &Path| p.file_name().unwrap_or_default().to_string_lossy().to_lowercase();
        match &self.cur {
            None => {
                for d in drives() {
                    let k = d.display().to_string().to_lowercase();
                    self.folders.push(Row { path: d, key: k, scripted: false });
                }
            }
            Some(dir) => {
                // Both halves of the pass: the folders and videos to list, and
                // the bare stems of every `.funscript` seen on the way past.
                let mut scripts: std::collections::HashSet<String> = std::collections::HashSet::new();
                if let Ok(rd) = std::fs::read_dir(dir) {
                    for e in rd.flatten() {
                        let name = e.file_name().to_string_lossy().to_string();
                        if name.starts_with('.')
                            || name.starts_with('$')
                            || name == "System Volume Information"
                        {
                            continue;
                        }
                        let path = e.path();
                        // `file_type()` here is free: it comes out of the
                        // directory entry the walk already read.
                        match e.file_type() {
                            Ok(t) if t.is_dir() => {
                                let k = key(&path);
                                self.folders.push(Row { path, key: k, scripted: false });
                            }
                            Ok(t) if t.is_file() => {
                                if is_video(&path) {
                                    let k = key(&path);
                                    self.videos.push(Row { path, key: k, scripted: false });
                                } else if path
                                    .extension()
                                    .and_then(|e| e.to_str())
                                    .is_some_and(|e| e.eq_ignore_ascii_case("funscript"))
                                {
                                    scripts.insert(
                                        path.file_stem()
                                            .unwrap_or_default()
                                            .to_string_lossy()
                                            .to_lowercase(),
                                    );
                                }
                            }
                            _ => {}
                        }
                    }
                }
                for v in &mut self.videos {
                    let stem = v.path.file_stem().unwrap_or_default().to_string_lossy().to_lowercase();
                    v.scripted = scripts.contains(&stem);
                }
                self.folders.sort_by(|a, b| a.key.cmp(&b.key));
                self.videos.sort_by(|a, b| a.key.cmp(&b.key));
                self.n_videos = self.videos.len();
                self.n_scripted = self.videos.iter().filter(|v| v.scripted).count();
            }
        }
        self.spawn_kind_scan();
        self.refilter();
        self.list.select((!self.entries.is_empty()).then_some(0));
        self.offset = 0;
    }

    /// Rebuild the visible listing from the model already in memory. A
    /// substring compare per row and no I/O, which is what a keystroke is
    /// allowed to cost: `refresh` used to run here, and in a folder of
    /// thousands it re-read every script beside every video, per character.
    fn refilter(&mut self) {
        let needle = self.filter.to_lowercase();
        let hit = |r: &Row| needle.is_empty() || r.key.contains(&needle);
        self.entries.clear();
        if self.cur.is_some() {
            self.entries.push(Entry::Up);
        }
        self.entries
            .extend(self.folders.iter().filter(|r| hit(r)).map(|r| Entry::Dir(r.path.clone())));
        self.entries
            .extend(self.videos.iter().filter(|r| hit(r)).map(|r| Entry::File(r.path.clone())));
        // The cursor keeps its row where it can, and stops at the end where it
        // cannot: narrowing a listing under a cursor that was near the bottom
        // must not leave it pointing past the last row.
        if self.entries.is_empty() {
            self.list.select(None);
            self.offset = 0;
        } else if let Some(i) = self.list.selected() {
            self.list.select(Some(i.min(self.entries.len() - 1)));
        } else {
            self.list.select(Some(0));
        }
    }

    /// The probe pool: spawned once, alive for the picker's lifetime, reading
    /// whatever `want` currently holds. One ffprobe per video, for the duration
    /// column and the VR guess.
    ///
    /// It reads a WANT-LIST rather than draining the folder, because a probe is
    /// a process launch: a folder of thousands would spend minutes spawning
    /// them for rows nobody scrolls to, competing with the app for the same
    /// disk, and the duration column only ever shows the rows on screen. The
    /// draw rewrites the list every frame, so the work follows the cursor.
    /// A path is probed once per session, and a failed probe is recorded as
    /// failed rather than retried on every re-listing.
    fn spawn_probe_pool(&self) {
        for _ in 0..PROBE_WORKERS {
            let want = Arc::clone(&self.want);
            let probes = Arc::clone(&self.probes);
            let stop = Arc::clone(&self.stop);
            std::thread::spawn(move || {
                while !stop.load(Ordering::Relaxed) {
                    let next = want.lock().unwrap().pop();
                    let Some(p) = next else {
                        // Nothing on screen needs a probe. Idle cheaply rather
                        // than spinning: the draw refills the list at its own
                        // rate and there is nothing to race for.
                        std::thread::sleep(Duration::from_millis(40));
                        continue;
                    };
                    if probes.lock().unwrap().contains_key(&p) {
                        continue;
                    }
                    let got = crate::vr::probe_summary(&p);
                    let probed = Probed {
                        dur_ms: got.map(|(d, _)| d).filter(|d| *d > 0.0),
                        vr: got.is_some_and(|(_, vr)| vr),
                    };
                    probes.lock().unwrap().insert(p, probed);
                }
            });
        }
    }

    /// Hand the probe pool the rows actually on screen (plus a lookahead either
    /// side, so an unhurried scroll finds its durations already home). Called
    /// from the draw, which is the only place that knows how tall the list is.
    /// Newest window wins outright -- an abandoned one is not worth finishing.
    fn aim_probes(&self, height: usize) {
        let lo = self.offset.saturating_sub(PROBE_LOOKAHEAD);
        let hi = (self.offset + height + PROBE_LOOKAHEAD).min(self.entries.len());
        let have = self.probes.lock().unwrap();
        let mut want: Vec<PathBuf> = self.entries[lo.min(hi)..hi]
            .iter()
            .filter_map(|e| match e {
                Entry::File(p) if !have.contains_key(p) => Some(p.clone()),
                _ => None,
            })
            .collect();
        drop(have);
        want.reverse(); // workers pop from the end, so the top of the window goes first
        *self.want.lock().unwrap() = want;
    }

    /// Classify every script in the folder in the background: what wrote it is
    /// a file READ, and the listing does not wait for one. The location line's
    /// ai/hand split fills in as the answers land; the scripted TOTAL beside it
    /// was already exact from the directory pass.
    ///
    /// Generation-cancelled, so leaving a folder abandons its scan instead of
    /// draining it, and cached per path, so walking back into a folder is free.
    fn spawn_kind_scan(&self) {
        self.n_ai.store(0, Ordering::Relaxed);
        self.n_hand.store(0, Ordering::Relaxed);
        let mut todo: Vec<PathBuf> = Vec::new();
        {
            let have = self.kinds.lock().unwrap();
            for v in self.videos.iter().filter(|v| v.scripted) {
                let script = v.path.with_extension("funscript");
                match have.get(&script) {
                    // already known this session: count it now, read nothing
                    Some(ScriptKind::Ai) => {
                        self.n_ai.fetch_add(1, Ordering::Relaxed);
                    }
                    Some(ScriptKind::Hand) => {
                        self.n_hand.fetch_add(1, Ordering::Relaxed);
                    }
                    Some(ScriptKind::Unknown) => {}
                    None => todo.push(script),
                }
            }
        }
        let gen = self.probe_gen.fetch_add(1, Ordering::Relaxed) + 1;
        if todo.is_empty() {
            return;
        }
        todo.reverse();
        let queue = Arc::new(Mutex::new(todo));
        for _ in 0..KIND_WORKERS {
            let queue = Arc::clone(&queue);
            let kinds = Arc::clone(&self.kinds);
            let cur = Arc::clone(&self.probe_gen);
            let (n_ai, n_hand) = (Arc::clone(&self.n_ai), Arc::clone(&self.n_hand));
            let stop = Arc::clone(&self.stop);
            std::thread::spawn(move || {
                while cur.load(Ordering::Relaxed) == gen && !stop.load(Ordering::Relaxed) {
                    let Some(p) = queue.lock().unwrap().pop() else { return };
                    let k = script_kind(&p);
                    match k {
                        ScriptKind::Ai => {
                            n_ai.fetch_add(1, Ordering::Relaxed);
                        }
                        ScriptKind::Hand => {
                            n_hand.fetch_add(1, Ordering::Relaxed);
                        }
                        ScriptKind::Unknown => {}
                    }
                    kinds.lock().unwrap().insert(p, k);
                }
            });
        }
    }

    /// Begin fetching the typed link into the folder being browsed. The fetch
    /// itself runs on its own thread (`dl::start`), so the picker keeps drawing,
    /// keeps dancing, and can cancel.
    fn start_download(&mut self) {
        let Some(url) = self.url.take() else { return };
        let url = url.trim().to_string();
        if url.is_empty() {
            return;
        }
        if !crate::dl::is_url(&url) {
            self.dl_line = Some(t!("picker.dl.notalink").into());
            return;
        }
        // --dl-dir wins; otherwise the folder being browsed, which is the folder
        // the user chose by walking to it. The drive list is not a folder, so
        // there is nowhere for a video to go and saying so beats inventing one.
        let Some(dir) = self.dl_dir.clone().or_else(|| self.cur.clone()) else {
            self.dl_line = Some(t!("picker.dl.nofolder").into());
            return;
        };
        match crate::dl::start(&url, &dir, &self.dl_args) {
            Ok(j) => {
                self.dl_line = Some(t!("picker.dl.asking").into());
                self.job = Some(j);
            }
            Err(e) => self.dl_line = Some(format!("{e:#}")),
        }
    }

    /// Move a finished download into the listing: shown, selected, under the
    /// cursor. The filter goes, because a link just fetched is the most recent
    /// thing the user asked for and a narrowed listing could hide it.
    fn arrived(&mut self, path: PathBuf) {
        let name =
            path.file_name().unwrap_or_default().to_string_lossy().to_string();
        self.dl_line = Some(t!("picker.dl.fetched", name = name));
        self.clear_filter();
        self.refresh();
        self.selected.insert(path.clone());
        if let Some(i) = self
            .entries
            .iter()
            .position(|e| matches!(e, Entry::File(p) if *p == path))
        {
            self.list.select(Some(i));
        }
    }

    /// Take whatever the running fetch has to say. Called once per draw, and
    /// cheap when there is nothing running.
    fn pump_download(&mut self) {
        let Some(job) = self.job.as_mut() else { return };
        let name = job
            .dest()
            .map(|d| d.file_name().unwrap_or_default().to_string_lossy().to_string());
        let mut latest = None;
        let done = job.pump(&mut |p| latest = Some(p));
        if let Some(p) = latest {
            let what = name.clone().unwrap_or_else(|| t!("picker.dl.thevideo").into());
            let pct = format!("{:.0}", p.pct);
            self.dl_line = Some(match (p.size.as_str(), p.eta.as_str()) {
                ("", _) => t!("picker.dl.fetching", what = what, pct = pct),
                (size, "") => t!("picker.dl.fetching.size", what = what, pct = pct, size = size),
                (size, eta) => {
                    t!("picker.dl.fetching.eta", what = what, pct = pct, size = size, eta = eta)
                }
            });
        }
        match done {
            None => {}
            Some(Ok(p)) => {
                self.job = None;
                self.arrived(p);
            }
            Some(Err(e)) => {
                self.job = None;
                self.dl_line = Some(format!("{e:#}"));
            }
        }
    }

    /// Mark (or unmark) the video under the cursor as VR. The detector is
    /// recall-biased and still misses some, and the cost of a miss is a whole
    /// draft spent on a warped bubble -- so this says "open the aiming page for
    /// this one anyway".
    fn toggle_vr_at_cursor(&mut self) {
        let Some(Entry::File(p)) = self.list.selected().and_then(|i| self.entries.get(i)) else {
            return;
        };
        let p = p.clone();
        if !self.vr_marks.remove(&p) {
            self.vr_marks.insert(p);
        }
    }

    /// Select (or, if all already are, deselect) every video currently listed --
    /// respects the active filter, so `/finale` then `A` grabs just those.
    fn select_all_listed(&mut self) {
        let files: Vec<PathBuf> = self
            .entries
            .iter()
            .filter_map(|e| match e {
                Entry::File(f) => Some(f.clone()),
                _ => None,
            })
            .collect();
        if files.is_empty() {
            return;
        }
        let all_selected = files.iter().all(|f| self.selected.contains(f));
        for f in files {
            if all_selected {
                self.selected.remove(&f);
            } else {
                self.selected.insert(f);
            }
        }
    }

    /// A filter belongs to one listing; moving to another directory drops it.
    fn clear_filter(&mut self) {
        self.filter.clear();
        self.filtering = false;
    }

    fn go_up(&mut self) {
        self.cur = match self.cur.take() {
            Some(d) => d.parent().map(Path::to_path_buf),
            None => None,
        };
        self.clear_filter();
        self.refresh();
    }

    fn enter(&mut self, dir: PathBuf) {
        self.cur = Some(dir);
        self.clear_filter();
        self.refresh();
    }

    fn toggle(&mut self, file: PathBuf) {
        if !self.selected.remove(&file) {
            self.selected.insert(file);
        }
    }

    /// How many of a folder's OWN videos are in the selection -- what the folder
    /// row draws. A `range` rather than a scan of the whole selection: paths
    /// compare component by component, so a folder's contents sort together and
    /// this costs a seek plus one step per video it actually holds.
    fn n_selected_in(&self, dir: &Path) -> usize {
        self.selected
            .range(dir.to_path_buf()..)
            .take_while(|p| p.starts_with(dir))
            .filter(|p| p.parent() == Some(dir))
            .count()
    }

    /// Everything picked from anywhere below `dir`, which is what THIS
    /// listing accounts for: a folder toggled from its parent puts its videos
    /// one level down, and the folder's own row is showing their tally.
    fn n_selected_under(&self, dir: &Path) -> usize {
        self.selected.range(dir.to_path_buf()..).take_while(|p| p.starts_with(dir)).count()
    }

    /// Space on a folder: take every video it holds directly, or give them all
    /// back when they are already taken -- the same toggle A does over a
    /// listing, aimed at a folder the cursor never has to enter.
    ///
    /// One level, never deeper (`videos_in`). Folders nest without limit and a
    /// recursive grab reads exactly like this one right up until the batch
    /// starts, so the picker does not offer the choice.
    fn toggle_folder(&mut self, dir: PathBuf) {
        let vids = videos_in(&dir);
        if vids.is_empty() {
            self.status = Some(t!("picker.folder.empty").into());
            return;
        }
        let all = vids.iter().all(|v| self.selected.contains(v));
        for v in vids {
            if all {
                self.selected.remove(&v);
            } else {
                self.selected.insert(v);
            }
        }
    }

    /// Enter on whatever the cursor is on.
    fn activate(&mut self) {
        enum Act {
            Up,
            Dir(PathBuf),
            File(PathBuf),
        }
        let act = match self.list.selected().and_then(|i| self.entries.get(i)) {
            Some(Entry::Up) => Act::Up,
            Some(Entry::Dir(d)) => Act::Dir(d.clone()),
            Some(Entry::File(f)) => Act::File(f.clone()),
            None => return,
        };
        match act {
            Act::Up => self.go_up(),
            Act::Dir(d) => self.enter(d),
            Act::File(f) => self.toggle(f),
        }
    }

    /// Space: take what the cursor is on. A video is itself; a folder is the
    /// videos inside it, which is how a drop of clips is started from the row
    /// above it instead of a trip in and out.
    fn toggle_at_cursor(&mut self) {
        match self.list.selected().and_then(|i| self.entries.get(i)) {
            Some(Entry::File(f)) => {
                let f = f.clone();
                self.toggle(f);
            }
            Some(Entry::Dir(d)) => {
                let d = d.clone();
                self.toggle_folder(d);
            }
            _ => {}
        }
    }

    /// Re-list the current directory without moving the cursor (R / F5). The
    /// folder changes under the picker all the time -- a download lands, a
    /// player writes a script beside a video -- and a refresh that jumped back
    /// to the top would cost the user their place in a long listing.
    fn reload(&mut self) {
        let (keep, top) = (self.list.selected(), self.offset);
        self.refresh();
        if let (Some(i), false) = (keep, self.entries.is_empty()) {
            self.list.select(Some(i.min(self.entries.len() - 1)));
            self.offset = top.min(self.entries.len() - 1);
        }
    }

    /// A path explorer.exe will actually accept: absolute, because it resolves a
    /// relative one against the ALREADY-RUNNING shell's directory rather than
    /// ours, and backslash-separated, because a forward slash ends the path as
    /// far as it is concerned (a remembered `last_dir` can be either). Not
    /// `canonicalize()`: that yields a `\\?\` prefix explorer cannot parse at
    /// all. Wide throughout, so a name that is not valid UTF-8 survives.
    #[cfg(windows)]
    fn explorer_path(p: &Path) -> std::ffi::OsString {
        use std::os::windows::ffi::{OsStrExt, OsStringExt};
        let abs = if p.is_absolute() {
            p.to_path_buf()
        } else {
            std::env::current_dir().map_or_else(|_| p.to_path_buf(), |c| c.join(p))
        };
        let wide: Vec<u16> = abs
            .as_os_str()
            .encode_wide()
            .map(|c| if c == b'/' as u16 { b'\\' as u16 } else { c })
            .collect();
        std::ffi::OsString::from_wide(&wide)
    }

    /// Open the system file browser on the current listing, with the entry
    /// under the cursor selected when there is one -- the escape hatch for
    /// everything the picker deliberately cannot do (rename, delete, look at a
    /// non-video file). Best effort: the picker stays up either way, and a
    /// failed spawn is not worth an error line over.
    fn open_in_explorer(&self) {
        let Some(dir) = self.cur.clone() else { return };
        let here = match self.list.selected().and_then(|i| self.entries.get(i)) {
            Some(Entry::Dir(p)) | Some(Entry::File(p)) => Some(p.clone()),
            _ => None,
        };
        Self::reveal(&dir, here.as_deref());
    }

    /// The failure log, in the system file browser -- the same escape hatch E
    /// gives the listing, aimed at the file instead. The path is on screen
    /// beside it, so this is a convenience rather than the only way in.
    fn open_log(&self) {
        let Some(log) = self.log.clone() else { return };
        let Some(dir) = log.parent().map(Path::to_path_buf) else { return };
        Self::reveal(&dir, Some(&log));
    }

    /// Open a folder, with one entry in it selected where there is one. Best
    /// effort: the picker stays up either way, and a failed spawn is not worth
    /// an error line over.
    fn reveal(dir: &Path, here: Option<&Path>) {
        #[cfg(windows)]
        let r = {
            use std::os::windows::process::CommandExt;
            // explorer.exe parses its own command line, and `/select,` plus the
            // path is ONE token whose path carries its own quotes. The normal
            // argument escaping quotes the WHOLE token as soon as the path holds
            // a space ("/select,D:\My Videos\a.mp4"), which explorer answers by
            // opening the default folder instead -- so build the line by hand.
            let raw = match here {
                Some(p) => {
                    let mut s = std::ffi::OsString::from("/select,\"");
                    s.push(Self::explorer_path(p));
                    s.push("\"");
                    s
                }
                None => {
                    let mut s = std::ffi::OsString::from("\"");
                    s.push(Self::explorer_path(dir));
                    s.push("\"");
                    s
                }
            };
            std::process::Command::new("explorer.exe").raw_arg(raw).spawn()
        };
        #[cfg(not(windows))]
        let r = {
            let _ = here; // no portable "open the folder AND select this" verb
            std::process::Command::new("xdg-open").arg(dir).spawn()
        };
        let _ = r;
    }

    fn move_by(&mut self, delta: i64) {
        if self.entries.is_empty() {
            return;
        }
        let i = self.list.selected().unwrap_or(0) as i64 + delta;
        self.list
            .select(Some(i.clamp(0, self.entries.len() as i64 - 1) as usize));
    }
}

/// The goblin mascot + wordmark, four lines, shown atop both screens. The
/// double rule under it is the top edge of the screen's frame, so the header
/// and the panel below it read as one enclosure -- and the goblin stands ON
/// that rule, which is the floor he idles and dances on.
fn goblin_logo(t: &Theme, width: u16) -> Vec<Line<'static>> {
    // His pose and the parade behind him are one question: whether he is
    // watching depends on whether there is a file crossing to watch.
    logo_rows(t, width, crate::sound::music_on(), mascot::scene_now(width))
}

/// The four rows for a GIVEN scene, which is the seam the tests drive a parade
/// through -- one turns up every few minutes, and a header that could only be
/// rendered by waiting for it is one nothing holds a check against.
fn logo_rows(
    t: &Theme,
    width: u16,
    dancing: bool,
    (pose, par): (mascot::Pose, Option<mascot::Parade>),
) -> Vec<Line<'static>> {
    // Bold while he dances, exactly as the band brightens when it plays: the
    // mood reads across the header before any single frame does. Never dimmed,
    // though -- he is the wordmark's own goblin, and a greyed-out logo would
    // read as a disabled app rather than as quiet.
    let g = if dancing {
        Style::new().fg(rat(t.logo)).add_modifier(Modifier::BOLD)
    } else {
        Style::new().fg(rat(t.logo))
    };
    let gb = Style::new().fg(rat(t.accent)).add_modifier(Modifier::BOLD);
    let tag = Style::new().fg(rat(t.accent));
    let ver = Style::new().fg(rat(t.muted));
    let rule = Style::new().fg(rat(t.muted));
    // Padded through the same helper the startup report uses, so whatever he is
    // doing he occupies one block and the wordmark keeps its column.
    let art = |i: usize| Span::styled(crate::theme::art_line(pose[i], 2), g);
    // Four spaces after the padded art on BOTH text rows: the wordmark and the
    // tagline start in one column, and it is the same column the startup
    // report puts its text in.
    let mut l1 = vec![
        art(1),
        Span::styled("    GoblinScript", gb),
        Span::styled(concat!(" v", env!("CARGO_PKG_VERSION")), ver),
    ];
    let mut l2 = vec![art(2), Span::styled(format!("    {}", t!("app.tagline")), tag)];
    let mut l3 = vec![art(3)];
    // The parade, when one is crossing: the same file the processing header
    // draws, off the same beat, laid in with ratatui's primitives instead of
    // the console's. Only the painting differs -- where the marchers ARE is
    // `mascot`'s answer for both screens, so the two cannot drift.
    let text_col = crate::theme::MASCOT_W + 2;
    match par {
        Some(p) => {
            let span = |(s, ink): &(String, u8)| Span::styled(s.clone(), Style::new().fg(rat(*ink)));
            let head = (p.left as usize).saturating_sub(text_col);
            let corridor = (p.right - p.left) as usize;
            let tail = (width as usize).saturating_sub(text_col + head + corridor);
            l3.push(Span::styled("\u{2550}".repeat(head), rule));
            l3.extend(p.rows[mascot::MARCH_H - 1].iter().map(span));
            l3.push(Span::styled("\u{2550}".repeat(tail), rule));
            for (line, row) in [(&mut l1, &p.rows[0]), (&mut l2, &p.rows[1])] {
                let at: usize = line.iter().map(|s| s.content.chars().count()).sum();
                line.push(Span::raw(" ".repeat((p.left as usize).saturating_sub(at))));
                line.extend(row.iter().map(span));
            }
        }
        None => l3.push(Span::styled(
            "\u{2550}".repeat(width.saturating_sub(text_col as u16) as usize),
            rule,
        )),
    }
    vec![Line::from(art(0)), Line::from(l1), Line::from(l2), Line::from(l3)]
}

// The header's goblins -- which pose the mascot strikes, which frame the band
// is on, and the beat both of them keep -- live in `mascot`, because the
// startup report and the processing header draw the same fellow doing the same
// thing at the same moment. What stays here is only how ratatui puts him on the
// screen.

/// Draw the corner band flush right in `area` (the four-row header block).
/// Three rows tall against a four-row header, so the deck and the drum land ON
/// the rule -- the band plays along the top edge of the frame.
fn draw_corner_goblin(f: &mut Frame, area: Rect, t: &Theme) {
    if area.width < SOLO_MIN_W || area.height < GOBLIN_H {
        return;
    }
    let full = area.width >= BAND_MIN_W;
    let playing = crate::sound::music_on();
    let tick = mascot::beat_tick();
    let rows = mascot::band_rows(full, playing, tick);

    // Bright and brand-coloured while they play, dropped to the secondary ink
    // when they do not -- the mood reads before the drawing does.
    let style = if playing {
        Style::new().fg(rat(t.logo)).add_modifier(Modifier::BOLD)
    } else {
        Style::new().fg(rat(t.muted))
    };
    let width = if full { BAND_W } else { BLOCK_W };
    let rect = Rect {
        x: area.right() - width - 1,
        y: area.y,
        width,
        height: GOBLIN_H,
    };
    let lines: Vec<Line> = rows.into_iter().map(|r| Line::styled(r, style)).collect();
    f.render_widget(Paragraph::new(lines), rect);

    // The sound itself, radiating off the band into the header. Only while
    // something is actually playing -- silence that draws waves is the one way
    // this corner could lie about the audio layer.
    if playing && full && area.width >= WAVE_MIN_W {
        draw_sound_waves(f, rect.x, width, area.y, tick, t);
    }
}

/// The columns of sound either side of the band, blinking out and in together.
///
/// Rendered on the quiet beat too, as blanks, so the rays are cleared by this
/// function rather than by whatever else happens to paint the header.
fn draw_sound_waves(f: &mut Frame, band_x: u16, band_w: u16, y: u16, tick: usize, t: &Theme) {
    let style = Style::new().fg(rat(t.accent));
    let out = mascot::wave_out(tick);
    // saturating: the caller's width guard already keeps the band well clear of
    // the left edge, but a draw function is the last place to risk an underflow
    for (x, right) in [
        (band_x.saturating_sub(WAVE_W), false),
        (band_x + band_w, true),
    ] {
        let lines: Vec<Line> = mascot::wave_rows(out, right)
            .into_iter()
            .map(|r| Line::styled(r, style))
            .collect();
        let rect = Rect { x, y, width: WAVE_W, height: GOBLIN_H };
        f.render_widget(Paragraph::new(lines), rect);
    }
}

/// The status strip that closes every screen: which palette is on, what the
/// goblins are making noise with, and the keys that change any of it. Drawn in
/// inverse video so it reads as machine chrome rather than content.
///
/// The volume and skip controls only appear while music is actually playing --
/// a key that would do nothing is worse than no key at all.
fn chrome_strip(t: &Theme) -> Line<'static> {
    let chip = Style::new().fg(Color::Black).bg(rat(t.chrome_bg));
    let val = Style::new().fg(rat(t.accent)).add_modifier(Modifier::BOLD);
    let lbl = Style::new().fg(rat(t.muted));
    let mut spans = vec![
        Span::styled(" T ", chip),
        Span::styled(t!("picker.chrome.theme"), lbl),
        Span::styled(crate::theme::active().label(), val),
        Span::styled("   ", lbl),
        Span::styled(" M ", chip),
        Span::styled(t!("picker.chrome.audio"), lbl),
        Span::styled(crate::sound::audio_mode().label(), val),
    ];
    if crate::sound::music_on() {
        spans.extend([
            Span::styled("   ", lbl),
            Span::styled(" V ", chip),
            Span::styled(t!("picker.chrome.volume"), lbl),
            Span::styled(crate::sound::volume().label(), val),
            Span::styled("   ", lbl),
            Span::styled(" N ", chip),
            Span::styled(t!("picker.chrome.next"), lbl),
        ]);
    }
    // The language the goblins are speaking, and the key that changes it. Shown
    // only where there is more than one catalog to move between -- the strip
    // says what a key would do, and a key that does nothing says nothing.
    if crate::lang::available().len() > 1 {
        spans.extend([
            Span::styled("   ", lbl),
            Span::styled(" G ", chip),
            Span::styled(" ", lbl),
            Span::styled(crate::lang::current_name(), val),
        ]);
    }
    Line::from(spans)
}

// ===========================================================================
// Startup demo: a demoscene-style flourish shown once when the app is opened by
// double-click. A full-screen colour-cycling plasma, the wordmark assembled
// letter-by-letter in a big block font with a rainbow sweep that settles to
// goblin green, and a scrolling greetz marquee. Any key skips it.
// ===========================================================================

/// The 12 letters of GOBLINSCRIPT in a 5x5 block font (`#` = a lit pixel), laid
/// out left to right with a one-column gap, so the whole word is 71 columns.
const GLYPH_H: i32 = 5;
const GLYPH_W: i32 = 5;
const WORD: &str = "GOBLINSCRIPT";
const WORD_COLS: i32 = 12 * (GLYPH_W + 1) - 1; // 71

fn glyph(c: char) -> [&'static str; 5] {
    match c {
        'G' => [".####", "#....", "#..##", "#...#", ".####"],
        'O' => [".###.", "#...#", "#...#", "#...#", ".###."],
        'B' => ["####.", "#...#", "####.", "#...#", "####."],
        'L' => ["#....", "#....", "#....", "#....", "#####"],
        'I' => ["#####", "..#..", "..#..", "..#..", "#####"],
        'N' => ["#...#", "##..#", "#.#.#", "#..##", "#...#"],
        'S' => [".####", "#....", ".###.", "....#", "####."],
        'C' => [".####", "#....", "#....", "#....", ".####"],
        'R' => ["####.", "#...#", "####.", "#.#..", "#..##"],
        'P' => ["####.", "#...#", "####.", "#....", "#...."],
        'T' => ["#####", "..#..", "..#..", "..#..", "..#.."],
        _ => ["     ", "     ", "     ", "     ", "     "],
    }
}

/// Classic four-sine plasma. `y` is doubled so the field stays roughly circular
/// despite cells being about twice as tall as wide. Range ~[-4, 4].
fn plasma(x: f32, y: f32, t: f32) -> f32 {
    let y = y * 2.0;
    let a = (x * 0.12 + t * 1.4).sin();
    let b = (y * 0.11 + t * 1.1).sin();
    let c = ((x + y) * 0.08 + t * 0.9).sin();
    let (dx, dy) = (x - 60.0, y - 22.0);
    let d = ((dx * dx + dy * dy).sqrt() * 0.09 - t * 1.8).sin();
    a + b + c + d
}

fn hsv(h: f32, s: f32, v: f32) -> (f32, f32, f32) {
    let h = h.rem_euclid(360.0) / 60.0;
    let i = h.floor() as i32;
    let f = h - i as f32;
    let (p, q, u) = (v * (1.0 - s), v * (1.0 - s * f), v * (1.0 - s * (1.0 - f)));
    match i.rem_euclid(6) {
        0 => (v, u, p),
        1 => (q, v, p),
        2 => (p, v, u),
        3 => (p, q, v),
        4 => (u, p, v),
        _ => (v, p, q),
    }
}

fn lerp3(a: (f32, f32, f32), b: (f32, f32, f32), k: f32) -> (f32, f32, f32) {
    (a.0 + (b.0 - a.0) * k, a.1 + (b.1 - a.1) * k, a.2 + (b.2 - a.2) * k)
}

fn rgb(c: (f32, f32, f32)) -> Color {
    let ch = |v: f32| (v.clamp(0.0, 1.0) * 255.0) as u8;
    Color::Rgb(ch(c.0), ch(c.1), ch(c.2))
}

/// The wordmark's colour for a letter at horizontal fraction `cf`: a sweep
/// across the palette's hue band for the first stretch, then a settle onto its
/// brand colour. `land` fades each letter up as it drops in; `flash` is the
/// white pop as it lands; `dim` is the global fade-out at the end.
///
/// The sweep is expressed as FRACTIONS of the band, not absolute degrees, so a
/// palette narrows the rainbow rather than losing it: phosphor sweeps all 360,
/// amber shimmers across 45, and mono (a zero-width band at zero saturation)
/// resolves to a clean greyscale ramp.
fn letter_color(t: f32, cf: f32, land: f32, flash: f32, dim: f32) -> Color {
    const SWEEP_END: f32 = 2.4;
    let th = theme();
    let sweep = hsv(th.hue_base + (cf * 0.833 - t * 0.555) * th.hue_span, th.sat, 1.0);
    let mut c = if t < SWEEP_END {
        sweep
    } else {
        lerp3(sweep, th.brand, ((t - SWEEP_END) / 0.9).clamp(0.0, 1.0))
    };
    let b = (0.25 + 0.75 * land) * dim;
    c = (c.0 * b, c.1 * b, c.2 * b);
    if flash > 0.0 {
        c = lerp3(c, (dim, dim, dim), flash * 0.85);
    }
    rgb(c)
}

/// One frame of the intro at elapsed time `t` (seconds). Paints straight into
/// the cell buffer: plasma everywhere, the block wordmark over it, a marquee.
fn draw_intro(f: &mut Frame, t: f32) {
    let area = f.area();
    let (w, h) = (area.width, area.height);
    if w == 0 || h == 0 {
        return;
    }
    let buf = f.buffer_mut();

    // The tail is a hold on the settled wordmark, then a fade to black, so the
    // hand-off to the picker resolves instead of cutting. `dim` scales every
    // layer: 1.0 until the hold ends, ramping to 0 across the fade.
    const HOLD_END: f32 = 3.9;
    const FADE_DUR: f32 = 0.7;
    let dim = (1.0 - (t - HOLD_END) / FADE_DUR).clamp(0.0, 1.0);
    let edge = rgb((0.03 * dim, 0.04 * dim, 0.03 * dim)); // crisp letter/marquee backing

    // full-screen plasma background, confined to the palette's hue band
    let th = theme();
    for y in 0..h {
        for x in 0..w {
            let v = plasma(x as f32, y as f32, t);
            let hue = th.hue_base + ((v + 4.0) / 8.0 + t * 0.153) * th.hue_span;
            let val = (0.30 + 0.26 * ((v * 1.3).sin() * 0.5 + 0.5)) * dim; // dim: wordmark reads + fade
            let cell = &mut buf[(x, y)];
            cell.set_char(' ');
            cell.set_bg(rgb(hsv(hue, th.sat, val)));
        }
    }

    // wordmark, scaled to the biggest block font that fits (else a plain line)
    let scale = (((w as i32 - 2) / WORD_COLS).min((h as i32 - 4) / GLYPH_H)).max(0);
    if scale >= 1 {
        let (word_cols, word_rows) = (WORD_COLS * scale, GLYPH_H * scale);
        let ox = (w as i32 - word_cols) / 2;
        let oy = (h as i32 - word_rows) / 2 - 1;
        for (i, c) in WORD.chars().enumerate() {
            let local = t - i as f32 * 0.055; // staggered entrance
            if local < 0.0 {
                continue;
            }
            let land = (local / 0.28).clamp(0.0, 1.0);
            let ease = 1.0 - (1.0 - land) * (1.0 - land); // ease-out
            let drop = ((1.0 - ease) * 6.0 * scale as f32).round() as i32; // falls from above
            let flash = (1.0 - (local - 0.28) / 0.18).clamp(0.0, 1.0) * (local > 0.28) as i32 as f32;
            let cf = i as f32 / (WORD.chars().count() - 1) as f32;
            let color = letter_color(t, cf, land, flash, dim);
            let lx = ox + i as i32 * (GLYPH_W + 1) * scale;
            let g = glyph(c);
            for (fy, row) in g.iter().enumerate() {
                for (fx, px) in row.chars().enumerate() {
                    if px != '#' {
                        continue;
                    }
                    for sy in 0..scale {
                        for sx in 0..scale {
                            let cx = lx + fx as i32 * scale + sx;
                            let cy = oy + fy as i32 * scale + sy - drop;
                            if cx < 0 || cy < 0 || cx >= w as i32 || cy >= h as i32 {
                                continue;
                            }
                            let cell = &mut buf[(cx as u16, cy as u16)];
                            cell.set_char('\u{2588}'); // full block
                            cell.set_fg(color);
                            cell.set_bg(edge); // crisp edge on any terminal
                        }
                    }
                }
            }
        }

        // He arrives under his own wordmark once it has landed, and dances the
        // chime out. Two blank rows below the version: the block needs air, or
        // his ears read as an ascender on the version string.
        draw_intro_goblin(buf, (w as i32, h as i32), t, oy + word_rows + 3, dim, edge);

        // The version, centred just under the wordmark, easing in once the
        // letters have settled -- the goblin signs its work with a build number.
        let ver: Vec<char> = concat!("v", env!("CARGO_PKG_VERSION")).chars().collect();
        let vin = ((t - 1.0) / 0.5).clamp(0.0, 1.0) * dim; // fade in after the drop
        if vin > 0.0 {
            let vx = ox + (word_cols - ver.len() as i32) / 2;
            let vy = oy + word_rows + 1;
            let vc = rgb((th.brand.0 * vin, th.brand.1 * vin, th.brand.2 * vin)); // brand, faded
            for (i, ch) in ver.iter().enumerate() {
                let cx = vx + i as i32;
                if cx < 0 || vy < 0 || cx >= w as i32 || vy >= h as i32 {
                    continue;
                }
                let cell = &mut buf[(cx as u16, vy as u16)];
                cell.set_char(*ch);
                cell.set_fg(vc);
                cell.set_bg(edge);
            }
        }
    } else {
        let sx = ((w as i32 - WORD.len() as i32) / 2).max(0) as u16;
        let sy = h / 2;
        for (i, ch) in WORD.chars().enumerate() {
            let cx = sx + i as u16;
            if cx >= w {
                break;
            }
            let cf = i as f32 / (WORD.chars().count() - 1) as f32;
            let cell = &mut buf[(cx, sy)];
            cell.set_char(ch);
            cell.set_fg(letter_color(t, cf, 1.0, 0.0, dim));
            cell.set_style(Style::new().add_modifier(Modifier::BOLD));
        }
        draw_intro_goblin(buf, (w as i32, h as i32), t, sy as i32 + 2, dim, edge);
    }

    // scrolling greetz marquee along the bottom row
    if h >= 6 {
        let msg = marquee(&format!(
            "  ***  {}  ***  GOBLINSCRIPT v{}  ***  {}  ",
            t!("app.tagline"),
            env!("CARGO_PKG_VERSION"),
            t!("picker.intro.anykey"),
        ));
        let n = msg.len() as i32;
        let off = (t * 20.0) as i32;
        let cy = h - 1;
        for x in 0..w as i32 {
            // one CELL per column: a wide glyph paints its own two, and the
            // column it swallowed is left for it rather than given a letter
            let Some(ch) = msg[(((x + off) % n + n) % n) as usize] else { continue };
            if ch == ' ' {
                continue;
            }
            let c = hsv(
                th.hue_base + (x as f32 * 0.0167 - t * 0.333) * th.hue_span,
                th.sat * 0.82,
                1.0,
            );
            let cell = &mut buf[(x as u16, cy)];
            cell.set_char(ch);
            cell.set_fg(rgb((c.0 * dim, c.1 * dim, c.2 * dim)));
            cell.set_bg(edge);
        }
    }
}

/// The mascot, dancing the intro out under his own wordmark, with `top` the
/// first of his four rows.
///
/// He dances rather than idles: the boot chime is playing over this, and the
/// one screen where the app is unambiguously showing off is no place for him to
/// stand about. The beat comes off the animation's own clock rather than the
/// shared one, so he starts the piece with it and every replay of the intro is
/// the same performance.
///
/// Drawn only where he fits WHOLE. A goblin with his legs off the bottom of the
/// screen is worse than no goblin, and the intro is the one screen with no
/// obligation to say anything.
fn draw_intro_goblin(
    buf: &mut ratatui::buffer::Buffer,
    (w, h): (i32, i32),
    t: f32,
    top: i32,
    dim: f32,
    edge: Color,
) {
    /// He waits for the wordmark to finish landing, then fades up over this
    /// long -- the letters are the entrance, and two entrances at once is one
    /// busy screen.
    const IN_AT: f32 = 1.6;
    const IN_DUR: f32 = 0.5;

    let fade = ((t - IN_AT) / IN_DUR).clamp(0.0, 1.0) * dim;
    if fade <= 0.0 {
        return;
    }
    // Never watching: the intro has no header to march across, and its clock
    // starts at zero, which is the one cycle a parade is never scheduled in.
    let rows = mascot::mascot_pose((t * 1000.0 / mascot::BEAT_MS as f32) as usize, true, false);
    // his own block, centred, and the marquee's row kept clear
    let ox = (w - crate::theme::MASCOT_W as i32) / 2;
    if ox < 0 || top < 0 || top + rows.len() as i32 >= h {
        return;
    }
    let th = theme();
    let c = rgb((th.brand.0 * fade, th.brand.1 * fade, th.brand.2 * fade));
    for (r, row) in rows.iter().enumerate() {
        for (i, ch) in row.chars().enumerate() {
            let (cx, cy) = (ox + i as i32, top + r as i32);
            if ch == ' ' || cx < 0 || cx >= w {
                continue;
            }
            let cell = &mut buf[(cx as u16, cy as u16)];
            cell.set_char(ch);
            cell.set_fg(c);
            cell.set_bg(edge); // the same crisp backing the letters get
        }
    }
}

#[cfg(test)]
mod intro_tests {
    use super::draw_intro;
    use ratatui::{backend::TestBackend, Terminal};

    // draw_intro writes into the cell buffer with hand-computed coordinates, so
    // the risk is an out-of-bounds index. Render every regime -- degenerate,
    // fallback-font, and huge -- across the whole timeline and assert no panic.
    #[test]
    fn no_panic_any_size_or_time() {
        let sizes = [(1u16, 1u16), (10, 3), (40, 10), (72, 5), (80, 24), (200, 60)];
        let times = [0.0f32, 0.1, 0.5, 1.0, 2.4, 3.0, 3.9, 4.3, 4.69];
        for (w, h) in sizes {
            let mut term = Terminal::new(TestBackend::new(w, h)).unwrap();
            for &t in &times {
                term.draw(|f| draw_intro(f, t)).unwrap();
            }
        }
    }
}

#[cfg(all(test, windows))]
mod explorer_tests {
    use super::*;

    // explorer.exe is unforgiving about all three of these, and answers a path
    // it dislikes by opening the default folder rather than failing -- so the
    // wrongness is silent, and only a test sees it.
    #[test]
    fn explorer_paths_are_absolute_and_backslashed() {
        let p = App::explorer_path(Path::new(r"D:/My Videos/a.mp4"));
        assert_eq!(p.to_string_lossy(), r"D:\My Videos\a.mp4");

        let cwd = std::env::current_dir().unwrap();
        let rel = App::explorer_path(Path::new("sub/clip.mp4"));
        assert_eq!(
            rel.to_string_lossy(),
            cwd.join(r"sub\clip.mp4").to_string_lossy()
        );

        // a `\\?\` prefix is what canonicalize() would add, and explorer cannot
        // parse it -- this path must reach it exactly as given
        let plain = App::explorer_path(Path::new(r"C:\vids"));
        assert_eq!(plain.to_string_lossy(), r"C:\vids");
    }
}

#[cfg(test)]
use crate::lang::speaking;

#[cfg(test)]
mod summary_tests {
    use super::*;

    #[test]
    fn summary_names_the_kinds_present_only() {
        let _lang = speaking("en-US");
        assert_eq!(scripted_summary(0, 40, 0, 0), None);
        assert_eq!(
            scripted_summary(12, 40, 9, 3).unwrap(),
            "(12/40 scripted: 9 ai, 3 hand)"
        );
        assert_eq!(scripted_summary(9, 40, 9, 0).unwrap(), "(9/40 scripted: 9 ai)");
        // whatever neither classified is the remainder, and says so
        assert_eq!(
            scripted_summary(5, 40, 3, 1).unwrap(),
            "(5/40 scripted: 3 ai, 1 hand, 1 ?)"
        );
    }
}

#[cfg(test)]
mod screen_tests {
    use super::*;
    use ratatui::{backend::TestBackend, Terminal};

    // The picker paints the parade with ratatui's spans while the processing
    // header paints the same file with the console's escapes, and only `mascot`
    // knows where the marchers are -- so what the two screens show cannot
    // drift. What CAN drift is this side's splicing: the rule is laid in three
    // pieces around the corridor here, and a piece measured wrong is a header
    // row that runs long once every few minutes.
    #[test]
    fn the_picker_paints_the_parade_into_its_rule() {
        crate::theme::set(crate::theme::Palette::Phosphor);
        let t = theme();
        let width = 140u16;
        let (tick, scene) = (0..100_000)
            .map(|k| (k, mascot::scene_at(width, k, true)))
            .find(|(_, (_, par))| par.is_some())
            .expect("no parade is ever scheduled");
        let rows = logo_rows(&t, width, true, scene);
        assert_eq!(rows.len(), 4);
        let plain = |l: &Line| -> String { l.spans.iter().map(|s| s.content.as_ref()).collect() };
        // every row lands inside the frame, parade or not
        for (i, l) in rows.iter().enumerate() {
            let w = plain(l).chars().count();
            assert!(w <= width as usize, "row {i} is {w} columns at tick {tick}");
        }
        // the marchers are IN the rule, with no blanks punched around them
        let rule = plain(&rows[3]);
        assert!(rule.contains("|_\\") || rule.contains("/_|"), "no legs in the rule: {rule:?}");
        assert!(
            rule.chars().skip(crate::theme::MASCOT_W + 2).all(|c| c != ' '),
            "the rule has holes punched in it: {rule:?}"
        );
        assert_eq!(rule.chars().count(), width as usize, "the rule row is short: {rule:?}");
        // and the rest of him is up beside the wordmark and the tagline
        assert!(plain(&rows[1]).contains("/\\,/\\"), "no ears beside the wordmark");
        assert!(plain(&rows[2]).contains("(o.o)"), "no faces beside the tagline");
    }

    // A marcher entering or leaving is drawn in the secondary ink, which is the
    // whole answer to the one thing a parade cannot do gracefully: a goblin is
    // only ever drawn WHOLE, so he arrives at the corridor's edge all at once,
    // and the step down in brightness is what turns that pop into an arrival.
    //
    // Checked here rather than against the processing header because `console`
    // emits no colour at all off a terminal, so the console rows are plain text
    // under test -- ratatui carries its styles as data either way.
    #[test]
    fn marchers_arriving_and_leaving_are_dimmed() {
        crate::theme::set(crate::theme::Palette::Phosphor);
        let t = theme();
        let width = 140u16;
        // a beat with somebody mid-corridor AND somebody at an edge
        let scene = (0..100_000)
            .map(|k| mascot::scene_at(width, k, true))
            .find(|(_, par)| {
                par.as_ref().is_some_and(|p| {
                    let inks: Vec<u8> = p.rows[1].iter().map(|(_, i)| *i).collect();
                    inks.contains(&t.logo) && inks.contains(&t.muted)
                })
            })
            .expect("no crossing ever has a marcher at an edge and one in the middle");
        let rows = logo_rows(&t, width, true, scene);
        let inks: Vec<Color> = rows[2].spans.iter().filter_map(|s| s.style.fg).collect();
        assert!(inks.contains(&rat(t.logo)), "nobody is drawn in the goblins' own ink");
        assert!(inks.contains(&rat(t.muted)), "nobody at the corridor's edge is dimmed");
    }

    // Both screens split the frame with a fixed-size `areas()` destructure,
    // which PANICS if the constraint count and the binding count ever disagree
    // -- and neither screen is reachable except by a human double-clicking the
    // exe, so nothing else would catch it. Render every palette at sizes from
    // degenerate to large.
    #[test]
    fn screens_render_at_any_size_and_palette() {
        let _lang = speaking("en-US");
        let sizes = [(1u16, 1u16), (20, 6), (80, 24), (200, 60)];
        for p in crate::theme::Palette::ALL {
            crate::theme::set(p);
            for (w, h) in sizes {
                let mut term = Terminal::new(TestBackend::new(w, h)).unwrap();
                let mut app =
                    App::new(false, false, false, None, Report::default(), None, Vec::new());
                term.draw(|f| draw(f, &mut app)).unwrap();
                // and again mid-filter, which adds the blinking cursor span
                app.filtering = true;
                app.filter.push('a');
                term.draw(|f| draw(f, &mut app)).unwrap();

                let items = [ReviewItem { name: "a.mp4".into(), actions: 12 }];
                let params = [style::Params::default()];
                term.draw(|f| draw_review(f, &items, &params, 0)).unwrap();
            }
        }
        crate::theme::set(crate::theme::Palette::Phosphor);
    }

    /// A failed batch used to be a red line the picker painted over on its way
    /// back up. The report is what replaced it, so the words that matter have
    /// to be ON it: which video, which stage, why, and where it was written
    /// down. Rendered at every size for the same `areas()` reason as above.
    #[test]
    fn a_failed_batch_opens_its_report_and_says_what_happened() {
        let _lang = speaking("en-US");
        crate::theme::set(crate::theme::Palette::Phosphor);
        let report = Report {
            status: Some("1 failed".into()),
            failures: vec![crate::errlog::Failure {
                what: "clip.mp4".into(),
                stage: Some("encode".into()),
                causes: vec!["the goblin fell over".into(), "out of memory".into()],
            }],
            log: Some(PathBuf::from("C:/goblins/goblinscript.log")),
        };
        let mut app = App::new(false, false, false, None, report, None, Vec::new());
        assert!(app.errors, "a batch that failed has to say so without being asked");

        for (w, h) in [(1u16, 1u16), (20, 6), (80, 24), (200, 60)] {
            let mut term = Terminal::new(TestBackend::new(w, h)).unwrap();
            term.draw(|f| draw(f, &mut app)).unwrap();
            if w < 80 {
                continue; // narrow frames clip by design; the point there is not panicking
            }
            let buf = term.backend().buffer();
            let screen: String = (0..h)
                .map(|y| (0..w).map(|x| buf.cell((x, y)).unwrap().symbol()).collect::<String>())
                .collect::<Vec<_>>()
                .join("\n");
            for want in ["clip.mp4", "during encode", "the goblin fell over", "out of memory"] {
                assert!(screen.contains(want), "{w}x{h} report has no {want:?}:\n{screen}");
            }
            assert!(screen.contains("goblinscript.log"), "no path to the log:\n{screen}");
        }

        // any key that is not about reading it puts the listing back -- and Q
        // among them, which must not take the session with it
        assert!(on_key(&mut app, KeyEvent::from(KeyCode::Char('q'))).is_none());
        assert!(!app.errors, "Q left the report up");

        // ...and the listing underneath says both that a video failed and how
        // to read about it again -- the status line points at X, so X has to
        // be in the guide beside it
        let mut term = Terminal::new(TestBackend::new(80, 24)).unwrap();
        term.draw(|f| draw(f, &mut app)).unwrap();
        let buf = term.backend().buffer();
        let screen: String = (0..24)
            .map(|y| (0..80).map(|x| buf.cell((x, y)).unwrap().symbol()).collect::<String>())
            .collect::<Vec<_>>()
            .join("
");
        assert!(screen.contains("1 failed"), "the picker does not say a video failed:
{screen}");
        assert!(screen.contains("X what went wrong"), "X is not offered:
{screen}");

        // the report is still there to be read again, for as long as it is the
        // news: the batch that failed is not re-runnable from memory
        on_key(&mut app, KeyEvent::from(KeyCode::Char('x')));
        assert!(app.errors, "X did not bring the report back");
    }

    /// Clearing a filter is a RUN of backspaces, and the run does not stop at
    /// the empty box: the last stroke finds nothing to delete, and the one
    /// after it -- or the key repeat that is really one long press -- used to
    /// walk out of the folder. The listing the user was narrowing is gone, and
    /// nothing on screen said it would be. Left is what leaves a folder.
    #[test]
    fn backspacing_a_filter_away_stays_in_the_folder() {
        let _lang = speaking("en-US");
        let root = std::env::temp_dir().join("goblin_backspace_probe");
        let here = root.join("here");
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&here).unwrap();
        for n in ["clip_a.mp4", "clip_b.mp4", "other.mp4"] {
            std::fs::write(here.join(n), b"").unwrap();
        }

        let mut app = App::new(false, false, false, None, Report::default(), None, Vec::new());
        app.cur = Some(here.clone());
        app.refresh();

        on_key(&mut app, KeyEvent::from(KeyCode::Char('/')));
        for c in "clip".chars() {
            on_key(&mut app, KeyEvent::from(KeyCode::Char(c)));
        }
        assert_eq!(app.filter, "clip", "the filter never took the keystrokes");
        let files = |a: &App| a.entries.iter().filter(|e| matches!(e, Entry::File(_))).count();
        assert_eq!(files(&app), 2, "the filter did not narrow the listing");

        // Far more strokes than the filter is long: a held key does not stop
        // politely at the empty box.
        for _ in 0..10 {
            on_key(&mut app, KeyEvent::from(KeyCode::Backspace));
        }
        assert!(app.filter.is_empty(), "the filter survived its own backspaces");
        assert_eq!(app.cur.as_deref(), Some(here.as_path()), "backspace left the folder");

        // ...and the key that IS the way out still is.
        on_key(&mut app, KeyEvent::from(KeyCode::Left));
        assert_eq!(app.cur.as_deref(), Some(root.as_path()), "Left no longer goes up");

        let _ = std::fs::remove_dir_all(&root);
    }

    /// A bare count promises a batch the listing can show. Pick a video, walk
    /// into the next folder, and "2 selected" sits over a screen holding one of
    /// them -- while S starts both. The count says where they are.
    #[test]
    fn the_footer_counts_selections_this_folder_cannot_show() {
        let _lang = speaking("en-US");
        let root = std::env::temp_dir().join("goblin_selection_probe");
        let (a, b) = (root.join("a"), root.join("b"));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&a).unwrap();
        std::fs::create_dir_all(&b).unwrap();
        std::fs::write(a.join("one.mp4"), b"").unwrap();
        std::fs::write(b.join("two.mp4"), b"").unwrap();

        let mut app = App::new(false, false, false, None, Report::default(), None, Vec::new());
        app.selected.insert(a.join("one.mp4"));
        app.selected.insert(b.join("two.mp4"));

        // Standing in one of the two: the other pick is off screen, and the
        // footer is the only thing that can say so.
        app.cur = Some(a.clone());
        let here = app.n_selected_under(&a);
        assert_eq!(here, 1, "this folder's own pick was not counted");
        assert_eq!(
            selected_summary(app.selected.len(), here),
            "2 selected (1 not in this folder)"
        );

        // Standing above both: a folder toggled from its parent keeps its
        // videos one level down, and the folder's row is showing their tally.
        // Nothing is hidden, so nothing is announced.
        app.cur = Some(root.clone());
        let here = app.n_selected_under(&root);
        assert_eq!(here, 2, "picks one level down were called elsewhere");
        assert_eq!(selected_summary(app.selected.len(), here), "2 selected");

        let _ = std::fs::remove_dir_all(&root);
    }

    /// End asks for the end of the report, which is where the last failure is.
    /// Unbound, it fell through to the catch-all that dismisses the report --
    /// the one thing a scroll key must never do.
    #[test]
    fn end_scrolls_the_report_to_its_last_failure() {
        let _lang = speaking("en-US");
        let failures: Vec<crate::errlog::Failure> = (0..20)
            .map(|i| crate::errlog::Failure {
                what: format!("clip{i:02}.mp4"),
                stage: None,
                causes: vec!["ffprobe could not read the video".into()],
            })
            .collect();
        let mut app = App::new(
            false,
            false,
            false,
            None,
            Report { status: None, failures, log: None },
            None,
            Vec::new(),
        );
        let mut term = Terminal::new(TestBackend::new(80, 24)).unwrap();
        term.draw(|f| draw(f, &mut app)).unwrap();
        assert!(app.errors, "a failed batch did not open its report");

        on_key(&mut app, KeyEvent::from(KeyCode::End));
        assert!(app.errors, "End took the report off the screen");
        term.draw(|f| draw(f, &mut app)).unwrap();

        let buf = term.backend().buffer();
        let screen: String = (0..24)
            .map(|y| (0..80).map(|x| buf.cell((x, y)).unwrap().symbol()).collect::<String>())
            .collect::<Vec<_>>()
            .join("
");
        assert!(screen.contains("clip19.mp4"), "End did not reach the last failure:
{screen}");
    }

    /// A report longer than the screen is the case the scroll exists for, and
    /// the end of it is where the last failure is. Holding a key down must
    /// park on that end rather than scrolling the text off the top.
    #[test]
    fn a_long_report_scrolls_and_stops_at_its_end() {
        let _lang = speaking("en-US");
        let failures: Vec<crate::errlog::Failure> = (0..20)
            .map(|i| crate::errlog::Failure {
                what: format!("clip{i:02}.mp4"),
                stage: None,
                causes: vec!["ffprobe could not read the video".into()],
            })
            .collect();
        let mut app = App::new(
            false,
            false,
            false,
            None,
            Report { status: None, failures, log: None },
            None,
            Vec::new(),
        );
        let mut term = Terminal::new(TestBackend::new(80, 24)).unwrap();
        term.draw(|f| draw(f, &mut app)).unwrap();
        for _ in 0..200 {
            on_key(&mut app, KeyEvent::from(KeyCode::Down));
            term.draw(|f| draw(f, &mut app)).unwrap();
        }
        let buf = term.backend().buffer();
        let screen: String = (0..24)
            .map(|y| (0..80).map(|x| buf.cell((x, y)).unwrap().symbol()).collect::<String>())
            .collect::<Vec<_>>()
            .join("\n");
        assert!(screen.contains("clip19.mp4"), "the end of the report is off screen:\n{screen}");
        assert!(
            screen.contains("the log could not be written"),
            "a report with no log has to say so:\n{screen}"
        );
    }

    /// The report is the one screen made of prose rather than rows, so it is
    /// the one that wraps -- and a Windows path is a single word longer than
    /// most terminals are wide.
    #[test]
    fn wrapping_keeps_every_word_inside_the_width() {
        let long = "could not read \
C:/videos/a/very/deeply/nested/folder/with/an/extremely-long-file-name-indeed.mp4 twice";
        for width in [8usize, 20, 40, 80] {
            let rows = wrap(long, width);
            for r in &rows {
                assert!(cols(r) <= width, "{width}: {r:?} overruns");
            }
            let back: String = rows.join(" ");
            assert_eq!(
                back.split_whitespace().collect::<String>(),
                long.split_whitespace().collect::<String>(),
                "{width}: wrapping lost or invented text"
            );
        }
    }

    /// The report is the one screen made of sentences rather than rows, and
    /// reading it back is the only way to see whether it reads. Ignored by
    /// default and run by hand -- `cargo test preview_report -- --ignored
    /// --nocapture` -- like the loudness table in `sound`.
    #[test]
    #[ignore]
    fn preview_report() {
        let _lang = speaking("en-US");
        crate::theme::set(crate::theme::Palette::Phosphor);
        let report = Report {
            status: Some("2 failed".into()),
            failures: vec![
                crate::errlog::Failure {
                    what: r"D:\clips\a really long name for a holiday clip.mp4".into(),
                    stage: Some("encode".into()),
                    causes: vec![
                        "could not start ffmpeg".into(),
                        "The system cannot find the file specified. (os error 2)".into(),
                    ],
                },
                crate::errlog::Failure {
                    what: r"D:\clips\b.mkv".into(),
                    stage: None,
                    causes: vec!["ffprobe could not read the video".into()],
                },
            ],
            log: Some(PathBuf::from("C:/Program Files/goblinscript/goblinscript.log")),
        };
        let mut app = App::new(false, false, false, None, report, None, Vec::new());
        let mut term = Terminal::new(TestBackend::new(80, 26)).unwrap();
        term.draw(|f| draw(f, &mut app)).unwrap();
        let buf = term.backend().buffer();
        for y in 0..26 {
            let row: String = (0..80).map(|x| buf.cell((x, y)).unwrap().symbol()).collect();
            println!("|{}|", row.trim_end());
        }
    }

    /// A listing row is assembled by hand -- box, name, chips, and a duration
    /// padded out to the frame's right edge -- which makes it the one row that
    /// can overrun its width or split a multi-byte name. The drive list the
    /// size sweep above renders has no file rows in it at all.
    #[test]
    fn file_rows_hold_their_column_at_every_width() {
        let _lang = speaking("en-US");
        let dir = PathBuf::from("Q:/vids");
        // 92 characters, multi-byte at the front: too long for an 80-column
        // frame and comfortable in a 200-column one, so the same row proves
        // both that the name gives way and that it only gives way when it must
        let long_name = format!("ünïcödé_{}.mp4", "long_".repeat(16));
        let (short, hand, long) =
            (dir.join("short.mp4"), dir.join("hand.mp4"), dir.join(&long_name));
        for p in crate::theme::Palette::ALL {
            crate::theme::set(p);
            for (w, h) in [(1u16, 1u16), (12, 9), (40, 12), (80, 24), (200, 60)] {
                let mut term = Terminal::new(TestBackend::new(w, h)).unwrap();
                let mut app = App::new(false, false, false, None, Report::default(), None, Vec::new());
                app.cur = Some(dir.clone());
                app.entries = vec![
                    Entry::Up,
                    Entry::File(short.clone()),
                    Entry::File(hand.clone()),
                    Entry::File(long.clone()),
                ];
                // the classifier's answers, as they reach the draw: one hand,
                // one ai, and one video with no script at all
                {
                    let mut k = app.kinds.lock().unwrap();
                    k.insert(hand.with_extension("funscript"), ScriptKind::Hand);
                    k.insert(long.with_extension("funscript"), ScriptKind::Ai);
                }
                // one probe home at 1:42:07, one marked by hand, one still out
                app.probes
                    .lock()
                    .unwrap()
                    .insert(short.clone(), Probed { dur_ms: Some(6_127_000.0), vr: true });
                app.vr_marks.insert(hand.clone());
                app.list.select(Some(1));
                term.draw(|f| draw(f, &mut app)).unwrap();

                if w < 80 {
                    continue; // narrow frames clip by design; the point there is not panicking
                }
                let buf = term.backend().buffer();
                let row = |y: u16| -> String {
                    (0..w).map(|x| buf.cell((x, y)).unwrap().symbol()).collect()
                };
                // the list starts under the 4-row logo and the location line
                let (probed, marked, unprobed) = (row(6), row(7), row(8));
                assert!(probed.contains("short.mp4"), "row 6 is not the probed video: {probed:?}");
                // the column is the contract: every row's last written cell is
                // one short of the edge, whatever the name did to get there
                for r in [&probed, &marked, &unprobed] {
                    assert_eq!(
                        r.trim_end().chars().count(),
                        w as usize - 1,
                        "the duration column does not reach the edge: {r:?}"
                    );
                }
                assert!(probed.trim_end().ends_with("1:42:07"), "no duration: {probed:?}");
                assert!(probed.contains("[vr?]"), "detector guess missing: {probed:?}");
                assert!(marked.contains("[vr]"), "hand mark missing: {marked:?}");
                assert!(unprobed.trim_end().ends_with("--:--"), "unprobed row: {unprobed:?}");
                if w == 80 {
                    assert!(unprobed.contains('\u{2026}'), "long name not elided: {unprobed:?}");
                } else {
                    assert!(unprobed.contains(&long_name), "name cut early: {unprobed:?}");
                }
            }
        }
        crate::theme::set(crate::theme::Palette::Phosphor);
    }

    /// The intro banner scrolls one CELL per column. Indexed by character, a
    /// translated banner would put a two-cell glyph in one cell -- the text
    /// squeezed, and the rainbow that is computed per column drifting off the
    /// letters it colours.
    #[test]
    fn the_intro_banner_scrolls_by_the_column() {
        // one guard for the whole test: the lock is not reentrant, so taking a
        // second while this one is alive would wait on this thread forever
        let lang = speaking("zh-CN");
        let msg = format!("  ***  {}  ***  ", t!("app.tagline"));
        let cells = marquee(&msg);
        assert_eq!(cells.len(), cols(&msg), "one entry per cell");
        // every wide glyph owns the cell after it, and nothing else is blank
        for (i, c) in cells.iter().enumerate() {
            if c.is_none() {
                let prev = cells[i - 1].expect("a skipped cell follows a glyph");
                assert_eq!(cols(prev.encode_utf8(&mut [0u8; 4])), 2, "{prev:?} is not wide");
            }
        }
        // and the English banner, which has no wide glyph in it, is unchanged
        assert!(crate::lang::set("en-US"));
        let plain = format!("  ***  {}  ***  ", t!("app.tagline"));
        assert_eq!(marquee(&plain).len(), plain.chars().count());
        drop(lang);
    }

    /// A translated picker is laid out in COLUMNS, and a CJK glyph is two of
    /// them. Measured as characters -- which is what every one of these rows
    /// used to do -- a Chinese footer reads as half its real width, so it packs
    /// past the right edge and the key guide loses its tail; and the file rows'
    /// duration column, which is padded to land one short of the edge, lands
    /// somewhere else on every row that has a wide glyph in it.
    ///
    /// So this is the same contract as the English row test, in a language that
    /// can break it: nothing overruns, and the column is still a column.
    #[test]
    fn a_translated_picker_still_holds_its_columns() {
        let _lang = speaking("zh-CN");
        let dir = PathBuf::from("Q:/vids");
        let (short, wide) = (dir.join("short.mp4"), dir.join("影片_很长的名字_测试.mp4"));
        for (w, h) in [(80u16, 24u16), (100, 30), (200, 60)] {
            let mut term = Terminal::new(TestBackend::new(w, h)).unwrap();
            let mut app = App::new(false, false, false, None, Report::default(), None, Vec::new());
            app.cur = Some(dir.clone());
            app.entries =
                vec![Entry::Up, Entry::File(short.clone()), Entry::File(wide.clone())];
            app.probes
                .lock()
                .unwrap()
                .insert(short.clone(), Probed { dur_ms: Some(6_127_000.0), vr: true });
            app.list.select(Some(1));
            term.draw(|f| draw(f, &mut app)).unwrap();

            let buf = term.backend().buffer();
            // One CELL per entry, which is one column: a wide glyph sits in the
            // first of its two cells and ratatui blanks the second, so counting
            // cells is what measures the drawn row. (Measuring the rebuilt
            // string instead counts a wide glyph as 2 and its blank as 1 more.)
            let rows: Vec<String> = (0..h)
                .map(|y| (0..w).map(|x| buf.cell((x, y)).unwrap().symbol()).collect())
                .collect();
            // The file rows still end their duration one short of the edge --
            // the padding is computed from the name's COLUMNS, and a name of
            // wide glyphs is the case that says whether it really is.
            for y in [6u16, 7, 8] {
                let r = &rows[y as usize];
                if r.trim().is_empty() {
                    continue;
                }
                assert_eq!(
                    r.trim_end().chars().count(),
                    w as usize - 1,
                    "the duration column does not reach the edge at width {w}: {r:?}"
                );
            }
            // And the key guide keeps its TAIL. A footer packed as if Chinese
            // were half its real width runs past the edge, where ratatui cuts
            // it -- and what it cuts is the end of the row, which is where
            // start and quit are.
            // Matched with the blanks taken out: ratatui puts a wide glyph in
            // one cell and blanks the next, so a word rebuilt from cells reads
            // as "哥 布 林" and no substring of it is the word itself.
            let screen = rows.join("\n").replace(' ', "");
            assert!(screen.contains("哥布林巢穴"), "the location line is not translated");
            for word in ["开始", "退出"] {
                assert!(screen.contains(word), "{word} was cut from the guide at width {w}");
            }
        }
        crate::lang::set("en-US");
    }

    /// A big folder is drawn through a WINDOW -- the rows on screen and no
    /// others -- so the cursor and the scroll offset are now two separate
    /// things that have to agree. They disagree by one at exactly the edges,
    /// which is where this walks: the top, the bottom, and the jumps between.
    /// A cursor the window has lost is a listing that scrolls without moving.
    #[test]
    fn the_cursor_stays_on_screen_in_a_folder_too_tall_to_draw() {
        let dir = PathBuf::from("Q:/vids");
        let (w, h) = (100u16, 30u16);
        let mut term = Terminal::new(TestBackend::new(w, h)).unwrap();
        let mut app = App::new(false, false, false, None, Report::default(), None, Vec::new());
        app.cur = Some(dir.clone());
        app.folders.clear(); // `App::new` listed the launch directory; this is a folder of its own
        app.videos = (0..5000)
            .map(|i| {
                let path = dir.join(format!("clip_{i:05}.mp4"));
                let key = path.file_name().unwrap().to_string_lossy().to_lowercase();
                Row { path, key, scripted: false }
            })
            .collect();
        app.refilter();
        assert_eq!(app.entries.len(), 5001, "the `..` row plus every video");

        // Home, End, and a page either side of the join: after every one the
        // selected row has to be a row the frame actually PAINTED. Reading it
        // off the screen is the contract itself, and needs no second copy of
        // the layout's constraints to compare against.
        for jump in [0i64, i64::MAX / 2, i64::MIN / 2, 20, -20, i64::MAX / 2, -1] {
            app.move_by(jump);
            term.draw(|f| draw(f, &mut app)).unwrap();
            let sel = app.list.selected().expect("a listing this long always has a cursor");
            assert!(
                sel >= app.offset,
                "the window starts past the cursor: offset {} > sel {sel}",
                app.offset
            );
            if sel == 0 {
                continue; // the `..` row, which carries no name to look for
            }
            let name = format!("clip_{:05}.mp4", sel - 1);
            let buf = term.backend().buffer();
            let screen: String = (0..h)
                .map(|y| (0..w).map(|x| buf.cell((x, y)).unwrap().symbol()).collect::<String>())
                .collect::<Vec<_>>()
                .join("\n");
            assert!(screen.contains(&name), "{name} is selected but was never drawn");
        }
    }

    /// What a big folder actually costs, printed rather than asserted: build a
    /// throwaway folder of videos and scripts, list it once, then filter it the
    /// way a keystroke does. `cargo test -- --ignored big_folder --nocapture`.
    ///
    /// The two numbers are the whole point of the split. Listing is one
    /// `read_dir`; filtering is a substring compare per row and must not show
    /// up at all -- it used to re-walk the folder AND re-open every script
    /// beside every video, on every character typed.
    #[test]
    #[ignore]
    fn big_folder_lists_once_and_filters_free() {
        let dir = std::env::temp_dir().join("goblin_big_folder_probe");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let body = format!(
            "{{\"actions\":[{}],\"metadata\":{{\"creator\":\"someone\"}}}}",
            (0..4000).map(|i| format!("{{\"at\":{},\"pos\":{}}}", i * 100, i % 100))
                .collect::<Vec<_>>()
                .join(",")
        );
        const N: usize = 4000;
        for i in 0..N {
            std::fs::write(dir.join(format!("clip_{i:05}.mp4")), b"").unwrap();
            if i % 2 == 0 {
                std::fs::write(dir.join(format!("clip_{i:05}.funscript")), &body).unwrap();
            }
        }

        let mut app = App::new(false, false, false, None, Report::default(), None, Vec::new());
        app.cur = Some(dir.clone());
        let t = Instant::now();
        app.refresh();
        let listed = t.elapsed();
        assert_eq!(app.n_videos, N, "every video listed");
        assert_eq!(app.n_scripted, N / 2, "and every script found, without a stat apiece");

        let t = Instant::now();
        for c in "clip_012".chars() {
            app.filter.push(c);
            app.refilter();
        }
        let filtered = t.elapsed();
        println!(
            "  {N} videos: list {:?}, then {} filter keystrokes in {:?} ({:?} each)",
            listed,
            "clip_012".len(),
            filtered,
            filtered / 8
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The filter narrows the listing already in memory and reads nothing --
    /// it used to re-walk the folder and re-open every script beside every
    /// video, per keystroke. A model with no folder behind it can only still
    /// filter if that is true, so the paths here point nowhere on purpose.
    #[test]
    fn filtering_narrows_the_listing_without_a_folder_to_read() {
        let dir = PathBuf::from("Q:/nowhere-at-all");
        let mut app = App::new(false, false, false, None, Report::default(), None, Vec::new());
        app.cur = Some(dir.clone());
        app.folders.clear(); // `App::new` listed the launch directory; this is a folder of its own
        for name in ["alpha.mp4", "beta.mp4", "gamma.mp4", "ALPHABET.mp4"] {
            let path = dir.join(name);
            let key = name.to_lowercase();
            app.videos.push(Row { path, key, scripted: false });
        }
        app.refilter();
        assert_eq!(app.entries.len(), 5, "`..` plus four videos");

        app.filter.push_str("alpha");
        app.refilter();
        // case-folded, substring, `..` still first
        assert_eq!(app.entries.len(), 3, "alpha.mp4 and ALPHABET.mp4");
        assert!(matches!(app.entries[0], Entry::Up));

        app.filter.push('z');
        app.refilter();
        assert_eq!(app.entries.len(), 1, "nothing matches but `..` is always reachable");
        assert!(app.list.selected().is_some(), "a listing with a row has a cursor on it");

        app.clear_filter();
        app.refilter();
        assert_eq!(app.entries.len(), 5, "clearing the filter restores the whole listing");
    }

    /// Space on a folder row takes the videos in it and stops there, so a drop
    /// of clips starts from the row above them. What the row then draws is the
    /// count of the folder's OWN videos -- the one number that stays honest
    /// when the selection also holds videos from somewhere deeper.
    #[test]
    fn a_folder_row_takes_one_level_and_says_how_many() {
        let _lang = speaking("en-US");
        crate::theme::set(crate::theme::Palette::Phosphor);
        let root = std::env::temp_dir().join("goblin_folder_pick");
        let (drop, deep, bare) = (root.join("drop"), root.join("drop/deeper"), root.join("bare"));
        let _ = std::fs::remove_dir_all(&root);
        std::fs::create_dir_all(&deep).unwrap();
        std::fs::create_dir_all(&bare).unwrap();
        for name in ["b.mkv", "a.mp4", "sleeve.jpg"] {
            std::fs::write(drop.join(name), b"").unwrap();
        }
        std::fs::write(deep.join("deeper.mp4"), b"").unwrap();

        let mut app =
            App::new(false, false, false, Some(root.clone()), Report::default(), None, Vec::new());
        assert!(matches!(app.entries[1], Entry::Dir(ref d) if *d == bare), "bare sorts first");
        app.list.select(Some(2)); // the drop
        app.toggle_at_cursor();
        assert_eq!(
            app.selected.iter().cloned().collect::<Vec<_>>(),
            vec![drop.join("a.mp4"), drop.join("b.mkv")],
            "the folder's own videos, and neither the jpg nor the one below it"
        );
        assert_eq!(app.n_selected_in(&drop), 2);

        // a video from deeper down belongs to ITS folder, not to the one above
        app.selected.insert(deep.join("deeper.mp4"));
        assert_eq!(app.n_selected_in(&drop), 2, "the deeper video is not the drop's");
        assert_eq!(app.n_selected_in(&deep), 1);
        app.selected.remove(&deep.join("deeper.mp4"));

        // the row says so, in the same right-hand column the videos use
        let mut term = Terminal::new(TestBackend::new(80, 24)).unwrap();
        term.draw(|f| draw(f, &mut app)).unwrap();
        let buf = term.backend().buffer();
        let row: String = (0..80).map(|x| buf.cell((x, 7)).unwrap().symbol()).collect();
        assert!(row.contains("[x] drop\\"), "the folder is not shown as taken: {row:?}");
        assert!(row.trim_end().ends_with("2 selected"), "no tally: {row:?}");

        // pressed again it gives them all back -- the same toggle A does
        app.toggle_at_cursor();
        assert!(app.selected.is_empty(), "a second press did not clear the folder");

        // and a folder with nothing to take says so instead of doing nothing
        app.list.select(Some(1));
        app.toggle_at_cursor();
        assert!(app.selected.is_empty());
        assert!(app.status.as_deref().is_some_and(|s| s.contains("no videos")), "{:?}", app.status);

        let _ = std::fs::remove_dir_all(&root);
    }

    /// The footers are the rows that overrun: the picker's key guide is ~150
    /// columns of keys, and a single-row footer cuts whatever does not fit --
    /// which is its TAIL, where start and quit are. So the keys are looked for
    /// on the SCREEN, not in the string that was handed to the renderer, at the
    /// 80 columns a default terminal opens with.
    #[test]
    fn every_key_reaches_the_screen_at_the_default_width() {
        let _lang = speaking("en-US");
        for p in crate::theme::Palette::ALL {
            crate::theme::set(p);
            for (w, h) in [(80u16, 24u16), (100, 30), (200, 60)] {
                let mut term = Terminal::new(TestBackend::new(w, h)).unwrap();
                let mut app = App::new(false, false, false, None, Report::default(), None, Vec::new());
                term.draw(|f| draw(f, &mut app)).unwrap();
                let buf = term.backend().buffer();
                let rows: Vec<String> = (0..h)
                    .map(|y| (0..w).map(|x| buf.cell((x, y)).unwrap().symbol()).collect())
                    .collect();
                let screen = rows.join("\n");
                for key in ["Space select", "B mark VR", "C auto-crop", "S start", "Q quit"] {
                    assert!(screen.contains(key), "{key:?} never drawn at {w}x{h}");
                }
                // the toggles are a footer too, and wrap by the same rule
                assert!(screen.contains("auto-crop: off"), "the crop toggle is cut at {w}x{h}");

                let items = [ReviewItem { name: "a.mp4".into(), actions: 12 }];
                let params = [style::Params::default()];
                term.draw(|f| draw_review(f, &items, &params, 0)).unwrap();
                let buf = term.backend().buffer();
                let screen: String = (0..h)
                    .map(|y| {
                        (0..w).map(|x| buf.cell((x, y)).unwrap().symbol()).collect::<String>()
                    })
                    .collect::<Vec<_>>()
                    .join("\n");
                for key in ["intensity: 1.0", "D/S/U/+- restyle it", "the goblins keep these"] {
                    assert!(screen.contains(key), "{key:?} never drawn at {w}x{h}");
                }
            }
        }
        crate::theme::set(crate::theme::Palette::Phosphor);
    }

    /// Packing is what makes the above true: items wrap between rows, never
    /// inside one, and no row is wider than the frame.
    #[test]
    fn packed_rows_wrap_between_items_and_stay_inside_the_width() {
        crate::theme::set(crate::theme::Palette::Phosphor);
        let t = theme();
        let entries: Vec<(&str, &str)> =
            (0..12).map(|_| ("Key", "does a thing")).collect(); // 16 columns each
        for w in [20u16, 40, 80, 200] {
            let rows = pack_lines(help_items(&t, &entries), "  ", w, FOOT_MAX_ROWS);
            let plain = |l: &Line| -> String { l.spans.iter().map(|s| s.content.as_ref()).collect() };
            for (i, l) in rows.iter().enumerate() {
                let n = cols(&plain(l));
                assert!(n <= w as usize, "row {i} is {n} columns at width {w}");
                // an item is never split: each row holds whole "Key does a thing"s
                assert_eq!(
                    plain(l).matches("Key does a thing").count(),
                    plain(l).matches("Key").count(),
                    "an item was cut in half at width {w}"
                );
            }
            assert!(rows.len() <= FOOT_MAX_ROWS, "footer took {} rows", rows.len());
            if w >= 80 {
                let drawn: usize =
                    rows.iter().map(|l| plain(l).matches("Key does a thing").count()).sum();
                assert_eq!(drawn, entries.len(), "items dropped at width {w}");
            }
        }
    }

    /// Every label in the Chinese picker is double-width, so a row counted by
    /// CHARACTER is half the row that draws. Wrapping, the inset and the
    /// balance step read one width function, and this is what says so.
    #[test]
    fn packed_rows_measure_double_width_labels_in_columns() {
        // The picker's own footer, in the language that makes it widest.
        let labels = ["上/下 移动", "已选 3 个", "回车 打开", "S 开始", "Q 退出"];
        let items: Vec<Vec<Span<'static>>> =
            labels.iter().map(|s| vec![Span::raw(s.to_string())]).collect();
        let plain = |l: &Line| -> String { l.spans.iter().map(|s| s.content.as_ref()).collect() };
        for w in [12u16, 20, 24, 30, 40, 45, 60, 80] {
            let rows = pack_lines(items.clone(), "  ", w, FOOT_MAX_ROWS);
            let mut seen: Vec<String> = Vec::new();
            for (i, l) in rows.iter().enumerate() {
                let p = plain(l);
                assert!(cols(&p) <= w as usize, "row {i} draws {} columns at width {w}", cols(&p));
                // The inset is blank cells, and there are exactly as many as
                // the packing arithmetic charged for.
                assert_eq!(
                    p.chars().take_while(|c| *c == ' ').count(),
                    FOOT_INSET,
                    "row {i} draws a different inset than it was measured at, width {w}"
                );
                for it in p.trim_start().split("  ") {
                    seen.push(it.to_string());
                }
            }
            // Items are whole, in order, and none is invented.
            assert!(
                labels.starts_with(&seen.iter().map(|s| s.as_str()).collect::<Vec<_>>()[..]),
                "items reordered or split at width {w}: {seen:?}"
            );
            if w >= 60 {
                assert_eq!(seen.len(), labels.len(), "items dropped at width {w}");
            }
            // The orphan rule: a lone item on the last row is allowed ONLY
            // when the item above it could not have come down to join it.
            if rows.len() >= 2 {
                let (last, prev) = (rows.len() - 1, rows.len() - 2);
                let n_last = plain(&rows[last]).trim_start().split("  ").count();
                let prev_items: Vec<String> =
                    plain(&rows[prev]).trim_start().split("  ").map(|s| s.to_string()).collect();
                if n_last == 1 && prev_items.len() > 1 {
                    let together = FOOT_INSET
                        + cols(prev_items.last().unwrap())
                        + 2
                        + cols(plain(&rows[last]).trim_start());
                    assert!(
                        together > w as usize,
                        "an orphan was left on the last row at width {w}: it fits in {together}"
                    );
                }
            }
        }
    }

    /// The balance step itself, pinned on the width where it fires: five
    /// Chinese labels wrap 4+1, and the fourth comes down to sit with the
    /// fifth rather than leaving it alone.
    #[test]
    fn a_lone_last_item_pulls_its_neighbour_down() {
        let labels = ["上/下 移动", "已选 3 个", "回车 打开", "S 开始", "Q 退出"];
        let items: Vec<Vec<Span<'static>>> =
            labels.iter().map(|s| vec![Span::raw(s.to_string())]).collect();
        let plain = |l: &Line| -> String { l.spans.iter().map(|s| s.content.as_ref()).collect() };
        let rows = pack_lines(items, "  ", 45, FOOT_MAX_ROWS);
        let drawn: Vec<String> = rows.iter().map(|l| plain(l)).collect();
        assert_eq!(
            drawn,
            vec![
                " 上/下 移动  已选 3 个  回车 打开".to_string(),
                " S 开始  Q 退出".to_string(),
            ],
            "the last row was left holding one item"
        );
    }
}

/// Play the startup demo until it finishes (~3.6 s) or the user presses a key.
/// Own alternate screen; restores the terminal on every exit path.
pub fn intro() -> Result<()> {
    // The chime plays for as long as this binding lives -- i.e. the animation.
    // Skipping the intro drops it and cuts the sound too, which is right.
    let _chime = crate::sound::play_boot();
    let mut term = ratatui::init();
    let r = intro_run(&mut term);
    ratatui::restore();
    r
}

fn intro_run(term: &mut ratatui::DefaultTerminal) -> Result<()> {
    let start = Instant::now();
    // assemble + sweep + settle (~3.3 s), a brief hold, then the fade to black
    let total = Duration::from_millis(4700);
    loop {
        let elapsed = start.elapsed();
        if elapsed >= total {
            return Ok(());
        }
        let t = elapsed.as_secs_f32();
        term.draw(|f| draw_intro(f, t))?;
        // any key press skips the rest; the ~28 ms poll doubles as the frame delay
        if event::poll(Duration::from_millis(28))? {
            if let Event::Key(k) = event::read()? {
                if k.kind == KeyEventKind::Press {
                    return Ok(());
                }
            }
        }
    }
}

fn onoff(t: &Theme, label: &str, key: &str, on: bool) -> Vec<Span<'static>> {
    vec![
        Span::styled(format!(" {key} "), Style::new().fg(Color::Black).bg(rat(t.chrome_bg))),
        Span::raw(format!(" {label}: ")),
        if on {
            Span::styled(t!("common.on"), Style::new().fg(rat(t.accent)).add_modifier(Modifier::BOLD))
        } else {
            Span::styled(t!("common.off"), Style::new().fg(rat(t.muted)))
        },
    ]
}

/// A footer row is only as wide as the terminal, and what runs past the right
/// edge is silently CUT -- which on a key guide is its tail, where start and
/// quit live. So the footers are built as items and packed here: an item that
/// does not fit wraps to the next row, and the caller sizes the row's area
/// from how many rows came back. `max_rows` is the floor under the list above:
/// a terminal too narrow for even that many rows loses the remainder, which by
/// then is a window no screen fits in.
/// The blank cells a packed row opens with. The wrap arithmetic, the balance
/// step and the drawn prefix all read THIS -- a row measured at one inset and
/// drawn at another is a row that fits on paper and overflows on screen.
const FOOT_INSET: usize = 1;

fn pack_lines(
    items: Vec<Vec<Span<'static>>>,
    sep: &str,
    width: u16,
    max_rows: usize,
) -> Vec<Line<'static>> {
    let w = width.max(8) as usize;
    let span_w = |s: &[Span<'static>]| s.iter().map(|x| cols(&x.content)).sum::<usize>();
    let sep_w = cols(sep);
    // Items stay whole and stay APART until wrapping is done: the balance step
    // below moves one between rows, which a flat run of spans cannot express.
    let (mut rows, mut cur, mut cw) = (Vec::<Vec<Vec<Span<'static>>>>::new(), Vec::new(), 0usize);
    for it in items {
        let n = span_w(&it);
        if cur.is_empty() {
            cw = FOOT_INSET + n;
        } else if cw + sep_w + n > w {
            rows.push(std::mem::take(&mut cur));
            cw = FOOT_INSET + n;
        } else {
            cw += sep_w + n;
        }
        cur.push(it);
    }
    if !cur.is_empty() {
        rows.push(cur);
    }
    rows.truncate(max_rows.max(1));

    // One short item alone under a full row reads as a mistake, and wrapping
    // cannot see it: a row is committed before the row after it exists. So the
    // last pair is balanced once packing is finished, and only where the two
    // still fit the frame together. This is what the double-width languages
    // hit -- a CJK key guide holds fewer items per row, so the remainder is
    // more often exactly one. Balancing AFTER the truncation is deliberate:
    // the other order would move an item into a row nobody sees.
    if rows.len() >= 2 {
        let last = rows.len() - 1;
        if rows[last].len() == 1 && rows[last - 1].len() > 1 {
            let moved = span_w(rows[last - 1].last().expect("row holds two or more"));
            if FOOT_INSET + moved + sep_w + span_w(&rows[last][0]) <= w {
                let it = rows[last - 1].pop().expect("row holds two or more");
                rows[last].insert(0, it);
            }
        }
    }

    rows.into_iter()
        .map(|row| {
            let mut spans = vec![Span::raw(" ".repeat(FOOT_INSET))];
            for (i, it) in row.into_iter().enumerate() {
                if i != 0 {
                    spans.push(Span::raw(sep.to_string()));
                }
                spans.extend(it);
            }
            Line::from(spans)
        })
        .collect()
}

/// How many rows a footer may take before the list starts paying for it.
const FOOT_MAX_ROWS: usize = 4;

/// The key guide: one item per key, so the packer only ever wraps BETWEEN
/// keys -- a key split across two rows would read as two keys.
fn help_items(t: &Theme, entries: &[(&str, &str)]) -> Vec<Vec<Span<'static>>> {
    entries
        .iter()
        .map(|(k, what)| vec![Span::styled(format!("{k} {what}"), Style::new().fg(rat(t.help)))])
        .collect()
}

/// The folder's script tally, or `None` when nothing here is scripted (the
/// picker then says nothing at all). Kinds that came back unreadable are the
/// remainder, and are only named when there are any.
fn scripted_summary(
    n_scripted: usize,
    n_videos: usize,
    n_ai: usize,
    n_hand: usize,
) -> Option<String> {
    if n_scripted == 0 {
        return None;
    }
    let mut by: Vec<String> = Vec::new();
    if n_ai > 0 {
        by.push(t!("picker.scripted.ai", n = n_ai));
    }
    if n_hand > 0 {
        by.push(t!("picker.scripted.hand", n = n_hand));
    }
    let unknown = n_scripted.saturating_sub(n_ai + n_hand);
    if unknown > 0 {
        by.push(t!("picker.scripted.unknown", n = unknown));
    }
    Some(t!("picker.scripted", done = n_scripted, total = n_videos, by = by.join(", ")))
}

/// The selection as the footer says it. A bare count is a promise the listing
/// cannot keep: pick five videos, walk into the next folder, and "5 selected"
/// sits over a screen holding none of them -- and S starts a batch the user
/// cannot see. So the ones this folder does not hold are counted out loud.
fn selected_summary(n_selected: usize, n_here: usize) -> String {
    let elsewhere = n_selected.saturating_sub(n_here);
    if elsewhere == 0 {
        t!("picker.selected", n = n_selected).to_string()
    } else {
        t!("picker.selected.elsewhere", n = n_selected, k = elsewhere).to_string()
    }
}

/// `1:42:07` past the hour, `18:33` under it, `--:--` while the probe is still
/// out (or came back empty). One shape per row, so the column stays a column.
fn dur_label(dur_ms: Option<f64>) -> String {
    let Some(ms) = dur_ms else {
        return t!("picker.duration.unknown").to_string();
    };
    let s = (ms / 1000.0).round().max(0.0) as u64;
    let (h, m, sec) = (s / 3600, (s % 3600) / 60, s % 60);
    if h > 0 {
        format!("{h}:{m:02}:{sec:02}")
    } else {
        format!("{m}:{sec:02}")
    }
}

/// The COLUMNS a string occupies, which is not the number of characters in it:
/// a CJK glyph takes two terminal cells, so a translated label measured by
/// `chars().count()` lays out at half its real width and every column after it
/// in the row lands somewhere else. `console` already measures this for the
/// processing log, which is what keeps the picker and the log agreeing.
fn cols(s: &str) -> usize {
    console::measure_text_width(s)
}

/// The intro banner as COLUMNS: one entry per terminal cell, `None` for the
/// second cell of a double-width glyph. The marquee scrolls by column, so a
/// translated banner indexed by character would squeeze its own text and let
/// the rainbow drift off the letters it is colouring.
fn marquee(msg: &str) -> Vec<Option<char>> {
    let mut out = Vec::with_capacity(msg.len());
    for ch in msg.chars() {
        out.push(Some(ch));
        for _ in 1..cols(ch.encode_utf8(&mut [0u8; 4])) {
            out.push(None);
        }
    }
    out
}

/// A style knob's value, read for display only. The label itself is the WIRE
/// value -- `Params::from_label` round-trips it and `flags_line` writes it onto
/// a command line -- so the translation happens here, at the last moment, out
/// of the value's own way. The review page shows these same words through the
/// same `page.preset.*` keys.
fn preset(label: &str) -> String {
    crate::lang::try_t(&format!("page.preset.{label}"))
        .unwrap_or(label)
        .to_string()
}

/// A name cut to `max` COLUMNS with an ellipsis. Characters are taken whole --
/// a byte slice would split a multi-byte name mid-character and panic, and a
/// column slice would leave half a double-width glyph in the row.
fn elide(name: &str, max: usize) -> String {
    if cols(name) <= max {
        return name.to_string();
    }
    match max {
        0 => String::new(),
        1 => "\u{2026}".to_string(),
        _ => {
            let mut out = String::new();
            let mut w = 0usize;
            let mut buf = [0u8; 4];
            for c in name.chars() {
                let cw = cols(c.encode_utf8(&mut buf));
                if w + cw > max - 1 {
                    break;
                }
                out.push(c);
                w += cw;
            }
            out.push('\u{2026}');
            out
        }
    }
}

/// Break a line of English to a width: on spaces where there are any, and
/// through the middle of a word where there are none -- a path is one long
/// word and still has to fit. Measured in COLUMNS, like every other width
/// here, so a translated line lays out where it is drawn.
fn wrap(s: &str, width: usize) -> Vec<String> {
    let width = width.max(8);
    let mut out: Vec<String> = Vec::new();
    let mut line = String::new();
    for word in s.split_whitespace() {
        if !line.is_empty() && cols(&line) + 1 + cols(word) > width {
            out.push(std::mem::take(&mut line));
        }
        if cols(word) > width {
            for ch in word.chars() {
                if cols(&line) + cols(ch.encode_utf8(&mut [0u8; 4])) > width {
                    out.push(std::mem::take(&mut line));
                }
                line.push(ch);
            }
            continue;
        }
        if !line.is_empty() {
            line.push(' ');
        }
        line.push_str(word);
    }
    if !line.is_empty() {
        out.push(line);
    }
    if out.is_empty() {
        out.push(String::new());
    }
    out
}

/// The failure report's body: every failure named, the stage it reached, then
/// the error and each cause under it -- the same words in the same order as
/// the log file's entry, because they are read off the same value.
fn failure_lines(fs: &[crate::errlog::Failure], width: usize) -> Vec<Line<'static>> {
    let t = theme();
    let mut out: Vec<Line<'static>> = Vec::new();
    let mut push = |prefix: &str, text: &str, style: Style| {
        let indent = " ".repeat(cols(prefix));
        for (i, part) in wrap(text, width.saturating_sub(cols(prefix))).into_iter().enumerate() {
            let head = if i == 0 { prefix } else { &indent };
            out.push(Line::styled(format!("{head}{part}"), style));
        }
    };
    for f in fs {
        push("  ", &f.what, Style::new().fg(rat(t.bad)).add_modifier(Modifier::BOLD));
        if let Some(st) = &f.stage {
            push("    ", &t!("picker.errors.during", stage = st), Style::new().fg(rat(t.muted)));
        }
        let mut causes = f.causes.iter();
        if let Some(first) = causes.next() {
            push("    ", first, Style::new().fg(rat(t.text)));
        }
        for c in causes {
            push("      ", &t!("picker.errors.because", cause = c), Style::new().fg(rat(t.muted)));
        }
        // one blank row between failures: three of them run together otherwise
        push("", "", Style::default());
    }
    out
}

/// The screen a failed batch opens on, and the one X brings back: what could
/// not be done, in as many lines as it takes, over the listing rather than
/// through it.
///
/// This exists because the console cannot do the job. The picker reopens into
/// the alternate screen as soon as a batch ends, so a red line printed on the
/// way out is on screen for a frame and then gone -- which reads as the app
/// having thrown the work away for no reason. The words survive here and in
/// the log; this is where they are read.
fn draw_errors(f: &mut Frame, app: &mut App) {
    let t = theme();
    let area = f.area();
    let help_rows = pack_lines(
        help_items(
            &t,
            &[
                (t!("key.updown"), t!("picker.errors.act.scroll")),
                ("E", t!("picker.errors.act.log")),
                (t!("key.enterq"), t!("picker.errors.act.close")),
            ],
        ),
        "   ",
        area.width,
        FOOT_MAX_ROWS,
    );

    // The one line that outlives the session: where to look tomorrow, or that
    // there is nothing to look at because the log could not be written. WRAPPED
    // rather than cut -- a path whose tail is missing is not a path.
    let foot = match &app.log {
        Some(p) => t!("picker.errors.log", path = p.display()),
        None => t!("picker.errors.nolog").to_string(),
    };
    let log_rows: Vec<String> = wrap(&foot, (area.width as usize).saturating_sub(2))
        .into_iter()
        .map(|l| format!("  {l}"))
        .collect();

    let [logo_a, title_a, body_a, log_a, chrome_a, help_a] = Layout::vertical([
        Constraint::Length(4),
        Constraint::Length(1),
        Constraint::Min(1),
        Constraint::Length(log_rows.len() as u16),
        Constraint::Length(1),
        Constraint::Length(help_rows.len() as u16),
    ])
    .areas(area);

    f.render_widget(Paragraph::new(goblin_logo(&t, area.width)), logo_a);
    draw_corner_goblin(f, logo_a, &t);

    f.render_widget(
        Line::styled(
            format!("  {}", t!("picker.errors.title", n = app.failures.len())),
            Style::new().fg(rat(t.bad)).add_modifier(Modifier::BOLD),
        ),
        title_a,
    );

    let lines = failure_lines(&app.failures, area.width as usize);
    let view_h = (body_a.height as usize).max(1);
    // Clamped where the height is known: the last screenful is the end of the
    // report, so holding a key down parks on it instead of scrolling into
    // blank rows below the last cause.
    app.err_off = app.err_off.min(lines.len().saturating_sub(view_h));
    let shown: Vec<Line> = lines.into_iter().skip(app.err_off).take(view_h).collect();
    f.render_widget(Paragraph::new(shown), body_a);

    f.render_widget(
        Paragraph::new(
            log_rows
                .into_iter()
                .map(|l| Line::styled(l, Style::new().fg(rat(t.help))))
                .collect::<Vec<_>>(),
        ),
        log_a,
    );

    f.render_widget(chrome_strip(&t), chrome_a);
    f.render_widget(Paragraph::new(help_rows), help_a);
}

fn draw(f: &mut Frame, app: &mut App) {
    // The report covers the picker whole rather than sharing it: an error is
    // paragraphs, and a listing is rows.
    if app.errors {
        draw_errors(f, app);
        return;
    }
    let t = theme();
    let area = f.area();

    // The two footers are packed to the terminal's width first: how many rows
    // they need is what the layout gives them, so nothing is cut off the right
    // edge on an 80-column window (the guide alone is ~150 columns of keys).
    let here = app.cur.as_deref().map_or(0, |d| app.n_selected_under(d));
    let mut opts: Vec<Vec<Span>> = vec![vec![Span::styled(
        selected_summary(app.selected.len(), here),
        if app.selected.is_empty() {
            Style::new().fg(rat(t.muted))
        } else {
            Style::new().fg(rat(t.accent)).add_modifier(Modifier::BOLD)
        },
    )]];
    opts.push(onoff(&t, t!("picker.toggle.overwrite"), "F", app.force));
    opts.push(onoff(&t, t!("picker.toggle.autocrop"), "C", app.autocrop));
    opts.push(onoff(&t, t!("picker.toggle.cropedit"), "K", app.crop_edit));
    let opt_rows = pack_lines(opts, "   ", area.width, FOOT_MAX_ROWS);

    // The link key is advertised only when yt-dlp is installed: an offer the app
    // cannot honour is worse than no offer, and the startup report says why.
    // The G key is offered only where there is somewhere to go: with one
    // catalog installed, a language key would cycle between English and
    // English. Adding a file to `languages/` is what turns it on.
    let mut entries: Vec<(&str, &str)> = if app.url.is_some() {
        vec![
            (t!("key.pasteortype"), t!("picker.act.alink")),
            (t!("key.enter"), t!("picker.act.fetch")),
            (t!("key.esc"), t!("picker.act.cancel")),
            (t!("key.backspace"), t!("picker.act.delete")),
        ]
    } else if app.filtering {
        vec![
            (t!("key.type"), t!("picker.act.tofilter")),
            (t!("key.enter"), t!("picker.act.apply")),
            (t!("key.esc"), t!("picker.act.clear")),
            (t!("key.updown"), t!("picker.act.move")),
            (t!("key.backspace"), t!("picker.act.delete")),
        ]
    } else {
        let mut v = vec![
            (t!("key.updown"), t!("picker.act.move")),
            (t!("key.enter"), t!("picker.act.open")),
            (t!("key.space"), t!("picker.act.select")),
            ("A", t!("picker.act.all")),
            ("B", t!("picker.act.markvr")),
            ("C", t!("picker.act.autocrop")),
            ("K", t!("picker.act.cropedit")),
        ];
        if crate::dl::available() {
            v.push(("L", t!("picker.act.pastelink")));
        }
        v.extend([
            ("/", t!("picker.act.filter")),
            ("R", t!("picker.act.refresh")),
            ("E", t!("picker.act.explorer")),
        ]);
        // Offered only while there is something to read: a key that opens an
        // empty report is a key that teaches the user to ignore it.
        if !app.failures.is_empty() {
            v.push(("X", t!("picker.act.errors")));
        }
        if crate::lang::available().len() > 1 {
            v.push(("G", t!("picker.act.language")));
        }
        v.extend([
            (t!("key.left"), t!("picker.act.back")),
            ("S", t!("picker.act.start")),
            ("Q", t!("picker.act.quit")),
        ]);
        v
    };
    entries.shrink_to_fit();
    let help_rows = pack_lines(help_items(&t, &entries), "  ", area.width, FOOT_MAX_ROWS);

    let [logo_a, loc_a, list_a, status_a, opts_a, chrome_a, help_a] = Layout::vertical([
        Constraint::Length(4),
        Constraint::Length(1),
        Constraint::Min(1),
        Constraint::Length(1),
        Constraint::Length(opt_rows.len() as u16),
        Constraint::Length(1),
        Constraint::Length(help_rows.len() as u16),
    ])
    .areas(area);

    f.render_widget(Paragraph::new(goblin_logo(&t, area.width)), logo_a);
    draw_corner_goblin(f, logo_a, &t);

    let loc = match &app.cur {
        Some(d) => d.display().to_string(),
        None => t!("picker.drives").to_string(),
    };
    let mut loc_spans = vec![
        Span::styled(format!("  {}", t!("picker.den")), Style::new().fg(rat(t.text)).bold()),
        Span::styled(loc, Style::new().fg(rat(t.muted))),
    ];
    // how much of this folder is already done, and by what -- so a
    // mostly-scripted drop is obvious before the batch runs and skips it. A
    // folder with nothing scripted says nothing: silence is the same news.
    if let Some(s) = scripted_summary(
        app.n_scripted,
        app.n_videos,
        app.n_ai.load(Ordering::Relaxed),
        app.n_hand.load(Ordering::Relaxed),
    ) {
        loc_spans.push(Span::styled(format!("   {s}"), Style::new().fg(rat(t.muted))));
    }
    // the live filter query, so a narrowed listing is never a mystery
    if app.filtering || !app.filter.is_empty() {
        loc_spans.push(Span::styled(
            format!("   {}", t!("picker.filter", q = app.filter)),
            Style::new().fg(rat(t.warn)).add_modifier(Modifier::BOLD),
        ));
        if app.filtering {
            // A block cursor blinking on wall time. The key loop polls with a
            // timeout precisely so this (and anything else that animates) has a
            // clock to run on -- a purely event-driven loop would freeze it
            // between keystrokes.
            let on = (app.started.elapsed().as_millis() / 450).is_multiple_of(2);
            loc_spans.push(Span::styled(
                if on { "\u{2588}" } else { " " },
                Style::new().fg(rat(t.warn)),
            ));
        }
    }
    f.render_widget(Line::from(loc_spans), loc_a);

    // Only the rows that will actually be drawn are ASSEMBLED. `List` renders a
    // window whatever it is handed, so building an item per entry spent a
    // folder-sized pass of formatting and allocation on every 120 ms frame to
    // throw all but a screenful away. The scroll offset is ours for the same
    // reason: it is what says which rows those are.
    let view_h = list_a.height as usize;
    if let Some(sel) = app.list.selected() {
        // keep the cursor on screen, moving the window the least that does it
        if sel < app.offset {
            app.offset = sel;
        } else if view_h > 0 && sel >= app.offset + view_h {
            app.offset = sel + 1 - view_h;
        }
    }
    let max_off = app.entries.len().saturating_sub(view_h.max(1));
    app.offset = app.offset.min(max_off);
    let hi = (app.offset + view_h).min(app.entries.len());
    let window = &app.entries[app.offset.min(hi)..hi];
    app.aim_probes(view_h);

    let items: Vec<ListItem> = {
        // held for the whole build: the workers only ever insert, so a row
        // rendered from this snapshot is one consistent answer per video
        let probes = app.probes.lock().unwrap();
        let kinds = app.kinds.lock().unwrap();
        window
            .iter()
            .map(|e| match e {
                Entry::Up => ListItem::new("     .."),
                Entry::Dir(p) => {
                    // A folder carries the same box as a video, because Space
                    // does the same thing to it: an affordance that is only in
                    // the key guide is one nobody finds. What it counts is the
                    // folder's OWN videos -- a checked box says "these are in
                    // the batch", never "and everything under them".
                    let n = app.n_selected_in(p);
                    let mark = if n > 0 { "[x]" } else { "[ ]" };
                    let name_style = if n > 0 {
                        Style::new().fg(rat(t.accent)).add_modifier(Modifier::BOLD)
                    } else {
                        Style::new().fg(rat(t.text))
                    };
                    let base = match &app.cur {
                        None => p.display().to_string(),
                        Some(_) => {
                            format!("{}\\", p.file_name().unwrap_or_default().to_string_lossy())
                        }
                    };
                    // the same fixed right column the videos end at, so a
                    // folder's tally lines up with their durations
                    let tail = if n > 0 {
                        t!("picker.selected", n = n).to_string()
                    } else {
                        String::new()
                    };
                    let row = (list_a.width as usize).saturating_sub(1);
                    let name = elide(&base, row.saturating_sub(5 + 2 + cols(&tail)));
                    let pad = row.saturating_sub(5 + cols(&name) + cols(&tail));
                    ListItem::new(Line::from(vec![
                        Span::styled(format!(" {mark} {name}"), name_style),
                        Span::raw(" ".repeat(pad)),
                        Span::styled(tail, Style::new().fg(rat(t.muted))),
                    ]))
                }
                Entry::File(p) => {
                    // What wrote the script beside it, once the classifier pool
                    // has opened it. Absent until then, which draws as no chip
                    // at all -- the same as a video with no script, because the
                    // row that matters (hand work, in warn colour) is the one
                    // worth waiting a frame to be sure of.
                    let kind = kinds.get(&p.with_extension("funscript")).copied();
                    let sel = app.selected.contains(p);
                    let mark = if sel { "[x]" } else { "[ ]" };
                    let name_style = if sel {
                        Style::new().fg(rat(t.accent)).add_modifier(Modifier::BOLD)
                    } else {
                        Style::new()
                    };
                    let probed = probes.get(p);
                    // The chips, in the order they earn attention. A hand-made
                    // script is the one worth a colour of its own: overwriting
                    // hand work is a different mistake from re-drafting an AI
                    // script. The VR chip says which way the answer came --
                    // `[vr]` is the user's own mark, `[vr?]` the detector's
                    // guess, and either one opens the aiming page.
                    let mut chips: Vec<Span> = Vec::new();
                    if app.vr_marks.contains(p) {
                        chips.push(Span::styled(
                            t!("picker.chip.vr"),
                            Style::new().fg(rat(t.accent)).add_modifier(Modifier::BOLD),
                        ));
                    } else if probed.is_some_and(|q| q.vr) {
                        chips.push(Span::styled(
                            t!("picker.chip.vr.guess"),
                            Style::new().fg(rat(t.accent)),
                        ));
                    }
                    if let Some(k) = kind {
                        chips.push(Span::styled(
                            format!("[{}]", k.label()),
                            match k {
                                ScriptKind::Hand => Style::new().fg(rat(t.warn)).bold(),
                                _ => Style::new().fg(rat(t.muted)),
                            },
                        ));
                    }
                    // The duration ends the row at a fixed column, so a folder's
                    // lengths compare down the list instead of zig-zagging after
                    // whatever each name happens to measure. The name is what
                    // gives way when the row is too narrow for all of it.
                    let dur = dur_label(probed.and_then(|q| q.dur_ms));
                    let chips_w: usize = chips.iter().map(|s| cols(&s.content) + 2).sum();
                    // " [x] " + chips + a two-space gap + the duration + one
                    // column left unwritten, so the row never touches the edge
                    let row = (list_a.width as usize).saturating_sub(1);
                    let fixed = 5 + chips_w + 2 + cols(&dur);
                    let name = elide(
                        &p.file_name().unwrap_or_default().to_string_lossy(),
                        row.saturating_sub(fixed),
                    );
                    let pad = row.saturating_sub(5 + cols(&name) + chips_w + cols(&dur));
                    let mut spans = vec![Span::styled(format!(" {mark} {name}"), name_style)];
                    for c in chips {
                        spans.push(Span::raw("  "));
                        spans.push(c);
                    }
                    spans.push(Span::raw(" ".repeat(pad)));
                    spans.push(Span::styled(dur, Style::new().fg(rat(t.muted))));
                    ListItem::new(Line::from(spans))
                }
            })
            .collect()
    };
    let list = List::new(items).highlight_style(
        Style::new().bg(rat(t.chrome_bg)).fg(Color::Black).add_modifier(Modifier::BOLD),
    );
    // The widget is handed the window and nothing else, so its state addresses
    // the window: the cursor's row within it, and no scrolling left to do.
    // `app.list` stays the selection over the WHOLE listing, which is what every
    // key handler reads.
    let mut view_state = ListState::default();
    view_state.select(
        app.list
            .selected()
            .filter(|&i| i >= app.offset && i < app.offset + view_h.max(1))
            .map(|i| i - app.offset),
    );
    f.render_stateful_widget(list, list_a, &mut view_state);

    // One line, three possible tenants, most urgent first: a link being typed,
    // then what a fetch is doing, then feedback from the batch that just ran (an
    // instant, all-skipped start is otherwise wiped by the picker reopening).
    if let Some(buf) = &app.url {
        let on = (app.started.elapsed().as_millis() / 450).is_multiple_of(2);
        f.render_widget(
            Line::from(vec![
                Span::styled(t!("picker.link.prompt"), Style::new().fg(rat(t.text)).bold()),
                Span::styled(buf.clone(), Style::new().fg(rat(t.accent))),
                Span::styled(
                    if on { "\u{2588}" } else { " " },
                    Style::new().fg(rat(t.accent)),
                ),
                Span::styled(t!("picker.link.hint"), Style::new().fg(rat(t.help))),
            ]),
            status_a,
        );
    } else if let Some(s) = &app.dl_line {
        f.render_widget(
            Line::styled(format!("  {s}"), Style::new().fg(rat(t.accent))),
            status_a,
        );
    } else if let Some(s) = &app.status {
        f.render_widget(
            Line::styled(
                format!("  {s}"),
                Style::new().fg(rat(t.warn)).add_modifier(Modifier::BOLD),
            ),
            status_a,
        );
    }

    f.render_widget(Paragraph::new(opt_rows), opts_a);

    f.render_widget(chrome_strip(&t), chrome_a);

    f.render_widget(Paragraph::new(help_rows), help_a);
}

fn run(term: &mut ratatui::DefaultTerminal, app: &mut App) -> Result<Option<Pick>> {
    loop {
        // a running fetch reports on the same clock the screen redraws on
        app.pump_download();
        term.draw(|f| draw(f, app))?;
        // Poll rather than block: the screen has to redraw on its own clock for
        // the filter cursor to blink and the corner goblin to dance. 120 ms is
        // far below the eye's notice for keypress latency, and two polls per
        // dance frame is what keeps the dance from dropping steps.
        if !event::poll(Duration::from_millis(120))? {
            continue;
        }
        // Every key ALREADY QUEUED is handled before the next frame. A paste
        // arrives as a burst of keystrokes and a held arrow outruns the frame
        // clock; redrawing between each would spend the burst on frames nobody
        // sees and leave the listing trailing the keyboard.
        loop {
            if let Event::Key(k) = event::read()? {
                if k.kind == KeyEventKind::Press {
                    if let Some(done) = on_key(app, k) {
                        return Ok(done);
                    }
                }
            }
            if !event::poll(Duration::ZERO)? {
                break;
            }
        }
    }
}

/// One keypress, applied to the picker. `None` carries on; `Some` finishes --
/// `Some(None)` is quit, `Some(Some(pick))` a batch the user started.
fn on_key(app: &mut App, k: KeyEvent) -> Option<Option<Pick>> {
    // The failure report covers the picker, so it takes the keyboard with it.
    // Anything that is not about reading it closes it -- including Q, which
    // here means "done reading", not "quit the session": the user is looking
    // at work that failed, and taking the app away from them is the last thing
    // that screen should be able to do.
    if app.errors {
        match k.code {
            KeyCode::Up => app.err_off = app.err_off.saturating_sub(1),
            KeyCode::Down => app.err_off += 1,
            KeyCode::PageUp => app.err_off = app.err_off.saturating_sub(10),
            KeyCode::PageDown => app.err_off += 10,
            KeyCode::Home => app.err_off = 0,
            // The last failure is the end of the report, and End is the key
            // that asks for it. Unbound, it fell to the catch-all below and
            // took the report off the screen instead.
            KeyCode::End => app.err_off = usize::MAX / 2,
            KeyCode::Char('e') | KeyCode::Char('E') => {
                app.open_log();
                crate::sound::play_click();
            }
            // the chrome strip advertises these here too, so they work here too
            KeyCode::Char('t') | KeyCode::Char('T') => {
                crate::theme::cycle();
                crate::sound::play_click();
            }
            KeyCode::Char('m') | KeyCode::Char('M') => {
                crate::sound::cycle_audio();
                crate::sound::play_click();
            }
            KeyCode::Char('v') | KeyCode::Char('V') if crate::sound::music_on() => {
                crate::sound::cycle_volume();
                crate::sound::play_click();
            }
            KeyCode::Char('n') | KeyCode::Char('N') if crate::sound::music_on() => {
                crate::sound::next_track();
                crate::sound::play_click();
            }
            _ => {
                app.errors = false;
                app.err_off = 0;
                crate::sound::play_click();
            }
        }
        return None;
    }
    // A link being typed captures the keyboard the same way the filter does,
    // and for the same reason: a pasted URL is full of letters that are
    // otherwise hotkeys. Terminals paste as keystrokes, so this is also what
    // makes Ctrl-V work without anything special.
    if let Some(buf) = app.url.as_mut() {
        match k.code {
            KeyCode::Esc => {
                app.url = None;
                app.dl_line = None;
            }
            KeyCode::Enter => app.start_download(),
            KeyCode::Backspace => {
                buf.pop();
            }
            KeyCode::Char(c) => buf.push(c),
            _ => {}
        }
        return None;
    }
    // Filtering captures typed characters, so the letter hotkeys (A, B, L,
    // V, F, S, Q) only fire outside it. Arrows still navigate the narrowed
    // list.
    if app.filtering {
        match k.code {
            KeyCode::Esc => {
                app.clear_filter();
                app.refilter();
            }
            KeyCode::Enter => app.filtering = false, // keep the filter, resume hotkeys
            KeyCode::Backspace => {
                if app.filter.pop().is_none() {
                    app.filtering = false;
                }
                app.refilter();
            }
            KeyCode::Char(c) => {
                app.filter.push(c);
                app.refilter();
            }
            KeyCode::Up => app.move_by(-1),
            KeyCode::Down => app.move_by(1),
            KeyCode::PageUp => app.move_by(-20),
            KeyCode::PageDown => app.move_by(20),
            _ => {}
        }
        return None;
    }
    match k.code {
        // Q is the only way out. Esc is the key a user hits to back out of
        // whatever they just started -- binding it to "quit the app" turns
        // a reflex into a lost session, so here it only drops the filter.
        KeyCode::Char('q') | KeyCode::Char('Q') => return Some(None),
        // Esc reaches for the most recent thing started: a fetch, then a
        // filter. The part-file stays, so pasting the link again resumes it.
        KeyCode::Esc if app.job.is_some() => {
            if let Some(j) = app.job.as_mut() {
                j.cancel();
            }
            app.dl_line = Some(t!("picker.dl.stopping").into());
        }
        KeyCode::Esc if !app.filter.is_empty() => {
            app.clear_filter();
            app.refilter();
        }
        // Paste a link and the goblins fetch it into this folder first. The
        // key exists only when yt-dlp does -- see the startup report.
        KeyCode::Char('l') | KeyCode::Char('L') if crate::dl::available() => {
            app.url = Some(String::new());
            crate::sound::play_click();
        }
        // Starting a batch mid-fetch would walk off with a half-written file
        KeyCode::Char('s') | KeyCode::Char('S')
            if !app.selected.is_empty() && app.job.is_some() =>
        {
            app.dl_line = Some(t!("picker.dl.stillcoming").into());
        }
        KeyCode::Char('s') | KeyCode::Char('S') if !app.selected.is_empty() => {
            crate::sound::play_click();
            return Some(Some(Pick {
                videos: app.selected.iter().cloned().collect(),
                force: app.force,
                autocrop: app.autocrop,
                crop_edit: app.crop_edit,
                dir: app.cur.clone(),
                // only the marks that are in this batch: a mark left on a
                // video the user did not start is not an instruction
                vr: app.vr_marks.intersection(&app.selected).cloned().collect(),
            }));
        }
        // The detector reads most VR sources, and the aiming page has K for
        // the ones it calls VR wrongly. B is the other direction: this one IS
        // VR, whatever the probe made of it.
        KeyCode::Char('b') | KeyCode::Char('B') => {
            app.toggle_vr_at_cursor();
            crate::sound::play_click();
        }
        KeyCode::Char('f') | KeyCode::Char('F') => {
            app.force = !app.force;
            crate::sound::play_click();
        }
        KeyCode::Char('c') | KeyCode::Char('C') => {
            app.autocrop = !app.autocrop;
            crate::sound::play_click();
        }
        KeyCode::Char('k') | KeyCode::Char('K') => {
            app.crop_edit = !app.crop_edit;
            crate::sound::play_click();
        }
        KeyCode::Char('a') | KeyCode::Char('A') => {
            app.select_all_listed();
            crate::sound::play_click();
        }
        KeyCode::Char('t') | KeyCode::Char('T') => {
            crate::theme::cycle();
            crate::sound::play_click();
        }
        // The language, on the same terms as the palette: one key, live, and
        // remembered when a batch starts. Offered only where a second catalog
        // is installed -- see the key guide.
        KeyCode::Char('g') | KeyCode::Char('G') if crate::lang::available().len() > 1 => {
            crate::lang::cycle();
            crate::sound::play_click();
        }
        // One key for the whole audio layer: music -> blips -> silent. The
        // click confirms the new state, so it fires after the switch and
        // stays silent (correctly) on the way into silence.
        KeyCode::Char('m') | KeyCode::Char('M') => {
            crate::sound::cycle_audio();
            crate::sound::play_click();
        }
        // Volume and skip only mean anything while the music is playing;
        // pressed otherwise they are ignored rather than silently arming a
        // setting the user cannot hear the effect of.
        KeyCode::Char('v') | KeyCode::Char('V') if crate::sound::music_on() => {
            crate::sound::cycle_volume();
            crate::sound::play_click();
        }
        KeyCode::Char('n') | KeyCode::Char('N') if crate::sound::music_on() => {
            crate::sound::next_track();
            crate::sound::play_click();
        }
        // The folder moves under us -- a download finishes, a script gets
        // written next door. R (or F5, the habit every file browser trains)
        // re-lists it in place.
        KeyCode::Char('r') | KeyCode::Char('R') | KeyCode::F(5) => {
            app.reload();
            crate::sound::play_click();
        }
        KeyCode::Char('e') | KeyCode::Char('E') => {
            app.open_in_explorer();
            crate::sound::play_click();
        }
        // What the last batch could not do, back whenever it is wanted -- the
        // report opened on its own once, and a user who pressed a key through
        // it should not have to run the batch again to read it.
        KeyCode::Char('x') | KeyCode::Char('X') if !app.failures.is_empty() => {
            app.errors = true;
            app.err_off = 0;
            crate::sound::play_click();
        }
        KeyCode::Char('/') => app.filtering = true,
        KeyCode::Up => app.move_by(-1),
        KeyCode::Down => app.move_by(1),
        KeyCode::PageUp => app.move_by(-20),
        KeyCode::PageDown => app.move_by(20),
        KeyCode::Home => app.move_by(i64::MIN / 2),
        KeyCode::End => app.move_by(i64::MAX / 2),
        // Left alone leaves the folder. Backspace does NOT: clearing a filter
        // is a run of backspaces, the last of which finds an empty box, and a
        // key that walks out of the folder on that stroke throws away the
        // listing the user was reading.
        KeyCode::Left => app.go_up(),
        KeyCode::Right | KeyCode::Enter => app.activate(),
        KeyCode::Char(' ') => {
            app.toggle_at_cursor();
            crate::sound::play_click();
        }
        _ => {}
    }
    None
}

/// Open the picker; `None` means the user quit without starting. `start` is
/// where the browser opens (the previous session location, or the launch dir);
/// `report` is what the batch that just ran left behind -- its outcome line,
/// and its failures, which open on their own screen before the listing.
pub fn pick(
    force: bool,
    autocrop: bool,
    crop_edit: bool,
    start: Option<PathBuf>,
    report: Report,
    dl_dir: Option<PathBuf>,
    dl_args: Vec<String>,
) -> Result<Option<Pick>> {
    let mut term = ratatui::init();
    let mut app = App::new(force, autocrop, crop_edit, start, report, dl_dir, dl_args);
    let r = run(&mut term, &mut app);
    // The pools outlive the screen otherwise, and the batch about to run wants
    // the disk (and, for a probe, the process table) to itself.
    app.stop.store(true, Ordering::Relaxed);
    app.want.lock().unwrap().clear();
    ratatui::restore();
    r
}

/// One drafted video on the review screen.
pub struct ReviewItem {
    pub name: String,
    pub actions: usize,
}

/// The post-draft review loop: the scripts are already on disk, the head's
/// tracks are in memory, and every keypress that changes a parameter
/// re-styles and REWRITES every script in milliseconds -- the user eyeballs
/// the result in their player, reloads, adjusts, repeats. `apply(i, params)`
/// re-styles video `i` and returns its action count.
pub fn review(
    items: &mut [ReviewItem],
    params: &mut [style::Params],
    mut apply: impl FnMut(usize, &style::Params) -> Result<usize>,
) -> Result<()> {
    let mut term = ratatui::init();
    let r = review_run(&mut term, items, params, &mut apply);
    ratatui::restore();
    r
}

fn review_run(
    term: &mut ratatui::DefaultTerminal,
    items: &mut [ReviewItem],
    params: &mut [style::Params],
    apply: &mut impl FnMut(usize, &style::Params) -> Result<usize>,
) -> Result<()> {
    for i in 0..items.len() {
        items[i].actions = apply(i, &params[i])?;
    }
    // style is per script: up/down selects which one the knobs edit
    let mut sel = 0usize;
    loop {
        term.draw(|f| draw_review(f, items, params, sel))?;
        // Same polled loop as the picker, for the same reason: the corner
        // goblin needs a clock of its own to dance on.
        if !event::poll(Duration::from_millis(120))? {
            continue;
        }
        let Event::Key(k) = event::read()? else { continue };
        if k.kind != KeyEventKind::Press {
            continue;
        }
        // The chrome strip advertises the theme and audio keys on this screen
        // too, so they are handled here as well -- and toggling the music is
        // what the goblin in the corner is reacting to.
        match k.code {
            KeyCode::Char('t') | KeyCode::Char('T') => {
                crate::theme::cycle();
                crate::sound::play_click();
                continue;
            }
            KeyCode::Char('m') | KeyCode::Char('M') => {
                crate::sound::cycle_audio();
                crate::sound::play_click();
                continue;
            }
            KeyCode::Char('v') | KeyCode::Char('V') if crate::sound::music_on() => {
                crate::sound::cycle_volume();
                crate::sound::play_click();
                continue;
            }
            KeyCode::Char('n') | KeyCode::Char('N') if crate::sound::music_on() => {
                crate::sound::next_track();
                crate::sound::play_click();
                continue;
            }
            _ => {}
        }
        let before = params[sel];
        match k.code {
            // Enter or Q, not Esc -- same reasoning as the picker: leaving a
            // screen should be a key you meant to press.
            KeyCode::Char('q') | KeyCode::Char('Q') | KeyCode::Enter => return Ok(()),
            KeyCode::Up => sel = sel.saturating_sub(1),
            KeyCode::Down => {
                if sel + 1 < items.len() {
                    sel += 1
                }
            }
            KeyCode::Char('d') | KeyCode::Char('D') => {
                params[sel].dwells = params[sel].dwells.cycle();
                params[sel].dwell_ramp = None; // the preset takes over from any override
            }
            KeyCode::Char('s') | KeyCode::Char('S') => {
                params[sel].stillness = params[sel].stillness.cycle();
                params[sel].still_eps = None;
            }
            KeyCode::Char('u') | KeyCode::Char('U') => {
                params[sel].depth = params[sel].depth.cycle();
                params[sel].depth_dose = None; // the preset takes over from any override
                params[sel].depth_window = None;
            }
            KeyCode::Char('+') | KeyCode::Char('=') => {
                params[sel].intensity = (params[sel].intensity + 0.1).min(2.0)
            }
            KeyCode::Char('-') => params[sel].intensity = (params[sel].intensity - 0.1).max(0.5),
            _ => {}
        }
        if params[sel] != before {
            items[sel].actions = apply(sel, &params[sel])?;
        }
    }
}

fn draw_review(f: &mut Frame, items: &[ReviewItem], params: &[style::Params], sel: usize) {
    let t = theme();
    let area = f.area();

    // knobs and guide, packed to the width before anything is laid out -- the
    // knob row runs past 80 columns as soon as a value is a long word
    let knob = |key: &str, label: &str, value: String| -> Vec<Span<'static>> {
        vec![
            Span::styled(format!(" {key} "), Style::new().fg(Color::Black).bg(rat(t.chrome_bg))),
            Span::raw(format!(" {label}: ")),
            Span::styled(value, Style::new().fg(rat(t.accent)).add_modifier(Modifier::BOLD)),
        ]
    };
    let p = &params[sel];
    let opt_rows = pack_lines(
        vec![
            knob("D", t!("review.tui.knob.dwells"), preset(p.dwells.label())),
            knob("S", t!("review.tui.knob.stillness"), preset(p.stillness.label())),
            knob("U", t!("review.tui.knob.depth"), preset(p.depth.label())),
            knob("+/-", t!("review.tui.knob.intensity"), format!("{:.1}", p.intensity)),
        ],
        "  ",
        area.width,
        FOOT_MAX_ROWS,
    );
    let help_rows = pack_lines(
        help_items(
            &t,
            &[
                (t!("key.updown"), t!("review.tui.act.pick")),
                (t!("review.tui.key.restyle"), t!("review.tui.act.restyle")),
                (t!("key.enterq"), t!("review.tui.act.back")),
            ],
        ),
        "   ",
        area.width,
        FOOT_MAX_ROWS,
    );

    let [logo_a, sub_a, list_a, opts_a, chrome_a, help_a] = Layout::vertical([
        Constraint::Length(4),
        Constraint::Length(1),
        Constraint::Min(1),
        Constraint::Length(opt_rows.len() as u16),
        Constraint::Length(1),
        Constraint::Length(help_rows.len() as u16),
    ])
    .areas(area);

    f.render_widget(Paragraph::new(goblin_logo(&t, area.width)), logo_a);
    draw_corner_goblin(f, logo_a, &t);

    f.render_widget(
        Line::from(vec![
            Span::styled(t!("review.tui.title"), Style::new().fg(rat(t.text)).bold()),
            Span::styled(t!("review.tui.subtitle"), Style::new().fg(rat(t.muted))),
        ]),
        sub_a,
    );

    let rows: Vec<ListItem> = items
        .iter()
        .enumerate()
        .map(|(i, it)| {
            let here = i == sel;
            let mark = if here { " > " } else { "   " };
            ListItem::new(Line::from(vec![
                Span::styled(
                    format!("{mark}[OK] "),
                    Style::new().fg(rat(t.ok)).add_modifier(Modifier::BOLD),
                ),
                Span::styled(
                    it.name.clone(),
                    if here {
                        Style::new().fg(rat(t.accent)).add_modifier(Modifier::BOLD)
                    } else {
                        Style::new().fg(rat(t.text))
                    },
                ),
                Span::styled(
                    t!("review.tui.actions", n = it.actions),
                    Style::new().fg(rat(t.muted)),
                ),
            ]))
        })
        .collect();
    f.render_widget(List::new(rows), list_a);

    f.render_widget(Paragraph::new(opt_rows), opts_a);

    f.render_widget(chrome_strip(&t), chrome_a);

    f.render_widget(Paragraph::new(help_rows), help_a);
}

