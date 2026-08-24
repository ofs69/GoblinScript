//! The optional VR prep step: aim a flat viewport at the action once, and the
//! rest of the pipeline never learns the source was VR.
//!
//! An equirectangular eye is off-distribution for the frozen encoder -- every
//! clip the model has seen is flat 2D footage -- so a VR source has to become
//! flat before it becomes latents. That reprojection needs one human decision
//! (WHERE to point) which no heuristic here is going to make, and a browser is
//! the only place to make it: it is an aiming task with a picture in it.
//!
//! Two things keep this cheap rather than a second pipeline:
//!
//! * **The aim folds into the normalize transcode.** `crop` the eye, `v360` it
//!   to flat, and hand that to the scale/fps chain the flat path already runs
//!   -- ONE ffmpeg pass, no intermediate render. A static aim also means v360
//!   builds its projection LUT once, so the pass costs about what a flat
//!   transcode of the same footage costs.
//! * **The preview is the render.** `projector.js` is one file shared with the
//!   training-ingest tool (`vr_project.py serve`), embedded here unchanged, and
//!   it reproduces v360's mapping per pixel. The page projects server-supplied
//!   eye frames itself, so re-aiming is a redraw rather than a round trip -- and
//!   V flips to a frame ffmpeg's own v360 produced, the one check that catches
//!   the two engines disagreeing.
//!
//! A render RANGE is the other half of the step: VR sources run long and the
//! encoder is the expensive stage, so the page can trim one. The draft then
//! runs on the CLIP clock, and the written funscript is shifted back onto the
//! source clock at the end -- what lands on disk is always for the full video.

use anyhow::{Context, Result};
use ratatui::crossterm::event::{self, Event, KeyCode, KeyEventKind, KeyModifiers};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tiny_http::{Header, Method, Response};

/// Eye width fetched for client-side projection. Matches the page's `PROJ_W`:
/// one size is ever asked for, so one size is ever cached.
const PROJ_W: u32 = 1600;
/// Eye frames held in memory, per session. NSFW frames never touch disk here --
/// unlike the Python tool, which needs a temp dir to re-encode from.
const FRAME_CAP: usize = 48;

const BURST_COLS: u32 = 5;
const BURST_ROWS: u32 = 4;
const BURST_FPS: f64 = 10.0;

// --------------------------------------------------------------------------
// the aim
// --------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Aim {
    pub yaw: f64,
    pub pitch: f64,
}

/// The sidecar. Field-for-field the one `vr_project.py` writes, so an aim made
/// in either tool is readable by the other -- there is one projection and one
/// way to describe it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Config {
    #[serde(default = "version_2")]
    pub version: u32,
    /// Which eye to cut out: `sbs` (left half), `tb` (top half), `mono` (all).
    pub layout: String,
    /// v360's input format for that eye: `hequirect`, `fisheye`, `equirect`.
    pub projection: String,
    /// Input lens FOV -- fisheye only (190/200/... lenses).
    pub ih_fov: f64,
    /// Output viewport FOV. FIXED for the whole video: a constant angular
    /// scale is what keeps motion magnitudes comparable for the fixed-grid
    /// perception, which a zooming rect would not.
    pub h_fov: f64,
    /// `None` = derived from `h_fov` and the output aspect.
    pub v_fov: Option<f64>,
    #[serde(default)]
    pub roll: f64,
    /// The viewport's ASPECT. Not a resolution and not cosmetic: the encoder
    /// decodes every clip at a flat `scale=enc_res:enc_res`, so the frame it
    /// sees is the normalized copy SQUASHED into a square, and the squash
    /// factor is exactly this aspect. 16:9 is 94% of the training corpus and
    /// the shape every VR clip in it was reprojected at, so anything else
    /// hands the goblins a stretch they have effectively never seen. (The
    /// height comes from the encode spec -- only the RATIO is ever used.)
    pub out_w: u32,
    pub out_h: u32,
    /// `[t0, t1]` on the SOURCE clock, or `None` for the whole video.
    #[serde(default)]
    pub range_ms: Option<[f64; 2]>,
    /// ONE direction for the whole render. Measured equivalent to an
    /// 11-keyframe human track, which is why there is no keyframe list here.
    pub aim: Aim,
    /// A human saved something. An untouched sidecar has never been aimed.
    #[serde(default)]
    pub touched: bool,
    /// The user said "not VR after all" -- draft it as plain 2D.
    #[serde(default)]
    pub skip: bool,
}

fn version_2() -> u32 {
    2
}

/// The default pitch: aim the viewport at the LOWER half of the frame, centred
/// so its top edge sits on the horizon and it looks down over the bottom half
/// -- where POV VR action sits, so straight-ahead is rarely the right aim.
/// Derived from the default viewport's vertical FOV (the same `auto_v_fov`
/// rule), matching `vr_project.py`'s `_bottom_half_pitch`.
fn default_pitch() -> f64 {
    let h: f64 = (90.0_f64).to_radians();
    let v = 2.0 * ((h / 2.0).tan() * 1080.0 / 1920.0).atan();
    (-v.to_degrees() / 2.0 * 1000.0).round() / 1000.0
}

impl Default for Config {
    fn default() -> Self {
        Self {
            version: 2,
            layout: "sbs".into(),
            projection: "hequirect".into(),
            ih_fov: 180.0,
            h_fov: 90.0,
            v_fov: None,
            roll: 0.0,
            out_w: 1920,
            out_h: 1080,
            range_ms: None,
            // ONE direction for the whole render; defaults to the bottom half
            // (yaw 0, pitch tilted down) rather than dead centre.
            aim: Aim { yaw: 0.0, pitch: default_pitch() },
            touched: false,
            skip: false,
        }
    }
}

pub fn wrap180(a: f64) -> f64 {
    (a + 180.0).rem_euclid(360.0) - 180.0
}

impl Config {
    /// `v_fov` from `h_fov` and the output aspect (rectilinear), unless pinned.
    /// The same rule `projector.js` applies before it draws, and
    /// `vr_project.py`'s `auto_v_fov` before it calls v360.
    pub fn auto_v_fov(&self) -> f64 {
        let h = self.h_fov.to_radians();
        let v = 2.0 * ((h / 2.0).tan() * self.out_h as f64 / self.out_w as f64).atan();
        (v.to_degrees() * 1000.0).round() / 1000.0
    }

    pub fn v_fov_eff(&self) -> f64 {
        self.v_fov.unwrap_or_else(|| self.auto_v_fov())
    }

    /// The ffmpeg crop that isolates the projected eye; `None` = whole frame.
    fn crop_expr(&self) -> Option<&'static str> {
        match self.layout.as_str() {
            "sbs" => Some("crop=iw/2:ih:0:0"),
            "tb" => Some("crop=iw:ih/2:0:0"),
            _ => None,
        }
    }

    /// The single v360 filter string, used by the A/B reference AND the render.
    /// `yaw`/`pitch` are passed rather than read off `self`, because the A/B
    /// reference has to follow an UNSAVED drag -- that is the whole point of it.
    pub fn v360_expr(&self, yaw: f64, pitch: f64, w: u32, h: u32) -> String {
        let mut parts = vec![
            format!("input={}", self.projection),
            "output=flat".to_string(),
        ];
        if self.projection == "fisheye" {
            parts.push(format!("ih_fov={}", self.ih_fov));
            parts.push(format!("iv_fov={}", self.ih_fov));
        }
        parts.push(format!("h_fov={}", self.h_fov));
        parts.push(format!("v_fov={}", self.v_fov_eff()));
        parts.push(format!("w={w}"));
        parts.push(format!("h={h}"));
        parts.push(format!("yaw={yaw:.4}"));
        parts.push(format!("pitch={pitch:.4}"));
        parts.push(format!("roll={}", self.roll));
        format!("v360={}", parts.join(":"))
    }

    /// The filter chain that turns this VR source into flat 2D at `out_h`, for
    /// the front of the normalize transcode. The aspect is the config's; the
    /// height is the encode spec's, so the projected video lands in exactly the
    /// shape every flat source lands in.
    pub fn filter_prefix(&self, out_h: u32) -> String {
        let out_h = out_h.max(2) & !1;
        let out_w = ((out_h as f64 * self.out_w as f64 / self.out_h as f64).round() as u32).max(2)
            & !1;
        let mut v: Vec<String> = Vec::new();
        if let Some(c) = self.crop_expr() {
            v.push(c.to_string());
        }
        v.push(self.v360_expr(self.aim.yaw, self.aim.pitch, out_w, out_h));
        v.join(",")
    }

    /// The effective `[t0, t1]` on the source clock, clamped into the video.
    pub fn range(&self, dur_ms: f64) -> (f64, f64) {
        match self.range_ms {
            None => (0.0, dur_ms),
            Some([a, b]) => {
                let t0 = a.max(0.0).min((dur_ms - 100.0).max(0.0));
                let t1 = b.max(t0 + 100.0).min(dur_ms);
                (t0, t1)
            }
        }
    }

    /// Where the draft's clock starts on the source clock -- what the written
    /// funscript is shifted by. 0 unless a range was trimmed.
    pub fn t0_ms(&self, dur_ms: f64) -> f64 {
        self.range(dur_ms).0
    }

    /// The viewport aspect the perception stack expects, and what every flat
    /// source effectively arrives as.
    pub const NATIVE_ASPECT: f64 = 16.0 / 9.0;

    /// Complains when the viewport aspect would hand the encoder a stretch the
    /// corpus never contained.
    ///
    /// `scale=enc_res:enc_res` (`encode.rs`, and `jepa_extract` on the training
    /// side) squashes the normalized copy into a square, so the aspect chosen
    /// here IS the anamorphic factor the model sees. A 16:9 viewport squashes
    /// the way 94% of the corpus squashed -- and the way every VR clip in it
    /// was reprojected; a square one does not squash at all, and the goblins
    /// read the motion as narrower than it is. Not refused -- the pixels are
    /// the user's call -- but it must not pass unseen, because the preview
    /// looks perfectly aimed either way.
    pub fn aspect_warning(&self) -> Option<String> {
        let a = self.out_w as f64 / self.out_h.max(1) as f64;
        // +-5%: 1920x1080, 1280x720 and 854x480 all read as native; 4:3 and
        // square do not.
        if (a / Self::NATIVE_ASPECT - 1.0).abs() <= 0.05 {
            return None;
        }
        Some(format!(
            "viewport is {a:.2}:1, not the 16:9 the goblins were trained on -- \
             the picture reaches them squashed into a square either way, so a \
             different shape is a stretch they have never seen"
        ))
    }

    /// Has a human actually aimed this? An untouched sidecar still on the
    /// default bottom-half aim has not, and drafting it would silently point
    /// the goblins at whatever happens to be at the default aim.
    pub fn aimed(&self) -> bool {
        self.touched
    }

    /// A stable identity for everything about this config that changes the
    /// pixels the encoder sees. Folded into the cache key, so re-aiming a video
    /// gets its own working directory instead of silently reusing a transcode
    /// of the old aim.
    pub fn key(&self) -> String {
        // the render-affecting fields only: `touched` is bookkeeping, and a
        // skipped config produces no VR stage at all
        let mut h = blake3::Hasher::new();
        for s in [&self.layout, &self.projection] {
            h.update(s.as_bytes());
        }
        for f in [
            self.ih_fov,
            self.h_fov,
            self.v_fov_eff(),
            self.roll,
            self.out_w as f64,
            self.out_h as f64,
            self.aim.yaw,
            self.aim.pitch,
            self.range_ms.map(|r| r[0]).unwrap_or(-1.0),
            self.range_ms.map(|r| r[1]).unwrap_or(-1.0),
        ] {
            h.update(&f.to_le_bytes());
        }
        h.finalize().to_hex()[..12].to_string()
    }
}

// --------------------------------------------------------------------------
// detection
// --------------------------------------------------------------------------

/// What a probe made of a source, when it looks like VR.
pub struct Detected {
    pub cfg: Config,
    /// Sanitized, for the console: dimensions and the guess, never a path.
    pub why: String,
    pub w: u32,
    pub h: u32,
    pub dur_ms: f64,
}

/// Does this look like a VR source, and if so how is it laid out?
///
/// Deliberately recall-biased, and deliberately not the last word: a wrong
/// guess costs one glance at the prep page (where the layout and projection are
/// dropdowns and a mistake is instantly visible), while a MISSED VR source
/// costs a full-length draft of nonsense. Stereo/spherical side data decides it
/// when the container carries any; dimensions are the fallback, and the common
/// VR shapes are unmistakable at the sizes VR is shot at.
pub fn detect(path: &Path) -> Option<Detected> {
    let (w, h, dur_ms, stereo, spherical) = probe(path)?;
    classify(w, h, dur_ms, stereo, spherical)
}

/// How long the source runs, and whether `detect` reads it as VR -- both off
/// ONE probe. The picker asks both questions of every video in a folder, and
/// asking them separately would double a listing's ffprobe bill.
pub fn probe_summary(path: &Path) -> Option<(f64, bool)> {
    let (w, h, dur_ms, stereo, spherical) = probe(path)?;
    Some((dur_ms, classify(w, h, dur_ms, stereo, spherical).is_some()))
}

/// The verdict on an already-probed source.
fn classify(
    w: u32,
    h: u32,
    dur_ms: f64,
    stereo: Option<String>,
    spherical: bool,
) -> Option<Detected> {
    if w == 0 || h == 0 {
        return None;
    }
    let ratio = w as f64 / h as f64;
    let mut cfg = Config::default();

    // Side data is definitive where it exists -- most VR files carry neither.
    let (layout, why) = match stereo.as_deref() {
        Some("side by side") | Some("side by side (quincunx subsampling)") => {
            ("sbs", "tagged side-by-side stereo")
        }
        Some("top and bottom") => ("tb", "tagged top-and-bottom stereo"),
        _ if spherical => {
            // spherical but mono (or untagged stereo): the frame shape decides
            if ratio >= 1.9 {
                ("sbs", "tagged spherical, 2:1 frame")
            } else {
                ("tb", "tagged spherical, square frame")
            }
        }
        // No tags: fall back to the shapes VR actually ships in. A full SBS
        // equirect frame is 2:1 (two 1:1 eyes); a full TB frame is 1:1 (two 2:1
        // eyes). Both need to be BIG -- a 2.39:1 scope film is ~1920 wide and a
        // 1:1 crop is somebody's phone video, and neither is VR.
        _ if (1.9..=2.1).contains(&ratio) && w >= 2560 => {
            ("sbs", "2:1 frame at VR resolution")
        }
        _ if (0.95..=1.05).contains(&ratio) && w >= 2160 => {
            ("tb", "square frame at VR resolution")
        }
        _ => return None,
    };
    cfg.layout = layout.into();
    Some(Detected {
        cfg,
        why: format!("{w}x{h}, {why}"),
        w,
        h,
        dur_ms,
    })
}

/// `(width, height, duration_ms, stereo3d type, has spherical mapping)`.
fn probe(path: &Path) -> Option<(u32, u32, f64, Option<String>, bool)> {
    let out = Command::new("ffprobe")
        .args([
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height:stream_side_data=side_data_type,type:format=duration",
            "-of",
            "json",
        ])
        .arg(path)
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).ok()?;
    // The first entry WITH a picture, not simply the first: a video tied to a
    // timecode track by a track reference is reported twice by an ffprobe new
    // enough to read stream groups, and a member that carries no picture has
    // no width to read.
    let st = v["streams"].as_array()?.iter().find(|s| {
        s["width"].as_u64().is_some_and(|w| w > 0) && s["height"].as_u64().is_some_and(|h| h > 0)
    })?;
    let w = st["width"].as_u64()? as u32;
    let h = st["height"].as_u64()? as u32;
    let dur_ms = v["format"]["duration"]
        .as_str()
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or(0.0)
        * 1000.0;
    let mut stereo = None;
    let mut spherical = false;
    if let Some(sd) = st["side_data_list"].as_array() {
        for d in sd {
            let kind = d["side_data_type"].as_str().unwrap_or("");
            if kind.eq_ignore_ascii_case("Stereo 3D") {
                stereo = d["type"].as_str().map(|s| s.to_ascii_lowercase());
            }
            if kind.to_ascii_lowercase().contains("spherical") {
                spherical = true;
            }
        }
    }
    Some((w, h, dur_ms, stereo, spherical))
}

// --------------------------------------------------------------------------
// sidecars
// --------------------------------------------------------------------------

/// Where a video's aim is remembered: keyed on the SOURCE bytes only, so it
/// survives re-aiming (which changes the draft's cache key, not the video's
/// identity) and is found again on a later run.
pub fn sidecar_path(cache_root: &Path, video: &Path) -> Result<PathBuf> {
    let dir = cache_root.join("vr");
    std::fs::create_dir_all(&dir)
        .with_context(|| format!("could not create {}", dir.display()))?;
    Ok(dir.join(format!("{}.json", crate::cache::fingerprint(video)?)))
}

pub fn load(path: &Path) -> Option<Config> {
    serde_json::from_slice(&std::fs::read(path).ok()?).ok()
}

pub fn save(path: &Path, cfg: &Config) -> Result<()> {
    std::fs::write(path, serde_json::to_vec_pretty(cfg)?)
        .with_context(|| format!("could not write {}", path.display()))?;
    Ok(())
}

// --------------------------------------------------------------------------
// the prep session
// --------------------------------------------------------------------------

/// One video waiting to be aimed. The whole batch is aimed in ONE session,
/// before any drafting starts -- the batch loop normalizes the next video while
/// this one is on the GPU, and it cannot do that for a video whose aim is still
/// a question.
pub struct Clip {
    pub name: String,
    pub src: PathBuf,
    pub sidecar: PathBuf,
    pub cfg: Config,
    pub dur_ms: f64,
    pub src_w: u32,
    pub src_h: u32,
}

struct Session {
    clips: Vec<Clip>,
    idx: usize,
    /// Eye frames, keyed by (rounded ms, layout). Held across a clip switch is
    /// wrong -- a new source means new frames -- so a switch clears it.
    frames: HashMap<(i64, String), Arc<Vec<u8>>>,
    order: Vec<(i64, String)>,
}

impl Session {
    fn cur(&self) -> &Clip {
        &self.clips[self.idx]
    }

    fn take_frame(&mut self, key: &(i64, String)) -> Option<Arc<Vec<u8>>> {
        self.frames.get(key).cloned()
    }

    fn put_frame(&mut self, key: (i64, String), data: Arc<Vec<u8>>) {
        if self.frames.insert(key.clone(), data).is_none() {
            self.order.push(key);
        }
        while self.order.len() > FRAME_CAP {
            let old = self.order.remove(0);
            self.frames.remove(&old);
        }
    }

    fn drop_frames(&mut self) {
        self.frames.clear();
        self.order.clear();
    }
}

/// Aim every clip, then hand them back with their configs saved.
///
/// Blocks until the user is done (the page's Done button, or Enter/Q/Esc in the
/// console). Ctrl-C propagates as a cancel, same as everywhere else.
pub fn prep(clips: Vec<Clip>, open: bool) -> Result<Vec<Clip>> {
    let server = tiny_http::Server::http("127.0.0.1:0")
        .map_err(|e| anyhow::anyhow!("could not start the prep server: {e}"))?;
    let port = server
        .server_addr()
        .to_ip()
        .context("prep server has no TCP address")?
        .port();
    let url = format!("http://127.0.0.1:{port}/");

    let state = Arc::new(Mutex::new(Session {
        clips,
        idx: 0,
        frames: HashMap::new(),
        order: Vec::new(),
    }));

    let th = crate::theme::theme();
    println!(
        "\n  {} {}  {}",
        console::style("[ VR ]").fg(crate::theme::con(th.accent)).bold(),
        console::style(&url).fg(crate::theme::con(th.accent)).bold(),
        console::style(crate::t!("console.vr.hint")).fg(crate::theme::con(th.muted))
    );
    if open {
        open_browser(&url);
    }

    let raw = crate::RawMode::enable();
    loop {
        crate::cancel::check()?;
        // Console side: Enter/Q/Esc finish, same as the page's Done. No console
        // (a piped run) means the page's button is the only finish, which is
        // exactly right there.
        while raw.on() && event::poll(Duration::ZERO)? {
            let k = match event::read()? {
                Event::Key(k) if k.kind == KeyEventKind::Press => k,
                _ => continue,
            };
            match k.code {
                KeyCode::Enter | KeyCode::Esc | KeyCode::Char('q') | KeyCode::Char('Q') => {
                    return finish(state);
                }
                KeyCode::Char('c') if k.modifiers.contains(KeyModifiers::CONTROL) => {
                    crate::cancel::request();
                    crate::cancel::check()?;
                }
                _ => {}
            }
        }
        let req = match server.recv_timeout(Duration::from_millis(50))? {
            Some(r) => r,
            None => continue,
        };
        if handle(req, &state)? {
            return finish(state);
        }
    }
}

fn finish(state: Arc<Mutex<Session>>) -> Result<Vec<Clip>> {
    let mut s = state.lock().unwrap();
    s.drop_frames();
    Ok(std::mem::take(&mut s.clips))
}

/// Returns true when the session is over.
fn handle(mut req: tiny_http::Request, state: &Arc<Mutex<Session>>) -> Result<bool> {
    let url = req.url().to_string();
    let (path, query) = match url.split_once('?') {
        Some((p, q)) => (p.to_string(), q.to_string()),
        None => (url, String::new()),
    };
    match (req.method().clone(), path.as_str()) {
        (Method::Get, "/") | (Method::Get, "/index.html") => {
            let _ = req.respond(
                Response::from_string(include_str!("vr.html"))
                    .with_header(header("Content-Type", "text/html; charset=utf-8")),
            );
        }
        // The ONE projection definition, shared verbatim with vr_project.py's
        // aiming page in the training tree, which keeps a byte-identical copy
        // and checks it (`grid_check.py`). The preview and the render agree
        // per pixel because there is exactly one mapping, spelled once.
        (Method::Get, "/projector.js") => {
            let _ = req.respond(
                Response::from_string(include_str!("projector.js"))
                    .with_header(header("Content-Type", "text/javascript; charset=utf-8")),
            );
        }
        (Method::Get, "/api/state") => {
            let s = state.lock().unwrap();
            let _ = req.respond(json_response(&state_json(&s)));
        }
        // The active catalog, same as the review page's -- both are surfaces of
        // the app that opened them and speak the language the picker is set to.
        (Method::Get, "/api/lang") => {
            let _ = req.respond(json_response(&crate::lang::catalog_json()));
        }
        (Method::Get, "/api/clips") => {
            let s = state.lock().unwrap();
            let _ = req.respond(json_response(&clips_json(&s)));
        }
        // Frames are ffmpeg seeks into an 8K source -- seconds each. They go to
        // a thread so a slow decode never freezes the page's controls (or the
        // Done button) behind it.
        (Method::Get, "/api/frame") => {
            let st = state.clone();
            std::thread::spawn(move || frame_request(req, &st, &query));
        }
        (Method::Get, "/api/burst") => {
            let st = state.clone();
            std::thread::spawn(move || burst_request(req, &st, &query));
        }
        (Method::Post, "/api/config") => {
            let body = read_body(&mut req);
            let mut s = state.lock().unwrap();
            match apply_patch(&mut s, &body) {
                Ok(relayout) => {
                    if relayout {
                        s.drop_frames();
                    }
                    let cfg = s.cur().cfg.clone();
                    let path = s.cur().sidecar.clone();
                    save(&path, &cfg)?;
                    let _ = req.respond(json_response(&serde_json::json!({
                        "ok": true, "cfg": cfg_json(&cfg)
                    })));
                }
                Err(e) => {
                    let _ = req.respond(json_response(&serde_json::json!({
                        "ok": false, "error": format!("{e:#}")
                    })));
                }
            }
        }
        (Method::Post, "/api/aim") => {
            let body = read_body(&mut req);
            let mut s = state.lock().unwrap();
            match parse_aim(&body) {
                Some(aim) => {
                    let i = s.idx;
                    s.clips[i].cfg.aim = aim;
                    s.clips[i].cfg.touched = true;
                    s.clips[i].cfg.skip = false;
                    let (cfg, path) = (s.clips[i].cfg.clone(), s.clips[i].sidecar.clone());
                    save(&path, &cfg)?;
                    let _ = req.respond(json_response(&serde_json::json!({
                        "ok": true, "aim": aim
                    })));
                }
                None => {
                    let _ = req.respond(json_response(&serde_json::json!({
                        "ok": false, "error": "bad aim"
                    })));
                }
            }
        }
        // "not VR after all" -- a false positive on the detector must have a
        // one-click way out that is not "quit and pass a flag".
        (Method::Post, "/api/skip") => {
            let mut s = state.lock().unwrap();
            let i = s.idx;
            s.clips[i].cfg.skip = !s.clips[i].cfg.skip;
            s.clips[i].cfg.touched = true;
            let (cfg, path) = (s.clips[i].cfg.clone(), s.clips[i].sidecar.clone());
            save(&path, &cfg)?;
            let _ = req.respond(json_response(&serde_json::json!({
                "ok": true, "skip": cfg.skip
            })));
        }
        (Method::Post, "/api/clip") => {
            let body = read_body(&mut req);
            let mut s = state.lock().unwrap();
            let i = serde_json::from_str::<serde_json::Value>(&body)
                .ok()
                .and_then(|v| v["index"].as_u64())
                .map(|i| i as usize);
            match i {
                Some(i) if i < s.clips.len() => {
                    if i != s.idx {
                        s.idx = i;
                        s.drop_frames(); // a new clip means new eye frames entirely
                    }
                    let _ = req.respond(json_response(&serde_json::json!({
                        "ok": true, "idx": i
                    })));
                }
                _ => {
                    let _ = req.respond(json_response(&serde_json::json!({
                        "ok": false, "error": "bad index"
                    })));
                }
            }
        }
        (Method::Post, "/api/done") => {
            let _ = req.respond(Response::from_string("ok"));
            return Ok(true);
        }
        _ => {
            let _ = req.respond(Response::from_string("not found").with_status_code(404));
        }
    }
    Ok(false)
}

fn read_body(req: &mut tiny_http::Request) -> String {
    let mut s = String::new();
    let _ = std::io::Read::read_to_string(req.as_reader(), &mut s);
    s
}

/// Apply a config patch. Returns true when the change re-crops the eye, which
/// makes every held frame wrong.
fn apply_patch(s: &mut Session, body: &str) -> Result<bool> {
    let v: serde_json::Value = serde_json::from_str(body).context("bad config payload")?;
    let obj = v.as_object().context("config patch is not an object")?;
    let i = s.idx;
    let dur = s.clips[i].dur_ms;
    let cfg = &mut s.clips[i].cfg;
    let mut relayout = false;
    for (k, val) in obj {
        match k.as_str() {
            "layout" => {
                let l = val.as_str().context("layout must be a string")?;
                if !["sbs", "tb", "mono"].contains(&l) {
                    anyhow::bail!("unknown layout {l:?}");
                }
                relayout |= cfg.layout != l;
                cfg.layout = l.into();
            }
            "projection" => {
                let p = val.as_str().context("projection must be a string")?;
                if !["hequirect", "fisheye", "equirect"].contains(&p) {
                    anyhow::bail!("unknown projection {p:?}");
                }
                cfg.projection = p.into();
            }
            // same bounds vr_project.py enforces, so a config valid in one
            // tool is valid in the other
            "ih_fov" => cfg.ih_fov = num(val, 120.0, 360.0)?,
            "h_fov" => cfg.h_fov = num(val, 10.0, 150.0)?,
            "v_fov" => {
                cfg.v_fov = if val.is_null() {
                    None
                } else {
                    Some(num(val, 10.0, 150.0)?)
                }
            }
            "roll" => cfg.roll = num(val, -180.0, 180.0)?,
            "out_w" => cfg.out_w = num(val, 64.0, 4096.0)? as u32,
            "out_h" => cfg.out_h = num(val, 64.0, 4096.0)? as u32,
            "range_ms" => {
                cfg.range_ms = if val.is_null() {
                    None
                } else {
                    let a = val.as_array().context("range_ms must be [t0, t1]")?;
                    if a.len() != 2 {
                        anyhow::bail!("range_ms must be [t0, t1]");
                    }
                    let t0 = num(&a[0], 0.0, dur.max(1.0))?;
                    let t1 = num(&a[1], 0.0, dur.max(1.0))?;
                    if t1 - t0 < 1000.0 {
                        anyhow::bail!("a render range must be at least a second long");
                    }
                    Some([t0, t1])
                }
            }
            other => anyhow::bail!("unknown config field {other:?}"),
        }
    }
    cfg.touched = true;
    cfg.skip = false;
    Ok(relayout)
}

fn num(v: &serde_json::Value, lo: f64, hi: f64) -> Result<f64> {
    let n = v.as_f64().context("expected a number")?;
    if !n.is_finite() || n < lo || n > hi {
        anyhow::bail!("{n} is out of range ({lo}..{hi})");
    }
    Ok(n)
}

fn parse_aim(body: &str) -> Option<Aim> {
    let v: serde_json::Value = serde_json::from_str(body).ok()?;
    let a = &v["aim"];
    let yaw = a["yaw"].as_f64()?;
    let pitch = a["pitch"].as_f64()?;
    if !yaw.is_finite() || !pitch.is_finite() {
        return None;
    }
    Some(Aim {
        yaw: wrap180(yaw),
        pitch: pitch.clamp(-90.0, 90.0),
    })
}

fn cfg_json(c: &Config) -> serde_json::Value {
    let mut v = serde_json::to_value(c).expect("config serializes");
    v["v_fov_auto"] = serde_json::json!(c.auto_v_fov());
    v["aspect_warning"] = match c.aspect_warning() {
        Some(w) => serde_json::json!(w),
        None => serde_json::Value::Null,
    };
    v
}

fn state_json(s: &Session) -> serde_json::Value {
    let c = s.cur();
    serde_json::json!({
        "id": c.name,
        "cfg": cfg_json(&c.cfg),
        "src_w": c.src_w,
        "src_h": c.src_h,
        "duration_ms": c.dur_ms,
        "idx": s.idx,
        "n_clips": s.clips.len(),
        "theme": crate::theme::css_vars(),
    })
}

fn clips_json(s: &Session) -> serde_json::Value {
    serde_json::json!({
        "idx": s.idx,
        "clips": s.clips.iter().map(|c| serde_json::json!({
            "id": c.name,
            "aimed": c.cfg.aimed() && !c.cfg.skip,
            "ranged": c.cfg.range_ms.is_some(),
            "skipped": c.cfg.skip,
        })).collect::<Vec<_>>(),
    })
}

// --------------------------------------------------------------------------
// frame endpoints
// --------------------------------------------------------------------------

fn qparam(query: &str, key: &str) -> Option<String> {
    query.split('&').find_map(|kv| {
        let (k, v) = kv.split_once('=')?;
        (k == key).then(|| v.to_string())
    })
}

fn qnum(query: &str, key: &str) -> Option<f64> {
    qparam(query, key)?.parse().ok()
}

/// `mode=equi` is the working frame: the equirect eye, which the page projects
/// itself. `mode=proj` re-projects it here through the render's own v360 -- the
/// A/B reference (V), and the only thing that can catch the shader disagreeing
/// with the renderer.
fn frame_request(req: tiny_http::Request, state: &Arc<Mutex<Session>>, query: &str) {
    let (src, layout, cfg, dur) = {
        let s = state.lock().unwrap();
        let c = s.cur();
        (c.src.clone(), c.cfg.layout.clone(), c.cfg.clone(), c.dur_ms)
    };
    let t_ms = match qnum(query, "t_ms") {
        Some(t) => t.clamp(0.0, (dur - 50.0).max(0.0)),
        None => {
            let _ = req.respond(Response::from_string("bad t_ms").with_status_code(400));
            return;
        }
    };
    let key = (t_ms.round() as i64, layout.clone());

    let cached = state.lock().unwrap().take_frame(&key);
    let eye = match cached {
        Some(d) => d,
        None => match eye_jpeg(&src, &cfg, t_ms) {
            Some(d) => {
                let d = Arc::new(d);
                state.lock().unwrap().put_frame(key, d.clone());
                d
            }
            None => {
                let _ =
                    req.respond(Response::from_string("frame failed").with_status_code(500));
                return;
            }
        },
    };

    let mode = qparam(query, "mode").unwrap_or_else(|| "proj".into());
    let (yaw, pitch) = match (qnum(query, "yaw"), qnum(query, "pitch")) {
        (Some(y), Some(p)) => (wrap180(y), p.clamp(-90.0, 90.0)),
        _ => (cfg.aim.yaw, cfg.aim.pitch),
    };
    let body = if mode == "equi" {
        Some((*eye).clone())
    } else {
        project_jpeg(&eye, &cfg, yaw, pitch)
    };
    match body {
        Some(data) => {
            let _ = req.respond(
                Response::from_data(data)
                    .with_header(header("Content-Type", "image/jpeg"))
                    .with_header(header("Cache-Control", "no-store")),
            );
        }
        None => {
            let _ = req.respond(Response::from_string("projection failed").with_status_code(500));
        }
    }
}

/// ~2 s of eye frames as ONE tiled sheet. Frames, not a projection: the page
/// loops them under the LIVE aim, so the motion check -- does the stroke stay in
/// frame WHILE it moves, which a still cannot answer -- survives a re-aim
/// without refetching. Twenty seeks into an 8K source would cost far more than
/// this single decode.
fn burst_request(req: tiny_http::Request, state: &Arc<Mutex<Session>>, query: &str) {
    let (src, cfg, dur) = {
        let s = state.lock().unwrap();
        let c = s.cur();
        (c.src.clone(), c.cfg.clone(), c.dur_ms)
    };
    let t_ms = match qnum(query, "t_ms") {
        Some(t) => t.clamp(0.0, (dur - 2100.0).max(0.0)),
        None => {
            let _ = req.respond(Response::from_string("bad t_ms").with_status_code(400));
            return;
        }
    };
    match burst_sheet(&src, &cfg, t_ms) {
        Some(data) => {
            let _ = req.respond(
                Response::from_data(data)
                    .with_header(header("Content-Type", "image/jpeg"))
                    .with_header(header("X-Cols", &BURST_COLS.to_string()))
                    .with_header(header("X-Rows", &BURST_ROWS.to_string()))
                    .with_header(header("X-Fps", &BURST_FPS.to_string()))
                    .with_header(header("Cache-Control", "no-store")),
            );
        }
        None => {
            let _ = req.respond(Response::from_string("burst failed").with_status_code(500));
        }
    }
}

/// The equirect eye at `t_ms`, as JPEG bytes. Fast (keyframe) seek: this is an
/// aiming preview, and a frame or two of imprecision is invisible against the
/// seconds an accurate seek into an 8K HEVC source would cost.
fn eye_jpeg(src: &Path, cfg: &Config, t_ms: f64) -> Option<Vec<u8>> {
    let mut vf: Vec<String> = Vec::new();
    if let Some(c) = cfg.crop_expr() {
        vf.push(c.to_string());
    }
    vf.push(format!("scale={PROJ_W}:-2"));
    // Software decode on purpose, matching vr_project.py's frame cache: one
    // frame does not amortize hardware-decoder setup, and a preview that
    // silently fails on a fussy driver is worse than one that is a second
    // slower. The transcode is where hardware decode is worth its risk.
    let out = Command::new("ffmpeg")
        .args(["-y", "-nostdin", "-loglevel", "error"])
        .args(["-ss", &format!("{:.3}", t_ms / 1000.0)])
        .arg("-i")
        .arg(src)
        .args(["-frames:v", "1", "-vf", &vf.join(",")])
        .args(["-c:v", "mjpeg", "-q:v", "3", "-f", "image2pipe", "-"])
        .stderr(Stdio::null())
        .output()
        .ok()?;
    (out.status.success() && !out.stdout.is_empty()).then_some(out.stdout)
}

/// The cached eye frame through ffmpeg's own v360 -- the A/B reference. Fed on
/// stdin, so the source is never seeked a second time.
fn project_jpeg(eye: &[u8], cfg: &Config, yaw: f64, pitch: f64) -> Option<Vec<u8>> {
    let w = PROJ_W - (PROJ_W % 2);
    let h = {
        let h = (w as f64 * cfg.out_h as f64 / cfg.out_w as f64).round() as u32;
        h.max(2) & !1
    };
    let mut child = Command::new("ffmpeg")
        .args(["-y", "-nostdin", "-loglevel", "error"])
        .args(["-f", "image2pipe", "-i", "pipe:0"])
        .args(["-frames:v", "1", "-vf", &cfg.v360_expr(yaw, pitch, w, h)])
        .args(["-c:v", "mjpeg", "-q:v", "3", "-f", "image2pipe", "-"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .ok()?;
    // The JPEG is bigger than a pipe buffer, so the write has to run alongside
    // the read -- feeding it inline would deadlock on the first full buffer.
    let mut stdin = child.stdin.take()?;
    let buf = eye.to_vec();
    std::thread::spawn(move || {
        let _ = stdin.write_all(&buf);
    });
    let out = child.wait_with_output().ok()?;
    (out.status.success() && !out.stdout.is_empty()).then_some(out.stdout)
}

fn burst_sheet(src: &Path, cfg: &Config, t_ms: f64) -> Option<Vec<u8>> {
    let n = BURST_COLS * BURST_ROWS;
    let tile_w = (PROJ_W / BURST_COLS).max(2) & !1;
    let mut vf: Vec<String> = Vec::new();
    if let Some(c) = cfg.crop_expr() {
        vf.push(c.to_string());
    }
    vf.push(format!("fps={BURST_FPS}"));
    vf.push(format!("scale={tile_w}:-2"));
    vf.push(format!("tile={BURST_COLS}x{BURST_ROWS}"));
    // a little more source than the tile needs, so it never flushes short
    let span = n as f64 / BURST_FPS * 1.25;
    let out = Command::new("ffmpeg")
        .args(["-y", "-nostdin", "-loglevel", "error"])
        .args(["-ss", &format!("{:.3}", t_ms / 1000.0)])
        .args(["-t", &format!("{span:.2}")])
        .arg("-i")
        .arg(src)
        .args(["-an", "-vf", &vf.join(",")])
        .args(["-frames:v", "1"])
        .args(["-c:v", "mjpeg", "-q:v", "4", "-f", "image2pipe", "-"])
        .stderr(Stdio::null())
        .output()
        .ok()?;
    (out.status.success() && !out.stdout.is_empty()).then_some(out.stdout)
}

// --------------------------------------------------------------------------

fn header(field: &str, value: &str) -> Header {
    Header::from_bytes(field.as_bytes(), value.as_bytes()).expect("static header")
}

fn json_response(v: &serde_json::Value) -> Response<std::io::Cursor<Vec<u8>>> {
    Response::from_data(v.to_string().into_bytes())
        .with_header(header("Content-Type", "application/json"))
}

fn open_browser(url: &str) {
    #[cfg(windows)]
    let r = Command::new("cmd").args(["/C", "start", "", url]).spawn();
    #[cfg(not(windows))]
    let r = Command::new("xdg-open").arg(url).spawn();
    let _ = r;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn v_fov_follows_the_output_aspect() {
        let c = Config::default(); // 90 deg across 16:9
        let v = c.auto_v_fov();
        assert!((v - 58.72).abs() < 0.1, "got {v}");
        // pinned wins over derived
        let c = Config { v_fov: Some(70.0), ..Config::default() };
        assert_eq!(c.v_fov_eff(), 70.0);
    }

    // The filter that goes in front of the normalize chain has to cut the right
    // eye out and hand v360 the aim that was saved. A wrong crop is a draft of
    // half a face and half a wall.
    #[test]
    fn the_transcode_prefix_crops_the_eye_and_carries_the_aim() {
        let c = Config {
            aim: Aim { yaw: -35.0, pitch: -12.5 },
            ..Config::default()
        };
        let f = c.filter_prefix(480);
        assert!(f.starts_with("crop=iw/2:ih:0:0,v360="), "got {f}");
        assert!(f.contains("input=hequirect"), "got {f}");
        assert!(f.contains("yaw=-35.0000"), "got {f}");
        assert!(f.contains("pitch=-12.5000"), "got {f}");
        // the encode spec's height at the aim's aspect, both even (libx264
        // refuses odd dimensions in yuv420p, so the rounding is not cosmetic)
        assert!(f.contains("w=852:h=480"), "got {f}");
    }

    #[test]
    fn mono_is_not_cropped_and_tb_takes_the_top() {
        let c = Config { layout: "mono".into(), ..Config::default() };
        assert!(c.filter_prefix(480).starts_with("v360="));
        let c = Config { layout: "tb".into(), ..Config::default() };
        assert!(c.filter_prefix(480).starts_with("crop=iw:ih/2:0:0,"));
    }

    // The range is the draft's whole clock: t0 is what the written funscript
    // gets shifted by, so an out-of-bounds or inverted range must clamp rather
    // than produce a negative shift.
    #[test]
    fn ranges_clamp_into_the_video() {
        let c = Config { range_ms: Some([1000.0, 5000.0]), ..Config::default() };
        assert_eq!(c.range(60_000.0), (1000.0, 5000.0));
        assert_eq!(c.t0_ms(60_000.0), 1000.0);
        // past the end
        let c = Config { range_ms: Some([1000.0, 90_000.0]), ..Config::default() };
        assert_eq!(c.range(60_000.0), (1000.0, 60_000.0));
        // inverted
        let c = Config { range_ms: Some([5000.0, 1000.0]), ..Config::default() };
        let (a, b) = c.range(60_000.0);
        assert!(a < b, "got {a}..{b}");
        // absent
        assert_eq!(Config::default().range(60_000.0), (0.0, 60_000.0));
        assert_eq!(Config::default().t0_ms(60_000.0), 0.0);
    }

    // The cache key exists so a re-aim never reuses the old aim's transcode.
    #[test]
    fn the_cache_key_moves_with_every_render_affecting_field() {
        let base = Config::default();
        let k = base.key();
        for c in [
            Config { aim: Aim { yaw: 1.0, pitch: 0.0 }, ..base.clone() },
            Config { aim: Aim { yaw: 0.0, pitch: 1.0 }, ..base.clone() },
            Config { layout: "tb".into(), ..base.clone() },
            Config { projection: "fisheye".into(), ..base.clone() },
            Config { h_fov: 100.0, ..base.clone() },
            Config { range_ms: Some([0.0, 1000.0]), ..base.clone() },
            Config { out_w: 1280, out_h: 720, ..base.clone() },
        ] {
            assert_ne!(k, c.key(), "key did not move");
        }
        // bookkeeping does NOT move it: marking a sidecar touched must not
        // throw away a transcode that is still correct
        assert_eq!(k, Config { touched: true, ..base }.key());
    }

    // The encoder squashes every clip into a square (`scale=enc_res:enc_res`),
    // so the viewport aspect IS the anamorphic factor the model sees. The
    // default has to be the corpus's 16:9, and a departure has to be visible --
    // it is the one setting on this page that can quietly put a draft
    // off-distribution while the preview still looks perfectly aimed.
    #[test]
    fn the_default_viewport_matches_the_corpus_anamorphic_shape() {
        let c = Config::default();
        let a = c.out_w as f64 / c.out_h as f64;
        assert!((a - Config::NATIVE_ASPECT).abs() < 1e-9, "default is {a}");
        assert!(c.aspect_warning().is_none());
        // the shapes a user might reasonably type, all still 16:9
        for (w, h) in [(1280u32, 720u32), (854, 480), (3840, 2160)] {
            let c = Config { out_w: w, out_h: h, ..Config::default() };
            assert!(c.aspect_warning().is_none(), "{w}x{h} warned");
        }
        // and the ones that change the squash
        for (w, h) in [(1080u32, 1080u32), (1440, 1080), (2560, 1080)] {
            let c = Config { out_w: w, out_h: h, ..Config::default() };
            assert!(c.aspect_warning().is_some(), "{w}x{h} did not warn");
        }
    }

    // The flat path scales a source to the spec height keeping its aspect, then
    // the encoder squares it. A VR viewport has to arrive at that same square
    // by the same route, or the one VR clip the model trained on is not the
    // shape the deploy path produces.
    #[test]
    fn a_default_viewport_reaches_the_encoder_as_a_16_9_source_would() {
        let f = Config::default().filter_prefix(480);
        // 16:9 at the spec height -- exactly what `scale=-2:480` makes of any
        // 16:9 flat source, so the downstream square-scale squashes identically
        assert!(f.contains("w=852:h=480"), "got {f}");
    }

    #[test]
    fn wrap_keeps_yaw_in_half_turns() {
        assert_eq!(wrap180(0.0), 0.0);
        assert_eq!(wrap180(190.0), -170.0);
        assert_eq!(wrap180(-190.0), 170.0);
        assert_eq!(wrap180(540.0), -180.0);
    }
}
