//! The optional crop review: see what the goblins will be shown, and move it.
//!
//! `--crop-edit` puts the auto-crop plan in front of a person BEFORE the
//! expensive stage reads a single frame through it. The page plays the
//! normalized copy with each shot's rect drawn on it, and the rect is
//! draggable: a hand-drawn one replaces the attention's own for that shot, for
//! every shot, or nowhere at all.
//!
//! Three things make this cheap rather than a second pipeline:
//!
//! * **It runs between the probe and the encode.** The normalized copy already
//!   exists (the crop is applied to it), the plan already exists, and nothing
//!   downstream has been computed yet -- so a correction costs the person's
//!   attention and no GPU seconds at all. Correcting AFTER a draft would mean
//!   re-encoding the whole video.
//! * **The rect is already the page's language.** A plan carries fractions of
//!   the frame (`autocrop::Rect`), which is exactly what a browser overlay
//!   wants, so the drawing, the saving and the decode all read one number.
//! * **The correction outlives the recipe.** A hand-drawn plan is written to
//!   the same `autocrop.json` the probe writes, marked `manual`, and
//!   `autocrop::read_cached` then keeps it through a new checkpoint or a
//!   retuned constant. The aim was made against the picture, not against the
//!   recipe.
//!
//! What the page deliberately does NOT do is change WHEN a rect changes: the
//! segment starts are the detected cuts, and a rect that moved mid-shot would
//! be a camera move the source never had. It edits which rect a shot gets.

use anyhow::{Context, Result};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tiny_http::{Method, Response};

use crate::autocrop::{Plan, Rect, IDENTITY};
use crate::review::{header, json_response, open_browser, serve_file_threaded, video_ctype};

/// What the page edits and what the caller gets back.
struct Session {
    /// One entry per shot, in plan order: where it starts (frame), and the
    /// rect it is to be drafted with.
    segs: Vec<(usize, Rect)>,
    /// The probe's own rects for those same shots -- what "auto" restores.
    auto: Vec<(usize, Rect)>,
    /// A rect here was drawn by hand rather than voted for.
    hand: Vec<bool>,
    fps: f64,
    dur_ms: f64,
    /// The frame's own aspect (w/h), so the page can keep a drawn rect the
    /// shape of the source -- the shape the encoder's square squash expects.
    aspect: f64,
    done: bool,
}

impl Session {
    fn json(&self) -> serde_json::Value {
        let segs: Vec<serde_json::Value> = self
            .segs
            .iter()
            .zip(&self.auto)
            .zip(&self.hand)
            .map(|(((f, r), (_, a)), &hand)| {
                serde_json::json!({
                    "t_ms": *f as f64 / self.fps * 1000.0,
                    "rect": [r.0, r.1, r.2, r.3],
                    "auto": [a.0, a.1, a.2, a.3],
                    "hand": hand,
                })
            })
            .collect();
        serde_json::json!({
            "segs": segs,
            "dur_ms": self.dur_ms,
            "aspect": self.aspect,
            "cap": crate::autocrop::MIN_SIDE_FRAC,
        })
    }
}

/// Show `plan` for `video`, let the person correct it, and hand back what they
/// settled on. `None` when they left it exactly as the probe drew it -- the
/// caller then keeps the automatic plan, cache stamp and all.
///
/// `video` is the normalized copy: the picture the crop is actually applied
/// to, so a rect drawn here is a rect of the frames the encoder will read.
pub fn edit(
    video: &Path,
    plan: &Plan,
    dur_ms: f64,
    fps: f64,
    aspect: f64,
    open: bool,
    announce: &dyn Fn(String),
) -> Result<Option<Plan>> {
    let server = tiny_http::Server::http("127.0.0.1:0")
        .map_err(|e| anyhow::anyhow!("could not start the crop page: {e}"))?;
    let port = server
        .server_addr()
        .to_ip()
        .context("the crop page has no TCP address")?
        .port();
    let url = format!("http://127.0.0.1:{port}/");

    let state = Arc::new(Mutex::new(Session {
        segs: plan.segs.clone(),
        auto: if plan.auto.len() == plan.segs.len() {
            plan.auto.clone()
        } else {
            plan.segs.clone()
        },
        hand: vec![false; plan.segs.len()],
        fps,
        dur_ms,
        aspect,
        done: false,
    }));

    announce(format!(
        "  {} {}  {}",
        crate::t!("console.cropedit.label"),
        console::style(&url).cyan().bold(),
        console::style(crate::t!("console.cropedit.hint")).dim()
    ));
    if open {
        open_browser(&url);
    }

    // No console reading here: the draft's own display already owns the
    // keyboard (Ctrl-C included), so this stage waits on the page and on the
    // cancel flag that display sets.
    let path = video.to_path_buf();
    loop {
        crate::cancel::check()?;
        let Some(req) = server.recv_timeout(Duration::from_millis(100))? else {
            continue;
        };
        handle(req, &state, &path)?;
        if state.lock().unwrap().done {
            return Ok(finish(&state, plan));
        }
    }
}

/// The edited plan, or `None` when nothing was actually moved. `src` is the
/// plan that went in: the rects are the page's, everything else is still the
/// probe's -- the grid they were read on, and the escape the probe measured
/// (which the review page prints, and which a drawn rect does not re-measure).
fn finish(state: &Arc<Mutex<Session>>, src: &Plan) -> Option<Plan> {
    let s = state.lock().unwrap();
    if !s.hand.iter().any(|&h| h) {
        return None;
    }
    // consecutive equal rects are one decode segment again: the merge the
    // probe does for its own rects, redone now that some of them moved
    let mut segs: Vec<(usize, Rect)> = Vec::with_capacity(s.segs.len());
    for &(start, r) in &s.segs {
        match segs.last() {
            Some(&(_, last)) if same(last, r) => {}
            _ => segs.push((start, r)),
        }
    }
    Some(Plan {
        auto: s.auto.clone(),
        segs,
        manual: true,
        grid: src.grid,
        escape_share: src.escape_share,
        placed: 0,
    })
}

fn same(a: Rect, b: Rect) -> bool {
    (a.0 - b.0).abs() < 1e-9
        && (a.1 - b.1).abs() < 1e-9
        && (a.2 - b.2).abs() < 1e-9
        && (a.3 - b.3).abs() < 1e-9
}

fn handle(mut req: tiny_http::Request, state: &Arc<Mutex<Session>>, video: &PathBuf) -> Result<()> {
    let url = req.url().to_string();
    let path = url.split('?').next().unwrap_or("/").to_string();
    match (req.method().clone(), path.as_str()) {
        (Method::Get, "/") | (Method::Get, "/index.html") => {
            let _ = req.respond(
                Response::from_string(include_str!("cropedit.html"))
                    .with_header(header("Content-Type", "text/html; charset=utf-8")),
            );
        }
        (Method::Get, "/api/state") => {
            let v = state.lock().unwrap().json();
            let _ = req.respond(json_response(&v));
        }
        (Method::Get, "/api/lang") => {
            let _ = req.respond(json_response(&crate::lang::catalog_json()));
        }
        (Method::Get, "/video") => {
            serve_file_threaded(req, video.clone(), video_ctype(video));
        }
        (Method::Post, "/api/rects") => {
            let mut body = String::new();
            let _ = req.as_reader().read_to_string(&mut body);
            let v = apply(state, &body);
            let _ = req.respond(json_response(&v));
        }
        (Method::Post, "/api/done") => {
            state.lock().unwrap().done = true;
            let _ = req.respond(json_response(&serde_json::json!({"ok": true})));
        }
        _ => {
            let _ = req.respond(Response::from_string("not here").with_status_code(404));
        }
    }
    Ok(())
}

/// One posted correction: `{"i": <segment or -1 for all>, "rect": [x,y,w,h] or
/// null for "back to the probe's own"}`.
///
/// Everything about a posted rect is checked here rather than trusted: the
/// page is a page, and a rect that left the frame or shrank past the zoom cap
/// would reach ffmpeg as a crop filter.
fn apply(state: &Arc<Mutex<Session>>, body: &str) -> serde_json::Value {
    let mut s = state.lock().unwrap();
    let v: serde_json::Value = match serde_json::from_str(body) {
        Ok(v) => v,
        Err(_) => return serde_json::json!({"ok": false}),
    };
    let i = v.get("i").and_then(|x| x.as_i64()).unwrap_or(-1);
    let want: Option<Rect> = v.get("rect").and_then(|r| {
        let a = r.as_array()?;
        (a.len() == 4).then(|| {
            let n = |k: usize| a[k].as_f64().unwrap_or(f64::NAN);
            (n(0), n(1), n(2), n(3))
        })
    });
    let n = s.segs.len();
    let targets: Vec<usize> = if i < 0 { (0..n).collect() } else { vec![(i as usize).min(n - 1)] };
    for t in targets {
        match want {
            None => {
                s.segs[t].1 = s.auto[t].1;
                s.hand[t] = false;
            }
            Some(r) => {
                let Some(r) = legal(r) else { continue };
                s.segs[t].1 = r;
                s.hand[t] = true;
            }
        }
    }
    let v = s.json();
    drop(s);
    v
}

/// A rect the decode can be handed: inside the frame, no smaller than the zoom
/// cap on either axis, and made of real numbers. `None` refuses it outright --
/// the page keeps what it had rather than showing a rect nothing will honour.
fn legal(r: Rect) -> Option<Rect> {
    let (x, y, w, h) = r;
    if ![x, y, w, h].iter().all(|v| v.is_finite()) {
        return None;
    }
    let cap = crate::autocrop::MIN_SIDE_FRAC;
    let w = w.clamp(cap, 1.0);
    let h = h.clamp(cap, 1.0);
    let x = x.clamp(0.0, 1.0 - w);
    let y = y.clamp(0.0, 1.0 - h);
    Some(if w >= 1.0 && h >= 1.0 { IDENTITY } else { (x, y, w, h) })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A probe's plan and the page that opens on it -- one shot per rect,
    /// each starting 900 frames after the one before.
    fn opened(rects: &[Rect]) -> (Plan, Arc<Mutex<Session>>) {
        let segs: Vec<(usize, Rect)> =
            rects.iter().enumerate().map(|(i, &r)| (i * 900, r)).collect();
        let plan = Plan {
            auto: segs.clone(),
            segs: segs.clone(),
            manual: false,
            grid: 24,
            escape_share: 0.03,
            placed: 0,
        };
        let page = Arc::new(Mutex::new(Session {
            auto: segs.clone(),
            hand: vec![false; segs.len()],
            segs,
            fps: 30.0,
            dur_ms: 60_000.0,
            aspect: 16.0 / 9.0,
            done: false,
        }));
        (plan, page)
    }

    /// A page that posts nothing leaves the probe's plan exactly as it was --
    /// including its cache stamp, which a manual plan would have replaced.
    #[test]
    fn an_untouched_page_hands_back_no_plan() {
        let (src, s) = opened(&[(0.1, 0.1, 0.7, 0.7), IDENTITY]);
        assert!(finish(&s, &src).is_none());
    }

    #[test]
    fn a_drawn_rect_replaces_one_shot_and_survives_the_merge() {
        let (src, s) = opened(&[(0.1, 0.1, 0.7, 0.7), (0.1, 0.1, 0.7, 0.7), IDENTITY]);
        apply(&s, r#"{"i":1,"rect":[0.2,0.05,0.75,0.75]}"#);
        let p = finish(&s, &src).expect("a hand-drawn rect is a plan");
        assert!(p.manual);
        // three shots, three rects, none of them equal to its neighbour now
        assert_eq!(p.segs.len(), 3);
        assert!((p.segs[1].1 .0 - 0.2).abs() < 1e-12);
        assert_eq!(p.segs[1].0, 900);
        // and the probe's own rects are still there for the next page to show
        assert!((p.auto[1].1 .0 - 0.1).abs() < 1e-12);
    }

    #[test]
    fn one_rect_for_every_shot_merges_into_one_segment() {
        let (src, s) = opened(&[(0.1, 0.1, 0.7, 0.7), IDENTITY, (0.2, 0.2, 0.7, 0.7)]);
        apply(&s, r#"{"i":-1,"rect":[0.15,0.15,0.7,0.7]}"#);
        let p = finish(&s, &src).expect("a hand-drawn rect is a plan");
        assert_eq!(p.segs.len(), 1, "one rect throughout is one decode segment");
        assert_eq!(p.segs[0].0, 0);
    }

    /// "Auto" is a real undo: the shot goes back to the probe's rect AND stops
    /// counting as hand-drawn, so a page fully reset hands back nothing.
    #[test]
    fn auto_puts_a_shot_back_and_a_full_reset_hands_back_nothing() {
        let (src, s) = opened(&[(0.1, 0.1, 0.7, 0.7), IDENTITY]);
        apply(&s, r#"{"i":0,"rect":[0.2,0.2,0.7,0.7]}"#);
        assert!(finish(&s, &src).is_some());
        apply(&s, r#"{"i":0,"rect":null}"#);
        assert!(finish(&s, &src).is_none());
        assert!((s.lock().unwrap().segs[0].1 .0 - 0.1).abs() < 1e-12);
    }

    /// The page end to end, over its own socket: the state it publishes, a
    /// rect posted into it, and Done handing the plan back to the pipeline.
    /// The unit tests above reach past the server; this is the only thing that
    /// says the routes, the JSON and the finish are wired to each other.
    #[test]
    fn the_page_serves_its_state_takes_a_rect_and_finishes() {
        use std::io::{BufRead, BufReader, Write};
        // the page's loop reads the process-global stop flag; no test that
        // sets it may run beside this one
        let _guard = crate::cancel::TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner());
        crate::cancel::reset();
        let (plan, _) = opened(&[(0.1, 0.1, 0.7, 0.7), IDENTITY]);
        let (tx, rx) = std::sync::mpsc::channel::<String>();
        let worker = std::thread::spawn(move || {
            edit(Path::new("no-such-video.mp4"), &plan, 60_000.0, 30.0, 16.0 / 9.0, false, &|line| {
                let _ = tx.send(line);
            })
        });
        // the announce line carries the URL, which is where the port is
        let line = rx.recv_timeout(Duration::from_secs(5)).expect("the page announces itself");
        let addr = line
            .split_whitespace()
            .find_map(|w| {
                let w = console::strip_ansi_codes(w).to_string();
                w.strip_prefix("http://").map(|a| a.trim_end_matches('/').to_string())
            })
            .expect("an http address");

        // Lenient on purpose: the LAST request is Done, and the page tears
        // its server down the moment it has it -- a reset there is the stage
        // ending, not a failure.
        let ask = |req: &str| -> String {
            let mut s = std::net::TcpStream::connect(&addr).expect("the page is listening");
            if s.write_all(req.as_bytes()).is_err() {
                return String::new();
            }
            let mut r = BufReader::new(s);
            let mut len = 0usize;
            loop {
                let mut l = String::new();
                match r.read_line(&mut l) {
                    Ok(0) | Err(_) => return String::new(),
                    Ok(_) => {}
                }
                if let Some(n) = l.to_ascii_lowercase().strip_prefix("content-length:") {
                    len = n.trim().parse().unwrap_or(0);
                }
                if l.trim().is_empty() {
                    break;
                }
            }
            let mut body = vec![0u8; len];
            match std::io::Read::read_exact(&mut r, &mut body) {
                Ok(()) => String::from_utf8_lossy(&body).to_string(),
                Err(_) => String::new(),
            }
        };
        let post = |path: &str, body: &str| {
            format!(
                "POST {path} HTTP/1.1\r\nHost: x\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            )
        };

        let state: serde_json::Value =
            serde_json::from_str(&ask("GET /api/state HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n"))
                .expect("state is json");
        assert_eq!(state["segs"].as_array().unwrap().len(), 2);
        assert_eq!(state["cap"].as_f64().unwrap(), crate::autocrop::MIN_SIDE_FRAC);
        assert!(!state["segs"][0]["hand"].as_bool().unwrap());

        let after: serde_json::Value =
            serde_json::from_str(&ask(&post("/api/rects", r#"{"i":1,"rect":[0.2,0.2,0.7,0.7]}"#)))
                .expect("the post answers with the new state");
        assert!(after["segs"][1]["hand"].as_bool().unwrap());
        assert!((after["segs"][1]["rect"][0].as_f64().unwrap() - 0.2).abs() < 1e-9);

        ask(&post("/api/done", "{}"));
        let got = worker.join().expect("the page thread ends").expect("no error");
        let got = got.expect("a drawn rect is a plan");
        assert!(got.manual);
        assert_eq!(got.segs.len(), 2);
        assert!((got.segs[1].1 .0 - 0.2).abs() < 1e-9);
    }

    /// The page is a page: every posted rect is checked before it can become a
    /// crop filter.
    #[test]
    fn an_impossible_rect_is_refused_or_brought_back_in() {
        assert_eq!(legal((0.0, 0.0, 1.0, 1.0)), Some(IDENTITY));
        assert!(legal((f64::NAN, 0.0, 0.5, 0.5)).is_none());
        // past the zoom cap on either axis: lifted to it
        let r = legal((0.4, 0.4, 0.2, 0.2)).unwrap();
        assert!((r.2 - crate::autocrop::MIN_SIDE_FRAC).abs() < 1e-12);
        // off the frame: brought back inside it, whole
        let r = legal((0.9, -0.5, 0.8, 0.8)).unwrap();
        assert!(r.0 + r.2 <= 1.0 + 1e-12 && r.1 >= 0.0);
    }
}
