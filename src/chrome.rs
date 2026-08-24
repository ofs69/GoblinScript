//! The header pinned over the processing log: the mascot, the wordmark and the
//! corner band, in the top rows of the screen, animating for as long as the
//! goblins are working.
//!
//! It is the picker's header, still there after the picker has closed -- one
//! screen the app hands to itself, rather than a browser that dances and a
//! draft that does not. The startup report is cleared on the way in: it has
//! said everything it had to say, and a still drawing of a goblin sat above a
//! moving one only makes the moving one look like a mistake.
//!
//! Pinning is a SCROLL REGION (`DECSTBM`), not a redraw: the log below prints
//! and scrolls entirely inside its own margins, so the header rows are never
//! moved by it and nothing has to know how tall the log is. That also keeps the
//! single-line live display in `main` exactly as it was -- it erases and
//! repaints one line, at whatever row the region has left it on.
//!
//! Every write here is one locked, cursor-preserving burst (`DECSC`/`DECRC`),
//! because the live display is being painted from another thread against the
//! same stdout: interleaving mid-sequence would put a progress bar in the
//! goblin's ear.

use crate::mascot;
use std::io::{IsTerminal, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

/// Rows the header occupies at the top of the screen: the mascot's block, with
/// the rule he stands on as its last row.
pub const HEADER_H: u16 = 4;

/// Take the terminal's cursor away, and give it back. Every path that stops
/// pinning the header sends the second one -- including the one that leaves by
/// `exit` and runs no destructor -- because a cursor left hidden outlives the
/// process and belongs to the user's next shell.
const HIDE_CURSOR: &str = "\u{1b}[?25l";
const SHOW_CURSOR: &str = "\u{1b}[?25h";

/// The shortest screen worth splitting. Below this the log would be reduced to
/// a few rows to keep a decoration, which is the wrong trade -- the header is
/// simply skipped and the run looks exactly as it always did.
const MIN_H: u16 = HEADER_H + 8;

/// Whether the top rows are currently pinned. The live display below has to
/// know how many rows the LOG has, not how many the screen does, and the header
/// is the only thing that takes any away.
static PINNED: AtomicBool = AtomicBool::new(false);

/// Rows the log has to itself on a screen `screen_h` tall.
pub fn log_rows(screen_h: u16) -> u16 {
    if PINNED.load(Ordering::Relaxed) {
        screen_h.saturating_sub(HEADER_H)
    } else {
        screen_h
    }
}

pub struct Header {
    stop: Arc<AtomicBool>,
    handle: Option<std::thread::JoinHandle<()>>,
}

impl Header {
    /// Take the top of the screen and start animating, or `None` where that
    /// cannot be done: a redirected run (a log file has no top) and a terminal
    /// too short to spare the rows.
    pub fn begin() -> Option<Header> {
        if !std::io::stdout().is_terminal() {
            return None;
        }
        let (h, _w) = console::Term::stdout().size();
        if h < MIN_H {
            return None;
        }
        {
            // All of this before the caller prints a single line, or the log's
            // first lines land in rows the header is about to take.
            let mut out = std::io::stdout().lock();
            // Erase what is on the screen, but never the scrollback: the
            // startup report stays exactly one scroll away, which is where a
            // user pasting a bug report goes looking for it.
            let _ = write!(out, "\u{1b}[2J\u{1b}[H");
            // Pin the top rows, then open the log on the first row under them
            // -- setting a region homes the cursor, so the move is not optional.
            let _ = write!(out, "\u{1b}[{};{h}r", HEADER_H + 1);
            let _ = write!(out, "\u{1b}[{};1H", HEADER_H + 1);
            // And put the cursor away for as long as the header is up.
            //
            // `frame` paints by ADDRESSING the header rows and handing the
            // cursor back (`\x1b7` ... `\x1b8`), which is correct and still
            // leaves it standing in the header for the length of that burst.
            // A terminal is free to draw its cursor from wherever it has read
            // to, so several times a second it appears under the goblin and
            // goes again -- which is the flicker in the bottom-left of the
            // header. There is nothing for a cursor to mark on this screen
            // anyway: every key the draft takes is a hotkey, and nothing is
            // being typed into.
            let _ = write!(out, "{HIDE_CURSOR}");
            let _ = out.flush();
        }
        // the goblins are up before the first video is
        PINNED.store(true, Ordering::Relaxed);
        let mut pinned = h;
        paint(&mut pinned);
        let stop = Arc::new(AtomicBool::new(false));
        let s = stop.clone();
        let handle = std::thread::spawn(move || {
            // A frame at half the beat, so no pose the goblins strike is ever
            // skipped by the redraw clock.
            let frame = Duration::from_millis((mascot::BEAT_MS as u64 / 2).max(1));
            let mut pinned = h;
            while !s.load(Ordering::Relaxed) {
                paint(&mut pinned);
                std::thread::sleep(frame);
            }
        });
        Some(Header { stop, handle: Some(handle) })
    }
}

impl Drop for Header {
    fn drop(&mut self) {
        PINNED.store(false, Ordering::Relaxed);
        self.stop.store(true, Ordering::Relaxed);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
        let mut out = std::io::stdout().lock();
        // The whole screen goes back to whatever comes next. Releasing the
        // region homes the cursor, so it is saved and restored around the
        // release -- the painter has stopped and handed it back, which makes
        // the position under it the log's own, and the summary that prints next
        // carries straight on from the last stage line. The header stays where
        // it is until the output that follows scrolls it away.
        //
        // The cursor comes back with it: the screen after this one is a summary
        // and a picker, where a caret is the thing that says the app is waiting
        // on a person.
        let _ = write!(out, "\u{1b}7\u{1b}[r\u{1b}8{SHOW_CURSOR}");
        let _ = out.flush();
    }
}

/// Give the screen back from a path that cannot run `Drop` -- the second Ctrl-C
/// leaves by `exit`, and a shell inherited with a scroll region still set is a
/// shell that scrolls in the top four rows of itself forever. A shell inherited
/// with no cursor is the same kind of debt, so both are handed back here.
///
/// Safe to call when no header was ever pinned: showing a cursor that was never
/// hidden is what a terminal is already doing.
pub fn release(out: &mut impl Write) {
    PINNED.store(false, Ordering::Relaxed);
    let _ = write!(out, "\u{1b}[r{SHOW_CURSOR}");
}

/// One frame: re-assert the region if the terminal has been resized, then paint
/// the four rows and give the cursor straight back.
///
/// `pinned` is the screen height the current region was set for. Setting a
/// region homes the cursor, so it is re-sent only when the height has actually
/// changed rather than every frame.
fn paint(pinned: &mut u16) {
    let (h, w) = console::Term::stdout().size();
    let mut out = std::io::stdout().lock();
    frame(&mut out, w, h, pinned);
    let _ = out.flush();
}

/// The frame itself, written into `out` -- the escape sequence with none of the
/// stdout around it, which is the part worth holding a test against.
fn frame(out: &mut impl Write, w: u16, h: u16, pinned: &mut u16) {
    let rows = mascot::header_rows(w);
    let _ = write!(out, "\u{1b}7"); // the log's cursor, to be handed back intact
    if h != *pinned {
        let _ = write!(out, "\u{1b}[{};{}r", HEADER_H + 1, h.max(HEADER_H + 1));
        *pinned = h;
    }
    for (i, row) in rows.iter().enumerate() {
        // absolute rows: the header sits ABOVE the region, where nothing the
        // log does can reach it. `\x1b[K` clears the tail a narrower pose or a
        // dropped band leaves behind.
        let _ = write!(out, "\u{1b}[{};1H{row}\u{1b}[0m\u{1b}[K", i + 1);
    }
    let _ = write!(out, "\u{1b}8");
}

#[cfg(test)]
mod tests {
    use super::*;

    // The log is being painted from another thread against the same stdout, so
    // a frame that does not hand the cursor back exactly as it found it puts
    // the next progress bar wherever this left off. Save first, restore last,
    // every frame, no exceptions -- and the header itself only ever addressed
    // by absolute row, never by moving relative to the log's position.
    // The header paints by addressing its rows and handing the cursor back,
    // which leaves the cursor standing up there for the length of the burst --
    // several times a second, visibly, in the bottom-left of the header. So the
    // cursor is hidden for as long as the header is pinned, and the debt that
    // creates has exactly one rule: EVERY path that stops pinning gives it
    // back, including the one that leaves by `exit` and runs no destructor.
    #[test]
    fn releasing_hands_the_cursor_back_with_the_screen() {
        let mut buf: Vec<u8> = Vec::new();
        release(&mut buf);
        let s = String::from_utf8(buf).unwrap();
        assert!(s.contains("\u{1b}[r"), "the scroll region goes back: {s:?}");
        assert!(s.contains(SHOW_CURSOR), "the cursor goes back: {s:?}");
    }

    #[test]
    fn a_frame_gives_the_cursor_back() {
        let mut pinned = 0u16;
        let mut buf: Vec<u8> = Vec::new();
        frame(&mut buf, 80, 40, &mut pinned);
        let s = String::from_utf8(buf).unwrap();
        assert!(s.starts_with("\u{1b}7"), "a frame opens by saving the cursor: {s:?}");
        assert!(s.ends_with("\u{1b}8"), "a frame closes by restoring it: {s:?}");
        // no relative cursor motion anywhere in it
        for m in ['A', 'B', 'C', 'D'] {
            assert!(!s.contains(&format!("\u{1b}[{m}")), "relative move {m} in a frame");
        }
        for i in 1..=HEADER_H {
            assert!(s.contains(&format!("\u{1b}[{i};1H")), "row {i} is not addressed");
        }
    }

    // Setting the region homes the cursor, so it is sent when the height has
    // actually changed and not otherwise -- 8 times a second of homing the
    // cursor is a log that prints its next line in the header.
    #[test]
    fn the_region_is_set_once_per_size() {
        let region = format!("\u{1b}[{};40r", HEADER_H + 1);
        let render = |h: u16, pinned: &mut u16| {
            let mut buf: Vec<u8> = Vec::new();
            frame(&mut buf, 80, h, pinned);
            String::from_utf8(buf).unwrap()
        };
        let mut pinned = 0u16;
        assert!(render(40, &mut pinned).contains(&region), "the first frame pins the region");
        assert!(!render(40, &mut pinned).contains(&region), "the region is re-sent unchanged");
        // a resize re-pins it against the new height
        assert!(
            render(30, &mut pinned).contains(&format!("\u{1b}[{};30r", HEADER_H + 1)),
            "a resize does not re-pin the region"
        );
    }
}
