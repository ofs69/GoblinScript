//! Stopping a draft in flight.
//!
//! The console is read by the live display's render thread while a draft runs
//! (`main::render_loop`), so the user's Ctrl-C arrives as a KEY, not a signal --
//! raw mode is what makes M/V work mid-draft, and it takes the signal with it.
//! This is the flag that key sets and every long loop reads.
//!
//! Cancelling is always SAFE to do abruptly: nothing downstream of a stage
//! commits until that stage finishes (latents land in a `.part` file, the
//! funscript is written once), and the cache is built to be resumed. So the
//! loops check between chunks and unwind; they never have to unwind cleanly.

use std::sync::atomic::{AtomicUsize, Ordering};

/// How many times the user has asked to stop. Counted, not just flagged: the
/// first ask unwinds the draft, and a second one -- pressed because the first
/// looked ignored, which it will while a 20 s forward finishes -- exits on the
/// spot.
static ASKS: AtomicUsize = AtomicUsize::new(0);

/// Record an ask to stop and return how many there have now been.
pub fn request() -> usize {
    ASKS.fetch_add(1, Ordering::Relaxed) + 1
}

pub fn requested() -> bool {
    ASKS.load(Ordering::Relaxed) > 0
}

/// The error a cancelled stage returns, so `main` can tell "the user stopped
/// this" from "this failed" -- one is a clean exit, the other is a red line and
/// a non-zero code.
#[derive(Debug)]
pub struct Cancelled;

impl std::fmt::Display for Cancelled {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "stopped")
    }
}

impl std::error::Error for Cancelled {}

/// `?` this anywhere a long loop can afford to stop.
pub fn check() -> anyhow::Result<()> {
    if requested() {
        anyhow::bail!(Cancelled);
    }
    Ok(())
}

/// Was this error a cancellation rather than a failure?
pub fn is_cancel(e: &anyhow::Error) -> bool {
    e.downcast_ref::<Cancelled>().is_some()
}

#[cfg(test)]
mod tests {
    use super::*;

    // A cancel has to stay recognizable after a caller adds context to it.
    // Otherwise it reaches `main` as an ordinary failure and the user who
    // pressed Ctrl-C gets a red error line and a dead batch instead of a stop.
    #[test]
    fn a_cancel_survives_being_given_context() {
        let e = anyhow::Error::from(Cancelled).context("while encoding 000123");
        assert!(is_cancel(&e), "{e:#}");
    }

    // ...and nothing else is mistaken for one.
    #[test]
    fn an_ordinary_failure_is_not_a_cancel() {
        let e = anyhow::anyhow!("ffmpeg exited 1").context("while transcoding");
        assert!(!is_cancel(&e), "{e:#}");
    }

    // The count is what separates "stop at the next chunk" from "stop NOW",
    // and the second press has to be distinguishable even though the first
    // already set the flag.
    #[test]
    fn asks_are_counted_not_just_flagged() {
        assert!(!requested());
        assert_eq!(request(), 1);
        assert!(requested());
        assert_eq!(request(), 2);
        assert!(check().is_err());
    }
}
