//! Normalizing the NEXT video while the goblins work on this one.
//!
//! The pipeline's stages are not all on the same machine. `normalize` is
//! libx264 on the CPU; `shot cuts`, `encode` and `model` are the GPU. Run
//! strictly in sequence -- which is what a batch loop does -- each one idles
//! the other's hardware for its whole duration.
//!
//! That used to be a rounding error and is not any more. The encoder went from
//! ~13 s per minute of video to ~3.4 s (see `export_bundle.fuse_attention`),
//! while the transcode stayed at 3-6 s: the CPU stage is now the *larger* half
//! of a draft. Overlapping the two is free wall clock, and an overnight batch
//! -- what the picker exists for -- is where it pays.
//!
//! One video ahead, never more. The gain is already collected at one (the next
//! transcode has a whole draft to hide behind) and each one in flight is
//! another normalized copy on disk.
//!
//! A head start is only ever an OPTIMIZATION: every path here degrades to
//! "the main loop transcodes it itself". Failures are swallowed on purpose --
//! a video that cannot be transcoded must produce its error in the draft that
//! the user is watching, with that video's name on it, not from a thread that
//! ran ten minutes earlier.

use anyhow::{anyhow, Result};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::Duration;

use crate::bundle::Transcode;
use crate::cache::Cache;

/// Progress is published as permille so the waiting bar has something finer
/// than percent to draw.
const SCALE: f64 = 1000.0;

pub struct Job {
    progress: Arc<AtomicU64>,
    handle: JoinHandle<Result<()>>,
}

impl Job {
    /// Wait for the head start to finish, reporting its progress in permille.
    ///
    /// Returns the time spent WAITING HERE -- not the transcode's own run time,
    /// which mostly happened while the previous video was on the GPU and is
    /// exactly the part that cost nothing. A head start that landed in time
    /// returns ~0, and that is what the stage line reads.
    pub fn wait(self, mut on_progress: impl FnMut(u64)) -> Result<Duration> {
        let t = std::time::Instant::now();
        while !self.handle.is_finished() {
            on_progress(self.progress.load(Ordering::Relaxed));
            std::thread::sleep(Duration::from_millis(50));
        }
        let waited = t.elapsed();
        on_progress(SCALE as u64);
        // The thread's own Result: a cancel stays a cancel, a failure stays a
        // failure. Both are the caller's to report, under this video's name.
        self.handle
            .join()
            .map_err(|_| anyhow!("the transcoding goblin panicked"))??;
        Ok(waited)
    }
}

pub struct Prefetch {
    /// `None` when prefetching is off (`--no-transcode`: there is no transcode
    /// to run ahead) or when nothing is in flight.
    job: Mutex<Option<(PathBuf, Job)>>,
    enabled: bool,
    /// The run's decode dial, so a head start and the stage it stands in for
    /// are the same pass -- and land on the same cache entry either way.
    hw: crate::ffmpeg::HwAccel,
}

impl Prefetch {
    pub fn new(enabled: bool, hw: crate::ffmpeg::HwAccel) -> Self {
        Self { job: Mutex::new(None), enabled, hw }
    }

    /// Begin normalizing `video` in the background.
    ///
    /// A no-op if prefetching is off, something is already in flight, the
    /// source is unreadable, or the normalized copy is already on disk. None of
    /// those is an error: they all just mean the main loop does the work.
    ///
    /// `vr` is why the whole batch is aimed BEFORE any of it is drafted: this
    /// runs a video's transcode a whole draft early, and it cannot do that for
    /// a video whose aim is still an open question.
    pub fn start(
        &self,
        video: &Path,
        cache_root: &Path,
        spec: &Transcode,
        vr: Option<&crate::vr::Config>,
    ) {
        if !self.enabled || crate::cancel::requested() {
            return;
        }
        let mut slot = self.job.lock().unwrap();
        if slot.is_some() {
            return;
        }
        let cache = match Cache::open(cache_root, video, vr) {
            Ok(c) => c,
            Err(_) => return,
        };
        // the caller derived the spec's height for THIS source (norm_height)
        let norm = cache.norm_video(spec);
        if norm.exists() {
            return;
        }
        let progress = Arc::new(AtomicU64::new(0));

        let (v, s, p) = (video.to_path_buf(), spec.clone(), progress.clone());
        let vr = vr.cloned();
        let hw = self.hw;
        let handle = std::thread::spawn(move || -> Result<()> {
            let dur = crate::ffmpeg::duration_ms(&v)?;
            // the head start must measure the same timeline the main loop will
            // wait on -- for a trimmed VR source that is the range, not the file
            let total = match &vr {
                Some(c) => {
                    let (t0, t1) = c.range(dur);
                    t1 - t0
                }
                None => dur,
            };
            crate::ffmpeg::transcode(&v, &norm, &s, total, vr.as_ref(), hw, |f| {
                p.store((f * SCALE) as u64, Ordering::Relaxed);
            })
        });
        *slot = Some((
            video.to_path_buf(),
            Job { progress, handle },
        ));
    }

    /// Take the in-flight head start if it is for `video`.
    ///
    /// A job for some OTHER video is left alone: it was started for a video
    /// still ahead in the batch, and claiming it here would be a bug, not a
    /// cleanup.
    pub fn claim(&self, video: &Path) -> Option<Job> {
        let mut slot = self.job.lock().unwrap();
        match slot.as_ref() {
            Some((v, _)) if v == video => slot.take().map(|(_, j)| j),
            _ => None,
        }
    }
}

#[cfg(test)]
impl Prefetch {
    /// Park a job that just sleeps, so the bookkeeping can be tested without
    /// ffmpeg, a video file, or a GPU.
    fn park_if_free(&self, video: &Path) {
        if self.job.lock().unwrap().is_some() {
            return;
        }
        self.park(video, 0);
    }

    fn park(&self, video: &Path, ms: u64) {
        let mut slot = self.job.lock().unwrap();
        let progress = Arc::new(AtomicU64::new(0));
        let p = progress.clone();
        let handle = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(ms));
            p.store(SCALE as u64, Ordering::Relaxed);
            Ok(())
        });
        *slot = Some((
            video.to_path_buf(),
            Job { progress, handle },
        ));
    }
}

impl Drop for Prefetch {
    /// Never leave a transcode running behind the process.
    ///
    /// Dropping a `JoinHandle` detaches the thread, which here would mean an
    /// ffmpeg still writing into the cache after GoblinScript has exited. The
    /// only way to reach this with work in flight is a stopped batch -- the
    /// last video never starts a head start, so a batch that runs to the end
    /// has nothing pending -- and a stopped batch means the cancel flag is
    /// already set, which `ffmpeg::transcode` polls between progress lines. So
    /// this join is the few hundred milliseconds it takes that loop to notice,
    /// kill its child and delete the half-written file.
    fn drop(&mut self) {
        if let Some((_, job)) = self.job.lock().unwrap().take() {
            let _ = job.handle.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // The whole safety story is that a head start is claimed by the video it
    // was started FOR. Claiming another video's transcode would hand the draft
    // a normalized copy of the wrong film -- silently, since both are just
    // `norm<h>.mp4` in their own cache dir.
    #[test]
    fn a_job_is_only_claimed_by_its_own_video() {
        let pf = Prefetch::new(true, crate::ffmpeg::HwAccel::Off);
        pf.park(Path::new("b.mp4"), 0);
        assert!(pf.claim(Path::new("a.mp4")).is_none(), "claimed the wrong video");
        assert!(pf.claim(Path::new("b.mp4")).is_some());
        // and once taken it is gone, so a second video cannot claim it again
        assert!(pf.claim(Path::new("b.mp4")).is_none());
    }

    // One video ahead, never more: a second start while one is in flight must
    // not replace (and so orphan) the first.
    #[test]
    fn only_one_head_start_runs_at_a_time() {
        let pf = Prefetch::new(true, crate::ffmpeg::HwAccel::Off);
        pf.park(Path::new("b.mp4"), 0);
        pf.park_if_free(Path::new("c.mp4"));
        assert!(pf.claim(Path::new("c.mp4")).is_none(), "c displaced b");
        assert!(pf.claim(Path::new("b.mp4")).is_some());
    }

    // Waiting reports how long the caller actually waited, which is what the
    // "ready (ran during the last video)" line keys off. A job that finished
    // while the GPU was busy must read as ~0, not as its own run time.
    #[test]
    fn waiting_on_a_finished_job_costs_nothing() {
        let pf = Prefetch::new(true, crate::ffmpeg::HwAccel::Off);
        pf.park(Path::new("b.mp4"), 0);
        std::thread::sleep(Duration::from_millis(120));
        let waited = pf.claim(Path::new("b.mp4")).unwrap().wait(|_| {}).unwrap();
        assert!(waited < Duration::from_millis(100), "waited {waited:?}");
    }

    // `--no-transcode` has no transcode to run ahead, and `--no-prefetch` is
    // the measurement baseline. Both must leave the batch strictly sequential.
    #[test]
    fn disabled_never_starts_anything() {
        let pf = Prefetch::new(false, crate::ffmpeg::HwAccel::Off);
        pf.start(Path::new("b.mp4"), Path::new("."), &Transcode {
            height: 480, fps: 30.0, crf: 23, preset: "medium".into(),
        }, None);
        assert!(pf.claim(Path::new("b.mp4")).is_none());
    }
}
