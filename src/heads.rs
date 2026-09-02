//! The trained head over the latent rows -> the tracks composed styling reads.
//!
//! **The head runs INSIDE the encode stage**, chunk by chunk, on rows the
//! encoder wrote seconds ago. There is no separate model pass: the head is
//! cheap next to the encoder (a TCN over 1024 rows against 16 windows of a
//! ViT-B), so folding it in costs a hitch every ~34 s of video and saves a
//! stage, a second read of the cache, and the wait for both.
//!
//! Three things here are not free choices, they are what the model was trained
//! and evaluated with:
//!
//! * **Chunking.** The timeline is forwarded in 1024-row chunks that run
//!   straight across cuts (the cut-flag channel tells the model where the seams
//!   are), each padded with `man.ctx` rows of real timeline on both sides that
//!   are forwarded and thrown away -- kept rows carry a warm receptive field
//!   and a warm envelope history instead of a chunk edge's zero padding. A
//!   chunk shorter than 2 s carries no usable context, so it is re-forwarded
//!   with a 30-row lookback and only its own rows are kept -- which is why
//!   `Heads` holds that much history behind the rows it has forwarded.
//! * **The envelope is autoregressive.** Its context buffer reseeds to zero at
//!   the start of every FORWARD (inside the discarded ctx prefix), exactly as
//!   the Python loop does inside `model.forward`. This is a stepped graph
//!   driven from Rust, which makes it the same computation, not an
//!   approximation of it.
//! * **The chunk boundaries do not move.** They are counted from row 0 and the
//!   short-tail lookback is decided at the end, so the chunks a streamed run
//!   forwards are the chunks a whole-cache run would have -- the tracks are
//!   identical either way, which is the only reason streaming is allowed to be
//!   the way this runs. The ctx lookahead only delays WHEN a chunk can be
//!   forwarded, never where its boundaries sit.
//!
//! The forward runs on its OWN THREAD, and that is not a micro-optimization.
//! The head has no GPU kernel it can use (`bundle::head_session`), so a chunk
//! is ~0.5 s of CPU -- and run inline it was 0.5 s in which the encode thread
//! was not feeding the encoder, with the graphics card measurably parked at 1%
//! for the whole hitch. Rows go to the head through a queue instead, and the
//! encoder keeps working through the forward. `Heads` is the handle; `Decode`
//! is the same bookkeeping as before, running behind it.
//!
//! The queue changes no output. One worker takes the rows in the order they
//! were pushed and the tracks come back from `finish`, so the chunks, their
//! boundaries and their envelope histories are what a single thread produced.
//! What it does change is WHEN a failure is reported: a chunk that fails does
//! so at the next `push` or at `finish` rather than the instant it happened.

use anyhow::{Context, Result};
use indicatif::ProgressBar;
use ort::session::Session;
use std::fs::File;
use std::io::Read;
use std::sync::mpsc::{sync_channel, Receiver, SyncSender};
use std::thread::JoinHandle;

use crate::bundle::{ort_err, Manifest};
use crate::encode::Latents;
use crate::style::Tracks;

/// One chunk's outputs, in the head's own units and one value per OWNED row
/// (the lookback rows a short tail re-forwards are already dropped).
struct ChunkOut {
    vmarg: Vec<f64>,
    level: Vec<f64>,
    blo: Vec<f64>,
    bhi: Vec<f64>,
    eg: Vec<f64>,
    /// Empty where the bundle carries no such head.
    ptop: Vec<f64>,
    pbot: Vec<f64>,
    rtop: Vec<f64>,
    rbot: Vec<f64>,
    /// Local period, seconds per row; empty without the head.
    period: Vec<f64>,
    conf: Vec<f64>,
}

/// What the head's thread is sent.
enum Msg {
    /// Latent rows in video order, in a buffer to be handed back when spent.
    Rows(Vec<i8>),
    /// The clip's true length and its shot edges -- everything `finish` needs,
    /// and the last thing the thread ever receives.
    Finish(usize, Vec<usize>),
}

/// Rows per queued message. A bigger push is split into these, so the queue's
/// depth is a fixed amount of memory whether the rows arrive a window at a time
/// from the encoder or a whole chunk at a time off a cache.
const PUSH_ROWS: usize = 128;

/// How far the encoder may run ahead of the head before it has to wait.
///
/// A chunk forward is ~0.5 s of CPU, and the encoder makes a few hundred rows
/// in that time, so this is roughly double the slack it takes to cover one.
/// Past that the pushing thread blocks -- which is exactly what it did when the
/// forward was inline, so a head that cannot keep up costs what it always cost.
const QUEUE: usize = 4;

/// The head and the envelope decode, fed latent rows in video order.
///
/// Rows go in through `push` as the encoder makes them (or as the cache is
/// read back); `finish` hands over the tracks. The work happens on the thread
/// this owns, so a `push` costs a memcpy and the encoder carries on.
pub struct Heads {
    /// `None` once the thread has been handed its `Finish`, or given up on.
    rows: Option<SyncSender<Msg>>,
    /// What a row weighs, so a push can be split on a row boundary.
    row_bytes: usize,
    /// Row buffers coming back from the thread, to be filled again rather than
    /// allocated (and first-touched) once per window for the length of a film.
    spare: Receiver<Vec<i8>>,
    worker: Option<JoinHandle<Result<Tracks>>>,
}

impl Heads {
    pub fn new(head: Session, env: Session, man: &Manifest, cut_rows: Vec<usize>) -> Self {
        let row_bytes = man.dim * man.grid * man.grid;
        let (rows_tx, rows_rx) = sync_channel::<Msg>(QUEUE);
        let (spare_tx, spare_rx) = sync_channel::<Vec<i8>>(QUEUE);
        let man = man.clone();
        let worker = std::thread::spawn(move || -> Result<Tracks> {
            let mut dec = Decode::new(head, env, man, cut_rows);
            while let Ok(msg) = rows_rx.recv() {
                match msg {
                    Msg::Rows(v) => {
                        dec.push(&v)?;
                        // the buffer is spent the moment its rows are copied
                        // into the decode's own; hand it straight back
                        let _ = spare_tx.try_send(v);
                    }
                    Msg::Finish(n_rows, edges) => return dec.finish(n_rows, &edges),
                }
            }
            // the handle went away without asking for tracks: a cancelled or
            // failed draft, and nobody is waiting on this
            anyhow::bail!("the head goblin was dismissed before the clip ended")
        });
        Self { rows: Some(rows_tx), row_bytes, spare: spare_rx, worker: Some(worker) }
    }

    /// Hand over freshly made latent rows, in video order.
    ///
    /// Blocks only when the head is a whole queue behind whoever is pushing.
    pub fn push(&mut self, rows: &[i8]) -> Result<()> {
        for part in rows.chunks(PUSH_ROWS * self.row_bytes) {
            let mut buf = self.spare.try_recv().unwrap_or_default();
            buf.clear();
            buf.extend_from_slice(part);
            match self.rows.as_ref().map(|tx| tx.send(Msg::Rows(buf))) {
                Some(Ok(())) => {}
                // the thread is gone, which means a chunk of ITS work failed --
                // that error is the one worth reporting, not "the queue closed"
                _ => return Err(self.stopped()),
            }
        }
        Ok(())
    }

    /// Forward the rows left over and hand back the finished tracks.
    ///
    /// `n_rows` is the clip's true length -- the tail chunk's lookback is
    /// decided from it, which is the one decision that could not be made while
    /// rows were still arriving.
    pub fn finish(mut self, n_rows: usize, shot_edges: &[usize]) -> Result<Tracks> {
        let sent = self
            .rows
            .take()
            .map(|tx| tx.send(Msg::Finish(n_rows, shot_edges.to_vec())));
        if !matches!(sent, Some(Ok(()))) {
            return Err(self.stopped());
        }
        let Some(worker) = self.worker.take() else { return Err(self.stopped()) };
        match worker.join() {
            Ok(tracks) => tracks,
            Err(_) => anyhow::bail!("the head goblin panicked"),
        }
    }

    /// Why the thread is not there any more, as an error to report in its place.
    fn stopped(&mut self) -> anyhow::Error {
        self.rows = None;
        match self.worker.take().map(|w| w.join()) {
            Some(Ok(Err(e))) => e,
            Some(Err(_)) => anyhow::anyhow!("the head goblin panicked"),
            _ => anyhow::anyhow!("the head goblin stopped early"),
        }
    }
}

impl Drop for Heads {
    /// Never leave the head running behind the draft. Dropping the queue is
    /// what tells the thread to stop, and the join is the moment it takes to
    /// notice -- at most one chunk forward, and `cancel::check` cuts that short
    /// on a stopped draft.
    fn drop(&mut self) {
        self.rows = None;
        if let Some(w) = self.worker.take() {
            let _ = w.join();
        }
    }
}

/// The bookkeeping behind `Heads`: the chunking, the sessions, and the outputs
/// as they accumulate. Lives on the head's thread and is touched by nothing
/// else, which is why it can own its sessions outright.
struct Decode {
    head: Session,
    env: Session,
    man: Manifest,
    /// The rows carrying a cut flag, ascending (`boundaries::cut_rows`).
    cut_rows: Vec<usize>,

    row_bytes: usize,
    /// Rows held in memory, starting at `buf_start`: everything not yet
    /// forwarded, behind it the lookback a short final chunk may need.
    buf: Vec<i8>,
    buf_start: usize,
    /// The next row whose outputs are still owed -- the start of the chunk
    /// after the last one forwarded.
    next: usize,

    vmarg: Vec<f64>,
    level: Vec<f64>,
    blo: Vec<f64>,
    bhi: Vec<f64>,
    eg: Vec<f64>,
    ptop: Vec<f64>,
    pbot: Vec<f64>,
    rtop: Vec<f64>,
    rbot: Vec<f64>,
    /// Local period, seconds per row; empty without the head.
    period: Vec<f64>,
    conf: Vec<f64>,
    /// The confidence head is always present in a current bundle, but an older
    /// one exported before it has no such output -- detect it once and skip the
    /// read rather than crash.
    has_conf: bool,
}

impl Decode {
    fn new(head: Session, env: Session, man: Manifest, cut_rows: Vec<usize>) -> Self {
        let has_conf = head.outputs().iter().any(|o| o.name() == "conf");
        let row_bytes = man.dim * man.grid * man.grid;
        Self {
            head,
            env,
            man,
            cut_rows,
            row_bytes,
            buf: Vec::new(),
            buf_start: 0,
            next: 0,
            vmarg: Vec::new(),
            level: Vec::new(),
            blo: Vec::new(),
            bhi: Vec::new(),
            eg: Vec::new(),
            ptop: Vec::new(),
            pbot: Vec::new(),
            rtop: Vec::new(),
            rbot: Vec::new(),
            period: Vec::new(),
            conf: Vec::new(),
            has_conf,
        }
    }

    fn buffered_rows(&self) -> usize {
        self.buf_start + self.buf.len() / self.row_bytes
    }

    /// Take freshly made latent rows, in video order. Every whole chunk whose
    /// ctx lookahead they complete is forwarded on the spot.
    fn push(&mut self, rows: &[i8]) -> Result<()> {
        self.buf.extend_from_slice(rows);
        while self.buffered_rows() >= self.next + self.man.chunk + self.man.ctx {
            let (s, e) = (self.next, self.next + self.man.chunk);
            let fs = s.saturating_sub(self.man.ctx).max(self.buf_start);
            let fe = e + self.man.ctx;
            self.forward(fs, s, e, fe)?;
            self.next = e;
            self.drop_history();
        }
        Ok(())
    }

    /// Everything before the lookback the NEXT forward could ask for (its ctx
    /// prefix, or the short-tail min_chunk window) is spent -- the buffer is
    /// the only thing here that would otherwise grow with the film's length.
    fn drop_history(&mut self) {
        let keep_from = self
            .next
            .saturating_sub(self.man.min_chunk.max(self.man.ctx));
        if keep_from > self.buf_start {
            self.buf.drain(..(keep_from - self.buf_start) * self.row_bytes);
            self.buf_start = keep_from;
        }
    }

    /// Forward rows `fs..fe`, keeping the outputs for `s..e`. The rows outside
    /// `s..e` are context -- the short-tail lookback and the ctx padding -- and
    /// recompute identically wherever forwards overlap.
    fn forward(&mut self, fs: usize, s: usize, e: usize, fe: usize) -> Result<()> {
        crate::cancel::check()?;
        let (n, off) = (fe - fs, s - fs);
        let keep = e - s;
        let a = (fs - self.buf_start) * self.row_bytes;
        let x = &self.buf[a..a + n * self.row_bytes];
        let cut = crate::boundaries::cut_flags(&self.cut_rows, fs, fe);
        let out = run_chunk(
            &mut self.head, &mut self.env, &self.man, x, &cut, n, off, keep, self.has_conf, fs,
        )?;

        self.vmarg.extend_from_slice(&out.vmarg);
        self.level.extend_from_slice(&out.level);
        self.blo.extend_from_slice(&out.blo);
        self.bhi.extend_from_slice(&out.bhi);
        self.eg.extend_from_slice(&out.eg);
        self.ptop.extend_from_slice(&out.ptop);
        self.pbot.extend_from_slice(&out.pbot);
        self.rtop.extend_from_slice(&out.rtop);
        self.rbot.extend_from_slice(&out.rbot);
        self.period.extend_from_slice(&out.period);
        self.conf.extend_from_slice(&out.conf);
        Ok(())
    }

    /// Forward the rows left over and hand back the finished tracks.
    ///
    /// `n_rows` is the clip's true length -- the tail chunk's lookback is
    /// decided here, which is the one decision that could not be made while
    /// rows were still arriving.
    fn finish(mut self, n_rows: usize, shot_edges: &[usize]) -> Result<Tracks> {
        // whole chunks the ctx lookahead was still holding back: now that the
        // clip's end is known, forward them with the lookahead they have
        while n_rows > self.next + self.man.chunk {
            let (s, e) = (self.next, self.next + self.man.chunk);
            let fs = s.saturating_sub(self.man.ctx).max(self.buf_start);
            let fe = (e + self.man.ctx).min(n_rows);
            self.forward(fs, s, e, fe)?;
            self.next = e;
            self.drop_history();
        }
        if n_rows > self.next {
            let (s, e) = (self.next, n_rows);
            let mc = self.man.min_chunk;
            // a short tail still owns rows: forward it with a lookback window
            // (the overlapping rows recompute identically) and write only its own
            let fs = if e - s >= mc { s } else { s.saturating_sub(mc - (e - s)) };
            let fs = fs.saturating_sub(self.man.ctx).max(self.buf_start);
            // less than 2 s of signal in the whole clip -- nothing to forward,
            // and the rows stay unknown
            if e - fs >= mc {
                self.forward(fs, s, e, e)?;
                self.next = e;
            }
        }
        // rows no chunk covered read as unknown, and `tracks_of` gives each
        // head its own neutral value for them
        let nan = f64::NAN;
        for v in [
            &mut self.vmarg, &mut self.level, &mut self.blo, &mut self.bhi, &mut self.eg,
        ] {
            v.resize(n_rows, nan);
        }
        if self.man.heads.plat {
            self.ptop.resize(n_rows, nan);
            self.pbot.resize(n_rows, nan);
        }
        if self.man.heads.rev {
            self.rtop.resize(n_rows, nan);
            self.rbot.resize(n_rows, nan);
        }
        if self.man.heads.period {
            self.period.resize(n_rows, nan);
        }
        if self.has_conf {
            self.conf.resize(n_rows, nan);
        }

        // The last predicted frame of every shot sits on a static tail pair (zero
        // observed motion). Holding the previous velocity there keeps a cut from
        // minting a phantom stroke -- on the RAW phase track, before scaling, which
        // is where jepa_infer does it.
        hold_shot_tails(&mut self.vmarg, shot_edges);
        Ok(tracks_of(
            &self.vmarg, &self.level, &self.blo, &self.bhi, &self.eg,
            &self.ptop, &self.pbot, &self.rtop, &self.rbot, &self.period, self.conf,
            self.man.v_std,
        ))
    }
}

/// The flow envelope's base sample for one ABSOLUTE row, uniform on
/// `[0, hi)` -- `common.env_base_draw` in Rust.
///
/// The head's sampler is an ODE, so its output is a pure function of this
/// draw, which makes the draw the one place this decoder and `jepa_infer`
/// could disagree. Keying it on the absolute row is what lets both reach
/// the same sample without exchanging state: a decode chunk that reseeds
/// its AR buffer does not reseed this. SplitMix64's finalizer, then the
/// top 53 bits as the mantissa -- integer mixing and one multiply, with no
/// `ln` or `cos` to disagree in the last few ULP.
fn env_base_draw(row: u64, seed: u64, hi: f64) -> f32 {
    const G: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut z = row
        .wrapping_add(seed.wrapping_mul(G))
        .wrapping_add(G);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    (((z >> 11) as f64) * (-53f64).exp2() * hi) as f32
}

/// One chunk through the head, then the envelope stepped over its rows.
/// `off..off + keep` is the owned slice; everything outside it is context.
#[allow(clippy::too_many_arguments)]
fn run_chunk(
    head: &mut Session,
    env: &mut Session,
    man: &Manifest,
    x: &[i8],
    cut: &[f32],
    n: usize,
    off: usize,
    keep: usize,
    has_conf: bool,
    // the chunk's first ABSOLUTE row: the flow envelope's base draw is
    // keyed to it, so a chunk has to say where it starts
    fs: usize,
) -> Result<ChunkOut> {
    let xs = vec![1i64, n as i64, man.dim as i64, man.grid as i64, man.grid as i64];
    let xt = ort::value::TensorRef::from_array_view((xs, x)).map_err(ort_err)?;
    let ct = ort::value::TensorRef::from_array_view((vec![1i64, n as i64], cut))
        .map_err(ort_err)?;
    let out = head
        .run(ort::inputs!["x_i8" => xt, "cut" => ct])
        .map_err(ort_err)?;

    let take = |name: &str| -> Result<Vec<f64>> {
        let (_s, v) = out[name].try_extract_tensor::<f32>().map_err(ort_err)?;
        Ok(v[off..off + keep].iter().map(|&x| x as f64).collect())
    };
    let pair = |on: bool, a: &str, b: &str| -> Result<(Vec<f64>, Vec<f64>)> {
        Ok(if on { (take(a)?, take(b)?) } else { (Vec::new(), Vec::new()) })
    };
    let (ptop, pbot) = pair(man.heads.plat, "plat_top", "plat_bot")?;
    let (rtop, rbot) = pair(man.heads.rev, "rev_top", "rev_bot")?;
    let period = if man.heads.period { take("period")? } else { Vec::new() };

    // envelope: one step per row, buffer reseeded at the chunk boundary
    let (hshape, h) = out["h"].try_extract_tensor::<f32>().map_err(ort_err)?;
    let c = hshape[1] as usize;
    let w = hshape[2] as usize;
    let mut buf = vec![0.0f32; man.env_ctx];
    let mut ht = vec![0.0f32; c];
    let mut eg = Vec::with_capacity(keep);
    for i in 0..w {
        for j in 0..c {
            ht[j] = h[j * w + i]; // (1, C, W), row-major
        }
        let hv = ort::value::TensorRef::from_array_view((vec![1i64, c as i64, 1], &ht[..]))
            .map_err(ort_err)?;
        let bv = ort::value::TensorRef::from_array_view((vec![1i64, man.env_ctx as i64], &buf[..]))
            .map_err(ort_err)?;
        let eo = if man.env_flow {
            let e0 = [env_base_draw((fs + i) as u64, man.env_seed, man.env_base_hi)];
            let ev0 = ort::value::TensorRef::from_array_view((vec![1i64, 1], &e0[..]))
                .map_err(ort_err)?;
            env.run(ort::inputs!["h_t" => hv, "buf" => bv, "e0" => ev0])
                .map_err(ort_err)?
        } else {
            env.run(ort::inputs!["h_t" => hv, "buf" => bv])
                .map_err(ort_err)?
        };
        let (_s, ev) = eo["env"].try_extract_tensor::<f32>().map_err(ort_err)?;
        let et = ev[0];
        if i >= off && i < off + keep {
            eg.push(et as f64);
        }
        buf.rotate_right(1);
        buf[0] = et;
    }

    Ok(ChunkOut {
        vmarg: take("vmarg")?,
        level: take("level")?,
        blo: take("ext_lo")?,
        bhi: take("ext_hi")?,
        eg,
        ptop,
        pbot,
        rtop,
        rbot,
        period,
        conf: if has_conf { take("conf")? } else { Vec::new() },
    })
}

/// Read a latent cache back through the head, for the run that already has one.
///
/// Same chunking, same viewport: the rows come off a file instead of the GPU,
/// and everything downstream cannot tell the difference.
pub fn stream_cache(
    lat: &Latents,
    heads: &mut Heads,
    mask: Option<Session>,
    man: &Manifest,
    pb: &ProgressBar,
) -> Result<()> {
    let mut f = File::open(&lat.path).context("latent cache vanished")?;
    let view = crate::viz::Viewport::spawn(mask, man.dim, man.grid);
    let mut raw = vec![0u8; man.chunk * lat.row_bytes];
    let mut done = 0usize;
    while done < lat.rows {
        let n = man.chunk.min(lat.rows - done);
        let need = n * lat.row_bytes;
        f.read_exact(&mut raw[..need])?;
        let rows: &[i8] =
            unsafe { std::slice::from_raw_parts(raw.as_ptr() as *const i8, need) };
        view.publish(&rows[(n / 2) * lat.row_bytes..(n / 2 + 1) * lat.row_bytes]);
        heads.push(rows)?;
        done += n;
        pb.set_position(done as u64);
    }
    Ok(())
}

/// The head's raw per-row outputs -> the tracks styling reads: the velocity
/// scaling, and one neutral value per head for a row nothing forwarded.
#[allow(clippy::too_many_arguments)]
fn tracks_of(
    vmarg: &[f64],
    level: &[f64],
    blo: &[f64],
    bhi: &[f64],
    eg: &[f64],
    ptop: &[f64],
    pbot: &[f64],
    rtop: &[f64],
    rbot: &[f64],
    period: &[f64],
    conf: Vec<f64>,
    v_std: f64,
) -> Tracks {
    let fill = |v: &[f64], d: f64| -> Vec<f64> {
        v.iter().map(|&x| if x.is_finite() { x } else { d }).collect()
    };
    let scaled = |v: &[f64]| -> Vec<f64> {
        v.iter().map(|&x| if x.is_finite() { x * v_std } else { 0.0 }).collect()
    };
    // NaN probability rows are "no reversal here", exactly like the dwells
    let two = |a: &[f64], b: &[f64]| -> Option<(Vec<f64>, Vec<f64>)> {
        (!a.is_empty()).then(|| (fill(a, 0.0), fill(b, 0.0)))
    };
    Tracks {
        // confidence is already 0..1; keep NaN as "unknown" for the page, and
        // empty where the bundle has no such head
        conf,
        vmarg: scaled(vmarg),
        level: fill(level, 50.0),
        env: scaled(eg),
        band: Some((fill(blo, 0.0), fill(bhi, 100.0))),
        plat: two(ptop, pbot),
        rev: two(rtop, rbot),
        // NaN stays NaN: the decode reads an unforwarded row as a hold
        period: period.to_vec(),
    }
}

fn hold_shot_tails(vmarg: &mut [f64], shot_edges: &[usize]) {
    for w in shot_edges.windows(2) {
        let (lo, hi) = (w[0], w[1]);
        if hi - lo >= 2 && vmarg[hi - 2].is_finite() {
            vmarg[hi - 1] = vmarg[hi - 2];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::env_base_draw;

    /// The base draw is the one place this decoder and `jepa_infer` could
    /// integrate a different ODE, and nothing else in either language would
    /// notice: the envelope would still be smooth, still bounded, still
    /// plausible, and only the artifact columns would drift. So the two
    /// implementations are pinned to each other by VALUE here, cut from
    /// `common.env_base_draw` -- the same way `grid_check` binds the styling
    /// constants the two languages share.
    #[test]
    fn the_base_draw_matches_common_env_base_draw() {
        const ROWS: [u64; 7] = [0, 1, 2, 17, 1000, 123456, 999999];
        const HI: f64 = 6.0;
        const REF: [[f32; 7]; 3] = [
            [5.299864849, 3.399369451, 3.547138405, 3.012127139,
             1.409063295, 1.357027354, 2.671600102],
            [2.589167982, 4.474690544, 4.494898103, 2.348602075,
             4.886225764, 1.545966942, 4.033623147],
            [0.158602630, 5.826016522, 3.573828488, 1.969978886,
             4.640479842, 3.537259637, 4.671366349],
        ];
        for (seed, want) in REF.iter().enumerate() {
            for (row, &w) in ROWS.iter().zip(want.iter()) {
                let got = env_base_draw(*row, seed as u64, HI);
                assert!(
                    (got - w).abs() < 1e-6,
                    "row {row} seed {seed}: {got} != {w}"
                );
            }
        }
    }

    /// Every draw lands inside the support the head's clamp assumes.
    #[test]
    fn the_base_draw_stays_inside_its_support() {
        for row in 0..5000u64 {
            let e = env_base_draw(row, 0, 6.0);
            assert!((0.0..6.0).contains(&e), "row {row}: {e}");
        }
    }
}
