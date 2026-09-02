//! The ONNX bundle: the graphs, the manifest, and the GPU they run on.
//!
//! Everything the model IS lives here. `export_bundle.py` produces the bundle
//! from a champion checkpoint; a release build bakes it into the exe
//! (`--features embed`), a dev build reads it from a directory. Either way the
//! rest of the program sees the same thing: a `Manifest` of frozen constants
//! and a handful of `Session`s.

use anyhow::{Context, Result};
use ort::session::{builder::GraphOptimizationLevel, Session};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// A one-time provider note (the CUDA->DirectML fallback), stashed rather than
/// printed where it happens -- session building runs mid-draft, and a stray
/// print there corrupts the live progress display. `main` drains it through the
/// progress bars instead.
static PROVIDER_NOTE: std::sync::Mutex<Option<String>> = std::sync::Mutex::new(None);

/// Take the pending provider note, if any -- drained once by `main`.
pub fn take_provider_note() -> Option<String> {
    PROVIDER_NOTE.lock().unwrap().take()
}

/// Whether this run is pinned to the CPU (`--cpu`). Process-global because the
/// backend is a property of the RUN, while the choice is made deep inside a
/// per-graph call with no argument path back to `main` -- and every session,
/// wherever it is built, has to make the same one.
static FORCE_CPU: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Pin every graph to the CPU. Set from the command line before the first
/// session is built; a later change would leave a half-CPU run behind it.
pub fn set_force_cpu(on: bool) {
    FORCE_CPU.store(on, std::sync::atomic::Ordering::Relaxed);
}

/// Is this run pinned to the CPU?
pub fn force_cpu() -> bool {
    FORCE_CPU.load(std::sync::atomic::Ordering::Relaxed)
}

/// Frozen perception + deploy config, written by `export_bundle.py`. Nothing
/// in here is a Rust default: a value that disagrees with the checkpoint is a
/// silently wrong draft, so every one of them is read, never assumed. The
/// manifest JSON carries more keys than this struct (values baked into the
/// graphs, e.g. `tcn_ch`, `pos_temp`); serde skips what Rust has no use for.
#[derive(Debug, Clone, Deserialize)]
pub struct Manifest {
    pub bundle_version: u32,
    pub checkpoint: String,
    pub epoch: i64,
    pub basis_id: String,
    pub encoder: String,
    /// Which `MultiHeadAttention` input layout the encoder was fused with.
    /// `packed` is 2.4x faster on DirectML and has no CPU kernel at all, so
    /// `--cpu` is refused up front rather than dying on an opaque kernel error
    /// once the encode stage is already running. Bundles exported before the
    /// field existed are the separate layout, which is what the default says.
    #[serde(default = "attn_separate")]
    pub attn: String,
    pub graphs: Graphs,
    pub transcode: Transcode,
    pub grid_fps: f64,
    /// Latent ROW rate (rows/s). Distinct from `grid_fps`, the DECODE frame
    /// rate: with tubelet stride k and n alignment passes the encoder emits
    /// `grid_fps * n / (2k)` rows/s, and every row->ms conversion, styling
    /// window and progress budget rides THIS clock, not the decode one.
    /// Bundles from before the field (all 15 rows/s: stride 1, alignments 2)
    /// deserialize to 0 and `row_hz()` falls back to that formula.
    #[serde(default)]
    pub row_hz: f64,
    pub enc_res: usize,
    pub grid: usize,
    pub dim: usize,
    pub clip_len: usize,
    pub tubelet_stride: usize,
    pub alignments: usize,
    pub int8_scale: f32,
    pub v_std: f64,
    pub chunk: usize,
    pub min_chunk: usize,
    /// Rows of real context forwarded on both sides of a chunk and cut from
    /// the output, so kept rows carry a warm receptive field and a warm
    /// envelope AR history across chunk boundaries. Pre-ctx bundles have no
    /// field, which deserializes to 0 = the bare tiling they were tested
    /// under.
    #[serde(default)]
    pub ctx: usize,
    pub env_ctx: usize,
    /// The envelope head SAMPLES rather than publishing an expectation, so
    /// `env_step` takes a base draw per row. A categorical bundle has the
    /// field false (and a pre-flow one has no field at all, which
    /// deserializes to the same).
    #[serde(default)]
    pub env_flow: bool,
    /// Authorship seed the base draw is keyed by, beside the absolute row.
    #[serde(default)]
    pub env_seed: u64,
    /// Support of the base draw, `[0, env_base_hi)`.
    #[serde(default)]
    pub env_base_hi: f64,
    pub heads: Heads,
    pub still_eps: f64,
    pub ext_snap: f64,
    /// Composed styling's amplitude bound: a stroke's written excursion is
    /// at most this multiple of the marginal's own travel over the segment,
    /// so the written speed stays under that multiple of the speed the
    /// model reads there. 0 (absent = the serde default) = off.
    #[serde(default)]
    pub amp_cap_x: f64,
    /// Hz: the bound loosens with the segment's own frequency above this on
    /// the hedge curve `(f / f0) ^ env_gain_p`, flat below. 0 = flat.
    #[serde(default)]
    pub amp_cap_f0: f64,
    /// Exponent of the envelope's measured frequency hedge, which the bound
    /// loosens on. jepa_train's `ENV_GAIN_P`; a bundle without the field
    /// carries no bound, so the value is never reached.
    #[serde(default = "default_env_gain_p")]
    pub env_gain_p: f64,
    pub plat_thr: f64,
    pub plat_lo: f64,
    /// Dwell-call peak filter; 0 disables it (and pre-filter bundles have no
    /// field, which deserializes to the same off).
    #[serde(default)]
    pub plat_peak: f64,
    /// Dwell-call stroke veto (pos-units/s): calls whose ~2 s flanks both
    /// read below this mean predicted |vmarg| are dropped -- no adjoining
    /// stroke, no plateau. 0 disables it (pre-veto bundles have no field,
    /// which deserializes to the same off).
    #[serde(default)]
    pub plat_veto: f64,
    /// Per-row rail target inside a dwell lock (follows slow plateau
    /// drift) instead of one constant level. Absent (pre-variant bundles)
    /// deserializes to false = the constant-level lock.
    #[serde(default)]
    pub plat_rail_track: bool,
    /// Peak-confidence lock scaling [p0, p1]: each dwell's correction is
    /// scaled by clamp((peak - p0) / (p1 - p0), 0, 1). [0, 0] (absent =
    /// the serde default) = full strength, the pre-variant behavior.
    #[serde(default)]
    pub plat_soft: [f64; 2],
    /// Cap on the dwell lock's per-row mean correction, position units:
    /// a confident call shifting too far is half the gross-error rate.
    /// 0 (absent = the serde default, pre-cap bundles) = uncapped.
    #[serde(default)]
    pub plat_shift_cap: f64,
    /// Reversal-event crossing snap radius, SECONDS: composed styling moves
    /// each vmarg crossing to the rev head's direction-aware local argmax
    /// within this radius. 0 disables it.
    pub rev_snap_s: f64,
    /// Sub-frame reversal times for the written actions: "rev" = 3-point
    /// parabola on the event head around the snapped apex (the champion
    /// decode). Empty (pre-subframe bundles) = actions stay on the grid.
    #[serde(default)]
    pub subframe: String,
    /// Reversal SEGMENTATION source: "viterbi" = the alternating event
    /// decode over the rev head, which is what fast capture rides on;
    /// anything else (incl. the empty default of pre-graft bundles) = the
    /// marginal's zero crossings. The carrier is identical either way.
    #[serde(default)]
    pub rev_source: String,
    /// Pre-derivative crossing smoother, SECONDS. 0.1 is the operating
    /// point; 0.2 collapses fast capture.
    pub rev_smooth_s: f64,
    /// Event refractory, SECONDS. 0 = unconstrained.
    pub rev_gap_s: f64,
    /// Fit the emission prior separately on STILL and MOVING rows, so an
    /// unscripted gap's posterior mass cannot inflate the prior that governs
    /// moving rows. Defaults false, which is the single global fit that
    /// pre-gap-prior bundles carry.
    #[serde(default)]
    pub rev_gap_prior: bool,
    /// Box width, SECONDS, for the |vmarg| the gap prior reads its still /
    /// moving membership off. Defaults to jepa_infer's `SPEED_REF_S`.
    #[serde(default = "default_speed_ref_s")]
    pub speed_ref_s: f64,
    /// Emission-prior fit window, SECONDS (a cap: normally the whole clip).
    pub bias_fit_s: f64,
    pub transnet: Transnet,
}

/// jepa_infer's `SPEED_REF_S`. Only reached by a bundle exported before the
/// field existed; every current export writes it.
fn default_speed_ref_s() -> f64 {
    0.6
}

/// jepa_train's `ENV_GAIN_P`, for a bundle exported before the field.
fn default_env_gain_p() -> f64 {
    0.6
}

#[derive(Debug, Clone, Deserialize)]
pub struct Graphs {
    pub encoder: String,
    pub transnet: String,
    pub head: String,
    pub env_step: Option<String>,
    /// The mask net on its own, for the viewport. Absent in bundles exported
    /// before it, which draw latent band energy instead.
    pub mask: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Transcode {
    pub height: u32,
    pub fps: f64,
    pub crf: u32,
    pub preset: String,
}

#[derive(Debug, Clone, Copy, Deserialize)]
pub struct Heads {
    pub pos: bool,
    pub env: bool,
    pub ext: bool,
    pub plat: bool,
    /// Reversal-event head (pre-rev bundles have no field -> false).
    #[serde(default)]
    pub rev: bool,
}

fn attn_separate() -> String {
    "separate".to_string()
}

/// The manifest format this build understands. A bump means field semantics
/// changed, and an old binary must refuse the bundle rather than misread it.
/// Version 3: every duration in the manifest is SECONDS -- `rev_snap_s` and
/// `rev_gap_s` replace the row-valued fields, and all three are required, so
/// a bundle that omits one is refused instead of silently defaulted.
const BUNDLE_VERSION: u32 = 3;

impl Manifest {
    /// A manifest duration as ROWS on this bundle's grid -- the mirror of
    /// `common.rows_at`, and the ONE place seconds become rows.
    pub fn rows_at(&self, seconds: f64) -> usize {
        (seconds * self.row_hz()).round().max(1.0) as usize
    }

    /// The emission-prior fit window as ROWS on this bundle's grid.
    pub fn bias_fit_rows(&self) -> usize {
        self.rows_at(self.bias_fit_s)
    }

    /// The latent row rate: the stamped field, or (pre-field bundles) the
    /// grid formula over the stamped decode config.
    pub fn row_hz(&self) -> f64 {
        if self.row_hz > 0.0 {
            self.row_hz
        } else {
            self.grid_fps * self.alignments as f64
                / (2.0 * self.tubelet_stride as f64)
        }
    }

    /// Video milliseconds of latent row `i` -- THE row clock, and the one
    /// definition of it.
    ///
    /// A row is not an instant: it is the tubelet PAIR it was encoded from,
    /// frames `a` and `a + tubelet_stride`, so it sits at that pair's
    /// midpoint. That is what the training caches store as `times_ms` and
    /// therefore the clock every scorer, every lag fit and every target the
    /// heads were trained against already speak. Half a frame of it is not a
    /// rounding detail -- the whole draft is written on this clock, and an
    /// offset here moves every action in it.
    pub fn row_ms(&self, i: usize) -> f64 {
        let k = self.tubelet_stride as f64;
        // rows are 1:1 with frames when the alignments tile the window
        // (`alignments == 2k`); a coarser alignment count decimates them
        let frame = i as f64 * 2.0 * k / self.alignments as f64;
        (frame + k / 2.0) / self.grid_fps * 1000.0
    }


    /// Loud load-time refusal of a bundle this build cannot honor: a format
    /// version it does not understand, or a missing head the styler reads
    /// unconditionally (composed styling needs pos + env + ext; only the
    /// dwell head is optional at decode).
    fn validate(self) -> Result<Self> {
        anyhow::ensure!(
            self.bundle_version == BUNDLE_VERSION,
            "bundle format v{} (this build understands v{BUNDLE_VERSION})",
            self.bundle_version
        );
        anyhow::ensure!(
            self.heads.pos && self.heads.env && self.heads.ext,
            "bundle is missing heads the styler requires \
             (pos {}, env {}, ext {})",
            self.heads.pos,
            self.heads.env,
            self.heads.ext
        );
        // The preset ladders are compile-time constants standing beside these
        // runtime numbers, and this is the first and only point at which the
        // two meet. A bundle whose tuning has flattened a rung is a bundle
        // whose menus lie, so it does not load.
        if let Err(e) =
            crate::style::presets_ordered(self.still_eps, (self.plat_soft[0], self.plat_soft[1]))
        {
            anyhow::bail!("{e}");
        }
        Ok(self)
    }
}

#[derive(Debug, Clone, Copy, Deserialize, Serialize, PartialEq)]
pub struct Transnet {
    /// TransNetV2's fixed input frame size.
    pub input_h: usize,
    pub input_w: usize,
    /// Sliding-window inference: `window` frames per forward, advanced by
    /// `step`; the middle `step` rows of each window's output are kept.
    pub window: usize,
    pub step: usize,
    /// A per-frame transition probability above `thr` marks a transition;
    /// consecutive such frames are one cut, no closer than `min_gap_s`.
    pub thr: f64,
    pub min_gap_s: f64,
}

/// Graph bytes + the manifest. Sessions are built lazily: the boundary pass and
/// the encoder pass never need to be resident at the same time, and the encoder
/// alone is ~180 MB of weights on the GPU.
pub struct Bundle {
    pub manifest: Manifest,
    encoder: Vec<u8>,
    transnet: Vec<u8>,
    head: Vec<u8>,
    env_step: Option<Vec<u8>>,
    mask: Option<Vec<u8>>,
}

#[cfg(feature = "embed")]
mod embedded {
    macro_rules! bundled {
        ($name:literal) => {
            include_bytes!(concat!(env!("CARGO_MANIFEST_DIR"), "/bundle/", $name))
        };
    }
    pub const MANIFEST: &[u8] = bundled!("manifest.json");
    pub const ENCODER: &[u8] = bundled!("vjepa_pca.onnx");
    pub const TRANSNET: &[u8] = bundled!("transnet.onnx");
    pub const HEAD: &[u8] = bundled!("head.onnx");
    pub const ENV_STEP: &[u8] = bundled!("env_step.onnx");
    pub const MASK: &[u8] = bundled!("mask.onnx");
}

/// Write the bundle baked into this binary out to `dir`, byte for byte.
///
/// The graphs are a build product of the training tree and are not in the
/// repository, so a build from source has no goblins in it. A release binary
/// carries a set, and these are the same bytes `--bundle` reads: dump them
/// once and any build drafts exactly as that release does.
///
/// Named from the manifest rather than from a fixed list, so what lands in the
/// directory is what `from_dir` will look for -- including leaving out a graph
/// the manifest does not claim.
#[cfg(feature = "embed")]
pub fn dump(dir: &Path) -> Result<Vec<(String, usize)>> {
    let manifest: Manifest = serde_json::from_slice(embedded::MANIFEST)
        .context("the baked-in manifest.json is not one this build understands")?;
    std::fs::create_dir_all(dir)
        .with_context(|| format!("cannot make {}", dir.display()))?;
    let mut wrote: Vec<(String, usize)> = Vec::new();
    let mut put = |name: &str, bytes: &'static [u8]| -> Result<()> {
        let path = dir.join(name);
        std::fs::write(&path, bytes)
            .with_context(|| format!("cannot write {}", path.display()))?;
        wrote.push((name.to_string(), bytes.len()));
        Ok(())
    };
    put("manifest.json", embedded::MANIFEST)?;
    put(&manifest.graphs.encoder, embedded::ENCODER)?;
    put(&manifest.graphs.transnet, embedded::TRANSNET)?;
    put(&manifest.graphs.head, embedded::HEAD)?;
    if let Some(name) = &manifest.graphs.env_step {
        put(name, embedded::ENV_STEP)?;
    }
    if let Some(name) = &manifest.graphs.mask {
        put(name, embedded::MASK)?;
    }
    Ok(wrote)
}

/// The same door, in a build that has nothing behind it.
#[cfg(not(feature = "embed"))]
pub fn dump(_dir: &Path) -> Result<Vec<(String, usize)>> {
    anyhow::bail!(
        "this build has no bundle baked into it, so there is nothing to write \
         out. --dump-bundle is for a released goblinscript.exe; a build from \
         source is already being handed a bundle directory with --bundle."
    )
}

impl Bundle {
    /// Load from a bundle directory (dev / `--bundle`).
    pub fn from_dir(dir: &Path) -> Result<Self> {
        let read = |name: &str| -> Result<Vec<u8>> {
            std::fs::read(dir.join(name))
                .with_context(|| format!("bundle: missing {name} in {}", dir.display()))
        };
        let manifest: Manifest = serde_json::from_slice::<Manifest>(&read("manifest.json")?)
            .context("bundle: manifest.json is not a manifest this build understands")?
            .validate()?;
        Ok(Self {
            encoder: read(&manifest.graphs.encoder)?,
            transnet: read(&manifest.graphs.transnet)?,
            head: read(&manifest.graphs.head)?,
            env_step: match &manifest.graphs.env_step {
                Some(n) => Some(read(n)?),
                None => None,
            },
            mask: match &manifest.graphs.mask {
                Some(n) => Some(read(n)?),
                None => None,
            },
            manifest,
        })
    }

    /// Load the bundle baked into this binary.
    #[cfg(feature = "embed")]
    pub fn embedded() -> Result<Self> {
        Ok(Self {
            manifest: serde_json::from_slice::<Manifest>(embedded::MANIFEST)?.validate()?,
            encoder: embedded::ENCODER.to_vec(),
            transnet: embedded::TRANSNET.to_vec(),
            head: embedded::HEAD.to_vec(),
            env_step: Some(embedded::ENV_STEP.to_vec()),
            mask: Some(embedded::MASK.to_vec()),
        })
    }

    pub fn encoder_session(&self) -> Result<Session> {
        session(&self.encoder, "encoder")
    }
    pub fn transnet_session(&self) -> Result<Session> {
        session(&self.transnet, "transnet")
    }
    /// The head runs on the CPU, and not by preference.
    ///
    /// DirectML BUILDS this graph and then crashes executing it (an access
    /// violation inside the EP, reproducible from Python with the same 5-D int8
    /// input, so it is not a binding bug on our side). The head is also the
    /// cheap half -- the encoder is ~95% of a draft's compute -- and the CPU
    /// path is the one the parity check validated to corr 1.00000000 against
    /// jepa_infer, so this costs accuracy nothing and wall clock little.
    pub fn head_session(&self) -> Result<Session> {
        cpu_session(&self.head, "head")
    }
    /// The envelope's step graph runs on the CPU too, for its own reason: the
    /// decode is autoregressive -- one call per row, ~36k calls for a 40-minute
    /// video -- over three 1x1 convolutions. On a GPU that is pure dispatch
    /// overhead with no compute to amortize it.
    pub fn env_session(&self) -> Result<Option<Session>> {
        match &self.env_step {
            Some(b) => Ok(Some(cpu_session(b, "env_step")?)),
            None => Ok(None),
        }
    }

    /// The mask net, for the viewport -- on the CPU, and pinned to ONE thread.
    ///
    /// It runs on the encode thread between windows, and everything around it
    /// is already spending the machine: the encoder has the GPU, and in a batch
    /// `prefetch` has libx264 on the cores. A decoration that took either would
    /// cost real wall clock. One thread is enough by a wide margin -- ~40 M
    /// multiply-adds against a window that arrives every ~120 ms -- so this is
    /// the cheap end of the trade, not a compromise.
    ///
    /// `None` when the bundle predates the graph; the viewport draws latent
    /// band energy instead and nothing else notices.
    pub fn mask_session(&self) -> Result<Option<Session>> {
        use ort::execution_providers::CPUExecutionProvider;
        let Some(bytes) = &self.mask else { return Ok(None) };
        Ok(Some(
            Session::builder()
                .map_err(ort_err)?
                .with_execution_providers([CPUExecutionProvider::default().build()])
                .map_err(ort_err)?
                .with_intra_threads(1)
                .map_err(ort_err)?
                .with_optimization_level(GraphOptimizationLevel::Level3)
                .map_err(ort_err)?
                .commit_from_memory(bytes)
                .map_err(ort_err)
                .context("failed to build the mask session")?,
        ))
    }
}

fn cpu_session(bytes: &[u8], what: &str) -> Result<Session> {
    use ort::execution_providers::CPUExecutionProvider;
    Session::builder()
        .map_err(ort_err)?
        .with_execution_providers([CPUExecutionProvider::default().build()])
        .map_err(ort_err)?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(ort_err)?
        .commit_from_memory(bytes)
        .map_err(ort_err)
        .with_context(|| format!("failed to build the {what} session"))
}

/// `ort`'s errors carry a recovery payload that is neither Send nor Sync, so
/// they do not convert into `anyhow::Error` on their own. The payload is a way
/// to get the half-built session back, which we never want: a session that
/// failed to build is a bug in the bundle, not something to recover from.
pub fn ort_err<R>(e: ort::Error<R>) -> anyhow::Error {
    anyhow::anyhow!("{}", e.message())
}

/// One session on the best GPU we can reach.
///
/// DirectML is the floor: it is the only backend that covers every vendor's
/// GPU on Windows, which is what makes the binary distributable. A `cuda`
/// build tries a CUDA session FIRST -- ONNX Runtime refuses CUDA and DML in
/// one session, so it is a separate attempt, and `error_on_failure` makes a
/// machine without CUDA/cuDNN fail that attempt instead of silently landing
/// on CPU -- then falls back to DirectML, so the same binary serves both.
/// CPU is registered last, but on a shipped bundle it can only ever serve the
/// small graphs: the encoder's packed attention has no CPU kernel at all, so a
/// GPU is not optional here. `main` refuses `--cpu` outright for that bundle.
///
/// `--cpu` skips the chain entirely, and needs an `--cpu-attn` export to work.
/// It exists because a GPU backend can MISCOMPUTE this graph rather than fail
/// it (the fp16 overflow `encode` guards against), and the CPU provider --
/// fp32, no vendor EP in the path -- is the reference a suspect draft gets
/// bisected against.
fn session(bytes: &[u8], what: &str) -> Result<Session> {
    if force_cpu() {
        return cpu_session(bytes, what);
    }
    use ort::execution_providers::{CPUExecutionProvider, DirectMLExecutionProvider};
    #[cfg(feature = "cuda")]
    {
        use ort::execution_providers::CUDAExecutionProvider;
        let cuda = (|| -> Result<Session> {
            Session::builder()
                .map_err(ort_err)?
                .with_execution_providers([
                    CUDAExecutionProvider::default().build().error_on_failure(),
                    CPUExecutionProvider::default().build(),
                ])
                .map_err(ort_err)?
                .with_optimization_level(GraphOptimizationLevel::Level3)
                .map_err(ort_err)?
                .commit_from_memory(bytes)
                .map_err(ort_err)
        })();
        match cuda {
            Ok(s) => return Ok(s),
            Err(_) => {
                // The raw ORT error is a screenful of build-server paths; the
                // user-relevant content is one line. Stash it (once) instead of
                // printing here: session building runs mid-draft, and a stray
                // print corrupts the live progress display. main drains it
                // through the progress bars.
                static ONCE: std::sync::Once = std::sync::Once::new();
                ONCE.call_once(|| {
                    *PROVIDER_NOTE.lock().unwrap() = Some(
                        "note: CUDA not available, running on DirectML \
                         (the fast path needs the CUDA 12 runtime + cuDNN 9 on PATH)"
                            .to_string(),
                    );
                });
            }
        }
    }
    Session::builder()
        .map_err(ort_err)?
        .with_execution_providers([
            DirectMLExecutionProvider::default().build(),
            CPUExecutionProvider::default().build(),
        ])
        .map_err(ort_err)?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(ort_err)?
        .commit_from_memory(bytes)
        .map_err(ort_err)
        .with_context(|| format!("failed to build the {what} session"))
}

#[cfg(test)]
mod tests {
    use super::Manifest;
    use std::path::PathBuf;

    /// Where row 0 sits on the shipped grid, in milliseconds. The mirror of
    /// the extractor's own arithmetic, which `grid_check.py` recomputes from
    /// the bundle manifest and compares against this line.
    const SHIPPED_ROW0_MS: f64 = 33.333333;

    /// The row clock a bundle on the shipped grid produces, against the clock
    /// the training caches were written on. Row `i` is the midpoint of the
    /// tubelet pair `(i, i + stride)`, which at the shipped 30 fps / stride 2
    /// puts row 0 at 33.333 ms and every row a frame later than the index
    /// alone would suggest. This is the one number that decides where every
    /// action in every draft is written, and there is nothing downstream that
    /// would notice it drifting.
    ///
    /// Read from the FIXTURE, which carries the shipped grid's constants: the
    /// arithmetic is the thing under test, and a bundle is a ~250 MB build
    /// product a source checkout does not have. That the bundle actually
    /// packed agrees with this is `cargo xtask dist`'s check, on the real
    /// manifest, at the moment it is about to ship.
    #[test]
    fn the_row_clock_is_the_tubelet_pairs_midpoint() {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/manifest.json");
        let man: Manifest = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        let step = 1000.0 / man.row_hz();
        for i in [0usize, 1, 17, 4096] {
            let want = man.row_ms(0) + i as f64 * step;
            assert!(
                (man.row_ms(i) - want).abs() < 1e-9,
                "row {i}: the clock is not uniform at 1/row_hz"
            );
        }
        assert!(
            (man.row_ms(0) - 1000.0 * man.tubelet_stride as f64 / 2.0 / man.grid_fps).abs() < 1e-9,
            "row 0 is not half a tubelet in"
        );
        assert!((man.row_ms(0) - SHIPPED_ROW0_MS).abs() < 1e-4,
                "the shipped grid's row 0 moved off {SHIPPED_ROW0_MS} ms (got {})",
                man.row_ms(0));
    }
}
