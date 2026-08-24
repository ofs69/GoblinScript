//! The working cache: the transcode and the encoder's latents.
//!
//! Both are expensive and both are pure functions of (source, flags, bundle),
//! so a re-run should never pay for them twice. Everything here is keyed on a
//! fingerprint of the source file plus the settings that would change the
//! result -- and every entry records WHAT IT COVERS as well as what made it,
//! so a cache written over part of a clip is never mistaken for one written
//! over all of it.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::io::Read;
use std::path::{Path, PathBuf};

use crate::bundle::{Manifest, Transcode, Transnet};

/// The normalize's identity: the spec every later stage decodes from, as one
/// string that both names the file on disk and rides in the latent meta. The
/// height is the clip's own (`main::norm_height`) and the rest is the
/// bundle's -- a bundle that ships a different frame rate, quality or preset
/// is a different picture, and gets its own file rather than inheriting one.
pub fn norm_key(spec: &Transcode) -> String {
    format!("norm{}-{}-{}-{}", spec.height, spec.fps, spec.crf, spec.preset)
}

/// What `--no-transcode` decoded from instead: the source, untouched. It sits
/// in the same slot as a `norm_key`, so a parity run's latents and cuts are
/// never handed to a normal one.
pub const NORM_SOURCE: &str = "source";

#[derive(Serialize, Deserialize)]
pub struct CacheMeta {
    pub rows: usize,
    pub dim: usize,
    pub grid: usize,
    /// The basis these latents were projected on. A refit basis makes them
    /// meaningless, and meaningless features are worse than absent ones.
    pub basis_id: String,
    pub encoder: String,
    /// The auto-crop plan these latents were decoded under (`Plan::key`), or
    /// `None` for an uncropped decode. Part of the identity: cropped and
    /// uncropped latents of the same clip are different features.
    pub crop: Option<String>,
    /// The ONE file these latents were decoded from (`norm_key`, or
    /// `NORM_SOURCE`). Part of the identity: another normalize feeds the
    /// decode chain different pixels.
    pub norm: String,
    /// The exposure gamma the decode applied. Part of the identity for the
    /// same reason as `norm`.
    pub gamma: f64,
    /// The decode chain that read these frames (`ffmpeg::DECODE_REV`). The
    /// fields above pin the INPUT; this pins the reader, so a decode fix
    /// invalidates the caches the older reader wrote rather than handing the
    /// heads frames nobody would produce today.
    pub decode_rev: u32,
    /// The span of video the decode was asked to cover, in milliseconds: the
    /// clip's own duration, or what `--minutes` bounded it to. `rows` says
    /// how much was MADE; this says how much was asked for, and only the two
    /// together tell a complete cache from a bounded one over the same clip.
    /// Without it a two-minute development run leaves behind a cache that a
    /// later full run reads as the whole film.
    pub span_ms: i64,
}

/// The cut list, and what found it. Cuts are as much a function of the bundle
/// as the latents are -- the detector's thresholds, its input size, the frame
/// clock it counted on, the file it read -- so they carry the same kind of
/// stamp, and a new bundle re-detects instead of inheriting.
#[derive(Serialize, Deserialize)]
struct CutsFile {
    detector: Transnet,
    grid_fps: f64,
    norm: String,
    decode_rev: u32,
    span_ms: i64,
    cuts: Vec<f64>,
}

pub struct Cache {
    pub dir: PathBuf,
}

impl Cache {
    /// Per-video cache directory, created on demand.
    ///
    /// A VR aim is part of the key, not a detail on the side: re-aiming a clip
    /// changes every pixel the encoder will see, so it has to get its own
    /// working directory rather than silently reuse a transcode of the old aim.
    pub fn open(root: &Path, video: &Path, vr: Option<&crate::vr::Config>) -> Result<Self> {
        let mut key = fingerprint(video)?;
        if let Some(v) = vr {
            key.push('-');
            key.push_str(&v.key());
        }
        let dir = root.join(key);
        std::fs::create_dir_all(&dir)
            .with_context(|| format!("could not create the cache dir {}", dir.display()))?;
        Ok(Self { dir })
    }

    /// The ONE normalized file every stage decodes from, named by the spec
    /// that made it (`norm_key`): the clip's own crop-capable height, and the
    /// bundle's frame rate, quality and preset. A file made under one spec
    /// must never be mistaken for another's, and the name is where that is
    /// settled -- the reuse check upstream is the file's existence.
    pub fn norm_video(&self, spec: &Transcode) -> PathBuf {
        self.dir.join(format!("{}.mp4", norm_key(spec)))
    }
    pub fn latents(&self) -> PathBuf {
        self.dir.join("latents.i8")
    }
    fn cuts(&self) -> PathBuf {
        self.dir.join("cuts.json")
    }
    fn meta(&self) -> PathBuf {
        self.dir.join("meta.json")
    }

    /// Cached latents, but only if this run would have written the same ones:
    /// the same bundle, the same auto-crop plan (`None` = uncropped), the same
    /// normalize, the same exposure, the same reader -- and the same span of
    /// video, so a bounded run's cache is never read as a whole clip's.
    pub fn valid_latents(
        &self,
        man: &Manifest,
        crop: Option<&str>,
        norm: &str,
        gamma: f64,
        span_ms: i64,
    ) -> Option<CacheMeta> {
        let m: CacheMeta = serde_json::from_slice(&std::fs::read(self.meta()).ok()?).ok()?;
        let ok = m.basis_id == man.basis_id
            && m.encoder == man.encoder
            && m.dim == man.dim
            && m.grid == man.grid
            && m.crop.as_deref() == crop
            && m.norm == norm
            && m.gamma == gamma
            && m.decode_rev == crate::ffmpeg::DECODE_REV
            && m.span_ms == span_ms;
        // A cache IS its rows: an empty one is the shape a decode that
        // delivered nothing leaves behind, and it validates against itself
        // forever -- zero bytes are exactly zero rows.
        let sized = m.rows > 0
            && std::fs::metadata(self.latents()).ok()?.len() as usize
                == m.rows * m.dim * m.grid * m.grid;
        (ok && sized).then_some(m)
    }

    pub fn write_meta(&self, m: &CacheMeta) -> Result<()> {
        std::fs::write(self.meta(), serde_json::to_vec_pretty(m)?)?;
        Ok(())
    }

    /// The cached cuts, if this run's detector, clock, file and span are the
    /// ones that found them.
    pub fn read_cuts(&self, man: &Manifest, norm: &str, span_ms: i64) -> Option<Vec<f64>> {
        let c: CutsFile = serde_json::from_slice(&std::fs::read(self.cuts()).ok()?).ok()?;
        (c.detector == man.transnet
            && c.grid_fps == man.grid_fps
            && c.norm == norm
            && c.decode_rev == crate::ffmpeg::DECODE_REV
            && c.span_ms == span_ms)
            .then_some(c.cuts)
    }

    pub fn write_cuts(
        &self,
        man: &Manifest,
        norm: &str,
        span_ms: i64,
        cuts: &[f64],
    ) -> Result<()> {
        let f = CutsFile {
            detector: man.transnet,
            grid_fps: man.grid_fps,
            norm: norm.to_string(),
            decode_rev: crate::ffmpeg::DECODE_REV,
            span_ms,
            cuts: cuts.to_vec(),
        };
        std::fs::write(self.cuts(), serde_json::to_vec(&f)?)?;
        Ok(())
    }
}

/// A cheap, stable identity for a source file: its size, and the head and tail
/// of its bytes. Hashing a 40 GB video in full would cost more than the draft.
///
/// Also what a VR sidecar is filed under (`vr::sidecar_path`) -- the SOURCE
/// identity, so an aim survives being re-aimed and is found again next run.
pub fn fingerprint(video: &Path) -> Result<String> {
    const CHUNK: usize = 1 << 20;
    let mut f = std::fs::File::open(video)
        .with_context(|| format!("cannot open {}", video.display()))?;
    let len = f.metadata()?.len();

    let mut h = blake3::Hasher::new();
    h.update(&len.to_le_bytes());

    let mut buf = vec![0u8; CHUNK.min(len as usize)];
    f.read_exact(&mut buf)?;
    h.update(&buf);
    if len > 2 * CHUNK as u64 {
        use std::io::Seek;
        f.seek(std::io::SeekFrom::End(-(CHUNK as i64)))?;
        f.read_exact(&mut buf)?;
        h.update(&buf);
    }
    Ok(h.finalize().to_hex()[..16].to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A manifest to size caches against. The FIXTURE, not `bundle/`: these
    /// tests are about the cache's own arithmetic, and a bundle is a ~250 MB
    /// build product that a source checkout does not have.
    fn manifest() -> Manifest {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/manifest.json");
        serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap()
    }

    /// A cache dir of its own, with a latent file of `rows` rows in it.
    fn cache_with(name: &str, man: &Manifest, rows: usize) -> Cache {
        let dir = std::env::temp_dir().join(format!("goblin_cache_{name}"));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let c = Cache { dir };
        std::fs::write(c.latents(), vec![0u8; rows * man.dim * man.grid * man.grid])
            .unwrap();
        c
    }

    fn meta(man: &Manifest, rows: usize, norm: &str, span_ms: i64) -> CacheMeta {
        CacheMeta {
            rows,
            dim: man.dim,
            grid: man.grid,
            basis_id: man.basis_id.clone(),
            encoder: man.encoder.clone(),
            crop: None,
            norm: norm.to_string(),
            gamma: 1.0,
            decode_rev: crate::ffmpeg::DECODE_REV,
            span_ms,
        }
    }

    /// The bug this field exists for: `--minutes 2` leaves a cache of two
    /// minutes, and a later FULL run of the same clip must not read it as the
    /// whole film. Nothing else distinguishes the two -- the rows agree with
    /// the file's size either way, which is all the check used to be.
    #[test]
    fn a_bounded_cache_is_not_a_whole_clips() {
        let man = manifest();
        let c = cache_with("bounded", &man, 8);
        let norm = norm_key(&man.transcode);
        c.write_meta(&meta(&man, 8, &norm, 120_000)).unwrap();

        assert!(
            c.valid_latents(&man, None, &norm, 1.0, 120_000).is_some(),
            "the run that wrote it reads it back"
        );
        assert!(
            c.valid_latents(&man, None, &norm, 1.0, 3_600_000).is_none(),
            "a full run took a two-minute cache for the whole video"
        );
    }

    /// Zero rows agree with a zero-byte file forever, so the size check alone
    /// calls an empty cache valid. An empty cache is a decode that delivered
    /// nothing, and it must never be the answer to a later run.
    #[test]
    fn an_empty_cache_is_never_valid() {
        let man = manifest();
        let c = cache_with("empty", &man, 0);
        let norm = norm_key(&man.transcode);
        c.write_meta(&meta(&man, 0, &norm, 120_000)).unwrap();
        assert!(c.valid_latents(&man, None, &norm, 1.0, 120_000).is_none());
    }

    /// The normalize is part of what made the latents: a different spec, or
    /// the untouched source under `--no-transcode`, is a different picture of
    /// the same clip and gets its own entry -- and its own file on disk.
    #[test]
    fn a_different_normalize_is_a_different_cache() {
        let man = manifest();
        let c = cache_with("norm", &man, 8);
        let norm = norm_key(&man.transcode);
        c.write_meta(&meta(&man, 8, &norm, 120_000)).unwrap();
        assert!(c.valid_latents(&man, None, &norm, 1.0, 120_000).is_some());
        assert!(
            c.valid_latents(&man, None, NORM_SOURCE, 1.0, 120_000).is_none(),
            "a parity run read the normalized clip's latents"
        );

        let faster = Transcode { fps: man.transcode.fps + 1.0, ..man.transcode.clone() };
        assert!(
            c.valid_latents(&man, None, &norm_key(&faster), 1.0, 120_000).is_none(),
            "a bundle with another frame rate read the old spec's latents"
        );
        assert_ne!(
            c.norm_video(&man.transcode),
            c.norm_video(&faster),
            "two normalize specs share one file"
        );
    }

    /// Cuts are a function of the bundle too -- the detector's thresholds, the
    /// frame clock, the file, the span -- and carry the stamp that says so.
    #[test]
    fn cuts_are_reread_only_by_the_run_that_would_find_them() {
        let man = manifest();
        let c = cache_with("cuts", &man, 0);
        let norm = norm_key(&man.transcode);
        c.write_cuts(&man, &norm, 120_000, &[1000.0, 2000.0]).unwrap();

        assert_eq!(c.read_cuts(&man, &norm, 120_000), Some(vec![1000.0, 2000.0]));
        assert!(
            c.read_cuts(&man, &norm, 3_600_000).is_none(),
            "a full run took a two-minute cut list for the whole video"
        );
        assert!(c.read_cuts(&man, NORM_SOURCE, 120_000).is_none());

        let mut keener = manifest();
        keener.transnet.thr /= 2.0;
        assert!(
            c.read_cuts(&keener, &norm, 120_000).is_none(),
            "another detector read the old one's cuts"
        );
    }
}
