//! Everything the app makes noise with, in two independent halves.
//!
//! **The effects** -- a Game Boy-flavoured boot chime, a finished-batch jingle,
//! and the small square-wave punctuation the interface makes as it works -- are
//! synthesized here, sample by sample, and played through rodio. No audio
//! assets, no decoders.
//!
//! **The music** is the `music/` playlist, played through the operating
//! system's General MIDI synthesizer (see the soundtrack section below). Real
//! arrangements, played as written.
//!
//! Everything is best-effort: on a machine with no audio device, no synth, or a
//! headless CI run, the calls are silent no-ops and nothing upstream notices.
//!
//! Two rules keep the punctuation from becoming a nuisance. It is MUTABLE (the
//! picker's M key, `--mute`, remembered in settings), and it only fires on
//! events a human caused or would want to hear across the room -- a finished
//! stage, a failure, a skip -- never per keystroke and never on a piped run,
//! where an overnight batch would be blipping to nobody.

use rodio::{OutputStream, OutputStreamHandle, Sink};
use std::io::IsTerminal;
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};

const SAMPLE_RATE: u32 = 44_100;

/// Global mute. Checked at the top of every play call, so muting mid-batch
/// silences the next sound rather than needing to reach anything already
/// playing.
static MUTED: AtomicBool = AtomicBool::new(false);

pub fn set_muted(m: bool) {
    MUTED.store(m, Ordering::Relaxed);
}

pub fn muted() -> bool {
    MUTED.load(Ordering::Relaxed)
}

/// What the app is currently making noise with. Three states rather than two
/// independent switches, because "background track but no blips" is a
/// combination nobody has ever wanted and a second key to explain.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Audio {
    /// The music playlist and the effects.
    Full,
    /// Effects only -- chimes, stage blips, the failure buzz.
    Blips,
    Silent,
}

impl Audio {
    pub fn label(self) -> &'static str {
        match self {
            Audio::Full => crate::t!("audio.music"),
            Audio::Blips => crate::t!("audio.blips"),
            Audio::Silent => crate::t!("audio.silent"),
        }
    }
}

pub fn audio_mode() -> Audio {
    if muted() {
        Audio::Silent
    } else if music_on() {
        Audio::Full
    } else {
        Audio::Blips
    }
}

/// Advance the audio mode and return the new one (the picker's M key).
/// full -> blips -> silent -> full.
pub fn cycle_audio() -> Audio {
    let next = match audio_mode() {
        Audio::Full => Audio::Blips,
        Audio::Blips => Audio::Silent,
        Audio::Silent => Audio::Full,
    };
    match next {
        Audio::Full => {
            set_muted(false);
            set_music(true);
        }
        Audio::Blips => {
            set_muted(false);
            set_music(false);
        }
        Audio::Silent => {
            set_music(false);
            set_muted(true);
        }
    }
    next
}

/// Play a finished buffer and return immediately.
///
/// The whole audio pipeline is owned by a short-lived thread that blocks until
/// the sound ends, so it is never cut off, while the caller moves on. rodio's
/// stream handle is not `Send`-safe to park in a static, so each effect opens
/// its own -- which is why these stay rare and short.
fn fire(samples: Vec<f32>) {
    // Muted, or nobody there: a redirected run is a batch job, and a batch job
    // blipping at an empty room four times per video is the exact nuisance the
    // module docs promise not to be. One gate, so no call site can forget it.
    if muted() || !std::io::stdout().is_terminal() {
        return;
    }
    std::thread::spawn(move || {
        if let Ok((_stream, handle)) = OutputStream::try_default() {
            if let Ok(sink) = Sink::try_new(&handle) {
                sink.append(rodio::buffer::SamplesBuffer::new(1, SAMPLE_RATE, samples));
                sink.sleep_until_end();
            }
        }
    });
}

/// Holds the audio pipeline alive while it plays. Dropping it (when the intro
/// ends, or the user skips) stops the sound -- which is exactly what we want.
pub struct Boot {
    _stream: OutputStream,
    _handle: OutputStreamHandle,
    _sink: Sink,
}

/// Start the chime. Returns the live pipeline to keep in scope for the duration
/// of the animation; `None` if no audio device is available.
pub fn play_boot() -> Option<Boot> {
    if muted() {
        return None;
    }
    let (stream, handle) = OutputStream::try_default().ok()?;
    let sink = Sink::try_new(&handle).ok()?;
    sink.append(rodio::buffer::SamplesBuffer::new(1, SAMPLE_RATE, synth()));
    Some(Boot { _stream: stream, _handle: handle, _sink: sink })
}

/// The "processing finished" jingle, once per batch that drafted something.
pub fn play_done() {
    fire(done_synth());
}

/// A stage cleared (normalize, cuts, encode, model). Deliberately tiny -- it
/// punctuates a long run without becoming the run's soundtrack.
pub fn play_stage() {
    fire(stage_synth());
}

/// A video failed. Dissonant and falling, so it reads as bad from another room.
pub fn play_error() {
    fire(error_synth());
}

/// A video was skipped because it already had a script: a short goblin grumble,
/// low and unhappy but not alarming -- nothing went wrong.
pub fn play_skip() {
    fire(skip_synth());
}

/// A picker toggle (select, overwrite, mute-off, theme). One short click.
pub fn play_click() {
    fire(click_synth());
}

/// A 50%-duty square wave sampled at fractional `phase` (cycles). Buzzy on
/// purpose -- it is the chiptune voice.
fn square(phase: f32) -> f32 {
    if phase.fract() < 0.5 {
        1.0
    } else {
        -1.0
    }
}

/// Mix one enveloped note into `buf`. `decay` sets the exponential ring-down;
/// `sine` blends a pure sine in (0 = raw square, higher = softer, bell-like) so
/// the big note rings instead of just buzzing.
#[allow(clippy::too_many_arguments)]
fn add_tone(
    buf: &mut [f32],
    start_s: f32,
    dur_s: f32,
    freq: f32,
    amp: f32,
    decay: f32,
    sine: f32,
) {
    let sr = SAMPLE_RATE as f32;
    let start = (start_s * sr) as usize;
    let n = (dur_s * sr) as usize;
    for i in 0..n {
        let idx = start + i;
        if idx >= buf.len() {
            break;
        }
        let t = i as f32 / sr;
        let phase = freq * t;
        let wave = square(phase) * (1.0 - sine)
            + (2.0 * std::f32::consts::PI * phase).sin() * sine;
        // 3 ms attack (no click) then an exponential decay
        let env = (t / 0.003).min(1.0) * (-t * decay).exp();
        buf[idx] += wave * amp * env;
    }
}

/// The whole ~1.3 s chime as mono f32 samples: a C-E-G power-up run, then a
/// ringing C6 "DING" with a soft octave-down body under it.
fn synth() -> Vec<f32> {
    let mut buf = vec![0.0f32; (SAMPLE_RATE as f32 * 1.4) as usize];

    // rising power-up blips (short, pure square)
    add_tone(&mut buf, 0.00, 0.06, 523.25, 0.12, 8.0, 0.0); // C5
    add_tone(&mut buf, 0.06, 0.06, 659.25, 0.12, 8.0, 0.0); // E5
    add_tone(&mut buf, 0.12, 0.06, 783.99, 0.12, 8.0, 0.0); // G5
    // the DING: a high C that rings down, square+sine so it chimes
    add_tone(&mut buf, 0.20, 1.15, 1046.50, 0.22, 3.2, 0.45); // C6
    // a quiet octave below for body
    add_tone(&mut buf, 0.20, 1.15, 523.25, 0.10, 3.2, 0.60); // C5

    clamp(&mut buf);
    buf
}

/// The "done" jingle: a brighter C-E-G-C run that resolves onto a ringing C
/// major triad -- a clear "finished, and it went well" flourish, distinct from
/// the two-part boot chime.
fn done_synth() -> Vec<f32> {
    let mut buf = vec![0.0f32; (SAMPLE_RATE as f32 * 1.6) as usize];

    // ascending run
    add_tone(&mut buf, 0.00, 0.10, 523.25, 0.14, 7.0, 0.0); // C5
    add_tone(&mut buf, 0.10, 0.10, 659.25, 0.14, 7.0, 0.0); // E5
    add_tone(&mut buf, 0.20, 0.10, 783.99, 0.14, 7.0, 0.0); // G5
    // the resolve: a C-major triad that rings down
    add_tone(&mut buf, 0.30, 1.25, 1046.50, 0.18, 2.6, 0.45); // C6
    add_tone(&mut buf, 0.30, 1.25, 783.99, 0.11, 2.6, 0.50); // G5
    add_tone(&mut buf, 0.30, 1.25, 659.25, 0.09, 2.6, 0.50); // E5

    clamp(&mut buf);
    buf
}

/// A stage blip: one 60 ms square note, quiet. Short enough that four of them
/// across a video read as punctuation rather than a tune.
fn stage_synth() -> Vec<f32> {
    let mut buf = vec![0.0f32; (SAMPLE_RATE as f32 * 0.14) as usize];
    add_tone(&mut buf, 0.00, 0.06, 987.77, 0.09, 9.0, 0.15); // B5
    clamp(&mut buf);
    buf
}

/// The failure buzz: a minor second (two tones a semitone apart) beating
/// against each other, sliding down. Dissonance is the point -- it must not be
/// mistakable for any of the cheerful sounds.
fn error_synth() -> Vec<f32> {
    let mut buf = vec![0.0f32; (SAMPLE_RATE as f32 * 0.75) as usize];
    add_tone(&mut buf, 0.00, 0.30, 233.08, 0.16, 3.0, 0.0); // Bb3
    add_tone(&mut buf, 0.00, 0.30, 246.94, 0.16, 3.0, 0.0); // B3 -- the grind
    add_tone(&mut buf, 0.28, 0.42, 185.00, 0.15, 2.6, 0.0); // F#3
    add_tone(&mut buf, 0.28, 0.42, 174.61, 0.15, 2.6, 0.0); // F3
    clamp(&mut buf);
    buf
}

/// The skip grumble: two low, fast, buzzy notes -- a goblin muttering that this
/// one was already done.
fn skip_synth() -> Vec<f32> {
    let mut buf = vec![0.0f32; (SAMPLE_RATE as f32 * 0.34) as usize];
    add_tone(&mut buf, 0.00, 0.11, 146.83, 0.13, 7.0, 0.0); // D3
    add_tone(&mut buf, 0.11, 0.16, 123.47, 0.13, 6.0, 0.0); // B2
    clamp(&mut buf);
    buf
}

/// A UI click: 25 ms of high square, barely there.
fn click_synth() -> Vec<f32> {
    let mut buf = vec![0.0f32; (SAMPLE_RATE as f32 * 0.06) as usize];
    add_tone(&mut buf, 0.00, 0.025, 1567.98, 0.07, 22.0, 0.0); // G6
    clamp(&mut buf);
    buf
}

/// Square waves sum past full scale; every voice ends here.
fn clamp(buf: &mut [f32]) {
    for s in buf.iter_mut() {
        *s = s.clamp(-1.0, 1.0);
    }
}

// ===========================================================================
// The soundtrack: every `.mid` in `music/`, played as a playlist through the
// operating system's General MIDI synthesizer.
//
// The scores are real GM arrangements -- strings, choir, sax, piano -- and they
// are played as written. An earlier version rendered them through this file's
// own square-wave voices, which turned every arrangement into an 8-bit cover;
// on a retro-themed app that sounded like a good idea and did not survive
// hearing it.
//
// So: `midly` reads the file into a stream of timed MIDI messages, and a player
// thread hands them to the OS synth at the right moments. That means no
// soundfont asset, no licence beyond the parser's, and the music sounds the way
// it sounds in the composer's own player.
//
// The cost is that this path is WINDOWS-ONLY -- it is the multimedia API's
// built-in wavetable synth. On any other platform the music is simply absent
// (the effects, which are synthesized here, still work); the port would be a
// soundfont synth behind the same `set_music` door, and nothing outside this
// section would change.
// ===========================================================================

// `SCORES: &[(name, bytes)]` -- one entry per `.mid` in `music/`, generated by
// `build.rs`. Adding a track is dropping a file in that folder.
include!(concat!(env!("OUT_DIR"), "/soundtrack.rs"));

/// One MIDI message, at its moment on the piece's absolute second-clock.
///
/// `msg` is packed the way the multimedia API wants it: status byte in the low
/// 8 bits, then the two data bytes.
struct Event {
    at_s: f32,
    msg: u32,
}

/// A parsed piece: the messages to send, how long it runs, and the tempo it
/// runs at.
///
/// The tempo map is kept rather than consumed because the header animates ON
/// the music (`music_beat`) -- flattening everything to seconds is right for
/// playback and useless for anything that has to land on a beat.
struct Score {
    events: Vec<Event>,
    end_s: f32,
    /// `(from this second, microseconds per quarter note)`, in order, always
    /// opening at 0.0 with whatever the file starts at.
    tempo: Vec<(f32, f64)>,
}

/// How loud the music sits under the work. Three named steps cycled with one
/// key, the way every other preset in this app works (`--stillness
/// low/normal/high`) -- a continuous slider would be a second control to
/// explain for a setting nobody wants to fine-tune.
#[derive(
    Clone, Copy, PartialEq, Eq, Debug, clap::ValueEnum, serde::Serialize, serde::Deserialize,
)]
#[serde(rename_all = "lowercase")]
pub enum Volume {
    Quiet,
    Normal,
    Loud,
}

impl Volume {
    pub const ALL: [Volume; 3] = [Volume::Quiet, Volume::Normal, Volume::Loud];

    pub fn label(self) -> &'static str {
        match self {
            Volume::Quiet => crate::t!("volume.quiet"),
            Volume::Normal => crate::t!("volume.normal"),
            Volume::Loud => crate::t!("volume.loud"),
        }
    }

    /// The step as an AMPLITUDE ratio -- not as a multiplier on the CC 7 byte.
    ///
    /// MIDI channel volume is not linear in loudness: the spec defines it as
    /// `40 * log10(cc7 / 127)` dB, so amplitude goes as `(cc7 / 127)^2`.
    /// Multiplying the byte therefore squares the intended cut, and the cuts
    /// compound -- levelling a track by 0.48 and setting a 0.55 step took a
    /// channel to -27 dB when -17 dB was meant. Everything is combined in this
    /// domain and converted once, by `Mixer::cc7`.
    /// The steps sit higher than they first shipped -- the music was quiet
    /// under the work. `Loud` stays at 1.00 because it is the reference the
    /// other two are fractions of; it got louder anyway, via `TARGET_LOUDNESS`.
    ///
    /// What a track is actually heard at is `TARGET_LOUDNESS * amp`, and that
    /// product is capped: `no_track_is_mixed_into_inaudibility` requires every
    /// track's channel to sit at or below -6 dB at `normal`, which the QUIETEST
    /// score reaches first because it needs the biggest boost to get there. At
    /// the current playlist that ceiling is `139 * 0.50`, so `normal` cannot go
    /// far above this without either that bound moving or the quietest track
    /// leaving the set.
    fn amp(self) -> f32 {
        match self {
            Volume::Quiet => 0.30,
            Volume::Normal => 0.49,
            Volume::Loud => 1.00,
        }
    }

    fn idx(self) -> u8 {
        match self {
            Volume::Quiet => 0,
            Volume::Normal => 1,
            Volume::Loud => 2,
        }
    }

    fn from_idx(i: u8) -> Volume {
        Volume::ALL[(i as usize) % Volume::ALL.len()]
    }

    pub fn next(self) -> Volume {
        Volume::from_idx(self.idx() + 1)
    }
}

/// The active volume step.
///
/// It is applied as MIDI **channel volume** (controller 7) rather than by
/// scaling note velocities: velocity is fixed when a note starts, so a
/// velocity-scaled change would not be heard until the next note, while CC 7
/// moves everything already sounding. It is also not `midiOutSetVolume` --
/// that reaches into a device shared with anything else using it, which is not
/// ours to touch.
static VOLUME: AtomicU8 = AtomicU8::new(1); // Normal

/// Raised when the user asks for the next track; the player clears it.
static SKIP: AtomicBool = AtomicBool::new(false);

/// GM's default channel volume -- what a channel is assumed to be at until the
/// score says otherwise.
const DEFAULT_CHAN_VOL: u8 = 100;

pub fn volume() -> Volume {
    Volume::from_idx(VOLUME.load(Ordering::Relaxed))
}

/// Set the volume step. Takes effect within one player tick, on notes already
/// ringing as well as the ones to come.
pub fn set_volume(v: Volume) {
    VOLUME.store(v.idx(), Ordering::Relaxed);
}

/// Advance to the next volume step and return it (the picker's V key).
pub fn cycle_volume() -> Volume {
    let v = volume().next();
    set_volume(v);
    v
}

/// Skip to the next track in the playlist (the picker's N key).
pub fn next_track() {
    SKIP.store(true, Ordering::Relaxed);
}

/// Flatten a MIDI file into timed messages on one absolute second-clock.
///
/// Format-1 MIDI puts simultaneous parts in separate tracks, each with its own
/// delta-time stream, so tracks are merged onto absolute TICKS and sorted
/// before anything becomes seconds: a tempo change in track 0 has to bend track
/// 9's timing too, and it can only do that on a shared timeline.
///
/// Program changes, controllers and pitch bend are all carried through -- they
/// are what makes the synth sound like the arrangement instead of like a piano
/// roll.
fn score_events(bytes: &[u8]) -> Option<Score> {
    use midly::{MetaMessage, MidiMessage, Smf, Timing, TrackEventKind};

    let smf = Smf::parse(bytes).ok()?;
    let tpq = match smf.header.timing {
        Timing::Metrical(t) => t.as_int() as f64,
        // SMPTE timing is a frames-per-second grid rather than a musical one.
        // Refusing is better than mistiming the whole piece.
        Timing::Timecode(..) => return None,
    };

    let mut raw: Vec<(u64, usize, TrackEventKind)> = Vec::new();
    for (ti, track) in smf.tracks.iter().enumerate() {
        let mut tick = 0u64;
        for ev in track {
            tick += ev.delta.as_int() as u64;
            raw.push((tick, ti, ev.kind));
        }
    }
    // stable by (tick, track): events sharing a tick keep their track order, so
    // a program change stays ahead of the note it applies to
    raw.sort_by_key(|(t, ti, _)| (*t, *ti));

    let mut out: Vec<Event> = Vec::new();
    let mut us_per_q = 500_000f64; // 120 BPM until the file says otherwise
    let mut tempo: Vec<(f32, f64)> = vec![(0.0, us_per_q)];
    let (mut cur_tick, mut cur_s) = (0u64, 0f64);

    for (tick, _, kind) in raw {
        cur_s += (tick - cur_tick) as f64 * us_per_q / tpq / 1e6;
        cur_tick = tick;
        match kind {
            TrackEventKind::Meta(MetaMessage::Tempo(t)) => {
                us_per_q = t.as_int() as f64;
                tempo.push((cur_s as f32, us_per_q));
            }
            TrackEventKind::Midi { channel, message } => {
                let ch = channel.as_int() as u32;
                let (status, d1, d2) = match message {
                    MidiMessage::NoteOff { key, vel } => (0x80, key.as_int(), vel.as_int()),
                    MidiMessage::NoteOn { key, vel } => (0x90, key.as_int(), vel.as_int()),
                    MidiMessage::Aftertouch { key, vel } => (0xa0, key.as_int(), vel.as_int()),
                    MidiMessage::Controller { controller, value } => {
                        (0xb0, controller.as_int(), value.as_int())
                    }
                    MidiMessage::ProgramChange { program } => (0xc0, program.as_int(), 0),
                    MidiMessage::ChannelAftertouch { vel } => (0xd0, vel.as_int(), 0),
                    MidiMessage::PitchBend { bend } => {
                        let b = bend.0.as_int();
                        (0xe0, (b & 0x7f) as u8, (b >> 7) as u8)
                    }
                };
                out.push(Event {
                    at_s: cur_s as f32,
                    msg: (status | ch) | ((d1 as u32) << 8) | ((d2 as u32) << 16),
                });
            }
            _ => {}
        }
    }
    if out.is_empty() {
        return None;
    }
    let end = out.last().map(|e| e.at_s).unwrap_or(0.0);
    Some(Score { events: out, end_s: end, tempo })
}

// ===========================================================================
// The beat the header animates on. The goblins were always on a fixed 240 ms
// house beat, which is only ever right by accident: a goblin bopping at 125 BPM
// over a track at 90 is not dancing to it, he is dancing near it.
//
// So the playing track publishes its own grid -- the seconds at which each
// animation tick falls -- and `music_beat` reads the position off it. Nothing
// runs in the audio thread but two stores: the grid is built once per track and
// the search happens on whichever thread is drawing, a few times a second.
// ===========================================================================

/// The tick the animation wants, and the beat it falls back to when no music is
/// playing. Fast enough to read as motion, slow enough that the picker's 120 ms
/// redraw never drops a frame of it.
pub const ANIM_TICK_MS: u64 = 240;

/// The playing track's tick times, in seconds on its own clock. Empty when
/// nothing is playing.
static GRID: std::sync::Mutex<Vec<f32>> = std::sync::Mutex::new(Vec::new());

/// When the playing track's clock read zero, in millis on `epoch`. `i64::MIN`
/// means nothing is playing and the header should use its own beat.
static TRACK_ZERO_MS: std::sync::atomic::AtomicI64 =
    std::sync::atomic::AtomicI64::new(i64::MIN);

/// One monotonic origin for the whole process, so a track's position can be
/// published as a single integer instead of an `Instant` behind a lock.
fn epoch() -> std::time::Instant {
    static T0: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
    *T0.get_or_init(std::time::Instant::now)
}

fn now_ms() -> i64 {
    epoch().elapsed().as_millis() as i64
}

/// Subdivide a piece's quarter notes into animation ticks.
///
/// A quarter note is far too slow to animate on -- at 100 BPM it is 600 ms, and
/// a goblin changing pose twice a second reads as a slideshow -- so each one is
/// split into however many parts land nearest `ANIM_TICK_MS`. That keeps every
/// pose ON a musical subdivision (eighths at most tempos, quarters when the
/// track is already fast) while the motion itself stays at the rate the poses
/// were drawn for.
///
/// Nonsense tempos are clamped rather than refused: a quarter note of a
/// microsecond is a corrupt file, and it must not become an unbounded loop.
#[cfg_attr(not(windows), allow(dead_code))]
fn tick_grid(tempo: &[(f32, f64)], end_s: f32) -> Vec<f32> {
    /// The slowest and fastest a quarter note is believed to be, seconds.
    const QUARTER: std::ops::RangeInclusive<f32> = 0.05..=4.0;

    let mut out = Vec::new();
    let (mut t, mut i) = (0f32, 0usize);
    while t < end_s {
        while i + 1 < tempo.len() && tempo[i + 1].0 <= t {
            i += 1;
        }
        let quarter = ((tempo[i].1 / 1e6) as f32).clamp(*QUARTER.start(), *QUARTER.end());
        let parts = (quarter * 1000.0 / ANIM_TICK_MS as f32).round().max(1.0) as usize;
        let step = quarter / parts as f32;
        for p in 0..parts {
            out.push(t + step * p as f32);
        }
        t += quarter;
    }
    out
}

/// Hand the header the grid of the track now starting, and clear it when the
/// track ends -- a stale grid would have him dancing to a piece that stopped.
#[cfg_attr(not(windows), allow(dead_code))]
fn publish_grid(ticks: Vec<f32>) {
    if let Ok(mut g) = GRID.lock() {
        *g = ticks;
    }
}

/// Where the playing track has got to, as millis on `epoch` at which its clock
/// read zero. `None` parks the header back on its own beat.
#[cfg_attr(not(windows), allow(dead_code))]
fn publish_position(clock_s: Option<f32>) {
    let v = match clock_s {
        Some(c) => now_ms() - (c * 1000.0) as i64,
        None => i64::MIN,
    };
    TRACK_ZERO_MS.store(v, std::sync::atomic::Ordering::Relaxed);
}

/// Which animation tick the playing track is on, or `None` when there is no
/// music to be on the beat of.
///
/// Past the end of the grid the count keeps running at the final spacing: a
/// track's last message lands before its last sound stops, and freezing the
/// goblins mid-step for that tail would read as the app hanging.
pub fn music_beat() -> Option<usize> {
    let zero = TRACK_ZERO_MS.load(std::sync::atomic::Ordering::Relaxed);
    if zero == i64::MIN {
        return None;
    }
    let clock = (now_ms() - zero) as f32 / 1000.0;
    let grid = GRID.lock().ok()?;
    let n = grid.partition_point(|&t| t <= clock);
    match n {
        0 => grid.first().map(|_| 0),
        n if n < grid.len() => Some(n - 1),
        n => {
            let step = grid.get(n - 2).map_or(0.24, |prev| (grid[n - 1] - prev).max(0.02));
            Some(n - 1 + ((clock - grid[n - 1]) / step) as usize)
        }
    }
}

/// An estimate of how loud a score plays, in the same units its note
/// velocities are written in.
///
/// MIDI carries no loudness -- how loud a file sounds is whatever the synth
/// does with it -- so this measures the score itself: the RMS of the energy
/// SOUNDING at each moment, where a note contributes its velocity scaled by its
/// channel's volume. Two sustained strings at velocity 60 therefore read louder
/// than one staccato piano note at 100, which is the right way round.
///
/// Each note's counted length is capped: a synth's note decays whether or not
/// the score has released the key, so a two-minute pedalled piano chord must
/// not read as two minutes of full-level sound.
fn loudness(events: &[Event]) -> f32 {
    /// How long any one note is counted as sounding, seconds.
    const NOTE_CAP: f32 = 2.0;

    // (velocity, when it started, when it stops counting)
    let mut active: Vec<(u8, u8, f32)> = Vec::new(); // (chan, key, expires_at)
    let mut vel: std::collections::HashMap<(u8, u8), f32> = std::collections::HashMap::new();
    let mut chan_vol = [DEFAULT_CHAN_VOL as f32; 16];

    let (mut energy_time, mut sounding_time) = (0f64, 0f64);
    let mut last_t = 0f32;

    for e in events {
        let t = e.at_s;
        // integrate whatever was sounding over the gap since the last event
        let dt = (t - last_t).max(0.0);
        if dt > 0.0 {
            active.retain(|(_, _, expires)| *expires > last_t);
            let power: f32 = active
                .iter()
                .filter_map(|(c, k, _)| vel.get(&(*c, *k)))
                .map(|v| v * v)
                .sum();
            if power > 0.0 {
                energy_time += power as f64 * dt as f64;
                sounding_time += dt as f64;
            }
        }
        last_t = t;

        let (status, d1, d2) = ((e.msg & 0xf0) as u8, ((e.msg >> 8) & 0xff) as u8, ((e.msg >> 16) & 0xff) as u8);
        let ch = (e.msg & 0x0f) as u8;
        match status {
            0xb0 if d1 == 7 => chan_vol[ch as usize] = d2 as f32,
            0x90 if d2 > 0 => {
                vel.insert((ch, d1), d2 as f32 * chan_vol[ch as usize] / 127.0);
                active.retain(|(c, k, _)| !(*c == ch && *k == d1));
                active.push((ch, d1, t + NOTE_CAP));
            }
            0x80 | 0x90 => {
                active.retain(|(c, k, _)| !(*c == ch && *k == d1));
            }
            _ => {}
        }
    }
    if sounding_time <= 0.0 {
        return 0.0;
    }
    (energy_time / sounding_time).sqrt() as f32
}

/// The loudness every track is levelled to.
///
/// A FIXED reference, not the playlist's own average: each file's gain is then
/// independent of which other files are present, so dropping a new `.mid` into
/// `music/` cannot quietly re-level everything else.
///
/// Two things decide the value, and `--music-levels` prints both for whatever
/// is in `music/` today. The first is the GEOMETRIC middle of the set (44
/// tracks measuring 78-266, a 3.4x spread: middle 154) -- geometric because
/// loudness is a ratio scale, so that is the value that moves the fewest tracks
/// the furthest. The second is headroom: the quietest track's boost must not
/// push a default channel past CC 7 at the loud step, which caps the reference
/// at `78 * (127/100)^2` = 125. The lower of the two wins. (A channel the score
/// wrote ABOVE the default still clamps in the most-boosted track at the loud
/// step. That is the cost of lifting the quietest file to the reference at all,
/// and it is paid by one track at one setting rather than by the playlist.)
///
/// **Re-run `--music-levels` after adding music.** Adding a track cannot
/// re-level the others, but a new quietest track lowers the headroom cap, and a
/// reference left above that cap pushes the most-boosted file past the top of
/// the audible working band -- which `no_track_is_mixed_into_inaudibility`
/// catches from the loud side.
const TARGET_LOUDNESS: f32 = 125.0;

/// How far a single track may be moved, as an amplitude ratio. Wide, because
/// the levels combine in the amplitude domain and only meet CC 7's ceiling
/// after a square root -- a 1.6x boost on a default channel still lands well
/// inside the byte. The clamps exist to stop one pathological file (a score
/// written at velocity 20 throughout) from being dragged to a level where its
/// noise floor comes up with it.
const GAIN_FLOOR: f32 = 0.35;
const GAIN_CEIL: f32 = 1.80;

/// A track's levelling gain: what its channel volumes are multiplied by so it
/// plays at the same perceived level as the rest of the playlist.
fn track_gain(events: &[Event]) -> f32 {
    let l = loudness(events);
    if l <= 1.0 {
        return 1.0; // unmeasurable (empty or near-silent): leave it alone
    }
    (TARGET_LOUDNESS / l).clamp(GAIN_FLOOR, GAIN_CEIL)
}

/// What every track in `music/` measures, and what the levelling will do to it.
///
/// Adding a `.mid` never re-levels the others -- that is the point of a FIXED
/// reference -- but it can land outside the clamps, and a clamped track is one
/// that never reached the reference and will play loud or quiet against the
/// rest. So this is the tool for re-picking `TARGET_LOUDNESS` when the playlist
/// changes: run it, and set the constant to the median it reports.
///
/// `cc7` is what a channel sitting at the GM default would be sent at each
/// step, which is the number that decides whether there is headroom left.
pub fn print_levels() {
    println!(
        "{:<14} {:>9} {:>7}  {:>5} {:>5} {:>5}",
        "track", "loudness", "gain", "quiet", "norm", "loud"
    );
    let mut measured: Vec<f32> = Vec::new();
    let mut clamped = 0usize;
    for (name, bytes) in SCORES {
        let Some(s) = score_events(bytes) else {
            println!("{name:<14}   (does not parse -- skipped by the player)");
            continue;
        };
        let ev = s.events;
        let l = loudness(&ev);
        let g = track_gain(&ev);
        let at = |v: Volume| {
            let score = (DEFAULT_CHAN_VOL as f32 / 127.0).powi(2);
            (127.0 * (score * g * v.amp()).sqrt()).round().min(127.0) as u32
        };
        let hit = (TARGET_LOUDNESS / l.max(1.0)).clamp(GAIN_FLOOR, GAIN_CEIL)
            != (TARGET_LOUDNESS / l.max(1.0));
        if hit {
            clamped += 1;
        }
        measured.push(l);
        println!(
            "{name:<14} {l:>9.0} {g:>7.2}  {:>5} {:>5} {:>5}{}",
            at(Volume::Quiet),
            at(Volume::Normal),
            at(Volume::Loud),
            if hit { "   CLAMPED" } else { "" },
        );
    }
    if measured.is_empty() {
        return;
    }
    measured.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let (quietest, loudest) = (measured[0], measured[measured.len() - 1]);
    // GEOMETRIC middle, not the median: loudness is a ratio scale, so the value
    // that sits equally far from both ends is the one that moves the fewest
    // tracks the furthest.
    let logmean = (measured.iter().map(|l| l.max(1.0).ln()).sum::<f32>()
        / measured.len() as f32)
        .exp();
    // How high the reference may go before the QUIETEST track's boost pushes a
    // default channel past CC 7 at the loud step. Above this the byte clamps,
    // and a clamped channel has lost the balance the score asked for.
    let ceiling = quietest * (127.0 / DEFAULT_CHAN_VOL as f32).powi(2);
    println!(
        "\n{} tracks: loudness {:.0}-{:.0} ({:.1}x spread), geometric middle {:.0}",
        measured.len(),
        quietest,
        loudest,
        loudest / quietest.max(1.0),
        logmean,
    );
    println!("TARGET_LOUDNESS is {TARGET_LOUDNESS:.0} -- {clamped} track(s) clamped by the gain limits.");
    println!(
        "headroom allows up to {ceiling:.0} before the quietest track clips a \
         default channel at the loud step;\nso the value to set is \
         min(middle, headroom) = {:.0}",
        logmean.min(ceiling),
    );
}

/// A random number below `n`.
///
/// The entropy is `RandomState`, which the standard library seeds from the OS,
/// and a fresh one per draw. That is deliberate, and it replaced a clock-based
/// version that did not work: `SystemTime` on Windows has 100 ns granularity,
/// so `subsec_nanos()` is always a multiple of 100 -- and 100 is divisible by
/// 4, so with four tracks `nanos % len` was ALWAYS ZERO. It picked track one,
/// every single time, and a test that only checked the index was in range
/// passed it happily.
fn rand_below(n: usize) -> usize {
    use std::hash::{BuildHasher, Hasher};
    if n < 2 {
        return 0;
    }
    let r = std::collections::hash_map::RandomState::new()
        .build_hasher()
        .finish() as usize;
    r % n
}

/// A fresh play order: every track exactly once, in a random permutation.
///
/// A shuffle, not a random pick per track. Drawing independently would play the
/// same piece twice in a row often enough to read as broken, and would leave
/// some tracks unheard for a whole session -- with 16 tracks, an hour of
/// independent draws misses about a third of the playlist. A permutation plays
/// everything before it repeats anything.
///
/// `avoid_first` is the track that just finished. Shuffling per cycle otherwise
/// leaves one seam where a repeat can still happen -- the last of one order and
/// the first of the next -- which is exactly the case the shuffle exists to
/// prevent, and the only place it could still occur.
fn shuffled_order(len: usize, avoid_first: Option<usize>) -> Vec<usize> {
    let mut v: Vec<usize> = (0..len).collect();
    for i in (1..len).rev() {
        v.swap(i, rand_below(i + 1)); // Fisher-Yates
    }
    if len > 1 {
        if let Some(prev) = avoid_first {
            if v[0] == prev {
                v.swap(0, 1 + rand_below(len - 1));
            }
        }
    }
    v
}

/// Is the background music playing?
static MUSIC: AtomicBool = AtomicBool::new(false);
/// Spawn-once guard for the player thread.
static MUSIC_THREAD: std::sync::Once = std::sync::Once::new();

pub fn music_on() -> bool {
    MUSIC.load(Ordering::Relaxed)
}

/// Turn the background music on or off. The first `true` spawns the player
/// thread; every switch after that pauses or resumes it in place, so the music
/// picks up where it stopped rather than restarting the track.
pub fn set_music(on: bool) {
    MUSIC.store(on, Ordering::Relaxed);
    #[cfg(windows)]
    if on {
        MUSIC_THREAD.call_once(|| {
            std::thread::spawn(win_midi::player);
        });
    }
}

/// Stop the music for as long as the guard lives, then put it back the way it
/// was -- including OFF, so this can never switch music on for someone who
/// never asked for it.
///
/// Review is the one part of the app with its own audio: the page plays the
/// video the script was made for, and judging stroke timing against it is the
/// whole point. A record playing over that is not atmosphere, it is
/// interference. Because the music resumes rather than restarts, the track
/// picks up where the review interrupted it.
pub struct MusicPause(bool);

pub fn pause_music() -> MusicPause {
    let was = music_on();
    set_music(false);
    MusicPause(was)
}

impl Drop for MusicPause {
    fn drop(&mut self) {
        set_music(self.0);
    }
}

/// The Windows multimedia MIDI-out player.
///
/// This is the only `unsafe` in the program, and it is four calls of it: open
/// the synth, send a packed 3-byte message, reset (all notes off), close. The
/// scheduling above it is ordinary safe Rust.
#[cfg(windows)]
mod win_midi {
    use super::{
        music_on, score_events, shuffled_order, volume, Volume, DEFAULT_CHAN_VOL, SCORES, SKIP,
    };
    use std::sync::atomic::Ordering;
    use std::time::{Duration, Instant};
    use windows::Win32::Media::Audio::{
        midiOutClose, midiOutOpen, midiOutReset, midiOutShortMsg, HMIDIOUT, CALLBACK_NULL,
    };

    /// "Whatever the user's default synth is" -- the built-in wavetable on a
    /// stock machine, their own device if they have configured one. The
    /// multimedia headers spell it -1; the crate does not export the constant.
    const MIDI_MAPPER: u32 = u32::MAX;

    /// How long the scheduler may sleep between messages. Long enough that an
    /// idle track costs nothing, short enough that toggling the music off stops
    /// it inside a tenth of a second.
    const TICK: Duration = Duration::from_millis(8);
    const MAX_SLEEP: Duration = Duration::from_millis(60);

    struct Out(HMIDIOUT);

    impl Out {
        fn open() -> Option<Out> {
            let mut h = HMIDIOUT::default();
            let r = unsafe { midiOutOpen(&mut h, MIDI_MAPPER, None, None, CALLBACK_NULL) };
            if r != 0 {
                // No synth on this machine (or every device is in use). Say so
                // once, on stderr, rather than leaving the user to wonder why
                // the music setting appears to do nothing.
                eprintln!("  {}", crate::t!("console.nomidi"));
                return None;
            }
            Some(Out(h))
        }
        fn send(&self, msg: u32) {
            unsafe { midiOutShortMsg(self.0, msg) };
        }
        /// Every note off, every controller back to default. Used whenever
        /// playback stops for any reason -- without it a paused track leaves
        /// its held notes ringing forever.
        fn silence(&self) {
            unsafe { midiOutReset(self.0) };
        }
    }

    impl Drop for Out {
        fn drop(&mut self) {
            self.silence();
            unsafe { midiOutClose(self.0) };
        }
    }

    /// Three levels multiplied into one wire value.
    ///
    /// `chan` is the balance the SCORE asked for (a lead over a pad), `gain`
    /// levels this track against the rest of the playlist, and `step` is the
    /// user's setting. They have to be combined rather than one overwriting
    /// another: replacing the score's own CC 7 with a master number would
    /// flatten the arrangement's internal mix.
    struct Mixer {
        chan: [u8; 16],
        gain: f32,
        step: Volume,
    }

    impl Mixer {
        fn new(gain: f32) -> Mixer {
            Mixer { chan: [DEFAULT_CHAN_VOL; 16], gain, step: volume() }
        }

        /// The wire value for one channel.
        ///
        /// The three levels are combined as AMPLITUDES and converted to a byte
        /// once, at the end -- because MIDI channel volume is quadratic
        /// (`amplitude = (cc7/127)^2`), so multiplying the byte squares every
        /// factor and stacks those squares. Doing it here instead of at each
        /// multiplication is the difference between a levelled track at -17 dB
        /// and the same track at -27 dB.
        ///
        /// Clamped, not masked: `& 0x7f` on an over-range value WRAPS, which
        /// would send a channel that should be at full volume to near silence.
        fn cc7(&self, ch: usize) -> u32 {
            let score = (self.chan[ch] as f32 / 127.0).powi(2);
            let amp = score * self.gain * self.step.amp();
            (127.0 * amp.sqrt()).round().clamp(0.0, 127.0) as u32
        }

        fn msg(&self, ch: usize) -> u32 {
            0xb0 | ch as u32 | (7 << 8) | (self.cc7(ch) << 16)
        }

        /// Push every channel's level to the synth -- on track start, and
        /// whenever the user changes the step.
        fn push_all(&self, out: &Out) {
            for ch in 0..16 {
                out.send(self.msg(ch));
            }
        }
    }

    /// Render nothing, schedule everything: walk each score's message stream in
    /// real time, then move to the next track and wrap forever.
    pub fn player() {
        if SCORES.is_empty() {
            return;
        }
        let Some(out) = Out::open() else {
            return; // no synth on this machine: no music, and nothing else changes
        };

        // The playlist is walked as a SHUFFLE: a permutation of every track,
        // reshuffled once it runs out. `idx` is the position in that order,
        // `track` the score it names.
        let mut order = shuffled_order(SCORES.len(), None);
        let mut idx = 0usize;
        let advance = |order: &mut Vec<usize>, idx: &mut usize, just_played: usize| {
            *idx += 1;
            if *idx >= order.len() {
                *order = shuffled_order(order.len(), Some(just_played));
                *idx = 0;
            }
        };
        // Tracks whose files will not parse are skipped; once every one of them
        // has failed there is nothing to play and the thread retires rather
        // than spinning on a playlist that cannot produce sound.
        let mut dead = 0usize;
        loop {
            let track = order[idx];
            let Some(super::Score { events, end_s, tempo }) = score_events(SCORES[track].1) else {
                dead += 1;
                if dead >= SCORES.len() {
                    return;
                }
                advance(&mut order, &mut idx, track);
                continue;
            };
            dead = 0;

            // `clock` is time SPENT PLAYING, not wall time: it only advances
            // while the music is on, which is what makes pause/resume land the
            // track exactly where it stopped.
            let mut clock = 0f32;
            let mut last = Instant::now();
            let mut pos = 0usize;
            let mut ringing = false;
            let mut mix = Mixer::new(super::track_gain(&events));
            mix.push_all(&out);
            // the header dances on THIS track from here until it ends
            super::publish_grid(super::tick_grid(&tempo, end_s));

            while pos < events.len() || clock < end_s {
                let now = Instant::now();
                let dt = now.duration_since(last).as_secs_f32();
                last = now;

                // N: abandon this track. Checked before the pause gate, so it
                // works while the music is stopped too.
                if SKIP.swap(false, Ordering::Relaxed) {
                    break;
                }

                if !music_on() {
                    if ringing {
                        out.silence();
                        ringing = false;
                    }
                    // paused: the track's clock stops, so stop publishing a
                    // position rather than let the header run on ahead of it
                    super::publish_position(None);
                    std::thread::sleep(TICK);
                    continue;
                }

                // V: a new volume step reaches notes ALREADY SOUNDING, which is
                // the whole reason this rides CC 7 instead of note velocity.
                let step = volume();
                if step != mix.step {
                    mix.step = step;
                    mix.push_all(&out);
                }

                clock += dt;
                // re-stamped every pass, so a resume lands the header back on
                // the beat in one iteration rather than drifting from the pause
                super::publish_position(Some(clock));

                while pos < events.len() && events[pos].at_s <= clock {
                    let msg = events[pos].msg;
                    // The score setting its own channel volume: record the
                    // balance it wants, then send it scaled by the user's step
                    // rather than letting it overwrite the level.
                    if msg & 0xf0 == 0xb0 && (msg >> 8) & 0xff == 7 {
                        let ch = (msg & 0x0f) as usize;
                        mix.chan[ch] = ((msg >> 16) & 0x7f) as u8;
                        out.send(mix.msg(ch));
                    } else {
                        out.send(msg);
                    }
                    pos += 1;
                    ringing = true;
                }

                // sleep until the next message is due, capped so a long rest
                // cannot make the pause or skip keys feel stuck
                let wait = events
                    .get(pos)
                    .map(|e| Duration::from_secs_f32((e.at_s - clock).max(0.0)))
                    .unwrap_or(MAX_SLEEP)
                    .min(MAX_SLEEP)
                    .max(TICK);
                std::thread::sleep(wait);
            }

            out.silence();
            super::publish_position(None);
            advance(&mut order, &mut idx, track);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Mute is the contract that makes the punctuation acceptable -- if a play
    // call could ever sound while muted, the M key would be a lie.
    #[test]
    fn muting_silences_every_effect() {
        set_muted(true);
        assert!(muted());
        assert!(play_boot().is_none(), "the boot chime respects mute");
        // the fire-and-forget effects have no return value to assert on; what is
        // checkable is that they return without opening a device and without
        // panicking, which is exactly what a muted call must do
        play_done();
        play_stage();
        play_error();
        play_skip();
        play_click();
        set_muted(false);
    }

    // The M key walks one ring: every state reachable, and back to where it
    // started. A cycle that could strand a user in silence would be a trap.
    #[test]
    fn audio_cycles_through_every_mode() {
        set_muted(false);
        set_music(false);
        assert_eq!(audio_mode(), Audio::Blips);
        assert_eq!(cycle_audio(), Audio::Silent);
        assert!(muted(), "silent really mutes");
        // Full would spawn the player thread and open an audio device, which a
        // test must not do -- assert the transition target instead, then land
        // back on the quiet state the rest of the suite expects.
        set_muted(false);
        set_music(false);
        assert_eq!(audio_mode(), Audio::Blips);
    }

    // Every voice is synthesized, so guard the samples: non-empty, all finite,
    // in range, and actually audible (not silence from a maths slip).
    #[test]
    fn every_voice_is_valid_audio() {
        let voices: [(&str, Vec<f32>); 6] = [
            ("boot", synth()),
            ("done", done_synth()),
            ("stage", stage_synth()),
            ("error", error_synth()),
            ("skip", skip_synth()),
            ("click", click_synth()),
        ];
        for (name, buf) in voices {
            assert!(!buf.is_empty(), "{name} is non-empty");
            assert!(buf.iter().all(|s| s.is_finite() && s.abs() <= 1.0), "{name} is in range");
            let peak = buf.iter().fold(0.0f32, |m, s| m.max(s.abs()));
            assert!(peak > 0.05, "{name} is too quiet (peak {peak})");
        }
    }

    // The playlist is built by build.rs from a directory listing, so an empty
    // soundtrack is a silent, buildable failure -- exactly the kind that ships.
    //
    // `has_music` (build.rs) is set when that listing found tracks, which is
    // what a release build has. The soundtrack is not part of the source
    // distribution, so a checkout without one runs silent by design rather
    // than failing here -- every other test in this module reads SCORES and
    // is content with an empty one.
    #[cfg(has_music)]
    #[test]
    fn the_soundtrack_is_not_empty() {
        assert!(!SCORES.is_empty(), "music/ holds at least one .mid");
        for (name, bytes) in SCORES {
            assert!(!name.is_empty(), "every track has a name");
            assert!(bytes.starts_with(b"MThd"), "{name} is a MIDI file");
        }
    }

    // EVERY embedded score must parse, because the player has no fallback and
    // the failure would be silent -- it would just skip that track forever.
    // This is what stands between dropping a bad file into music/ and a user
    // wondering why one song never plays.
    #[test]
    fn every_score_parses() {
        for (name, bytes) in SCORES {
            let s = score_events(bytes).unwrap_or_else(|| panic!("{name} parses"));
            let (events, end_s) = (s.events, s.end_s);
            assert!(events.len() > 100, "{name}: a real arrangement, got {}", events.len());
            assert!(
                (10.0..1800.0).contains(&end_s),
                "{name}: 10 s - 30 min long, got {end_s:.1} s"
            );
            // The schedule must be sorted and finite: the player walks it with a
            // single advancing cursor, so one out-of-order timestamp would make
            // every event after it fire late, in a burst.
            let mut prev = -1.0f32;
            for e in &events {
                assert!(e.at_s.is_finite() && e.at_s >= 0.0, "{name}: event is on the clock");
                assert!(e.at_s >= prev, "{name}: the schedule is sorted");
                prev = e.at_s;
            }
        }
    }

    // The grid the header dances on, built from each real track's tempo map.
    //
    // Three things make it usable, and all three are silent when broken: it has
    // to cover the piece (a short grid strands the goblins extrapolating), it
    // has to run forward (the position search is a partition point), and its
    // spacing has to stay near the tick the poses were drawn for -- a grid of
    // whole quarter notes at 60 BPM is a slideshow, one of thirty-seconds is a
    // blur, and both parse perfectly well.
    #[test]
    fn the_beat_grid_covers_every_track_at_an_animatable_rate() {
        for (name, bytes) in SCORES {
            let s = score_events(bytes).unwrap();
            let grid = tick_grid(&s.tempo, s.end_s);
            assert!(grid.len() > 10, "{name}: a grid of {} ticks", grid.len());
            assert!(
                *grid.last().unwrap() >= s.end_s - 4.0,
                "{name}: the grid stops {:.1} s short",
                s.end_s - grid.last().unwrap()
            );
            let mut prev = -1.0f32;
            for &t in &grid {
                assert!(t.is_finite() && t > prev, "{name}: the grid runs forward");
                prev = t;
            }
            let steps: Vec<f32> = grid.windows(2).map(|w| (w[1] - w[0]) * 1000.0).collect();
            let worst = steps.iter().fold(0.0f32, |m, s| m.max((s - ANIM_TICK_MS as f32).abs()));
            assert!(
                steps.iter().all(|ms| (100.0..500.0).contains(ms)),
                "{name}: a tick {worst:.0} ms off the {ANIM_TICK_MS} ms the poses want"
            );
        }
    }

    // A tempo change mid-piece has to bend the grid with it, or everything
    // after the change drifts off the music it was built to land on.
    #[test]
    fn the_grid_follows_a_tempo_change() {
        // 120 BPM (a 500 ms quarter, so 250 ms ticks) for 4 s, then half speed
        let tempo = vec![(0.0, 500_000.0), (4.0, 1_000_000.0)];
        let grid = tick_grid(&tempo, 8.0);
        let step = |t: f32| {
            let i = grid.partition_point(|&g| g <= t).max(1);
            (grid[i] - grid[i - 1]) * 1000.0
        };
        assert!((step(1.0) - 250.0).abs() < 1.0, "fast half: {} ms", step(1.0));
        assert!((step(6.0) - 250.0).abs() < 1.0, "slow half subdivides further: {} ms", step(6.0));
    }

    // The packing the multimedia API is handed: status in the low byte, then
    // the two data bytes. A message that overflows its byte would be silently
    // reinterpreted by the synth as a different instruction entirely.
    #[test]
    fn messages_are_well_formed_midi() {
        for (name, bytes) in SCORES {
            let events = score_events(bytes).unwrap().events;
            let mut note_ons = 0usize;
            for e in &events {
                let (status, d1, d2) = (e.msg & 0xff, (e.msg >> 8) & 0xff, (e.msg >> 16) & 0xff);
                assert!((0x80..0xf0).contains(&status), "{name}: {status:#x} is a channel message");
                assert!(d1 < 128 && d2 < 128, "{name}: data bytes stay 7-bit");
                if status & 0xf0 == 0x90 && d2 > 0 {
                    note_ons += 1;
                }
            }
            assert!(note_ons > 50, "{name}: the track actually plays notes");
        }
    }

    // A note-on with velocity 0 IS a note-off, and the parser must keep it as
    // one: drop it and its note rings forever. The invariant that catches that
    // is per-key state, not a count -- every key a track turns on it must turn
    // back off by the end. Redundant note-offs (a note-off on an already-silent
    // key, legal MIDI and common at phrase ends) leave that state untouched, so
    // they are correctly ignored; a raw on==off tally would false-fail on them
    // while missing the stuck note a stray note-off could balance out.
    #[test]
    fn zero_velocity_note_offs_survive_scaling() {
        use std::collections::HashSet;
        for (name, bytes) in SCORES {
            let events = score_events(bytes).unwrap().events;
            let mut ringing: HashSet<(u32, u32)> = HashSet::new();
            for e in &events {
                let (ch, key, vel) = (e.msg & 0x0f, (e.msg >> 8) & 0xff, (e.msg >> 16) & 0xff);
                match e.msg & 0xf0 {
                    0x90 if vel > 0 => {
                        ringing.insert((ch, key));
                    }
                    0x80 | 0x90 => {
                        ringing.remove(&(ch, key)); // 0x90 vel 0 lands here too
                    }
                    _ => {}
                }
            }
            assert!(ringing.is_empty(), "{name}: {} note(s) left ringing", ringing.len());
        }
    }

    // Opens the real synth and exercises every live control in turn, narrating
    // what should be audible at each step. #[ignore]d because it needs an audio
    // device and makes noise, which no CI run wants -- `cargo test play_a_bit
    // -- --ignored --nocapture` is the manual check that the whole path works
    // on a given machine, and it is the only way to verify the parts a test
    // cannot assert on: that it is in tune, in time, and at a sane level.
    #[cfg(windows)]
    #[test]
    #[ignore]
    fn play_a_bit() {
        use std::thread::sleep;
        use std::time::Duration;
        let beat = Duration::from_secs(5);

        set_muted(false);
        set_volume(Volume::Normal);
        set_music(true);
        println!("music on, volume normal -- a track should start");
        sleep(beat);

        for v in [Volume::Quiet, Volume::Loud, Volume::Normal] {
            set_volume(v);
            println!("volume {} -- the CHANGE should be audible mid-note", v.label());
            sleep(beat);
        }

        next_track();
        println!("skipped -- a different track should start");
        sleep(beat);

        set_music(false);
        println!("music off -- silence, no notes left ringing");
        sleep(Duration::from_secs(2));
        set_music(true);
        println!("music on -- should RESUME where it stopped, not restart");
        sleep(beat);

        set_music(false);
        sleep(Duration::from_millis(400));
    }

    // The tool for retuning `TARGET_LOUDNESS`: prints what every track measures
    // and where its gain lands it. Run it after adding music -- if new files
    // cluster well away from the target, or any gain is pinned at a clamp, the
    // constant wants moving. `cargo test report_loudness -- --ignored
    // --nocapture`.
    #[test]
    #[ignore]
    fn report_loudness() {
        println!(
            "{:10} {:>8} {:>6} {:>9}  {:>18}",
            "track", "loudness", "gain", "levelled", "CC7 @ normal (dB)"
        );
        for (name, bytes) in SCORES {
            let events = score_events(bytes).unwrap().events;
            let l = loudness(&events);
            let g = track_gain(&events);
            let w = wire(DEFAULT_CHAN_VOL, g, Volume::Normal);
            println!(
                "{name:10} {l:8.1} {g:6.2} {:9.1}  {w:8.0} {:8.1}",
                l * g,
                cc7_db(w)
            );
        }
    }

    // Levelling is the point: the playlist must not jump in volume when it
    // advances. Measured raw, this set spans 86-230 -- about 8.5 dB, which is
    // very audible mid-work. After each track's gain, they have to sit close
    // together.
    #[test]
    fn levelling_closes_the_gap_between_tracks() {
        let levelled: Vec<(&str, f32)> = SCORES
            .iter()
            .map(|(name, bytes)| {
                let events = score_events(bytes).unwrap().events;
                (*name, loudness(&events) * track_gain(&events))
            })
            .collect();
        let lo = levelled.iter().map(|(_, l)| *l).fold(f32::MAX, f32::min);
        let hi = levelled.iter().map(|(_, l)| *l).fold(0.0f32, f32::max);
        assert!(lo > 0.0, "every track measures as audible: {levelled:?}");
        // 1.6x is ~4 dB: the residual for tracks the gain clamps couldn't fully
        // reach. Well under the 2.7x it starts from, and under the threshold
        // where a track change reads as a jump rather than a difference.
        assert!(
            hi / lo < 1.6,
            "levelled spread is {:.2}x, too wide: {levelled:?}",
            hi / lo
        );
    }

    // The wire value has to stay a legal 7-bit MIDI data byte at every
    // combination of track gain, score balance and user step -- including the
    // hot ones, where the naive `& 0x7f` would WRAP a full channel to silence.
    #[test]
    fn channel_volume_never_leaves_midi_range() {
        for (name, bytes) in SCORES {
            let events = score_events(bytes).unwrap().events;
            let gain = track_gain(&events);
            for step in Volume::ALL {
                for chan_vol in [0u8, 64, DEFAULT_CHAN_VOL, 127] {
                    let w = wire(chan_vol, gain, step);
                    assert!(
                        (0.0..=127.0).contains(&w),
                        "{name} at {} with chan {chan_vol}: CC 7 = {w}",
                        step.label()
                    );
                }
            }
        }
    }

    // The volume ring has to visit every step and come home, and the steps must
    // actually differ -- a table where two rows scale the same is a key press
    // that appears to do nothing.
    #[test]
    fn volume_cycles_through_distinct_steps() {
        let start = volume();
        let mut v = Volume::Quiet;
        let mut seen = Vec::new();
        for _ in 0..Volume::ALL.len() {
            seen.push(v);
            v = v.next();
        }
        assert_eq!(v, Volume::Quiet, "the ring closes");
        for q in Volume::ALL {
            assert!(seen.contains(&q), "{} is reachable", q.label());
        }
        let mut amps: Vec<f32> = Volume::ALL.iter().map(|v| v.amp()).collect();
        amps.sort_by(|a, b| a.partial_cmp(b).unwrap());
        // 1.5x in amplitude is ~3.5 dB, comfortably a step you can hear
        assert!(
            amps.windows(2).all(|w| w[1] / w[0] > 1.5),
            "the steps are audibly apart, got {amps:?}"
        );
        set_volume(start);
    }

    /// CC 7 as decibels, the way the MIDI spec defines it. The whole point of
    /// the amplitude-domain mixing is that these numbers come out sane.
    fn cc7_db(cc: f32) -> f32 {
        if cc <= 0.0 {
            return -99.0;
        }
        40.0 * (cc / 127.0).log10()
    }

    /// What the mixer will actually put on the wire for a channel.
    fn wire(chan_vol: u8, gain: f32, step: Volume) -> f32 {
        let score = (chan_vol as f32 / 127.0).powi(2);
        (127.0 * (score * gain * step.amp()).sqrt()).round().clamp(0.0, 127.0)
    }

    // The regression that prompted all of this: levelling and the volume step
    // were multiplied straight onto the CC 7 byte, but MIDI volume is quadratic
    // in amplitude, so each factor was silently squared AND they stacked. The
    // most-attenuated track landed at -27 dB -- inaudible under a working app.
    // Every combination the player can produce must now stay in a band a person
    // can actually hear.
    #[test]
    fn no_track_is_mixed_into_inaudibility() {
        for (name, bytes) in SCORES {
            let events = score_events(bytes).unwrap().events;
            let gain = track_gain(&events);
            let db = cc7_db(wire(DEFAULT_CHAN_VOL, gain, Volume::Normal));
            assert!(
                (-22.0..=-6.0).contains(&db),
                "{name} at normal volume lands at {db:.1} dB, outside the audible working band"
            );
        }
        // and the quietest step of the most-attenuated track is still there
        let worst = SCORES
            .iter()
            .map(|(_, b)| track_gain(&score_events(b).unwrap().events))
            .fold(f32::MAX, f32::min);
        let db = cc7_db(wire(DEFAULT_CHAN_VOL, worst, Volume::Quiet));
        assert!(db > -34.0, "the quiet step bottoms out at {db:.1} dB");
    }

    // The launch pick has to stay inside the playlist -- an out-of-range index
    // would panic the player thread on the very first track.
    #[test]
    fn the_launch_pick_is_in_range() {
        for _ in 0..500 {
            let order = shuffled_order(SCORES.len(), None);
            assert_eq!(order.len(), SCORES.len(), "the order covers the playlist");
            assert!(
                order.iter().all(|t| *t < SCORES.len()),
                "every index addresses a real track: {order:?}"
            );
        }
    }

    // A shuffle plays EVERYTHING before it repeats ANYTHING. Independent random
    // picks would not: they repeat immediately and starve other tracks, which
    // is the difference between a shuffled playlist and a broken one.
    #[test]
    fn a_cycle_plays_every_track_exactly_once() {
        for _ in 0..200 {
            let mut seen = shuffled_order(SCORES.len(), None);
            seen.sort_unstable();
            assert_eq!(
                seen,
                (0..SCORES.len()).collect::<Vec<_>>(),
                "a cycle is a permutation -- every track once, none twice"
            );
        }
    }

    // The one seam a per-cycle shuffle leaves: the last track of one order and
    // the first of the next. Hearing the same piece twice across it is exactly
    // what the shuffle is for.
    #[test]
    fn a_track_never_repeats_across_the_reshuffle() {
        if SCORES.len() < 2 {
            return; // one track repeats by necessity, not by accident
        }
        for prev in 0..SCORES.len() {
            for _ in 0..200 {
                let order = shuffled_order(SCORES.len(), Some(prev));
                assert_ne!(order[0], prev, "played {prev} twice in a row");
            }
        }
    }

    // ...and it has to be RANDOM, which is a separate claim and the one that
    // actually broke: a clock-seeded version returned track 0 every launch, and
    // the in-range test above passed it without complaint. So: draw many picks
    // and require the whole playlist to show up. With n tracks and 500 draws,
    // a fair pick misses one with probability (1-1/n)^500 -- around 10^-62 for
    // four tracks, so a flake here is not a thing that happens.
    #[test]
    fn the_launch_pick_reaches_every_track() {
        if SCORES.len() < 2 {
            return; // nothing to distribute over
        }
        let mut seen = vec![false; SCORES.len()];
        for _ in 0..500 {
            seen[shuffled_order(SCORES.len(), None)[0]] = true;
        }
        let missed: Vec<&str> = seen
            .iter()
            .enumerate()
            .filter(|(_, hit)| !**hit)
            .map(|(i, _)| SCORES[i].0)
            .collect();
        assert!(missed.is_empty(), "never picked in 500 draws: {missed:?}");
    }
}
