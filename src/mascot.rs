//! The goblin, animated: every pose he can strike and the clock he strikes them
//! on, in one place because four surfaces draw him -- the startup report
//! (`bios.rs`), the intro demo and the picker header (`tui.rs`), and the header
//! pinned above the processing log (`chrome.rs`).
//!
//! What he is doing is a pure function of the beat clock and the audio state, so
//! every surface agrees on it without any of them owning animation state, and a
//! screen that opens mid-dance opens mid-dance. The drawing itself starts from
//! `theme::MASCOT` -- the resting pose every gesture here returns to -- so all of
//! them are one goblin moving rather than several drawings of one.
//!
//! Rarely -- about once every five minutes -- a file of small goblins marches
//! across the header, out from behind the band and away behind the wordmark,
//! standing in the same rule the mascot stands on, and he stops what he is
//! doing to watch them go past. That is the one thing here big enough to draw a
//! user's eye off their work, which is exactly why it is rationed: it stops
//! being worth looking up at the moment it becomes expected.
//!
//! His EARS never move: they are up and alert in every pose, asleep or dancing.
//! Two strokes at that size cannot be folded -- an ear drawn any other way stops
//! reading as an ear that moved and starts reading as a notch cut out of his
//! head. Everything a lowered ear was reaching for (tired, startled, sulking)
//! the eyes, arms and legs say better, and they are what carry it here.
//! `the_ears_are_always_up` is the test that holds it.

use crate::theme::{art_line, con, theme};
use console::{measure_text_width, style, truncate_str};
use std::time::Instant;

/// One pose: four rows, the block `theme::MASCOT` occupies. Rows are padded to
/// `MASCOT_W` at draw time, so a pose only has to stay INSIDE that width --
/// overrun would push the wordmark out of its column. Rows may sit at different
/// indents from each other (that is how he leans) and may be blank (that is how
/// he leaves the ground).
pub type Pose = [&'static str; 4];

/// Stood there, the drawing every gesture returns to.
const REST: Pose = crate::theme::MASCOT;

/// Every face he is allowed to pull, and what each one DELIVERS -- not what it
/// was meant to mean. Three characters between the parentheses, always:
///
/// - `o.o` open, level, the resting face
/// - `-.-` shut: a closed eye is a horizontal line, in every alphabet there is
/// - `>.<` screwed up: both eyes squeezing inward, which is the drawing, not a
///   convention that has to be known
/// - `-.o` one shut, one open -- squinting at whatever the hands are doing
/// - `^.^` pleased: the arc of a squinting smile, and the one borrowed idiom
///   here that a reader either knows or reads as harmless
/// - `___` no face at all, which is the back of his head (the spin)
///
/// A glyph that means something ELSE in the reader's head is worse than no
/// glyph: `x.x` was drawn as the peak of a sneeze and reads, universally, as a
/// character who has died. Anything expressive that is not on this list belongs
/// in the arms, the legs or the stance, which say things by drawing them.
///
/// Nothing DRAWS from this list -- the poses carry their own faces. It is the
/// rule written down where the art is, and `the_faces_are_ones_that_read` is
/// the test that reads it, which is why it is built only for the tests.
#[cfg(test)]
const FACES: [&str; 6] = ["o.o", "-.-", ">.<", "-.o", "^.^", "___"];

// --- idle gestures -------------------------------------------------------
// Each is a short sequence played one pose per beat, dropped somewhere inside
// an idle cycle with the rest of the cycle spent at REST. Small and specific
// beats big and generic: a blink nobody consciously notices is worth more than
// a flourish that draws the eye off the listing every four seconds.

/// A slow blink, and the double-take of a second one.
const G_BLINK: [Pose; 1] = [[" /\\,/\\", "\\(-.-)/", " |___|", "  / \\"]];
const G_BLINK2: [Pose; 3] = [
    [" /\\,/\\", "\\(-.-)/", " |___|", "  / \\"],
    REST,
    [" /\\,/\\", "\\(-.-)/", " |___|", "  / \\"],
];

/// Craning to look at something: the HEAD leans out over a planted body, ears
/// travelling with it because they are attached to it. Held four beats, because
/// a look that snaps back never happened.
///
/// It leans one way only. The face already sits at the left edge of the block,
/// so there is no column to lean into on that side, and moving the body instead
/// would read as shifting his weight -- the eye follows the larger mass. This
/// is also what makes it a different gesture from `G_PEEK`, which leans the
/// whole goblin.
const G_CRANE: [Pose; 4] = [
    ["  /\\,/\\", " \\(o.o)/", " |___|", "  / \\"],
    ["  /\\,/\\", " \\(o.o)/", " |___|", "  / \\"],
    ["  /\\,/\\", " \\(o.o)/", " |___|", "  / \\"],
    ["  /\\,/\\", " \\(o.o)/", " |___|", "  / \\"],
];

/// Working at an itch behind one ear, squinting at it -- the arm does the
/// scratching, since the ear itself never moves (`the_ears_are_always_up`).
const G_SCRATCH: [Pose; 4] = [
    [" /\\,/\\", "\\(-.o)|", " |___|", "  / \\"],
    [" /\\,/\\", "\\(-.o)/", " |___|", "  / \\"],
    [" /\\,/\\", "\\(-.o)|", " |___|", "  / \\"],
    [" /\\,/\\", "\\(-.o)/", " |___|", "  / \\"],
];

/// A stretch: arms down, out, then up on his toes with his eyes shut.
const G_STRETCH: [Pose; 5] = [
    [" /\\,/\\", "/(o.o)\\", " |___|", "  / \\"],
    [" /\\,/\\", "-(o.o)-", " |___|", "  / \\"],
    [" /\\,/\\", "\\(-.-)/", " |___|", "  | |"],
    [" /\\,/\\", "\\(-.-)/", " |___|", "  | |"],
    [" /\\,/\\", "-(o.o)-", " |___|", "  / \\"],
];

/// A shrug: arms out, eyes half shut. Held, because a shrug that snaps back is
/// a twitch.
const G_SHRUG: [Pose; 3] = [
    [" /\\,/\\", "-(-.-)-", " |___|", "  / \\"],
    [" /\\,/\\", "-(-.-)-", " |___|", "  / \\"],
    [" /\\,/\\", "-(-.-)-", " |___|", "  / \\"],
];

/// A wave at whoever is reading the listing.
const G_WAVE: [Pose; 4] = [
    [" /\\,/\\", "\\(^.^)-", " |___|", "  / \\"],
    [" /\\,/\\", "\\(^.^)/", " |___|", "  / \\"],
    [" /\\,/\\", "\\(^.^)-", " |___|", "  / \\"],
    [" /\\,/\\", "\\(^.^)/", " |___|", "  / \\"],
];

/// Hands on hips, tapping a foot -- the pose of a goblin waiting on a human to
/// pick a folder.
const G_TAP: [Pose; 6] = [
    [" /\\,/\\", "|(o.o)|", " |___|", "  / |"],
    [" /\\,/\\", "|(o.o)|", " |___|", "  / \\"],
    [" /\\,/\\", "|(o.o)|", " |___|", "  / |"],
    [" /\\,/\\", "|(o.o)|", " |___|", "  / \\"],
    [" /\\,/\\", "|(o.o)|", " |___|", "  / |"],
    [" /\\,/\\", "|(o.o)|", " |___|", "  / \\"],
];

/// A shudder: he screws his eyes up, it goes through him -- arms flung out,
/// feet thrown apart -- and two beats to come back down.
///
/// A shudder rather than the sneeze it was drawn as, because that is what the
/// drawing delivers: with no sound and no "achoo" the frames say a whole-body
/// convulsion and cannot say what caused it. It is the only gesture where his
/// stance breaks, which is the whole reason to keep it.
const G_SHIVER: [Pose; 5] = [
    [" /\\,/\\", "\\(>.<)/", " |___|", "  / \\"],
    [" /\\,/\\", "\\(>.<)/", " |___|", "  / \\"],
    [" /\\,/\\", "<(>.<)>", " |___|", " /   \\"],
    [" /\\,/\\", "/(-.-)\\", " |___|", "  / \\"],
    [" /\\,/\\", "/(-.-)\\", " |___|", "  / \\"],
];

/// Dozing off, with a z drifting up out of the block's spare column. The long
/// one -- half an idle cycle -- because a nap that lasts a beat is a blink.
const G_DOZE: [Pose; 8] = [
    [" /\\,/\\", "\\(-.-)/", " |___|", "  / \\"],
    [" /\\,/\\", "\\(-.-)/", " |___|", "  / \\"],
    [" /\\,/\\", "\\(-.-)/z", " |___|", "  / \\"],
    [" /\\,/\\ z", "\\(-.-)/", " |___|", "  / \\"],
    [" /\\,/\\", "\\(-.-)/", " |___|", "  / \\"],
    [" /\\,/\\", "\\(-.-)/z", " |___|", "  / \\"],
    [" /\\,/\\ z", "\\(-.-)/", " |___|", "  / \\"],
    [" /\\,/\\", "\\(-.-)/", " |___|", "  / \\"],
];

/// Leaning out of his column to peer down the screen at the listing.
const G_PEEK: [Pose; 3] = [
    ["  /\\,/\\", " |(o.o)/", "  |___|", "   / \\"],
    ["  /\\,/\\", " |(o.o)/", "  |___|", "   / \\"],
    ["  /\\,/\\", " |(o.o)\\", "  |___|", "   / \\"],
];

/// Every idle gesture, plus one empty entry: a cycle that draws it is a cycle
/// he simply stands there, which is what keeps the idling from reading as a
/// loop of tics.
const IDLE: [&[Pose]; 12] = [
    &[],
    &G_BLINK,
    &G_BLINK2,
    &G_CRANE,
    &G_SCRATCH,
    &G_STRETCH,
    &G_SHRUG,
    &G_WAVE,
    &G_TAP,
    &G_SHIVER,
    &G_DOZE,
    &G_PEEK,
];

/// What he does while the machine boots: he is watching the report land line by
/// line, and it is the gestures of LOOKING that say so -- craning at it,
/// leaning out to read further down, waving at whoever is watching him do it.
///
/// The ones left out are left out for the same reason. Dozing and the tapped
/// foot say he has been abandoned, which is the opposite of a machine that is
/// mid-way through waking up; the single blink and the shrug are over before
/// anybody looks up. A separate table because a separate CYCLE is the point --
/// see `BOOT_CYCLE`.
const BOOT: [&[Pose]; 5] = [&G_BLINK2, &G_CRANE, &G_SCRATCH, &G_PEEK, &G_WAVE];

/// What he does while a parade crosses the header: the gestures of LOOKING,
/// the same ones the boot report leans on and for the same reason. He is
/// watching them go past, and a mascot who blinks and dozes through a parade is
/// two animations sharing a header rather than one scene.
///
/// The empty entry stays, because most of watching something is standing there
/// watching it -- a goblin gesturing without pause for a whole crossing is a
/// goblin having a fit.
///
/// On a header too narrow to draw a parade in he watches one anyway, since the
/// pose is a function of the beat and not of the width. Every gesture here is
/// in the idle table too, so what that looks like is a goblin idling.
const WATCH: [&[Pose]; 4] = [&[], &G_CRANE, &G_PEEK, &G_WAVE];

// --- dance moves ---------------------------------------------------------
// One pose per beat, looped, so a two-pose move is a bop at half the beat rate
// -- the drummer's tempo, and the tempo the pair of them therefore share.

/// The bounce: he drops a whole row and folds his legs under him, which is the
/// only way to read as leaving the ground in a block four rows tall.
const D_BOUNCE: [Pose; 2] = [REST, ["", " /\\,/\\", "\\(o.o)/", " |/_\\|"]];

/// Weight left, weight right, the body swinging a column with it.
const D_SWAY: [Pose; 4] = [
    [" /\\,/\\", "\\(o.o)|", " |___|", "  |_\\"],
    [" /\\,/\\", "\\(o.o)|", " |___|", "  |_\\"],
    ["  /\\,/\\", " |(o.o)/", "  |___|", "   /_|"],
    ["  /\\,/\\", " |(o.o)/", "  |___|", "   /_|"],
];

/// Both arms windmilling, feet apart on the wide beats.
const D_WINDMILL: [Pose; 4] = [
    [" /\\,/\\", "\\(o.o)/", " |___|", "  / \\"],
    [" /\\,/\\", "-(o.o)-", " |___|", " /   \\"],
    [" /\\,/\\", "/(o.o)\\", " |___|", "  / \\"],
    [" /\\,/\\", "-(o.o)-", " |___|", " /   \\"],
];

/// Headbanging: arms thrown down, eyes shut, knees taking it.
const D_HEADBANG: [Pose; 2] = [
    REST,
    [" /\\,/\\", "/(-.-)\\", " |___|", "  |_|"],
];

/// Fists pumping, one and then the other, grinning about it.
const D_PUMP: [Pose; 2] = [
    [" /\\,/\\", "\\(^.^)|", " |___|", "  |_\\"],
    [" /\\,/\\", "|(^.^)/", " |___|", "  /_|"],
];

/// A can-can: arms out, kicking one way and then the other.
const D_KICK: [Pose; 4] = [
    [" /\\,/\\", "-(o.o)-", " |___|", "  / \\"],
    [" /\\,/\\", "-(o.o)-", " |___|", "  |_/"],
    [" /\\,/\\", "-(o.o)-", " |___|", "  / \\"],
    [" /\\,/\\", "-(o.o)-", " |___|", "  \\_|"],
];

/// A full turn. The back of his head has no face on it, which is the whole
/// trick -- three frames of arms would just be another windmill.
const D_SPIN: [Pose; 4] = [
    REST,
    [" /\\,/\\", "|(o.o)\\", " |___|", "  / \\"],
    [" /\\,/\\", "|(___)|", " |___|", "  / \\"],
    [" /\\,/\\", "/(o.o)|", " |___|", "  / \\"],
];

/// Every move he knows. A new one is picked each phrase, so the dance keeps
/// changing for as long as the track does.
const DANCES: [&[Pose]; 7] = [
    &D_BOUNCE,
    &D_SWAY,
    &D_WINDMILL,
    &D_HEADBANG,
    &D_PUMP,
    &D_KICK,
    &D_SPIN,
];

/// Beats in one idle cycle: at most one gesture happens per cycle, dropped at a
/// varying offset inside it. ~3.8 s, which is often enough to notice him and
/// rare enough to ignore him while reading the listing.
const IDLE_CYCLE: usize = 16;

/// Beats in one BOOT cycle. Barely longer than the gestures in it, and with no
/// empty entry in `BOOT`, because the startup report is over in about five
/// seconds: the picker's long cycle is right for a screen somebody sits at, and
/// would let a whole boot pass with the goblin stood perfectly still, which is
/// the one thing this must not do.
const BOOT_CYCLE: usize = 5;

/// Beats a dance move is held before the next is drawn. Two seconds of a move
/// is long enough to read as a step and short enough that nobody watches the
/// same one twice.
const PHRASE: usize = 8;

/// Phrases in a set: the block he either dances right through or takes off.
/// Whole sets rather than phrases, because a goblin who decides every two
/// seconds whether to dance is not resting, he is flickering.
const SET: usize = 4;

/// One set in this many is a breather. Nobody dances for the whole record --
/// and a mascot who NEVER stops is back to being furniture that happens to
/// move, which is the thing the idle table exists to avoid.
const SETS_PER_BREATHER: u64 = 4;

/// Is he sitting this set out?
///
/// Two sets are never taken off in a row -- a breather is a breather, and a
/// scramble left to itself will hand out four in a row eventually, which is
/// half a minute of a mascot ignoring the music that is plainly playing. The
/// first set of a track is never one either: a track grid restarts the count,
/// so that is the top of every piece, and the music starting is the one moment
/// he is certainly meant to be moving.
fn on_a_breather(tick: usize) -> bool {
    let set = (tick / (PHRASE * SET)) as u64;
    let picked = |s: u64| s > 0 && scramble(s).is_multiple_of(SETS_PER_BREATHER);
    picked(set) && !picked(set - 1)
}

/// How long one frame is held when there is no music to hold it to. Two draws
/// per frame at the key loops' poll interval, so no frame is ever skipped by
/// the redraw clock. It is the same figure the playing track's grid subdivides
/// toward, which is why the goblins keep their tempo either way.
pub const BEAT_MS: u128 = crate::sound::ANIM_TICK_MS as u128;

/// One clock for everything that animates. Process-wide rather than per-screen
/// so the band keeps its tempo across the picker closing and reopening between
/// batches, and across the hand-off from the picker to the processing header.
fn anim_clock() -> Instant {
    static T0: std::sync::OnceLock<Instant> = std::sync::OnceLock::new();
    *T0.get_or_init(Instant::now)
}

/// The beat everything animates on -- the mascot, the band and the sound coming
/// off it, so the three of them are never off from each other.
///
/// The playing track's own beat when there is one (`sound::music_beat`), and
/// the house beat off the process clock when there is not. A track's grid
/// starts at zero, so every track begins the count again: a dance starts when
/// the music does rather than wherever the last one left off.
pub fn beat_tick() -> usize {
    crate::sound::music_beat()
        .unwrap_or_else(|| (anim_clock().elapsed().as_millis() / BEAT_MS) as usize)
}

/// A cheap deterministic scramble (splitmix64's finaliser), so the gestures
/// arrive in an order nobody can hum along to. Deterministic on purpose: the
/// pose stays a pure function of the clock, which is what lets several screens
/// draw him without sharing any state.
fn scramble(n: u64) -> u64 {
    let mut z = n
        .wrapping_add(0x9E37_79B9_7F4A_7C15)
        .wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 30)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// One gesture per cycle, at an offset drawn from the same scramble, and REST
/// for the rest of the cycle. The offset is what stops him twitching on a
/// metronome -- a gesture that always began on the cycle's first beat would be
/// a loop however many gestures were in the table.
fn idle_pose(tick: usize, cycle: usize, table: &[&[Pose]]) -> Pose {
    let c = (tick / cycle) as u64;
    let g = table[scramble(c) as usize % table.len()];
    // saturating: `idle_gestures_fit_a_cycle` is what actually holds this, but
    // a table edit should shorten a gesture, never panic a screen
    let room = cycle.saturating_sub(g.len());
    let start = scramble(c ^ 0xA5A5_A5A5) as usize % (room + 1);
    let beat = tick % cycle;
    match beat.checked_sub(start) {
        Some(i) if i < g.len() => g[i],
        _ => REST,
    }
}

/// What the mascot is doing on beat `tick`.
///
/// `music` is whether anything is playing; it is not the same question as
/// whether he is dancing, because he takes a set off now and then and loiters
/// through it (`on_a_breather`) exactly as he does in silence.
///
/// Dancing: the move for this phrase, indexed by the GLOBAL beat rather than by
/// a beat counted from the move's own start, so a two-pose bop always lands on
/// the same beats as the drummer's stick.
/// `watching` is whether a parade is on screen RIGHT NOW -- not merely whether
/// the clock has scheduled one. The two differ: a crossing is scheduled a fixed
/// number of beats but takes as long as the header is wide, and a goblin still
/// craning at a corridor the file left twenty seconds ago is a goblin staring
/// at nothing. The caller has the width, so the caller answers it.
pub fn mascot_pose(tick: usize, music: bool, watching: bool) -> Pose {
    // A parade outranks the music. He is a goblin and that is an army of
    // goblins: whatever he was doing, he is watching it go past.
    if watching {
        return idle_pose(tick, IDLE_CYCLE, &WATCH);
    }
    if music && !on_a_breather(tick) {
        let mv = DANCES[scramble((tick / PHRASE) as u64) as usize % DANCES.len()];
        return mv[tick % mv.len()];
    }
    idle_pose(tick, IDLE_CYCLE, &IDLE)
}

/// What he is doing on beat `tick` of the startup report: always something.
pub fn boot_pose(tick: usize) -> Pose {
    idle_pose(tick, BOOT_CYCLE, &BOOT)
}

/// The header for right now, on the shared clock and the live audio state --
/// the mascot's pose and the parade crossing behind him, which are one question
/// because whether he is watching depends on whether there is anything to
/// watch. Every surface that is not driving its own timeline (the intro) asks
/// for both together.
pub fn scene_now(width: u16) -> (Pose, Option<Parade>) {
    scene_at(width, beat_tick(), crate::sound::music_on())
}

/// The scene on a GIVEN beat, which is what the tests drive: a crossing is a
/// once-in-minutes event, and one that could only be rendered by waiting for it
/// is one nothing holds a check against.
pub fn scene_at(width: u16, tick: usize, playing: bool) -> (Pose, Option<Parade>) {
    let par = parade(width, playing, tick);
    (mascot_pose(tick, playing, par.is_some()), par)
}

// ===========================================================================
// The corner band: two small goblins in the top right of the header, a drummer
// and a DJ. They play while the music is playing and slump when it is not, so
// the state of the audio layer is visible as a MOOD and not only as the word
// next to `audio:` in the chrome strip.
// ===========================================================================

/// One band member's block, and the height they share. Every row of every pose
/// is exactly this wide, so a goblin animates in place instead of jittering
/// around its corner.
pub const BLOCK_W: u16 = 9;
pub const GOBLIN_H: u16 = 3;
/// The pair plus the column between them.
pub const BAND_W: u16 = BLOCK_W * 2 + 1;

/// The DJ: one hand up, one on the deck, the record's spoke turning under
/// them. Four frames -- the spoke is what makes the deck read as SPINNING
/// rather than as a drawing of a turntable.
const DJ: [[&str; 3]; 4] = [
    ["  /\\,/\\  ", " \\(o.o)/ ", " [_(|)_] "],
    ["  /\\,/\\  ", " |(o.o)| ", " [_(/)_] "],
    ["  /\\,/\\  ", " /(o.o)\\ ", " [_(-)_] "],
    ["  /\\,/\\  ", " |(o.o)| ", " [_(\\)_] "],
];

/// The drummer: sticks alternating, the impact flashing to the side the stick
/// just came down on. Two frames, played at HALF the DJ's rate -- a hit every
/// other frame is a beat around 125 BPM, and a drummer at the spoke's rate
/// reads as a blur rather than as hitting anything.
const DRUM: [[&str; 3]; 2] = [
    ["  /\\,/\\  ", " \\(o.o)| ", " *[====] "],
    ["  /\\,/\\  ", " |(o.o)/ ", " [====]* "],
];

/// No music: arms down, eyes shut, the deck stopped and the sticks laid across
/// the drum. Ears up even here -- same rule as the mascot's, and these two are
/// the same goblin.
const DJ_SULK: [&str; 3] = ["  /\\,/\\  ", "  (-.-)  ", " [_(.)_] "];
const DRUM_SULK: [&str; 3] = ["  /\\,/\\  ", "  (-.-)  ", "  [====] "];

/// The narrowest header with room for the whole band, and for the DJ alone.
/// Below the first the drummer is dropped, below the second the tagline would
/// run into the deck -- and no band beats a broken one.
pub const BAND_MIN_W: u16 = 60;
pub const SOLO_MIN_W: u16 = 50;

/// The sound leaving the band is one column wide, and this is the header width
/// that affords it. It is the FIRST thing dropped as the header narrows -- the
/// band itself carries the playing/stopped state, so the sound is the part that
/// can go.
pub const WAVE_W: u16 = 1;
pub const WAVE_MIN_W: u16 = BAND_MIN_W + WAVE_W;

/// The band's three rows on beat `tick`: the drummer and the DJ side by side,
/// or the DJ alone in a header too narrow for both.
///
/// The DJ is the one who stays when the header narrows: the deck carries the
/// playing/stopped state in its own right, while a drummer alone is just a
/// goblin holding sticks.
pub fn band_rows(full: bool, playing: bool, tick: usize) -> Vec<String> {
    let dj = if playing { DJ[tick % DJ.len()] } else { DJ_SULK };
    let drum = if playing { DRUM[(tick / 2) % DRUM.len()] } else { DRUM_SULK };
    (0..GOBLIN_H as usize)
        .map(|r| if full { format!("{} {}", drum[r], dj[r]) } else { dj[r].to_string() })
        .collect()
}

/// Is the sound out on beat `tick`? Two beats out, two beats in, so the blink
/// is about half a second either way -- fast enough to read as sound, slow
/// enough not to flicker in the corner of the eye while somebody works.
pub fn wave_out(tick: usize) -> bool {
    (tick / 2).is_multiple_of(2)
}

/// One side's column of sound, out or in. `right` mirrors it.
///
/// The orientation is the ordinary radiating burst,
///
/// ```text
///  \ | /
/// -- O --
///  / | \
/// ```
///
/// so the column LEFT of the band reads `\ - /` downward and the column right
/// of it is that mirrored, `/ - \`. Upside down is the difference between sound
/// leaving the band and sound arriving at it, and the two sides must be each
/// other's mirror or the corner looks lopsided.
///
/// Rendered on the quiet beat too, as blanks, so the rays are cleared by this
/// function rather than by whatever else happens to paint the header.
pub fn wave_rows(out: bool, right: bool) -> Vec<String> {
    let rays = if right { ['/', '-', '\\'] } else { ['\\', '-', '/'] };
    (0..GOBLIN_H as usize)
        .map(|r| if out { rays[r] } else { ' ' }.to_string())
        .collect()
}

// ===========================================================================
// The parade: rarely, a file of small goblins marches across the header, out
// from behind the band and off behind the wordmark, on the same rule the mascot
// stands on. He stops to watch them (`WATCH`).
//
// It is the same goblin again, one row shorter -- the trade the band already
// makes, legs where their kit is. Everything about where the file is, is a pure
// function of the beat and the width, so both stacks paint the same march
// without either of them owning it.
// ===========================================================================

/// One marcher's block. Every row of every pose is exactly this wide, so he
/// animates in place rather than jittering along the rule.
pub const MARCH_W: u16 = 7;
/// Rows he occupies -- the mascot's four with the torso dropped, which puts his
/// head a row below the mascot's and his feet on the same floor.
pub const MARCH_H: usize = 3;

/// One marcher's pose.
type March = [&'static str; MARCH_H];

/// The walk: arms and legs swapping over, one pose per beat. A step every
/// 240 ms over two columns of ground is a walking pace; a pose per column would
/// be a shuffle nobody could see the legs of.
const MARCH_POSES: [March; 2] = [
    [" /\\,/\\ ", "\\(o.o)|", "  |_\\  "],
    [" /\\,/\\ ", "|(o.o)/", "  /_|  "],
];

/// The straggler catching his own feet: eyes screwed up, arms out, legs gone
/// from under him. He is the only one who ever strikes it.
const STUMBLE: March = [" /\\,/\\ ", "\\(>.<)/", "  /  \\ "];

/// Marchers in the file, and the columns from one block to the next.
const FILE: usize = 8;
const PITCH: u16 = MARCH_W + 1;

/// Columns of ground covered per beat.
const SPEED: u16 = 2;

/// Beats in one parade cycle: at most one crossing happens per cycle, at a
/// varying offset inside it. ~61 s, so a parade is something you catch rather
/// than something you wait for.
const PARADE_CYCLE: usize = 256;

/// One cycle in this many carries a parade -- about one crossing every five
/// minutes. Rare is the whole point: this is the only thing the header does
/// that a user will look up at, and it stops being that the moment it is
/// expected.
const CYCLES_PER_PARADE: u64 = 5;

/// Beats a crossing is allowed, which is the room the start offset has to leave
/// at the end of its cycle. Long enough for the file to clear a corridor about
/// 250 columns wide; a narrower header finishes early and simply draws nothing
/// for the rest of it.
const PARADE_ROOM: usize = 160;

/// The corridor a parade needs before it is worth running one. Four blocks --
/// below that the file is a queue rather than an army, and an 80-column header
/// (22 columns of corridor) gets no parade at all, exactly as it gets no
/// drummer.
const MIN_CORRIDOR: u16 = PITCH * 4;

/// Where the header's text runs out: the longer of the two lines beside the
/// art, both of which start four columns past the block. In COLUMNS -- the
/// tagline is translated, and a byte count would put the rule's end inside a
/// multi-byte glyph.
const WORDMARK: &str = concat!("GoblinScript v", env!("CARGO_PKG_VERSION"));

fn tagline() -> &'static str {
    crate::t!("app.tagline")
}

fn text_end() -> u16 {
    let w = console::measure_text_width(WORDMARK).max(console::measure_text_width(tagline()));
    (TEXT_COL + 4 + w) as u16
}

/// Which beat of a crossing `tick` is, or `None` when no parade is on.
///
/// Never in the first cycle: a track grid restarts the beat count and the intro
/// animates on a clock of its own, so cycle zero is the one stretch where a
/// parade would be a certainty rather than a surprise.
fn parade_beat(tick: usize) -> Option<usize> {
    let c = (tick / PARADE_CYCLE) as u64;
    if c == 0 || !scramble(c ^ 0x9E37_79B9).is_multiple_of(CYCLES_PER_PARADE) {
        return None;
    }
    let start = scramble(c ^ 0x5EED_1E55) as usize % (PARADE_CYCLE - PARADE_ROOM + 1);
    (tick % PARADE_CYCLE)
        .checked_sub(start)
        .filter(|e| *e < PARADE_ROOM)
}

/// Does the straggler catch his feet on this beat? Two beats at a time, a
/// handful of times a crossing, on no rhythm anybody can predict -- the same
/// scramble everything else here is spaced by.
fn stumbling(e: usize) -> bool {
    scramble((e / 2) as u64 ^ 0x5711_4D1E).is_multiple_of(8)
}

/// The first column the band's block touches, its sound included, or `None` in
/// a header too narrow to carry one. This is where the parade's corridor ends.
fn band_left(width: u16, playing: bool) -> Option<u16> {
    if width < SOLO_MIN_W {
        return None;
    }
    let full = width >= BAND_MIN_W;
    let waves = playing && full && width >= WAVE_MIN_W;
    let band_w = if full { BAND_W } else { BLOCK_W };
    let band_x = width.saturating_sub(band_w + if waves { 2 } else { 1 });
    Some(band_x.saturating_sub(u16::from(waves)))
}

/// The stretch of header a parade has to march through: past the tagline, short
/// of the band. `None` where there is not enough of it to be worth one.
fn corridor(width: u16, playing: bool) -> Option<(u16, u16)> {
    let right = band_left(width, playing)?.checked_sub(1)?;
    let left = text_end() + 1;
    (right >= left + MIN_CORRIDOR).then_some((left, right))
}

/// One marcher, placed.
struct Marcher {
    /// Left column of his block.
    x: u16,
    rows: March,
    /// Drawn in the secondary ink because he is entering or leaving. A goblin
    /// is only ever drawn WHOLE (a cut-off one reads as a rendering fault, not
    /// as a goblin), and the cost of that is a body appearing all at once at
    /// the corridor's edge -- the step down in brightness is what turns the pop
    /// into an arrival.
    fading: bool,
}

/// The file on beat `tick`, in column order. Empty whenever no parade is
/// crossing, or the header has no corridor to hold one.
///
/// They march right to left: out from behind the band, which is an opaque
/// object standing on the same rule and painted after them, and away behind the
/// wordmark.
fn marchers(width: u16, playing: bool, tick: usize) -> Vec<Marcher> {
    let Some(e) = parade_beat(tick) else {
        return Vec::new();
    };
    let Some((left, right)) = corridor(width, playing) else {
        return Vec::new();
    };
    let (l, r, w) = (left as i32, right as i32, MARCH_W as i32);
    (0..FILE)
        .filter_map(|i| {
            let x = r - (e as i32) * SPEED as i32 + (i as i32) * PITCH as i32;
            // whole goblins only, and the last column of the corridor is air
            if x < l || x + w > r {
                return None;
            }
            // The straggler is a beat behind the file he is meant to be in, and
            // now and then he goes over his own feet. He is the reason the
            // parade reads as goblins rather than as a repeated sprite.
            let back = i == FILE - 1;
            let rows = if back && stumbling(e) {
                STUMBLE
            } else {
                MARCH_POSES[(e + usize::from(back)) % MARCH_POSES.len()]
            };
            let pitch = PITCH as i32;
            Some(Marcher {
                x: x as u16,
                rows,
                fading: x < l + pitch || x + w + pitch > r,
            })
        })
        .collect()
}

/// A parade laid out for painting: `left` is the column its rows start at, and
/// each row is a run of (text, xterm-256 ink) pairs covering the corridor.
///
/// Runs rather than cells because both stacks paint this -- the console header
/// styles them with `console`, the picker with ratatui spans -- and a run is
/// the largest piece both of them can colour in one go. The marchers are drawn
/// TRANSPARENTLY: a space in a pose paints nothing, which is what leaves the
/// rule running between a marcher's legs instead of a hole punched around them.
pub struct Parade {
    pub left: u16,
    pub right: u16,
    /// Header rows 1, 2 and 3 -- his head beside the wordmark, his body beside
    /// the tagline, his feet in the rule.
    pub rows: [Vec<(String, u8)>; MARCH_H],
}

/// The parade crossing a `width`-column header on beat `tick`, or `None`.
pub fn parade(width: u16, playing: bool, tick: usize) -> Option<Parade> {
    let ms = marchers(width, playing, tick);
    if ms.is_empty() {
        return None;
    }
    let (left, right) = corridor(width, playing)?;
    let t = theme();
    let n = (right - left) as usize;
    let rows = std::array::from_fn(|r| {
        // the bottom row's ground is the rule itself; the two above it are air
        let bg = if r == MARCH_H - 1 { '\u{2550}' } else { ' ' };
        let mut cells = vec![(bg, t.muted); n];
        for m in &ms {
            let ink = if m.fading { t.muted } else { t.logo };
            for (i, ch) in m.rows[r].chars().enumerate() {
                if ch == ' ' {
                    continue;
                }
                let c = (m.x - left) as usize + i;
                if c < n {
                    cells[c] = (ch, ink);
                }
            }
        }
        runs(&cells)
    });
    Some(Parade { left, right, rows })
}

/// Neighbouring cells that share an ink, gathered into one run.
fn runs(cells: &[(char, u8)]) -> Vec<(String, u8)> {
    let mut out: Vec<(String, u8)> = Vec::new();
    for &(ch, ink) in cells {
        match out.last_mut() {
            Some((s, i)) if *i == ink => s.push(ch),
            _ => out.push((ch.to_string(), ink)),
        }
    }
    out
}

// ===========================================================================
// The header as CONSOLE text: the same four rows the picker draws through
// ratatui, for the surfaces that write escape sequences straight to stdout (the
// header pinned over the processing log). Both stacks read the same pose and
// band functions above, so what the two screens show cannot drift -- only the
// styling primitives differ.
// ===========================================================================

/// Where the wordmark and the tagline start, measured from the left edge: the
/// two-column indent plus the mascot's block. The startup report puts its text
/// in this same column.
const TEXT_COL: usize = 2 + crate::theme::MASCOT_W;

/// The four header rows, styled and fitted to `width`: the mascot on the left
/// with the wordmark beside him, the band flush right, and the rule closing the
/// block -- the goblin stands ON that rule, which is the floor he idles and
/// dances on.
///
/// Every row comes back at most `width - 1` columns wide, so nothing wraps into
/// a second physical row on any terminal.
pub fn header_rows(width: u16) -> Vec<String> {
    header_rows_at(width, beat_tick(), crate::sound::music_on())
}

/// The header on a GIVEN beat and audio state, which is what the tests drive:
/// a parade is a once-in-minutes event, and one that could only be rendered by
/// waiting for it is one nothing holds a check against.
pub fn header_rows_at(width: u16, tick: usize, dancing: bool) -> Vec<String> {
    let t = theme();
    let par = parade(width, dancing, tick);
    let pose = mascot_pose(tick, dancing, par.is_some());
    // Bold while he dances, exactly as the band brightens when it plays: the
    // mood reads across the header before any single frame does. Never dimmed,
    // though -- he is the wordmark's own goblin, and a greyed-out logo would
    // read as a disabled app rather than as quiet.
    let art = |i: usize| {
        let s = style(art_line(pose[i], 2)).fg(con(t.logo));
        if dancing { s.bold() } else { s }
    };
    let rule = |n: usize| style("\u{2550}".repeat(n)).fg(con(t.muted)).to_string();
    // out to the last cell the row is allowed to touch, which is one short of
    // the screen's -- a rule that fills the final column wraps on a terminal
    // that does not defer it
    let rule_w = (width as usize).saturating_sub(TEXT_COL + 1);
    let mut rows = vec![
        art(0).to_string(),
        format!(
            "{}    {}{}",
            art(1),
            style("GoblinScript").fg(con(t.accent)).bold(),
            style(concat!(" v", env!("CARGO_PKG_VERSION"))).fg(con(t.muted)),
        ),
        format!("{}    {}", art(2), style(tagline()).fg(con(t.accent))),
        art(3).to_string(),
    ];
    // The rule the goblins stand on, laid in three pieces where a parade is
    // crossing it so the marchers stand IN it rather than on a hole cut out of
    // it, and in one piece the rest of the time.
    match &par {
        Some(p) => {
            let head = (p.left as usize).saturating_sub(TEXT_COL);
            let tail = rule_w.saturating_sub(head + (p.right - p.left) as usize);
            rows[3].push_str(&rule(head));
            rows[3].push_str(&paint(&p.rows[MARCH_H - 1]));
            rows[3].push_str(&rule(tail));
        }
        None => rows[3].push_str(&rule(rule_w)),
    }
    if let Some(p) = &par {
        add_parade(&mut rows, p);
    }
    add_band(&mut rows, width, dancing, tick);
    rows.into_iter()
        .map(|r| format!("{}\u{1b}[0m", truncate_str(&r, width.saturating_sub(1) as usize, "")))
        .collect()
}

/// One row of a parade as console text.
fn paint(row: &[(String, u8)]) -> String {
    row.iter()
        .map(|(s, ink)| style(s.clone()).fg(con(*ink)).to_string())
        .collect()
}

/// Lay the marchers' top two rows into the header, past the text. The rule row
/// is built with the rule itself, in `header_rows_at`, because the ground the
/// marchers stand in has to be laid around them rather than over them.
fn add_parade(rows: &mut [String], p: &Parade) {
    for (i, row) in rows.iter_mut().enumerate().take(MARCH_H).skip(1) {
        let pad = (p.left as usize).saturating_sub(measure_text_width(row));
        row.push_str(&" ".repeat(pad));
        row.push_str(&paint(&p.rows[i - 1]));
    }
}

/// Lay the band into the top three rows, flush right, with its sound either
/// side. Silently does nothing in a header too narrow to hold it -- the width
/// guards are the whole reason this is a separate step.
fn add_band(rows: &mut [String], width: u16, playing: bool, tick: usize) {
    let Some(at) = band_left(width, playing) else {
        return;
    };
    let t = theme();
    let full = width >= BAND_MIN_W;
    // Bright and brand-coloured while they play, dropped to the secondary ink
    // when they do not -- the mood reads before the drawing does.
    let ink = if playing { con(t.logo) } else { con(t.muted) };
    // The sound rides in the column either side of the band, so the pair sits
    // one column further in when it is drawn -- and the right-hand column stops
    // short of the last cell, which is the one a terminal may wrap on.
    let waves = playing && full && width >= WAVE_MIN_W;
    let band = band_rows(full, playing, tick);
    let out = wave_out(tick);
    for (r, row) in rows.iter_mut().enumerate().take(GOBLIN_H as usize) {
        let mut cell = String::new();
        if waves {
            cell.push_str(&style(wave_rows(out, false)[r].clone()).fg(con(t.accent)).to_string());
        }
        let b = style(band[r].clone()).fg(ink);
        cell.push_str(&if playing { b.bold() } else { b }.to_string());
        if waves {
            cell.push_str(&style(wave_rows(out, true)[r].clone()).fg(con(t.accent)).to_string());
        }
        let pad = (at as usize).saturating_sub(measure_text_width(row));
        row.push_str(&" ".repeat(pad));
        row.push_str(&cell);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::theme::MASCOT_W;
    use std::collections::HashSet;

    fn poses() -> Vec<Pose> {
        IDLE.iter()
            .chain(DANCES.iter())
            .chain(BOOT.iter())
            .flat_map(|s| s.iter().copied())
            .collect()
    }

    /// Every pose a marcher in the parade can strike.
    fn march_poses() -> Vec<March> {
        MARCH_POSES.iter().copied().chain([STUMBLE]).collect()
    }

    /// The first beat of the first crossing the clock ever schedules.
    fn first_parade() -> usize {
        (0..PARADE_CYCLE * 64)
            .find(|&t| parade_beat(t) == Some(0))
            .expect("the clock never schedules a parade at all")
    }

    /// Every pose either band member can strike.
    fn band_poses() -> Vec<&'static [&'static str; 3]> {
        DJ.iter()
            .chain(DRUM.iter())
            .chain([&DJ_SULK, &DRUM_SULK])
            .collect()
    }

    // The header prints the wordmark and the tagline beside the art, in a fixed
    // column that the padding provides. A pose row wider than the block pushes
    // that column right for one frame, which reads as the whole header
    // twitching. It also has to stay ASCII: this is drawn on whatever console
    // the user double-clicked into, and one wide character shifts the row.
    #[test]
    fn every_pose_fits_the_block() {
        for pose in poses() {
            for row in pose {
                assert!(row.is_ascii(), "{row:?} is ASCII");
                assert!(
                    row.chars().count() <= MASCOT_W,
                    "{row:?} is {} columns, the block is {MASCOT_W}",
                    row.chars().count()
                );
            }
        }
    }

    // The band is drawn into a fixed rect and swapped frame by frame, so every
    // row of every pose has to be exactly one block wide: a short row leaves
    // the previous frame's tail on screen, a long one pushes its neighbour out
    // of the rect. Eyeballed art drifts the first time somebody edits it.
    #[test]
    fn every_band_row_fills_its_block() {
        for pose in band_poses() {
            for row in pose.iter() {
                assert_eq!(
                    row.chars().count(),
                    BLOCK_W as usize,
                    "{row:?} is {} columns, want {BLOCK_W}",
                    row.chars().count()
                );
                assert!(row.is_ascii(), "{row:?} is ASCII");
            }
        }
    }

    // The two of them play in one enclosure, so they must stand on one
    // baseline: heads on the top row, hands in the middle, gear on the rule.
    // (Checked as "the gear row is the one with the deck and the drum in it" --
    // a pose assembled a row out of order would put a head on the rule.)
    #[test]
    fn the_band_shares_a_baseline() {
        for pose in DJ.iter().chain([&DJ_SULK]) {
            assert!(pose[2].contains("[_("), "the DJ's deck is on the bottom row: {pose:?}");
        }
        for pose in DRUM.iter().chain([&DRUM_SULK]) {
            assert!(pose[2].contains("[===="), "the drum is on the bottom row: {pose:?}");
        }
    }

    // Up and alert in every pose he can strike, and in every pose the band can
    // strike either -- asleep, sneezing, sulking or mid-headbang. An ear is two
    // strokes at this size and cannot be folded: drawn any other way it reads
    // as a notch cut out of his head rather than as an ear that moved, so the
    // mood goes in the eyes, arms and legs instead.
    //
    // Checked through the comma, which is the head top and the only comma in
    // any pose: exactly one row carries it, and that row OPENS with the ears.
    // Opens rather than equals, because that row's spare columns are real
    // drawing space -- the z he sleeps under rises through them.
    #[test]
    fn the_ears_are_always_up() {
        let mascot = poses().into_iter().map(|p| p.to_vec());
        let band = band_poses().into_iter().map(|p| p.to_vec());
        let march = march_poses().into_iter().map(|p| p.to_vec());
        for pose in mascot.chain(band).chain(march) {
            let ears: Vec<&str> = pose.iter().copied().filter(|r| r.contains(',')).collect();
            assert_eq!(ears.len(), 1, "one ear row per pose, got {ears:?} in {pose:?}");
            assert!(
                ears[0].trim_start().starts_with("/\\,/\\"),
                "the ears are up: {pose:?}"
            );
        }
    }

    // Nothing may pull a face that is not on the list, because the failure mode
    // is silent: the art parses, renders, passes every width check, and reads
    // as the wrong emotion. Checked through the parentheses, which no other row
    // of any pose contains.
    #[test]
    fn the_faces_are_ones_that_read() {
        let mascot = poses().into_iter().map(|p| p.to_vec());
        let march = march_poses().into_iter().map(|p| p.to_vec());
        for pose in mascot.chain(march) {
            let faces: Vec<&str> = pose
                .iter()
                .filter_map(|r| r.split_once('(').and_then(|(_, t)| t.split_once(')')))
                .map(|(f, _)| f)
                .collect();
            assert_eq!(faces.len(), 1, "one face per pose, got {faces:?} in {pose:?}");
            assert!(FACES.contains(&faces[0]), "{:?} is not a face that reads", faces[0]);
        }
    }

    // A gesture longer than its cycle would have nowhere to start, and the
    // offset arithmetic would have to clamp it into a stutter.
    #[test]
    fn idle_gestures_fit_a_cycle() {
        for g in IDLE {
            assert!(g.len() <= IDLE_CYCLE, "a {}-beat gesture in a {IDLE_CYCLE}-beat cycle", g.len());
        }
        for g in BOOT {
            assert!(g.len() <= BOOT_CYCLE, "a {}-beat gesture in a {BOOT_CYCLE}-beat cycle", g.len());
        }
    }

    // The scramble picks from both tables, so a degenerate one (or a table that
    // outgrew the mixing) would quietly leave gestures nobody ever sees. Every
    // one of them must come up over a long enough run.
    #[test]
    fn every_gesture_and_move_gets_drawn() {
        let idle: HashSet<Pose> = (0..IDLE_CYCLE * 4096).map(|t| mascot_pose(t, false, false)).collect();
        for g in IDLE {
            assert!(g.iter().all(|p| idle.contains(p)), "an idle gesture never comes up: {g:?}");
        }
        let dance: HashSet<Pose> = (0..PHRASE * 4096).map(|t| mascot_pose(t, true, false)).collect();
        for m in DANCES {
            assert!(m.iter().all(|p| dance.contains(p)), "a dance move never comes up: {m:?}");
        }
        let boot: HashSet<Pose> = (0..BOOT_CYCLE * 4096).map(boot_pose).collect();
        for g in BOOT {
            assert!(g.iter().all(|p| boot.contains(p)), "a boot gesture never comes up: {g:?}");
        }
    }

    // Idling is punctuated, not constant: he spends most of it stood there, or
    // the corner of the eye never gets a rest and the gestures stop registering
    // as gestures.
    #[test]
    fn idle_mostly_rests() {
        let span = IDLE_CYCLE * 512;
        let resting = (0..span).filter(|&t| mascot_pose(t, false, false) == REST).count();
        assert!(resting * 2 > span, "he idles at rest {resting}/{span} beats, want over half");
    }

    // The boot is the opposite case: the report is over in about five seconds,
    // so a cycle he might spend entirely at rest would read as a still drawing.
    // Every cycle has to carry a gesture, and he has to be moving for a good
    // share of the report rather than technically-not-frozen.
    #[test]
    fn the_boot_goblin_is_never_still_for_long() {
        // ~5 s of report at the house beat, from any point the clock starts on
        let report = 5_000 / BEAT_MS as usize;
        for t0 in 0..4096 {
            let moving = (t0..t0 + report).filter(|&t| boot_pose(t) != REST).count();
            assert!(
                moving * 5 >= report * 2,
                "only {moving}/{report} beats of the report move, from tick {t0}"
            );
        }
        // and no cycle is ever a blank one
        for c in 0..4096 {
            let t0 = c * BOOT_CYCLE;
            assert!(
                (t0..t0 + BOOT_CYCLE).any(|t| boot_pose(t) != REST),
                "boot cycle {c} passes with nothing happening"
            );
        }
    }

    // A danced phrase must never hold one pose the whole way through, which is
    // what a stalled clock or a one-pose move would look like. Breather phrases
    // are exempt -- standing still is what they are for.
    #[test]
    fn a_danced_phrase_never_stalls() {
        for phrase in 0..512 {
            let t0 = phrase * PHRASE;
            if on_a_breather(t0) {
                continue;
            }
            let moved = (t0..t0 + PHRASE).any(|t| mascot_pose(t, true, false) != mascot_pose(t0, true, false));
            assert!(moved, "the dance holds one pose through phrase {phrase}");
        }
    }

    // He dances the music, but not every bar of it: the breathers have to
    // actually happen, have to end, and have to stay the minority -- music
    // playing while the mascot loiters through most of it reads as a mascot
    // that has not noticed the music.
    #[test]
    fn he_breaks_from_dancing_without_giving_it_up() {
        let span = PHRASE * SET * 4096;
        let dancing = (0..span).filter(|&t| !on_a_breather(t)).count();
        let share = dancing as f64 / span as f64;
        assert!(
            (0.6..0.95).contains(&share),
            "he dances {:.0}% of the music, want a clear majority with real breaks",
            share * 100.0
        );
        // and the first set of any track is always danced
        assert!((0..PHRASE * SET).all(|t| !on_a_breather(t)), "he sits out the top of the track");
        // no two sets off in a row, so the longest he ignores the music is one
        // set however the scramble falls
        for set in 1..4096 {
            let off = |s: usize| on_a_breather(s * PHRASE * SET);
            assert!(!(off(set) && off(set - 1)), "sets {} and {set} are both off", set - 1);
        }
        // a breather is a whole set, never a fragment of one
        for set in 0..4096 {
            let t0 = set * PHRASE * SET;
            let first = on_a_breather(t0);
            assert!(
                (t0..t0 + PHRASE * SET).all(|t| on_a_breather(t) == first),
                "set {set} changes its mind part way through"
            );
        }
    }

    // The sound has to actually BLINK -- a column that settles into one state
    // is a character sitting next to the band, not something being emitted.
    #[test]
    fn the_sound_blinks_out_and_in() {
        let seq: Vec<bool> = (0..16).map(wave_out).collect();
        assert!(seq.contains(&true) && seq.contains(&false), "never blinks: {seq:?}");
        // even duty, and slow enough to read: two beats out, two beats in
        assert_eq!(seq.iter().filter(|b| **b).count(), seq.len() / 2, "{seq:?}");
        assert_eq!((0..4).map(wave_out).collect::<Vec<_>>(), [true, true, false, false]);
    }

    // The burst radiates OUTWARD from the band, and the two sides are each
    // other's mirror. Upside down would read as sound arriving rather than
    // leaving, and a non-mirrored pair makes the whole corner look lopsided.
    #[test]
    fn the_burst_radiates_outward_and_mirrors() {
        assert_eq!(wave_rows(true, false), vec!["\\", "-", "/"], "left of the band");
        assert_eq!(wave_rows(true, true), vec!["/", "-", "\\"], "right of the band");

        // the mirror is the property, not the two literals above: flipping one
        // side's slashes has to produce the other, row for row
        let mirror = |s: &str| match s {
            "/" => "\\".to_string(),
            "\\" => "/".to_string(),
            other => other.to_string(),
        };
        let (l, r) = (wave_rows(true, false), wave_rows(true, true));
        assert_eq!(l.iter().map(|s| mirror(s)).collect::<Vec<_>>(), r, "sides are mirrored");

        // and on the quiet beat both sides clear completely -- rendered as
        // blanks rather than skipped, so nothing is left behind
        for right in [false, true] {
            assert!(wave_rows(false, right).iter().all(|s| s == " "), "in: blank");
            for r in wave_rows(true, right) {
                assert_eq!(r.chars().count(), WAVE_W as usize, "{r:?} is not one column");
            }
        }
    }

    // The console header is written straight into a terminal with no layout
    // engine to catch an overrun: a row wider than the screen wraps, and a
    // wrapped header row pushes the whole processing log down one line per
    // frame. Every width, every palette, both audio states.
    #[test]
    fn header_rows_never_outgrow_the_terminal() {
        // Parade beats included: the file is laid into the same rows, and a
        // crossing that pushed one of them a column wide would be a header that
        // wraps once every few minutes -- the hardest kind of fault to catch by
        // looking at it.
        let t0 = first_parade();
        let ticks = [0usize, 1, 7]
            .into_iter()
            .chain((0..PARADE_ROOM).step_by(3).map(|e| t0 + e));
        for p in crate::theme::Palette::ALL {
            crate::theme::set(p);
            for tick in ticks.clone() {
                for width in [1u16, 2, 10, 40, 49, 50, 59, 60, 61, 80, 90, 100, 120, 400] {
                    for dancing in [false, true] {
                        let rows = header_rows_at(width, tick, dancing);
                        assert_eq!(rows.len(), 4, "the header is four rows at width {width}");
                        for r in &rows {
                            assert!(
                                measure_text_width(r) < width.max(1) as usize,
                                "a {}-column row in a {width}-column terminal: {r:?}",
                                measure_text_width(r)
                            );
                        }
                    }
                }
            }
        }
        crate::theme::set(crate::theme::Palette::Phosphor);
    }

    // The band only appears where it fits whole, and the sound only where there
    // is a column spare either side of it -- the checks are on visible columns,
    // since every row here carries colour escapes too.
    #[test]
    fn the_band_arrives_with_the_width() {
        crate::theme::set(crate::theme::Palette::Phosphor);
        let plain = |w: u16| -> Vec<String> {
            header_rows(w)
                .iter()
                .map(|r| console::strip_ansi_codes(r).to_string())
                .collect()
        };
        // too narrow for even the DJ: nothing but the mascot and the wordmark.
        // Checked through the deck, which is the one drawing the header's own
        // goblin does not have -- his block ends in a pair of legs.
        assert!(
            !plain(SOLO_MIN_W - 1).iter().any(|r| r.contains("[_(")),
            "a band in a narrow header"
        );
        // the DJ alone, then the pair
        let solo = plain(SOLO_MIN_W);
        assert!(solo[2].contains("[_("), "the DJ's deck is missing at {SOLO_MIN_W}");
        assert!(!solo[2].contains("[===="), "the drummer came along too early");
        assert!(plain(BAND_MIN_W)[2].contains("[===="), "no drummer at {BAND_MIN_W}");
    }

    // The file is swapped block for block down a corridor, exactly as the band
    // is swapped frame for frame in its corner: a row that is not the full
    // block wide leaves the last frame's tail on the rule or shoves its
    // neighbour along. Eyeballed art drifts the first time somebody edits it.
    #[test]
    fn every_marcher_fills_his_block() {
        for pose in march_poses() {
            for row in pose {
                assert_eq!(
                    row.chars().count(),
                    MARCH_W as usize,
                    "{row:?} is {} columns, want {MARCH_W}",
                    row.chars().count()
                );
                assert!(row.is_ascii(), "{row:?} is ASCII");
            }
        }
    }

    // He is the mascot with his torso dropped, which is the trade the band
    // already makes -- so his head is on the row the wordmark is on, his body
    // on the tagline's, and his feet in the rule. Checked as "the bottom row is
    // the one with the legs in it": a pose assembled out of order would march a
    // head along the floor.
    #[test]
    fn the_marchers_stand_on_the_rule() {
        for pose in march_poses() {
            assert!(
                pose[MARCH_H - 1].trim().starts_with(['|', '/']),
                "the legs are on the bottom row: {pose:?}"
            );
            assert!(pose[0].contains(','), "the ears are on the top row: {pose:?}");
        }
    }

    // Rare is the entire design. A parade that turns up every minute is a
    // feature of the header; one that turns up every few minutes is something a
    // user looks up at. Measured on what is actually ON SCREEN rather than on
    // what the clock scheduled -- a crossing takes as long as the header is
    // wide, and the schedule is deliberately longer than the widest of them.
    #[test]
    fn a_parade_is_a_rare_thing() {
        let span = PARADE_CYCLE * 4096;
        let on = (0..span).filter(|&t| !marchers(120, true, t).is_empty()).count();
        let share = on as f64 / span as f64;
        assert!(
            (0.01..0.10).contains(&share),
            "goblins are marching {:.1}% of the time",
            share * 100.0
        );
        // and the gap between one crossing and the next is minutes, not seconds
        let crossings = (1..span)
            .filter(|&t| {
                marchers(120, true, t).is_empty() && !marchers(120, true, t - 1).is_empty()
            })
            .count();
        let minutes = span as f64 * BEAT_MS as f64 / 60_000.0 / crossings as f64;
        assert!(
            (3.0..10.0).contains(&minutes),
            "a parade every {minutes:.1} minutes, want one every few"
        );
    }

    // A goblin is drawn WHOLE or not at all -- the intro keeps the same rule
    // for the same reason, that a cut-off one reads as a rendering fault rather
    // than as a goblin. So over a crossing every marcher has to enter, the
    // whole file has to be on screen at once somewhere in the middle, and all
    // of them have to leave again before the window is up.
    #[test]
    fn the_file_crosses_the_corridor_whole() {
        // wide enough that all eight fit between the tagline and the band
        let (w, playing) = (140u16, true);
        let (left, right) = corridor(w, playing).expect("no corridor at 140 columns");
        let t0 = first_parade();
        let counts: Vec<usize> = (0..PARADE_ROOM)
            .map(|e| {
                let ms = marchers(w, playing, t0 + e);
                let mut prev: Option<u16> = None;
                for m in &ms {
                    assert!(
                        m.x >= left && m.x + MARCH_W <= right,
                        "a marcher is cut by the corridor at beat {e}: {} in {left}..{right}",
                        m.x
                    );
                    if let Some(p) = prev {
                        assert!(m.x >= p + MARCH_W, "two marchers overlap at beat {e}");
                    }
                    prev = Some(m.x);
                }
                ms.len()
            })
            .collect();
        assert_eq!(counts[0], 0, "the file is on screen before it has marched in");
        assert_eq!(*counts.iter().max().unwrap(), FILE, "the whole file is never on screen at once");
        assert_eq!(counts[PARADE_ROOM - 1], 0, "a marcher is still standing there at the end");
        // One crossing, not several: the beats with anybody on screen are one
        // unbroken stretch, so the file passes by once and does not come back
        // round for the rest of the schedule.
        let on: Vec<usize> = (0..PARADE_ROOM).filter(|&e| counts[e] > 0).collect();
        assert!(
            on.windows(2).all(|p| p[1] == p[0] + 1),
            "the file leaves and comes back: on screen at {on:?}"
        );

        // They keep formation and they only ever walk one way. Both fall out of
        // one invariant: the file is a rigid comb of blocks `PITCH` apart, so
        // every marcher on a beat shares a column residue, and that residue
        // steps LEFT by `SPEED` from each beat to the next.
        let residue = |e: usize| {
            let ms = marchers(w, playing, t0 + e);
            let r = ms[0].x % PITCH;
            assert!(ms.iter().all(|m| m.x % PITCH == r), "the file breaks formation at beat {e}");
            r
        };
        for pair in on.windows(2) {
            let (a, b) = (residue(pair[0]), residue(pair[1]));
            assert_eq!(
                b,
                (a + PITCH - SPEED % PITCH) % PITCH,
                "the file does not march left at beat {}",
                pair[1]
            );
        }
    }

    // The corridor is what is left of the header between the tagline and the
    // band, and below four blocks of it the file is a queue rather than an
    // army. An 80-column header gets no parade for the same reason it gets no
    // drummer: no band beats a broken one.
    #[test]
    fn a_parade_needs_a_corridor_to_march_down() {
        let t0 = first_parade();
        let busiest = |w: u16| {
            (0..PARADE_ROOM).map(|e| marchers(w, true, t0 + e).len()).max().unwrap()
        };
        for w in [1u16, 40, SOLO_MIN_W, BAND_MIN_W, 80] {
            assert_eq!(busiest(w), 0, "a parade squeezed into a {w}-column header");
        }
        assert!(busiest(120) >= 6, "no parade in a 120-column header");
        assert!(corridor(80, true).is_none() && corridor(120, true).is_some());
    }

    // The straggler is the reason the file reads as goblins and not as one
    // sprite repeated: he is a beat behind the rest of them, and now and then
    // he goes over his own feet. Both have to actually happen over a crossing,
    // and neither may ever happen to anybody else.
    #[test]
    fn the_straggler_is_out_of_step_and_falls_over() {
        let t0 = first_parade();
        let (mut behind, mut tripped) = (0, 0);
        for e in 0..PARADE_ROOM {
            let ms = marchers(140, true, t0 + e);
            // whoever is NOT the straggler marches in lockstep
            if ms.len() > 1 {
                let front = &ms[..ms.len() - 1];
                assert!(
                    front.windows(2).all(|p| p[0].rows == p[1].rows),
                    "the file breaks step at beat {e}"
                );
                assert!(
                    front.iter().all(|m| m.rows != STUMBLE),
                    "somebody other than the straggler fell over at beat {e}"
                );
                let back = &ms[ms.len() - 1];
                if back.rows == STUMBLE {
                    tripped += 1;
                } else if back.rows != front[0].rows {
                    behind += 1;
                }
            }
        }
        assert!(behind > 0, "the straggler keeps perfect step with the file");
        assert!(tripped > 0, "the straggler never trips");
        // but he trips occasionally, not constantly -- a goblin who is always
        // falling over is not clumsy, he is broken
        assert!(tripped * 3 < behind, "the straggler trips {tripped} beats and marches {behind}");
    }

    // The marchers are painted TRANSPARENTLY: a space in a pose paints nothing.
    // That is what leaves the rule running between a marcher's legs instead of
    // a rectangle of blanks punched out around each of them -- and the rule row
    // is the only place in the header where the difference is visible, because
    // it is the only row with anything drawn under them.
    #[test]
    fn the_marchers_stand_in_the_rule_rather_than_a_hole_in_it() {
        let _lang = crate::lang::speaking("en-US");
        crate::theme::set(crate::theme::Palette::Phosphor);
        let t0 = first_parade();
        let e = (0..PARADE_ROOM)
            .find(|&e| marchers(140, true, t0 + e).len() == FILE)
            .expect("the file is never all on screen");
        let rows: Vec<String> = header_rows_at(140, t0 + e, true)
            .iter()
            .map(|r| console::strip_ansi_codes(r).to_string())
            .collect();
        // legs in the rule, and not one blank anywhere past the art: every
        // column the marchers do not paint is still rule
        assert!(
            rows[3].contains("|_\\") || rows[3].contains("/_|"),
            "no legs in the rule: {:?}",
            rows[3]
        );
        assert!(
            rows[3].chars().skip(TEXT_COL).all(|c| c != ' '),
            "the rule has holes punched in it: {:?}",
            rows[3]
        );
        // and the rest of him is up where he belongs, past the header's text
        assert!(rows[1].contains("/\\,/\\"), "no ears beside the wordmark: {:?}", rows[1]);
        assert!(rows[2].contains("(o.o)"), "no faces beside the tagline: {:?}", rows[2]);
        assert!(
            rows[2].starts_with(&format!("   |___|      {}", crate::t!("app.tagline"))),
            "{:?}",
            rows[2]
        );
    }

    // Watching them go past is the thing that makes it one scene rather than
    // two animations sharing a header. It outranks the music, it is only ever
    // the gestures of LOOKING, and it stops when they do.
    #[test]
    fn he_stops_to_watch_them_go_past() {
        let looks: HashSet<Pose> = WATCH
            .iter()
            .flat_map(|g| g.iter().copied())
            .chain([REST])
            .collect();
        for t in 0..IDLE_CYCLE * 512 {
            assert!(looks.contains(&mascot_pose(t, true, true)), "he is not watching at {t}");
            // the music makes no difference to a goblin watching a parade
            assert_eq!(mascot_pose(t, true, true), mascot_pose(t, false, true), "at {t}");
        }
        // and it is a real difference: he does something else when there is
        // nothing to watch
        assert!(
            (0..IDLE_CYCLE * 512).any(|t| mascot_pose(t, true, false) != mascot_pose(t, true, true)),
            "watching a parade looks exactly like not watching one"
        );
        // he watches for as long as they are there, and not a beat longer --
        // the schedule runs well past the crossing on every real header width
        let t0 = first_parade();
        let last = (0..PARADE_ROOM)
            .filter(|&e| !marchers(120, true, t0 + e).is_empty())
            .max()
            .expect("no crossing to watch");
        assert!(last + 1 < PARADE_ROOM, "the file marches for the whole schedule");
        let (_, par) = (0, parade(120, true, t0 + last + 1));
        assert!(par.is_none(), "he would still be craning at an empty corridor");
    }
}


