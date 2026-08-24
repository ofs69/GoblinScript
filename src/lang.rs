//! The interface's words, kept OUT of the code and read from `languages/` at
//! startup -- so someone who does not read English can copy `en-US.json`,
//! translate the values, and have the app speak their language without a build.
//!
//! Every catalog is one flat JSON map of key -> string. Flat because a
//! translator should never have to understand a structure to work: open the
//! file, replace the right-hand sides, save. `language.name` is the one key
//! that names the file's own language, and it is what the picker's menu shows.
//!
//! **English is also EMBEDDED.** A key missing from a catalog -- a translation
//! that predates a new screen, a file half-finished, a typo in a key --
//! falls back to the English string rather than drawing a blank or a raw key.
//! The shipped `languages/en-US.json` is therefore both a working catalog and
//! the template: it is complete by construction, because the fallback is the
//! same text.
//!
//! Lookups return `&'static str` and cost an index plus a hash. Catalogs are
//! parsed once and leaked, and the active one is an `AtomicUsize` -- the same
//! shape as the palette in `theme`, and for the same reason: every surface
//! reads it, and a keypress has to change all of them on the next frame with
//! nothing threaded through call stacks that exist for other reasons.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::OnceLock;

/// The English catalog, compiled in. Shipped as a file too -- that copy is what
/// a translator starts from, and this one is what guarantees no screen can go
/// blank because a catalog on disk was incomplete.
const EN_US: &str = include_str!("../languages/en-US.json");

/// The tag the built-in catalog answers to.
pub const FALLBACK: &str = "en-US";

pub struct Catalog {
    /// The BCP-47 tag, which is also the file's stem: `zh-CN` <- `zh-CN.json`.
    pub code: String,
    /// What this language calls itself, from `language.name`. Shown in the
    /// picker, so it is written in the language it names -- a menu entry
    /// reading "Chinese" is no use to someone looking for 简体中文.
    pub name: String,
    strings: HashMap<String, String>,
}

impl Catalog {
    fn parse(code: &str, json: &str) -> Result<Catalog, serde_json::Error> {
        let strings: HashMap<String, String> = serde_json::from_str(json)?;
        let name = strings
            .get("language.name")
            .cloned()
            .unwrap_or_else(|| code.to_string());
        Ok(Catalog { code: code.to_string(), name, strings })
    }
}

/// A catalog file as text, whatever a text editor decided a text file is.
///
/// The invitation this module rests on is "copy `en-US.json` and edit it", and
/// the tools nearest to hand disagree about how to save the result: Notepad
/// offers UTF-8 WITH a byte-order mark, PowerShell's `>` writes UTF-16 with
/// one, and an editor left on the machine's own codepage writes bytes that are
/// not Unicode at all. `serde_json` reads none of those -- a byte-order mark
/// alone fails the parse on the first character -- so the mark is honoured
/// here, and bytes that are still not UTF-8 are read lossily rather than
/// dropped: a replacement character standing in one string is something the
/// translator can see and fix, where a file that quietly does not load reads
/// as our bug, in the one place where the user cannot open a debugger.
fn read_catalog(path: &Path) -> std::io::Result<String> {
    let raw = std::fs::read(path)?;
    let utf16 = |rest: &[u8], to: fn([u8; 2]) -> u16| {
        let units: Vec<u16> = rest.chunks_exact(2).map(|c| to([c[0], c[1]])).collect();
        String::from_utf16_lossy(&units)
    };
    Ok(match raw.as_slice() {
        [0xEF, 0xBB, 0xBF, rest @ ..] => String::from_utf8_lossy(rest).into_owned(),
        [0xFF, 0xFE, rest @ ..] => utf16(rest, u16::from_le_bytes),
        [0xFE, 0xFF, rest @ ..] => utf16(rest, u16::from_be_bytes),
        _ => String::from_utf8_lossy(&raw).into_owned(),
    })
}

fn catalogs() -> &'static [&'static Catalog] {
    static CATALOGS: OnceLock<Vec<&'static Catalog>> = OnceLock::new();
    CATALOGS.get_or_init(|| {
        // English first and always: index 0 is the fallback every lookup ends
        // at, so it must exist whatever is or is not on disk.
        let built_in = Catalog::parse(FALLBACK, EN_US)
            .expect("the built-in en-US catalog is not valid JSON");
        let mut out: Vec<&'static Catalog> = vec![Box::leak(Box::new(built_in))];
        let mut broken: Vec<String> = Vec::new();
        for (code, path) in on_disk() {
            // A file for a language already loaded REPLACES it -- including
            // en-US, so a user can correct our own wording without a build. The
            // embedded copy stays behind it as the per-key fallback.
            //
            // A file that is there and does not load is REMEMBERED rather than
            // passed over: the user put it there on purpose, and "your language
            // is not installed" would be the one answer that sends them looking
            // in the wrong place.
            let text = match read_catalog(&path) {
                Ok(t) => t,
                Err(e) => {
                    broken.push(format!("{code}.json ({e})"));
                    continue;
                }
            };
            let c = match Catalog::parse(&code, &text) {
                Ok(c) => c,
                Err(e) => {
                    broken.push(format!("{code}.json ({e})"));
                    continue;
                }
            };
            let c: &'static Catalog = Box::leak(Box::new(c));
            match out.iter().position(|o| o.code == c.code) {
                Some(i) => out[i] = c,
                None => out.push(c),
            }
        }
        // English keeps index 0 and the rest sort behind it. Index 0 is where
        // `ACTIVE` starts and where the picker's cycle begins, so a catalog
        // whose tag happens to sort before `en-US` must not be able to take
        // the seat of the language the app opens in.
        out[1..].sort_by(|a, b| a.code.cmp(&b.code));
        broken.sort();
        let _ = UNREADABLE.set(broken);
        out
    })
}

static UNREADABLE: OnceLock<Vec<String>> = OnceLock::new();

/// The files in `languages/` that are there and did not load, each with the
/// reason, for the one line at startup that says so. Empty in every ordinary
/// run -- and never empty and silent, which is the state this exists to end.
pub fn unreadable() -> &'static [String] {
    catalogs();
    UNREADABLE.get().map(Vec::as_slice).unwrap_or(&[])
}

/// The English compiled into the exe, whatever `languages/` holds.
///
/// Two readers, both of which need English that no file can move: the per-key
/// fallback below, so a half-written `en-US.json` on disk leaves the app
/// speaking English rather than drawing raw keys; and `en`, which is what the
/// failure log is written in.
fn builtin() -> &'static Catalog {
    static BUILT_IN: OnceLock<&'static Catalog> = OnceLock::new();
    BUILT_IN.get_or_init(|| {
        let c = Catalog::parse(FALLBACK, EN_US)
            .expect("the built-in en-US catalog is not valid JSON");
        Box::leak(Box::new(c))
    })
}

/// Every `languages/*.json` we can see, as (tag, path).
///
/// Two directories, in order: beside the executable, which is where a release
/// unpacks and where a user's own translation goes, and then the working
/// directory, which is what makes `cargo run` from the repo find the tree's own
/// copies without an install step.
fn on_disk() -> Vec<(String, PathBuf)> {
    let mut dirs: Vec<PathBuf> = Vec::new();
    if let Ok(exe) = std::env::current_exe() {
        if let Some(d) = exe.parent() {
            dirs.push(d.join("languages"));
        }
    }
    if let Ok(cwd) = std::env::current_dir() {
        let d = cwd.join("languages");
        if !dirs.contains(&d) {
            dirs.push(d);
        }
    }
    let mut out = Vec::new();
    for dir in dirs {
        let Ok(rd) = std::fs::read_dir(&dir) else { continue };
        for e in rd.flatten() {
            let p = e.path();
            if p.extension().and_then(|x| x.to_str()).is_some_and(|x| x.eq_ignore_ascii_case("json"))
            {
                if let Some(stem) = p.file_stem().and_then(|s| s.to_str()) {
                    out.push((stem.to_string(), p.clone()));
                }
            }
        }
    }
    out.sort();
    out
}

static ACTIVE: AtomicUsize = AtomicUsize::new(0);

/// Every language the app can currently speak, English first by sort order.
/// What the picker cycles through and what `--lang` is checked against.
pub fn available() -> &'static [&'static Catalog] {
    catalogs()
}

/// The tag in force.
pub fn active() -> &'static str {
    &catalogs()[ACTIVE.load(Ordering::Relaxed)].code
}

/// What the active language calls itself -- the chrome strip's label, so the
/// name beside the key that changes it is readable to whoever wants it.
pub fn current_name() -> &'static str {
    &catalogs()[ACTIVE.load(Ordering::Relaxed)].name
}

/// Switch languages, by tag. Matching is case-insensitive, and a bare primary
/// tag takes the first catalog that starts with it -- `zh` finds `zh-CN`, which
/// is what a system locale of `zh-Hans-CN` should also land on. Returns false
/// (changing nothing) when no catalog answers, so a bad `--lang` can say so.
pub fn set(tag: &str) -> bool {
    let all = catalogs();
    let exact = all.iter().position(|c| c.code.eq_ignore_ascii_case(tag));
    let prefix = || {
        let head = tag.split(['-', '_']).next().unwrap_or(tag);
        all.iter().position(|c| {
            c.code.split('-').next().is_some_and(|p| p.eq_ignore_ascii_case(head))
        })
    };
    match exact.or_else(prefix) {
        Some(i) => {
            ACTIVE.store(i, Ordering::Relaxed);
            true
        }
        None => false,
    }
}

/// Move to the next language (the picker's key). A no-op with only English
/// installed, which is the state the key is not offered in.
pub fn cycle() {
    let n = catalogs().len();
    if n > 1 {
        ACTIVE.store((ACTIVE.load(Ordering::Relaxed) + 1) % n, Ordering::Relaxed);
    }
}

/// The language the machine is set to, as a tag -- the default when nothing has
/// been chosen. Someone who has never opened the settings should still find the
/// app speaking their language if a catalog for it is installed.
pub fn system_tag() -> Option<String> {
    #[cfg(windows)]
    {
        use windows::Win32::Globalization::GetUserDefaultLocaleName;
        let mut buf = [0u16; 85]; // LOCALE_NAME_MAX_LENGTH
        let n = unsafe { GetUserDefaultLocaleName(&mut buf) };
        if n > 1 {
            // the count includes the terminating null
            return Some(String::from_utf16_lossy(&buf[..(n as usize - 1)]));
        }
        None
    }
    #[cfg(not(windows))]
    {
        // `zh_CN.UTF-8` -> `zh-CN`: the encoding suffix and the underscore are
        // POSIX spelling of the same tag.
        std::env::var("LC_ALL")
            .or_else(|_| std::env::var("LANG"))
            .ok()
            .map(|v| v.split('.').next().unwrap_or("").replace('_', "-"))
            .filter(|v| !v.is_empty())
    }
}

/// One string, in the active language.
///
/// A key with no translation falls back to English, and a key in NO catalog
/// returns the key itself -- visible, greppable, and never a blank label. That
/// last case is a bug in this repo rather than in anyone's translation, which
/// is why it is loud rather than silent.
pub fn t(key: &str) -> &'static str {
    // Leaked so the signature can stay `&'static str` for the common path. Only
    // reachable from a key this repo failed to ship, so it happens once.
    try_t(key).unwrap_or_else(|| Box::leak(key.to_string().into_boxed_str()))
}

/// The same lookup, but saying so when nobody has the key.
///
/// This is what the CLI help uses: an argument's English help text lives in its
/// own doc comment (where the developer reading the code finds it), and a
/// translation OVERRIDES it. So "no catalog has this" has to mean "leave
/// clap's own text alone" rather than "draw the key".
pub fn try_t(key: &str) -> Option<&'static str> {
    let all = catalogs();
    let i = ACTIVE.load(Ordering::Relaxed);
    all[i]
        .strings
        .get(key)
        .or_else(|| all[0].strings.get(key))
        .or_else(|| builtin().strings.get(key))
        .map(|s| s.as_str())
}

/// One string in English, whatever the interface is speaking.
///
/// For the failure log, which is written to be READ BACK -- by whoever
/// receives it, in the language this repo is written in. Everything a user
/// sees goes through `t` instead.
pub fn en(key: &str) -> &'static str {
    // Leaked on a miss, exactly as `t` does, and for the same reason: a key
    // this repo never shipped is a bug here, and a visible one is the cheapest
    // kind. It can only happen once per key.
    builtin()
        .strings
        .get(key)
        .map(|s| s.as_str())
        .unwrap_or_else(|| Box::leak(key.to_string().into_boxed_str()))
}

/// The active catalog as JSON, for the browser pages -- which are surfaces of
/// the same app and speak the same language as the picker that launched them.
///
/// English is MERGED underneath, so the page gets one complete map and needs no
/// fallback logic of its own: the same per-key rule the terminal side has.
pub fn catalog_json() -> serde_json::Value {
    let all = catalogs();
    let mut out = all[0].strings.clone();
    out.extend(all[ACTIVE.load(Ordering::Relaxed)].strings.clone());
    serde_json::json!({ "code": active(), "strings": out })
}

/// Substitute `{name}` placeholders in a template. Used by `t!` when arguments
/// are given; a translator moves a placeholder wherever their grammar wants it,
/// and one they drop simply does not appear.
pub fn fill(template: &str, args: &[(&str, &str)]) -> String {
    let mut out = String::with_capacity(template.len() + 16);
    let mut rest = template;
    while let Some(open) = rest.find('{') {
        out.push_str(&rest[..open]);
        // An unbalanced brace is text, not a placeholder. The prefix is already
        // out, so what is left to copy starts AT the brace -- taking `rest`
        // whole here would write the prefix twice.
        let Some(close) = rest[open..].find('}').map(|i| i + open) else {
            rest = &rest[open..];
            break;
        };
        let name = &rest[open + 1..close];
        match args.iter().find(|(k, _)| *k == name) {
            Some((_, v)) => out.push_str(v),
            // Left as written: a placeholder we have no argument for is a
            // mismatch between catalog and code, and hiding it hides the bug.
            None => out.push_str(&rest[open..=close]),
        }
        rest = &rest[close + 1..];
    }
    out.push_str(rest);
    out
}

/// `t!("some.key")` is the string; `t!("some.key", n = 3)` fills `{n}` in it.
///
/// The two forms differ in type on purpose -- `&'static str` when nothing is
/// substituted, so the common case (a label, a menu entry, a key name) costs
/// nothing per frame, and `String` only where a value actually goes in.
#[macro_export]
macro_rules! t {
    ($key:expr) => { $crate::lang::t($key) };
    ($key:expr, $($name:ident = $value:expr),+ $(,)?) => {
        $crate::lang::fill($crate::lang::t($key), &[$((stringify!($name), &$value.to_string())),+])
    };
}

/// Take the language for the length of a test.
///
/// The active language is process-wide -- it has to be, every surface reads it
/// -- and tests run in parallel, so any test that asserts on WORDS holds this
/// while it runs. Without it a test rendering Chinese flips the global under a
/// test looking for "Space select", and the failure is intermittent, which is
/// the worst kind. `theme` gets away without one only because no test asserts
/// on a colour.
#[cfg(test)]
pub fn speaking(tag: &str) -> std::sync::MutexGuard<'static, ()> {
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    // A poisoned lock means an earlier language test panicked. The language is
    // set below regardless, so the state this hands back is still sound.
    let g = LOCK.lock().unwrap_or_else(|e| e.into_inner());
    assert!(set(tag), "no catalog for {tag}");
    g
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every way a Windows editor saves a file a translator was told to copy
    /// and edit. All of them have to come back as the same catalog: the
    /// app cannot answer "your language is not installed" about a file the user
    /// is looking at, and what separates these is a menu setting nobody reading
    /// a JSON file has any reason to have noticed.
    #[test]
    fn a_catalog_survives_whatever_encoding_it_was_saved_in() {
        const TEXT: &str = r#"{"language.name":"Francais","page.done":"Termine"}"#;
        let mut utf16 = vec![0xFFu8, 0xFE];
        for u in TEXT.encode_utf16() {
            utf16.extend_from_slice(&u.to_le_bytes());
        }
        let mut utf16be = vec![0xFEu8, 0xFF];
        for u in TEXT.encode_utf16() {
            utf16be.extend_from_slice(&u.to_be_bytes());
        }
        let mut utf8_bom = vec![0xEFu8, 0xBB, 0xBF];
        utf8_bom.extend_from_slice(TEXT.as_bytes());

        let dir = std::env::temp_dir().join(format!("gs_lang_enc_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        for (how, bytes) in [
            ("plain utf-8", TEXT.as_bytes().to_vec()),
            ("utf-8 with a BOM", utf8_bom),
            ("utf-16 le", utf16),
            ("utf-16 be", utf16be),
        ] {
            let f = dir.join("fr-FR.json");
            std::fs::write(&f, &bytes).unwrap();
            let text = read_catalog(&f).unwrap_or_else(|e| panic!("{how}: {e}"));
            let c = Catalog::parse("fr-FR", &text)
                .unwrap_or_else(|e| panic!("{how} did not parse: {e}"));
            assert_eq!(c.name, "Francais", "{how}");
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A catalog saved in the machine's own codepage rather than in Unicode
    /// still loads, with the bytes that are not UTF-8 standing out as
    /// replacement characters. Visibly wrong in one string beats a language
    /// that is not there: only one of the two tells the translator what to fix.
    #[test]
    fn a_codepage_catalog_loads_with_its_damage_visible() {
        let mut raw = br#"{"language.name":"Fran"#.to_vec();
        raw.push(0xE7); // cp1252 c-cedilla, which is not UTF-8
        raw.extend_from_slice(br#"ais"}"#);
        let dir = std::env::temp_dir().join(format!("gs_lang_cp_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let f = dir.join("fr-FR.json");
        std::fs::write(&f, &raw).unwrap();
        let c = Catalog::parse("fr-FR", &read_catalog(&f).unwrap())
            .expect("a codepage catalog still has to parse");
        assert!(
            c.name.contains(char::REPLACEMENT_CHARACTER),
            "the byte that is not Unicode has to be visible, got {}",
            c.name
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The embedded catalog is the fallback every lookup ends at, so it has to
    /// parse and it has to name itself. A broken JSON edit here would otherwise
    /// only show up as a panic on someone's first launch.
    #[test]
    fn the_built_in_catalog_parses_and_names_itself() {
        let c = Catalog::parse(FALLBACK, EN_US).expect("en-US.json is not valid JSON");
        assert_eq!(c.code, "en-US");
        assert_eq!(c.name, "English", "language.name is missing from en-US.json");
        assert!(!c.strings.is_empty());
    }

    /// The app opens in English unless something says otherwise, and every
    /// lookup falls back to it. Both of those are "index 0 is English", and
    /// catalogs are sorted by tag -- so a language whose tag sorts before
    /// `en-US` (de-DE, cs-CZ, ar-EG...) is exactly the case that could quietly
    /// take the seat.
    #[test]
    fn english_holds_the_first_seat_whatever_sorts_before_it() {
        assert_eq!(catalogs()[0].code, FALLBACK);
        let rest: Vec<&str> = catalogs()[1..].iter().map(|c| c.code.as_str()).collect();
        let mut sorted = rest.clone();
        sorted.sort_unstable();
        assert_eq!(rest, sorted, "the languages behind English are out of order");
    }

    /// The failure log is written in English while the screen is in whatever
    /// the user chose -- so the two lookups have to be able to disagree.
    #[test]
    fn the_log_reads_english_while_the_screen_reads_chinese() {
        let _lang = speaking("zh-CN");
        let key = "console.stage.encode";
        assert_eq!(en(key), "encode");
        assert_ne!(t(key), en(key), "zh-CN has not translated {key}");
    }

    /// Every catalog in the tree, read from the repo rather than from wherever
    /// the test binary happens to sit.
    fn shipped() -> Vec<(String, HashMap<String, String>)> {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("languages");
        let mut out = Vec::new();
        for e in std::fs::read_dir(&dir).expect("the languages/ directory is missing").flatten() {
            let p = e.path();
            if p.extension().and_then(|x| x.to_str()) != Some("json") {
                continue;
            }
            let stem = p.file_stem().unwrap().to_string_lossy().to_string();
            let text = std::fs::read_to_string(&p).expect("unreadable catalog");
            let map: HashMap<String, String> = serde_json::from_str(&text)
                .unwrap_or_else(|e| panic!("{stem}.json is not valid JSON: {e}"));
            out.push((stem, map));
        }
        out
    }

    /// The catalogs we SHIP have to stay level with each other. A key added to
    /// the code and to English but forgotten in Chinese is invisible -- the
    /// fallback quietly draws English into a Chinese screen -- and a key that
    /// exists ONLY in a translation is a typo that will never be read. Neither
    /// shows up at runtime, so both are caught here.
    ///
    /// This is about the files in this repo. A translation someone else writes
    /// is under no such obligation: it falls back per key, by design.
    #[test]
    fn the_shipped_catalogs_carry_the_same_keys() {
        let all = shipped();
        let en: &HashMap<String, String> = &all
            .iter()
            .find(|(c, _)| c == FALLBACK)
            .expect("languages/en-US.json is missing")
            .1;
        for (code, map) in &all {
            if code == FALLBACK {
                continue;
            }
            // `cli.*` is the one family English does not carry: an argument's
            // English help IS its doc comment in `main.rs`, where the developer
            // reading the code finds it, and a catalog only ever OVERRIDES that.
            // So a translation may add `cli.` keys freely, and is not asked for
            // them -- but every other key has to line up exactly.
            let mine = |k: &&String| !k.starts_with("cli.");
            let mut missing: Vec<&str> = en
                .keys()
                .filter(mine)
                .filter(|k| !map.contains_key(*k))
                .map(|k| k.as_str())
                .collect();
            let mut unknown: Vec<&str> = map
                .keys()
                .filter(mine)
                .filter(|k| !en.contains_key(*k))
                .map(|k| k.as_str())
                .collect();
            missing.sort();
            unknown.sort();
            assert!(missing.is_empty(), "{code}.json is missing keys: {missing:?}");
            assert!(unknown.is_empty(), "{code}.json has keys English does not: {unknown:?}");
        }
    }

    /// Every key the browser pages ask for has to EXIST. A key the page names
    /// and the catalog does not have is silent at runtime -- the element simply
    /// keeps the English in the markup -- so a typo in a `data-t` would ship a
    /// page that is translated except for one label nobody can explain.
    ///
    /// Four ways a page names a key, all covered: `data-t` on an element,
    /// `data-t-title` for a tooltip, `data-t-placeholder` for an input's ghost
    /// text, and `T(...)` in its script, in either quote. The help modal's keys
    /// are built from each `?` button's `data-h`, so those are checked as the
    /// pair of keys they resolve to.
    #[test]
    fn the_pages_only_ask_for_keys_the_catalog_has() {
        let en = Catalog::parse(FALLBACK, EN_US).unwrap().strings;
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
        for page in ["review.html", "vr.html"] {
            let html = std::fs::read_to_string(dir.join(page)).expect("page is missing");
            let mut want: Vec<String> = Vec::new();
            for (attr, prefix) in [
                ("data-t=\"", ""),
                ("data-t-title=\"", ""),
                ("data-t-placeholder=\"", ""),
            ] {
                let mut rest = html.as_str();
                while let Some(i) = rest.find(attr) {
                    rest = &rest[i + attr.len()..];
                    let end = rest.find('"').expect("unterminated attribute");
                    want.push(format!("{prefix}{}", &rest[..end]));
                    rest = &rest[end..];
                }
            }
            // T('key') -- the LITERAL calls only, in whichever quote the page
            // writes them. A `T('prefix.' + x)` names a family rather than a
            // key; those families reach the catalog through the markup's own
            // `data-t`/`data-h` instead, which is why both are scanned.
            for (open, quote) in [("T('", '\''), ("T(\"", '"')] {
                let mut rest = html.as_str();
                while let Some(i) = rest.find(open) {
                    rest = &rest[i + open.len()..];
                    if let Some(end) = rest.find(quote) {
                        let concatenated = rest[end + 1..].trim_start().starts_with('+');
                        if !concatenated {
                            want.push(rest[..end].to_string());
                        }
                        rest = &rest[end..];
                    }
                }
            }
            // each ? button resolves to a title and a body
            let mut rest = html.as_str();
            while let Some(i) = rest.find("data-h=\"") {
                rest = &rest[i + 8..];
                let end = rest.find('"').expect("unterminated attribute");
                let h = &rest[..end];
                want.push(format!("page.help.{h}.title"));
                want.push(format!("page.help.{h}.body"));
                rest = &rest[end..];
            }
            let mut missing: Vec<&str> = want
                .iter()
                .filter(|k| !en.contains_key(*k))
                .map(|k| k.as_str())
                .collect();
            missing.sort();
            missing.dedup();
            assert!(missing.is_empty(), "{page} asks for keys en-US.json does not have: {missing:?}");
        }
    }

    /// A translated `<option>` carries an explicit `value`.
    ///
    /// An option with no `value` attribute takes its value from its own TEXT,
    /// and translating the page is exactly the act of replacing that text. The
    /// value is a recipe's identity -- `steady`, `cautious`, `off` -- and it
    /// crosses to `style.rs` through `from_label`, so it has to read the same
    /// in every language. Without the attribute the page posts the Chinese
    /// word, or (once the script's own `select.value = 'steady'` matches no
    /// option) the empty string, and the styling request is refused.
    #[test]
    fn a_translated_option_carries_its_value_as_an_attribute() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
        for page in ["review.html", "vr.html"] {
            let html = std::fs::read_to_string(dir.join(page)).expect("page is missing");
            let mut rest = html.as_str();
            while let Some(i) = rest.find("<option") {
                rest = &rest[i..];
                let end = rest.find('>').expect("unterminated tag");
                let tag = &rest[..end];
                assert!(
                    !tag.contains("data-t=\"") || tag.contains("value=\""),
                    "{page} has a translated option with no value attribute: {tag}>"
                );
                rest = &rest[end..];
            }
        }
    }

    /// A placeholder is the one thing a translation can get wrong that changes
    /// what the user is TOLD rather than how it reads -- a `{n}` dropped from a
    /// count, a `{size}` that became `{sz}`. The set per key has to match
    /// English exactly, whatever order the grammar puts them in.
    #[test]
    fn every_translation_keeps_english_s_placeholders() {
        let holes = |s: &str| -> std::collections::BTreeSet<String> {
            let mut out = std::collections::BTreeSet::new();
            let mut rest = s;
            while let Some(o) = rest.find('{') {
                let Some(c) = rest[o..].find('}').map(|i| i + o) else { break };
                out.insert(rest[o + 1..c].to_string());
                rest = &rest[c + 1..];
            }
            out
        };
        let all = shipped();
        let en = &all.iter().find(|(c, _)| c == FALLBACK).unwrap().1;
        for (code, map) in &all {
            if code == FALLBACK {
                continue;
            }
            for (key, en_text) in en.iter() {
                let Some(text) = map.get(key) else { continue };
                assert_eq!(
                    holes(text),
                    holes(en_text),
                    "{code}.json key {key:?} does not carry English's placeholders"
                );
            }
        }
    }

    /// Placeholders are the one thing a translator can get wrong in a way that
    /// reaches the screen, so every outcome is defined: filled, left alone, or
    /// passed through as text.
    #[test]
    fn placeholders_fill_move_and_survive_being_dropped() {
        assert_eq!(fill("{n} selected", &[("n", "3")]), "3 selected");
        // a translation may reorder, repeat, or omit them
        assert_eq!(fill("selected: {n}", &[("n", "3")]), "selected: 3");
        assert_eq!(fill("{n}/{n}", &[("n", "3")]), "3/3");
        assert_eq!(fill("nothing here", &[("n", "3")]), "nothing here");
        // an argument with no placeholder, and a placeholder with no argument
        assert_eq!(fill("{missing}", &[("n", "3")]), "{missing}");
        // braces that are not placeholders are text
        assert_eq!(fill("a { b", &[]), "a { b");
        assert_eq!(fill("100%{", &[]), "100%{");
    }

    /// `--lang zh` and a system locale of `zh-Hans-CN` both have to land on the
    /// `zh-CN` catalog, and a tag nobody ships has to fail rather than pick
    /// something arbitrary.
    #[test]
    fn a_language_is_found_by_tag_or_by_its_primary_subtag() {
        let _lang = speaking("en-US");
        assert!(set("EN-us"), "tags are case-insensitive");
        assert!(set("en"), "a bare primary subtag finds its catalog");
        assert!(!set("qq-ZZ"), "a language nobody ships is refused, not guessed");
        assert_eq!(active(), "en-US", "a refused tag leaves the active one alone");
    }
}
