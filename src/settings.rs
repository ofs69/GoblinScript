//! A small JSON settings file next to the executable, so preferences survive
//! between runs. Best-effort throughout: any read/parse/write error just falls
//! back to (or keeps) the defaults -- settings are a convenience, never a
//! dependency of a draft.

use crate::style;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Field defaults come from `Default` (below), not from the derive: with
/// `#[serde(default)]` at container level, serde fills a missing field from it
/// too, so "never chosen" means the same thing whether the settings file is
/// absent, truncated, or simply predates the field.
#[derive(Serialize, Deserialize)]
#[serde(default)]
pub struct Settings {
    /// The directory the last batch was started from -- the picker reopens here
    /// on the next launch. `None` (or a path that no longer exists) falls back
    /// to the launch directory.
    pub last_dir: Option<String>,
    /// The style accepted at the end of the last review -- the picker seeds the
    /// next session with it, so a preferred recipe carries across launches.
    pub last_params: Option<style::Params>,
    /// The colour scheme, cycled with T in the picker. `None` = never chosen,
    /// which means the house phosphor green.
    pub theme: Option<crate::theme::Palette>,
    /// The language, as the tag of a file in `languages/` (cycled with G in the
    /// picker). `None` = never chosen, which means the machine's own language
    /// where a catalog for it is installed, and English otherwise -- so someone
    /// who never opens this file still gets their own language if it is there.
    pub lang: Option<String>,
    /// Sound effects off (M in the picker). Defaults to on -- a user who wants
    /// silence says so once and it sticks.
    #[serde(default)]
    pub muted: bool,
    /// Background music on. ON by default -- the goblins put a record on while
    /// they work. One press of M turns it off and that choice sticks, so the
    /// default costs an unwanted listener exactly one keystroke, once.
    ///
    /// Read by the PICKER only. A named-video run is silent unless `--music`
    /// says otherwise, so a preference set here never surprises someone who
    /// called the tool from a terminal rather than opening it.
    pub music: bool,
    /// How loud that music is (V in the picker). `None` = never chosen.
    pub volume: Option<crate::sound::Volume>,
    /// Auto-crop on (C in the picker): zoom the encoder onto the attention's
    /// region before drafting. ON by default, like the CLI -- a clip whose
    /// attention wants the whole frame is left alone anyway, so the default
    /// costs a probe and nothing else. The choice sticks when a batch is
    /// started with it. A named-video run reads the flags only.
    pub autocrop: bool,
    /// The crop check on (K in the picker): show each video's crop rects in
    /// the browser before the goblins read it, so a rect that is wrong can be
    /// dragged onto the action. ON by default -- the crop is the one decision
    /// a person makes better than the goblins in a glance, and it is free to
    /// make here. Switched off, a batch runs start to finish with nobody at
    /// the keyboard.
    pub crop_edit: bool,
}

impl Default for Settings {
    fn default() -> Self {
        Settings {
            last_dir: None,
            last_params: None,
            theme: None,
            lang: None,
            muted: false,
            music: true,
            volume: None,
            autocrop: true,
            crop_edit: true,
        }
    }
}

/// `settings.json` beside the exe (where the cache lives too).
fn path() -> Option<PathBuf> {
    std::env::current_exe()
        .ok()?
        .parent()
        .map(|d| d.join("settings.json"))
}

impl Settings {
    pub fn load() -> Self {
        path()
            .and_then(|p| std::fs::read_to_string(p).ok())
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_default()
    }

    pub fn save(&self) {
        if let (Some(p), Ok(s)) = (path(), serde_json::to_string_pretty(self)) {
            let _ = std::fs::write(p, s);
        }
    }

    /// Capture the live presentation state (theme, mute, music) into the
    /// settings about to be saved. Those three live in process-wide globals
    /// because every surface reads them; this is the one place they are copied
    /// back out to disk.
    pub fn remember_presentation(&mut self) {
        self.theme = Some(crate::theme::active());
        self.lang = Some(crate::lang::active().to_string());
        self.muted = crate::sound::muted();
        self.music = crate::sound::music_on();
        self.volume = Some(crate::sound::volume());
    }

    /// The remembered directory, if it still exists.
    pub fn start_dir(&self) -> Option<PathBuf> {
        self.last_dir
            .as_ref()
            .map(PathBuf::from)
            .filter(|p| p.is_dir())
    }
}
