"""Central design tokens for the entire UI.

Usage in src/:      from .theme import ...
Usage in mixins/:   from ..theme import ...
"""

# ── Fonts ─────────────────────────────────────────────────────────────────────
FONT_MONO    = ("Monospace", 12)
FONT_MONO_SM = ("Monospace", 10)   # compact table cells
FONT_SMALL   = ("Sans", 11)
FONT_BOLD    = ("Sans", 12, "bold")
FONT_TINY    = ("Sans", 10)        # hints, counters, small labels

# ── Primary actions ───────────────────────────────────────────────────────────
CLR_RUN          = "#1e6e1e"
CLR_RUN_HOV      = "#278a27"

CLR_PRIMARY      = "#1f6aa5"       # blue accent: buttons, bars
CLR_PRIMARY_HOV  = "#1a5a8a"

CLR_STOP         = "#7a2222"       # stop / prominent delete
CLR_STOP_HOV     = "#992222"

CLR_DANGER       = "#5a1a1a"       # secondary destructive: clear, remove
CLR_DANGER_HOV   = "#7a2222"

CLR_JUDGE        = "#4a3a7a"
CLR_JUDGE_HOV    = "#5a4a9a"

CLR_DISABLED     = "#444444"       # single-btn in batch mode
CLR_DISABLED_HOV = "#555555"
CLR_DISABLED_TB  = ("#303030", "#252525")   # CTkTextbox fg_color when disabled

# ── Status / indicator ────────────────────────────────────────────────────────
CLR_WIN  = "#c8a000"   # leaderboard winner gold
CLR_OK   = "#44aa44"   # success (token ratio, copy flash)
CLR_WARN = "#ddaa00"   # warning (token ratio near limit)
CLR_ERR  = "#cc4444"   # error text / status

# ── Text shades (darkest → lightest) ─────────────────────────────────────────
CLR_TXT_GHOST  = "#666666"
CLR_TXT_FAINT  = "#777777"
CLR_TXT_DIM    = "#888888"
CLR_TXT_MUTED  = "#aaaaaa"
CLR_TXT_NORMAL = "#cccccc"
CLR_TXT_BRIGHT = "#eeeeee"

CLR_ERR_TEXT   = "#cc6666"         # error cell text in batch table

# ── Canvas / surface ─────────────────────────────────────────────────────────
CLR_CANVAS_BG = "#2b2b2b"
