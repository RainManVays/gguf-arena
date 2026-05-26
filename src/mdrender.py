from __future__ import annotations
import re
import tkinter.font as tkfont

_INLINE_RE = re.compile(
    r'(`+)(.+?)\1'
    r'|\*\*\*(\S[^\n]*?\S|\S)\*\*\*'
    r'|\*\*(\S[^\n]*?\S|\S)\*\*'
    r'|\*(\S[^\n]*?\S|\S)\*'
    r'|_(\S[^\n]*?\S|\S)_',
)

_HDR_RE     = re.compile(r'^(#{1,3})\s+(.*)')
_FENCE_RE   = re.compile(r'^\s*```')
_LIST_RE    = re.compile(r'^(\s*)([-*+]|\d+\.)\s+(.*)')
_QUOTE_RE   = re.compile(r'^>\s?(.*)')
_HR_RE      = re.compile(r'^[-*_]{3,}\s*$')
_TBL_ROW_RE = re.compile(r'^\|(.+)\|$')
_TBL_SEP_RE = re.compile(r'^\|(\s*:?-+:?\s*\|)+\s*$')


def _cfg_tags(t) -> None:
    # Read the widget's actual (possibly DPI-scaled) font so tags match exactly.
    try:
        f = tkfont.Font(font=t.cget("font"))
        fam  = f.cget("family")
        fsz  = abs(f.cget("size"))  # negative means pixels on some systems
    except Exception:
        fam, fsz = "Monospace", 12

    mono = (fam, fsz)
    t.tag_configure("h1", font=(fam, fsz + 6, "bold"), foreground="#ffffff",
                    spacing1=8, spacing3=6)
    t.tag_configure("h2", font=(fam, fsz + 3, "bold"), foreground="#eeeeee",
                    spacing1=6, spacing3=4)
    t.tag_configure("h3", font=(fam, fsz + 1, "bold"), foreground="#dddddd",
                    spacing1=4, spacing3=2)
    t.tag_configure("bold",        font=(fam, fsz, "bold"))
    t.tag_configure("italic",      font=(fam, fsz, "italic"))
    t.tag_configure("bold_italic", font=(fam, fsz, "bold italic"))
    t.tag_configure("code_block",  font=mono,
                    background="#181818", foreground="#c8d3f5",
                    lmargin1=12, lmargin2=12, spacing1=2, spacing3=2)
    t.tag_configure("code_inline", font=mono,
                    background="#383838", foreground="#f9a97e")
    t.tag_configure("bullet",      lmargin1=8,  lmargin2=28)
    t.tag_configure("blockquote",  foreground="#888888",
                    lmargin1=24, lmargin2=24)
    t.tag_configure("hr",          foreground="#555555")
    t.tag_configure("tbl_head",    font=(fam, fsz, "bold"), foreground="#88ccff")
    t.tag_configure("tbl_sep",     foreground="#555555")


def _inline(t, text: str, block: str | None = None) -> None:
    """Insert `text` into tk.Text `t`, applying inline MD tags.
    `block` is an optional extra tag applied to every fragment (e.g. 'blockquote').
    Uses insert(text, tags) — no position tracking needed.
    """
    pos = 0
    for m in _INLINE_RE.finditer(text):
        if m.start() > pos:
            _ins(t, text[pos:m.start()], block)

        if m.group(1):                        # `code`
            _ins(t, m.group(2), block, "code_inline")
        elif m.group(3):                      # ***bold+italic***
            _ins(t, m.group(3), block, "bold_italic")
        elif m.group(4):                      # **bold**
            _ins(t, m.group(4), block, "bold")
        elif m.group(5) or m.group(6):        # *italic* or _italic_
            _ins(t, m.group(5) or m.group(6), block, "italic")
        pos = m.end()

    if pos < len(text):
        _ins(t, text[pos:], block)


def _ins(t, text: str, *tags) -> None:
    """Insert text with the given tags (None entries are filtered out)."""
    active = tuple(tag for tag in tags if tag)
    if active:
        t.insert("end", text, active)
    else:
        t.insert("end", text)


def render(tb, raw: str) -> None:
    """Render markdown into a CTkTextbox. Uses tb._textbox for all tk ops."""
    t = tb._textbox

    tb.configure(state="normal")
    t.delete("1.0", "end")
    _cfg_tags(t)

    lines = raw.split("\n")
    in_code = False

    for idx, line in enumerate(lines):
        nl = "\n" if idx < len(lines) - 1 else ""

        # ── code fence ──────────────────────────────────────────────────────
        if _FENCE_RE.match(line):
            in_code = not in_code
            if not in_code and nl:
                t.insert("end", nl)
            continue

        if in_code:
            t.insert("end", line + nl, ("code_block",))
            continue

        # ── HR ──────────────────────────────────────────────────────────────
        if _HR_RE.match(line):
            t.insert("end", "─" * 48 + nl, ("hr",))
            continue

        # ── table separator — skip ───────────────────────────────────────────
        if _TBL_SEP_RE.match(line):
            continue

        # ── table row ───────────────────────────────────────────────────────
        if _TBL_ROW_RE.match(line):
            next_line = lines[idx + 1] if idx + 1 < len(lines) else ""
            is_head   = bool(_TBL_SEP_RE.match(next_line))
            cells     = [c.strip() for c in line.strip("|").split("|")]
            block     = "tbl_head" if is_head else None
            t.insert("end", "  ")
            for i, cell in enumerate(cells):
                if i > 0:
                    t.insert("end", " │ ", ("tbl_sep",))
                _inline(t, cell, block)
            t.insert("end", nl)
            continue

        # ── header ──────────────────────────────────────────────────────────
        m = _HDR_RE.match(line)
        if m:
            _inline(t, m.group(2), f"h{len(m.group(1))}")
            t.insert("end", nl)
            continue

        # ── blockquote ──────────────────────────────────────────────────────
        m = _QUOTE_RE.match(line)
        if m:
            _inline(t, m.group(1), "blockquote")
            t.insert("end", nl)
            continue

        # ── list item ───────────────────────────────────────────────────────
        m = _LIST_RE.match(line)
        if m:
            bullet = "•" if not m.group(2)[0].isdigit() else m.group(2)
            t.insert("end", f"  {bullet} ", ("bullet",))
            _inline(t, m.group(3), "bullet")
            t.insert("end", nl)
            continue

        # ── normal text ─────────────────────────────────────────────────────
        _inline(t, line)
        t.insert("end", nl)

    tb.configure(state="disabled")
