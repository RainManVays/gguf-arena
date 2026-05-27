from __future__ import annotations

import tkinter as tk

import customtkinter as ctk


class UIRulesMixin:
    """Common keyboard and mouse rules applied uniformly across all interactive widgets.

    Call _activate_ui_rules() once after the window exists.
    Call _setup_scroll_wheel(sf) for each CTkScrollableFrame.
    Call _rebind_scroll_wheel(sf) after dynamic content is added to sf.
    Call _bind_textbox(tb) for each editable CTkTextbox.
    """

    # ── One-time class-level setup ────────────────────────────────────────────

    def _activate_ui_rules(self) -> None:
        """Register class-level key bindings. Call once after the window exists."""
        self.bind_class("Entry", "<Control-a>", self._on_ctrl_a)
        self.bind_class("Text",  "<Control-a>", self._on_ctrl_a)
        # Override default paste so it replaces the selection instead of concatenating.
        self.bind_class("Text",  "<Control-v>", self._on_ctrl_v_replace)

    # ── Scrollable frame mouse-wheel support ──────────────────────────────────

    def _setup_scroll_wheel(self, sf: ctk.CTkScrollableFrame) -> None:
        """Bind mouse-wheel scrolling to sf and all its current children.

        Stores sf._rebind_scroll so _rebind_scroll_wheel() can re-apply after
        dynamic content is added.
        """
        canvas = sf._parent_canvas

        def _scroll(event):
            delta = -1 if (event.num == 4 or event.delta > 0) else 1
            canvas.yview_scroll(delta, "units")

        def _bind_tree(widget):
            widget.bind("<MouseWheel>", _scroll, add="+")
            widget.bind("<Button-4>",   _scroll, add="+")
            widget.bind("<Button-5>",   _scroll, add="+")
            for child in widget.winfo_children():
                _bind_tree(child)

        _bind_tree(sf)
        _bind_tree(canvas)
        sf._rebind_scroll = lambda: _bind_tree(sf)  # type: ignore[attr-defined]

    def _rebind_scroll_wheel(self, sf: ctk.CTkScrollableFrame) -> None:
        """Re-apply scroll bindings after sf content is repopulated."""
        rebind = getattr(sf, "_rebind_scroll", None)
        if rebind:
            rebind()

    # ── Text widget keyboard rules ────────────────────────────────────────────

    def _bind_textbox(self, tb: ctk.CTkTextbox) -> None:
        """Bind undo/redo to a CTkTextbox. Ctrl+A and Ctrl+V are handled class-wide."""
        tb.bind("<Control-z>", self._on_undo)
        tb.bind("<Control-y>", self._on_redo)
        tb.bind("<Control-Z>", self._on_redo)

    def _on_ctrl_a(self, event) -> str:
        w = event.widget
        if isinstance(w, tk.Entry):
            w.select_range(0, "end")
            w.icursor("end")
        elif isinstance(w, tk.Text):
            w.tag_add("sel", "1.0", "end")
        return "break"

    def _on_ctrl_v_replace(self, event) -> str:
        w = event.widget
        if not isinstance(w, tk.Text) or str(w.cget("state")) != "normal":
            return "break"
        try:
            text = w.clipboard_get()
        except tk.TclError:
            return "break"
        if w.tag_ranges("sel"):
            w.delete("sel.first", "sel.last")
        w.insert("insert", text)
        return "break"

    def _on_undo(self, event) -> str:
        try:
            event.widget.edit_undo()
        except Exception:
            pass
        return "break"

    def _on_redo(self, event) -> str:
        try:
            event.widget.edit_redo()
        except Exception:
            pass
        return "break"
