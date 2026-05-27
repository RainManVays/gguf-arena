from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, simpledialog

import customtkinter as ctk

from ..runner import _estimate_tokens
from ..storage import (REGISTRY_DIR, _slug, delete_from_registry,
                       load_registry, save_to_registry)


class PromptsMixin:

    def _update_token_counter(self, event=None) -> None:
        sys_tok  = _estimate_tokens(self._sys_text.get("1.0", "end"))
        user_tok = _estimate_tokens(self._user_text.get("1.0", "end"))
        total    = sys_tok + user_tok
        try:
            ctx = int(self._ctx_var.get())
        except ValueError:
            ctx = 0
        ratio = total / ctx if ctx > 0 else 0
        color = "#44aa44" if ratio < 0.8 else ("#ddaa00" if ratio <= 1.0 else "#cc4444")
        ctx_str = f" / {ctx}" if ctx > 0 else ""
        self._tok_label.configure(
            text=f"SYS: ~{sys_tok}  USR: ~{user_tok}  Total: ~{total}{ctx_str} tok",
            text_color=color,
        )

    def _prompt_names(self) -> list[str]:
        return [e["name"] for e in load_registry()] or ["—"]

    def _on_load_prompt(self, name: str) -> None:
        for e in load_registry():
            if e["name"] == name:
                self._sys_text.delete("1.0", "end")
                self._sys_text.insert("end", e.get("system", ""))
                self._user_text.delete("1.0", "end")
                self._user_text.insert("end", e.get("user", ""))
                break

    def _save_prompt(self) -> None:
        current = self._prompt_combo.get()
        default = current if current and current != "—" else ""
        name = simpledialog.askstring(
            "Save Prompt", "Prompt name:", initialvalue=default, parent=self)
        if not name:
            return
        system = self._sys_text.get("1.0", "end").strip()
        user   = self._user_text.get("1.0", "end").strip()
        if (REGISTRY_DIR / f"{_slug(name)}.yaml").exists():
            if not messagebox.askyesno("Overwrite?", f'Overwrite prompt "{name}"?',
                                       parent=self):
                return
        save_to_registry(name, system, user)
        names = self._prompt_names()
        self._prompt_combo.configure(values=names)
        self._prompt_combo.set(name)

    def _delete_prompt(self) -> None:
        name = self._prompt_combo.get()
        if not name or name == "—":
            return
        delete_from_registry(name)
        names = self._prompt_names()
        self._prompt_combo.configure(values=names)
        self._prompt_combo.set(names[0])

    def _bind_text_keys(self, widget: ctk.CTkTextbox) -> None:
        widget.bind("<Control-z>", self._undo_text)
        widget.bind("<Control-y>", self._redo_text)
        widget.bind("<Control-Z>", self._redo_text)  # Ctrl+Shift+Z

    def _on_ctrl_a(self, event) -> str:
        w = event.widget
        if isinstance(w, tk.Entry):
            w.select_range(0, "end")
            w.icursor("end")
        elif isinstance(w, tk.Text):
            w.tag_add("sel", "1.0", "end")
        return "break"

    def _undo_text(self, event) -> str:
        try:
            event.widget.edit_undo()
        except Exception:
            pass
        return "break"

    def _redo_text(self, event) -> str:
        try:
            event.widget.edit_redo()
        except Exception:
            pass
        return "break"
