from __future__ import annotations

import logging

import customtkinter as ctk

from ..storage import delete_history_entry, load_history
from ..theme import (
    FONT_SMALL as _FONT_SMALL, FONT_BOLD as _FONT_BOLD,
    CLR_DANGER, CLR_DANGER_HOV, CLR_TXT_GHOST, CLR_TXT_FAINT,
)

log = logging.getLogger(__name__)


class HistoryMixin:

    def _refresh_history_tab(self) -> None:
        for w in self._history_scroll.winfo_children():
            w.destroy()
        history = load_history()
        if not history:
            ctk.CTkLabel(self._history_scroll, text="No runs yet.",
                         font=_FONT_SMALL, text_color=CLR_TXT_GHOST).pack(
                anchor="w", padx=8, pady=8)
        else:
            for entry in history:
                self._add_history_card(entry)
        self._rebind_scroll_wheel(self._history_scroll)

    def _add_history_card(self, entry: dict) -> None:
        card = ctk.CTkFrame(self._history_scroll)
        card.pack(fill="x", padx=4, pady=3)
        card.grid_columnconfigure(0, weight=1)

        ts = entry.get("timestamp", "")[:19].replace("T", " ")
        models = entry.get("models", [])
        model_summary = "  |  ".join(
            "{} {}".format(
                m["name"],
                "%.1f tok/s" % m["tps"] if m.get("tps") else ("✓" if m.get("success") else "✗"),
            )
            for m in models
        )
        ctk.CTkLabel(card, text=f"{ts}   {model_summary}",
                     font=ctk.CTkFont(size=11, weight="bold"), anchor="w").grid(
            row=0, column=0, sticky="ew", padx=8, pady=(6, 2))

        sys_prev  = entry.get("system_prompt", "").replace("\n", " ")[:70]
        user_prev = entry.get("user_input",    "").replace("\n", " ")[:70]
        ctk.CTkLabel(
            card,
            text=f"SYS: {sys_prev}\nUSR: {user_prev}",
            font=ctk.CTkFont(size=10), text_color=CLR_TXT_FAINT,
            anchor="w", justify="left",
        ).grid(row=1, column=0, sticky="ew", padx=8, pady=(0, 6))

        btn_frame = ctk.CTkFrame(card, fg_color="transparent")
        btn_frame.grid(row=0, column=1, rowspan=2, padx=8)

        ctk.CTkButton(btn_frame, text="Load", width=60, height=26,
                      command=lambda e=entry: self._load_history_entry(e)).pack(pady=(0, 4))
        ctk.CTkButton(btn_frame, text="✕", width=60, height=26,
                      fg_color=CLR_DANGER, hover_color=CLR_DANGER_HOV,
                      command=lambda e=entry: self._on_delete_history(e)).pack()

    def _on_delete_history(self, entry: dict) -> None:
        ts = entry.get("timestamp", "")
        delete_history_entry(ts)
        log.info("History entry deleted: %s", ts)
        self._refresh_history_tab()

    def _load_history_entry(self, entry: dict) -> None:
        self._sys_text.delete("1.0", "end")
        self._sys_text.insert("end", entry.get("system_prompt", ""))
        self._user_text.delete("1.0", "end")
        self._user_text.insert("end", entry.get("user_input", ""))
