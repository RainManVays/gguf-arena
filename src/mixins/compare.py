from __future__ import annotations

import tkinter as tk

import customtkinter as ctk

from ..theme import (
    FONT_MONO as _FONT_MONO, FONT_SMALL as _FONT_SMALL, FONT_BOLD as _FONT_BOLD,
    FONT_TINY,
    CLR_CANVAS_BG, CLR_PRIMARY, CLR_TXT_NORMAL, CLR_TXT_MUTED,
)


class CompareMixin:

    def _build_compare_tab(self, results: list[dict]) -> None:
        successful = [r for r in results if r.get("success") and r.get("output")]
        if len(successful) < 2:
            return

        for name in list(self._tabs._tab_dict.keys()):
            if name == "Compare":
                self._tabs.delete(name)

        self._tabs.add("Compare")
        tab = self._tabs.tab("Compare")
        n = len(successful)
        tab.grid_rowconfigure(1, weight=1)
        for i in range(n):
            tab.grid_columnconfigure(i, weight=1)

        for i, r in enumerate(successful):
            lbl = ctk.CTkLabel(tab, text=r["name"], font=_FONT_BOLD, anchor="center",
                               cursor="hand2")
            lbl.grid(row=0, column=i, sticky="ew", padx=(4 if i else 8, 4), pady=(4, 2))
            lbl.bind("<Double-Button-1>",
                     lambda e, l=lbl, n=r["name"]: self._copy_model_name(l, n))
            tb = ctk.CTkTextbox(tab, wrap="word", state="disabled", font=_FONT_MONO)
            tb.grid(row=1, column=i, sticky="nsew", padx=(4 if i else 8, 4), pady=(0, 4))
            tb.configure(state="normal")
            tb.insert("end", r["output"])
            tb.configure(state="disabled")

    def _build_stats_tab(self, results: list[dict]) -> None:
        has_tps = [r for r in results if r.get("tps") is not None]
        if len(has_tps) < 2:
            return

        for name in list(self._tabs._tab_dict.keys()):
            if name == "Stats":
                self._tabs.delete(name)

        self._tabs.add("Stats")
        tab = self._tabs.tab("Stats")
        tab.grid_rowconfigure(0, weight=1)
        tab.grid_columnconfigure(0, weight=1)

        canvas = tk.Canvas(tab, bg=CLR_CANVAS_BG, highlightthickness=0)
        canvas.grid(row=0, column=0, sticky="nsew")

        def _draw(event=None):
            canvas.delete("all")
            w = canvas.winfo_width()
            h = canvas.winfo_height()
            if w < 10 or h < 10:
                return
            pad_left  = 240
            pad_right = 100
            pad_top   = 20
            row_h     = 24  # bar 20px + 4px gap
            max_tps   = max(r["tps"] for r in has_tps)
            bar_area  = max(1, w - pad_left - pad_right)
            for i, r in enumerate(has_tps):
                y = pad_top + i * row_h + row_h // 2
                bar_w = int(r["tps"] / max_tps * bar_area)
                canvas.create_rectangle(
                    pad_left, y - 10, pad_left + bar_w, y + 10,
                    fill=CLR_PRIMARY, outline="")
                label = r["name"][:36]
                canvas.create_text(
                    pad_left - 8, y, text=label, anchor="e",
                    fill=CLR_TXT_NORMAL, font=FONT_TINY)
                canvas.create_text(
                    pad_left + bar_w + 6, y,
                    text=f"{r['tps']:.1f} tok/s", anchor="w",
                    fill=CLR_TXT_MUTED, font=FONT_TINY)

        canvas.bind("<Configure>", _draw)
        canvas.after(50, _draw)
