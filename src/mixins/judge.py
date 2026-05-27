from __future__ import annotations

import logging
import threading
import tkinter as tk
from pathlib import Path
from tkinter import messagebox

import customtkinter as ctk

from ..storage import save_config

log = logging.getLogger(__name__)

_FONT_MONO  = ("Monospace", 12)
_FONT_SMALL = ("Sans", 11)


class JudgeMixin:

    def _build_judge_tab(self, results: list[dict], system_prompt: str, user_input: str) -> None:
        successful = [r for r in results if r.get("success") and r.get("output")]
        if len(successful) < 2:
            return

        for name in list(self._tabs._tab_dict.keys()):
            if name == "Judge":
                self._tabs.delete(name)

        self._tabs.add("Judge")
        tab = self._tabs.tab("Judge")
        tab.grid_rowconfigure(2, weight=1)
        tab.grid_columnconfigure(0, weight=1)

        self._judge_results_data    = successful
        self._judge_system_prompt   = system_prompt
        self._judge_user_input      = user_input
        self._judge_last_scores     = None
        self._judge_last_winner     = None
        self._judge_last_wins_label = None

        ctrl = ctk.CTkFrame(tab, fg_color="transparent")
        ctrl.grid(row=0, column=0, sticky="ew", padx=8, pady=(8, 4))

        ctk.CTkLabel(ctrl, text="Judge:", font=_FONT_SMALL).pack(side="left", padx=(0, 4))

        folder = Path(self._models_dir_var.get())
        self._judge_model_paths = {}
        display_names: list[str] = []
        for p in self.model_vars:
            f = Path(p)
            try:
                disp = str(f.relative_to(folder)) if self._recursive_var.get() else f.name
            except ValueError:
                disp = f.name
            if disp not in self._judge_model_paths:
                self._judge_model_paths[disp] = p
                display_names.append(disp)

        self._judge_model_combo = ctk.CTkComboBox(
            ctrl, values=display_names, width=260, font=_FONT_SMALL)
        saved_judge = self.cfg.get("judge_model", "")
        saved_path  = self._judge_model_paths.get(saved_judge, "")
        if saved_judge and saved_judge in display_names and Path(saved_path).is_file():
            self._judge_model_combo.set(saved_judge)
        elif display_names:
            self._judge_model_combo.set(display_names[0])
        self._judge_model_combo.pack(side="left", padx=(0, 10))

        saved_mode = self.cfg.get("judge_mode", "Pairwise")
        self._judge_mode_var = ctk.StringVar(value=saved_mode)
        ctk.CTkSegmentedButton(
            ctrl, values=["All at once", "Pairwise"],
            variable=self._judge_mode_var, font=_FONT_SMALL,
        ).pack(side="left", padx=(0, 10))

        self._judge_btn = ctk.CTkButton(
            ctrl, text="⚖ Judge", width=100, height=32,
            fg_color="#4a3a7a", hover_color="#5a4a9a",
            command=self._on_judge_run,
        )
        self._judge_btn.pack(side="left")

        self._judge_status = ctk.CTkLabel(
            ctrl, text="", font=_FONT_SMALL, text_color="#888888")
        self._judge_status.pack(side="left", padx=(10, 0))

        bar_height = len(successful) * 36 + 20
        self._judge_canvas = tk.Canvas(
            tab, bg="#2b2b2b", highlightthickness=0, height=bar_height)
        self._judge_canvas.grid(row=1, column=0, sticky="ew", padx=8, pady=(4, 0))
        self._judge_canvas.bind("<Configure>", lambda _: self._redraw_judge_leaderboard())

        self._judge_log = ctk.CTkTextbox(
            tab, wrap="word", state="disabled", font=_FONT_MONO)
        self._judge_log.grid(row=2, column=0, sticky="nsew", padx=8, pady=(4, 8))

        if self._auto_judge_var.get():
            self.after(0, self._on_judge_run)

    def _redraw_judge_leaderboard(self) -> None:
        if not self._judge_last_scores:
            return
        canvas = self._judge_canvas
        canvas.delete("all")
        w        = canvas.winfo_width() or 600
        scores   = self._judge_last_scores
        winner   = self._judge_last_winner
        names    = list(scores.keys())
        pad_left  = 240
        pad_right = 70
        pad_top   = 10
        row_h     = 36
        bar_area  = max(1, w - pad_left - pad_right)
        for i, name in enumerate(names):
            score = scores[name]
            y     = pad_top + i * row_h + row_h // 2
            bar_w = int(score / 100 * bar_area) if score > 0 else 0
            color = "#c8a000" if name == winner else "#1f6aa5"
            canvas.create_rectangle(
                pad_left, y - 12, pad_left + bar_w, y + 12,
                fill=color, outline="")
            crown = "★ " if name == winner else "  "
            canvas.create_text(
                pad_left - 8, y,
                text=f"{crown}{name[:34]}",
                anchor="e",
                fill="#eeeeee" if name == winner else "#cccccc",
                font=("Sans", 10, "bold" if name == winner else "normal"))
            wins_lbl   = (self._judge_last_wins_label or {}).get(name)
            right_text = wins_lbl if wins_lbl else f"{score}%"
            canvas.create_text(
                pad_left + bar_w + 6, y,
                text=right_text, anchor="w",
                fill="#aaaaaa", font=("Sans", 10))

    def _on_judge_run(self) -> None:
        if self._judge_running:
            return

        judge_name = self._judge_model_combo.get()
        judge_path = self._judge_model_paths.get(judge_name, "")
        if not judge_path or not Path(judge_path).is_file():
            self.cfg.pop("judge_model", None)
            save_config(self.cfg)
            messagebox.showerror("Judge", f"Model file not found:\n{judge_name}", parent=self)
            return

        self.cfg["judge_model"] = judge_name
        self.cfg["judge_mode"]  = self._judge_mode_var.get()
        save_config(self.cfg)

        llama_path    = self._llama_var.get().strip()
        params        = self._collect_params()
        if params is None:
            return

        judge_params = dict(params)
        judge_params["max_tokens"] = max(params["max_tokens"], 512)

        mode          = self._judge_mode_var.get()
        results       = self._judge_results_data
        system_prompt = self._judge_system_prompt
        user_input    = self._judge_user_input
        extra_args    = self._extra_var.get().strip()

        self._judge_running = True
        self._judge_btn.configure(state="disabled", text="Judging…")
        self._judge_status.configure(text="", text_color="#888888")
        self._judge_log.configure(state="normal")
        self._judge_log.delete("1.0", "end")
        self._judge_log.configure(state="disabled")

        def _log(text: str) -> None:
            def _do() -> None:
                self._judge_log.configure(state="normal")
                self._judge_log.insert("end", text + "\n")
                self._judge_log.see("end")
                self._judge_log.configure(state="disabled")
            self.after(0, _do)

        def _show_leaderboard(scores: dict, winner: str | None) -> None:
            self._judge_last_scores = scores
            self._judge_last_winner = winner
            self._redraw_judge_leaderboard()

        def worker() -> None:
            from ..judge import run_judge_all, run_judge_pairwise
            try:
                if mode == "All at once":
                    self.after(0, lambda: self._judge_status.configure(text="Running…"))
                    result = run_judge_all(
                        llama_path=llama_path,
                        judge_model_path=judge_path,
                        system_prompt=system_prompt,
                        user_input=user_input,
                        results=results,
                        params=judge_params,
                        extra_args=extra_args,
                    )
                    if "error" in result:
                        _log(f"✗ {result['error']}\n\nRaw output:\n{result.get('raw', '')}")
                        self.after(0, lambda: self._judge_status.configure(
                            text="✗ Parse error", text_color="#cc4444"))
                    else:
                        raw_scores = result.get("scores", {})
                        pct: dict[str, int] = {}
                        for r in results:
                            n = r["name"]
                            v = raw_scores.get(n, 0)
                            try:
                                pct[n] = int(float(v) / 10 * 100)
                            except (TypeError, ValueError):
                                pct[n] = 0
                        winner    = result.get("winner")
                        reasoning = result.get("reasoning", "")
                        _log(f"Reasoning:\n{reasoning}")
                        self.after(0, lambda s=pct, w=winner: _show_leaderboard(s, w))
                        self.after(0, lambda w=winner: self._judge_status.configure(
                            text=f"Winner: {w}", text_color="#c8a000"))

                else:  # Pairwise
                    total = len(results) * (len(results) - 1) // 2

                    def on_match(idx: int, tot: int, na: str, nb: str,
                                 wname: str | None, reasoning: str) -> None:
                        line = (f"✓ Match {idx+1}/{tot}: {na}  vs  {nb}  →  {wname}"
                                if wname else
                                f"? Match {idx+1}/{tot}: {na}  vs  {nb}  →  unclear")
                        _log(line)
                        self.after(0, lambda i=idx+1, t=tot:
                                   self._judge_status.configure(text=f"Match {i}/{t}"))

                    result = run_judge_pairwise(
                        llama_path=llama_path,
                        judge_model_path=judge_path,
                        system_prompt=system_prompt,
                        user_input=user_input,
                        results=results,
                        params=judge_params,
                        extra_args=extra_args,
                        on_match_done=on_match,
                        stop_flag=lambda: self._stop_requested,
                    )

                    scores   = result.get("scores", {})
                    winner   = result.get("winner")
                    wins     = result.get("wins", {})
                    max_wins = max(1, len(results) - 1)
                    self._judge_last_wins_label = {
                        name: f"{wins.get(name, 0)}/{max_wins} wins"
                        for name in scores
                    }

                    _log("\n─── Reasoning ───")
                    for m in result.get("matches", []):
                        _log(f"  A = {m['a']}  |  B = {m['b']}")
                        if m.get("reasoning"):
                            _log(f"  {m['reasoning']}")
                        _log("")

                    self.after(0, lambda s=scores, w=winner: _show_leaderboard(s, w))
                    self.after(0, lambda w=winner: self._judge_status.configure(
                        text=f"Winner: {w}", text_color="#c8a000"))

            except Exception:
                log.exception("Judge worker exception")
                self.after(0, lambda: self._judge_status.configure(
                    text="Internal error — see log", text_color="#cc4444"))
            finally:
                self._judge_running = False
                self.after(0, lambda: self._judge_btn.configure(
                    state="normal", text="⚖ Judge"))

        threading.Thread(target=worker, daemon=True).start()
