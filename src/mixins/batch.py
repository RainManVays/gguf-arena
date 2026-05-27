from __future__ import annotations

import logging
import threading
import tkinter as tk
from pathlib import Path
from tkinter import messagebox

import customtkinter as ctk

from ..runner import run_model, truncate_user_input
from ..storage import save_config
from ..theme import (
    FONT_MONO as _FONT_MONO, FONT_SMALL as _FONT_SMALL, FONT_BOLD as _FONT_BOLD,
    FONT_MONO_SM, FONT_TINY,
    CLR_JUDGE, CLR_JUDGE_HOV,
    CLR_CANVAS_BG, CLR_WIN, CLR_ERR,
    CLR_TXT_DIM, CLR_TXT_MUTED, CLR_TXT_NORMAL, CLR_ERR_TEXT,
)

log = logging.getLogger(__name__)


class BatchMixin:

    def _run_batch_worker(self, llama_path: str, selected: list,
                          system_prompt: str, params: dict,
                          chat_mode: bool, extra_args: str) -> None:
        self._batch_run_results = []
        cases = self._batch_cases or []
        total_cases  = len(cases)
        total_models = len(selected)

        try:
            for ci, case in enumerate(cases):
                if self._stop_requested:
                    break

                try:
                    ui, _ = truncate_user_input(
                        system_prompt, case["input"],
                        params["ctx_size"], params["max_tokens"])
                except ValueError:
                    ui = case["input"]

                case_result: dict = {
                    "case_id":    case["id"],
                    "user_input": case["input"],
                    "expected":   case.get("expected"),
                    "models":     [],
                }

                for mi, (path, _) in enumerate(selected):
                    if self._stop_requested:
                        break
                    name = Path(path).name
                    txt = f"Case {ci+1}/{total_cases} · {name} [{mi+1}/{total_models}]"
                    self.after(0, lambda t=txt: self._status_lbl.configure(text=t))

                    def _on_proc(proc, n=name) -> None:
                        self._current_proc = proc
                        log.debug("Process started  model=%s  pid=%d", n, proc.pid)

                    res = run_model(
                        llama_path=llama_path,
                        model_path=path,
                        system_prompt=system_prompt,
                        user_input=ui,
                        params=params,
                        chat_mode=chat_mode,
                        extra_args=extra_args,
                        proc_started=_on_proc,
                    )
                    case_result["models"].append({
                        "name":     name,
                        "success":  res.get("success", False),
                        "output":   res.get("output", ""),
                        "elapsed":  res.get("elapsed", 0),
                        "tps":      res.get("tps"),
                        "n_tokens": res.get("n_tokens"),
                    })

                self._batch_run_results.append(case_result)

            if self._batch_run_results:
                agg: dict[str, list] = {}
                for cr in self._batch_run_results:
                    for mr in cr["models"]:
                        n = mr["name"]
                        agg.setdefault(n, [])
                        if mr.get("tps"):
                            agg[n].append(mr["tps"])
                agg_results = [
                    {"name": n, "tps": sum(vals) / len(vals) if vals else None}
                    for n, vals in agg.items()
                ]
                self.after(0, lambda r=agg_results: self._build_stats_tab(r))
                self.after(0, self._build_batch_results_tab)
                self.after(0, self._build_batch_judge_tab)
                self.after(0, lambda: self._export_btn.configure(state="normal"))

            done_n  = len(self._batch_run_results)
            stopped = self._stop_requested
            txt = (f"Stopped ({done_n}/{total_cases} cases)" if stopped
                   else f"Done (batch: {done_n} × {total_models})")
            self.after(0, lambda t=txt: self._status_lbl.configure(text=t))

        except Exception:
            log.exception("Batch worker exception")
            self.after(0, lambda: self._status_lbl.configure(text="Internal error — see log"))
        finally:
            self._running        = False
            self._stop_requested = False
            self._current_proc   = None
            self.after(0, lambda: self._set_running(False))

    def _build_batch_results_tab(self) -> None:
        for name in list(self._tabs._tab_dict.keys()):
            if name == "Batch":
                self._tabs.delete(name)
        if not self._batch_run_results:
            return

        self._tabs.add("Batch")
        tab = self._tabs.tab("Batch")
        tab.grid_rowconfigure(1, weight=1)
        tab.grid_columnconfigure(0, weight=1)

        toolbar = ctk.CTkFrame(tab, fg_color="transparent")
        toolbar.grid(row=0, column=0, sticky="ew", padx=8, pady=(4, 2))
        ctk.CTkButton(toolbar, text="💾 Save YAML", width=120, height=28,
                      font=_FONT_SMALL,
                      command=self._on_save_batch_yaml).pack(side="left")

        scroll = ctk.CTkScrollableFrame(tab)
        scroll.grid(row=1, column=0, sticky="nsew")
        self._setup_scroll_wheel(scroll)

        model_names = [m["name"] for m in self._batch_run_results[0]["models"]]

        ctk.CTkLabel(scroll, text="Case / Input", font=_FONT_BOLD,
                     anchor="w", width=160).grid(row=0, column=0, padx=4, pady=2, sticky="ew")
        for j, mname in enumerate(model_names):
            ctk.CTkLabel(scroll, text=mname[:28], font=_FONT_BOLD,
                         anchor="w", width=220).grid(row=0, column=j+1, padx=4, pady=2, sticky="ew")

        for i, cr in enumerate(self._batch_run_results):
            row = i + 1
            inp = cr["user_input"]
            preview = inp[:55].replace("\n", " ")
            if len(inp) > 55:
                preview += "…"
            cell_text = f"{cr['case_id']}\n{preview}"
            exp = cr.get("expected")
            if exp:
                exp_preview = exp[:45].replace("\n", " ")
                cell_text += f"\n✓ {exp_preview}{'…' if len(exp) > 45 else ''}"
            ctk.CTkLabel(scroll, text=cell_text,
                         font=FONT_TINY, anchor="nw", width=160,
                         justify="left").grid(row=row, column=0, padx=4, pady=2, sticky="nw")

            for j, mr in enumerate(cr["models"]):
                if mr.get("success") and mr.get("output"):
                    out = mr["output"][:110].replace("\n", " ")
                    tps_s = f"  [{mr['tps']:.1f} t/s]" if mr.get("tps") else ""
                    cell_text = out + tps_s
                    color = CLR_TXT_NORMAL
                else:
                    cell_text = f"✗ {mr.get('output','')[:60]}"
                    color = CLR_ERR_TEXT
                ctk.CTkLabel(scroll, text=cell_text, font=FONT_MONO_SM,
                             anchor="nw", width=220, justify="left",
                             wraplength=215, text_color=color).grid(
                    row=row, column=j+1, padx=4, pady=2, sticky="nw")

        self._rebind_scroll_wheel(scroll)
        self._tabs.set("Batch")

    def _build_batch_judge_tab(self) -> None:
        for name in list(self._tabs._tab_dict.keys()):
            if name == "Judge":
                self._tabs.delete(name)
        if not self._batch_run_results:
            return

        first = self._batch_run_results[0]
        ok_models = [m for m in first["models"] if m.get("success") and m.get("output")]
        if len(ok_models) < 2:
            return

        self._tabs.add("Judge")
        tab = self._tabs.tab("Judge")
        tab.grid_rowconfigure(2, weight=1)
        tab.grid_columnconfigure(0, weight=1)

        self._judge_system_prompt   = self._sys_text.get("1.0", "end").strip()
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
        saved = self.cfg.get("judge_model", "")
        saved_path = self._judge_model_paths.get(saved, "")
        if saved and saved in display_names and Path(saved_path).is_file():
            self._judge_model_combo.set(saved)
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
            ctrl, text="⚖ Judge All Cases", width=150, height=32,
            fg_color=CLR_JUDGE, hover_color=CLR_JUDGE_HOV,
            command=self._on_batch_judge_run,
        )
        self._judge_btn.pack(side="left")

        self._judge_status = ctk.CTkLabel(
            ctrl, text="", font=_FONT_SMALL, text_color=CLR_TXT_DIM)
        self._judge_status.pack(side="left", padx=(10, 0))

        bar_height = len(ok_models) * 36 + 20
        self._judge_canvas = tk.Canvas(
            tab, bg=CLR_CANVAS_BG, highlightthickness=0, height=bar_height)
        self._judge_canvas.grid(row=1, column=0, sticky="ew", padx=8, pady=(4, 0))
        self._judge_canvas.bind("<Configure>", lambda _: self._redraw_judge_leaderboard())

        self._judge_log = ctk.CTkTextbox(
            tab, wrap="word", state="disabled", font=_FONT_MONO)
        self._judge_log.grid(row=2, column=0, sticky="nsew", padx=8, pady=(4, 8))

    def _on_batch_judge_run(self) -> None:
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
        batch_results = self._batch_run_results
        system_prompt = self._judge_system_prompt
        extra_args    = self._extra_var.get().strip()
        total_cases   = len(batch_results)

        self._judge_running = True
        self._judge_btn.configure(state="disabled", text="Judging…")
        self._judge_status.configure(text="", text_color=CLR_TXT_DIM)
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

        def worker() -> None:
            from ..judge import run_judge_all, run_judge_pairwise
            total_wins: dict[str, int] = {}
            try:
                for ci, case in enumerate(batch_results):
                    if self._stop_requested:
                        break

                    ok = [m for m in case["models"] if m.get("success") and m.get("output")]
                    if len(ok) < 2:
                        continue

                    self.after(0, lambda i=ci: self._judge_status.configure(
                        text=f"Case {i+1}/{total_cases}"))
                    _log(f"\n── Case {ci+1}/{total_cases}: {case['case_id']} ──")
                    inp = case["user_input"]
                    _log(f"   {inp[:80]}{'…' if len(inp) > 80 else ''}")

                    if mode == "All at once":
                        result = run_judge_all(
                            llama_path=llama_path,
                            judge_model_path=judge_path,
                            system_prompt=system_prompt,
                            user_input=case["user_input"],
                            results=ok,
                            params=judge_params,
                            extra_args=extra_args,
                        )
                        winner = result.get("winner")
                    else:
                        result = run_judge_pairwise(
                            llama_path=llama_path,
                            judge_model_path=judge_path,
                            system_prompt=system_prompt,
                            user_input=case["user_input"],
                            results=ok,
                            params=judge_params,
                            extra_args=extra_args,
                            on_match_done=lambda *_: None,
                            stop_flag=lambda: self._stop_requested,
                        )
                        winner = result.get("winner")

                    if winner:
                        total_wins[winner] = total_wins.get(winner, 0) + 1
                        _log(f"   → Winner: {winner}")
                    else:
                        _log("   → Unclear")

                if total_wins:
                    all_names = [m["name"] for m in batch_results[0]["models"]
                                 if m.get("success") and m.get("output")]
                    for n in all_names:
                        total_wins.setdefault(n, 0)

                    max_w = max(total_wins.values()) or 1
                    scores = {n: int(w / max_w * 100) for n, w in total_wins.items()}
                    winner = max(total_wins, key=total_wins.get)
                    wins_label = {n: f"{w}/{total_cases} wins" for n, w in total_wins.items()}
                    self._judge_last_wins_label = wins_label

                    def _update_lb(s=scores, w=winner) -> None:
                        self._judge_last_scores = s
                        self._judge_last_winner = w
                        self._redraw_judge_leaderboard()
                    self.after(0, _update_lb)
                    self.after(0, lambda w=winner: self._judge_status.configure(
                        text=f"Winner: {w} ({total_wins[w]}/{total_cases} cases)",
                        text_color=CLR_WIN))

                    _log("\n── Leaderboard ──")
                    for n, w in sorted(total_wins.items(), key=lambda x: -x[1]):
                        _log(f"   {n}: {w}/{total_cases} wins")

            except Exception:
                log.exception("Batch judge worker exception")
                self.after(0, lambda: self._judge_status.configure(
                    text="Internal error — see log", text_color=CLR_ERR))
            finally:
                self._judge_running = False
                self.after(0, lambda: self._judge_btn.configure(
                    state="normal", text="⚖ Judge All Cases"))

        threading.Thread(target=worker, daemon=True).start()
