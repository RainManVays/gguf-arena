from __future__ import annotations

import logging
import subprocess
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox

import customtkinter as ctk

from .mixins.compare import CompareMixin
from .mixins.export import ExportMixin
from .mixins.history import HistoryMixin
from .mixins.models_panel import ModelsPanelMixin
from .mixins.prompts import PromptsMixin
from .mixins.run import RunMixin
from .runner import run_model, truncate_user_input
from .mdrender import render as _md_render
from .storage import (append_history, load_config, load_registry,
                      make_history_entry, migrate_prompts_to_registry, save_config)

log = logging.getLogger(__name__)

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

_FONT_MONO  = ("Monospace", 12)
_FONT_SMALL = ("Sans", 11)
_FONT_BOLD  = ("Sans", 12, "bold")


# ──────────────────────────────────────────────────────────────────────────────

class App(HistoryMixin, CompareMixin, ExportMixin, PromptsMixin, ModelsPanelMixin, RunMixin, ctk.CTk):
    def __init__(self) -> None:
        super().__init__()
        self.title("GGUF Arena")
        self.geometry("1440x900")
        self.minsize(1000, 680)

        self.cfg = load_config()
        migrate_prompts_to_registry(self.cfg)
        self.model_vars: dict[str, ctk.BooleanVar] = {}

        # state for run / stop
        self._running        = False
        self._stop_requested = False
        self._current_proc: subprocess.Popen | None = None
        self._last_run: dict | None = None
        self._model_widgets: dict = {}
        self._raw_outputs:   dict[str, str]  = {}
        self._md_mode:       dict[str, bool] = {}

        self._judge_running = False
        self._judge_results_data: list[dict] = []
        self._judge_model_paths: dict[str, str] = {}
        self._judge_system_prompt = ""
        self._judge_user_input = ""
        self._judge_last_scores: dict | None = None
        self._judge_last_winner: str | None = None
        self._judge_last_wins_label: dict | None = None  # name → "W/N wins" for pairwise

        # batch state
        self._batch_cases: list[dict] | None = None
        self._batch_mode = False
        self._batch_run_results: list[dict] = []
        self._batch_source_path: str = ""

        log.info("App started")
        self._build_ui()
        self._load_model_list()
        self._refresh_history_tab()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ── helpers ───────────────────────────────────────────────────────────────

    def _ui(self, func, **kwargs) -> None:
        """Schedule a widget configure call safely from any thread."""
        self.after(0, lambda: func(**kwargs))

    # ── Layout ────────────────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self._left = ctk.CTkFrame(self, width=290, corner_radius=0)
        self._left.grid(row=0, column=0, sticky="nsew")
        self._left.grid_propagate(False)
        self._left.grid_columnconfigure(0, weight=1)

        self._right = ctk.CTkFrame(self, corner_radius=0, fg_color="transparent")
        self._right.grid(row=0, column=1, sticky="nsew")
        self._right.grid_rowconfigure(1, weight=1)
        self._right.grid_columnconfigure(0, weight=1)

        self._build_left()
        self._build_right()
        self.bind_class("Entry", "<Control-a>", self._on_ctrl_a)
        self.bind_class("Text",  "<Control-a>", self._on_ctrl_a)

    # ── Left panel ────────────────────────────────────────────────────────────

    def _build_left(self) -> None:
        p = self._left
        p.grid_rowconfigure(2, weight=1)

        ctk.CTkLabel(p, text="llama-cli binary", font=_FONT_BOLD).grid(
            row=0, column=0, sticky="w", padx=10, pady=(12, 2))

        row0 = ctk.CTkFrame(p, fg_color="transparent")
        row0.grid(row=1, column=0, sticky="ew", padx=10, pady=2)
        row0.grid_columnconfigure(0, weight=1)

        self._llama_var = ctk.StringVar(value=self.cfg.get("llama_path", ""))
        ctk.CTkEntry(row0, textvariable=self._llama_var, font=_FONT_SMALL).grid(
            row=0, column=0, sticky="ew", padx=(0, 4))
        ctk.CTkButton(row0, text="…", width=32,
                      command=self._browse_llama).grid(row=0, column=1)

        self._models_tabview = ctk.CTkTabview(p)
        self._models_tabview.grid(row=2, column=0, sticky="nsew", padx=6, pady=4)
        self._models_tabview.add("Local")
        self._models_tabview.add("HF Download")

        self._build_local_tab()
        self._build_hf_download_tab()

        param_box = ctk.CTkFrame(p)
        param_box.grid(row=3, column=0, sticky="ew", padx=8, pady=(4, 10))
        param_box.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(param_box, text="Parameters", font=_FONT_BOLD).grid(
            row=0, column=0, columnspan=2, sticky="w", padx=8, pady=(6, 4))

        params = self.cfg.get("params", {})
        self._temp_var    = ctk.StringVar(value=str(params.get("temperature", 0.8)))
        self._topp_var    = ctk.StringVar(value=str(params.get("top_p", 0.9)))
        self._maxtok_var  = ctk.StringVar(value=str(params.get("max_tokens", 512)))
        self._ctx_var     = ctk.StringVar(value=str(params.get("ctx_size", 8192)))
        self._threads_var = ctk.StringVar(value=str(params.get("threads", -1)))
        self._gpu_var     = ctk.StringVar(value=str(params.get("gpu_layers", -1)))
        self._seed_var    = ctk.StringVar(value=str(params.get("seed", -1)))
        self._topk_var    = ctk.StringVar(value=str(params.get("top_k", 40)))
        self._rep_var     = ctk.StringVar(value=str(params.get("repeat_penalty", 1.1)))

        for i, (lbl, var) in enumerate([
            ("temp",       self._temp_var),
            ("top_p",      self._topp_var),
            ("max_tokens", self._maxtok_var),
            ("ctx_size",   self._ctx_var),
            ("threads",    self._threads_var),
            ("gpu_layers", self._gpu_var),
            ("seed",       self._seed_var),
            ("top_k",      self._topk_var),
            ("rep_pen",    self._rep_var),
        ]):
            ctk.CTkLabel(param_box, text=lbl, font=_FONT_SMALL).grid(
                row=i + 1, column=0, sticky="w", padx=(8, 4), pady=2)
            ctk.CTkEntry(param_box, textvariable=var, width=80, height=28,
                         font=_FONT_SMALL).grid(
                row=i + 1, column=1, sticky="ew", padx=(0, 8), pady=2)

        ctk.CTkLabel(param_box, text="(−1 = auto)", font=("Sans", 10),
                     text_color="#777777").grid(
            row=7, column=0, columnspan=2, padx=8, pady=(0, 6))

    def _build_local_tab(self) -> None:
        tab = self._models_tabview.tab("Local")
        tab.grid_rowconfigure(2, weight=1)
        tab.grid_columnconfigure(0, weight=1)

        row1 = ctk.CTkFrame(tab, fg_color="transparent")
        row1.grid(row=0, column=0, sticky="ew", padx=4, pady=(4, 2))
        row1.grid_columnconfigure(0, weight=1)

        self._models_dir_var = ctk.StringVar(value=self.cfg.get("models_dir", ""))
        ctk.CTkEntry(row1, textvariable=self._models_dir_var, font=_FONT_SMALL).grid(
            row=0, column=0, sticky="ew", padx=(0, 4))
        ctk.CTkButton(row1, text="…", width=32,
                      command=self._browse_models_dir).grid(row=0, column=1)

        scan_row = ctk.CTkFrame(tab, fg_color="transparent")
        scan_row.grid(row=1, column=0, sticky="ew", padx=4, pady=2)
        scan_row.grid_columnconfigure(1, weight=1)

        self._recursive_var = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(scan_row, text="Subdirs", variable=self._recursive_var,
                        font=_FONT_SMALL, width=80,
                        command=self._load_model_list).grid(row=0, column=0)
        ctk.CTkButton(scan_row, text="↺", height=26, width=32,
                      command=self._load_model_list).grid(row=0, column=1, padx=(4, 0), sticky="w")
        ctk.CTkButton(scan_row, text="+ Add", height=26,
                      command=self._add_model_file).grid(row=0, column=2, padx=(4, 0))

        self._models_scroll = ctk.CTkScrollableFrame(tab, label_text="Models")
        self._models_scroll.grid(row=2, column=0, sticky="nsew", padx=2, pady=2)
        self._models_scroll.grid_columnconfigure(0, weight=1)

        sel_row = ctk.CTkFrame(tab, fg_color="transparent")
        sel_row.grid(row=3, column=0, padx=4, pady=(2, 4), sticky="ew")
        sel_row.grid_columnconfigure((0, 1), weight=1)
        ctk.CTkButton(sel_row, text="All", height=26,
                      command=self._select_all).grid(row=0, column=0, padx=(0, 3), sticky="ew")
        ctk.CTkButton(sel_row, text="None", height=26,
                      command=self._select_none).grid(row=0, column=1, padx=(3, 0), sticky="ew")

    def _build_hf_download_tab(self) -> None:
        tab = self._models_tabview.tab("HF Download")
        tab.grid_rowconfigure(2, weight=1)
        tab.grid_columnconfigure(0, weight=1)

        form = ctk.CTkFrame(tab, fg_color="transparent")
        form.grid(row=0, column=0, sticky="ew", padx=4, pady=(4, 2))
        form.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(form, text="Model ID", font=_FONT_SMALL).pack(anchor="w")
        self._hf_repo_var = ctk.StringVar()
        ctk.CTkEntry(form, textvariable=self._hf_repo_var,
                     placeholder_text="org/model-name",
                     font=_FONT_SMALL).pack(fill="x", pady=(0, 6))

        ctk.CTkLabel(form, text="Save to", font=_FONT_SMALL).pack(anchor="w")
        dir_row = ctk.CTkFrame(form, fg_color="transparent")
        dir_row.pack(fill="x")
        dir_row.grid_columnconfigure(0, weight=1)
        self._nlp_dir_var = ctk.StringVar(value=self.cfg.get("nlp_models_dir", ""))
        ctk.CTkEntry(dir_row, textvariable=self._nlp_dir_var,
                     font=_FONT_SMALL).grid(row=0, column=0, sticky="ew", padx=(0, 4))
        ctk.CTkButton(dir_row, text="…", width=32,
                      command=self._browse_nlp_dir).grid(row=0, column=1)

        self._hf_download_btn = ctk.CTkButton(
            tab, text="↓ Download", height=30,
            fg_color="#1e6e1e", hover_color="#278a27",
            command=self._on_hf_download,
        )
        self._hf_download_btn.grid(row=1, column=0, padx=4, pady=(6, 4), sticky="ew")

        self._hf_log = ctk.CTkTextbox(tab, state="disabled", font=_FONT_MONO, wrap="word")
        self._hf_log.grid(row=2, column=0, sticky="nsew", padx=2, pady=(0, 4))

    # ── Right panel ───────────────────────────────────────────────────────────

    def _build_right(self) -> None:
        top = ctk.CTkFrame(self._right, fg_color="transparent")
        top.grid(row=0, column=0, sticky="ew", padx=8, pady=(8, 4))
        top.grid_columnconfigure(0, weight=1)

        # System prompt header
        sp_hdr = ctk.CTkFrame(top, fg_color="transparent")
        sp_hdr.grid(row=0, column=0, sticky="ew")
        sp_hdr.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(sp_hdr, text="System Prompt", font=_FONT_BOLD).grid(
            row=0, column=0, sticky="w")
        self._prompt_combo = ctk.CTkComboBox(
            sp_hdr, values=self._prompt_names(), width=200,
            font=_FONT_SMALL, command=self._on_load_prompt)
        self._prompt_combo.grid(row=0, column=1, padx=(0, 4))
        ctk.CTkButton(sp_hdr, text="Save", width=55, height=28,
                      command=self._save_prompt).grid(row=0, column=2, padx=2)
        ctk.CTkButton(sp_hdr, text="Del", width=40, height=28,
                      fg_color="#7a2222", hover_color="#992222",
                      command=self._delete_prompt).grid(row=0, column=3)

        self._sys_text = ctk.CTkTextbox(top, height=85, wrap="word", font=_FONT_MONO,
                                        undo=True)
        self._sys_text.grid(row=1, column=0, sticky="ew", pady=(2, 6))
        self._bind_text_keys(self._sys_text)
        self._sys_text.bind("<KeyRelease>", self._update_token_counter)
        self._ctx_var.trace_add("write", lambda *_: self._update_token_counter())

        user_hdr = ctk.CTkFrame(top, fg_color="transparent")
        user_hdr.grid(row=2, column=0, sticky="ew")
        user_hdr.grid_columnconfigure(2, weight=1)

        self._single_btn = ctk.CTkButton(
            user_hdr, text="User Input", width=110, height=28,
            font=_FONT_BOLD,
            fg_color="#1f6aa5", hover_color="#1a5a8a",
            command=self._on_toggle_single)
        self._single_btn.grid(row=0, column=0, sticky="w", padx=(0, 8))

        ctk.CTkButton(
            user_hdr, text="Load Batch…", width=100, height=28,
            font=_FONT_SMALL,
            command=self._on_batch_load).grid(row=0, column=1)

        self._batch_info_lbl = ctk.CTkLabel(
            user_hdr, text="", font=("Sans", 10), text_color="#888888", anchor="w")
        self._batch_info_lbl.grid(row=0, column=2, sticky="w", padx=(8, 0))

        self._batch_clear_btn = ctk.CTkButton(
            user_hdr, text="✕", width=28, height=28,
            fg_color="#5a1a1a", hover_color="#7a2222",
            command=self._on_batch_clear)

        self._user_text = ctk.CTkTextbox(top, height=110, wrap="word", font=_FONT_MONO,
                                         undo=True)
        self._user_text.grid(row=3, column=0, sticky="ew", pady=(2, 2))
        self._bind_text_keys(self._user_text)
        self._user_text.bind("<KeyRelease>", self._update_token_counter)

        reg = load_registry()
        if reg:
            self._sys_text.insert("end", reg[0].get("system", ""))
            self._user_text.insert("end", reg[0].get("user", ""))
            self._prompt_combo.set(reg[0]["name"])

        self._tok_label = ctk.CTkLabel(top, text="", font=("Sans", 10),
                                       text_color="#666666", anchor="w")
        self._tok_label.grid(row=4, column=0, sticky="w", pady=(0, 4))

        # Controls row: Run | Stop | chat mode | extra args | status
        ctrl = ctk.CTkFrame(top, fg_color="transparent")
        ctrl.grid(row=5, column=0, sticky="ew")

        self._run_btn = ctk.CTkButton(
            ctrl, text="▶  Run", width=110, height=36,
            fg_color="#1e6e1e", hover_color="#278a27",
            command=self._on_run)
        self._run_btn.grid(row=0, column=0, padx=(0, 6))

        self._stop_btn = ctk.CTkButton(
            ctrl, text="■  Stop", width=90, height=36,
            fg_color="#7a2222", hover_color="#992222",
            command=self._on_stop, state="disabled")
        self._stop_btn.grid(row=0, column=1, padx=(0, 12))

        self._chat_mode_var = ctk.BooleanVar(value=self.cfg.get("chat_mode", True))
        ctk.CTkCheckBox(ctrl, text="Chat mode", variable=self._chat_mode_var,
                        font=_FONT_SMALL).grid(row=0, column=2, padx=(0, 14))

        self._auto_judge_var = ctk.BooleanVar(value=self.cfg.get("auto_judge", False))
        ctk.CTkCheckBox(ctrl, text="Run judge", variable=self._auto_judge_var,
                        font=_FONT_SMALL).grid(row=0, column=3, padx=(0, 14))

        ctk.CTkLabel(ctrl, text="Extra args:", font=_FONT_SMALL).grid(
            row=0, column=4, padx=(0, 4))
        self._extra_var = ctk.StringVar(value=self.cfg.get("extra_args", ""))
        ctk.CTkEntry(ctrl, textvariable=self._extra_var, width=220,
                     height=32, font=_FONT_SMALL).grid(row=0, column=5, padx=(0, 14))

        self._status_lbl = ctk.CTkLabel(ctrl, text="", font=_FONT_SMALL,
                                         text_color="#888888")
        self._status_lbl.grid(row=0, column=6, padx=(0, 10))

        self._export_btn = ctk.CTkButton(
            ctrl, text="Export…", width=80, height=32,
            state="disabled", command=self._on_export)
        self._export_btn.grid(row=0, column=7)

        # Results tabs
        self._tabs = ctk.CTkTabview(self._right)
        self._tabs.grid(row=1, column=0, sticky="nsew", padx=8, pady=(0, 8))

        self._tabs.add("History")
        self._history_scroll = ctk.CTkScrollableFrame(self._tabs.tab("History"))
        self._history_scroll.pack(fill="both", expand=True)


    # ── Batch worker & results ────────────────────────────────────────────────

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
                    "case_id":  case["id"],
                    "user_input": case["input"],
                    "expected": case.get("expected"),
                    "models":   [],
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
                # aggregate tok/s per model for Stats tab
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

        model_names = [m["name"] for m in self._batch_run_results[0]["models"]]

        # Header
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
                         font=("Sans", 10), anchor="nw", width=160,
                         justify="left").grid(row=row, column=0, padx=4, pady=2, sticky="nw")

            for j, mr in enumerate(cr["models"]):
                if mr.get("success") and mr.get("output"):
                    out = mr["output"][:110].replace("\n", " ")
                    tps_s = f"  [{mr['tps']:.1f} t/s]" if mr.get("tps") else ""
                    cell_text = out + tps_s
                    color = "#cccccc"
                else:
                    cell_text = f"✗ {mr.get('output','')[:60]}"
                    color = "#cc6666"
                ctk.CTkLabel(scroll, text=cell_text, font=("Monospace", 10),
                             anchor="nw", width=220, justify="left",
                             wraplength=215, text_color=color).grid(
                    row=row, column=j+1, padx=4, pady=2, sticky="nw")

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

        self._judge_system_prompt    = self._sys_text.get("1.0", "end").strip()
        self._judge_last_scores      = None
        self._judge_last_winner      = None
        self._judge_last_wins_label  = None

        # Controls
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
            fg_color="#4a3a7a", hover_color="#5a4a9a",
            command=self._on_batch_judge_run,
        )
        self._judge_btn.pack(side="left")

        self._judge_status = ctk.CTkLabel(
            ctrl, text="", font=_FONT_SMALL, text_color="#888888")
        self._judge_status.pack(side="left", padx=(10, 0))

        # Leaderboard canvas (sized by number of models)
        bar_height = len(ok_models) * 36 + 20
        self._judge_canvas = tk.Canvas(
            tab, bg="#2b2b2b", highlightthickness=0, height=bar_height)
        self._judge_canvas.grid(row=1, column=0, sticky="ew", padx=8, pady=(4, 0))
        self._judge_canvas.bind("<Configure>", lambda _: self._redraw_judge_leaderboard())

        # Log / per-case results
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

        llama_path = self._llama_var.get().strip()
        params = self._collect_params()
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

        def worker() -> None:
            from .judge import run_judge_all, run_judge_pairwise
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
                        text_color="#c8a000"))

                    _log("\n── Leaderboard ──")
                    for n, w in sorted(total_wins.items(), key=lambda x: -x[1]):
                        _log(f"   {n}: {w}/{total_cases} wins")

            except Exception:
                log.exception("Batch judge worker exception")
                self.after(0, lambda: self._judge_status.configure(
                    text="Internal error — see log", text_color="#cc4444"))
            finally:
                self._judge_running = False
                self.after(0, lambda: self._judge_btn.configure(
                    state="normal", text="⚖ Judge All Cases"))

        threading.Thread(target=worker, daemon=True).start()

    # ── Judge tab ─────────────────────────────────────────────────────────────

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

        self._judge_results_data  = successful
        self._judge_system_prompt = system_prompt
        self._judge_user_input    = user_input
        self._judge_last_scores      = None
        self._judge_last_winner      = None
        self._judge_last_wins_label  = None

        # Controls row
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
        saved_path = self._judge_model_paths.get(saved_judge, "")
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

        # Leaderboard canvas
        bar_height = len(successful) * 36 + 20
        self._judge_canvas = tk.Canvas(
            tab, bg="#2b2b2b", highlightthickness=0, height=bar_height)
        self._judge_canvas.grid(row=1, column=0, sticky="ew", padx=8, pady=(4, 0))
        self._judge_canvas.bind("<Configure>", lambda _: self._redraw_judge_leaderboard())

        # Log / reasoning textbox
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
        w = canvas.winfo_width() or 600
        scores = self._judge_last_scores
        winner = self._judge_last_winner
        names  = list(scores.keys())
        pad_left  = 240
        pad_right = 70
        pad_top   = 10
        row_h     = 36
        bar_area  = max(1, w - pad_left - pad_right)
        for i, name in enumerate(names):
            score = scores[name]
            y = pad_top + i * row_h + row_h // 2
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
            wins_lbl = (self._judge_last_wins_label or {}).get(name)
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

        llama_path = self._llama_var.get().strip()
        params = self._collect_params()
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
            from .judge import run_judge_all, run_judge_pairwise
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
                        # Convert 1-10 scores to percentages
                        pct: dict[str, int] = {}
                        for r in results:
                            n = r["name"]
                            v = raw_scores.get(n, 0)
                            try:
                                pct[n] = int(float(v) / 10 * 100)
                            except (TypeError, ValueError):
                                pct[n] = 0
                        winner   = result.get("winner")
                        reasoning = result.get("reasoning", "")
                        _log(f"Reasoning:\n{reasoning}")
                        self.after(0, lambda s=pct, w=winner: _show_leaderboard(s, w))
                        self.after(0, lambda w=winner: self._judge_status.configure(
                            text=f"Winner: {w}", text_color="#c8a000"))

                else:  # Pairwise
                    total = len(results) * (len(results) - 1) // 2

                    def on_match(idx: int, tot: int, na: str, nb: str,
                                 wname: str | None, reasoning: str) -> None:
                        if wname:
                            line = f"✓ Match {idx+1}/{tot}: {na}  vs  {nb}  →  {wname}"
                        else:
                            line = f"? Match {idx+1}/{tot}: {na}  vs  {nb}  →  unclear"
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

                    scores = result.get("scores", {})
                    winner = result.get("winner")
                    wins   = result.get("wins", {})
                    n_models = len(results)
                    max_wins = max(1, n_models - 1)

                    # Build "W/N wins" labels for the leaderboard
                    wins_label = {
                        name: f"{wins.get(name, 0)}/{max_wins} wins"
                        for name in scores
                    }
                    self._judge_last_wins_label = wins_label

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

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _copy(self, tb: ctk.CTkTextbox) -> None:
        self.clipboard_clear()
        self.clipboard_append(tb.get("1.0", "end").strip())

    def _copy_model_name(self, lbl: ctk.CTkLabel, name: str) -> None:
        self.clipboard_clear()
        self.clipboard_append(name)
        orig_color = lbl.cget("text_color")
        lbl.configure(text="✓ Copied!", text_color="#44aa44")
        self.after(1200, lambda: lbl.configure(text=name, text_color=orig_color))

    def _save_settings(self) -> None:
        self.cfg["llama_path"]     = self._llama_var.get()
        self.cfg["models_dir"]     = self._models_dir_var.get()
        self.cfg["nlp_models_dir"] = self._nlp_dir_var.get()
        self.cfg["extra_args"]     = self._extra_var.get()
        self.cfg["chat_mode"]      = self._chat_mode_var.get()
        self.cfg["auto_judge"]     = self._auto_judge_var.get()
        try:
            self.cfg["params"] = {
                "temperature":    float(self._temp_var.get()),
                "top_p":          float(self._topp_var.get()),
                "max_tokens":     int(self._maxtok_var.get()),
                "ctx_size":       int(self._ctx_var.get()),
                "threads":        int(self._threads_var.get()),
                "gpu_layers":     int(self._gpu_var.get()),
                "seed":           int(self._seed_var.get()),
                "top_k":          int(self._topk_var.get()),
                "repeat_penalty": float(self._rep_var.get()),
            }
        except ValueError:
            pass
        save_config(self.cfg)
