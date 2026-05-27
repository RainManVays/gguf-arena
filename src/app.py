from __future__ import annotations

import logging
import subprocess
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox

import customtkinter as ctk

from .mixins.batch import BatchMixin
from .mixins.compare import CompareMixin
from .mixins.export import ExportMixin
from .mixins.history import HistoryMixin
from .mixins.judge import JudgeMixin
from .mixins.models_panel import ModelsPanelMixin
from .mixins.prompts import PromptsMixin
from .mixins.run import RunMixin
from .mixins.ui_rules import UIRulesMixin
from .storage import (load_config, load_registry,
                      migrate_prompts_to_registry, save_config)
from .theme import (
    FONT_MONO as _FONT_MONO, FONT_SMALL as _FONT_SMALL, FONT_BOLD as _FONT_BOLD,
    FONT_TINY,
    CLR_RUN, CLR_RUN_HOV, CLR_PRIMARY, CLR_PRIMARY_HOV,
    CLR_STOP, CLR_STOP_HOV, CLR_DANGER, CLR_DANGER_HOV,
    CLR_OK, CLR_TXT_DIM, CLR_TXT_FAINT, CLR_TXT_GHOST,
)

log = logging.getLogger(__name__)

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


# ──────────────────────────────────────────────────────────────────────────────

class App(HistoryMixin, CompareMixin, ExportMixin, PromptsMixin, ModelsPanelMixin, RunMixin, BatchMixin, JudgeMixin, UIRulesMixin, ctk.CTk):
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
        self._activate_ui_rules()
        self._setup_scroll_wheel(self._models_scroll)
        self._setup_scroll_wheel(self._history_scroll)

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

        ctk.CTkLabel(param_box, text="(−1 = auto)", font=FONT_TINY,
                     text_color=CLR_TXT_FAINT).grid(
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
            fg_color=CLR_RUN, hover_color=CLR_RUN_HOV,
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
                      fg_color=CLR_STOP, hover_color=CLR_STOP_HOV,
                      command=self._delete_prompt).grid(row=0, column=3)

        self._sys_text = ctk.CTkTextbox(top, height=85, wrap="word", font=_FONT_MONO,
                                        undo=True)
        self._sys_text.grid(row=1, column=0, sticky="ew", pady=(2, 6))
        self._bind_textbox(self._sys_text)
        self._sys_text.bind("<KeyRelease>", self._update_token_counter)
        self._ctx_var.trace_add("write", lambda *_: self._update_token_counter())

        user_hdr = ctk.CTkFrame(top, fg_color="transparent")
        user_hdr.grid(row=2, column=0, sticky="ew")
        user_hdr.grid_columnconfigure(2, weight=1)

        self._single_btn = ctk.CTkButton(
            user_hdr, text="User Input", width=110, height=28,
            font=_FONT_BOLD,
            fg_color=CLR_PRIMARY, hover_color=CLR_PRIMARY_HOV,
            command=self._on_toggle_single)
        self._single_btn.grid(row=0, column=0, sticky="w", padx=(0, 8))

        ctk.CTkButton(
            user_hdr, text="Load Batch…", width=100, height=28,
            font=_FONT_SMALL,
            command=self._on_batch_load).grid(row=0, column=1)

        self._batch_info_lbl = ctk.CTkLabel(
            user_hdr, text="", font=FONT_TINY, text_color=CLR_TXT_DIM, anchor="w")
        self._batch_info_lbl.grid(row=0, column=2, sticky="w", padx=(8, 0))

        self._batch_clear_btn = ctk.CTkButton(
            user_hdr, text="✕", width=28, height=28,
            fg_color=CLR_DANGER, hover_color=CLR_DANGER_HOV,
            command=self._on_batch_clear)

        self._user_text = ctk.CTkTextbox(top, height=110, wrap="word", font=_FONT_MONO,
                                         undo=True)
        self._user_text.grid(row=3, column=0, sticky="ew", pady=(2, 2))
        self._bind_textbox(self._user_text)
        self._user_text.bind("<KeyRelease>", self._update_token_counter)

        reg = load_registry()
        if reg:
            self._sys_text.insert("end", reg[0].get("system", ""))
            self._user_text.insert("end", reg[0].get("user", ""))
            self._prompt_combo.set(reg[0]["name"])

        self._tok_label = ctk.CTkLabel(top, text="", font=FONT_TINY,
                                       text_color=CLR_TXT_GHOST, anchor="w")
        self._tok_label.grid(row=4, column=0, sticky="w", pady=(0, 4))

        # Controls row: Run | Stop | chat mode | extra args | status
        ctrl = ctk.CTkFrame(top, fg_color="transparent")
        ctrl.grid(row=5, column=0, sticky="ew")

        self._run_btn = ctk.CTkButton(
            ctrl, text="▶  Run", width=110, height=36,
            fg_color=CLR_RUN, hover_color=CLR_RUN_HOV,
            command=self._on_run)
        self._run_btn.grid(row=0, column=0, padx=(0, 6))

        self._stop_btn = ctk.CTkButton(
            ctrl, text="■  Stop", width=90, height=36,
            fg_color=CLR_STOP, hover_color=CLR_STOP_HOV,
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
                                         text_color=CLR_TXT_DIM)
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


    # ── Helpers ───────────────────────────────────────────────────────────────

    def _copy(self, tb: ctk.CTkTextbox) -> None:
        self.clipboard_clear()
        self.clipboard_append(tb.get("1.0", "end").strip())

    def _copy_model_name(self, lbl: ctk.CTkLabel, name: str) -> None:
        self.clipboard_clear()
        self.clipboard_append(name)
        orig_color = lbl.cget("text_color")
        lbl.configure(text="✓ Copied!", text_color=CLR_OK)
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
