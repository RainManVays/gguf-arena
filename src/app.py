from __future__ import annotations

import logging
import subprocess
import threading
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, simpledialog

import customtkinter as ctk

from .runner import _estimate_tokens, run_model, truncate_user_input
from .mdrender import render as _md_render
from .storage import (append_history, delete_from_registry, delete_history_entry,
                      load_config, load_history, load_registry, make_history_entry,
                      migrate_prompts_to_registry, save_config, save_to_registry)

log = logging.getLogger(__name__)

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

_FONT_MONO  = ("Monospace", 12)
_FONT_SMALL = ("Sans", 11)
_FONT_BOLD  = ("Sans", 12, "bold")


def _fmt_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


# ──────────────────────────────────────────────────────────────────────────────

class App(ctk.CTk):
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

        log.info("App started")
        self._build_ui()
        self._load_model_list()
        self._refresh_history_tab()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ── helpers ───────────────────────────────────────────────────────────────

    def _ui(self, func, **kwargs) -> None:
        """Schedule a widget configure call safely from any thread."""
        self.after(0, lambda: func(**kwargs))

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

        ctk.CTkLabel(top, text="User Input", font=_FONT_BOLD).grid(
            row=2, column=0, sticky="w")
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

        ctk.CTkLabel(ctrl, text="Extra args:", font=_FONT_SMALL).grid(
            row=0, column=3, padx=(0, 4))
        self._extra_var = ctk.StringVar(value=self.cfg.get("extra_args", ""))
        ctk.CTkEntry(ctrl, textvariable=self._extra_var, width=220,
                     height=32, font=_FONT_SMALL).grid(row=0, column=4, padx=(0, 14))

        self._status_lbl = ctk.CTkLabel(ctrl, text="", font=_FONT_SMALL,
                                         text_color="#888888")
        self._status_lbl.grid(row=0, column=5, padx=(0, 10))

        self._export_btn = ctk.CTkButton(
            ctrl, text="Export…", width=80, height=32,
            state="disabled", command=self._on_export)
        self._export_btn.grid(row=0, column=6)

        # Results tabs
        self._tabs = ctk.CTkTabview(self._right)
        self._tabs.grid(row=1, column=0, sticky="nsew", padx=8, pady=(0, 8))

        self._tabs.add("History")
        self._history_scroll = ctk.CTkScrollableFrame(self._tabs.tab("History"))
        self._history_scroll.pack(fill="both", expand=True)


    # ── HF Download ───────────────────────────────────────────────────────────

    def _browse_nlp_dir(self) -> None:
        path = filedialog.askdirectory(title="Select NLP models folder")
        if path:
            self._nlp_dir_var.set(path)
            self.cfg["nlp_models_dir"] = path
            save_config(self.cfg)

    def _on_hf_download(self) -> None:
        repo_id   = self._hf_repo_var.get().strip()
        local_dir = self._nlp_dir_var.get().strip()
        if not repo_id:
            messagebox.showwarning("No model ID", "Enter a HuggingFace model ID.", parent=self)
            return
        if not local_dir:
            messagebox.showwarning("No directory", "Select a save directory.", parent=self)
            return

        self.cfg["nlp_models_dir"] = local_dir
        save_config(self.cfg)

        self._hf_log.configure(state="normal")
        self._hf_log.delete("1.0", "end")
        self._hf_log.configure(state="disabled")
        self._hf_download_btn.configure(state="disabled", text="Downloading…")

        def _append(msg: str) -> None:
            def _do() -> None:
                self._hf_log.configure(state="normal")
                self._hf_log.insert("end", msg + "\n")
                self._hf_log.see("end")
                self._hf_log.configure(state="disabled")
            self.after(0, _do)

        def worker() -> None:
            try:
                from .hf_downloader import download_model
                download_model(repo_id, local_dir, on_progress=_append)
            except Exception as e:
                _append(f"✗ Error: {e}")
                log.exception("HF download failed: %s → %s", repo_id, local_dir)
            finally:
                self.after(0, lambda: self._hf_download_btn.configure(
                    state="normal", text="↓ Download"))

        threading.Thread(target=worker, daemon=True).start()

    # ── Model list ────────────────────────────────────────────────────────────

    def _load_model_list(self) -> None:
        for w in self._models_scroll.winfo_children():
            w.destroy()
        self.model_vars.clear()

        folder = Path(self._models_dir_var.get())
        log.debug("Scanning models in %s (recursive=%s)", folder, self._recursive_var.get())

        if not folder.is_dir():
            ctk.CTkLabel(self._models_scroll, text="No folder selected",
                         font=_FONT_SMALL, text_color="#777777").pack(anchor="w", pady=4)
            return

        pattern = "**/*.gguf" if self._recursive_var.get() else "*.gguf"
        files = sorted(folder.glob(pattern))

        if not files:
            ctk.CTkLabel(self._models_scroll, text="No .gguf files found",
                         font=_FONT_SMALL, text_color="#777777").pack(anchor="w", pady=4)
            return

        log.info("Found %d model(s) in %s", len(files), folder)
        for f in files:
            base = str(f.relative_to(folder)) if self._recursive_var.get() else f.name
            try:
                size_str = _fmt_size(f.stat().st_size)
            except OSError:
                size_str = "?"
            display = f"{base}  [{size_str}]"
            var = ctk.BooleanVar(value=False)
            ctk.CTkCheckBox(self._models_scroll, text=display, variable=var,
                            font=_FONT_SMALL).pack(anchor="w", pady=2)
            self.model_vars[str(f)] = var

    def _add_model_file(self) -> None:
        path = filedialog.askopenfilename(
            title="Select .gguf model",
            filetypes=[("GGUF models", "*.gguf"), ("All files", "*.*")],
        )
        if not path or path in self.model_vars:
            return
        var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(self._models_scroll, text=Path(path).name,
                        variable=var, font=_FONT_SMALL).pack(anchor="w", pady=2)
        self.model_vars[path] = var

    def _select_all(self)  -> None:
        for v in self.model_vars.values(): v.set(True)

    def _select_none(self) -> None:
        for v in self.model_vars.values(): v.set(False)

    def _browse_llama(self) -> None:
        path = filedialog.askopenfilename(title="Select llama-cli binary")
        if path:
            self._llama_var.set(path)
            log.info("llama-cli set to %s", path)
            self._save_settings()

    def _browse_models_dir(self) -> None:
        path = filedialog.askdirectory(title="Select models folder")
        if path:
            self._models_dir_var.set(path)
            log.info("Models dir set to %s", path)
            self._save_settings()
            self._load_model_list()

    # ── Prompt library ────────────────────────────────────────────────────────

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

    def _save_prompt(self) -> None:
        current = self._prompt_combo.get()
        default = current if current and current != "—" else ""
        name = simpledialog.askstring(
            "Save Prompt", "Prompt name:", initialvalue=default, parent=self)
        if not name:
            return
        system = self._sys_text.get("1.0", "end").strip()
        user   = self._user_text.get("1.0", "end").strip()
        from .storage import REGISTRY_DIR, _slug
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

    def _toggle_md(self, name: str) -> None:
        tb, stat, md_btn = self._model_widgets[name]
        raw = self._raw_outputs.get(name, "")
        if self._md_mode.get(name, False):
            tb.configure(state="normal")
            tb.delete("1.0", "end")
            tb.insert("end", raw)
            tb.configure(state="disabled")
            self._md_mode[name] = False
            md_btn.configure(text="MD")
        else:
            _md_render(tb, raw)
            self._md_mode[name] = True
            md_btn.configure(text="Raw")

    # ── Run / Stop ────────────────────────────────────────────────────────────

    def _collect_params(self) -> dict | None:
        try:
            return {
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
        except ValueError as e:
            messagebox.showerror("Bad parameters", str(e))
            return None

    def _set_running(self, running: bool) -> None:
        if running:
            self._run_btn.configure(state="disabled", text="Running…")
            self._stop_btn.configure(state="normal")
        else:
            self._run_btn.configure(state="normal", text="▶  Run")
            self._stop_btn.configure(state="disabled")

    def _on_close(self) -> None:
        log.info("Window close requested  running=%s", self._running)
        if self._running:
            self._stop_requested = True
            proc = self._current_proc
            if proc is not None and proc.poll() is None:
                log.info("Terminating subprocess pid=%d on app exit", proc.pid)
                proc.terminate()
        log.info("App exiting")
        self.destroy()

    def _on_stop(self) -> None:
        if not self._running:
            return
        log.info("Stop requested by user")
        self._stop_requested = True
        proc = self._current_proc
        if proc is not None and proc.poll() is None:
            proc.terminate()
        self._status_lbl.configure(text="Stopping…")

    def _on_run(self) -> None:
        if self._running:
            return

        selected = [(p, v) for p, v in self.model_vars.items() if v.get()]
        if not selected:
            messagebox.showwarning("No models", "Select at least one model.")
            return

        llama_path = self._llama_var.get().strip()
        if not Path(llama_path).is_file():
            log.error("llama-cli not found: %s", llama_path)
            messagebox.showerror("Not found", f"llama-cli not found:\n{llama_path}")
            return

        params = self._collect_params()
        if params is None:
            return

        system_prompt = self._sys_text.get("1.0", "end").strip()
        user_input    = self._user_text.get("1.0", "end").strip()
        chat_mode     = self._chat_mode_var.get()
        extra_args    = self._extra_var.get().strip()

        if not user_input and not system_prompt:
            messagebox.showwarning("Empty", "Enter system prompt or user input.")
            return

        try:
            user_input, was_truncated = truncate_user_input(
                system_prompt, user_input,
                params["ctx_size"], params["max_tokens"],
            )
        except ValueError as exc:
            messagebox.showerror("Prompt too long", str(exc))
            return

        if was_truncated:
            messagebox.showwarning(
                "User input truncated",
                "User input was too long and has been trimmed to fit "
                f"ctx_size={params['ctx_size']} (max_tokens={params['max_tokens']}).",
            )
            self._user_text.delete("1.0", "end")
            self._user_text.insert("end", user_input)

        # Remove previous model tabs (keep History)
        for tab_name in list(self._tabs._tab_dict.keys()):
            if tab_name != "History":
                self._tabs.delete(tab_name)
        self._model_widgets.clear()
        self._raw_outputs.clear()
        self._md_mode.clear()

        # Create result tabs
        for path, _ in selected:
            name = Path(path).name
            self._tabs.add(name)
            tab = self._tabs.tab(name)
            tab.grid_rowconfigure(0, weight=1)
            tab.grid_columnconfigure(0, weight=1)

            tb = ctk.CTkTextbox(tab, wrap="word", state="disabled", font=_FONT_MONO)
            tb.grid(row=0, column=0, sticky="nsew")

            footer = ctk.CTkFrame(tab, fg_color="transparent")
            footer.grid(row=1, column=0, sticky="ew", pady=(2, 4))
            footer.grid_columnconfigure(0, weight=1)

            stat = ctk.CTkLabel(footer, text="⏳ Waiting…",
                                 font=_FONT_SMALL, text_color="#777777", anchor="w")
            stat.grid(row=0, column=0, sticky="ew", padx=8)

            ctk.CTkButton(footer, text="Copy", width=60, height=24,
                          command=lambda t=tb: self._copy(t)).grid(row=0, column=1, padx=8)

            md_btn = ctk.CTkButton(footer, text="MD", width=50, height=24,
                                   state="disabled")
            md_btn.configure(command=lambda n=name: self._toggle_md(n))
            md_btn.grid(row=0, column=2, padx=(0, 8))

            self._model_widgets[name] = (tb, stat, md_btn)

        self._running        = True
        self._stop_requested = False
        self._current_proc   = None
        self._set_running(True)
        self._save_settings()

        if selected:
            self._tabs.set(Path(selected[0][0]).name)

        log.info("Run started: %d model(s)  chat_mode=%s  params=%s",
                 len(selected), chat_mode, params)

        def worker() -> None:
            model_results = []
            total = len(selected)
            try:
                for i, (path, _) in enumerate(selected):
                    if self._stop_requested:
                        log.info("Stop flag set, skipping remaining models")
                        break

                    name = Path(path).name
                    status_txt = f"[{i+1}/{total}] {name}…"
                    self.after(0, lambda t=status_txt: self._status_lbl.configure(text=t))
                    self.after(0, lambda n=name: self._tabs.set(n))
                    self.after(0, lambda n=name: self._model_widgets[n][1].configure(
                        text="⏳ Generating…", text_color="#aaaaaa"))

                    def _on_proc(proc, n=name) -> None:
                        self._current_proc = proc
                        log.debug("Process started  model=%s  pid=%d", n, proc.pid)

                    res = run_model(
                        llama_path    = llama_path,
                        model_path    = path,
                        system_prompt = system_prompt,
                        user_input    = user_input,
                        params        = params,
                        chat_mode     = chat_mode,
                        extra_args    = extra_args,
                        proc_started  = _on_proc,
                    )

                    def _update(n=name, r=res) -> None:
                        tb, stat, md_btn = self._model_widgets[n]
                        tb.configure(state="normal")
                        tb.delete("1.0", "end")
                        if r.get("success"):
                            raw = r["output"]
                            tb.insert("end", raw)
                            self._raw_outputs[n] = raw
                            self._md_mode[n] = False
                            md_btn.configure(state="normal")
                            parts = [f"⏱ {r['elapsed']:.1f}s"]
                            if r.get("tps"):
                                parts.append(f"{r['tps']:.1f} tok/s")
                            if r.get("n_tokens"):
                                parts.append(f"{r['n_tokens']} tokens")
                            stat.configure(text="  ".join(parts), text_color="#888888")
                        else:
                            err = r.get("error", "Unknown error")
                            partial = r.get("output", "")
                            text = f"[{err}]"
                            if partial:
                                text += f"\n\n── partial output ──\n{partial}"
                            text += f"\n\n── stderr ──\n{r.get('stderr', '')}"
                            tb.insert("end", text)
                            stat.configure(text=f"✗ {err}", text_color="#cc4444")
                        tb.configure(state="disabled")

                    self.after(0, _update)

                    model_results.append({
                        "name":     name,
                        "success":  res.get("success", False),
                        "output":   res.get("output", ""),
                        "elapsed":  res.get("elapsed", 0),
                        "tps":      res.get("tps"),
                        "n_tokens": res.get("n_tokens"),
                    })

                # persist history only if at least one model ran
                if model_results:
                    entry = make_history_entry(system_prompt, user_input, params,
                                               chat_mode, model_results)
                    append_history(entry)
                    self.after(0, self._refresh_history_tab)
                    self._last_run = {
                        "timestamp":     entry["timestamp"],
                        "system_prompt": system_prompt,
                        "user_input":    user_input,
                        "params":        params,
                        "chat_mode":     chat_mode,
                        "models":        model_results,
                    }
                    self.after(0, lambda r=model_results: self._build_compare_tab(r))
                    self.after(0, lambda r=model_results: self._build_stats_tab(r))
                    self.after(0, lambda r=model_results, s=system_prompt, u=user_input:
                               self._build_judge_tab(r, s, u))
                    self.after(0, lambda: self._export_btn.configure(state="normal"))

                done_n = len(model_results)
                stopped = self._stop_requested
                log.info("Run finished: %d/%d model(s) ran  stopped=%s",
                         done_n, total, stopped)
                done_txt = (f"Stopped ({done_n}/{total})" if stopped
                            else f"Done ({done_n} model{'s' if done_n > 1 else ''})")
                self.after(0, lambda t=done_txt: self._status_lbl.configure(text=t))

            except Exception:
                log.exception("Unhandled exception in worker thread")
                self.after(0, lambda: self._status_lbl.configure(
                    text="Internal error — see log"))
            finally:
                self._running        = False
                self._stop_requested = False
                self._current_proc   = None
                self.after(0, lambda: self._set_running(False))

        threading.Thread(target=worker, daemon=True).start()

    # ── History tab ───────────────────────────────────────────────────────────

    def _refresh_history_tab(self) -> None:
        for w in self._history_scroll.winfo_children():
            w.destroy()
        history = load_history()
        if not history:
            ctk.CTkLabel(self._history_scroll, text="No runs yet.",
                         font=_FONT_SMALL, text_color="#666666").pack(
                anchor="w", padx=8, pady=8)
            return
        for entry in history:
            self._add_history_card(entry)

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
            font=ctk.CTkFont(size=10), text_color="#777777",
            anchor="w", justify="left",
        ).grid(row=1, column=0, sticky="ew", padx=8, pady=(0, 6))

        btn_frame = ctk.CTkFrame(card, fg_color="transparent")
        btn_frame.grid(row=0, column=1, rowspan=2, padx=8)

        ctk.CTkButton(btn_frame, text="Load", width=60, height=26,
                      command=lambda e=entry: self._load_history_entry(e)).pack(pady=(0, 4))
        ctk.CTkButton(btn_frame, text="✕", width=60, height=26,
                      fg_color="#5a1a1a", hover_color="#7a2222",
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

    # ── Compare tab ───────────────────────────────────────────────────────────

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
            ctk.CTkLabel(tab, text=r["name"], font=_FONT_BOLD, anchor="center").grid(
                row=0, column=i, sticky="ew", padx=(4 if i else 8, 4), pady=(4, 2))
            tb = ctk.CTkTextbox(tab, wrap="word", state="disabled", font=_FONT_MONO)
            tb.grid(row=1, column=i, sticky="nsew", padx=(4 if i else 8, 4), pady=(0, 4))
            tb.configure(state="normal")
            tb.insert("end", r["output"])
            tb.configure(state="disabled")

    # ── Stats tab (tok/s chart) ───────────────────────────────────────────────

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

        canvas = tk.Canvas(tab, bg="#2b2b2b", highlightthickness=0)
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
            row_h     = max(28, (h - pad_top) // len(has_tps))
            max_tps   = max(r["tps"] for r in has_tps)
            bar_area  = max(1, w - pad_left - pad_right)
            for i, r in enumerate(has_tps):
                y = pad_top + i * row_h + row_h // 2
                bar_w = int(r["tps"] / max_tps * bar_area)
                canvas.create_rectangle(
                    pad_left, y - 10, pad_left + bar_w, y + 10,
                    fill="#1f6aa5", outline="")
                label = r["name"][:36]
                canvas.create_text(
                    pad_left - 8, y, text=label, anchor="e",
                    fill="#cccccc", font=("Sans", 10))
                canvas.create_text(
                    pad_left + bar_w + 6, y,
                    text=f"{r['tps']:.1f} tok/s", anchor="w",
                    fill="#aaaaaa", font=("Sans", 10))

        canvas.bind("<Configure>", _draw)
        canvas.after(50, _draw)

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
        if display_names:
            self._judge_model_combo.set(display_names[0])
        self._judge_model_combo.pack(side="left", padx=(0, 10))

        self._judge_mode_var = ctk.StringVar(value="Pairwise")
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
            messagebox.showerror("Judge", f"Model file not found:\n{judge_name}", parent=self)
            return

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

    # ── Export ────────────────────────────────────────────────────────────────

    def _on_export(self) -> None:
        if not self._last_run:
            return
        path = filedialog.asksaveasfilename(
            title="Export results",
            defaultextension=".md",
            filetypes=[("Markdown", "*.md"), ("Text", "*.txt")],
        )
        if not path:
            return

        run = self._last_run
        params = run["params"]
        ts = run["timestamp"]
        lines: list[str] = [
            f"# LLM Benchmark — {ts}",
            "",
            f"**System prompt:** {run['system_prompt'][:200] or '—'}",
            f"**User input:** {run['user_input'][:200] or '—'}",
            "",
            "## Parameters",
            "",
            "| Key | Value |",
            "|-----|-------|",
        ]
        for k, v in params.items():
            lines.append(f"| {k} | {v} |")

        lines += [
            "",
            "## Results",
            "",
            "| Model | Elapsed | tok/s | Tokens | Status |",
            "|-------|---------|-------|--------|--------|",
        ]
        for m in run["models"]:
            tps  = f"{m['tps']:.1f}" if m.get("tps") else "—"
            tok  = str(m["n_tokens"]) if m.get("n_tokens") else "—"
            ok   = "✓" if m.get("success") else "✗"
            lines.append(
                f"| {m['name']} | {m['elapsed']:.1f}s | {tps} | {tok} | {ok} |")

        lines += ["", "## Outputs", ""]
        for m in run["models"]:
            lines.append(f"### {m['name']}")
            lines.append("")
            lines.append("```")
            lines.append(m.get("output", ""))
            lines.append("```")
            lines.append("")

        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        log.info("Results exported to %s", path)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _copy(self, tb: ctk.CTkTextbox) -> None:
        self.clipboard_clear()
        self.clipboard_append(tb.get("1.0", "end").strip())

    def _save_settings(self) -> None:
        self.cfg["llama_path"]     = self._llama_var.get()
        self.cfg["models_dir"]     = self._models_dir_var.get()
        self.cfg["nlp_models_dir"] = self._nlp_dir_var.get()
        self.cfg["extra_args"]     = self._extra_var.get()
        self.cfg["chat_mode"]      = self._chat_mode_var.get()
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
