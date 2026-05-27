from __future__ import annotations

import logging
import threading
from pathlib import Path
from tkinter import filedialog, messagebox

import customtkinter as ctk

from ..storage import save_config
from ..theme import FONT_SMALL as _FONT_SMALL, CLR_TXT_FAINT

log = logging.getLogger(__name__)


def _fmt_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


class ModelsPanelMixin:

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
                from ..hf_downloader import download_model
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
                         font=_FONT_SMALL, text_color=CLR_TXT_FAINT).pack(anchor="w", pady=4)
            self._rebind_scroll_wheel(self._models_scroll)
            return

        pattern = "**/*.gguf" if self._recursive_var.get() else "*.gguf"
        files = sorted(folder.glob(pattern))

        if not files:
            ctk.CTkLabel(self._models_scroll, text="No .gguf files found",
                         font=_FONT_SMALL, text_color=CLR_TXT_FAINT).pack(anchor="w", pady=4)
            self._rebind_scroll_wheel(self._models_scroll)
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
        self._rebind_scroll_wheel(self._models_scroll)

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

    def _select_all(self) -> None:
        for v in self.model_vars.values():
            v.set(True)

    def _select_none(self) -> None:
        for v in self.model_vars.values():
            v.set(False)

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
