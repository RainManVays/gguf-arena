from __future__ import annotations

import logging
import threading
from pathlib import Path
from tkinter import filedialog, messagebox

import customtkinter as ctk

from ..mdrender import render as _md_render
from ..runner import run_model, truncate_user_input
from ..storage import append_history, make_history_entry
from ..theme import (
    FONT_MONO as _FONT_MONO, FONT_SMALL as _FONT_SMALL,
    CLR_PRIMARY, CLR_PRIMARY_HOV,
    CLR_DISABLED, CLR_DISABLED_HOV, CLR_DISABLED_TB,
    CLR_TXT_FAINT, CLR_TXT_DIM, CLR_TXT_MUTED, CLR_ERR,
)

log = logging.getLogger(__name__)


class RunMixin:

    # ── Batch mode controls ───────────────────────────────────────────────────

    def _on_batch_load(self) -> None:
        path = filedialog.askopenfilename(
            title="Load batch cases",
            filetypes=[("YAML / JSON", "*.yaml *.yml *.json"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            from ..batch_runner import load_batch_file
            cases = load_batch_file(path)
        except ValueError as exc:
            messagebox.showerror("Batch load error", str(exc), parent=self)
            return
        self._batch_cases = cases
        self._batch_source_path = path
        fname = Path(path).name
        n = len(cases)
        self._batch_info_lbl.configure(
            text=f"{fname} · {n} case{'s' if n != 1 else ''}", text_color=CLR_TXT_MUTED)
        self._batch_clear_btn.grid(row=0, column=3, padx=(4, 0))
        self._set_batch_mode(True)

    def _on_batch_clear(self) -> None:
        self._batch_cases = None
        self._batch_info_lbl.configure(text="")
        self._batch_clear_btn.grid_remove()
        self._set_batch_mode(False)

    def _set_batch_mode(self, active: bool) -> None:
        self._batch_mode = active
        if active:
            self._user_text.configure(state="disabled", fg_color=CLR_DISABLED_TB)
            self._single_btn.configure(fg_color=CLR_DISABLED, hover_color=CLR_DISABLED_HOV)
        else:
            self._user_text.configure(state="normal", fg_color=("gray86", "gray17"))
            self._single_btn.configure(fg_color=CLR_PRIMARY, hover_color=CLR_PRIMARY_HOV)

    def _on_toggle_single(self) -> None:
        if self._batch_mode:
            self._set_batch_mode(False)

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

        # Remove previous model tabs (keep History)
        for tab_name in list(self._tabs._tab_dict.keys()):
            if tab_name != "History":
                self._tabs.delete(tab_name)
        self._model_widgets.clear()
        self._raw_outputs.clear()
        self._md_mode.clear()

        self._running        = True
        self._stop_requested = False
        self._current_proc   = None
        self._set_running(True)
        self._save_settings()

        # ── Batch path ────────────────────────────────────────────────────────
        if self._batch_mode and self._batch_cases:
            log.info("Batch run: %d case(s) × %d model(s)  chat_mode=%s",
                     len(self._batch_cases), len(selected), chat_mode)
            threading.Thread(
                target=self._run_batch_worker,
                args=(llama_path, selected, system_prompt, params, chat_mode, extra_args),
                daemon=True,
            ).start()
            return

        # ── Single path ───────────────────────────────────────────────────────
        if not user_input and not system_prompt:
            messagebox.showwarning("Empty", "Enter system prompt or user input.")
            self._running = False
            self._set_running(False)
            return

        try:
            user_input, was_truncated = truncate_user_input(
                system_prompt, user_input,
                params["ctx_size"], params["max_tokens"],
            )
        except ValueError as exc:
            messagebox.showerror("Prompt too long", str(exc))
            self._running = False
            self._set_running(False)
            return

        if was_truncated:
            messagebox.showwarning(
                "User input truncated",
                "User input was too long and has been trimmed to fit "
                f"ctx_size={params['ctx_size']} (max_tokens={params['max_tokens']}).",
            )
            self._user_text.delete("1.0", "end")
            self._user_text.insert("end", user_input)

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
                                font=_FONT_SMALL, text_color=CLR_TXT_FAINT, anchor="w")
            stat.grid(row=0, column=0, sticky="ew", padx=8)

            ctk.CTkButton(footer, text="Copy", width=60, height=24,
                          command=lambda t=tb: self._copy(t)).grid(row=0, column=1, padx=8)

            md_btn = ctk.CTkButton(footer, text="MD", width=50, height=24,
                                   state="disabled")
            md_btn.configure(command=lambda n=name: self._toggle_md(n))
            md_btn.grid(row=0, column=2, padx=(0, 8))

            self._model_widgets[name] = (tb, stat, md_btn)

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
                        text="⏳ Generating…", text_color=CLR_TXT_MUTED))

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
                            stat.configure(text="  ".join(parts), text_color=CLR_TXT_DIM)
                        else:
                            err = r.get("error", "Unknown error")
                            partial = r.get("output", "")
                            text = f"[{err}]"
                            if partial:
                                text += f"\n\n── partial output ──\n{partial}"
                            text += f"\n\n── stderr ──\n{r.get('stderr', '')}"
                            tb.insert("end", text)
                            stat.configure(text=f"✗ {err}", text_color=CLR_ERR)
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

                done_n  = len(model_results)
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
