from __future__ import annotations

import logging
import platform
import re
import subprocess
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox

import yaml

log = logging.getLogger(__name__)


class ExportMixin:

    def _collect_hw_info(self) -> dict:
        cpu = platform.processor() or platform.machine() or "unknown"

        ram = "unknown"
        try:
            with open("/proc/meminfo", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        gb = int(line.split()[1]) / 1024 / 1024
                        ram = f"{gb:.0f} GB"
                        break
        except Exception:
            pass

        gpu = "unknown"
        try:
            res = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=3,
            )
            if res.returncode == 0 and res.stdout.strip():
                gpu = res.stdout.strip().split("\n")[0]
        except Exception:
            pass

        return {"cpu": cpu, "gpu": gpu, "ram": ram}

    def _on_save_batch_yaml(self) -> None:
        if not self._batch_run_results:
            return

        hw       = self._collect_hw_info()
        now      = datetime.now()
        dt_str   = now.strftime("%Y%m%d_%H%M%S")
        dt_iso   = now.isoformat(timespec="seconds")
        stem     = Path(self._batch_source_path).stem if self._batch_source_path else "batch"
        save_dir = Path(self._batch_source_path).parent if self._batch_source_path else Path.home()

        model_names = [m["name"] for m in self._batch_run_results[0]["models"]]
        saved: list[str] = []

        for model_name in model_names:
            cases_out = []
            for cr in self._batch_run_results:
                mr = next((m for m in cr["models"] if m["name"] == model_name), None)
                if mr is None:
                    continue
                entry: dict = {"id": cr["case_id"], "input": cr["user_input"]}
                if cr.get("expected") is not None:
                    entry["expected"] = cr["expected"]
                entry["output"] = (mr.get("output", "") if mr.get("success")
                                   else f"[ERROR] {mr.get('output', '')}")
                if mr.get("tps"):
                    entry["speed"] = f"{mr['tps']:.1f} t/s"
                cases_out.append(entry)

            clean = re.sub(r"[^\w\-]", "_", Path(model_name).stem)[:40]
            filepath = save_dir / f"{stem}_{dt_str}_{clean}.yaml"

            data = {
                "model_name": model_name,
                "datetime":   dt_iso,
                "hw":         hw,
                "cases":      cases_out,
            }
            with open(filepath, "w", encoding="utf-8") as f:
                yaml.dump(data, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
            log.info("Batch results saved: %s", filepath)
            saved.append(str(filepath))

        messagebox.showinfo(
            "Saved",
            f"Saved {len(saved)} file(s):\n" + "\n".join(saved),
            parent=self,
        )

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
            tps = f"{m['tps']:.1f}" if m.get("tps") else "—"
            tok = str(m["n_tokens"]) if m.get("n_tokens") else "—"
            ok  = "✓" if m.get("success") else "✗"
            lines.append(f"| {m['name']} | {m['elapsed']:.1f}s | {tps} | {tok} | {ok} |")

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
