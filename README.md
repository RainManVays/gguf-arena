# GGUF Arena

Desktop GUI for benchmarking local GGUF models via [llama.cpp](https://github.com/ggerganov/llama.cpp).

Run multiple models side-by-side, compare outputs and generation speed, save prompts to a library, and let a model judge which output is best.

---

## Features

- Select and run multiple GGUF models simultaneously
- System prompt + user input with live token counter
- Prompt library backed by YAML files (`proms_reg/`) — save / load / delete
- Per-model result tabs with Markdown rendering
- **Compare** — all model outputs side-by-side
- **Stats** — horizontal bar chart of generation speed (tok/s)
- **Judge** — let any loaded model score or run a pairwise tournament between outputs
- **HF Download** — download any HuggingFace model directly into a local folder
- Run history with one-click reload
- Export results to Markdown
- Configurable inference parameters: temperature, top_p, top_k, ctx_size, max_tokens, threads, gpu_layers, seed, repeat_penalty

---

## Requirements

- Python 3.12+
- [llama.cpp](https://github.com/ggerganov/llama.cpp) compiled binary `llama-cli`
- Linux with X11 (`DISPLAY` set)

---

## Quick start

**Option A — launcher script** (no setup required):
```bash
git clone https://github.com/RainManVays/gguf-arena
cd gguf-arena
./run.sh
```
`run.sh` creates a virtualenv, installs dependencies and launches the app. Subsequent runs reuse the existing venv.

**Option B — install as a command** (editable install):
```bash
git clone https://github.com/RainManVays/gguf-arena
cd gguf-arena
pip install -e .
llm-stand
```
After this `llm-stand` is available system-wide (or in the active virtualenv).

> **Note:** use `pip install -e .` (editable). A regular `pip install .` works for the command itself, but the prompt library (`proms_reg/`) is looked up relative to the source tree, so editable install keeps everything in one place.

### Python dependencies

```
customtkinter>=5.2.0
pyyaml>=6.0
huggingface_hub>=0.20.0
```

---

## Project layout

```
main.py          — entry point
run.sh           — launcher (venv, DISPLAY, exec)
src/
  app.py         — UI (CustomTkinter), run / stop / history logic
  runner.py      — llama-cli command builder, subprocess runner, stats parser
  storage.py     — config / history (JSON), prompt registry (YAML)
  judge.py       — all-at-once and pairwise judge modes
  hf_downloader.py — HuggingFace model downloader
  mdrender.py    — lightweight Markdown renderer for CTkTextbox
  log.py         — rotating log → logs/stand.log
proms_reg/       — prompt library (*.yaml, versioned in git)
logs/            — runtime logs (gitignored)
```

---

## Configuration

Stored in `~/.llm_stand/config.json`:

| Field | Default | Description |
|---|---|---|
| `llama_path` | *(set manually)* | Path to the `llama-cli` binary |
| `models_dir` | `""` | Folder scanned for `.gguf` files |
| `chat_mode` | `true` | Pass `-cnv -st` flags to llama-cli |
| `extra_args` | `""` | Extra CLI arguments |
| `params.*` | see below | Inference parameters |

Default parameters: `temperature=0.8`, `top_p=0.9`, `max_tokens=512`, `ctx_size=8192`, `threads=-1`, `gpu_layers=-1`, `seed=-1`, `top_k=40`, `repeat_penalty=1.1`.

Run history: `~/.llm_stand/history.json` (last 100 entries).

---

## Prompt library

Prompts are YAML files in `proms_reg/`. Format:

```yaml
name: "Prompt name"
system: |
  System prompt text
user: |
  Optional user input template
tags: [coding, general]  # optional
```

Manage via **Save** / **Del** buttons in the UI. Files can be committed to git.

---

## Prompt truncation

If `sys_tokens + user_tokens + max_tokens > ctx_size`, user input is trimmed from the end until everything fits. The system prompt is never trimmed. If even an empty user input doesn't fit, an error is shown and the run is aborted. Token estimate: **2 chars = 1 token** (conservative; covers Cyrillic-heavy text).

---

## Logs

`logs/stand.log` — rotating file (5 MB × 5 files).

---

## DevContainer (optional)

Copy `.devcontainer/.env.example` to `.devcontainer/.env` and set `MODELS_DIR` to your local models folder. The compose file defaults to a CPU image; see comments inside for GPU (NVIDIA) setup.

---

## License

MIT — see [LICENSE](LICENSE).
