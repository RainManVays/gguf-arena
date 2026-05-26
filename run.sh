#!/usr/bin/env bash
# GGUF Arena — launcher
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$SCRIPT_DIR/logs"

# ── venv (только на хост-машине, не в контейнере) ────────────────────────────
if [[ -z "${DEVCONTAINER:-}" ]]; then
    VENV="$SCRIPT_DIR/.venv"
    REQ="$SCRIPT_DIR/requirements.txt"
    STAMP="$VENV/.req_stamp"
    if [[ ! -d "$VENV" ]]; then
        echo "[run.sh] Creating venv..."
        python3 -m venv "$VENV"
    fi
    if [[ ! -f "$STAMP" ]] || [[ "$REQ" -nt "$STAMP" ]]; then
        echo "[run.sh] Installing dependencies..."
        "$VENV/bin/pip" install -q -r "$REQ"
        touch "$STAMP"
    fi
    PYTHON="$VENV/bin/python"
else
    PYTHON="python3"
fi

# ── display ───────────────────────────────────────────────────────────────────
if [[ -z "${DISPLAY:-}" ]]; then
    export DISPLAY=:0
fi

# ── launch ────────────────────────────────────────────────────────────────────
mkdir -p "$LOG_DIR"

echo "[run.sh] Starting GGUF Arena..."
echo "[run.sh] Log file: $LOG_DIR/stand.log"

exec "$PYTHON" "$SCRIPT_DIR/main.py" "$@"
