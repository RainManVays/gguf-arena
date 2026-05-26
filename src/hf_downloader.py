from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Callable

log = logging.getLogger(__name__)


def download_model(
    repo_id: str,
    local_dir: str | Path,
    on_progress: Callable[[str], None] | None = None,
) -> None:
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

    from huggingface_hub import hf_hub_download, list_repo_files

    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    def _log(msg: str) -> None:
        log.info("[hf] %s", msg)
        if on_progress:
            on_progress(msg)

    _log(f"Connecting: {repo_id}")

    try:
        files = list(list_repo_files(repo_id))
    except Exception as e:
        raise RuntimeError(f"Cannot list files for '{repo_id}': {e}") from e

    _log(f"Found {len(files)} file(s)")

    for i, filename in enumerate(files):
        _log(f"[{i + 1}/{len(files)}] {filename}")
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=str(local_dir),
        )

    _log(f"✓ Done → {local_dir}")
