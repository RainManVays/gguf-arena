"""Central logging configuration for GGUF Arena.

Call setup_logging() once at startup. All modules then use:
    import logging
    log = logging.getLogger(__name__)
"""

import logging
import logging.handlers
import os
from pathlib import Path

LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_FILE = LOG_DIR / "stand.log"

# 5 MB per file, keep 5 rotated files → max 25 MB total
_MAX_BYTES = 5 * 1024 * 1024
_BACKUP_COUNT = 5

_FMT = "%(asctime)s  %(levelname)-8s  %(name)-20s  %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"


def setup_logging(level: int = logging.DEBUG) -> None:
    """Configure root logger: DEBUG → file, LOG_LEVEL (default WARNING) → stderr."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)

    fh = logging.handlers.RotatingFileHandler(
        LOG_FILE,
        maxBytes=_MAX_BYTES,
        backupCount=_BACKUP_COUNT,
        encoding="utf-8",
    )
    fh.setLevel(level)
    fh.setFormatter(logging.Formatter(_FMT, datefmt=_DATEFMT))

    console_level = getattr(logging, os.environ.get("LOG_LEVEL", "WARNING").upper(), logging.WARNING)
    ch = logging.StreamHandler()
    ch.setLevel(console_level)
    ch.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))

    root.addHandler(fh)
    root.addHandler(ch)

    # Silence noisy third-party loggers
    for name in ("PIL", "customtkinter"):
        logging.getLogger(name).setLevel(logging.WARNING)

    logging.getLogger(__name__).info(
        "Logging started  file=%s  level=%s", LOG_FILE, logging.getLevelName(level)
    )
