"""Centralized logging setup, used by every long-running entry point
(`experiment/*.py`, `scripts/*.py`, `api/main.py`) instead of ad hoc
`print()` statements.

Plain `print()`-to-a-file only captures raw text with no timestamp, level,
or source module - fine for "save this pipeline's final answer to a file"
(still done separately via `output_writer.py::write_output()`) but not for
"tell me what stage a long-running experiment is at, or filter out DEBUG
noise, while it's running." `setup_logging()` configures the standard
library's `logging`
module instead: every line gets a timestamp and level, console output stays
at INFO by default (readable while watching a run live), and everything
including DEBUG goes to a timestamped file under `logs/` for later
inspection - independent of whatever redirected the process's stdout.
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path

LOG_DIR = Path("logs")
_LOG_FORMAT = "%(asctime)s [%(levelname)-8s] %(name)s: %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Third-party libraries that log at INFO by default and are pure noise for
# this project's purposes (confirmed live: httpx alone logs a HEAD request
# line for every single file HuggingFace Hub resolves when loading a model,
# completely burying the actually useful progress messages). Quieted to
# WARNING so only their genuine problems surface, not routine traffic.
_NOISY_LOGGERS = ["httpx", "httpcore", "urllib3", "filelock", "sentence_transformers", "chromadb", "google_genai"]


def setup_logging(run_name: str, console_level: int = logging.INFO) -> Path:
    """Configure the root logger for this process: a console handler at
    `console_level` (INFO by default - readable progress, not a firehose)
    and a file handler at DEBUG (everything) writing to
    `logs/<run_name>_<timestamp>.log`. Safe to call more than once in the
    same interpreter (clears any handlers from a prior call first, so
    re-entrant test/notebook usage doesn't duplicate every log line).

    Call this once near the top of a script's `main()`, then use
    `logging.getLogger(__name__)` (or any name) anywhere in the codebase -
    every logger propagates to this shared root configuration automatically,
    with no per-module setup required.
    """
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"{run_name}_{timestamp}.log"

    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.handlers.clear()

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(formatter)
    root.addHandler(console_handler)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)

    return log_path
