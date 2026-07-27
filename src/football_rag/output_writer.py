"""Saves a pipeline run's final, human-readable output to
outputs/<prefix>_<timestamp>.txt - a clean deliverable text file (the
generated summary/evidence), distinct from the timestamped, leveled
execution log that `logging_config.py::setup_logging()` produces alongside
it. Used by the CLI scripts (`scripts/run_vertical_slice.py`,
`scripts/run_agentic_pipeline.py`) and the API's `/pipeline/run` endpoint.
`outputs/` is gitignored, same as `data/` and `logs/`.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

OUTPUT_DIR = Path("outputs")


def _timestamped_path(prefix: str) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # microseconds included so concurrent API requests in the same second
    # never collide on a filename.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return OUTPUT_DIR / f"{prefix}_{timestamp}.txt"


def write_output(prefix: str, text: str) -> Path:
    """Write `text` to a new timestamped file under outputs/ and return its
    path. Safe to call from concurrent requests (each gets its own file)."""
    path = _timestamped_path(prefix)
    path.write_text(text, encoding="utf-8")
    return path
