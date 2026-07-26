"""Lets CLI scripts save their printed output to a timestamped file under
outputs/, while still printing to the console as normal. tqdm/sentence-
transformers progress bars go to stderr, not stdout, so the saved file
stays clean (just the script's own print() calls).
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

OUTPUT_DIR = Path("outputs")


class _Tee:
    def __init__(self, *streams):
        self._streams = streams

    def write(self, data: str) -> None:
        for stream in self._streams:
            stream.write(data)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()


@contextmanager
def tee_to_file(prefix: str):
    """Redirect stdout to both the console and outputs/<prefix>_<timestamp>.txt
    for the duration of the `with` block. Yields the file path."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = OUTPUT_DIR / f"{prefix}_{timestamp}.txt"

    original_stdout = sys.stdout
    with open(path, "w", encoding="utf-8") as f:
        sys.stdout = _Tee(original_stdout, f)
        try:
            yield path
        finally:
            sys.stdout = original_stdout
