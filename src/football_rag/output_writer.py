"""Saves generated output to outputs/<prefix>_<timestamp>.txt - shared by
every CLI script (`scripts/run_*.py`, via `tee_to_file()`) and the API's
`/pipeline/run` endpoint (via `write_output()`), so every run or request
leaves a persistent, timestamped record. `outputs/` is gitignored, same as
`data/`.
"""

from __future__ import annotations

import sys
from contextlib import contextmanager
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
    """Redirect stdout to both the console and a new
    outputs/<prefix>_<timestamp>.txt for the duration of the `with` block.
    CLI scripts only - not safe for concurrent use, unlike `write_output()`,
    since it swaps out the process-global `sys.stdout`. Yields the path."""
    path = _timestamped_path(prefix)
    original_stdout = sys.stdout
    # Line-buffered (not the default block-buffering for a regular file) so
    # a long-running script's log file can actually be tailed for live
    # progress instead of only catching up once the buffer fills or the
    # process exits.
    with open(path, "w", encoding="utf-8", buffering=1) as f:
        sys.stdout = _Tee(original_stdout, f)
        try:
            yield path
        finally:
            sys.stdout = original_stdout
