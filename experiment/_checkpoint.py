"""Shared resumable-CSV helper for `run_full_comparison.py` and
`run_full_ablation.py` - both need to survive being stopped daily by
Gemini's free-tier quota (20 requests/day/model) and picked back up the
next day without redoing already-completed rows or losing partial
progress to a crash.

Design: a plain CSV under `experiment/results/`, one row per
completed (key columns...) combination, appended and flushed
immediately after each row - not batched - so a Ctrl-C, a crash, or a
quota cutoff mid-run never loses more than the one row in flight.
"""

from __future__ import annotations

import csv
from pathlib import Path

import httpx
from google.genai import errors as genai_errors


def load_done_keys(csv_path: Path, key_columns: list[str]) -> set[tuple]:
    """Already-completed rows, keyed by `key_columns`, so a re-run skips
    them. Empty set if the CSV doesn't exist yet (first run)."""
    if not csv_path.exists():
        return set()
    done = set()
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            done.add(tuple(row[col] for col in key_columns))
    return done


class CheckpointWriter:
    """Appends one row at a time, flushing after each write. Writes the
    header only when creating a new file, so resuming a partial CSV keeps
    appending rather than duplicating the header."""

    def __init__(self, csv_path: Path, fieldnames: list[str]):
        self.csv_path = csv_path
        self.fieldnames = fieldnames
        is_new = not csv_path.exists()
        self._file = csv_path.open("a", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=fieldnames)
        if is_new:
            self._writer.writeheader()
            self._file.flush()

    def write(self, row: dict) -> None:
        self._writer.writerow(row)
        self._file.flush()

    def close(self) -> None:
        self._file.close()


def is_quota_error(exc: Exception) -> bool:
    """True if `exc` is a Gemini rate-limit/quota error (HTTP 429) - the
    expected, frequent way a daily run gets cut short on the free tier,
    not a bug to crash on. See `api/main.py` for the same check used to
    turn this into a clean HTTP response instead of an opaque 500."""
    return isinstance(exc, genai_errors.APIError) and exc.code == 429


def is_transient_network_error(exc: Exception) -> bool:
    """True if `exc` is a network/transport-level failure (dropped
    connection, timeout, etc.) rather than a real bug. Confirmed live: an
    ablation run crashed with `httpx.RemoteProtocolError: Server
    disconnected without sending a response` after a ~12-minute gap between
    rows - almost certainly the machine went idle/sleep and the
    long-held HTTP connection was dropped by the time it woke up. This is
    exactly as recoverable as a quota 429 (just re-run to resume from the
    checkpoint), so it's handled the same way rather than crashing with a
    full traceback. `httpx.TransportError` is the right base class here -
    it covers RemoteProtocolError, ConnectError, ReadTimeout, ConnectTimeout,
    and PoolTimeout, all of which are transient by nature."""
    return isinstance(exc, httpx.TransportError)
