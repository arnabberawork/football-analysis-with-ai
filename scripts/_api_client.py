"""Thin HTTP client for scripts/app.py (Streamlit UI) to talk to the
FastAPI service in src/football_rag/api/main.py - scripts/app.py no longer
imports pipeline code directly, only this module.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator

import requests

API_BASE_URL = os.environ.get("FOOTBALL_RAG_API_URL", "http://localhost:8000")

# First-time ingestion (~1 min) plus up to 4 Gemini calls can take a while;
# generous timeout so a slow first run doesn't look like a hung request.
RUN_PIPELINE_TIMEOUT = 300


def _raise_for_status(response: requests.Response) -> None:
    if response.ok:
        return
    try:
        detail = response.json().get("detail", response.text)
    except ValueError:
        detail = response.text
    raise RuntimeError(detail)


def get_competitions() -> list[dict]:
    response = requests.get(f"{API_BASE_URL}/competitions", timeout=30)
    _raise_for_status(response)
    return response.json()


def get_matches(competition_id: int, season_id: int) -> list[dict]:
    response = requests.get(
        f"{API_BASE_URL}/matches",
        params={"competition_id": competition_id, "season_id": season_id},
        timeout=30,
    )
    _raise_for_status(response)
    return response.json()


def run_pipeline(
    match_id: int,
    query: str,
    enable_evaluator: bool = True,
    max_retries: int = 1,
) -> dict:
    response = requests.post(
        f"{API_BASE_URL}/pipeline/run",
        json={
            "match_id": match_id,
            "query": query,
            "enable_evaluator": enable_evaluator,
            "max_retries": max_retries,
        },
        timeout=RUN_PIPELINE_TIMEOUT,
    )
    _raise_for_status(response)
    return response.json()


def stream_pipeline(
    match_id: int,
    query: str,
    enable_evaluator: bool = True,
    max_retries: int = 1,
) -> Iterator[dict]:
    """Generator over `/pipeline/run/stream`'s newline-delimited JSON events,
    one per completed agent stage, so a caller (`scripts/app.py`) can update
    a live progress display instead of blocking on a single `run_pipeline()`
    call. Raises `RuntimeError` immediately if the pipeline itself reports a
    mid-stream error (the HTTP status is already 200 by the time streaming
    starts, so that can't be signalled via the status code the way
    `_raise_for_status` normally does)."""
    with requests.post(
        f"{API_BASE_URL}/pipeline/run/stream",
        json={
            "match_id": match_id,
            "query": query,
            "enable_evaluator": enable_evaluator,
            "max_retries": max_retries,
        },
        timeout=RUN_PIPELINE_TIMEOUT,
        stream=True,
    ) as response:
        _raise_for_status(response)
        for line in response.iter_lines(decode_unicode=True):
            if not line:
                continue
            payload = json.loads(line)
            if payload.get("event") == "error":
                raise RuntimeError(payload.get("detail", "Pipeline run failed"))
            yield payload
