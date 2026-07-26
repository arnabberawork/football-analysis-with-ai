"""Thin HTTP client for scripts/app.py (Streamlit UI) to talk to the
FastAPI service in src/football_rag/api/main.py - scripts/app.py no longer
imports pipeline code directly, only this module.
"""

from __future__ import annotations

import os

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
