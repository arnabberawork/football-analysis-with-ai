"""FastAPI service over the agentic pipeline (proposal §6.2.3, §6.3).

Exposes the same ingestion + 6-agent pipeline that `scripts/run_agentic_pipeline.py`
runs from the CLI, so `scripts/app.py` (Streamlit) can drive it purely over
HTTP instead of importing pipeline code directly.

Run via `uv run python scripts/run_api.py`.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from google.genai import errors as genai_errors

from football_rag.agents.formatting import NODE_LABELS, format_pipeline_result, next_stage_hint, summarize_stage
from football_rag.agents.graph import run_agentic_pipeline, stream_agentic_pipeline
from football_rag.api.schemas import PipelineRunRequest, PipelineRunResponse
from football_rag.chunking import build_hierarchical_chunks
from football_rag.ingestion.statsbomb import fetch_competitions, fetch_matches, fetch_raw_events, parse_events
from football_rag.logging_config import setup_logging
from football_rag.output_writer import write_output
from football_rag.retrieval.vector_store import ChunkVectorStore
from football_rag.schema import Chunk, Event

# Configured once at import time (this module is loaded once per server
# process by uvicorn, unlike the CLI scripts' one-shot main()) - every
# request handled by this process logs into the same file.
log_path = setup_logging("api")
logger = logging.getLogger(__name__)
logger.info("API logging to %s", log_path)

CACHE_DIR = Path("data/statsbomb_cache")
CHROMA_DIR = Path("data/chroma")

app = FastAPI(title="Football Agentic RAG API")

# Process-lifetime caches - the server-side equivalent of scripts/app.py's
# former @st.cache_resource/@st.cache_data: one vector store per process,
# one embed-and-ingest per distinct match_id.
_vector_store: ChunkVectorStore | None = None
_chunks_cache: dict[int, list[Chunk]] = {}
_events_cache: dict[int, list[Event]] = {}


def get_vector_store() -> ChunkVectorStore:
    global _vector_store
    if _vector_store is None:
        _vector_store = ChunkVectorStore(persist_dir=CHROMA_DIR)
    return _vector_store


def get_or_ingest_chunks(match_id: int) -> tuple[list[Chunk], list[Event]]:
    """Returns (chunks, events) - events are needed by StatsAgent (via
    `run_agentic_pipeline`), which reads the full event list directly
    rather than a retrieved chunk sample."""
    if match_id not in _chunks_cache:
        logger.info("Ingesting match %s for the first time this process...", match_id)
        raw_events = fetch_raw_events(match_id, cache_dir=CACHE_DIR)
        events = parse_events(raw_events, match_id)
        chunks = build_hierarchical_chunks(events, raw_events=raw_events)
        get_vector_store().ingest(chunks)
        _chunks_cache[match_id] = chunks
        _events_cache[match_id] = events
        logger.info("Match %s ingested: %d chunks", match_id, len(chunks))
    return _chunks_cache[match_id], _events_cache[match_id]


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/competitions")
def competitions() -> list[dict]:
    return fetch_competitions(cache_dir=CACHE_DIR)


@app.get("/matches")
def matches(competition_id: int, season_id: int) -> list[dict]:
    return fetch_matches(competition_id, season_id, cache_dir=CACHE_DIR)


@app.post("/pipeline/run", response_model=PipelineRunResponse)
def run_pipeline(request: PipelineRunRequest) -> PipelineRunResponse:
    logger.info("POST /pipeline/run match=%s query=%r", request.match_id, request.query)
    try:
        chunks, events = get_or_ingest_chunks(request.match_id)
        result = run_agentic_pipeline(
            str(request.match_id),
            request.query,
            get_vector_store(),
            chunks,
            events,
            enable_evaluator=request.enable_evaluator,
            max_retries=request.max_retries,
            top_n=request.top_n,
        )
    except RuntimeError as e:
        logger.exception("Pipeline run failed (RuntimeError): match=%s query=%r", request.match_id, request.query)
        raise HTTPException(status_code=500, detail=str(e)) from e
    except genai_errors.APIError as e:
        # Surfaces Gemini-side errors (rate limits, quota exhaustion, etc.)
        # as a real HTTP response instead of an opaque 500 - these are
        # expected/frequent on the free tier, not bugs.
        logger.warning("Gemini API error (code=%s) for match=%s query=%r: %s", e.code, request.match_id, request.query, e.message)
        raise HTTPException(status_code=e.code or 502, detail=e.message) from e

    # Every call gets its own saved record, same as the CLI scripts -
    # write_output() doesn't touch global state (unlike stdout redirection),
    # so it's safe under concurrent requests.
    output_text = format_pipeline_result(str(request.match_id), request.query, result)
    output_path = write_output(f"api_pipeline_run_{request.match_id}", output_text)
    logger.info("Pipeline run complete: match=%s output_file=%s", request.match_id, output_path)

    return PipelineRunResponse(**result, output_file=str(output_path))


@app.post("/pipeline/run/stream")
def run_pipeline_stream(request: PipelineRunRequest) -> StreamingResponse:
    """Same pipeline as `/pipeline/run`, but streams one JSON line per agent
    stage as it completes (newline-delimited JSON) instead of blocking until
    the whole run finishes - lets a UI (`scripts/app.py`) show live,
    agent-by-agent progress. Each line is one of:
      {"event": "stage", "stage": <node name>, "label": <human label>, "detail": <one-line summary>, "next": <node name or null>}
      {"event": "error", "detail": <message>}
      {"event": "done", "result": <same shape as /pipeline/run's response body>}
    `next` is the node expected to run immediately after this one (or
    `null` if the graph ends here) - see `next_stage_hint()` - so a client
    can highlight which branch (retrieval vs. StatsAgent) or retry loop is
    coming next without parsing `detail`'s free text.
    """
    logger.info("POST /pipeline/run/stream match=%s query=%r", request.match_id, request.query)

    def event_stream():
        try:
            if request.match_id not in _chunks_cache:
                yield json.dumps(
                    {
                        "event": "stage",
                        "stage": "ingest",
                        "label": NODE_LABELS["ingest"],
                        "detail": "first run for this match - fetching events and embedding chunks (may take ~1 min)",
                        "next": "query_understanding",
                    }
                ) + "\n"
            chunks, events = get_or_ingest_chunks(request.match_id)

            final_state: dict = {}
            for node_name, state in stream_agentic_pipeline(
                str(request.match_id),
                request.query,
                get_vector_store(),
                chunks,
                events,
                enable_evaluator=request.enable_evaluator,
                max_retries=request.max_retries,
                top_n=request.top_n,
            ):
                final_state = state
                yield json.dumps(
                    {
                        "event": "stage",
                        "stage": node_name,
                        "label": NODE_LABELS.get(node_name, node_name),
                        "detail": summarize_stage(node_name, state),
                        "next": next_stage_hint(node_name, state, enable_evaluator=request.enable_evaluator),
                    }
                ) + "\n"
        except RuntimeError as e:
            logger.exception("Pipeline stream failed (RuntimeError): match=%s query=%r", request.match_id, request.query)
            yield json.dumps({"event": "error", "detail": str(e)}) + "\n"
            return
        except genai_errors.APIError as e:
            logger.warning(
                "Gemini API error (code=%s) for match=%s query=%r: %s", e.code, request.match_id, request.query, e.message
            )
            yield json.dumps({"event": "error", "detail": e.message}) + "\n"
            return

        output_text = format_pipeline_result(str(request.match_id), request.query, final_state)
        output_path = write_output(f"api_pipeline_run_{request.match_id}", output_text)
        logger.info("Pipeline run complete: match=%s output_file=%s", request.match_id, output_path)
        response = PipelineRunResponse(**final_state, output_file=str(output_path))
        yield json.dumps({"event": "done", "result": response.model_dump(mode="json")}) + "\n"

    return StreamingResponse(event_stream(), media_type="application/x-ndjson")
