"""FastAPI service over the agentic pipeline (proposal §6.2.3, §6.3).

Exposes the same ingestion + 6-agent pipeline that `scripts/run_agentic_pipeline.py`
runs from the CLI, so `scripts/app.py` (Streamlit) can drive it purely over
HTTP instead of importing pipeline code directly.

Run via `uv run python scripts/run_api.py`.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from google.genai import errors as genai_errors

from football_rag.agents.formatting import format_pipeline_result
from football_rag.agents.graph import run_agentic_pipeline
from football_rag.api.schemas import PipelineRunRequest, PipelineRunResponse
from football_rag.chunking import build_hierarchical_chunks
from football_rag.ingestion.statsbomb import fetch_competitions, fetch_matches, load_match_events
from football_rag.output_writer import write_output
from football_rag.retrieval.vector_store import ChunkVectorStore
from football_rag.schema import Chunk

CACHE_DIR = Path("data/statsbomb_cache")
CHROMA_DIR = Path("data/chroma")

app = FastAPI(title="Football Agentic RAG API")

# Process-lifetime caches - the server-side equivalent of scripts/app.py's
# former @st.cache_resource/@st.cache_data: one vector store per process,
# one embed-and-ingest per distinct match_id.
_vector_store: ChunkVectorStore | None = None
_chunks_cache: dict[int, list[Chunk]] = {}


def get_vector_store() -> ChunkVectorStore:
    global _vector_store
    if _vector_store is None:
        _vector_store = ChunkVectorStore(persist_dir=CHROMA_DIR)
    return _vector_store


def get_or_ingest_chunks(match_id: int) -> list[Chunk]:
    if match_id not in _chunks_cache:
        events = load_match_events(match_id, cache_dir=CACHE_DIR)
        chunks = build_hierarchical_chunks(events)
        get_vector_store().ingest(chunks)
        _chunks_cache[match_id] = chunks
    return _chunks_cache[match_id]


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
    try:
        chunks = get_or_ingest_chunks(request.match_id)
        result = run_agentic_pipeline(
            str(request.match_id),
            request.query,
            get_vector_store(),
            chunks,
            enable_evaluator=request.enable_evaluator,
            max_retries=request.max_retries,
            top_n=request.top_n,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    except genai_errors.APIError as e:
        # Surfaces Gemini-side errors (rate limits, quota exhaustion, etc.)
        # as a real HTTP response instead of an opaque 500 - these are
        # expected/frequent on the free tier, not bugs.
        raise HTTPException(status_code=e.code or 502, detail=e.message) from e

    # Every call gets its own saved record, same as the CLI scripts -
    # write_output() (unlike tee_to_file()) doesn't touch global state, so
    # it's safe under concurrent requests.
    output_text = format_pipeline_result(str(request.match_id), request.query, result)
    output_path = write_output(f"api_pipeline_run_{request.match_id}", output_text)

    return PipelineRunResponse(**result, output_file=str(output_path))
