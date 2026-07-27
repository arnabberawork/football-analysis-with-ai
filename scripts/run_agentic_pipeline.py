"""End-to-end run of the full 6-agent Agentic RAG pipeline (proposal §6.2.3,
§6.3): query_understanding -> retriever -> context_aggregator -> reasoning
-> generator -> evaluator.

Usage:
    uv run python scripts/run_agentic_pipeline.py [match_id] ["your query"]

Same ingestion steps as scripts/run_vertical_slice.py (that script remains
the naive RAG baseline, per CLAUDE.md, and is unaffected by this one).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from football_rag.agents.formatting import format_pipeline_result
from football_rag.agents.graph import run_agentic_pipeline
from football_rag.chunking import build_hierarchical_chunks
from football_rag.ingestion.statsbomb import fetch_raw_events, parse_events
from football_rag.logging_config import setup_logging
from football_rag.output_writer import write_output
from football_rag.retrieval.vector_store import ChunkVectorStore

logger = logging.getLogger(__name__)

CACHE_DIR = Path("data/statsbomb_cache")
CHROMA_DIR = Path("data/chroma")

DEFAULT_MATCH_ID = 3869685  # 2022 FIFA World Cup Final: Argentina vs France
DEFAULT_QUERY = "Summarize the match and provide key tactical insights."


def main() -> None:
    match_id = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_MATCH_ID
    query = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_QUERY

    log_path = setup_logging(f"run_agentic_pipeline_{match_id}")
    logger.info("Logging to %s", log_path)

    logger.info("Loading events for match %s...", match_id)
    raw_events = fetch_raw_events(match_id, cache_dir=CACHE_DIR)
    events = parse_events(raw_events, match_id)
    logger.info("  %d events parsed", len(events))

    chunks = build_hierarchical_chunks(events, raw_events=raw_events)
    logger.info("  %d chunks generated (event + time-window + sequence + lineup tiers)", len(chunks))

    logger.info("Ingesting chunks into vector store...")
    store = ChunkVectorStore(persist_dir=CHROMA_DIR)
    store.ingest(chunks)
    logger.info("  vector store now holds %d chunks", store.count())

    result = run_agentic_pipeline(str(match_id), query, store, chunks, events)
    output_text = format_pipeline_result(str(match_id), query, result)

    # Clean, final deliverable saved separately from the execution log - see
    # run_vertical_slice.py's identical rationale.
    output_path = write_output(f"run_agentic_pipeline_{match_id}", output_text)
    logger.info("Output saved to %s", output_path)
    logger.info("Log saved to %s", log_path)


if __name__ == "__main__":
    main()
