"""End-to-end vertical slice: one match, ingest -> chunk -> embed -> retrieve -> generate.

Usage:
    uv run python scripts/run_vertical_slice.py [match_id] ["your query"]

Defaults to the 2022 FIFA World Cup Final (Argentina vs France, match_id
3869685) and a match-summary query. This is the naive RAG baseline from
proposal §6.11.5 - no metadata filtering, no reasoning agent, no evaluator.
The full Agentic RAG pipeline is built on top of these same building blocks
in later phases.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from football_rag.chunking import build_hierarchical_chunks
from football_rag.generation.generator import NaiveRAGGenerator
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

    log_path = setup_logging(f"run_vertical_slice_{match_id}")
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

    logger.info("Query: %s", query)
    generator = NaiveRAGGenerator(vector_store=store)
    result = generator.generate(query, where={"match_id": str(match_id)})
    logger.info("Response generated (%d chars)", len(result["response_text"]))

    # The clean, final deliverable (evidence + generated summary) is saved
    # separately from the execution log via write_output() - not routed
    # through the logging formatter, so it stays a plain, readable text file
    # under outputs/ (correctly UTF-8 encoded even when the Windows console
    # itself mangles accented characters on screen), matching every other
    # script's saved-output convention.
    lines = [f"Query: {query}", "", "Retrieved evidence:"]
    for chunk in result["retrieved_chunks"]:
        lines.append(f"  [{chunk['chunk_id']}] ({chunk['distance']:.3f}) {chunk['text']}")
    lines += ["", "Generated response:", "", result["response_text"]]
    output_path = write_output(f"run_vertical_slice_{match_id}", "\n".join(lines))

    logger.info("Output saved to %s", output_path)
    logger.info("Log saved to %s", log_path)


if __name__ == "__main__":
    main()
