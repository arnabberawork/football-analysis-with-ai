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

import sys
from pathlib import Path

from football_rag.chunking.event_chunker import chunk_events
from football_rag.generation.generator import NaiveRAGGenerator
from football_rag.ingestion.statsbomb import load_match_events
from football_rag.retrieval.vector_store import ChunkVectorStore

CACHE_DIR = Path("data/statsbomb_cache")
CHROMA_DIR = Path("data/chroma")

DEFAULT_MATCH_ID = 3869685  # 2022 FIFA World Cup Final: Argentina vs France
DEFAULT_QUERY = "Summarize the match and provide key tactical insights."


def main() -> None:
    match_id = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_MATCH_ID
    query = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_QUERY

    print(f"Loading events for match {match_id}...")
    events = load_match_events(match_id, cache_dir=CACHE_DIR)
    print(f"  {len(events)} events parsed")

    chunks = chunk_events(events)
    print(f"  {len(chunks)} event-level chunks generated")

    print("Ingesting chunks into vector store...")
    store = ChunkVectorStore(persist_dir=CHROMA_DIR)
    store.ingest(chunks)
    print(f"  vector store now holds {store.count()} chunks")

    print(f"\nQuery: {query}\n")
    generator = NaiveRAGGenerator(vector_store=store)
    result = generator.generate(query, where={"match_id": str(match_id)})

    print("Retrieved evidence:")
    for chunk in result["retrieved_chunks"]:
        print(f"  [{chunk['chunk_id']}] ({chunk['distance']:.3f}) {chunk['text']}")

    print("\nGenerated response:\n")
    print(result["response_text"])


if __name__ == "__main__":
    main()
