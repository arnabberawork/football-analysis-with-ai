"""End-to-end run of the full 6-agent Agentic RAG pipeline (proposal §6.2.3,
§6.3): query_understanding -> retriever -> context_aggregator -> reasoning
-> generator -> evaluator.

Usage:
    uv run python scripts/run_agentic_pipeline.py [match_id] ["your query"]

Same ingestion steps as scripts/run_vertical_slice.py (that script remains
the naive RAG baseline, per CLAUDE.md, and is unaffected by this one).
"""

from __future__ import annotations

import sys
from pathlib import Path

from _output_writer import tee_to_file

from football_rag.agents.graph import run_agentic_pipeline
from football_rag.chunking import build_hierarchical_chunks
from football_rag.ingestion.statsbomb import load_match_events
from football_rag.retrieval.vector_store import ChunkVectorStore

CACHE_DIR = Path("data/statsbomb_cache")
CHROMA_DIR = Path("data/chroma")

DEFAULT_MATCH_ID = 3869685  # 2022 FIFA World Cup Final: Argentina vs France
DEFAULT_QUERY = "Summarize the match and provide key tactical insights."


def main() -> None:
    match_id = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_MATCH_ID
    query = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_QUERY

    with tee_to_file(f"run_agentic_pipeline_{match_id}") as output_path:
        print(f"Loading events for match {match_id}...")
        events = load_match_events(match_id, cache_dir=CACHE_DIR)
        print(f"  {len(events)} events parsed")

        chunks = build_hierarchical_chunks(events)
        print(f"  {len(chunks)} chunks generated (event + time-window + sequence tiers)")

        print("Ingesting chunks into vector store...")
        store = ChunkVectorStore(persist_dir=CHROMA_DIR)
        store.ingest(chunks)
        print(f"  vector store now holds {store.count()} chunks")

        print(f"\nQuery: {query}\n")
        result = run_agentic_pipeline(str(match_id), query, store, chunks)

        print("1. Query Understanding:")
        print(f"   {result['structured_query']!r}")
        print(f"   where filter: {result.get('where_filter')}")

        print("\n2. Retriever (fused dense + BM25 hits, re-ranked):")
        for hit in result["retrieved_chunks"]:
            score = hit.get("rerank_score", hit.get("fusion_score"))
            print(f"   [{hit['chunk_id']}] ({score:.4f}) {hit['text']}")

        print(f"\n3. Context Aggregator ({len(result['aggregated_context'])} deduplicated chunks kept)")

        print("\n4. Reasoning insights:")
        for insight in result["reasoning_insights"]:
            print(f"   - {insight.insight} (evidence: {', '.join(insight.supporting_chunk_ids)})")

        print("\n5. Generated response:\n")
        print(result["response_text"])

        evaluation = result.get("evaluation")
        if evaluation is not None:
            print("\n6. Evaluator verdict:")
            print(
                f"   passed={evaluation.passed} relevance={evaluation.relevance:.2f} "
                f"coherence={evaluation.coherence:.2f} factual_consistency={evaluation.factual_consistency:.2f}"
            )
            print(f"   feedback: {evaluation.feedback}")

    print(f"\nOutput saved to {output_path}")


if __name__ == "__main__":
    main()
