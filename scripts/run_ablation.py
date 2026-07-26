"""Ablation study (proposal §6.11.6).

Re-runs `evaluation/dataset.py`'s STARTER_DATASET through several agentic
pipeline configurations, each toggling one component off from the full
configuration, to isolate its contribution to retrieval precision and
generation quality.

Quota-heavy: 5 configs x N queries, several Gemini calls each - best run
once your Gemini free-tier daily quota has reset (see TESTING.md).

Usage:
    uv run python scripts/run_ablation.py
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from football_rag.agents.graph import run_agentic_pipeline
from football_rag.chunking import build_hierarchical_chunks
from football_rag.evaluation.dataset import STARTER_DATASET, LabeledQuery, resolve_relevant_chunk_ids
from football_rag.evaluation.generation_metrics import rouge_scores
from football_rag.evaluation.retrieval_metrics import precision_at_k
from football_rag.ingestion.statsbomb import load_match_events
from football_rag.retrieval.vector_store import ChunkVectorStore

CACHE_DIR = Path("data/statsbomb_cache")
CHROMA_DIR = Path("data/chroma")
TOP_K = 5

# Each config flips exactly one flag off from "full", so the delta versus
# "full" isolates that component's contribution.
CONFIGS: dict[str, dict] = {
    "full": dict(enable_hybrid=True, enable_reranker=True, enable_reasoning=True, enable_evaluator=True),
    "no_hybrid": dict(enable_hybrid=False, enable_reranker=True, enable_reasoning=True, enable_evaluator=True),
    "no_reranker": dict(enable_hybrid=True, enable_reranker=False, enable_reasoning=True, enable_evaluator=True),
    "no_reasoning": dict(enable_hybrid=True, enable_reranker=True, enable_reasoning=False, enable_evaluator=True),
    "no_evaluator": dict(enable_hybrid=True, enable_reranker=True, enable_reasoning=True, enable_evaluator=False),
}


def main() -> None:
    store = ChunkVectorStore(persist_dir=CHROMA_DIR)

    by_match: dict[int, list[LabeledQuery]] = defaultdict(list)
    for labeled_query in STARTER_DATASET:
        by_match[labeled_query.match_id].append(labeled_query)

    for match_id, labeled_queries in by_match.items():
        events = load_match_events(match_id, cache_dir=CACHE_DIR)
        chunks = build_hierarchical_chunks(events)
        store.ingest(chunks)

        for labeled_query in labeled_queries:
            relevant_ids = resolve_relevant_chunk_ids(labeled_query.relevance_criteria, chunks)
            print(f"\nQuery: {labeled_query.query!r}")
            for config_name, config in CONFIGS.items():
                result = run_agentic_pipeline(str(match_id), labeled_query.query, store, chunks, **config)
                retrieved_ids = [c["chunk_id"] for c in result["retrieved_chunks"]]
                p_at_k = precision_at_k(retrieved_ids, relevant_ids, TOP_K)
                rouge = rouge_scores(result["response_text"], labeled_query.reference_summary)
                print(f"  [{config_name:12s}] P@{TOP_K}={p_at_k:.2f} rougeL={rouge['rougeL']:.2f}")


if __name__ == "__main__":
    main()
