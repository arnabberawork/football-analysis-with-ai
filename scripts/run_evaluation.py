"""Evaluation harness (proposal §6.11): naive-vs-agentic baseline comparison.

Runs both pipelines over `evaluation/dataset.py`'s STARTER_DATASET,
reporting retrieval metrics (P@5/R@5) and generation metrics (ROUGE-L/BLEU)
side by side. See `dataset.py`'s docstring for what this starter dataset is
(and isn't) a substitute for.

Usage:
    uv run python scripts/run_evaluation.py
"""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path

from football_rag.agents.graph import run_agentic_pipeline
from football_rag.chunking import build_hierarchical_chunks
from football_rag.evaluation.dataset import STARTER_DATASET, LabeledQuery, resolve_relevant_chunk_ids
from football_rag.evaluation.generation_metrics import bleu_score, rouge_scores
from football_rag.evaluation.retrieval_metrics import precision_at_k, recall_at_k
from football_rag.generation.generator import NaiveRAGGenerator
from football_rag.ingestion.statsbomb import load_match_events
from football_rag.logging_config import setup_logging
from football_rag.retrieval.vector_store import ChunkVectorStore

logger = logging.getLogger(__name__)

CACHE_DIR = Path("data/statsbomb_cache")
CHROMA_DIR = Path("data/chroma")
TOP_K = 5


def main() -> None:
    log_path = setup_logging("run_evaluation")
    logger.info("Logging to %s", log_path)

    store = ChunkVectorStore(persist_dir=CHROMA_DIR)
    naive_generator = NaiveRAGGenerator(vector_store=store)

    by_match: dict[int, list[LabeledQuery]] = defaultdict(list)
    for labeled_query in STARTER_DATASET:
        by_match[labeled_query.match_id].append(labeled_query)

    for match_id, labeled_queries in by_match.items():
        logger.info("=== Match %s ===", match_id)
        events = load_match_events(match_id, cache_dir=CACHE_DIR)
        chunks = build_hierarchical_chunks(events)
        store.ingest(chunks)

        for labeled_query in labeled_queries:
            relevant_ids = resolve_relevant_chunk_ids(labeled_query.relevance_criteria, chunks)
            logger.info("Query: %r (%d relevant chunks resolved)", labeled_query.query, len(relevant_ids))

            naive_result = naive_generator.generate(labeled_query.query, where={"match_id": str(match_id)})
            agentic_result = run_agentic_pipeline(str(match_id), labeled_query.query, store, chunks, events)

            for name, retrieved_chunks, response_text in [
                ("naive", naive_result["retrieved_chunks"], naive_result["response_text"]),
                ("agentic", agentic_result["retrieved_chunks"], agentic_result["response_text"]),
            ]:
                retrieved_ids = [c["chunk_id"] for c in retrieved_chunks]
                p_at_k = precision_at_k(retrieved_ids, relevant_ids, TOP_K)
                r_at_k = recall_at_k(retrieved_ids, relevant_ids, TOP_K)
                rouge = rouge_scores(response_text, labeled_query.reference_summary)
                bleu = bleu_score(response_text, labeled_query.reference_summary)
                logger.info(
                    "  [%-7s] P@%d=%.2f R@%d=%.2f rougeL=%.2f bleu=%.1f",
                    name, TOP_K, p_at_k, TOP_K, r_at_k, rouge["rougeL"], bleu,
                )

    logger.info("Log saved to %s", log_path)


if __name__ == "__main__":
    main()
