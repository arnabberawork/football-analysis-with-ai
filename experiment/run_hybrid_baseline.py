"""Stronger baseline: naive RAG with the agentic retrieval stack, no agents.

Answers the question the main comparison leaves open: how much of the
Agentic RAG gain comes from better retrieval, and how much from the agents?

"Hybrid naive RAG" uses exactly the same retrieval as the agentic pipeline
(`RetrieverAgent` with dense + BM25 retrieval, Reciprocal Rank Fusion,
cross-encoder re-ranking and salience forcing, top 10 chunks, scoped to
the match), followed by the same single-pass prompt and generation call
as `NaiveRAGGenerator`. It has no Query Understanding (so no
team/player/event filters), no Stats Agent routing, no Reasoning Agent and
no Evaluator.

It runs on the same 25 queries that both systems answered by retrieval in
the main comparison (rows of `results/main_comparison.csv` whose agentic
Precision@5 is not "N/A"), with the same relevance criteria and reference
summaries, so the three systems can be compared query by query.

Two stages, both resumable:
  1. Retrieval (no API calls): P@5/R@5/P@10/R@10 for all 25 queries ->
     results/hybrid_baseline_retrieval.csv. Finishes in one run.
  2. Generation (one Gemini call per query, 25 in total): response text,
     ROUGE, BLEU and BERTScore -> results/hybrid_baseline.csv. With the
     free-tier limit this may take two days: re-run until it prints
     "All 25 rows done." The model version Gemini reports is recorded per
     row, because GEMINI_MODEL may be an alias such as
     "gemini-flash-latest" whose underlying model changes over time.

Usage:
    uv run python experiment/run_hybrid_baseline.py
"""

from __future__ import annotations

import csv
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _checkpoint import CheckpointWriter, is_quota_error, is_transient_network_error, load_done_keys  # noqa: E402
from query_dataset import EXPERIMENT_DATASET  # noqa: E402

import pandas as pd
from google import genai

from football_rag.agents._llm import DEFAULT_MODEL, get_client
from football_rag.agents.retriever import RetrieverAgent
from football_rag.chunking import build_hierarchical_chunks
from football_rag.evaluation.dataset import resolve_relevant_chunk_ids
from football_rag.evaluation.generation_metrics import bertscore_per_example, bleu_score, rouge_scores
from football_rag.evaluation.retrieval_metrics import precision_at_k, recall_at_k
from football_rag.generation.prompts import SYSTEM_PROMPT, build_user_prompt
from football_rag.ingestion.statsbomb import fetch_raw_events, parse_events
from football_rag.logging_config import setup_logging
from football_rag.retrieval.vector_store import ChunkVectorStore

logger = logging.getLogger(__name__)

CACHE_DIR = Path("data/statsbomb_cache")
CHROMA_DIR = Path("data/chroma")
RESULTS_DIR = Path(__file__).resolve().parent / "results"
MAIN_CSV = RESULTS_DIR / "main_comparison.csv"
RETRIEVAL_CSV = RESULTS_DIR / "hybrid_baseline_retrieval.csv"
CSV_PATH = RESULTS_DIR / "hybrid_baseline.csv"
SYSTEM = "hybrid_naive"
TOP_N = 10
KEY_COLUMNS = ["match_id", "query", "system"]
RETRIEVAL_FIELDS = KEY_COLUMNS + ["retrieved_ids", "precision_at_5", "recall_at_5", "precision_at_10", "recall_at_10"]
FIELDNAMES = KEY_COLUMNS + [
    "reference_summary", "response_text", "precision_at_5", "recall_at_5", "precision_at_10", "recall_at_10",
    "rouge1", "rouge2", "rougeL", "bleu", "bertscore_f1", "model_version",
]


def common_queries() -> list:
    """The 25 (match, query) pairs both systems answered by retrieval."""
    with open(MAIN_CSV, encoding="utf-8") as fh:
        keys = {(r["match_id"], r["query"]) for r in csv.DictReader(fh)
                if r["system"] == "agentic" and r["precision_at_5"] not in ("N/A", "")}
    selected = [lq for lq in EXPERIMENT_DATASET if (str(lq.match_id), lq.query) in keys]
    if len(selected) != len(keys):
        raise SystemExit(f"Expected {len(keys)} queries from main_comparison.csv, matched {len(selected)} in the dataset.")
    return selected


def main() -> None:
    log_path = setup_logging("run_hybrid_baseline")
    logger.info("Logging to %s", log_path)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    queries = common_queries()
    logger.info("%d common retrieval queries", len(queries))

    store = ChunkVectorStore(persist_dir=CHROMA_DIR)
    chunks_by_match: dict[int, list] = {}
    for match_id in sorted({lq.match_id for lq in queries}):
        raw_events = fetch_raw_events(match_id, cache_dir=CACHE_DIR)
        chunks = build_hierarchical_chunks(parse_events(raw_events, match_id), raw_events=raw_events)
        store.ingest(chunks, skip_existing=True)
        chunks_by_match[match_id] = chunks

    retrievers: dict[int, RetrieverAgent] = {}

    def retrieve(lq) -> list[dict]:
        if lq.match_id not in retrievers:
            retrievers[lq.match_id] = RetrieverAgent(store, chunks_by_match[lq.match_id], enable_hybrid=True, enable_reranker=True)
        return retrievers[lq.match_id].retrieve(lq.query, where={"match_id": str(lq.match_id)}, top_n=TOP_N, ensure_salient=True)

    # ---- stage 1: retrieval only (no API calls)
    done = load_done_keys(RETRIEVAL_CSV, KEY_COLUMNS)
    writer = CheckpointWriter(RETRIEVAL_CSV, RETRIEVAL_FIELDS)
    try:
        for lq in queries:
            key = (str(lq.match_id), lq.query, SYSTEM)
            if key in done:
                continue
            relevant = resolve_relevant_chunk_ids(lq.relevance_criteria, chunks_by_match[lq.match_id])
            ids = [h["chunk_id"] for h in retrieve(lq)]
            writer.write({
                "match_id": str(lq.match_id), "query": lq.query, "system": SYSTEM, "retrieved_ids": " ".join(ids),
                "precision_at_5": precision_at_k(ids, relevant, 5), "recall_at_5": recall_at_k(ids, relevant, 5),
                "precision_at_10": precision_at_k(ids, relevant, 10), "recall_at_10": recall_at_k(ids, relevant, 10),
            })
            done.add(key)
    finally:
        writer.close()
    ret = pd.read_csv(RETRIEVAL_CSV)
    logger.info("Stage 1 done: mean P@5 %.3f, R@5 %.3f over %d queries", ret.precision_at_5.mean(), ret.recall_at_5.mean(), len(ret))

    # ---- stage 2: single-pass generation (one Gemini call per query)
    client = get_client()
    done = load_done_keys(CSV_PATH, KEY_COLUMNS)
    writer = CheckpointWriter(CSV_PATH, FIELDNAMES)
    try:
        for lq in queries:
            key = (str(lq.match_id), lq.query, SYSTEM)
            if key in done:
                continue
            relevant = resolve_relevant_chunk_ids(lq.relevance_criteria, chunks_by_match[lq.match_id])
            hits = retrieve(lq)
            ids = [h["chunk_id"] for h in hits]
            response = client.models.generate_content(
                model=DEFAULT_MODEL,
                contents=build_user_prompt(lq.query, hits),
                config=genai.types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT, max_output_tokens=4096),
            )
            text = response.text or ""
            rouge = rouge_scores(text, lq.reference_summary)
            writer.write({
                "match_id": str(lq.match_id), "query": lq.query, "system": SYSTEM,
                "reference_summary": lq.reference_summary, "response_text": text,
                "precision_at_5": precision_at_k(ids, relevant, 5), "recall_at_5": recall_at_k(ids, relevant, 5),
                "precision_at_10": precision_at_k(ids, relevant, 10), "recall_at_10": recall_at_k(ids, relevant, 10),
                "rouge1": rouge["rouge1"], "rouge2": rouge["rouge2"], "rougeL": rouge["rougeL"],
                "bleu": bleu_score(text, lq.reference_summary), "bertscore_f1": "",
                "model_version": getattr(response, "model_version", "") or DEFAULT_MODEL,
            })
            done.add(key)
            logger.info("[hybrid_naive] match=%s query=%r done (%d/%d)", lq.match_id, lq.query, len(done), len(queries))
    except Exception as e:
        if is_quota_error(e):
            logger.warning("Gemini quota hit - %d/%d generated. Re-run tomorrow to continue.", len(done), len(queries))
        elif is_transient_network_error(e):
            logger.warning("Transient network error (%s) - %d/%d generated. Re-run to continue.", type(e).__name__, len(done), len(queries))
        else:
            raise
    finally:
        writer.close()

    # ---- BERTScore (local, no API)
    if CSV_PATH.exists():
        df = pd.read_csv(CSV_PATH, keep_default_na=False)
        missing = df["bertscore_f1"] == ""
        if missing.any():
            scores = bertscore_per_example(df.loc[missing, "response_text"].tolist(), df.loc[missing, "reference_summary"].tolist())
            df.loc[missing, "bertscore_f1"] = [str(s) for s in scores]
            df.to_csv(CSV_PATH, index=False)
    if len(done) >= len(queries):
        logger.info("All %d rows done.", len(queries))
    else:
        logger.info("%d/%d rows generated so far - re-run to continue.", len(done), len(queries))
    logger.info("Log saved to %s", log_path)


if __name__ == "__main__":
    main()
