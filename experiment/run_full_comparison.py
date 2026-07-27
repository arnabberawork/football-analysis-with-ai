"""Stages 9+10+13: the core naive-vs-agentic comparison, at full scale.

Resumable by design (see `_checkpoint.py`): re-run this once a day until
it prints "All queries done." `gemini-flash-latest` is shared by
`NaiveRAGGenerator` and the agentic pipeline's reasoning/generator/
evaluator agents - one naive-vs-agentic query pair costs ~4 calls against
the same free-tier daily cap (20 requests/day/model), so a ~60-query sweep
(2 systems each) takes roughly 8-12 days at this pace. Each completed
(match_id, query, system) row is written to
`experiment/results/main_comparison.csv` immediately, so stopping this
script (Ctrl-C, a quota 429, a crash) never loses more than the row in
flight, and the next run picks up exactly where it left off.

P@5/R@5/P@10/R@10 and ROUGE/BLEU are computed inline per row (cheap).
BERTScore is a separate batched pass at the end, over every row still
missing it - quota-free (no Gemini calls) so it always runs to
completion regardless of how the main loop above was cut short.

Usage:
    uv run python experiment/run_full_comparison.py
"""

from __future__ import annotations

import logging
import shutil
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _checkpoint import CheckpointWriter, is_quota_error, load_done_keys  # noqa: E402
from matches import FROZEN_MATCHES  # noqa: E402
from query_dataset import EXPERIMENT_DATASET  # noqa: E402

import pandas as pd

from football_rag.agents.graph import run_agentic_pipeline
from football_rag.chunking import build_hierarchical_chunks
from football_rag.evaluation.dataset import resolve_relevant_chunk_ids
from football_rag.evaluation.generation_metrics import bertscore_per_example, bleu_score, rouge_scores
from football_rag.evaluation.retrieval_metrics import precision_at_k, recall_at_k
from football_rag.generation.generator import NaiveRAGGenerator
from football_rag.ingestion.statsbomb import fetch_raw_events, parse_events
from football_rag.logging_config import setup_logging
from football_rag.retrieval.vector_store import ChunkVectorStore

logger = logging.getLogger(__name__)

CACHE_DIR = Path("data/statsbomb_cache")
CHROMA_DIR = Path("data/chroma")
RESULTS_DIR = Path(__file__).resolve().parent / "results"
CSV_PATH = RESULTS_DIR / "main_comparison.csv"
TOP_KS = [5, 10]
KEY_COLUMNS = ["match_id", "query", "system"]
FIELDNAMES = KEY_COLUMNS + [
    "reference_summary",
    "response_text",
    "precision_at_5",
    "recall_at_5",
    "precision_at_10",
    "recall_at_10",
    "rouge1",
    "rouge2",
    "rougeL",
    "bleu",
    "bertscore_f1",
]


def _metrics_row(
    match_id: int,
    query: str,
    system: str,
    reference_summary: str,
    retrieved_ids: list[str],
    relevant_ids: set[str],
    response_text: str,
    retrieval_not_applicable: bool = False,
) -> dict:
    rouge = rouge_scores(response_text, reference_summary)
    # StatsAgent-routed ("statistic" intent) responses bypass retrieval
    # entirely by design (agents/graph.py's stats_node returns
    # retrieved_chunks=[]) - scoring that as Precision/Recall = 0.0 would
    # unfairly penalise the agentic system for doing exactly what it's
    # supposed to do (answer with an exact deterministic count instead of
    # sampling top-N chunks). Recorded as "N/A" and excluded from
    # retrieval-metric aggregation in make_tables.py instead.
    if retrieval_not_applicable:
        precision_5 = recall_5 = precision_10 = recall_10 = "N/A"
    else:
        precision_5 = precision_at_k(retrieved_ids, relevant_ids, 5)
        recall_5 = recall_at_k(retrieved_ids, relevant_ids, 5)
        precision_10 = precision_at_k(retrieved_ids, relevant_ids, 10)
        recall_10 = recall_at_k(retrieved_ids, relevant_ids, 10)
    return {
        "match_id": str(match_id),
        "query": query,
        "system": system,
        "reference_summary": reference_summary,
        "response_text": response_text,
        "precision_at_5": precision_5,
        "recall_at_5": recall_5,
        "precision_at_10": precision_10,
        "recall_at_10": recall_10,
        "rouge1": rouge["rouge1"],
        "rouge2": rouge["rouge2"],
        "rougeL": rouge["rougeL"],
        "bleu": bleu_score(response_text, reference_summary),
        "bertscore_f1": "",
    }


def run_main_sweep() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    done = load_done_keys(CSV_PATH, KEY_COLUMNS)
    writer = CheckpointWriter(CSV_PATH, FIELDNAMES)

    store = ChunkVectorStore(persist_dir=CHROMA_DIR)
    naive_generator = NaiveRAGGenerator(vector_store=store)

    chunks_by_match: dict[int, list] = {}
    events_by_match: dict[int, list] = {}
    for match in FROZEN_MATCHES:
        raw_events = fetch_raw_events(match.match_id, cache_dir=CACHE_DIR)
        events = parse_events(raw_events, match.match_id)
        chunks = build_hierarchical_chunks(events, raw_events=raw_events)
        store.ingest(chunks, skip_existing=True)
        chunks_by_match[match.match_id] = chunks
        events_by_match[match.match_id] = events

    total = len(EXPERIMENT_DATASET) * 2
    completed_this_run = 0
    try:
        for lq in EXPERIMENT_DATASET:
            relevant_ids = resolve_relevant_chunk_ids(lq.relevance_criteria, chunks_by_match[lq.match_id])
            key_naive = (str(lq.match_id), lq.query, "naive")
            key_agentic = (str(lq.match_id), lq.query, "agentic")

            if key_naive not in done:
                logger.debug("Calling naive generator: match=%s query=%r", lq.match_id, lq.query)
                result = naive_generator.generate(lq.query, where={"match_id": str(lq.match_id)})
                retrieved_ids = [c["chunk_id"] for c in result["retrieved_chunks"]]
                writer.write(_metrics_row(lq.match_id, lq.query, "naive", lq.reference_summary, retrieved_ids, relevant_ids, result["response_text"]))
                done.add(key_naive)
                completed_this_run += 1
                logger.info("[naive]   match=%s query=%r done (%d/%d)", lq.match_id, lq.query, len(done), total)

            if key_agentic not in done:
                logger.debug("Calling agentic pipeline: match=%s query=%r", lq.match_id, lq.query)
                result = run_agentic_pipeline(str(lq.match_id), lq.query, store, chunks_by_match[lq.match_id], events_by_match[lq.match_id])
                retrieved_ids = [c["chunk_id"] for c in result["retrieved_chunks"]]
                is_statistic = result["structured_query"].intent == "statistic"
                writer.write(
                    _metrics_row(
                        lq.match_id,
                        lq.query,
                        "agentic",
                        lq.reference_summary,
                        retrieved_ids,
                        relevant_ids,
                        result["response_text"],
                        retrieval_not_applicable=is_statistic,
                    )
                )
                done.add(key_agentic)
                completed_this_run += 1
                logger.info("[agentic] match=%s query=%r done (%d/%d)", lq.match_id, lq.query, len(done), total)
    except Exception as e:
        if is_quota_error(e):
            logger.warning("Gemini quota hit for today - %d/%d done, %d completed this run.", len(done), total, completed_this_run)
            logger.warning("Re-run this script tomorrow (or once quota resets) to continue.")
        else:
            logger.exception("Unexpected error during main sweep (not a quota error) - %d/%d done", len(done), total)
            raise
    finally:
        writer.close()

    if len(done) >= total:
        logger.info("All %d rows done.", total)
    else:
        logger.info("%d/%d done so far - re-run to continue.", len(done), total)


def run_bertscore_pass() -> None:
    """Idempotent, quota-free: fills in `bertscore_f1` for any row that's
    missing it. Safe to call every time regardless of how the main sweep
    above was cut short."""
    if not CSV_PATH.exists():
        return
    df = pd.read_csv(CSV_PATH, keep_default_na=False)
    missing = df["bertscore_f1"] == ""
    if not missing.any():
        logger.info("BERTScore already computed for every row.")
        return

    logger.info("Computing BERTScore for %d row(s)...", missing.sum())
    candidates = df.loc[missing, "response_text"].tolist()
    references = df.loc[missing, "reference_summary"].tolist()
    per_example_f1 = bertscore_per_example(candidates, references)
    # pandas' default (pyarrow-backed) string dtype for a column read via
    # keep_default_na=False rejects raw floats on assignment - stringify
    # here since the value round-trips through CSV as text anyway, and
    # make_tables.py already does pd.to_numeric() when reading it back.
    df.loc[missing, "bertscore_f1"] = [str(f) for f in per_example_f1]
    df.to_csv(CSV_PATH, index=False)
    logger.info("Mean F1 for this batch: %.3f; per-row scores written to %s", sum(per_example_f1) / len(per_example_f1), CSV_PATH)


def save_snapshot() -> None:
    """Copies the current (possibly still-partial) main_comparison.csv into
    its own permanent, timestamped folder every time this script is run.

    The resumable checkpoint file (CSV_PATH) has to stay a single,
    continuously-appended file across many days of quota-limited runs -
    that's what load_done_keys()/CheckpointWriter above resume from, and it
    would be pointless to reset it per run. But that's an implementation
    detail, not something the user should have to dig through: every
    invocation of this script should leave behind its own permanent record
    of "what results existed as of this run", never silently overwritten by
    a later run and never deleted except by the user themselves.
    """
    if not CSV_PATH.exists():
        return

    snapshot_dir = RESULTS_DIR / f"main_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(CSV_PATH, snapshot_dir / CSV_PATH.name)
    logger.info("Snapshot saved: %s", snapshot_dir / CSV_PATH.name)


def main() -> None:
    log_path = setup_logging("run_full_comparison")
    logger.info("Logging to %s", log_path)
    run_main_sweep()
    run_bertscore_pass()
    save_snapshot()
    logger.info("Log saved to %s", log_path)


if __name__ == "__main__":
    main()
