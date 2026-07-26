"""Stage 12: ablation study, at research scale (but not full scale).

Same 5 configs as `scripts/run_ablation.py` (each toggles exactly one
component off from "full"), but the 5x multiplier over ~60 queries makes
the *entire* query set infeasible within reasonable Gemini free-tier
quota timelines (20 requests/day/model - see `experiment/README.md`).
Instead this runs on a **representative subset of 4 matches, one per
major tournament stage** (Final, Quarter-final, Round of 16, Group
Stage - chosen for a spread of scorelines: a 3-3 classic, a 1-1
penalties game, a 6-1 blowout, a 7-0 blowout), taking every query
generated for those 4 matches (~14-15 queries). This is a standard,
defensible thesis practice for an ablation - documented here rather than
silently narrowed.

Resumable exactly like `run_full_comparison.py` (see `_checkpoint.py`):
re-run once a day until it reports done.

Usage:
    uv run python experiment/run_full_ablation.py
"""

from __future__ import annotations

import shutil
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _checkpoint import CheckpointWriter, is_quota_error, load_done_keys  # noqa: E402
from matches import REPRESENTATIVE_SUBSET_MATCH_IDS  # noqa: E402
from query_dataset import EXPERIMENT_DATASET  # noqa: E402

from football_rag.agents.graph import run_agentic_pipeline
from football_rag.chunking import build_hierarchical_chunks
from football_rag.evaluation.dataset import resolve_relevant_chunk_ids
from football_rag.evaluation.generation_metrics import rouge_scores
from football_rag.evaluation.retrieval_metrics import precision_at_k
from football_rag.ingestion.statsbomb import fetch_raw_events, parse_events
from football_rag.retrieval.vector_store import ChunkVectorStore

CACHE_DIR = Path("data/statsbomb_cache")
CHROMA_DIR = Path("data/chroma")
RESULTS_DIR = Path(__file__).resolve().parent / "results"
CSV_PATH = RESULTS_DIR / "ablation.csv"
TOP_K = 5
KEY_COLUMNS = ["match_id", "query", "config"]
FIELDNAMES = KEY_COLUMNS + ["precision_at_5", "rougeL", "response_text"]

CONFIGS: dict[str, dict] = {
    "full": dict(enable_hybrid=True, enable_reranker=True, enable_reasoning=True, enable_evaluator=True),
    "no_hybrid": dict(enable_hybrid=False, enable_reranker=True, enable_reasoning=True, enable_evaluator=True),
    "no_reranker": dict(enable_hybrid=True, enable_reranker=False, enable_reasoning=True, enable_evaluator=True),
    "no_reasoning": dict(enable_hybrid=True, enable_reranker=True, enable_reasoning=False, enable_evaluator=True),
    "no_evaluator": dict(enable_hybrid=True, enable_reranker=True, enable_reasoning=True, enable_evaluator=False),
}


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    done = load_done_keys(CSV_PATH, KEY_COLUMNS)
    writer = CheckpointWriter(CSV_PATH, FIELDNAMES)

    store = ChunkVectorStore(persist_dir=CHROMA_DIR)
    subset = [lq for lq in EXPERIMENT_DATASET if lq.match_id in REPRESENTATIVE_SUBSET_MATCH_IDS]
    print(f"Ablation subset: {len(subset)} queries across {len(REPRESENTATIVE_SUBSET_MATCH_IDS)} matches x {len(CONFIGS)} configs = {len(subset) * len(CONFIGS)} rows")

    chunks_by_match: dict[int, list] = {}
    events_by_match: dict[int, list] = {}
    for match_id in REPRESENTATIVE_SUBSET_MATCH_IDS:
        raw_events = fetch_raw_events(match_id, cache_dir=CACHE_DIR)
        events = parse_events(raw_events, match_id)
        chunks = build_hierarchical_chunks(events, raw_events=raw_events)
        store.ingest(chunks, skip_existing=True)
        chunks_by_match[match_id] = chunks
        events_by_match[match_id] = events

    total = len(subset) * len(CONFIGS)
    try:
        for lq in subset:
            relevant_ids = resolve_relevant_chunk_ids(lq.relevance_criteria, chunks_by_match[lq.match_id])
            for config_name, config in CONFIGS.items():
                key = (str(lq.match_id), lq.query, config_name)
                if key in done:
                    continue
                result = run_agentic_pipeline(
                    str(lq.match_id), lq.query, store, chunks_by_match[lq.match_id], events_by_match[lq.match_id], **config
                )
                retrieved_ids = [c["chunk_id"] for c in result["retrieved_chunks"]]
                p_at_5 = precision_at_k(retrieved_ids, relevant_ids, TOP_K)
                rouge = rouge_scores(result["response_text"], lq.reference_summary)
                writer.write(
                    {
                        "match_id": str(lq.match_id),
                        "query": lq.query,
                        "config": config_name,
                        "precision_at_5": p_at_5,
                        "rougeL": rouge["rougeL"],
                        "response_text": result["response_text"],
                    }
                )
                done.add(key)
                print(f"  [{config_name:12s}] match={lq.match_id} query={lq.query!r} P@5={p_at_5:.2f} rougeL={rouge['rougeL']:.2f} ({len(done)}/{total})")
    except Exception as e:
        if is_quota_error(e):
            print(f"\nGemini quota hit for today - {len(done)}/{total} done.")
            print("Re-run this script tomorrow (or once quota resets) to continue.")
        else:
            raise
    finally:
        writer.close()

    if len(done) >= total:
        print(f"\nAll {total} rows done.")
    else:
        print(f"\n{len(done)}/{total} done so far - re-run to continue.")

    save_snapshot()


def save_snapshot() -> None:
    """Copies the current (possibly still-partial) ablation.csv into its own
    permanent, timestamped folder every time this script is run - same
    rationale as run_full_comparison.py::save_snapshot(): CSV_PATH itself
    has to stay one continuously-appended file for the resumable
    load_done_keys()/CheckpointWriter logic above to work across multiple
    quota-limited days, but every invocation should still leave behind its
    own permanent, never-overwritten record of "what existed as of this
    run"."""
    if not CSV_PATH.exists():
        return

    snapshot_dir = RESULTS_DIR / f"ablation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(CSV_PATH, snapshot_dir / CSV_PATH.name)
    print(f"Snapshot saved: {snapshot_dir / CSV_PATH.name}")


if __name__ == "__main__":
    main()
