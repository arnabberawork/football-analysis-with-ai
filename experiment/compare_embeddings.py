"""Stage 3: MiniLM vs E5 retrieval-only comparison.

Embeds a representative subset of the corpus (`matches.py::
REPRESENTATIVE_SUBSET_MATCH_IDS` - the same 4 matches
`run_full_ablation.py` uses, for consistency) into two separate Chroma
collections (one per embedding model) and reports P@K/R@K for each, using
`query_dataset.py`'s relevance criteria. Dense-only (not the full hybrid
pipeline) - BM25 and the cross-encoder reranker are identical regardless
of embedder, so comparing dense retrieval in isolation is what actually
isolates the embedding model's contribution, per the proposal's §6.7
SBERT-vs-E5 discussion.

Confirmed live that embedding the *full* ~32,000-chunk corpus twice was
impractically slow on CPU-only hardware (a partial run projected multiple
hours); a representative subset keeps this quota-free step tractable
while still giving a directionally valid comparison - documented here
rather than silently narrowed. The 18-match corpus itself is still used
everywhere else (freeze_dataset.py, run_full_comparison.py) - only this
specific embedding-model comparison is scoped down.

Quota-free - no Gemini calls, pure local embedding + retrieval metrics.
Whichever embedder wins here is the one downstream generation stages use;
re-running full agentic generation twice (once per embedder) is out of
scope to protect the scarce Gemini quota for the actual naive-vs-agentic
comparison (see experiment/run_full_comparison.py).

Usage:
    uv run python experiment/compare_embeddings.py
"""

from __future__ import annotations

import shutil
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from matches import REPRESENTATIVE_SUBSET_MATCH_IDS  # noqa: E402
from query_dataset import EXPERIMENT_DATASET  # noqa: E402

from football_rag.chunking import build_hierarchical_chunks
from football_rag.embeddings.embedder import get_default_embedder, get_e5_embedder
from football_rag.evaluation.dataset import resolve_relevant_chunk_ids
from football_rag.evaluation.retrieval_metrics import precision_at_k, recall_at_k
from football_rag.ingestion.statsbomb import fetch_raw_events, parse_events
from football_rag.output_writer import tee_to_file
from football_rag.retrieval.vector_store import ChunkVectorStore

CACHE_DIR = Path("data/statsbomb_cache")
CHROMA_DIR = Path("data/chroma")
# RESULTS_DIR stays the base results/ folder (same "latest" fixed-path
# convention as main_comparison.csv/ablation.csv/thesis_tables.md) so
# make_tables.py's fixed read path keeps working; a separate, permanent
# timestamped snapshot is saved on top of that (see save_snapshot()) rather
# than replacing the fixed path entirely - an earlier version of this script
# wrote *only* to a timestamped folder, which silently broke make_tables.py
# (it kept reading a stale results/embedding_comparison.csv that no run
# ever updated again).
RESULTS_DIR = Path(__file__).resolve().parent / "results"
OUTPUT_PATH = RESULTS_DIR / "embedding_comparison.csv"
TOP_KS = [5, 10]

EMBEDDERS = {
    "minilm": ("football_chunks_minilm", get_default_embedder),
    "e5": ("football_chunks_e5", get_e5_embedder),
}


def main() -> None:
    with tee_to_file("compare_embeddings") as log_path:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)

        chunks_by_match: dict[int, list] = {}
        all_chunks = []
        for match_id in REPRESENTATIVE_SUBSET_MATCH_IDS:
            raw_events = fetch_raw_events(match_id, cache_dir=CACHE_DIR)
            events = parse_events(raw_events, match_id)
            chunks = build_hierarchical_chunks(events, raw_events=raw_events)
            chunks_by_match[match_id] = chunks
            all_chunks.extend(chunks)

        subset_queries = [lq for lq in EXPERIMENT_DATASET if lq.match_id in REPRESENTATIVE_SUBSET_MATCH_IDS]
        print(f"{len(all_chunks)} chunks across {len(REPRESENTATIVE_SUBSET_MATCH_IDS)} matches, {len(subset_queries)} queries")

        rows = []
        for embedder_name, (collection_name, get_embedder) in EMBEDDERS.items():
            print(f"\n=== {embedder_name} ===")
            store = ChunkVectorStore(persist_dir=CHROMA_DIR, collection_name=collection_name, embedder=get_embedder())
            store.ingest(all_chunks, skip_existing=True)
            print(f"  ingested {store.count()} chunks")

            scores_by_k: dict[int, dict[str, list[float]]] = {k: defaultdict(list) for k in TOP_KS}
            for lq in subset_queries:
                relevant_ids = resolve_relevant_chunk_ids(lq.relevance_criteria, chunks_by_match[lq.match_id])
                if not relevant_ids:
                    continue
                hits = store.query(lq.query, top_n=max(TOP_KS), where={"match_id": str(lq.match_id)})
                retrieved_ids = [h["chunk_id"] for h in hits]
                for k in TOP_KS:
                    scores_by_k[k]["precision"].append(precision_at_k(retrieved_ids, relevant_ids, k))
                    scores_by_k[k]["recall"].append(recall_at_k(retrieved_ids, relevant_ids, k))

            for k in TOP_KS:
                mean_p = sum(scores_by_k[k]["precision"]) / len(scores_by_k[k]["precision"])
                mean_r = sum(scores_by_k[k]["recall"]) / len(scores_by_k[k]["recall"])
                print(f"  P@{k}={mean_p:.3f}  R@{k}={mean_r:.3f}  (n={len(scores_by_k[k]['precision'])} queries)")
                rows.append({"embedder": embedder_name, "k": k, "mean_precision": mean_p, "mean_recall": mean_r, "n_queries": len(scores_by_k[k]["precision"])})

        df = pd.DataFrame(rows)
        df.to_csv(OUTPUT_PATH, index=False)
        print(f"\nSaved: {OUTPUT_PATH}")

        # Permanent, timestamped copy for auditing - same convention as
        # run_full_comparison.py/run_full_ablation.py/make_tables.py/
        # error_analysis.py's save_snapshot(). OUTPUT_PATH stays the single
        # "latest" file make_tables.py reads.
        snapshot_dir = RESULTS_DIR / f"compare_embeddings_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(OUTPUT_PATH, snapshot_dir / OUTPUT_PATH.name)
        print(f"Snapshot saved: {snapshot_dir / OUTPUT_PATH.name}")
    print(f"\nLog saved to {log_path}")


if __name__ == "__main__":
    main()
