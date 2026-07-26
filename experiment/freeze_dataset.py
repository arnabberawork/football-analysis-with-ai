"""Stage 1 (freeze dataset) + stage 2 (hierarchical chunking) EDA pass.

Downloads and caches raw StatsBomb events for every match in
`experiment/matches.py::FROZEN_MATCHES` (reusing the same disk cache as
every other script in this repo - `data/statsbomb_cache/`), runs them
through `build_hierarchical_chunks()`, and writes two CSVs under
`experiment/results/` that are the direct input to the thesis's
Dataset/EDA section:

- `dataset_eda.csv` - one row per match: event counts (raw and after
  low-signal filtering), and chunk counts per tier.
- `event_type_distribution.csv` - one row per (match_id, event_type):
  counts, for a more detailed breakdown or a bar chart.

Quota-free - no Gemini calls, pure ingestion + chunking.

Usage:
    uv run python experiment/freeze_dataset.py
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from matches import FROZEN_MATCHES  # noqa: E402

from football_rag.chunking import build_hierarchical_chunks
from football_rag.ingestion.statsbomb import fetch_raw_events, parse_events

CACHE_DIR = Path("data/statsbomb_cache")
RESULTS_DIR = Path(__file__).resolve().parent / "results"


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    eda_rows: list[dict] = []
    distribution_rows: list[dict] = []

    for match in FROZEN_MATCHES:
        print(f"Match {match.match_id}: {match.home_team} {match.home_score}-{match.away_score} {match.away_team} ({match.stage})")
        raw_events = fetch_raw_events(match.match_id, cache_dir=CACHE_DIR)
        events = parse_events(raw_events, match.match_id)
        chunks = build_hierarchical_chunks(events, raw_events=raw_events)

        chunk_counts = Counter(c.metadata.chunk_level for c in chunks)
        event_type_counts = Counter(e.event_type for e in events)

        eda_rows.append(
            {
                "match_id": match.match_id,
                "stage": match.stage,
                "home_team": match.home_team,
                "away_team": match.away_team,
                "home_score": match.home_score,
                "away_score": match.away_score,
                "date": match.date,
                "went_to_penalties": match.went_to_penalties,
                "raw_event_count": len(raw_events),
                "parsed_event_count": len(events),
                "chunks_event": chunk_counts.get("event", 0),
                "chunks_time_window": chunk_counts.get("time_window", 0),
                "chunks_sequence": chunk_counts.get("sequence", 0),
                "chunks_lineup": chunk_counts.get("lineup", 0),
                "chunks_total": len(chunks),
            }
        )
        for event_type, count in event_type_counts.items():
            distribution_rows.append({"match_id": match.match_id, "event_type": event_type, "count": count})

        print(f"  {len(raw_events)} raw -> {len(events)} parsed events, {len(chunks)} chunks")

    eda_df = pd.DataFrame(eda_rows)
    eda_df.to_csv(RESULTS_DIR / "dataset_eda.csv", index=False)

    distribution_df = pd.DataFrame(distribution_rows)
    distribution_df.to_csv(RESULTS_DIR / "event_type_distribution.csv", index=False)

    print(f"\n{len(eda_rows)} matches frozen.")
    print(f"Total parsed events across corpus: {eda_df['parsed_event_count'].sum()}")
    print(f"Total chunks across corpus: {eda_df['chunks_total'].sum()}")
    print(f"Saved: {RESULTS_DIR / 'dataset_eda.csv'}")
    print(f"Saved: {RESULTS_DIR / 'event_type_distribution.csv'}")


if __name__ == "__main__":
    main()
