"""Chunking pipeline (proposal §6.6): converts `Event` objects into
retrievable `Chunk` objects across four granularity tiers - event-level
(`event_chunker.py`), time-window (`time_window_chunker.py`),
possession-sequence (`sequence_chunker.py`), and lineup/formation
(`lineup_chunker.py` - a deliberate fourth tier beyond the proposal's
original three, see its module docstring). `build_hierarchical_chunks()`
combines all of them into the single knowledge base both the naive
baseline and the agentic pipeline retrieve from.
"""

from __future__ import annotations

from football_rag.chunking.event_chunker import chunk_events
from football_rag.chunking.lineup_chunker import chunk_lineups
from football_rag.chunking.sequence_chunker import chunk_sequences
from football_rag.chunking.time_window_chunker import chunk_time_windows
from football_rag.schema import Chunk, Event


def build_hierarchical_chunks(events: list[Event], raw_events: list[dict] | None = None) -> list[Chunk]:
    """All chunking tiers combined - the standard ingestion call site.
    Chunk IDs across tiers are disjoint (`-ev-`/`-tw-`/`-seq-`/`-lineup-`),
    so `ChunkVectorStore.ingest()`'s upsert-by-chunk_id grows the same
    collection additively regardless of call order.

    `raw_events` (the pre-`parse_events()` dicts) is optional - lineup
    chunks are only produced when it's given, since formation/starting-XI
    data lives on the `Starting XI` event that `parse_events()` already
    filters out of `events` as low-signal.
    """
    chunks = chunk_events(events) + chunk_time_windows(events) + chunk_sequences(events)
    if raw_events is not None and events:
        chunks += chunk_lineups(int(events[0].match_id), raw_events)
    return chunks
