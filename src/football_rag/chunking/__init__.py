"""Chunking pipeline (proposal §6.6): converts `Event` objects into
retrievable `Chunk` objects across three granularity tiers - event-level
(`event_chunker.py`), time-window (`time_window_chunker.py`), and
possession-sequence (`sequence_chunker.py`). `build_hierarchical_chunks()`
combines all three into the single knowledge base both the naive baseline
and the agentic pipeline retrieve from.
"""

from __future__ import annotations

from football_rag.chunking.event_chunker import chunk_events
from football_rag.chunking.sequence_chunker import chunk_sequences
from football_rag.chunking.time_window_chunker import chunk_time_windows
from football_rag.schema import Chunk, Event


def build_hierarchical_chunks(events: list[Event]) -> list[Chunk]:
    """All three chunking tiers combined - the standard ingestion call
    site. Chunk IDs across tiers are disjoint (`-ev-`/`-tw-`/`-seq-`), so
    `ChunkVectorStore.ingest()`'s upsert-by-chunk_id grows the same
    collection additively regardless of call order.
    """
    return chunk_events(events) + chunk_time_windows(events) + chunk_sequences(events)
