"""Possession-sequence chunking (proposal §6.6 — third chunking tier).

Groups events by StatsBomb's own `possession_id` (already on `Event`) into
one chunk per possession sequence - a team's uninterrupted spell in
control of the ball - giving move-level narrative rather than isolated
events.
"""

from __future__ import annotations

from collections import defaultdict

from football_rag.chunking.event_chunker import render_events
from football_rag.schema import Chunk, ChunkMetadata, Event


def chunk_sequences(events: list[Event]) -> list[Chunk]:
    """Convert events into possession-sequence chunks.

    Chunk IDs are `<match_id>-seq-<index>`, ordered by each sequence's
    first event. Events with no `possession_id` are skipped (already
    filtered to have one by the time they reach chunking, in practice).
    """
    sequences: dict[str, list[Event]] = defaultdict(list)
    for event in events:
        if event.possession_id is not None:
            sequences[event.possession_id].append(event)

    chunks: list[Chunk] = []
    ordered_ids = sorted(sequences, key=lambda pid: sequences[pid][0].minute)
    for index, possession_id in enumerate(ordered_ids):
        seq_events = sequences[possession_id]
        first, last = seq_events[0], seq_events[-1]
        team_names = {e.team_name for e in seq_events}
        text = render_events(seq_events)
        metadata = ChunkMetadata(
            match_id=first.match_id,
            team_name=first.team_name if len(team_names) == 1 else None,
            minute_start=first.minute,
            minute_end=last.minute,
            period=first.period,
            possession_id=possession_id,
            chunk_level="sequence",
        )
        chunks.append(Chunk(chunk_id=f"{first.match_id}-seq-{index}", text=text, metadata=metadata))
    return chunks
