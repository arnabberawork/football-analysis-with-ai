"""Time-window chunking (proposal §6.6 — second chunking tier).

Groups events into fixed-size minute windows within each period, producing
one chunk per non-empty window that narrates everything that happened in
it. This is the tier a query like "what happened 20'-24'" needs — a single
event-level sentence can't give the generator that kind of narrative span.
"""

from __future__ import annotations

from collections import defaultdict

from football_rag.chunking.event_chunker import event_to_sentence
from football_rag.schema import Chunk, ChunkMetadata, Event

DEFAULT_WINDOW_MINUTES = 5


def chunk_time_windows(events: list[Event], window_minutes: int = DEFAULT_WINDOW_MINUTES) -> list[Chunk]:
    """Convert events into time-window chunks.

    Chunk IDs are `<match_id>-tw-<index>`, stable given a fixed (sorted)
    window order.
    """
    windows: dict[tuple[int, int], list[Event]] = defaultdict(list)
    for event in events:
        window_start = (event.minute // window_minutes) * window_minutes
        windows[(event.period, window_start)].append(event)

    chunks: list[Chunk] = []
    for index, (period, window_start) in enumerate(sorted(windows)):
        window_events = windows[(period, window_start)]
        match_id = window_events[0].match_id
        window_end = window_start + window_minutes - 1
        text = f"{window_start}'-{window_end}': " + " ".join(event_to_sentence(e) for e in window_events)
        metadata = ChunkMetadata(
            match_id=match_id,
            minute_start=window_start,
            minute_end=window_end,
            period=period,
            chunk_level="time_window",
        )
        chunks.append(Chunk(chunk_id=f"{match_id}-tw-{index}", text=text, metadata=metadata))
    return chunks
