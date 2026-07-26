"""Event-level chunking (proposal §6.4.4, §6.6 — the most granular tier).

Each individual event is converted into a single natural-language sentence
following the fixed template style specified in the proposal:

    "<minute>'[EventType] <Team> - <Player> <description> (x, y); outcome: <detail>."

e.g. "23' [Shot] Argentina - Messi took a shot (108.0, 40.0); outcome: Goal."

Fixed templates (rather than free LLM phrasing) keep the knowledge base
consistent, as called for in §6.6, and every chunk retains a link back to its
source event's metadata for retrieval-time filtering.
"""

from __future__ import annotations

from football_rag.schema import Chunk, ChunkMetadata, Event

# Maps a StatsBomb event type to a short verb phrase used in the sentence
# template. Event types not listed fall back to a generic "recorded a
# <type>" phrasing so the chunker never silently drops an event.
VERB_PHRASES: dict[str, str] = {
    "Pass": "played a pass",
    "Shot": "took a shot",
    "Foul Committed": "committed a foul",
    "Foul Won": "won a foul",
    "Dribble": "attempted a dribble",
    "Duel": "contested a duel",
    "Interception": "made an interception",
    "Substitution": "was substituted",
    "Bad Behaviour": "was booked",
    "Ball Recovery": "recovered the ball",
    "Clearance": "made a clearance",
    "Block": "blocked the ball",
    "Own Goal Against": "conceded an own goal",
    "Own Goal For": "scored an own goal",
}


def _verb_phrase(event_type: str) -> str:
    return VERB_PHRASES.get(event_type, f"recorded a {event_type.lower()}")


def event_to_sentence(event: Event) -> str:
    """Render a single Event as one fixed-template natural-language sentence."""
    time_str = f"{event.minute}'" + (f"{event.second:02d}" if event.second else "")
    location_str = f" at ({event.x:.1f}, {event.y:.1f})" if event.x is not None and event.y is not None else ""
    player = event.player_name or "Unknown player"

    parts = [f"{time_str} [{event.event_type}] {event.team_name} - {player} {_verb_phrase(event.event_type)}{location_str}"]

    if event.detail:
        parts.append(event.detail)
    if event.outcome:
        parts.append(f"outcome: {event.outcome}")

    return "; ".join(parts) + "."


MAX_EVENTS_PER_CHUNK = 12
"""Cap on events rendered into a single multi-event chunk (time-window,
sequence). A busy passing sequence or a dense 5-minute window can contain
dozens of raw events - left unbounded, a chunk can balloon past 10,000
characters, bloating the evidence prompt and risking the generator hitting
its output token budget (`finish_reason=MAX_TOKENS`) with little or no
visible text, since some Gemini models spend part of that budget on
internal reasoning before their visible answer. See TESTING.md."""


def render_events(events: list[Event], max_events: int = MAX_EVENTS_PER_CHUNK) -> str:
    """Render a list of events as joined sentences, capped to
    `max_events` with a note about any events omitted. Shared by
    `time_window_chunker.py` and `sequence_chunker.py` so both tiers get
    the same size bound."""
    rendered = [event_to_sentence(e) for e in events[:max_events]]
    text = " ".join(rendered)
    remaining = len(events) - max_events
    if remaining > 0:
        text += f" (+{remaining} more events omitted for brevity)"
    return text


def chunk_events(events: list[Event]) -> list[Chunk]:
    """Convert a list of parsed events into event-level chunks.

    Chunk IDs are `<match_id>-ev-<index>` so they are stable given a fixed
    (sorted) event order and traceable back to the source match.
    """
    chunks: list[Chunk] = []
    for index, event in enumerate(events):
        metadata = ChunkMetadata(
            match_id=event.match_id,
            team_id=event.team_id,
            team_name=event.team_name,
            player_id=event.player_id,
            player_name=event.player_name,
            minute_start=event.minute,
            minute_end=event.minute,
            period=event.period,
            event_type=event.event_type,
            possession_id=event.possession_id,
            chunk_level="event",
        )
        chunks.append(
            Chunk(
                chunk_id=f"{event.match_id}-ev-{index}",
                text=event_to_sentence(event),
                metadata=metadata,
            )
        )
    return chunks
