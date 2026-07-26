"""Lineup/formation chunking - a fourth tier beyond the proposal's original
three (§6.6: event/time-window/sequence), added because "who started" and
"what formation did each team play" are otherwise structurally
unanswerable: `Starting XI` is filtered as low-signal for event-level
chunking (correctly - it has no per-event narrative value), which silently
discards the `tactics.formation`/lineup payload it carries. Reuses the raw
`Starting XI` events already present in the same `raw_events` list every
ingestion call site fetches - no new HTTP endpoint (`fetch_lineups()`
stays unused; its data mostly overlaps what event-level `Substitution`/
`Bad Behaviour`/`Foul Committed` chunks already capture).
"""

from __future__ import annotations

from football_rag.schema import Chunk, ChunkMetadata


def _format_formation(formation: int | None) -> str | None:
    """StatsBomb encodes a formation as concatenated per-line digits, e.g.
    433 -> "4-3-3", 4231 -> "4-2-3-1"."""
    if formation is None:
        return None
    return "-".join(str(formation))


def chunk_lineups(match_id: int, raw_events: list[dict]) -> list[Chunk]:
    """One chunk per team, built from that team's `Starting XI` raw event."""
    chunks: list[Chunk] = []
    starting_xi_events = [e for e in raw_events if e["type"]["name"] == "Starting XI"]

    for index, raw in enumerate(starting_xi_events):
        team = raw["team"]
        formation = _format_formation(raw.get("tactics", {}).get("formation"))
        lineup = raw.get("tactics", {}).get("lineup", [])
        players = ", ".join(
            f"{p['player']['name']} ({p['position']['name']}, #{p['jersey_number']})" for p in lineup
        )

        intro = f"{team['name']} started in a {formation} formation:" if formation else f"{team['name']} starting lineup:"
        text = f"{intro} {players}."

        metadata = ChunkMetadata(
            match_id=str(match_id),
            team_id=str(team["id"]),
            team_name=team["name"],
            minute_start=0,
            minute_end=0,
            period=1,
            chunk_level="lineup",
        )
        chunks.append(Chunk(chunk_id=f"{match_id}-lineup-{index}", text=text, metadata=metadata))

    return chunks
