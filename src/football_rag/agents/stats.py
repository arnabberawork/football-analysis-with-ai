"""Stats Agent - not part of the proposal's original 6-agent design
(§6.2.3); added because RAG retrieval structurally cannot answer
counting/aggregate questions reliably (it samples top-N chunks, never "all"
of anything). Routed to directly when Query Understanding sets
`intent="statistic"` (see `agents/graph.py`), bypassing
retrieval/aggregation/reasoning/generation/evaluation entirely.

No LLM call: the answer is computed deterministically by filtering the
full event list and formatting the result directly, so an exact count can
never be altered by generation - and it costs zero Gemini quota.
"""

from __future__ import annotations

from collections import defaultdict

from football_rag.agents.state import StructuredQuery
from football_rag.schema import Event

MAX_DETAIL_EVENTS = 20
"""Cap on how many matching events get listed individually - a broad
count (e.g. "how many passes") can match hundreds of events; the total
count is always exact regardless, only the itemized detail list is capped
(same reasoning as chunking's MAX_EVENTS_PER_CHUNK - an unbounded listing
isn't useful and just bloats the response)."""

SHOOTOUT_PERIOD = 5
_OWN_GOAL_EVENT_TYPES = {"Own Goal Against", "Own Goal For"}


def _is_goal_count_query(structured: StructuredQuery) -> bool:
    """True for statistic queries counting goals "scored in the match" (e.g.
    "how many goals did each team score") - these must exclude the penalty
    shootout (StatsBomb period 5). Confirmed live: a shootout penalty is
    encoded as an ordinary Shot/outcome="Goal" event indistinguishable from
    a regulation-time goal, so without this exclusion StatsAgent counted
    Argentina 7-5 France for the 2022 WC Final instead of the real 3-3,
    silently including six shootout kicks in "goals scored in the match".
    Mirrors the same period > 4 exclusion already applied when building
    evaluation reference data
    (experiment/build_query_dataset.py::_goal_moments()). Only kicks in when
    the user hasn't already explicitly scoped the query to one period
    themselves - if they ask about period 5 specifically, that's the
    shootout and should not be filtered back out.
    """
    if structured.period is not None:
        return False
    outcome_is_goal = structured.outcome is not None and "goal" in structured.outcome.lower()
    event_types = set(structured.event_types or [])
    return outcome_is_goal or bool(event_types & _OWN_GOAL_EVENT_TYPES)


def _matches(event: Event, structured: StructuredQuery, exclude_shootout: bool) -> bool:
    if exclude_shootout and event.period == SHOOTOUT_PERIOD:
        return False
    if structured.teams and not any(t.lower() in event.team_name.lower() for t in structured.teams):
        return False
    if structured.players:
        if event.player_name is None or not any(p.lower() in event.player_name.lower() for p in structured.players):
            return False
    if structured.event_types and event.event_type not in structured.event_types:
        return False
    if structured.outcome:
        if event.outcome is None or structured.outcome.lower() not in event.outcome.lower():
            return False
    if structured.minute_min is not None and event.minute < structured.minute_min:
        return False
    if structured.minute_max is not None and event.minute > structured.minute_max:
        return False
    if structured.period is not None and event.period != structured.period:
        return False
    return True


class StatsAgent:
    def answer(self, query: str, structured: StructuredQuery, events: list[Event]) -> str:
        exclude_shootout = _is_goal_count_query(structured)
        matches = [e for e in events if _matches(e, structured, exclude_shootout)]
        if not matches:
            return f'No matching events found for "{query}".'

        by_team: dict[str, int] = defaultdict(int)
        by_player: dict[str, int] = defaultdict(int)
        for e in matches:
            by_team[e.team_name] += 1
            if e.player_name:
                by_player[e.player_name] += 1

        lines = [f"{len(matches)} matching event(s) found."]
        lines.append("")
        lines.append("By team: " + ", ".join(f"{team}: {count}" for team, count in sorted(by_team.items())))
        if by_player and (structured.players or len(by_player) > 1):
            by_player_sorted = sorted(by_player.items(), key=lambda kv: -kv[1])
            lines.append("By player: " + ", ".join(f"{name}: {count}" for name, count in by_player_sorted))

        lines.append("")
        lines.append("Details:")
        for e in matches[:MAX_DETAIL_EVENTS]:
            time_str = f"{e.minute}'" + (f"{e.second:02d}" if e.second else "")
            detail_str = "; ".join(p for p in (e.detail, e.outcome) if p)
            suffix = f" {detail_str}" if detail_str else ""
            lines.append(f"  {time_str} {e.team_name} - {e.player_name or 'Unknown'} [{e.event_type}]{suffix}")
        if len(matches) > MAX_DETAIL_EVENTS:
            lines.append(f"  ... and {len(matches) - MAX_DETAIL_EVENTS} more (already included in the count above)")

        return "\n".join(lines)
