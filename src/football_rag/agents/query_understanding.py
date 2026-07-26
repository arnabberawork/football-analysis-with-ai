"""Query Understanding Agent (proposal §6.3.1).

Interprets the user's request - identifying intent (match summary vs.
tactical insight vs. player focus, etc.) and extracting key entities (teams,
players, event types, timeframe) - and normalises them into a
`StructuredQuery` so downstream retrieval operates on a well-defined
representation instead of free text.
"""

from __future__ import annotations

from football_rag.agents._llm import DEFAULT_MODEL_LITE, call_structured, get_client
from football_rag.agents.state import StructuredQuery

MODEL = DEFAULT_MODEL_LITE

SYSTEM_PROMPT = (
    "You are a query analysis assistant for a football match analytics system. "
    "Extract the user's intent and any named entities from their request. "
    'Use intent="statistic" for any counting/aggregate question (e.g. "how many '
    'shots did Messi take", "how many yellow cards were shown", "how many fouls '
    'did each team commit") - these are answered by exact computation over every '
    "matching event, not by retrieving example passages, so they must be routed "
    'differently from a narrative question. When intent="statistic" and the '
    "question names a specific result (a goal, a card, a save, etc.), set "
    '"outcome" to that result (e.g. "Goal", "Yellow Card", "Saved") so the count '
    "can be scoped precisely.\n\n"
    "event_types MUST only use these exact StatsBomb event type strings - never "
    "invent a type that isn't in this list, even if it seems like an obvious name:\n"
    "Pass, Shot, Dribble, Duel, Dribbled Past, Interception, Ball Recovery, "
    "Clearance, Block, Shield, Foul Committed, Foul Won, Bad Behaviour, "
    "Substitution, Player On, Player Off, Offside, Injury Stoppage.\n"
    'There is no "Card" event type. Cards are recorded as the outcome of a '
    '"Foul Committed" event (a card given for a foul) or a "Bad Behaviour" event '
    "(a card not tied to a foul, e.g. dissent, time-wasting). A question about "
    'cards (yellow or red) must set event_types=["Foul Committed", "Bad Behaviour"] '
    'and outcome to the card name (e.g. "Yellow Card", "Red Card"). Goals are the '
    'outcome of a "Shot" event (outcome="Goal"), not a separate event type.'
)


class QueryUnderstandingAgent:
    def __init__(self, model: str = MODEL):
        self.model = model
        self._client = get_client()

    def understand(self, query: str) -> StructuredQuery:
        return call_structured(
            self._client,
            self.model,
            SYSTEM_PROMPT,
            f"User request: {query}",
            response_schema=StructuredQuery,
        )


def to_where_filter(structured: StructuredQuery) -> dict | None:
    """Map extracted entities to a Chroma `where` metadata filter over the
    fields in `schema.ChunkMetadata`. Only single-value equality is used for
    team/player/event_type (Chroma needs `$and`/`$in` for multi-value
    matches) - kept to the common case of one of each per query.
    """
    clauses: list[dict] = []
    if len(structured.teams) == 1:
        clauses.append({"team_name": structured.teams[0]})
    if len(structured.players) == 1:
        clauses.append({"player_name": structured.players[0]})
    if len(structured.event_types) == 1:
        clauses.append({"event_type": structured.event_types[0]})
    if structured.period is not None:
        clauses.append({"period": structured.period})
    if structured.minute_min is not None:
        clauses.append({"minute_start": {"$gte": structured.minute_min}})
    if structured.minute_max is not None:
        clauses.append({"minute_start": {"$lte": structured.minute_max}})

    if not clauses:
        return None
    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}
