"""Illustrative starter dataset for the evaluation harness.

**This is NOT a rigorous benchmark.** The proposal's evaluation design
(§6.4.2, §7.1) calls for SoccerNet's labeled annotations and real
human-written match reports as ground truth. Neither is pulled in here:
SoccerNet requires its own separate dataset access/download this project
doesn't have, and scraping published match reports raises copyright
concerns (verbatim reproduction, redistribution). Expanding real coverage
is a follow-up requiring that data access, not something this harness can
manufacture on its own.

What's here instead: a small, hand-written set of queries against the 2022
World Cup Final (match_id 3869685), each with a `relevance_criteria` list
(matched against chunk *text* and metadata, not hardcoded chunk_ids - see
`resolve_relevant_chunk_ids()` - so it stays valid even if chunking or
ingestion changes) and an originally-written reference summary based on
the public factual record of the match (final score, goal-scorers,
timing), not copied from any article.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel

from football_rag.schema import Chunk

WC_FINAL_MATCH_ID = 3869685


class RelevanceCriterion(BaseModel):
    """A chunk is relevant to a query if it matches ALL of the fields set
    here. `event_type`/`outcome`/`player_name` are matched as substrings
    against the chunk's rendered text (every chunk tier's text always
    contains these verbatim, per `event_chunker.event_to_sentence()`'s
    template); `minute_min`/`minute_max` are matched against the chunk's
    metadata minute range.
    """

    event_type: Optional[str] = None
    outcome: Optional[str] = None
    player_name: Optional[str] = None
    minute_min: Optional[int] = None
    minute_max: Optional[int] = None


class LabeledQuery(BaseModel):
    match_id: int
    query: str
    relevance_criteria: list[RelevanceCriterion]
    reference_summary: str


def resolve_relevant_chunk_ids(criteria: list[RelevanceCriterion], chunks: list[Chunk]) -> set[str]:
    """A chunk is relevant if it matches at least one criterion in the list
    (criteria are OR'd; fields within a single criterion are AND'd)."""
    relevant: set[str] = set()
    for chunk in chunks:
        for criterion in criteria:
            if criterion.event_type and criterion.event_type not in chunk.text:
                continue
            if criterion.outcome and criterion.outcome not in chunk.text:
                continue
            if criterion.player_name and criterion.player_name.lower() not in chunk.text.lower():
                continue
            if criterion.minute_min is not None and chunk.metadata.minute_end < criterion.minute_min:
                continue
            if criterion.minute_max is not None and chunk.metadata.minute_start > criterion.minute_max:
                continue
            relevant.add(chunk.chunk_id)
            break
    return relevant


STARTER_DATASET: list[LabeledQuery] = [
    LabeledQuery(
        match_id=WC_FINAL_MATCH_ID,
        query="Summarize the match and provide key tactical insights.",
        relevance_criteria=[RelevanceCriterion(event_type="Shot", outcome="Goal")],
        reference_summary=(
            "Argentina beat France on penalties after a 3-3 draw in the 2022 "
            "World Cup Final. Messi and Di Maria scored in normal time to put "
            "Argentina 2-0 up before Mbappe's second-half penalty and immediate "
            "follow-up goal levelled the match at 2-2. Messi restored "
            "Argentina's lead in extra time before Mbappe completed a hat-trick "
            "from the penalty spot to make it 3-3 and force a shootout, which "
            "Argentina won."
        ),
    ),
    LabeledQuery(
        match_id=WC_FINAL_MATCH_ID,
        query="What did Mbappe do in the second half?",
        relevance_criteria=[RelevanceCriterion(player_name="Mbapp", minute_min=45)],
        reference_summary=(
            "In the second half and extra time, Mbappe scored a penalty and an "
            "immediate follow-up goal to level the match at 2-2, then scored "
            "another penalty in extra time to make it 3-3 and force a shootout."
        ),
    ),
    LabeledQuery(
        match_id=WC_FINAL_MATCH_ID,
        query="Describe Messi's goals in the match.",
        relevance_criteria=[RelevanceCriterion(player_name="Messi", event_type="Shot", outcome="Goal")],
        reference_summary=(
            "Messi scored twice for Argentina: a first-half penalty to open "
            "the scoring, and a second goal in extra time to put Argentina "
            "back ahead at 3-2 after France had equalized."
        ),
    ),
    LabeledQuery(
        match_id=WC_FINAL_MATCH_ID,
        query="What happened in extra time?",
        relevance_criteria=[RelevanceCriterion(minute_min=90)],
        reference_summary=(
            "In extra time, Messi scored to put Argentina ahead 3-2, then "
            "Mbappe equalized again from the penalty spot to make it 3-3, "
            "sending the match to a penalty shootout."
        ),
    ),
    LabeledQuery(
        match_id=WC_FINAL_MATCH_ID,
        query="Were there any fouls or cards in the match?",
        relevance_criteria=[
            RelevanceCriterion(event_type="Foul Committed"),
            RelevanceCriterion(event_type="Bad Behaviour"),
        ],
        reference_summary=(
            "The match included multiple fouls from both sides typical of a "
            "high-stakes final, with players booked for cautionable challenges "
            "over the course of normal and extra time."
        ),
    ),
]
