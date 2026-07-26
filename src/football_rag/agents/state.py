"""Shared state and structured intermediate models for the agent pipeline
(proposal §6.3). Every node in the LangGraph in `agents/graph.py` reads from
and writes to the same `AgentState`.
"""

from __future__ import annotations

from typing import Literal, Optional, TypedDict

from pydantic import BaseModel, Field


class StructuredQuery(BaseModel):
    """Normalised representation of the user's request (§6.3.1)."""

    intent: Literal["match_summary", "tactical_insight", "player_focus", "event_query", "other"]
    teams: list[str] = Field(default_factory=list)
    players: list[str] = Field(default_factory=list)
    event_types: list[str] = Field(default_factory=list)
    minute_min: Optional[int] = None
    minute_max: Optional[int] = None
    period: Optional[int] = None


class ReasoningInsight(BaseModel):
    """A single analytical finding produced by the Reasoning Agent (§6.3.4)."""

    insight: str
    supporting_chunk_ids: list[str] = Field(default_factory=list)


class EvaluationResult(BaseModel):
    """Verdict produced by the Evaluator Agent (§6.3.6)."""

    passed: bool
    relevance: float
    coherence: float
    factual_consistency: float
    feedback: str


class AgentState(TypedDict, total=False):
    match_id: str
    raw_query: str
    structured_query: StructuredQuery
    where_filter: Optional[dict]
    retrieved_chunks: list[dict]
    aggregated_context: list[dict]
    reasoning_insights: list[ReasoningInsight]
    response_text: str
    evaluation: Optional[EvaluationResult]
    retry_count: int
