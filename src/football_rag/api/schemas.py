"""Request/response models for the API (`api/main.py`).

Reuses the existing domain models (`StructuredQuery`, `ReasoningInsight`,
`EvaluationResult`) as field types directly rather than redefining them, so
the API's response shape and the in-process `AgentState` shape
(`agents/state.py`) never drift apart.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel

from football_rag.agents.state import EvaluationResult, ReasoningInsight, StructuredQuery


class PipelineRunRequest(BaseModel):
    match_id: int
    query: str
    enable_evaluator: bool = True
    max_retries: int = 1
    top_n: int = 10


class PipelineRunResponse(BaseModel):
    structured_query: StructuredQuery
    where_filter: Optional[dict] = None
    retrieved_chunks: list[dict]
    aggregated_context: list[dict]
    reasoning_insights: list[ReasoningInsight]
    response_text: str
    evaluation: Optional[EvaluationResult] = None
    retry_count: int
