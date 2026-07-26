"""Reasoning Agent (proposal §6.3.4).

The core analytical component: performs multi-step reasoning over the
aggregated context to identify tactical patterns and trends (e.g. attacking
dominance, possession shifts) as structured findings, each grounded in
specific evidence chunk_ids - rather than generating prose directly. This is
what distinguishes the pipeline from the naive RAG baseline, which generates
straight from retrieved chunks with no intermediate analysis step.
"""

from __future__ import annotations

from football_rag.agents._llm import call_structured, get_client
from football_rag.agents.state import ReasoningInsight

MODEL = "gemini-flash-latest"

SYSTEM_PROMPT = (
    "You are a football tactics analyst. Analyse the provided match evidence "
    "and identify concrete tactical patterns and trends. Do not write a prose "
    "summary - only structured analytical findings, each grounded in specific "
    "evidence chunk_ids."
)


class ReasoningAgent:
    def __init__(self, model: str = MODEL):
        self.model = model
        self._client = get_client()

    def reason(self, query: str, aggregated_context: list[dict]) -> list[ReasoningInsight]:
        evidence_lines = "\n".join(f"[{item['chunk_id']}] {item['text']}" for item in aggregated_context)
        user_prompt = (
            f"User request: {query}\n\n"
            f"Evidence:\n{evidence_lines}\n\n"
            "Identify the key tactical patterns and trends supported by this evidence."
        )
        return call_structured(
            self._client,
            self.model,
            SYSTEM_PROMPT,
            user_prompt,
            response_schema=list[ReasoningInsight],
            max_tokens=2048,
        )
