"""Reasoning Agent (proposal §6.3.4).

The core analytical component: performs multi-step reasoning over the
aggregated context to identify tactical patterns and trends (e.g. attacking
dominance, possession shifts) as structured findings, each grounded in
specific evidence chunk_ids - rather than generating prose directly. This is
what distinguishes the pipeline from the naive RAG baseline, which generates
straight from retrieved chunks with no intermediate analysis step.
"""

from __future__ import annotations

from football_rag.agents._llm import DEFAULT_MODEL, call_structured, get_client
from football_rag.agents.state import ReasoningInsight

MODEL = DEFAULT_MODEL

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
        insights = call_structured(
            self._client,
            self.model,
            SYSTEM_PROMPT,
            user_prompt,
            response_schema=list[ReasoningInsight],
            max_tokens=4096,
        )
        # response.parsed can be None outright (not just individual fields)
        # if the model exhausts its output budget on internal reasoning
        # before finishing valid JSON - confirmed live under
        # GEMINI_MODEL=gemini-3.5-flash, which crashed this loop with
        # TypeError: 'NoneType' object is not iterable. An empty insight
        # list is already a supported state elsewhere (enable_reasoning=False
        # skips this node entirely and generator_node reads
        # state.get("reasoning_insights", [])), so falling back to [] here
        # degrades gracefully to "no analytical insights this run" instead
        # of crashing the whole pipeline over one under-budgeted call.
        if insights is None:
            return []
        # Gemini's structured output treats fields with a Python default (like
        # supporting_chunk_ids's default_factory=list) as nullable and can emit
        # an explicit null, which bypasses pydantic's default via the SDK's
        # parsing path - normalize here so nothing downstream sees None.
        for insight in insights:
            if insight.supporting_chunk_ids is None:
                insight.supporting_chunk_ids = []
        return insights
