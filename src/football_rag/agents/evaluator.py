"""Evaluator Agent (proposal §6.3.6, optional).

Validates and refines the generated output: checks relevance, coherence,
and factual consistency of the response against the evidence it was
supposed to be grounded in, flagging unsupported claims. When it fails, the
pipeline (`agents/graph.py`) feeds its `feedback` back into the Generator
Agent for a bounded number of retries.
"""

from __future__ import annotations

from football_rag.agents._llm import DEFAULT_MODEL, call_structured, get_client
from football_rag.agents.state import EvaluationResult

MODEL = DEFAULT_MODEL

SYSTEM_PROMPT = (
    "You are a strict reviewer for football match summaries. Judge whether the "
    "response is relevant to the request, coherent, and factually consistent "
    "with the evidence, and flag any claim not supported by the evidence."
)


class EvaluatorAgent:
    def __init__(self, model: str = MODEL):
        self.model = model
        self._client = get_client()

    def evaluate(self, query: str, response_text: str, aggregated_context: list[dict]) -> EvaluationResult:
        evidence_lines = "\n".join(f"[{item['chunk_id']}] {item['text']}" for item in aggregated_context)
        user_prompt = (
            f"User request: {query}\n\n"
            f"Evidence:\n{evidence_lines}\n\n"
            f"Generated response:\n{response_text}\n\n"
            "Score relevance, coherence, and factual_consistency each from 0 to 1, "
            "and set passed=true only if all three are at least 0.7 and no claim is "
            "unsupported by the evidence."
        )
        return call_structured(
            self._client,
            self.model,
            SYSTEM_PROMPT,
            user_prompt,
            response_schema=EvaluationResult,
            # Confirmed live with GEMINI_MODEL=gemini-3.5-flash: the default
            # 1024-token budget let the model exhaust its output on internal
            # reasoning before finishing the JSON, so response.parsed came
            # back None and crashed graph.py's route_after_evaluator on
            # evaluation.passed. ReasoningAgent hit the identical failure
            # mode earlier and was already bumped to 2048 for this reason -
            # same fix here.
            max_tokens=2048,
        )
