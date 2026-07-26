"""Generator Agent (proposal §6.3.5).

Produces the final natural-language output, reusing the same structured
prompt template as the naive baseline (`generation/prompts.py`'s
`SYSTEM_PROMPT`) but extended with the Reasoning Agent's structured findings
via `build_agentic_user_prompt`. When the Evaluator Agent rejects a prior
attempt, its feedback is appended as a revision instruction.
"""

from __future__ import annotations

from typing import Optional

from google import genai

from football_rag.agents._llm import get_client
from football_rag.agents.state import ReasoningInsight
from football_rag.generation.prompts import SYSTEM_PROMPT, build_agentic_user_prompt

MODEL = "gemini-flash-latest"


class GeneratorAgent:
    def __init__(self, model: str = MODEL):
        self.model = model
        self._client = get_client()

    def generate(
        self,
        query: str,
        aggregated_context: list[dict],
        insights: list[ReasoningInsight],
        feedback: Optional[str] = None,
    ) -> str:
        user_prompt = build_agentic_user_prompt(query, aggregated_context, insights)
        if feedback:
            user_prompt += f"\n\nRevision feedback from evaluator (address this in the new response): {feedback}"

        response = self._client.models.generate_content(
            model=self.model,
            contents=user_prompt,
            config=genai.types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT, max_output_tokens=4096),
        )
        return response.text or ""
