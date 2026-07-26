"""Shared Gemini plumbing used by every LLM-backed agent (Query
Understanding, Reasoning, Generator, Evaluator) and the naive baseline
(`generation.generator.NaiveRAGGenerator`) so each doesn't duplicate the
same client setup and structured-output parsing.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
from google import genai

load_dotenv()

DEFAULT_MODEL = os.environ.get("GEMINI_MODEL", "gemini-flash-latest")
"""Model used by ReasoningAgent, GeneratorAgent, EvaluatorAgent, and
NaiveRAGGenerator - deliberately the *same* model across all four call
sites (naive and agentic alike) so the naive-vs-agentic comparison isolates
architecture differences, not model differences (§6.10). Overridable via a
GEMINI_MODEL env var (in .env) rather than editing each file separately -
previously each of these five call sites hardcoded its own literal model
string, which meant swapping models (e.g. to route around a daily quota
limit on one model) required editing multiple files and risked the naive/
agentic comparison silently drifting onto different models if only some
were updated."""

DEFAULT_MODEL_LITE = os.environ.get("GEMINI_MODEL_LITE", "gemini-3.5-flash-lite")
"""Lighter/cheaper model used only by QueryUnderstandingAgent, whose
extraction task is simpler than reasoning/generation/evaluation. Separate
from DEFAULT_MODEL since it's intentionally a different tier, not part of
the naive-vs-agentic model-parity requirement above. Overridable via a
GEMINI_MODEL_LITE env var."""


def get_client() -> genai.Client:
    if not os.environ.get("GEMINI_API_KEY"):
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Add it to a .env file in the project root "
            "or export it in your shell before running the pipeline. Get a free "
            "key at https://aistudio.google.com/apikey."
        )
    return genai.Client()


def call_structured(
    client: genai.Client,
    model: str,
    system: str,
    user_prompt: str,
    response_schema,
    max_tokens: int = 1024,
):
    """Call Gemini with a Pydantic model (or `list[Model]`) as
    `response_schema`, forcing schema-conformant JSON output, and return the
    already-validated parsed instance - no manual JSON parsing required.
    """
    response = client.models.generate_content(
        model=model,
        contents=user_prompt,
        config=genai.types.GenerateContentConfig(
            system_instruction=system,
            response_mime_type="application/json",
            response_schema=response_schema,
            max_output_tokens=max_tokens,
        ),
    )
    return response.parsed
