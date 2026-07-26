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
