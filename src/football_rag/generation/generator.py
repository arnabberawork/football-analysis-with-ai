"""Naive RAG generation baseline (proposal §6.11.5).

Single-stage retrieve-then-generate with no metadata filtering, no
reasoning agent, and no evaluator — this is the baseline the full Agentic
RAG pipeline is benchmarked against.
"""

from __future__ import annotations

from google import genai

from football_rag.agents._llm import DEFAULT_MODEL, get_client
from football_rag.generation.prompts import SYSTEM_PROMPT, build_user_prompt
from football_rag.retrieval.vector_store import ChunkVectorStore


class NaiveRAGGenerator:
    def __init__(
        self,
        vector_store: ChunkVectorStore,
        model: str = DEFAULT_MODEL,
        top_n: int = 10,
    ):
        self.vector_store = vector_store
        self.model = model
        self.top_n = top_n
        self._client = get_client()

    def generate(self, user_query: str, where: dict | None = None) -> dict:
        """Retrieve top-N chunks for the query and generate a grounded response.

        Returns {"query", "retrieved_chunks", "response_text"}.
        """
        retrieved_chunks = self.vector_store.query(user_query, top_n=self.top_n, where=where)
        user_prompt = build_user_prompt(user_query, retrieved_chunks)

        response = self._client.models.generate_content(
            model=self.model,
            contents=user_prompt,
            config=genai.types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT, max_output_tokens=4096),
        )

        return {
            "query": user_query,
            "retrieved_chunks": retrieved_chunks,
            "response_text": response.text or "",
        }
