"""Cross-encoder re-ranking (proposal §6.8's optional retrieval-refinement
step).

Re-scores the RRF-fused candidates from `agents/retriever.py` with a small
local cross-encoder so salient events (goals, cards, shots) outrank generic
ones (duels, ball receipts) within the top-N - at zero added Gemini quota
cost, unlike an LLM-based re-rank.
"""

from __future__ import annotations

from functools import lru_cache

from sentence_transformers import CrossEncoder

DEFAULT_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class Reranker:
    def __init__(self, model_name: str = DEFAULT_MODEL_NAME):
        self.model_name = model_name
        self._model = CrossEncoder(model_name)

    def rerank(self, query: str, hits: list[dict], top_n: int) -> list[dict]:
        """Re-score `hits` (each with a `text` field) against `query` and
        return the top-N, highest-scoring first."""
        if not hits:
            return hits
        pairs = [(query, hit["text"]) for hit in hits]
        scores = self._model.predict(pairs)
        ranked = sorted(zip(hits, scores), key=lambda pair: pair[1], reverse=True)
        return [{**hit, "rerank_score": float(score)} for hit, score in ranked[:top_n]]


@lru_cache(maxsize=1)
def get_default_reranker() -> Reranker:
    """Process-wide cached reranker so repeated calls don't reload the model."""
    return Reranker()
