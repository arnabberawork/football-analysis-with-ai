"""Context Aggregator Agent (proposal §6.3.3, and §6.8's de-duplication step).

Integrates the Retriever's fused hits into a unified, deduplicated context
list. This is a structuring/filtering step, not a generative one (per
§6.3.3: "integrates ... filtering out redundant or irrelevant content"), so
it makes no LLM call and stays free per the resource-efficiency goal (§6.13).
"""

from __future__ import annotations

from difflib import SequenceMatcher

DEFAULT_SIMILARITY_THRESHOLD = 0.9


class ContextAggregatorAgent:
    def __init__(self, similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD):
        self.similarity_threshold = similarity_threshold

    def aggregate(self, hits: list[dict]) -> list[dict]:
        """Drop near-duplicate chunk text, keeping retrieval-rank order, so
        overlapping evidence doesn't crowd out distinct evidence downstream.
        """
        kept: list[dict] = []
        for hit in hits:
            if not any(self._is_near_duplicate(hit["text"], k["text"]) for k in kept):
                kept.append(hit)
        return kept

    def _is_near_duplicate(self, a: str, b: str) -> bool:
        return SequenceMatcher(None, a, b).ratio() >= self.similarity_threshold
