"""Context Aggregator Agent (proposal §6.3.3, and §6.8's de-duplication step).

Integrates the Retriever's fused hits into a unified, deduplicated context
list. This is a structuring/filtering step, not a generative one (per
§6.3.3: "integrates ... filtering out redundant or irrelevant content"), so
it makes no LLM call and stays free per the resource-efficiency goal (§6.13).
"""

from __future__ import annotations

from difflib import SequenceMatcher

DEFAULT_SIMILARITY_THRESHOLD = 0.9

MAX_DUPLICATE_MINUTE_GAP = 5
"""Two chunks are never treated as duplicates if their `minute_start` is
further apart than this, no matter how similar their text - confirmed live
that the fixed event-sentence template renders two structurally identical
events (e.g. two penalties by the same player, same technique/body part/xG)
almost identically except for the timestamp prefix: Mbappe's 79' and 117'
penalties in the 2022 WC Final (`Penalty, Normal, Right Foot; xG 0.78` for
both) scored a 0.97 text-similarity ratio, well above the 0.9 threshold, so
the second - his decisive extra-time equalizer - was silently dropped as a
"duplicate" of the earlier one and never reached the Generator. Kept
generous (not exact-minute equality) rather than tight, so a time-window
chunk and an overlapping sequence chunk describing the same passage of
play - the actual redundancy this dedup step is meant to catch - can still
be recognised as duplicates even when their window start times don't align
exactly."""


class ContextAggregatorAgent:
    def __init__(self, similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD):
        self.similarity_threshold = similarity_threshold

    def aggregate(self, hits: list[dict]) -> list[dict]:
        """Drop near-duplicate chunk text, keeping retrieval-rank order, so
        overlapping evidence doesn't crowd out distinct evidence downstream.
        """
        kept: list[dict] = []
        for hit in hits:
            if not any(self._is_near_duplicate(hit, k) for k in kept):
                kept.append(hit)
        return kept

    def _is_near_duplicate(self, a: dict, b: dict) -> bool:
        minute_a = a.get("metadata", {}).get("minute_start")
        minute_b = b.get("metadata", {}).get("minute_start")
        if minute_a is not None and minute_b is not None and abs(minute_a - minute_b) > MAX_DUPLICATE_MINUTE_GAP:
            return False
        return SequenceMatcher(None, a["text"], b["text"]).ratio() >= self.similarity_threshold
