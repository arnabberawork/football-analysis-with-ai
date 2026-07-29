"""Retriever Agent (proposal §6.3.2, §6.8).

Hybrid retrieval: dense semantic search (via `ChunkVectorStore`) combined
with BM25 keyword search (via `BM25Index`), both restricted by the same
metadata `where` filter, fused with Reciprocal Rank Fusion since cosine
distance and BM25 scores are not on comparable scales. The fused pool is
then optionally re-ranked by a local cross-encoder (`retrieval/reranker.py`).
`enable_hybrid`/`enable_reranker` exist so the ablation study
(`scripts/run_ablation.py`) can toggle each independently.
"""

from __future__ import annotations

import unicodedata

from football_rag.retrieval.bm25_index import BM25Index
from football_rag.retrieval.reranker import get_default_reranker
from football_rag.retrieval.vector_store import ChunkVectorStore
from football_rag.schema import Chunk, ChunkMetadata

RRF_K = 60

# team_name/player_name are LLM-extracted (e.g. "Mbappe") and won't always
# match the stored metadata verbatim (e.g. "Kylian Mbappé"), so those two
# fields match on accent-/case-insensitive substring rather than equality.
_FUZZY_NAME_FIELDS = {"team_name", "player_name"}


def _normalize(text: str) -> str:
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii").lower()


SHOOTOUT_PERIOD = 5
"""StatsBomb period convention: 5 = the penalty shootout. A shootout kick
is encoded as an ordinary Shot/outcome=Goal event indistinguishable by
event_type from an in-match goal - excluded here for the same reason
`agents/stats.py`/`chunking/event_chunker.py` exclude it: a shootout kick
being forced into "salient" evidence alongside real goals would let a
generated summary mistake it for one."""


def _salience_priority(chunk: Chunk) -> int | None:
    """Lower = more narratively important; `None` = not a "salient" event.
    Used by `retrieve(ensure_salient=True)` to force goals/cards/subs into
    the context for broad, unscoped summary queries - confirmed live
    (TESTING.md §9/§12) that similarity ranking alone can leave a
    "summarize the match" query with zero goal-related evidence, and that
    neither hierarchical chunking nor re-ranking alone closes that gap.
    Restricted to event-level chunks, which map 1:1 to a single event
    unambiguously (time-window/sequence chunks mix many events together).

    Card outcome is checked by substring across both `Foul Committed` and
    `Bad Behaviour` event types (StatsBomb records a card as the outcome of
    either), not gated by event_type first - confirmed live that a
    second-booking dismissal is recorded with outcome "Second Yellow", not
    "Red Card", so a straight-red-only text check silently excluded it from
    salience entirely (a real sending-off dropped from evidence, not merely
    outranked) - see TESTING.md.
    """
    if chunk.metadata.chunk_level != "event":
        return None
    if chunk.metadata.period == SHOOTOUT_PERIOD:
        return None
    event_type = chunk.metadata.event_type
    if event_type == "Shot" and "outcome: Goal" in chunk.text:
        return 0
    if event_type in ("Own Goal Against", "Own Goal For"):
        return 0
    if event_type in ("Foul Committed", "Bad Behaviour"):
        if "Red Card" in chunk.text or "Second Yellow" in chunk.text:
            return 1
        if "Yellow Card" in chunk.text:
            return 2
    if event_type == "Substitution":
        return 3
    return None


def _matches_where(metadata: ChunkMetadata, where: dict) -> bool:
    """Evaluate the subset of Chroma `where` syntax that
    `query_understanding.to_where_filter` can produce (equality, `$and`,
    `$gte`, `$lte`), so BM25 candidates are filtered the same way Chroma
    filters dense candidates - modulo the fuzzy name matching above, which
    Chroma's own exact-match `where` filtering can't do.
    """
    if "$and" in where:
        return all(_matches_where(metadata, clause) for clause in where["$and"])

    (field, condition), = where.items()
    value = getattr(metadata, field, None)
    if isinstance(condition, dict):
        op, operand = next(iter(condition.items()))
        if op == "$gte":
            return value is not None and value >= operand
        if op == "$lte":
            return value is not None and value <= operand
        raise ValueError(f"Unsupported where operator: {op}")
    if field in _FUZZY_NAME_FIELDS:
        return value is not None and _normalize(condition) in _normalize(value)
    return value == condition


class RetrieverAgent:
    def __init__(
        self,
        vector_store: ChunkVectorStore,
        chunks: list[Chunk],
        enable_hybrid: bool = True,
        enable_reranker: bool = True,
    ):
        self.vector_store = vector_store
        self._chunks = chunks
        self._chunks_by_id = {c.chunk_id: c for c in chunks}
        self._bm25 = BM25Index(chunks)
        self.enable_hybrid = enable_hybrid
        self.enable_reranker = enable_reranker
        self._reranker = get_default_reranker() if enable_reranker else None

    def retrieve(
        self,
        query: str,
        where: dict | None = None,
        top_n: int = 10,
        fusion_pool: int = 30,
        ensure_salient: bool = False,
    ) -> list[dict]:
        dense_hits = self.vector_store.query(query, top_n=fusion_pool, where=where)

        bm25_hits: list[tuple[str, float]] = []
        if self.enable_hybrid:
            bm25_index = self._bm25
            if where is not None:
                candidates = [c for c in self._chunks if _matches_where(c.metadata, where)]
                bm25_index = BM25Index(candidates)
            bm25_hits = bm25_index.query(query, top_n=fusion_pool)

        fused = self._fuse(dense_hits, bm25_hits)
        ranked = self._reranker.rerank(query, fused[:fusion_pool], top_n) if self._reranker is not None else fused[:top_n]

        if not ensure_salient:
            return ranked
        return self._force_salient(ranked, top_n)

    def _force_salient(self, ranked: list[dict], top_n: int) -> list[dict]:
        """Force every real goal (priority 0) into the context, uncapped,
        then reserve up to half of `top_n` more for the next-highest-priority
        card/substitution chunks not already present, filling the rest from
        the similarity-ranked `ranked` list.

        Goals are deliberately never budget-capped - confirmed live that a
        fixed `top_n // 2` cap silently dropped a real goal (Mbappe's 117'
        extra-time penalty, the 6th goal in the 2022 WC Final) because 5
        earlier goals already filled the old shared budget first. Omitting
        a real goal from a match summary is a correctness bug, not a
        ranking nuance, so goals get an unconditional pass while cards/subs
        (lower narrative stakes, and unbounded in principle - a scrappy
        match can have many more fouls/subs than goals) stay budget-limited
        to protect `top_n` headroom for the similarity-ranked pool.
        """
        seen_ids = {h["chunk_id"] for h in ranked}

        candidates = sorted(
            (c for c in self._chunks if _salience_priority(c) is not None),
            key=_salience_priority,
        )
        goal_candidates = [c for c in candidates if _salience_priority(c) == 0]
        other_candidates = [c for c in candidates if _salience_priority(c) != 0]

        def _force(chunk: Chunk) -> dict:
            seen_ids.add(chunk.chunk_id)
            return {"chunk_id": chunk.chunk_id, "text": chunk.text, "metadata": chunk.metadata.model_dump()}

        forced: list[dict] = [_force(c) for c in goal_candidates if c.chunk_id not in seen_ids]

        other_budget = max(1, top_n // 2) - len(forced)
        for chunk in other_candidates:
            if other_budget <= 0:
                break
            if chunk.chunk_id in seen_ids:
                continue
            forced.append(_force(chunk))
            other_budget -= 1

        remaining_slots = max(0, top_n - len(forced))
        return forced + ranked[:remaining_slots]

    def _fuse(self, dense_hits: list[dict], bm25_hits: list[tuple[str, float]]) -> list[dict]:
        scores: dict[str, float] = {}
        known: dict[str, dict] = {}

        for rank, hit in enumerate(dense_hits):
            scores[hit["chunk_id"]] = scores.get(hit["chunk_id"], 0.0) + 1.0 / (RRF_K + rank + 1)
            known[hit["chunk_id"]] = hit

        for rank, (chunk_id, _score) in enumerate(bm25_hits):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (RRF_K + rank + 1)

        ranked_ids = sorted(scores, key=lambda cid: scores[cid], reverse=True)

        fused = []
        for chunk_id in ranked_ids:
            hit = known.get(chunk_id)
            if hit is None:
                chunk = self._chunks_by_id[chunk_id]
                hit = {"chunk_id": chunk_id, "text": chunk.text, "metadata": chunk.metadata.model_dump()}
            fused.append({**hit, "fusion_score": scores[chunk_id]})
        return fused
