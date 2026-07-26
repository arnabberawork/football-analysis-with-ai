"""Retriever Agent (proposal §6.3.2, §6.8).

Hybrid retrieval: dense semantic search (via `ChunkVectorStore`) combined
with BM25 keyword search (via `BM25Index`), both restricted by the same
metadata `where` filter, fused with Reciprocal Rank Fusion since cosine
distance and BM25 scores are not on comparable scales.
"""

from __future__ import annotations

import unicodedata

from football_rag.retrieval.bm25_index import BM25Index
from football_rag.retrieval.vector_store import ChunkVectorStore
from football_rag.schema import Chunk, ChunkMetadata

RRF_K = 60

# team_name/player_name are LLM-extracted (e.g. "Mbappe") and won't always
# match the stored metadata verbatim (e.g. "Kylian Mbappé"), so those two
# fields match on accent-/case-insensitive substring rather than equality.
_FUZZY_NAME_FIELDS = {"team_name", "player_name"}


def _normalize(text: str) -> str:
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii").lower()


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
    def __init__(self, vector_store: ChunkVectorStore, chunks: list[Chunk]):
        self.vector_store = vector_store
        self._chunks = chunks
        self._chunks_by_id = {c.chunk_id: c for c in chunks}
        self._bm25 = BM25Index(chunks)

    def retrieve(
        self,
        query: str,
        where: dict | None = None,
        top_n: int = 10,
        fusion_pool: int = 30,
    ) -> list[dict]:
        dense_hits = self.vector_store.query(query, top_n=fusion_pool, where=where)

        bm25_index = self._bm25
        if where is not None:
            candidates = [c for c in self._chunks if _matches_where(c.metadata, where)]
            bm25_index = BM25Index(candidates)
        bm25_hits = bm25_index.query(query, top_n=fusion_pool)

        return self._fuse(dense_hits, bm25_hits)[:top_n]

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
