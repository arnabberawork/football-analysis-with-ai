"""Keyword-based retrieval index (proposal §6.8's "optional keyword-enhanced
retrieval step" - combined with dense retrieval by `agents.retriever` via
reciprocal rank fusion).

`rank-bm25` was already a dependency (per CLAUDE.md) but unused until now.
"""

from __future__ import annotations

import re

from rank_bm25 import BM25Okapi

from football_rag.schema import Chunk

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class BM25Index:
    """BM25 index over a fixed corpus of chunks (built once per match)."""

    def __init__(self, chunks: list[Chunk]):
        self._chunk_ids = [c.chunk_id for c in chunks]
        # BM25Okapi divides by corpus size internally, so an empty corpus
        # (e.g. a metadata filter that matches nothing) must short-circuit
        # here rather than crash on construction.
        self._bm25 = BM25Okapi([_tokenize(c.text) for c in chunks]) if chunks else None

    def query(self, text: str, top_n: int = 10) -> list[tuple[str, float]]:
        """Return up to `top_n` (chunk_id, score) pairs, highest score first."""
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(_tokenize(text))
        ranked = sorted(zip(self._chunk_ids, scores), key=lambda pair: pair[1], reverse=True)
        return ranked[:top_n]
