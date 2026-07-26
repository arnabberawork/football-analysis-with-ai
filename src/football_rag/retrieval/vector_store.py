"""ChromaDB-backed vector store for retrieval (proposal §6.7, §6.8).

ChromaDB is chosen over a raw FAISS index specifically because it supports
metadata filtering natively at query time (§6.7: "databases such as ChromaDB
allow filtering to be applied during the retrieval process, improving
efficiency and precision"), which the metadata-aware retrieval strategy in
§6.8 depends on.

Embeddings are computed by our own `Embedder` (not Chroma's built-in
embedding function) so the embedding model choice stays explicit and
swappable, per §6.7's discussion of SBERT vs. E5 vs. API embeddings.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import chromadb
from chromadb.api.models.Collection import Collection

from football_rag.embeddings.embedder import Embedder, get_default_embedder
from football_rag.schema import Chunk, ChunkMetadata

DEFAULT_PERSIST_DIR = Path("data/chroma")
DEFAULT_COLLECTION_NAME = "football_chunks"


def _metadata_to_chroma_dict(metadata: ChunkMetadata) -> dict:
    """Chroma only accepts flat str/int/float/bool metadata values (no
    None), so optional fields are dropped rather than sent as null."""
    raw = metadata.model_dump()
    return {k: v for k, v in raw.items() if v is not None}


class ChunkVectorStore:
    def __init__(
        self,
        persist_dir: Path = DEFAULT_PERSIST_DIR,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        embedder: Optional[Embedder] = None,
    ):
        self.embedder = embedder or get_default_embedder()
        self._client = chromadb.PersistentClient(path=str(persist_dir))
        self._collection: Collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def ingest(self, chunks: list[Chunk], batch_size: int = 256) -> None:
        """Embed and upsert chunks. Upsert (not add) so re-running ingestion
        on the same match is idempotent instead of creating duplicates."""
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            embeddings = self.embedder.encode_documents([c.text for c in batch])
            self._collection.upsert(
                ids=[c.chunk_id for c in batch],
                embeddings=embeddings,
                documents=[c.text for c in batch],
                metadatas=[_metadata_to_chroma_dict(c.metadata) for c in batch],
            )

    def count(self) -> int:
        return self._collection.count()

    def query(
        self,
        query_text: str,
        top_n: int = 10,
        where: Optional[dict] = None,
    ) -> list[dict]:
        """Return up to `top_n` chunks most semantically similar to
        `query_text`, optionally restricted by a Chroma `where` metadata
        filter (e.g. {"team_name": "Argentina"} or
        {"event_type": {"$in": ["Shot", "Foul Committed"]}}).

        Each result dict has: chunk_id, text, metadata, distance.
        """
        query_embedding = self.embedder.encode_query(query_text)
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_n,
            where=where,
        )

        hits = []
        ids = results["ids"][0]
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]
        for chunk_id, text, metadata, distance in zip(ids, documents, metadatas, distances):
            hits.append({"chunk_id": chunk_id, "text": text, "metadata": metadata, "distance": distance})
        return hits
