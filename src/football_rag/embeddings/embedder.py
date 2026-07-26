"""Local embedding model wrapper (proposal §6.7).

Defaults to a small open-source Sentence-BERT model that runs on CPU with no
API cost, in line with the resource-efficiency constraint in §6.13 ("priority
is given to open-source embedding models that can be executed locally").
Swapping to an E5 variant later is a one-line change since callers only ever
see `encode_documents` / `encode_query`.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[3] / ".env")

if not os.environ.get("HF_TOKEN") and not os.environ.get("HF_API_TOKEN") and os.environ.get("HF_API_KEY"):
    os.environ["HF_TOKEN"] = os.environ["HF_API_KEY"]
    os.environ["HF_API_TOKEN"] = os.environ["HF_API_KEY"]

try:
    from sentence_transformers import SentenceTransformer
except Exception as exc:  # pragma: no cover - import may vary by platform
    SentenceTransformer = None
    _SENTENCE_TRANSFORMERS_IMPORT_ERROR = exc
else:
    _SENTENCE_TRANSFORMERS_IMPORT_ERROR = None

DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class Embedder:
    def __init__(self, model_name: str = DEFAULT_MODEL_NAME):
        if SentenceTransformer is None:
            raise ImportError(
                "sentence-transformers is required to create an Embedder. "
                "Install the project dependencies with 'uv sync'."
            ) from _SENTENCE_TRANSFORMERS_IMPORT_ERROR

        self.model_name = model_name
        self._model = SentenceTransformer(model_name)

    def encode_documents(self, texts: list[str]) -> list[list[float]]:
        embeddings = self._model.encode(texts, convert_to_numpy=True, show_progress_bar=len(texts) > 200)
        return embeddings.tolist()

    def encode_query(self, text: str) -> list[float]:
        return self._model.encode(text, convert_to_numpy=True).tolist()


@lru_cache(maxsize=1)
def get_default_embedder() -> Embedder:
    """Process-wide cached embedder so repeated calls don't reload the model."""
    return Embedder()
