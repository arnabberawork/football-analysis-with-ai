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
E5_MODEL_NAME = "intfloat/e5-small-v2"  # small variant - confirmed live that e5-base-v2 was too slow on CPU-only
# hardware to embed even a representative subset of the corpus in reasonable time; e5-small-v2 is a closer
# parameter-count match to MiniLM anyway (~33M vs ~22M) for a fairer comparison.


class Embedder:
    def __init__(self, model_name: str = DEFAULT_MODEL_NAME):
        if SentenceTransformer is None:
            raise ImportError(
                "sentence-transformers is required to create an Embedder. "
                "Install the project dependencies with 'uv sync'."
            ) from _SENTENCE_TRANSFORMERS_IMPORT_ERROR

        self.model_name = model_name
        # E5 models are trained with "query: "/"passage: " prefixes baked
        # into their objective - omitting them doesn't error, it just
        # silently degrades retrieval quality, so this is load-bearing for
        # a fair MiniLM-vs-E5 comparison (experiment/compare_embeddings.py).
        self._is_e5 = "e5" in model_name.lower()
        self._model = SentenceTransformer(model_name)

    def encode_documents(self, texts: list[str]) -> list[list[float]]:
        if self._is_e5:
            texts = [f"passage: {t}" for t in texts]
        embeddings = self._model.encode(texts, convert_to_numpy=True, show_progress_bar=len(texts) > 200)
        return embeddings.tolist()

    def encode_query(self, text: str) -> list[float]:
        if self._is_e5:
            text = f"query: {text}"
        return self._model.encode(text, convert_to_numpy=True).tolist()


@lru_cache(maxsize=1)
def get_default_embedder() -> Embedder:
    """Process-wide cached embedder so repeated calls don't reload the model."""
    return Embedder()


@lru_cache(maxsize=1)
def get_e5_embedder() -> Embedder:
    """Process-wide cached E5 embedder, parallel to `get_default_embedder()` -
    used by `experiment/compare_embeddings.py` to compare against MiniLM."""
    return Embedder(E5_MODEL_NAME)
