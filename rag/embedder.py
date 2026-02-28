"""Sentence-transformer embedding wrapper (loaded once)."""

import numpy as np
from sentence_transformers import SentenceTransformer


class Embedder:
    """Thin wrapper around a SentenceTransformer model."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self.model = SentenceTransformer(model_name)
        self.dimension: int = self.model.get_sentence_embedding_dimension()  # type: ignore[assignment]

    def embed_text(self, text: str) -> np.ndarray:
        """Return a normalised float32 embedding for *text*."""
        vec = self.model.encode(text, convert_to_numpy=True).astype(np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """Return normalised float32 embeddings for a list of texts."""
        vecs = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=True).astype(np.float32)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1
        return vecs / norms
