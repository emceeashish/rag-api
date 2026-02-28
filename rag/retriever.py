"""Retrieve nearest-neighbour text chunks from FAISS."""

import numpy as np
import faiss


def retrieve(
    query_embedding: np.ndarray,
    index: faiss.IndexFlatL2,
    chunks: list[str],
    top_k: int = 3,
) -> list[str]:
    """Return *top_k* chunks closest to *query_embedding*.

    If the index is empty or *top_k* exceeds the number of stored vectors,
    return as many as available.
    """
    n_vectors = index.ntotal
    if n_vectors == 0:
        return []
    k = min(top_k, n_vectors)
    query = query_embedding.reshape(1, -1).astype(np.float32)
    _, indices = index.search(query, k)  # type: ignore[arg-type]
    return [chunks[i] for i in indices[0] if 0 <= i < len(chunks)]
