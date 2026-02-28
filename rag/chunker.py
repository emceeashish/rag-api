"""Split documents into overlapping text chunks."""


def chunk_documents(
    documents: list[str],
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[str]:
    """Split *documents* into fixed-size character chunks with overlap.

    Raises ``ValueError`` if inputs are invalid.
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be > 0, got {chunk_size}")
    if chunk_overlap < 0:
        raise ValueError(f"chunk_overlap must be >= 0, got {chunk_overlap}")
    if chunk_overlap >= chunk_size:
        raise ValueError(
            f"chunk_overlap ({chunk_overlap}) must be < chunk_size ({chunk_size})"
        )

    chunks: list[str] = []
    for doc in documents:
        start = 0
        while start < len(doc):
            end = start + chunk_size
            chunk = doc[start:end].strip()
            if chunk:
                chunks.append(chunk)
            start += chunk_size - chunk_overlap
    return chunks
