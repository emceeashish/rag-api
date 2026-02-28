"""FAISS index management: build, save, load."""

import json
import os

import faiss
import numpy as np


def build_index(embeddings: np.ndarray) -> faiss.IndexFlatL2:
    """Build a FAISS L2 index from normalised embeddings."""
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)  # type: ignore[arg-type]
    return index


def save_index(index: faiss.IndexFlatL2, chunks: list[str], index_path: str) -> None:
    """Persist FAISS index and associated chunks to disk."""
    os.makedirs(index_path, exist_ok=True)
    faiss.write_index(index, os.path.join(index_path, "index.faiss"))
    with open(os.path.join(index_path, "chunks.json"), "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False)


def load_index(index_path: str) -> tuple[faiss.IndexFlatL2, list[str]] | None:
    """Load FAISS index and chunks from disk.  Returns *None* if not found."""
    idx_file = os.path.join(index_path, "index.faiss")
    chunks_file = os.path.join(index_path, "chunks.json")
    if not (os.path.isfile(idx_file) and os.path.isfile(chunks_file)):
        return None
    index = faiss.read_index(idx_file)
    with open(chunks_file, "r", encoding="utf-8") as f:
        chunks: list[str] = json.load(f)
    return index, chunks


def empty_index(dimension: int) -> faiss.IndexFlatL2:
    """Return an empty FAISS index with the given dimension."""
    return faiss.IndexFlatL2(dimension)
