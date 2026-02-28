"""Simple in-memory dict cache for query results."""

from typing import Optional


class QueryCache:
    """Thread-safe-ish dict cache keyed on (question, top_k)."""

    def __init__(self) -> None:
        self._store: dict[str, str] = {}

    @staticmethod
    def _key(question: str, top_k: int) -> str:
        return f"{question}:::{top_k}"

    def get(self, question: str, top_k: int) -> Optional[str]:
        return self._store.get(self._key(question, top_k))

    def set(self, question: str, top_k: int, answer: str) -> None:
        self._store[self._key(question, top_k)] = answer
