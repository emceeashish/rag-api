"""Pydantic models for API request/response schemas."""

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=10_000)
    top_k: int = Field(default=3, ge=1, le=50)


class QueryResponse(BaseModel):
    answer: str
    chunks: list[str] = Field(default_factory=list, description="Retrieved context chunks")
    latency_ms: float
    embedding_ms: float
    retrieval_ms: float
    generation_ms: float
    top_k: int
    cache_hit: bool
