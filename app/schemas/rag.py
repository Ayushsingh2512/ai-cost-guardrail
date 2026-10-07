from typing import Literal

from pydantic import BaseModel, Field


class RAGRequest(BaseModel):
    query: str = Field(
        min_length=1,
        max_length=4000,
        description="Question to answer using the tenant's knowledge base.",
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of relevant chunks to retrieve.",
    )


class RAGCitation(BaseModel):
    ref: tuple[str, int]
    source: str | None = None
    page: int | None = None


class RAGResponse(BaseModel):
    status: Literal["answered", "no_context", "blocked"]
    answer: str | None
    citations: list[RAGCitation]
    top_k: int
    retrieved_count: int
    timings_ms: dict[str, int]
    embedding_tokens: int | None = None