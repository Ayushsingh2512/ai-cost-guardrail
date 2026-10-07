from __future__ import annotations

from collections.abc import Sequence

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.rag.generation import ContextPassage
from app.services.models import Document, DocumentChunkRecord


class Retriever:
    """Exact tenant-scoped cosine-similarity retrieval using pgvector."""

    def __init__(self, db: Session) -> None:
        self._db = db

    def retrieve(
        self,
        *,
        tenant_id: int,
        query_vector: Sequence[float],
        top_k: int = 5,
    ) -> list[ContextPassage]:
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        if not query_vector:
            raise ValueError("query_vector must not be empty")

        distance = DocumentChunkRecord.embedding.cosine_distance(
            list(query_vector)
        )

        statement = (
            select(DocumentChunkRecord)
            .join(
                Document,
                Document.id == DocumentChunkRecord.document_id,
            )
            .where(
                Document.tenant_id == tenant_id,
                DocumentChunkRecord.tenant_id == tenant_id,
            )
            .order_by(distance)
            .limit(top_k)
        )

        chunks = self._db.scalars(statement).all()

        passages: list[ContextPassage] = []

        for chunk in chunks:
            metadata = chunk.metadata_ or {}

            source = metadata.get("source")
            page = chunk.page

            passages.append(
                ContextPassage(
                    ref=(str(chunk.document_id), chunk.chunk_index),
                    text=chunk.text,
                    source=source,
                    page=page,
                )
            )

        return passages