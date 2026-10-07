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

    def search(
        self,
        *,
        tenant_id: int,
        vector: Sequence[float],
        k: int = 5,
        fingerprint: str,
        max_distance: float | None = None,
    ) -> list[ContextPassage]:
        if k <= 0:
            raise ValueError("k must be greater than zero")

        if not vector:
            raise ValueError("vector must not be empty")

        if not fingerprint:
            raise ValueError("fingerprint must not be empty")

        if max_distance is not None and max_distance < 0:
            raise ValueError("max_distance must not be negative")

        distance = DocumentChunkRecord.embedding.cosine_distance(
            list(vector)
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
                DocumentChunkRecord.fingerprint == fingerprint,
            )
            .order_by(distance)
            .limit(k)
        )

        if max_distance is not None:
            statement = statement.where(distance <= max_distance)

        chunks = self._db.scalars(statement).all()

        passages: list[ContextPassage] = []

        for chunk in chunks:
            metadata = chunk.metadata_ or {}

            passages.append(
                ContextPassage(
                    ref=(str(chunk.document_id), chunk.chunk_index),
                    text=chunk.text,
                    source=metadata.get("source"),
                    page=chunk.page,
                )
            )

        return passages