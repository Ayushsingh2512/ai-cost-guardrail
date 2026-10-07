import pytest
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from app.rag.embeddings import EmbeddingConfig
from app.services.models import (
    Document,
    DocumentChunkRecord,
    Tenant,
    EMBEDDING_DIMS,
)


def make_chunk(
    *,
    tenant_id: int,
    document_id,
    chunk_index: int = 0,
) -> DocumentChunkRecord:
    return DocumentChunkRecord(
        tenant_id=tenant_id,
        document_id=document_id,
        chunk_index=chunk_index,
        page=1,
        text=f"Test chunk {chunk_index}",
        metadata_={},
        embedding=[0.0] * EMBEDDING_DIMS,
        fingerprint="test/gemini-embedding-001/768",
    )


def test_cross_tenant_chunk_insert_is_rejected(db):
    tenant_a = Tenant(
        name="RAG Tenant A",
        monthly_budget=100.0,
        current_spend=0.0,
    )
    tenant_b = Tenant(
        name="RAG Tenant B",
        monthly_budget=100.0,
        current_spend=0.0,
    )

    db.add_all([tenant_a, tenant_b])
    db.flush()

    document = Document(
        tenant_id=tenant_a.id,
        source="tenant-a.pdf",
    )

    db.add(document)
    db.flush()

    cross_tenant_chunk = make_chunk(
        tenant_id=tenant_b.id,
        document_id=document.id,
    )

    db.add(cross_tenant_chunk)

    with pytest.raises(IntegrityError):
        db.flush()

    db.rollback()


def test_deleting_document_cascades_to_chunks(db):
    tenant = Tenant(
        name="Cascade Test Tenant",
        monthly_budget=100.0,
        current_spend=0.0,
    )
    db.add(tenant)
    db.flush()

    document = Document(
        tenant_id=tenant.id,
        source="cascade-test.pdf",
    )
    db.add(document)
    db.flush()

    chunk_0 = make_chunk(
        tenant_id=tenant.id,
        document_id=document.id,
        chunk_index=0,
    )
    chunk_1 = make_chunk(
        tenant_id=tenant.id,
        document_id=document.id,
        chunk_index=1,
    )

    db.add_all([chunk_0, chunk_1])
    db.commit()

    db.delete(document)
    db.commit()

    remaining_chunks = db.scalars(
        select(DocumentChunkRecord).where(
            DocumentChunkRecord.document_id == document.id
        )
    ).all()

    assert remaining_chunks == []


def test_embedding_dimensions_match_config():
    assert EMBEDDING_DIMS == EmbeddingConfig().dimensions