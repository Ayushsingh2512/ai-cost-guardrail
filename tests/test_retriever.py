from uuid import uuid4

import pytest

from app.rag.embeddings import EmbeddingConfig
from app.rag.retriever import Retriever
from app.services.models import (
    Document,
    DocumentChunkRecord,
    Tenant,
)


EMBEDDING_DIMS = EmbeddingConfig().dimensions


def make_vector(index: int) -> list[float]:
    vector = [0.0] * EMBEDDING_DIMS
    vector[index] = 1.0
    return vector


def make_chunk(
    *,
    tenant_id: int,
    document_id,
    chunk_index: int,
    text: str,
    embedding: list[float],
    source: str,
    page: int,
) -> DocumentChunkRecord:
    return DocumentChunkRecord(
        tenant_id=tenant_id,
        document_id=document_id,
        chunk_index=chunk_index,
        page=page,
        text=text,
        metadata_={"source": source},
        embedding=embedding,
        fingerprint="test/gemini-embedding-001/768",
    )


def test_retriever_returns_nearest_chunks_in_cosine_order(db):
    tenant = Tenant(
        name="Retriever Tenant",
        monthly_budget=100.0,
        current_spend=0.0,
    )
    db.add(tenant)
    db.flush()

    document = Document(
        tenant_id=tenant.id,
        source="retrieval-test.pdf",
    )
    db.add(document)
    db.flush()

    first_chunk = make_chunk(
        tenant_id=tenant.id,
        document_id=document.id,
        chunk_index=0,
        text="Closest chunk",
        embedding=make_vector(0),
        source="retrieval-test.pdf",
        page=1,
    )

    second_chunk = make_chunk(
        tenant_id=tenant.id,
        document_id=document.id,
        chunk_index=1,
        text="Less similar chunk",
        embedding=make_vector(1),
        source="retrieval-test.pdf",
        page=2,
    )

    db.add_all([first_chunk, second_chunk])
    db.commit()

    retriever = Retriever(db)

    results = retriever.retrieve(
        tenant_id=tenant.id,
        query_vector=make_vector(0),
        top_k=2,
    )

    assert len(results) == 2
    assert results[0].text == "Closest chunk"
    assert results[1].text == "Less similar chunk"


def test_retriever_isolates_tenants(db):
    tenant_a = Tenant(
        name="Retriever Tenant A",
        monthly_budget=100.0,
        current_spend=0.0,
    )
    tenant_b = Tenant(
        name="Retriever Tenant B",
        monthly_budget=100.0,
        current_spend=0.0,
    )

    db.add_all([tenant_a, tenant_b])
    db.flush()

    document_a = Document(
        tenant_id=tenant_a.id,
        source="tenant-a.pdf",
    )
    document_b = Document(
        tenant_id=tenant_b.id,
        source="tenant-b.pdf",
    )

    db.add_all([document_a, document_b])
    db.flush()

    chunk_a = make_chunk(
        tenant_id=tenant_a.id,
        document_id=document_a.id,
        chunk_index=0,
        text="Tenant A secret",
        embedding=make_vector(0),
        source="tenant-a.pdf",
        page=1,
    )

    chunk_b = make_chunk(
        tenant_id=tenant_b.id,
        document_id=document_b.id,
        chunk_index=0,
        text="Tenant B secret",
        embedding=make_vector(0),
        source="tenant-b.pdf",
        page=1,
    )

    db.add_all([chunk_a, chunk_b])
    db.commit()

    retriever = Retriever(db)

    results = retriever.retrieve(
        tenant_id=tenant_a.id,
        query_vector=make_vector(0),
        top_k=10,
    )

    assert len(results) == 1
    assert results[0].text == "Tenant A secret"


def test_retriever_preserves_provenance(db):
    tenant = Tenant(
        name="Provenance Tenant",
        monthly_budget=100.0,
        current_spend=0.0,
    )
    db.add(tenant)
    db.flush()

    document = Document(
        tenant_id=tenant.id,
        source="employee-handbook.pdf",
    )
    db.add(document)
    db.flush()

    chunk = make_chunk(
        tenant_id=tenant.id,
        document_id=document.id,
        chunk_index=7,
        text="Leave policy",
        embedding=make_vector(0),
        source="employee-handbook.pdf",
        page=14,
    )

    db.add(chunk)
    db.commit()

    retriever = Retriever(db)

    results = retriever.retrieve(
        tenant_id=tenant.id,
        query_vector=make_vector(0),
        top_k=1,
    )

    assert len(results) == 1
    assert results[0].ref == (str(document.id), 7)
    assert results[0].text == "Leave policy"
    assert results[0].source == "employee-handbook.pdf"
    assert results[0].page == 14


def test_retriever_returns_empty_for_tenant_without_documents(db):
    tenant = Tenant(
        name="Empty Retriever Tenant",
        monthly_budget=100.0,
        current_spend=0.0,
    )
    db.add(tenant)
    db.commit()

    retriever = Retriever(db)

    results = retriever.retrieve(
        tenant_id=tenant.id,
        query_vector=make_vector(0),
        top_k=5,
    )

    assert results == []


def test_retriever_rejects_invalid_arguments(db):
    retriever = Retriever(db)

    with pytest.raises(ValueError):
        retriever.retrieve(
            tenant_id=1,
            query_vector=make_vector(0),
            top_k=0,
        )

    with pytest.raises(ValueError):
        retriever.retrieve(
            tenant_id=1,
            query_vector=[],
            top_k=5,
        )