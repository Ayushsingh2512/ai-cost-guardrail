import pytest
from pydantic import ValidationError

from app.schemas.rag import (
    RAGCitation,
    RAGRequest,
    RAGResponse,
)


def test_rag_request_defaults_top_k():
    request = RAGRequest(query="What is the leave policy?")

    assert request.query == "What is the leave policy?"
    assert request.top_k == 5


def test_rag_request_accepts_valid_top_k():
    request = RAGRequest(
        query="What is the leave policy?",
        top_k=10,
    )

    assert request.top_k == 10


def test_rag_request_rejects_empty_query():
    with pytest.raises(ValidationError):
        RAGRequest(query="")


def test_rag_request_rejects_query_over_max_length():
    with pytest.raises(ValidationError):
        RAGRequest(query="a" * 4001)


@pytest.mark.parametrize("top_k", [0, -1, 21])
def test_rag_request_rejects_invalid_top_k(top_k):
    with pytest.raises(ValidationError):
        RAGRequest(
            query="What is the leave policy?",
            top_k=top_k,
        )


@pytest.mark.parametrize("top_k", [1, 20])
def test_rag_request_accepts_top_k_boundaries(top_k):
    request = RAGRequest(
        query="What is the leave policy?",
        top_k=top_k,
    )

    assert request.top_k == top_k


def test_rag_citation():
    citation = RAGCitation(
        ref=("document-123", 2),
        source="employee_handbook.pdf",
        page=4,
    )

    assert citation.ref == ("document-123", 2)
    assert citation.source == "employee_handbook.pdf"
    assert citation.page == 4


def test_rag_citation_allows_missing_provenance():
    citation = RAGCitation(
        ref=("document-123", 2),
    )

    assert citation.source is None
    assert citation.page is None


@pytest.mark.parametrize(
    "status",
    ["answered", "no_context", "blocked"],
)
def test_rag_response_accepts_valid_status(status):
    response = RAGResponse(
        status=status,
        answer=None,
        citations=[],
        top_k=5,
        retrieved_count=0,
        timings_ms={
            "embed": 10,
            "retrieve": 5,
            "generate": 20,
        },
    )

    assert response.status == status
    assert response.embedding_tokens is None


def test_rag_response_rejects_invalid_status():
    with pytest.raises(ValidationError):
        RAGResponse(
            status="error",
            answer=None,
            citations=[],
            top_k=5,
            retrieved_count=0,
            timings_ms={},
        )


def test_rag_response_with_citations():
    response = RAGResponse(
        status="answered",
        answer="The leave policy allows 20 days.",
        citations=[
            RAGCitation(
                ref=("document-123", 2),
                source="employee_handbook.pdf",
                page=4,
            )
        ],
        top_k=5,
        retrieved_count=1,
        timings_ms={
            "embed": 12,
            "retrieve": 4,
            "generate": 300,
        },
        embedding_tokens=25,
    )

    assert response.answer == "The leave policy allows 20 days."
    assert len(response.citations) == 1
    assert response.retrieved_count == 1
    assert response.embedding_tokens == 25