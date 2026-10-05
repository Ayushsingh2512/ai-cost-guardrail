import pytest

from app.rag.chunking import DocumentChunk, chunk_text


def test_empty_text_returns_no_chunks():
    chunks = chunk_text(
        "",
        document_id="doc-1",
    )

    assert chunks == []


def test_whitespace_only_text_returns_no_chunks():
    chunks = chunk_text(
        "   \n\n   ",
        document_id="doc-1",
    )

    assert chunks == []


def test_short_text_returns_one_chunk():
    chunks = chunk_text(
        "Hello world.",
        document_id="doc-1",
    )

    assert len(chunks) == 1
    assert isinstance(chunks[0], DocumentChunk)
    assert chunks[0].document_id == "doc-1"
    assert chunks[0].chunk_index == 0
    assert chunks[0].text == "Hello world."


def test_metadata_is_preserved():
    metadata = {
        "source": "employee_handbook.pdf",
        "page": 17,
        "tenant_id": 123,
    }

    chunks = chunk_text(
        "Employees must provide written notice.",
        document_id="doc-1",
        metadata=metadata,
    )

    assert len(chunks) == 1
    assert chunks[0].metadata == metadata


def test_metadata_is_copied_for_each_chunk():
    metadata = {
        "source": "document.pdf",
        "tenant_id": 123,
    }

    chunks = chunk_text(
        "This is a long document. " * 100,
        document_id="doc-1",
        metadata=metadata,
        chunk_size=100,
        overlap=20,
    )

    assert len(chunks) > 1

    for chunk in chunks:
        assert chunk.metadata == metadata
        assert chunk.metadata is not metadata


def test_invalid_chunk_size_is_rejected():
    with pytest.raises(ValueError):
        chunk_text(
            "some text",
            document_id="doc-1",
            chunk_size=0,
        )


def test_negative_overlap_is_rejected():
    with pytest.raises(ValueError):
        chunk_text(
            "some text",
            document_id="doc-1",
            chunk_size=100,
            overlap=-1,
        )


def test_overlap_equal_to_chunk_size_is_rejected():
    with pytest.raises(ValueError):
        chunk_text(
            "some text",
            document_id="doc-1",
            chunk_size=100,
            overlap=100,
        )


def test_overlap_greater_than_chunk_size_is_rejected():
    with pytest.raises(ValueError):
        chunk_text(
            "some text",
            document_id="doc-1",
            chunk_size=100,
            overlap=101,
        )


def test_long_text_produces_multiple_chunks():
    text = "This is a sentence about an employee policy. " * 50

    chunks = chunk_text(
        text,
        document_id="doc-1",
        chunk_size=200,
        overlap=40,
    )

    assert len(chunks) > 1


def test_chunk_indexes_are_sequential():
    text = "This is a sentence about an employee policy. " * 50

    chunks = chunk_text(
        text,
        document_id="doc-1",
        chunk_size=200,
        overlap=40,
    )

    assert [chunk.chunk_index for chunk in chunks] == list(range(len(chunks)))


def test_document_id_is_preserved_for_every_chunk():
    text = "This is a sentence about an employee policy. " * 50

    chunks = chunk_text(
        text,
        document_id="employee-handbook-001",
        chunk_size=200,
        overlap=40,
    )

    assert all(
        chunk.document_id == "employee-handbook-001"
        for chunk in chunks
    )


def test_chunks_do_not_exceed_configured_size():
    text = "This is a sentence about an employee policy. " * 100

    chunks = chunk_text(
        text,
        document_id="doc-1",
        chunk_size=200,
        overlap=40,
    )

    assert all(len(chunk.text) <= 200 for chunk in chunks)


def test_zero_overlap_does_not_duplicate_previous_content():
    text = (
        "First paragraph contains important information.\n\n"
        "Second paragraph contains different information."
    )

    chunks = chunk_text(
        text,
        document_id="doc-1",
        chunk_size=60,
        overlap=0,
    )

    assert len(chunks) > 1

    for index in range(1, len(chunks)):
        assert chunks[index - 1].text not in chunks[index].text


def test_overlap_preserves_previous_context():
    text = (
        "The resignation policy requires written notice. "
        "Employees must submit the request to HR. "
        "The standard notice period is thirty days. "
        "Managers must acknowledge the request."
    )

    chunks = chunk_text(
        text,
        document_id="doc-1",
        chunk_size=100,
        overlap=30,
    )

    assert len(chunks) > 1

    first = chunks[0].text
    second = chunks[1].text

    # Some tail of the previous chunk should appear in the next chunk.
    previous_tail = first[-30:].strip()

    assert previous_tail in second or second.startswith(
        previous_tail.split(None, 1)[-1]
    )


def test_paragraph_boundaries_are_preferred():
    text = (
        "Paragraph one contains information about employees.\n\n"
        "Paragraph two contains information about salaries.\n\n"
        "Paragraph three contains information about leave."
    )

    chunks = chunk_text(
        text,
        document_id="doc-1",
        chunk_size=100,
        overlap=0,
    )

    assert len(chunks) >= 2

    # We should not arbitrarily split the middle of a paragraph
    # when a paragraph boundary is available.
    combined = "\n".join(chunk.text for chunk in chunks)

    assert "employees" in combined
    assert "salaries" in combined
    assert "leave" in combined


def test_large_single_token_does_not_crash():
    huge_token = "A" * 500

    chunks = chunk_text(
        huge_token,
        document_id="doc-1",
        chunk_size=100,
        overlap=20,
    )

    assert len(chunks) > 1
    assert all(len(chunk.text) <= 100 for chunk in chunks)


def test_default_configuration_is_usable():
    text = "This is a sentence. " * 100

    chunks = chunk_text(
        text,
        document_id="doc-1",
    )

    assert len(chunks) > 1
    assert all(len(chunk.text) <= 1000 for chunk in chunks)