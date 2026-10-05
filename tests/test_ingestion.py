import fitz
import pytest

from app.rag.ingestion import DocumentPage, extract_pdf


def create_test_pdf(path, pages: list[str]) -> None:
    """Create a small PDF for testing."""
    document = fitz.open()

    for text in pages:
        page = document.new_page()
        page.insert_text((72, 72), text)

    document.save(path)
    document.close()


def test_extract_pdf_returns_document_pages(tmp_path):
    pdf_path = tmp_path / "test.pdf"

    create_test_pdf(
        pdf_path,
        ["Hello from page one."],
    )

    pages = extract_pdf(
        pdf_path,
        document_id="doc-1",
    )

    assert len(pages) == 1
    assert isinstance(pages[0], DocumentPage)


def test_page_text_is_extracted(tmp_path):
    pdf_path = tmp_path / "test.pdf"

    create_test_pdf(
        pdf_path,
        ["Employees are entitled to annual leave."],
    )

    pages = extract_pdf(
        pdf_path,
        document_id="doc-1",
    )

    assert pages[0].text == "Employees are entitled to annual leave."


def test_multiple_pages_are_extracted_in_order(tmp_path):
    pdf_path = tmp_path / "test.pdf"

    create_test_pdf(
        pdf_path,
        [
            "This is page one.",
            "This is page two.",
            "This is page three.",
        ],
    )

    pages = extract_pdf(
        pdf_path,
        document_id="doc-1",
    )

    assert len(pages) == 3

    assert pages[0].page_number == 1
    assert pages[1].page_number == 2
    assert pages[2].page_number == 3

    assert pages[0].text == "This is page one."
    assert pages[1].text == "This is page two."
    assert pages[2].text == "This is page three."


def test_document_id_is_preserved(tmp_path):
    pdf_path = tmp_path / "test.pdf"

    create_test_pdf(
        pdf_path,
        [
            "Page one.",
            "Page two.",
        ],
    )

    pages = extract_pdf(
        pdf_path,
        document_id="employee-handbook-123",
    )

    assert all(
        page.document_id == "employee-handbook-123"
        for page in pages
    )


def test_metadata_is_preserved_on_every_page(tmp_path):
    pdf_path = tmp_path / "handbook.pdf"

    create_test_pdf(
        pdf_path,
        [
            "Page one.",
            "Page two.",
        ],
    )

    metadata = {
        "tenant_id": 42,
        "department": "engineering",
    }

    pages = extract_pdf(
        pdf_path,
        document_id="doc-1",
        metadata=metadata,
    )

    for page in pages:
        assert page.metadata["tenant_id"] == 42
        assert page.metadata["department"] == "engineering"


def test_source_filename_is_added_to_metadata(tmp_path):
    pdf_path = tmp_path / "employee_handbook.pdf"

    create_test_pdf(
        pdf_path,
        ["Employee handbook content."],
    )

    pages = extract_pdf(
        pdf_path,
        document_id="doc-1",
    )

    assert pages[0].metadata["source"] == "employee_handbook.pdf"


def test_page_number_is_added_to_metadata(tmp_path):
    pdf_path = tmp_path / "test.pdf"

    create_test_pdf(
        pdf_path,
        [
            "Page one.",
            "Page two.",
        ],
    )

    pages = extract_pdf(
        pdf_path,
        document_id="doc-1",
    )

    assert pages[0].metadata["page"] == 1
    assert pages[1].metadata["page"] == 2


def test_metadata_does_not_override_source_or_page(tmp_path):
    pdf_path = tmp_path / "actual.pdf"

    create_test_pdf(
        pdf_path,
        ["Actual content."],
    )

    pages = extract_pdf(
        pdf_path,
        document_id="doc-1",
        metadata={
            "source": "fake.pdf",
            "page": 999,
            "tenant_id": 42,
        },
    )

    assert pages[0].metadata["source"] == "actual.pdf"
    assert pages[0].metadata["page"] == 1
    assert pages[0].metadata["tenant_id"] == 42


def test_missing_pdf_raises_file_not_found(tmp_path):
    pdf_path = tmp_path / "missing.pdf"

    with pytest.raises(FileNotFoundError):
        extract_pdf(
            pdf_path,
            document_id="doc-1",
        )


def test_directory_is_rejected(tmp_path):
    directory = tmp_path / "documents"
    directory.mkdir()

    with pytest.raises(ValueError):
        extract_pdf(
            directory,
            document_id="doc-1",
        )


def test_non_pdf_file_is_rejected(tmp_path):
    text_file = tmp_path / "document.txt"
    text_file.write_text("This is not a PDF.")

    with pytest.raises(ValueError):
        extract_pdf(
            text_file,
            document_id="doc-1",
        )



def test_page_with_no_text_returns_empty_text(tmp_path):
    pdf_path = tmp_path / "image_like.pdf"

    document = fitz.open()
    document.new_page()
    document.save(pdf_path)
    document.close()

    pages = extract_pdf(
        pdf_path,
        document_id="doc-1",
    )

    assert len(pages) == 1
    assert pages[0].text == ""
    assert pages[0].page_number == 1