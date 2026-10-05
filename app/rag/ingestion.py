from dataclasses import dataclass, field
from pathlib import Path

import fitz


@dataclass(frozen=True)
class DocumentPage:
    document_id: str
    page_number: int
    text: str
    metadata: dict = field(default_factory=dict)


def extract_pdf(
    file_path: str | Path,
    document_id: str,
    metadata: dict | None = None,
) -> list[DocumentPage]:
    """
    Extract text from a PDF page by page.

    Each page becomes a DocumentPage so that page-level
    provenance can be preserved during later chunking
    and retrieval.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    if not path.is_file():
        raise ValueError(f"Path is not a file: {path}")

    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a PDF file, got: {path.suffix}")

    pages: list[DocumentPage] = []
    base_metadata = dict(metadata or {})

    with fitz.open(path) as document:
        for page_index, page in enumerate(document):
            text = page.get_text("text").strip()

            page_metadata = {
                **base_metadata,
                "source": path.name,
                "page": page_index + 1,
            }

            pages.append(
                DocumentPage(
                    document_id=document_id,
                    page_number=page_index + 1,
                    text=text,
                    metadata=page_metadata,
                )
            )

    return pages