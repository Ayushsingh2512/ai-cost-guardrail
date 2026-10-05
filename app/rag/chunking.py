import re
from dataclasses import dataclass, field


_SENTENCE_PATTERN = re.compile(r"(?<=[.!?])\s+")

_SEPARATORS = (
    "\n\n",  # paragraph
    "\n",    # line
    " ",     # word
)


@dataclass(frozen=True)
class DocumentChunk:
    document_id: str
    chunk_index: int
    text: str
    metadata: dict = field(default_factory=dict)


def _split_text(text: str, max_length: int, separator_index: int = 0) -> list[str]:
    """
    Recursively split text while preferring larger semantic boundaries.

    Order:
        paragraph -> line -> sentence -> word -> characters
    """
    text = text.strip()

    if not text:
        return []

    if len(text) <= max_length:
        return [text]

    # Paragraph / line / word separators.
    if separator_index < len(_SEPARATORS):
        separator = _SEPARATORS[separator_index]
        parts = text.split(separator)

        if len(parts) > 1:
            result: list[str] = []

            for part in parts:
                part = part.strip()
                if not part:
                    continue

                result.extend(
                    _split_text(
                        part,
                        max_length,
                        separator_index + 1,
                    )
                )

            return result

        # Separator doesn't exist in this text.
        return _split_text(
            text,
            max_length,
            separator_index + 1,
        )

    # Sentence splitting is deliberately after paragraph/line splitting.
    # This gives us a better chance of preserving document structure.
    sentences = _SENTENCE_PATTERN.split(text)

    if len(sentences) > 1:
        result = []

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            result.extend(
                _split_text(
                    sentence,
                    max_length,
                    len(_SEPARATORS),
                )
            )

        return result

    # Final fallback: hard character split.
    return [
        text[i : i + max_length]
        for i in range(0, len(text), max_length)
    ]


def _word_safe_tail(text: str, overlap: int) -> str:
    """
    Return up to `overlap` characters from the end of text,
    avoiding a partial word when possible.
    """
    if overlap <= 0:
        return ""

    if len(text) <= overlap:
        return text

    tail = text[-overlap:]

    # If the overlap begins in the middle of a word,
    # discard that partial word.
    if not tail[0].isspace() and not text[-overlap - 1].isspace():
        parts = tail.split(None, 1)

        if len(parts) > 1:
            tail = parts[1]
        else:
            # One giant token with no whitespace.
            return tail

    return tail.strip()


def chunk_text(
    text: str,
    document_id: str,
    metadata: dict | None = None,
    chunk_size: int = 1000,
    overlap: int = 200,
) -> list[DocumentChunk]:
    """
    Split document text into retrieval-friendly chunks.

    Parameters
    ----------
    text:
        Extracted document text.

    document_id:
        Stable identifier for the source document.

    metadata:
        Metadata attached to every generated chunk.

    chunk_size:
        Maximum number of characters in a final chunk.

    overlap:
        Maximum amount of previous-chunk text reused as context.

    Returns
    -------
    list[DocumentChunk]
        Deterministic chunks ordered by chunk_index.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")

    if overlap < 0:
        raise ValueError("overlap cannot be negative")

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    normalized_text = text.strip()

    if not normalized_text:
        return []

    # We need room for the overlap and the newline joining
    # overlap + body.
    body_limit = chunk_size - overlap - 1

    # If chunk_size is too small to leave room for a body,
    # fail explicitly rather than producing broken chunks.
    if body_limit <= 0:
        raise ValueError(
            "chunk_size must be large enough for the requested overlap"
        )

    atoms = _split_text(
        normalized_text,
        max_length=body_limit,
    )

    chunks: list[DocumentChunk] = []
    current = ""

    for atom in atoms:
        atom = atom.strip()

        if not atom:
            continue

        candidate = (
            f"{current}\n{atom}"
            if current
            else atom
        )

        if current and len(candidate) > body_limit:
            chunks.append(
                DocumentChunk(
                    document_id=document_id,
                    chunk_index=len(chunks),
                    text=current,
                    metadata=dict(metadata or {}),
                )
            )

            current = atom
        else:
            current = candidate

    if current:
        chunks.append(
            DocumentChunk(
                document_id=document_id,
                chunk_index=len(chunks),
                text=current,
                metadata=dict(metadata or {}),
            )
        )

    # Add overlap after the initial body chunks have been created.
    final_chunks: list[DocumentChunk] = []

    for index, chunk in enumerate(chunks):
        if index == 0 or overlap == 0:
            final_text = chunk.text
        else:
            tail = _word_safe_tail(
                chunks[index - 1].text,
                overlap,
            )

            final_text = (
                f"{tail}\n{chunk.text}"
                if tail
                else chunk.text
            )

        # Defensive invariant: no final chunk may exceed chunk_size.
        if len(final_text) > chunk_size:
            raise RuntimeError(
                "Chunking invariant violated: "
                f"chunk {index} has {len(final_text)} characters "
                f"(limit={chunk_size})"
            )

        final_chunks.append(
            DocumentChunk(
                document_id=document_id,
                chunk_index=index,
                text=final_text,
                metadata=dict(chunk.metadata),
            )
        )

    return final_chunks