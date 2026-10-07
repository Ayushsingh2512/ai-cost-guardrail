from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass
from typing import Callable, Sequence

from google import genai
from google.genai import types

from app.rag.chunking import DocumentChunk


DEFAULT_MODEL = "gemini-embedding-001"
DEFAULT_DIMENSIONS = 768
DEFAULT_BATCH_SIZE = 50
DEFAULT_MAX_RETRIES = 3



class EmbeddingError(Exception):
    """Base class for embedding-related failures."""


class EmbeddingInputError(EmbeddingError):
    """Raised when embedding input is invalid."""


class EmbeddingRateLimitError(EmbeddingError):
    """Raised when the embedding provider rate limit is exhausted."""


class EmbeddingProviderError(EmbeddingError):
    """Raised when the embedding provider fails."""


@dataclass(frozen=True)
class EmbeddingConfig:
    model: str = DEFAULT_MODEL
    dimensions: int = DEFAULT_DIMENSIONS
    batch_size: int = DEFAULT_BATCH_SIZE
    max_retries: int = DEFAULT_MAX_RETRIES

    @property
    def fingerprint(self) -> str:
        return f"gemini/{self.model}/{self.dimensions}"


@dataclass(frozen=True)
class EmbeddedChunk:
    chunk: DocumentChunk
    embedding: list[float]
    fingerprint: str


class GeminiEmbedder:
    """
    Gemini embedding adapter.

    Responsibilities:
    - validate embedding input
    - batch requests
    - call Gemini
    - retry transient failures
    - normalize vectors
    - validate provider output
    - expose provider-independent errors
    """

    def __init__(
        self,
        config: EmbeddingConfig | None = None,
        *,
        client=None,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.config = config or EmbeddingConfig()
        self.client = client or genai.Client()
        self.sleep = sleep

        self._validate_config()
        
    @property
    def fingerprint(self) -> str:
        return self.config.fingerprint

    def _validate_config(self) -> None:
        if self.config.dimensions <= 0:
            raise ValueError("dimensions must be greater than zero")

        if self.config.batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")

        if self.config.max_retries < 0:
            raise ValueError("max_retries cannot be negative")

    def embed_documents(
        self,
        texts: Sequence[str],
    ) -> list[list[float]]:
        """
        Embed document texts using RETRIEVAL_DOCUMENT.
        """
        texts = list(texts)

        if not texts:
            return []

        self._validate_texts(texts)

        return self._embed(
            texts,
            task_type="RETRIEVAL_DOCUMENT",
        )

    def embed_query(
        self,
        query: str,
    ) -> list[float]:
        """
        Embed a search query using RETRIEVAL_QUERY.
        """
        self._validate_text(query)

        vectors = self._embed(
            [query],
            task_type="RETRIEVAL_QUERY",
        )

        return vectors[0]

    def _embed(
        self,
        texts: list[str],
        *,
        task_type: str,
    ) -> list[list[float]]:
        all_vectors: list[list[float]] = []

        for start in range(0, len(texts), self.config.batch_size):
            batch = texts[
                start : start + self.config.batch_size
            ]

            vectors = self._embed_batch(
                batch,
                task_type=task_type,
            )

            all_vectors.extend(vectors)

        return all_vectors

    def _embed_batch(
        self,
        texts: list[str],
        *,
        task_type: str,
    ) -> list[list[float]]:
        last_error: Exception | None = None

        for attempt in range(self.config.max_retries + 1):
            try:
                response = self.client.models.embed_content(
                    model=self.config.model,
                    contents=texts,
                    config=types.EmbedContentConfig(
                        task_type=task_type,
                        output_dimensionality=self.config.dimensions,
                    ),
                )

                vectors = self._extract_vectors(response)

                if len(vectors) != len(texts):
                    raise EmbeddingProviderError(
                        "Embedding provider returned "
                        f"{len(vectors)} vectors for "
                        f"{len(texts)} inputs"
                    )

                return [
                    self._normalize_and_validate(vector)
                    for vector in vectors
                ]

            except EmbeddingProviderError:
                raise

            except Exception as exc:
                last_error = exc

                code = getattr(exc, "code", None)

                if self._is_rate_limit(code):
                    if attempt >= self.config.max_retries:
                        raise EmbeddingRateLimitError(
                            "Embedding provider rate limit "
                            "persisted after retries"
                        ) from exc

                    self._sleep_before_retry(attempt)
                    continue

                if self._is_retryable(code, exc):
                    if attempt >= self.config.max_retries:
                        raise EmbeddingProviderError(
                            "Embedding provider failed "
                            "after retries"
                        ) from exc

                    self._sleep_before_retry(attempt)
                    continue

                raise EmbeddingProviderError(
                    "Embedding provider request failed"
                ) from exc

        raise EmbeddingProviderError(
            "Embedding provider failed"
        ) from last_error

    def _extract_vectors(self, response) -> list[list[float]]:
        embeddings = getattr(response, "embeddings", None)

        if embeddings is None:
            raise EmbeddingProviderError(
                "Embedding provider returned no embeddings"
            )

        vectors: list[list[float]] = []

        for embedding in embeddings:
            values = getattr(embedding, "values", None)

            if values is None:
                raise EmbeddingProviderError(
                    "Embedding provider returned an embedding "
                    "without vector values"
                )

            vectors.append(list(values))

        return vectors

    def _normalize_and_validate(
        self,
        vector: list[float],
    ) -> list[float]:
        if len(vector) != self.config.dimensions:
            raise EmbeddingProviderError(
                "Embedding provider returned vector with "
                f"{len(vector)} dimensions; expected "
                f"{self.config.dimensions}"
            )

        values = [float(value) for value in vector]

        if not all(math.isfinite(value) for value in values):
            raise EmbeddingProviderError(
                "Embedding provider returned a non-finite vector"
            )

        magnitude = math.sqrt(
            sum(value * value for value in values)
        )

        if magnitude == 0:
            raise EmbeddingProviderError(
                "Embedding provider returned a zero vector"
            )

        return [
            value / magnitude
            for value in values
        ]

    def _validate_texts(
        self,
        texts: Sequence[str],
    ) -> None:
        for text in texts:
            self._validate_text(text)

    def _validate_text(
        self,
        text: str,
    ) -> None:
        if not isinstance(text, str):
            raise EmbeddingInputError(
                "Embedding input must be a string"
            )

        if not text.strip():
            raise EmbeddingInputError(
                "Embedding input cannot be empty"
            )

    def _is_rate_limit(
        self,
        code,
    ) -> bool:
        return code == 429

    def _is_retryable(
        self,
        code,
        exc: Exception,
    ) -> bool:
        if isinstance(code, int) and 500 <= code < 600:
            return True

        # Connection/timeout errors generally have no HTTP code.
        return isinstance(
            exc,
            (
                TimeoutError,
                ConnectionError,
            ),
        )

    def _sleep_before_retry(
        self,
        attempt: int,
    ) -> None:
        # Exponential backoff:
        # retry 1 -> 1 second
        # retry 2 -> 2 seconds
        # retry 3 -> 4 seconds
        delay = 2**attempt
        self.sleep(delay)

    @classmethod
    def from_env(
        cls,
        config: EmbeddingConfig | None = None,
        *,
        sleep: Callable[[float], None] = time.sleep,
    ) -> "GeminiEmbedder":
        """
        Build the real Gemini client using the environment-based
        authentication handled by the Google GenAI SDK.
        """
        client = genai.Client()

        return cls(
            config=config,
            client=client,
            sleep=sleep,
        )


class FakeEmbedder:
    """
    Deterministic local embedder for tests.

    It intentionally does not attempt to reproduce Gemini's
    embedding space. Its purpose is to provide a predictable
    embedding implementation for unit tests.
    """

    def __init__(
        self,
        *,
        dimensions: int = 8,
        model: str = "fake-embedding",
    ):
        if dimensions <= 0:
            raise ValueError(
                "dimensions must be greater than zero"
            )

        self.dimensions = dimensions
        self.model = model

    @property
    def fingerprint(self) -> str:
        return f"fake/{self.model}/{self.dimensions}"

    def embed_documents(
        self,
        texts: Sequence[str],
    ) -> list[list[float]]:
        return [
            self._embed_text(text)
            for text in texts
        ]

    def embed_query(
        self,
        query: str,
    ) -> list[float]:
        return self._embed_text(query)

    def _embed_text(
        self,
        text: str,
    ) -> list[float]:
        if not isinstance(text, str) or not text.strip():
            raise EmbeddingInputError(
                "Embedding input cannot be empty"
            )

        digest = hashlib.sha256(
            text.encode("utf-8")
        ).digest()

        raw: list[float] = []

        for i in range(self.dimensions):
            byte = digest[i % len(digest)]
            raw.append((byte / 255.0) * 2.0 - 1.0)

        magnitude = math.sqrt(
            sum(value * value for value in raw)
        )

        if magnitude == 0:
            raw[0] = 1.0
            magnitude = 1.0

        return [
            value / magnitude
            for value in raw
        ]


def embed_chunks(
    embedder,
    chunks: Sequence[DocumentChunk],
) -> list[EmbeddedChunk]:
    """
    Embed DocumentChunks while preserving their association
    with the resulting vectors.
    """
    chunks = list(chunks)

    if not chunks:
        return []

    texts = [chunk.text for chunk in chunks]

    vectors = embedder.embed_documents(texts)

    if len(vectors) != len(chunks):
        raise EmbeddingProviderError(
            "Number of embeddings does not match "
            "number of chunks"
        )

    fingerprint = embedder.fingerprint

    return [
        EmbeddedChunk(
            chunk=chunk,
            embedding=vector,
            fingerprint=fingerprint,
        )
        for chunk, vector in zip(
            chunks,
            vectors,
            strict=True,
        )
    ]