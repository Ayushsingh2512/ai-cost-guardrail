import math
from types import SimpleNamespace

import pytest

from app.rag.chunking import DocumentChunk
from app.rag.embeddings import (
    EmbeddedChunk,
    EmbeddingConfig,
    EmbeddingError,
    EmbeddingInputError,
    EmbeddingProviderError,
    EmbeddingRateLimitError,
    FakeEmbedder,
    GeminiEmbedder,
    embed_chunks,
)


class StubAPIError(Exception):
    def __init__(self, code):
        super().__init__(f"stub error {code}")
        self.code = code


class StubClient:
    """
    Mimics client.models.embed_content.

    `script` items:
        None      -> return a normal response
        Exception -> raise that exception

    Returned vectors encode the input number as:
        [number + 1, 1, ...]

    This lets tests verify that the association between input
    and output is preserved even after normalization.
    """

    def __init__(
        self,
        script=None,
        dims=2,
        wrong_count=False,
        wrong_dims=False,
    ):
        self.calls = []
        self.script = list(script or [])
        self.dims = dims
        self.wrong_count = wrong_count
        self.wrong_dims = wrong_dims
        self.models = self

    def embed_content(self, model, contents, config):
        self.calls.append(
            SimpleNamespace(
                model=model,
                contents=list(contents),
                config=config,
            )
        )

        if self.script:
            step = self.script.pop(0)

            if step is not None:
                raise step

        output_contents = (
            list(contents)[:-1]
            if self.wrong_count
            else list(contents)
        )

        dimensions = (
            self.dims + 1
            if self.wrong_dims
            else self.dims
        )

        vectors = []

        for text in output_contents:
            number = float(text[1:])

            vectors.append(
                SimpleNamespace(
                    values=[
                        number + 1.0
                    ]
                    + [1.0] * (dimensions - 1)
                )
            )

        return SimpleNamespace(
            embeddings=vectors
        )


def make(client, **config):
    base = {
        "dimensions": 2,
        "batch_size": 2,
        "max_retries": 3,
    }

    base.update(config)

    return GeminiEmbedder(
        EmbeddingConfig(**base),
        client=client,
        sleep=lambda _: None,
    )


def test_order_preserved_across_batches():
    client = StubClient()

    texts = [
        "t0",
        "t1",
        "t2",
        "t3",
        "t4",
    ]

    vectors = make(client).embed_documents(texts)

    assert [call.contents for call in client.calls] == [
        ["t0", "t1"],
        ["t2", "t3"],
        ["t4"],
    ]

    # The vector direction should preserve the input identity.
    # For [n + 1, 1], the ratio is n + 1.
    for i, vector in enumerate(vectors):
        assert vector[0] / vector[1] == pytest.approx(i + 1)


def test_vectors_are_unit_length_with_configured_dimensions():
    vectors = make(
        StubClient()
    ).embed_documents(
        ["t1", "t2", "t3"]
    )

    for vector in vectors:
        assert len(vector) == 2

        magnitude = math.sqrt(
            sum(value * value for value in vector)
        )

        assert magnitude == pytest.approx(1.0)


def test_task_types_differ_for_documents_and_queries():
    client = StubClient()

    embedder = make(client)

    embedder.embed_documents(["t1"])
    embedder.embed_query("t1")

    assert (
        client.calls[0].config.task_type
        == "RETRIEVAL_DOCUMENT"
    )

    assert (
        client.calls[1].config.task_type
        == "RETRIEVAL_QUERY"
    )


def test_empty_list_returns_empty_without_api_call():
    client = StubClient()

    assert make(client).embed_documents([]) == []
    assert client.calls == []


@pytest.mark.parametrize(
    "bad",
    [
        "",
        "   ",
        "\n",
    ],
)
def test_empty_text_rejected_before_any_api_call(bad):
    client = StubClient()

    with pytest.raises(EmbeddingInputError):
        make(client).embed_documents(
            ["t1", bad]
        )

    assert client.calls == []


def test_count_mismatch_raises():
    client = StubClient(
        wrong_count=True
    )

    with pytest.raises(EmbeddingProviderError):
        make(client).embed_documents(
            ["t1", "t2"]
        )


def test_wrong_dimension_from_provider_raises():
    client = StubClient(
        wrong_dims=True
    )

    with pytest.raises(EmbeddingProviderError):
        make(client).embed_documents(
            ["t1"]
        )


def test_429_is_retried_then_succeeds():
    client = StubClient(
        script=[
            StubAPIError(429),
            StubAPIError(503),
            None,
        ]
    )

    vectors = make(
        client
    ).embed_documents(["t1"])

    assert len(vectors) == 1
    assert len(client.calls) == 3


def test_429_exhausts_retries():
    client = StubClient(
        script=[
            StubAPIError(429),
            StubAPIError(429),
            StubAPIError(429),
            StubAPIError(429),
        ]
    )

    with pytest.raises(EmbeddingRateLimitError):
        make(
            client,
            max_retries=2,
        ).embed_documents(["t1"])

    # Initial request + 2 retries.
    assert len(client.calls) == 3


def test_400_is_not_retried():
    client = StubClient(
        script=[
            StubAPIError(400),
        ]
    )

    with pytest.raises(EmbeddingProviderError):
        make(client).embed_documents(["t1"])

    assert len(client.calls) == 1


def test_errors_share_a_base_class():
    for exception in (
        EmbeddingInputError,
        EmbeddingRateLimitError,
        EmbeddingProviderError,
    ):
        assert issubclass(
            exception,
            EmbeddingError,
        )


def test_batch_failure_is_atomic_no_partial_result():
    client = StubClient(
        script=[
            None,
            StubAPIError(400),
        ]
    )

    with pytest.raises(EmbeddingProviderError):
        make(client).embed_documents(
            [
                "t1",
                "t2",
                "t3",
            ]
        )


def test_embed_chunks_pairs_chunks_with_vectors_and_fingerprint():
    chunks = [
        DocumentChunk(
            "doc",
            i,
            f"t{i}",
            {"tenant_id": 7},
        )
        for i in range(3)
    ]

    embedder = make(
        StubClient()
    )

    output = embed_chunks(
        embedder,
        chunks,
    )

    assert [
        item.chunk
        for item in output
    ] == chunks

    assert all(
        isinstance(item, EmbeddedChunk)
        for item in output
    )

    assert {
        item.fingerprint
        for item in output
    } == {
        "gemini/gemini-embedding-001/2"
    }

    assert (
        output[2].embedding[0]
        / output[2].embedding[1]
        == pytest.approx(3)
    )


def test_embed_chunks_empty():
    client = StubClient()

    assert (
        embed_chunks(
            make(client),
            [],
        )
        == []
    )

    assert client.calls == []


def test_fake_embedder_is_deterministic_and_normalized():
    fake = FakeEmbedder()

    first = fake.embed_documents(
        ["hello"]
    )[0]

    second = fake.embed_documents(
        ["hello"]
    )[0]

    assert first == second
    assert len(first) == fake.dimensions

    magnitude = math.sqrt(
        sum(value * value for value in first)
    )

    assert magnitude == pytest.approx(1.0)