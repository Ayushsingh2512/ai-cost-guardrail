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
    EmbedResult,
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

        output_contents = list(contents)[:-1] if self.wrong_count else list(contents)

        dimensions = self.dims + 1 if self.wrong_dims else self.dims

        vectors = []

        for text in output_contents:
            number = float(text[1:])

            vectors.append(
                SimpleNamespace(values=[number + 1.0] + [1.0] * (dimensions - 1))
            )

        return SimpleNamespace(embeddings=vectors)


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
    vectors = make(StubClient()).embed_documents(["t1", "t2", "t3"])

    for vector in vectors:
        assert len(vector) == 2

        magnitude = math.sqrt(sum(value * value for value in vector))

        assert magnitude == pytest.approx(1.0)


def test_task_types_differ_for_documents_and_queries():
    client = StubClient()

    embedder = make(client)

    embedder.embed_documents(["t1"])
    embedder.embed_query("t1")

    assert client.calls[0].config.task_type == "RETRIEVAL_DOCUMENT"

    assert client.calls[1].config.task_type == "RETRIEVAL_QUERY"


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
        make(client).embed_documents(["t1", bad])

    assert client.calls == []


def test_count_mismatch_raises():
    client = StubClient(wrong_count=True)

    with pytest.raises(EmbeddingProviderError):
        make(client).embed_documents(["t1", "t2"])


def test_wrong_dimension_from_provider_raises():
    client = StubClient(wrong_dims=True)

    with pytest.raises(EmbeddingProviderError):
        make(client).embed_documents(["t1"])


def test_429_is_retried_then_succeeds():
    client = StubClient(
        script=[
            StubAPIError(429),
            StubAPIError(503),
            None,
        ]
    )

    vectors = make(client).embed_documents(["t1"])

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

    embedder = make(StubClient())

    output = embed_chunks(
        embedder,
        chunks,
    )

    assert [item.chunk for item in output] == chunks

    assert all(isinstance(item, EmbeddedChunk) for item in output)

    assert {item.fingerprint for item in output} == {"gemini/gemini-embedding-001/2"}

    assert output[2].embedding[0] / output[2].embedding[1] == pytest.approx(3)


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

    first = fake.embed_documents(["hello"])[0]

    second = fake.embed_documents(["hello"])[0]

    assert first == second
    assert len(first) == fake.dimensions

    magnitude = math.sqrt(sum(value * value for value in first))

    assert magnitude == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# count_tokens tests
# ---------------------------------------------------------------------------


class StubCountTokensClient:
    """
    Mimics client.models.count_tokens for unit-testing GeminiEmbedder.count_tokens.

    `ct_script` items (consumed in order):
        None             -> return SimpleNamespace(total_tokens=<ct_tokens>)
        Exception        -> raise that exception

    `ec_script` items are forwarded to the embed_content stub path (unused by
    these tests but kept so the client satisfies the full interface).
    """

    def __init__(
        self,
        ct_script=None,
        ct_tokens: int = 7,
        dims: int = 2,
    ):
        self.ct_calls: list = []
        self.ct_script = list(ct_script or [])
        self.ct_tokens = ct_tokens
        self.dims = dims
        self.models = self

    def count_tokens(self, model, contents):
        self.ct_calls.append(SimpleNamespace(model=model, contents=contents))

        if self.ct_script:
            step = self.ct_script.pop(0)
            if step is not None:
                raise step

        return SimpleNamespace(total_tokens=self.ct_tokens)

    def embed_content(self, model, contents, config):
        # Minimal stub so GeminiEmbedder can be instantiated with this client.
        vectors = [
            SimpleNamespace(values=[1.0] + [0.0] * (self.dims - 1)) for _ in contents
        ]
        return SimpleNamespace(embeddings=vectors)


def make_ct(client, **config):
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


def test_count_tokens_success_returns_api_value():
    """Success path: API returns total_tokens; method propagates it."""
    client = StubCountTokensClient(ct_tokens=42)
    embedder = make_ct(client)

    result = embedder.count_tokens("hello world")

    assert result == 42
    assert len(client.ct_calls) == 1
    assert client.ct_calls[0].contents == "hello world"


def test_count_tokens_retries_on_5xx_then_succeeds():
    """5xx errors are retried; eventual success returns token count."""
    client = StubCountTokensClient(
        ct_script=[
            StubAPIError(503),
            StubAPIError(500),
            None,
        ],
        ct_tokens=5,
    )
    embedder = make_ct(client, max_retries=3)

    result = embedder.count_tokens("retry me")

    assert result == 5
    assert len(client.ct_calls) == 3


def test_count_tokens_rate_limit_exhausted_raises():
    """429s that exhaust all retries raise EmbeddingRateLimitError."""
    client = StubCountTokensClient(
        ct_script=[
            StubAPIError(429),
            StubAPIError(429),
            StubAPIError(429),
        ]
    )
    embedder = make_ct(client, max_retries=2)

    with pytest.raises(EmbeddingRateLimitError):
        embedder.count_tokens("rate limited")

    # Initial + 2 retries = 3 attempts.
    assert len(client.ct_calls) == 3


def test_count_tokens_unsupported_model_falls_back_to_estimate(caplog):
    """
    A 400 error with 'unsupported' in the message triggers the fallback path.
    The fallback returns ceil(len(text) / 3) and logs a warning.
    """

    class UnsupportedModelError(Exception):
        code = 400

        def __str__(self):
            return "model does not support token counting: unsupported operation"

    client = StubCountTokensClient(
        ct_script=[UnsupportedModelError()],
    )
    embedder = make_ct(client)

    text = "hello world!"  # 12 chars -> ceil(12/3) = 4
    import math

    expected = math.ceil(len(text) / 3)

    import logging

    with caplog.at_level(logging.WARNING, logger="app.rag.embeddings"):
        result = embedder.count_tokens(text)

    assert result == expected
    assert len(client.ct_calls) == 1  # no retry on unsupported model
    assert any("estimate" in r.message for r in caplog.records)


def test_count_tokens_empty_input_raises_for_fake_embedder():
    """
    FakeEmbedder.count_tokens raises EmbeddingInputError on empty/blank input,
    matching the validation contract of embed_query.
    """
    from app.rag.embeddings import FakeEmbedder

    fake = FakeEmbedder()

    with pytest.raises(EmbeddingInputError):
        fake.count_tokens("")

    with pytest.raises(EmbeddingInputError):
        fake.count_tokens("   ")


def test_gemini_embed_query_with_usage_calls_count_tokens():
    """GeminiEmbedder.embed_query_with_usage calls count_tokens exactly once with right model and query."""
    client = StubCountTokensClient(ct_tokens=27)
    embedder = make_ct(client)

    res = embedder.embed_query_with_usage("search query")

    assert isinstance(res, EmbedResult)
    assert res.tokens == 27
    assert len(res.vector) == 2
    assert len(client.ct_calls) == 1
    assert client.ct_calls[0].model == embedder.config.model
    assert client.ct_calls[0].contents == "search query"


def test_gemini_embed_query_with_usage_count_tokens_failure_raises():
    """count_tokens failure in embed_query_with_usage raises EmbeddingProviderError."""
    client = StubCountTokensClient(ct_script=[RuntimeError("count failed")])
    embedder = make_ct(client)

    with pytest.raises(
        EmbeddingProviderError, match="Embedding token counting is unavailable"
    ):
        embedder.embed_query_with_usage("search query")


def test_fake_embedder_embed_query_with_usage_deterministic():
    """FakeEmbedder.embed_query_with_usage returns a stable deterministic count for the same input."""
    fake = FakeEmbedder(dimensions=4)
    query = "Deterministic test query"

    res1 = fake.embed_query_with_usage(query)
    res2 = fake.embed_query_with_usage(query)

    assert isinstance(res1, EmbedResult)
    assert res1.tokens == res2.tokens
    assert res1.vector == res2.vector
    assert isinstance(res1.tokens, int)
    assert res1.tokens > 0


def test_fake_embedder_both_methods():
    """FakeEmbedder supports both embed_query and embed_query_with_usage."""
    fake = FakeEmbedder(dimensions=4)
    query = "Search query for testing"

    vector = fake.embed_query(query)
    assert isinstance(vector, list)
    assert len(vector) == 4

    result = fake.embed_query_with_usage(query)
    assert isinstance(result, EmbedResult)
    assert result.vector == vector
    assert result.tokens == fake.count_tokens(query)
