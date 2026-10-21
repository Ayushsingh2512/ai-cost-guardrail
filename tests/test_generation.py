from types import SimpleNamespace

import pytest

from app.rag.generation import (
    PROMPT_VERSION,
    ContextPassage,
    GenerationConfig,
    GenerationInputError,
    GenerationProviderError,
    GenerationRateLimitError,
    GenerationService,
    GenerationTimeoutError,
)


class FakeError(Exception):
    def __init__(self, message: str, code=None):
        super().__init__(message)
        self.code = code


class FakeModels:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.calls = []

    async def generate_content(
        self,
        *,
        model,
        contents,
        config,
    ):
        self.calls.append(
            {
                "model": model,
                "contents": contents,
                "config": config,
            }
        )

        if self.error is not None:
            raise self.error

        return self.response


class FakeClient:
    def __init__(self, response=None, error=None):
        self.aio = SimpleNamespace(
            models=FakeModels(
                response=response,
                error=error,
            )
        )


def make_response(
    *,
    text: str | None = "Grounded answer [1].",
    finish_reason: str = "STOP",
    prompt_tokens: int = 100,
    completion_tokens: int = 20,
    response_id: str = "resp-123",
):
    return SimpleNamespace(
        text=text,
        candidates=[
            SimpleNamespace(
                finish_reason=finish_reason,
            )
        ],
        usage_metadata=SimpleNamespace(
            prompt_token_count=prompt_tokens,
            candidates_token_count=completion_tokens,
        ),
        response_id=response_id,
        prompt_feedback=None,
    )


@pytest.fixture
def passages():
    return [
        ContextPassage(
            ref=("doc-1", 3),
            text="Refunds are available within 30 days.",
            source="policy.pdf",
            page=3,
        ),
        ContextPassage(
            ref=("doc-1", 4),
            text="Refund requests require the order number.",
            source="policy.pdf",
            page=4,
        ),
    ]


@pytest.mark.asyncio
async def test_generate_returns_grounded_answer(passages):
    response = make_response(text="Refunds are available within 30 days [1].")

    client = FakeClient(response=response)

    service = GenerationService(
        config=GenerationConfig(
            model="gemini-3-flash-preview",
            temperature=0.0,
        ),
        client=client,
    )

    result = await service.generate(
        query="What is the refund period?",
        passages=passages,
    )

    assert result.status == "answered"
    assert result.answer == ("Refunds are available within 30 days [1].")
    assert result.passages_sent == [
        ("doc-1", 3),
        ("doc-1", 4),
    ]
    assert len(result.citations) == 1
    assert result.citations[0].ref == ("doc-1", 3)
    assert result.citations[0].source == "policy.pdf"
    assert result.citations[0].page == 3
    assert result.invalid_citation_markers == 0
    assert result.finish_reason == "stop"
    assert result.prompt_version == PROMPT_VERSION

    assert result.provider is not None
    assert result.provider.provider == "gemini"
    assert result.provider.model == "gemini-3-flash-preview"
    assert result.provider.prompt_tokens == 100
    assert result.provider.completion_tokens == 20
    assert result.provider.request_id == "resp-123"
    assert result.provider.latency_ms >= 0


@pytest.mark.asyncio
async def test_generate_sends_numbered_evidence_and_question(passages):
    response = make_response()

    client = FakeClient(response=response)

    service = GenerationService(
        config=GenerationConfig(),
        client=client,
    )

    await service.generate(
        query="What is the refund policy?",
        passages=passages,
    )

    assert len(client.aio.models.calls) == 1

    call = client.aio.models.calls[0]

    assert call["model"] == "gemini-3-flash-preview"

    prompt = call["contents"]

    assert "EVIDENCE" in prompt
    assert "[1]" in prompt
    assert "[2]" in prompt
    assert "Refunds are available within 30 days." in prompt
    assert "Refund requests require the order number." in prompt
    assert "QUESTION" in prompt
    assert "What is the refund policy?" in prompt

    assert call["config"].temperature == 0.0


@pytest.mark.asyncio
async def test_generate_deduplicates_valid_citations(passages):
    response = make_response(
        text="Refunds are available [1]. "
        "They require an order number [2]. "
        "The same refund period is confirmed [1]."
    )

    service = GenerationService(client=FakeClient(response=response))

    result = await service.generate(
        query="What is the refund policy?",
        passages=passages,
    )

    assert [citation.ref for citation in result.citations] == [
        ("doc-1", 3),
        ("doc-1", 4),
    ]
    assert result.invalid_citation_markers == 0


@pytest.mark.asyncio
async def test_generate_drops_invalid_citations(passages):
    response = make_response(
        text="Answer [1]. Unknown source [99]. Invalid source [0]."
    )

    service = GenerationService(client=FakeClient(response=response))

    result = await service.generate(
        query="Question?",
        passages=passages,
    )

    assert [citation.ref for citation in result.citations] == [("doc-1", 3)]
    assert result.invalid_citation_markers == 2


@pytest.mark.asyncio
async def test_generate_empty_context_short_circuits():
    client = FakeClient(response=make_response())

    service = GenerationService(client=client)

    result = await service.generate(
        query="What is the answer?",
        passages=[],
    )

    assert result.status == "no_context"
    assert result.answer is None
    assert result.citations == []
    assert result.passages_sent == []
    assert result.provider is None
    assert result.finish_reason == "other"
    assert result.invalid_citation_markers == 0

    assert client.aio.models.calls == []


@pytest.mark.asyncio
async def test_generate_maps_safety_block_to_result():
    response = make_response(
        text=None,
        finish_reason="SAFETY",
    )

    client = FakeClient(response=response)

    service = GenerationService(client=client)

    result = await service.generate(
        query="Question?",
        passages=[
            ContextPassage(
                ref=("doc-1", 1),
                text="Evidence",
            )
        ],
    )

    assert result.status == "blocked"
    assert result.answer is None
    assert result.citations == []
    assert result.finish_reason == "safety"
    assert result.provider is not None


@pytest.mark.asyncio
async def test_generate_maps_prompt_block_to_result():
    response = make_response(
        text=None,
        finish_reason="OTHER",
    )

    response.prompt_feedback = SimpleNamespace(block_reason="SAFETY")

    client = FakeClient(response=response)

    service = GenerationService(client=client)

    result = await service.generate(
        query="Question?",
        passages=[
            ContextPassage(
                ref=("doc-1", 1),
                text="Evidence",
            )
        ],
    )

    assert result.status == "blocked"
    assert result.answer is None
    assert result.finish_reason == "safety"


@pytest.mark.asyncio
async def test_generate_maps_max_tokens_to_length(passages):
    response = make_response(
        text="Partial answer [1].",
        finish_reason="MAX_TOKENS",
    )

    service = GenerationService(client=FakeClient(response=response))

    result = await service.generate(
        query="Question?",
        passages=passages,
    )

    assert result.status == "answered"
    assert result.finish_reason == "length"


@pytest.mark.asyncio
async def test_generate_raises_rate_limit_error(passages):
    client = FakeClient(
        error=FakeError(
            "rate limited",
            code=429,
        )
    )

    service = GenerationService(client=client)

    with pytest.raises(GenerationRateLimitError):
        await service.generate(
            query="Question?",
            passages=passages,
        )


@pytest.mark.asyncio
async def test_generate_raises_provider_error_for_5xx(passages):
    client = FakeClient(
        error=FakeError(
            "server error",
            code=503,
        )
    )

    service = GenerationService(client=client)

    with pytest.raises(GenerationProviderError):
        await service.generate(
            query="Question?",
            passages=passages,
        )


@pytest.mark.asyncio
async def test_generate_raises_timeout_error(passages):
    client = FakeClient(error=TimeoutError("timed out"))

    service = GenerationService(client=client)

    with pytest.raises(GenerationTimeoutError):
        await service.generate(
            query="Question?",
            passages=passages,
        )


@pytest.mark.asyncio
async def test_generate_raises_provider_error_for_connection_failure(
    passages,
):
    client = FakeClient(error=ConnectionError("connection failed"))

    service = GenerationService(client=client)

    with pytest.raises(GenerationProviderError):
        await service.generate(
            query="Question?",
            passages=passages,
        )


@pytest.mark.asyncio
async def test_generate_rejects_empty_query(passages):
    service = GenerationService(client=FakeClient(response=make_response()))

    with pytest.raises(GenerationInputError):
        await service.generate(
            query="   ",
            passages=passages,
        )


@pytest.mark.asyncio
async def test_generate_rejects_empty_passage_text():
    service = GenerationService(client=FakeClient(response=make_response()))

    passages = [
        ContextPassage(
            ref=("doc-1", 1),
            text="   ",
        )
    ]

    with pytest.raises(GenerationInputError):
        await service.generate(
            query="Question?",
            passages=passages,
        )


def test_generation_config_rejects_invalid_temperature():
    with pytest.raises(ValueError):
        GenerationConfig(temperature=-0.1)

    with pytest.raises(ValueError):
        GenerationConfig(temperature=2.1)


def test_generation_config_rejects_empty_model():
    with pytest.raises(ValueError):
        GenerationConfig(model="   ")


def test_generation_config_defaults_max_output_tokens():
    config = GenerationConfig()

    assert config.max_output_tokens == 2000


def test_generation_config_accepts_custom_max_output_tokens():
    config = GenerationConfig(
        max_output_tokens=1000,
    )

    assert config.max_output_tokens == 1000


@pytest.mark.parametrize("max_output_tokens", [0, -1])
def test_generation_config_rejects_invalid_max_output_tokens(
    max_output_tokens,
):
    with pytest.raises(
        ValueError,
        match="max_output_tokens must be greater than zero",
    ):
        GenerationConfig(
            max_output_tokens=max_output_tokens,
        )


@pytest.mark.asyncio
async def test_generate_passes_max_output_tokens_to_provider():
    captured = {}

    class FakeModels:
        async def generate_content(self, **kwargs):
            captured.update(kwargs)

            return SimpleNamespace(
                text="The answer is supported. [1]",
                usage_metadata=SimpleNamespace(
                    prompt_token_count=10,
                    candidates_token_count=5,
                ),
                candidates=[
                    SimpleNamespace(
                        finish_reason="STOP",
                    )
                ],
                prompt_feedback=None,
            )

    class FakeAio:
        def __init__(self):
            self.models = FakeModels()

    class FakeClient:
        def __init__(self):
            self.aio = FakeAio()

    service = GenerationService(
        config=GenerationConfig(
            max_output_tokens=1500,
        ),
        client=FakeClient(),
    )

    passage = ContextPassage(
        ref=("doc-1", 0),
        text="The company provides 20 days of annual leave.",
        source="handbook.pdf",
        page=3,
    )

    result = await service.generate(
        query="How many annual leave days are provided?",
        passages=[passage],
    )

    assert result.status == "answered"
    assert captured["config"].max_output_tokens == 1500


# ---------------------------------------------------------------------------
# New tests for task: build_prompt, thinking/total tokens, count_prompt_tokens,
# optional prompt= kwarg in generate().
# ---------------------------------------------------------------------------


def test_build_prompt_is_public_and_matches_private(passages):
    """build_prompt() produces the same string as the internal _build_prompt()."""
    service = GenerationService()

    public = service.build_prompt(
        query="What is the refund period?",
        passages=passages,
    )
    private = service._build_prompt(
        query="What is the refund period?",
        passages=passages,
    )

    assert public == private
    assert "EVIDENCE" in public
    assert "[1]" in public
    assert "[2]" in public
    assert "QUESTION" in public
    assert "What is the refund period?" in public


def test_build_prompt_numbers_passages_sequentially():
    """Passages are numbered [1], [2], … in order in the prompt."""
    service = GenerationService()

    p1 = ContextPassage(ref=("d", 0), text="First passage.")
    p2 = ContextPassage(ref=("d", 1), text="Second passage.")
    p3 = ContextPassage(ref=("d", 2), text="Third passage.")

    prompt = service.build_prompt(query="Q?", passages=[p1, p2, p3])

    # The marker lines appear in ascending order.
    idx1 = prompt.index("[1]")
    idx2 = prompt.index("[2]")
    idx3 = prompt.index("[3]")
    assert idx1 < idx2 < idx3


def test_provider_meta_thinking_and_total_tokens_default_none():
    """Existing code that doesn't pass thinking/total_tokens still works."""
    from app.rag.generation import ProviderMeta

    meta = ProviderMeta(
        provider="gemini",
        model="gemini-3-flash-preview",
        prompt_tokens=10,
        completion_tokens=5,
        latency_ms=42,
    )

    assert meta.thinking_tokens is None
    assert meta.total_tokens is None


def test_provider_meta_accepts_thinking_and_total_tokens():
    """thinking_tokens and total_tokens are stored correctly when provided."""
    from app.rag.generation import ProviderMeta

    meta = ProviderMeta(
        provider="gemini",
        model="gemini-3-flash-preview",
        prompt_tokens=10,
        completion_tokens=5,
        latency_ms=42,
        thinking_tokens=7,
        total_tokens=22,
    )

    assert meta.thinking_tokens == 7
    assert meta.total_tokens == 22


@pytest.mark.asyncio
async def test_build_provider_meta_populates_thinking_and_total_tokens(passages):
    """
    When usage_metadata contains thoughts_token_count and total_token_count
    the returned ProviderMeta exposes them.
    """
    response = SimpleNamespace(
        text="Answer [1].",
        candidates=[SimpleNamespace(finish_reason="STOP")],
        usage_metadata=SimpleNamespace(
            prompt_token_count=50,
            candidates_token_count=10,
            thoughts_token_count=8,
            total_token_count=68,
        ),
        response_id="rid-1",
        prompt_feedback=None,
    )

    service = GenerationService(client=FakeClient(response=response))
    result = await service.generate(query="Q?", passages=passages)

    assert result.provider is not None
    assert result.provider.thinking_tokens == 8
    assert result.provider.total_tokens == 68


@pytest.mark.asyncio
async def test_build_provider_meta_thinking_none_when_absent(passages):
    """
    When usage_metadata does not include thinking/total fields they are None.
    (make_response() does not include them, so this also guards existing tests.)
    """
    service = GenerationService(client=FakeClient(response=make_response()))
    result = await service.generate(query="Q?", passages=passages)

    assert result.provider is not None
    assert result.provider.thinking_tokens is None
    assert result.provider.total_tokens is None


# ---------------------------------------------------------------------------
# count_prompt_tokens tests
# ---------------------------------------------------------------------------


class FakeModelsWithCounting:
    """
    Stub that handles both generate_content and count_tokens.
    `ct_response` is returned by count_tokens; `ct_error` is raised instead.
    """

    def __init__(
        self,
        *,
        gen_response=None,
        gen_error=None,
        ct_response=None,
        ct_error=None,
    ):
        self.gen_response = gen_response
        self.gen_error = gen_error
        self.ct_response = ct_response
        self.ct_error = ct_error
        self.ct_calls: list = []

    async def generate_content(self, *, model, contents, config):
        if self.gen_error is not None:
            raise self.gen_error
        return self.gen_response

    async def count_tokens(self, *, model, contents, config=None):
        self.ct_calls.append(
            SimpleNamespace(model=model, contents=contents, config=config)
        )
        if self.ct_error is not None:
            raise self.ct_error
        return self.ct_response


def make_counting_client(*, ct_tokens: int = 42, ct_error=None, gen_response=None):
    models = FakeModelsWithCounting(
        gen_response=gen_response or make_response(),
        ct_response=SimpleNamespace(total_tokens=ct_tokens),
        ct_error=ct_error,
    )
    client = SimpleNamespace(aio=SimpleNamespace(models=models))
    return client, models


@pytest.mark.asyncio
async def test_count_prompt_tokens_returns_total_tokens():
    """count_prompt_tokens returns the total_tokens from the SDK response."""
    client, models = make_counting_client(ct_tokens=55)

    service = GenerationService(client=client)
    result = await service.count_prompt_tokens("Hello world")

    assert result == 55
    assert len(models.ct_calls) == 1
    call = models.ct_calls[0]
    assert call.contents == "Hello world"
    assert call.model == service.config.model


@pytest.mark.asyncio
async def test_count_prompt_tokens_passes_system_instruction():
    """
    count_prompt_tokens passes a CountTokensConfig with system_instruction
    so that the system instruction tokens are counted in a single API call.
    """
    from google.genai import types

    from app.rag.generation import SYSTEM_INSTRUCTION

    client, models = make_counting_client(ct_tokens=10)
    service = GenerationService(client=client)

    await service.count_prompt_tokens("test prompt")

    call = models.ct_calls[0]
    assert call.config is not None
    assert isinstance(call.config, types.CountTokensConfig)
    assert call.config.system_instruction == SYSTEM_INSTRUCTION


@pytest.mark.asyncio
async def test_count_prompt_tokens_maps_rate_limit_error():
    """Fail-closed: 429 from count_tokens is mapped to GenerationProviderError."""
    client, _ = make_counting_client(ct_error=FakeError("rate limit", code=429))

    service = GenerationService(client=client)

    with pytest.raises(GenerationProviderError):
        await service.count_prompt_tokens("prompt")


@pytest.mark.asyncio
async def test_count_prompt_tokens_maps_server_error():
    """5xx from count_tokens is mapped to GenerationProviderError."""
    client, _ = make_counting_client(ct_error=FakeError("server error", code=503))

    service = GenerationService(client=client)

    with pytest.raises(GenerationProviderError):
        await service.count_prompt_tokens("prompt")


@pytest.mark.asyncio
async def test_count_prompt_tokens_with_query_and_passages(passages):
    """count_prompt_tokens(query, passages) builds grounded prompt and counts tokens."""
    client, models = make_counting_client(ct_tokens=88)
    service = GenerationService(client=client)

    tokens = await service.count_prompt_tokens(
        query="What is the refund period?",
        passages=passages,
    )

    assert tokens == 88
    assert len(models.ct_calls) == 1
    sent_prompt = models.ct_calls[0].contents
    assert "EVIDENCE" in sent_prompt
    assert "QUESTION" in sent_prompt
    assert "What is the refund period?" in sent_prompt


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "invalid_tokens",
    [
        None,
        True,
        False,
        "5",
        2.5,
        -1,
        -10,
    ],
)
async def test_count_prompt_tokens_invalid_or_missing_total_tokens_raises(
    invalid_tokens,
):
    """
    total_tokens that is missing (None), Boolean, non-integer, or negative
    must not silently become zero; it must raise GenerationProviderError.
    """
    client = SimpleNamespace(
        aio=SimpleNamespace(
            models=FakeModelsWithCounting(
                ct_response=SimpleNamespace(total_tokens=invalid_tokens),
            )
        )
    )
    service = GenerationService(client=client)

    with pytest.raises(GenerationProviderError, match="invalid prompt token count"):
        await service.count_prompt_tokens("x")


@pytest.mark.asyncio
async def test_count_prompt_tokens_zero_tokens_is_valid():
    """Zero is a valid non-negative integer token count."""
    client = SimpleNamespace(
        aio=SimpleNamespace(
            models=FakeModelsWithCounting(
                ct_response=SimpleNamespace(total_tokens=0),
            )
        )
    )
    service = GenerationService(client=client)
    result = await service.count_prompt_tokens("x")

    assert result == 0


# ---------------------------------------------------------------------------
# generate() with optional pre-built prompt
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_uses_supplied_prompt_without_rebuilding(passages):
    """
    When a prompt is passed to generate(), it is sent to the provider as-is;
    build_prompt is NOT called again.
    """
    response = make_response()
    client = FakeClient(response=response)
    service = GenerationService(client=client)

    pre_built = "PRE-BUILT PROMPT"

    result = await service.generate(
        query="What is the refund period?",
        passages=passages,
        prompt=pre_built,
    )

    assert result.status == "answered"
    # The provider received exactly our pre-built string.
    assert client.aio.models.calls[0]["contents"] == pre_built


@pytest.mark.asyncio
async def test_generate_builds_prompt_when_none_supplied(passages):
    """
    When prompt=None (the default), generate() builds the prompt from
    query + passages as it always did.
    """
    response = make_response()
    client = FakeClient(response=response)
    service = GenerationService(client=client)

    await service.generate(
        query="What is the refund period?",
        passages=passages,
    )

    sent_prompt = client.aio.models.calls[0]["contents"]
    assert "EVIDENCE" in sent_prompt
    assert "QUESTION" in sent_prompt
    assert "What is the refund period?" in sent_prompt
