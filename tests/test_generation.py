from types import SimpleNamespace

import pytest

from app.rag.generation import (
    ContextPassage,
    GenerationConfig,
    GenerationInputError,
    GenerationProviderError,
    GenerationRateLimitError,
    GenerationService,
    GenerationTimeoutError,
    PROMPT_VERSION,
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
    response = make_response(
        text="Refunds are available within 30 days [1]."
    )

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
    assert result.answer == (
        "Refunds are available within 30 days [1]."
    )
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

    service = GenerationService(
        client=FakeClient(response=response)
    )

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
        text="Answer [1]. "
        "Unknown source [99]. "
        "Invalid source [0]."
    )

    service = GenerationService(
        client=FakeClient(response=response)
    )

    result = await service.generate(
        query="Question?",
        passages=passages,
    )

    assert [citation.ref for citation in result.citations] == [
        ("doc-1", 3)
    ]
    assert result.invalid_citation_markers == 2


@pytest.mark.asyncio
async def test_generate_empty_context_short_circuits():
    client = FakeClient(
        response=make_response()
    )

    service = GenerationService(
        client=client
    )

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

    service = GenerationService(
        client=client
    )

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

    response.prompt_feedback = SimpleNamespace(
        block_reason="SAFETY"
    )

    client = FakeClient(response=response)

    service = GenerationService(
        client=client
    )

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

    service = GenerationService(
        client=FakeClient(response=response)
    )

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

    service = GenerationService(
        client=client
    )

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

    service = GenerationService(
        client=client
    )

    with pytest.raises(GenerationProviderError):
        await service.generate(
            query="Question?",
            passages=passages,
        )


@pytest.mark.asyncio
async def test_generate_raises_timeout_error(passages):
    client = FakeClient(
        error=TimeoutError("timed out")
    )

    service = GenerationService(
        client=client
    )

    with pytest.raises(GenerationTimeoutError):
        await service.generate(
            query="Question?",
            passages=passages,
        )


@pytest.mark.asyncio
async def test_generate_raises_provider_error_for_connection_failure(
    passages,
):
    client = FakeClient(
        error=ConnectionError("connection failed")
    )

    service = GenerationService(
        client=client
    )

    with pytest.raises(GenerationProviderError):
        await service.generate(
            query="Question?",
            passages=passages,
        )


@pytest.mark.asyncio
async def test_generate_rejects_empty_query(passages):
    service = GenerationService(
        client=FakeClient(response=make_response())
    )

    with pytest.raises(GenerationInputError):
        await service.generate(
            query="   ",
            passages=passages,
        )


@pytest.mark.asyncio
async def test_generate_rejects_empty_passage_text():
    service = GenerationService(
        client=FakeClient(response=make_response())
    )

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
        GenerationConfig(
            temperature=-0.1
        )

    with pytest.raises(ValueError):
        GenerationConfig(
            temperature=2.1
        )


def test_generation_config_rejects_empty_model():
    with pytest.raises(ValueError):
        GenerationConfig(
            model="   "
        )
        
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