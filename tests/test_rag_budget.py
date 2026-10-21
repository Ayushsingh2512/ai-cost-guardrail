from collections.abc import Sequence
from decimal import Decimal
from unittest.mock import patch
from uuid import uuid4

from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.api.v1.rag import get_rag_service
from app.core.model_registry import DEFAULT_GENERATION_MODEL
from app.main import app
from app.rag.embeddings import (
    EmbeddingProviderError,
    EmbedResult,
)
from app.rag.generation import (
    Citation,
    ContextPassage,
    GenerationProviderError,
    GenerationResult,
    GenerationTimeoutError,
    ProviderMeta,
)
from app.rag.pipeline import RAGService
from app.services.database import get_db
from app.services.models import Tenant, UsageRecord, User
from app.services.usage import usage_service

client = TestClient(app, raise_server_exceptions=False)


# ---------------------------------------------------------------------------
# Test Doubles (Stubs)
# ---------------------------------------------------------------------------


class StubEmbedder:
    def __init__(
        self,
        vector: list[float] | None = None,
        tokens: int = 10,
    ) -> None:
        self.vector = vector or [0.1] * 768
        self.tokens = tokens
        self.calls: list[str] = []
        self.count_tokens_calls: list[str] = []

    @property
    def fingerprint(self) -> str:
        return "stub/embed/768"

    def embed_query(self, query: str) -> list[float]:
        self.calls.append(query)
        return self.vector

    def embed_query_with_usage(
        self,
        query: str,
        *,
        precomputed_tokens: int | None = None,
    ) -> EmbedResult:
        self.calls.append(query)
        tokens = (
            precomputed_tokens
            if precomputed_tokens is not None
            else self.count_tokens(query)
        )
        return EmbedResult(vector=self.vector, tokens=tokens)

    def count_tokens(self, text: str) -> int:
        self.count_tokens_calls.append(text)
        return self.tokens


class StubRetriever:
    def __init__(
        self,
        passages: list[ContextPassage] | None = None,
    ) -> None:
        self.passages = (
            passages
            if passages is not None
            else [
                ContextPassage(
                    ref=("policy.pdf", 1),
                    text="Refunds are supported within 30 calendar days.",
                    source="policy.pdf",
                    page=1,
                )
            ]
        )
        self.calls: list[dict] = []

    def search(self, **kwargs) -> list[ContextPassage]:
        self.calls.append(kwargs)
        return self.passages


class StubGenerator:
    def __init__(
        self,
        result: GenerationResult | None = None,
        prompt_tokens: int = 30,
        completion_tokens: int = 20,
        thinking_tokens: int = 10,
        model: str = DEFAULT_GENERATION_MODEL,
        max_output_tokens: int = 2000,
    ) -> None:
        self.result = result
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.thinking_tokens = thinking_tokens
        self.config = type(
            "Config",
            (),
            {
                "model": model,
                "max_output_tokens": max_output_tokens,
            },
        )()
        self.generate_called = False
        self.calls: list[dict] = []

    def build_prompt(
        self,
        query: str,
        passages: Sequence[ContextPassage],
    ) -> str:
        return f"PROMPT: {query}"

    async def count_prompt_tokens(self, prompt: str) -> int:
        return self.prompt_tokens

    async def generate(
        self,
        query: str,
        passages,
        *,
        prompt: str | None = None,
    ) -> GenerationResult:
        self.generate_called = True
        self.calls.append(
            {"query": query, "passages": list(passages), "prompt": prompt}
        )
        if self.result is not None:
            return self.result

        passages_sent = [p.ref for p in passages]
        citations = [
            Citation(ref=p.ref, source=p.source, page=p.page) for p in passages
        ]
        total_tokens = (
            self.prompt_tokens + self.thinking_tokens + self.completion_tokens
        )
        return GenerationResult(
            status="answered",
            answer="Grounded answer based on evidence.",
            citations=citations,
            passages_sent=passages_sent,
            finish_reason="stop",
            invalid_citation_markers=0,
            prompt_version="test-v1",
            provider=ProviderMeta(
                provider="gemini",
                model=self.config.model,
                prompt_tokens=self.prompt_tokens,
                completion_tokens=self.completion_tokens,
                latency_ms=75,
                thinking_tokens=self.thinking_tokens,
                total_tokens=total_tokens,
            ),
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def create_tenant_and_user(
    db: Session,
    *,
    monthly_budget: Decimal = Decimal("50.000000"),
    current_spend: Decimal = Decimal("0.000000"),
) -> tuple[Tenant, User]:
    tenant = Tenant(
        name=f"RAG Budget Test Tenant {uuid4()}",
        monthly_budget=monthly_budget,
        current_spend=current_spend,
    )
    db.add(tenant)
    db.commit()
    db.refresh(tenant)

    user = User(
        tenant_id=tenant.id,
        email=f"rag-budget-{uuid4()}@example.com",
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    return tenant, user


def call_rag_query(
    *,
    tenant_id: int,
    user_id: int,
    query: str = "What is the refund policy?",
    top_k: int = 5,
):
    with patch(
        "app.api.dependencies.verify_access_token",
        return_value={
            "tenant_id": str(tenant_id),
            "user_id": str(user_id),
        },
    ):
        return client.post(
            "/api/v1/rag/query",
            headers={"Authorization": "Bearer test-token"},
            json={"query": query, "top_k": top_k},
        )


def get_usage_record(
    db: Session,
    tenant_id: int,
    operation: str,
    request_id: str | None = None,
) -> UsageRecord | None:
    query = db.query(UsageRecord).filter(
        UsageRecord.tenant_id == tenant_id,
        UsageRecord.operation == operation,
    )
    if request_id is not None:
        query = query.filter(UsageRecord.request_id == request_id)
    return query.first()


# ---------------------------------------------------------------------------
# 11 Budget Accounting Tests
# ---------------------------------------------------------------------------


def test_1_happy_path(db: Session):
    """
    1. Happy path: embed + generate records both 'completed';
    current_spend == embed actual + generate actual (exact Decimal equality);
    response embedding_tokens is an int.
    """
    tenant, user = create_tenant_and_user(db)
    service = RAGService(
        embedder=StubEmbedder(tokens=10),
        retriever=StubRetriever(),
        generator=StubGenerator(
            prompt_tokens=30,
            completion_tokens=20,
            thinking_tokens=10,
        ),
    )

    app.dependency_overrides[get_db] = lambda: db
    app.dependency_overrides[get_rag_service] = lambda: service

    try:
        response = call_rag_query(tenant_id=tenant.id, user_id=user.id)
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    data = response.json()
    assert isinstance(data["embedding_tokens"], int)
    assert data["embedding_tokens"] == 10
    assert data["status"] == "answered"

    req_id = data["request_id"]
    db.refresh(tenant)

    embed_record = get_usage_record(db, tenant.id, "embed", req_id)
    generate_record = get_usage_record(db, tenant.id, "generate", req_id)

    assert embed_record is not None
    assert embed_record.status == "completed"
    assert embed_record.input_tokens == 10
    assert embed_record.output_tokens == 0
    assert embed_record.thinking_tokens == 0
    assert embed_record.total_tokens == 10
    assert embed_record.actual_cost > Decimal(0)
    assert embed_record.reserved_cost > Decimal(0)

    assert generate_record is not None
    assert generate_record.status == "completed"
    assert generate_record.input_tokens == 30
    assert generate_record.output_tokens == 20
    assert generate_record.thinking_tokens == 10
    assert generate_record.total_tokens == 60
    assert generate_record.actual_cost > Decimal(0)
    assert generate_record.reserved_cost > Decimal(0)

    # Exact Decimal equality
    assert (
        tenant.current_spend == embed_record.actual_cost + generate_record.actual_cost
    )


def test_2_no_passages_skips_generation_reservation(db: Session):
    """
    2. No passages: embed record 'completed', generator.generate() IS still called and
    returns status 'no_context', but NO generate UsageRecord is created (no reservation
    was made for it) and current_spend reflects only the embed cost.
    """
    tenant, user = create_tenant_and_user(db)
    generator = StubGenerator(
        result=GenerationResult(
            status="no_context",
            answer=None,
            citations=[],
            passages_sent=[],
            finish_reason="other",
            invalid_citation_markers=0,
            prompt_version="test-v1",
            provider=None,
        )
    )
    service = RAGService(
        embedder=StubEmbedder(tokens=10),
        retriever=StubRetriever(passages=[]),
        generator=generator,
    )

    app.dependency_overrides[get_db] = lambda: db
    app.dependency_overrides[get_rag_service] = lambda: service

    try:
        response = call_rag_query(tenant_id=tenant.id, user_id=user.id)
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "no_context"
    assert generator.generate_called, (
        "generator.generate() must be called on empty context"
    )

    req_id = data["request_id"]
    db.refresh(tenant)

    embed_record = get_usage_record(db, tenant.id, "embed", req_id)
    generate_record = get_usage_record(db, tenant.id, "generate", req_id)

    assert embed_record is not None
    assert embed_record.status == "completed"
    assert embed_record.input_tokens == 10
    assert embed_record.actual_cost > Decimal(0)

    # No reservation was made for generation
    assert generate_record is None
    assert tenant.current_spend == embed_record.actual_cost


def test_3_embedding_fails_releases_spend(db: Session):
    """
    3. Embedding fails: embed record 'failed', spend back to starting value.
    """
    tenant, user = create_tenant_and_user(db)

    class FailingEmbedder(StubEmbedder):
        def embed_query_with_usage(
            self,
            query: str,
            *,
            precomputed_tokens: int | None = None,
        ) -> EmbedResult:
            raise EmbeddingProviderError("Embedding provider crashed")

    service = RAGService(
        embedder=FailingEmbedder(),
        retriever=StubRetriever(),
        generator=StubGenerator(),
    )

    app.dependency_overrides[get_db] = lambda: db
    app.dependency_overrides[get_rag_service] = lambda: service

    try:
        response = call_rag_query(tenant_id=tenant.id, user_id=user.id)
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 502
    db.refresh(tenant)

    embed_record = get_usage_record(db, tenant.id, "embed")
    generate_record = get_usage_record(db, tenant.id, "generate")

    assert embed_record is not None
    assert embed_record.status == "failed"
    assert embed_record.actual_cost == Decimal("0.000000")
    assert embed_record.reserved_cost > Decimal(0)

    assert generate_record is None
    assert tenant.current_spend == Decimal("0.000000")


def test_4_generation_fails_provider_error(db: Session):
    """
    4. Generation fails (provider error): embed 'completed', generate 'failed',
    spend == embed actual only.
    """
    tenant, user = create_tenant_and_user(db)

    class FailingGenerator(StubGenerator):
        async def generate(
            self,
            query: str,
            passages,
            *,
            prompt: str | None = None,
        ) -> GenerationResult:
            raise GenerationProviderError("LLM API unavailable")

    service = RAGService(
        embedder=StubEmbedder(tokens=10),
        retriever=StubRetriever(),
        generator=FailingGenerator(),
    )

    app.dependency_overrides[get_db] = lambda: db
    app.dependency_overrides[get_rag_service] = lambda: service

    try:
        response = call_rag_query(tenant_id=tenant.id, user_id=user.id)
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 502
    db.refresh(tenant)

    embed_record = get_usage_record(db, tenant.id, "embed")
    generate_record = get_usage_record(db, tenant.id, "generate")

    assert embed_record is not None
    assert embed_record.status == "completed"
    assert embed_record.actual_cost > Decimal(0)

    assert generate_record is not None
    assert generate_record.status == "failed"
    assert generate_record.actual_cost == Decimal("0.000000")

    assert tenant.current_spend == embed_record.actual_cost


def test_5_generation_times_out_returns_504(db: Session):
    """
    5. Generation times out -> 504 and the same ledger result as (4).
    """
    tenant, user = create_tenant_and_user(db)

    class TimeoutGenerator(StubGenerator):
        async def generate(
            self,
            query: str,
            passages,
            *,
            prompt: str | None = None,
        ) -> GenerationResult:
            raise GenerationTimeoutError("Generation request timed out after 30s")

    service = RAGService(
        embedder=StubEmbedder(tokens=10),
        retriever=StubRetriever(),
        generator=TimeoutGenerator(),
    )

    app.dependency_overrides[get_db] = lambda: db
    app.dependency_overrides[get_rag_service] = lambda: service

    try:
        response = call_rag_query(tenant_id=tenant.id, user_id=user.id)
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 504
    db.refresh(tenant)

    embed_record = get_usage_record(db, tenant.id, "embed")
    generate_record = get_usage_record(db, tenant.id, "generate")

    assert embed_record is not None
    assert embed_record.status == "completed"
    assert embed_record.actual_cost > Decimal(0)

    assert generate_record is not None
    assert generate_record.status == "failed"
    assert generate_record.actual_cost == Decimal("0.000000")

    assert tenant.current_spend == embed_record.actual_cost


def test_6_safety_blocked_with_usage_metadata(db: Session):
    """
    6. Safety-blocked with usage metadata: generate record 'completed' with real tokens.
    """
    tenant, user = create_tenant_and_user(db)
    blocked_provider_meta = ProviderMeta(
        provider="gemini",
        model=DEFAULT_GENERATION_MODEL,
        prompt_tokens=30,
        completion_tokens=0,
        latency_ms=60,
        thinking_tokens=0,
        total_tokens=30,
    )
    generator = StubGenerator(
        prompt_tokens=30,
        result=GenerationResult(
            status="blocked",
            answer=None,
            citations=[],
            passages_sent=[("policy.pdf", 1)],
            finish_reason="safety",
            invalid_citation_markers=0,
            prompt_version="test-v1",
            provider=blocked_provider_meta,
        ),
    )
    service = RAGService(
        embedder=StubEmbedder(tokens=10),
        retriever=StubRetriever(),
        generator=generator,
    )

    app.dependency_overrides[get_db] = lambda: db
    app.dependency_overrides[get_rag_service] = lambda: service

    try:
        response = call_rag_query(tenant_id=tenant.id, user_id=user.id)
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json()["status"] == "blocked"

    db.refresh(tenant)
    embed_record = get_usage_record(db, tenant.id, "embed")
    generate_record = get_usage_record(db, tenant.id, "generate")

    assert embed_record is not None
    assert embed_record.status == "completed"

    assert generate_record is not None
    assert generate_record.status == "completed"
    assert generate_record.input_tokens == 30
    assert generate_record.output_tokens == 0
    assert generate_record.thinking_tokens == 0
    assert generate_record.total_tokens == 30
    assert generate_record.actual_cost > Decimal(0)

    assert (
        tenant.current_spend == embed_record.actual_cost + generate_record.actual_cost
    )


def test_7_budget_too_small_for_embed_reservation_returns_400(db: Session):
    """
    7. Budget too small for the embed reservation -> 400, no provider call, no records
    left in 'reserved'.
    """
    # Set current spend equal to budget so any reservation will exceed it
    tenant, user = create_tenant_and_user(
        db,
        monthly_budget=Decimal("0.000001"),
        current_spend=Decimal("0.000001"),
    )
    embedder = StubEmbedder(tokens=10)
    generator = StubGenerator()
    service = RAGService(
        embedder=embedder,
        retriever=StubRetriever(),
        generator=generator,
    )

    app.dependency_overrides[get_db] = lambda: db
    app.dependency_overrides[get_rag_service] = lambda: service

    try:
        response = call_rag_query(tenant_id=tenant.id, user_id=user.id)
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 400
    assert "exceed budget" in response.json()["detail"].lower()

    # Preflight token count was performed, but neither provider operation was called
    assert len(embedder.count_tokens_calls) == 1
    assert len(embedder.calls) == 0
    assert len(generator.calls) == 0

    tenant = db.query(Tenant).filter(Tenant.id == tenant.id).first()
    assert tenant is not None
    assert tenant.current_spend == Decimal("0.000001")

    # No records stuck in 'reserved' and no records created
    reserved_records = (
        db.query(UsageRecord)
        .filter(UsageRecord.tenant_id == tenant.id, UsageRecord.status == "reserved")
        .all()
    )
    assert len(reserved_records) == 0
    all_records = db.query(UsageRecord).filter(UsageRecord.tenant_id == tenant.id).all()
    assert len(all_records) == 0


def test_8_budget_enough_for_embed_not_generation_returns_400(db: Session):
    """
    8. Budget enough for embed but not for generation -> 400, embed settled correctly,
    no generate record in 'reserved'.
    """
    # 10 embed tokens costs $0.000002.
    # Generation reservation for 30 prompt tokens + 2000 output is > $0.006.
    # Setting budget to $0.001000 allows embed reservation but rejects generation.
    tenant, user = create_tenant_and_user(
        db,
        monthly_budget=Decimal("0.001000"),
        current_spend=Decimal("0.000000"),
    )
    service = RAGService(
        embedder=StubEmbedder(tokens=10),
        retriever=StubRetriever(),
        generator=StubGenerator(prompt_tokens=30),
    )

    app.dependency_overrides[get_db] = lambda: db
    app.dependency_overrides[get_rag_service] = lambda: service

    try:
        response = call_rag_query(tenant_id=tenant.id, user_id=user.id)
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 400
    assert "exceed budget" in response.json()["detail"].lower()

    db.refresh(tenant)
    embed_record = get_usage_record(db, tenant.id, "embed")
    generate_record = get_usage_record(db, tenant.id, "generate")

    assert embed_record is not None
    assert embed_record.status == "completed"
    assert embed_record.actual_cost == Decimal("0.000002")

    # Generation reservation was rejected; no generate record exists
    assert generate_record is None
    reserved_records = (
        db.query(UsageRecord)
        .filter(UsageRecord.tenant_id == tenant.id, UsageRecord.status == "reserved")
        .all()
    )
    assert len(reserved_records) == 0

    assert tenant.current_spend == embed_record.actual_cost


def test_9_actual_generation_cost_exceeds_reservation_shortfall_charged(db: Session):
    """
    9. Actual generation cost > reservation (thinking tokens overflow): spend is charged
    the shortfall, not left under-counted.
    """
    tenant, user = create_tenant_and_user(db)
    # 30 prompt tokens, 3000 thinking tokens + 500 completion tokens = 3500 billable output tokens
    # Output cost for 3500 tokens at $3.00/1M is $0.010500, which exceeds the 2000-token
    # output reservation of $0.006000.
    generator = StubGenerator(
        prompt_tokens=30,
        completion_tokens=500,
        thinking_tokens=3000,
    )
    service = RAGService(
        embedder=StubEmbedder(tokens=10),
        retriever=StubRetriever(),
        generator=generator,
    )

    app.dependency_overrides[get_db] = lambda: db
    app.dependency_overrides[get_rag_service] = lambda: service

    try:
        response = call_rag_query(tenant_id=tenant.id, user_id=user.id)
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    db.refresh(tenant)

    embed_record = get_usage_record(db, tenant.id, "embed")
    generate_record = get_usage_record(db, tenant.id, "generate")

    assert embed_record is not None
    assert embed_record.status == "completed"

    assert generate_record is not None
    assert generate_record.status == "completed"
    assert generate_record.thinking_tokens == 3000
    assert generate_record.output_tokens == 500
    # Actual cost strictly exceeds reserved cost
    assert generate_record.actual_cost > generate_record.reserved_cost

    # Shortfall is charged, matching the exact sum
    expected_total_spend = embed_record.actual_cost + generate_record.actual_cost
    assert tenant.current_spend == expected_total_spend


def test_10_settlement_db_failure_releases_reservation_returns_500(db: Session):
    """
    10. Settlement DB failure after successful generation: reservation is released,
    API returns 500, no record stuck in 'reserved'.
    """
    tenant, user = create_tenant_and_user(db)
    service = RAGService(
        embedder=StubEmbedder(tokens=10),
        retriever=StubRetriever(),
        generator=StubGenerator(prompt_tokens=30),
    )

    real_settle_success = usage_service.settle_success
    call_count = 0

    def failing_settle_success(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        # 1st call is embed settlement (succeeds)
        # 2nd call is generation settlement (fails with DB error)
        if call_count == 2:
            raise RuntimeError(
                "Simulated DB connection failure during generation settlement"
            )
        return real_settle_success(*args, **kwargs)

    app.dependency_overrides[get_db] = lambda: db
    app.dependency_overrides[get_rag_service] = lambda: service

    try:
        with patch.object(
            usage_service, "settle_success", side_effect=failing_settle_success
        ):
            response = call_rag_query(tenant_id=tenant.id, user_id=user.id)
    finally:
        app.dependency_overrides.clear()

    # Route fails with 500 on unhandled settlement error
    assert response.status_code == 500
    db.refresh(tenant)

    embed_record = get_usage_record(db, tenant.id, "embed")
    generate_record = get_usage_record(db, tenant.id, "generate")

    assert embed_record is not None
    assert embed_record.status == "completed"
    assert embed_record.actual_cost > Decimal(0)

    # Generate record was released to 'failed' and refunded
    assert generate_record is not None
    assert generate_record.status == "failed"
    assert generate_record.actual_cost == Decimal("0.000000")

    # No record stuck in 'reserved'
    reserved_records = (
        db.query(UsageRecord)
        .filter(UsageRecord.tenant_id == tenant.id, UsageRecord.status == "reserved")
        .all()
    )
    assert len(reserved_records) == 0

    # Tenant spend is only embed cost
    assert tenant.current_spend == embed_record.actual_cost


def test_11_two_sequential_requests_accumulate_spend(db: Session):
    """
    11. Two sequential requests: spend accumulates correctly across requests.
    """
    tenant, user = create_tenant_and_user(db)
    service = RAGService(
        embedder=StubEmbedder(tokens=10),
        retriever=StubRetriever(),
        generator=StubGenerator(
            prompt_tokens=30,
            completion_tokens=20,
            thinking_tokens=10,
        ),
    )

    app.dependency_overrides[get_db] = lambda: db
    app.dependency_overrides[get_rag_service] = lambda: service

    try:
        # Request 1
        resp1 = call_rag_query(
            tenant_id=tenant.id, user_id=user.id, query="First question"
        )
        assert resp1.status_code == 200
        req1_id = resp1.json()["request_id"]

        db.refresh(tenant)
        embed1 = get_usage_record(db, tenant.id, "embed", req1_id)
        gen1 = get_usage_record(db, tenant.id, "generate", req1_id)
        assert embed1 is not None and gen1 is not None
        cost1 = embed1.actual_cost + gen1.actual_cost
        assert tenant.current_spend == cost1

        # Request 2
        resp2 = call_rag_query(
            tenant_id=tenant.id, user_id=user.id, query="Second question"
        )
        assert resp2.status_code == 200
        req2_id = resp2.json()["request_id"]

        db.refresh(tenant)
        embed2 = get_usage_record(db, tenant.id, "embed", req2_id)
        gen2 = get_usage_record(db, tenant.id, "generate", req2_id)
        assert embed2 is not None and gen2 is not None
        cost2 = embed2.actual_cost + gen2.actual_cost

        # Spend accumulates accurately
        assert tenant.current_spend == cost1 + cost2

        # 4 total completed UsageRecords
        all_records = (
            db.query(UsageRecord)
            .filter(UsageRecord.tenant_id == tenant.id)
            .order_by(UsageRecord.id)
            .all()
        )
        assert len(all_records) == 4
        assert all(r.status == "completed" for r in all_records)
        assert sum(r.actual_cost for r in all_records) == tenant.current_spend

    finally:
        app.dependency_overrides.clear()


def test_12_single_count_tokens_call_for_rag_request(db: Session):
    """
    12. Exactly one count_tokens call for a successful RAG request.
    Verifies that preflight token count is reused for reservation and settlement,
    with no second count_tokens call.
    """
    tenant, user = create_tenant_and_user(db)
    client = StubEmbedder(tokens=10)
    service = RAGService(
        embedder=client,
        retriever=StubRetriever(),
        generator=StubGenerator(
            prompt_tokens=30,
            completion_tokens=20,
            thinking_tokens=10,
        ),
    )

    app.dependency_overrides[get_db] = lambda: db
    app.dependency_overrides[get_rag_service] = lambda: service

    try:
        response = call_rag_query(tenant_id=tenant.id, user_id=user.id)
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert len(client.count_tokens_calls) == 1


def test_13_generation_succeeds_with_none_provider_metadata_preserves_reservation_returns_502(
    db: Session,
):
    """
    13. Generation succeeds but provider returned None for usage metadata.
    Fails closed with 502, does NOT create a completed zero-cost record,
    and preserves the generation reservation to protect tenant budget.
    """
    tenant, user = create_tenant_and_user(db)
    embedder = StubEmbedder(tokens=10)
    result_with_none_meta = GenerationResult(
        status="answered",
        answer="Grounded answer.",
        citations=[],
        passages_sent=[("policy.pdf", 1)],
        finish_reason="stop",
        invalid_citation_markers=0,
        prompt_version="test-v1",
        provider=None,
    )
    generator = StubGenerator(result=result_with_none_meta)
    service = RAGService(
        embedder=embedder,
        retriever=StubRetriever(),
        generator=generator,
    )

    app.dependency_overrides[get_db] = lambda: db
    app.dependency_overrides[get_rag_service] = lambda: service

    try:
        response = call_rag_query(tenant_id=tenant.id, user_id=user.id)
    finally:
        app.dependency_overrides.clear()

    # Provider call occurred
    assert generator.generate_called is True

    # Failed closed with 502 Bad Gateway
    assert response.status_code == 502
    assert "no usage metadata" in response.json()["detail"].lower()

    db.refresh(tenant)
    embed_record = get_usage_record(db, tenant.id, "embed")
    generate_record = get_usage_record(db, tenant.id, "generate")

    # Embed operation settled normally
    assert embed_record is not None
    assert embed_record.status == "completed"
    assert embed_record.actual_cost > Decimal(0)

    # Generation operation is PRESERVED in 'reserved' status (not 'completed' with zero cost, not 'failed' refunded)
    assert generate_record is not None
    assert generate_record.status == "reserved"
    assert generate_record.reserved_cost > Decimal(0)

    # Tenant spend preserves the reservation, preventing budget overspend
    expected_spend = embed_record.actual_cost + generate_record.reserved_cost
    assert tenant.current_spend == expected_spend


def test_14_generation_succeeds_with_missing_prompt_tokens_preserves_reservation_returns_502(
    db: Session,
):
    """
    14. Generation provider metadata exists but prompt_tokens is None.
    Must not silently become zero or record a successfully settled, zero-cost generation.
    Fails closed with 502 and preserves the generation reservation.
    """
    tenant, user = create_tenant_and_user(db)
    embedder = StubEmbedder(tokens=10)
    meta = ProviderMeta(
        provider="gemini",
        model=DEFAULT_GENERATION_MODEL,
        prompt_tokens=None,
        completion_tokens=25,
        latency_ms=80,
    )
    result = GenerationResult(
        status="answered",
        answer="Grounded answer.",
        citations=[],
        passages_sent=[("policy.pdf", 1)],
        finish_reason="stop",
        invalid_citation_markers=0,
        prompt_version="test-v1",
        provider=meta,
    )
    generator = StubGenerator(result=result)
    service = RAGService(
        embedder=embedder,
        retriever=StubRetriever(),
        generator=generator,
    )

    app.dependency_overrides[get_db] = lambda: db
    app.dependency_overrides[get_rag_service] = lambda: service

    try:
        response = call_rag_query(tenant_id=tenant.id, user_id=user.id)
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 502
    assert "prompt_tokens" in response.json()["detail"].lower()

    db.refresh(tenant)
    generate_record = get_usage_record(db, tenant.id, "generate")
    embed_record = get_usage_record(db, tenant.id, "embed")

    assert generate_record is not None
    assert generate_record.status == "reserved"
    assert generate_record.actual_cost == Decimal("0.000000")
    # Budget remains protected by the reservation
    assert (
        tenant.current_spend == embed_record.actual_cost + generate_record.reserved_cost
    )


def test_15_generation_succeeds_with_missing_completion_tokens_preserves_reservation_returns_502(
    db: Session,
):
    """
    15. Generation provider metadata exists but completion_tokens is None.
    Fails closed with 502 and preserves the generation reservation.
    """
    tenant, user = create_tenant_and_user(db)
    embedder = StubEmbedder(tokens=10)
    meta = ProviderMeta(
        provider="gemini",
        model=DEFAULT_GENERATION_MODEL,
        prompt_tokens=35,
        completion_tokens=None,
        latency_ms=80,
    )
    result = GenerationResult(
        status="answered",
        answer="Grounded answer.",
        citations=[],
        passages_sent=[("policy.pdf", 1)],
        finish_reason="stop",
        invalid_citation_markers=0,
        prompt_version="test-v1",
        provider=meta,
    )
    generator = StubGenerator(result=result)
    service = RAGService(
        embedder=embedder,
        retriever=StubRetriever(),
        generator=generator,
    )

    app.dependency_overrides[get_db] = lambda: db
    app.dependency_overrides[get_rag_service] = lambda: service

    try:
        response = call_rag_query(tenant_id=tenant.id, user_id=user.id)
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 502
    assert "completion_tokens" in response.json()["detail"].lower()

    db.refresh(tenant)
    generate_record = get_usage_record(db, tenant.id, "generate")
    embed_record = get_usage_record(db, tenant.id, "embed")

    assert generate_record is not None
    assert generate_record.status == "reserved"
    assert generate_record.actual_cost == Decimal("0.000000")
    assert (
        tenant.current_spend == embed_record.actual_cost + generate_record.reserved_cost
    )


def test_16_generation_succeeds_with_invalid_boolean_tokens_preserves_reservation_returns_502(
    db: Session,
):
    """
    16. Generation provider metadata contains Boolean tokens (e.g. prompt_tokens=True).
    Rejects Boolean values, fails closed with 502, and preserves the generation reservation.
    """
    tenant, user = create_tenant_and_user(db)
    embedder = StubEmbedder(tokens=10)
    meta = ProviderMeta(
        provider="gemini",
        model=DEFAULT_GENERATION_MODEL,
        prompt_tokens=True,  # type: ignore[arg-type]
        completion_tokens=20,
        latency_ms=80,
    )
    result = GenerationResult(
        status="answered",
        answer="Grounded answer.",
        citations=[],
        passages_sent=[("policy.pdf", 1)],
        finish_reason="stop",
        invalid_citation_markers=0,
        prompt_version="test-v1",
        provider=meta,
    )
    generator = StubGenerator(result=result)
    service = RAGService(
        embedder=embedder,
        retriever=StubRetriever(),
        generator=generator,
    )

    app.dependency_overrides[get_db] = lambda: db
    app.dependency_overrides[get_rag_service] = lambda: service

    try:
        response = call_rag_query(tenant_id=tenant.id, user_id=user.id)
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 502
    assert "prompt_tokens" in response.json()["detail"].lower()

    db.refresh(tenant)
    generate_record = get_usage_record(db, tenant.id, "generate")
    embed_record = get_usage_record(db, tenant.id, "embed")

    assert generate_record is not None
    assert generate_record.status == "reserved"
    assert (
        tenant.current_spend == embed_record.actual_cost + generate_record.reserved_cost
    )
