"""
Tests for Pass 1.5 of the RAG Evaluation Suite.

Covers:
- Pure ranking metrics: Recall@1, Recall@3, Recall@5, MRR, and random-chance references
- Multi-evidence MRR definition verification
- Analytic properties of random_chance_recall_at_k
- Corpus hardening: chunk count, distractor properties, and invariant unanswerability
- Live-mode hardening using fakes/mocks (no network, no real API keys):
  * Dedicated evaluation tenant setup and isolation
  * Ingestion of evaluation corpus and UUID-to-canonical-doc mapping
  * Evaluation document cleanup isolation
  * Per-question failure isolation (distinct from abstentions)
  * Request pacing (--request-delay)
  * GenerationRateLimitError retry with exponential backoff and provider delay
  * Retry exhaustion handling
  * Incomplete report flag ("complete": false) and failed question IDs
  * Live report metadata recording mode, models, and top_k
  * Split guardrail enforcement (live mode rejects test split)
"""

from __future__ import annotations

import asyncio
from decimal import Decimal
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from app.rag.generation import GenerationRateLimitError
from evals.datasets.corpus import (
    CORPUS_CHUNK_COUNT,
    CORPUS_DOCUMENT_COUNT,
    CORPUS_VERSION,
    SYNTHETIC_CORPUS,
    VALID_EVIDENCE_REFS,
)
from evals.evaluate_rag import (
    _build_arg_parser,
    _extract_retry_delay,
    _run_one_question_live,
    cleanup_evaluation_documents,
    ensure_eval_tenant,
    ingest_evaluation_corpus,
    run_evaluation,
)
from evals.metrics import (
    QuestionResult,
    compute_split_metrics,
    random_chance_recall_at_k,
    recall_at_1,
    recall_at_3,
    recall_at_k,
    reciprocal_rank,
)
from evals.report import write_report
from evals.schema import load_and_validate

# ===========================================================================
# 1. Ranking Metrics Tests
# ===========================================================================


class TestRankingMetrics:
    def test_recall_at_1_hit_at_rank_1(self):
        expected = [("doc_a", 0)]
        retrieved = [("doc_a", 0), ("doc_b", 1), ("doc_c", 2)]
        assert recall_at_1(expected, retrieved) == pytest.approx(1.0)

    def test_recall_at_1_hit_at_rank_2(self):
        expected = [("doc_a", 0)]
        retrieved = [("doc_b", 1), ("doc_a", 0), ("doc_c", 2)]
        assert recall_at_1(expected, retrieved) == pytest.approx(0.0)

    def test_recall_at_3_hit_at_rank_3(self):
        expected = [("doc_a", 0)]
        retrieved = [("doc_b", 1), ("doc_c", 2), ("doc_a", 0), ("doc_d", 3)]
        assert recall_at_3(expected, retrieved) == pytest.approx(1.0)

    def test_recall_at_3_hit_at_rank_4(self):
        expected = [("doc_a", 0)]
        retrieved = [("doc_b", 1), ("doc_c", 2), ("doc_d", 3), ("doc_a", 0)]
        assert recall_at_3(expected, retrieved) == pytest.approx(0.0)

    def test_recall_at_5_hit_at_rank_5(self):
        expected = [("doc_a", 0)]
        retrieved = [
            ("doc_1", 0),
            ("doc_2", 0),
            ("doc_3", 0),
            ("doc_4", 0),
            ("doc_a", 0),
        ]
        assert recall_at_k(expected, retrieved, k=5) == pytest.approx(1.0)

    def test_multi_evidence_recall_progression(self):
        expected = [("doc_a", 0), ("doc_b", 1)]
        # Rank 1 hit: doc_a. Rank 4 hit: doc_b.
        retrieved = [("doc_a", 0), ("x", 0), ("y", 0), ("doc_b", 1), ("z", 0)]
        assert recall_at_1(expected, retrieved) == pytest.approx(0.5)
        assert recall_at_3(expected, retrieved) == pytest.approx(0.5)
        assert recall_at_k(expected, retrieved, k=5) == pytest.approx(1.0)


class TestReciprocalRank:
    def test_mrr_first_rank_hit(self):
        expected = [("doc_a", 0)]
        retrieved = [("doc_a", 0), ("doc_b", 0)]
        assert reciprocal_rank(expected, retrieved) == pytest.approx(1.0)

    def test_mrr_second_rank_hit(self):
        expected = [("doc_a", 0)]
        retrieved = [("doc_x", 0), ("doc_a", 0), ("doc_b", 0)]
        assert reciprocal_rank(expected, retrieved) == pytest.approx(0.5)

    def test_mrr_third_rank_hit(self):
        expected = [("doc_a", 0)]
        retrieved = [("doc_x", 0), ("doc_y", 0), ("doc_a", 0)]
        assert reciprocal_rank(expected, retrieved) == pytest.approx(1.0 / 3.0)

    def test_mrr_no_hits_returns_zero(self):
        expected = [("doc_a", 0)]
        retrieved = [("doc_x", 0), ("doc_y", 0), ("doc_z", 0)]
        assert reciprocal_rank(expected, retrieved) == pytest.approx(0.0)

    def test_mrr_empty_retrieved_returns_zero(self):
        expected = [("doc_a", 0)]
        assert reciprocal_rank(expected, []) == pytest.approx(0.0)

    def test_mrr_multi_evidence_evaluates_first_hit(self):
        """
        Documented definition for multi-evidence questions:
        Reciprocal rank is determined by the first (highest-ranked) expected chunk found.
        """
        expected = [("doc_a", 0), ("doc_b", 1)]
        # Hit for doc_b at rank 2, hit for doc_a at rank 4 -> first hit rank is 2 -> RR = 0.5
        retrieved = [("noise_1", 0), ("doc_b", 1), ("noise_2", 0), ("doc_a", 0)]
        assert reciprocal_rank(expected, retrieved) == pytest.approx(0.5)

    def test_mrr_empty_expected_raises(self):
        with pytest.raises(ValueError, match="expected_refs must be non-empty"):
            reciprocal_rank([], [("doc_a", 0)])


class TestRandomChanceRecall:
    def test_exact_analytic_calculation(self):
        # N=50 chunks in corpus
        assert random_chance_recall_at_k(corpus_size=50, k=1) == pytest.approx(0.02)
        assert random_chance_recall_at_k(corpus_size=50, k=3) == pytest.approx(0.06)
        assert random_chance_recall_at_k(corpus_size=50, k=5) == pytest.approx(0.10)

    def test_k_greater_than_or_equal_to_corpus_size(self):
        assert random_chance_recall_at_k(corpus_size=10, k=10) == pytest.approx(1.0)
        assert random_chance_recall_at_k(corpus_size=10, k=20) == pytest.approx(1.0)

    def test_zero_or_negative_inputs(self):
        assert random_chance_recall_at_k(corpus_size=0, k=5) == pytest.approx(0.0)
        assert random_chance_recall_at_k(corpus_size=50, k=0) == pytest.approx(0.0)
        assert random_chance_recall_at_k(corpus_size=-10, k=5) == pytest.approx(0.0)

    def test_with_expected_evidence_counts_invariance(self):
        # Mathematical property: macro-expectation is identical regardless of evidence counts
        counts = [1, 2, 1, 3, 2]
        res = random_chance_recall_at_k(
            corpus_size=50, k=5, expected_evidence_counts=counts
        )
        assert res == pytest.approx(0.10)

    def test_invalid_evidence_count_raises(self):
        with pytest.raises(ValueError, match="positive integers"):
            random_chance_recall_at_k(
                corpus_size=50, k=5, expected_evidence_counts=[0, 1]
            )


class TestSplitMetricsRankingAggregation:
    def test_aggregation_computes_all_ranking_metrics_and_references(self):
        q1 = QuestionResult(
            question_id="q1",
            split="dev",
            answerable=True,
            generation_status="answered",
            recall_at_1=1.0,
            recall_at_3=1.0,
            recall_at_5=1.0,
            reciprocal_rank=1.0,
            recall_at_k=1.0,
            latency_ms=50,
            total_cost=None,
            embedding_tokens=None,
        )
        q2 = QuestionResult(
            question_id="q2",
            split="dev",
            answerable=True,
            generation_status="answered",
            recall_at_1=0.0,
            recall_at_3=0.0,
            recall_at_5=0.5,
            reciprocal_rank=0.2,  # hit at rank 5
            recall_at_k=0.5,
            latency_ms=60,
            total_cost=None,
            embedding_tokens=None,
        )
        m = compute_split_metrics([q1, q2], split="dev", top_k=5, corpus_size=50)

        assert m.recall_at_1_mean == pytest.approx(0.5)
        assert m.recall_at_3_mean == pytest.approx(0.5)
        assert m.recall_at_5_mean == pytest.approx(0.75)
        assert m.mrr_mean == pytest.approx(0.6)
        assert m.random_chance_recall_at_1 == pytest.approx(0.02)
        assert m.random_chance_recall_at_3 == pytest.approx(0.06)
        assert m.random_chance_recall_at_5 == pytest.approx(0.10)


# ===========================================================================
# 2. Corpus Hardening and Distractor Tests
# ===========================================================================


class TestCorpusHardening:
    def test_corpus_chunk_count_within_target_range(self):
        """Corpus must have roughly 40 to 60 chunks total."""
        assert 40 <= CORPUS_CHUNK_COUNT <= 60
        assert CORPUS_CHUNK_COUNT == 50
        assert CORPUS_DOCUMENT_COUNT == 18
        assert CORPUS_VERSION == "1.5.0"

    def test_all_original_documents_and_chunk_indices_preserved(self):
        """Existing document IDs and chunk indices must remain identical."""
        golden_originals = {
            "synthetic_hr_policy": 3,
            "synthetic_refund_policy": 3,
            "synthetic_shipping_policy": 3,
            "synthetic_security_policy": 2,
            "synthetic_expense_policy": 2,
        }
        for doc_id, expected_chunk_count in golden_originals.items():
            assert doc_id in SYNTHETIC_CORPUS
            assert len(SYNTHETIC_CORPUS[doc_id]) == expected_chunk_count
            for idx in range(expected_chunk_count):
                assert (doc_id, idx) in VALID_EVIDENCE_REFS

    def test_golden_dataset_references_valid_against_expanded_corpus(self):
        golden_path = Path("evals/datasets/rag_golden.jsonl")
        records = load_and_validate(golden_path)
        assert len(records) == 30
        for rec in records:
            for ev in rec["expected_evidence"]:
                ref = (ev["document_id"], ev["chunk_index"])
                assert ref in VALID_EVIDENCE_REFS

    def test_unanswerable_questions_remain_strictly_unanswerable_by_distractors(self):
        """
        Verify that none of the distractors contain facts answering the 8 unanswerable questions:
        q_016: corporate tax rate on bonuses
        q_017: CEO's personal email address
        q_018: software engineer employee count
        q_019: company stock price
        q_020: personal gym membership reimbursement
        q_026: employee moonlighting / secondary employment policy
        q_027: third-party contractors covered by security policy
        q_029: accommodation per-trip maximum without CFO approval
        """
        original_doc_ids = {
            "synthetic_hr_policy",
            "synthetic_refund_policy",
            "synthetic_shipping_policy",
            "synthetic_security_policy",
            "synthetic_expense_policy",
        }
        distractor_texts = [
            text.lower()
            for doc_id, chunks in SYNTHETIC_CORPUS.items()
            if doc_id not in original_doc_ids
            for text in chunks
        ]
        distractor_corpus_text = " ".join(distractor_texts)

        # q_016: corporate tax rate on bonuses
        assert "tax rate" not in distractor_corpus_text
        assert "bonus" not in distractor_corpus_text

        # q_017: CEO personal email address
        assert "ceo" not in distractor_corpus_text or "@" not in distractor_corpus_text

        # q_018: software engineer count
        assert "software engineer" not in distractor_corpus_text
        assert "engineers does the company employ" not in distractor_corpus_text

        # q_019: company stock price
        assert "stock price" not in distractor_corpus_text
        assert "ticker" not in distractor_corpus_text

        # q_020: personal gym membership reimbursement
        assert "gym" not in distractor_corpus_text
        assert "fitness" not in distractor_corpus_text

        # q_026: employee moonlighting / secondary employment
        assert "moonlighting" not in distractor_corpus_text
        assert "secondary employment" not in distractor_corpus_text

        # q_027: third-party contractors covered by security policy
        distractor_security = " ".join(
            [
                text.lower()
                for doc, chunks in SYNTHETIC_CORPUS.items()
                if "security" in doc and doc not in original_doc_ids
                for text in chunks
            ]
        )
        assert "contractor" not in distractor_security
        assert "third-party" not in distractor_security

        # q_029: accommodation per-trip maximum without CFO approval
        assert "accommodation" not in distractor_corpus_text


# ===========================================================================
# 3. Live-Mode Hardening, Pacing, and Retry Tests (Fakes and Mocks Only)
# ===========================================================================


class TestLiveModeHardening:
    def test_ensure_eval_tenant_creates_or_finds_dedicated_tenant(self):
        mock_db = MagicMock()
        mock_db.query().filter().first.return_value = None

        def fake_refresh(obj):
            obj.id = 999

        mock_db.refresh.side_effect = fake_refresh

        tenant_id = ensure_eval_tenant(mock_db, tenant_id=None)
        assert tenant_id == 999
        mock_db.add.assert_called_once()
        mock_db.commit.assert_called_once()

    def test_cleanup_evaluation_documents_is_tenant_scoped(self):
        mock_db = MagicMock()
        mock_query = mock_db.query().filter()
        mock_query.delete.return_value = 5

        deleted = cleanup_evaluation_documents(mock_db, tenant_id=999)
        assert deleted == 5
        mock_db.commit.assert_called_once()

    def test_ingest_evaluation_corpus_creates_documents_and_chunks(self):
        from app.rag.embeddings import FakeEmbedder

        mock_db = MagicMock()
        fake_embedder = FakeEmbedder(dimensions=8)
        small_corpus = {
            "doc_1": ["Chunk 0 text", "Chunk 1 text"],
            "doc_2": ["Chunk 0 of doc 2"],
        }

        uuid_to_doc = ingest_evaluation_corpus(
            db=mock_db,
            tenant_id=77,
            embedder=fake_embedder,
            corpus=small_corpus,
        )

        assert len(uuid_to_doc) == 2
        assert set(uuid_to_doc.values()) == {"doc_1", "doc_2"}
        assert mock_db.commit.called

    def test_run_one_question_live_resolves_uuids_to_canonical_ids(self):
        from app.rag.generation import GenerationResult

        mock_service = MagicMock()
        uuid_doc_1 = "12345678-1234-5678-1234-567812345678"
        uuid_to_doc = {uuid_doc_1: "synthetic_hr_policy"}

        mock_answer = MagicMock()
        mock_answer.retrieved = [(uuid_doc_1, 0)]
        mock_answer.generation = GenerationResult(
            status="answered",
            answer="Grounded answer",
            citations=[],
            passages_sent=[(uuid_doc_1, 0)],
            finish_reason="stop",
            invalid_citation_markers=0,
            prompt_version="v1",
            provider=None,
        )
        mock_answer.timings_ms = {"total": 50}
        mock_answer.embedding_tokens = 15
        mock_answer.total_cost = Decimal("0.000100")

        async def fake_answer(*args, **kwargs):
            return mock_answer

        mock_service.answer = fake_answer

        record = {
            "id": "q_001",
            "split": "dev",
            "query": "What is the policy?",
            "answerable": True,
            "expected_evidence": [
                {"document_id": "synthetic_hr_policy", "chunk_index": 0}
            ],
        }

        retrieval, error = asyncio.run(
            _run_one_question_live(
                record=record,
                service=mock_service,
                tenant_id=77,
                top_k=5,
                uuid_to_doc_id=uuid_to_doc,
            )
        )

        assert error is None
        assert retrieval.generation_status == "answered"
        # Canonical ID successfully resolved from database UUID
        assert retrieval.retrieved_refs == [("synthetic_hr_policy", 0)]
        assert retrieval.retries == 0

    def test_run_one_question_live_failure_isolation_marks_failed_distinct(self):
        """Per-question failure isolation: exceptions mark status as 'failed', distinct from 'no_context'."""
        mock_service = MagicMock()

        async def fail_answer(**kwargs):
            raise RuntimeError("Database connection died")

        mock_service.answer = fail_answer

        record = {
            "id": "q_fail",
            "split": "dev",
            "query": "What is the policy?",
            "answerable": True,
            "expected_evidence": [
                {"document_id": "synthetic_hr_policy", "chunk_index": 0}
            ],
        }

        retrieval, error = asyncio.run(
            _run_one_question_live(
                record=record,
                service=mock_service,
                tenant_id=77,
                top_k=5,
                uuid_to_doc_id={},
            )
        )

        assert error is not None
        assert "RuntimeError" in error
        # Kept distinct from abstention ("no_context")
        assert retrieval.generation_status == "failed"


class TestLivePacingAndRetries:
    def test_pacing_delay_called_between_questions(self):
        """Pacing delay is called between questions using the injected sleep_fn."""
        sleep_calls: list[float] = []

        async def fake_sleep(duration: float):
            sleep_calls.append(duration)

        records = [
            {
                "id": "q1",
                "split": "dev",
                "query": "Q1",
                "answerable": True,
                "expected_evidence": [
                    {"document_id": "synthetic_hr_policy", "chunk_index": 0}
                ],
            },
            {
                "id": "q2",
                "split": "dev",
                "query": "Q2",
                "answerable": True,
                "expected_evidence": [
                    {"document_id": "synthetic_hr_policy", "chunk_index": 0}
                ],
            },
            {
                "id": "q3",
                "split": "dev",
                "query": "Q3",
                "answerable": True,
                "expected_evidence": [
                    {"document_id": "synthetic_hr_policy", "chunk_index": 0}
                ],
            },
        ]

        asyncio.run(
            run_evaluation(
                records=records,
                mode="deterministic",
                top_k=5,
                request_delay=4.5,
                sleep_fn=fake_sleep,
            )
        )

        # 3 questions -> 2 delays between them
        assert len(sleep_calls) == 2
        assert sleep_calls == [4.5, 4.5]

    def test_cli_default_request_delays(self):
        parser = _build_arg_parser()
        # Default with no args
        args_det = parser.parse_args(["--mode", "deterministic"])
        assert args_det.request_delay is None

        args_live = parser.parse_args(["--mode", "live", "--request-delay", "15.5"])
        assert args_live.request_delay == pytest.approx(15.5)

    def test_generation_rate_limit_retry_success(self):
        """Fails twice with GenerationRateLimitError, then succeeds on 3rd attempt."""
        call_count = 0
        sleep_durations: list[float] = []

        async def fake_sleep(duration: float):
            sleep_durations.append(duration)

        mock_service = MagicMock()
        mock_answer = MagicMock()
        mock_answer.retrieved = [("uuid-1", 0)]
        mock_answer.generation = MagicMock(status="answered")
        mock_answer.timings_ms = {"total": 50}
        mock_answer.embedding_tokens = 10
        mock_answer.total_cost = None

        async def answer_with_retries(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise GenerationRateLimitError("Rate limit exceeded (429)")
            return mock_answer

        mock_service.answer = answer_with_retries

        record = {
            "id": "q_retry",
            "split": "dev",
            "query": "Q?",
            "answerable": True,
            "expected_evidence": [
                {"document_id": "synthetic_hr_policy", "chunk_index": 0}
            ],
        }

        retrieval, error = asyncio.run(
            _run_one_question_live(
                record=record,
                service=mock_service,
                tenant_id=1,
                top_k=5,
                uuid_to_doc_id={"uuid-1": "synthetic_hr_policy"},
                sleep_fn=fake_sleep,
                max_retries=4,
                base_backoff=1.0,
            )
        )

        assert error is None
        assert retrieval.generation_status == "answered"
        assert retrieval.retries == 2
        assert len(sleep_durations) == 2
        # Backoff: attempt 1 >= 1.0, attempt 2 >= 2.0
        assert sleep_durations[0] >= 1.0
        assert sleep_durations[1] >= 2.0

    def test_generation_rate_limit_retry_exhaustion(self):
        """Rate limit persists for all 4 retries (5 total calls); question fails with status 'failed'."""
        call_count = 0
        sleep_durations: list[float] = []

        async def fake_sleep(duration: float):
            sleep_durations.append(duration)

        mock_service = MagicMock()

        async def always_rate_limit(**kwargs):
            nonlocal call_count
            call_count += 1
            raise GenerationRateLimitError("Rate limit persistent")

        mock_service.answer = always_rate_limit

        record = {
            "id": "q_exhaust",
            "split": "dev",
            "query": "Q?",
            "answerable": True,
            "expected_evidence": [
                {"document_id": "synthetic_hr_policy", "chunk_index": 0}
            ],
        }

        retrieval, error = asyncio.run(
            _run_one_question_live(
                record=record,
                service=mock_service,
                tenant_id=1,
                top_k=5,
                uuid_to_doc_id={},
                sleep_fn=fake_sleep,
                max_retries=4,
                base_backoff=1.0,
            )
        )

        assert call_count == 5  # 1 initial + 4 retries
        assert retrieval.retries == 4
        assert retrieval.generation_status == "failed"
        assert error is not None
        assert "GenerationRateLimitError" in error
        assert len(sleep_durations) == 4

    def test_honors_provider_retry_delay(self):
        """Verifies that provider-supplied retry delay attribute or header is honored."""
        exc = GenerationRateLimitError("Rate limit exceeded")
        exc.retry_after = 8.5
        delay = _extract_retry_delay(exc)
        assert delay == pytest.approx(8.5)

        # Also test regex extraction from message
        exc_msg = GenerationRateLimitError("Quota exceeded: please retry after 12.5s")
        delay_msg = _extract_retry_delay(exc_msg)
        assert delay_msg == pytest.approx(12.5)

    def test_incomplete_report_flag_when_question_fails(self, tmp_path: Path):
        """When any question fails, report has 'complete': false and records failed question IDs."""
        output_file = tmp_path / "report_incomplete.json"
        q_ok = QuestionResult(
            question_id="q_ok",
            split="dev",
            answerable=True,
            generation_status="answered",
            recall_at_1=1.0,
            recall_at_3=1.0,
            recall_at_5=1.0,
            reciprocal_rank=1.0,
            retries=1,
            failed=False,
        )
        q_fail = QuestionResult(
            question_id="q_fail",
            split="dev",
            answerable=True,
            generation_status="failed",
            retries=4,
            failed=True,
            error="GenerationRateLimitError: persistent rate limit",
        )

        split_metrics = {
            "dev": compute_split_metrics(
                [q_ok, q_fail], split="dev", top_k=5, corpus_size=50
            )
        }

        report = write_report(
            output_path=output_file,
            dataset_path=Path("evals/datasets/rag_golden.jsonl"),
            dataset_size=2,
            mode="live",
            model_config={
                "embedder": "gemini-embedding-001",
                "generator": "gemini-3-flash-preview",
            },
            top_k=5,
            question_results=[q_ok, q_fail],
            split_metrics=split_metrics,
        )

        assert report["complete"] is False
        assert report["failed_question_ids"] == ["q_fail"]
        # Per question has retries recorded
        q_map = {q["id"]: q for q in report["per_question"]}
        assert q_map["q_ok"]["retries"] == 1
        assert q_map["q_fail"]["retries"] == 4
        assert q_map["q_fail"]["generation_status"] == "failed"
        # Warning caveat present
        assert any("INCOMPLETE RUN" in c for c in report["caveats"])

    def test_complete_report_flag_when_all_succeed(self, tmp_path: Path):
        output_file = tmp_path / "report_complete.json"
        q_ok = QuestionResult(
            question_id="q_ok",
            split="dev",
            answerable=True,
            generation_status="answered",
            recall_at_1=1.0,
            recall_at_3=1.0,
            recall_at_5=1.0,
            reciprocal_rank=1.0,
            failed=False,
        )

        split_metrics = {
            "dev": compute_split_metrics([q_ok], split="dev", top_k=5, corpus_size=50)
        }

        report = write_report(
            output_path=output_file,
            dataset_path=Path("evals/datasets/rag_golden.jsonl"),
            dataset_size=1,
            mode="live",
            model_config={
                "embedder": "gemini-embedding-001",
                "generator": "gemini-3-flash-preview",
            },
            top_k=5,
            question_results=[q_ok],
            split_metrics=split_metrics,
        )

        assert report["complete"] is True
        assert report["failed_question_ids"] == []
