"""
Tests for RAG Evaluation hardening:
1. --retrieval-only (skips generation, records scores for top_k results)
2. Per-question details in report and checkpoint JSONL (answer, citations, scores, validity, secrets sanitized)
3. Report header fields (git_commit, dataset_sha256, corpus_version, prompt_version, model, temp, top_k, config_fingerprint)
4. --resume with config-fingerprint check and refusal on mismatch
5. --only-ids question filtering
6. --order unanswerable-first
7. --skip-ingest when fingerprint matches
8. Immediate stop on 429 daily quota
9. Immediate stop on 2 consecutive retry exhaustions
10. Retrieval score calibration script (dev-only enforcement, distributions, margin)
"""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.rag.generation import (
    ContextPassage,
    GenerationRateLimitError,
)
from evals.calibrate_retrieval import (
    analyze_retrieval_calibration,
    compute_distribution_stats,
)
from evals.evaluate_rag import (
    _build_arg_parser,
    _is_daily_quota_error,
    _run_one_question_deterministic,
    _run_one_question_live,
    run_evaluation,
)
from evals.fakes import FakeCorpusRetriever, FakeGenerator
from evals.metrics import (
    QuestionResult,
    evaluate_citation_validity,
)
from evals.report import (
    compute_config_fingerprint,
    read_checkpoint,
    sanitize_secrets,
    write_report,
)
from evals.schema import GoldenRecord

# ===========================================================================
# 1. --retrieval-only tests
# ===========================================================================


class TestRetrievalOnly:
    @pytest.mark.asyncio
    async def test_deterministic_retrieval_only_skips_generation(self):
        retriever = FakeCorpusRetriever(
            passages={
                ("doc_a", 0): ContextPassage(
                    ref=("doc_a", 0), text="Alpha chunk text", source="doc_a", page=0
                ),
                ("doc_b", 1): ContextPassage(
                    ref=("doc_b", 1), text="Beta chunk text", source="doc_b", page=1
                ),
            }
        )
        generator = FakeGenerator()
        generator.generate = AsyncMock()  # type: ignore[method-assign]

        record: GoldenRecord = {
            "id": "q_ret_only",
            "split": "dev",
            "query": "Alpha information?",
            "answerable": True,
            "expected_evidence": [{"document_id": "doc_a", "chunk_index": 0}],
        }

        retrieval, error = await _run_one_question_deterministic(
            record=record,
            retriever=retriever,
            generator=generator,
            top_k=2,
            retrieval_only=True,
        )

        assert error is None
        assert retrieval.generation_status == "retrieval_only"
        assert retrieval.answer is None
        assert retrieval.citations == []
        assert len(retrieval.retrieved_scores) == 2
        # Generator was never called
        generator.generate.assert_not_called()

        # Check that scores are present and sorted
        score_1 = retrieval.retrieved_scores[0][2]
        score_2 = retrieval.retrieved_scores[1][2]
        assert isinstance(score_1, float)
        assert score_1 >= score_2

    @pytest.mark.asyncio
    async def test_live_retrieval_only_embeds_and_retrieves_without_generate(self):
        mock_service = MagicMock()
        mock_embedder = MagicMock()
        mock_embedder.embed_query.return_value = [0.1, 0.2, 0.3]
        mock_embedder.fingerprint = "test-fp"
        mock_service.embedder = mock_embedder

        mock_retriever = MagicMock()
        p1 = ContextPassage(ref=("uuid-1", 0), text="Passage 1", source="doc_1", page=0)
        p2 = ContextPassage(ref=("uuid-2", 0), text="Passage 2", source="doc_2", page=0)
        mock_retriever.search_with_scores.return_value = [(p1, 0.95), (p2, 0.82)]
        mock_service.retriever = mock_retriever

        # Generator answer must NOT be called
        mock_service.answer = AsyncMock()

        record: GoldenRecord = {
            "id": "q_live_ret",
            "split": "dev",
            "query": "Test query?",
            "answerable": True,
            "expected_evidence": [{"document_id": "doc_1", "chunk_index": 0}],
        }

        uuid_to_doc = {"uuid-1": "doc_1", "uuid-2": "doc_2"}

        retrieval, error = await _run_one_question_live(
            record=record,
            service=mock_service,
            tenant_id=42,
            top_k=2,
            uuid_to_doc_id=uuid_to_doc,
            retrieval_only=True,
        )

        assert error is None
        assert retrieval.generation_status == "retrieval_only"
        assert retrieval.retrieved_refs == [("doc_1", 0), ("doc_2", 0)]
        assert retrieval.retrieved_scores == [("doc_1", 0, 0.95), ("doc_2", 0, 0.82)]
        assert retrieval.answer is None
        assert retrieval.citations == []
        mock_service.answer.assert_not_called()


# ===========================================================================
# 2. Per-question details & Secret sanitization
# ===========================================================================


class TestPerQuestionAndSecrets:
    def test_sanitize_secrets_redacts_api_keys(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "real-super-secret-api-key-12345")
        text = "Failed to call Gemini with key=real-super-secret-api-key-12345 AIzaSyDummyKey1234567890abcdefghijkl"
        sanitized = sanitize_secrets(text)
        assert "real-super-secret-api-key-12345" not in sanitized
        assert "AIzaSyDummyKey1234567890abcdefghijkl" not in sanitized
        assert "[REDACTED" in sanitized

    def test_evaluate_citation_validity(self):
        # Valid citations
        valid_res = evaluate_citation_validity(
            generation_status="answered",
            citations=[("doc_a", 0)],
            retrieved_refs=[("doc_a", 0), ("doc_b", 1)],
            invalid_markers_count=0,
        )
        assert valid_res["valid"] is True
        assert valid_res["unretrieved_citations_count"] == 0

        # Unretrieved citation
        invalid_res1 = evaluate_citation_validity(
            generation_status="answered",
            citations=[("doc_c", 0)],
            retrieved_refs=[("doc_a", 0)],
            invalid_markers_count=0,
        )
        assert invalid_res1["valid"] is False
        assert invalid_res1["unretrieved_citations_count"] == 1

        # Invalid citation markers ([99])
        invalid_res2 = evaluate_citation_validity(
            generation_status="answered",
            citations=[("doc_a", 0)],
            retrieved_refs=[("doc_a", 0)],
            invalid_markers_count=1,
        )
        assert invalid_res2["valid"] is False
        assert invalid_res2["invalid_markers_count"] == 1

        # Retrieval-only or no_context
        no_ctx_res = evaluate_citation_validity(
            generation_status="no_context",
            citations=[],
            retrieved_refs=[],
            invalid_markers_count=0,
        )
        assert no_ctx_res["valid"] is True

    def test_per_question_recorded_in_checkpoint_and_report(self, tmp_path):
        out_json = tmp_path / "report.json"

        qr = QuestionResult(
            question_id="q_100",
            split="dev",
            answerable=True,
            generation_status="answered",
            recall_at_1=1.0,
            recall_at_3=1.0,
            recall_at_5=1.0,
            reciprocal_rank=1.0,
            latency_ms=120,
            total_cost=Decimal("0.000200"),
            embedding_tokens=15,
            error=None,
            failed=False,
            retries=1,
            answer="According to [1], vacation is 15 days.",
            citations=[{"document_id": "doc_hr", "chunk_index": 0}],
            retrieved_chunks=[
                {"document_id": "doc_hr", "chunk_index": 0, "score": 0.92}
            ],
            citation_validity={"valid": True, "reason": "All citations valid"},
        )

        rep = write_report(
            output_path=out_json,
            dataset_path=Path("evals/datasets/rag_golden.jsonl"),
            dataset_size=1,
            mode="live",
            model_config={"generator": "gemini-3-flash-preview"},
            top_k=5,
            question_results=[qr],
            split_metrics={},
        )

        pq = rep["per_question"][0]
        assert pq["id"] == "q_100"
        assert pq["answer"] == "According to [1], vacation is 15 days."
        assert pq["citations"] == [{"document_id": "doc_hr", "chunk_index": 0}]
        assert pq["retrieved"] == [
            {"document_id": "doc_hr", "chunk_index": 0, "score": 0.92}
        ]
        assert pq["citation_validity"]["valid"] is True
        assert pq["retries"] == 1


# ===========================================================================
# 3. Report Header Metadata & Fingerprint
# ===========================================================================


class TestReportHeader:
    def test_report_header_contains_all_required_fields(self, tmp_path):
        out_json = tmp_path / "header_test.json"
        rep = write_report(
            output_path=out_json,
            dataset_path=Path("evals/datasets/rag_golden.jsonl"),
            dataset_size=10,
            mode="live",
            model_config={
                "generator": "gemini-3-flash-preview",
                "temperature": 0.0,
                "prompt_version": "rag-grounded-v1",
            },
            top_k=5,
            question_results=[],
            split_metrics={},
            git_commit="abc1234567890",
            corpus_version="1.5.0",
        )

        assert rep["git_commit"] == "abc1234567890"
        assert len(rep["dataset_sha256"]) == 64
        assert rep["corpus_version"] == "1.5.0"
        assert rep["prompt_version"] == "rag-grounded-v1"
        assert rep["generator_model"] == "gemini-3-flash-preview"
        assert rep["temperature"] == 0.0
        assert rep["top_k"] == 5
        assert len(rep["config_fingerprint"]) == 64


# ===========================================================================
# 4. --resume, --only-ids, and --order
# ===========================================================================


class TestResumeAndCliFiltering:
    def test_resume_refuses_on_config_fingerprint_mismatch(self, tmp_path):
        chk_path = tmp_path / "test.checkpoint.jsonl"
        with chk_path.open("w", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "type": "header",
                        "config_fingerprint": "different_hash_value_12345",
                    }
                )
                + "\n"
            )

        header, _ = read_checkpoint(chk_path)
        assert header is not None
        assert header["config_fingerprint"] == "different_hash_value_12345"

        current_fp = compute_config_fingerprint(
            dataset_sha256="abc",
            corpus_version="1.5.0",
            prompt_version="v1",
            generator_model="model_x",
            temperature=0.0,
            top_k=5,
            mode="deterministic",
        )
        assert current_fp != header["config_fingerprint"]

    @pytest.mark.asyncio
    async def test_resume_skips_completed_questions(self):
        records: list[GoldenRecord] = [
            {
                "id": "q_done",
                "split": "dev",
                "query": "Q1",
                "answerable": True,
                "expected_evidence": [{"document_id": "doc_a", "chunk_index": 0}],
            },
            {
                "id": "q_new",
                "split": "dev",
                "query": "Q2",
                "answerable": False,
                "expected_evidence": [],
            },
        ]

        completed_qr = QuestionResult(
            question_id="q_done",
            split="dev",
            answerable=True,
            generation_status="answered",
            recall_at_1=1.0,
            recall_at_5=1.0,
        )

        completed_map = {"q_done": completed_qr}

        results = await run_evaluation(
            records=records,
            mode="deterministic",
            top_k=2,
            corpus={"doc_a": ["Text A"]},
            completed_results=completed_map,
        )

        assert len(results) == 2
        assert results[0].question_id == "q_done"
        assert results[0] is completed_qr
        assert results[1].question_id == "q_new"

    def test_cli_parsing_options(self):
        parser = _build_arg_parser()
        args = parser.parse_args(
            [
                "--retrieval-only",
                "--resume",
                "--only-ids",
                "q_001",
                "q_002",
                "--order",
                "unanswerable-first",
                "--skip-ingest",
            ]
        )
        assert args.retrieval_only is True
        assert args.resume is True
        assert args.only_ids == ["q_001", "q_002"]
        assert args.order == "unanswerable-first"
        assert args.skip_ingest is True


# ===========================================================================
# 5. 429 Daily Quota & 2 Consecutive Retry Exhaustion Immediate Stops
# ===========================================================================


class TestImmediateStopConditions:
    def test_daily_quota_error_detection(self):
        e1 = Exception(
            "429 Resource has been exhausted: quota limit 'GenerateContent requests per day' exceeded"
        )
        is_daily, details = _is_daily_quota_error(e1)
        assert is_daily is True
        assert "per day" in (details or "").lower()

        e2 = GenerationRateLimitError("Rate limit hit, retry after 5s")
        is_daily2, _ = _is_daily_quota_error(e2)
        assert is_daily2 is False

    @pytest.mark.asyncio
    async def test_stops_immediately_on_429_daily_quota(self):
        mock_service = MagicMock()
        mock_service.answer = AsyncMock(
            side_effect=Exception(
                "429 Resource exhausted: free_tier_requests_per_day exceeded"
            )
        )

        records: list[GoldenRecord] = [
            {
                "id": "q_quota",
                "split": "dev",
                "query": "Q1",
                "answerable": True,
                "expected_evidence": [{"document_id": "doc_1", "chunk_index": 0}],
            },
            {
                "id": "q_pending_1",
                "split": "dev",
                "query": "Q2",
                "answerable": False,
                "expected_evidence": [],
            },
            {
                "id": "q_pending_2",
                "split": "dev",
                "query": "Q3",
                "answerable": False,
                "expected_evidence": [],
            },
        ]

        run_state: dict[str, Any] = {}
        results = await run_evaluation(
            records=records,
            mode="live",
            top_k=5,
            tenant_id=1,
            service=mock_service,
            uuid_to_doc_id={"uuid-1": "doc_1"},
            run_state=run_state,
        )

        # Only 1 question was run; the rest are pending
        assert len(results) == 1
        assert results[0].question_id == "q_quota"
        assert results[0].failed is True

        assert run_state["stopped_early"] is True
        assert run_state["stop_reason"] == "daily_quota"
        assert run_state["pending_ids"] == ["q_pending_1", "q_pending_2"]
        assert run_state["quota_error"]["type"] == "daily_quota"

    @pytest.mark.asyncio
    async def test_stops_immediately_on_2_consecutive_retry_exhaustions(self):
        call_count = 0

        async def fail_rate_limit(**kwargs):
            nonlocal call_count
            call_count += 1
            raise GenerationRateLimitError("Rate limit exceeded")

        mock_service = MagicMock()
        mock_service.answer = fail_rate_limit

        records: list[GoldenRecord] = [
            {
                "id": "q_fail_1",
                "split": "dev",
                "query": "Q1",
                "answerable": True,
                "expected_evidence": [{"document_id": "doc_1", "chunk_index": 0}],
            },
            {
                "id": "q_fail_2",
                "split": "dev",
                "query": "Q2",
                "answerable": True,
                "expected_evidence": [{"document_id": "doc_1", "chunk_index": 0}],
            },
            {
                "id": "q_pending",
                "split": "dev",
                "query": "Q3",
                "answerable": False,
                "expected_evidence": [],
            },
        ]

        run_state: dict[str, Any] = {}

        async def fast_sleep(_):
            pass

        results = await run_evaluation(
            records=records,
            mode="live",
            top_k=5,
            tenant_id=1,
            service=mock_service,
            uuid_to_doc_id={"uuid-1": "doc_1"},
            sleep_fn=fast_sleep,
            run_state=run_state,
        )

        # Two consecutive questions failed after retries, 3rd question was not run
        assert len(results) == 2
        assert results[0].question_id == "q_fail_1"
        assert results[1].question_id == "q_fail_2"
        assert run_state["stopped_early"] is True
        assert run_state["stop_reason"] == "consecutive_retries_exhausted"
        assert run_state["pending_ids"] == ["q_pending"]


# ===========================================================================
# 6. Retrieval Score Calibration Tool Tests
# ===========================================================================


class TestCalibrationTool:
    def test_calibration_distribution_stats(self):
        scores = [0.80, 0.85, 0.90, 0.95]
        stats = compute_distribution_stats(scores)
        assert stats["count"] == 4
        assert stats["min"] == 0.80
        assert stats["max"] == 0.95
        assert stats["mean"] == pytest.approx(0.875, abs=1e-4)
        assert stats["median"] == pytest.approx(0.875, abs=1e-4)

    def test_calibration_analysis_separation_margin(self):
        report_data = {
            "per_question": [
                {
                    "id": "q_ans_1",
                    "split": "dev",
                    "answerable": True,
                    "retrieved": [
                        {"document_id": "doc_1", "chunk_index": 0, "score": 0.88}
                    ],
                },
                {
                    "id": "q_ans_2",
                    "split": "dev",
                    "answerable": True,
                    "retrieved": [
                        {"document_id": "doc_1", "chunk_index": 1, "score": 0.92}
                    ],
                },
                {
                    "id": "q_unans_1",
                    "split": "dev",
                    "answerable": False,
                    "retrieved": [
                        {"document_id": "doc_2", "chunk_index": 0, "score": 0.55}
                    ],
                },
                {
                    "id": "q_unans_2",
                    "split": "dev",
                    "answerable": False,
                    "retrieved": [
                        {"document_id": "doc_2", "chunk_index": 1, "score": 0.60}
                    ],
                },
            ]
        }

        calib = analyze_retrieval_calibration(report_data)
        assert calib["split"] == "dev"
        assert calib["answerable"]["min"] == 0.88
        assert calib["unanswerable"]["max"] == 0.60
        # Separation margin: 0.88 - 0.60 = 0.28
        assert calib["separation"]["margin"] == pytest.approx(0.28, abs=1e-4)
        assert calib["separation"]["separable"] is True
        assert calib["separation"]["suggested_threshold"] == pytest.approx(
            0.74, abs=1e-4
        )

    def test_calibration_refuses_test_split(self):
        report_data = {
            "per_question": [
                {
                    "id": "q_test_1",
                    "split": "test",
                    "answerable": True,
                    "retrieved": [
                        {"document_id": "doc_1", "chunk_index": 0, "score": 0.90}
                    ],
                }
            ]
        }

        with pytest.raises(ValueError, match="dev.*split only"):
            analyze_retrieval_calibration(report_data)
