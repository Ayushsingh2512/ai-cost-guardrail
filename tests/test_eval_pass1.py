"""
Tests for the RAG evaluation suite — Pass 1.

Covers:
- Golden dataset schema validation (valid records, missing fields, wrong types,
  invalid splits, duplicate IDs, invalid evidence references)
- Recall@5 (single expected chunk, multiple expected chunks, no retrieved results,
  partial hit, K smaller than corpus)
- Unanswerable question handling in metrics
- Abstention categorisation (correct abstention, unsupported answer,
  unnecessary abstention)
- Deterministic end-to-end run (fake embeddings + fake generation, no network)
- Partial execution failures and report generation

No network calls, no database access, no API keys required.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Imports under test
# ---------------------------------------------------------------------------
from evals.datasets.corpus import SYNTHETIC_CORPUS, VALID_EVIDENCE_REFS
from evals.fakes import (
    FakeCorpusRetriever,
    FakeGenerator,
    build_passages_from_corpus,
)
from evals.metrics import (
    QuestionResult,
    SplitMetrics,
    compute_split_metrics,
    recall_at_k,
)
from evals.report import write_report
from evals.schema import DatasetValidationError, load_and_validate, validate_record

# ===========================================================================
# Helpers
# ===========================================================================


def _make_record(
    id: str = "q_001",
    split: str = "dev",
    query: str = "What is the vacation policy?",
    reference_answer: str | None = "15 vacation days.",
    answerable: bool = True,
    expected_evidence: list | None = None,
) -> dict:
    rec: dict = {
        "id": id,
        "split": split,
        "query": query,
        "answerable": answerable,
        "expected_evidence": expected_evidence
        if expected_evidence is not None
        else [{"document_id": "synthetic_hr_policy", "chunk_index": 0}],
    }
    if reference_answer is not None:
        rec["reference_answer"] = reference_answer
    else:
        rec["reference_answer"] = None
    return rec


def _make_q_result(
    *,
    question_id: str = "q_001",
    split: str = "dev",
    answerable: bool = True,
    generation_status: str = "answered",
    recall: float | None = 1.0,
    latency_ms: int | None = 100,
    failed: bool = False,
    error: str | None = None,
) -> QuestionResult:
    return QuestionResult(
        question_id=question_id,
        split=split,
        answerable=answerable,
        generation_status=generation_status,  # type: ignore[arg-type]
        recall_at_k=recall,
        latency_ms=latency_ms,
        total_cost=None,
        embedding_tokens=None,
        failed=failed,
        error=error,
    )


# ===========================================================================
# 1. Dataset schema validation
# ===========================================================================


class TestValidateRecord:
    def test_valid_answerable_record_returns_no_errors(self):
        rec = _make_record()
        errors = validate_record(rec, line_number=1)
        assert errors == []

    def test_valid_unanswerable_record_returns_no_errors(self):
        rec = _make_record(
            answerable=False,
            reference_answer=None,
            expected_evidence=[],
        )
        errors = validate_record(rec, line_number=1)
        assert errors == []

    def test_missing_required_field_is_reported(self):
        rec = _make_record()
        del rec["query"]
        errors = validate_record(rec, line_number=2)
        assert any("query" in e for e in errors)

    def test_missing_multiple_fields_reports_all(self):
        rec = {}
        errors = validate_record(rec, line_number=3)
        # At minimum: id, split, query, answerable, expected_evidence missing.
        assert len(errors) >= 5

    def test_wrong_type_for_split_reported(self):
        rec = _make_record()
        rec["split"] = 42
        errors = validate_record(rec, line_number=1)
        assert any("split" in e for e in errors)

    def test_invalid_split_value_reported(self):
        rec = _make_record()
        rec["split"] = "train"
        errors = validate_record(rec, line_number=1)
        assert any("split" in e for e in errors)

    def test_both_dev_and_test_splits_accepted(self):
        for split in ("dev", "test"):
            rec = _make_record(split=split)
            errors = validate_record(rec, line_number=1)
            assert errors == [], f"Unexpected errors for split={split}: {errors}"

    def test_answerable_with_null_reference_answer_rejected(self):
        rec = _make_record(answerable=True)
        rec["reference_answer"] = None
        errors = validate_record(rec, line_number=1)
        assert any("reference_answer" in e for e in errors)

    def test_answerable_with_empty_evidence_rejected(self):
        rec = _make_record(expected_evidence=[])
        errors = validate_record(rec, line_number=1)
        assert any("expected_evidence" in e for e in errors)

    def test_unanswerable_with_reference_answer_rejected(self):
        rec = _make_record(
            answerable=False,
            reference_answer="Some answer",
            expected_evidence=[],
        )
        errors = validate_record(rec, line_number=1)
        assert any("reference_answer" in e for e in errors)

    def test_unanswerable_with_non_empty_evidence_rejected(self):
        rec = _make_record(
            answerable=False,
            reference_answer=None,
            expected_evidence=[
                {"document_id": "synthetic_hr_policy", "chunk_index": 0}
            ],
        )
        errors = validate_record(rec, line_number=1)
        assert any("expected_evidence" in e for e in errors)

    def test_invalid_evidence_document_id_rejected(self):
        rec = _make_record(
            expected_evidence=[{"document_id": "does_not_exist", "chunk_index": 0}]
        )
        errors = validate_record(rec, line_number=1)
        assert any("does_not_exist" in e for e in errors)

    def test_invalid_evidence_chunk_index_rejected(self):
        rec = _make_record(
            expected_evidence=[
                {"document_id": "synthetic_hr_policy", "chunk_index": 999}
            ]
        )
        errors = validate_record(rec, line_number=1)
        assert any("synthetic_hr_policy" in e and "999" in e for e in errors)

    def test_evidence_missing_document_id_reported(self):
        rec = _make_record(expected_evidence=[{"chunk_index": 0}])
        errors = validate_record(rec, line_number=1)
        assert any("document_id" in e for e in errors)

    def test_evidence_missing_chunk_index_reported(self):
        rec = _make_record(expected_evidence=[{"document_id": "synthetic_hr_policy"}])
        errors = validate_record(rec, line_number=1)
        assert any("chunk_index" in e for e in errors)


class TestLoadAndValidate:
    def test_loads_canonical_golden_dataset(self):
        golden_path = Path("evals/datasets/rag_golden.jsonl")
        records = load_and_validate(golden_path)
        assert len(records) == 30
        dev_count = sum(1 for r in records if r["split"] == "dev")
        test_count = sum(1 for r in records if r["split"] == "test")
        assert dev_count == 20
        assert test_count == 10

    def test_all_ids_unique_in_golden_dataset(self):
        golden_path = Path("evals/datasets/rag_golden.jsonl")
        records = load_and_validate(golden_path)
        ids = [r["id"] for r in records]
        assert len(ids) == len(set(ids))

    def test_duplicate_id_raises(self, tmp_path: Path):
        content = (
            '{"id": "q_001", "split": "dev", "query": "Q?", "answerable": false, '
            '"reference_answer": null, "expected_evidence": []}\n'
            '{"id": "q_001", "split": "dev", "query": "Q2?", "answerable": false, '
            '"reference_answer": null, "expected_evidence": []}\n'
        )
        f = tmp_path / "dup.jsonl"
        f.write_text(content, encoding="utf-8")
        with pytest.raises(DatasetValidationError, match="duplicate id"):
            load_and_validate(f)

    def test_invalid_json_line_raises(self, tmp_path: Path):
        f = tmp_path / "bad.jsonl"
        f.write_text("{invalid json}\n", encoding="utf-8")
        with pytest.raises(DatasetValidationError, match="invalid JSON"):
            load_and_validate(f)

    def test_file_not_found_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            load_and_validate(tmp_path / "missing.jsonl")

    def test_empty_file_raises(self, tmp_path: Path):
        f = tmp_path / "empty.jsonl"
        f.write_text("", encoding="utf-8")
        with pytest.raises(DatasetValidationError, match="empty"):
            load_and_validate(f)

    def test_all_evidence_refs_in_golden_dataset_exist_in_corpus(self):
        golden_path = Path("evals/datasets/rag_golden.jsonl")
        records = load_and_validate(golden_path)
        for rec in records:
            for ev in rec["expected_evidence"]:
                ref = (ev["document_id"], ev["chunk_index"])
                assert ref in VALID_EVIDENCE_REFS, (
                    f"Evidence ref {ref!r} in {rec['id']!r} not in corpus"
                )


# ===========================================================================
# 2. Recall@5
# ===========================================================================


class TestRecallAtK:
    def test_perfect_single_chunk_recall(self):
        assert recall_at_k(
            expected_refs=[("doc_a", 0)],
            retrieved_refs=[("doc_a", 0), ("doc_b", 1)],
            k=5,
        ) == pytest.approx(1.0)

    def test_zero_recall_when_nothing_retrieved(self):
        assert recall_at_k(
            expected_refs=[("doc_a", 0)],
            retrieved_refs=[],
            k=5,
        ) == pytest.approx(0.0)

    def test_zero_recall_when_wrong_chunk_retrieved(self):
        assert recall_at_k(
            expected_refs=[("doc_a", 0)],
            retrieved_refs=[("doc_b", 0), ("doc_c", 1)],
            k=5,
        ) == pytest.approx(0.0)

    def test_multiple_expected_chunks_all_found(self):
        assert recall_at_k(
            expected_refs=[("doc_a", 0), ("doc_a", 1)],
            retrieved_refs=[("doc_a", 0), ("doc_a", 1), ("doc_b", 0)],
            k=5,
        ) == pytest.approx(1.0)

    def test_multiple_expected_chunks_half_found(self):
        assert recall_at_k(
            expected_refs=[("doc_a", 0), ("doc_a", 1)],
            retrieved_refs=[("doc_a", 0), ("doc_b", 0)],
            k=5,
        ) == pytest.approx(0.5)

    def test_k_truncation_applies(self):
        # k=1 means only the first retrieved result counts.
        assert recall_at_k(
            expected_refs=[("doc_a", 0)],
            retrieved_refs=[("doc_b", 0), ("doc_a", 0)],
            k=1,
        ) == pytest.approx(0.0)

    def test_fewer_than_k_retrieved_evaluates_what_was_returned(self):
        # Only 2 passages returned; k=5 but we evaluate what we have.
        assert recall_at_k(
            expected_refs=[("doc_a", 0)],
            retrieved_refs=[("doc_a", 0), ("doc_b", 1)],
            k=5,
        ) == pytest.approx(1.0)

    def test_empty_expected_refs_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            recall_at_k(expected_refs=[], retrieved_refs=[("doc_a", 0)])

    def test_multi_chunk_none_found(self):
        assert recall_at_k(
            expected_refs=[("doc_a", 0), ("doc_a", 1)],
            retrieved_refs=[],
            k=5,
        ) == pytest.approx(0.0)


# ===========================================================================
# 3. Abstention metrics
# ===========================================================================


class TestAbstentionMetrics:
    def _run(self, results: list[QuestionResult], split: str = "dev") -> SplitMetrics:
        return compute_split_metrics(results, split=split)

    def test_unanswerable_no_context_is_correct_abstention(self):
        r = _make_q_result(
            answerable=False,
            generation_status="no_context",
            recall=None,
        )
        m = self._run([r])
        assert m.correct_abstentions == 1
        assert m.unsupported_answers == 0
        assert m.unnecessary_abstentions == 0

    def test_unanswerable_answered_is_unsupported_answer(self):
        r = _make_q_result(
            answerable=False,
            generation_status="answered",
            recall=None,
        )
        m = self._run([r])
        assert m.unsupported_answers == 1
        assert m.correct_abstentions == 0

    def test_answerable_no_context_is_unnecessary_abstention(self):
        r = _make_q_result(
            answerable=True,
            generation_status="no_context",
            recall=0.0,
        )
        m = self._run([r])
        assert m.unnecessary_abstentions == 1
        assert m.correct_abstentions == 0

    def test_failed_questions_excluded_from_abstention_counts(self):
        r = _make_q_result(
            answerable=False,
            generation_status="no_context",
            recall=None,
            failed=True,
            error="some error",
        )
        m = self._run([r])
        assert m.correct_abstentions == 0
        assert m.failed_count == 1

    def test_mixed_results_aggregated_correctly(self):
        results = [
            # Unanswerable → correct abstention
            _make_q_result(
                question_id="q1",
                answerable=False,
                generation_status="no_context",
                recall=None,
            ),
            # Unanswerable → unsupported answer
            _make_q_result(
                question_id="q2",
                answerable=False,
                generation_status="answered",
                recall=None,
            ),
            # Answerable → correct answer
            _make_q_result(
                question_id="q3",
                answerable=True,
                generation_status="answered",
                recall=1.0,
            ),
            # Answerable → unnecessary abstention
            _make_q_result(
                question_id="q4",
                answerable=True,
                generation_status="no_context",
                recall=0.0,
            ),
        ]
        m = self._run(results)
        assert m.correct_abstentions == 1
        assert m.unsupported_answers == 1
        assert m.unnecessary_abstentions == 1
        assert m.answerable_count == 2
        assert m.unanswerable_count == 2


# ===========================================================================
# 4. Split metrics aggregation
# ===========================================================================


class TestSplitMetrics:
    def test_recall_macro_average(self):
        results = [
            _make_q_result(question_id="q1", recall=1.0),
            _make_q_result(question_id="q2", recall=0.0),
        ]
        m = compute_split_metrics(results, split="dev")
        assert m.recall_at_k_mean == pytest.approx(0.5)

    def test_no_answerable_questions_gives_none_recall(self):
        results = [
            _make_q_result(
                question_id="q1",
                answerable=False,
                generation_status="no_context",
                recall=None,
            )
        ]
        m = compute_split_metrics(results, split="dev")
        assert m.recall_at_k_mean is None

    def test_latency_statistics_computed(self):
        results = [
            _make_q_result(question_id="q1", latency_ms=100),
            _make_q_result(question_id="q2", latency_ms=200),
            _make_q_result(question_id="q3", latency_ms=300),
        ]
        m = compute_split_metrics(results, split="dev")
        assert m.latency_ms_mean == pytest.approx(200.0)
        assert m.latency_ms_p50 is not None
        assert m.latency_ms_p95 is not None

    def test_split_filter_applies(self):
        results = [
            _make_q_result(question_id="q1", split="dev", recall=1.0),
            _make_q_result(question_id="q2", split="test", recall=0.0),
        ]
        m_dev = compute_split_metrics(results, split="dev")
        m_test = compute_split_metrics(results, split="test")
        assert m_dev.total == 1
        assert m_test.total == 1
        assert m_dev.recall_at_k_mean == pytest.approx(1.0)
        assert m_test.recall_at_k_mean == pytest.approx(0.0)


# ===========================================================================
# 5. Fake retriever and generator
# ===========================================================================


class TestFakeCorpusRetriever:
    def setup_method(self):
        self.passages = build_passages_from_corpus(SYNTHETIC_CORPUS)
        self.retriever = FakeCorpusRetriever(passages=self.passages)

    def test_returns_at_most_k_results(self):
        from app.rag.embeddings import FakeEmbedder

        e = FakeEmbedder(dimensions=8)
        vector = e.embed_query("vacation days policy")
        results = self.retriever.search(
            tenant_id=0,
            vector=vector,
            k=3,
            fingerprint="fake/fake-embedding/8",
        )
        assert len(results) <= 3

    def test_returns_context_passages(self):
        from app.rag.embeddings import FakeEmbedder
        from app.rag.generation import ContextPassage

        e = FakeEmbedder(dimensions=8)
        vector = e.embed_query("vacation days policy")
        results = self.retriever.search(
            tenant_id=0,
            vector=vector,
            k=5,
            fingerprint="fake/fake-embedding/8",
        )
        assert all(isinstance(r, ContextPassage) for r in results)

    def test_k_zero_raises(self):
        from app.rag.embeddings import FakeEmbedder

        e = FakeEmbedder(dimensions=8)
        vector = e.embed_query("test")
        with pytest.raises(ValueError, match="k must be greater than zero"):
            self.retriever.search(
                tenant_id=0,
                vector=vector,
                k=0,
                fingerprint="fake/fake-embedding/8",
            )

    def test_empty_vector_raises(self):
        with pytest.raises(ValueError, match="vector must not be empty"):
            self.retriever.search(
                tenant_id=0,
                vector=[],
                k=5,
                fingerprint="fake/fake-embedding/8",
            )


class TestFakeGenerator:
    def test_empty_passages_returns_no_context(self):
        gen = FakeGenerator()
        result = asyncio.run(gen.generate("What is the policy?", []))
        assert result.status == "no_context"
        assert result.answer is None
        assert result.citations == []

    def test_non_empty_passages_returns_answered(self):
        from app.rag.generation import ContextPassage

        gen = FakeGenerator()
        passages = [ContextPassage(ref=("doc_a", 0), text="Some evidence text.")]
        result = asyncio.run(gen.generate("What is the policy?", passages))
        assert result.status == "answered"
        assert result.answer is not None
        assert result.citations  # at least one citation


# ===========================================================================
# 6. End-to-end deterministic run
# ===========================================================================


class TestDeterministicEndToEnd:
    def test_deterministic_run_completes_without_network(self, tmp_path: Path):
        """Full harness run: load dataset, run deterministic eval, write report."""
        from evals.evaluate_rag import run_evaluation
        from evals.schema import load_and_validate

        golden_path = Path("evals/datasets/rag_golden.jsonl")
        records = load_and_validate(golden_path)

        results = asyncio.run(
            run_evaluation(records=records, mode="deterministic", top_k=5)
        )

        assert len(results) == len(records)

        # No result should have failed.
        failures = [r for r in results if r.failed]
        assert failures == [], (
            f"Some questions failed: {[r.question_id for r in failures]}"
        )

    def test_deterministic_run_produces_valid_report(self, tmp_path: Path):
        from evals.evaluate_rag import run_evaluation
        from evals.schema import load_and_validate

        golden_path = Path("evals/datasets/rag_golden.jsonl")
        records = load_and_validate(golden_path)

        results = asyncio.run(
            run_evaluation(records=records, mode="deterministic", top_k=5)
        )

        split_metrics = {}
        for split in ("dev", "test"):
            split_metrics[split] = compute_split_metrics(results, split=split)

        output_path = tmp_path / "report.json"
        report_dict = write_report(
            output_path=output_path,
            dataset_path=golden_path,
            dataset_size=len(records),
            mode="deterministic",
            model_config={"embedder": "FakeEmbedder", "generator": "FakeGenerator"},
            top_k=5,
            question_results=results,
            split_metrics=split_metrics,
        )

        assert output_path.exists()

        with output_path.open(encoding="utf-8") as fh:
            loaded = json.load(fh)

        assert loaded["mode"] == report_dict["mode"]

        assert loaded["mode"] == "deterministic"
        assert loaded["dataset"]["size"] == 30
        assert "dev" in loaded["splits"]
        assert "test" in loaded["splits"]

        # Deterministic disclaimer is present.
        assert any("DETERMINISTIC" in c for c in loaded["caveats"])

    def test_deterministic_mode_all_answerable_questions_get_recall(
        self, tmp_path: Path
    ):
        from evals.evaluate_rag import run_evaluation
        from evals.schema import load_and_validate

        golden_path = Path("evals/datasets/rag_golden.jsonl")
        records = load_and_validate(golden_path)
        results = asyncio.run(
            run_evaluation(records=records, mode="deterministic", top_k=5)
        )

        answerable_results = [r for r in results if r.answerable and not r.failed]
        for qr in answerable_results:
            assert qr.recall_at_k is not None, (
                f"Answerable question {qr.question_id} has no recall score"
            )

    def test_unanswerable_questions_excluded_from_recall(self, tmp_path: Path):
        from evals.evaluate_rag import run_evaluation
        from evals.schema import load_and_validate

        golden_path = Path("evals/datasets/rag_golden.jsonl")
        records = load_and_validate(golden_path)
        results = asyncio.run(
            run_evaluation(records=records, mode="deterministic", top_k=5)
        )

        unanswerable_results = [r for r in results if not r.answerable]
        for qr in unanswerable_results:
            assert qr.recall_at_k is None, (
                f"Unanswerable question {qr.question_id} should not have a recall score"
            )


# ===========================================================================
# 7. Partial failure handling
# ===========================================================================


class TestPartialFailures:
    def test_failed_question_marked_as_failed_in_report(self, tmp_path: Path):
        results = [
            _make_q_result(question_id="q_ok", failed=False, recall=1.0),
            _make_q_result(
                question_id="q_fail",
                failed=True,
                error="Pipeline exploded",
                recall=None,
                generation_status="no_context",
            ),
        ]
        split_metrics = {
            "dev": compute_split_metrics(results, split="dev"),
        }
        output_path = tmp_path / "report.json"
        report = write_report(
            output_path=output_path,
            dataset_path=Path("evals/datasets/rag_golden.jsonl"),
            dataset_size=2,
            mode="deterministic",
            model_config={},
            top_k=5,
            question_results=results,
            split_metrics=split_metrics,
        )

        # Report written even with failures.
        assert output_path.exists()

        failed_entries = [q for q in report["per_question"] if q["failed"]]
        assert len(failed_entries) == 1
        assert failed_entries[0]["id"] == "q_fail"
        assert failed_entries[0]["error"] == "Pipeline exploded"

    def test_failed_question_not_counted_in_recall(self):
        results = [
            _make_q_result(question_id="q_ok", recall=1.0, failed=False),
            _make_q_result(
                question_id="q_fail",
                recall=None,
                failed=True,
                error="boom",
                generation_status="no_context",
            ),
        ]
        m = compute_split_metrics(results, split="dev")
        # Only q_ok is answerable and not failed → one recall value.
        assert len(m.recall_at_k_values) == 1
        assert m.failed_count == 1
