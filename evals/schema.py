"""
Schema validation for the RAG evaluation golden dataset.

Validates that each record in the JSONL file conforms to the required schema
and that all expected_evidence references exist in the committed corpus.

This module is intentionally dependency-free (stdlib only) so it can be
imported by tests without requiring any project dependencies.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from evals.datasets.corpus import VALID_EVIDENCE_REFS

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

EvidenceRef = dict[str, Any]  # {"document_id": str, "chunk_index": int}
GoldenRecord = dict[str, Any]

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

REQUIRED_FIELDS: dict[str, type | tuple[type, ...]] = {
    "id": str,
    "split": str,
    "query": str,
    "answerable": bool,
    "expected_evidence": list,
}

VALID_SPLITS = {"dev", "test"}


class DatasetValidationError(ValueError):
    """Raised when the golden dataset fails schema validation."""


def validate_record(record: GoldenRecord, line_number: int) -> list[str]:
    """
    Validate a single golden record.

    Returns a (possibly empty) list of human-readable error messages.
    Does not raise; accumulate errors and let the caller decide.
    """
    errors: list[str] = []
    prefix = f"Line {line_number} id={record.get('id', '<missing>')!r}"

    # --- required fields ---
    for field, expected_type in REQUIRED_FIELDS.items():
        if field not in record:
            errors.append(f"{prefix}: missing required field {field!r}")
            continue
        value = record[field]
        if not isinstance(value, expected_type):
            errors.append(
                f"{prefix}: field {field!r} must be {expected_type}, "
                f"got {type(value).__name__}"
            )

    if errors:
        # Cannot safely check semantic rules without required fields.
        return errors

    # --- split ---
    if record["split"] not in VALID_SPLITS:
        errors.append(
            f"{prefix}: split must be one of {sorted(VALID_SPLITS)}, "
            f"got {record['split']!r}"
        )

    # --- answerable / reference_answer / expected_evidence consistency ---
    answerable: bool = record["answerable"]
    ref_answer = record.get("reference_answer")
    expected_evidence: list = record["expected_evidence"]

    if answerable:
        if ref_answer is None:
            errors.append(f"{prefix}: answerable=true but reference_answer is null")
        elif not isinstance(ref_answer, str):
            errors.append(
                f"{prefix}: reference_answer must be a string, "
                f"got {type(ref_answer).__name__}"
            )
        if not expected_evidence:
            errors.append(f"{prefix}: answerable=true but expected_evidence is empty")
    else:
        # Unanswerable: reference_answer must be absent or null,
        # expected_evidence must be empty.
        if ref_answer is not None:
            errors.append(
                f"{prefix}: answerable=false but reference_answer is not null"
            )
        if expected_evidence:
            errors.append(
                f"{prefix}: answerable=false but expected_evidence is non-empty"
            )

    # --- expected_evidence items ---
    for i, ev in enumerate(expected_evidence):
        ev_prefix = f"{prefix} expected_evidence[{i}]"

        if not isinstance(ev, dict):
            errors.append(f"{ev_prefix}: must be an object, got {type(ev).__name__}")
            continue

        if "document_id" not in ev:
            errors.append(f"{ev_prefix}: missing 'document_id'")
        elif not isinstance(ev["document_id"], str):
            errors.append(f"{ev_prefix}: 'document_id' must be a string")

        if "chunk_index" not in ev:
            errors.append(f"{ev_prefix}: missing 'chunk_index'")
        elif not isinstance(ev["chunk_index"], int) or isinstance(
            ev["chunk_index"], bool
        ):
            errors.append(f"{ev_prefix}: 'chunk_index' must be an integer")

        # Validate against the committed corpus.
        if "document_id" in ev and "chunk_index" in ev:
            ref = (ev["document_id"], ev["chunk_index"])
            if ref not in VALID_EVIDENCE_REFS:
                errors.append(
                    f"{ev_prefix}: reference {ref!r} does not exist in the "
                    "committed synthetic corpus (evals/datasets/corpus.py)"
                )

    return errors


def load_and_validate(path: Path) -> list[GoldenRecord]:
    """
    Load and fully validate the golden JSONL dataset at *path*.

    Raises:
        DatasetValidationError: on any schema or corpus-reference violation.
        FileNotFoundError: if *path* does not exist.

    Returns:
        A list of validated records.
    """
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    records: list[GoldenRecord] = []
    all_errors: list[str] = []
    seen_ids: dict[str, int] = {}  # id -> first line number

    with path.open(encoding="utf-8") as fh:
        for line_number, raw_line in enumerate(fh, start=1):
            raw_line = raw_line.strip()
            if not raw_line:
                continue  # skip blank lines

            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                all_errors.append(f"Line {line_number}: invalid JSON — {exc}")
                continue

            if not isinstance(record, dict):
                all_errors.append(
                    f"Line {line_number}: each line must be a JSON object"
                )
                continue

            # Duplicate ID check.
            record_id = record.get("id")
            if isinstance(record_id, str):
                if record_id in seen_ids:
                    all_errors.append(
                        f"Line {line_number}: duplicate id {record_id!r} "
                        f"(first seen at line {seen_ids[record_id]})"
                    )
                else:
                    seen_ids[record_id] = line_number

            line_errors = validate_record(record, line_number)
            all_errors.extend(line_errors)

            if not line_errors:
                records.append(record)

    if all_errors:
        detail = "\n  ".join(all_errors)
        raise DatasetValidationError(
            f"Dataset validation failed with {len(all_errors)} error(s):\n  {detail}"
        )

    if not records:
        raise DatasetValidationError("Dataset is empty (no valid records found)")

    return records
