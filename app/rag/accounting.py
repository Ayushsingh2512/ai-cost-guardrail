"""
RAG budget accounting.

RAGBudget manages the reserve-and-settle lifecycle for the two provider
operations that make up a single RAG request (embedding + generation).
It follows the same pattern as app/api/v1/chat.py:

  reserve(...) -> db.commit()
  <provider call>
  settle_success(...) -> db.commit()
  # on any failure:
  settle_failure(...) -> db.commit()

It intentionally does not know about FastAPI, HTTP, or routing.
"""

from __future__ import annotations

import logging
from decimal import Decimal
from typing import TypeGuard

from sqlalchemy.orm import Session

from app.rag.generation import GenerationProviderError, ProviderMeta
from app.services.cost_engine import cost_engine
from app.services.models import UsageRecord
from app.services.usage import usage_service

logger = logging.getLogger(__name__)

_EMBED_MODEL = "gemini-embedding-001"


class RAGBudget:
    """
    Owns the usage-accounting state for one RAG HTTP request.

    Lifecycle
    ---------
    1. reserve_embedding(query_tokens) -> reserves embed cost; commits.
    2. settle_embedding(tokens)        -> settles actual embed cost; commits.
    3. reserve_generation(...)         -> reserves generation cost; commits.
    4. settle_generation(provider_meta)-> settles actual generation cost; commits.
    5. release_all()                   -> settle_failure for any still-reserved
                                         records; called on every exception path;
                                         NEVER raises.

    Properties
    ----------
    embed_record    : the UsageRecord for the embedding operation (or None).
    generate_record : the UsageRecord for the generation operation (or None).
    total_cost      : sum of actual_cost across settled records (Decimal).
    """

    def __init__(
        self,
        db: Session,
        tenant_id: int,
        user_id: int,
        request_id: str,
    ) -> None:
        self.db = db
        self.tenant_id = tenant_id
        self.user_id = user_id
        self.request_id = request_id

        self._embed_record: UsageRecord | None = None
        self._generate_record: UsageRecord | None = None
        self._preserve_generate_reservation: bool = False

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def embed_record(self) -> UsageRecord | None:
        return self._embed_record

    @property
    def generate_record(self) -> UsageRecord | None:
        return self._generate_record

    @property
    def is_generation_preserved(self) -> bool:
        """True if the generation reservation is preserved due to incomplete usage."""
        return self._preserve_generate_reservation

    def preserve_generation_reservation(self) -> None:
        """Mark the generation reservation to be preserved rather than released on failure."""
        self._preserve_generate_reservation = True

    @staticmethod
    def _is_valid_token_count(val: object) -> TypeGuard[int]:
        """Accept only non-Boolean, non-negative integers."""
        return isinstance(val, int) and not isinstance(val, bool) and val >= 0

    @property
    def total_cost(self) -> Decimal:
        """Sum of actual_cost for all settled records."""
        total = Decimal(0)
        for record in (self._embed_record, self._generate_record):
            if record is not None and record.status == "completed":
                total += Decimal(str(record.actual_cost))
        return total

    # ------------------------------------------------------------------
    # Embedding phase
    # ------------------------------------------------------------------

    def reserve_embedding(self, query_tokens: int) -> None:
        """
        Reserve the embedding budget based on the token estimate.
        Raises BudgetExceededError (subclass of ValueError) if the tenant
        would exceed their monthly budget.
        Commits on success.
        """
        reserved_cost = cost_engine.estimate_embedding_cost(
            model=_EMBED_MODEL,
            input_tokens=query_tokens,
        )

        try:
            self._embed_record = usage_service.reserve_budget(
                db=self.db,
                tenant_id=self.tenant_id,
                request_id=self.request_id,
                model=_EMBED_MODEL,
                user_id=self.user_id,
                reserved_cost=reserved_cost,
                operation="embed",
            )
            self.db.commit()

        except Exception:
            self.db.rollback()
            raise

    def settle_embedding(self, tokens: int) -> None:
        """
        Settle the embedding reservation with the actual token count.
        On DB failure: rollback, attempt settle_failure in a fresh
        transaction (mirrors the chat.py error-recovery pattern).
        """
        assert self._embed_record is not None, "reserve_embedding must be called first"

        if self._embed_record.status != "reserved":
            raise RuntimeError(
                f"Embedding reservation is in status {self._embed_record.status!r}; "
                "cannot settle"
            )

        actual_cost = cost_engine.calculate_embedding_cost(
            model=_EMBED_MODEL,
            input_tokens=tokens,
        )

        try:
            usage_service.settle_success(
                db=self.db,
                usage=self._embed_record,
                actual_cost=actual_cost,
                input_tokens=tokens,
                thinking_tokens=0,
                output_tokens=0,
                total_tokens=tokens,
            )
            self.db.commit()

        except Exception:
            self.db.rollback()
            self._try_release(self._embed_record)
            raise

    # ------------------------------------------------------------------
    # Generation phase
    # ------------------------------------------------------------------

    def reserve_generation(
        self,
        prompt_tokens: int,
        max_output_tokens: int,
        model: str,
    ) -> None:
        """
        Reserve the generation budget.
        prompt_tokens comes from count_prompt_tokens() — the estimate that
        includes SYSTEM_INSTRUCTION.
        Raises BudgetExceededError if the tenant would exceed their budget.
        Commits on success.
        """
        reserved_cost = cost_engine.estimate_reservation(
            model=model,
            input_tokens=prompt_tokens,
            max_output_tokens=max_output_tokens,
        )

        try:
            self._generate_record = usage_service.reserve_budget(
                db=self.db,
                tenant_id=self.tenant_id,
                request_id=self.request_id,
                model=model,
                user_id=self.user_id,
                reserved_cost=reserved_cost,
                operation="generate",
            )
            self.db.commit()

        except Exception:
            self.db.rollback()
            raise

    def settle_generation(self, provider_meta: ProviderMeta | None) -> None:
        """
        Settle the generation reservation from provider-reported usage.

        Uses prompt_tokens, thinking_tokens, completion_tokens, and
        total_tokens from ProviderMeta.

        Validates required usage fields before normal generation settlement:
        prompt_tokens and completion_tokens must be non-Boolean, non-negative integers.
        Incomplete or missing usage metadata fails closed: preserves the reservation
        to protect the tenant budget and raises GenerationProviderError.

        On DB failure: rollback, attempt settle_failure in a fresh
        transaction (mirrors the chat.py error-recovery pattern).
        """
        assert self._generate_record is not None, (
            "reserve_generation must be called first"
        )

        if self._generate_record.status != "reserved":
            raise RuntimeError(
                f"Generation reservation is in status {self._generate_record.status!r}; "
                "cannot settle"
            )

        if provider_meta is None:
            self._preserve_generate_reservation = True
            raise GenerationProviderError(
                "Generation provider returned no usage metadata; "
                "reservation retained to protect tenant budget"
            )

        if not self._is_valid_token_count(provider_meta.prompt_tokens):
            self._preserve_generate_reservation = True
            raise GenerationProviderError(
                f"Generation provider returned invalid or missing prompt_tokens: "
                f"{provider_meta.prompt_tokens!r}; reservation retained to protect tenant budget"
            )

        if not self._is_valid_token_count(provider_meta.completion_tokens):
            self._preserve_generate_reservation = True
            raise GenerationProviderError(
                f"Generation provider returned invalid or missing completion_tokens: "
                f"{provider_meta.completion_tokens!r}; reservation retained to protect tenant budget"
            )

        if provider_meta.thinking_tokens is not None and not self._is_valid_token_count(
            provider_meta.thinking_tokens
        ):
            self._preserve_generate_reservation = True
            raise GenerationProviderError(
                f"Generation provider returned invalid thinking_tokens: "
                f"{provider_meta.thinking_tokens!r}; reservation retained to protect tenant budget"
            )

        if provider_meta.total_tokens is not None and not self._is_valid_token_count(
            provider_meta.total_tokens
        ):
            self._preserve_generate_reservation = True
            raise GenerationProviderError(
                f"Generation provider returned invalid total_tokens: "
                f"{provider_meta.total_tokens!r}; reservation retained to protect tenant budget"
            )

        input_tokens: int = provider_meta.prompt_tokens
        thinking_tokens: int = provider_meta.thinking_tokens or 0
        output_tokens: int = provider_meta.completion_tokens
        total_tokens: int = (
            provider_meta.total_tokens
            if provider_meta.total_tokens is not None
            else input_tokens + thinking_tokens + output_tokens
        )

        model = self._generate_record.model

        actual_cost = cost_engine.calculate_actual_cost(
            model=model,
            input_tokens=input_tokens,
            thinking_tokens=thinking_tokens,
            output_tokens=output_tokens,
        )

        try:
            usage_service.settle_success(
                db=self.db,
                usage=self._generate_record,
                actual_cost=actual_cost,
                input_tokens=input_tokens,
                thinking_tokens=thinking_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
            )
            self.db.commit()

        except Exception:
            self.db.rollback()
            self._try_release(self._generate_record)
            raise

    # ------------------------------------------------------------------
    # Release (exception path)
    # ------------------------------------------------------------------

    def release_all(self) -> None:
        """
        Release every record that is still in 'reserved' status,
        except when a reservation is explicitly preserved to protect
        the tenant budget on incomplete provider usage.
        Called on any exception path. NEVER raises — logs instead.
        """
        for record in (self._embed_record, self._generate_record):
            if record is None or record.status != "reserved":
                continue
            if record is self._generate_record and self._preserve_generate_reservation:
                continue
            self._try_release(record)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _try_release(self, record: UsageRecord) -> None:
        """
        Best-effort settle_failure in a fresh transaction.
        Logs on failure; never raises.
        """
        try:
            usage_service.settle_failure(db=self.db, usage=record)
            self.db.commit()
        except Exception:
            try:
                self.db.rollback()
            except Exception:
                logger.warning("Rollback failed during release", exc_info=True)
            logger.exception(
                "Failed to release reservation for request_id=%s operation=%s",
                record.request_id,
                record.operation,
            )
