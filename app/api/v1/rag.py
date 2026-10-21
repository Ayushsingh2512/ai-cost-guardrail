from decimal import Decimal
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.api.dependencies import enforce_rag_guardrails, get_current_user
from app.rag.accounting import RAGBudget
from app.rag.embeddings import (
    EmbeddingError,
    EmbeddingRateLimitError,
    GeminiEmbedder,
)
from app.rag.generation import (
    GenerationError,
    GenerationRateLimitError,
    GenerationTimeoutError,
    generation_service,
)
from app.rag.pipeline import RAGService
from app.rag.retriever import Retriever
from app.schemas.rag import RAGCitation, RAGRequest, RAGResponse
from app.services.database import get_db
from app.services.models import Tenant
from app.services.usage import BudgetExceededError

router = APIRouter(
    prefix="/api/v1",
    tags=["RAG"],
)


def get_rag_service(
    db: Session = Depends(get_db),
) -> RAGService:
    embedder = GeminiEmbedder()
    retriever = Retriever(db=db)

    return RAGService(
        embedder=embedder,
        retriever=retriever,
        generator=generation_service,
    )


@router.post(
    "/rag/query",
    response_model=RAGResponse,
)
async def rag_query(
    request: RAGRequest = Depends(enforce_rag_guardrails),
    current_user: dict = Depends(get_current_user),
    rag_service: RAGService = Depends(get_rag_service),
    db: Session = Depends(get_db),
):
    tenant_id = int(current_user["tenant_id"])
    user_id = int(current_user["user_id"])

    tenant = db.query(Tenant).filter(Tenant.id == tenant_id).first()

    if tenant is None:
        raise HTTPException(
            status_code=404,
            detail=f"Tenant {tenant_id} not found",
        )

    request_id = str(uuid4())
    budget = RAGBudget(
        db=db,
        tenant_id=tenant_id,
        user_id=user_id,
        request_id=request_id,
    )

    try:
        result = await rag_service.answer(
            tenant_id=tenant_id,
            query=request.query,
            top_k=request.top_k,
            budget=budget,
        )

    except BudgetExceededError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc
    except (EmbeddingRateLimitError, GenerationRateLimitError) as exc:
        raise HTTPException(
            status_code=429,
            detail=str(exc),
        ) from exc
    except GenerationTimeoutError as exc:
        raise HTTPException(
            status_code=504,
            detail=str(exc),
        ) from exc
    except (EmbeddingError, GenerationError) as exc:
        raise HTTPException(
            status_code=502,
            detail=str(exc),
        ) from exc

    return RAGResponse(
        status=result.generation.status,
        answer=result.generation.answer,
        citations=[
            RAGCitation(
                ref=citation.ref,
                source=citation.source,
                page=citation.page,
            )
            for citation in result.generation.citations
        ],
        top_k=request.top_k,
        retrieved_count=len(result.retrieved),
        timings_ms=result.timings_ms,
        embedding_tokens=result.embedding_tokens,
        request_id=getattr(result, "request_id", None) or request_id,
        total_cost=getattr(result, "total_cost", Decimal(0)),
    )
