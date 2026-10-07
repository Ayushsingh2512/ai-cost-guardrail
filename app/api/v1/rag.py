from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.api.dependencies import enforce_rag_guardrails, get_current_user
from app.rag.embeddings import GeminiEmbedder
from app.rag.generation import GenerationService, generation_service
from app.rag.pipeline import RAGService
from app.rag.retriever import Retriever
from app.schemas.rag import RAGRequest, RAGResponse
from app.services.database import get_db
from app.services.models import Tenant


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

    tenant = (
        db.query(Tenant)
        .filter(Tenant.id == tenant_id)
        .first()
    )

    if tenant is None:
        raise HTTPException(
            status_code=404,
            detail=f"Tenant {tenant_id} not found",
        )

    try:
        result = await rag_service.answer(
            tenant_id=tenant_id,
            query=request.query,
            top_k=request.top_k,
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    return RAGResponse(
        status=result.generation.status,
        answer=result.generation.answer,
        citations=[
            {
                "ref": citation.ref,
                "source": citation.source,
                "page": citation.page,
            }
            for citation in result.generation.citations
        ],
        top_k=request.top_k,
        retrieved_count=len(result.retrieved),
        timings_ms=result.timings_ms,
        embedding_tokens=result.embedding_tokens,
    )