"""
API endpoints do chat usando Django Ninja.
"""
from ninja import Router
from .schemas import ChatRequest, ChatResponse, HealthResponse, Source, LogStep
from .services.rag import rag_service

router = Router(tags=["Chat"])


@router.get("/health", response=HealthResponse)
def health_check(request):
    """
    Endpoint de health check para monitoramento.
    """
    return HealthResponse(
        status="ok" if rag_service.is_loaded else "degraded",
        message="Serviço funcionando" if rag_service.is_loaded else "Dados não carregados",
        data_loaded=rag_service.is_loaded,
        embeddings_count=rag_service.embeddings_count
    )


@router.post("/", response=ChatResponse)
def chat(request, payload: ChatRequest):
    """
    Endpoint principal de chat.
    
    Recebe uma mensagem e retorna a resposta do assistente junto com
    as fontes utilizadas e logs de processamento.
    """
    result = rag_service.process_query(payload.message)
    
    sources = [
        Source(
            video_id=s["video_id"],
            chunk_id=s["chunk_id"],
            similarity=s["similarity"],
            metadata=s.get("metadata")
        )
        for s in result["sources"]
    ]
    
    logs = [
        LogStep(
            step=log["step"],
            duration_ms=log["duration_ms"],
            details=log.get("details", {})
        )
        for log in result["logs"]
    ]
    
    return ChatResponse(
        answer=result["answer"],
        sources=sources,
        logs=logs
    )
