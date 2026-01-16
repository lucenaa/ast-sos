"""
Pydantic schemas para a API de chat.
"""
from typing import Optional
from ninja import Schema


class ChatRequest(Schema):
    """Request para enviar mensagem ao chat."""
    message: str


class Source(Schema):
    """Fonte/referência de onde veio a informação."""
    video_id: str
    chunk_id: str
    similarity: float
    metadata: Optional[str] = None


class LogStep(Schema):
    """Passo de log para debug/transparência."""
    step: str
    duration_ms: int
    details: dict = {}


class ChatResponse(Schema):
    """Response completa do chat com resposta, fontes e logs."""
    answer: str
    sources: list[Source]
    logs: list[LogStep]


class HealthResponse(Schema):
    """Response do endpoint de health check."""
    status: str
    message: str
    data_loaded: bool
    embeddings_count: int
