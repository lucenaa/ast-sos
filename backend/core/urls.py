"""
URL configuration for SOS Educação Chat API.
"""
from django.urls import path
from ninja import NinjaAPI
from chat.api import router as chat_router

api = NinjaAPI(
    title="SOS Educação Chat API",
    version="1.0.0",
    description="API de chat para alfabetização com RAG baseado em transcrições da SOS Educação"
)

api.add_router("/chat", chat_router)

urlpatterns = [
    path("api/", api.urls),
]
