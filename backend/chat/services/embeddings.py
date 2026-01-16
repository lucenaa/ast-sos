"""
Serviço de embeddings usando OpenAI.
"""
import time
from typing import Optional
import numpy as np
from django.conf import settings

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


class EmbeddingService:
    """Serviço para gerar embeddings de queries usando OpenAI."""
    
    def __init__(self):
        self.client: Optional[OpenAI] = None
        self.model = settings.EMBEDDING_MODEL
        self._initialize_client()
    
    def _initialize_client(self):
        """Inicializa o cliente OpenAI se a API key estiver configurada."""
        if OpenAI is None:
            print("[EmbeddingService] OpenAI não instalado")
            return
        
        api_key = settings.OPENAI_API_KEY
        if not api_key:
            print("[EmbeddingService] OPENAI_API_KEY não configurada")
            return
        
        try:
            self.client = OpenAI(api_key=api_key)
            print(f"[EmbeddingService] Cliente OpenAI inicializado com modelo {self.model}")
        except Exception as e:
            print(f"[EmbeddingService] Erro ao inicializar cliente: {e}")
    
    def get_embedding(self, text: str) -> tuple[Optional[np.ndarray], int]:
        """
        Gera embedding para o texto usando OpenAI.
        
        Returns:
            Tuple com (embedding normalizado ou None, tempo em ms)
        """
        if self.client is None:
            return None, 0
        
        start = time.time()
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=text
            )
            vec = np.array(response.data[0].embedding, dtype=np.float32)
            
            # Normaliza para similaridade coseno via produto escalar
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            
            duration_ms = int((time.time() - start) * 1000)
            return vec, duration_ms
            
        except Exception as e:
            print(f"[EmbeddingService] Erro ao gerar embedding: {e}")
            duration_ms = int((time.time() - start) * 1000)
            return None, duration_ms
    
    @property
    def is_available(self) -> bool:
        """Verifica se o serviço está disponível."""
        return self.client is not None


# Instância global singleton
embedding_service = EmbeddingService()
