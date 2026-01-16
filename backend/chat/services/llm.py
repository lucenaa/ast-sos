"""
Serviço de LLM usando Google Gemini.
"""
import time
from typing import Optional
from django.conf import settings

try:
    import google.generativeai as genai
except ImportError:
    genai = None


SYSTEM_PROMPT = """Você é uma tutora da SOS Educação (Taís e Roberta). Responda apenas sobre ALFABETIZAÇÃO.
Baseie-se estritamente no contexto fornecido. Se não houver contexto suficiente, diga que não sabe e sugira perguntar de outra forma.
Seja prática, clara e breve; cite estratégias, exemplos e passos quando adequado.
Seja atenciosa e acolhedora. Estruture em tópicos quando for útil e ofereça próxima etapa ao final."""


class LLMService:
    """Serviço para geração de respostas usando Gemini."""
    
    def __init__(self):
        self.model = None
        self.model_name = settings.CHAT_MODEL
        self._initialize_model()
    
    def _initialize_model(self):
        """Inicializa o modelo Gemini se a API key estiver configurada."""
        if genai is None:
            print("[LLMService] google-generativeai não instalado")
            return
        
        api_key = settings.GOOGLE_API_KEY
        if not api_key:
            print("[LLMService] GOOGLE_API_KEY não configurada")
            return
        
        try:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(
                model_name=self.model_name,
                system_instruction=SYSTEM_PROMPT
            )
            print(f"[LLMService] Modelo {self.model_name} inicializado")
        except Exception as e:
            print(f"[LLMService] Erro ao inicializar modelo: {e}")
    
    def generate_response(self, question: str, context: str) -> tuple[str, int]:
        """
        Gera resposta para a pergunta usando o contexto fornecido.
        
        Returns:
            Tuple com (resposta ou mensagem de erro, tempo em ms)
        """
        if self.model is None:
            return "[Erro] Modelo de chat não configurado. Verifique GOOGLE_API_KEY.", 0
        
        prompt = f"Pergunta: {question}\n\nContexto:\n{context}"
        
        start = time.time()
        try:
            response = self.model.generate_content(prompt)
            answer = response.text
            duration_ms = int((time.time() - start) * 1000)
            return answer, duration_ms
            
        except Exception as e:
            print(f"[LLMService] Erro ao gerar resposta: {e}")
            duration_ms = int((time.time() - start) * 1000)
            return f"[Erro] Falha ao gerar resposta: {str(e)}", duration_ms
    
    @property
    def is_available(self) -> bool:
        """Verifica se o serviço está disponível."""
        return self.model is not None


# Instância global singleton
llm_service = LLMService()
