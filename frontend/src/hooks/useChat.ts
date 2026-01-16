import { useState, useCallback } from 'react';
import type { Message } from '../types';
import { sendMessage } from '../services/api';

const WELCOME_MESSAGE: Message = {
  id: 'welcome',
  role: 'assistant',
  content: `Olá! 👋 Eu sou sua assistente de alfabetização da **SOS Educação**.

Posso ajudar gestores e professores a tirar dúvidas sobre planejamento, estratégias e intervenções na alfabetização.

**Para obter respostas de qualidade:**
- Faça perguntas claras (ex.: "Como trabalhar consciência fonológica no 1º ano?")
- Se quiser, diga a série/ano e o objetivo da aula
- Posso sugerir atividades, sequências didáticas e orientações para famílias

*Importante: respondo apenas sobre ALFABETIZAÇÃO e uso as transcrições da SOS Educação como base.*`,
  timestamp: new Date(),
};

function generateId(): string {
  return `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
}

export function useChat() {
  const [messages, setMessages] = useState<Message[]>([WELCOME_MESSAGE]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const sendUserMessage = useCallback(async (text: string) => {
    if (!text.trim() || isLoading) return;

    const userMessage: Message = {
      id: generateId(),
      role: 'user',
      content: text.trim(),
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setIsLoading(true);
    setError(null);

    try {
      const response = await sendMessage(text.trim());

      const assistantMessage: Message = {
        id: generateId(),
        role: 'assistant',
        content: response.answer,
        sources: response.sources,
        logs: response.logs,
        timestamp: new Date(),
      };

      setMessages((prev) => [...prev, assistantMessage]);
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Erro desconhecido';
      setError(errorMessage);

      const errorAssistantMessage: Message = {
        id: generateId(),
        role: 'assistant',
        content: `Desculpe, ocorreu um erro ao processar sua mensagem. Por favor, tente novamente.\n\n*Erro: ${errorMessage}*`,
        timestamp: new Date(),
      };

      setMessages((prev) => [...prev, errorAssistantMessage]);
    } finally {
      setIsLoading(false);
    }
  }, [isLoading]);

  const clearMessages = useCallback(() => {
    setMessages([WELCOME_MESSAGE]);
    setError(null);
  }, []);

  return {
    messages,
    isLoading,
    error,
    sendMessage: sendUserMessage,
    clearMessages,
  };
}
