import type { ChatRequest, ChatResponse } from '../types';

// Em produção (Netlify), as chamadas /api/* são redirecionadas via netlify.toml
// Em desenvolvimento, usa localhost:8000
const API_URL = import.meta.env.PROD ? '' : (import.meta.env.VITE_API_URL || 'http://localhost:8000');

export async function sendMessage(message: string): Promise<ChatResponse> {
  const payload: ChatRequest = { message };

  const response = await fetch(`${API_URL}/api/chat/`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    throw new Error(`Erro na API: ${response.status} ${response.statusText}`);
  }

  return response.json();
}

export async function checkHealth(): Promise<{ status: string; data_loaded: boolean; embeddings_count: number }> {
  const response = await fetch(`${API_URL}/api/chat/health`);
  
  if (!response.ok) {
    throw new Error(`Erro no health check: ${response.status}`);
  }

  return response.json();
}
