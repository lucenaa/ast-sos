export interface Source {
  video_id: string;
  chunk_id: string;
  similarity: number;
  metadata?: string;
}

export interface LogStep {
  step: string;
  duration_ms: number;
  details: Record<string, unknown>;
}

export interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  sources?: Source[];
  logs?: LogStep[];
  timestamp: Date;
}

export interface ChatResponse {
  answer: string;
  sources: Source[];
  logs: LogStep[];
}

export interface ChatRequest {
  message: string;
}
