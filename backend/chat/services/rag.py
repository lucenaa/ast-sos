"""
Serviço de RAG (Retrieval-Augmented Generation).
"""
import os
import time
import ast
import json
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
from django.conf import settings

from .embeddings import embedding_service
from .llm import llm_service


class RAGService:
    """Serviço de RAG para recuperação de contexto e geração de respostas."""
    
    def __init__(self):
        self.df: Optional[pd.DataFrame] = None
        self.embeddings_matrix: Optional[np.ndarray] = None
        self.valid_indices: list[int] = []
        self.cols_map: dict = {}
        self._data_loaded = False
        
        # Carrega dados na inicialização
        self._load_data()
    
    def _get_data_path(self) -> Path:
        """Retorna o caminho para os arquivos de dados."""
        return Path(__file__).parent.parent / "data"
    
    def _parse_embedding_cell(self, cell) -> Optional[list[float]]:
        """Converte célula do CSV para lista de floats."""
        if cell is None or (isinstance(cell, float) and np.isnan(cell)):
            return None
        if isinstance(cell, list):
            try:
                return [float(x) for x in cell]
            except Exception:
                return None
        s = str(cell).strip()
        if not s:
            return None
        try:
            parsed = json.loads(s)
            if isinstance(parsed, list):
                return [float(x) for x in parsed]
        except Exception:
            pass
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, list):
                return [float(x) for x in parsed]
        except Exception:
            return None
        return None
    
    def _load_data(self):
        """Carrega CSV e embeddings (ou matriz .npy se existir)."""
        data_path = self._get_data_path()
        csv_path = data_path / "documents_rows.csv"
        npy_path = data_path / "embeddings.npy"
        
        if not csv_path.exists():
            print(f"[RAGService] CSV não encontrado em {csv_path}")
            return
        
        print(f"[RAGService] Carregando CSV de {csv_path}")
        try:
            self.df = pd.read_csv(csv_path, dtype=str, low_memory=False)
            self.cols_map = {c.lower().strip(): c for c in self.df.columns}
            print(f"[RAGService] CSV carregado: {len(self.df)} linhas")
        except Exception as e:
            print(f"[RAGService] Erro ao carregar CSV: {e}")
            return
        
        # Tenta carregar matriz .npy pré-computada
        if npy_path.exists():
            print(f"[RAGService] Carregando embeddings de {npy_path}")
            try:
                self.embeddings_matrix = np.load(npy_path)
                self.valid_indices = list(range(len(self.embeddings_matrix)))
                
                # Normaliza se necessário
                norms = np.linalg.norm(self.embeddings_matrix, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                self.embeddings_matrix = self.embeddings_matrix / norms
                
                print(f"[RAGService] Embeddings carregados: {self.embeddings_matrix.shape}")
                self._data_loaded = True
                return
            except Exception as e:
                print(f"[RAGService] Erro ao carregar .npy: {e}")
        
        # Fallback: parsear embeddings do CSV
        print("[RAGService] Parseando embeddings do CSV...")
        embedding_col = None
        for key in ["embedding", "embeddings", "vector"]:
            if key in self.cols_map:
                embedding_col = self.cols_map[key]
                break
        
        if embedding_col is None or embedding_col not in self.df.columns:
            print("[RAGService] Coluna de embedding não encontrada")
            self._data_loaded = True  # Dados carregados, mas sem embeddings
            return
        
        try:
            parsed = [self._parse_embedding_cell(x) for x in self.df[embedding_col].tolist()]
            self.valid_indices = [i for i, v in enumerate(parsed) if isinstance(v, list) and len(v) > 0]
            
            if not self.valid_indices:
                print("[RAGService] Nenhum embedding válido encontrado")
                self._data_loaded = True
                return
            
            emb_list = [parsed[i] for i in self.valid_indices]
            self.embeddings_matrix = np.array(emb_list, dtype=np.float32)
            
            # Normaliza
            norms = np.linalg.norm(self.embeddings_matrix, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            self.embeddings_matrix = self.embeddings_matrix / norms
            
            print(f"[RAGService] Embeddings parseados: {self.embeddings_matrix.shape}")
            self._data_loaded = True
            
        except Exception as e:
            print(f"[RAGService] Erro ao parsear embeddings: {e}")
            self._data_loaded = True
    
    def _top_k_by_similarity(self, query_vec: np.ndarray, k: int) -> list[tuple[int, float]]:
        """Retorna os top-k índices mais similares com suas similaridades."""
        if self.embeddings_matrix is None:
            return []
        
        sims = self.embeddings_matrix @ query_vec
        
        if k >= len(sims):
            indices = np.argsort(-sims)
        else:
            indices = np.argpartition(-sims, k)[:k]
            indices = indices[np.argsort(-sims[indices])]
        
        return [(int(idx), float(sims[idx])) for idx in indices]
    
    def _build_context(self, row_indices: list[int], max_chars: int) -> tuple[str, list[dict]]:
        """Monta o contexto concatenando conteúdos até max_chars."""
        if self.df is None:
            return "", []
        
        content_col = self.cols_map.get("content", "content")
        video_col = self.cols_map.get("video_id", self.cols_map.get("video", "video_id"))
        chunk_col = self.cols_map.get("chunk_id", self.cols_map.get("chunk", "chunk_id"))
        meta_col = self.cols_map.get("metadata", self.cols_map.get("meta", "metadata"))
        
        pieces: list[str] = []
        sources: list[dict] = []
        total = 0
        
        for i in row_indices:
            try:
                row = self.df.iloc[i]
            except Exception:
                continue
            
            content = str(row.get(content_col, "")).strip()
            if not content:
                continue
            
            piece = content
            if total + len(piece) > max_chars:
                piece = piece[:max(0, max_chars - total)]
            
            if piece:
                pieces.append(piece)
                total += len(piece)
                sources.append({
                    "video_id": str(row.get(video_col, "")),
                    "chunk_id": str(row.get(chunk_col, "")),
                    "metadata": str(row.get(meta_col, "")),
                })
            
            if total >= max_chars:
                break
        
        return "\n\n---\n\n".join(pieces), sources
    
    def process_query(self, question: str) -> dict:
        """
        Processa uma query completa: embedding -> retrieval -> geração.
        
        Returns:
            Dict com answer, sources e logs
        """
        logs = []
        top_k = settings.RAG_TOP_K
        max_chars = settings.RAG_MAX_CONTEXT_CHARS
        
        # Step 1: Gerar embedding da query
        query_vec, embed_ms = embedding_service.get_embedding(question)
        logs.append({
            "step": "embed_query",
            "duration_ms": embed_ms,
            "details": {
                "model": settings.EMBEDDING_MODEL,
                "ok": query_vec is not None
            }
        })
        
        # Step 2: Recuperar top-k
        row_indices = []
        topk_details = []
        
        if query_vec is not None and self.embeddings_matrix is not None:
            start = time.time()
            top_results = self._top_k_by_similarity(query_vec, top_k)
            row_indices = [self.valid_indices[idx] for idx, _ in top_results]
            
            for rank, (mat_idx, sim) in enumerate(top_results, start=1):
                df_idx = self.valid_indices[mat_idx]
                row = self.df.iloc[df_idx]
                topk_details.append({
                    "rank": rank,
                    "df_index": df_idx,
                    "similarity": round(sim, 4),
                    "video_id": str(row.get(self.cols_map.get("video_id", "video_id"), "")),
                    "chunk_id": str(row.get(self.cols_map.get("chunk_id", "chunk_id"), "")),
                })
            
            retrieve_ms = int((time.time() - start) * 1000)
            logs.append({
                "step": "retrieve_topk",
                "duration_ms": retrieve_ms,
                "details": {"top_k": top_k, "results": topk_details}
            })
        else:
            # Fallback: busca por texto
            start = time.time()
            if self.df is not None:
                tokens = [t.lower() for t in question.split() if t]
                content_col = self.cols_map.get("content", "content")
                scores = []
                for i, text in enumerate(self.df[content_col].fillna("").astype(str).tolist()):
                    t = text.lower()
                    score = sum(t.count(tok) for tok in tokens)
                    if score > 0:
                        scores.append((score, i))
                scores.sort(key=lambda x: x[0], reverse=True)
                row_indices = [i for _, i in scores[:top_k]]
            
            retrieve_ms = int((time.time() - start) * 1000)
            logs.append({
                "step": "text_fallback",
                "duration_ms": retrieve_ms,
                "details": {"top_k": top_k, "num_matches": len(row_indices)}
            })
        
        # Step 3: Construir contexto
        start = time.time()
        context_text, sources = self._build_context(row_indices, max_chars)
        context_ms = int((time.time() - start) * 1000)
        logs.append({
            "step": "build_context",
            "duration_ms": context_ms,
            "details": {"chars": len(context_text), "num_sources": len(sources)}
        })
        
        # Step 4: Gerar resposta
        answer, gen_ms = llm_service.generate_response(question, context_text)
        logs.append({
            "step": "generate",
            "duration_ms": gen_ms,
            "details": {"model": settings.CHAT_MODEL, "answer_length": len(answer)}
        })
        
        # Adiciona similaridade às sources
        sources_with_sim = []
        for i, src in enumerate(sources):
            sim = topk_details[i]["similarity"] if i < len(topk_details) else 0.0
            sources_with_sim.append({
                "video_id": src["video_id"],
                "chunk_id": src["chunk_id"],
                "similarity": sim,
                "metadata": src["metadata"]
            })
        
        return {
            "answer": answer,
            "sources": sources_with_sim,
            "logs": logs
        }
    
    @property
    def is_loaded(self) -> bool:
        """Verifica se os dados foram carregados."""
        return self._data_loaded
    
    @property
    def embeddings_count(self) -> int:
        """Retorna a quantidade de embeddings carregados."""
        if self.embeddings_matrix is None:
            return 0
        return len(self.embeddings_matrix)


# Instância global singleton
rag_service = RAGService()
