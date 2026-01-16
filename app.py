import os
import ast
import json
import time
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None  # type: ignore


# --------------
# Configuração
# --------------
load_dotenv(override=True)

DEFAULT_CSV_PATH = os.getenv("CSV_PATH", "documents_rows.csv")
DEFAULT_EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
DEFAULT_CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-5")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


# --------------
# Utilitários
# --------------
def parse_embedding_cell(cell: str) -> Optional[List[float]]:
    """Converte a célula do CSV (string) para lista de floats.

    Aceita formatos como "[0.1, 0.2, ...]" ou JSON equivalente.
    Retorna None se não conseguir parsear.
    """
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
        # Tenta JSON primeiro
        parsed = json.loads(s)
        if isinstance(parsed, list):
            return [float(x) for x in parsed]
    except Exception:
        pass
    try:
        # Fallback para literal_eval
        parsed = ast.literal_eval(s)
        if isinstance(parsed, list):
            return [float(x) for x in parsed]
    except Exception:
        return None
    return None


@st.cache_resource(show_spinner=False)
def load_dataframe(csv_path: str, max_rows: int) -> pd.DataFrame:
    """Carrega o CSV. Se max_rows > 0, carrega apenas as primeiras N linhas.

    Importante: usa dtype=str para evitar coerção de colunas e preservar o conteúdo.
    """
    if max_rows and max_rows > 0:
        df = pd.read_csv(csv_path, nrows=max_rows, dtype=str, low_memory=False)
    else:
        df = pd.read_csv(csv_path, dtype=str, low_memory=False)
    # Normaliza nomes de colunas esperados
    cols_map = {c.lower().strip(): c for c in df.columns}
    # Garante presence de 'content'
    if "content" not in cols_map:
        # tenta alternativas comuns
        for cand in ["texto", "text", "conteudo"]:
            if cand in cols_map:
                cols_map["content"] = cols_map[cand]
                break
    return df, cols_map


@st.cache_resource(show_spinner=False)
def build_embedding_matrix(df: pd.DataFrame, cols_map: dict) -> Tuple[Optional[np.ndarray], List[int]]:
    """Cria a matriz de embeddings (normalizada por linha) a partir do DF, se houver coluna embedding.

    Retorna (matrix_or_none, valid_row_indices). Se não houver embedding válido, retorna (None, []).
    """
    embedding_col = None
    for key in ["embedding", "embeddings", "vector"]:
        if key in cols_map:
            embedding_col = cols_map[key]
            break
    if embedding_col is None or embedding_col not in df.columns:
        return None, []

    parsed: List[Optional[List[float]]] = [parse_embedding_cell(x) for x in df[embedding_col].tolist()]
    valid_indices = [i for i, v in enumerate(parsed) if isinstance(v, list) and len(v) > 0]
    if not valid_indices:
        return None, []
    emb_list = [parsed[i] for i in valid_indices]
    # Cria matriz e normaliza para produto escalar = similaridade coseno
    mat = np.array(emb_list, dtype=np.float32)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    mat = mat / norms
    return mat, valid_indices


def ensure_openai_client() -> Optional[OpenAI]:
    if OpenAI is None:
        return None
    if not OPENAI_API_KEY:
        return None
    try:
        return OpenAI(api_key=OPENAI_API_KEY)
    except Exception:
        return None


def get_query_embedding(client: OpenAI, text: str, model: str) -> Optional[np.ndarray]:
    try:
        res = client.embeddings.create(model=model, input=text)
        vec = np.array(res.data[0].embedding, dtype=np.float32)
        norm = np.linalg.norm(vec)
        if norm == 0:
            return None
        return vec / norm
    except Exception as e:
        st.error(f"Falha ao gerar embedding da consulta: {e}")
        return None


def top_k_by_dot(matrix: np.ndarray, query_vec: np.ndarray, k: int) -> List[int]:
    sims = matrix @ query_vec
    if k >= len(sims):
        return list(np.argsort(-sims))
    idx = np.argpartition(-sims, k)[:k]
    # Ordena top-k pela similaridade
    idx_sorted = idx[np.argsort(-sims[idx])]
    return idx_sorted.tolist()


def build_context(df: pd.DataFrame, row_indices: List[int], cols_map: dict, max_chars: int) -> Tuple[str, List[dict]]:
    """Monta o contexto concatenando conteúdos até max_chars (para caber no prompt).
    Retorna o texto e a lista de metadados das fontes.
    """
    content_col = cols_map.get("content", "content")
    video_col = cols_map.get("video_id", cols_map.get("video", cols_map.get("titulo", "video_id")))
    chunk_col = cols_map.get("chunk_id", cols_map.get("chunk", "chunk_id"))
    meta_col = cols_map.get("metadata", cols_map.get("meta", "metadata"))

    pieces: List[str] = []
    sources: List[dict] = []
    total = 0
    for i in row_indices:
        try:
            row = df.iloc[i]
        except Exception:
            continue
        content = str(row.get(content_col, "")).strip()
        if not content:
            continue
        piece = content
        if total + len(piece) > max_chars:
            piece = piece[: max(0, max_chars - total)]
        if piece:
            pieces.append(piece)
            total += len(piece)
            sources.append(
                {
                    "video_id": str(row.get(video_col, "")),
                    "chunk_id": str(row.get(chunk_col, "")),
                    "metadata": str(row.get(meta_col, "")),
                }
            )
        if total >= max_chars:
            break
    return "\n\n---\n\n".join(pieces), sources


def call_chat(client: OpenAI, messages: List[dict], model: str) -> str:
    try:
        resp = client.chat.completions.create(model=model, messages=messages)
        return resp.choices[0].message.content or ""
    except Exception as e:
        st.error(f"Falha ao chamar o modelo de chat: {e}")
        return ""


# --------------
# UI Streamlit
# --------------
st.set_page_config(page_title="Chat SOS Educação - Alfabetização (RAG)", layout="wide")

# Estilos tipo WhatsApp
st.markdown(
    """
    <style>
    /* Container do chat para um look mais clean */
    .main .block-container { padding-top: 1rem; padding-bottom: 2rem; }

    /* Bolhas de conversa */
    .chat-bubble { border-radius: 16px; padding: 10px 14px; margin: 6px 0; max-width: 900px; line-height: 1.45; }
    .chat-bubble.user { background: #dcf8c6; color: #111; margin-left: auto; border-bottom-right-radius: 4px; }
    .chat-bubble.assistant { background: #ffffff; color: #111; border: 1px solid #e5e5e5; border-bottom-left-radius: 4px; }
    .chat-meta { font-size: 12px; color: #666; margin-top: 2px; }
    .welcome-card { border: 1px solid #e5e5e5; padding: 14px 16px; border-radius: 12px; background: #ffffff; }
    .sugg-container { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 6px; }
    .sugg-chip { background: #f0f0f0; border: 1px solid #e5e5e5; padding: 6px 10px; border-radius: 16px; cursor: pointer; }
    .sugg-chip:hover { background: #e8e8e8; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Chat SOS Educação - Alfabetização")
st.caption("Assistente para gestores e professores, com base nas transcrições da SOS Educação (Taís e Roberta)")

with st.sidebar:
    st.header("Configurações")
    csv_path = st.text_input("Caminho do CSV", value=DEFAULT_CSV_PATH)
    max_rows = st.number_input("Limite de linhas carregadas (0 = todas)", min_value=0, max_value=2_000_000, value=10000, step=1000)
    top_k = st.slider("Top-K contextos", min_value=1, max_value=10, value=5)
    max_ctx_chars = st.slider("Limite de caracteres do contexto", min_value=1000, max_value=20000, value=6000, step=500)

    embedding_model = st.selectbox(
        "Modelo de Embedding",
        options=["text-embedding-3-small", "text-embedding-3-large"],
        index=0 if DEFAULT_EMBEDDING_MODEL.endswith("small") else 1,
    )
    chat_model = st.text_input("Modelo de Chat", value=DEFAULT_CHAT_MODEL)

    st.markdown(
        """
        - É recomendado que a coluna `embedding` exista. Caso contrário, o app tentará usar apenas busca por texto (menos precisa).
        - Defina `OPENAI_API_KEY` no `.env` ou nas variáveis de ambiente.
        """
    )
    # Visualizador de logs da sessão
    if "all_logs" not in st.session_state:
        st.session_state.all_logs = []
    with st.expander("Logs da sessão (debug)"):
        if st.session_state.all_logs:
            for i, log in enumerate(reversed(st.session_state.all_logs[-10:])):
                st.write(f"Interação #{log.get('id', i+1)} - Pergunta: {log.get('question','')}")
                st.json(log)
        else:
            st.caption("Os logs aparecerão aqui após cada pergunta.")


if not os.path.exists(csv_path):
    st.error(f"CSV não encontrado em: {csv_path}")
    st.stop()

with st.spinner("Carregando CSV..."):
    df, cols_map = load_dataframe(csv_path, max_rows)

required_content_col = cols_map.get("content", "content")
if required_content_col not in df.columns:
    st.error("Coluna de conteúdo não encontrada. Esperado: 'content' (ou similar).")
    st.stop()

with st.spinner("Preparando índice de embeddings (se disponível)..."):
    matrix, valid_idx = build_embedding_matrix(df, cols_map)

if matrix is None:
    st.warning(
        "Nenhuma coluna de `embedding` válida encontrada. O app funcionará com menor precisão (busca por texto simples)."
    )

client = ensure_openai_client()
if client is None:
    st.warning("OPENAI_API_KEY não configurada. Você poderá pesquisar contextos, mas não conversar com o modelo.")


# Histórico de chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# Mensagem inicial acolhedora do assistente (se primeira visita)
if not st.session_state.messages:
    welcome = (
        "Oi! Eu sou sua assistente de alfabetização da SOS Educação. Posso ajudar gestores e professores a tirar dúvidas "
        "sobre planejamento, estratégias e intervenções na alfabetização. Para obter respostas de qualidade:\n\n"
        "- Faça perguntas claras (ex.: 'Como trabalhar consciência fonológica no 1º ano?').\n"
        "- Se quiser, diga a série/ano e o objetivo da aula.\n"
        "- Posso sugerir atividades, sequências didáticas e orientações para famílias.\n\n"
        "Importante: respondo apenas sobre ALFABETIZAÇÃO e uso as transcrições da SOS Educação como base."
    )
    st.session_state.messages.append({"role": "assistant", "content": welcome, "sources": []})

def render_message(msg: dict):
    role = msg.get("role", "assistant")
    content = msg.get("content", "")
    sources = msg.get("sources", [])
    # Usar avatar padrão do Streamlit para evitar erro com emojis como imagem
    with st.chat_message(role):
        # Bolha customizada
        bubble_role = "assistant" if role == "assistant" else "user"
        st.markdown(f"<div class='chat-bubble {bubble_role}'>{content}</div>", unsafe_allow_html=True)
        if role == "assistant" and sources:
            with st.expander("Ver referências (trechos recuperados)"):
                st.json(sources)

for msg in st.session_state.messages:
    render_message(msg)

user_input = st.chat_input("Escreva sua mensagem...")

# Sugestões rápidas (chips)
SUGGESTIONS = [
    "Como incentivar o gosto pela leitura?",
    "Ideias de atividades de consciência fonológica",
    "Sequência didática para o 1º ano",
    "Como envolver as famílias na alfabetização?",
    "Estratégias para alunos com dificuldade em correspondência grafema-fonema",
]

suggestion_clicked = None
cols = st.columns(min(3, len(SUGGESTIONS)))
for i, text in enumerate(SUGGESTIONS):
    col = cols[i % len(cols)]
    if col.button(text, key=f"sugg_{i}"):
        suggestion_clicked = text

if not user_input and suggestion_clicked:
    user_input = suggestion_clicked

if user_input:
    # Exibe imediatamente a mensagem do usuário (eco otimista)
    with st.chat_message("user"):
        st.markdown(f"<div class='chat-bubble user'>{user_input}</div>", unsafe_allow_html=True)

    # Inicializa log desta interação
    current_log = {"id": len(st.session_state.all_logs) + 1, "question": user_input, "steps": []}

    def log_step(name: str, info: dict):
        current_log["steps"].append({"step": name, **info})

    # Placeholder para resposta e status
    with st.chat_message("assistant"):
        status = st.status("Analisando sua pergunta...", expanded=True)
        status.write("Gerando embedding da consulta (se disponível)...")
        answer_placeholder = st.empty()
        answer_placeholder.markdown("<div class='chat-bubble assistant'>Digitando...</div>", unsafe_allow_html=True)

        # Recuperação
        start = time.time()
        row_indices: List[int] = []
        topk_details: List[dict] = []
        if matrix is not None and client is not None:
            qvec = get_query_embedding(client, user_input, embedding_model)
            embed_ms = int((time.time() - start) * 1000)
            log_step("embed_query", {"model": embedding_model, "duration_ms": embed_ms, "ok": qvec is not None})
            status.update(label="Recuperando trechos relevantes...", state="running")
            start = time.time()
            if qvec is not None:
                sims = matrix @ qvec
                idxs = top_k_by_dot(matrix, qvec, top_k)
                row_indices = [valid_idx[i] for i in idxs]
                for rank, i_m in enumerate(idxs, start=1):
                    df_i = valid_idx[i_m]
                    row = df.iloc[df_i]
                    topk_details.append(
                        {
                            "rank": rank,
                            "df_index": int(df_i),
                            "similarity": float(sims[i_m]),
                            "video_id": str(row.get(cols_map.get("video_id", "video_id"), "")),
                            "chunk_id": str(row.get(cols_map.get("chunk_id", "chunk_id"), "")),
                        }
                    )
            retrieve_ms = int((time.time() - start) * 1000)
            log_step("retrieve_topk", {"top_k": top_k, "duration_ms": retrieve_ms, "results": topk_details})
        else:
            status.update(label="Busca por texto (fallback)...", state="running")
            start = time.time()
            tokens = [t.lower() for t in user_input.split() if t]
            scores = []
            content_col = cols_map.get("content", "content")
            series = df[content_col].fillna("").astype(str)
            for i, text in enumerate(series.tolist()):
                t = text.lower()
                score = sum(t.count(tok) for tok in tokens)
                if score > 0:
                    scores.append((score, i))
            scores.sort(key=lambda x: x[0], reverse=True)
            row_indices = [i for _, i in scores[:top_k]]
            retrieve_ms = int((time.time() - start) * 1000)
            log_step("text_fallback", {"top_k": top_k, "duration_ms": retrieve_ms, "num_matches": len(row_indices)})

        context_text, sources = build_context(df, row_indices, cols_map, max_ctx_chars)
        log_step("build_context", {"chars": len(context_text), "num_sources": len(sources)})

        system_prompt = (
            "Você é uma tutora da SOS Educação (Taís e Roberta). Responda apenas sobre ALFABETIZAÇÃO. "
            "Baseie-se estritamente no contexto fornecido. Se não houver contexto suficiente, diga que não sabe e sugira perguntar de outra forma. "
            "Seja prática, clara e breve; cite estratégias, exemplos e passos quando adequado."
        )

        if client is not None:
            status.update(label="Redigindo resposta...", state="running")
            start = time.time()
            messages = [
                {"role": "system", "content": system_prompt + " Seja atenciosa e acolhedora. Estruture em tópicos quando for útil e ofereça próxima etapa ao final."},
                {
                    "role": "user",
                    "content": f"Pergunta: {user_input}\n\nContexto:\n{context_text}",
                },
            ]
            answer = call_chat(client, messages, chat_model)
            gen_ms = int((time.time() - start) * 1000)
            log_step("generate", {"model": chat_model, "duration_ms": gen_ms, "tokens": len(answer or "")})
        else:
            answer = "[Sem OPENAI_API_KEY] Contexto encontrado:\n\n" + (context_text[:1000] + ("..." if len(context_text) > 1000 else ""))
            log_step("no_api_key", {"info": "mostrando apenas contexto"})

        # Atualiza UI com resposta final
        answer_placeholder.markdown(f"<div class='chat-bubble assistant'>{answer}</div>", unsafe_allow_html=True)
        with st.expander("Fontes e logs detalhados"):
            st.subheader("Fontes usadas")
            st.json(sources)
            st.subheader("Logs desta interação")
            st.json({**current_log, "sources": sources})
        status.update(label="Concluído", state="complete", expanded=False)

    # Persiste no histórico e nos logs da sessão
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
    current_log["sources"] = sources
    st.session_state.all_logs.append(current_log)


