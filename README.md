# Chat SOS Educação - Alfabetização

Chatbot com RAG (Retrieval-Augmented Generation) para gestores e professores tirarem dúvidas sobre alfabetização, baseado nas transcrições dos vídeos da SOS Educação (Taís e Roberta).

## Arquitetura

```
┌─────────────────────────────────────────────────────────────────┐
│                         FRONTEND                                 │
│                    React + Vite + TypeScript                     │
│                        (Netlify)                                 │
└──────────────────────────┬──────────────────────────────────────┘
                           │ HTTP POST /api/chat
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                         BACKEND                                  │
│                   Django + Django Ninja                          │
│                        (Railway)                                 │
├─────────────────────────────────────────────────────────────────┤
│  ┌───────────┐   ┌────────────────┐   ┌──────────────────────┐  │
│  │    RAG    │──▶│   Embeddings   │──▶│  OpenAI Embeddings   │  │
│  │  Service  │   │    Service     │   │ text-embedding-3-small│  │
│  └─────┬─────┘   └────────────────┘   └──────────────────────┘  │
│        │                                                         │
│        ▼                                                         │
│  ┌───────────┐   ┌────────────────┐                             │
│  │    LLM    │──▶│  Google Gemini │                             │
│  │  Service  │   │gemini-3-pro-preview│                         │
│  └───────────┘   └────────────────┘                             │
└─────────────────────────────────────────────────────────────────┘
```

## Estrutura do Projeto

```
ast-sos/
├── backend/                    # Django + Django Ninja
│   ├── core/                   # Configurações Django
│   │   ├── settings.py
│   │   ├── urls.py
│   │   └── wsgi.py
│   ├── chat/                   # App principal
│   │   ├── api.py              # Endpoints da API
│   │   ├── schemas.py          # Schemas Pydantic
│   │   ├── services/           # Serviços de negócio
│   │   │   ├── embeddings.py   # OpenAI embeddings
│   │   │   ├── llm.py          # Google Gemini
│   │   │   └── rag.py          # RAG (retrieval)
│   │   └── data/               # Dados
│   │       ├── documents_rows.csv
│   │       └── embeddings.npy  # (gerado)
│   ├── requirements.txt
│   ├── Procfile                # Deploy Railway
│   └── env.template
├── frontend/                   # React + Vite
│   ├── src/
│   │   ├── components/         # Componentes UI
│   │   ├── hooks/              # Custom hooks
│   │   ├── services/           # API client
│   │   └── types/              # TypeScript types
│   ├── netlify.toml            # Deploy Netlify
│   └── env.template
└── documents_rows.csv          # Dados originais
```

## Configuração Local

### 1. Backend

```bash
cd backend

# Criar e ativar ambiente virtual
python -m venv venv
.\venv\Scripts\Activate.ps1  # Windows
# ou: source venv/bin/activate  # Linux/Mac

# Instalar dependências
pip install -r requirements.txt

# Configurar variáveis de ambiente
copy env.template .env
# Edite .env com suas chaves

# (Opcional) Pré-computar embeddings para performance
cd chat/data
python precompute.py
cd ../..

# Rodar servidor de desenvolvimento
python manage.py runserver
```

O backend estará disponível em `http://localhost:8000`.

**Endpoints:**
- `POST /api/chat/` - Enviar mensagem
- `GET /api/chat/health` - Health check
- `GET /api/docs` - Documentação OpenAPI

### 2. Frontend

```bash
cd frontend

# Instalar dependências
npm install

# Configurar variáveis de ambiente
copy env.template .env
# A URL padrão já é http://localhost:8000

# Rodar servidor de desenvolvimento
npm run dev
```

O frontend estará disponível em `http://localhost:5173`.

## Variáveis de Ambiente

### Backend (.env)

| Variável | Descrição | Obrigatório |
|----------|-----------|-------------|
| `GOOGLE_API_KEY` | Chave da API Google (Gemini) | Sim |
| `OPENAI_API_KEY` | Chave da API OpenAI (embeddings) | Sim |
| `DJANGO_SECRET_KEY` | Secret key do Django | Sim (prod) |
| `DEBUG` | Modo debug | Não |
| `ALLOWED_HOSTS` | Hosts permitidos | Sim (prod) |
| `CORS_ORIGINS` | Origens CORS permitidas | Sim |
| `CHAT_MODEL` | Modelo Gemini | Não |
| `EMBEDDING_MODEL` | Modelo de embeddings | Não |

### Frontend (.env)

| Variável | Descrição |
|----------|-----------|
| `VITE_API_URL` | URL do backend |

## Deploy

### Railway (Backend)

1. Conecte o repositório ao Railway
2. Configure as variáveis de ambiente no painel
3. O `Procfile` já está configurado

### Netlify (Frontend)

1. Conecte o repositório ao Netlify
2. Configure:
   - Build command: `npm run build`
   - Publish directory: `dist`
   - Base directory: `frontend`
3. Atualize `VITE_API_URL` para a URL do Railway
4. Atualize o redirect em `netlify.toml` para a URL do Railway

## API Reference

### POST /api/chat/

Envia uma mensagem e recebe resposta do assistente.

**Request:**
```json
{
  "message": "Como incentivar o gosto pela leitura?"
}
```

**Response:**
```json
{
  "answer": "Para incentivar o gosto pela leitura...",
  "sources": [
    {
      "video_id": "AULA 16 para escrever melhor_original",
      "chunk_id": "3",
      "similarity": 0.6504,
      "metadata": "{\"file\": \"...\", \"source\": \"video\"}"
    }
  ],
  "logs": [
    {
      "step": "embed_query",
      "duration_ms": 250,
      "details": {"model": "text-embedding-3-small", "ok": true}
    },
    {
      "step": "retrieve_topk",
      "duration_ms": 5,
      "details": {"top_k": 5, "results": [...]}
    },
    {
      "step": "build_context",
      "duration_ms": 1,
      "details": {"chars": 4264, "num_sources": 5}
    },
    {
      "step": "generate",
      "duration_ms": 2500,
      "details": {"model": "gemini-3-pro-preview", "answer_length": 850}
    }
  ]
}
```

## Tecnologias

- **Backend:** Python, Django, Django Ninja, NumPy, Pandas
- **Frontend:** React 18, TypeScript, Vite
- **LLM:** Google Gemini 3 Pro Preview
- **Embeddings:** OpenAI text-embedding-3-small
- **Deploy:** Railway (backend), Netlify (frontend)