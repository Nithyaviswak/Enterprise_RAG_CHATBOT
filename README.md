# 🤖 RAG Chatbot

A production-grade Retrieval-Augmented Generation (RAG) chatbot with a modern Claude-like UI, powered by **Google Gemini API** and integrated with industry-leading open-source plugins.

![Python](https://img.shields.io/badge/Python-3.12-blue) ![Next.js](https://img.shields.io/badge/Next.js-15-black) ![LLM](https://img.shields.io/badge/LLM-Google%20Gemini%202.5-orange) ![RAG](https://img.shields.io/badge/RAG-RAGFlow-green) ![Embeddings](https://img.shields.io/badge/Embeddings-Sentence%20Transformers-purple) ![Docker](https://img.shields.io/badge/Docker-Ready-2496ED)

---

## ✨ Features

### Core RAG Pipeline
- **🧠 Hybrid Search** — Semantic (ChromaDB) + keyword (BM25) retrieval with cross-encoder re-ranking
- **🤖 Google Gemini 2.5 Flash** — Streaming responses via Server-Sent Events (SSE)
- **📄 Multi-Format Ingestion** — Upload PDF, DOCX, TXT, MD, CSV files with intelligent chunking
- **🔍 RAGFlow Integration** — Deep document understanding with agent-based retrieval (optional plugin)
- **🧬 Embedding Fine-Tuning** — Improve retrieval accuracy with synthetic data via LlamaIndex

### Advanced Features
- **📊 RAGAS Evaluation** — Automated RAG quality benchmarking (faithfulness, relevancy, precision, recall)
- **🔬 Hallucination Detection** — Built-in hallucination scoring and grounding verification
- **📈 LangSmith Observability** — Full pipeline tracing and debugging
- **🗃️ Enhanced Document Processing** — OCR-capable processing via Unstructured + Tesseract
- **⚡ Redis Caching** — Performance optimization with Redis for retrieval caching

### User Experience
- **💎 Glassmorphism UI** — Premium dark/light mode with smooth animations
- **💬 Persistent Chat History** — Auto-titled conversations stored in SQLite
- **📋 Code Highlighting** — Syntax highlighting with one-click copy
- **📎 Drag & Drop Upload** — Intuitive file upload modal
- **🔒 Firebase Auth** — Optional authentication layer
- **🚦 Rate Limiting** — Built-in API rate limiting via SlowAPI
- **🐳 Docker Ready** — Full Docker Compose deployment

---

## 🏗 Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  Frontend (Next.js 15)                   │
│  ┌────────────┐ ┌────────────┐ ┌─────────────┐         │
│  │ ChatInterface│ │ ChatSidebar│ │ FileUpload  │         │
│  │ (Streaming  │ │ (History)  │ │ (Drag&Drop) │         │
│  │  Markdown)  │ │            │ │             │         │
│  └────────────┘ └────────────┘ └─────────────┘         │
│  ┌────────────┐ ┌────────────┐                          │
│  │ ChatMessage │ │ThemeToggle │                          │
│  │ (Highlight) │ │(Dark/Light)│                          │
│  └────────────┘ └────────────┘                          │
└─────────────────────┬───────────────────────────────────┘
                      │ SSE / REST API
┌─────────────────────▼───────────────────────────────────┐
│              Backend (FastAPI + Python 3.12)              │
│  ┌───────────────────────────────────────────────────┐  │
│  │             Chat Service (Orchestrator)             │  │
│  └──────┬────────────┬──────────────┬────────────────┘  │
│         │            │              │                    │
│  ┌──────▼──────┐ ┌───▼──────────┐ ┌▼───────────────┐   │
│  │ Gemini API  │ │  Enhanced    │ │ Document       │   │
│  │ (Streaming) │ │  Retrieval   │ │ Processor      │   │
│  └─────────────┘ │  Service     │ │ (PDF/DOCX/TXT) │   │
│                  │  (Hybrid +   │ └────────────────┘   │
│                  │   Reranking)  │                      │
│                  └──┬─────┬─────┘                      │
│        ┌────────────┘     └───────────┐                 │
│  ┌─────▼───────┐            ┌────────▼─────────────┐   │
│  │  ChromaDB   │            │  RAGFlow Client      │   │
│  │  (Vector DB)│            │  (Optional Plugin)   │   │
│  └─────────────┘            └──────────────────────┘   │
│                                                          │
│  ┌────────────────────────────────────────────────────┐  │
│  │ Services Layer                                      │  │
│  │ • EmbeddingService (sentence-transformers)          │  │
│  │ • EvaluationService (RAGAS metrics)                 │  │
│  │ • HallucinationService (grounding checks)           │  │
│  │ • ObservabilityService (LangSmith tracing)          │  │
│  │ • FinetuneService (LlamaIndex embedding tuning)     │  │
│  └────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────┘
```

---

## 🔗 Plugin Integrations

| Plugin | Repository | Role |
|--------|-----------|------|
| **RAGFlow** | [infiniflow/ragflow](https://github.com/infiniflow/ragflow) | Core RAG engine — document parsing, retrieval |
| **Sentence-Transformers** | [huggingface/sentence-transformers](https://github.com/huggingface/sentence-transformers) | Embedding generation, cross-encoder re-ranking |
| **Finetune-Embedding** | [run-llama/finetune-embedding](https://github.com/run-llama/finetune-embedding) | Embedding fine-tuning with synthetic data |
| **RAGAS** | [explodinggradients/ragas](https://github.com/explodinggradients/ragas) | RAG evaluation framework |
| **LangSmith** | [langchain-ai/langsmith](https://github.com/langchain-ai/langsmith-sdk) | Observability and tracing |
| **Unstructured** | [Unstructured-IO/unstructured](https://github.com/Unstructured-IO/unstructured) | Advanced document parsing with OCR |

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.12**
- **Node.js 20+**
- **Docker** (optional, for containerized deployment)
- **Google Gemini API Key** — [Get one here](https://aistudio.google.com/apikey)

### 1. Clone & Setup

```bash
git clone https://github.com/Nithyaviswak/rag-chatbot.git
cd rag-chatbot
```

### 2. Backend Setup

```bash
cd backend

# Create virtual environment
python -m venv venv
venv\Scripts\activate      # Windows
# source venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Configure environment
copy .env.example .env
# Edit .env and add your GEMINI_API_KEY
```

### 3. Frontend Setup

```bash
cd frontend

# Install dependencies
npm install
```

### 4. Run Locally

**Terminal 1 — Backend:**
```bash
cd backend
venv\Scripts\activate
uvicorn app.main:app --reload --port 8000
```

**Terminal 2 — Frontend:**
```bash
cd frontend
npm run dev
```

Open **http://localhost:3000** in your browser.

---

## 🐳 Docker Deployment

```bash
# Build and start all services
docker compose up -d

# Check status
docker compose ps

# View logs
docker compose logs -f backend
```

### With RAGFlow (Optional)

RAGFlow requires its own Docker stack (16GB+ RAM):

```bash
git clone https://github.com/infiniflow/ragflow.git
cd ragflow/docker
docker compose up -d
```

Then configure `RAGFLOW_BASE_URL` and `RAGFLOW_API_KEY` in `backend/.env`.

---

## 📡 API Endpoints

### Chat & Conversations

| Method | Endpoint | Description |
|--------|---------|-------------|
| `POST` | `/api/chat` | Send message (SSE streaming) |
| `GET` | `/api/chat/history` | List conversations |
| `GET` | `/api/chat/{id}` | Get conversation messages |
| `DELETE` | `/api/chat/{id}` | Delete conversation |

### Documents

| Method | Endpoint | Description |
|--------|---------|-------------|
| `POST` | `/api/documents/upload` | Upload document (PDF/DOCX/TXT) |
| `GET` | `/api/documents` | List documents |
| `DELETE` | `/api/documents/{id}` | Delete document |

### Evaluation & Benchmarking

| Method | Endpoint | Description |
|--------|---------|-------------|
| `POST` | `/api/evaluation/evaluate` | Evaluate single RAG sample |
| `POST` | `/api/evaluation/evaluate/batch` | Batch evaluation |
| `GET` | `/api/evaluation/benchmark` | Run built-in benchmarks |
| `GET` | `/api/evaluation/statistics` | Get evaluation statistics |
| `POST` | `/api/evaluation/generate-dataset` | Generate synthetic test data |

### Fine-Tuning & Health

| Method | Endpoint | Description |
|--------|---------|-------------|
| `POST` | `/api/finetune/start` | Start embedding fine-tuning |
| `GET` | `/api/finetune/status` | Check fine-tuning progress |
| `GET` | `/api/health` | Health check + service status |

Full interactive API docs at **http://localhost:8000/docs**

---

## 🧬 Embedding Fine-Tuning

To improve retrieval accuracy (~5-10% improvement):

1. Upload documents to build a corpus
2. Trigger fine-tuning via API:
   ```bash
   curl -X POST http://localhost:8000/api/finetune/start \
     -H "Content-Type: application/json" \
     -d '{"base_model": "BAAI/bge-small-en-v1.5", "epochs": 2}'
   ```
3. Monitor progress: `GET /api/finetune/status`
4. Set `FINETUNED_MODEL_PATH` in `.env` and restart

---

## 📊 RAGAS Evaluation

Run automated quality benchmarks on your RAG pipeline:

```bash
# Evaluate a single sample
curl -X POST http://localhost:8000/api/evaluation/evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is machine learning?",
    "retrieved_context": [{"content": "Machine learning is..."}],
    "generated_answer": "Machine learning is a subset of AI..."
  }'

# Run built-in benchmarks
curl http://localhost:8000/api/evaluation/benchmark

# Generate synthetic evaluation dataset
curl -X POST "http://localhost:8000/api/evaluation/generate-dataset?num_questions=10"
```

**Metrics measured:** Context Precision, Context Recall, Context Relevancy, Faithfulness, Answer Relevancy, Overall Score.

---

## 📂 Project Structure

```
rag-chatbot/
├── backend/
│   ├── app/
│   │   ├── main.py                        # FastAPI entry point
│   │   ├── config.py                      # Environment configuration
│   │   ├── models/
│   │   │   ├── schemas.py                 # Pydantic request/response models
│   │   │   └── database.py                # SQLite chat history operations
│   │   ├── services/
│   │   │   ├── gemini_service.py          # Google Gemini API integration
│   │   │   ├── ragflow_client.py          # RAGFlow plugin client
│   │   │   ├── embedding_service.py       # sentence-transformers embeddings
│   │   │   ├── finetune_service.py        # LlamaIndex embedding fine-tuning
│   │   │   ├── vector_store.py            # ChromaDB vector storage
│   │   │   ├── retrieval_service.py       # Hybrid search + BM25
│   │   │   ├── enhanced_retrieval_service.py  # Advanced retrieval pipeline
│   │   │   ├── chat_service.py            # RAG orchestration
│   │   │   ├── evaluation_service.py      # RAGAS evaluation framework
│   │   │   ├── hallucination_service.py   # Hallucination detection
│   │   │   └── observability_service.py   # LangSmith tracing
│   │   ├── routers/
│   │   │   ├── chat.py                    # Chat endpoints
│   │   │   ├── documents.py               # Document upload/management
│   │   │   ├── evaluation.py              # RAGAS evaluation endpoints
│   │   │   ├── finetune.py                # Fine-tuning endpoints
│   │   │   └── health.py                  # Health check
│   │   └── middleware/
│   │       └── rate_limit.py              # SlowAPI rate limiting
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── src/
│   │   ├── app/                           # Next.js 15 app router
│   │   │   ├── layout.tsx                 # Root layout
│   │   │   ├── page.tsx                   # Main page
│   │   │   └── globals.css                # Design system (Glassmorphism)
│   │   ├── components/
│   │   │   ├── ChatInterface.tsx           # Main chat area + input
│   │   │   ├── ChatMessage.tsx             # Message bubble + markdown
│   │   │   ├── ChatSidebar.tsx             # Conversation history panel
│   │   │   ├── FileUpload.tsx              # Drag & drop upload modal
│   │   │   └── ThemeToggle.tsx             # Dark/light mode switch
│   │   ├── hooks/
│   │   │   ├── useChat.ts                  # Chat state management
│   │   │   └── useTheme.ts                 # Theme persistence
│   │   ├── lib/                            # API client utilities
│   │   └── types/                          # TypeScript type definitions
│   ├── package.json
│   └── Dockerfile
├── docker-compose.yml
├── .env.example
├── .gitignore
└── README.md
```

---

## ⚙️ Configuration

All settings in `backend/.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `GEMINI_API_KEY` | — | Google Gemini API key **(required)** |
| `GEMINI_MODEL` | `gemini-2.5-flash` | Gemini model name |
| `RAGFLOW_BASE_URL` | `http://localhost:9380` | RAGFlow server URL |
| `RAGFLOW_API_KEY` | — | RAGFlow API key |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Base embedding model |
| `RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Re-ranking model |
| `FINETUNED_MODEL_PATH` | — | Path to fine-tuned embedding model |
| `RATE_LIMIT` | `30/minute` | API rate limit |
| `REDIS_URL` | `redis://localhost:6379` | Redis URL for caching |
| `LANGSMITH_API_KEY` | — | LangSmith API key for tracing |

---

## 🚢 Production Deployment

### Backend → AWS/GCP/Railway
```bash
cd backend
docker build -t rag-chatbot-backend .
docker push <registry>/rag-chatbot-backend
```

### Frontend → Vercel
```bash
cd frontend
npx vercel --prod
```

Set `NEXT_PUBLIC_API_URL` to your backend URL in Vercel environment variables.

---

## 🛠 Tech Stack

| Layer | Technology |
|-------|-----------|
| **Frontend** | Next.js 15, React 19, TypeScript |
| **Styling** | Vanilla CSS (Glassmorphism + Dark Mode) |
| **Backend** | FastAPI, Python 3.12, Uvicorn |
| **LLM** | Google Gemini 2.5 Flash |
| **Vector DB** | ChromaDB |
| **Embeddings** | sentence-transformers (all-MiniLM-L6-v2) |
| **Search** | Hybrid (Semantic + BM25) + Cross-Encoder Reranking |
| **Evaluation** | RAGAS, LangSmith |
| **Auth** | Firebase Authentication |
| **Caching** | Redis |
| **Database** | SQLite (chat history) |
| **Deployment** | Docker Compose |

---

## 📄 License

MIT
