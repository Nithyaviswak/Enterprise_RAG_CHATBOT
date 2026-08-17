# 🤖 Enterprise RAG Chatbot

A production-grade Retrieval-Augmented Generation (RAG) chatbot with a modern Claude-inspired UI, powered by **Google Gemini 2.5 Flash**, a **Neo4j Knowledge Graph**, and integrated with industry-leading open-source plugins.

![Python](https://img.shields.io/badge/Python-3.11-blue) ![Next.js](https://img.shields.io/badge/Next.js-15-black) ![React](https://img.shields.io/badge/React-19-61DAFB) ![LLM](https://img.shields.io/badge/LLM-Google%20Gemini%202.5-orange) ![RAG](https://img.shields.io/badge/RAG-Hybrid%20Pipeline-green) ![Embeddings](https://img.shields.io/badge/Embeddings-Sentence%20Transformers-purple) ![Neo4j](https://img.shields.io/badge/Graph-Neo4j-008CC1) ![Docker](https://img.shields.io/badge/Docker-Ready-2496ED)

---

## ✨ Features

### Core RAG Pipeline
- **🧠 Hybrid Search** — Semantic (ChromaDB) + keyword (BM25) retrieval with cross-encoder re-ranking
- **🤖 Google Gemini 2.5 Flash** — Streaming responses via Server-Sent Events (SSE) with retry/backoff that honors Gemini's `RetryInfo.retryDelay`
- **📄 Multi-Format Ingestion** — Upload PDF, DOCX, TXT, MD, CSV files with intelligent chunking
- **🔍 RAGFlow Integration** — Deep document understanding with agent-based retrieval (optional plugin)
- **🧬 Embedding Fine-Tuning** — Improve retrieval accuracy with synthetic data via LlamaIndex

### Knowledge Graph
- **🕸️ Neo4j Graph Database** — Entity and relationship storage with multi-hop traversal
- **🔎 Entity Extraction** — LLM-powered entity extraction from uploaded documents
- **🔗 Relationship Extraction** — Automatic relationship discovery between entities
- **🗺️ Interactive Graph Explorer** — Force-directed SVG visualization with search, filter, and drill-down
- **🛤️ Path Finding** — Discover connections between entities via multi-hop graph queries
- **🔄 Hybrid Graph + Vector Retrieval** — Combines graph traversal with vector similarity for richer context

### Enterprise Security & Governance
- **🛡️ Prompt Injection Guardrails** — Regex + LLM-based detection blocks jailbreaks, system prompt leaks, and data exfiltration
- **🔐 Access Control Lists (ACL)** — Organization, role, and department-level document access filtering
- **📝 Immutable Audit Logs** — Hash-chained, append-only JSONL logs for document access, search, chat, and security events
- **🔒 Firebase Authentication** — Optional auth middleware layer
- **🚦 Rate Limiting** — Built-in API rate limiting via SlowAPI

### Intelligence & Quality
- **🧭 Query Router** — LLM + keyword intent classification routes queries to optimal handlers (Small Talk, Knowledge Base, Coding, Document)
- **📊 Evaluation & Release Gates** — Deterministic RAG metrics (faithfulness, relevancy, precision, recall) with gated CI via `evaluation/evaluate.py`
- **🔬 Hallucination Detection** — Built-in hallucination scoring and grounding verification
- **📈 LangSmith Observability** — Full pipeline tracing and debugging
- **🗃️ Enhanced Document Processing** — OCR-capable processing via Unstructured + Tesseract
- **⚡ Redis Caching** — Performance optimization with Redis for retrieval caching

### User Experience
- **💎 Glassmorphism UI** — Premium dark/light mode with smooth animations
- **💬 Persistent Chat History** — Auto-titled conversations stored in SQLite
- **📋 Code Highlighting** — Syntax highlighting with one-click copy
- **📎 Drag & Drop Upload** — Intuitive file upload modal
- **🕸️ Graph Explorer Modal** — Interactive knowledge graph visualization with entity filtering
- **🐳 Docker Ready** — Full Docker Compose deployment with Neo4j, ChromaDB, and app services

---

## 🔒 Production Hardening (this upgrade)

The codebase was upgraded from a basic RAG demo into a measurable, observable,
gate-protected pipeline. Everything in this section is implemented and verified
by the automated test suite (`tests/`, 45 tests) and the evaluation harness.

### End-to-end RAG pipeline (`backend/app/rag/`)
`query → hybrid retrieval → guardrails → grounded generation → hallucination check → confidence → source attribution`

- **Configurable chunking** (`chunking.py`) — word and recursive strategies with
  size/overlap/min-length driven by `backend/.env`, no longer hard-coded.
- **Page-aware ingestion with dedup** (`ingestion.py`) — PDF/DOCX/TXT/MD/CSV,
  provenance metadata (source, document_id, chunk_id, page, content_hash), and
  a SHA-256 content-hash check that skips already-ingested content.
- **Hybrid retriever** (`retriever.py`) — semantic + BM25 (+ optional RAGFlow)
  fusion with dedup, metadata filters, similarity threshold, and a retrieval
  confidence score (fixed a numpy `max()` bug that broke BM25).
- **Cross-encoder re-ranking with cosine fallback** (`reranker.py`) — injectable
  scorer keeps tests deterministic and offline.
- **Guardrails** (`guardrails.py`) — refuses on empty/low-confidence retrieval
  instead of hallucinating, using a configurable threshold and refusal message.
- **Grounded generation** (`generator.py`) — the system prompt instructs the LLM
  to answer only from numbered, page-annotated sources and to refuse otherwise.
- **Hallucination detector** (`hallucination.py`) — deterministic lexical
  grounding verifier: per-claim support vs. the retrieved context, risk level,
  and a combined confidence score.
- **Failure taxonomy** (`failures.py`) — every pipeline outcome is categorised:
  `RETRIEVAL_FAILURE`, `LOW_CONFIDENCE`, `EMPTY_CONTEXT`, `LLM_FAILURE`,
  `TIMEOUT`, `INVALID_DOCUMENT`, `PARSING_FAILURE`, `HALLUCINATION_RISK`.

### Observability (`backend/app/observability/`)
Every request gets a `request_id`; retrieval/generation stages are timed and
emitted as structured logs. An in-memory `MetricsStore` aggregates live
metrics shown in the UI dashboard and exposed at
`GET /api/evaluation/metrics/live`: retrieval confidence, grounding ratio,
latency, refusal rate, and hallucination-risk rate.

### Developer mode
The UI has a **`</>` toggle** that enables `debug_mode` on chat requests. The
API responds with a `debug` SSE event containing retrieved documents (source,
page, scores, method), retrieval methods and confidence, final context, the
system prompt, and per-stage timings. Normal users only ever see answer +
sources + confidence.

### Automated tests (`tests/`, CI at `.github/workflows/rag-ci.yml`)
- 45 deterministic unit + integration tests — no ChromaDB, no model downloads,
  no network (hermetic fakes for the store, embeddings, reranker and LLM).
- The pipeline is tested end-to-end: answer path, empty/low-confidence refusal,
  LLM failure fallback, debug payload.
- CI runs tests + an **offline evaluation smoke** on every push.

---

```
┌──────────────────────────────────────────────────────────────┐
│                    Frontend (Next.js 15)                      │
│  ┌─────────────┐ ┌─────────────┐ ┌──────────────┐           │
│  │ChatInterface│ │ ChatSidebar │ │  FileUpload  │           │
│  │ (Streaming  │ │  (History)  │ │ (Drag&Drop)  │           │
│  │  Markdown)  │ │             │ │              │           │
│  └─────────────┘ └─────────────┘ └──────────────┘           │
│  ┌─────────────┐ ┌─────────────┐ ┌──────────────┐           │
│  │ ChatMessage │ │ThemeToggle  │ │GraphExplorer │           │
│  │ (Highlight) │ │(Dark/Light) │ │(SVG Force    │           │
│  └─────────────┘ └─────────────┘ │ Layout)      │           │
│                                   └──────────────┘           │
└─────────────────────┬────────────────────────────────────────┘
                      │ SSE / REST API
┌─────────────────────▼────────────────────────────────────────┐
│              Backend (FastAPI + Python 3.11)                   │
│  ┌────────────────────────────────────────────────────────┐  │
│  │            Chat Service (Orchestrator)                  │  │
│  └───────┬─────────────┬──────────────┬───────────────────┘  │
│          │             │              │                       │
│  ┌───────▼───────┐ ┌───▼──────────┐ ┌▼────────────────┐    │
│  │  Gemini API   │ │  Enhanced    │ │ Document         │    │
│  │  (Streaming + │ │  Retrieval   │ │ Processor        │    │
│  │   Retry)      │ │  Service     │ │ (PDF/DOCX/TXT)   │    │
│  └───────────────┘ │  (Hybrid +   │ └─────────────────┘    │
│                    │   Reranking)  │                         │
│                    └──┬─────┬─────┘                         │
│       ┌───────────────┘     └──────────────┐                │
│  ┌────▼────────┐            ┌──────────────▼────────────┐   │
│  │  ChromaDB   │            │  RAGFlow Client           │   │
│  │ (Vector DB) │            │  (Optional Plugin)        │   │
│  └─────────────┘            └───────────────────────────┘   │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐  │
│  │ Knowledge Graph Layer                                   │  │
│  │ • Neo4j GraphService (CRUD + traversal)                │  │
│  │ • EntityExtractor (LLM-powered)                        │  │
│  │ • RelationshipExtractor (LLM-powered)                  │  │
│  │ • GraphRetriever (hybrid graph + vector search)        │  │
│  │ • GraphEmbeddingService (entity embeddings)            │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐  │
│  │ Enterprise Services                                     │  │
│  │ • GuardrailService (prompt injection defense)          │  │
│  │ • QueryRouter (intent classification + routing)        │  │
│  │ • ACLService (access control filtering)                │  │
│  │ • AuditService (immutable hash-chained logs)           │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌────────────────────────────────────────────────────────┐  │
│  │ ML & Observability Services                             │  │
│  │ • EmbeddingService (sentence-transformers)             │  │
│  │ • EvaluationService (RAGAS metrics)                    │  │
│  │ • HallucinationService (grounding checks)              │  │
│  │ • ObservabilityService (LangSmith tracing)             │  │
│  │ • FinetuneService (LlamaIndex embedding tuning)        │  │
│  └────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
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
| **Neo4j** | [neo4j/neo4j](https://github.com/neo4j/neo4j) | Knowledge graph database |

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.11**
- **Node.js 20+**
- **Docker** (optional, for containerized deployment)
- **Google Gemini API Key** — [Get one here](https://aistudio.google.com/apikey)

### 1. Clone & Setup

```bash
git clone https://github.com/Nithyaviswak/Enterprise_RAG_CHATBOT.git
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
# Build and start all services (Backend + Frontend + ChromaDB + Neo4j)
docker compose up -d

# Check status
docker compose ps

# View logs
docker compose logs -f backend
```

### Services Started

| Service | Port | Description |
|---------|------|-------------|
| **Backend** | `8000` | FastAPI application |
| **Frontend** | `3000` | Next.js UI |
| **ChromaDB** | `8200` | Vector store |
| **Neo4j Browser** | `7474` | Graph database UI |
| **Neo4j Bolt** | `7687` | Graph database protocol |

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

### Knowledge Graph

| Method | Endpoint | Description |
|--------|---------|-------------|
| `GET` | `/api/graph/stats` | Graph statistics (node/edge counts) |
| `GET` | `/api/graph/entities` | List entities (with search & type filter) |
| `GET` | `/api/graph/entities/{id}` | Entity detail + relationships |
| `GET` | `/api/graph/entities/{id}/neighborhood` | Explore entity neighborhood (for visualization) |
| `POST` | `/api/graph/entities` | Create entity |
| `DELETE` | `/api/graph/entities/{id}` | Delete entity |
| `GET` | `/api/graph/relationships` | List relationships |
| `POST` | `/api/graph/relationships` | Create relationship |
| `POST` | `/api/graph/search` | Smart graph search (auto-selects strategy) |
| `POST` | `/api/graph/traverse` | Multi-hop graph traversal |
| `GET` | `/api/graph/paths` | Find paths between two entities |
| `POST` | `/api/graph/index-document` | Extract entities/relationships from a document |
| `DELETE` | `/api/graph/clear` | Clear entire knowledge graph |

### Evaluation & Benchmarking

| Method | Endpoint | Description |
|--------|---------|-------------|
| `POST` | `/api/chat` | Send message (SSE streaming; body may set `debug_mode`) |
| `GET` | `/api/evaluation/metrics/live` | Live runtime metrics (confidence, latency, refusal/hallucination rates) |
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

## 🕸️ Knowledge Graph

The Knowledge Graph system provides structured, relational context that complements vector-based retrieval.

### How It Works

1. **Upload** a document via the `/api/documents/upload` endpoint
2. **Index** the document to the graph via `/api/graph/index-document?document_id=...`
3. The system automatically:
   - Extracts entities (People, Organizations, Concepts, Technologies, etc.)
   - Discovers relationships between entities
   - Stores everything in Neo4j
4. **Query** the graph via the API or **explore visually** via the Graph Explorer in the UI

### Graph Explorer UI

The built-in Graph Explorer provides an interactive, force-directed visualization:

- **Search** entities by name
- **Filter** by entity type (Organization, Person, Concept, Technology, etc.)
- **Click** a node to see its details and connections
- **Double-click** to expand and explore its neighborhood
- **Detail panel** shows entity description and all connected relationships

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

## 📊 Evaluation & Release Gates

A dedicated evaluation harness lives in `evaluation/` and is wired to the
actual knowledge base (the resume + AI-agent notes currently indexed in
ChromaDB):

- `evaluation/dataset.json` — 24 questions: factual, multi-step, per-document,
  unanswerable and hallucination-bait examples with reference answers grounded
  in the real KB.
- `evaluation/metrics.py` — deterministic, reproducible metrics (no API needed):
  **faithfulness**, **answer relevancy**, **context precision** (rank-weighted),
  **context recall**, **hallucination rate**, **refusal rate**.
- `evaluation/thresholds.yaml` — release gates the runner fails (non-zero exit)
  when a metric is below/above target.
- `evaluation/evaluate.py` — runs every question through the real `RagPipeline`,
  prints a scoreboard, writes a timestamped JSON report to `evaluation/reports/`,
  and fails CI on a breached gate.

Run it (local, uses the live Gemini key from `backend/.env`):

```bash
backend/venv/Scripts/python.exe evaluation/evaluate.py        # Windows
python evaluation/evaluate.py                                  # macOS / Linux
# pace requests to respect the free-tier per-minute quota:
python evaluation/evaluate.py --sample-delay 12
# override a gate:
python evaluation/evaluate.py --min-faithfulness 0.85
```

Run it hermetically in CI (stub retrieval + generation, no network):

```bash
python evaluation/evaluate.py --offline
```

### Measured results (as of this build)

These are real numbers from runs against the actual KB with `gemini-2.5-flash`.
A **3-sample smoke on the fixed metric pipeline** recorded:

| Metric | Value |
|--------|-------|
| Faithfulness | **0.889** |
| Answer relevancy | **0.681** |
| Context recall | **0.674** |
| Hallucination rate | **0.000** |

The **full 24-sample live gate run** was partially interrupted by the Gemini
free tier **daily quota (20 req/day)** — several answers fell back to the
refusal path on `LLM_FAILURE`, so its aggregates are not representative
(final: `FAIL` on context precision/recall because attribution bug zeroed them
at the time; both bugs fixed since). No fabricated numbers are published — run
`evaluation/evaluate.py` on a tier with headroom to get the definitive
scoreboard. The **offline harness currently passes all gates (RESULT: PASS)**.

### Live runtime metrics (dashboard / Dev panel)

```
GET /api/evaluation/metrics/live
```
Returns in-memory aggregates for the running server: total requests, avg
retrieval/generation/total latency, avg retrieval confidence, avg grounding,
refusal rate, hallucination-risk rate, and failure category counts.

### Legacy REST evaluation endpoints (kept)

```bash
curl -X POST http://localhost:8000/api/evaluation/evaluate \
  -H "Content-Type: application/json" \
  -d '{"query":"What is machine learning?","retrieved_context":[{"content":"Machine learning is..."}],"generated_answer":"Machine learning is a subset of AI..."}'

curl http://localhost:8000/api/evaluation/benchmark
curl -X POST "http://localhost:8000/api/evaluation/generate-dataset?num_questions=10"
```

**Metrics measured:** Context Precision, Context Recall, Context Relevancy, Faithfulness, Answer Relevancy, Overall Score.

---

## 🛡️ Security Features

### Prompt Injection Guardrails

The `GuardrailService` defends against prompt injection attacks using a layered approach:

- **High-confidence blocking** — Detects "ignore previous instructions", system prompt leakage, jailbreak patterns (`DAN`, developer mode), and data exfiltration attempts
- **Suspicious pattern detection** — Flags subtle override attempts, prompt manipulation, and token injection
- **Leak detection** — Identifies accidentally leaked system prompts in responses
- **Threat levels** — `SAFE → LOW → MEDIUM → HIGH → BLOCKED`

### Access Control (ACL)

Fine-grained document access control:
- Organization-level isolation
- Role-based access (`admin > manager > employee > contractor > guest`)
- Department-based filtering
- Per-document allow/deny lists

### Audit Logging

Immutable, hash-chained audit logs (JSONL format):
- Document access, upload, delete events
- Search queries with result counts
- Chat messages (length + RAG usage)
- Security block events
- Integrity verification via hash chain

---

## 📂 Project Structure

```
rag-chatbot/
├── backend/
│   ├── app/
│   │   ├── main.py                           # FastAPI entry point + lifespan
│   │   ├── config.py                         # Pydantic settings (env vars)
│   │   ├── models/
│   │   │   ├── schemas.py                    # Pydantic request/response models
│   │   │   └── database.py                   # SQLite chat history (aiosqlite)
│   │   ├── rag/                              # ★ production RAG pipeline
│   │   │   ├── pipeline.py                   # End-to-end orchestrator
│   │   │   ├── retriever.py                  # Hybrid retrieval + confidence
│   │   │   ├── reranker.py                   # Cross-encoder + cosine fallback
│   │   │   ├── chunking.py                   # Configurable chunking strategies
│   │   │   ├── ingestion.py                  # Page-aware parsing + dedup
│   │   │   ├── generator.py                  # Grounded prompt building
│   │   │   ├── guardrails.py                 # Refusal on weak/no context
│   │   │   ├── hallucination.py              # Deterministic grounding checks
│   │   │   └── failures.py                   # Failure taxonomy
│   │   ├── observability/
│   │   │   └── tracing.py                    # Tracer + MetricsStore
│   │   ├── services/
│   │   │   ├── gemini_service.py             # Gemini API + retry/backoff
│   │   │   ├── chat_service.py               # RAG orchestration
│   │   │   ├── embedding_service.py          # sentence-transformers embeddings
│   │   │   ├── vector_store.py               # ChromaDB vector storage
│   │   │   ├── retrieval_service.py          # Hybrid search + BM25
│   │   │   ├── enhanced_retrieval_service.py # Advanced retrieval pipeline
│   │   │   ├── ragflow_client.py             # RAGFlow plugin client
│   │   │   ├── finetune_service.py           # LlamaIndex embedding tuning
│   │   │   ├── evaluation_service.py         # RAGAS evaluation framework
│   │   │   ├── hallucination_service.py      # Hallucination detection
│   │   │   ├── observability_service.py      # LangSmith tracing
│   │   │   ├── guardrail_service.py          # Prompt injection defense
│   │   │   ├── query_router.py               # Intent classification + routing
│   │   │   ├── acl_service.py                # Access control lists
│   │   │   ├── audit_service.py              # Immutable audit logging
│   │   │   └── knowledge_graph/
│   │   │       ├── graph_service.py          # Neo4j CRUD + traversal
│   │   │       ├── entity_extractor.py       # LLM entity extraction
│   │   │       ├── relationship_extractor.py # LLM relationship extraction
│   │   │       ├── graph_retriever.py        # Hybrid graph + vector search
│   │   │       └── graph_embeddings.py       # Entity embeddings
│   │   ├── routers/
│   │   │   ├── chat.py                       # Chat endpoints
│   │   │   ├── documents.py                  # Document upload/management
│   │   │   ├── graph.py                      # Knowledge graph API
│   │   │   ├── evaluation.py                 # Evaluation + live metrics endpoints
│   │   │   ├── finetune.py                   # Fine-tuning endpoints
│   │   │   └── health.py                     # Health check
│   │   └── middleware/
│   │       ├── rate_limit.py                 # SlowAPI rate limiting
│   │       └── auth.py                       # Firebase auth middleware
│   ├── requirements.txt
│   ├── requirements-runtime.txt              # lean set for CI/tests/harness
│   └── Dockerfile
├── evaluation/                               # ★ quality gates
│   ├── dataset.json                          # 24 questions grounded in the KB
│   ├── metrics.py                            # deterministic RAG metrics
│   ├── thresholds.yaml                       # release gates
│   ├── thresholds.offline.yaml               # hermetic CI gate profile
│   ├── evaluate.py                           # runner + gate enforcement
│   └── reports/                              # timestamped scoreboards
├── tests/                                    # ★ 45 hermetic tests
│   ├── conftest.py                           # deterministic fakes
│   ├── test_chunking.py
│   ├── test_ingestion.py
│   ├── test_reranker.py
│   ├── test_retriever.py
│   ├── test_guardrails.py
│   ├── test_hallucination.py
│   ├── test_metrics.py
│   └── test_pipeline_integration.py
├── .github/workflows/rag-ci.yml              # CI: tests + offline gates + live eval
├── frontend/
│   ├── src/
│   │   ├── app/                              # Next.js 15 app router
│   │   │   ├── layout.tsx                    # Root layout
│   │   │   ├── page.tsx                      # Main page
│   │   │   └── globals.css                   # Design system (Glassmorphism)
│   │   ├── components/
│   │   │   ├── ChatInterface.tsx             # Main chat area + input
│   │   │   ├── ChatMessage.tsx               # Message bubble + markdown
│   │   │   ├── DevPanel.tsx                  # Developer insights + live metrics
│   │   │   ├── ChatSidebar.tsx               # Conversation history panel
│   │   │   ├── FileUpload.tsx                # Drag & drop upload modal
│   │   │   ├── GraphExplorer.tsx             # Interactive knowledge graph viz
│   │   │   └── ThemeToggle.tsx               # Dark/light mode switch
│   │   ├── hooks/
│   │   │   ├── useChat.ts                    # Chat state management
│   │   │   └── useTheme.ts                   # Theme persistence
│   │   ├── lib/                              # API client utilities
│   │   └── types/                            # TypeScript type definitions
│   ├── package.json
│   └── Dockerfile
├── docker-compose.yml
├── .env.example
├── .python-version                           # Pinned to 3.11.11
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
| `LLM_TEMPERATURE` | `0.2` | Generation temperature |
| `LLM_MAX_OUTPUT_TOKENS` | `2048` | Max generated tokens |
| `RAGFLOW_BASE_URL` | `http://localhost:9380` | RAGFlow server URL |
| `RAGFLOW_API_KEY` | — | RAGFlow API key |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Base embedding model |
| `RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Re-ranking model |
| `FINETUNED_MODEL_PATH` | — | Path to fine-tuned embedding model |
| `RATE_LIMIT` | `30/minute` | API rate limit |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` / `CHUNK_MIN_LENGTH` | `512` / `64` / `20` | Chunking config (words) |
| `RETRIEVAL_TOP_K` | `5` | Final context size for the LLM |
| `RETRIEVAL_HYBRID` | `true` | Combine semantic + BM25 |
| `RETRIEVAL_SEMANTIC_WEIGHT` | `0.7` | Semantic share in hybrid fusion |
| `SIMILARITY_THRESHOLD` | `0.30` | Min cosine similarity for semantic hits |
| `RERANK_ENABLED` / `RERANK_TOP_K` | `true` / `8` | Cross-encoder re-ranking |
| `LOW_CONFIDENCE_THRESHOLD` | `0.40` | Below this the guardrail refuses to answer |
| `REFUSAL_MESSAGE` | *(configured)* | Refusal copy returned by guardrails |
| `MAX_CONTEXT_CHARS` | `12000` | Context budget sent to the LLM |
| `HALLUCINATION_CHECK_ENABLED` / `HALLUCINATION_MIN_OVERLAP` | `true` / `2` | Grounding verifier |
| `EVALUATION_DATASET` / `EVALUATION_THRESHOLDS` | `evaluation/dataset.json` / `evaluation/thresholds.yaml` | Harness paths |
| `EVALUATION_REPORT_DIR` | `evaluation/reports` | Scoreboard JSON reports |
| `NEO4J_URI` | `bolt://localhost:7687` | Neo4j connection URI |
| `NEO4J_USER` | `neo4j` | Neo4j username |
| `NEO4J_PASSWORD` | `changeme-password` | Neo4j password |
| `NEO4J_DATABASE` | `neo4j` | Neo4j database name |
| `GRAPH_EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Model for graph embeddings |
| `ENTITY_EXTRACTION_MODEL` | `gemini-2.5-flash` | Model for entity extraction |
| `FIREBASE_CREDENTIALS_PATH` | — | Path to Firebase service account JSON |

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
|-------|-----------:|
| **Frontend** | Next.js 15, React 19, TypeScript |
| **Styling** | Vanilla CSS (Glassmorphism + Dark Mode) |
| **Backend** | FastAPI, Python 3.11, Uvicorn |
| **LLM** | Google Gemini 2.5 Flash (with retry/backoff) |
| **Vector DB** | ChromaDB |
| **Graph DB** | Neo4j 5 Community (with APOC) |
| **Embeddings** | sentence-transformers (all-MiniLM-L6-v2) |
| **Search** | Hybrid (Semantic + BM25) + Cross-Encoder Reranking |
| **Graph Search** | Hybrid Graph Traversal + Vector Similarity |
| **Evaluation** | RAGAS, LangSmith |
| **Security** | Guardrails, ACL, Audit Logs, Firebase Auth |
| **Auth** | Firebase Authentication |
| **Caching** | Redis |
| **Database** | SQLite (chat history) |
| **Deployment** | Docker Compose |

---

## 📄 License

MIT
