CLAUDE.md — Project 1: Production RAG System

> Drop this file at the root of `production-rag/`. Claude Code reads it automatically on every session.

---

Project Overview
What this is: A production-grade Retrieval-Augmented Generation system. Users upload PDFs/text files, the system indexes them, and answers natural-language questions with cited sources.
Stack (all free, all local):
LLM: Ollama + Llama3.2 (local, $0/call)
Embeddings: BAAI/bge-small-en-v1.5 via sentence-transformers (local, $0)
Vector DB: ChromaDB (local persistent)
Keyword Search: rank-bm25 (pure Python)
Reranking: cross-encoder/ms-marco-MiniLM-L-6-v2 (local)
Cache: Redis (semantic similarity cache)
Queue: Celery + Redis (async ingestion)
Observability: Arize Phoenix + Prometheus + Grafana
Evals: RAGAS with local Ollama judge
Security: Presidio PII masking
API: FastAPI + API key auth + rate limiting
CI: GitHub Actions eval quality gate

---

Project Structure

```
production-rag/
├── src/
│   ├── config.py                  # Pydantic Settings — all config here
│   ├── ingestion/
│   │   ├── parser.py              # PDF/text → raw text pages
│   │   ├── chunker.py             # Text → overlapping chunks (500w, 100w overlap)
│   │   └── tasks.py               # Celery async tasks
│   ├── retrieval/
│   │   ├── embedder.py            # Embedding model (cached with lru_cache)
│   │   ├── vectorstore.py         # ChromaDB add/search
│   │   ├── bm25.py                # BM25 in-memory index
│   │   └── retriever.py           # RRF fusion + CrossEncoder rerank
│   ├── generation/
│   │   ├── llm.py                 # Ollama chat wrapper
│   │   ├── prompts/rag_v1.yaml    # Versioned prompt template
│   │   └── generator.py           # Full RAG pipeline
│   ├── api/
│   │   ├── main.py                # FastAPI app entry point
│   │   ├── models.py              # Pydantic request/response models
│   │   ├── auth.py                # X-API-Key header verification
│   │   └── cache.py               # Redis semantic cache
│   ├── security/
│   │   └── pii.py                 # Presidio PII detection + masking
│   └── observability/
│       └── metrics.py             # Prometheus counters/histograms
├── tests/
│   ├── unit/                      # Fast isolated tests (pytest)
│   ├── integration/               # Full pipeline tests
│   └── evals/
│       ├── dataset.json           # 20-30 Q&A pairs for RAGAS
│       └── run_evals.py           # RAGAS evaluation runner
├── docker/
│   ├── docker-compose.yml         # Redis + Prometheus + Grafana
│   └── prometheus.yml             # Scrape config
├── .github/workflows/
│   └── eval_gate.yml              # CI quality gate (fails PR if eval drops)
├── prompts/                       # Versioned YAML prompts
├── data/
│   ├── raw_docs/                  # Input documents (gitignored)
│   └── chroma_db/                 # ChromaDB storage (gitignored)
├── .env                           # Local secrets (NEVER commit)
├── .env.example                   # Template for teammates
└── requirements.txt
```

---

Environment Setup (Run Once)

```bash
# 1. Install Ollama
curl -fsSL https://ollama.com/install.sh | sh
ollama serve &
ollama pull llama3.2

# 2. Start Redis (Docker)
docker run -d -p 6379:6379 --name rag-redis redis:7-alpine

# 3. Python environment
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# 4. Copy env file
cp .env.example .env
# Edit .env — set your API_KEY
```

---

Daily Development Commands

```bash
# Activate environment (every session)
source venv/bin/activate

# Start services
docker start rag-redis
ollama serve &

# Run API
python -m src.api.main
# → http://localhost:8000/docs

# Start observability (separate terminal)
python -m phoenix.server.main
# → http://localhost:6006

# Start Celery worker (separate terminal)
celery -A src.ingestion.tasks.celery_app worker --loglevel=info

# Start Prometheus + Grafana
cd docker && docker compose up -d
# Grafana → http://localhost:3000 (admin/admin)
# Prometheus → http://localhost:9090
```

---

Testing Commands

```bash
# Unit tests
pytest tests/unit/ -v

# Integration tests
pytest tests/integration/ -v

# Manual eval run
python tests/evals/run_evals.py

# Health check
curl http://localhost:8000/health

# Query test
curl -X POST http://localhost:8000/query \
  -H "X-API-Key: dev-key-change-in-production" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the main topic of the document?"}'

# Upload document
curl -X POST http://localhost:8000/ingest \
  -H "X-API-Key: dev-key-change-in-production" \
  -F "file=@your_document.pdf"
```

---

Key Design Decisions (Do Not Change Without Understanding Why)
Decision Why
`@lru_cache(maxsize=1)` on embedding model Models are large. Load once, reuse. Thread-safe.
Chunk size 500 words, overlap 100 words Empirically best for Llama3.2 4096 context window
BM25 + Vector → RRF → CrossEncoder Three-stage funnel. Wide net (both), smart merge (RRF), precision rerank (CrossEncoder)
Presidio runs BEFORE ChromaDB indexing PII must never reach the vector store — masking in-flight is not enough
YAML prompt versioning Prompt changes are git-tracked separately from code. Rollback is `git revert prompt_file.yaml`
`task_acks_late=True` in Celery Tasks are not acknowledged until complete. If worker dies, task re-queues. No data loss.
Faithfulness threshold 0.70 Below this, the LLM is making things up 30% of the time — unacceptable for enterprise

---

Retrieval Pipeline (The Core)

```
Query
  │
  ├── embed_query()           → query_embedding (384-dim vector)
  │
  ├── vector_search()         → top-20 semantic hits from ChromaDB
  │
  ├── bm25_search()           → top-20 keyword hits from BM25 index
  │
  ├── reciprocal_rank_fusion()→ merged ranked list (RRF formula: 1/(k + rank))
  │
  └── rerank()                → CrossEncoder scores top-20, returns top-5
```

---

Configuration Reference (.env variables)

```env
# LLM
OLLAMA_BASE_URL=http://localhost:11434
LLM_MODEL=llama3.2
LLM_TEMPERATURE=0.1
LLM_NUM_CTX=4096

# Embeddings
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5

# Vector store
CHROMA_PATH=./data/chroma_db
CHROMA_COLLECTION=rag_documents

# Retrieval tuning
VECTOR_TOP_K=20
BM25_TOP_K=20
RERANK_TOP_N=5

# Cache
REDIS_URL=redis://localhost:6379/0
CACHE_TTL_SECONDS=3600
CACHE_SIMILARITY_THRESHOLD=0.92

# Security
API_KEY=your-secret-key-here
RATE_LIMIT_PER_MINUTE=100

# Eval thresholds
FAITHFULNESS_THRESHOLD=0.70
CONTEXT_PRECISION_THRESHOLD=0.65

# Observability
LOG_LEVEL=INFO
```

---

Production Layer Coverage
Layer Implementation
Scaling Stateless FastAPI, Celery parallel workers, Redis semantic cache reduces LLM load
Observability Arize Phoenix (LLM traces), Prometheus (metrics), Grafana (dashboards)
Evals RAGAS faithfulness/precision/recall, GitHub Actions blocks bad PRs
Queuing Celery + Redis, dead letter queue, exponential backoff retry
Security Presidio PII masking, X-API-Key auth, rate limiting, .env secrets
Cost Control $0 stack: Ollama local, semantic cache, local embeddings

---

Common Issues & Fixes

```
Issue: ChromaDB "collection already exists" error
Fix:   Collection is created with get_or_create_collection() — should not happen.
       If it does: delete data/chroma_db/ folder and restart.

Issue: Ollama connection refused
Fix:   Run: ollama serve &
       Wait 5 seconds, then try again.

Issue: Celery "Cannot connect to Redis" error
Fix:   Run: docker start rag-redis
       Verify: docker exec rag-redis redis-cli ping → should return PONG

Issue: "No module named spacy"
Fix:   pip install spacy && python -m spacy download en_core_web_sm

Issue: RAGAS eval fails with import error
Fix:   pip install ragas datasets langchain-ollama langchain-community

Issue: BM25 returns empty results
Fix:   BM25 index must be built AFTER documents are ingested.
       Call build_bm25_index() in your startup or after ingestion.
```

---

Phases Completed

> Status reflects git history, not aspiration. Commits are labelled `Stage N`.
> Stage 0-3 = scaffold/config, parse+chunk, PII, retrieval primitives.
> Stage 4 = ingestion pipeline + hybrid retriever (RRF + CrossEncoder).
> Stage 5-9 = generation, API, async+Docker, CI+evals, observability+docs.

[x] Phase 1: Environment Setup
[x] Phase 2: Configuration (Pydantic Settings)
[x] Phase 3: Ingestion (Parser + Chunker; now also .md)
[x] Phase 4: Embeddings + ChromaDB (client cached, upsert)
[x] Phase 5: BM25 Keyword Search
[x] Phase 6: Hybrid Retrieval + RRF            → src/retrieval/retriever.py
[x] Phase 7: CrossEncoder Reranking            → src/retrieval/retriever.py
[x] Phase 8: LLM Generation + Versioned Prompts → prompts/rag_v1.yaml, src/generation/*
[x] Phase 9: Security — PII Masking (Presidio)
[x] Phase 10: FastAPI + Auth + Rate Limiting   → src/api/*
[x] Phase 11: Semantic Cache (Redis)           → src/api/cache.py (degrades gracefully)
[~] Phase 12: Observability — Prometheus + Grafana + JSON logs done;
              Arize Phoenix wired as optional/no-op (src/observability/tracing.py)
[x] Phase 13: Async Ingestion (Celery)         → src/ingestion/tasks.py
[~] Phase 14: Eval gate — local Llama judge (faithfulness + context precision) +
              GitHub Actions. RAGAS itself is disabled: v0.4.3 is incompatible
              with the installed langchain 1.x stack. See ARCHITECTURE.md §7.
[x] Phase 15: Tests — 89 unit + 7 integration (marked, Ollama-gated)

See ARCHITECTURE.md for the full component map, runtime topology, what is still
left, alternative tools, and the production-methodology self-review.

---

This CLAUDE.md is for use with Claude Code. Drop it at the project root and Claude Code will read it automatically at session start.
