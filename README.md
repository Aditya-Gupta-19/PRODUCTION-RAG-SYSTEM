# Production RAG System

## What This Is

A production-grade Retrieval-Augmented Generation (RAG) system. Users upload PDF or text documents, the system indexes them, and answers natural-language questions about their content with cited sources — page numbers and originating file included, not just a bare answer.

The defining constraint of this project is that the entire stack runs **locally, at $0 marginal cost per call**: local LLM (Ollama + Llama 3.2), local embeddings, local reranking, local vector storage. No API keys to a hosted model provider are required for the system to function. What's built on top of that local core is what makes it "production-grade" rather than a weekend script: hybrid retrieval, PII-safe ingestion, semantic caching, async processing, observability, and an automated quality gate that blocks regressions.

## How It Works

There are two flows through the system: **ingestion** (getting a document in) and **query** (getting an answer out).

### Ingestion

```
PDF / text file
   │
   ├─ parse            → raw text, per page
   ├─ chunk            → overlapping ~500-word windows (100-word overlap)
   ├─ mask PII         → Presidio strips names, emails, phone numbers, etc.
   ├─ embed            → each chunk → 384-dim vector (BAAI/bge-small-en-v1.5)
   └─ index            → vector → ChromaDB, text → BM25 keyword index
```

PII masking happens **before** anything is embedded or stored. This ordering is deliberate: once sensitive text is embedded into a vector and persisted, redacting the source text afterward doesn't undo the fact that the vector store now encodes it. The only safe point to strip PII is in-flight, before indexing.

### Query

```
Question
   │
   ├─ embed query
   ├─ vector search (top 20, semantic)  ─┐
   ├─ BM25 search   (top 20, keyword)   ─┼─ Reciprocal Rank Fusion → merged ranking
   ├─ CrossEncoder rerank (top 20 → top 5, precision pass)
   └─ LLM generates an answer grounded only in those 5 chunks, with citations
```

A semantic cache sits in front of the LLM call: if a sufficiently similar question has been answered recently, the cached answer is returned instead of re-running generation.

## Implementation

**Retrieval is a three-stage funnel, not a single search.** Vector search alone misses exact keyword or acronym matches (e.g. a product code or a specific error string); BM25 alone misses semantic paraphrases (asking "how do I get my money back" when the document says "refund policy"). Running both in parallel casts a wide net that hedges against either retriever's blind spot. The two ranked lists — differently scored and not directly comparable (cosine distance vs. BM25 score) — are merged with **Reciprocal Rank Fusion**, `score = Σ 1/(k + rank)`, which combines rankings by position rather than by raw score, sidestepping the normalization problem entirely. RRF is a cheap heuristic, though, so a **CrossEncoder** — a model that reads the query and each candidate chunk jointly, rather than comparing precomputed embeddings — makes the final precision pass, scoring only the narrowed-down top 20 and returning the top 5. This funnel shape (cheap-and-wide → merge → expensive-and-narrow) is what makes strong retrieval affordable.

**Chunking** uses 500-word windows with 100-word overlap — sized empirically against Llama 3.2's context window so that several chunks plus the question plus the system prompt fit comfortably, while the overlap prevents a fact from being severed exactly at a chunk boundary.

**PII masking is a hard ingestion gate**, not a display-time filter — see above. It uses Microsoft Presidio's entity recognizers (spaCy-backed) to detect and redact names, emails, phone numbers, and other sensitive entities from every chunk before it reaches the embedder.

**Prompts are version-controlled YAML files, tracked independently of application code.** A prompt regression — the LLM starting to hallucinate more, or citing sources incorrectly — is diagnosed and rolled back with `git log` / `git revert` on a single YAML file, without touching or redeploying the Python codebase.

**Async ingestion runs through Celery with `task_acks_late=True`.** A task is only marked complete once it actually finishes; if the worker process dies mid-ingestion, the task is automatically re-queued rather than silently lost. This matters specifically for document ingestion, which can be slow (large PDFs, model inference for embeddings) and shouldn't block the API or lose work on a crash.

**Every model load is cached (`@lru_cache(maxsize=1)`).** The embedding model, reranker, and PII analyzer are all large objects with real load latency; the cache ensures each is loaded once per process and reused across every request, rather than reloaded per call.

**An automated quality gate protects answer quality.** RAGAS evaluates faithfulness (does the answer actually follow from the retrieved context, or is the model making things up?) and context precision (are the retrieved chunks actually relevant?) against a fixed Q&A dataset, using a local Ollama model as the judge. Faithfulness below 0.70 means the system is fabricating unsupported claims roughly 30% of the time — the threshold exists to fail a pull request automatically rather than let that regression reach users.

**Observability and security wrap the whole pipeline rather than being bolted onto one part of it:** Prometheus/Grafana track latency and volume metrics, Arize Phoenix traces individual LLM calls, X-API-Key auth and per-client rate limiting guard the API surface, and the Redis semantic cache reduces both cost and latency for repeated or near-duplicate questions.

## Advantages

- **Zero marginal cost.** Every model in the pipeline — LLM, embeddings, reranker, PII detection — runs locally. Answering one question or ten thousand costs the same $0 in API fees.
- **Data never leaves the machine, and PII never reaches storage.** No document content or query is sent to a third-party API, and Presidio strips sensitive entities before indexing, not after.
- **Retrieval quality from combining two different strengths.** Hybrid (semantic + keyword) search with RRF fusion and CrossEncoder reranking outperforms either retrieval method alone, especially on real-world queries that mix exact terms with paraphrased intent.
- **Regressions are caught automatically, not by eyeballing outputs.** The RAGAS-based CI gate fails a pull request if answer faithfulness or retrieval precision drops below threshold — quality is measured, not assumed.
- **Answers are auditable.** Every response cites its source document and page, and every prompt change is tracked and revertible independently of code.
- **Resilient by design.** Async ingestion with late acknowledgment means a crashed worker doesn't silently drop a document; the semantic cache and rate limiting protect the system under repeated or bursty load.

## What I Learned

Building this project meant working through the full width of what separates a RAG prototype from something closer to production:

- **Hybrid retrieval design** — why semantic and keyword search fail differently, and how rank-fusion (RRF) lets you merge incomparable ranking signals without ad hoc score normalization.
- **The role of reranking** — the tradeoff between cheap-and-approximate retrieval and expensive-and-accurate scoring, and why you fan out wide before narrowing precisely.
- **Evaluation-driven LLM development** — treating answer quality as a metric to gate on (RAGAS faithfulness/precision) rather than a subjective impression, and wiring that into CI so regressions are caught before merge, not after deployment.
- **Privacy-by-construction in a data pipeline** — why *where* in the pipeline a safeguard runs (before vs. after indexing) is itself a correctness property, not just an implementation detail.
- **Operating a fully local LLM stack** — the practical differences between calling a hosted model API and running inference, embedding, and reranking models locally: latency characteristics, resource management, and the `lru_cache`-once-load pattern for expensive model objects.
- **Production concerns beyond the model call itself** — async task processing with failure recovery (Celery `task_acks_late`), observability (metrics and tracing, not just logs), API security (auth, rate limiting), and versioning prompts as first-class, revertible artifacts alongside code.

## Running It

```bash
# 0. prerequisites: Ollama serving llama3.2, Docker for Redis
ollama serve &            # then: ollama pull llama3.2

# 1. install
uv venv && uv pip install -r requirements.txt
uv run python -m spacy download en_core_web_sm
cp .env.example .env      # set API_KEY

# 2. run the API
uv run uvicorn src.api.main:app --reload --port 8000     # http://localhost:8000/docs

# 3. (optional) async ingestion worker + infra
docker run -d -p 6379:6379 --name rag-redis redis:7-alpine
uv run celery -A src.ingestion.tasks.celery_app worker --pool=solo

# 4. full stack with metrics dashboards (Grafana :3000, Prometheus :9090)
docker compose -f docker/docker-compose.yml up --build
#    + LLM tracing (Phoenix :6006):
make up-observability

# 5. use it
curl -s localhost:8000/health
curl -s -XPOST localhost:8000/ingest -H "X-API-Key: $API_KEY" -F "file=@handbook.pdf"
curl -s -XPOST localhost:8000/query  -H "X-API-Key: $API_KEY" \
  -H 'Content-Type: application/json' -d '{"question":"what is the leave policy?"}'
```

Endpoints: `GET /health` · `POST /ingest` (`?background=true` for async) ·
`GET /tasks/{id}` · `POST /query` · `GET /metrics` · `GET /docs`.

**Prove it works:** `make validate` runs lint + unit + integration + eval gate +
a live API call and writes [`VALIDATION.md`](VALIDATION.md).
Individually: `make test` · `make test-int` (needs Ollama) · `make evals` ·
`make docker-build`.

Docs: [ARCHITECTURE.md](ARCHITECTURE.md) (topology, gaps, tool alternatives) ·
[docs/WALKTHROUGH.md](docs/WALKTHROUGH.md) (every stage explained, with commands
and proof steps).

## Project Structure

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
│   │   ├── prompt.py              # Loads/renders versioned YAML prompt
│   │   ├── llm.py                 # Ollama chat wrapper (raises LLMError)
│   │   └── generator.py           # Full RAG pipeline + citation parsing + refusal
│   ├── api/
│   │   ├── main.py                # FastAPI app: /health /ingest /query /tasks /metrics
│   │   ├── models.py              # Pydantic request/response models
│   │   ├── auth.py                # X-API-Key header verification (constant-time)
│   │   └── cache.py               # Redis semantic cache (degrades gracefully)
│   ├── security/
│   │   └── pii.py                 # Presidio PII detection + masking
│   ├── ingestion/
│   │   ├── pipeline.py            # ingest_document(): parse→chunk→mask→embed→index
│   │   └── tasks.py               # Celery async ingestion (task_acks_late)
│   └── observability/
│       ├── metrics.py             # Prometheus counters/histograms
│       ├── logging.py             # JSON structured logging + request-id
│       └── tracing.py             # Optional Arize Phoenix / OTel (no-op if absent)
├── tests/
│   ├── unit/                      # Fast isolated tests (mocked I/O)
│   ├── integration/               # Real models / Ollama (marker: integration)
│   └── evals/
│       ├── dataset.json           # 17 Q&A pairs (15 answerable + 2 refusal)
│       ├── fixtures/acme_handbook.md
│       └── run_evals.py           # Local-judge faithfulness + context-precision gate
├── docker/
│   ├── docker-compose.yml         # api + worker + Redis + Prometheus + Grafana
│   ├── prometheus.yml
│   └── grafana/provisioning/      # datasource + RAG overview dashboard
├── .github/workflows/
│   ├── ci.yml                     # lint + unit gate (every PR)
│   └── eval_gate.yml              # quality gate (retrieval/prompt changes + weekly)
├── prompts/rag_v1.yaml            # Versioned prompt template
├── Dockerfile                     # Multi-stage, non-root, models baked in
├── Makefile
├── ARCHITECTURE.md
├── .env / .env.example
└── requirements.txt
```
