# Architecture — Production RAG System

Local, $0-per-call Retrieval-Augmented Generation. Upload PDF / text / Markdown,
ask natural-language questions, get answers grounded **only** in the retrieved
passages, with source + page citations.

---

## 1. Component map

| Layer | Module | Tool | Why this tool |
|---|---|---|---|
| Config | `src/config.py` | pydantic-settings | Typed config, one source of truth, env/`.env` override |
| Parse | `src/ingestion/parser.py` | `pypdf` | Pure-Python, no native deps, fine for digital PDFs |
| Chunk | `src/ingestion/chunker.py` | hand-rolled sliding window | 35 lines, auditable, no LangChain tree |
| PII | `src/security/pii.py` | Microsoft Presidio + spaCy `en_core_web_sm` | De-facto OSS PII engine; small model keeps image lean |
| Embed | `src/retrieval/embedder.py` | `sentence-transformers` + `BAAI/bge-small-en-v1.5` | 384-d, strong MTEB retrieval, $0, query/doc asymmetry |
| Vector store | `src/retrieval/vectorstore.py` | ChromaDB (embedded, cosine HNSW) | Zero-ops, on-disk, good to ~1M chunks |
| Lexical | `src/retrieval/bm25.py` | `rank-bm25` (`BM25Okapi`) | Exact-token recall (codes, acronyms); pure Python |
| Fuse + rerank | `src/retrieval/retriever.py` | RRF + `cross-encoder/ms-marco-MiniLM-L-6-v2` | Merge incomparable scores by rank; precision pass |
| Prompt | `prompts/rag_v1.yaml` + `src/generation/prompt.py` | YAML | Prompt versioned & revertible independent of code |
| LLM | `src/generation/llm.py` | Ollama + Llama 3.2 | Local, $0/call, 4096 ctx |
| Orchestrate | `src/generation/generator.py` | — | retrieve → ground → generate → parse citations → refuse-safe |
| API | `src/api/main.py` | FastAPI + slowapi | Async, typed, OpenAPI, per-client rate limit |
| Auth | `src/api/auth.py` | `secrets.compare_digest` | Constant-time API-key check |
| Cache | `src/api/cache.py` | Redis + cosine over query embeddings | Skip generation for near-duplicate questions |
| Async ingest | `src/ingestion/tasks.py` | Celery + Redis | `task_acks_late` → no lost work on worker crash |
| Metrics | `src/observability/metrics.py` | prometheus-client | Counters + latency histograms on `/metrics` |
| Logs | `src/observability/logging.py` | stdlib + JSON formatter | One JSON line per event, request-id correlated |
| Tracing | `src/observability/tracing.py` | Arize Phoenix / OTel (optional) | No-op unless installed + endpoint set |
| Evals | `tests/evals/run_evals.py` | local Llama judge | Faithfulness + context-precision gate, no heavy deps |
| CI | `.github/workflows/` | GitHub Actions | Lint + unit gate on every PR; eval gate on retrieval/prompt changes |
| Deploy | `Dockerfile`, `docker/docker-compose.yml` | Docker | Multi-stage, non-root, models baked, healthcheck |

---

## 2. Ingestion flow

```
 PDF / .txt / .md
      │
      ▼  parser.parse_document           list[{page, text, source}]
      ▼  chunker.chunk_text              500-word windows, 100 overlap, deterministic chunk_id
      ▼  pii.mask_pii   (per chunk)      <PERSON> <EMAIL_ADDRESS> <PHONE_NUMBER> ... BEFORE embed/store
      ├─▶ embedder.embed_texts           384-d normalized vectors
      │        │
      ▼        ▼
  vectorstore.add_documents (UPSERT)     ChromaDB  ./data/chroma_db  (cosine)
      │
      ▼  bm25.rebuild_bm25_from_vectorstore   in-memory BM25 rebuilt from the store
      │
      ▼  IngestResult(source, pages, chunks, chunk_ids)
```

- **PII is a hard gate, not a filter.** Once a chunk is embedded and persisted the
  vector encodes its PII; redacting the source afterwards cannot undo that. The
  masked text is what gets embedded, stored, **and** BM25-indexed.
- **Idempotent.** `chunk_id = "<source>::chunk_<n>"` + `upsert` ⇒ re-ingesting a
  changed document overwrites its chunks in place.
- **Sync** (`POST /ingest`) or **async** (`POST /ingest?background=true` → Celery
  → `GET /tasks/{id}`).

## 3. Query flow

```
 question ──▶ [semantic cache] ──hit──▶ cached QueryResponse (cached=true)
                    │ miss
                    ▼
        generator.answer(question)
          │
          ├─ retriever.retrieve(question)
          │     ├─ embed_query ──▶ vector_search(top-20)   ─┐  ids + text + meta
          │     ├─ bm25_search(top-20) ─────────────────────┤  ids + score
          │     ├─ reciprocal_rank_fusion([vec, bm25], k=60)┤  fused {id: score}
          │     ├─ fetch_documents(bm25-only ids)           ┤  back-fill text
          │     └─ CrossEncoder.predict([(q, chunk)…]) ─────▶  top-5 RetrievedChunk (+ per-retriever ranks)
          │
          ├─ no chunks?  ──▶ deterministic REFUSAL (LLM never called)
          │
          ├─ prompt = load_prompt("rag_v1");  context = "[1] (src, p.N)\n<text>\n\n[2] …"
          ├─ llm.chat(system, user)   ──LLMError──▶ degraded fallback (never a 500 stack trace)
          ├─ refusal text?  ──▶ normalised REFUSAL
          └─ parse [n] markers, validate against shown passages ──▶ Citation[]
                    │
                    ▼
      QueryResponse{answer, citations[], contexts[], prompt_version, refused, degraded, cached, latency_ms}
                    │
                    └─ if answered: cache.store(question, response)
```

## 4. Runtime topology (docker-compose)

```
                 ┌────────────┐        scrape /metrics        ┌────────────┐
   client ──────▶│  api :8000 │◀──────────────────────────────│ prometheus │◀── grafana :3000
                 │  (FastAPI) │                                │   :9090    │
                 └─────┬──────┘                                └────────────┘
                       │ enqueue ingest            reads/writes
                       ▼                                   │
                 ┌────────────┐   broker + result   ┌──────┴─────┐   ┌──────────────────┐
                 │ redis :6379│◀───────────────────▶│  worker    │──▶│ chroma-data vol  │
                 │ cache+queue│                     │  (Celery)  │   │ (shared, sqlite) │
                 └────────────┘                     └─────┬──────┘   └──────────────────┘
                                                         │ chat/embeddings
                                                         ▼
                                            ┌───────────────────────────┐
                                            │ Ollama on host :11434       │
                                            │ (host.docker.internal)      │
                                            └───────────────────────────┘
```

---

## 5. What is still left / not done

Kept deliberately (not defects — the constraints are $0, local, single-node):

| Area | Gap | Impact | Note |
|---|---|---|---|
| Vector store concurrency | `api` + `worker` share one on-disk Chroma (sqlite). Concurrent writers can lock. | Medium at real multi-writer load | **Keeping Chroma** (per project decision). If it ever matters: `chromadb run` server mode + `HttpClient` — `vectorstore.py` is 4 functions, the swap is contained. |
| Evals | RAGAS replaced by a local-judge harness (dependency conflict, §7). Judge is Llama 3.2 — noisier than a frontier judge. | Low — gate still catches large regressions (0.80 / 0.73 last run) | Pin a compatible `langchain`+`ragas` set in a separate venv if standard metric names are wanted. |
| Frontend | No web UI | Cosmetic — `/docs` (Swagger) is usable | Any SPA can consume the typed OpenAPI schema. |
| Deploy target | compose only; no Kubernetes / Helm | Medium for scale-out | k8s Deployment + HPA on `rag_query_latency` / queue depth. |
| Auth | Single shared API key | OK for an internal tool | Per-tenant keys + scopes, or OIDC/JWT at a gateway. |
| Cache eviction | `lookup` does a full `SCAN` per query — O(n) in cached questions | Low at TTL-bounded scale | RediSearch vector index at large scale. |
| PDF coverage | No OCR; scanned PDFs yield empty text | Corpus-dependent | `unstructured` / `pytesseract` fallback when `extract_text()` is empty. |
| PII entities | No URL / `DOMAIN_NAME` recognizer | Low | Add a Presidio pattern recognizer for URLs. |

Done in Stages 10–14 (previously "left"):

| Area | Status |
|---|---|
| Docker | **Build-verified** — `production-rag:local` (3.13 GB), non-root, boots healthy, containerised `/query` returns grounded cited answers. CI `docker` job added. |
| Full compose stack | **Verified** — api + worker + redis + prometheus + grafana up; cache hit/miss, async Celery task SUCCESS, Prometheus target `up`, Grafana healthy. |
| Tracing | Real `arize-phoenix-otel` integration behind `requirements-observability.txt` + `docker/docker-compose.observability.yml` overlay (`make up-observability`). Base image stays lean. |
| Security | Security headers, 20 MB upload cap (413), path-traversal neutralised, `/health` info-leak fixed, non-ASCII-key → 401. `/security-review` run — no HIGH/MEDIUM findings. |
| Reproducible proof | `scripts/validate.py` (`make validate`) → `VALIDATION.md`. |

---

## 6. Alternative tools / features considered

| Current | Alternatives | When to switch |
|---|---|---|
| ChromaDB embedded | Qdrant, Weaviate, pgvector, Milvus, LanceDB | Need HA, filtered search at scale, or SQL-adjacent ops |
| `rank-bm25` in-memory | OpenSearch / Elasticsearch, Tantivy, `bm25s` | Corpus > ~1M chunks, or want persistence + analyzers |
| RRF fusion | Weighted score fusion, learned fusion, ColBERT late-interaction | Have labelled relevance data to tune weights |
| CrossEncoder MiniLM | `bge-reranker-v2-m3`, Cohere Rerank, `mxbai-rerank` | Need multilingual / higher ceiling and can spend latency |
| bge-small-en | `bge-base/large`, `nomic-embed`, `e5`, OpenAI `text-embedding-3` | Recall ceiling hit; willing to trade size/cost |
| Ollama + Llama 3.2 3B | Llama 3.1 8B, Qwen2.5, Mistral, hosted Claude/GPT | Answer quality ceiling; have GPU or budget |
| Local-judge evals | RAGAS (pinned), DeepEval, promptfoo, Phoenix evals | Want standard metric names / dashboards |
| Celery + Redis | RQ, Dramatiq, Arq, cloud queues (SQS/PubSub) | Simpler needs, or managed infra preference |
| Redis semantic cache | GPTCache, semantic-router, no cache | Need cross-node cache semantics or richer policies |
| prometheus-client | OpenTelemetry metrics + OTLP | Standardising on OTel end-to-end |
| Arize Phoenix (tracing) | Langfuse, LangSmith, Jaeger/Tempo + manual OTel | Want prompt management / datasets alongside traces |
| compose | Kubernetes + Helm/Kustomize, Nomad, ECS | Multi-node, autoscaling, rolling deploys |
| single shared API key | per-tenant keys + scopes, OIDC/JWT at a gateway | More than one consumer, or per-consumer quotas/audit |

Full reasoning for each row is in [`docs/WALKTHROUGH.md`](docs/WALKTHROUGH.md).
**We are not switching any of these** — the table is for when the constraints change.

---

## 7. Is the production methodology being followed?  (self-review, third-party lens)

**Followed:**
- **Separation of concerns** — every stage is one small module with a narrow
  surface; swapping any tool touches one file.
- **Config as data** — all tunables in `pydantic-settings`, env-overridable, no
  magic numbers in logic. Secrets only in `.env` (gitignored) / env vars.
- **Fail safe, degrade loud** — no context ⇒ deterministic refusal; LLM down ⇒
  `degraded` response, not a 500; Redis down ⇒ cache is a silent no-op;
  BM25-rebuild failure at startup is logged, not fatal.
- **Idempotency** — deterministic chunk ids + upsert; re-ingest is safe.
- **Privacy by construction** — PII masked *before* the irreversible step
  (embed/store), verified end-to-end by an integration test.
- **Testing pyramid** — ~97 fast unit tests (mocked I/O) form the base; 7
  `integration`-marked tests exercise real models/Ollama and skip when the host
  is absent; a separate eval gate guards answer quality.
- **Version control discipline** — one focused commit per stage; prompts
  versioned as data; `.gitattributes` normalises line endings.
- **Observability** — RED metrics as counters + histograms; JSON logs with
  request-id correlation; provisioned Grafana board; optional Phoenix tracing.
- **CI/CD** — lint + unit gate + image-build-and-boot on every PR; heavier eval
  gate scoped to quality-affecting paths; multi-stage non-root pinned image.
- **Security** — constant-time key check, security headers, upload cap,
  path-traversal neutralised, unauthenticated-endpoint info-leak fixed;
  `/security-review` run clean.
- **Supply chain** — deps pinned by floor + a lockable uv venv; model weights
  baked into the image for reproducible, offline-capable startup.
- **Reproducible proof** — `make validate` runs everything and writes
  `VALIDATION.md`.

**Not yet at production bar (tracked in §5, all deliberate for a $0 single-node
system):**
- Single-writer vector store under a multi-process compose (Chroma is kept by
  project decision; server mode is the escape hatch).
- No k8s / autoscaling / blue-green.
- Eval judge is a small local model (weaker than a frontier judge).
- Single shared API key.

**Verdict:** the system is production-*shaped* and now production-*verified* at
single-node scale — it builds, boots as non-root, answers grounded-and-cited,
degrades safely on every dependency failure, passes its own quality gate, and the
whole thing is reproducible with one command. The open items are genuine
scale-out concerns and one dependency pin, all tracked rather than hidden.
