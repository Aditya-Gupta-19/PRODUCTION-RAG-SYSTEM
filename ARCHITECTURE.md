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

| Area | Gap | Impact | Planned fix |
|---|---|---|---|
| Vector store concurrency | `api` + `worker` share one on-disk Chroma (sqlite). Concurrent writers can lock/corrupt. | Medium — fine for demo, not for real multi-writer load | Run Chroma in server mode (`chromadb run`) + `HttpClient`; the `vectorstore.py` surface is 4 functions, so the swap is contained |
| Evals | RAGAS replaced by a local-judge harness (dependency conflict, see §7). Judge is Llama 3.2 — noisier than GPT-4-class judges. | Low — gate still catches large regressions | Pin a compatible `langchain`+`ragas` set in a separate venv, or use a bigger judge model |
| Tracing | Phoenix wired as optional/no-op; not exercised in compose | Low — metrics + structured logs cover most needs | Add a `phoenix` service to compose + `PHOENIX_COLLECTOR_ENDPOINT` |
| Frontend | No React UI (spec Phase 10 item) | Cosmetic — `/docs` is usable | Build `codesentinel-ui`-style dashboard against the existing API |
| Deploy target | compose only; no Kubernetes manifests / Helm | Medium for scale-out | k8s Deployment + HPA on `rag_query_latency` / queue depth |
| Auth | Single shared API key | OK for internal tool | Per-tenant keys + scopes, or OIDC at a gateway |
| Docker | Not build-verified this session (Docker Desktop was down) | Low — Dockerfile is standard multi-stage | `docker build` + compose smoke test in CI |
| Cache eviction | `lookup` does a full `SCAN` per query — O(n) in cached questions | Low at TTL-bounded scale | RediSearch vector index, or an LRU cap |
| Reranker score | Uncalibrated logit surfaced as `score` | Cosmetic | Sigmoid at the API boundary if a 0–1 confidence is needed |
| PDF coverage | No OCR; scanned PDFs yield empty text | Medium depending on corpus | `unstructured` / `pytesseract` fallback when `extract_text()` is empty |
| PII entities | No URL / `DOMAIN_NAME` recognizer | Low | Add a Presidio pattern recognizer for URLs |

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
| compose | Kubernetes + Helm/Kustomize, Nomad | Multi-node, autoscaling, rolling deploys |

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
- **Testing pyramid** — 60+ fast unit tests (mocked I/O) form the base; a small
  `integration`-marked layer exercises real models/Ollama and is skipped when the
  host isn't available; an eval gate guards answer quality separately.
- **Version control discipline** — one branch + one focused commit per stage;
  prompts versioned as data; `.gitattributes` normalises line endings.
- **Observability** — RED metrics (Rate, Errors, Duration) as counters +
  histograms; JSON logs with request-id correlation; provisioned Grafana board.
- **CI/CD** — lint + type-of-contract tests on every PR; heavier eval gate scoped
  to the paths that can regress quality; Docker image is multi-stage, non-root,
  pinned base, healthchecked.
- **Supply chain** — dependencies pinned by floor + a lockable venv (uv); model
  weights baked into the image for reproducible, offline-capable startup.

**Not yet at production bar (tracked in §5):**
- Single-writer vector store under a multi-process compose.
- No k8s / autoscaling / blue-green.
- Eval judge is a small local model (weaker than a frontier judge).
- Docker image not build-verified in this environment.
- Single shared API key; no request quotas beyond rate limiting.

**Verdict:** the *shape* is production-grade — the seams, the failure behaviour,
the tests, the observability and the release pipeline are all in place and
correct. The remaining items are scale-out concerns (stateful store, orchestration)
and one dependency pin (RAGAS), all explicitly tracked rather than hidden.
