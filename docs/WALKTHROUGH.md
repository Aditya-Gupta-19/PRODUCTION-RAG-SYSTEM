# Production RAG — Full Build Walkthrough (Stages 0–14)

This is the architect's narrative of the whole system: for every stage, **what
was built, why, which tool and why that tool, what problem it solves, the exact
steps/commands, the problems hit, and how to prove it works.**

Companion docs: [`ARCHITECTURE.md`](../ARCHITECTURE.md) (component map + topology),
[`README.md`](../README.md) (quickstart), [`VALIDATION.md`](../VALIDATION.md)
(the machine-generated proof run).

---

## The one-paragraph summary

A user uploads a PDF / text / Markdown file. The system splits it into
overlapping ~500-word chunks, strips PII from each chunk, turns each chunk into a
384-dimension vector with a local embedding model, and stores those vectors in a
local database plus a keyword index. When the user asks a question, the system
searches **both** indexes, merges the two ranked lists by position, re-scores the
survivors with a slower but sharper model, hands the top 5 chunks to a **local**
LLM, and returns an answer that is grounded only in those chunks — with a citation
(file + page) on every fact, or an honest "I don't know" if the chunks don't
contain the answer. Every model runs locally, so answering costs **$0**. Around
that core: an HTTP API with auth + rate limiting, a semantic cache, async
ingestion, Prometheus metrics, an automated answer-quality gate, and a Docker
deployment.

---

## Stage 0 — Scaffolding & configuration

**Goal:** a typed, single-source-of-truth config and a clean package layout
before any feature code.

**What was built:** `src/` package tree (`ingestion`, `retrieval`, `generation`,
`api`, `security`, `observability`), `tests/{unit,integration}`, `requirements.txt`,
`.env.example`, and `src/config.py`.

**Tool: `pydantic-settings`.** `Settings(BaseSettings)` reads every value from the
environment (or `.env`), coerces it to the declared type, and fails loudly at
startup if something required (`API_KEY`) is missing or mistyped.

```python
class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    llm_model: str = "llama3.2"
    chunk_size: int = 500
    api_key: str  # no default -> must be provided


settings = Settings()
```

**Why:** config drift ("it worked on my machine") is one of the most common
production failures. One typed object, imported everywhere, is the fix. No
feature module ever calls `os.getenv` directly.

**Concept — 12-factor config:** config that varies between deploys (dev / staging
/ prod) lives in the environment, not in code. Secrets never enter git (`.env` is
`.gitignore`d; `.env.example` is the checked-in template).

**Proof:**
```bash
python -c "from src.config import settings; print(settings.llm_model)"   # -> llama3.2
API_KEY= python -c "from src.config import settings"                     # -> ValidationError
pytest tests/unit/test_config.py -q                                      # env->field mapping
```

---

## Stage 1 — Ingestion front-half: parse & chunk

**Goal:** turn a file into a list of retrievable text chunks with page provenance.

### `parser.py`

**Tool: `pypdf`.** Pure-Python PDF reader. `PdfReader(path).pages` → per page
`page.extract_text()`. `.txt` / `.md` become a single synthetic "page 1".

```python
def parse_document(path):
    if suffix == ".pdf":  return [{"page": i+1, "text": p.extract_text() or "", "source": name} ...]
    if suffix in {".txt", ".md", ".markdown"}: return [{"page": 1, "text": read_text(), "source": name}]
    raise ValueError(...)
```

**Why pypdf:** no system libraries (works in a slim container), permissive
license, good enough for digitally-generated PDFs. *Limitation:* it does not OCR
scanned images — a documented trade-off.

### `chunker.py`

**Concept — why chunk at all.** An LLM has a fixed context window (Llama 3.2 here:
4096 tokens ≈ 3000 words). You cannot paste a 40-page document into the prompt.
And even if you could, retrieval works better on small focused passages than on
whole documents. So: split into pieces small enough that *several* fit in the
prompt alongside the question.

**What it does:** a sliding window over the word stream — 500-word windows, moved
forward 400 words each step, so **consecutive chunks share a 100-word overlap**.

```
words:  [w0 ................ w499][w400 ............... w899][w800 ...]
chunk0: |------ 500 words -------|
chunk1:                    |------ 500 words -------|
                           ^--- 100-word overlap ---^
```

**Why the overlap:** a fact that lands exactly on a window boundary ("...the
refund period is | 30 days...") would be cut in half and retrievable from
neither chunk. The 100-word overlap guarantees any ~sentence-length fact appears
whole in at least one chunk.

**Why 500 / 100 (not a library):** 500 words ≈ 650 tokens. Five chunks + question
+ system prompt ≈ 3500 tokens, comfortably inside 4096. A 35-line function is
auditable; LangChain's splitter drags in a large dependency tree for the same
logic. Each chunk gets a **deterministic id** `"<source>::chunk_<n>"` — this is
what makes re-ingesting a document idempotent later.

**Problem hit:** the loop's termination — an exact multiple of the step size
produced a tiny leftover chunk. Fixed with a boundary check + covered by
`test_exact_boundary_produces_no_short_tail`.

**Proof:** `pytest tests/unit/test_chunker.py -q` — overlap content equality,
page-range tracking, boundary cases, invalid-overlap rejection.

---

## Stage 2 — PII masking

**Goal:** sensitive entities must never reach the vector store.

**Tools: Microsoft Presidio (`presidio-analyzer` + `presidio-anonymizer`) +
spaCy `en_core_web_sm`.** The analyzer runs spaCy NER + regex/checksum
recognizers and returns typed spans with confidence scores; the anonymizer
replaces each span above threshold with `<ENTITY_TYPE>`.

```python
mask_pii("Reach John Smith at john@acme.com")
# -> "Reach <PERSON> at <EMAIL_ADDRESS>"
```

Entities masked: `PERSON, EMAIL_ADDRESS, PHONE_NUMBER, CREDIT_CARD, US_SSN,
IP_ADDRESS, LOCATION, IBAN_CODE`.

**Concept — why the *ordering* is a correctness property, not a detail.** Once a
chunk is embedded into a vector and persisted, that vector *encodes* whatever PII
the chunk contained. Deleting the source text afterwards does not un-encode the
vector. The **only** safe place to mask is in-flight, before `embed_texts()`.
This is "privacy by construction" — the unsafe state is unreachable, not merely
discouraged.

**Why `en_core_web_sm` (12 MB) over the default `en_core_web_lg` (~700 MB):** the
small model is accurate enough for these entity types and keeps the Docker image
lean. Pinned explicitly in an `NlpEngineProvider` config.

**Problem hit + fix (commit `a970549`):** Presidio's default `score_threshold=0`
lets its "very weak" (0.05-confidence) *bare-9-digit* US_SSN pattern through, so
every invoice / tracking / order number got masked as `<US_SSN>`. Raised the
threshold to `0.4` — keeps properly-delimited SSNs, emails, phones; drops the
standalone-digit guesses. Covered by
`test_bare_nine_digit_number_is_not_treated_as_ssn`.

**Known quirks (learned from red tests — see the memory note):**
`123-45-6789` is Presidio's denylisted canonical example and is never flagged
(use another fake). The email recognizer **validates the TLD** — `foo@bar.example`
(RFC-2606 reserved) is not recognized; fixtures must use `.com`. "WiFi" was once
tagged `<PERSON>` — false positives are the safe direction.

**Both models are `@lru_cache(maxsize=1)` singletons** — large objects, loaded
once per process.

**Proof:** `pytest tests/unit/test_pii.py -q`, and end-to-end
`tests/integration/test_ingest_and_retrieve.py::test_pii_is_masked_before_it_reaches_the_store`
which ingests a doc with a name/email/phone and reads the raw Chroma documents
back to assert the literals are gone.

---

## Stage 3 — Retrieval primitives: embed, store, keyword-index

### `embedder.py` — dense vectors

**Tool: `sentence-transformers` running `BAAI/bge-small-en-v1.5` locally.**
Encodes text → **384-dim L2-normalised** vector.

**Concept — embeddings.** A neural model maps text to a point in 384-space such
that *semantically similar* texts land near each other. "how do I get my money
back" and "refund policy" end up close even with zero shared words. Similarity =
cosine of the angle; because vectors are normalised, cosine = plain dot product.

**Subtlety — query/document asymmetry.** bge is trained so that *queries* get a
prefix (`"Represent this sentence for searching relevant passages: "`) and
*documents* do not. `embed_query()` adds it; `embed_texts()` does not. Getting
this wrong silently hurts recall.

### `vectorstore.py` — ChromaDB

**Tool: ChromaDB `PersistentClient`** — an embedded (in-process, on-disk) vector
database at `./data/chroma_db`. No server to run.

**Concept — ANN / HNSW.** Exact nearest-neighbour over millions of vectors is too
slow. Chroma uses **HNSW** (Hierarchical Navigable Small World) — a graph index
that finds *approximate* nearest neighbours in ~log time. The collection is
pinned to `metadata={"hnsw:space": "cosine"}` at creation — Chroma's unset
default is L2 (Euclidean), which is the wrong metric for normalised vectors and
fails silently.

**Why Chroma:** zero-ops, embedded, no separate service, good to ~1M chunks —
perfect for a $0 local-first system. (We are **keeping** Chroma; the alternatives
table below is reference only.)

### `bm25.py` — sparse / keyword index

**Tool: `rank-bm25` (`BM25Okapi`)** — pure-Python, in-memory.

**Concept — BM25.** A bag-of-words ranking function: a document scores high for a
query term if the term appears **often in that document** (term frequency) but
**rarely across the corpus** (inverse document frequency). It is exact-match:
"E-4021" retrieves the chunk containing literally "E-4021".

**Why have both dense and sparse:** they fail differently. Embeddings blur exact
tokens (error codes, SKUs, acronyms, names); BM25 misses paraphrases. Running
both hedges each one's blind spot.

**Operational constraint:** the BM25 index is **in-memory only**. It does not
survive a restart, so it is rebuilt from Chroma at every startup and after every
ingest (`rebuild_bm25_from_vectorstore()`).

**Problem hit (Stage 4):** `PersistentClient(...)` was being constructed on *every*
`vector_search` call. Fixed with an `@lru_cache` keyed on `(path, name)` — the
full test suite went 111s → 57s. Also switched `.add()` → `.upsert()` so
re-ingesting a document overwrites rather than throwing on duplicate ids.

**Proof:** `pytest tests/unit/test_embedder.py tests/unit/test_vectorstore.py
tests/unit/test_bm25.py -q` — 384-dim, normalisation, cosine-vs-magnitude, BM25
IDF edge cases.

---

## Stage 4 — The retrieval core: ingestion pipeline + hybrid retriever

**Goal:** wire the primitives into one `ingest_document()` and one `retrieve()`.

### `ingestion/pipeline.py`

```
parse_document -> chunk_text -> mask_pii(each) -> embed_texts -> upsert to Chroma -> rebuild BM25
```

Returns `IngestResult(source, pages, chunks, chunk_ids)`. Idempotent
(deterministic ids + upsert). One gotcha fixed here: the chunker emits
`page_range` as a **list**, and Chroma metadata values must be scalar — flattened
to `"1,2,3"` on the way in.

### `retrieval/retriever.py` — the three-stage funnel

```
query
  ├─ embed_query -> vector_search(top-20)   ─┐
  ├─ bm25_search(top-20) ────────────────────┤
  │                                          ▼
  │            reciprocal_rank_fusion([vec_ids, bm25_ids], k=60)
  │                                          │  fetch text for BM25-only ids
  │                                          ▼
  └────────────────▶ CrossEncoder.predict([(query, chunk)…]) -> top-5
```

**Concept — Reciprocal Rank Fusion (RRF).** Vector search returns cosine
distances (0–2, lower better); BM25 returns unbounded corpus-relative scores.
These are **not comparable**, and normalising them (min-max, z-score) is fragile.
RRF ignores the scores and uses only **rank position**:

```
score(doc) = Σ_retrievers  1 / (k + rank_in_that_retriever)      k = 60
```

A doc at rank 1 in both retrievers beats a doc at rank 1 in one and absent from
the other. `k=60` (from the 2009 Cormack paper) damps the top ranks so
cross-retriever *agreement* matters more than one retriever's confidence. This is
what Elasticsearch/Weaviate hybrid search use internally.

**Concept — cross-encoder reranking.** The embedding model is a *bi-encoder*: it
encodes query and document **separately**, so it never sees them together. A
**cross-encoder** (`cross-encoder/ms-marco-MiniLM-L-6-v2`, ~22M params) feeds
`[query] [SEP] [chunk]` through one transformer, so attention runs across both —
far more accurate at judging relevance, but too slow to run on the whole corpus.
So it only scores the ~20 candidates the funnel already narrowed to. **Funnel
shape: cheap-and-wide → merge → expensive-and-narrow.**

`RetrievedChunk` carries `vector_rank` / `bm25_rank` (which retriever found it,
at what position) — this feeds the metrics and the `/query` debug payload.

**Problem hit:** a chunk that BM25 ranked but vector search missed had no text to
give the reranker. Added `fetch_documents(ids)` to back-fill from Chroma.

**Proof:** `pytest tests/unit/test_retriever.py -q` (RRF arithmetic, stubbed
funnel) + `pytest -m integration tests/integration/test_ingest_and_retrieve.py`
(real models: semantic paraphrase retrieval, exact-token retrieval, idempotent
re-ingest).

---

## Stage 5 — Grounded generation

**Goal:** an answer grounded **only** in retrieved chunks, with citations, and an
honest refusal when the chunks don't contain the answer.

### `prompts/rag_v1.yaml` — the prompt as versioned data

```yaml
version: rag_v1
system: |
  Answer using ONLY the numbered CONTEXT passages. Cite every fact as [n].
  If the context is insufficient, reply exactly:
  "I don't have enough information in the provided documents to answer that."
user: |
  CONTEXT:
  {context}
  QUESTION: {question}
```

**Why a YAML file, not a Python string:** an LLM regression (more hallucination,
worse citations) is almost always a *prompt* problem. Keeping the prompt in its
own file means the fix is `git revert <hash> -- prompts/rag_v1.yaml` — no code
change, no Python redeploy, and `git log prompts/` is the complete history of
"how we ask the model". `load_prompt()` caches it per version.

### `generation/llm.py` — Ollama wrapper

**Tool: Ollama + Llama 3.2 (3B), local.** `ollama.Client(host=...).chat(...)`.
The 2 GB of model weights live in the Ollama server process, not the app.

```python
def chat(system, user, *, temperature=None):
    try:
        r = get_client().chat(model=settings.llm_model, messages=[...], options={...})
    except Exception as exc:
        raise LLMError(...) from exc  # one typed error for the caller
    return r.message.content or ""
```

**Why Ollama + a 3B model:** $0 per call, no data leaves the machine, 4096
context is enough for this funnel. (Quality ceiling is real — the alternatives
table covers when to move up.)

### `generation/generator.py` — the pipeline + three guarantees

```
answer(question):
  chunks = retrieve(question)
  contexts = [{marker, chunk_id, source, page, text} ...]     # ALL passages shown, for audit
  if not chunks:              return refusal                  # LLM never called
  raw = chat(system, render(context, question))               # LLMError -> degraded fallback
  if looks_like_refusal(raw): return refusal (normalised)
  citations = [Citation(marker, source, page, chunk_id) for n in [n]-markers if 1<=n<=len(chunks)]
  return RagAnswer(answer=raw, citations, contexts, prompt_version, refused=False, degraded=False)
```

1. **No retrieved context → deterministic refusal, the LLM is not called.** No
   wasted inference, no chance to invent an answer from nothing.
2. **`LLMError` → `degraded=True` fallback string**, never a 500 + stack trace.
3. **Citation validation** — `[7]` when only 3 passages were shown produces *no*
   citation, not a crash or a fake reference.

**Concept — grounding / faithfulness.** "Grounded" = every claim traces to a
retrieved passage. The prompt enforces it; the eval gate (Stage 8) *measures* it.

**Proof:** `pytest tests/unit/test_generator.py -q` (7 tests, LLM mocked:
citation parse, refusal paths, degraded path, out-of-range markers) +
`pytest -m integration tests/integration/test_generation.py` (real Ollama:
grounded question → cited "25 days"; off-topic question → refusal).

---

## Stage 6 — HTTP API

**Goal:** a deployable service with auth, rate limiting, a cache, and metrics.

**Tool: FastAPI (ASGI).** Routes: `GET /health`, `POST /query`, `POST /ingest`
(`?background=true`), `GET /tasks/{id}`, `GET /metrics`, `GET /docs`. Request and
response bodies are Pydantic models → automatic 422 on bad input + a generated
OpenAPI schema.

**Lifespan.** On startup: rebuild BM25 from Chroma (in-memory index doesn't
survive a restart — without this the first queries after a deploy have keyword
search silently off).

### Auth — `auth.py`

```python
if not secrets.compare_digest(x_api_key, settings.api_key):
    raise HTTPException(401, ...)
```

**Concept — timing attack.** A normal `==` on strings returns as soon as it hits
a mismatched byte, so an attacker measuring response time can recover the key one
byte at a time. `secrets.compare_digest` always compares the full length.

### Rate limiting — `slowapi`

```python
@limiter.limit(lambda: f"{settings.rate_limit_per_minute}/minute")
```

**Concept — token bucket, per client IP.** Each client gets N requests/minute;
over that → `429`. The limit is a *callable* so it re-reads config per request
(tunable without redeploy; testable). Proven: 4 requests at limit 2 →
`[200, 200, 429, 429]`.

### Semantic cache — `cache.py`

**Concept.** Before generating, embed the incoming question and compare (cosine =
dot product, vectors normalised) against every cached entry. Best match ≥ `0.92`
→ return the stored answer, `cached=true`, skip the LLM entirely. So "what's the
leave policy" and "how much annual leave do we get" hit the same cache entry.

**The headline property is graceful degradation:** `get_redis()` pings once and
returns `None` on any failure; every cache function is then a **silent no-op**.
Redis down ⇒ `/health` shows `"cache":"unavailable"` and queries still work.

*Known limitation (documented):* `lookup` does a full `SCAN` — O(n) in cached
questions. Fine under a TTL; RediSearch vector index at scale.

### Metrics — `observability/metrics.py` on `/metrics`

**Concept — the RED method:** **R**ate, **E**rrors, **D**uration.
`prometheus-fastapi-instrumentator` exposes per-route HTTP metrics; the custom
RAG counters/histograms (`rag_queries_total{outcome}`,
`rag_query_latency_seconds`, `rag_cache_lookups_total{result}`, …) share the same
registry and endpoint. Grafana's `histogram_quantile(0.95, ...)` over the
`_bucket` series gives p95 latency without storing every sample.

**Proof:** `pytest tests/unit/test_api.py tests/unit/test_auth.py
tests/unit/test_cache.py -q` (TestClient — auth 401, cache hit/miss, 415, 429,
`/metrics` contents). Live: see Stage 12 / VALIDATION.md.

---

## Stage 7 — Async ingestion + containerisation

### `ingestion/tasks.py` — Celery

**Concept — why async.** Ingesting a 200-page PDF = parse + chunk + *embed every
chunk* (model inference) + upsert. Seconds to minutes. It must not block the HTTP
request and must not be lost if the worker crashes.

**Celery reliability config, and what each line buys:**

| Setting | Effect |
|---|---|
| `task_acks_late=True` | broker removes the task only **after** it finishes → worker dies at 90%, task is redelivered, not lost |
| `task_reject_on_worker_lost=True` | hard kill (OOM/SIGKILL) requeues instead of failing |
| `worker_prefetch_multiplier=1` | a worker holds one task at a time → a slow ingest can't starve queued siblings |
| `autoretry_for=(Exception,)` + exp backoff, `max_retries=3` | transient Ollama/Chroma blips retry then give up cleanly |

**Concept — at-least-once delivery.** `acks_late` means a task can run twice (die
after the work, before the ack). That is only safe because **ingestion is
idempotent** (deterministic ids + upsert, Stage 4). The two design choices lock
together.

**Wiring:** `POST /ingest?background=true` → `.delay(path)` → returns `task_id`
immediately; `GET /tasks/{id}` → `AsyncResult` state.

### Docker — exact steps

**`Dockerfile` — multi-stage.**

*Stage `builder`* (`python:3.12-slim`):
1. `pip install uv` — fast resolver/installer.
2. `COPY requirements.txt requirements-observability.txt ./`
3. `uv pip install --system torch --index-url https://download.pytorch.org/whl/cpu`
   — **CPU-only torch**. The default torch wheel bundles CUDA libraries (~2 GB
   we'll never use on a CPU box). This one line is the biggest size win.
4. `uv pip install --system -r requirements.txt` then `python -m spacy download en_core_web_sm`.
5. `ARG INSTALL_OBSERVABILITY=0` + conditional `uv pip install -r requirements-observability.txt`
   — the heavy Phoenix tree only when the observability compose overlay asks for it.
6. `RUN python - <<'PY' ... SentenceTransformer(...) ; CrossEncoder(...) ... PY`
   — **bakes the model weights into the image** under `/models`. The container
   then starts deterministically and runs fully offline (`HF_HUB_OFFLINE=1`) — no
   first-request download, no dependency on HuggingFace being up.

*Stage `runtime`* (fresh `python:3.12-slim`):
7. `apt-get install libgomp1` — the OpenMP runtime torch/onnxruntime need on slim.
8. `COPY --from=builder` only `site-packages`, `/usr/local/bin`, `/models`, then
   `src/` + `prompts/`. The build toolchain (`uv`, pip caches, apt lists) is left
   behind → smaller, smaller attack surface.
9. `useradd --uid 10001 appuser` + `chown` + `USER appuser` — **runs as
   non-root**. A container escape lands as an unprivileged user.
10. `HEALTHCHECK` curls `/health`; `CMD ["uvicorn", "src.api.main:app", ...]`.

**Commands actually run this session:**
```bash
docker build -t production-rag:local .                     # ~11 min first time, layer-cached after
docker images production-rag:local                          # -> 3.13GB
docker run -d --name rag-test -p 8288:8000 \
  -e API_KEY=test-key-123 \
  -e OLLAMA_BASE_URL=http://host.docker.internal:11434 \
  --add-host host.docker.internal:host-gateway \
  production-rag:local
docker inspect --format '{{.State.Health.Status}}' rag-test # starting -> healthy (~78s: model load)
curl localhost:8288/health                                  # {"status":"ok",...}
curl -XPOST localhost:8288/ingest -H "X-API-Key: test-key-123" \
  -F "file=@tests/evals/fixtures/acme_handbook.md;type=text/markdown"
curl -XPOST localhost:8288/query  -H "X-API-Key: test-key-123" \
  -H 'Content-Type: application/json' -d '{"question":"how many sick days do employees get?"}'
# -> {"answer":"Employees are entitled to 10 paid sick days per year. [1]", "citations":[...]}
docker rm -f rag-test
```

**Problem hit:** the first container was built before Stage 12/13, so the
security headers and the 413 upload-cap weren't in it (a 30 MB upload → 500). The
image was rebuilt after Stage 12/13 and re-verified.

### `docker/docker-compose.yml` — the runtime topology

`api` + `worker` (share a Chroma volume) + `redis` (broker **and** cache) +
`prometheus` (scrapes `api:8000/metrics`) + `grafana` (provisioned datasource +
dashboard JSON). Ollama stays on the **host** (`host.docker.internal:11434`) so
2 GB of weights aren't copied into containers.

```bash
docker compose -f docker/docker-compose.yml up --build -d
docker compose -f docker/docker-compose.yml ps          # all services healthy
# Grafana  http://localhost:3000   Prometheus http://localhost:9090
docker compose -f docker/docker-compose.yml down
```

**Documented caveat:** `api` + `worker` share one on-disk Chroma (SQLite);
concurrent writers can lock. Fine for a single-node demo; the fix (Chroma server
mode) is contained to `vectorstore.py`'s 4 functions. **We are not switching** —
this is a noted limitation, not a blocker.

**Proof:** `pytest tests/unit/test_tasks.py -q` (reliability config, eager run,
failure propagation) + the container commands above + CI `docker` job.

---

## Stage 8 — CI + eval quality gate

### `.github/workflows/ci.yml` — three jobs

- **lint** — `uvx ruff check .` + `uvx ruff format --check .`
- **test** — `uv venv`, install, `spacy download`, `pytest -m "not integration"`,
  with `~/.cache/huggingface` cached so the ~150 MB of model weights download once.
- **docker** — `docker/build-push-action` (build, no push, GHA layer cache),
  then boot the container and assert `/health` returns 200.

**Concept — a CI gate.** A pull request cannot merge until lint + unit + image
build all pass. Regressions are caught *before* merge, mechanically, not by
someone remembering to check.

### `.github/workflows/eval_gate.yml` — the quality gate

Installs Ollama, `ollama pull llama3.2`, runs `python -m tests.evals.run_evals`.
Heavy (~30 min), so it's **scoped**: only on PRs touching `src/retrieval/**`,
`src/generation/**`, `prompts/**` (the things that can regress answer quality),
plus weekly and on-demand.

### `tests/evals/run_evals.py` — what "quality" means, measured

**Faithfulness** = *fraction of the answer's atomic claims that the retrieved
context supports.* The judge splits the answer into claims, then for each: "is
this supported by the context — YES/NO". Score = supported / total. **Below 0.70
= the model is inventing unsupported claims ~30% of the time → the PR fails.**

**Context precision** = *weighted precision@k over the retrieved passages in rank
order:* `Σ_k (precision@k · relevant_k) / total_relevant`. Rewards putting the
relevant chunks *first*.

Plus `answer_hit_rate` (did the expected fact appear) and `refusal_accuracy`
(refused iff unanswerable).

**Concept — LLM-as-judge.** The same local `llama3.2` grades the pipeline's
output. Cheap, fully local; noisier than a frontier judge (a documented
trade-off).

**Problem hit + third-party decision.** The spec said "RAGAS". But
`ragas==0.4.3` (the latest release) hard-imports
`langchain_community.chat_models.vertexai.ChatVertexAI` and
`langchain_openai.chat_models.AzureChatOpenAI` at module load — **neither exists
in the installed langchain 1.x stack**, so `import ragas` fails outright with no
runtime workaround. Following the spec literally would ship a broken gate. The
**correct** call (recorded in `ARCHITECTURE.md §7` and a memory note) was to
implement the *same two metrics* RAGAS's CI gate uses, with zero extra
dependencies and the local judge, and keep `dataset.json` in RAGAS-compatible
shape so RAGAS drops back in behind a pinned stack. This is the "$0, keep it
lean" principle winning over "follow the doc verbatim".

**Latest run (this session):** faithfulness **0.80**, context precision **0.73**,
answer-hit **1.00**, refusal accuracy **1.00**, 15/17 answered → **PASS**.

**Proof:** `pytest tests/unit/test_eval_harness.py -q` (metric arithmetic, judge
stubbed) + `python -m tests.evals.run_evals` (full run, needs Ollama).

---

## Stage 9 — Structured logging, optional tracing, docs

### `observability/logging.py`

`JsonFormatter` emits **one JSON object per line**:
`{ts, level, logger, msg, request_id, route, latency_ms, outcome}`. Directly
ingestible by Loki / CloudWatch / ELK; `grep | jq` in dev.

### Request-ID middleware (`api/main.py`)

Every request gets an `X-Request-ID` (from the header or generated), echoed on
the response, and emitted in one access-log line with route + latency + status —
so a slow or failing request is greppable end-to-end.

### `observability/tracing.py` — the "no-op unless configured" pattern

`setup_tracing()` returns `False` unless `PHOENIX_COLLECTOR_ENDPOINT` is set
**and** `arize-phoenix-otel` imports. `span(name, **attrs)` is a context manager:
a real OTel span when tracing is active, **nothing** otherwise. The generator
wraps `retrieve` and `generate` in `span(...)` unconditionally — **zero cost when
off**. Phoenix stays an *optional* dependency (`requirements-observability.txt`),
so the base install stays lean. See Stage 13.

**Proof:** `pytest tests/unit/test_tracing.py -q`.

---

## Stage 10 — Docker verified for real

Docker Desktop was down in earlier stages; this stage started it and actually
built + ran the image. Full command transcript is in Stage 7 above and in
`VALIDATION.md`. Outcomes:

- `docker build` → `production-rag:local`, **3.13 GB**, exit 0.
- Container reached `healthy` in ~78s (startup model load).
- Containerised `/query` returned a grounded, cited answer from the fixture doc.
- Added the CI `docker` job (build + boot + `/health` 200).

---

## Stage 11 — Reproducible validation harness (the proof)

**`scripts/validate.py`** (`make validate`) runs, records, and reports:

1. `ruff check` + `ruff format --check`  (required)
2. `pytest -m "not integration"`  — all pure-logic tests  (required)
3. `pytest -m integration -v`  — real models + **the full HTTP end-to-end**:
   `tests/integration/test_api_e2e.py` runs the assembled FastAPI app in-process
   (its lifespan does the BM25 rebuild), ingests
   `tests/evals/fixtures/acme_handbook.md`, asserts unauthenticated `/query` →
   401, asks *"How many paid annual leave days…"* and checks the answer contains
   "25" **and** carries a citation to `acme_handbook.md` **and**
   `refused/degraded/cached` are all false, checks an unanswerable question is
   refused, checks a 20 MB+ upload → 413, and scrapes `/metrics`.
   (required when Ollama is up)
4. `python -m tests.evals.run_evals`  (skipped if Ollama down, or `RAG_SKIP_EVALS=1`)

The app-in-process approach (vs. spawning `uvicorn`) is deliberate — it is
reliable and fast; that `uvicorn` itself boots is proved by the CI `docker` job
and the container transcript in Stage 7.

It writes **`VALIDATION.md`** — each step's status, duration and output tail — and
exits non-zero if any required step fails. That file *is* the proof; it is
committed so the repo always carries a current run.

**How you get the proof yourself:**
```bash
make install                       # once
ollama serve & ; ollama pull llama3.2   # for steps 3–4 (they skip cleanly if absent)
make validate                       # ~8 min (RAG_SKIP_EVALS=1) .. ~18 min (full)
cat VALIDATION.md ; echo $?         # the evidence; 0 = all required steps passed
```

---

## Stage 12 — Security hardening

| Hardening | What & why |
|---|---|
| **Security headers** on every response | `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`, `Referrer-Policy: no-referrer`, `Cache-Control: no-store` — via one middleware. Stops MIME-sniffing, clickjacking, referrer leakage, and caching of API responses. |
| **Upload size cap** (`MAX_UPLOAD_BYTES`, 20 MB) | `/ingest` streams the upload in 1 MB chunks and aborts with **413** past the cap, deleting the partial file. Without it, a 30 MB junk upload became thousands of chunks → embedding storm → **500** (seen in the Stage 10 container test). |
| **Path-traversal neutralised** | `Path(filename).name` strips every directory component, so `../../../etc/passwd.txt` can only land as `passwd.txt` inside `RAW_DOCS_DIR`. Test-pinned. |
| **`/health` no longer leaks internals** | it's unauthenticated; it now returns `"error"` per failing check and logs the exception (which can contain filesystem paths / internal URLs) instead of putting it in the response body. |
| **Question length bound** | `QueryRequest.question` `max_length=4000` → 422, before any work. |

**Proof:** `pytest tests/unit/test_security.py -q` (5 tests) + `/security-review`
run on the branch (Stage 13).

---

## Stage 13 — Optional Phoenix tracing + CI docker job

**`requirements-observability.txt`** — `arize-phoenix-otel` +
`openinference-instrumentation` (5 packages — far lighter than full
`arize-phoenix`, which drags in `strawberry-graphql`, `pydantic-ai`, `boto3`…).

**`docker/docker-compose.observability.yml`** — an *overlay*:
```bash
docker compose -f docker/docker-compose.yml \
               -f docker/docker-compose.observability.yml up --build
# Phoenix UI: http://localhost:6006  — one trace per /query with retrieve + generate spans
```
It adds a `phoenix` service, sets `PHOENIX_COLLECTOR_ENDPOINT` on `api`+`worker`,
and rebuilds the image with `INSTALL_OBSERVABILITY=1`.

**Why an overlay, not the base:** tracing is valuable but its dependency tree is
heavy. Opt-in keeps the default image lean ($0, fast). This is the same
"degrade gracefully / pay only for what you use" principle as the Redis cache.

**CI `docker` job** — builds the image on every PR (GHA layer cache) and asserts
the container boots and serves `/health`.

---

## Stage 14 — This document + final validation + merge

- `docs/WALKTHROUGH.md` (this file).
- `ARCHITECTURE.md`, `README.md`, `claude.md` reconciled with reality.
- Full suite + eval gate + container + `make validate` re-run green.
- `feat/rag-generation-to-deploy` merged to `main` with no conflicts.

---

# Alternative tools — reference only (we are NOT switching)

Every choice below is deliberate for a **$0, local-first, single-node** system.
This table is for *when the constraints change*.

| We use | Alternatives | Switch when… |
|---|---|---|
| **ChromaDB** (embedded) | Qdrant, Weaviate, Milvus, pgvector, LanceDB | you need HA / replication, metadata-filtered search at >1M vectors, or multi-writer without app coordination |
| **rank-bm25** (in-memory) | OpenSearch / Elasticsearch, Tantivy, `bm25s` | corpus > ~1M chunks, or you want a persistent index with language analyzers/stemming |
| **RRF fusion** | weighted score fusion, learned-to-rank, ColBERT late interaction | you have labelled relevance judgements to train a fusion model |
| **CrossEncoder MiniLM-L-6** | `bge-reranker-v2-m3`, `mxbai-rerank`, Cohere Rerank API | multilingual corpus, or you can spend more latency for a higher ceiling |
| **bge-small-en-v1.5** (384-d) | `bge-base`/`large`, `nomic-embed-text`, `e5-*`, OpenAI `text-embedding-3` | recall ceiling reached and you'll trade size/cost/privacy |
| **Ollama + Llama 3.2 3B** | Llama 3.1 8B, Qwen2.5, Mistral, hosted Claude/GPT | answer-quality ceiling; you have a GPU or an API budget |
| **local LLM-judge evals** | RAGAS (pinned stack), DeepEval, promptfoo, Phoenix evals | you want standard metric names / dashboards and can pin deps |
| **Celery + Redis** | RQ, Dramatiq, Arq, SQS/PubSub/Cloud Tasks | simpler needs (RQ/Arq), or you're already on managed cloud infra |
| **Redis semantic cache** | GPTCache, semantic-router, no cache | you need a cache shared/consistent across nodes with eviction policies |
| **prometheus-client** | OpenTelemetry metrics + OTLP collector | standardising the whole stack on OTel |
| **Arize Phoenix** (tracing) | Langfuse, LangSmith, Jaeger/Tempo + manual OTel | you want prompt management / datasets / annotations alongside traces |
| **docker-compose** | Kubernetes + Helm/Kustomize, Nomad, ECS | multi-node, autoscaling, rolling deploys, per-tenant isolation |
| **single shared API key** | per-tenant keys + scopes, OIDC/JWT at a gateway | more than one consumer, or you need per-consumer quotas/audit |

---

# How to prove the whole system works — copy/paste

```bash
# ── 0. one-time setup ────────────────────────────────────────────────
uv venv && uv pip install -r requirements.txt
uv run python -m spacy download en_core_web_sm
cp .env.example .env                       # then edit API_KEY
ollama serve &                             # separate terminal; then: ollama pull llama3.2

# ── 1. static checks + unit tests (no model host needed) ─────────────
uvx ruff check . && uvx ruff format --check .
uv run pytest -m "not integration" -q      # expect: ~90+ passed

# ── 2. real models end to end ───────────────────────────────────────
uv run pytest -m integration -q            # expect: 7 passed (needs Ollama)

# ── 3. answer-quality gate ─────────────────────────────────────────
uv run python -m tests.evals.run_evals     # expect: PASS, faithfulness ≥ 0.70

# ── 4. one command that does 1–3 + a live API call and writes proof ──
uv run python scripts/validate.py          # writes VALIDATION.md ; echo $?  -> 0
cat VALIDATION.md

# ── 5. the container ───────────────────────────────────────────────
docker build -t production-rag:local .
docker run -d --name rag -p 8000:8000 -e API_KEY=dev-key-change-in-production \
  -e OLLAMA_BASE_URL=http://host.docker.internal:11434 \
  --add-host host.docker.internal:host-gateway production-rag:local
until [ "$(docker inspect -f '{{.State.Health.Status}}' rag)" = healthy ]; do sleep 5; done
curl -s localhost:8000/health
curl -s -XPOST localhost:8000/ingest -H "X-API-Key: dev-key-change-in-production" \
  -F "file=@tests/evals/fixtures/acme_handbook.md;type=text/markdown"
curl -s -XPOST localhost:8000/query -H "X-API-Key: dev-key-change-in-production" \
  -H 'Content-Type: application/json' \
  -d '{"question":"What is the international daily meal allowance?"}'
# expect: {"answer":"... 90 dollars international. [1]", "citations":[{...}], "refused":false}
docker rm -f rag

# ── 6. full stack with dashboards ─────────────────────────────────
docker compose -f docker/docker-compose.yml up --build -d
open http://localhost:3000     # Grafana → "RAG — Overview" dashboard
docker compose -f docker/docker-compose.yml down
```

**What each step proves:**
- 1 → code style consistent, all pure logic correct (retrieval math, citation
  parsing, auth, cache, security, Celery config).
- 2 → the real embedding/rerank/Ollama pipeline retrieves the right passage and
  the LLM produces a grounded, cited answer *and* refuses when it should.
- 3 → answer quality is above threshold, measured not assumed.
- 4 → all of the above **plus** the assembled HTTP service works, in one
  reproducible artefact (`VALIDATION.md`).
- 5 → the exact image that would deploy builds, boots as non-root, and answers.
- 6 → the multi-service topology (queue, cache, metrics, dashboards) comes up.
