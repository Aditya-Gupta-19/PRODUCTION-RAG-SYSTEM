import logging
import shutil
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import Depends, FastAPI, File, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from src.api.auth import require_api_key
from src.api.cache import get_redis, lookup, store
from src.api.models import (
    HealthResponse,
    IngestResponse,
    QueryRequest,
    QueryResponse,
    TaskStatusResponse,
)
from src.config import REPO_ROOT, settings
from src.generation.generator import answer as run_rag
from src.ingestion.pipeline import ingest_document
from src.observability import metrics
from src.observability.logging import configure_logging
from src.observability.tracing import setup_tracing
from src.retrieval.bm25 import rebuild_bm25_from_vectorstore
from src.retrieval.vectorstore import get_collection

configure_logging()
logger = logging.getLogger("rag.api")

RAW_DOCS_DIR = REPO_ROOT / "data" / "raw_docs"
ALLOWED_SUFFIXES = {".pdf", ".txt", ".md", ".markdown"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Rebuild the in-memory BM25 index from whatever is already persisted in
    # Chroma — it does not survive a restart. Best-effort: a fresh deployment
    # with an empty store is fine.
    try:
        rebuild_bm25_from_vectorstore()
        logger.info("BM25 index rebuilt from vector store on startup")
    except Exception:  # pragma: no cover - defensive
        logger.exception("BM25 rebuild on startup failed; continuing")
    if setup_tracing():  # pragma: no cover - only when phoenix is installed
        logger.info("Phoenix/OTel tracing enabled")
    yield


limiter = Limiter(key_func=get_remote_address, default_limits=[f"{settings.rate_limit_per_minute}/minute"])

app = FastAPI(title="Production RAG System", version="1.0.0", lifespan=lifespan)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in settings.cors_allowed_origins.split(",") if o.strip()],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID", uuid.uuid4().hex[:12])
    started = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    logger.info(
        "request",
        extra={
            "request_id": request_id,
            "route": request.url.path,
            "latency_ms": round((time.perf_counter() - started) * 1000, 1),
            "outcome": str(response.status_code),
        },
    )
    return response


# /metrics — built-in HTTP metrics + everything registered in src/observability/metrics.py
Instrumentator().instrument(app).expose(app, include_in_schema=False)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    checks: dict[str, str] = {}

    try:
        get_collection().count()
        checks["vector_store"] = "ok"
    except Exception as exc:
        checks["vector_store"] = f"error: {exc}"

    checks["cache"] = "ok" if get_redis() is not None else "unavailable"

    try:
        httpx.get(f"{settings.ollama_base_url}/api/tags", timeout=2.0).raise_for_status()
        checks["llm"] = "ok"
    except Exception as exc:
        checks["llm"] = f"error: {exc}"

    status = "ok" if checks.get("vector_store") == "ok" and checks.get("llm") == "ok" else "degraded"
    return HealthResponse(status=status, checks=checks)


@app.post("/query", response_model=QueryResponse, dependencies=[Depends(require_api_key)])
@limiter.limit(lambda: f"{settings.rate_limit_per_minute}/minute")
def query(request: Request, body: QueryRequest) -> QueryResponse:
    started = time.perf_counter()

    cached = lookup(body.question)
    if cached is not None:
        metrics.CACHE_LOOKUPS.labels(result="hit").inc()
        metrics.QUERIES.labels(outcome="cached").inc()
        cached = {**cached, "cached": True, "latency_ms": round((time.perf_counter() - started) * 1000, 1)}
        return QueryResponse(**cached)

    metrics.CACHE_LOOKUPS.labels(result="miss" if get_redis() is not None else "disabled").inc()

    result = run_rag(body.question, rerank_top_n=body.rerank_top_n)

    outcome = "degraded" if result.degraded else "refused" if result.refused else "answered"
    metrics.QUERIES.labels(outcome=outcome).inc()

    response = QueryResponse(
        question=result.question,
        answer=result.answer,
        citations=[c.__dict__ for c in result.citations],
        contexts=result.contexts,
        prompt_version=result.prompt_version,
        refused=result.refused,
        degraded=result.degraded,
        cached=False,
        latency_ms=round((time.perf_counter() - started) * 1000, 1),
    )
    metrics.QUERY_LATENCY.observe(time.perf_counter() - started)

    if outcome == "answered":
        store(body.question, response.model_dump(exclude={"cached", "latency_ms"}))
    return response


@app.post("/ingest", response_model=IngestResponse, dependencies=[Depends(require_api_key)])
@limiter.limit(lambda: f"{settings.rate_limit_per_minute}/minute")
def ingest(request: Request, file: UploadFile = File(...), background: bool = False) -> IngestResponse:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        from fastapi import HTTPException

        raise HTTPException(status_code=415, detail=f"Unsupported file type {suffix!r}; allowed: .pdf .txt")

    RAW_DOCS_DIR.mkdir(parents=True, exist_ok=True)
    dest = RAW_DOCS_DIR / Path(file.filename).name
    with dest.open("wb") as out:
        shutil.copyfileobj(file.file, out)

    if background:
        from src.ingestion.tasks import ingest_document_task

        task = ingest_document_task.delay(str(dest))
        metrics.INGESTS.labels(mode="async").inc()
        return IngestResponse(
            source=dest.name, pages=0, chunks=0, chunk_ids=[], mode="async", task_id=task.id
        )

    result = ingest_document(dest)
    metrics.INGESTS.labels(mode="sync").inc()
    metrics.INGEST_CHUNKS.inc(result.chunks)
    return IngestResponse(
        source=result.source,
        pages=result.pages,
        chunks=result.chunks,
        chunk_ids=result.chunk_ids,
        mode="sync",
    )


@app.get("/tasks/{task_id}", response_model=TaskStatusResponse, dependencies=[Depends(require_api_key)])
def task_status(task_id: str) -> TaskStatusResponse:
    from celery.result import AsyncResult

    from src.ingestion.tasks import celery_app

    res = AsyncResult(task_id, app=celery_app)
    return TaskStatusResponse(
        task_id=task_id,
        state=res.state,
        result=res.result if isinstance(res.result, dict) else None,
    )
