from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    rerank_top_n: int | None = Field(default=None, ge=1, le=20)


class CitationModel(BaseModel):
    marker: int
    source: str
    page: int | None = None
    chunk_id: str


class ContextModel(BaseModel):
    marker: int
    chunk_id: str
    source: str | None = None
    page: int | None = None
    text: str


class QueryResponse(BaseModel):
    question: str
    answer: str
    citations: list[CitationModel]
    contexts: list[ContextModel]
    prompt_version: str
    refused: bool
    degraded: bool
    cached: bool
    latency_ms: float


class IngestResponse(BaseModel):
    source: str
    pages: int
    chunks: int
    chunk_ids: list[str]
    mode: str  # "sync" or "async"
    task_id: str | None = None  # set when mode == "async"


class TaskStatusResponse(BaseModel):
    task_id: str
    state: str  # PENDING | STARTED | SUCCESS | FAILURE | RETRY
    result: dict | None = None


class HealthResponse(BaseModel):
    status: str  # "ok" | "degraded"
    checks: dict[str, str]
