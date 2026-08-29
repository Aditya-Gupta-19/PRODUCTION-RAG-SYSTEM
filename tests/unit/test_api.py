import io

import pytest
from fastapi.testclient import TestClient

from src.api import main
from src.config import settings
from src.generation.generator import RagAnswer
from src.ingestion.pipeline import IngestResult

KEY = {"X-API-Key": settings.api_key}


@pytest.fixture(autouse=True)
def _fast_and_isolated(monkeypatch, tmp_path):
    main.limiter.enabled = False
    monkeypatch.setattr(main, "RAW_DOCS_DIR", tmp_path / "raw_docs")
    monkeypatch.setattr(main, "lookup", lambda q: None)
    monkeypatch.setattr(main, "store", lambda q, r: None)
    yield
    main.limiter.enabled = False


@pytest.fixture
def client():
    return TestClient(main.app)


def _answer(**kw):
    base = {
        "question": "q",
        "answer": "Because the sky scatters blue light [1].",
        "citations": [],
        "contexts": [{"marker": 1, "chunk_id": "c1", "source": "s.txt", "page": 1, "text": "..."}],
        "prompt_version": "rag_v1",
        "refused": False,
        "degraded": False,
    }
    base.update(kw)
    return RagAnswer(**base)


def test_health_is_reachable_without_a_key(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert "vector_store" in resp.json()["checks"]


def test_query_requires_api_key(client):
    assert client.post("/query", json={"question": "hi"}).status_code == 401


def test_query_returns_answer_shape(client, monkeypatch):
    monkeypatch.setattr(main, "run_rag", lambda q, rerank_top_n=None: _answer(question=q))
    resp = client.post("/query", json={"question": "why is the sky blue"}, headers=KEY)

    assert resp.status_code == 200
    body = resp.json()
    assert body["question"] == "why is the sky blue"
    assert body["cached"] is False
    assert body["prompt_version"] == "rag_v1"
    assert body["latency_ms"] >= 0


def test_query_serves_from_cache_without_calling_the_pipeline(client, monkeypatch):
    cached = {
        "question": "why is the sky blue",
        "answer": "cached answer [1].",
        "citations": [],
        "contexts": [],
        "prompt_version": "rag_v1",
        "refused": False,
        "degraded": False,
    }
    monkeypatch.setattr(main, "lookup", lambda q: cached)

    def _boom(*a, **k):
        raise AssertionError("pipeline must not run on a cache hit")

    monkeypatch.setattr(main, "run_rag", _boom)
    resp = client.post("/query", json={"question": "why is the sky blue"}, headers=KEY)

    assert resp.status_code == 200
    assert resp.json()["cached"] is True


def test_query_validates_empty_question(client):
    assert client.post("/query", json={"question": ""}, headers=KEY).status_code == 422


def test_ingest_requires_api_key(client):
    files = {"file": ("a.txt", io.BytesIO(b"hello"), "text/plain")}
    assert client.post("/ingest", files=files).status_code == 401


def test_ingest_rejects_unsupported_type(client):
    files = {"file": ("a.docx", io.BytesIO(b"hello"), "application/octet-stream")}
    resp = client.post("/ingest", files=files, headers=KEY)
    assert resp.status_code == 415


def test_ingest_sync_indexes_the_file(client, monkeypatch):
    monkeypatch.setattr(
        main,
        "ingest_document",
        lambda path: IngestResult(source="a.txt", pages=1, chunks=3, chunk_ids=["a::0", "a::1", "a::2"]),
    )
    files = {"file": ("a.txt", io.BytesIO(b"some text"), "text/plain")}
    resp = client.post("/ingest", files=files, headers=KEY)

    assert resp.status_code == 200
    body = resp.json()
    assert body["mode"] == "sync"
    assert body["chunks"] == 3
    assert body["task_id"] is None


def test_metrics_endpoint_exposes_custom_counters(client):
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert "rag_queries_total" in resp.text


def test_rate_limit_returns_429(client, monkeypatch):
    monkeypatch.setattr(main, "run_rag", lambda q, rerank_top_n=None: _answer())
    monkeypatch.setattr(settings, "rate_limit_per_minute", 2)
    main.limiter.enabled = True
    main.limiter.reset()

    codes = [client.post("/query", json={"question": "q"}, headers=KEY).status_code for _ in range(4)]
    assert codes.count(200) == 2
    assert 429 in codes
