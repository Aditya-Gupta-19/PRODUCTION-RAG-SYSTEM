"""End-to-end HTTP smoke: the assembled FastAPI app, real retriever, real Ollama.

Marked ``integration``; auto-skips when Ollama is not serving. Runs the app
in-process via TestClient (its lifespan does the BM25 rebuild) — no server
subprocess, so it is reliable. That uvicorn itself boots is covered by the CI
``docker`` job and the container run in docs/WALKTHROUGH.md.
"""

from pathlib import Path

import httpx
import pytest

from src.config import settings
from src.retrieval import bm25

FIXTURE = "tests/evals/fixtures/acme_handbook.md"


def _ollama_up() -> bool:
    try:
        httpx.get(f"{settings.ollama_base_url}/api/tags", timeout=2.0).raise_for_status()
        return True
    except Exception:
        return False


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not _ollama_up(), reason="Ollama is not serving"),
]


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "chroma_path", str(tmp_path / "chroma"))
    monkeypatch.setattr(settings, "chroma_collection", "api_e2e")
    monkeypatch.setattr(bm25, "_index", bm25.BM25Index())

    from fastapi.testclient import TestClient

    from src.api import main

    monkeypatch.setattr(main, "RAW_DOCS_DIR", tmp_path / "raw_docs")
    main.limiter.enabled = False
    with TestClient(main.app) as c:
        yield c


KEY = {"X-API-Key": settings.api_key}


def test_health_ok_and_carries_security_headers(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.headers["x-content-type-options"] == "nosniff"
    assert resp.headers["x-frame-options"] == "DENY"
    assert "x-request-id" in resp.headers


def test_full_flow_ingest_then_grounded_cited_answer(client):
    assert client.post("/query", json={"question": "hi"}).status_code == 401

    doc = Path(FIXTURE).read_bytes()
    ing = client.post("/ingest", headers=KEY, files={"file": ("acme_handbook.md", doc, "text/markdown")})
    assert ing.status_code == 200
    assert ing.json()["chunks"] >= 1

    r = client.post(
        "/query",
        headers=KEY,
        json={"question": "How many paid annual leave days do full-time employees get?"},
    )
    assert r.status_code == 200
    body = r.json()
    assert "25" in body["answer"]
    assert body["citations"] and body["citations"][0]["source"] == "acme_handbook.md"
    assert body["refused"] is False and body["degraded"] is False
    assert body["cached"] is False

    metrics = client.get("/metrics").text
    assert 'rag_queries_total{outcome="answered"}' in metrics


def test_oversized_upload_is_rejected(client):
    big = b"x" * (settings.max_upload_bytes + 1)
    resp = client.post("/ingest", headers=KEY, files={"file": ("big.txt", big, "text/plain")})
    assert resp.status_code == 413


def test_unanswerable_question_is_refused(client):
    doc = Path(FIXTURE).read_bytes()
    client.post("/ingest", headers=KEY, files={"file": ("acme_handbook.md", doc, "text/markdown")})
    r = client.post("/query", headers=KEY, json={"question": "What is the CEO's personal phone number?"})
    assert r.status_code == 200
    assert r.json()["refused"] is True
