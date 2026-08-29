import io

import pytest
from fastapi.testclient import TestClient

from src.api import main
from src.config import settings
from src.ingestion.pipeline import IngestResult

KEY = {"X-API-Key": settings.api_key}


@pytest.fixture(autouse=True)
def _isolated(monkeypatch, tmp_path):
    main.limiter.enabled = False
    monkeypatch.setattr(main, "RAW_DOCS_DIR", tmp_path / "raw_docs")
    monkeypatch.setattr(main, "lookup", lambda q: None)
    monkeypatch.setattr(main, "store", lambda q, r: None)


@pytest.fixture
def client():
    return TestClient(main.app)


def test_security_headers_present_on_every_response(client):
    resp = client.get("/health")
    assert resp.headers["X-Content-Type-Options"] == "nosniff"
    assert resp.headers["X-Frame-Options"] == "DENY"
    assert resp.headers["Referrer-Policy"] == "no-referrer"
    assert resp.headers["Cache-Control"] == "no-store"


def test_health_does_not_leak_exception_detail(client, monkeypatch):
    def _boom():
        raise RuntimeError("/secret/path/chroma.sqlite is locked")

    monkeypatch.setattr(main, "get_collection", lambda: type("C", (), {"count": staticmethod(_boom)})())
    body = client.get("/health").json()
    assert body["checks"]["vector_store"] == "error"
    assert "secret" not in str(body)


def test_ingest_rejects_oversized_upload(client, monkeypatch):
    monkeypatch.setattr(settings, "max_upload_bytes", 1024)
    big = io.BytesIO(b"x" * 4096)
    resp = client.post("/ingest", files={"file": ("big.txt", big, "text/plain")}, headers=KEY)
    assert resp.status_code == 413


def test_ingest_path_traversal_filename_is_neutralised(client, monkeypatch):
    seen = {}

    def _capture(path):
        seen["path"] = str(path)
        return IngestResult(source="passwd.txt", pages=1, chunks=0, chunk_ids=[])

    monkeypatch.setattr(main, "ingest_document", _capture)
    files = {"file": ("../../../etc/passwd.txt", io.BytesIO(b"data"), "text/plain")}
    resp = client.post("/ingest", files=files, headers=KEY)

    assert resp.status_code == 200
    assert "etc" not in seen["path"]
    assert seen["path"].replace("\\", "/").endswith("/passwd.txt")


def test_query_rejects_overlong_question(client):
    resp = client.post("/query", json={"question": "a" * 5000}, headers=KEY)
    assert resp.status_code == 422
