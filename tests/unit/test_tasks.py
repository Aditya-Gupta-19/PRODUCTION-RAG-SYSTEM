import pytest

from src.ingestion import tasks
from src.ingestion.pipeline import IngestResult


def test_reliability_config_is_set():
    conf = tasks.celery_app.conf
    assert conf.task_acks_late is True
    assert conf.task_reject_on_worker_lost is True
    assert conf.worker_prefetch_multiplier == 1
    assert conf.accept_content == ["json"]


def test_task_runs_pipeline_and_returns_summary(monkeypatch):
    monkeypatch.setattr(
        tasks,
        "ingest_document",
        lambda path: IngestResult(source="d.pdf", pages=4, chunks=9, chunk_ids=[f"d::{i}" for i in range(9)]),
    )
    tasks.celery_app.conf.task_always_eager = True
    tasks.celery_app.conf.task_eager_propagates = True
    try:
        result = tasks.ingest_document_task.delay("/tmp/d.pdf").get()
    finally:
        tasks.celery_app.conf.task_always_eager = False

    assert result == {
        "source": "d.pdf",
        "pages": 4,
        "chunks": 9,
        "chunk_ids": [f"d::{i}" for i in range(9)],
    }


def test_task_is_configured_to_retry_transient_failures():
    task = tasks.ingest_document_task
    assert task.max_retries == 3
    assert task.retry_backoff is True
    assert Exception in task.autoretry_for


def test_task_propagates_failure_after_exhausting_retries(monkeypatch):
    def _always_fails(path):
        raise RuntimeError("permanent")

    monkeypatch.setattr(tasks, "ingest_document", _always_fails)
    monkeypatch.setattr(tasks.ingest_document_task, "max_retries", 0, raising=False)
    tasks.celery_app.conf.task_always_eager = True
    tasks.celery_app.conf.task_eager_propagates = True
    try:
        with pytest.raises(RuntimeError):
            tasks.ingest_document_task.apply(args=["/tmp/d.pdf"]).get()
    finally:
        tasks.celery_app.conf.task_always_eager = False
        tasks.ingest_document_task.max_retries = 3
