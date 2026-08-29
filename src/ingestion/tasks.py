"""Async ingestion via Celery.

Ingestion is slow (large PDFs, embedding inference) and must not block the API
or be lost on a crash. Design choices:

- ``task_acks_late = True``: the broker only drops a task once it *finishes*. If
  the worker dies mid-ingestion the task is redelivered, not silently lost.
- ``task_reject_on_worker_lost = True``: a hard worker kill (OOM, SIGKILL)
  requeues the task rather than marking it failed.
- ``worker_prefetch_multiplier = 1``: a worker holds one task at a time, so a
  slow ingestion does not starve siblings sitting behind it in the prefetch buffer.
- ``autoretry_for=(Exception,)`` with exponential backoff: transient failures
  (Ollama/Chroma blips) retry up to 3 times before going to the dead set.
"""

from celery import Celery

from src.config import settings
from src.ingestion.pipeline import ingest_document

celery_app = Celery("rag", broker=settings.redis_url, backend=settings.redis_url)
celery_app.conf.update(
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    task_track_started=True,
    worker_prefetch_multiplier=1,
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    result_expires=86_400,
)


@celery_app.task(
    bind=True,
    name="ingestion.ingest_document",
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_backoff_max=60,
    retry_jitter=True,
    max_retries=3,
)
def ingest_document_task(self, file_path: str) -> dict:
    result = ingest_document(file_path)
    return {
        "source": result.source,
        "pages": result.pages,
        "chunks": result.chunks,
        "chunk_ids": result.chunk_ids,
    }
