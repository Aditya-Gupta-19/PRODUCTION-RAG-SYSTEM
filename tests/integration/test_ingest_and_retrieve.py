"""End-to-end: real parser, chunker, Presidio, embedder, Chroma and cross-encoder.

Marked ``integration`` — loads real models and writes a real on-disk vector
store (slow, first run downloads the cross-encoder).
"""

import pytest

from src.config import settings
from src.ingestion.pipeline import ingest_document
from src.retrieval import bm25
from src.retrieval.retriever import retrieve
from src.retrieval.vectorstore import get_collection

pytestmark = pytest.mark.integration

SAMPLE = """\
The Apollo programme was a series of crewed spaceflights undertaken by NASA between 1961 and 1972.
Apollo 11 was the spaceflight that first landed humans on the Moon, in July 1969.
Neil Armstrong and Buzz Aldrin walked on the lunar surface while Michael Collins orbited above.
The refund policy for damaged goods allows a full return within thirty days of delivery,
provided the original packaging and the receipt are retained by the customer.
Error code E-4021 indicates a checksum mismatch was detected during firmware upload.
"""


@pytest.fixture(autouse=True)
def isolated_stores(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "chroma_path", str(tmp_path / "chroma"))
    monkeypatch.setattr(settings, "chroma_collection", "integration_test")
    # the BM25 index is a module-level singleton — give each test a fresh one
    monkeypatch.setattr(bm25, "_index", bm25.BM25Index())


def _write(tmp_path, name="handbook.txt", body=SAMPLE):
    path = tmp_path / name
    path.write_text(body, encoding="utf-8")
    return path


def test_ingest_reports_pages_and_chunks(tmp_path):
    result = ingest_document(_write(tmp_path), chunk_size=30, overlap=8)

    assert result.source == "handbook.txt"
    assert result.pages == 1
    assert result.chunks >= 2
    assert len(result.chunk_ids) == result.chunks
    assert get_collection().count() == result.chunks


def test_semantic_retrieval_matches_paraphrase(tmp_path):
    ingest_document(_write(tmp_path), chunk_size=30, overlap=8)

    # the document says "refund policy" / "full return"; the query says "money back"
    hits = retrieve("how do I get my money back for a broken item", rerank_top_n=3)

    assert hits
    assert any("refund" in h.text.lower() for h in hits)
    assert all(h.metadata["source"] == "handbook.txt" for h in hits)


def test_keyword_retrieval_matches_exact_token(tmp_path):
    ingest_document(_write(tmp_path), chunk_size=30, overlap=8)

    # an exact code string is BM25's job — embeddings tend to blur it
    hits = retrieve("E-4021", rerank_top_n=3)

    assert any("E-4021" in h.text for h in hits)


def test_reingest_is_idempotent(tmp_path):
    first = ingest_document(_write(tmp_path), chunk_size=30, overlap=8)
    second = ingest_document(_write(tmp_path), chunk_size=30, overlap=8)

    assert first.chunk_ids == second.chunk_ids
    assert get_collection().count() == len(first.chunk_ids)  # upsert overwrote, did not duplicate


def test_pii_is_masked_before_it_reaches_the_store(tmp_path):
    # NB: Presidio's EmailRecognizer validates the TLD, so the fixture must use
    # a real one (.com) — an RFC-2606 reserved TLD like .example is silently
    # not recognised as an email, the same way 123-45-6789 is a denylisted SSN.
    body = (
        "For escalations contact Sarah Whitfield at sarah.whitfield@acme.com or call 415-555-0142.\n" + SAMPLE
    )
    ingest_document(_write(tmp_path, "contacts.txt", body), chunk_size=40, overlap=10)

    stored = " ".join(get_collection().get()["documents"])
    assert "sarah.whitfield@acme.com" not in stored
    assert "415-555-0142" not in stored
    assert "<EMAIL_ADDRESS>" in stored
    assert "<PHONE_NUMBER>" in stored
