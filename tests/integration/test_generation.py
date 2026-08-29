"""End-to-end generation against the real local Ollama model.

Marked ``integration`` and skipped automatically when Ollama is not serving, so
CI without a model host stays green.
"""

import httpx
import pytest

from src.config import settings
from src.generation.generator import REFUSAL, answer
from src.ingestion.pipeline import ingest_document
from src.retrieval import bm25


def _ollama_up() -> bool:
    try:
        httpx.get(f"{settings.ollama_base_url}/api/tags", timeout=2.0).raise_for_status()
        return True
    except Exception:
        return False


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not _ollama_up(), reason="Ollama is not serving on settings.ollama_base_url"),
]

HANDBOOK = """\
ACME Corp employee handbook.
Annual leave: full-time employees accrue 25 days of paid annual leave per year.
Sick leave: employees are entitled to 10 paid sick days per year, no doctor's note
required for absences of two days or fewer.
Remote work: employees may work remotely up to three days per week with manager approval.
Expenses: receipts must be submitted within 60 days; claims over 500 dollars require
director sign-off.
"""


@pytest.fixture(autouse=True)
def isolated_stores(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "chroma_path", str(tmp_path / "chroma"))
    monkeypatch.setattr(settings, "chroma_collection", "gen_integration")
    monkeypatch.setattr(bm25, "_index", bm25.BM25Index())
    doc = tmp_path / "handbook.txt"
    doc.write_text(HANDBOOK, encoding="utf-8")
    ingest_document(doc, chunk_size=40, overlap=10)


def test_answers_grounded_question_with_citation():
    result = answer("How many paid annual leave days do full-time staff get?")

    assert not result.refused and not result.degraded
    assert "25" in result.answer
    assert result.citations, "a grounded answer must cite at least one passage"
    assert result.citations[0].source == "handbook.txt"


def test_refuses_when_answer_is_not_in_the_documents():
    result = answer("What is the CEO's home address?")

    assert result.answer == REFUSAL
    assert result.refused is True
    assert result.citations == []
