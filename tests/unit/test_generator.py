import pytest

from src.generation import generator
from src.generation.generator import REFUSAL, RagAnswer, answer
from src.generation.llm import LLMError
from src.retrieval.retriever import RetrievedChunk


def _chunk(chunk_id, text, source="handbook.txt", page=1):
    return RetrievedChunk(
        id=chunk_id,
        text=text,
        metadata={"source": source, "page": page},
        score=1.0,
        rrf_score=0.1,
        vector_rank=1,
        bm25_rank=1,
    )


@pytest.fixture
def three_chunks(monkeypatch):
    chunks = [
        _chunk("c1", "The warranty lasts 24 months from purchase.", page=2),
        _chunk("c2", "Returns are accepted within 30 days.", page=3),
        _chunk("c3", "Support is available on weekdays only.", page=5),
    ]
    monkeypatch.setattr(generator, "retrieve", lambda q, rerank_top_n=None: chunks)
    return chunks


def test_grounded_answer_parses_citations(three_chunks, monkeypatch):
    monkeypatch.setattr(
        generator,
        "chat",
        lambda system, user, temperature=None: "The warranty is 24 months [1]. Returns take 30 days [2].",
    )
    result = answer("warranty and returns?")

    assert isinstance(result, RagAnswer)
    assert not result.refused and not result.degraded
    assert [c.marker for c in result.citations] == [1, 2]
    assert result.citations[0].chunk_id == "c1"
    assert result.citations[0].page == 2
    assert result.citations[1].source == "handbook.txt"
    assert len(result.contexts) == 3  # every retrieved passage is reported, not just cited ones


def test_no_retrieved_context_refuses_without_calling_llm(monkeypatch):
    monkeypatch.setattr(generator, "retrieve", lambda q, rerank_top_n=None: [])

    def _boom(*a, **k):
        raise AssertionError("LLM must not be called when there is no context")

    monkeypatch.setattr(generator, "chat", _boom)
    result = answer("anything")

    assert result.refused is True
    assert result.answer == REFUSAL
    assert result.citations == []


def test_llm_refusal_text_is_normalised(three_chunks, monkeypatch):
    monkeypatch.setattr(
        generator,
        "chat",
        lambda system, user, temperature=None: (
            "I don't have enough information in the provided documents to answer that"
        ),
    )
    result = answer("something off-topic")
    assert result.refused is True
    assert result.answer == REFUSAL
    assert result.citations == []


def test_out_of_range_citation_markers_are_dropped(three_chunks, monkeypatch):
    monkeypatch.setattr(
        generator,
        "chat",
        lambda system, user, temperature=None: "Fact A [1]. Fact B [7]. Fact C [0].",
    )
    result = answer("q")
    assert [c.marker for c in result.citations] == [1]
    # the invalid markers must also be scrubbed from the answer text
    assert "[7]" not in result.answer
    assert "[0]" not in result.answer
    assert "[1]" in result.answer
    assert "Fact B ." not in result.answer  # no dangling space before the period


def test_llm_error_returns_degraded_answer(three_chunks, monkeypatch):
    def _raise(*a, **k):
        raise LLMError("connection refused")

    monkeypatch.setattr(generator, "chat", _raise)
    result = answer("q")

    assert result.degraded is True
    assert result.refused is False
    assert result.citations == []
    assert "unavailable" in result.answer.lower()


def test_prompt_version_is_recorded(three_chunks, monkeypatch):
    monkeypatch.setattr(generator, "chat", lambda system, user, temperature=None: "answer [1]")
    assert answer("q").prompt_version == "rag_v1"
