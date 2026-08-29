import pytest

from src.retrieval import retriever
from src.retrieval.retriever import RetrievedChunk, reciprocal_rank_fusion, retrieve


class _FakeReranker:
    """Scores a (query, passage) pair by how many query words the passage contains."""

    def predict(self, pairs):
        return [
            float(sum(word in passage.lower() for word in query.lower().split())) for query, passage in pairs
        ]


# --- reciprocal_rank_fusion: pure function ---------------------------------


def test_rrf_scores_by_1_over_k_plus_rank():
    fused = reciprocal_rank_fusion([["a", "b", "c"]], k=60)
    assert fused["a"] == pytest.approx(1 / 61)
    assert fused["b"] == pytest.approx(1 / 62)
    assert fused["c"] == pytest.approx(1 / 63)
    assert fused["a"] > fused["b"] > fused["c"]


def test_rrf_accumulates_agreement_across_lists():
    # 'b' is rank 2 in both lists; 'a' is rank 1 in one list only.
    fused = reciprocal_rank_fusion([["a", "b"], ["c", "b"]], k=60)
    assert fused["b"] == pytest.approx(1 / 62 + 1 / 62)
    assert fused["a"] == pytest.approx(1 / 61)
    assert fused["b"] > fused["a"]  # cross-retriever agreement beats a lone first place


def test_rrf_empty_inputs_return_empty():
    assert reciprocal_rank_fusion([]) == {}
    assert reciprocal_rank_fusion([[], []]) == {}


def test_rrf_k_controls_damping():
    steep = reciprocal_rank_fusion([["a", "b"]], k=1)
    flat = reciprocal_rank_fusion([["a", "b"]], k=1000)
    assert steep["a"] / steep["b"] > flat["a"] / flat["b"]


# --- retrieve: wired, with primitives + reranker stubbed ------------------

_DOCS = {
    "c0": {"text": "Reciprocal rank fusion merges ranked lists.", "metadata": {"source": "d.txt", "page": 1}},
    "c1": {
        "text": "The cross encoder reranks the fused candidates.",
        "metadata": {"source": "d.txt", "page": 1},
    },
    "c2": {"text": "Bananas are a good source of potassium.", "metadata": {"source": "d.txt", "page": 2}},
}


@pytest.fixture
def stubbed(monkeypatch):
    monkeypatch.setattr(retriever, "embed_query", lambda q: [0.0])
    monkeypatch.setattr(retriever, "get_reranker", lambda: _FakeReranker())
    monkeypatch.setattr(retriever, "fetch_documents", lambda ids: {i: _DOCS[i] for i in ids if i in _DOCS})


def _vector(*ids):
    return lambda vec, top_k: [{"id": i, **_DOCS[i]} for i in ids]


def _bm25(*ids):
    return lambda q, top_k: [{"id": i, "score": 1.0} for i in ids]


def test_retrieve_fuses_then_reranks(stubbed, monkeypatch):
    monkeypatch.setattr(retriever, "vector_search", _vector("c2", "c0"))
    monkeypatch.setattr(retriever, "bm25_search", _bm25("c0", "c1"))

    results = retrieve("how does rank fusion work", rerank_top_n=2)

    assert [r.id for r in results] == ["c0", "c1"]  # reranker: most query-word overlap wins
    assert isinstance(results[0], RetrievedChunk)
    assert results[0].vector_rank == 2
    assert results[0].bm25_rank == 1
    assert results[0].metadata["source"] == "d.txt"


def test_retrieve_fetches_text_for_bm25_only_hits(stubbed, monkeypatch):
    monkeypatch.setattr(retriever, "vector_search", _vector())  # vector returns nothing
    monkeypatch.setattr(retriever, "bm25_search", _bm25("c1"))

    results = retrieve("cross encoder", rerank_top_n=5)

    assert [r.id for r in results] == ["c1"]
    assert results[0].vector_rank is None
    assert results[0].bm25_rank == 1
    assert results[0].text == _DOCS["c1"]["text"]  # text was fetched, not dropped


def test_retrieve_returns_empty_when_no_retriever_hits(stubbed, monkeypatch):
    monkeypatch.setattr(retriever, "vector_search", _vector())
    monkeypatch.setattr(retriever, "bm25_search", _bm25())

    assert retrieve("anything at all") == []


def test_retrieve_without_rerank_orders_by_rrf_score(stubbed, monkeypatch):
    monkeypatch.setattr(retriever, "vector_search", _vector("c2", "c0"))
    monkeypatch.setattr(retriever, "bm25_search", _bm25("c0", "c1"))

    results = retrieve("irrelevant text", rerank=False, rerank_top_n=3)

    # c0 = 1/61 + 1/62 ; c2 = 1/61 ; c1 = 1/62  ->  c0 > c2 > c1
    assert [r.id for r in results] == ["c0", "c2", "c1"]
    assert results[0].score == results[0].rrf_score


def test_retrieve_respects_rerank_top_n(stubbed, monkeypatch):
    monkeypatch.setattr(retriever, "vector_search", _vector("c0", "c1", "c2"))
    monkeypatch.setattr(retriever, "bm25_search", _bm25("c2", "c1", "c0"))

    assert len(retrieve("rank fusion", rerank_top_n=1)) == 1
    assert len(retrieve("rank fusion", rerank_top_n=2)) == 2
