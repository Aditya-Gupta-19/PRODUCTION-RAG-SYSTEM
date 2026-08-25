import pytest

from src.config import settings
from src.retrieval.vectorstore import add_documents, get_collection, vector_search


@pytest.fixture(autouse=True)
def isolated_chroma_path(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "chroma_path", str(tmp_path))
    monkeypatch.setattr(settings, "chroma_collection", "test_collection")


def test_add_and_search_round_trip():
    add_documents(
        ids=["a", "b", "c"],
        texts=["cats are great", "dogs are loyal", "paris is in france"],
        embeddings=[[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]],
        metadatas=[{"source": "doc1"}, {"source": "doc1"}, {"source": "doc2"}],
    )
    results = vector_search([1.0, 0.0], top_k=3)
    assert len(results) == 3
    assert results[0]["id"] == "a"
    assert results[0]["text"] == "cats are great"
    assert results[0]["metadata"] == {"source": "doc1"}
    assert results[0]["distance"] == pytest.approx(0.0, abs=1e-6)


def test_cosine_ranking_orders_by_direction_not_magnitude():
    # 'a' and 'c' point in the same direction as the query but 'c' has a
    # much larger magnitude — cosine distance must not be fooled by that.
    # Note: Chroma 1.x rejects an empty {} metadata dict outright, so every
    # fixture below carries a placeholder key even where the value is unused.
    add_documents(
        ids=["a", "b", "c"],
        texts=["close direction, unit length", "orthogonal", "close direction, large magnitude"],
        embeddings=[[1.0, 0.0], [0.0, 1.0], [100.0, 0.0]],
        metadatas=[{"idx": 0}, {"idx": 1}, {"idx": 2}],
    )
    results = vector_search([1.0, 0.0], top_k=3)
    ranked_ids = [r["id"] for r in results]
    assert set(ranked_ids[:2]) == {"a", "c"}
    assert ranked_ids[2] == "b"
    assert results[0]["distance"] == pytest.approx(results[1]["distance"], abs=1e-4)


def test_top_k_limits_results():
    add_documents(
        ids=[f"doc{i}" for i in range(5)],
        texts=[f"text {i}" for i in range(5)],
        embeddings=[[float(i), 0.0] for i in range(5)],
        metadatas=[{"idx": i} for i in range(5)],
    )
    results = vector_search([0.0, 1.0], top_k=2)
    assert len(results) == 2


def test_top_k_defaults_to_settings_vector_top_k(monkeypatch):
    monkeypatch.setattr(settings, "vector_top_k", 2)
    add_documents(
        ids=[f"doc{i}" for i in range(5)],
        texts=[f"text {i}" for i in range(5)],
        embeddings=[[float(i), 0.0] for i in range(5)],
        metadatas=[{"idx": i} for i in range(5)],
    )
    results = vector_search([0.0, 1.0])
    assert len(results) == 2


def test_get_or_create_collection_is_idempotent():
    first = get_collection()
    second = get_collection()
    assert first.name == second.name == "test_collection"
