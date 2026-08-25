from src.config import settings
from src.retrieval.bm25 import BM25Index, bm25_search, rebuild_bm25_from_vectorstore
from src.retrieval.vectorstore import add_documents


def _corpus():
    ids = ["a", "b", "c"]
    texts = [
        "The cat sat on the mat",
        "Dogs are loyal companions",
        "Cats and dogs can be friends",
    ]
    return ids, texts


def test_search_ranks_keyword_distinctive_doc_first():
    idx = BM25Index()
    idx.build(*_corpus())
    results = idx.search("loyal companions", top_k=5)
    assert results
    assert results[0][0] == "b"


def test_search_returns_plain_floats_not_numpy():
    idx = BM25Index()
    idx.build(*_corpus())
    results = idx.search("loyal", top_k=5)
    assert isinstance(results[0][1], float)


def test_no_stemming_exact_token_match_only():
    # "cat" (singular) only matches doc "a" verbatim; doc "c" only contains
    # "Cats" (plural) — with no stemming, that's a different token and must
    # NOT match. This is a direct consequence of the no-stemming design
    # choice and is worth pinning down explicitly, not just assuming.
    idx = BM25Index()
    idx.build(*_corpus())
    results = idx.search("cat", top_k=5)
    matched_ids = [doc_id for doc_id, _ in results]
    assert matched_ids == ["a"]


def test_unmatched_query_returns_empty_list():
    idx = BM25Index()
    idx.build(*_corpus())
    assert idx.search("nonexistent word xyz", top_k=5) == []


def test_top_k_limits_results():
    # A term present in every document has zero/negative IDF and gets
    # filtered out entirely (see test_term_in_every_doc_scores_are_filtered).
    # A term in exactly half the corpus (5/10) also lands at idf=log(1)=0
    # by coincidence of the standard BM25 formula. Use a 3/10 split so IDF
    # is unambiguously positive, giving more than top_k positive matches to
    # actually truncate.
    ids = [f"doc{i}" for i in range(10)]
    texts = [f"shared keyword appears in document {i}" for i in range(3)] + [
        f"completely unrelated filler content number {i}" for i in range(7)
    ]
    idx = BM25Index()
    idx.build(ids, texts)
    results = idx.search("shared keyword", top_k=2)
    assert len(results) == 2


def test_term_in_every_doc_scores_are_filtered():
    # BM25's IDF for a term appearing in every document is zero/negative (no
    # discriminating power) — rank_bm25 doesn't clip this, so such a term
    # produces negative scores across the board, and search() must filter
    # them out via `score > 0` rather than return a full, meaningless ranking.
    ids = [f"doc{i}" for i in range(10)]
    texts = ["shared keyword appears here" for _ in range(10)]
    idx = BM25Index()
    idx.build(ids, texts)
    assert idx.search("shared keyword", top_k=5) == []


def test_empty_corpus_returns_empty_list():
    idx = BM25Index()
    idx.build([], [])
    assert idx.search("anything", top_k=5) == []


def test_fresh_index_before_build_returns_empty_list():
    idx = BM25Index()
    assert idx.search("anything", top_k=5) == []


def test_rebuild_from_vectorstore_and_search(tmp_path, monkeypatch):
    # BM25 is in-memory only — this verifies the actual wiring that recovers
    # it from the persistent Chroma store, not just BM25Index in isolation.
    # Uses 3 docs (not 2) with "alpha" in only 1 — at exactly 1/2 document
    # frequency, idf lands at log(1)=0 again (same coincidence as the
    # top_k test above); 1/3 keeps it unambiguously positive.
    monkeypatch.setattr(settings, "chroma_path", str(tmp_path))
    monkeypatch.setattr(settings, "chroma_collection", "bm25_rebuild_test")
    add_documents(
        ids=["x", "y", "z"],
        texts=[
            "unique keyword alpha appears here",
            "totally different content",
            "yet another unrelated passage",
        ],
        embeddings=[[1.0, 0.0], [0.0, 1.0], [0.0, -1.0]],
        metadatas=[{"idx": 0}, {"idx": 1}, {"idx": 2}],
    )
    rebuild_bm25_from_vectorstore()
    results = bm25_search("alpha", top_k=5)
    assert results
    assert results[0]["id"] == "x"
