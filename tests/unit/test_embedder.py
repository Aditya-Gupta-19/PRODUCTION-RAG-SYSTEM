import numpy as np

from src.retrieval.embedder import QUERY_INSTRUCTION, embed_query, embed_texts, get_embedder


def test_embed_texts_returns_384_dim_vectors():
    vecs = embed_texts(["hello world", "a second sentence"])
    assert len(vecs) == 2
    assert len(vecs[0]) == 384
    assert len(vecs[1]) == 384


def test_embeddings_are_normalized():
    vecs = embed_texts(["some text to embed"])
    norm = np.linalg.norm(vecs[0])
    assert abs(norm - 1.0) < 1e-5


def test_embed_query_is_also_384_dim_and_normalized():
    vec = embed_query("what is this document about?")
    assert len(vec) == 384
    assert abs(np.linalg.norm(vec) - 1.0) < 1e-5


def test_embed_query_uses_instruction_prefix_not_raw_text():
    query = "example query"
    query_vec = np.array(embed_query(query))
    raw_vec = np.array(embed_texts([query])[0])
    # embed_query prefixes with QUERY_INSTRUCTION, so it must differ from
    # embedding the bare string the way embed_texts (document-side) would.
    assert not np.allclose(query_vec, raw_vec)
    prefixed_vec = np.array(embed_texts([QUERY_INSTRUCTION + query])[0])
    assert np.allclose(query_vec, prefixed_vec)


def test_get_embedder_is_cached_singleton():
    assert get_embedder() is get_embedder()


def test_query_ranks_semantically_relevant_doc_highest():
    docs = [
        "The mitochondria is the powerhouse of the cell.",
        "Reciprocal rank fusion merges ranked lists from multiple retrievers.",
        "Paris is the capital of France.",
    ]
    doc_vecs = np.array(embed_texts(docs))
    q_vec = np.array(embed_query("How does RRF combine search results?"))

    sims = doc_vecs @ q_vec
    best_index = int(np.argmax(sims))
    assert docs[best_index] == "Reciprocal rank fusion merges ranked lists from multiple retrievers."
