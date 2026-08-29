from dataclasses import dataclass
from functools import lru_cache

from sentence_transformers import CrossEncoder

from src.config import settings
from src.retrieval.bm25 import bm25_search
from src.retrieval.embedder import embed_query
from src.retrieval.vectorstore import fetch_documents, vector_search

# Cross-encoder that scores a (query, passage) pair jointly rather than
# comparing two precomputed embeddings. ms-marco-MiniLM-L-6-v2 (~22M params)
# is the standard small reranker: trained on MS MARCO passage ranking, it
# returns one relevance logit per pair — higher means more relevant. Only the
# relative order of the logits matters here, not their absolute value.
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# RRF damping constant. 60 is the value from Cormack et al. (2009) and the
# de-facto default. A larger k flattens the contribution of top ranks so no
# single retriever can dominate the fused order on its own.
RRF_K = 60


@dataclass(frozen=True)
class RetrievedChunk:
    id: str
    text: str
    metadata: dict
    score: float             # reranker relevance score, or RRF score if rerank is off
    rrf_score: float
    vector_rank: int | None  # 1-indexed rank in vector search; None if it did not return this chunk
    bm25_rank: int | None    # 1-indexed rank in BM25 search; None if it did not return this chunk


def reciprocal_rank_fusion(ranked_lists: list[list[str]], k: int = RRF_K) -> dict[str, float]:
    """Merge ranked id lists by position: ``score(d) = Σ 1 / (k + rank_i(d))``.

    Rank is 1-indexed. An id missing from a list contributes nothing from that
    list. Only rank position is used, so the retrievers' own incomparable
    scores (cosine distance vs. BM25 magnitude) never need normalizing.
    """
    fused: dict[str, float] = {}
    for ranking in ranked_lists:
        for rank, doc_id in enumerate(ranking, start=1):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + rank)
    return fused


@lru_cache(maxsize=1)
def get_reranker() -> CrossEncoder:
    return CrossEncoder(RERANKER_MODEL)


def retrieve(
    query: str,
    *,
    vector_top_k: int | None = None,
    bm25_top_k: int | None = None,
    rerank_top_n: int | None = None,
    rerank: bool = True,
) -> list[RetrievedChunk]:
    """Hybrid retrieval: vector + BM25 → RRF fusion → CrossEncoder rerank → top-N.

    The funnel is cheap-and-wide (both retrievers, ``*_top_k`` candidates each),
    then a cheap position-only merge (RRF), then an expensive-and-narrow
    precision pass (the cross-encoder) over only the fused candidates.
    """
    vector_top_k = vector_top_k if vector_top_k is not None else settings.vector_top_k
    bm25_top_k = bm25_top_k if bm25_top_k is not None else settings.bm25_top_k
    rerank_top_n = rerank_top_n if rerank_top_n is not None else settings.rerank_top_n

    vector_hits = vector_search(embed_query(query), top_k=vector_top_k)
    bm25_hits = bm25_search(query, top_k=bm25_top_k)

    vector_ids = [h["id"] for h in vector_hits]
    bm25_ids = [h["id"] for h in bm25_hits]
    vector_rank = {doc_id: i for i, doc_id in enumerate(vector_ids, start=1)}
    bm25_rank = {doc_id: i for i, doc_id in enumerate(bm25_ids, start=1)}

    fused = reciprocal_rank_fusion([vector_ids, bm25_ids])
    if not fused:
        return []

    # Text + metadata for every fused candidate. Vector hits already carry
    # their text; a chunk that only BM25 ranked must be fetched from the store.
    pool: dict[str, dict] = {
        h["id"]: {"text": h["text"], "metadata": h["metadata"]} for h in vector_hits
    }
    pool.update(fetch_documents([doc_id for doc_id in fused if doc_id not in pool]))

    candidates = sorted(
        (doc_id for doc_id in fused if doc_id in pool),
        key=lambda doc_id: fused[doc_id],
        reverse=True,
    )

    if rerank and len(candidates) > 1:
        scores = get_reranker().predict([(query, pool[doc_id]["text"]) for doc_id in candidates])
        ranked = sorted(zip(candidates, scores), key=lambda pair: pair[1], reverse=True)
        final = [(doc_id, float(score)) for doc_id, score in ranked[:rerank_top_n]]
    else:
        final = [(doc_id, fused[doc_id]) for doc_id in candidates[:rerank_top_n]]

    return [
        RetrievedChunk(
            id=doc_id,
            text=pool[doc_id]["text"],
            metadata=pool[doc_id]["metadata"],
            score=score,
            rrf_score=fused[doc_id],
            vector_rank=vector_rank.get(doc_id),
            bm25_rank=bm25_rank.get(doc_id),
        )
        for doc_id, score in final
    ]
