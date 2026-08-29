from functools import lru_cache

import chromadb
from chromadb.api.models.Collection import Collection

from src.config import settings


@lru_cache(maxsize=8)
def _client(path: str) -> chromadb.ClientAPI:
    # PersistentClient opens the on-disk DB (sqlite + HNSW segments). It is
    # expensive to construct and safe to reuse, so it is cached per path
    # rather than rebuilt on every call. Keyed on path so tests that point
    # settings.chroma_path at a tmp dir each get their own isolated client.
    return chromadb.PersistentClient(path=path)


@lru_cache(maxsize=8)
def _collection(path: str, name: str) -> Collection:
    # hnsw:space must be set explicitly at creation time — Chroma's un-set
    # default is L2, which would silently be the wrong metric for the
    # normalized embeddings this project uses everywhere.
    return _client(path).get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},
    )


def get_collection() -> Collection:
    return _collection(settings.chroma_path, settings.chroma_collection)


def add_documents(
    ids: list[str],
    texts: list[str],
    embeddings: list[list[float]],
    metadatas: list[dict],
) -> None:
    # upsert (not add) so re-ingesting a document overwrites its existing
    # chunks in place — chunk_ids are deterministic, so this is the intended
    # idempotent path, not an error.
    get_collection().upsert(ids=ids, documents=texts, embeddings=embeddings, metadatas=metadatas)


def vector_search(query_embedding: list[float], top_k: int | None = None) -> list[dict]:
    top_k = top_k if top_k is not None else settings.vector_top_k
    result = get_collection().query(query_embeddings=[query_embedding], n_results=top_k)
    ids = result["ids"][0]
    docs = result["documents"][0]
    metadatas = result["metadatas"][0]
    distances = result["distances"][0]
    return [
        {"id": doc_id, "text": doc, "metadata": meta, "distance": dist}
        for doc_id, doc, meta, dist in zip(ids, docs, metadatas, distances, strict=True)
    ]


def fetch_documents(ids: list[str]) -> dict[str, dict]:
    """Look up stored text + metadata for specific chunk ids.

    Needed by the retriever: a chunk that BM25 ranked but vector search did
    not return still needs its text pulled from the store before it can be
    handed to the reranker.
    """
    if not ids:
        return {}
    result = get_collection().get(ids=ids)
    return {
        doc_id: {"text": doc, "metadata": meta}
        for doc_id, doc, meta in zip(result["ids"], result["documents"], result["metadatas"], strict=True)
    }
