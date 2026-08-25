import chromadb

from src.config import settings


def get_collection():
    client = chromadb.PersistentClient(path=settings.chroma_path)
    # hnsw:space must be set explicitly at creation time — Chroma's un-set
    # default is L2, which would silently be the wrong metric for the
    # normalized embeddings this project uses everywhere.
    return client.get_or_create_collection(
        name=settings.chroma_collection,
        metadata={"hnsw:space": "cosine"},
    )


def add_documents(ids: list[str], texts: list[str], embeddings: list[list[float]], metadatas: list[dict]) -> None:
    get_collection().add(ids=ids, documents=texts, embeddings=embeddings, metadatas=metadatas)


def vector_search(query_embedding: list[float], top_k: int | None = None) -> list[dict]:
    top_k = top_k if top_k is not None else settings.vector_top_k
    result = get_collection().query(query_embeddings=[query_embedding], n_results=top_k)
    ids = result["ids"][0]
    docs = result["documents"][0]
    metadatas = result["metadatas"][0]
    distances = result["distances"][0]
    return [
        {"id": doc_id, "text": doc, "metadata": meta, "distance": dist}
        for doc_id, doc, meta, dist in zip(ids, docs, metadatas, distances)
    ]
