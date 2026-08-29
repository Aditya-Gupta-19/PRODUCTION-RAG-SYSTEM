from rank_bm25 import BM25Okapi

from src.config import settings


class BM25Index:
    def __init__(self) -> None:
        self._bm25: BM25Okapi | None = None
        self._ids: list[str] = []

    def build(self, ids: list[str], texts: list[str]) -> None:
        tokenized = [text.lower().split() for text in texts]
        self._ids = list(ids)
        self._bm25 = BM25Okapi(tokenized) if tokenized else None

    def search(self, query: str, top_k: int = 20) -> list[tuple[str, float]]:
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(query.lower().split())
        ranked = sorted(zip(self._ids, scores, strict=True), key=lambda pair: pair[1], reverse=True)
        # rank_bm25 returns numpy.float64 — cast to plain float so scores stay
        # JSON-serializable once they flow into API responses later.
        return [(doc_id, float(score)) for doc_id, score in ranked[:top_k] if score > 0]


# BM25 is in-memory only — it does not survive a process restart, and must be
# rebuilt from the persistent vector store after every ingest and at startup.
_index = BM25Index()


def rebuild_bm25_from_vectorstore() -> None:
    from src.retrieval.vectorstore import get_collection

    data = get_collection().get()
    _index.build(data.get("ids", []), data.get("documents", []))


def bm25_search(query: str, top_k: int | None = None) -> list[dict]:
    top_k = top_k if top_k is not None else settings.bm25_top_k
    return [{"id": doc_id, "score": score} for doc_id, score in _index.search(query, top_k=top_k)]
