from functools import lru_cache

from sentence_transformers import SentenceTransformer

from src.config import settings

# BAAI/bge-small-en-v1.5's model card recommends prefixing the QUERY side
# only (never documents) with this instruction for retrieval tasks — it's
# what the model was fine-tuned to expect for asymmetric query-to-passage
# matching. Applying it to documents too would be wrong and would hurt
# retrieval quality, not improve it.
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


@lru_cache(maxsize=1)
def get_embedder() -> SentenceTransformer:
    return SentenceTransformer(settings.embedding_model)


def embed_texts(texts: list[str]) -> list[list[float]]:
    embeddings = get_embedder().encode(texts, normalize_embeddings=True)
    return embeddings.tolist()


def embed_query(query: str) -> list[float]:
    embedding = get_embedder().encode(QUERY_INSTRUCTION + query, normalize_embeddings=True)
    return embedding.tolist()
