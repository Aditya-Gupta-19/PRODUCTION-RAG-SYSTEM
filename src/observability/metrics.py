"""Prometheus instruments for the RAG pipeline.

All metrics register in prometheus_client's default registry, so
``prometheus_fastapi_instrumentator`` picks them up on the same ``/metrics``
endpoint as the built-in HTTP metrics. Import this module for its side effects
(metric registration) and use the objects directly.
"""

from prometheus_client import Counter, Histogram

_LATENCY_BUCKETS = (0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0)

QUERIES = Counter(
    "rag_queries_total",
    "Answered /query requests, labelled by outcome",
    ["outcome"],  # answered | refused | degraded | cached
)

INGESTS = Counter("rag_ingests_total", "Documents ingested", ["mode"])  # sync | async
INGEST_CHUNKS = Counter("rag_ingest_chunks_total", "Chunks written to the vector store")

CACHE_LOOKUPS = Counter(
    "rag_cache_lookups_total",
    "Semantic cache lookups, labelled by result",
    ["result"],  # hit | miss | disabled
)

QUERY_LATENCY = Histogram("rag_query_latency_seconds", "End-to-end /query latency", buckets=_LATENCY_BUCKETS)
RETRIEVAL_LATENCY = Histogram(
    "rag_retrieval_latency_seconds", "Hybrid retrieval + rerank latency", buckets=_LATENCY_BUCKETS
)
GENERATION_LATENCY = Histogram(
    "rag_generation_latency_seconds", "LLM generation latency", buckets=_LATENCY_BUCKETS
)
