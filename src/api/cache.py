"""Redis-backed semantic cache for /query.

A question is cached under a key derived from its text, together with its query
embedding. On lookup, the incoming question is embedded and compared (cosine, via
dot product — embeddings are already L2-normalised) against every cached entry;
if the best match is above ``settings.cache_similarity_threshold`` the stored
response is returned and generation is skipped.

Every operation degrades to a no-op if Redis is unreachable — the cache is an
optimisation, never a dependency.

Scaling note: lookup scans all cache keys, which is O(n) in the number of cached
questions. Fine for a single-node deployment with a bounded TTL; for large-scale
use, back this with Redis Search / a vector index instead of SCAN.
"""

import contextlib
import hashlib
import json
from functools import lru_cache

import numpy as np
import redis

from src.config import settings
from src.retrieval.embedder import embed_query

_KEY_PREFIX = "ragcache:"


@lru_cache(maxsize=1)
def get_redis() -> redis.Redis | None:
    try:
        client = redis.from_url(
            settings.redis_url, decode_responses=True, socket_connect_timeout=1, socket_timeout=1
        )
        client.ping()
        return client
    except Exception:
        return None


def _vector(text: str) -> np.ndarray:
    return np.asarray(embed_query(text), dtype=np.float32)


def lookup(question: str) -> dict | None:
    """Return a cached response dict for a semantically-close question, or None."""
    client = get_redis()
    if client is None:
        return None

    query_vec = _vector(question)
    best_response: dict | None = None
    best_similarity = -1.0

    for key in client.scan_iter(match=f"{_KEY_PREFIX}*", count=200):
        raw = client.get(key)
        if not raw:
            continue
        entry = json.loads(raw)
        similarity = float(np.dot(query_vec, np.asarray(entry["embedding"], dtype=np.float32)))
        if similarity > best_similarity:
            best_similarity, best_response = similarity, entry["response"]

    if best_response is not None and best_similarity >= settings.cache_similarity_threshold:
        return best_response
    return None


def store(question: str, response: dict) -> None:
    """Cache ``response`` for ``question`` with the configured TTL. No-op if Redis is down."""
    client = get_redis()
    if client is None:
        return

    key = _KEY_PREFIX + hashlib.sha256(question.strip().lower().encode()).hexdigest()
    payload = json.dumps(
        {"question": question, "embedding": _vector(question).tolist(), "response": response}
    )
    with contextlib.suppress(Exception):
        client.setex(key, settings.cache_ttl_seconds, payload)
