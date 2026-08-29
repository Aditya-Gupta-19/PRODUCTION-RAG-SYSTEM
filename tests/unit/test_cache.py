import fakeredis
import numpy as np
import pytest

from src.api import cache
from src.config import settings


@pytest.fixture
def fake_redis(monkeypatch):
    server = fakeredis.FakeStrictRedis(decode_responses=True)
    monkeypatch.setattr(cache, "get_redis", lambda: server)
    return server


@pytest.fixture
def controlled_vectors(monkeypatch):
    table = {
        "east": np.array([1.0, 0.0, 0.0], dtype=np.float32),
        "east-ish": np.array([0.985, 0.174, 0.0], dtype=np.float32),  # ~0.985 cos with "east"
        "north": np.array([0.0, 1.0, 0.0], dtype=np.float32),
    }
    monkeypatch.setattr(cache, "_vector", lambda text: table[text])
    return table


def test_store_then_lookup_hits_a_close_question(fake_redis, controlled_vectors, monkeypatch):
    monkeypatch.setattr(settings, "cache_similarity_threshold", 0.92)
    cache.store("east", {"answer": "towards sunrise"})

    assert cache.lookup("east-ish") == {"answer": "towards sunrise"}


def test_lookup_misses_when_similarity_below_threshold(fake_redis, controlled_vectors, monkeypatch):
    monkeypatch.setattr(settings, "cache_similarity_threshold", 0.99)
    cache.store("east", {"answer": "towards sunrise"})

    assert cache.lookup("east-ish") is None  # 0.985 < 0.99
    assert cache.lookup("north") is None  # orthogonal


def test_cache_is_a_noop_when_redis_is_unavailable(monkeypatch):
    monkeypatch.setattr(cache, "get_redis", lambda: None)
    monkeypatch.setattr(cache, "_vector", lambda text: np.zeros(3, dtype=np.float32))

    assert cache.lookup("anything") is None
    cache.store("anything", {"answer": "x"})  # must not raise


def test_get_redis_returns_none_on_connection_failure(monkeypatch):
    monkeypatch.setattr(settings, "redis_url", "redis://127.0.0.1:6390/0")  # nothing listening
    cache.get_redis.cache_clear()
    try:
        assert cache.get_redis() is None
    finally:
        cache.get_redis.cache_clear()
