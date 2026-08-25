from src.config import Settings


def test_env_vars_map_to_expected_fields(monkeypatch):
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://example:11434")
    monkeypatch.setenv("LLM_MODEL", "llama3.2-custom")
    monkeypatch.setenv("LLM_TEMPERATURE", "0.5")
    monkeypatch.setenv("LLM_NUM_CTX", "8192")
    monkeypatch.setenv("EMBEDDING_MODEL", "some/other-model")
    monkeypatch.setenv("CHROMA_PATH", "./custom_chroma")
    monkeypatch.setenv("CHROMA_COLLECTION", "custom_collection")
    monkeypatch.setenv("VECTOR_TOP_K", "10")
    monkeypatch.setenv("BM25_TOP_K", "15")
    monkeypatch.setenv("RERANK_TOP_N", "3")
    monkeypatch.setenv("REDIS_URL", "redis://example:6380/1")
    monkeypatch.setenv("CACHE_TTL_SECONDS", "1800")
    monkeypatch.setenv("CACHE_SIMILARITY_THRESHOLD", "0.85")
    monkeypatch.setenv("API_KEY", "test-key")
    monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "50")
    monkeypatch.setenv("FAITHFULNESS_THRESHOLD", "0.75")
    monkeypatch.setenv("CONTEXT_PRECISION_THRESHOLD", "0.6")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")

    settings = Settings(_env_file=None)

    assert settings.ollama_base_url == "http://example:11434"
    assert settings.llm_model == "llama3.2-custom"
    assert settings.llm_temperature == 0.5
    assert settings.llm_num_ctx == 8192
    assert settings.embedding_model == "some/other-model"
    assert settings.chroma_path == "./custom_chroma"
    assert settings.chroma_collection == "custom_collection"
    assert settings.vector_top_k == 10
    assert settings.bm25_top_k == 15
    assert settings.rerank_top_n == 3
    assert settings.redis_url == "redis://example:6380/1"
    assert settings.cache_ttl_seconds == 1800
    assert settings.cache_similarity_threshold == 0.85
    assert settings.api_key == "test-key"
    assert settings.rate_limit_per_minute == 50
    assert settings.faithfulness_threshold == 0.75
    assert settings.context_precision_threshold == 0.6
    assert settings.log_level == "DEBUG"


def test_defaults_match_spec(monkeypatch):
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    monkeypatch.setenv("API_KEY", "test-key")  # only required field, no safe default

    settings = Settings(_env_file=None)

    assert settings.ollama_base_url == "http://localhost:11434"
    assert settings.llm_model == "llama3.2"
    assert settings.llm_temperature == 0.1
    assert settings.llm_num_ctx == 4096
    assert settings.embedding_model == "BAAI/bge-small-en-v1.5"
    assert settings.chroma_path == "./data/chroma_db"
    assert settings.chroma_collection == "rag_documents"
    assert settings.vector_top_k == 20
    assert settings.bm25_top_k == 20
    assert settings.rerank_top_n == 5
    assert settings.redis_url == "redis://localhost:6379/0"
    assert settings.cache_ttl_seconds == 3600
    assert settings.cache_similarity_threshold == 0.92
    assert settings.rate_limit_per_minute == 100
    assert settings.faithfulness_threshold == 0.70
    assert settings.context_precision_threshold == 0.65
    assert settings.log_level == "INFO"
