from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # LLM
    ollama_base_url: str = "http://localhost:11434"
    llm_model: str = "llama3.2"
    llm_temperature: float = 0.1
    llm_num_ctx: int = 4096

    # Embeddings
    embedding_model: str = "BAAI/bge-small-en-v1.5"

    # Vector store
    chroma_path: str = "./data/chroma_db"
    chroma_collection: str = "rag_documents"

    # Retrieval tuning
    vector_top_k: int = 20
    bm25_top_k: int = 20
    rerank_top_n: int = 5

    # Cache
    redis_url: str = "redis://localhost:6379/0"
    cache_ttl_seconds: int = 3600
    cache_similarity_threshold: float = 0.92

    # Security
    api_key: str
    rate_limit_per_minute: int = 100

    # Eval thresholds
    faithfulness_threshold: float = 0.70
    context_precision_threshold: float = 0.65

    # Observability
    log_level: str = "INFO"


settings = Settings()

# data/raw_docs and data/chroma_db are gitignored — recreate them on a fresh clone.
Path(settings.chroma_path).mkdir(parents=True, exist_ok=True)
(REPO_ROOT / "data" / "raw_docs").mkdir(parents=True, exist_ok=True)
