# syntax=docker/dockerfile:1

# ---- builder: install deps + bake models into the image ----------------------
FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_SYSTEM_PYTHON=1

RUN pip install --no-cache-dir uv

WORKDIR /app
COPY requirements.txt requirements-observability.txt ./

# CPU-only torch keeps the image ~2 GB smaller than the default CUDA build.
RUN uv pip install --system torch --index-url https://download.pytorch.org/whl/cpu \
 && uv pip install --system -r requirements.txt \
 && python -m spacy download en_core_web_sm

# Optional: install the Phoenix/OTel tracing tree. Off by default (large).
# Enabled by the docker-compose.observability.yml overlay.
ARG INSTALL_OBSERVABILITY=0
RUN if [ "$INSTALL_OBSERVABILITY" = "1" ]; then \
      uv pip install --system -r requirements-observability.txt ; \
    fi

# Pre-download the embedding + reranker weights so the container starts
# deterministically and works offline. Cached under /models.
ENV HF_HOME=/models
RUN python - <<'PY'
from sentence_transformers import SentenceTransformer, CrossEncoder
SentenceTransformer("BAAI/bge-small-en-v1.5")
CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
PY

# ---- runtime ----------------------------------------------------------------
FROM python:3.12-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/models \
    HF_HUB_OFFLINE=1

# libgomp1 is required by onnxruntime / torch CPU kernels on slim images.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /models /models

WORKDIR /app
COPY src ./src
COPY prompts ./prompts
COPY pyproject.toml ./

RUN useradd --create-home --uid 10001 appuser \
 && mkdir -p /app/data/chroma_db /app/data/raw_docs \
 && chown -R appuser:appuser /app /models
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=45s --retries=3 \
  CMD python -c "import httpx,sys; sys.exit(0 if httpx.get('http://localhost:8000/health',timeout=4).status_code==200 else 1)"

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
