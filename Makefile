# Production RAG — common tasks. Assumes the uv-managed venv in .venv/.
PY := .venv/Scripts/python.exe

.PHONY: help install test test-int lint fmt validate run worker evals \
        docker-build up up-observability down clean

help:
	@echo "install          install deps into .venv + spaCy model"
	@echo "test             unit tests (integration excluded)"
	@echo "test-int         integration tests (needs Ollama running)"
	@echo "lint / fmt       ruff check / ruff format"
	@echo "validate         full proof run -> writes VALIDATION.md"
	@echo "run              start the API on :8000"
	@echo "worker           start a Celery ingestion worker"
	@echo "evals            run the local LLM-judge quality gate"
	@echo "docker-build     build the app image"
	@echo "up / down        docker compose stack"
	@echo "up-observability compose stack + Arize Phoenix tracing"

install:
	uv venv
	uv pip install -r requirements.txt
	$(PY) -m spacy download en_core_web_sm

test:
	$(PY) -m pytest -m "not integration" -q

test-int:
	$(PY) -m pytest -m integration -q

lint:
	uvx ruff check .

fmt:
	uvx ruff format .

validate:
	$(PY) scripts/validate.py

run:
	$(PY) -m uvicorn src.api.main:app --reload --port 8000

worker:
	.venv/Scripts/celery.exe -A src.ingestion.tasks.celery_app worker --loglevel=info --pool=solo

evals:
	$(PY) -m tests.evals.run_evals

docker-build:
	docker build -t production-rag:local .

up:
	docker compose -f docker/docker-compose.yml up --build -d

up-observability:
	docker compose -f docker/docker-compose.yml -f docker/docker-compose.observability.yml up --build -d

down:
	docker compose -f docker/docker-compose.yml -f docker/docker-compose.observability.yml down

clean:
	rm -rf .pytest_cache .ruff_cache **/__pycache__ data/chroma_db
