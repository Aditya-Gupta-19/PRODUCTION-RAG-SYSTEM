# Production RAG — common tasks. Assumes the uv-managed venv in .venv/.
PY := .venv/Scripts/python.exe

.PHONY: help install test test-int lint fmt run worker evals up down clean

help:
	@echo "install   install deps into .venv + spaCy model"
	@echo "test      unit tests (integration excluded)"
	@echo "test-int  integration tests (needs Ollama running)"
	@echo "lint      ruff check"
	@echo "fmt       ruff format"
	@echo "run       start the API on :8000"
	@echo "worker    start a Celery ingestion worker"
	@echo "evals     run the local LLM-judge quality gate"
	@echo "up/down   docker compose stack (redis+prometheus+grafana+api+worker)"

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

run:
	$(PY) -m uvicorn src.api.main:app --reload --port 8000

worker:
	.venv/Scripts/celery.exe -A src.ingestion.tasks.celery_app worker --loglevel=info --pool=solo

evals:
	$(PY) -m tests.evals.run_evals

up:
	docker compose -f docker/docker-compose.yml up --build -d

down:
	docker compose -f docker/docker-compose.yml down

clean:
	rm -rf .pytest_cache .ruff_cache **/__pycache__ data/chroma_db
