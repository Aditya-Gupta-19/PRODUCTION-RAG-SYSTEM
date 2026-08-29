# Validation Report

- Generated: 2026-08-29 18:04 UTC
- Python: `3.12.4`  ·  Platform: `win32`
- Ollama reachable: **True**

| Step | Result | Time |
|------|--------|------|
| ruff check + ruff format --check | **PASS** | 0.2s |
| pytest -m 'not integration' | **PASS** | 82.5s |
| pytest -m integration (incl. HTTP end-to-end) | **PASS** | 137.3s |
| eval quality gate (faithfulness + context precision) | **PASS** | 678.4s |

## ruff check + ruff format --check — PASS

```
59 files already formatted
```

## pytest -m 'not integration' — PASS

```
........................................................................ [ 74%]
.........................                                                [100%]
============================== warnings summary ===============================
.venv\Lib\site-packages\fastapi\testclient.py:1
  C:\Users\sendf\Downloads\AI_PROJECT\RAG_IMPLEMENTATION\.venv\Lib\site-packages\fastapi\testclient.py:1: StarletteDeprecationWarning: Using `httpx` with `starlette.testclient` is deprecated; install `httpx2` instead.
    from starlette.testclient import TestClient as TestClient  # noqa

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
97 passed, 11 deselected, 1 warning in 76.45s (0:01:16)
```

## pytest -m integration (incl. HTTP end-to-end) — PASS

```
============================= test session starts =============================
collecting ... collected 108 items / 97 deselected / 11 selected

tests/integration/test_api_e2e.py::test_health_ok_and_carries_security_headers PASSED [  9%]
tests/integration/test_api_e2e.py::test_full_flow_ingest_then_grounded_cited_answer PASSED [ 18%]
tests/integration/test_api_e2e.py::test_oversized_upload_is_rejected PASSED [ 27%]
tests/integration/test_api_e2e.py::test_unanswerable_question_is_refused PASSED [ 36%]
tests/integration/test_generation.py::test_answers_grounded_question_with_citation PASSED [ 45%]
tests/integration/test_generation.py::test_refuses_when_answer_is_not_in_the_documents PASSED [ 54%]
tests/integration/test_ingest_and_retrieve.py::test_ingest_reports_pages_and_chunks PASSED [ 63%]
tests/integration/test_ingest_and_retrieve.py::test_semantic_retrieval_matches_paraphrase PASSED [ 72%]
tests/integration/test_ingest_and_retrieve.py::test_keyword_retrieval_matches_exact_token PASSED [ 81%]
tests/integration/test_ingest_and_retrieve.py::test_reingest_is_idempotent PASSED [ 90%]
tests/integration/test_ingest_and_retrieve.py::test_pii_is_masked_before_it_reaches_the_store PASSED [100%]

============================== warnings summary ===============================
.venv\Lib\site-packages\fastapi\testclient.py:1
  C:\Users\sendf\Downloads\AI_PROJECT\RAG_IMPLEMENTATION\.venv\Lib\site-packages\fastapi\testclient.py:1: StarletteDeprecationWarning: Using `httpx` with `starlette.testclient` is deprecated; install `httpx2` instead.
    from starlette.testclient import TestClient as TestClient  # noqa

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
========== 11 passed, 97 deselected, 1 warning in 128.29s (0:02:08) ===========
```

## eval quality gate (faithfulness + context precision) — PASS

```
  ANS  What is the notice period for a manager after probation?
  ANS  In which months are performance reviews held?
  REF  What is the company's policy on cryptocurrency trading by employees?
  REF  What health insurance provider does ACME use?

  faithfulness        0.800   (threshold 0.7)
  context_precision   0.733   (threshold 0.65)
  answer_hit_rate     1.000
  refusal_accuracy    1.000
  answered            15/17

PASS
Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.

Loading weights:   0%|          | 0/199 [00:00<?, ?it/s]
Loading weights:  44%|####4     | 88/199 [00:00<00:00, 875.48it/s]
Loading weights:  91%|#########1| 182/199 [00:00<00:00, 872.60it/s]
Loading weights: 100%|##########| 199/199 [00:00<00:00, 935.87it/s]
```
