"""Unit tests for the eval metric math — the LLM judge is stubbed."""

import pytest

from tests.evals import run_evals


def test_faithfulness_is_fraction_of_supported_claims(monkeypatch):
    monkeypatch.setattr(
        run_evals, "extract_claims", lambda text: ["claim a", "claim b", "claim c", "claim d"]
    )
    # judge says YES to claims mentioning "a" or "b", NO otherwise
    monkeypatch.setattr(run_evals, "_ask_yes_no", lambda body: ("claim a" in body) or ("claim b" in body))

    score = run_evals.faithfulness("irrelevant", [{"text": "ctx"}])
    assert score == pytest.approx(0.5)


def test_faithfulness_none_when_no_claims(monkeypatch):
    monkeypatch.setattr(run_evals, "extract_claims", lambda text: [])
    assert run_evals.faithfulness("I don't know.", [{"text": "ctx"}]) is None


def test_context_precision_weights_early_hits_higher(monkeypatch):
    # relevance pattern [rel, not, rel] -> P@1=1/1, P@3=2/3 ; total_relevant=2
    # score = (1 + 2/3) / 2 = 0.8333
    calls = iter([True, False, True])
    monkeypatch.setattr(run_evals, "_ask_yes_no", lambda body: next(calls))
    score = run_evals.context_precision("q", [{"text": "a"}, {"text": "b"}, {"text": "c"}])
    assert score == pytest.approx((1.0 + 2 / 3) / 2)


def test_context_precision_zero_when_nothing_relevant(monkeypatch):
    monkeypatch.setattr(run_evals, "_ask_yes_no", lambda body: False)
    assert run_evals.context_precision("q", [{"text": "a"}, {"text": "b"}]) == 0.0


def test_context_precision_none_when_no_contexts():
    assert run_evals.context_precision("q", []) is None


def test_report_pass_fail_uses_thresholds(monkeypatch):
    from src.config import settings

    monkeypatch.setattr(settings, "faithfulness_threshold", 0.7)
    monkeypatch.setattr(settings, "context_precision_threshold", 0.65)

    ok = run_evals.EvalReport(0.8, 0.7, 1.0, 1.0, 10, 12)
    bad = run_evals.EvalReport(0.6, 0.9, 1.0, 1.0, 10, 12)
    assert ok.passed() is True
    assert bad.passed() is False
