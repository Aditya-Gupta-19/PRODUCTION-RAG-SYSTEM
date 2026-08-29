from src.observability import tracing


def test_setup_returns_false_without_collector_endpoint(monkeypatch):
    monkeypatch.delenv("PHOENIX_COLLECTOR_ENDPOINT", raising=False)
    monkeypatch.setattr(tracing, "_tracer", None)
    assert tracing.setup_tracing() is False


def test_span_is_a_noop_when_tracing_is_off(monkeypatch):
    monkeypatch.setattr(tracing, "_tracer", None)
    with tracing.span("retrieve", question="x"):
        value = 1 + 1
    assert value == 2  # body ran, nothing raised


def test_span_opens_a_real_span_when_tracer_is_set(monkeypatch):
    events = []

    class _Span:
        def set_attribute(self, k, v):
            events.append((k, v))

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    class _Tracer:
        def start_as_current_span(self, name):
            events.append(("span", name))
            return _Span()

    monkeypatch.setattr(tracing, "_tracer", _Tracer())
    with tracing.span("generate", model_passages=3):
        pass

    assert ("span", "generate") in events
    assert ("model_passages", 3) in events
