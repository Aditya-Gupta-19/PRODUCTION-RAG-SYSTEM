"""Optional distributed tracing.

Arize Phoenix / OpenTelemetry is not a hard dependency. If it is installed
(``uv pip install arize-phoenix-otel openinference-instrumentation``) and
``PHOENIX_COLLECTOR_ENDPOINT`` is set, spans are exported; otherwise every helper
here is a no-op and the pipeline runs unchanged.
"""

import contextlib
import os

_tracer = None


def setup_tracing(service_name: str = "production-rag") -> bool:
    """Best-effort tracer init. Returns True if tracing is active."""
    global _tracer
    if _tracer is not None:
        return True
    if not os.getenv("PHOENIX_COLLECTOR_ENDPOINT"):
        return False
    try:  # pragma: no cover - exercised only when phoenix is installed
        from phoenix.otel import register

        provider = register(project_name=service_name, auto_instrument=True)
        _tracer = provider.get_tracer(service_name)
        return True
    except Exception:
        _tracer = None
        return False


@contextlib.contextmanager
def span(name: str, **attributes):
    """Context manager that opens a span when tracing is active, else does nothing."""
    if _tracer is None:
        yield
        return
    with _tracer.start_as_current_span(name) as current:  # pragma: no cover
        for key, value in attributes.items():
            current.set_attribute(key, value)
        yield
