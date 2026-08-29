"""Optional LLM tracing via Arize Phoenix / OpenTelemetry.

Not a hard dependency. If ``arize-phoenix-otel`` is installed
(``uv pip install -r requirements-observability.txt``) *and*
``PHOENIX_COLLECTOR_ENDPOINT`` is set, ``setup_tracing()`` wires an OTLP
exporter and every ``span()`` becomes a real span in the Phoenix UI. Otherwise
every helper here is a no-op and the pipeline is byte-for-byte unchanged.
"""

import contextlib
import logging
import os

logger = logging.getLogger("rag.tracing")

_tracer = None


def setup_tracing(service_name: str = "production-rag") -> bool:
    """Best-effort tracer init. Returns True only if tracing is now active."""
    global _tracer
    if _tracer is not None:
        return True

    endpoint = os.getenv("PHOENIX_COLLECTOR_ENDPOINT")
    if not endpoint:
        return False

    try:  # pragma: no cover - only when phoenix-otel is installed
        from phoenix.otel import register

        provider = register(
            endpoint=f"{endpoint.rstrip('/')}/v1/traces",
            project_name=service_name,
            batch=True,
            auto_instrument=True,
            set_global_tracer_provider=False,
            verbose=False,
        )
        _tracer = provider.get_tracer(service_name)
        logger.info("Phoenix tracing enabled -> %s", endpoint)
        return True
    except Exception:
        logger.warning("Phoenix tracing requested but could not initialise", exc_info=True)
        _tracer = None
        return False


@contextlib.contextmanager
def span(name: str, **attributes):
    """Open a span when tracing is active; a no-op otherwise."""
    if _tracer is None:
        yield
        return
    with _tracer.start_as_current_span(name) as current:  # pragma: no cover
        for key, value in attributes.items():
            current.set_attribute(key, value)
        yield
