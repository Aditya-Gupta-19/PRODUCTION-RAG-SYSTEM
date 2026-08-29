from functools import lru_cache

import ollama

from src.config import settings


class LLMError(RuntimeError):
    """Raised when the Ollama call fails (server down, model missing, timeout)."""


@lru_cache(maxsize=1)
def get_client() -> ollama.Client:
    # Connection pool + host config, built once. The heavy work (model weights)
    # lives in the Ollama server process, not here.
    return ollama.Client(host=settings.ollama_base_url)


def chat(
    system: str,
    user: str,
    *,
    temperature: float | None = None,
    num_ctx: int | None = None,
) -> str:
    """One-shot system+user chat completion against the local Ollama model.

    Returns the assistant text. Raises :class:`LLMError` on any transport or
    model error so the caller can emit a partial/degraded response instead of a
    500 with a stack trace.
    """
    try:
        response = get_client().chat(
            model=settings.llm_model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            options={
                "temperature": settings.llm_temperature if temperature is None else temperature,
                "num_ctx": settings.llm_num_ctx if num_ctx is None else num_ctx,
            },
        )
    except Exception as exc:  # ollama raises ResponseError, httpx.ConnectError, ...
        raise LLMError(f"Ollama chat failed: {exc}") from exc

    return response.message.content or ""
