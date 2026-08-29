import re
import time
from dataclasses import dataclass, field

from src.generation.llm import LLMError, chat
from src.generation.prompt import load_prompt
from src.observability import metrics
from src.observability.tracing import span
from src.retrieval.retriever import RetrievedChunk, retrieve

REFUSAL = "I don't have enough information in the provided documents to answer that."

_MARKER_RE = re.compile(r"\[(\d{1,2})\]")


@dataclass(frozen=True)
class Citation:
    marker: int  # the [n] used in the answer text
    source: str
    page: int | None
    chunk_id: str


@dataclass(frozen=True)
class RagAnswer:
    question: str
    answer: str
    citations: list[Citation]
    contexts: list[dict] = field(default_factory=list)  # every passage offered to the model
    prompt_version: str = "rag_v1"
    refused: bool = False
    degraded: bool = False  # LLM failed; answer is a safe fallback, not a real answer


def _format_context(chunks: list[RetrievedChunk]) -> str:
    blocks = []
    for i, chunk in enumerate(chunks, start=1):
        source = chunk.metadata.get("source", "unknown")
        page = chunk.metadata.get("page")
        location = f"{source}, page {page}" if page is not None else source
        blocks.append(f"[{i}] ({location})\n{chunk.text}")
    return "\n\n".join(blocks)


def _looks_like_refusal(text: str) -> bool:
    normalised = text.strip().lower().rstrip(".")
    return normalised.startswith("i don't have enough information in the provided documents")


def _cited_markers(answer: str, passage_count: int) -> list[int]:
    seen = {int(m) for m in _MARKER_RE.findall(answer)}
    return sorted(m for m in seen if 1 <= m <= passage_count)


def answer(
    question: str,
    *,
    prompt_version: str = "rag_v1",
    rerank_top_n: int | None = None,
) -> RagAnswer:
    """Full RAG pipeline: retrieve → build grounded prompt → generate → parse citations.

    Guarantees:
    - No retrieved context  → deterministic refusal, the LLM is never called.
    - LLM error             → ``degraded=True`` fallback, never a raised exception.
    - Citations in the answer are validated against the passages actually shown.
    """
    prompt = load_prompt(prompt_version)

    retrieval_started = time.perf_counter()
    with span("retrieve", question=question):
        chunks = retrieve(question, rerank_top_n=rerank_top_n)
    metrics.RETRIEVAL_LATENCY.observe(time.perf_counter() - retrieval_started)

    contexts = [
        {
            "marker": i,
            "chunk_id": chunk.id,
            "source": chunk.metadata.get("source"),
            "page": chunk.metadata.get("page"),
            "text": chunk.text,
        }
        for i, chunk in enumerate(chunks, start=1)
    ]

    if not chunks:
        return RagAnswer(question, REFUSAL, [], contexts, prompt.version, refused=True)

    try:
        generation_started = time.perf_counter()
        with span("generate", model_passages=len(chunks)):
            raw = chat(
                prompt.system,
                prompt.render_user(context=_format_context(chunks), question=question),
                temperature=prompt.model.get("temperature"),
            ).strip()
        metrics.GENERATION_LATENCY.observe(time.perf_counter() - generation_started)
    except LLMError:
        return RagAnswer(
            question,
            "The answer service is temporarily unavailable. Please retry shortly.",
            [],
            contexts,
            prompt.version,
            degraded=True,
        )

    if _looks_like_refusal(raw):
        return RagAnswer(question, REFUSAL, [], contexts, prompt.version, refused=True)

    citations = [
        Citation(
            marker=marker,
            source=chunks[marker - 1].metadata.get("source", ""),
            page=chunks[marker - 1].metadata.get("page"),
            chunk_id=chunks[marker - 1].id,
        )
        for marker in _cited_markers(raw, len(chunks))
    ]
    return RagAnswer(question, raw, citations, contexts, prompt.version)
