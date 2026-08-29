"""RAG quality gate — faithfulness + context precision, judged by the local LLM.

Why not RAGAS: ragas 0.4.3 (the latest release) hard-imports
``langchain_community.chat_models.vertexai`` and
``langchain_openai.chat_models.AzureChatOpenAI`` at module load, neither of which
exists in the installed langchain 1.x stack — ``import ragas`` fails outright.
This harness computes the same two gate metrics RAGAS uses, with the project's
own Ollama model as judge, and no extra dependencies. The dataset keeps
RAGAS-compatible field names so RAGAS can be dropped back in if the stack is
pinned to a compatible version.

Run:  python -m tests.evals.run_evals
CI:   .github/workflows/eval_gate.yml
Exit code 1 if faithfulness < FAITHFULNESS_THRESHOLD or
context precision < CONTEXT_PRECISION_THRESHOLD.
"""

from __future__ import annotations

import json
import re
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from src.config import settings
from src.generation.generator import answer
from src.generation.llm import chat
from src.ingestion.pipeline import ingest_document
from src.retrieval import bm25

EVALS_DIR = Path(__file__).parent
FIXTURES_DIR = EVALS_DIR / "fixtures"
DATASET_PATH = EVALS_DIR / "dataset.json"

_YESNO_RE = re.compile(r"\b(yes|no)\b", re.IGNORECASE)


def _ask_yes_no(body: str) -> bool:
    reply = chat(
        "You are a strict evaluator. Reply with exactly one word: YES or NO.",
        body,
        temperature=0.0,
    )
    match = _YESNO_RE.search(reply)
    return bool(match) and match.group(1).lower() == "yes"


def extract_claims(answer_text: str) -> list[str]:
    reply = chat(
        "Split the user's text into individual factual claims. Output one claim per "
        "line, each line starting with '- '. Output nothing else.",
        answer_text,
        temperature=0.0,
    )
    return [line.strip()[2:].strip() for line in reply.splitlines() if line.strip().startswith("- ")]


def faithfulness(answer_text: str, contexts: list[dict]) -> float | None:
    """Fraction of the answer's atomic claims that the retrieved context supports."""
    claims = extract_claims(answer_text)
    if not claims:
        return None
    joined = "\n\n".join(c["text"] for c in contexts)
    supported = sum(
        _ask_yes_no(
            f"CONTEXT:\n{joined}\n\nCLAIM: {claim}\n\nIs the CLAIM directly and fully supported by the CONTEXT?"
        )
        for claim in claims
    )
    return supported / len(claims)


def context_precision(question: str, contexts: list[dict]) -> float | None:
    """RAGAS-style weighted precision@k over the retrieved passages in rank order."""
    if not contexts:
        return None
    relevant = [
        _ask_yes_no(
            f"QUESTION: {question}\n\nPASSAGE: {c['text']}\n\nIs this PASSAGE useful for answering the QUESTION?"
        )
        for c in contexts
    ]
    total_relevant = sum(relevant)
    if total_relevant == 0:
        return 0.0
    hits = 0
    precision_sum = 0.0
    for k, is_relevant in enumerate(relevant, start=1):
        if is_relevant:
            hits += 1
            precision_sum += hits / k
    return precision_sum / total_relevant


@dataclass
class EvalReport:
    faithfulness: float
    context_precision: float
    answer_hit_rate: float
    refusal_accuracy: float
    n_answered: int
    n_total: int

    def passed(self) -> bool:
        return (
            self.faithfulness >= settings.faithfulness_threshold
            and self.context_precision >= settings.context_precision_threshold
        )


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def evaluate_dataset(dataset_path: Path = DATASET_PATH) -> EvalReport:
    spec = json.loads(dataset_path.read_text(encoding="utf-8"))
    fixture = FIXTURES_DIR / spec["fixture"]

    tmp = Path(tempfile.mkdtemp(prefix="rag-eval-"))
    settings.chroma_path = str(tmp / "chroma")
    settings.chroma_collection = "evals"
    bm25._index = bm25.BM25Index()
    ingest_document(fixture)

    faith_scores: list[float] = []
    ctxp_scores: list[float] = []
    answer_hits: list[float] = []
    refusal_hits: list[float] = []
    answered = 0

    for item in spec["items"]:
        result = answer(item["question"])
        gave_answer = not result.refused and not result.degraded

        if item["answerable"]:
            if gave_answer:
                answered += 1
                f = faithfulness(result.answer, result.contexts)
                if f is not None:
                    faith_scores.append(f)
                p = context_precision(item["question"], result.contexts)
                if p is not None:
                    ctxp_scores.append(p)
            lowered = result.answer.lower()
            alts = item.get("must_include") or []
            answer_hits.append(
                1.0 if (not alts or any(a.lower() in lowered for a in alts)) and gave_answer else 0.0
            )
        else:
            refusal_hits.append(1.0 if result.refused else 0.0)

        print(f"  {'ANS' if gave_answer else 'REF'}  {item['question'][:70]}")

    return EvalReport(
        faithfulness=_mean(faith_scores),
        context_precision=_mean(ctxp_scores),
        answer_hit_rate=_mean(answer_hits),
        refusal_accuracy=_mean(refusal_hits),
        n_answered=answered,
        n_total=len(spec["items"]),
    )


def main() -> int:
    print("Running RAG evals (local LLM judge)...\n")
    report = evaluate_dataset()
    print(
        "\n"
        f"  faithfulness        {report.faithfulness:.3f}   (threshold {settings.faithfulness_threshold})\n"
        f"  context_precision   {report.context_precision:.3f}   (threshold {settings.context_precision_threshold})\n"
        f"  answer_hit_rate     {report.answer_hit_rate:.3f}\n"
        f"  refusal_accuracy    {report.refusal_accuracy:.3f}\n"
        f"  answered            {report.n_answered}/{report.n_total}\n"
    )
    if report.passed():
        print("PASS")
        return 0
    print("FAIL — quality below threshold")
    return 1


if __name__ == "__main__":
    sys.exit(main())
