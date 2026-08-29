"""Reproducible end-to-end validation — the single source of proof.

Runs, in order:
  1. ruff check + ruff format --check              (required)
  2. pytest -m "not integration"                   (required)
  3. pytest -m integration -v                      (required if Ollama is up;
       includes the full HTTP end-to-end: ingest a doc, ask a grounded question,
       get a cited answer, refuse an unanswerable one, reject an oversized upload)
  4. eval quality gate (faithfulness + context precision)   (skipped if Ollama is down)

Writes VALIDATION.md with each command, its status, duration and output tail.
Exits non-zero if any required step fails.

    python scripts/validate.py            # or:  make validate
    RAG_SKIP_EVALS=1 python scripts/validate.py   # faster: skip step 4
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PY = sys.executable
REPORT = REPO / "VALIDATION.md"


def _ollama_up() -> bool:
    base = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    try:
        urllib.request.urlopen(f"{base}/api/tags", timeout=2)  # noqa: S310
        return True
    except Exception:
        return False


class Step:
    def __init__(self, name: str, required: bool = True):
        self.name = name
        self.required = required
        self.status = "PENDING"
        self.detail = ""
        self.seconds = 0.0

    def run_cmd(self, cmd: list[str], tail: int = 18) -> subprocess.CompletedProcess:
        started = time.perf_counter()
        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO)  # noqa: S603
        self.seconds = time.perf_counter() - started
        out = ((proc.stdout or "") + (proc.stderr or "")).strip().splitlines()
        self.detail = "\n".join(out[-tail:])
        self.status = "PASS" if proc.returncode == 0 else "FAIL"
        return proc

    def skip(self, why: str):
        self.status = "SKIP"
        self.detail = why


def main() -> int:
    ollama = _ollama_up()
    steps: list[Step] = []

    lint = Step("ruff check + ruff format --check")
    steps.append(lint)
    if lint.run_cmd(["uvx", "ruff", "check", "."]).returncode == 0:
        lint.run_cmd(["uvx", "ruff", "format", "--check", "."])

    unit = Step("pytest -m 'not integration'")
    steps.append(unit)
    unit.run_cmd([PY, "-m", "pytest", "-m", "not integration", "-q"])

    integ = Step("pytest -m integration (incl. HTTP end-to-end)", required=ollama)
    steps.append(integ)
    if ollama:
        integ.run_cmd([PY, "-m", "pytest", "-m", "integration", "-v", "--no-header"], tail=40)
    else:
        integ.skip("Ollama not serving")

    evals = Step("eval quality gate (faithfulness + context precision)", required=False)
    steps.append(evals)
    if ollama and not os.getenv("RAG_SKIP_EVALS"):
        evals.run_cmd([PY, "-m", "tests.evals.run_evals"])
    elif not ollama:
        evals.skip("Ollama not serving")
    else:
        evals.skip("RAG_SKIP_EVALS set")

    _write_report(steps, ollama)

    failed = [s for s in steps if s.status == "FAIL" and s.required]
    print("\n".join(f"{s.status:5} {s.name} ({s.seconds:.1f}s)" for s in steps))
    print(f"\nReport: {REPORT}  ->  {'FAIL' if failed else 'OK'}")
    return 1 if failed else 0


def _write_report(steps: list[Step], ollama: bool) -> None:
    now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "# Validation Report",
        "",
        f"- Generated: {now}",
        f"- Python: `{sys.version.split()[0]}`  ·  Platform: `{sys.platform}`",
        f"- Ollama reachable: **{ollama}**",
        "",
        "| Step | Result | Time |",
        "|------|--------|------|",
    ]
    lines += [f"| {s.name} | **{s.status}** | {s.seconds:.1f}s |" for s in steps]
    lines.append("")
    for s in steps:
        lines += [f"## {s.name} — {s.status}", "", "```", s.detail or "(no output)", "```", ""]
    REPORT.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
