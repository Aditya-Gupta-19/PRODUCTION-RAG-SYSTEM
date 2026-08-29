"""Reproducible end-to-end validation — the single source of proof.

Runs, in order:
  1. ruff check + ruff format --check          (required)
  2. pytest -m "not integration"               (required)
  3. pytest -m integration                     (skipped if Ollama is down)
  4. eval quality gate                         (skipped if Ollama is down)
  5. live API smoke: boot uvicorn, ingest a doc, ask a grounded question,
     scrape /metrics                           (required)

Writes VALIDATION.md with every command, its exit code, duration and key
output. Exits non-zero if any required step fails.

    python scripts/validate.py            # or:  make validate
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PY = sys.executable
REPORT = REPO / "VALIDATION.md"
API_KEY = os.getenv("API_KEY", "dev-key-change-in-production")


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


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

    def run_cmd(self, cmd: list[str], **kw) -> subprocess.CompletedProcess:
        started = time.perf_counter()
        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO, **kw)
        self.seconds = time.perf_counter() - started
        tail = (proc.stdout or "").strip().splitlines()[-12:]
        self.detail = "\n".join(tail)
        self.status = "PASS" if proc.returncode == 0 else "FAIL"
        return proc

    def skip(self, why: str):
        self.status = "SKIP"
        self.detail = why


def _smoke(step: Step) -> None:
    port = _free_port()
    base = f"http://localhost:{port}"
    lines: list[str] = []
    started = time.perf_counter()
    server_log = REPO / "scripts" / "_smoke_server.log"
    with server_log.open("w") as log:
        server = subprocess.Popen(
            [PY, "-m", "uvicorn", "src.api.main:app", "--port", str(port), "--log-level", "info"],
            cwd=REPO,
            env={**os.environ, "API_KEY": API_KEY},
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    try:
        for _ in range(120):  # up to ~240s for first-run model load
            if server.poll() is not None:
                raise RuntimeError(f"uvicorn exited early (code {server.returncode})")
            try:
                urllib.request.urlopen(f"{base}/health", timeout=20)  # noqa: S310
                break
            except Exception:
                time.sleep(2)
        else:
            raise RuntimeError("API did not become healthy within 240s")

        health = json.loads(urllib.request.urlopen(f"{base}/health", timeout=5).read())  # noqa: S310
        lines.append(f"health: {health}")

        unauth = _post(base, "/query", {"question": "hi"}, api_key=None)
        lines.append(f"unauth /query -> HTTP {unauth}")
        assert unauth == 401, unauth

        doc = REPO / "tests" / "evals" / "fixtures" / "acme_handbook.md"
        ing = _post_file(base, "/ingest", doc, API_KEY)
        lines.append(f"ingest -> {ing}")
        assert ing.get("chunks", 0) > 0

        question = "How many paid annual leave days do full-time employees get?"
        ans = _post(base, "/query", {"question": question}, API_KEY)
        lines.append(f"query answer: {ans['answer']!r}")
        lines.append(f"query citations: {ans['citations']}")
        assert "25" in ans["answer"]
        assert ans["citations"], "grounded answer must cite a passage"
        assert ans["refused"] is False and ans["degraded"] is False

        raw = urllib.request.urlopen(f"{base}/metrics", timeout=5).read().decode()  # noqa: S310
        counters = [ln for ln in raw.splitlines() if ln.startswith("rag_") and "_total" in ln and "{" in ln]
        lines.extend(counters[:6])

        step.status = "PASS"
    except Exception as exc:
        step.status = "FAIL"
        lines.append(f"ERROR: {exc}")
        tail = server_log.read_text(errors="replace").strip().splitlines()[-15:]
        lines.append("--- uvicorn log tail ---")
        lines.extend(tail)
    finally:
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()
        step.seconds = time.perf_counter() - started
        step.detail = "\n".join(lines)


def _post(base: str, path: str, body: dict, api_key: str | None) -> int | dict:
    data = json.dumps(body).encode()
    req = urllib.request.Request(f"{base}{path}", data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    if api_key:
        req.add_header("X-API-Key", api_key)
    try:
        resp = urllib.request.urlopen(req, timeout=120)  # noqa: S310
        return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return e.code


def _post_file(base: str, path: str, file_path: Path, api_key: str) -> dict:
    boundary = "----validate"
    body = (
        (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="file"; filename="{file_path.name}"\r\n'
            f"Content-Type: text/markdown\r\n\r\n"
        ).encode()
        + file_path.read_bytes()
        + f"\r\n--{boundary}--\r\n".encode()
    )
    req = urllib.request.Request(f"{base}{path}", data=body, method="POST")
    req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
    req.add_header("X-API-Key", api_key)
    resp = urllib.request.urlopen(req, timeout=300)  # noqa: S310
    return json.loads(resp.read())


def main() -> int:
    ollama = _ollama_up()
    steps: list[Step] = []

    lint = Step("ruff check + format")
    steps.append(lint)
    p1 = lint.run_cmd(["uvx", "ruff", "check", "."])
    if p1.returncode == 0:
        lint.run_cmd(["uvx", "ruff", "format", "--check", "."])

    unit = Step("pytest -m 'not integration'")
    steps.append(unit)
    unit.run_cmd([PY, "-m", "pytest", "-m", "not integration", "-q"])

    integ = Step("pytest -m integration", required=False)
    steps.append(integ)
    if ollama:
        integ.run_cmd([PY, "-m", "pytest", "-m", "integration", "-q"])
    else:
        integ.skip("Ollama not serving")

    evals = Step("eval quality gate", required=False)
    steps.append(evals)
    if ollama:
        evals.run_cmd([PY, "-m", "tests.evals.run_evals"])
    else:
        evals.skip("Ollama not serving")

    smoke = Step("live API smoke")
    steps.append(smoke)
    _smoke(smoke)

    _write_report(steps, ollama)

    failed = [s for s in steps if s.status == "FAIL" and s.required]
    print("\n".join(f"{s.status:5} {s.name} ({s.seconds:.1f}s)" for s in steps))
    print(f"\nReport: {REPORT}")
    return 1 if failed else 0


def _write_report(steps: list[Step], ollama: bool) -> None:
    now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "# Validation Report",
        "",
        f"- Generated: {now}",
        f"- Python: `{sys.version.split()[0]}`",
        f"- Ollama reachable: **{ollama}**",
        "",
        "| Step | Result | Time |",
        "|------|--------|------|",
    ]
    for s in steps:
        lines.append(f"| {s.name} | **{s.status}** | {s.seconds:.1f}s |")
    lines.append("")
    for s in steps:
        lines += [f"## {s.name} — {s.status}", "", "```", s.detail or "(no output)", "```", ""]
    REPORT.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
