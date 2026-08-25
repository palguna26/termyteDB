"""LongMemEval-S benchmark for TermyteDB through OpenRouter."""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import threading
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from termytedb import TermyteDB  # noqa: E402
from termytedb.api.schemas import ExtractionRequest  # noqa: E402
from termytedb.memory.provider import (  # noqa: E402
    OpenRouterExtractionProvider,
    ProviderError,
    ProviderResult,
)
from termytedb.retrieval.embedding import FastEmbedProvider  # noqa: E402

DEFAULT_EXTRACTION_MODEL = "mistralai/mistral-nemo"
DEFAULT_URL = "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json?download=true"
DEFAULT_DATA_PATH = ROOT / "benchmarks" / "longmemeval" / "longmemeval_s_cleaned.json"
STOPWORDS = {"the", "and", "for", "with", "that", "this", "from", "what", "when", "where", "were", "was", "are"}


class FixedOpenRouterProvider:
    """Paced retries for one fixed OpenRouter model with automatic routing."""

    name = "openrouter"

    def __init__(self, *, min_delay: float = 3.0) -> None:
        self.api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("TERMYTEDB_EXTRACTION_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY is required; put it in .env")
        self.model = (
            os.environ.get("TERMYTEDB_EXTRACTION_MODEL")
            or os.environ.get("OPENROUTER_MODEL")
            or DEFAULT_EXTRACTION_MODEL
        )
        self.min_delay = min_delay
        self._last_request = 0.0
        self._lock = threading.Lock()

    def _pace(self) -> None:
        with self._lock:
            delay = self.min_delay - (time.monotonic() - self._last_request)
            if delay > 0:
                time.sleep(delay)
            self._last_request = time.monotonic()

    @staticmethod
    def _retryable(error: ProviderError) -> bool:
        return error.retryable and ("429" in str(error) or "HTTP 4" not in str(error) or any(code in str(error) for code in ("500", "502", "503", "504")))

    def extract(self, request: ExtractionRequest, timeout_seconds: float = 30.0, cancellation: Any = None) -> ProviderResult:
        last_error: ProviderError | None = None
        provider = OpenRouterExtractionProvider(model=self.model, api_key=self.api_key)
        for attempt in range(3):
            self._pace()
            try:
                return provider.extract(request, timeout_seconds=timeout_seconds, cancellation=cancellation)
            except ProviderError as error:
                last_error = error
                if not self._retryable(error):
                    break
                if attempt < 2:
                    time.sleep((2**attempt) + random.uniform(0.0, 0.5))
        if last_error is not None:
            raise last_error
        raise ProviderError("OpenRouter extraction failed", retryable=False, error_class="configuration")


def parse_time(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def normalize_samples(raw: Any) -> list[dict[str, Any]]:
    if isinstance(raw, dict):
        raw = raw.get("data", raw.get("samples", []))
    if not isinstance(raw, list):
        raise ValueError("LongMemEval-S must be a JSON list")
    return [item for item in raw if isinstance(item, dict)]


def sample_sessions(sample: dict[str, Any]) -> list[tuple[str, list[dict[str, Any]], datetime | None]]:
    sessions = sample.get("sessions", sample.get("haystack_sessions", []))
    ids = sample.get("session_ids", sample.get("haystack_session_ids", []))
    output: list[tuple[str, list[dict[str, Any]], datetime | None]] = []
    for index, session in enumerate(sessions if isinstance(sessions, list) else []):
        if not isinstance(session, list):
            continue
        session_id = str(ids[index]) if index < len(ids) else f"session-{index}"
        messages = [item for item in session if isinstance(item, dict)]
        occurred_at = next((parse_time(item.get("timestamp", item.get("time", item.get("created_at")))) for item in messages), None)
        output.append((session_id, messages, occurred_at))
    return output


def tokens(text: str) -> int:
    try:
        import tiktoken

        return len(tiktoken.get_encoding("cl100k_base").encode(text))
    except Exception:
        return len(re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE))


def answer_hit(context: str, answer: Any) -> bool:
    expected = " ".join(str(answer).split()).casefold()
    actual = " ".join(context.split()).casefold()
    if expected and expected in actual:
        return True
    terms = [term for term in re.findall(r"[a-z0-9]+", expected) if len(term) > 2 and term not in STOPWORDS]
    return bool(terms) and sum(term in actual for term in terms) >= max(1, (len(terms) + 1) // 2)


def is_unanswerable(sample: dict[str, Any]) -> bool:
    return bool(sample.get("unanswerable", sample.get("is_unanswerable", False)))


def download_dataset(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    url = os.environ.get("LONGMEMEVAL_S_URL", DEFAULT_URL)
    print(f"Dataset missing; downloading {url}", flush=True)
    try:
        with urlopen(url, timeout=60) as response:
            path.write_bytes(response.read())
    except Exception as error:
        raise RuntimeError(
            f"could not download LongMemEval-S from {url}. "
            "Download longmemeval_s.json manually and pass it with --data-path."
        ) from error
    return path


def run(args: argparse.Namespace) -> dict[str, Any]:
    data_path = args.data_path
    if not data_path.exists():
        download_dataset(data_path)
    samples = normalize_samples(json.loads(data_path.read_text(encoding="utf-8")))
    if args.task:
        samples = [item for item in samples if str(item.get("task", item.get("category", "unknown"))) == args.task]
    if args.limit:
        samples = samples[: args.limit]
    if not samples:
        raise ValueError("no LongMemEval-S samples matched the requested filters")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    database_path = args.output_dir / f"longmemeval_s_{time.strftime('%Y%m%d-%H%M%S')}.sqlite"
    provider = FixedOpenRouterProvider()
    embedding = FastEmbedProvider()
    db = TermyteDB(database_path, extraction_provider=provider, embedding_provider=embedding)
    traces: list[dict[str, Any]] = []
    try:
        for number, sample in enumerate(samples, 1):
            question_id = str(sample.get("question_id", sample.get("sample_id", number)))
            namespace = f"lme_{question_id}"
            question = str(sample.get("question", ""))
            sessions = sample_sessions(sample)
            raw_text = "\n".join(json.dumps(message, ensure_ascii=False) for _, messages, _ in sessions for message in messages)
            extraction_latencies: list[float] = []
            process_summaries: list[dict[str, Any]] = []
            for session_id, messages, occurred_at in sessions:
                db.ingest({
                    "namespace_id": namespace,
                    "idempotency_key": f"longmemeval:{question_id}:{session_id}",
                    "type": "conversation.session",
                    "session_id": session_id,
                    "stream_id": session_id,
                    "occurred_at": occurred_at,
                    "payload": {"messages": messages},
                })
                extraction_started = time.perf_counter()
                process = db.process_with_timeout(
                    namespace,
                    limit=1,
                    lease_seconds=args.lease_seconds,
                    timeout_seconds=args.process_timeout,
                )
                process_summaries.append(process.model_dump())
                extraction_latencies.append((time.perf_counter() - extraction_started) * 1000)
                if process.failed or process.dead_lettered:
                    print(
                        f"[{number}/{len(samples)}] {question_id}/{session_id}: "
                        f"processing failed; continuing benchmark ({process})",
                        flush=True,
                    )
            retrieval_started = time.perf_counter()
            ctx = db.context(namespace, question, limit=5, token_budget=args.token_budget)
            retrieval_ms = (time.perf_counter() - retrieval_started) * 1000
            context_text = ctx.text
            old_answer = sample.get("old_answer")
            trace = {
                "question_id": question_id,
                "task": str(sample.get("task", sample.get("category", "unknown"))),
                "namespace": namespace,
                "target_hit": answer_hit(context_text, sample.get("answer", "")),
                "contradiction_leak": bool(old_answer) and answer_hit(context_text, old_answer),
                "unanswerable": is_unanswerable(sample),
                "abstained": ctx.abstained,
                "abstention_correct": ctx.abstained == is_unanswerable(sample),
                "retrieved_tokens": ctx.token_count,
                "raw_session_tokens": tokens(raw_text),
                "compression_ratio": 1 - (ctx.token_count / max(1, tokens(raw_text))),
                "extraction_latency_ms": [round(value, 2) for value in extraction_latencies],
                "retrieval_latency_ms": round(retrieval_ms, 2),
                "context": context_text,
                "process": process_summaries,
            }
            traces.append(trace)
            print(f"[{number}/{len(samples)}] {question_id}: recall={'hit' if trace['target_hit'] else 'miss'}; retrieval={retrieval_ms:.0f}ms", flush=True)
    finally:
        db.close()

    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for trace in traces:
        by_task[trace["task"]].append(trace)
    summary = []
    for task, rows in sorted(by_task.items()):
        summary.append({
            "Task Category": task,
            "Samples": len(rows),
            "Recall@5 (%)": round(100 * sum(row["target_hit"] for row in rows) / len(rows), 2),
            "Contradiction Leak (%)": round(100 * sum(row["contradiction_leak"] for row in rows) / len(rows), 2),
            "Abstention Acc (%)": round(100 * sum(row["abstention_correct"] for row in rows) / len(rows), 2),
            "Avg Tokens": round(sum(row["retrieved_tokens"] for row in rows) / len(rows), 2),
            "Avg Retrieval Latency (ms)": round(sum(row["retrieval_latency_ms"] for row in rows) / len(rows), 2),
        })
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output_path = args.output_dir / f"longmemeval_s_{timestamp}.json"
    result = {
        "dataset": str(data_path),
        "samples": len(traces),
        "cost_usd": "unknown",
        "summary": summary,
        "traces": traces,
    }
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    headers = list(summary[0]) if summary else [
        "Task Category", "Samples", "Recall@5 (%)", "Contradiction Leak (%)",
        "Abstention Acc (%)", "Avg Tokens", "Avg Retrieval Latency (ms)",
    ]
    print("\n" + " | ".join(headers))
    print("-+-".join("-" * len(header) for header in headers))
    for row in summary:
        print(" | ".join(str(row.get(header, "")) for header in headers))
    print(f"\nTrace: {output_path}")
    return result


def main() -> int:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    parser = argparse.ArgumentParser(description="Run the zero-cost LongMemEval-S TermyteDB benchmark")
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--task")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--token-budget", type=int, default=500)
    parser.add_argument("--process-timeout", type=float, default=300.0)
    parser.add_argument("--lease-seconds", type=int, default=180)
    args = parser.parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
