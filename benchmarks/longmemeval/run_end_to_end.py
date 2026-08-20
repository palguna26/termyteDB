from __future__ import annotations

import argparse
import json
import logging
import statistics
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from termytedb import TermyteDB
from termytedb.retrieval.embedding import FastEmbedProvider


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def reciprocal_rank(retrieved: list[str], expected: set[str]) -> float:
    for rank, session_id in enumerate(retrieved, 1):
        if session_id in expected:
            return 1.0 / rank
    return 0.0


def run(path: Path, top_k: int, database_path: Path | None = None, process_batch: int = 100, workers: int = 2) -> dict[str, object]:
    log(f"Loading dataset: {path}")
    questions = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(questions, list) or len(questions) != 500:
        raise ValueError("expected the official 500-question LongMemEval-S cleaned dataset")

    temporary: TemporaryDirectory[str] | None = None
    if database_path is None:
        temporary = TemporaryDirectory(prefix="termytedb-longmemeval-e2e-")
        database_path = Path(temporary.name) / "memory.sqlite"

    database_path.parent.mkdir(parents=True, exist_ok=True)
    embedding = FastEmbedProvider()
    benchmark_logger = logging.getLogger("termytedb.longmemeval")
    benchmark_logger.handlers.clear()
    benchmark_logger.addHandler(logging.NullHandler())
    benchmark_logger.setLevel(logging.WARNING)
    benchmark_logger.propagate = False
    db = TermyteDB(database_path, logger=benchmark_logger, embedding_provider=embedding)
    event_sessions: dict[str, str] = {}
    session_payloads: dict[str, list[dict[str, Any]]] = {}
    try:
        for question in questions:
            session_ids = question.get("haystack_session_ids", [])
            for session_index, session in enumerate(question.get("haystack_sessions", [])):
                if not isinstance(session, list):
                    continue
                if session_index >= len(session_ids):
                    continue
                session_id = str(session_ids[session_index])
                session_payloads.setdefault(session_id, session)

        log(f"Ingesting {len(session_payloads)} unique sessions through TermyteDB.ingest()")
        for index, (session_id, messages) in enumerate(session_payloads.items(), 1):
            receipt = db.ingest({
                "namespace_id": "longmemeval-e2e",
                "idempotency_key": f"longmemeval:{session_id}",
                "type": "conversation.session",
                "session_id": session_id,
                "stream_id": session_id,
                "payload": {"messages": messages},
            })
            event_sessions[str(receipt.event_id)] = session_id
            if index == 1 or index % 500 == 0:
                log(f"Ingested {index}/{len(session_payloads)} sessions")

        if workers < 1:
            raise ValueError("workers must be positive")
        log(f"Processing events through extraction, validation, reconciliation, and embedding with {workers} workers")
        processed = 0
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="termytedb-worker") as executor:
            while True:
                responses = list(executor.map(
                    lambda _: db.process_with_timeout("longmemeval-e2e", limit=process_batch, timeout_seconds=120.0),
                    range(workers),
                ))
                processed += sum(response.processed for response in responses)
                accepted = sum(response.accepted for response in responses)
                rejected = sum(response.rejected for response in responses)
                pending = int(db.metrics("longmemeval-e2e")["jobs_pending"])
                log(f"Processed {processed} jobs; accepted={accepted}; rejected={rejected}; pending={pending}")
                if pending == 0:
                    break

        rows: list[dict[str, object]] = []
        by_type: dict[str, list[bool]] = defaultdict(list)
        query_times: list[float] = []
        for index, question in enumerate(questions, 1):
            expected = {str(item) for item in question.get("answer_session_ids", [])}
            if not expected:
                continue
            query_started = time.perf_counter()
            hits = db.search("longmemeval-e2e", str(question["question"]), limit=top_k)
            query_times.append((time.perf_counter() - query_started) * 1000)
            retrieved: list[str] = []
            for hit in hits:
                for citation in hit.citations:
                    session_id = event_sessions.get(str(citation.event_id))
                    if session_id and session_id not in retrieved:
                        retrieved.append(session_id)
            retrieved = retrieved[:top_k]
            hit = bool(expected.intersection(retrieved))
            question_type = str(question.get("question_type", "unknown"))
            by_type[question_type].append(hit)
            rows.append({
                "question_id": question["question_id"],
                "question_type": question_type,
                "retrieved": retrieved,
                "expected": sorted(expected),
                "hit": hit,
                "mrr": reciprocal_rank(retrieved, expected),
            })
            if index == 1 or index % 25 == 0:
                log(f"Evaluated {index}/500 questions; latest hit={'yes' if hit else 'no'}")

        result = {
            "dataset": path.name,
            "pipeline": "TermyteDB.ingest -> process -> validate -> reconcile -> search",
            "questions": len(rows),
            "top_k": top_k,
            "unique_sessions": len(session_payloads),
            "processed_jobs": processed,
            "memories": int(db.metrics("longmemeval-e2e")["memories"]),
            "query_p50_ms": round(statistics.median(query_times), 2),
            "query_p95_ms": round(sorted(query_times)[max(0, int(len(query_times) * 0.95) - 1)], 2),
            "recall": round(sum(int(row["hit"]) for row in rows) / len(rows), 4),
            "mrr": round(sum(float(row["mrr"]) for row in rows) / len(rows), 4),
            "by_type": {key: round(sum(values) / len(values), 4) for key, values in sorted(by_type.items())},
            "misses": [row for row in rows if not row["hit"]],
        }
        log(f"Completed: Recall@{top_k}={result['recall']}; MRR@{top_k}={result['mrr']}")
        return result
    finally:
        db.close()
        if temporary is not None:
            temporary.cleanup()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--database", type=Path)
    parser.add_argument("--process-batch", type=int, default=100)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    result = run(args.dataset, args.top_k, args.database, args.process_batch, args.workers)
    rendered = json.dumps(result, indent=2, ensure_ascii=False)
    print(rendered)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
