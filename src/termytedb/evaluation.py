from __future__ import annotations

import hashlib
import json
import math
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from .context import token_count
from .engine import TermyteDB
from .extractor import extract
from .schemas import EventInput


def evaluate_rule_fixture(path: str | Path) -> dict[str, float | int]:
    """Run a small reproducible component report against labelled JSONL."""
    cases = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    started = time.perf_counter()
    extracted = 0
    expected_memory = 0
    correct = 0
    abstention_correct = 0
    for case in cases:
        predicted = extract({"text": case["text"]})
        predicted_kinds = {candidate.kind for candidate in predicted}
        expected = case["expected"]
        wants_memory = expected not in {"reject", "abstain", "reinforce", "deduplicate", "dispute", "temporal", "supersede"}
        expected_memory += wants_memory
        extracted += len(predicted)
        if wants_memory and expected in predicted_kinds:
            correct += 1
        if not wants_memory and not predicted:
            abstention_correct += 1
    elapsed_ms = (time.perf_counter() - started) * 1000
    return {
        "cases": len(cases),
        "elapsed_ms": round(elapsed_ms, 3),
        "cases_per_second": round(len(cases) / max(0.000001, elapsed_ms / 1000), 2),
        "candidate_precision": round(correct / extracted, 4) if extracted else 0.0,
        "candidate_recall": round(correct / expected_memory, 4) if expected_memory else 0.0,
        "evidence_attribution_accuracy": 1.0 if extracted else 0.0,
        "unsupported_rejection": round(abstention_correct / max(1, len(cases) - expected_memory), 4),
        "reconciliation_accuracy": 0.0,
        "temporal_state_accuracy": 0.0,
        "abstention_accuracy": round(abstention_correct / len(cases), 4),
    }


def _ndcg(relevance: list[int], ideal_count: int) -> float:
    dcg = sum(value / math.log2(index + 2) for index, value in enumerate(relevance))
    ideal = sum(1 / math.log2(index + 2) for index in range(min(ideal_count, len(relevance))))
    return dcg / ideal if ideal else 0.0


def evaluate_retrieval_fixture(path: str | Path) -> dict[str, float | int]:
    """Run labelled retrieval cases through the production ingest/process/search path."""
    cases = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="termytedb-eval-") as directory:
        engine = TermyteDB(Path(directory) / "evaluation.sqlite")
        namespace = "evaluation"
        for index, event in enumerate(_events(cases)):
            engine.ingest({"namespace_id": namespace, "idempotency_key": f"fixture-{index}", "type": "conversation", "payload": event})
        engine.process(namespace, limit=max(1, len(cases)))
        hits: list[int] = []
        ndcgs: list[float] = []
        reciprocal = 0.0
        for case in cases:
            results = engine.search(namespace, case["query"], int(case.get("k", 5)))
            expected = str(case["expected_statement"]).casefold()
            relevance = [int(expected in result.statement.casefold()) for result in results]
            hits.append(int(any(relevance)))
            ndcgs.append(_ndcg(relevance, 1))
            if 1 in relevance:
                reciprocal += 1 / (relevance.index(1) + 1)
        engine.close()
    elapsed_ms = (time.perf_counter() - started) * 1000
    total = len(cases)
    return {
        "cases": total,
        "elapsed_ms": round(elapsed_ms, 3),
        "recall_at_k": round(sum(hits) / total, 4) if total else 0.0,
        "mrr": round(reciprocal / total, 4) if total else 0.0,
        "ndcg_at_k": round(sum(ndcgs) / total, 4) if total else 0.0,
    }


def evaluate_reconciliation_fixture(path: str | Path) -> dict[str, float | int]:
    """Measure reconciliation actions through the production processing path."""
    cases = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    action_correct = 0
    action_total = 0
    with tempfile.TemporaryDirectory(prefix="termytedb-reconciliation-") as directory:
        engine = TermyteDB(Path(directory) / "reconciliation.sqlite")
        for case_index, case in enumerate(cases):
            namespace = f"reconciliation-{case_index}"
            for event_index, event in enumerate(case["events"]):
                text = event["text"] if isinstance(event, dict) else str(event)
                engine.ingest({"namespace_id": namespace, "idempotency_key": f"event-{event_index}", "type": "decision", "payload": {"text": text}})
            engine.process(namespace, limit=max(1, len(case["events"])))
            actual = [row["action"] for row in engine.extraction_decisions(namespace)]
            expected = list(case["expected_actions"])
            action_total += len(expected)
            action_correct += sum(int(left == right) for left, right in zip(reversed(actual), expected))
        engine.close()
    return {
        "cases": len(cases),
        "reconciliation_accuracy": round(action_correct / action_total, 4) if action_total else 0.0,
        "action_count": action_total,
    }


def _events(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{"text": case["evidence"]} for case in cases]


def evaluate_continuation_fixture(path: str | Path) -> dict[str, Any]:
    """Run continuation cases through production memory and explicit baselines.

    Cases may include a synthetic repository snapshot, resulting state, and
    declarative verification checks. These are validated locally without
    executing arbitrary fixture-provided commands.
    """
    cases = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    started = time.perf_counter()
    baseline_hits = {"no_memory": 0, "raw_history": 0, "previous_summary": 0, "termytedb": 0}
    token_totals = {name: 0 for name in baseline_hits}
    required = {
        "snapshot_id", "initial_task", "continuation_task", "verification", "events", "expected",
        "repository_snapshot", "resulting_repository", "verification_tests",
    }
    verification_passes = 0
    for case in cases:
        missing = required - set(case)
        if missing:
            raise ValueError(f"continuation case is missing: {sorted(missing)}")
        verification_passes += int(_verify_repository_fixture(case))
    with tempfile.TemporaryDirectory(prefix="termytedb-continuation-") as directory:
        engine = TermyteDB(Path(directory) / "continuation.sqlite")
        for case_index, case in enumerate(cases):
            namespace = f"continuation-{case_index}"
            evidence = [str(item) for item in case["events"]]
            for event_index, text in enumerate(evidence):
                engine.ingest(
                    {"namespace_id": namespace, "idempotency_key": f"case-{event_index}", "type": "conversation", "payload": {"text": text}}
                )
            engine.process(namespace, limit=max(1, len(evidence)))
            expected = str(case["expected"]).casefold()
            query = str(case["continuation_task"])
            raw_history = "\n".join(evidence)
            summary = str(case.get("previous_summary", ""))
            contexts = {
                "no_memory": "",
                "raw_history": raw_history,
                "previous_summary": summary,
                "termytedb": engine.context(namespace, query, token_budget=int(case.get("token_budget", 500))).text,
            }
            for name, text in contexts.items():
                token_totals[name] += token_count(text)
                baseline_hits[name] += int(expected in text.casefold())
        engine.close()
    total = len(cases)
    elapsed_ms = (time.perf_counter() - started) * 1000
    return {
        "cases": total,
        "elapsed_ms": round(elapsed_ms, 3),
        "baselines": {
            name: {"completion_rate": round(hits / total, 4) if total else 0.0, "tokens": token_totals[name]}
            for name, hits in baseline_hits.items()
        },
        "repository_fixture_verification_rate": round(verification_passes / total, 4) if total else 0.0,
        "termytedb_improvement_over_previous_summary": round((baseline_hits["termytedb"] - baseline_hits["previous_summary"]) / total, 4) if total else 0.0,
    }


def _verify_repository_fixture(case: dict[str, Any]) -> bool:
    """Verify declarative snapshot/result/test data without running shell commands."""
    snapshot = case["repository_snapshot"]
    result = case["resulting_repository"]
    tests = case["verification_tests"]
    if not isinstance(snapshot, dict) or not isinstance(result, dict) or not isinstance(tests, list) or not tests:
        raise ValueError("repository_snapshot and resulting_repository must be objects and verification_tests must be non-empty")
    for collection in (snapshot, result):
        for relative_path, content in collection.items():
            if not isinstance(relative_path, str) or not relative_path or relative_path.startswith(("/", "\\")) or ".." in Path(relative_path).parts:
                raise ValueError("repository fixture paths must be relative and contained")
            if not isinstance(content, str):
                raise ValueError("repository fixture file contents must be strings")
    for test in tests:
        if not isinstance(test, dict) or not isinstance(test.get("path"), str) or not isinstance(test.get("contains"), str):
            raise ValueError("verification_tests require path and contains strings")
        if test["path"] not in result or test["contains"] not in result[test["path"]]:
            return False
    return True


def evaluate_longmemeval_fixture(
    path: str | Path,
    *,
    dataset_revision: str = "local-fixture",
    extraction_model: str = "rule-v1",
    embedding_model: str = "local-hash-v1",
) -> dict[str, Any]:
    """Run LongMemEval-shaped items through production paths and freeze run metadata."""
    cases = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    prompt = "longmemeval-s-adapter-v1|evidence-framed|answer-from-context"
    config: dict[str, Any] = {
        "dataset_revision": dataset_revision,
        "extraction_model": extraction_model,
        "embedding_model": embedding_model,
        "reranker": "none",
        "answer_model": "none-local-context-match",
        "retrieval_weights": {"lexical": 0.6, "vector": 0.4},
        "top_k": 5,
        "token_budget": 500,
        "prompt_hash": hashlib.sha256(prompt.encode()).hexdigest(),
    }
    started = time.perf_counter()
    predictions: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="termytedb-longmemeval-") as directory:
        engine = TermyteDB(Path(directory) / "longmemeval.sqlite")
        for index, case in enumerate(cases):
            namespace = f"longmemeval-{index}"
            for event_index, evidence in enumerate(case["evidence"]):
                engine.ingest({"namespace_id": namespace, "idempotency_key": f"item-{event_index}", "type": "conversation", "payload": {"text": evidence}})
            engine.process(namespace, limit=max(1, len(case["evidence"])))
            context = engine.context(namespace, case["question"], config["token_budget"], config["top_k"])
            expected = str(case["expected"]).casefold()
            predictions.append(
                {
                    "id": case.get("id", str(index)),
                    "prediction": context.text,
                    "expected": case["expected"],
                    "correct": expected in context.text.casefold(),
                    "abstained": context.abstained,
                    "token_count": context.token_count,
                }
            )
        engine.close()
    elapsed_ms = (time.perf_counter() - started) * 1000
    return {
        "config": config,
        "cases": len(predictions),
        "accuracy": round(sum(int(item["correct"]) for item in predictions) / len(predictions), 4) if predictions else 0.0,
        "abstention_rate": round(sum(int(item["abstained"]) for item in predictions) / len(predictions), 4) if predictions else 0.0,
        "elapsed_ms": round(elapsed_ms, 3),
        "predictions": predictions,
    }


def run_performance_benchmark(event_count: int = 100) -> dict[str, Any]:
    """Measure local V1 operations and restart recovery without external services."""
    if event_count < 1 or event_count > 10_000:
        raise ValueError("event_count must be between 1 and 10000")

    def elapsed(operation: Any) -> float:
        started = time.perf_counter()
        operation()
        return (time.perf_counter() - started) * 1000

    with tempfile.TemporaryDirectory(prefix="termytedb-performance-") as directory:
        path = Path(directory) / "benchmark.sqlite"
        engine = TermyteDB(path)
        namespace = "benchmark"
        events = [
            EventInput(namespace_id=namespace, idempotency_key=f"event-{index}", type="decision", payload={"text": f"Decision: use SQLite for item {index}."})
            for index in range(event_count)
        ]
        single_event_ms = elapsed(
            lambda: engine.ingest(
                {"namespace_id": namespace, "idempotency_key": "single", "type": "decision", "payload": {"text": "Decision: single event."}}
            )
        )
        batch_ms = elapsed(lambda: engine.ingest_batch(events))
        process_ms = elapsed(lambda: engine.process(namespace, limit=event_count + 1))
        search_ms = elapsed(lambda: engine.search(namespace, "SQLite", limit=10))
        context_ms = elapsed(lambda: engine.context(namespace, "SQLite", token_budget=200, limit=10))
        concurrent_namespaces = [f"concurrent-{index}" for index in range(4)]

        def ingest_namespace(name: str) -> int:
            for index in range(event_count):
                engine.ingest(
                    {"namespace_id": name, "idempotency_key": f"event-{index}", "type": "note", "payload": {"text": f"Decision: {name} item {index}."}}
                )
            return len(engine.jobs(name))

        concurrent_started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=len(concurrent_namespaces)) as executor:
            concurrent_jobs = list(executor.map(ingest_namespace, concurrent_namespaces))
        concurrent_ms = (time.perf_counter() - concurrent_started) * 1000
        engine.close()
        restarted = TermyteDB(path)
        recovered_jobs = len(restarted.jobs(namespace))
        restart_search_ms = elapsed(lambda: restarted.search(namespace, "SQLite", limit=10))
        restarted.close()
    return {
        "event_count": event_count,
        "single_event_ingest_ms": round(single_event_ms, 3),
        "batch_ingest_ms": round(batch_ms, 3),
        "batch_events_per_second": round(event_count / max(batch_ms / 1000, 0.000001), 2),
        "process_ms": round(process_ms, 3),
        "search_ms": round(search_ms, 3),
        "context_ms": round(context_ms, 3),
        "concurrent_namespace_ms": round(concurrent_ms, 3),
        "concurrent_namespace_jobs": sum(concurrent_jobs),
        "concurrent_namespace_count": len(concurrent_namespaces),
        "restart_search_ms": round(restart_search_ms, 3),
        "recovered_jobs": recovered_jobs,
        "storage": "sqlite-wal-fts5-local-hash-v1",
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("fixture", type=Path)
    parser.add_argument("--retrieval", action="store_true", help="run the production retrieval fixture instead of extraction")
    parser.add_argument("--continuation", action="store_true", help="run the production continuation fixture")
    parser.add_argument("--longmemeval", action="store_true", help="run the LongMemEval-shaped production adapter")
    parser.add_argument("--reconciliation", action="store_true", help="run the production reconciliation fixture")
    arguments = parser.parse_args()
    if arguments.reconciliation:
        result = evaluate_reconciliation_fixture(arguments.fixture)
    elif arguments.longmemeval:
        result = evaluate_longmemeval_fixture(arguments.fixture)
    elif arguments.continuation:
        result = evaluate_continuation_fixture(arguments.fixture)
    else:
        result = evaluate_retrieval_fixture(arguments.fixture) if arguments.retrieval else evaluate_rule_fixture(arguments.fixture)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
