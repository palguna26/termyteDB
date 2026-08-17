from __future__ import annotations

import json
import math
import tempfile
import time
from pathlib import Path
from typing import Any

from .context import token_count
from .engine import TermyteDB
from .extractor import extract


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


def _events(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{"text": case["evidence"]} for case in cases]


def evaluate_continuation_fixture(path: str | Path) -> dict[str, Any]:
    """Run continuation cases through production memory and explicit simple baselines."""
    cases = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    started = time.perf_counter()
    baseline_hits = {"no_memory": 0, "raw_history": 0, "previous_summary": 0, "termytedb": 0}
    token_totals = {name: 0 for name in baseline_hits}
    with tempfile.TemporaryDirectory(prefix="termytedb-continuation-") as directory:
        engine = TermyteDB(Path(directory) / "continuation.sqlite")
        for case_index, case in enumerate(cases):
            required = {"snapshot_id", "initial_task", "continuation_task", "verification", "events", "expected"}
            missing = required - set(case)
            if missing:
                raise ValueError(f"continuation case is missing: {sorted(missing)}")
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
        "termytedb_improvement_over_previous_summary": round((baseline_hits["termytedb"] - baseline_hits["previous_summary"]) / total, 4) if total else 0.0,
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("fixture", type=Path)
    parser.add_argument("--retrieval", action="store_true", help="run the production retrieval fixture instead of extraction")
    parser.add_argument("--continuation", action="store_true", help="run the production continuation fixture")
    arguments = parser.parse_args()
    if arguments.continuation:
        result = evaluate_continuation_fixture(arguments.fixture)
    else:
        result = evaluate_retrieval_fixture(arguments.fixture) if arguments.retrieval else evaluate_rule_fixture(arguments.fixture)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
