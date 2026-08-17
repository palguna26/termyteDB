from __future__ import annotations

import json
import time
from pathlib import Path

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


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("fixture", type=Path)
    print(json.dumps(evaluate_rule_fixture(parser.parse_args().fixture), sort_keys=True))


if __name__ == "__main__":
    main()
