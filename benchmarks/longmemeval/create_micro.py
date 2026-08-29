"""Create a deterministic 30-question LongMemEval-S smoke-test subset."""

from __future__ import annotations

import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "benchmarks" / "longmemeval" / "longmemeval_s_cleaned.json"
OUTPUT = ROOT / "benchmarks" / "longmemeval" / "longmemeval_micro.json"
ALIAS = ROOT / "benchmarks" / "longmemeval" / "longmemeval-micro.json"
SEED = 42
PER_DOMAIN = 5
DOMAINS = (
    "single-session-user",
    "single-session-assistant",
    "single-session-preference",
    "knowledge-update",
    "temporal-reasoning",
    "multi-session",
)


def main() -> None:
    dataset = json.loads(SOURCE.read_text(encoding="utf-8"))
    rng = random.Random(SEED)
    selected = []
    for domain in DOMAINS:
        candidates = [
            item
            for item in dataset
            if item.get("question_type") == domain
            and "_abs" not in str(item.get("question_id", ""))
        ]
        if len(candidates) < PER_DOMAIN:
            raise ValueError(f"not enough non-abstention samples for {domain}")
        selected.extend(rng.sample(candidates, PER_DOMAIN))

    selected.sort(key=lambda item: (DOMAINS.index(item["question_type"]), item["question_id"]))
    payload = json.dumps(selected, ensure_ascii=False, indent=2) + "\n"
    OUTPUT.write_text(payload, encoding="utf-8")
    ALIAS.write_text(payload, encoding="utf-8")
    print(f"Wrote {len(selected)} questions to {OUTPUT}")
    for domain in DOMAINS:
        print(f"  {domain}: {sum(item['question_type'] == domain for item in selected)}")


if __name__ == "__main__":
    main()
