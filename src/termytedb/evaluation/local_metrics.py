"""Deterministic local metrics for the memory loop.

These metrics deliberately avoid an LLM judge. They measure whether the
retrieval system surfaced expected evidence and avoided known stale facts.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class LocalCase:
    query: str
    expected_terms: frozenset[str]
    answerable: bool = True
    forbidden_terms: frozenset[str] = frozenset()


def _terms(text: str) -> set[str]:
    return {part.casefold() for part in re.findall(r"[\w-]+", text) if len(part) > 2}


def evaluate_cases(
    cases: Iterable[LocalCase], retrieved: Iterable[list[str]], abstained: Iterable[bool],
    raw_tokens: Iterable[int], packed_tokens: Iterable[int],
) -> dict[str, float]:
    case_list = list(cases)
    result_list = list(retrieved)
    abstention_list = list(abstained)
    raw_list = list(raw_tokens)
    packed_list = list(packed_tokens)
    if not case_list or len(case_list) != len(result_list) or len(case_list) != len(abstention_list):
        raise ValueError("case, retrieval, and abstention lengths must match")
    evidence_hits = 0
    contradiction_leaks = 0
    abstention_hits = 0
    for case, items, is_abstained in zip(case_list, result_list, abstention_list, strict=True):
        retrieved_terms = set().union(*(_terms(item) for item in items[:5])) if items else set()
        evidence_hits += int(case.expected_terms.issubset(retrieved_terms))
        contradiction_leaks += int(bool(case.forbidden_terms & retrieved_terms))
        abstention_hits += int(is_abstained is (not case.answerable))
    raw_total = sum(raw_list)
    packed_total = sum(packed_list)
    return {
        "evidence_recall_at_5": evidence_hits / len(case_list),
        "contradiction_leak_rate": contradiction_leaks / len(case_list),
        "safe_abstention_accuracy": abstention_hits / len(case_list),
        "token_compression_ratio": 1.0 - (packed_total / raw_total) if raw_total else 0.0,
    }
