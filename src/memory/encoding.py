"""Deterministic, explainable memory encoding controls."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class EncodingDecision:
    importance_score: float
    novelty: float
    surprise: float
    task_relevance: float
    repetition: float
    outcome_signal: float
    correction_signal: float
    future_use: float
    privacy_penalty: float
    reason: str


_EMPHASIS = re.compile(r"\b(important|remember|always|never|required|must|do not|don't|prefer|decision)\b", re.I)
_SURPRISE = re.compile(r"\b(failed|failure|error|unexpected|wrong|fixed|corrected|blocked|regression)\b", re.I)
_OUTCOME = re.compile(r"\b(success|succeeded|passed|completed|shipped|resolved|works|worked)\b", re.I)
_PRIVATE = re.compile(r"\b(password|secret|token|api[_ -]?key|ssn|private key)\b", re.I)


def _signal(pattern: re.Pattern[str], text: str) -> float:
    return 1.0 if pattern.search(text) else 0.0


def score_observation(text: str, *, repeated: float = 0.0, task_relevance: float = 0.5) -> EncodingDecision:
    """Score an observation without deleting or hiding its raw evidence."""
    normalized = " ".join(text.split())
    emphasis = _signal(_EMPHASIS, normalized)
    surprise = _signal(_SURPRISE, normalized)
    outcome = _signal(_OUTCOME, normalized)
    correction = float(bool(re.search(r"\b(correct|instead|replace|supersed|no longer|changed)\b", normalized, re.I)))
    future_use = min(1.0, max(task_relevance, emphasis * 0.9, correction * 0.85))
    privacy = _signal(_PRIVATE, normalized)
    novelty = max(0.0, 1.0 - min(1.0, repeated))
    raw = (
        0.20 * emphasis
        + 0.20 * novelty
        + 0.18 * surprise
        + 0.14 * task_relevance
        + 0.10 * min(1.0, repeated)
        + 0.08 * outcome
        + 0.12 * correction
        + 0.10 * future_use
        - 0.15 * privacy
    )
    score = min(1.0, max(0.0, raw))
    reasons = [
        name
        for name, value in (
            ("explicit_emphasis", emphasis),
            ("novel", novelty),
            ("surprising", surprise),
            ("task_relevant", task_relevance),
            ("outcome", outcome),
            ("correction", correction),
            ("future_use", future_use),
            ("privacy_sensitive", privacy),
        )
        if value > 0
    ]
    return EncodingDecision(
        score,
        novelty,
        surprise,
        task_relevance,
        min(1.0, repeated),
        outcome,
        correction,
        future_use,
        privacy,
        ",".join(reasons) or "low_signal",
    )
