from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, cast
from uuid import UUID

from ..core.redaction import redact_text
from ..models import EvidenceSpan, ExtractionCandidate, MemoryKind
from .extractor import Candidate as RuleCandidate
from .extractor import payload_text


class CandidateRejected(ValueError):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class ValidatedCandidate:
    candidate: ExtractionCandidate
    fingerprint: str


def candidate_fingerprint(candidate: ExtractionCandidate) -> str:
    normalized = normalize_statement(candidate.statement)
    value = f"{candidate.kind}|{normalize_subject(candidate.subject)}|{normalized}"
    return hashlib.sha256(value.encode()).hexdigest()


def normalize_subject(value: str) -> str:
    return " ".join(value.strip().split()).casefold()


def normalize_statement(value: str) -> str:
    return " ".join(value.strip().split())


def validate_candidate(
    namespace_id: str,
    candidate: ExtractionCandidate,
    included_events: dict[UUID, str],
    *,
    strict: bool | None = None,
    included_chunks: set[str] | None = None,
    require_evidence: bool = False,
) -> ValidatedCandidate:
    # Structural checks remain strict; heuristic semantic check is relaxed for LLM-first flow
    if strict is None:
        # Default: relaxed unless explicitly requested, to honor LLM semantic judgement
        # Keep structural checks; relax semantic_support threshold for LLM candidates
        strict = False
    statement = normalize_statement(candidate.statement)
    subject = normalize_subject(candidate.subject)
    if not statement or len(statement) < 3:
        raise CandidateRejected("empty_or_generic_statement")
    if len(set(statement.casefold().split())) <= 1:
        raise CandidateRejected("generic_statement")
    if redact_text(statement) != statement:
        raise CandidateRejected("secret_in_statement")
    if candidate.valid_from and candidate.valid_until and candidate.valid_until <= candidate.valid_from:
        raise CandidateRejected("invalid_validity_interval")
    # Additional structural checks required by plan: valid schema, event IDs, evidence exists, confidence, durability, non-empty
    # These are partly enforced by Pydantic, but add explicit range checks for durability/confidence
    if not (0 <= candidate.confidence <= 1):
        raise CandidateRejected("invalid_confidence_range")
    if candidate.durability not in {"permanent", "session", "task"}:
        raise CandidateRejected("invalid_durability")
    if included_chunks is not None and any(chunk_id not in included_chunks for chunk_id in candidate.source_chunk_ids):
        raise CandidateRejected("unknown_source_chunk_id")
    if require_evidence and not candidate.evidence:
        raise CandidateRejected("missing_source_evidence")
    if not candidate.statement or not candidate.statement.strip():
        raise CandidateRejected("empty_statement")
    evidence_excerpts: list[str] = []
    if candidate.evidence:
        for span in candidate.evidence:
            if str(span.event_id) not in {str(event_id) for event_id in included_events}:
                raise CandidateRejected("evidence_not_in_extraction_input")
            source = included_events[span.event_id]
            if span.end_offset > len(source) or span.start_offset >= span.end_offset:
                raise CandidateRejected("invalid_evidence_span")
            actual = source[span.start_offset : span.end_offset]
            if actual != span.excerpt:
                raise CandidateRejected("evidence_excerpt_mismatch")
            if redact_text(actual) != actual:
                raise CandidateRejected("secret_in_evidence")
            evidence_excerpts.append(actual)
    # Evidence is now optional; skip semantic support check when no evidence is supplied
    if evidence_excerpts:
        if strict:
            if not semantic_support(statement, " ".join(evidence_excerpts), subject):
                raise CandidateRejected("unsupported_statement")
        else:
            if not semantic_support(statement, " ".join(evidence_excerpts), subject, threshold=0.35 if getattr(candidate, "source_stage", None) else 0.45):
                raise CandidateRejected("unsupported_statement")
    normalized = candidate.model_copy(update={"subject": subject, "statement": statement})
    return ValidatedCandidate(normalized, candidate_fingerprint(normalized))


def semantic_support(statement: str, excerpt: str, subject: str = "", *, threshold: float = 0.60) -> bool:
    """Require most meaningful terms, while allowing normal LLM paraphrasing.

    Threshold is relaxed for LLM-first flow (0.35-0.45) vs strict rule fallback (0.60).
    """
    statement_terms = _significant_terms(statement)
    excerpt_terms = _significant_terms(excerpt)
    if not statement_terms:
        return False
    overlap = len(statement_terms & excerpt_terms) / len(statement_terms)
    if overlap >= threshold:
        return True
    subject_terms = _significant_terms(subject)
    predicate_terms = statement_terms - subject_terms
    return bool(subject_terms and predicate_terms and subject_terms <= excerpt_terms and predicate_terms & excerpt_terms)


_STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "service",
    "the",
    "to",
    "use",
    "user",
    "was",
    "were",
}


def _significant_terms(value: str) -> set[str]:
    terms: set[str] = set()
    for raw in re.findall(r"[\w./:-]+", value.casefold()):
        if len(raw) <= 2 or raw in _STOP_WORDS:
            continue
        term = raw
        for suffix in ("ing", "ed", "es", "s"):
            if len(term) > len(suffix) + 2 and term.endswith(suffix):
                term = term[: -len(suffix)]
                break
        if len(term) > 3 and term.endswith("e"):
            term = term[:-1]
        terms.add(term)
    return terms


def rule_candidate_to_contract(candidate: RuleCandidate, event_id: UUID, source: str) -> ExtractionCandidate:
    excerpt = source[candidate.start_offset : candidate.end_offset]
    supported_kinds = {"decision", "failure", "outcome", "constraint", "procedure", "attempt", "task_state", "question", "correction"}
    kind = candidate.kind if candidate.kind in supported_kinds else "fact"
    return ExtractionCandidate(
        kind=cast(MemoryKind, kind),
        subject=candidate.subject_key,
        statement=candidate.statement,
        evidence=[EvidenceSpan(event_id=event_id, start_offset=candidate.start_offset, end_offset=candidate.end_offset, excerpt=excerpt)],
        confidence=1.0,
        durability="session",
        intent="insert",
    )


def event_texts(event: Any) -> dict[UUID, str]:
    return {UUID(event["id"]): payload_text(__import__("json").loads(event["payload_json"]), event["type"])}


def utc(value: datetime | None) -> str | None:
    return value.astimezone(UTC).isoformat() if value else None
