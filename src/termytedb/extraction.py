from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, cast
from uuid import UUID

from .extractor import Candidate as RuleCandidate
from .extractor import payload_text
from .redaction import redact_text
from .schemas import EvidenceSpan, ExtractionCandidate, MemoryKind


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
) -> ValidatedCandidate:
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
        if not semantic_support(statement, actual):
            raise CandidateRejected("unsupported_statement")
    normalized = candidate.model_copy(update={"subject": subject, "statement": statement})
    return ValidatedCandidate(normalized, candidate_fingerprint(normalized))


def semantic_support(statement: str, excerpt: str) -> bool:
    """Conservative deterministic verifier; model agreement is never authoritative."""
    statement_terms = {term.casefold() for term in re.findall(r"[\w./:-]+", statement) if len(term) > 2}
    excerpt_terms = {term.casefold() for term in re.findall(r"[\w./:-]+", excerpt)}
    return bool(statement_terms) and len(statement_terms & excerpt_terms) == len(statement_terms)


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
    return {UUID(event["id"]): payload_text(__import__("json").loads(event["payload_json"]))}


def utc(value: datetime | None) -> str | None:
    return value.astimezone(UTC).isoformat() if value else None
