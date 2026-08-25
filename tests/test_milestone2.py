from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError
from termytedb.errors import IdempotencyConflict
from termytedb.provider import FakeExtractionProvider
from termytedb.schemas import EvidenceSpan, ExtractionCandidate, ExtractionResponse

from termytedb import TermyteDB


def model_db(tmp_path: Path, text: str, candidate_factory):
    db = TermyteDB(tmp_path / "model.sqlite")
    receipt = db.ingest({"namespace_id": "model", "idempotency_key": "one", "type": "note", "payload": {"text": text}})
    end = len(text)
    candidate = candidate_factory(receipt.event_id, end)
    db.processor.provider = FakeExtractionProvider(ExtractionResponse(schema_version="extraction-v1", prompt_version="prompt-v1", candidates=[candidate]))
    return db


def test_model_proposal_is_validated_and_audited(tmp_path: Path):
    text = "The service uses SQLite."
    db = model_db(
        tmp_path,
        text,
        lambda event_id, end: ExtractionCandidate(
            kind="fact",
            subject="storage",
            statement=text,
            evidence=[EvidenceSpan(event_id=event_id, start_offset=0, end_offset=end, excerpt=text)],
            confidence=0.9,
            durability="permanent",
        ),
    )
    assert db.process("model").accepted == 1
    assert db.search("model", "SQLite")
    run = db.database.execute("SELECT provider_name, schema_version FROM extraction_runs").fetchone()
    assert tuple(run) == ("fake", "extraction-v1")
    assert db.database.execute("SELECT action FROM extraction_decisions").fetchone()[0] == "INSERT"
    db.close()


def test_unsupported_model_candidate_is_rejected_without_memory(tmp_path: Path):
    text = "The service uses SQLite."
    db = model_db(
        tmp_path,
        text,
        lambda event_id, end: ExtractionCandidate(
            kind="fact",
            subject="unrelated",
            statement="The service uses PostgreSQL.",
            evidence=[EvidenceSpan(event_id=event_id, start_offset=0, end_offset=end, excerpt=text)],
            confidence=0.9,
            durability="session",
        ),
    )
    result = db.process("model")
    assert (result.accepted, result.rejected) == (0, 1)
    assert db.repository.memory_count("model") == 0
    assert db.database.execute("SELECT rejection_reason FROM extraction_decisions").fetchone()[0] == "unsupported_statement"
    db.close()


def test_model_paraphrase_with_supported_evidence_is_accepted(tmp_path: Path):
    text = "I used to live in Paris."
    db = model_db(
        tmp_path,
        text,
        lambda event_id, end: ExtractionCandidate(
            kind="fact",
            subject="user",
            statement="User lived in Paris.",
            evidence=[EvidenceSpan(event_id=event_id, start_offset=0, end_offset=end, excerpt=text)],
            confidence=0.9,
            durability="session",
        ),
    )
    assert db.process("model").accepted == 1
    assert db.repository.memory_count("model") == 1
    db.close()


def test_candidate_support_can_be_combined_across_exact_evidence_spans(tmp_path: Path):
    text = "The service uses SQLite and it runs locally."
    db = model_db(
        tmp_path,
        text,
        lambda event_id, _end: ExtractionCandidate(
            kind="fact",
            subject="storage",
            statement="The service uses SQLite and runs locally.",
            evidence=[
                EvidenceSpan(event_id=event_id, start_offset=0, end_offset=27, excerpt="The service uses SQLite and"),
                EvidenceSpan(event_id=event_id, start_offset=28, end_offset=44, excerpt="it runs locally."),
            ],
            confidence=0.9,
            durability="session",
        ),
    )
    assert db.process("model").accepted == 1
    assert db.search("model", "runs locally")
    db.close()


def test_model_retry_is_idempotent_and_correction_is_explicit(tmp_path: Path):
    text = "Correction: the service uses PostgreSQL instead."
    db = model_db(
        tmp_path,
        text,
        lambda event_id, end: ExtractionCandidate(
            kind="decision",
            subject="storage",
            statement=text,
            evidence=[EvidenceSpan(event_id=event_id, start_offset=0, end_offset=end, excerpt=text)],
            confidence=0.9,
            durability="session",
            intent="supersede",
        ),
    )
    assert db.process("model").accepted == 1
    assert db.process("model").processed == 0
    assert db.database.execute("SELECT COUNT(*) FROM memory_versions").fetchone()[0] == 1
    assert db.database.execute("SELECT COUNT(*) FROM extraction_decisions").fetchone()[0] == 1
    db.close()


def test_cross_namespace_and_invented_span_are_rejected(tmp_path: Path):
    db = TermyteDB(tmp_path / "cross.sqlite")
    db.ingest({"namespace_id": "a", "idempotency_key": "a", "type": "note", "payload": {"text": "SQLite is used."}})
    second = db.ingest({"namespace_id": "b", "idempotency_key": "b", "type": "note", "payload": {"text": "PostgreSQL is used."}})
    candidate = ExtractionCandidate(
        kind="fact",
        subject="storage",
        statement="SQLite is used.",
        evidence=[EvidenceSpan(event_id=second.event_id, start_offset=0, end_offset=15, excerpt="SQLite is used.")],
        confidence=1,
        durability="session",
    )
    db.processor.provider = FakeExtractionProvider(ExtractionResponse(schema_version="extraction-v1", prompt_version="p", candidates=[candidate]))
    assert db.process("a").rejected == 1
    assert db.repository.memory_count("a") == 0
    db.close()


def test_timestamp_identity_contract(db):
    base = {"namespace_id": "n1", "idempotency_key": "time", "type": "note", "payload": {"text": "same"}}
    first = db.ingest(base)
    retry = db.ingest(base)
    assert retry.duplicate is True and retry.event_id == first.event_id
    supplied = {**base, "idempotency_key": "timed", "occurred_at": datetime(2025, 1, 1, tzinfo=UTC)}
    timed = db.ingest(supplied)
    assert timed.duplicate is False
    assert db.ingest(supplied).duplicate is True
    changed_time = {**supplied, "occurred_at": datetime(2025, 1, 2, tzinfo=UTC)}
    with pytest.raises(IdempotencyConflict):
        db.ingest(changed_time)


def test_extraction_schema_rejects_unknown_fields_and_bad_values():
    with pytest.raises(ValidationError):
        ExtractionCandidate.model_validate(
            {"kind": "unknown", "subject": "x", "statement": "claim", "evidence": [], "confidence": 2, "durability": "session", "extra": 1}
        )


def test_labelled_fixture_has_at_least_fifty_cases():
    fixture = Path(__file__).parent / "fixtures" / "extraction_cases.jsonl"
    cases = [json.loads(line) for line in fixture.read_text().splitlines() if line.strip()]
    assert len(cases) >= 50
    assert all({"id", "label", "text", "expected"} <= case.keys() for case in cases)
