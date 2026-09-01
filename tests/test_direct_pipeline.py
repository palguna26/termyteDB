import pytest

from src import TermyteDB
from src.memory.provider import FakeExtractionProvider, ProviderError, ProviderResult
from src.models import ExtractionCandidate, ExtractionResponse


class RecordingProvider(FakeExtractionProvider):
    def __init__(self):
        super().__init__()
        self.requests = []

    def extract(self, request, timeout_seconds=30.0, cancellation=None):
        self.requests.append(request)
        return super().extract(request, timeout_seconds, cancellation)


class RecordingEmbedding:
    name = "recording-v1"
    dimensions = 2

    def __init__(self):
        self.batches = []

    def embed(self, value):
        return [1.0, 0.0]

    def embed_many(self, values):
        self.batches.append(list(values))
        return [[1.0, 0.0] for _ in values]


class FailingProvider:
    name = "failing"
    model = "failing-v1"

    def extract(self, request, timeout_seconds=30.0, cancellation=None):
        raise ProviderError("provider unavailable", retryable=True, error_class="transport_error")


class SimpleResultProvider:
    name = "simple"
    model = "simple-v1"

    def extract(self, request, timeout_seconds=30.0, cancellation=None):
        return ProviderResult(
            response=ExtractionResponse(
                schema_version="extraction-v1",
                prompt_version="simple-v1",
                candidates=[
                    ExtractionCandidate(
                        kind="fact",
                        subject="user database preference",
                        statement="User prefers SQLite for local storage.",
                        evidence=[],
                        confidence=0.9,
                        importance=0.5,
                        durability="permanent",
                    )
                ],
            ),
            provider_name=self.name,
            model_name=self.model,
            prompt_version="simple-v1",
            raw_response_hash="test",
            input_tokens=1,
            output_tokens=1,
            latency_ms=1,
            stage="facts",
        )


def test_batch_is_one_extraction_call_and_indexes_chunks_and_memories(tmp_path):
    provider = RecordingProvider()
    embedding = RecordingEmbedding()
    db = TermyteDB(tmp_path / "batch.sqlite", extraction_provider=provider, embedding_provider=embedding)

    result = db.ingest_batch(
        [
            {
                "namespace_id": "batch",
                "idempotency_key": "one",
                "type": "decision",
                "stream_id": "session",
                "payload": {"text": "Decision: use SQLite."},
            },
            {
                "namespace_id": "batch",
                "idempotency_key": "two",
                "type": "constraint",
                "stream_id": "session",
                "payload": {"text": "Constraint: deploy in India."},
            },
        ]
    )

    assert len(provider.requests) == 1
    assert len(provider.requests[0].events) == 2
    assert len(embedding.batches) == 2
    assert len(embedding.batches[-1]) == result.accepted == 2
    assert len(db.memories("batch")) == 2
    assert db.database.execute("SELECT COUNT(*) FROM processing_jobs").fetchone()[0] == 0
    db.close()


def test_single_extraction_call_stays_single_when_legacy_multistage_env_is_set(tmp_path, monkeypatch):
    monkeypatch.setenv("TERMYTEDB_EXTRACTION_STAGES", "all")
    provider = RecordingProvider()
    db = TermyteDB(tmp_path / "single-call.sqlite", extraction_provider=provider, embedding_provider=RecordingEmbedding())

    db.ingest({"namespace_id": "single", "idempotency_key": "one", "type": "decision", "payload": {"text": "Decision: use SQLite."}})

    assert len(provider.requests) == 1
    db.close()


def test_legacy_ungrounded_candidate_is_rejected(tmp_path):
    db = TermyteDB(tmp_path / "simple.sqlite", extraction_provider=SimpleResultProvider(), embedding_provider=RecordingEmbedding())

    result = db.ingest({"namespace_id": "simple", "idempotency_key": "one", "type": "conversation", "payload": {"text": "I prefer SQLite for local storage."}})

    assert result.accepted == 0
    assert result.rejected == 1
    assert db.memories("simple") == []
    db.close()


def test_raw_session_search_survives_empty_memory_extraction(tmp_path):
    class EmptyProvider:
        name = "empty"
        model = "empty-v1"

        def extract(self, request, timeout_seconds=30.0, cancellation=None):
            return ProviderResult(
                response=ExtractionResponse(schema_version="extraction-v1", prompt_version="simple-v1", candidates=[]),
                provider_name=self.name, model_name=self.model, prompt_version="simple-v1", raw_response_hash="test",
                input_tokens=1, output_tokens=1, latency_ms=1, stage="facts",
            )

    db = TermyteDB(tmp_path / "raw-session.sqlite", extraction_provider=EmptyProvider(), embedding_provider=RecordingEmbedding())
    db.ingest({"namespace_id": "raw", "idempotency_key": "one", "type": "conversation", "stream_id": "s1", "payload": {"text": "My favorite database is SQLite."}})

    assert db.memories("raw") == []
    hit = db.search_sessions("raw", "Which database is my favorite?", limit=1)[0]
    assert hit.session_id == "s1"
    assert "SQLite" in hit.text
    assert db.search_context("raw", "favorite database", limit=1)["sessions"][0].session_id == "s1"
    db.close()


def test_memory_search_accepts_a_large_transcript_query(tmp_path):
    db = TermyteDB(tmp_path / "large-query.sqlite", extraction_provider=SimpleResultProvider(), embedding_provider=RecordingEmbedding())
    db.ingest({"namespace_id": "large", "idempotency_key": "one", "type": "conversation", "payload": {"text": "SQLite is used."}})

    # This previously produced more than 1,000 SQL OR expressions.
    assert isinstance(db.search("large", " ".join(f"token{index}" for index in range(1200)), limit=5), list)
    db.close()


def test_provider_failure_keeps_evidence_without_partial_memories(tmp_path):
    db = TermyteDB(tmp_path / "failure.sqlite", extraction_provider=FailingProvider(), embedding_provider=RecordingEmbedding())

    with pytest.raises(ProviderError, match="provider unavailable"):
        db.ingest(
            {
                "namespace_id": "failure",
                "idempotency_key": "one",
                "type": "decision",
                "payload": {"text": "Decision: use SQLite."},
            }
        )

    assert db.database.execute("SELECT COUNT(*) FROM events WHERE namespace_id='failure'").fetchone()[0] == 1
    assert db.database.execute("SELECT COUNT(*) FROM memories WHERE namespace_id='failure'").fetchone()[0] == 0
    # Durable: failed ingestion must leave a retryable job
    assert db.database.execute("SELECT COUNT(*) FROM processing_jobs WHERE namespace_id='failure'").fetchone()[0] == 1
    row = db.database.execute("SELECT status, next_attempt_at FROM processing_jobs WHERE namespace_id='failure'").fetchone()
    assert row[0] in {"failed", "dead", "pending"}
    # Retry via process() still sees the job (will fail again with same provider, immediately retryable)
    resp = db.process("failure")
    assert resp.failed == 1 or resp.dead_lettered == 1
    db.close()
