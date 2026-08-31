import pytest

from src import TermyteDB
from src.memory.provider import FakeExtractionProvider, ProviderError


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


def test_batch_is_one_extraction_call_and_one_memory_embedding_batch(tmp_path):
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
    assert len(embedding.batches) == 1
    assert len(embedding.batches[0]) == result.accepted == 2
    assert len(db.memories("batch")) == 2
    assert db.database.execute("SELECT COUNT(*) FROM processing_jobs").fetchone()[0] == 0
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
