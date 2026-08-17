import pytest


def test_event_protocol_identity_fields_are_persisted_and_hashed(db):
    event = {
        "namespace_id": "protocol",
        "idempotency_key": "one",
        "type": "tool",
        "protocol_version": "event-v1",
        "stream_id": "stream-1",
        "actor_id": "actor-1",
        "agent_id": "agent-1",
        "session_id": "session-1",
        "source_id": "source-1",
        "payload": {"text": "Decision: use SQLite."},
    }
    receipt = db.ingest(event)
    stored = db.event("protocol", str(receipt.event_id))
    assert stored is not None
    assert stored["protocol_version"] == "event-v1"
    assert stored["actor_id"] == "actor-1"
    assert stored["agent_id"] == "agent-1"
    assert stored["session_id"] == "session-1"
    assert stored["source_id"] == "source-1"


def test_event_payload_size_boundary_rejects_oversized_payload(db):
    with pytest.raises(ValueError, match="payload exceeds"):
        db.ingest({"namespace_id": "protocol", "idempotency_key": "large", "type": "document", "payload": {"blob": "x" * 1_100_000}})
