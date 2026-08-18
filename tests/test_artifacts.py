import pytest
from termytedb.errors import IdempotencyConflict


def test_artifact_metadata_is_persisted_and_namespace_scoped(db):
    content_hash = "sha256:" + "a" * 64
    receipt = db.ingest(
        {
            "namespace_id": "artifacts",
            "idempotency_key": "artifact-1",
            "type": "document",
            "source_id": "capture-1",
            "artifacts": [
                {"content_hash": content_hash, "media_type": "text/plain", "size_bytes": 12, "uri": "cas://" + "a" * 64, "metadata": {"name": "notes.txt"}}
            ],
            "payload": {"text": "Decision: preserve the document."},
        }
    )
    event = db.event("artifacts", str(receipt.event_id))
    assert event is not None
    assert event["artifacts"][0]["content_hash"] == content_hash
    assert event["artifacts"][0]["metadata_json"] == {"name": "notes.txt"}
    assert db.event("other", str(receipt.event_id)) is None


def test_artifact_hash_changes_event_identity(db):
    base = {"namespace_id": "artifacts", "idempotency_key": "same", "type": "document", "payload": {"text": "same"}}
    first = db.ingest({**base, "artifacts": [{"content_hash": "sha256:" + "a" * 64, "media_type": "text/plain", "size_bytes": 1}]})
    with pytest.raises(IdempotencyConflict):
        db.ingest({**base, "artifacts": [{"content_hash": "sha256:" + "b" * 64, "media_type": "text/plain", "size_bytes": 1}]})
    assert first.content_hash
