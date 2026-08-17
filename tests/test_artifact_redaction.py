def test_artifact_metadata_and_uri_are_redacted_before_persistence(db):
    secret = "SUPER-ARTIFACT-SECRET"
    receipt = db.ingest(
        {
            "namespace_id": "artifact-redaction",
            "idempotency_key": "one",
            "type": "document",
            "artifacts": [{
                "content_hash": "sha256:" + "a" * 64,
                "media_type": "text/plain",
                "size_bytes": 1,
                "uri": f"https://example.test/{secret}",
                "metadata": {"note": f"token={secret}"},
            }],
            "payload": {"text": "document"},
        }
    )
    stored = db.event("artifact-redaction", str(receipt.event_id))
    exported = db.export_namespace("artifact-redaction")
    assert stored is not None
    assert secret not in str(stored)
    assert secret not in str(exported)
