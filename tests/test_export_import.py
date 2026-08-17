from termytedb import TermyteDB


def test_namespace_export_import_preserves_memory_history_and_search(tmp_path):
    source = TermyteDB(tmp_path / "source.sqlite")
    source.ingest({"namespace_id": "roundtrip", "idempotency_key": "one", "type": "decision", "stream_id": "s", "payload": {"text": "Decision: use SQLite."}})
    source.process("roundtrip")
    document = source.export_namespace("roundtrip")
    assert document["extraction_runs"]
    assert document["episodes"]
    source.close()

    target = TermyteDB(tmp_path / "target.sqlite")
    counts = target.import_namespace(document, "roundtrip")
    assert counts["events"] == 1
    assert target.search("roundtrip", "SQLite")
    assert len(target.repository.history("roundtrip", str(target.search("roundtrip", "SQLite")[0].memory_id)) or []) == 1
    assert target.import_namespace(document, "roundtrip")["events"] == 0
    target.close()


def test_import_rejects_mixed_namespace_rows(tmp_path):
    db = TermyteDB(tmp_path / "import.sqlite")
    document = {"namespaces": [{"id": "safe", "org_id": "default", "created_at": "now", "deleted_at": None}],
                "events": [{"namespace_id": "other"}], "memories": [], "memory_versions": [], "evidence_refs": [], "processing_jobs": []}
    try:
        db.import_namespace(document, "safe")
    except ValueError as exc:
        assert "wrong namespace" in str(exc)
    else:
        raise AssertionError("mixed namespace export was accepted")
    db.close()
