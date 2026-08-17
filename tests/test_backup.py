from termytedb import TermyteDB


def test_sqlite_backup_reopens_with_preserved_evidence(tmp_path):
    source = tmp_path / "source.sqlite"
    target = tmp_path / "backups" / "copy.sqlite"
    db = TermyteDB(source)
    receipt = db.ingest(
        {"namespace_id": "backup", "idempotency_key": "one", "type": "note", "payload": {"text": "Decision: use SQLite."}}
    )
    db.process("backup")
    db.backup(target)
    db.close()

    restored = TermyteDB(target)
    assert restored.event("backup", str(receipt.event_id)) is not None
    assert restored.search("backup", "SQLite")
    restored.close()


def test_sqlite_backup_rejects_live_database_path(tmp_path):
    path = tmp_path / "source.sqlite"
    db = TermyteDB(path)
    try:
        import pytest

        with pytest.raises(ValueError, match="differ"):
            db.backup(path)
    finally:
        db.close()
