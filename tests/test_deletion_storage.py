from pathlib import Path

from termytedb import TermyteDB


def test_namespace_deletion_removes_secret_from_sqlite_files(tmp_path: Path):
    secret = "DELETE-ME-STORAGE-SECRET-31e9"
    database_path = tmp_path / "deletion.sqlite"
    db = TermyteDB(database_path)
    db.ingest(
        {
            "namespace_id": "delete-me",
            "idempotency_key": "secret",
            "type": "note",
            "payload": {"text": f"Decision: use API_KEY={secret}."},
        }
    )
    db.process("delete-me")
    assert db.delete_namespace("delete-me") is True
    for path in tmp_path.iterdir():
        if path.is_file():
            assert secret.encode() not in path.read_bytes(), path.name
    db.close()
    for path in tmp_path.iterdir():
        if path.is_file():
            assert secret.encode() not in path.read_bytes(), path.name
