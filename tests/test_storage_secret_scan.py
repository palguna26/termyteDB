from pathlib import Path

from termytedb import TermyteDB


def test_redacted_secret_is_absent_from_sqlite_files(tmp_path: Path):
    secret = "WAL-ONLY-SUPERSECRET-8f2c"
    database_path = tmp_path / "secrets.sqlite"
    db = TermyteDB(database_path)
    db.ingest(
        {
            "namespace_id": "storage-scan",
            "idempotency_key": "secret",
            "type": "note",
            "payload": {"text": f"Decision: use API_KEY={secret}."},
        }
    )
    db.process("storage-scan")
    db.close()

    for path in tmp_path.iterdir():
        if path.is_file():
            assert secret.encode() not in path.read_bytes(), path.name
