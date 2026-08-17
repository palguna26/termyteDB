from __future__ import annotations

import io
import logging
from pathlib import Path

import pytest

from termytedb.integrity import check_database, repair_fts
from termytedb.logging import JsonFormatter

from .conftest import event


def test_integrity_tool_and_deterministic_fts_repair(db):
    db.ingest(event("n1", "one", "Decision: storage uses SQLite."))
    db.process("n1")
    assert check_database(db.database).ok
    db.database.execute("DELETE FROM memory_fts")
    assert check_database(db.database).missing_fts == 1
    repair_fts(db.database)
    assert check_database(db.database).ok


def test_integrity_tool_repairs_embedding_index_too(db):
    db.ingest(event("n1", "one", "Decision: storage uses SQLite."))
    db.process("n1")
    db.database.execute("DELETE FROM memory_embeddings")
    assert check_database(db.database).missing_embeddings == 1
    repair_fts(db.database)
    assert check_database(db.database).missing_embeddings == 0


def test_nested_secrets_are_absent_from_all_persistent_surfaces(db, tmp_path: Path):
    secret = "SUPERSECRET123456789"
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JsonFormatter())
    logger = logging.getLogger("redaction-test")
    logger.handlers.clear()
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    from termytedb import TermyteDB

    local = TermyteDB(tmp_path / "secrets.sqlite", logger=logger)
    local.ingest(
        {
            "namespace_id": "n1",
            "idempotency_key": "secret",
            "type": "decision",
            "payload": {
                "nested": {"api_key": secret},
                "parts": ["API_KEY=", secret],
                "text": f"Decision: storage uses SQLite. API_KEY={secret}",
            },
        }
    )
    local.process("n1")
    local.checkpoint()
    database_text = "\n".join(
        str(row[0])
        for table, column in (
            ("events", "payload_json"),
            ("memories", "subject_key"),
            ("memory_versions", "statement"),
            ("evidence_refs", "excerpt"),
            ("processing_jobs", "last_error"),
            ("memory_fts", "statement"),
        )
        for row in local.database.execute(f"SELECT {column} FROM {table}").fetchall()
    )
    log_text = stream.getvalue()
    wal = Path(str(local.database.path) + "-wal")
    journal = Path(str(local.database.path) + "-journal")
    wal_text = wal.read_bytes().decode("utf-8", errors="ignore") if wal.exists() else ""
    journal_text = journal.read_bytes().decode("utf-8", errors="ignore") if journal.exists() else ""
    assert secret not in database_text
    assert secret not in log_text
    assert secret not in wal_text
    assert secret not in journal_text
    assert secret not in local.context("n1", "SQLite").text
    local.close()


def test_secret_is_redacted_from_job_error_and_structured_failure_log(db, monkeypatch):
    secret = "SUPERSECRET123456789"
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JsonFormatter())
    logger = logging.getLogger("failure-redaction-test")
    logger.handlers.clear()
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    from termytedb import TermyteDB

    local = TermyteDB(db.database.path + "-failure", logger=logger)
    local.ingest(event("n1", "failure", "Decision: storage uses SQLite."))
    import termytedb.processor as processor_module

    monkeypatch.setattr(
        processor_module,
        "extract",
        lambda payload: (_ for _ in ()).throw(RuntimeError(f"API_KEY={secret}")),
    )
    local.process("n1")
    error = local.database.execute("SELECT last_error FROM processing_jobs").fetchone()[0]
    assert secret not in error
    assert secret not in stream.getvalue()
    local.close()


def test_schema_version_mismatch_is_reported(db):
    with db.database.connection:
        db.database.execute("DELETE FROM schema_migrations WHERE version=2")
    assert check_database(db.database).schema_compatible is False


@pytest.mark.parametrize(
    "failure_prefix",
    [
        "INSERT INTO memory_versions",
        "INSERT INTO evidence_refs",
        "INSERT INTO memory_fts",
    ],
)
def test_processing_rolls_back_at_each_memory_write_stage(db, monkeypatch, failure_prefix):
    db.ingest(event("n1", failure_prefix, "Decision: storage uses SQLite."))
    original_execute = db.database.execute
    state = {"failed": False}

    def fail_stage(sql: str, parameters=()):
        if sql.lstrip().startswith(failure_prefix) and not state["failed"]:
            state["failed"] = True
            raise RuntimeError("forced stage failure")
        return original_execute(sql, parameters)

    monkeypatch.setattr(db.database, "execute", fail_stage)
    assert db.process("n1").failed == 1
    for table in ("memories", "memory_versions", "evidence_refs", "memory_fts"):
        assert db.database.execute(f"SELECT COUNT(*) FROM {table} WHERE namespace_id='n1'").fetchone()[0] == 0
