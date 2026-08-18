from __future__ import annotations

import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from termytedb.errors import IdempotencyConflict
from termytedb.service import create_app

from .conftest import event


def test_imports_do_not_create_files(tmp_path: Path):
    source_root = str(Path(__file__).parents[1] / "src")
    result = subprocess.run(
        [sys.executable, "-c", "import termytedb, termytedb.db, termytedb.service, termytedb.__main__"],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": source_root},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert list(tmp_path.iterdir()) == []


def test_database_path_is_explicit(db):
    from termytedb import TermyteDB

    with pytest.raises(ValueError):
        TermyteDB()
    with pytest.raises(ValueError):
        create_app()


def test_application_lifespan_closes_database(tmp_path):
    app = create_app(tmp_path / "lifespan.sqlite")
    with TestClient(app) as client:
        assert client.get("/v1/memories/not-found", params={"namespace_id": "n1"}).status_code == 404
    with pytest.raises(Exception):
        app.state.engine.database.connection.execute("SELECT 1")


def test_same_key_can_be_used_in_two_namespaces(db):
    first = db.ingest(event("n1", "same", "Decision: storage uses SQLite."))
    second = db.ingest(event("n2", "same", "Decision: storage uses SQLite."))
    assert first.event_id != second.event_id


def test_key_order_does_not_change_content_hash(db):
    first = db.ingest(
        {
            "namespace_id": "n1",
            "idempotency_key": "ordered",
            "type": "decision",
            "payload": {"a": "one", "b": {"x": 1, "y": 2}},
        }
    )
    second = db.ingest(
        {
            "namespace_id": "n1",
            "idempotency_key": "ordered",
            "type": "decision",
            "payload": {"b": {"y": 2, "x": 1}, "a": "one"},
        }
    )
    assert second.duplicate is True
    assert second.event_id == first.event_id


def test_changed_content_is_a_conflict_and_http_409(db, tmp_path):
    db.ingest(event("n1", "same", "Decision: storage uses SQLite."))
    with pytest.raises(IdempotencyConflict):
        db.ingest(event("n1", "same", "Decision: storage uses PostgreSQL."))

    client = TestClient(create_app(tmp_path / "http.sqlite"))
    client.post("/v1/events", json=event("n1", "same", "Decision: storage uses SQLite."))
    response = client.post("/v1/events", json=event("n1", "same", "Decision: storage uses PostgreSQL."))
    assert response.status_code == 409
    assert "PostgreSQL" not in response.text


def test_concurrent_duplicate_ingestion_creates_one_event_and_job(db):
    payload = event("n1", "concurrent", "Decision: storage uses SQLite.")
    with ThreadPoolExecutor(max_workers=8) as executor:
        receipts = list(executor.map(db.ingest, [payload] * 32))
    assert len({receipt.event_id for receipt in receipts}) == 1
    assert db.database.execute("SELECT COUNT(*) FROM events WHERE namespace_id='n1'").fetchone()[0] == 1
    assert db.database.execute("SELECT COUNT(*) FROM processing_jobs WHERE namespace_id='n1'").fetchone()[0] == 1
