from __future__ import annotations

from pathlib import Path

from termytedb import TermyteDB

from .conftest import event


def test_complete_vertical_slice_survives_restart(tmp_path: Path):
    path = tmp_path / "persistent.sqlite"
    first_db = TermyteDB(path)
    receipt = first_db.ingest(event("project-a", "run-1", "Decision: Use SQLite because it is portable."))
    first_db.process("project-a")
    before = first_db.context("project-a", "SQLite", token_budget=100)
    memory = first_db.get_memory("project-a", str(before.results[0].memory_id))
    first_db.close()

    second_db = TermyteDB(path)
    after = second_db.context("project-a", "SQLite", token_budget=100)
    assert after.text == before.text
    assert after.results[0].memory_version_id == before.results[0].memory_version_id
    assert after.results[0].citations[0].event_id == receipt.event_id
    assert memory is not None
    row = second_db.database.execute("SELECT payload_json FROM events WHERE id=?", (str(receipt.event_id),)).fetchone()
    assert "SUPERSECRET" not in row[0]
    second_db.close()
