from __future__ import annotations

import pytest

from .conftest import event


def test_memory_requires_evidence(db):
    db.repository.ensure_namespace("n1")
    with pytest.raises(Exception):
        with db.database.connection:
            db.database.execute(
                """INSERT INTO memory_versions
                (id, memory_id, namespace_id, version, statement, valid_from, recorded_at, status, reason)
                VALUES ('v', 'm', 'n1', 1, 'orphan', datetime('now'), datetime('now'), 'active', 'test')"""
            )


def test_new_version_supersedes_old_and_search_excludes_it(db):
    db.ingest(event("n1", "one", "Decision: storage uses SQLite."))
    db.process("n1")
    old = db.search("n1", "SQLite")[0]
    db.ingest(event("n1", "two", "Decision: storage uses PostgreSQL."))
    db.process("n1")
    assert db.search("n1", "SQLite") == []
    current = db.search("n1", "PostgreSQL")[0]
    assert current.memory_id == old.memory_id
    assert db.database.execute("SELECT COUNT(*) FROM memory_versions WHERE memory_id=?", (str(old.memory_id),)).fetchone()[0] == 2
