from __future__ import annotations

import sqlite3

from .conftest import event


def test_evidence_from_another_namespace_cannot_support_memory(db):
    db.ingest(event("n1", "one", "Decision: Use SQLite."))
    db.process("n1")
    memory = db.search("n1", "SQLite")[0]
    with db.database.connection:
        try:
            db.database.execute(
                "UPDATE evidence_refs SET namespace_id='n2' WHERE memory_version_id=?",
                (str(memory.memory_version_id),),
            )
        except sqlite3.IntegrityError:
            pass
        else:
            raise AssertionError("cross-namespace evidence link was accepted")


def test_search_and_context_cannot_leak_between_namespaces(db):
    db.ingest(event("n1", "one", "Decision: Use SQLite."))
    db.ingest(event("n2", "two", "Decision: Use PostgreSQL."))
    db.process("n1")
    db.process("n2")
    assert db.search("n1", "PostgreSQL") == []
    context = db.context("n1", "PostgreSQL")
    assert context.abstained is True
    assert context.results == []
