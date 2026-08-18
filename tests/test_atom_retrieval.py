from termytedb.db import Database
from termytedb.longmemeval_extraction import L1Atom, insert_atoms

from termytedb.retrieval import pack_context, search_atoms


def test_retrieval_filters_stale_and_history_restores_it(tmp_path):
    db = Database(tmp_path / "retrieval.sqlite")
    insert_atoms(
        db,
        [
            L1Atom("old", "s1", "User lived in London", "2023-01-01", "user"),
            L1Atom("new", "s2", "User lived in Berlin", "2024-01-01", "assistant"),
        ],
    )
    db.execute("UPDATE atoms SET invalid_at='2024-01-01', superseded_by='new' WHERE atom_id='old'")
    db.connection.commit()
    assert [hit.atom_id for hit in search_atoms(db, "London", 10)] == []
    assert "old" in [hit.atom_id for hit in search_atoms(db, "where previously lived London", 10)]
    assert "[Role: assistant]" in pack_context(search_atoms(db, "Berlin"))
    db.close()
