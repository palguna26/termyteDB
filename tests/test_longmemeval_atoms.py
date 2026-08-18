from termytedb.db import Database
from termytedb.longmemeval_extraction import L1Atom, insert_atoms


def test_atoms_schema_indexes_fact_and_timestamp(tmp_path):
    db = Database(tmp_path / "atoms.sqlite")
    insert_atoms(db, [L1Atom("a1", "s1", "User lived in Mysore", "2024-08-18", "user")])
    row = db.execute("SELECT fact, timestamp, source_role, invalid_at FROM atoms WHERE atom_id='a1'").fetchone()
    assert dict(row) == {"fact": "User lived in Mysore", "timestamp": "2024-08-18", "source_role": "user", "invalid_at": None}
    assert db.execute("SELECT atom_id FROM atoms_fts WHERE atoms_fts MATCH 'Mysore'").fetchone()[0] == "a1"
    assert db.execute("SELECT atom_id FROM atoms_fts WHERE atoms_fts MATCH '\"2024-08-18\"'").fetchone()[0] == "a1"
    db.close()


def test_invalidated_atoms_are_relationally_linked(tmp_path):
    db = Database(tmp_path / "invalid.sqlite")
    insert_atoms(db, [
        L1Atom("old", "s1", "User lived in London", "2023-01-01", "user"),
        L1Atom("new", "s2", "User lived in Berlin", "2024-01-01", "user"),
    ])
    db.execute("UPDATE atoms SET invalid_at=?, superseded_by=? WHERE atom_id=?", ("2024-01-01", "new", "old"))
    db.connection.commit()
    assert db.execute("SELECT COUNT(*) FROM atoms WHERE invalid_at IS NULL").fetchone()[0] == 1
    assert db.execute("SELECT superseded_by FROM atoms WHERE atom_id='old'").fetchone()[0] == "new"
    db.close()
