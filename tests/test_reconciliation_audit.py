def test_rule_mode_records_reinforce_for_duplicate_evidence(db):
    event = {"namespace_id": "audit-actions", "type": "decision", "payload": {"text": "Decision: use SQLite."}}
    db.ingest({**event, "idempotency_key": "one"})
    db.ingest({**event, "idempotency_key": "two"})
    assert db.process("audit-actions").accepted == 2
    actions = [row[0] for row in db.database.execute("SELECT action FROM extraction_decisions ORDER BY created_at, id").fetchall()]
    assert actions == ["INSERT", "REINFORCE"]
