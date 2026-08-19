def test_search_exposes_hybrid_component_scores_and_persists_index(db):
    db.ingest({"namespace_id": "hybrid", "idempotency_key": "one", "type": "decision", "payload": {"text": "Decision: use SQLite for local storage."}})
    db.process("hybrid")
    result = db.search("hybrid", "SQLite")[0]
    assert result.lexical_score > 0
    assert result.vector_score >= 0
    assert db.repository.db.execute("SELECT COUNT(*) FROM memory_embeddings WHERE namespace_id=?", ("hybrid",)).fetchone()[0] == 1


def test_vector_candidates_remain_namespace_scoped(db):
    db.ingest({"namespace_id": "left", "idempotency_key": "one", "type": "decision", "payload": {"text": "Decision: use SQLite."}})
    db.ingest({"namespace_id": "right", "idempotency_key": "two", "type": "decision", "payload": {"text": "Decision: use SQLite."}})
    db.process("left")
    db.process("right")
    left = db.search("left", "SQLite")
    assert len(left) == 1
    assert left[0].citations[0].event_id != db.search("right", "SQLite")[0].citations[0].event_id


def test_lexical_match_without_embedding_candidate_is_not_returned(db):
    db.ingest({"namespace_id": "dense-required", "idempotency_key": "one", "type": "decision", "payload": {"text": "Decision: use SQLite."}})
    db.process("dense-required")
    db.database.execute("DELETE FROM memory_embeddings WHERE namespace_id=?", ("dense-required",))
    db.database.connection.commit()
    assert db.search("dense-required", "SQLite") == []
