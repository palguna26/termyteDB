from termytedb import TermyteDB


def test_search_exposes_explainable_component_scores(tmp_path):
    db = TermyteDB(tmp_path / "components.sqlite")
    db.ingest({"namespace_id": "components", "idempotency_key": "one", "type": "note", "payload": {"text": "Decision: use SQLite."}})
    db.process("components")
    result = db.search("components", "decision SQLite")[0]
    assert result.component_scores["confidence"] == 1.0
    assert result.component_scores["evidence_quality"] > 0
    assert result.component_scores["memory_type_signal"] == 1.0
    assert result.component_scores["temporal_signal"] == 1.0
    db.close()
