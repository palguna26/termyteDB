def test_context_reports_selection_diagnostics_and_token_exclusions(db):
    db.ingest({"namespace_id": "context", "idempotency_key": "one", "type": "decision", "payload": {"text": "Decision: use SQLite for local persistence."}})
    db.process("context")
    result = db.context("context", "SQLite", token_budget=1)
    assert result.abstained is True
    assert result.diagnostics["candidate_count"] >= 1
    assert result.diagnostics["excluded"][0]["reason"] == "token_budget"


def test_irrelevant_query_can_abstain(db):
    db.ingest({"namespace_id": "context", "idempotency_key": "one", "type": "decision", "payload": {"text": "Decision: use SQLite."}})
    db.process("context")
    result = db.context("context", "unrelated quantum zebra", token_budget=100)
    assert result.abstained is True
