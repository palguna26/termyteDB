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


def test_context_groups_memory_kinds_and_reports_retrieval_modes(db):
    db.ingest({"namespace_id": "grouped", "idempotency_key": "one", "type": "decision", "payload": {"text": "Decision: use SQLite."}})
    db.process("grouped")
    result = db.context("grouped", "SQLite", token_budget=100)
    assert "[Decision memories]" in result.text
    assert result.diagnostics["selected_by_kind"] == {"decision": 1}
    assert "lexical" in result.diagnostics["retrieval_modes"]


def test_token_count_uses_conservative_fallback(monkeypatch):
    from src.retrieval import context

    monkeypatch.setattr(context, "_token_encoder", lambda: None)
    assert context.token_count("one two three four") == 6


def test_token_count_uses_bpe_encoder_when_available(monkeypatch):
    from src.retrieval import context

    class Encoder:
        def encode(self, text, disallowed_special=()):
            assert disallowed_special == ()
            return list(range(9))

    monkeypatch.setattr(context, "_token_encoder", lambda: Encoder())
    assert context.token_count("C:/code/x.py::function") == 9
