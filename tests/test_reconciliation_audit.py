def test_rule_mode_records_reinforce_for_duplicate_evidence(db):
    event = {"namespace_id": "audit-actions", "type": "decision", "payload": {"text": "Decision: use SQLite."}}
    db.ingest({**event, "idempotency_key": "one"})
    db.ingest({**event, "idempotency_key": "two"})
    assert db.process("audit-actions").accepted == 2
    actions = [row[0] for row in db.database.execute("SELECT action FROM extraction_decisions ORDER BY rowid").fetchall()]
    assert actions == ["INSERT", "REINFORCE"]


def test_model_reconciliation_records_dispute_and_ignore(tmp_path):
    from time import perf_counter

    from termytedb import TermyteDB
    from termytedb.provider import ProviderResult
    from termytedb.schemas import EvidenceSpan, ExtractionCandidate, ExtractionResponse

    database = TermyteDB(tmp_path / "actions.sqlite")
    first = database.ingest({"namespace_id": "model-actions", "idempotency_key": "one", "type": "note", "payload": {"text": "Decision: use SQLite."}})
    second = database.ingest({"namespace_id": "model-actions", "idempotency_key": "two", "type": "note", "payload": {"text": "The reports disagree."}})

    def candidate(event_id, text, intent, statement):
        return ExtractionCandidate(
            kind="decision", subject="storage", statement=statement,
            evidence=[EvidenceSpan(event_id=event_id, start_offset=0, end_offset=len(text), excerpt=text)],
            confidence=1, durability="session", intent=intent,
        )

    responses = [
        ExtractionResponse(
            schema_version="extraction-v1", prompt_version="p",
            candidates=[candidate(first.event_id, "Decision: use SQLite.", "insert", "Decision: use SQLite.")],
        ),
        ExtractionResponse(
            schema_version="extraction-v1", prompt_version="p",
            candidates=[candidate(second.event_id, "The reports disagree.", "dispute", "The reports disagree.")],
        ),
    ]

    class Provider:
        name = "sequence"
        model = "test"

        def extract(self, request, timeout_seconds=30.0, cancellation=None):
            del request, timeout_seconds, cancellation
            started = perf_counter()
            response = responses.pop(0)
            return ProviderResult(response, self.name, self.model, response.prompt_version, "hash", None, None, int((perf_counter() - started) * 1000))

    database.processor.provider = Provider()
    assert database.process("model-actions").accepted == 2
    actions = [row[0] for row in database.database.execute("SELECT action FROM extraction_decisions ORDER BY rowid").fetchall()]
    assert actions == ["INSERT", "DISPUTE"]
    database.close()
