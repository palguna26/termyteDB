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


def test_lexical_match_remains_available_without_embedding_candidate(db):
    db.ingest({"namespace_id": "dense-required", "idempotency_key": "one", "type": "decision", "payload": {"text": "Decision: use SQLite."}})
    db.process("dense-required")
    db.database.execute("DELETE FROM memory_embeddings WHERE namespace_id=?", ("dense-required",))
    db.database.connection.commit()
    results = db.search("dense-required", "SQLite")
    assert len(results) == 1
    assert results[0].statement == "Decision: use SQLite."
    assert results[0].lexical_score > 0
    assert results[0].vector_score == 0


def test_temporal_language_retrieves_superseded_memory(db):
    import hashlib
    import json
    from uuid import uuid4

    from termytedb.api.schemas import EvidenceSpan, ExtractionCandidate, ExtractionResponse
    from termytedb.memory.provider import ProviderResult

    texts = ["I lived in Delhi.", "I live in Mumbai."]
    responses = [
        ExtractionResponse(
            schema_version="extraction-v1",
            prompt_version="test",
            candidates=[
                ExtractionCandidate(
                    kind="fact", subject="user location", statement=texts[0],
                    evidence=[EvidenceSpan(event_id=uuid4(), start_offset=0, end_offset=len(texts[0]), excerpt=texts[0])],
                    confidence=1, durability="permanent", intent="insert",
                )
            ],
        ),
        ExtractionResponse(
            schema_version="extraction-v1",
            prompt_version="test",
            candidates=[
                ExtractionCandidate(
                    kind="fact", subject="user location", statement=texts[1],
                    evidence=[EvidenceSpan(event_id=uuid4(), start_offset=0, end_offset=len(texts[1]), excerpt=texts[1])],
                    confidence=1, durability="permanent", intent="update",
                )
            ],
        ),
    ]

    class Provider:
        name = "temporal-test"
        model = "test"

        def extract(self, request, timeout_seconds=30.0, cancellation=None):
            del timeout_seconds, cancellation
            response = responses.pop(0)
            event_id = request.events[0]
            candidate = response.candidates[0].model_copy(
                update={"evidence": [response.candidates[0].evidence[0].model_copy(update={"event_id": event_id})]}
            )
            response = response.model_copy(update={"candidates": [candidate]})
            raw = json.dumps(response.model_dump(mode="json"), sort_keys=True).encode()
            return ProviderResult(response, self.name, self.model, "test", hashlib.sha256(raw).hexdigest(), None, None, 1)

    db.processor.provider = Provider()
    for index, text in enumerate(texts):
        db.ingest({"namespace_id": "temporal", "idempotency_key": str(index), "type": "note", "payload": {"text": text}})
    db.process("temporal")

    assert all(result.statement != texts[0] for result in db.search("temporal", "Delhi"))
    historical = db.search("temporal", "Delhi in 2020")
    assert historical
    assert historical[0].statement == texts[0]


def test_entity_aliases_add_graph_candidates_to_search(db):
    db.ingest({"namespace_id": "aliases", "idempotency_key": "one", "type": "decision", "payload": {"text": "Decision: use SQLite."}})
    db.process("aliases")
    memory = db.memories("aliases")[0]
    subject = db.database.execute(
        "SELECT id FROM entities WHERE namespace_id=? AND canonical_key=?",
        ("aliases", f"subject:{memory.subject_key}"),
    ).fetchone()
    assert subject is not None
    db.repository.add_entity_alias("aliases", str(subject["id"]), "storage engine")

    results = db.search("aliases", "storage engine")
    assert results
    assert results[0].memory_id == memory.memory_id
    assert results[0].component_scores["alias_rank"] > 0
