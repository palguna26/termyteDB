from __future__ import annotations

from datetime import UTC, datetime

from termytedb.schemas import EvidenceSpan, ExtractionCandidate, ExtractionResponse

from termytedb import TermyteDB
from termytedb.memory.provider import FakeExtractionProvider


def candidate(event_id, text: str, *, confidence: float, intent: str, valid_from=None):
    return ExtractionCandidate(
        kind="decision",
        subject="database",
        statement=text,
        evidence=[EvidenceSpan(event_id=event_id, start_offset=0, end_offset=len(text), excerpt=text)],
        confidence=confidence,
        durability="permanent",
        intent=intent,
        valid_from=valid_from,
    )


def set_candidate(db: TermyteDB, item: ExtractionCandidate) -> None:
    db.processor.provider = FakeExtractionProvider(
        ExtractionResponse(schema_version="extraction-v1", prompt_version="test-v1", candidates=[item])
    )


def ingest_model_memory(db: TermyteDB, key: str, text: str, *, confidence: float = 0.9, intent: str = "insert"):
    receipt = db.ingest(
        {"namespace_id": "pipeline", "idempotency_key": key, "type": "conversation", "payload": {"text": text}}
    )
    set_candidate(db, candidate(receipt.event_id, text, confidence=confidence, intent=intent))
    assert db.process("pipeline").accepted == 1


def test_future_dated_memory_is_hidden_until_active(tmp_path):
    db = TermyteDB(tmp_path / "future.sqlite")
    text = "Decision: use the future database."
    receipt = db.ingest(
        {"namespace_id": "pipeline", "idempotency_key": "future", "type": "conversation", "payload": {"text": text}}
    )
    item = candidate(
        receipt.event_id,
        text,
        confidence=0.9,
        intent="insert",
        valid_from=datetime(2099, 1, 1, tzinfo=UTC),
    )
    set_candidate(db, item)

    assert db.process("pipeline").accepted == 1
    assert db.search("pipeline", "future database") == []
    assert db.search("pipeline", "future database", historical=True)
    db.close()


def test_transition_marker_allows_low_confidence_update(tmp_path):
    db = TermyteDB(tmp_path / "transition-marker.sqlite")
    ingest_model_memory(db, "initial", "Decision: use SQLite.")
    ingest_model_memory(
        db,
        "changed",
        "Decision: we switched to PostgreSQL.",
        confidence=0.7,
        intent="update",
    )

    versions = db.history("pipeline", str(db.search("pipeline", "PostgreSQL")[0].memory_id))
    assert versions is not None
    assert [row["status"] for row in versions] == ["superseded", "active"]
    assert db.database.execute("SELECT action FROM extraction_decisions ORDER BY rowid DESC LIMIT 1").fetchone()[0] == "UPDATE"
    db.close()


def test_high_confidence_structured_supersede_does_not_need_marker(tmp_path):
    db = TermyteDB(tmp_path / "transition-confidence.sqlite")
    ingest_model_memory(db, "initial", "Decision: use SQLite.")
    ingest_model_memory(
        db,
        "new",
        "Decision: use PostgreSQL.",
        confidence=0.9,
        intent="supersede",
    )

    assert db.search("pipeline", "PostgreSQL")
    assert db.database.execute("SELECT action FROM extraction_decisions ORDER BY rowid DESC LIMIT 1").fetchone()[0] == "SUPERSEDE"
    db.close()


def test_low_confidence_transition_without_marker_remains_disputed(tmp_path):
    db = TermyteDB(tmp_path / "transition-dispute.sqlite")
    ingest_model_memory(db, "initial", "Decision: use SQLite.")
    ingest_model_memory(
        db,
        "ambiguous",
        "Decision: use PostgreSQL.",
        confidence=0.7,
        intent="update",
    )

    assert db.database.execute("SELECT action FROM extraction_decisions ORDER BY rowid DESC LIMIT 1").fetchone()[0] == "DISPUTE"
    assert db.database.execute("SELECT status FROM memories").fetchone()[0] == "disputed"
    db.close()


def test_processing_writes_graph_edges_and_episode_summary(tmp_path):
    class SummaryProvider:
        name = "summary-test"

        def summarize(self, text: str, *, namespace_id: str, episode_id: str) -> str:
            del namespace_id, episode_id
            return f"SUMMARY::{text[:60]}"

    db = TermyteDB(tmp_path / "graph.sqlite", summary_provider=SummaryProvider())
    text = "Decision: use SQLite with WAL."
    receipt = db.ingest(
        {"namespace_id": "pipeline", "idempotency_key": "graph", "type": "conversation", "payload": {"text": text}}
    )
    db.processor.provider = FakeExtractionProvider(
        ExtractionResponse(
            schema_version="extraction-v1",
            prompt_version="test-v1",
            candidates=[
                ExtractionCandidate(
                    kind="decision",
                    subject="sqlite",
                    statement=text,
                    evidence=[EvidenceSpan(event_id=receipt.event_id, start_offset=0, end_offset=len(text), excerpt=text)],
                    confidence=0.95,
                    durability="permanent",
                )
            ],
        )
    )

    result = db.process("pipeline")
    assert result.accepted == 1

    predicates = [row[0] for row in db.database.execute("SELECT predicate FROM relationships ORDER BY rowid").fetchall()]
    assert "expresses" in predicates
    assert "contains" in predicates

    episode = db.episodes("pipeline")[0]
    assert episode["summary"].startswith("SUMMARY::")
    rebuilt = db.repository.rebuild_graph_index("pipeline")
    assert rebuilt["edges"] >= 1
    search = db.search("pipeline", "SQLite")
    assert search
    assert search[0].component_scores["graph_proximity"] > 0
    assert search[0].component_scores["session_summary"] > 0
    db.close()
