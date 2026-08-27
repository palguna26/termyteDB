from __future__ import annotations

from src.memory.consolidator import consolidate


def test_observation_encoding_is_explainable_and_episode_ordered(db):
    first = db.ingest(
        {
            "namespace_id": "memory-loop",
            "idempotency_key": "one",
            "type": "conversation",
            "stream_id": "task",
            "payload": {"text": "Decision: use SQLite. Remember this important constraint."},
        }
    )
    duplicate = db.ingest(
        {
            "namespace_id": "memory-loop",
            "idempotency_key": "one",
            "type": "conversation",
            "stream_id": "task",
            "payload": {"text": "Decision: use SQLite. Remember this important constraint."},
        }
    )
    assert duplicate.duplicate is True
    decisions = db.encoding_decisions("memory-loop")
    assert len(decisions) == 1
    assert decisions[0]["importance_score"] > 0
    assert "explicit_emphasis" in decisions[0]["reason"]
    event = db.event("memory-loop", str(first.event_id))
    assert event["episode_id"]
    assert event["sequence_number"] == 0


def test_replay_dry_run_and_apply_keep_lineage(db):
    db.ingest(
        {
            "namespace_id": "replay",
            "idempotency_key": "one",
            "type": "conversation",
            "stream_id": "task",
            "payload": {"text": "Decision: use SQLite."},
        }
    )
    preview = consolidate(db.repository, "replay", mode="dry-run")
    assert preview["proposal_count"] >= 1
    assert db.repository.list_consolidation_runs("replay")[0]["status"] == "completed"
    applied = consolidate(db.repository, "replay", mode="apply")
    assert applied["accepted"] >= 1
    assert db.memories("replay")
    assert db.repository.db.execute("SELECT COUNT(*) FROM consolidation_proposals WHERE namespace_id=? AND status='accepted'", ("replay",)).fetchone()[0] >= 1


def test_accessibility_is_control_not_deletion(db):
    db.ingest({"namespace_id": "forget", "idempotency_key": "one", "type": "conversation", "payload": {"text": "Decision: retain this."}})
    db.process("forget")
    memory = db.memories("forget")[0]
    assert db.repository.accessibility("forget") == 1
    assert db.get_memory("forget", str(memory.memory_id)) is not None
    assert db.repository.history("forget", str(memory.memory_id))
