from datetime import UTC, datetime, timedelta


def test_events_form_deterministic_stream_episodes(db):
    base = datetime(2026, 1, 1, tzinfo=UTC)
    db.ingest({"namespace_id": "episodes", "idempotency_key": "a", "type": "conversation", "stream_id": "s", "occurred_at": base, "payload": {"text": "start"}})
    db.ingest(
        {
            "namespace_id": "episodes", "idempotency_key": "b", "type": "conversation", "stream_id": "s",
            "occurred_at": base + timedelta(minutes=5), "payload": {"text": "continue"},
        }
    )
    db.ingest(
        {
            "namespace_id": "episodes", "idempotency_key": "c", "type": "conversation", "stream_id": "s",
            "occurred_at": base + timedelta(hours=2), "payload": {"text": "new task"},
        }
    )

    episodes = db.repository.list_episodes("episodes")
    assert len(episodes) == 2
    assert episodes[0]["status"] == "active"
    assert db.repository.db.execute("SELECT COUNT(*) FROM episode_events WHERE namespace_id=?", ("episodes",)).fetchone()[0] == 3


def test_episode_assignment_survives_restart(tmp_path):
    from termytedb import TermyteDB

    path = tmp_path / "episodes.sqlite"
    first = TermyteDB(path)
    first.ingest({"namespace_id": "restart", "idempotency_key": "a", "type": "conversation", "stream_id": "s", "payload": {"text": "start"}})
    first.close()
    second = TermyteDB(path)
    assert len(second.repository.list_episodes("restart")) == 1
    second.close()
