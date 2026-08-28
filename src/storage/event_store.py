"""Focused event persistence - extracted from god-object Repository.

Single responsibility: events, artifacts, episodes, encoding decisions,
and idempotency guard. No memory or retrieval logic here.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any, cast

from ..core.errors import IdempotencyConflict
from ..core.redaction import redact_text
from ..memory.encoding import score_observation
from ..memory.extractor import payload_text
from ..memory.provider import SessionSummaryProvider
from ..models import EventInput
from .db import Database

TRANSITION_MARKERS = (
    "correction",
    "corrected",
    "replace",
    "supersede",
    "instead",
    "switched",
    "moved",
    "migrated",
    "changed",
    "no longer",
    "now using",
    "updated",
    "prefer",
    "renamed",
    "deprecated",
    "stopped",
)


def iso(value: datetime | None = None) -> str:
    return (value or datetime.now(UTC)).astimezone(UTC).isoformat()


def stable_uuid(namespace_id: str, idempotency_key: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"termytedb:event:{namespace_id}:{idempotency_key}"))


def hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def safe_artifact_uri(uri: str | None, content_hash: str) -> str | None:
    if not uri:
        return None
    expected = f"cas://{content_hash.removeprefix('sha256:')}"
    return uri if uri == expected else "[REDACTED]"


def canonical_event_content(event: EventInput, redacted_payload: dict[str, Any]) -> str:
    artifacts = [
        {
            **item.model_dump(mode="json", exclude={"uri", "metadata"}),
            "uri": safe_artifact_uri(item.uri, item.content_hash),
            "metadata": {key: redact_text(value) for key, value in item.metadata.items()},
        }
        for item in event.artifacts
    ]
    return json.dumps(
        {
            "protocol_version": event.protocol_version,
            "type": event.type,
            "stream_id": event.stream_id,
            "actor_id": event.actor_id,
            "agent_id": event.agent_id,
            "session_id": event.session_id,
            "source_id": event.source_id,
            "artifacts": artifacts,
            "occurred_at": event.occurred_at.isoformat() if event.occurred_at else None,
            "payload": redacted_payload,
        },
        sort_keys=True,
        separators=(",", ":"),
    )


class EventStore:
    """Event log with deterministic episode assignment and idempotency."""

    def __init__(self, db: Database):
        self.db = db

    def ensure_namespace(self, namespace_id: str, org_id: str = "default") -> None:
        with self.db.lock, self.db.connection:
            self.db.execute(
                "INSERT OR IGNORE INTO namespaces(id, org_id, created_at) VALUES (?, ?, ?)",
                (namespace_id, org_id, iso()),
            )

    def ingest(self, namespace_id: str, event: EventInput, redacted_payload: dict[str, Any]) -> tuple[str, bool, str]:
        if event.namespace_id != namespace_id:
            raise ValueError("event namespace does not match repository namespace")
        with self.db.lock:
            self.ensure_namespace(namespace_id)
            event_id = stable_uuid(namespace_id, event.idempotency_key)
            idempotency_hash = hash_text(event.idempotency_key)
            payload_json = json.dumps(redacted_payload, sort_keys=True, separators=(",", ":"))
            content_hash = hash_text(canonical_event_content(event, redacted_payload))
            occurred = iso(event.occurred_at)
            observation_text = payload_text({**redacted_payload, "__termytedb_event_type": event.type})
            prior_count = (
                int(
                    self.db.execute(
                        "SELECT COUNT(*) FROM events WHERE namespace_id=? AND stream_id IS ? AND payload_json LIKE ?",
                        (namespace_id, event.stream_id, f"%{observation_text[:80]}%"),
                    ).fetchone()[0]
                )
                if observation_text
                else 0
            )
            encoding = score_observation(observation_text, repeated=min(1.0, prior_count / 3))
            with self.db.connection:
                existing = self.db.execute(
                    "SELECT id, content_hash FROM events WHERE namespace_id = ? AND idempotency_hash = ?",
                    (namespace_id, idempotency_hash),
                ).fetchone()
                if existing:
                    if existing["content_hash"] != content_hash:
                        raise IdempotencyConflict("idempotency key is already used for different content")
                    return existing["id"], True, existing["content_hash"]
                self.db.execute(
                    """INSERT INTO events
                    (id, namespace_id, protocol_version, stream_id, actor_id, agent_id, session_id, source_id, idempotency_hash, type, occurred_at,
                     payload_json, content_hash, redaction_state, created_at, observation_hash, importance_score, encoding_reason)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'redacted', ?, ?, ?, ?)""",
                    (
                        event_id,
                        namespace_id,
                        event.protocol_version,
                        event.stream_id,
                        event.actor_id,
                        event.agent_id,
                        event.session_id,
                        event.source_id,
                        idempotency_hash,
                        event.type,
                        occurred,
                        payload_json,
                        content_hash,
                        iso(),
                        content_hash,
                        encoding.importance_score,
                        encoding.reason,
                    ),
                )
                for artifact in event.artifacts:
                    safe_uri = safe_artifact_uri(artifact.uri, artifact.content_hash)
                    safe_metadata = {key: redact_text(value) for key, value in artifact.metadata.items()}
                    self.db.execute(
                        """INSERT INTO artifacts(id, namespace_id, event_id, media_type, size_bytes, uri, content_hash, metadata_json)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            str(uuid.uuid4()),
                            namespace_id,
                            event_id,
                            artifact.media_type,
                            artifact.size_bytes,
                            safe_uri,
                            artifact.content_hash,
                            json.dumps(safe_metadata, sort_keys=True, separators=(",", ":")),
                        ),
                    )
                episode_id = self._assign_episode(namespace_id, event_id, event.stream_id, occurred)
                ordinal = int(self.db.execute("SELECT ordinal FROM episode_events WHERE episode_id=? AND event_id=?", (episode_id, event_id)).fetchone()[0])
                self.db.execute(
                    "UPDATE events SET episode_id=?, sequence_number=? WHERE id=? AND namespace_id=?",
                    (episode_id, ordinal, event_id, namespace_id),
                )
                self.db.execute(
                    """INSERT INTO encoding_decisions
                    (id, namespace_id, event_id, importance_score, novelty, surprise, task_relevance,
                     repetition, outcome_signal, correction_signal, future_use, privacy_penalty, reason, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        str(uuid.uuid4()),
                        namespace_id,
                        event_id,
                        encoding.importance_score,
                        encoding.novelty,
                        encoding.surprise,
                        encoding.task_relevance,
                        encoding.repetition,
                        encoding.outcome_signal,
                        encoding.correction_signal,
                        encoding.future_use,
                        encoding.privacy_penalty,
                        encoding.reason,
                        iso(),
                    ),
                )
            return event_id, False, content_hash

    def events_by_id(self, namespace_id: str, event_ids: list[str]) -> list[sqlite3.Row]:
        if not event_ids:
            return []
        placeholders = ",".join("?" for _ in event_ids)
        rows = self.db.execute(
            f"SELECT * FROM events WHERE namespace_id=? AND id IN ({placeholders}) ORDER BY created_at, sequence_number, id",
            (namespace_id, *event_ids),
        ).fetchall()
        by_id = {str(row["id"]): row for row in rows}
        return [by_id[event_id] for event_id in event_ids if event_id in by_id]

    def _assign_episode(self, namespace_id: str, event_id: str, stream_id: str | None, occurred: str) -> str:
        event_time = datetime.fromisoformat(occurred)
        candidates = self.db.execute(
            """SELECT ep.id, ep.start_event_id, ep.end_event_id,
                      start.occurred_at AS start_at, end.occurred_at AS end_at
               FROM episodes ep
               JOIN events start ON start.id=ep.start_event_id
               JOIN events end ON end.id=ep.end_event_id
               WHERE ep.namespace_id=? AND (ep.stream_id IS ? OR ep.stream_id=?)
               ORDER BY ep.updated_at DESC""",
            (namespace_id, stream_id, stream_id),
        ).fetchall()
        selected: sqlite3.Row | None = None
        for row in candidates:
            start = datetime.fromisoformat(row["start_at"])
            end = datetime.fromisoformat(row["end_at"])
            if min(abs(event_time - start), abs(event_time - end)) <= timedelta(minutes=30):
                selected = row
                break
        if selected is None:
            episode_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"termytedb:episode:{namespace_id}:{event_id}"))
            self.db.execute(
                """INSERT INTO episodes(id, namespace_id, stream_id, start_event_id, end_event_id, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (episode_id, namespace_id, stream_id, event_id, event_id, iso(), iso()),
            )
            self.db.execute(
                "INSERT INTO episode_events(episode_id, namespace_id, event_id, ordinal) VALUES (?, ?, ?, 0)",
                (episode_id, namespace_id, event_id),
            )
            return episode_id
        episode_id = cast(str, selected["id"])
        count = int(self.db.execute("SELECT COUNT(*) FROM episode_events WHERE episode_id=?", (episode_id,)).fetchone()[0])
        self.db.execute(
            "INSERT INTO episode_events(episode_id, namespace_id, event_id, ordinal) VALUES (?, ?, ?, ?)",
            (episode_id, namespace_id, event_id, count),
        )
        start_id = selected["start_event_id"]
        end_id = selected["end_event_id"]
        if event_time < datetime.fromisoformat(selected["start_at"]):
            start_id = event_id
        if event_time > datetime.fromisoformat(selected["end_at"]):
            end_id = event_id
        self.db.execute(
            "UPDATE episodes SET start_event_id=?, end_event_id=?, updated_at=? WHERE id=? AND namespace_id=?",
            (start_id, end_id, iso(), episode_id, namespace_id),
        )
        return episode_id

    def list_episodes(self, namespace_id: str, limit: int = 100, offset: int = 0) -> list[dict[str, Any]]:
        rows = self.db.execute("SELECT * FROM episodes WHERE namespace_id=? ORDER BY created_at, id LIMIT ? OFFSET ?", (namespace_id, limit, offset)).fetchall()
        return [dict(row) for row in rows]

    def update_episode(self, namespace_id: str, episode_id: str, status: str, summary: str | None) -> bool:
        with self.db.lock, self.db.connection:
            cursor = self.db.execute(
                "UPDATE episodes SET status=?, summary=COALESCE(?, summary), updated_at=? WHERE id=? AND namespace_id=?",
                (status, redact_text(summary) if summary else None, iso(), episode_id, namespace_id),
            )
            return cursor.rowcount == 1

    def get_event(self, namespace_id: str, event_id: str) -> dict[str, Any] | None:
        row = self.db.execute("SELECT * FROM events WHERE id=? AND namespace_id=?", (event_id, namespace_id)).fetchone()
        if not row:
            return None
        result = dict(row)
        result["payload_json"] = json.loads(result["payload_json"])
        result["artifacts"] = [
            {**dict(item), "metadata_json": json.loads(item["metadata_json"])}
            for item in self.db.execute("SELECT * FROM artifacts WHERE event_id=? AND namespace_id=? ORDER BY id", (event_id, namespace_id)).fetchall()
        ]
        result["evidence_refs"] = [
            dict(item)
            for item in self.db.execute("SELECT * FROM evidence_refs WHERE event_id=? AND namespace_id=? ORDER BY id", (event_id, namespace_id)).fetchall()
        ]
        return result

    def list_events(self, namespace_id: str, limit: int = 100, offset: int = 0) -> list[dict[str, Any]]:
        rows = self.db.execute(
            "SELECT id FROM events WHERE namespace_id=? ORDER BY occurred_at, id LIMIT ? OFFSET ?",
            (namespace_id, limit, offset),
        ).fetchall()
        return [event for row in rows if (event := self.get_event(namespace_id, row["id"])) is not None]

    def extraction_window(self, namespace_id: str, event_id: str, *, limit: int = 4) -> dict[str, str]:
        current = self.get_event(namespace_id, event_id)
        if current is None:
            return {}
        scope_stream = current.get("stream_id")
        scope_session = current.get("session_id")
        current_sequence = current.get("sequence_number")
        params: list[Any] = [namespace_id]
        where = ["namespace_id = ?"]
        if scope_stream is not None:
            where.append("stream_id = ?")
            params.append(scope_stream)
        elif scope_session is not None:
            where.append("session_id = ?")
            params.append(scope_session)
        else:
            return {str(current["id"]): payload_text(current["payload_json"], current["type"])}
        if current_sequence is None:
            params.extend([current["occurred_at"], current["occurred_at"], current["id"], max(1, limit)])
            rows = self.db.execute(
                f"""SELECT id, type, payload_json FROM events WHERE {" AND ".join(where)}
                  AND (occurred_at < ? OR (occurred_at = ? AND id <= ?)) ORDER BY occurred_at DESC, id DESC LIMIT ?""",
                tuple(params),
            ).fetchall()
        else:
            params.extend([current["occurred_at"], current["occurred_at"], current_sequence, current_sequence, current["id"], max(1, limit)])
            rows = self.db.execute(
                f"""SELECT id, type, payload_json FROM events WHERE {" AND ".join(where)}
                  AND (occurred_at < ? OR (occurred_at = ? AND (COALESCE(sequence_number, -1) < ? OR (COALESCE(sequence_number, -1) = ? AND id <= ?))))
                  ORDER BY occurred_at DESC, COALESCE(sequence_number, -1) DESC, id DESC LIMIT ?""",
                tuple(params),
            ).fetchall()
        window: dict[str, str] = {}
        for row in reversed(rows):
            window[str(row["id"])] = payload_text(json.loads(row["payload_json"]), row["type"])
        if str(current["id"]) not in window:
            window[str(current["id"])] = payload_text(current["payload_json"], current["type"])
        return window

    def refresh_episode_summary(
        self,
        namespace_id: str,
        episode_id: str,
        *,
        limit: int = 8,
        summary_provider: SessionSummaryProvider | None = None,
    ) -> str | None:
        rows = self.db.execute(
            """SELECT e.* FROM episode_events ee JOIN events e ON e.id=ee.event_id
            WHERE ee.namespace_id=? AND ee.episode_id=? ORDER BY ee.ordinal LIMIT ?""",
            (namespace_id, episode_id, max(1, min(limit, 20))),
        ).fetchall()
        if not rows:
            return None
        snippets: list[str] = []
        seen: set[str] = set()
        for row in rows:
            text = payload_text({**json.loads(row["payload_json"]), "__termytedb_event_type": row["type"]})
            text = " ".join(text.split())
            if not text:
                continue
            short = text[:180]
            key = short.casefold()
            if key in seen:
                continue
            seen.add(key)
            role = str(row["actor_id"] or row["type"] or "event")
            snippets.append(f"{role}: {short}")
            if len(snippets) >= 4:
                break
        base_text = "\n".join(snippets)
        summary: str | None = None
        if summary_provider is not None:
            try:
                summary = summary_provider.summarize(base_text, namespace_id=namespace_id, episode_id=episode_id)
            except Exception:
                summary = None
        if not summary:
            summary = " | ".join(snippets) if snippets else None
        if summary:
            with self.db.lock, self.db.connection:
                self.db.execute(
                    "UPDATE episodes SET summary=?, updated_at=? WHERE id=? AND namespace_id=?",
                    (redact_text(summary) if summary else None, iso(), episode_id, namespace_id),
                )
        return summary
