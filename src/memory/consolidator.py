"""Small, deterministic replay worker for the first consolidation loop."""

from __future__ import annotations

import json
import uuid
from typing import Any

from ..storage.repository import Repository
from .extraction import rule_candidate_to_contract, validate_candidate
from .extractor import extract, payload_text


def consolidate(repository: Repository, namespace_id: str, *, limit: int = 5, mode: str = "dry-run") -> dict[str, Any]:
    """Replay high-signal episodes and optionally publish supported proposals.

    The worker is intentionally deterministic. A model-based consolidator can be
    added later, but it must use the same proposal and evidence boundary.
    """
    episodes = repository.db.execute(
        """SELECT id FROM episodes WHERE namespace_id=? ORDER BY updated_at DESC LIMIT ?""", (namespace_id, max(1, min(limit, 100)))
    ).fetchall()
    episode_ids = [str(row["id"]) for row in episodes]
    run_id = repository.create_consolidation_run(namespace_id, episode_ids, mode)
    accepted = rejected = 0
    proposals: list[dict[str, Any]] = []
    try:
        for episode_id in episode_ids:
            events = repository.db.execute(
                """SELECT e.* FROM episode_events ee JOIN events e ON e.id=ee.event_id
                WHERE ee.episode_id=? AND ee.namespace_id=? ORDER BY ee.ordinal""",
                (episode_id, namespace_id),
            ).fetchall()
            included: dict[uuid.UUID, str] = {}
            for event in events:
                text = payload_text({**json.loads(event["payload_json"]), "__termytedb_event_type": event["type"]})
                included[uuid.UUID(event["id"])] = text
            for event in events:
                source = included[uuid.UUID(event["id"])]
                for rule in extract({**json.loads(event["payload_json"]), "__termytedb_event_type": event["type"]}):
                    candidate = rule_candidate_to_contract(rule, uuid.UUID(event["id"]), source)
                    proposal = {
                        "episode_id": episode_id,
                        "event_id": event["id"],
                        "kind": candidate.kind,
                        "subject_key": candidate.subject,
                        "statement": candidate.statement,
                    }
                    if mode == "dry-run":
                        repository.record_consolidation_proposal(
                            namespace_id, run_id, candidate.kind, candidate.subject, candidate.statement, [event["id"]], "proposed", "dry_run"
                        )
                        proposals.append(proposal)
                        accepted += 1
                        continue
                    try:
                        validated = validate_candidate(namespace_id, candidate, included)
                        memory_id, action, version_id = repository.reconcile_candidate(namespace_id, event, validated, run_id)
                        if candidate.kind == "procedure":
                            repository.upsert_procedure(
                                namespace_id,
                                candidate.statement,
                                str(event["type"]),
                                [],
                                [candidate.statement],
                                candidate.statement,
                                candidate.statement,
                                [],
                                True,
                                [(str(event["id"]), candidate.statement)],
                            )
                        repository.record_consolidation_proposal(
                            namespace_id,
                            run_id,
                            candidate.kind,
                            candidate.subject,
                            candidate.statement,
                            [event["id"]],
                            "accepted",
                            action,
                            memory_id,
                            version_id,
                        )
                        proposals.append({**proposal, "action": action, "memory_id": memory_id})
                        accepted += 1
                    except Exception as exc:
                        repository.record_consolidation_proposal(
                            namespace_id, run_id, candidate.kind, candidate.subject, candidate.statement, [event["id"]], "rejected", type(exc).__name__
                        )
                        rejected += 1
        repository.finish_consolidation_run(namespace_id, run_id, "completed", accepted, rejected)
    except Exception as exc:
        repository.finish_consolidation_run(namespace_id, run_id, "failed", accepted, rejected, str(exc))
        raise
    repository.accessibility(namespace_id)
    return {
        "run_id": run_id,
        "mode": mode,
        "episode_count": len(episode_ids),
        "proposal_count": len(proposals),
        "accepted": accepted,
        "rejected": rejected,
        "proposals": proposals,
    }
