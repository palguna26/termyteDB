"""Normalize coding-agent hook payloads into observation events."""

from __future__ import annotations

from typing import Any


def coding_observation(namespace_id: str, idempotency_key: str, payload: dict[str, Any], *, agent: str, session_id: str | None = None,
                       stream_id: str | None = None, event_type: str = "agent_observation") -> dict[str, Any]:
    """Return the generic event-v1 shape used by Claude Code and Codex adapters."""
    return {
        "namespace_id": namespace_id,
        "idempotency_key": idempotency_key,
        "type": event_type,
        "agent_id": agent,
        "session_id": session_id,
        "stream_id": stream_id or session_id,
        "payload": payload,
    }


def claude_code_observation(namespace_id: str, idempotency_key: str, payload: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    return coding_observation(namespace_id, idempotency_key, payload, agent="claude-code", **kwargs)


def codex_observation(namespace_id: str, idempotency_key: str, payload: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    return coding_observation(namespace_id, idempotency_key, payload, agent="codex", **kwargs)
