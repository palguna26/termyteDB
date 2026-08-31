"""Small, source-grounded answer context formatter."""
from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any


def pack_evidence(memories: Iterable[Any], chunks_for_memory: Callable[[Any], list[dict[str, Any]]], *, token_budget: int = 3000, max_memories: int = 6, hard_max: int = 6000) -> dict[str, Any]:
    budget = min(max(1, token_budget), hard_max)
    selected: list[dict[str, Any]] = []
    used = 0
    def value(obj: Any, key: str, default: Any = "") -> Any:
        return getattr(obj, key, obj.get(key, default) if isinstance(obj, dict) else default)
    for memory in list(memories)[:max_memories]:
        statement = str(value(memory, "statement"))
        item = {"memory": statement, "memory_id": str(value(memory, "memory_id")), "chunks": []}
        for chunk in chunks_for_memory(memory)[:2]:
            text = str(chunk.get("text") or chunk.get("raw_text") or "")
            cost = len((statement + " " + text).split())
            if used + cost > budget:
                continue
            item["chunks"].append(chunk)
            used += cost
        if item["chunks"]:
            selected.append(item)
    return {"memories": selected, "token_count": used, "abstained": not bool(selected), "text": render_context(selected)}


def render_context(items: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for item in items:
        lines.append(f"Memory ({item['memory_id']}): {item['memory']}")
        for chunk in item["chunks"]:
            lines.append(f"Chunk ({chunk.get('chunk_id', 'unknown')}, documentDate={chunk.get('document_date', 'unknown')}): {chunk.get('text') or chunk.get('raw_text', '')}")
    return "\n".join(lines)
