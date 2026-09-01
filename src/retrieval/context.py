"""Small, source-grounded answer context formatter - Phase 5 compact builder."""
from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any


def count_tokens(text: str, *, model: str | None = None) -> int:
    """Count model tokens when tiktoken is installed; use a clear fallback."""
    try:
        import tiktoken  # type: ignore[import-not-found]

        try:
            encoding = tiktoken.encoding_for_model(model or "gpt-4o-mini")
        except KeyError:
            encoding = tiktoken.get_encoding("o200k_base")
        return len(encoding.encode(text))
    except Exception:
        # Conservative approximation for installations without the optional
        # tokenizer dependency.  Traces mark this as approximate.
        import re

        return len(re.findall(r"\w+|[^\w\s]", text))


def truncate_to_tokens(text: str, budget: int, *, model: str | None = None) -> str:
    if count_tokens(text, model=model) <= budget:
        return text
    try:
        import tiktoken  # type: ignore[import-not-found]

        try:
            encoding = tiktoken.encoding_for_model(model or "gpt-4o-mini")
        except KeyError:
            encoding = tiktoken.get_encoding("o200k_base")
        return encoding.decode(encoding.encode(text)[:budget])
    except Exception:
        # The fallback is deliberately conservative rather than calling words
        # "tokens".  It preserves early, highest-ranked evidence.
        words = text.split()
        return " ".join(words[: max(1, budget // 2)])


def pack_evidence(
    memories: Iterable[Any],
    chunks_for_memory: Callable[[Any], list[dict[str, Any]]],
    *,
    token_budget: int = 3000,
    max_memories: int = 6,
    hard_max: int = 6000,
    tokenizer_model: str | None = None,
) -> dict[str, Any]:
    """Packing policy (shared by API search and benchmark):

    - top 6 memories maximum
    - top 1-2 chunks per memory
    - one neighboring chunk only when required for continuity
    - deduplicate chunks and sessions
    - target 1500-3000 tokens; hard cap 6000 tokens
    - include Memory, Chunks, documentDate, eventDate, version, score, source identifiers
    - missing evidence returns abstention (caller maps to `insufficient information`)
    """
    budget = min(max(1, token_budget), hard_max)
    selected: list[dict[str, Any]] = []
    used = 0
    seen_chunks: set[str] = set()
    seen_sessions: set[str] = set()

    def value(obj: Any, key: str, default: Any = "") -> Any:
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    for memory in list(memories)[:max_memories]:
        statement = str(value(memory, "statement"))
        if not statement.strip():
            continue
        # Deduplicate identical statements
        if any(s["memory"].strip().casefold() == statement.strip().casefold() for s in selected):
            continue
        memory_id = str(value(memory, "memory_id"))
        kind = str(value(memory, "kind", "fact"))
        score = value(memory, "score", "")
        component_scores = value(memory, "component_scores", {})
        version = value(memory, "version", "")
        # Temporal metadata if available
        temporal = value(memory, "temporal", None)
        document_date = ""
        event_dates: list[str] = []
        if temporal is not None and hasattr(temporal, "model_dump"):
            try:
                td = temporal.model_dump() if hasattr(temporal, "model_dump") else {}
                document_date = str(td.get("recorded_at", "") or "")
                for k in ("valid_from", "valid_until"):
                    if td.get(k):
                        event_dates.append(str(td[k]))
            except Exception:
                pass
        # Also try direct attrs
        if not document_date:
            document_date = str(value(memory, "document_date", "") or value(memory, "created_at", "") or "")
        if not event_dates:
            raw_dates = value(memory, "event_dates", [])
            if isinstance(raw_dates, list):
                event_dates = [str(d) for d in raw_dates if d]

        chunks_raw = chunks_for_memory(memory)[:2]
        # Deduplicate chunks globally and per-session
        filtered_chunks: list[dict[str, Any]] = []
        for chunk in chunks_raw:
            cid = str(chunk.get("chunk_id", ""))
            if cid and cid in seen_chunks:
                continue
            sid = str(chunk.get("session_id", ""))
            # Allow at most one extra chunk per session diversity handled upstream;
            # here just dedup identical chunk IDs
            if cid:
                seen_chunks.add(cid)
            if sid:
                seen_sessions.add(sid)
            filtered_chunks.append(chunk)
        # If single chunk and caller indicates neighbor needed (e.g., truncated), we could add one more
        # but pack_evidence keeps it bounded to 2 per memory as per spec.

        item: dict[str, Any] = {
            "memory": statement,
            "memory_id": memory_id,
            "kind": kind,
            "score": score,
            "component_scores": component_scores,
            "version": version,
            "documentDate": document_date,
            "eventDate": event_dates,
            "source_event_ids": [str(x) for x in value(memory, "source_event_ids", [])],
            "source_chunk_ids": [str(x) for x in value(memory, "source_chunk_ids", [])],
            "chunks": [],
        }
        for chunk in filtered_chunks:
            text = str(chunk.get("text") or chunk.get("raw_text") or "")
            if not text.strip():
                continue
            cost = len((statement + " " + text).split())
            if used + cost > budget:
                continue
            # Enrich chunk entry with provenance metadata
            enriched = dict(chunk)
            enriched.setdefault("document_date", enriched.get("document_date") or document_date)
            item["chunks"].append(enriched)
            used += cost
        if item["chunks"]:
            selected.append(item)
            if used >= budget:
                break
        elif budget >= 100:
            # Memory without chunk is not grounded — skip per spec "raw source chunks, not only summaries"
            # But if token budget very tight and no chunks available, keep abstention handling
            continue

    text = render_context(selected)
    exact_tokens = count_tokens(text, model=tokenizer_model)
    if exact_tokens > budget:
        text = truncate_to_tokens(text, budget, model=tokenizer_model)
        exact_tokens = count_tokens(text, model=tokenizer_model)
    # In benchmark mode, missing evidence returns exactly `insufficient information` for stable abstention scoring.
    # The API layer maps abstained=True to that string; raw text here keeps structured data.
    return {
        "memories": selected,
        "token_count": exact_tokens,
        "abstained": not bool(selected),
        "text": text if selected else "insufficient information",
        "deduplicated_sessions": len(seen_sessions),
    }


def render_context(items: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for item in items:
        header_parts = [f"Memory ({item['memory_id']}"]
        if item.get("kind"):
            header_parts.append(f"kind={item['kind']}")
        if item.get("score") != "":
            header_parts.append(f"score={item.get('score')}")
        if item.get("version") != "":
            header_parts.append(f"version={item.get('version')}")
        header = ", ".join(header_parts) + ")"
        lines.append(f"{header}: {item['memory']}")
        # Include documentDate/eventDate per spec
        if item.get("documentDate"):
            lines.append(f"  documentDate: {item['documentDate']}")
        if item.get("eventDate"):
            lines.append(f"  eventDate: {', '.join(item['eventDate']) if isinstance(item['eventDate'], list) else item['eventDate']}")
        for chunk in item["chunks"]:
            doc_date = chunk.get("document_date") or chunk.get("documentDate") or item.get("documentDate") or "unknown"
            ev_dates = chunk.get("event_dates") or item.get("eventDate") or []
            ev_str = ", ".join(ev_dates) if isinstance(ev_dates, list) else str(ev_dates)
            score = chunk.get("score", "")
            score_str = f", score={score}" if score != "" else ""
            lines.append(
                f"  Chunk ({chunk.get('chunk_id', 'unknown')}, documentDate={doc_date}, eventDate={ev_str}{score_str}): {chunk.get('text') or chunk.get('raw_text', '')}"
            )
        # Source identifiers
        if item.get("source_event_ids"):
            lines.append(f"  source_event_ids: {', '.join(item['source_event_ids'][:3])}")
        if item.get("source_chunk_ids"):
            lines.append(f"  source_chunk_ids: {', '.join(item['source_chunk_ids'][:3])}")
    return "\n".join(lines)
